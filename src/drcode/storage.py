"""Task persistence: TWO files per task under `<project>/.drcode/tasks/`.

- `<id>.json` — the metadata file: every Task field except the bulky
  transcripts, plus the `skeleton` (the cheap user-prompts + final-response
  projection of the dev transcript the archive search indexes). Kilobytes,
  rewritten on every `save()`.
- `<id>.transcript.json` — the sidecar with the full serialized
  `messages` / `tester_messages` / `resolver_messages` (plain lists of
  serialized ModelMessage dicts, the same shape the monolithic pre-split
  files carried). Written only when the task's transcripts are in memory —
  every mutation goes through `task.messages` (which lazily loads the
  sidecar first), so an in-memory transcript is by definition the
  authoritative state and a save flushes it. A save of a task whose
  transcripts were never loaded (the common case at startup: hundreds of
  completed tasks) touches metadata only — a status change no longer
  rewrites megabytes.

`load_all` / `load_one` parse metadata only and attach a lazy loader to
each task; the sidecar is read on the first transcript access (agent run
resume, tester/resolver round, opening the task in chat, compaction,
context dialog — every reader goes through the properties, so the load
happens automatically at the right point).

A sidecar-transcript LRU (see `transcript_cache_limit`) evicts the
in-memory transcripts of idle COMPLETED tasks so long sessions don't
accumulate every transcript ever viewed.

A THIRD, append-only file exists per task when the main transcript was
ever compacted: `<id>.transcript.archive.jsonl` — one JSON line per
compaction of `task.messages`, carrying the full pre-compaction
transcript plus metadata (timestamp, gate level, compaction path, role).
It exists purely for incident forensics (the compacted transcript itself
loses the middle of the history — irreversibly so on the mechanical
floor): it is only ever appended to, is never read back by the app, and
is deleted together with the task.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import PROJECT_STATE_DIRNAME
from .gitignore import ensure_drcode_gitignored
from .models import TRANSCRIPT_KEYS, Task
from .transcript import build_skeleton

# Sidecar file suffix: `<id>.transcript.json` next to `<id>.json`. Every
# glob over the tasks dir must skip these (they are not task metadata).
TRANSCRIPT_SUFFIX = ".transcript.json"

# Append-only compaction archive suffix: `<id>.transcript.archive.jsonl`
# next to the metadata+sidecar pair. Also skipped by every metadata glob.
ARCHIVE_SUFFIX = ".transcript.archive.jsonl"


def is_transcript_path(path: Path) -> bool:
    """True for a sidecar transcript file or the compaction archive
    (neither is task metadata)."""
    return path.name.endswith((TRANSCRIPT_SUFFIX, ARCHIVE_SUFFIX))


def _write_tmp(path: Path, text: str) -> str:
    """Write `text` to a temp file next to `path`; returns the tmp path.
    The state dir can vanish mid-session (deleted externally): recreate
    it instead of crashing the pipeline worker — already-lost task files
    cannot be restored, but every write from now on must keep working."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return tmp


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically (tmp file + os.replace)."""
    tmp = _write_tmp(path, text)
    try:
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Storage:
    """Persists tasks as metadata+sidecar JSON pairs under
    `<project>/.drcode/tasks/`."""

    # In-memory sidecar transcripts kept per session before the LRU starts
    # evicting the idle COMPLETED ones (see the module docstring).
    transcript_cache_limit = 8

    def __init__(self, project_root: Path) -> None:
        self.root = project_root
        self.tasks_dir = project_root / PROJECT_STATE_DIRNAME / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        # `.drcode/` holds transcripts and the project config — never
        # commit it (the helper is a no-op outside a git repository).
        ensure_drcode_gitignored(project_root)
        # LRU of sidecar transcripts read this session (task ids, oldest
        # first). Enforced by _evict_hook (installed by the TaskManager —
        # only it can save+unload a Task object).
        self._transcript_order: OrderedDict[str, None] = OrderedDict()
        self._evict_hook: Callable[[str], bool] | None = None

    # -- paths ----------------------------------------------------------------

    def task_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def transcript_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}{TRANSCRIPT_SUFFIX}"

    def archive_path(self, task_id: str) -> Path:
        """The append-only compaction archive of `task_id` (may not exist —
        it is created on the first archived compaction)."""
        return self.tasks_dir / f"{task_id}{ARCHIVE_SUFFIX}"

    def task_file_paths(self) -> list[Path]:
        """The metadata files of the archive (sidecar transcripts and
        leftovers excluded), filename order."""
        return [
            p
            for p in sorted(self.tasks_dir.glob("*.json"))
            if not is_transcript_path(p)
        ]

    # -- loading ----------------------------------------------------------------

    def load_one(self, path: Path) -> Task | None:
        """Parse a single task METADATA file; None on failure.

        A corrupted file is backed up (`.json.bak`), never fatal; a file
        that vanished between the glob and the read (removed externally
        mid-scan) is silently skipped. Never raises on unreadable input —
        callers run inside the pipeline's hot paths.

        Legacy monolithic files (pre-split archives that carry the full
        transcripts inside `<id>.json`) are migrated on the spot: the
        transcripts are split into the sidecar file, the skeleton is
        derived from them and the metadata file is rewritten — both writes
        atomic, the sidecar replaced BEFORE the metadata (a crash in
        between leaves the legacy file in place and the migration simply
        reruns; the reverse order could lose the transcript). The
        migration is idempotent per file.
        """
        if is_transcript_path(path):
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupted task file: keep a backup, don't crash the app.
            try:
                os.replace(path, path.with_suffix(".json.bak"))
            except OSError:
                pass
            return None
        except OSError:
            # Unreadable / disappeared between the glob and the read.
            return None
        if not isinstance(data, dict):
            return self._backup_invalid(path)
        legacy_keys = [key for key in TRANSCRIPT_KEYS if key in data]
        if legacy_keys:
            return self._load_legacy(path, data)
        try:
            task = Task.model_validate(data)
        except ValidationError:
            return self._backup_invalid(path)
        # Lazy sidecar: the transcripts stay on disk until first access.
        task._transcripts.loaded = False
        task._transcripts.loader = lambda task_id=task.id: self.read_transcript(task_id)
        return task

    def _backup_invalid(self, path: Path) -> None:
        try:
            os.replace(path, path.with_suffix(".json.bak"))
        except OSError:
            pass

    def _load_legacy(self, path: Path, data: dict[str, Any]) -> Task | None:
        """Split a legacy monolithic task file into the metadata+sidecar
        pair (see load_one). Validation failure → `.json.bak`, never
        fatal (the load_one contract)."""
        transcripts = {key: data.pop(key, []) for key in TRANSCRIPT_KEYS}
        data["skeleton"] = build_skeleton(transcripts.get("messages") or [])
        try:
            task = Task.model_validate(data)
        except ValidationError:
            return self._backup_invalid(path)
        transcript_path = self.transcript_path(task.id)
        # Write BOTH temp files first, then replace the sidecar and only
        # then the metadata: any crash leaves either the legacy file
        # (migration reruns) or the fully migrated pair — never a new
        # metadata file without its sidecar.
        transcript_text = json.dumps(transcripts, ensure_ascii=False)
        metadata_text = task.model_dump_json(indent=2)
        migrated = False
        try:
            transcript_tmp = _write_tmp(transcript_path, transcript_text)
            metadata_tmp = _write_tmp(path, metadata_text)
            os.replace(transcript_tmp, transcript_path)
            os.replace(metadata_tmp, path)
            migrated = True
        except OSError:
            # A crash between the two replaces leaves the legacy file in
            # place — the next load reruns the migration. A failure of
            # the writes themselves is recovered by keeping the legacy
            # transcripts IN MEMORY: the task behaves as constructed, and
            # the next save() re-attempts the split from live data.
            pass
        if migrated:
            task._transcripts.loaded = False
        task._transcripts.loader = lambda task_id=task.id: self.read_transcript(task_id)
        return task

    def load_all(self) -> list[Task]:
        """Every readable task in the archive, created_at order — metadata
        only: the sidecar transcripts stay on disk until a task is
        actually touched (lazy properties), so the startup cost is
        milliseconds per task regardless of transcript sizes."""
        tasks = [t for t in (self.load_one(p) for p in self.task_file_paths()) if t]
        tasks.sort(key=lambda t: t.created_at)
        return tasks

    def read_transcript(self, task_id: str) -> dict[str, Any]:
        """The sidecar transcripts of `task_id` (lazy-load target of the
        Task properties). A missing/corrupted sidecar degrades to empty
        lists — the metadata file is the source of truth for everything
        else. Bookmarks the task in the sidecar LRU."""
        self._touch_transcript_use(task_id)
        try:
            data = json.loads(self.transcript_path(task_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        result: dict[str, Any] = {}
        for key in TRANSCRIPT_KEYS:
            value = data.get(key)
            result[key] = value if isinstance(value, list) else []
        return result

    def append_transcript_archive(self, task_id: str, record: dict[str, Any]) -> None:
        """Append one eviction record (a JSON line) to the task's
        compaction archive, creating the file on first use.

        The archive is APPEND-ONLY by contract: existing lines are never
        rewritten or dropped, so a crash mid-write can only truncate the
        very last line (the next append starts a fresh one). Raises OSError
        on failure — the caller decides how survivable that is (the
        compaction pipeline logs a service event and moves on)."""
        path = self.archive_path(task_id)
        line = json.dumps(record, ensure_ascii=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # -- sidecar LRU --------------------------------------------------------------

    def _touch_transcript_use(self, task_id: str) -> None:
        self._transcript_order.pop(task_id, None)
        self._transcript_order[task_id] = None
        while len(self._transcript_order) > self.transcript_cache_limit:
            oldest = next(iter(self._transcript_order), None)
            if oldest is None or self._evict_hook is None:
                break
            if not self._evict_hook(oldest):
                break  # hook refuses (task busy/not idle) — soft limit only
            self._transcript_order.pop(oldest, None)

    def set_evict_hook(self, hook: Callable[[str], bool] | None) -> None:
        """Install the sidecar-LRU eviction callback (TaskManager side:
        flush + unload a task's in-memory transcripts). The hook returns
        True when the task was evicted (or had nothing in memory)."""
        self._evict_hook = hook

    # -- saving -----------------------------------------------------------------

    def save(self, task: Task) -> None:
        """Persist `task`: metadata always, the sidecar transcript when
        the transcripts are in memory (then they ARE the newest state —
        every mutation loads them first). A save of a not-loaded task
        never rewrites the sidecar: megabyte status changes become
        kilobyte writes."""
        if task.transcripts_loaded:
            # Refresh the search skeleton from the live dev transcript,
            # then flush the sidecar.
            task.skeleton = build_skeleton(task.messages)
            self._flush_transcript(task)
        metadata_text = task.model_dump_json(indent=2)
        try:
            _atomic_write_text(self.task_path(task.id), metadata_text)
        except OSError:
            pass

    def _flush_transcript(self, task: Task) -> None:
        ts = task._transcripts
        transcripts = {key: getattr(ts, key) for key in TRANSCRIPT_KEYS}
        text = json.dumps(transcripts, ensure_ascii=False)
        try:
            _atomic_write_text(self.transcript_path(task.id), text)
        except OSError:
            pass

    def delete(self, task_id: str) -> None:
        self.task_path(task_id).unlink(missing_ok=True)
        self.transcript_path(task_id).unlink(missing_ok=True)
        self.archive_path(task_id).unlink(missing_ok=True)
        self._transcript_order.pop(task_id, None)
