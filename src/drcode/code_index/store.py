"""The persistent per-project code index (docs/code-index-design.md §3).

`.drcode/code_index.json`, deliberately the same pattern as
`task_search`'s `search_index.json`: per-file entries keyed by the
`(mtime_ns, size)` stat pair, a versioned JSON document written
atomically (`tmp` + `os.replace`), one module-level `threading.Lock`
with one in-memory cache per project root, and the full self-healing
contract — any corrupted/unexpected index shape quietly rebuilds from
disk, a stat mismatch quietly re-parses the file. The index is a CACHE,
never the source of truth: every failure degrades to "slower next
time", never to an agent-visible error.

The physical index lives in the MAIN checkout (`<storage root>/.drcode/`)
— worktree agents only read it; their own edits reach queries through
the stat keys (an edited file's stat pair no longer matches and the file
re-parses into the process cache — §6.2's stat overlay, base version).

BLOCKING file I/O: callers on an event loop must run `load_index` in a
worker thread (`tool_runtime._run_tool` already does).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drcode.version import APP_VERSION

from .ctags_adapter import CtagsExtractor
from .extract import (
    MAX_INDEXED_FILE_BYTES,
    FileIndex,
    SymbolExtractor,
    file_index_from_disk,
    file_index_to_disk,
)

__all__ = [
    "CODE_INDEX_FILENAME",
    "BuildInfo",
    "clear_code_index_cache",
    "load_index",
    "read_main_head",
    "take_build_notice",
]

CODE_INDEX_FILENAME = "code_index.json"
INDEX_VERSION = 1

# Directories never indexed (mirrors the spirit of tools.IGNORED_DIRS —
# this module must not import tools, that would be a cycle; the list is
# intentionally short: caches, VCS state and dependency trees).
_IGNORED_DIRS = {
    ".drcode",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "dist-desktop",
    "build",
    "build-desktop",
    ".tox",
    ".idea",
    ".vscode",
}

_DEFAULT_EXTRACTOR = SymbolExtractor()

# The §2.2 optional coverage adapter: consulted ONLY when the primary
# tree-sitter extractor yields no record (unknown extension / unavailable
# grammar). Inert (everything → None) when no universal-ctags binary is
# usable — the feature never requires it.
_FALLBACK_EXTRACTOR = CtagsExtractor()


def _extract_record(path: Path, text: str) -> FileIndex | None:
    """Primary tree-sitter extraction, then the ctags coverage fallback
    (§2.2) — the composition point that keeps `extract.py` pure."""
    record = _DEFAULT_EXTRACTOR.extract(path, text)
    if record is None:
        record = _FALLBACK_EXTRACTOR.extract(path, text)
    return record


def read_main_head(project_root: Path) -> str | None:
    """The main checkout's HEAD commit (the §6.4 branch-change sentinel).
    Pure file reads (`.git/HEAD` → ref file); the `git rev-parse`
    subprocess is only the packed-refs fallback. None = not a git
    checkout / detached-unresolvable / any failure — the guard then
    stays silent (it is an optimization, not correctness: the stat keys
    alone keep every record honest)."""
    root = Path(project_root)
    git_dir = root / ".git"
    if not git_dir.is_dir():
        return None  # not a checkout root (or a worktree .git FILE)
    try:
        content = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not content.startswith("ref: "):
        return content or None  # detached HEAD — the sha itself
    ref_file = git_dir / content.removeprefix("ref: ").strip()
    try:
        return ref_file.read_text(encoding="utf-8").strip() or None
    except OSError:
        pass
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None if proc.returncode == 0 else None


@dataclass(frozen=True, slots=True)
class BuildInfo:
    """Figures of the last (re)build: how many files/symbols/languages
    the index holds and how long the pass took. Surfaced as ONE service
    event at the first build in a process (§8.3) — chat visibility, never
    model context."""

    files: int
    symbols: int
    languages: int
    ms: int
    built_now: bool


@dataclass
class _CachedIndex:
    entries: dict[str, FileIndex]
    file_keys: dict[str, tuple[int, int]]


# One cache per project root, one module-level lock (task_search's
# `_ARCHIVE_LOCK` pattern): the tools run in worker threads, the TUI and
# the schema estimation may touch the store from others.
_INDEX_LOCK = threading.Lock()
_INDEX_CACHES: dict[str, _CachedIndex] = {}

# The ONE per-root service-event notice ("code index: N files, M symbols…"),
# produced by the first build in this process and consumed by the first
# code_* tool call that sees it (§8.3: lazy first build, one event).
_BUILD_NOTICES: dict[str, str] = {}


def _index_path(project_root: Path) -> Path:
    return Path(project_root) / ".drcode" / CODE_INDEX_FILENAME


def clear_code_index_cache(project_root: Path | None = None) -> None:
    """Drop the in-memory index cache (one project, or all when root is
    None). Purely operational/testing — the cache and the on-disk index
    rebuild lazily on the next `load_index`. Deleting the on-disk file
    forces a full re-index; this hook alone does not."""
    with _INDEX_LOCK:
        if project_root is None:
            _INDEX_CACHES.clear()
            _BUILD_NOTICES.clear()
        else:
            _INDEX_CACHES.pop(str(Path(project_root)), None)
            _BUILD_NOTICES.pop(str(Path(project_root)), None)


def take_build_notice(project_root: Path) -> str | None:
    """Consume (once) the pending first-build service-event line of the
    project, if the index was (re)built in this process since the last
    consumption. Cheap dict access — safe to call on the event loop."""
    return _BUILD_NOTICES.pop(str(Path(project_root)), None)


def reparse_entry(
    project_root: Path, rel: str, *, persist: bool = True
) -> FileIndex | None:
    """Re-parse ONE file unconditionally and refresh its record in the
    cache (and on disk when persisting) — the query-level self-heal
    (§6.4): a query that PROVED its record stale (a body range beyond
    the file, a vanished symbol) must not leave the wrong entry in
    place. Returns the fresh record (None when the file vanished or its
    language is not indexable)."""
    root = Path(project_root)
    path = root / rel
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        record = None
    else:
        record = _extract_record(path, text)
    try:
        stat = path.stat()
        key: tuple[int, int] | None = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        key = None
    with _INDEX_LOCK:
        cached = _INDEX_CACHES.get(str(root))
        if cached is not None and key is not None:
            if record is None:
                cached.entries.pop(rel, None)
            else:
                cached.entries[rel] = record
            cached.file_keys[rel] = key
        if persist and key is not None:
            disk, _head = _load_disk_index(root)
            disk[rel] = [
                list(key),
                file_index_to_disk(record) if record is not None else None,
            ]
            _write_disk_index(root, disk, read_main_head(root))
    return record


def _load_disk_index(project_root: Path) -> tuple[dict[str, list[Any]], str | None]:
    """The persisted index as a raw {relpath: [key, record]} dict plus the
    HEAD sentinel it was built under, or ({}, None) on any problem
    (missing, corrupted, unexpected shape) — the caller then re-parses
    the project and rewrites the file."""
    try:
        data = json.loads(_index_path(project_root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}, None
    if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
        return {}, None
    if data.get("version") != INDEX_VERSION:
        return {}, None
    head = data.get("head")
    return data["entries"], head if isinstance(head, str) else None


def _write_disk_index(
    project_root: Path, entries: dict[str, list[object]], head: str | None
) -> None:
    """Persist the index atomically (tmp + os.replace, §3.3) with the
    HEAD sentinel of the main checkout (§6.4); any failure is swallowed —
    the in-memory cache still serves this process and the next successful
    write catches up."""
    try:
        text = json.dumps(
            {
                "version": INDEX_VERSION,
                "generator": f"drcode-{APP_VERSION}",
                "head": head,
                "entries": entries,
            },
            ensure_ascii=False,
        )
        path = _index_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def _discover_files(root: Path) -> list[Path]:
    """Every candidate source file under the project root (known
    extension — tree-sitter's table OR the ctags fallback's, size cap,
    ignored dirs pruned) — os.walk with pruning, so a 10²–10³-file
    project stays milliseconds."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if (
                _DEFAULT_EXTRACTOR.language_for(path) is None
                and _FALLBACK_EXTRACTOR.language_for(path) is None
            ):
                continue
            found.append(path)
    return found


def load_index(
    project_root: Path, *, persist: bool = True
) -> tuple[dict[str, FileIndex], BuildInfo]:
    """The project's per-file index records, through the incremental
    cache: every candidate file is stat'ed (cheap), a matching
    `(mtime_ns, size)` entry is reused from the in-memory cache or —
    after a restart — from the persistent on-disk index; everything else
    is re-parsed. Returns the entries plus a BuildInfo for the first
    build's service event (§8.3).

    `persist=False` (worktree-isolated tasks: the agent reads a worktree
    whose `.drcode/` must never receive an index file — §7.1/§7.3) keeps
    the index purely in memory: same cache, same stat invalidation, no
    disk read/write. The worktree process-local cache IS the §6.2 stat
    overlay, base version.

    A stale/corrupted entry self-heals by re-parsing; a file that
    vanished since the last build simply drops out of the entries. The
    on-disk index is rewritten only when something actually changed (a
    re-parse, a vanished file, a broken entry) — the common warm call is
    a pure read."""
    root = Path(project_root)
    started = time.monotonic()
    # Fast path outside the lock: stat everything once (task_search's
    # load_archive does the same — no reason to serialize the walk).
    stat_entries: list[tuple[Path, str, tuple[int, int]]] = []
    for path in _discover_files(root):
        try:
            stat = path.stat()
        except OSError:
            continue  # Vanished mid-scan: skip silently.
        if stat.st_size > MAX_INDEXED_FILE_BYTES:
            continue
        rel = path.relative_to(root).as_posix()
        stat_entries.append((path, rel, (stat.st_mtime_ns, stat.st_size)))

    cache_key = str(root)
    with _INDEX_LOCK:
        cached = _INDEX_CACHES.get(cache_key)
        if not isinstance(cached, _CachedIndex):
            cached = None  # Broken/foreign entry: rebuild from scratch.
        old_keys = cached.file_keys if cached else {}
        disk: dict[str, list[Any]] = {}
        persisted_head: str | None = None
        if persist:
            disk, persisted_head = _load_disk_index(root)
        # §6.4 HEAD guard: the index was built on another branch of the
        # main checkout — the base layer is wholesale suspect, so it is
        # dropped and rebuilt (stat keys alone cannot see a branch switch
        # that happened while this process was down). An optimization in
        # the design; the simple full-drop is the honest version of it.
        current_head = read_main_head(root) if persist else None
        head_guard_fired = bool(
            persist
            and persisted_head
            and current_head
            and persisted_head != current_head
        )
        if head_guard_fired:
            disk = {}
        entries: dict[str, FileIndex] = {}
        file_keys: dict[str, tuple[int, int]] = {}
        disk_dirty = False
        changed = False
        for path, rel, key in stat_entries:
            record: FileIndex | None = None
            if old_keys.get(rel) == key and cached is not None:
                record = cached.entries.get(rel)
            if record is None:
                raw = disk.get(rel)
                if (
                    isinstance(raw, list)
                    and len(raw) == 2
                    and isinstance(raw[0], list)
                    and tuple(raw[0]) == key
                ):
                    if raw[1] is None:
                        # A known-unindexable file (unknown extension or
                        # unavailable grammar): honestly absent, and NOT
                        # re-extracted on every call.
                        file_keys[rel] = key
                        continue
                    record = file_index_from_disk(raw[1])
                    if record is None:
                        disk_dirty = True
            if record is None:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue  # Unreadable: skip (the index stays honest).
                record = _extract_record(path, text)
                changed = True
            if record is None:
                # Unknown/unavailable language: the file is honestly
                # absent (grep_search keeps the coverage) — but remember
                # the stat key so we do not re-extract every call.
                file_keys[rel] = key
                if rel in disk:
                    disk_dirty = True
                continue
            entries[rel] = record
            file_keys[rel] = key
        fresh = _CachedIndex(entries=entries, file_keys=file_keys)
        _INDEX_CACHES[cache_key] = fresh
        # Rewrite the persistent index only when something actually
        # changed — a re-parse, a vanished file, a broken entry, or a
        # file that fell out of coverage. The warm read never writes.
        stale_on_disk = any(
            rel not in disk
            or not isinstance(disk[rel], list)
            or len(disk[rel]) != 2
            or not isinstance(disk[rel][0], list)
            or tuple(disk[rel][0]) != key
            for rel, key in file_keys.items()
            if rel in entries
        )
        if persist and (
            changed
            or disk_dirty
            or stale_on_disk
            or head_guard_fired
            or (cached is None and stat_entries)
        ):
            serialized: dict[str, list[object]] = {}
            for rel, key in file_keys.items():
                record = entries.get(rel)
                serialized[rel] = [
                    list(key),
                    file_index_to_disk(record) if record is not None else None,
                ]
            _write_disk_index(root, serialized, current_head)
        ms = int((time.monotonic() - started) * 1000)
        languages = {rec.language for rec in entries.values()}
        symbols = sum(len(rec.symbols) for rec in entries.values())
        if changed:
            # First (re)build in this process — at least one real parse
            # happened (a disk-served warm load is NOT a build): queue
            # the ONE service event (§8.3) for the next code_* tool call
            # to consume.
            _BUILD_NOTICES[cache_key] = (
                f"code index: {len(entries)} files, {symbols} symbols, "
                f"{len(languages)} languages, {ms} ms"
            )
        return entries, BuildInfo(
            files=len(entries),
            symbols=symbols,
            languages=len(languages),
            ms=ms,
            built_now=changed,
        )
