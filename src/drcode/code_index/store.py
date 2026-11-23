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
— worktree agents only read it (§7.3: `load_index(root, persist=<storage
root>)` loads it as a READ-ONLY base layer under the worktree's stat
overlay); their own edits reach queries through the stat keys (an edited
file's stat pair no longer matches and the file re-parses into the
process cache — §6.2's stat overlay). A fresh `git worktree add`
checkout rewrites every file with fresh mtimes, so base reuse there
rides on the entries' content hashes (§7.3's identity fallback): same
sha256 of the decoded text ⇒ same parse ⇒ the base record is adopted
without a tree-sitter pass.

BLOCKING file I/O: callers on an event loop must run `load_index` in a
worker thread (`tool_runtime._run_tool` already does).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
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
    ".cache",
}

# Glob patterns for directory names that look like build/dependency
# trees (matched case-insensitively against each path segment). A
# general mechanism, not a per-name blacklist: PyTorch-style build
# forests (`build-torch214`, `cmake-build-debug`, `bazel-bin`, …) each
# hold 10⁵ files of vendor/compile junk that would otherwise inflate
# the index by orders of magnitude. grep_search keeps the coverage for
# anything pruned here.
_IGNORED_DIR_GLOBS = (
    "build*",  # build-torch214, BuildOutput, build-linux
    "_build",
    "cmake-*",  # cmake-build-debug, cmake-build-release-clang
    "bazel-*",
    "buck-out",
    "out",
    "output",
    "obj",
    "objs",
    "target",  # cargo/maven convention
    "deps",
    "_deps",  # CMake FetchContent
    "vendor",
    "third_party",
    "thirdparty",
    "3rdparty",
    "external",
    "extern",
    "pods",
    "deriveddata",
    "*.dSYM",
)

# Scale fuses for degenerate projects. MAX_INDEX_FILES caps the total
# entry count: a candidate file set beyond it is not indexed at all
# (grep_search keeps the coverage) — parsing and persisting that many
# records would make every code_* call a minutes-long freeze. The cap
# is generous real-project headroom over the 10⁴-scale a healthy index
# has after the build-tree pruning above; when it fires, a service
# event says so (§8.3 mechanism).
MAX_INDEX_FILES = 50_000
# A persisted index larger than this is not even parsed at cold start
# (a legacy/polluted file of that size costs ~a minute of json.loads
# alone) — treated as unusable, the project re-indexes from scratch.
_MAX_DISK_INDEX_BYTES = 128 * 1024 * 1024

# The file discovery (walk + stat) is cached per root between calls.
# Small/medium candidate sets (≤ _RESCAN_FILE_LIMIT) are re-discovered
# on EVERY call — the stat keys must be fresh for the edit→query
# self-heal flow, and a few thousand stats stay milliseconds. Beyond
# the limit a re-walk costs seconds-to-minutes, so the cached listing
# (paths AND stat keys) is reused until _DISCOVERY_TTL passes — a new
# or edited file becomes visible within that window, and the query-
# level reparse_entry self-heal covers proven-stale records regardless.
_RESCAN_FILE_LIMIT = 20_000
_DISCOVERY_TTL_SECONDS = 30.0

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


def _text_sha(text: str) -> str:
    """The content-identity hash of a file's decoded text (§7.3): the
    exact text the parser would see, so equal hashes ⇒ equal parse."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base_record_by_content(
    path: Path, base_record_raw: object
) -> tuple[FileIndex | None, str | None]:
    """§7.3's identity fallback for the worktree base layer: adopt the
    base checkout's record when the worktree file's text hashes to the
    sha256 the base entry persisted. Needed because a fresh
    `git worktree add` checkout rewrites every tracked file with a fresh
    mtime — the `(mtime_ns, size)` stat key can never match — while the
    CONTENT of the files the task has not touched is identical. Content
    identity is also the stronger guarantee (identical text ⇒ identical
    parse), so a match is never stale.

    Returns `(record, text)`: the adopted record (None when identity is
    not proven) and the file's decoded text (None when unreadable) — the
    caller reuses the read for the fallback parse instead of re-reading."""
    if not isinstance(base_record_raw, dict):
        return None, None
    sha = base_record_raw.get("sha")
    if not isinstance(sha, str) or not sha:
        return None, None  # Legacy entry without a hash: parse as usual.
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    if _text_sha(text) != sha:
        return None, text  # Genuinely changed: the caller parses `text`.
    return file_index_from_disk(base_record_raw), text


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
    # rel → sha256 of the file's decoded text, recorded when the file was
    # last parsed (or carried over from the persisted entry). Persisted
    # alongside the record so a worktree's base layer can prove content
    # identity without a parse (§7.3); never lives in FileIndex — the
    # parse record stays a pure parse result.
    shas: dict[str, str] = field(default_factory=dict)


# One cache per project root, one module-level lock (task_search's
# `_ARCHIVE_LOCK` pattern): the tools run in worker threads, the TUI and
# the schema estimation may touch the store from others.
_INDEX_LOCK = threading.Lock()
_INDEX_CACHES: dict[str, _CachedIndex] = {}


@dataclass
class _DiscoveryCache:
    """The cached file listing of one root: (path, relpath, stat key)
    triples plus the overflow flag and the build time (TTL anchor)."""

    entries: list[tuple[Path, str, tuple[int, int]]]
    overflow: bool
    at: float


_DISCOVERY_LOCK = threading.Lock()
_DISCOVERY_CACHES: dict[str, _DiscoveryCache] = {}
# Roots whose overflow service-event was already queued this process —
# the event is once-per-process, not once-per-TTL-window.
_OVERFLOW_NOTIFIED: set[str] = set()

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
            _OVERFLOW_NOTIFIED.clear()
        else:
            key = str(Path(project_root))
            _INDEX_CACHES.pop(key, None)
            _BUILD_NOTICES.pop(key, None)
            _OVERFLOW_NOTIFIED.discard(key)
    with _DISCOVERY_LOCK:
        if project_root is None:
            _DISCOVERY_CACHES.clear()
        else:
            _DISCOVERY_CACHES.pop(str(Path(project_root)), None)


def take_build_notice(project_root: Path) -> str | None:
    """Consume (once) the pending first-build service-event line of the
    project, if the index was (re)built in this process since the last
    consumption. Cheap dict access — safe to call on the event loop."""
    return _BUILD_NOTICES.pop(str(Path(project_root)), None)


def reparse_entry(
    project_root: Path, rel: str, *, persist: bool | Path = True
) -> FileIndex | None:
    """Re-parse ONE file unconditionally and refresh its record in the
    cache (and on disk when persisting) — the query-level self-heal
    (§6.4): a query that PROVED its record stale (a body range beyond
    the file, a vanished symbol) must not leave the wrong entry in
    place. `persist` is `load_index`'s base-layer selector (True — own
    disk; a Path — the read-only worktree mode, cache-only update;
    False — no disk at all). Returns the fresh record (None when the
    file vanished or its language is not indexable)."""
    root = Path(project_root)
    path = root / rel
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        record = None
        sha: str | None = None
    else:
        record = _extract_record(path, text)
        sha = _text_sha(text)
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
                cached.shas.pop(rel, None)
            else:
                cached.entries[rel] = record
                if sha is not None:
                    cached.shas[rel] = sha
            cached.file_keys[rel] = key
        if persist is True and key is not None:
            disk, _head = _load_disk_index(root)
            payload: dict[str, object] | None = (
                file_index_to_disk(record) if record is not None else None
            )
            if payload is not None and sha is not None:
                payload["sha"] = sha
            disk[rel] = [list(key), payload]
            _write_disk_index(root, disk, read_main_head(root))
    return record


def _is_ignored_dir(name: str) -> bool:
    """A directory that is never descended into: the explicit
    `_IGNORED_DIRS` set plus the case-insensitive build/dependency glob
    patterns (`build-torch214` → `build*`)."""
    if name in _IGNORED_DIRS:
        return True
    lowered = name.lower()
    return any(fnmatch(lowered, pattern) for pattern in _IGNORED_DIR_GLOBS)


def _load_disk_index(project_root: Path) -> tuple[dict[str, list[Any]], str | None]:
    """The persisted index as a raw {relpath: [key, record]} dict plus the
    HEAD sentinel it was built under, or ({}, None) on any problem
    (missing, corrupted, unexpected shape, or a degenerate oversized
    legacy index) — the caller then re-parses the project and rewrites
    the file. An oversized file is not even read: json.loads of a
    multi-hundred-megabyte document costs ~a minute, and a healthy
    pruned/capped index cannot reach that size in the first place."""
    path = _index_path(project_root)
    try:
        if path.stat().st_size > _MAX_DISK_INDEX_BYTES:
            return {}, None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}, None
    if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
        return {}, None
    if data.get("version") != INDEX_VERSION:
        return {}, None
    if len(data["entries"]) > MAX_INDEX_FILES:
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


def _discover_files(
    root: Path, *, limit: int | None = None
) -> tuple[list[tuple[Path, str, tuple[int, int]]], bool]:
    """Every candidate source file under the project root (known
    extension — tree-sitter's table OR the ctags fallback's, size cap,
    ignored dirs pruned by name AND by build-tree glob patterns) as
    `(path, relpath, (mtime_ns, size))` triples — os.scandir with
    pruning, so a 10²–10³-file project stays milliseconds. Symlinked
    directories are never descended into.

    Returns the triples plus an overflow flag: when more than `limit`
    candidates are found the walk stops early (the caller refuses to
    index a degenerate file set that large)."""
    if limit is None:
        limit = MAX_INDEX_FILES  # Late-bound: tests monkeypatch the cap.
    found: list[tuple[Path, str, tuple[int, int]]] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            scanner = os.scandir(directory)
        except OSError:
            continue
        with scanner:
            for entry in scanner:
                name = entry.name
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if not _is_ignored_dir(name):
                            stack.append(Path(entry.path))
                        continue
                except OSError:
                    pass  # Treated as a file below; its stat decides.
                path = Path(entry.path)
                if (
                    _DEFAULT_EXTRACTOR.language_for(path) is None
                    and _FALLBACK_EXTRACTOR.language_for(path) is None
                ):
                    continue
                try:
                    stat = entry.stat()
                except OSError:
                    continue  # Vanished mid-scan: skip silently.
                if stat.st_size > MAX_INDEXED_FILE_BYTES:
                    continue
                if len(found) >= limit:
                    return found, True
                rel = path.relative_to(root).as_posix()
                found.append((path, rel, (stat.st_mtime_ns, stat.st_size)))
    return found, False


def _discovery_for(root: Path) -> tuple[list[tuple[Path, str, tuple[int, int]]], bool]:
    """`_discover_files` through the per-root cache. Small/medium
    candidate sets are re-discovered on every call (fresh stat keys for
    the edit→query self-heal; still milliseconds); large sets reuse the
    cached listing until `_DISCOVERY_TTL_SECONDS` pass — a full walk
    over a huge tree costs seconds-to-minutes per call otherwise."""
    key = str(root)
    now = time.monotonic()
    with _DISCOVERY_LOCK:
        cached = _DISCOVERY_CACHES.get(key)
        if (
            cached is not None
            and (cached.overflow or len(cached.entries) > _RESCAN_FILE_LIMIT)
            and now - cached.at < _DISCOVERY_TTL_SECONDS
        ):
            return cached.entries, cached.overflow
    entries, overflow = _discover_files(root)
    with _DISCOVERY_LOCK:
        _DISCOVERY_CACHES[key] = _DiscoveryCache(
            entries=entries, overflow=overflow, at=now
        )
    return entries, overflow


def load_index(
    project_root: Path, *, persist: bool | Path = True
) -> tuple[dict[str, FileIndex], BuildInfo]:
    """The project's per-file index records, through the incremental
    cache: every candidate file is stat'ed (cheap), a matching
    `(mtime_ns, size)` entry is reused from the in-memory cache or —
    after a restart — from the persistent on-disk index; everything else
    is re-parsed. Returns the entries plus a BuildInfo for the first
    build's service event (§8.3).

    `persist` is the §7.3 base-layer selector:

    - `True` — the tree being indexed IS the main checkout: the own
      persistent index is read AND written (the historical behavior).
    - a `Path` — the persistent index of THAT root (the main checkout's
      storage root for a worktree task, §7.1) is loaded as a READ-ONLY
      base layer: matching stat keys are adopted without a parse, and a
      stat mismatch falls back to the entry's content hash (a fresh
      `git worktree add` checkout rewrites every file with fresh
      mtimes, so content identity is what makes unchanged files free).
      NOTHING is ever written — not the base file, not a worktree
      `.drcode/` (which does not exist, `_COPY_SKIP_DIRS`). The HEAD
      guard (§6.4) does not apply here: the worktree IS another branch
      by design, the base layer is trusted on stat keys + content
      identity alone.
    - `False` — purely in memory: same cache, same stat invalidation,
      no disk read/write at all (the §6.2 stat overlay, base version).

    In the Path mode the worktree's process-local cache IS the §6.2
    stat overlay over the borrowed base: a file the task edits no
    longer matches base (stat AND hash) and is re-parsed into the
    overlay; a file deleted in the worktree simply drops out; a file
    added there is parsed in.

    A stale/corrupted entry self-heals by re-parsing; a file that
    vanished since the last build simply drops out of the entries. The
    on-disk index is rewritten only when something actually changed (a
    re-parse, a vanished file, a broken entry) — the common warm call is
    a pure read.

    Scale fuses (a project with 10⁵+ files must not turn every call
    into a minutes-long freeze): build/dependency trees are pruned at
    discovery; a candidate file set beyond MAX_INDEX_FILES is not
    indexed at all (an empty index + a one-per-process service event;
    grep_search keeps the coverage); an oversized persisted index is
    not even parsed — it re-indexes from scratch."""
    root = Path(project_root)
    started = time.monotonic()
    # Fast path outside the lock: stat everything once (task_search's
    # load_archive does the same — no reason to serialize the walk).
    stat_entries, overflow = _discovery_for(root)
    cache_key = str(root)
    if overflow:
        # Degenerate file set (unpruned build/vendor forests at 10⁵+
        # scale): refuse to index, serve empty — parsing and persisting
        # that many records would freeze every call. One service event
        # per process says so; the entries dict stays honest.
        with _INDEX_LOCK:
            _INDEX_CACHES.pop(cache_key, None)
            if cache_key not in _OVERFLOW_NOTIFIED:
                _OVERFLOW_NOTIFIED.add(cache_key)
                _BUILD_NOTICES[cache_key] = (
                    f"code index: skipped — more than {MAX_INDEX_FILES} "
                    "candidate source files (build/dependency trees?); "
                    "grep_search keeps the coverage"
                )
        return {}, BuildInfo(
            files=0,
            symbols=0,
            languages=0,
            ms=int((time.monotonic() - started) * 1000),
            built_now=False,
        )
    with _INDEX_LOCK:
        cached = _INDEX_CACHES.get(cache_key)
        if not isinstance(cached, _CachedIndex):
            cached = None  # Broken/foreign entry: rebuild from scratch.
        old_keys = cached.file_keys if cached else {}
        old_shas = cached.shas if cached else {}
        # §7.3: the base layer to borrow. True = own persistent index
        # (read AND write); a Path = another checkout's persisted index,
        # READ-ONLY (the worktree case); False = no disk at all.
        base_root = persist if isinstance(persist, Path) else None
        persisting = persist is True
        disk: dict[str, list[Any]] = {}
        persisted_head: str | None = None
        if persisting:
            disk, persisted_head = _load_disk_index(root)
        elif base_root is not None:
            # A worktree task borrows the MAIN checkout's persisted
            # index as its base layer (§7.1: the path comes from the
            # storage root, never from the worktree's own `.drcode/`,
            # which must stay copy-skip clean). Stat keys + content
            # hashes decide what is re-parsed; nothing is written.
            disk, _base_head = _load_disk_index(base_root)
        # §6.4 HEAD guard: the index was built on another branch of the
        # main checkout — the base layer is wholesale suspect, so it is
        # dropped and rebuilt (stat keys alone cannot see a branch switch
        # that happened while this process was down). An optimization in
        # the design; the simple full-drop is the honest version of it.
        # Deliberately OWN-ROOT only: a worktree task loading the main
        # checkout's index IS the other-branch case (§7.3) — the base is
        # trusted there on stat keys + content identity.
        current_head = read_main_head(root) if persisting else None
        head_guard_fired = bool(
            persisting
            and persisted_head
            and current_head
            and persisted_head != current_head
        )
        if head_guard_fired:
            disk = {}
        entries: dict[str, FileIndex] = {}
        file_keys: dict[str, tuple[int, int]] = {}
        shas: dict[str, str] = {}
        disk_dirty = False
        changed = False
        for path, rel, key in stat_entries:
            record: FileIndex | None = None
            text: str | None = None  # The file's text, when already read.
            if old_keys.get(rel) == key and cached is not None:
                record = cached.entries.get(rel)
                sha = old_shas.get(rel)
                if record is not None and sha is not None:
                    shas[rel] = sha
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
                    elif isinstance(raw[1], dict) and isinstance(
                        raw[1].get("sha"), str
                    ):
                        shas[rel] = raw[1]["sha"]
                elif base_root is not None and isinstance(raw, list) and len(raw) == 2:
                    # The worktree base layer beyond stat keys: a fresh
                    # `git worktree add` checkout rewrote every file with
                    # fresh mtimes, so the stat key cannot match even for
                    # content the task never touched — prove identity by
                    # the persisted content hash instead (§7.3).
                    record, text = _base_record_by_content(path, raw[1])
            if record is None:
                if text is None:
                    try:
                        text = path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue  # Unreadable: skip (the index stays honest).
                record = _extract_record(path, text)
                changed = True
                if persisting:
                    shas[rel] = _text_sha(text)
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
        fresh = _CachedIndex(entries=entries, file_keys=file_keys, shas=shas)
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
        if persisting and (changed or disk_dirty or stale_on_disk or head_guard_fired):
            serialized: dict[str, list[object]] = {}
            for rel, key in file_keys.items():
                record = entries.get(rel)
                payload: dict[str, object] | None = (
                    file_index_to_disk(record) if record is not None else None
                )
                if payload is not None and rel in shas:
                    payload["sha"] = shas[rel]
                serialized[rel] = [list(key), payload]
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
