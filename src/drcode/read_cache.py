"""Per-run read_file deduplication: no full re-read of an unchanged file.

On long runs with active edits the model re-reads the same paths over and
over, and every full re-read result (up to MAX_READ_RESULT_BYTES) rides
EVERY subsequent request of the run until eviction — the largest single
untapped input-token saving (goal #c6ee24e4). A `ReadCache` maps each
resolved path to the last read's outcome; `read_file_cached` consults it:

- file unchanged (sha256 of the raw content — never mtime alone) and the
  same slice requested → a short "unchanged since read #N" note instead of
  the full text;
- file changed and the same slice requested → a unified diff against the
  previous read, but only when the diff is meaningfully smaller than the
  full text; otherwise the full text as before;
- anything else (different slice, no entry) → the full text as before.

The cache lives ONE run: it is created in `_prepare_agent_run` next to the
gates, shared across the per-turn deps rebuilds and a mid-run compaction
pause, and dies with the run closure — no orphans to sweep.

Safety: a stub claims "the content is already in your context", which is
only true while the previous read's tool result is still a live part of
the transcript. Before trusting an entry, `_entry_is_live` therefore scans
the task's serialized transcript for that read's tool_call_id and refuses
anything that is missing or already an eviction stub (`[tool result
evicted:` — a mark_unneeded collapse, an auto-eviction pass or a full
compaction all end here). A dead entry is dropped and the full text is
returned — the divergence guard is fail-open to the complete content, so
the cache can never cost the model information, only save it."""

from __future__ import annotations

import difflib
import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import tools
from .eviction import is_evicted_stub

# Return the diff only when it is meaningfully smaller than the full text;
# a diff approaching the full size saves nothing and is harder to read.
DIFF_MAX_SHARE = 0.8


@dataclass
class ReadCacheEntry:
    """One path's last read within a run.

    `content_sha` is the sha256 of the RAW file content at read time —
    the change test is byte-exact, not an mtime guess (a same-size write
    with a forged timestamp can never produce a false "unchanged").
    `result_text` is the exact numbered text the model received, so a
    later diff is against what the context actually holds. `tool_call_id`
    identifies the transcript part that carries `result_text` — the stub
    gate verifies it is still live before claiming "already in your
    context"."""

    content_sha: str
    result_text: str
    offset: int
    limit: int
    truncated: bool
    tool_call_id: str
    read_no: int


@dataclass
class ReadCache:
    """Per-run map of resolved path → last read. Plain in-memory state:
    nothing persists to disk (the transcript it must stay consistent with
    is itself rewritten by compaction, so persisting would only create
    stale claims). Created once per run in `_prepare_agent_run` and passed
    through every per-turn deps rebuild; a run restart (network retry,
    watchdog) simply starts a fresh cache."""

    entries: dict[str, ReadCacheEntry] = field(default_factory=dict)
    reads: int = 0

    def get(self, key: str) -> ReadCacheEntry | None:
        return self.entries.get(key)

    def drop(self, key: str) -> None:
        """Forget a path — its last read is no longer trustworthy (the
        transcript no longer carries the result verbatim)."""
        self.entries.pop(key, None)

    def record(
        self,
        key: str,
        *,
        raw: str,
        result_text: str,
        offset: int,
        limit: int,
        truncated: bool,
        tool_call_id: str | None,
    ) -> ReadCacheEntry:
        self.reads += 1
        entry = ReadCacheEntry(
            content_sha=_sha(raw),
            result_text=result_text,
            offset=offset,
            limit=limit,
            truncated=truncated,
            tool_call_id=tool_call_id or "",
            read_no=self.reads,
        )
        self.entries[key] = entry
        return entry


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entry_is_live(task_provider: Callable[[], Any] | None, tool_call_id: str) -> bool:
    """True when the transcript still holds the previous read's tool
    result verbatim (present, and not collapsed to an eviction stub).

    Fail-open to False everywhere: no task, a vanished task, any
    surprise — the caller then returns the full text instead of the
    stub, so a wrong True is the only dangerous direction and it requires
    a matching live tool-return to happen."""

    if not tool_call_id or task_provider is None:
        return False
    try:
        task = task_provider()
        messages = task.messages if task is not None else None
    except Exception:  # noqa: BLE001 - any surprise degrades to the full text
        return False
    for msg in messages or []:
        for part in msg.get("parts", []):
            if (
                isinstance(part, dict)
                and part.get("part_kind") == "tool-return"
                and part.get("tool_call_id") == tool_call_id
            ):
                return not is_evicted_stub(part.get("content"))
    return False


def _unified_diff(old: str, new: str, path: str) -> str:
    """Unified diff between the previous and the current numbered read of
    `path` (same slice requested). Empty string when the slices are
    identical — the file changed OUTSIDE the slice (the sha covers the
    whole file), so the content in the context is still current."""
    diff_lines = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=f"{path} (previous read)",
        tofile=f"{path} (current)",
        lineterm="",
    )
    return "\n".join(diff_lines)


def read_file_cached(
    cache: ReadCache | None,
    root: Path,
    path: str,
    offset: int = 1,
    limit: int = 2000,
    tool_call_id: str | None = None,
    *,
    task_provider: Callable[[], Any] | None = None,
    current_usage: int = tools.DEFAULT_CURRENT_USAGE,
    max_context_tokens: int = tools.DEFAULT_MAX_CONTEXT_TOKENS,
    compaction_threshold: int = tools.DEFAULT_COMPACTION_THRESHOLD,
) -> str:
    """read_file with per-run deduplication (see the module docstring).

    Drop-in for tools.read_file in the coding/tester/resolver tool
    wrappers: same signature plus the cache, the current call's
    tool_call_id and a task provider for the transcript liveness check.
    cache=None or a refused read degrades to the plain tools.read_file
    behavior with no cache bookkeeping."""

    p = tools.resolve_read_path(root, path)
    key = str(p)
    previous = cache.get(key) if cache is not None else None
    # The raw content is read either way: the change test is the sha of
    # the actual bytes, so a stub never rests on an mtime coincidence —
    # and a deleted/unreadable file still surfaces the normal OSError.
    raw = p.read_text(errors="replace")
    digest = _sha(raw)
    previous_live = previous is not None and _entry_is_live(
        task_provider, previous.tool_call_id
    )
    if previous is not None:
        if (
            previous_live
            and previous.content_sha == digest
            and previous.offset == offset
            and previous.limit == limit
        ):
            return (
                f"[read_file: '{path}' is unchanged since read "
                f"#{previous.read_no} — the numbered content from that read "
                f"is already in your context, do not spend tokens on it "
                f"again. To read a different slice, adjust `offset`/`limit`.]"
            )
        if not previous_live and cache is not None:
            # The claim "already in your context" no longer holds (the
            # result was evicted or the transcript rewritten) — drop the
            # stale entry and fall through to the full read.
            cache.drop(key)
    result = tools._read_file_text(
        p,
        path,
        offset,
        limit,
        current_usage=current_usage,
        max_context_tokens=max_context_tokens,
        compaction_threshold=compaction_threshold,
    )
    if result.text.startswith(tools.READ_REFUSAL_PREFIX):
        # Budget refusal: nothing was delivered, nothing to remember.
        return result.text
    if cache is not None:
        cache.record(
            key,
            raw=raw,
            result_text=result.text,
            offset=offset,
            limit=limit,
            truncated=result.truncated,
            tool_call_id=tool_call_id,
        )
        # A diff is only honest against a previous read the context still
        # holds verbatim, and only between two non-truncated slices of the
        # same window (a truncated slice is a head+tail cut, not a file
        # state — diffing those would fabricate a middle).
        if (
            previous is not None
            and previous_live
            and not previous.truncated
            and not result.truncated
            and previous.offset == offset
            and previous.limit == limit
        ):
            diff_text = _unified_diff(previous.result_text, result.text, path)
            if not diff_text:
                # The file changed outside the requested slice: the
                # numbered content in the context is still current.
                return (
                    f"[read_file: '{path}' changed since read "
                    f"#{previous.read_no}, but not within the slice of your "
                    f"last read — the numbered content in your context is "
                    f"still current.]"
                )
            if len(diff_text.encode("utf-8")) < int(
                len(result.text.encode("utf-8")) * DIFF_MAX_SHARE
            ):
                header = (
                    f"[read_file: '{path}' changed since read "
                    f"#{previous.read_no} — unified diff against the "
                    f"previous read; unchanged lines are already in your "
                    f"context]"
                )
                return f"{header}\n{diff_text}"
    return result.text
