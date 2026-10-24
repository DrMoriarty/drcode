"""Read-only keyword search over the project's task archive.

The research agent needs to assess the state of development around a topic —
across ALL statuses (queued, hold, active, testing, completed, clarification)
— without reading the raw task JSONs: a task file is `Task.model_dump_json`
with the full transcript (`messages`, `tester_messages`, `events`), and
reading several of those burns tens of thousands of tokens. The plain file
tools cannot help either: `.drcode/` is in `tools.IGNORED_DIRS`, so
grep/glob never see it, and `run_bash` + `grep`/`jq` is a fragile workaround
(jq may be absent, the JSON is multi-line).

This module gives the model a deliberately COMPACT view instead:

- a light parse — `json.loads` plus a hand-picked subset of keys, NEVER
  `Task.model_validate` (validating giant `messages` lists for a search is
  pure waste);
- indexing only the user-authored text: `title`, every `user-prompt` part of
  the transcript and `pending_input` texts. Model responses and tool results
  are NOT indexed — noise and volume, not signal;
- one output line per task: id, status, kind, date, title and a ±60-char
  snippet around the first keyword match (snippet capped at ~120 chars);
  never a transcript excerpt;
- resilience follows the `Storage.load_one` contract (`storage.py`): a
  corrupted or unreadable file is silently skipped, never fatal. Races need
  no handling — `Storage.save` is atomic (`os.replace`), the worst case is
  missing one task created mid-scan.

Worktree isolation note: under `task_isolation: worktree` the task files
live in the MAIN checkout's `.drcode/tasks/`, not in the task's worktree —
callers must pass the main project root (the pipeline's `manager.storage.root`),
never the agent's cwd/worktree root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PROJECT_STATE_DIRNAME
from .models import TaskKind, TaskStatus

# Output budget: the tool exists to SAVE context, so the result is hard-capped.
# No more than MAX_RESULTS lines per call; when the archive holds more
# matches, the first MAX_RESULTS are returned plus a "...and N more" tail
# telling the model to refine the query.
MAX_RESULTS = 50
# Snippet geometry: ±SNIPPET_CONTEXT characters around the first keyword
# occurrence, hard-capped at SNIPPET_MAX characters.
SNIPPET_CONTEXT = 60
SNIPPET_MAX = 120


def _validate_choice(value: str | None, choices: type, name: str) -> str | None:
    """Validate an optional status/kind filter against its enum.

    Raises ValueError (→ ModelRetry via _run_tool) for an unknown value so
    the model can correct itself instead of silently searching unfiltered.
    """
    if not value:
        return None
    allowed = [member.value for member in choices]  # type: ignore[attr-defined]
    if value not in allowed:
        raise ValueError(
            f"Unknown {name} filter {value!r}; allowed values: "
            + ", ".join(allowed)
            + " (omit the argument to search all of them)."
        )
    return value


def _prompt_texts(messages: list[dict[str, Any]]) -> list[str]:
    """The user-authored texts of a serialized transcript.

    Serialized `ModelRequest`s carry `parts`, and only the parts with
    `part_kind == "user-prompt"` are indexed. Content is usually a plain
    string; multimodal requests carry a list of content items (strings or
    {"text": ...} dicts) — their text is joined. Model responses, tool calls
    and tool results are deliberately ignored.
    """
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        for part in message.get("parts") or []:
            if not isinstance(part, dict):
                continue
            if part.get("part_kind") != "user-prompt":
                continue
            content = part.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        texts.append(item["text"])
    return texts


def _pending_texts(pending_input: list[Any]) -> list[str]:
    """`pending_input` texts (queued user messages not yet consumed).

    Tolerates both shapes the storage can hold: current `PendingInput`
    dicts and the legacy plain-string entries (see Task's migration).
    """
    texts: list[str] = []
    for entry in pending_input:
        if isinstance(entry, str):
            texts.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("text"), str):
            texts.append(entry["text"])
    return texts


def _snippet(text: str, keyword: str) -> str:
    """±SNIPPET_CONTEXT chars around the first (case-insensitive) keyword
    occurrence, whitespace-collapsed and hard-capped at SNIPPET_MAX chars."""
    flat = " ".join(text.split())
    lower = flat.lower()
    pos = lower.find(keyword)
    start = max(0, pos - SNIPPET_CONTEXT)
    end = min(len(flat), pos + len(keyword) + SNIPPET_CONTEXT)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(flat) else ""
    return (prefix + flat[start:end] + suffix)[:SNIPPET_MAX]


def _date(data: dict[str, Any]) -> str:
    """The task's most relevant date: completion when it has one, else
    creation — as a bare ISO date (`YYYY-MM-DD`)."""
    stamp = data.get("completed_at") or data.get("created_at") or ""
    return str(stamp)[:10]


def search(
    project_root: Path,
    keywords: list[str],
    *,
    exclude_task_id: str | None = None,
    status: str | None = None,
    kind: str | None = None,
    limit: int = MAX_RESULTS,
) -> str:
    """Search the project's task archive by keywords (all statuses).

    Matching is a case-insensitive substring over `title` + user prompts +
    pending input; multiple keywords are combined with AND (every keyword
    must occur). Optional filters: `status` (queued/hold/active/testing/
    completed/clarification) and `kind` (bug/feature/research). Returns at
    most `limit` (default MAX_RESULTS = 50) one-line results; a longer hit
    list is truncated with a "refine the query" tail. Corrupted task files
    are skipped silently. The caller's own task (`exclude_task_id`) is
    excluded.
    """
    cleaned = [k.strip().lower() for k in keywords if k and k.strip()]
    if not cleaned:
        raise ValueError(
            "search_task_history needs at least one non-empty keyword; "
            "pass the words you expect in a task's title or its user prompt."
        )
    status = _validate_choice(status, TaskStatus, "status")
    kind = _validate_choice(kind, TaskKind, "kind")
    tasks_dir = Path(project_root) / PROJECT_STATE_DIRNAME / "tasks"
    try:
        paths = sorted(tasks_dir.glob("*.json"))
    except OSError:
        return "Task archive is not readable — no results."
    lines: list[str] = []
    total = 0
    for path in paths:
        if exclude_task_id is not None and path.stem == exclude_task_id:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue  # Corrupted/unreadable: skip silently (load_one contract).
        if not isinstance(data, dict):
            continue
        if status is not None and data.get("status") != status:
            continue
        if kind is not None and data.get("kind") != kind:
            continue
        texts = (
            [str(data.get("title") or "")]
            + _prompt_texts(data.get("messages") or [])
            + _pending_texts(data.get("pending_input") or [])
        )
        haystack = "\n".join(texts).lower()
        if any(k not in haystack for k in cleaned):
            continue
        total += 1
        if len(lines) < limit:
            title = " ".join(str(data.get("title") or "").split())
            # Snippet source: the same indexed texts that were matched (the
            # match may live in a prompt or pending input, not the title).
            body = "\n".join(texts)
            first = next(k for k in cleaned if k in body.lower())
            snippet = _snippet(body, first)
            # A long title is as much a context hazard as a long snippet —
            # cap it to keep every line within the compact-output budget.
            if len(title) > SNIPPET_MAX:
                title = title[:SNIPPET_MAX] + "…"
            lines.append(
                f"#{path.stem} [{data.get('status')}/{data.get('kind')}] "
                f"{_date(data)} {title} — {snippet}"
            )
    if not lines:
        return (
            f"No tasks matched: {', '.join(cleaned)}"
            + (f" (status={status})" if status else "")
            + (f" (kind={kind})" if kind else "")
            + "."
        )
    if total > len(lines):
        lines.append(f"...and {total - len(lines)} more — refine the query.")
    return "\n".join(lines)
