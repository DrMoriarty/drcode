"""Read-only views of the project's state (design doc §7, the "read" block).

This module is the ONE place where task/goal/stat data is turned into
text. The MCP tools (`tools_read.py`) and the MCP resources
(`resources.py`) both call it, so a subscriber refetching a resource after
`notifications/resources/updated` and a poller calling the matching tool
receive byte-identical content — the subscription story and the polling
story cannot drift (§9: "агент без подписок ничего не теряет").

Hard rule of the whole layer: **cheap metadata only**. Nothing here ever
touches `task.messages` / `tester_messages` / `resolver_messages` — those
are megabyte sidecar files, and a read tool that parsed them would burn
more context than the work it reports on. The one transcript-derived field
exposed is `task.skeleton` (the user prompts + the final agent response,
already persisted in the metadata file), and only `get_task` returns it.

Blocking I/O: the archive search and the usage/stats reads touch disk. The
SDK runs sync handlers in a worker thread, so registering these bodies
directly is correct; async callers must hop through `asyncio.to_thread`.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .. import task_search
from ..models import Goal, Task, TaskKind, TaskStatus, VerificationStatus
from ..stats import DayStats, format_seconds
from ..task_manager import TaskManager
from ..task_overlap import (
    build_overlap_warning,
    extract_path_tokens,
    find_file_overlaps,
)
from ..usage_store import FIELDS, UsageStore, usage_file

#: How many compact rows `list_tasks` returns by default, and the hard cap
#: a caller can raise it to (a runaway model must not be able to ask for
#: the whole archive in one line-per-task answer).
DEFAULT_TASK_LIMIT = 50
MAX_TASK_LIMIT = 200

#: Service events `get_task` returns by default / at most. The events list
#: is unbounded on a long-lived task; the tail is what answers "what went
#: wrong last".
DEFAULT_EVENT_LIMIT = 10
MAX_EVENT_LIMIT = 100

#: Default/limit windows of `get_stats` and `get_usage` (days, newest last).
DEFAULT_STATS_DAYS = 14
MAX_DAYS = 366

#: Title cap inside a compact row — a long title is as much of a context
#: hazard as a long snippet.
TITLE_CAP = 80


class ProjectView:
    """Read source of the MCP read tools: the live manager plus the
    project root it belongs to.

    Deliberately a tiny holder, not a service: the tools are pure functions
    over it, so the tests can build one over a `TaskManager` on a tmp dir
    without mounting anything.

    `usage_store` is optional — token usage lives in the USER-level config
    dir, and the reader constructs a throwaway store per call when none was
    injected (the stats dialog's own pattern: read the file fresh, hold no
    state). That store is read-only here; nothing in this module records.
    """

    def __init__(
        self,
        manager: TaskManager,
        *,
        usage_store: UsageStore | None = None,
    ) -> None:
        self.manager = manager
        self._usage_store = usage_store

    @property
    def project_root(self) -> Path:
        """The project root the manager was built over."""
        return self.manager.storage.root

    @property
    def archive_root(self) -> Path:
        """Where the task archive lives: the MAIN checkout's `.drcode/`.

        Under worktree isolation the agent-side cwd is the task's worktree,
        whose `.drcode/` holds no task files — the manager's storage root
        always points at the main project root (the same rule the agent's
        search tool follows).
        """
        return self.manager.storage.root

    def usage(self) -> UsageStore:
        """The usage store to read (injected one, or a fresh read-only one)."""
        if self._usage_store is not None:
            return self._usage_store
        return UsageStore(path=usage_file())


# -- small formatting helpers --------------------------------------------------


def _one_line(text: str, cap: int = TITLE_CAP) -> str:
    """Whitespace-collapsed, capped text — the compact-row discipline."""
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= cap else collapsed[: cap - 1] + "…"


def _cap(value: int, *, default: int, maximum: int) -> int:
    """Clamp a caller-supplied limit into [1, maximum]."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    if number <= 0:
        return default
    return min(number, maximum)


def _iso(value: Any) -> str | None:
    """ISO-8601 of a datetime field, or None when unset."""
    if value is None:
        return None
    try:
        return value.isoformat()
    except AttributeError:  # pragma: no cover - defensive
        return str(value)


def _verification_marker(task: Task) -> str:
    """The compact form of the tester's label (`verification=passed`)."""
    if task.verification is VerificationStatus.PASSED:
        return "verification=passed"
    if task.verification is VerificationStatus.FAILED:
        return "verification=failed"
    return ""


def _task_markers(task: Task) -> list[str]:
    """The compact-row markers of a task, in a fixed order."""
    markers: list[str] = []
    verification = _verification_marker(task)
    if verification:
        markers.append(verification)
    if task.status is TaskStatus.HOLD:
        markers.append("hold")
    if task.unfinished:
        markers.append("unfinished")
    if task.needs_human:
        markers.append("needs_human")
    if task.hook_merge_failed:
        markers.append("hook_merge_failed")
    if task.cancelled:
        markers.append("cancelled")
    if task.github_issue is not None:
        markers.append(f"github={task.github_issue}")
    if task.goal_id:
        markers.append(f"goal={task.goal_id}")
    if task.pending_input:
        markers.append(f"input={len(task.pending_input)}")
    if task.clarification:
        markers.append("clarification")
    if task.depends_on:
        markers.append(f"depends_on={','.join(task.depends_on)}")
    return markers


def format_task_row(task: Task) -> str:
    """One compact row: `#<id> [<status>/<kind>] <title> — <markers>`."""
    markers = _task_markers(task)
    line = f"#{task.id} [{task.status.value}/{task.kind.value}] {_one_line(task.title)}"
    return f"{line} — {' '.join(markers)}" if markers else line


# -- list_tasks ----------------------------------------------------------------


def _parse_status(value: str | None) -> TaskStatus | None:
    """Validate a status filter against `TaskStatus` (None = no filter)."""
    if value is None or value == "" or value == "all":
        return None
    try:
        return TaskStatus(str(value))
    except ValueError:
        allowed = ", ".join(member.value for member in TaskStatus)
        raise ValueError(
            f"unknown status {value!r}; expected one of: {allowed}"
        ) from None


def _parse_kind(value: str | None) -> TaskKind | None:
    """Validate a kind filter against `TaskKind` (None = no filter)."""
    if value is None or value == "" or value == "all":
        return None
    try:
        return TaskKind(str(value))
    except ValueError:
        allowed = ", ".join(member.value for member in TaskKind)
        raise ValueError(
            f"unknown kind {value!r}; expected one of: {allowed}"
        ) from None


def list_tasks(
    view: ProjectView,
    status: str | None = None,
    kind: str | None = None,
    limit: int = DEFAULT_TASK_LIMIT,
) -> str:
    """Compact rows for every task, filtered by status and kind.

    Ordering is `manager.all_tasks()` (creation order, stable and cheap).
    Tasks of every status are included by default — including COMPLETED —
    because "what happened to the task I filed yesterday" is the most
    common external question.
    """
    status_filter = _parse_status(status)
    kind_filter = _parse_kind(kind)
    cap = _cap(limit, default=DEFAULT_TASK_LIMIT, maximum=MAX_TASK_LIMIT)
    tasks = [
        task
        for task in view.manager.all_tasks()
        if (status_filter is None or task.status is status_filter)
        and (kind_filter is None or task.kind is kind_filter)
    ]
    filters = []
    if status_filter is not None:
        filters.append(f"status={status_filter.value}")
    if kind_filter is not None:
        filters.append(f"kind={kind_filter.value}")
    header = f"{len(tasks)} task(s)"
    if filters:
        header += f" ({', '.join(filters)})"
    if not tasks:
        return header + " — nothing matches."
    shown = tasks[:cap]
    lines = [header + ("" if len(tasks) <= cap else f", showing the first {cap}")]
    lines.extend(format_task_row(task) for task in shown)
    if len(tasks) > cap:
        lines.append(f"...and {len(tasks) - cap} more — raise `limit` or filter.")
    return "\n".join(lines)


# -- get_task ------------------------------------------------------------------


def _clarification_payload(task: Task) -> dict[str, Any] | None:
    """The pending clarification package, structurally (never the markup)."""
    clarification = task.clarification
    if not clarification:
        return None
    return {
        "questions": [
            {
                "question": item.question,
                "options": list(item.options),
                "agent_proposals": list(item.agent_proposals),
                "multiple": item.multiple,
            }
            for item in clarification.questions
        ]
    }


def get_task(view: ProjectView, task_id: str, events: int = DEFAULT_EVENT_LIMIT) -> str:
    """Full metadata of one task as JSON — plus its search skeleton.

    Deliberately excludes the three transcripts (they live in a sidecar
    file, possibly megabytes). `skeleton` is the cheap projection the
    archive search indexes: the user's prompts and the agent's FINAL
    response — enough to explain what the task was about without reading a
    single tool result.

    Raises `KeyError` for an unknown id (both callers turn that into their
    protocol's own not-found answer).
    """
    task = view.manager.get(task_id)
    event_cap = _cap(events, default=DEFAULT_EVENT_LIMIT, maximum=MAX_EVENT_LIMIT)
    skeleton = task.skeleton if isinstance(task.skeleton, dict) else {}
    prompts = skeleton.get("prompts")
    payload: dict[str, Any] = {
        "id": task.id,
        "title": task.title,
        "kind": task.kind.value,
        "status": task.status.value,
        "hold": task.status is TaskStatus.HOLD,
        "verification": task.verification.value if task.verification else None,
        "unfinished": task.unfinished,
        "needs_human": task.needs_human,
        "cancelled": task.cancelled,
        "hook_merge_failed": task.hook_merge_failed,
        "review_pending": task.review_pending,
        "github_issue": task.github_issue,
        "goal_id": task.goal_id,
        "depends_on": list(task.depends_on),
        "blocking_dependencies": view.manager.blocking_dependencies(task_id),
        "queue_seq": task.queue_seq,
        "queue_head": task.queue_head,
        "created_at": _iso(task.created_at),
        "updated_at": _iso(task.updated_at),
        "status_changed_at": _iso(task.status_changed_at),
        "activated_at": _iso(task.activated_at),
        "completed_at": _iso(task.completed_at),
        "pending_input": {
            "count": len(task.pending_input),
            "items": [
                {"at": _iso(entry.at), "text": entry.text}
                for entry in task.pending_input
            ],
        },
        "clarification": _clarification_payload(task),
        "todos": [{"text": item.text, "done": item.done} for item in task.todos],
        "edited_files": list(task.edited_files),
        "skeleton": {
            "prompts": [p for p in prompts if isinstance(p, str)]
            if isinstance(prompts, list)
            else [],
            "final_response": skeleton.get("final_response") or "",
        },
        "events": {
            "total": len(task.events),
            "showing": min(event_cap, len(task.events)),
            "items": [
                {"at": _iso(event.at), "kind": event.kind, "text": event.text}
                for event in task.events[-event_cap:]
            ],
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# -- search_tasks --------------------------------------------------------------

_KEYWORD_SPLIT = re.compile(r"[,\s]+")


def parse_keywords(keywords: str | list[str]) -> list[str]:
    """Normalize the tool's keyword argument into `task_search`'s list.

    Accepts one space/comma-separated string (what an LLM naturally sends)
    or an explicit list; empties are dropped. Raises ValueError when
    nothing usable is left — the MCP `isError` answer tells the caller what
    to fix, right before the search would have scanned the whole archive
    for nothing.
    """
    if isinstance(keywords, str):
        raw = _KEYWORD_SPLIT.split(keywords)
    else:
        raw = [part for item in keywords for part in _KEYWORD_SPLIT.split(item)]
    cleaned = [part.strip() for part in raw if part and part.strip()]
    if not cleaned:
        raise ValueError("keywords must contain at least one non-empty word")
    return cleaned


def search_tasks(
    view: ProjectView,
    keywords: str | list[str],
    status: str | None = None,
    kind: str | None = None,
    limit: int = task_search.MAX_RESULTS,
) -> str:
    """Keyword search over the task archive — `task_search`'s own semantics.

    AND first, OR fallback under an explicit preamble, one compact line per
    hit (id, status/kind, verification, date, title, snippet), capped at
    `task_search.MAX_RESULTS` (50) rows. Reused wholesale rather than
    reimplemented: the archive index, the stemming heuristic and the
    "refine the query" tail are all battle-tested there.
    """
    cleaned = parse_keywords(keywords)
    cap = min(max(int(limit), 1), task_search.MAX_RESULTS)
    return task_search.search(
        view.archive_root,
        cleaned,
        status=status,
        kind=kind,
        limit=cap,
    )


# -- get_queue -----------------------------------------------------------------


def get_queue(view: ProjectView) -> str:
    """The queue in promotion order, with each entry's blockers.

    `manager.queued()` is the authoritative order (dependency sinks first,
    bugs newest-first, answered clarifications, then the queue stamp) — the
    same list the sidebar renders. Blockers are the STATUS half of the
    dependency gate (a dep that exists and is not COMPLETED yet); the git
    mergedness half is only checked at promotion time, so an entry may
    still be parked by an unmerged dependency branch after every status
    blocker is gone.
    """
    queue = view.manager.queued()
    active = view.manager.active_tasks()
    lines: list[str] = []
    active_text = (
        "active: " + ", ".join(f"#{t.id}" for t in active) if active else "active: none"
    )
    lines.append(f"queue: {len(queue)} task(s); {active_text}")
    if not queue:
        lines.append("(empty — nothing waits for development)")
        return "\n".join(lines)
    for position, task in enumerate(queue, 1):
        blockers = view.manager.blocking_dependencies(task.id)
        row = f"{position}. {format_task_row(task)}"
        if blockers:
            row += f" — blocked by {', '.join(f'#{dep}' for dep in blockers)}"
        lines.append(row)
    return "\n".join(lines)


# -- goals ---------------------------------------------------------------------


def _goal_payload(view: ProjectView, goal: Goal) -> dict[str, Any]:
    """The JSON body of one goal: roadmap + lifecycle + live progress."""
    done, total = view.manager.goal_progress(goal)
    linked = [task for task in view.manager.all_tasks() if task.goal_id == goal.id]
    return {
        "id": goal.id,
        "title": goal.title,
        "type": goal.type.value,
        "status": goal.status.value,
        "description": goal.description,
        "notes": goal.notes,
        "parent_id": goal.parent_id,
        "children": sorted(view.manager.goals.descendant_ids(goal.id)),
        "progress": {"done": done, "total": total},
        "complete": view.manager.goal_complete(goal),
        "created_at": _iso(goal.created_at),
        "updated_at": _iso(goal.updated_at),
        "tasks": [
            {
                "id": task.id,
                "status": task.status.value,
                "kind": task.kind.value,
                "title": task.title,
            }
            for task in linked
        ],
    }


def list_goals(view: ProjectView) -> str:
    """Every goal as one compact row, with its live done/total progress."""
    goals = list(view.manager.goals.goals.values())
    if not goals:
        return "goals: 0 (the registry is empty)"
    goals.sort(key=lambda g: (g.type.value, g.title.lower()))
    lines = [f"goals: {len(goals)}"]
    for goal in goals:
        done, total = view.manager.goal_progress(goal)
        progress = f"{done}/{total} tasks" if total else "no tasks"
        parent = f" parent={goal.parent_id}" if goal.parent_id else ""
        complete = " complete" if view.manager.goal_complete(goal) else ""
        lines.append(
            f"#{goal.id} [{goal.type.value}/{goal.status.value}]{complete}"
            f"{parent} {_one_line(goal.title)} — {progress}"
        )
    return "\n".join(lines)


def get_goal(view: ProjectView, goal_id: str) -> str:
    """One goal as JSON (roadmap text, sub-goals, linked tasks, progress).

    Raises `KeyError` for an unknown id, like `get_task`.
    """
    goal = view.manager.goals.get(goal_id)
    return json.dumps(_goal_payload(view, goal), indent=2, ensure_ascii=False)


# -- stats / usage -------------------------------------------------------------


def _average(day: DayStats, kind: str) -> str:
    """seconds / finished stays, human-formatted ("—" when none finished)."""
    rounds = getattr(day, f"{kind}_rounds")
    if rounds <= 0:
        return "-"
    return format_seconds(getattr(day, f"{kind}_seconds") / rounds)


def get_stats(view: ProjectView, days: int = DEFAULT_STATS_DAYS) -> str:
    """The project's per-day task statistics (created/completed/… + averages).

    Read through `manager.stats` — the live store of the running app, which
    is exactly the object every counter was folded into (a throwaway store
    would re-read the same file and could lag behind an in-flight write).
    """
    window = _cap(days, default=DEFAULT_STATS_DAYS, maximum=MAX_DAYS)
    snapshot = view.manager.stats.snapshot(window)
    rows = [
        "day        created completed cancelled failed dev_avg testing_avg resolver_avg"
    ]
    totals = DayStats()
    for day_key, day in snapshot:
        rows.append(
            f"{day_key} {day.created:>7} {day.completed:>9} {day.cancelled:>9} "
            f"{day.failed:>6} {_average(day, 'dev'):>9} "
            f"{_average(day, 'testing'):>11} {_average(day, 'resolver'):>11}"
        )
        totals.created += day.created
        totals.completed += day.completed
        totals.cancelled += day.cancelled
        totals.failed += day.failed
    rows.append(
        f"total (last {window} days): created={totals.created} "
        f"completed={totals.completed} cancelled={totals.cancelled} "
        f"failed={totals.failed}"
    )
    return "\n".join(rows)


def get_usage(view: ProjectView, days: int = DEFAULT_STATS_DAYS) -> str:
    """Per-day, per-`<role>/<model>` token usage, newest last.

    The counters are the ones `UsageStore` persists for the exact model
    calls of the pipeline; the window is measured backwards from today
    (UTC), inclusive.
    """
    window = _cap(days, default=DEFAULT_STATS_DAYS, maximum=MAX_DAYS)
    days_map = view.usage().load()
    if not days_map:
        return f"usage: no model calls recorded in the last {window} day(s)."
    oldest = (date.today() - timedelta(days=window - 1)).isoformat()
    rows: list[str] = []
    totals = dict.fromkeys(FIELDS, 0)
    for day_key in sorted(day_key for day_key in days_map if day_key >= oldest):
        entry = days_map[day_key]
        for model_key in sorted(entry):
            counters = entry[model_key]
            for field in FIELDS:
                totals[field] += counters.get(field, 0)
            rows.append(
                f"{day_key} {model_key} "
                + " ".join(f"{field}={counters.get(field, 0)}" for field in FIELDS)
            )
    if not rows:
        return f"usage: no model calls recorded in the last {window} day(s)."
    rows.append(
        f"total (last {window} days): "
        + " ".join(f"{field}={totals[field]}" for field in FIELDS)
    )
    return "\n".join(rows)


# -- check_file_overlaps -------------------------------------------------------


def check_file_overlaps(
    view: ProjectView, text: str, depends_on: list[str] | None = None
) -> str:
    """Pre-flight: which unfinished tasks mention the same files as `text`.

    The mechanical warning `create_task` runs anyway — exposed so an
    external agent can ask BEFORE creating (and then pass `depends_on` at
    creation, which is the only moment it can prevent the parallel merge
    race). A declared `depends_on` is excluded from the report: it already
    serializes the merge, so warning about it would be noise.
    """
    overlaps = find_file_overlaps(
        view.manager.tasks,
        text,
        exclude_id="",
        depends_on=list(depends_on or []),
    )
    tokens = extract_path_tokens(text)
    if not tokens:
        return "No path-like tokens found in the text — nothing to compare."
    if not overlaps:
        return (
            f"Scanned {len(tokens)} path token(s): no unfinished task mentions "
            "the same files."
        )
    lines = [f"Scanned {len(tokens)} path token(s) — overlapping unfinished tasks:"]
    for overlap in overlaps:
        lines.append(f"#{overlap.task_id}: {', '.join(overlap.files)}")
    lines.append(build_overlap_warning(overlaps))
    return "\n".join(lines)


__all__ = [
    "DEFAULT_EVENT_LIMIT",
    "DEFAULT_STATS_DAYS",
    "DEFAULT_TASK_LIMIT",
    "MAX_DAYS",
    "MAX_EVENT_LIMIT",
    "MAX_TASK_LIMIT",
    "ProjectView",
    "check_file_overlaps",
    "format_task_row",
    "get_goal",
    "get_queue",
    "get_stats",
    "get_task",
    "get_usage",
    "list_goals",
    "list_tasks",
    "parse_keywords",
    "search_tasks",
]
