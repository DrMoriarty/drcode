"""Read-only MCP tools (design doc §7, the "read" block).

Thin typed wrappers over `reads.py` — the shaping layer the resources use
too, so a subscriber's refetch and a poller's tool call return the same
body. Three rules hold here:

**Anticipated failures are `is_error`, not crashes** (§4: "не исключение,
а нормальный MCP isError"). An unknown task id, an unknown status filter or
an empty keyword list is the caller's mistake and comes back as readable
text (`ToolError` → `isError: true`); only a genuinely unexpected exception
crashes, and the SDK sanitizes that.

**Handlers are sync.** The SDK runs them in a worker thread, which is what
the disk-touching reads (archive search, usage file) want; an async handler
would block the server's event loop that every subscriber's notification
stream rides on.

**Nothing mutates.** The read block is deliberately side-effect free — no
listener registration, no store writes, no pipeline advance. `get_task`
never touches a transcript (only the already-persisted skeleton).
"""

from __future__ import annotations

from mcp.server.mcpserver.exceptions import ToolError

from . import reads
from .reads import ProjectView

#: Tool names — constants so tests and registration can never disagree.
LIST_TASKS = "list_tasks"
GET_TASK = "get_task"
SEARCH_TASKS = "search_tasks"
GET_QUEUE = "get_queue"
LIST_GOALS = "list_goals"
GET_GOAL = "get_goal"
GET_STATS = "get_stats"
GET_USAGE = "get_usage"
CHECK_FILE_OVERLAPS = "check_file_overlaps"

#: The whole read surface, in registration order (tests assert it in one place).
READ_TOOLS: tuple[str, ...] = (
    LIST_TASKS,
    GET_TASK,
    SEARCH_TASKS,
    GET_QUEUE,
    LIST_GOALS,
    GET_GOAL,
    GET_STATS,
    GET_USAGE,
    CHECK_FILE_OVERLAPS,
)

STATUS_HINT = "queued/draft/hold/active/testing/completed/clarification (or 'all')"
KIND_HINT = "bug/feature/research (or 'all')"


def register_tools(server, view: ProjectView) -> None:
    """Register the nine read tools on the SDK server for `view`."""
    _unknown_ids = lambda: ", ".join(f"#{t.id}" for t in view.manager.all_tasks()[-5:])

    @server.tool(
        name=LIST_TASKS,
        title="List drcode tasks",
        description=(
            "List the project's tasks as one compact line each: "
            "#id [status/kind] title — markers. Markers cover the tester's "
            "verification label (verification=passed/failed), hold, "
            "unfinished (an interrupted attempt), needs_human (automatic "
            "conflict resolution failed), hook_merge_failed, cancelled, "
            "github=<issue>, goal=<id>, input=<queued message count>, "
            "clarification (the agent waits for an answer) and "
            "depends_on=<ids>. The cheapest way to see the whole board; "
            "`status` ("
            + STATUS_HINT
            + ") and `kind` ("
            + KIND_HINT
            + ") narrow it. For one task's details use get_task, for the "
            "promotion order use get_queue, for keyword lookup over past "
            "work use search_tasks."
        ),
    )
    def list_tasks(
        status: str | None = None,
        kind: str | None = None,
        limit: int = reads.DEFAULT_TASK_LIMIT,
    ) -> str:
        try:
            return reads.list_tasks(view, status=status, kind=kind, limit=limit)
        except ValueError as e:
            raise ToolError(str(e)) from None

    @server.tool(
        name=GET_TASK,
        title="Get one drcode task",
        description=(
            "Full metadata of ONE task as JSON: status/kind/verification, "
            "timestamps (created/updated/status_changed/activated/"
            "completed), the queue stamp, github issue and goal references, "
            "depends_on plus the currently BLOCKING dependencies, pending "
            "input messages, the clarification package the agent waits on "
            "(question + options + the agent's own proposals), the todo "
            "checklist, tracked edited files, the last N service events and "
            "the task's search SKELETON — the user prompts and the agent's "
            "final response. Full model transcripts are never exposed (they "
            "live in a sidecar file and can be megabytes); the skeleton is "
            "the cheap projection that answers 'what was this about and how "
            "did it end'."
        ),
    )
    def get_task(task_id: str, events: int = reads.DEFAULT_EVENT_LIMIT) -> str:
        try:
            return reads.get_task(view, task_id, events=events)
        except KeyError:
            known = _unknown_ids()
            hint = f" Known task ids include: {known}." if known else ""
            raise ToolError(f"unknown task id {task_id!r}.{hint}") from None

    @server.tool(
        name=SEARCH_TASKS,
        title="Search the drcode task archive",
        description=(
            "Keyword search over the whole task archive (every status, "
            "completed work included) — the way to find past decisions, bug "
            "reports and 'was this done and merged?'. Case-insensitive "
            "substring matching with a light stemming variant; keywords "
            "combine with AND first, and when nothing matches AND an OR "
            "pass returns its results under an explicit 'no exact AND "
            "match' preamble. Ranked by occurrences, ties newest-first. "
            "Output is at most 50 compact lines — id, status/kind, "
            "verification and unmerged markers, date, title, snippet. "
            "`keywords` may be one space/comma-separated string or a list. "
            "Optional `status` ("
            + STATUS_HINT
            + ") and `kind` ("
            + KIND_HINT
            + ") filters."
        ),
    )
    def search_tasks(
        keywords: str,
        status: str | None = None,
        kind: str | None = None,
    ) -> str:
        try:
            return reads.search_tasks(view, keywords, status=status, kind=kind)
        except ValueError as e:
            raise ToolError(str(e)) from None

    @server.tool(
        name=GET_QUEUE,
        title="Get the drcode queue",
        description=(
            "The development queue in PROMOTION order (dependency-blocked "
            "tasks sink, bug reports newest-first at the head, answered "
            "clarifications first among the rest, then the explicit queue "
            "stamp — the same order the sidebar shows), plus which task(s) "
            "are ACTIVE right now. Each entry lists the dependencies that "
            "still block it. The git-mergedness half of the dependency gate "
            "is only evaluated at promotion time, so an entry may still be "
            "parked by an unmerged dependency branch after every status "
            "blocker cleared."
        ),
    )
    def get_queue() -> str:
        return reads.get_queue(view)

    @server.tool(
        name=LIST_GOALS,
        title="List drcode goals",
        description=(
            "Every long-lived product goal (idea/goal/milestone) with its "
            "lifecycle status and live progress (done/total of the tasks "
            "linked to it; a milestone sums its whole subtree). Goals are "
            "the roadmap layer — they never enter the task state machine. "
            "For one goal's full text (description + notes), sub-goals and "
            "linked tasks use get_goal."
        ),
    )
    def list_goals() -> str:
        return reads.list_goals(view)

    @server.tool(
        name=GET_GOAL,
        title="Get one drcode goal",
        description=(
            "One goal as JSON: title, type, lifecycle status, the short "
            "description and the long notes, parent_id, the sub-goal ids "
            "under it, the live done/total progress, whether the goal "
            "counts as complete, and every task linked to it (id, status, "
            "kind, title)."
        ),
    )
    def get_goal(goal_id: str) -> str:
        try:
            return reads.get_goal(view, goal_id)
        except KeyError:
            raise ToolError(f"unknown goal id {goal_id!r}") from None

    @server.tool(
        name=GET_STATS,
        title="Get drcode project statistics",
        description=(
            "Per-day project task statistics for the last `days` (default "
            "14) UTC days: created / completed / cancelled / failed counts "
            "and the average development, testing and conflict-resolution "
            "times of the finished stays. Folded at event time, never "
            "recomputed from the archive."
        ),
    )
    def get_stats(days: int = reads.DEFAULT_STATS_DAYS) -> str:
        return reads.get_stats(view, days=days)

    @server.tool(
        name=GET_USAGE,
        title="Get drcode token usage",
        description=(
            "Per-day token usage of the last `days` (default 14) UTC days, "
            "one row per <role>/<model> (coding/tester/... with the exact "
            "model id): requests, input, output, cache_read, cache_write. "
            "User-level counters — across the sessions of this machine, not "
            "just this project."
        ),
    )
    def get_usage(days: int = reads.DEFAULT_STATS_DAYS) -> str:
        return reads.get_usage(view, days=days)

    @server.tool(
        name=CHECK_FILE_OVERLAPS,
        title="Check drcode file overlaps",
        description=(
            "Pre-flight conflict warning: which UNFINISHED tasks mention "
            "the same files as the given text. `text` is the task "
            "description you are about to file; path-like tokens "
            "(src/x.py, name.ext) are extracted from it and compared "
            "against the descriptions and tracked edited_files of every "
            "unfinished task. A non-empty overlap is a WARNING, never a "
            "block — but act on it immediately: the only way to prevent a "
            "parallel merge race is depends_on at creation time (tasks "
            "already listed in `depends_on` are excluded from the report)."
        ),
    )
    def check_file_overlaps(text: str, depends_on: list[str] | None = None) -> str:
        return reads.check_file_overlaps(view, text, depends_on=depends_on)


__all__ = [
    "CHECK_FILE_OVERLAPS",
    "GET_GOAL",
    "GET_QUEUE",
    "GET_STATS",
    "GET_TASK",
    "GET_USAGE",
    "KIND_HINT",
    "LIST_GOALS",
    "LIST_TASKS",
    "READ_TOOLS",
    "SEARCH_TASKS",
    "STATUS_HINT",
    "register_tools",
]
