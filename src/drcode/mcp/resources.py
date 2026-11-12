"""MCP resources of the embedded drcode server (design doc §5/§7).

Resources are the *push* half of the read surface: an external agent
subscribes over the standard `subscriptions/listen` stream and gets a
standard `notifications/resources/updated` for the URIs whose underlying
data actually changed. Nothing custom is ever sent — a client that does
not understand our notifications simply ignores them, and a client that
never subscribes loses nothing at all: the same data is reachable by
polling the read tools (§9).

Four resources exist, and their content is EXACTLY what the matching read
tool returns (`reads.py` is the single shaping layer both of them use), so
a client can subscribe and refetch instead of polling:

- `drcode://queue` — `get_queue`
- `drcode://tasks/{task_id}` — `get_task`
- `drcode://goals/{goal_id}` — `get_goal`
- `drcode://stats` — `get_stats`

`drcode://tasks/*` and `drcode://goals/*` are URI TEMPLATES (one resource
per task/goal without enumerating the archive), while `queue`/`stats` are
static.

Routing lives here too (`affected_uris`), because it is the other half of
the same contract: the bridge must notify ONLY the URIs an event actually
touched — a status change of an unlinked task must not wake up every
`drcode://goals/*` subscriber, or coalescing would be pointless.
"""

from __future__ import annotations

from collections.abc import Iterable

from mcp.server.mcpserver.exceptions import ResourceNotFoundError

from ..models import Task

# The static resources (§7).
QUEUE_URI = "drcode://queue"
STATS_URI = "drcode://stats"

# The per-instance resource templates. The names double as the format
# strings the bridge builds notification URIs with, so a template and a
# concrete URI can never drift apart.
TASK_URI_TEMPLATE = "drcode://tasks/{task_id}"
GOAL_URI_TEMPLATE = "drcode://goals/{goal_id}"

STATIC_URIS: tuple[str, ...] = (QUEUE_URI, STATS_URI)


def task_uri(task_id: str) -> str:
    """The concrete `drcode://tasks/<id>` URI (the notification spelling)."""
    return TASK_URI_TEMPLATE.format(task_id=task_id)


def goal_uri(goal_id: str) -> str:
    """The concrete `drcode://goals/<id>` URI."""
    return GOAL_URI_TEMPLATE.format(goal_id=goal_id)


# -- routing: which URIs an event touches (§5) --------------------------------


def _task_uris(task: Task) -> list[str]:
    """The URIs a task's own state change touches: the task itself, the
    queue (order/composition), and its goal when it has one."""
    uris = [task_uri(task.id), QUEUE_URI]
    if task.goal_id:
        uris.append(goal_uri(task.goal_id))
    return uris


def affected_uris_for_status(task: Task) -> list[str]:
    """URIs affected by a STATUS transition (`fire_status_change`).

    No `drcode://stats`: a transition alone does not change the day
    buckets — only the transition INTO COMPLETED does, and that has its
    own listener (`affected_uris_for_completion`).
    """
    return _task_uris(task)


def affected_uris_for_completion(task: Task) -> list[str]:
    """URIs affected by a transition into COMPLETED: the task, the queue,
    its goal and the project statistics (completed/failed/cancelled
    counters and the finished-stay averages all move)."""
    return [*_task_uris(task), STATS_URI]


def affected_uris_for_creation(task: Task) -> list[str]:
    """URIs affected by a task entering the system: the new task, the
    queue, its goal and the statistics (`created` counter)."""
    return [*_task_uris(task), STATS_URI]


def affected_uris_for_removal(task_id: str) -> list[str]:
    """URIs affected by a task disappearing (cancel/delete/purge).

    The task's `goal_id` is not knowable here — the removal listeners fire
    AFTER the task left the manager — so a removed task cannot notify its
    goal. That is deliberate and harmless: a goal's own content (progress
    done/total) is derived from the LIVE tasks, so a client watching a goal
    still sees the number shrink on its next poll or on the next completion
    event of that goal; the alternative (notifying every goal) would defeat
    the "only affected URIs" rule for the common case.
    """
    return [task_uri(task_id), QUEUE_URI]


def dedupe(uris: Iterable[str]) -> list[str]:
    """Order-preserving deduplication of a URI list."""
    seen: dict[str, None] = {}
    for uri in uris:
        seen.setdefault(uri, None)
    return list(seen)


# -- registration --------------------------------------------------------------


def register_resources(server, view) -> None:
    """Register the four resources on the SDK server for `view`.

    `view` is the read source (`reads.ProjectView`): resource bodies are
    the very same rendering the read tools produce, so the subscription
    story and the polling story can never disagree. The handlers are sync:
    the SDK runs them in a worker thread, which is what we want for the
    (cheap, disk-touching) metadata reads.
    """
    from .reads import get_goal, get_queue, get_stats, get_task

    @server.resource(
        QUEUE_URI,
        name="queue",
        title="drcode task queue",
        description=(
            "The project's task queue in promotion order, with the dependency "
            "blockers of every entry (same content as the get_queue tool)."
        ),
        mime_type="text/plain",
    )
    def queue_resource() -> str:
        return get_queue(view)

    @server.resource(
        STATS_URI,
        name="stats",
        title="drcode project statistics",
        description=(
            "Per-day project task statistics: created/completed/cancelled/"
            "failed counts and the average development, testing and "
            "conflict-resolution times (same content as the get_stats tool)."
        ),
        mime_type="application/json",
    )
    def stats_resource() -> str:
        return get_stats(view)

    @server.resource(
        TASK_URI_TEMPLATE,
        name="task",
        title="drcode task metadata",
        description=(
            "One task's full metadata, its search skeleton (user prompts + "
            "final agent response), its last service events, the pending "
            "clarification and the blocking dependencies (same content as "
            "the get_task tool). Full transcripts are never exposed."
        ),
        mime_type="application/json",
    )
    def task_resource(task_id: str) -> str:
        try:
            return get_task(view, task_id)
        except KeyError:
            raise ResourceNotFoundError(f"unknown task id: {task_id}") from None

    @server.resource(
        GOAL_URI_TEMPLATE,
        name="goal",
        title="drcode goal",
        description=(
            "One long-lived product goal: its roadmap text, lifecycle "
            "status, sub-goals, linked tasks and the done/total progress "
            "(same content as the get_goal tool)."
        ),
        mime_type="application/json",
    )
    def goal_resource(goal_id: str) -> str:
        try:
            return get_goal(view, goal_id)
        except KeyError:
            raise ResourceNotFoundError(f"unknown goal id: {goal_id}") from None


__all__ = [
    "GOAL_URI_TEMPLATE",
    "QUEUE_URI",
    "STATIC_URIS",
    "STATS_URI",
    "TASK_URI_TEMPLATE",
    "affected_uris_for_completion",
    "affected_uris_for_creation",
    "affected_uris_for_removal",
    "affected_uris_for_status",
    "dedupe",
    "goal_uri",
    "register_resources",
    "task_uri",
]
