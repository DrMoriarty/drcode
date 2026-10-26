"""Service-event logging for the agent layer.

Turns tool refusals (path violations) and operational notes (git push outcomes)
into persisted ``TaskEvent`` lines shown in the chat under the "service events"
separator but never fed into the model context. Depends only on `models` at
runtime; the `AgentDeps`/`TesterDeps` types are imported under `TYPE_CHECKING`
to keep the agent layer free of a circular import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import TaskEvent

if TYPE_CHECKING:
    from .agent import AgentDeps, ResolverDeps, TesterDeps


def log_service_event(
    deps: AgentDeps | TesterDeps | ResolverDeps, kind: str, text: str
) -> None:
    """Persist a service event into the task's event log.

    Like all TaskEvents, it is shown in the chat under the "service events"
    separator but never reaches the model context.
    """
    task = deps.manager.get(deps.task_id)
    task.events.append(TaskEvent(kind=kind, text=text))
    task.touch()
    deps.manager.storage.save(task)


def log_violation(
    deps: AgentDeps | TesterDeps | ResolverDeps, tool: str, error: str
) -> None:
    """Persist a permission violation into the task's service-event log."""
    log_service_event(deps, "permission", f"{tool}: {error}")
