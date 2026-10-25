"""Dependency-gate commands of the Settings palette."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, cast

from textual.command import DiscoveryHit, Hit, Hits, Provider

if TYPE_CHECKING:
    from ..app import DrCodeApp


class DependencyCommandsProvider(Provider):
    """Dependency-gate commands for the viewed task in the Settings palette.

    One "Remove dependency: #<id>" entry per current dependency (the
    force-start: dropping a blocking dependency makes the task eligible
    immediately and logs the manual override) and one "Add dependency:
    #<id> <title>" entry per other task. Only offered while a queued or
    held task is open — dependencies of a running/completed task make no
    sense (the gate is evaluated at promotion time only)."""

    def _commands(self) -> list[tuple[str, str, Callable[[], object]]]:
        from ...models import TaskStatus

        app = cast("DrCodeApp", self.app)
        task = app.viewed
        if task is None or task.status not in (
            TaskStatus.QUEUED,
            TaskStatus.HOLD,
        ):
            return []
        commands: list[tuple[str, str, Callable[[], object]]] = []
        for dep in task.depends_on:
            dep_task = app.manager.tasks.get(dep)
            title = f" {dep_task.title_display}" if dep_task else ""
            commands.append(
                (
                    f"Remove dependency: #{dep}{title}",
                    (
                        "Force start: drop this dependency — the task stops "
                        "waiting for it (logged as a manual override into "
                        "the task's service events)"
                    ),
                    partial(app.action_remove_dependency, dep),
                )
            )
        for other in app.manager.all_tasks():
            if other.id == task.id or other.id in task.depends_on:
                continue
            commands.append(
                (
                    f"Add dependency: #{other.id} {other.title_display}",
                    (
                        "Make this task wait for it: the dependency must be "
                        "COMPLETED and (under worktree isolation) merged "
                        "into the base branch before this task is taken "
                        "into development"
                    ),
                    partial(app.action_add_dependency, other.id),
                )
            )
        return commands

    async def discover(self) -> Hits:
        for name, help_text, callback in self._commands():
            yield DiscoveryHit(name, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, help_text, callback in self._commands():
            if (match := matcher.match(name)) > 0:
                yield Hit(match, matcher.highlight(name), callback, help=help_text)
