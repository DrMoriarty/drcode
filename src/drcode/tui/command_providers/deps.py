"""Dependency-gate commands of the Settings palette."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

from .base import CommandEntry, CommandListProvider

if TYPE_CHECKING:
    from ..app import DrCodeApp


class DependencyCommandsProvider(CommandListProvider):
    """Dependency-gate commands for the viewed task in the Settings palette.

    One "Remove dependency: #<id>" entry per current dependency (the
    force-start: dropping a blocking dependency makes the task eligible
    immediately and logs the manual override). Attaching a dependency is
    deliberately NOT a palette entry — that lives on the task header (the
    cyan "+" link opening the filtered picker) and in the new-task/edit
    form (`ctrl+g`, comma-separated ids). Only offered while a queued or
    held task is open — dependencies of a running/completed task make no
    sense (the gate is evaluated at promotion time only)."""

    def _commands(self) -> list[CommandEntry]:
        from ...models import TaskStatus

        app = cast("DrCodeApp", self.app)
        task = app.viewed
        if task is None or task.status not in (
            TaskStatus.QUEUED,
            TaskStatus.HOLD,
        ):
            return []
        commands: list[CommandEntry] = []
        for dep in task.depends_on:
            dep_task = app.manager.tasks.get(dep)
            title = f" {dep_task.title_display}" if dep_task else ""
            commands.append(
                CommandEntry(
                    f"Remove dependency: #{dep}{title}",
                    partial(app.action_remove_dependency, dep),
                    "Force start: drop this dependency — the task stops "
                    "waiting for it (logged as a manual override into "
                    "the task's service events)",
                )
            )
        return commands
