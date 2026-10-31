"""Goal-capture command of the Settings palette."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

from .base import CommandEntry, CommandListProvider

if TYPE_CHECKING:
    from ..app import DrCodeApp


class GoalCommandsProvider(CommandListProvider):
    """Goal commands in the Settings palette.

    The single always-available capture entry: "Create goal…" — the
    unified edit-form dialog in create mode, with a pre-enabled checkbox
    to link the created goal to the viewed task when one is open (plain
    create without the checkbox when not). Linking and unlinking a
    task's goal live in the task header ("+ assign" / "✏" / "✕"),
    not here."""

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        task = app.viewed
        return [
            CommandEntry(
                "Create goal…",
                partial(
                    app.open_new_goal_dialog,
                    link_task_id=task.id if task is not None else None,
                ),
                "New product goal (title, type, description, notes) — "
                "when a task is open a pre-enabled checkbox links it "
                "to the created goal right away",
            )
        ]
