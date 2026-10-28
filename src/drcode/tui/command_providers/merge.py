"""Merge-conflict commands of the Settings palette."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

from .base import CommandEntry, CommandListProvider

if TYPE_CHECKING:
    from ..app import DrCodeApp


class MergeCommandsProvider(CommandListProvider):
    """Conflict-screen entries for the Settings palette.

    One "Resolve merge conflicts: #<id> <title>" entry per task the
    unmerged-work tracking currently knows about (a `needs_human`
    fallback flag or an unmerged task branch — both read from state the
    10 s poll already refreshed, no git here: the palette runs on the
    event loop). Empty when everything is merged — a healthy tree has
    nothing to offer."""

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        commands: list[CommandEntry] = []
        for task in app.manager.all_tasks():
            if task.needs_human or app._merge_state.get(task.id) == "unmerged":
                commands.append(
                    CommandEntry(
                        f"Resolve merge conflicts: #{task.id} {task.title_display}",
                        partial(app.open_merge_for, task.id),
                        "Open the conflict screen (Ctrl+M): resolve the "
                        "parked merge hunk by hunk, run the checks, then "
                        "commit on the task branch — by your explicit "
                        "confirmation only",
                    )
                )
        return commands
