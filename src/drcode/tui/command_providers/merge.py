"""Merge-conflict commands of the Settings palette."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, cast

from textual.command import DiscoveryHit, Hit, Hits, Provider

if TYPE_CHECKING:
    from ..app import DrCodeApp


class MergeCommandsProvider(Provider):
    """Conflict-screen entries for the Settings palette.

    One "Resolve merge conflicts: #<id> <title>" entry per task the
    unmerged-work tracking currently knows about (a `needs_human`
    fallback flag or an unmerged task branch — both read from state the
    10 s poll already refreshed, no git here: the palette runs on the
    event loop). Empty when everything is merged — a healthy tree has
    nothing to offer."""

    def _commands(self) -> list[tuple[str, str, Callable[[], object]]]:
        app = cast("DrCodeApp", self.app)
        commands: list[tuple[str, str, Callable[[], object]]] = []
        for task in app.manager.all_tasks():
            if task.needs_human or app._merge_state.get(task.id) == "unmerged":
                commands.append(
                    (
                        f"Resolve merge conflicts: #{task.id} {task.title_display}",
                        (
                            "Open the conflict screen (Ctrl+M): resolve the "
                            "parked merge hunk by hunk, run the checks, then "
                            "commit on the task branch — by your explicit "
                            "confirmation only"
                        ),
                        partial(app.open_merge_for, task.id),
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
