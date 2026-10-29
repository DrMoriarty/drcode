"""Goal-link commands of the Settings palette."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

from ..goal_screen import _goal_type_marker
from .base import CommandEntry, CommandListProvider

if TYPE_CHECKING:
    from ...models import Goal
    from ..app import DrCodeApp


def _goal_line(goal: Goal) -> str:
    """Palette entry label for a goal: the type marker + title + id."""
    return f"{_goal_type_marker(goal)} {goal.title_display} (#{goal.id})"


class GoalCommandsProvider(CommandListProvider):
    """Goal commands in the Settings palette.

    For the VIEWED task: one "Link goal: <title>" entry per attachable
    goal (not already linked, not archived — archived goals are out of
    the roadmap's active surface) and one "Unlink goal" entry when the
    task carries a link. Plus the always-available capture entries:
    "Create goal…" (links the created goal to the viewed task when one
    is open) and the plain "New goal" (the same dialog as Ctrl+Y)."""

    def _commands(self) -> list[CommandEntry]:
        from ...models import GoalStatus

        app = cast("DrCodeApp", self.app)
        task = app.viewed
        commands: list[CommandEntry] = []
        if task is not None:
            if task.goal_id and task.goal_id in app.manager.goals:
                goal = app.manager.goals.get(task.goal_id)
                commands.append(
                    CommandEntry(
                        f"Unlink goal: {_goal_line(goal)}",
                        app.unlink_viewed_goal,
                        "Detach this task from its goal — the ◆ marker "
                        "and the goal's n/m progress counter drop it",
                    )
                )
            for goal in app.manager.goals.goals.values():
                if goal.id == task.goal_id or goal.status is GoalStatus.ARCHIVED:
                    continue
                commands.append(
                    CommandEntry(
                        f"Link goal: {_goal_line(goal)}",
                        partial(app.link_viewed_goal, goal.id),
                        "Attach this task to the goal — it counts into "
                        "the goal's n/m progress in the sidebar's GOALS "
                        "section (a previous link is replaced)",
                    )
                )
        commands.append(
            CommandEntry(
                "Create goal…",
                partial(
                    app.open_new_goal_dialog,
                    link_task_id=task.id if task is not None else None,
                ),
                "New product goal (title + optional type) — when a task "
                "is open it is linked to the created goal right away",
            )
        )
        commands.append(
            CommandEntry(
                "New goal",
                app.action_new_goal,
                "Ctrl+Y — capture a product goal the research agent "
                "plans against (no task link)",
            )
        )
        return commands
