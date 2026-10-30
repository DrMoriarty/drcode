"""Goal actions layer — capture, linking and the GoalScreen entry.

Everything the UI does with product goals beyond the agent's own tools:
the Ctrl+Y capture dialog (optionally preset with a parent goal and/or a
task to link), the header assign/change/unlink links over the viewed
task, the research/dev task creation FROM a goal (R/T on the
GoalScreen) and opening the full-screen roadmap view
(``tui.goal_screen.GoalScreen``) from the sidebar's GOALS section.

Split out of :mod:`drcode.tui.actions_forms` (which owned the original
capture-only dialog) — see the composition in ``ActionsLayer``.
"""

from __future__ import annotations

from rich.markup import escape
from textual.widgets import Static

from ..models import Goal, GoalStatus, TaskKind
from .goal_screen import (
    GoalAssignPicker,
    GoalCaptureResult,
    GoalEditScreen,
    GoalScreen,
)
from .state import DrCodeState


class GoalsActionsLayer(DrCodeState):
    # -- the full-screen roadmap view -----------------------------------------

    def action_open_goals(self, goal_id: str | None = None) -> None:
        """Open the GoalScreen (a click/Enter on the sidebar's GOALS
        section; `goal_id` focuses and expands that goal's row)."""

        self.push_screen(GoalScreen(focus_goal_id=goal_id))

    # -- capturing a product goal (Ctrl+Y / "+ add goal" / palette) -----------

    def open_new_goal_dialog(
        self,
        *,
        parent_id: str | None = None,
        parent_label: str | None = None,
        link_task_id: str | None = None,
    ) -> None:
        """Push the capture dialog, remembering what the created goal
        should be pre-wired to: a parent (the GoalScreen's N) and/or a
        task to link (the palette's "Create goal…" over a viewed task).
        The plain Ctrl+Y passes nothing."""
        self._goal_capture_context = (parent_id, link_task_id)
        self.push_screen(
            GoalEditScreen(
                parent_id=parent_id,
                parent_label=parent_label,
                link_task_id=link_task_id,
            ),
            self._goal_captured,
        )

    def action_new_goal(self) -> None:
        """Ctrl+Y: the minimal goal-capture dialog (title required, type
        and description optional). The goal lands in .drcode/goals/
        immediately and is visible to the research agent's planning from
        its next activation — no restart, the goals block is recomputed
        per request."""
        self.open_new_goal_dialog()

    def _goal_captured(
        self, result: GoalCaptureResult | tuple[str, str] | None
    ) -> None:
        # The screen's create mode dismisses a GoalCaptureResult; the
        # edit-tuple and None (ESC/Cancel — nothing to do, the pipeline
        # is untouched) are not capture results.
        if not isinstance(result, GoalCaptureResult):
            return
        parent_id, link_task_id = self._goal_capture_context
        self.create_goal_from_capture(
            result, parent_id=parent_id, link_task_id=link_task_id
        )

    def create_goal_from_capture(
        self,
        result: GoalCaptureResult,
        *,
        parent_id: str | None = None,
        link_task_id: str | None = None,
    ) -> Goal:
        """Shared body of every capture path (Ctrl+Y, the sidebar's
        "+ add goal", the palette's "Create goal…", the GoalScreen's N):
        create the goal with its optional parent, apply the dialog's type
        classification and notes, and optionally link it to a task (only
        when the dialog's link checkbox was on AND a task id was preset).
        Returns the goal; persistence and the sidebar refresh are this
        method's job."""
        goal = self.manager.goals.create(
            result.title, description=result.description, parent_id=parent_id
        )
        # The captured item starts as an IDEA unless the user explicitly
        # classified it in the dialog (the agent refines it later too).
        self.manager.goals.update(goal.id, type=result.goal_type, notes=result.notes)
        goal = self.manager.goals.get(goal.id)
        if result.link_task and link_task_id and link_task_id in self.manager.tasks:
            self.manager.link_goal(link_task_id, goal.id)
        self.refresh_sidebar()
        self.notify(
            f"Goal #{goal.id} ({goal.type.value}) '{goal.title_display}' "
            "created — the research agent sees it from its next activation.",
            title="Goals",
            timeout=6,
        )
        return goal

    # -- linking the viewed task to a goal (the Settings palette) -------------

    def link_viewed_goal(self, goal_id: str) -> None:
        """The "+ assign"/"change" header link over the viewed task:
        attaches it to the goal (a previous link is replaced). The
        counters and the ◆ marker follow on the next sidebar refresh."""
        task = self.viewed
        if task is None:
            return
        try:
            self.manager.link_goal(task.id, goal_id)
        except ValueError as e:
            self.notify(str(e), title="Goals", severity="error", timeout=6)
            return
        goal = self.manager.goals.get(goal_id)
        self.notify(
            f"#{task.id} linked to goal #{goal.id} '{goal.title_display}'",
            title="Goals",
            timeout=5,
        )
        self.refresh_sidebar()
        self.refresh_task_status()

    def action_assign_goal(self) -> None:
        """Click on the task statusbar's "+ assign" fragment (the viewed
        task has no goal): open the goal picker and attach the picked
        goal. A drag-selection ending on the link is copying text, not
        navigating (same guard as action_open_task)."""
        if self.screen.selections:
            return
        task = self.viewed
        if task is None:
            return
        candidates = [
            goal
            for goal in self.manager.goals.goals.values()
            if goal.status is not GoalStatus.ARCHIVED and goal.id != task.goal_id
        ]
        if not candidates:
            self.notify(
                "No goals to assign — capture one with Ctrl+Y first.",
                title="Goals",
                severity="warning",
                timeout=6,
            )
            return
        self.push_screen(GoalAssignPicker(candidates), self._goal_assigned)

    def _goal_assigned(self, goal_id: str | None) -> None:
        # None = ESC / no pick: nothing to do.
        if goal_id is None:
            return
        self.link_viewed_goal(goal_id)

    def unlink_viewed_goal(self) -> None:
        """The "unlink" header fragment over the viewed task (also a
        click target — a drag-selection ending on it is copying text,
        not navigating, same guard as the other header links)."""
        if self.screen.selections:
            return
        task = self.viewed
        if task is None or not task.goal_id:
            return
        self.manager.unlink_goal(task.id)
        self.notify(
            f"#{task.id} detached from its goal",
            title="Goals",
            timeout=5,
        )
        self.refresh_sidebar()
        self.refresh_task_status()

    # -- creating tasks FROM a goal (the GoalScreen's R / T) -------------------

    def start_goal_research(self, goal_id: str) -> None:
        """R on the GoalScreen: a RESEARCH task linked to the goal. There
        is deliberately no separate "goal chat" — it is an ordinary
        research task with `goal_id`; the chat input is pre-filled with
        the «Цель: … Задание:» stub and the user completes and sends it
        (the task starts on that send — created without pending input it
        just waits in the queue)."""
        goal = self.manager.goals.get(goal_id)
        task = self.manager.create(
            goal.title, "", kind=TaskKind.RESEARCH, goal_id=goal.id
        )
        self.refresh_sidebar()
        self.view_task(task.id)
        stub = f"Цель: {goal.title}\n"
        if goal.description.strip():
            stub += f"{goal.description.strip()}\n"
        stub += "\nЗадание: "
        self.chat_input.load_text(stub)
        self.chat_input.focus()
        self.notify(
            f"Research task #{task.id} created for goal #{goal.id} — "
            "finish the «Задание:» line and press Enter",
            title="Goals",
            timeout=8,
        )

    def start_goal_task(self, goal_id: str) -> None:
        """T on the GoalScreen: the standard new-task form (Ctrl+N) with
        the goal link preset — the form shows the goal line and the
        created task is attached to it. The link is cleared together with
        the form (cancel/submit)."""
        goal = self.manager.goals.get(goal_id)
        self.action_new_task()
        self._new_task_goal_id = goal.id
        self.query_one("#nt-goal", Static).update(
            f"goal: [magenta]◆[/] {escape(goal.title_display)}"
        )
        self.new_task_input.focus()
