"""User capture of a product goal (stage A, the passive registry).

A minimal modal dialog — the title is the only required field, the type
(idea/goal/milestone) and the description are optional and can be refined
later (by the user here, or by the research agent via its update_goal
tool). Deliberately NOT the excluded Roadmap screen: no listing, no
editing of existing goals — capture only. The goal lands in
`<project>/.drcode/goals/` and from the next research activation is part
of the agent's planning surface.
"""

from __future__ import annotations

from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from ..models import GoalType

_TYPE_CYCLE: list[GoalType] = [
    GoalType.IDEA,
    GoalType.GOAL,
    GoalType.MILESTONE,
]

_TYPE_HINTS: dict[GoalType, str] = {
    GoalType.IDEA: "a captured thought, refined later",
    GoalType.GOAL: "an approved direction of development",
    GoalType.MILESTONE: "a decomposable checkpoint",
}


class GoalCaptureScreen(ModalScreen[tuple[str, str, GoalType] | None]):
    """Modal "New goal" form: title (required), description (optional),
    type cycled with ctrl+t. Dismisses with
    (title, description, type) on create, None on ESC/Cancel."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+t", "cycle_type", "Change type", priority=True),
    ]

    DEFAULT_CSS = """
    GoalCaptureScreen {
        align: center middle;
    }
    #goal-dialog {
        width: 72;
        max-width: 95%;
        height: auto;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #goal-title { margin-bottom: 1; }
    #goal-desc { margin-bottom: 1; }
    #goal-note { color: $text; margin-bottom: 1; }
    #goal-error { color: $error; height: auto; }
    #goal-type-label {
        width: auto;
        margin-right: 1;
        text-style: bold;
        color: yellow;
    }
    #goal-type-hint { color: $text-disabled; }
    #goal-buttons {
        height: auto;
        margin-top: 1;
    }
    #goal-buttons Button { width: auto; margin-right: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._goal_type = GoalType.IDEA

    # -- composition ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="goal-dialog"):
            yield Static(
                "New product goal — the roadmap item the research agent "
                "plans against. The title is enough to start; classification "
                "and description can be refined later.",
                id="goal-note",
            )
            yield Input(placeholder="Goal title (required)", id="goal-title")
            yield Input(placeholder="Description (optional)", id="goal-desc")
            yield Horizontal(
                Static("", id="goal-type-label"),
                Static("", id="goal-type-hint"),
                id="goal-type-line",
            )
            yield Static("", id="goal-error")
            with Horizontal(id="goal-buttons"):
                yield Button("Create goal", variant="primary", id="goal-create")
                yield Button("Cancel", id="goal-cancel")
        self._update_type_label()

    def on_mount(self) -> None:
        self.query_one("#goal-title", Input).focus()

    @property
    def type_label(self) -> Static:
        return self.query_one("#goal-type-label", Static)

    def _update_type_label(self) -> None:
        if not self.is_mounted:
            return
        self.type_label.update(f"type: {self._goal_type.value}")
        self.query_one("#goal-type-hint", Static).update(_TYPE_HINTS[self._goal_type])

    # -- actions ------------------------------------------------------------------

    def action_cycle_type(self) -> None:
        index = _TYPE_CYCLE.index(self._goal_type)
        self._goal_type = _TYPE_CYCLE[(index + 1) % len(_TYPE_CYCLE)]
        self._update_type_label()

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#goal-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#goal-create")
    def _create(self) -> None:
        self._submit()

    @on(Input.Submitted, "#goal-title")
    def _title_submitted(self) -> None:
        self._submit()

    @on(Input.Submitted, "#goal-desc")
    def _desc_submitted(self) -> None:
        self._submit()

    def _submit(self) -> None:
        title = self.query_one("#goal-title", Input).value.strip()
        if not title:
            self.query_one("#goal-error", Static).update(
                "The title is required — one line naming the goal."
            )
            self.query_one("#goal-title", Input).focus()
            return
        description = self.query_one("#goal-desc", Input).value.strip()
        self.dismiss((title, description, self._goal_type))
