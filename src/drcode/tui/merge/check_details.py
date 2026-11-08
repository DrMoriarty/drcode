"""The check-details modal of the conflict screen.

Full-screen viewer of one verification check's raw output, pushed OVER
the ConflictScreen: esc returns without touching the resolution state
(the conflict screen stays mounted underneath), ``[`` / ``]`` step
through the check list — skipped checks show their explanatory output
("no tests/ directory", "pytest unavailable").
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from ...merge_resolver import CheckResult
from .views import CheckOutputView


class CheckDetailsScreen(ModalScreen[None]):
    """Full-screen viewer of one check's raw output, pushed OVER the
    ConflictScreen: esc returns without touching the resolution state
    (the conflict screen stays mounted underneath), ``[`` / ``]`` step
    through the check list — skipped checks show their explanatory
    output ("no tests/ directory", "pytest unavailable")."""

    BINDINGS: ClassVar = [
        Binding("escape", "back", "Back", priority=True),
        Binding("left_square_bracket", "prev_check", "Previous check", show=False),
        Binding("right_square_bracket", "next_check", "Next check", show=False),
    ]

    DEFAULT_CSS = """
    CheckDetailsScreen {
        align: center middle;
    }
    #check-details-dialog {
        width: 95%;
        max-width: 160;
        height: 95%;
        padding: 0 1 1 1;
        background: $panel;
        border: thick $accent;
    }
    #check-details-header {
        height: 1;
        padding: 0 1;
        color: $text;
        text-style: bold;
    }
    #check-details-view { height: 1fr; }
    #check-details-footer {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, checks: list[CheckResult], index: int = 0) -> None:
        super().__init__()
        self._checks = checks
        self._index = max(0, min(index, len(checks) - 1))

    def compose(self) -> ComposeResult:
        with Vertical(id="check-details-dialog"):
            # markup=False: check names / the "[n/m]" counter would trip
            # Rich markup.
            yield Static("", id="check-details-header", markup=False)
            yield CheckOutputView(id="check-details-view")
            yield Static(
                "[ ] prev/next check · arrows scroll · esc back",
                id="check-details-footer",
                markup=False,
            )

    def on_mount(self) -> None:
        self._show()

    def _show(self) -> None:
        check = self._checks[self._index]
        self.query_one("#check-details-header", Static).update(
            f"[{self._index + 1}/{len(self._checks)}] {check.line}"
        )
        self.query_one("#check-details-view", CheckOutputView).set_output(check.output)

    def action_prev_check(self) -> None:
        if self._checks:
            self._index = (self._index - 1) % len(self._checks)
            self._show()

    def action_next_check(self) -> None:
        if self._checks:
            self._index = (self._index + 1) % len(self._checks)
            self._show()

    def action_back(self) -> None:
        self.dismiss()
