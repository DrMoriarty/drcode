"""Startup offer to create AGENTS.md.

Shown modally on app start when the model is configured but the project
root has no `AGENTS.md`. The user can accept (a pre-filled `research`
task with the bundled init-repo prompt is created — the same one the
Settings palette's "Update/create AGENTS.md" entry runs) or decline and
keep working without it. ESC counts as a decline.
"""

from __future__ import annotations

from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class AgentsMdDialog(ModalScreen[bool]):
    """Modal yes/no question: AGENTS.md is missing — create it now?

    Dismisses with True (create the task) or False (declined).
    """

    BINDINGS: ClassVar = [
        Binding("escape", "decline", "No", priority=True),
    ]

    DEFAULT_CSS = """
    AgentsMdDialog {
        align: center middle;
    }
    #agents-dialog {
        width: 64;
        max-width: 95%;
        height: auto;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #ag-note { color: $text; margin-bottom: 1; }
    #ag-hint { color: $text-disabled; margin-bottom: 1; }
    #ag-buttons {
        height: auto;
        margin-top: 1;
    }
    #ag-buttons Button { width: auto; margin-right: 1; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="agents-dialog"):
            yield Static(
                "This project has no [b]AGENTS.md[/b] — the file with "
                "project instructions that is added to the agent's system "
                "prompt on every run.",
                id="ag-note",
            )
            yield Static(
                "An agent task can analyze the repository and create "
                "(or later refresh) this file. Create it now?",
                id="ag-hint",
            )
            with Horizontal(id="ag-buttons"):
                yield Button("Yes, create", variant="primary", id="ag-yes")
                yield Button("No, not now", id="ag-no")

    def action_decline(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#ag-yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#ag-no")
    def _no(self) -> None:
        self.dismiss(False)
