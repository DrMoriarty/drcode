"""Modal path prompt for attaching an image file (ctrl+j in the chat
editor and the new-task form).

The screen only COLLECTS the path — validation (existence, image type,
size cap) and staging live in the app layer
(``TaskFormLayer._on_attach_path``), so a bad path reopens nothing and
shows the error as a toast while the composed text stays untouched.
"""

from __future__ import annotations

from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static


class AttachPathScreen(ModalScreen[str | None]):
    """Ask for the path of an image file to attach. Dismisses with the
    entered path, or None on ESC."""

    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel", priority=True)]

    DEFAULT_CSS = """
    AttachPathScreen {
        align: center middle;
    }
    #attach-dialog {
        width: 64;
        max-width: 95%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #attach-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #attach-hint {
        margin-top: 1;
        color: $text-disabled;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="attach-dialog"):
            yield Static("Attach image file", id="attach-title")
            yield Input(
                placeholder="/path/to/screenshot.png",
                id="attach-path-input",
            )
            yield Static(
                "enter — attach · esc — cancel   (png, jpeg, webp, gif · up to 5 MB)",
                id="attach-hint",
            )

    def on_mount(self) -> None:
        self.query_one("#attach-path-input", Input).focus()

    @on(Input.Submitted, "#attach-path-input")
    def _submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)
