"""The generic yes/no confirmation modal.

A single reusable screen for the destructive-ish header links that must
not act on the first click: the caller pushes it with its own wording and
acts only on an explicit "yes" (``True``). ESC and the "no" button both
dismiss with ``False`` — the safe answer is the default.

The layout copies :class:`~drcode.tui.agents_dialog.AgentsMdDialog` (the
compact ``Vertical`` + button row), so every confirmation in the app looks
the same. The widget ids are fixed (``cf-*``) — the screen is modal, so
two instances never share a screen.
"""

from __future__ import annotations

from typing import ClassVar

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ConfirmDialog(ModalScreen[bool]):
    """Modal yes/no question: `message` — confirm?

    Dismisses with True (the caller's action runs) or False (nothing
    changes). `body` is plain text and gets markup-escaped; the title is
    markup (it usually carries the task id in a color).
    """

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "No", priority=True),
    ]

    DEFAULT_CSS = """
    ConfirmDialog {
        align: center middle;
    }
    #confirm-dialog {
        width: 72;
        max-width: 95%;
        height: auto;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #cf-title { text-style: bold; margin-bottom: 1; }
    #cf-body { color: $text; margin-bottom: 1; }
    #cf-buttons {
        height: auto;
        margin-top: 1;
    }
    #cf-buttons Button { width: auto; margin-right: 1; }
    """

    def __init__(
        self,
        title: str,
        body: str,
        *,
        confirm_label: str = "Yes",
        cancel_label: str = "No",
    ) -> None:
        super().__init__()
        self._title = title
        self._body = body
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(self._title, id="cf-title")
            yield Static(escape(self._body), id="cf-body")
            with Horizontal(id="cf-buttons"):
                yield Button(self._confirm_label, variant="primary", id="cf-yes")
                yield Button(self._cancel_label, id="cf-no")

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_mount(self) -> None:
        # The safe answer holds the focus: a stray Enter must not confirm
        # a force-start. The yes button stays one Tab away.
        self.query_one("#cf-no", Button).focus()

    @on(Button.Pressed, "#cf-yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#cf-no")
    def _no(self) -> None:
        self.dismiss(False)
