from __future__ import annotations

import math
from typing import ClassVar

from rich.cells import cell_len
from textual import events
from textual.binding import Binding
from textual.widgets import TextArea


class WordNavTextArea(TextArea):
    """TextArea with macOS-style Option+Arrow word navigation.

    Terminals report Option+Left/Right as `alt+left`/`alt+right` (kitty
    keyboard protocol or CSI 1;3x sequences), but stock TextArea only binds
    `ctrl+left`/`ctrl+right` — so Option+Arrow arrives unbound. Some
    terminals (legacy esc+b / esc+f meta encoding) surface as `alt+b` /
    `alt+f`, so those aliases are bound too. Option+Shift+Arrow extends the
    selection by word, matching native macOS text fields.
    """

    BINDINGS: ClassVar = [
        Binding("alt+left,alt+b", "cursor_word_left", "Cursor word left", show=False),
        Binding(
            "alt+right,alt+f", "cursor_word_right", "Cursor word right", show=False
        ),
        Binding(
            "alt+shift+left,alt+shift+b",
            "cursor_word_left(True)",
            "Cursor word left select",
            show=False,
        ),
        Binding(
            "alt+shift+right,alt+shift+f",
            "cursor_word_right(True)",
            "Cursor word right select",
            show=False,
        ),
    ]


class NewTaskInput(WordNavTextArea):
    """Multi-line editor for a new task; Tab toggles the task type."""

    BINDINGS: ClassVar = [
        Binding("tab", "toggle_kind", "Toggle type", priority=True),
        Binding("ctrl+enter", "submit_new_task", "Create", priority=True),
        Binding("escape", "cancel_new_task", "Cancel", priority=True),
    ]

    def _app(self):
        from .app import DrCodeApp

        app = self.app
        assert isinstance(app, DrCodeApp)
        return app

    def action_toggle_kind(self) -> None:
        self._app().toggle_new_task_kind()

    def action_submit_new_task(self) -> None:
        self._app().submit_new_task()

    def action_cancel_new_task(self) -> None:
        self._app().cancel_new_task()


class ChatInput(WordNavTextArea):
    """Multi-line chat editor that grows with the text.

    Enter sends the message; Shift+Enter / Alt+Enter insert a newline.
    Soft wrapping breaks long lines at the right edge, and the widget's
    height follows the wrapped text (up to a cap) so the whole draft
    stays visible while typing; beyond the cap it scrolls internally.
    """

    MAX_LINES: ClassVar = 10

    BINDINGS: ClassVar = [
        Binding("enter", "submit", "Send", priority=True),
    ]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            soft_wrap=True,
            show_line_numbers=False,
            **kwargs,  # type: ignore[arg-type]
        )

    def _app(self):
        from .app import DrCodeApp

        app = self.app
        assert isinstance(app, DrCodeApp)
        return app

    def action_submit(self) -> None:
        self._app().submit_chat_input()

    async def _on_key(self, event: events.Key) -> None:
        # A "vertical" Enter variant inserts a newline; plain Enter is
        # bound to submit above. Terminals without the kitty keyboard
        # protocol report Shift+Enter as plain Enter — Alt+Enter is the
        # portable fallback for a newline.
        if event.key in ("shift+enter", "alt+enter"):
            event.stop()
            event.prevent_default()
            if not self.read_only:
                self.insert("\n")
                self._update_height()
            return
        await super()._on_key(event)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._update_height()

    def on_resize(self) -> None:
        self._update_height()

    def _wrapped_line_count(self) -> int:
        """Number of visual lines the draft occupies at the current width
        (soft-wrapped)."""
        # content_size excludes the 1+1 horizontal padding; before the first
        # layout it may be 0 — fall back to the widget/screen width so the
        # estimate stays sane.
        width = (
            self.content_size.width
            or (self.size.width - 2)
            or (self.screen.size.width - 2)
            or 80
        )
        width = max(width, 1)
        return sum(
            max(1, math.ceil(cell_len(line) / width))
            for line in (self.text.splitlines() or [""])
        )

    def _update_height(self) -> None:
        self.styles.height = max(2, min(self._wrapped_line_count(), self.MAX_LINES))
