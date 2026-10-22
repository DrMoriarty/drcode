from __future__ import annotations

import math
from typing import ClassVar

from rich.cells import cell_len
from textual import events
from textual.actions import SkipAction
from textual.binding import Binding
from textual.widgets import TextArea


class WordNavTextArea(TextArea):
    """TextArea with macOS-style Option+Arrow word navigation, a working
    copy/cut and edit-time cursor scrolling.

    Terminals report Option+Left/Right as `alt+left`/`alt+right` (kitty
    keyboard protocol or CSI 1;3x sequences), but stock TextArea only binds
    `ctrl+left`/`ctrl+right` — so Option+Arrow arrives unbound. Some
    terminals (legacy esc+b / esc+f meta encoding) surface as `alt+b` /
    `alt+f`, so those aliases are bound too. Option+Shift+Arrow extends the
    selection by word, matching native macOS text fields.
    """

    BINDINGS: ClassVar = [
        # TextArea's built-in `cut` (ctrl+x) and `delete_right` (ctrl+d)
        # bindings would shadow the app-level task actions whenever this
        # input has focus: Screen.active_bindings resolves a key conflict in
        # favor of the widget closest to the focus, and the keyboard event
        # never reaches the app. Re-declaring the keys here on the `app`
        # namespace displaces the TextArea bindings; visibility/disabled
        # state is still gated by the app's check_action (COMPLETED tasks
        # hide both, new-task mode hides both).
        Binding("ctrl+x", "app.cancel_task", "Cancel task"),
        Binding("ctrl+d", "app.done_task", "Done"),
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

    def _app(self):
        from .app import DrCodeApp

        app = self.app
        assert isinstance(app, DrCodeApp)
        return app

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        # Editing scrolls the cursor into view while virtual_size is still
        # stale (the layout hasn't caught up with the new line count yet):
        # scroll_to_region clamps against the old size, so the first scroll
        # is a no-op and the cursor stays below the fold — the next
        # keystroke finally scrolls. Re-running after the refresh settles
        # makes the cursor visible immediately (an empty new line, a paste,
        # a wrap that pushes the cursor down).
        self.call_after_refresh(self.scroll_cursor_visible)

    def action_copy(self) -> None:
        """Copy the selection via the app's clipboard fallback.

        The stock action only writes OSC 52, which macOS Terminal.app and
        iTerm2 (default settings) silently ignore — the copy then does
        nothing. _copy_with_fallback adds pbcopy, and the toast confirms
        that something was actually copied.
        """
        text = self.selected_text
        if not text:
            raise SkipAction()
        app = self._app()
        app._copy_with_fallback(text)
        app._show_copy_toast(text)

    def action_cut(self) -> None:
        # Super deletes the selection (or the cursor line) and copies via
        # OSC 52; add the system-clipboard fallback and the toast on top.
        text = self.selected_text
        super().action_cut()
        if text:
            app = self._app()
            app._copy_with_fallback(text)
            app._show_copy_toast(text)


class NewTaskInput(WordNavTextArea):
    """Multi-line editor for a new task; Tab toggles the task type."""

    BINDINGS: ClassVar = [
        Binding("tab", "toggle_kind", "Toggle type", priority=True),
        Binding("ctrl+enter", "submit_new_task", "Create", priority=True),
        Binding("escape", "cancel_new_task", "Cancel", priority=True),
    ]

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
        super().on_text_area_changed(event)
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
