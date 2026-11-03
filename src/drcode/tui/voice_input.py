"""Textual host adapter of the push-to-talk dictation state machine.

The dictation state machine itself lives UI-neutral in
:mod:`drcode.core.voice_input`; this module implements its
:class:`drcode.core.voice_input.VoiceDictationHost` seam over the
Textual app (toasts via ``app.notify``, the indicator re-render via
``call_from_thread``, target-field validation/insertion against the
Textual widget tree) and re-exports the controller under its historical
import path.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from textual.dom import DOMNode
from textual.widgets import TextArea

from ..core.voice_input import (  # noqa: F401 - historical import path
    _MIN_RECORDING,
    _TOGGLE_DEBOUNCE,
    VoiceDictationHost,
    VoiceInputController,
    _State,
)

if TYPE_CHECKING:
    from ..settings import ProjectSettings
    from .app import DrCodeApp


class TextualVoiceHost:
    """Implements the dictation host seam over the Textual app."""

    def __init__(self, app: DrCodeApp) -> None:
        self._app = app

    def voice_settings(self) -> ProjectSettings:
        return self._app.settings

    def voice_notify(self, message: str, *, error: bool = False) -> None:
        self._app.notify(
            message,
            title="Voice",
            severity="error" if error else "information",
            timeout=6 if error else 4,
        )

    def voice_sync_indicator(self) -> None:
        """Re-render the persistent header indicator right away (the 1 s
        activity poll would refresh it anyway). Also refreshes the
        bindings, so the footer's "Cancel recording" entry appears and
        disappears together with the state. Safe from any thread."""
        if threading.get_ident() == self._app._thread_id:
            self._app._render_model_status()
            self._app.refresh_bindings()
        else:

            def _sync() -> None:
                self._app._render_model_status()
                self._app.refresh_bindings()

            self._app.call_from_thread(_sync)

    def voice_on_ui_thread(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        self._app.call_from_thread(fn, *args, **kwargs)

    def voice_context_of(self, target: Any) -> tuple[str, Any]:
        """Freeze the UI context the dictation belongs to: which screen,
        and for the shared inputs which logical view (the ChatInput is one
        widget reused by every task view; the new-task editor serves both
        creation and held-task editing)."""
        from .inputs import ChatInput, DepsInput, NewTaskInput

        if isinstance(target, ChatInput):
            return ("chat", self._app.viewed_id)
        if isinstance(target, (NewTaskInput, DepsInput)):
            return ("new-task", self._app._editing_task_id)
        return ("other", None)

    def voice_context_valid(self, target: Any, context: tuple[str, Any]) -> bool:
        """Re-validate at insert time (runs on the UI thread): the target
        is still mounted, visible (no ``hidden`` class on it or an
        ancestor), shown by the app's active screen, and the context
        snapshot still matches the current state."""
        assert isinstance(target, TextArea)
        if not target.is_mounted:
            return False
        try:
            if target.screen is not self._app.screen:
                return False
        except Exception:  # noqa: BLE001 - not mounted / no screen stack
            return False
        node: DOMNode | None = target
        while node is not None:
            if node.has_class("hidden"):
                return False
            node = node.parent
        kind, token = context
        if kind == "chat":
            return self._app.viewed_id == token
        if kind == "new-task":
            return self._app._new_task_mode and self._app._editing_task_id == token
        return True

    def voice_insert_text(self, target: Any, text: str) -> None:
        """Insert the recognized text at the cursor of the field that
        started the recording (never sends it).

        Prefix-spacing detection is best-effort (a cosmetic failure
        never blocks the insertion); the insert itself propagates
        failures to the state machine (which turns them into a toast)."""
        prefix = ""
        try:
            row, col = target.cursor_location
            line = str(target.get_line(row))
            if col > 0 and not line[col - 1].isspace():
                prefix = " "
        except Exception:  # noqa: BLE001 - spacing is cosmetic
            prefix = ""
        target.insert(f"{prefix}{text.strip()}")
