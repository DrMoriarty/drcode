"""Push-to-talk voice dictation controller (the Ctrl+B hotkey in the input
fields).

Push-to-talk state machine, one instance per app:

- idle → recording: a worker thread opens the microphone
  (:class:`drcode.voice.MicrophoneRecorder`); the status toast appears via
  call_from_thread — the event loop never touches PortAudio.
- recording → idle (second press): the same (exclusive) worker thread stops
  the recorder, uploads the WAV to the ASR endpoint and inserts the
  recognized text at the cursor of the field that started the recording —
  the user edits before sending.
- recording → idle (ESC): the recording is discarded (stream closed on a
  worker thread, never on the UI thread).
- any failure (no config, no sounddevice, microphone busy, network) is a
  readable toast; the state machine resets to idle.

While a recording runs the persistent indicator lives in the header status
line (``_render_model_status`` re-renders it on every state change and on
the 1 s poll) — the status toast auto-dismisses after a few seconds and
must not be the only visible sign that dictation is still active.

Everything audio/network runs in short-lived daemon worker threads (the
tool_runtime pattern); the only UI-thread work is inserting the text and
showing toasts (via call_from_thread).
"""

from __future__ import annotations

import enum
import sys
import threading
import time
from typing import TYPE_CHECKING

from textual.dom import DOMNode
from textual.widgets import TextArea

from .. import voice
from ..voice import (
    SOUNDDEVICE_AVAILABLE,
    VOICE_UNAVAILABLE_HINT,
    MicrophoneRecorder,
    VoiceClient,
    VoiceError,
    config_from_settings,
)

if TYPE_CHECKING:
    from .app import DrCodeApp


class _State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


# Terminal auto-repeat sends a burst of synthetic presses while Ctrl+B is
# held (a natural push-to-talk reflex); Textual's Key event carries no
# repeat marker, so the toggle is guarded in time instead. Auto-repeat
# intervals (~15–90 ms) are far below the debounce window, and the first
# repeat — arriving after the initial delay, which can exceed it — is
# caught by the minimum-recording guard.
_TOGGLE_DEBOUNCE = 0.3  # s between distinct toggle presses
_MIN_RECORDING = 0.7  # a stop sooner than this cannot be real speech


class VoiceInputController:
    """Toggle-style push-to-talk: first press starts, second stops and
    inserts; ESC cancels. Created by DrCodeApp (``app.voice_input``)."""

    def __init__(self, app: DrCodeApp) -> None:
        self._app = app
        self._state = _State.IDLE
        self._recorder: MicrophoneRecorder | None = None
        # The field whose cursor receives the recognized text.
        self._target: TextArea | None = None
        # Context snapshot taken together with the target (see
        # _snapshot_context): the recognized text is only ever inserted
        # while the app is still in the same context the dictation was
        # started in (same screen, same viewed task / same new-task form).
        self._context: tuple[str, object] | None = None
        self._last_toggle = 0.0  # auto-repeat guard stamp (monotonic)
        self._recording_since = 0.0

    @property
    def active(self) -> bool:
        """A recording or transcription is in flight (drives the ESC
        handling of the input fields)."""
        return self._state is not _State.IDLE

    @property
    def indicator(self) -> str | None:
        """Persistent status text for the header line while busy (None
        when idle) — the auto-dismissing toast alone left any utterance
        longer than a few seconds with no visible sign that dictation
        was still running."""
        if self._state is _State.RECORDING:
            return "● запись…"
        if self._state is _State.TRANSCRIBING:
            return "● распознаю…"
        return None

    def _notify(self, message: str, *, error: bool = False) -> None:
        try:
            self._app.notify(
                message,
                title="Voice",
                severity="error" if error else "information",
                timeout=6 if error else 4,
            )
        except Exception as e:  # noqa: BLE001 - toasts are best-effort
            print(f"drcode voice: notify failed: {e}", file=sys.stderr)

    def toggle(self, target: TextArea) -> None:
        """The hotkey action: start recording, or stop it and insert the
        text (called on the UI thread).

        Guarded against terminal auto-repeat: holding Ctrl+B produces
        synthetic presses indistinguishable from real ones, and without
        the guard the first repeat would stop the recording it just
        started (a tiny fragment, flickering toasts). Every call —
        accepted or swallowed — refreshes the stamp, so a held key can
        never flip the state machine no matter how long it is held."""
        now = time.monotonic()
        repeated = now - self._last_toggle < _TOGGLE_DEBOUNCE
        self._last_toggle = now
        if repeated:
            return
        if self._state is _State.TRANSCRIBING:
            self._notify("Still recognizing the previous utterance…")
            return
        if self._state is _State.RECORDING:
            if now - self._recording_since < _MIN_RECORDING:
                # A stop within the first beat slipped past the debounce
                # (initial auto-repeat delay can exceed it) — it is not
                # intent, and must not cut the recording to a fragment.
                return
            self._state = _State.TRANSCRIBING
            self._notify("● распознаю…")
            self._sync_indicator()
            self._transcribe()
            return
        self._start(target)

    def _sync_indicator(self) -> None:
        """Re-render the persistent header indicator right away (the 1 s
        activity poll would refresh it anyway). Safe from any thread."""
        try:
            if threading.get_ident() == self._app._thread_id:
                self._app._render_model_status()
            else:
                self._app.call_from_thread(self._app._render_model_status)
        except Exception as e:  # noqa: BLE001 - the indicator is cosmetic
            print(f"drcode voice: indicator failed: {e}", file=sys.stderr)

    def cancel(self, *, silent: bool = False) -> bool:
        """ESC while a recording runs: drop it. True when something was
        cancelled (the caller then swallows the ESC).

        ``silent=True`` is for the automatic navigation hooks (task
        switch, new-task form leave, modal push): the discard is part of
        the navigation the user just performed, no extra toast.

        The stream close is offloaded to a worker thread — the UI thread
        never touches PortAudio. A TRANSCRIBING phase cannot be cancelled
        this way (no stream left); the insert-time context validation
        drops its text instead."""
        if self._state is not _State.RECORDING:
            return False
        recorder, self._recorder = self._recorder, None
        self._state = _State.IDLE
        self._target = None
        self._context = None
        if recorder is not None:
            _thread_start(_discard_recorder, recorder)
        self._sync_indicator()
        self._notify("Запись отменена")
        return True

    # -- phases ------------------------------------------------------------------

    def _start(self, target: TextArea) -> None:
        """Config/availability checks, then open the microphone in a
        worker thread."""
        settings = self._app.settings
        try:
            config = config_from_settings(settings)
        except Exception as e:  # noqa: BLE001 - config must never break typing
            self._notify(f"Voice settings unreadable: {e}", error=True)
            return
        if config is None or not config.asr_model:
            self._notify(
                "Voice dictation is not configured — Settings → Voice… "
                "(audio API base URL + ASR model).",
                error=True,
            )
            return
        if not SOUNDDEVICE_AVAILABLE:
            reason = (
                f" ({voice._SOUNDDEVICE_ERROR})" if voice._SOUNDDEVICE_ERROR else ""
            )
            self._notify(
                f"sounddevice is not available{reason} — {VOICE_UNAVAILABLE_HINT}",
                error=True,
            )
            return
        recorder = MicrophoneRecorder()
        self._recorder = recorder
        self._target = target
        self._context = self._snapshot_context(target)
        self._state = _State.RECORDING
        self._recording_since = time.monotonic()
        self._sync_indicator()
        _thread_start(self._open_microphone, recorder)

    def _snapshot_context(self, target: TextArea) -> tuple[str, object]:
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

    def _context_valid(self, target: TextArea) -> bool:
        """Re-validate at insert time (runs on the UI thread): the target
        is still mounted, visible (no ``hidden`` class on it or an
        ancestor), shown by the app's active screen, and the context
        snapshot still matches the current state."""
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
        if self._context is None:
            return True
        kind, token = self._context
        if kind == "chat":
            return self._app.viewed_id == token
        if kind == "new-task":
            return self._app._new_task_mode and self._app._editing_task_id == token
        return True

    def _open_microphone(self, recorder: MicrophoneRecorder) -> None:
        """Worker thread: open the PortAudio stream (a slow syscall — the
        event loop never waits for it), confirm the recording afterwards."""
        try:
            recorder.start()
        except VoiceError as e:
            self._fail(str(e))
            return
        if self._state is not _State.RECORDING:
            return
        try:
            self._app.call_from_thread(
                self._notify, "● запись…  Ctrl+B — готово, ESC — отмена"
            )
        except Exception as e:  # noqa: BLE001 - app may be shutting down
            print(f"drcode voice: status toast failed: {e}", file=sys.stderr)

    def _transcribe(self) -> None:
        """Stop the recorder, upload the WAV, insert the text at the
        cursor — all in a worker thread; only the insertion is UI work.

        The target and the context snapshot stay on the controller until
        the insert: ``_insert`` re-validates them against the CURRENT app
        state (they are not stale-cleared here), so a context change
        while ASR runs drops the text instead of inserting it."""
        recorder = self._recorder
        target = self._target
        self._recorder = None
        _thread_start(self._recognize, recorder, target)

    def _recognize(self, recorder: MicrophoneRecorder, target: TextArea | None) -> None:
        """Worker thread: close the stream, upload the WAV to the ASR
        endpoint and hop back into the app thread to insert the text."""
        try:
            wav = recorder.stop()
            config = config_from_settings(self._app.settings)
            if config is None or not config.asr_model:
                raise VoiceError("Voice dictation is not configured")
            text = VoiceClient(config).transcribe(wav)
        except VoiceError as e:
            self._fail(str(e))
            return
        except Exception as e:  # noqa: BLE001 - a dictation never breaks the app
            self._fail(f"{type(e).__name__}: {e}")
            return
        try:
            self._app.call_from_thread(self._insert, target, text)
        except Exception as e:  # noqa: BLE001 - app may be shutting down
            print(f"drcode voice: insert failed: {e}", file=sys.stderr)

    def _insert(self, target: TextArea | None, text: str) -> None:
        """Insert the recognized text at the cursor of the field that
        started the recording (never sends it).

        The dictation belongs to the context it was started in: if the
        user navigated away while ASR was in flight (another task, the
        new-task form, a modal dialog, a hidden target), the text is
        dropped with a toast instead of landing in a stale field."""
        self._state = _State.IDLE
        self._sync_indicator()
        if target is None:
            self._target = None
            self._context = None
            return
        valid = self._context_valid(target)
        self._target = None
        self._context = None
        if not valid:
            self._notify("Dictation discarded — context changed")
            return
        prefix = ""
        try:
            row, col = target.cursor_location
            line = str(target.get_line(row))
            if col > 0 and not line[col - 1].isspace():
                prefix = " "
        except Exception:  # noqa: BLE001 - spacing is cosmetic
            prefix = ""
        try:
            target.insert(f"{prefix}{text.strip()}")
        except Exception as e:  # noqa: BLE001 - insertion is best-effort
            self._notify(f"Could not insert the text: {e}", error=True)

    def _fail(self, message: str) -> None:
        """Abort the state machine with a readable error (from any
        thread)."""
        self._recorder = None
        self._target = None
        self._context = None
        self._state = _State.IDLE
        self._sync_indicator()
        try:
            self._app.call_from_thread(self._notify, message, error=True)
        except Exception:  # noqa: BLE001 - app may be shutting down
            self._notify(message, error=True)


def _discard_recorder(recorder: MicrophoneRecorder) -> None:
    """Worker body for the ESC path: close the PortAudio stream off the
    UI thread (cancel never blocks the event loop on audio syscalls)."""
    recorder.discard()


def _thread_start(func: object, *args: object) -> None:
    """A short-lived daemon worker (open-mic / transcribe phases): the
    audio and network calls must never touch the UI event loop (same
    pattern as tool_runtime's worker threads)."""
    import threading

    def wrapper() -> None:
        try:
            func(*args)  # type: ignore[operator]
        except Exception as e:  # noqa: BLE001 - a dictation never breaks the app
            print(f"drcode voice: {type(e).__name__}: {e}", file=sys.stderr)

    threading.Thread(target=wrapper, daemon=True, name="drcode-voice").start()
