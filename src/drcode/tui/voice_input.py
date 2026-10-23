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
- recording → idle (ESC): the recording is discarded.
- any failure (no config, no sounddevice, microphone busy, network) is a
  readable toast; the state machine resets to idle.

Everything audio/network runs in short-lived daemon worker threads (the
tool_runtime pattern); the only UI-thread work is inserting the text and
showing toasts (via call_from_thread).
"""

from __future__ import annotations

import enum
import sys
from typing import TYPE_CHECKING

from textual.widgets import TextArea

from .. import voice
from ..voice import (
    SOUNDDEVICE_AVAILABLE,
    VOICE_EXTRA_HINT,
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


class VoiceInputController:
    """Toggle-style push-to-talk: first press starts, second stops and
    inserts; ESC cancels. Created by DrCodeApp (``app.voice_input``)."""

    def __init__(self, app: DrCodeApp) -> None:
        self._app = app
        self._state = _State.IDLE
        self._recorder: MicrophoneRecorder | None = None
        # The field whose cursor receives the recognized text.
        self._target: TextArea | None = None

    @property
    def active(self) -> bool:
        """A recording or transcription is in flight (drives the ESC
        handling of the input fields)."""
        return self._state is not _State.IDLE

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
        text (called on the UI thread)."""
        if self._state is _State.TRANSCRIBING:
            self._notify("Still recognizing the previous utterance…")
            return
        if self._state is _State.RECORDING:
            self._state = _State.TRANSCRIBING
            self._notify("● распознаю…")
            self._transcribe()
            return
        self._start(target)

    def cancel(self) -> bool:
        """ESC while a recording runs: drop it. True when something was
        cancelled (the caller then swallows the ESC)."""
        if self._state is not _State.RECORDING:
            return False
        recorder, self._recorder = self._recorder, None
        self._state = _State.IDLE
        self._target = None
        if recorder is not None:
            recorder.discard()
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
                f"sounddevice is not available{reason} — {VOICE_EXTRA_HINT}",
                error=True,
            )
            return
        recorder = MicrophoneRecorder()
        self._recorder = recorder
        self._target = target
        self._state = _State.RECORDING
        _thread_start(self._open_microphone, recorder)

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
        cursor — all in a worker thread; only the insertion is UI work."""
        recorder = self._recorder
        target = self._target
        self._recorder = None
        self._target = None
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
        started the recording (never sends it)."""
        self._state = _State.IDLE
        if target is None:
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
        self._state = _State.IDLE
        try:
            self._app.call_from_thread(self._notify, message, error=True)
        except Exception:  # noqa: BLE001 - app may be shutting down
            self._notify(message, error=True)


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
