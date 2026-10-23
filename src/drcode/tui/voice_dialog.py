"""Voice settings dialog (ASR dictation + TTS announcements).

Opened from the Settings palette ("Voice…"). The form collects:

- the OpenAI-compatible audio-API base URL (default:
  ``https://api-copilot.x5.ru/v1`` — deliberately NOT the chat provider's
  base URL, gateways serve audio on their own route),
- the API key (pre-filled from the user-catalog provider whose host
  matches the audio URL; can be overridden),
- the ASR model id (full id, e.g. ``x5-asr/x5-asr-large``),
- the TTS model id + speaker voice,
- three checkboxes: which events to voice-announce (task completed,
  tester failure, empty queue).

"Check" probes ``GET /models`` in a worker thread. Save persists everything
into the project settings (``<project>/.drcode/settings.json``); ESC — close
without saving. Everything empty = the voice features are off.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, Switch

from ..voice import DEFAULT_TTS_VOICE, DEFAULT_VOICE_API_URL, VoiceClient, VoiceConfig

_VOICE_URLS = {
    "transcriptions": "/audio/transcriptions",
    "speech": "/audio/speech",
}


@dataclass
class VoiceSetup:
    """Values collected by the dialog (handed back to the app on Save)."""

    api_url: str
    api_key: str
    asr_model: str
    tts_model: str
    tts_voice: str
    tts_completed: bool
    tts_tester_failed: bool
    tts_queue_empty: bool


class VoiceSettingsDialog(ModalScreen[VoiceSetup | None]):
    """Modal form: audio API connection, ASR/TTS models, announcement
    checkboxes, connection check, Save / ESC-to-cancel."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Close", priority=True),
    ]

    DEFAULT_CSS = """
    VoiceSettingsDialog {
        align: center middle;
    }
    #voice-dialog {
        width: 78;
        max-width: 95%;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
        overflow-y: auto;
    }
    #voice-title { color: $text; margin-bottom: 1; text-style: bold; }
    #voice-note { color: $text-disabled; }
    .voice-label {
        width: auto;
        margin-top: 1;
        color: $text-muted;
        text-style: bold;
    }
    #voice-url, #voice-key, #voice-asr, #voice-tts-model, #voice-tts-voice {
        height: 3;
    }
    .voice-row { height: auto; }
    .voice-row Switch { margin-right: 1; }
    #voice-tts-voice { width: 30; }
    #voice-check-row { height: auto; margin-top: 1; }
    #voice-check { width: auto; margin-right: 1; }
    #voice-status {
        height: auto;
        min-height: 1;
        margin-top: 1;
        color: $text-muted;
    }
    #voice-status.ok { color: $success; }
    #voice-status.error { color: $error; }
    #voice-buttons {
        dock: bottom;
        height: auto;
        margin-top: 1;
        background: $panel;
    }
    #voice-save { width: auto; }
    #voice-hint {
        width: 1fr;
        padding: 1 0 0 1;
        color: $text-disabled;
        text-align: right;
    }
    """

    def __init__(
        self,
        *,
        api_url: str = "",
        api_key: str = "",
        catalog_api_key: str = "",
        asr_model: str = "",
        tts_model: str = "",
        tts_voice: str = "",
        tts_completed: bool = False,
        tts_tester_failed: bool = False,
        tts_queue_empty: bool = False,
    ) -> None:
        super().__init__()
        self._initial = {
            "api_url": api_url or DEFAULT_VOICE_API_URL,
            "api_key": api_key or catalog_api_key,
            "asr_model": asr_model,
            "tts_model": tts_model,
            "tts_voice": tts_voice,
        }
        self._initial_tts = {
            "completed": tts_completed,
            "tester_failed": tts_tester_failed,
            "queue_empty": tts_queue_empty,
        }
        self._checking = False

    def compose(self) -> ComposeResult:
        with Vertical(id="voice-dialog"):
            yield Static("Voice", id="voice-title")
            yield Static(
                "Speech recognition (dictation into the input fields, "
                "push-to-talk) and spoken announcements, via an "
                "OpenAI-compatible audio API. Leave everything empty to "
                "keep the voice features off.",
                id="voice-note",
            )
            yield Static("Audio API base URL", classes="voice-label")
            yield Input(
                placeholder=DEFAULT_VOICE_API_URL,
                value=self._initial["api_url"],
                id="voice-url",
            )
            yield Static("API key", classes="voice-label")
            yield Input(
                placeholder="reused from the provider catalog by host",
                value=self._initial["api_key"],
                password=True,
                id="voice-key",
            )
            yield Static("ASR model (full id)", classes="voice-label")
            yield Input(
                placeholder="x5-asr/x5-asr-large",
                value=self._initial["asr_model"],
                id="voice-asr",
            )
            yield Static("TTS model (full id)", classes="voice-label")
            yield Input(
                placeholder="x5-tts/tts-1",
                value=self._initial["tts_model"],
                id="voice-tts-model",
            )
            yield Static("TTS voice", classes="voice-label")
            yield Input(
                placeholder=DEFAULT_TTS_VOICE,
                value=self._initial["tts_voice"],
                id="voice-tts-voice",
            )
            yield Static("Announce (TTS)", classes="voice-label")
            with Horizontal(classes="voice-row"):
                yield Switch(
                    value=self._initial_tts["completed"], id="voice-tts-completed"
                )
                yield Static("task completed", classes="voice-label")
            with Horizontal(classes="voice-row"):
                yield Switch(
                    value=self._initial_tts["tester_failed"],
                    id="voice-tts-tester-failed",
                )
                yield Static(
                    "tester failure (bug / back to queue)", classes="voice-label"
                )
            with Horizontal(classes="voice-row"):
                yield Switch(
                    value=self._initial_tts["queue_empty"],
                    id="voice-tts-queue-empty",
                )
                yield Static("queue empty", classes="voice-label")
            with Horizontal(id="voice-check-row"):
                yield Button("Check", variant="default", id="voice-check")
                yield Static("", id="voice-status")
            with Horizontal(id="voice-buttons"):
                yield Button("Save", variant="primary", id="voice-save")
                yield Static("esc — close without saving", id="voice-hint")

    def on_mount(self) -> None:
        self.query_one("#voice-url", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    # -- values -----------------------------------------------------------------

    def _config(self) -> VoiceConfig:
        return VoiceConfig(
            base_url=self.query_one("#voice-url", Input).value.strip(),
            api_key=self.query_one("#voice-key", Input).value.strip(),
            asr_model=self.query_one("#voice-asr", Input).value.strip(),
            tts_model=self.query_one("#voice-tts-model", Input).value.strip(),
            tts_voice=(
                self.query_one("#voice-tts-voice", Input).value.strip()
                or DEFAULT_TTS_VOICE
            ),
        )

    def _collect(self) -> VoiceSetup:
        return VoiceSetup(
            api_url=self.query_one("#voice-url", Input).value.strip(),
            api_key=self.query_one("#voice-key", Input).value.strip(),
            asr_model=self.query_one("#voice-asr", Input).value.strip(),
            tts_model=self.query_one("#voice-tts-model", Input).value.strip(),
            tts_voice=self.query_one("#voice-tts-voice", Input).value.strip(),
            tts_completed=self.query_one("#voice-tts-completed", Switch).value,
            tts_tester_failed=self.query_one("#voice-tts-tester-failed", Switch).value,
            tts_queue_empty=self.query_one("#voice-tts-queue-empty", Switch).value,
        )

    # -- connection check ---------------------------------------------------------

    @on(Button.Pressed, "#voice-check")
    async def _check_clicked(self) -> None:
        """Probe ``GET /models`` in a worker thread (asyncio.to_thread —
        the network call must never freeze the modal)."""
        config = self._config()
        status = self.query_one("#voice-status", Static)
        status.remove_class("ok", "error")
        if not config.base_url:
            status.add_class("error")
            status.update("✗ No base URL given.")
            return
        if self._checking:
            return
        self._checking = True
        status.update("checking…")
        try:
            ok, detail = await asyncio.to_thread(self._probe, config)
        finally:
            self._checking = False
        if not self.is_running:
            return
        status.remove_class("ok", "error")
        status.add_class("ok" if ok else "error")
        hint = "" if ok else f" (endpoints: {', '.join(_VOICE_URLS.values())})"
        status.update(f"{'✓' if ok else '✗'} {detail}{hint}")

    @staticmethod
    def _probe(config: VoiceConfig) -> tuple[bool, str]:
        try:
            return True, VoiceClient(config).check()
        except Exception as e:  # noqa: BLE001 - the probe reports, never raises
            return False, str(e)

    # -- save ---------------------------------------------------------------------

    @on(Button.Pressed, "#voice-save")
    def _save_clicked(self) -> None:
        self.dismiss(self._collect())
