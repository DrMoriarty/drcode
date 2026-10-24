"""Voice subsystem tests (stubs only — no real API, no audio devices).

- VoiceConfig resolution from project settings (empty = off, api-key host
  fallback to the user catalog),
- VoiceClient against httpx.MockTransport (request shape, error mapping),
- TtsSpeaker queueing/serialization/off-switch with injected synth+play,
- Notifier voice hooks (checkbox filtering, user-origin silence,
  bug-report hook), over a FIFO speaker stub,
- TaskManager creation listeners (origin propagation),
- WAV assembly of MicrophoneRecorder with a faked PortAudio stream.
"""

from __future__ import annotations

import io
import threading
import time
import wave
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from drcode.notifications import Notifier
from drcode.providers import Provider, UserModelsConfig, UserModelsStore
from drcode.settings import ProjectSettings
from drcode.voice import (
    DEFAULT_TTS_VOICE,
    DEFAULT_VOICE_API_URL,
    MicrophoneRecorder,
    TtsSpeaker,
    VoiceClient,
    VoiceConfig,
    VoiceError,
    catalog_api_key_for_host,
    config_from_settings,
    resolve_api_key,
)

# A minimal valid PCM16 WAV file (one sample) for the request-shape tests.
_WAV_BYTES = (
    b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x02\x00\x00\x00\x00\x00"
)


def _mock_transport(
    handler: Any,
) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# -- config resolution ----------------------------------------------------------


def test_config_off_when_url_empty() -> None:
    assert config_from_settings(ProjectSettings()) is None


def test_config_resolved_with_defaults() -> None:
    settings = ProjectSettings(
        voice_api_url=DEFAULT_VOICE_API_URL,
        voice_asr_model="x5-asr/x5-asr-large",
        voice_tts_model="x5-tts/tts-1",
    )
    config = config_from_settings(settings)
    assert config is not None
    assert config.base_url == DEFAULT_VOICE_API_URL
    assert config.asr_model == "x5-asr/x5-asr-large"
    assert config.tts_voice == DEFAULT_TTS_VOICE


def test_api_key_host_fallback_from_catalog(user_config_dir: Any) -> None:
    # A provider on the same host carries the token: the audio config
    # reuses it without a second key.
    store = UserModelsStore()
    store.save(
        UserModelsConfig(
            providers=[
                Provider(
                    name="x5",
                    base_url="https://api-copilot.x5.ru/aigw/v1",
                    api_key="sekret-token",
                    models=[],
                )
            ]
        )
    )
    settings = ProjectSettings(voice_api_url="https://api-copilot.x5.ru/v1")
    assert resolve_api_key(settings, settings.voice_api_url) == "sekret-token"
    config = config_from_settings(settings)
    assert config is not None
    assert config.api_key == "sekret-token"


def test_api_key_explicit_wins(user_config_dir: Any) -> None:
    store = UserModelsStore()
    store.save(
        UserModelsConfig(
            providers=[
                Provider(
                    name="x5",
                    base_url="https://api-copilot.x5.ru/aigw/v1",
                    api_key="catalog-token",
                    models=[],
                )
            ]
        )
    )
    settings = ProjectSettings(
        voice_api_url="https://api-copilot.x5.ru/v1",
        voice_api_key="own-token",
    )
    assert resolve_api_key(settings, settings.voice_api_url) == "own-token"


def test_api_key_no_host_match(user_config_dir: Any) -> None:
    store = UserModelsStore()
    store.save(
        UserModelsConfig(
            providers=[
                Provider(name="other", base_url="https://example.com/v1", models=[])
            ]
        )
    )
    assert catalog_api_key_for_host("https://api-copilot.x5.ru/v1") == ""


# -- VoiceClient (httpx.MockTransport) -------------------------------------------


def _client(handler: Any, config: VoiceConfig | None = None) -> VoiceClient:
    return VoiceClient(
        config
        or VoiceConfig(
            base_url="https://audio.local/v1",
            api_key="key-1",
            asr_model="x5-asr/x5-asr-large",
            tts_model="x5-tts/tts-1",
        ),
        transport=_mock_transport(handler),
    )


def test_transcribe_request_and_answer() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        body = request.content
        seen["wav"] = _WAV_BYTES in body
        # The ASR model id travels in the multipart form.
        seen["model"] = b"x5-asr/x5-asr-large" in body
        return httpx.Response(200, json={"text": " Привет, мир. ", "usage": None})

    text = _client(handler).transcribe(_WAV_BYTES)
    assert text == "Привет, мир."
    assert seen["url"] == "https://audio.local/v1/audio/transcriptions"
    assert seen["auth"] == "Bearer key-1"
    assert seen["wav"] is True
    assert seen["model"] is True


def test_transcribe_http_error_is_voice_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "bad key"})

    with pytest.raises(VoiceError, match="401"):
        _client(handler).transcribe(_WAV_BYTES)


def test_transcribe_empty_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "  "})

    with pytest.raises(VoiceError, match="no text"):
        _client(handler).transcribe(_WAV_BYTES)


def test_transcribe_requires_asr_model() -> None:
    client = VoiceClient(VoiceConfig(base_url="https://audio.local/v1", api_key=""))
    with pytest.raises(VoiceError, match="ASR"):
        client.transcribe(_WAV_BYTES)


def test_speech_request_and_answer() -> None:
    import json as _json

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = _json.loads(request.read())
        return httpx.Response(200, content=_WAV_BYTES)

    data = _client(handler).speech("Проверка")
    assert data == _WAV_BYTES
    assert seen["url"] == "https://audio.local/v1/audio/speech"
    assert seen["auth"] == "Bearer key-1"
    assert seen["body"] == {
        "model": "x5-tts/tts-1",
        "input": "Проверка",
        "voice": DEFAULT_TTS_VOICE,
        "response_format": "wav",
    }


def test_speech_http_error_is_voice_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Invalid model name"})

    with pytest.raises(VoiceError, match="404"):
        _client(handler).speech("Проверка")


def test_check_models_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://audio.local/v1/models"
        return httpx.Response(200, json={"data": [{"id": "x5-asr/x5-asr-large"}]})

    assert "1" in _client(handler).check()


def test_check_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(VoiceError, match="ConnectError"):
        _client(handler).check()


def test_production_requests_use_truststore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the test transport the client must verify against the system
    trust store (corporate CAs are absent from certifi)."""
    import ssl as ssl_mod

    seen: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        seen["verify"] = kwargs.get("verify")
        return httpx.Response(
            200,
            json={"text": "ok"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    VoiceClient(
        VoiceConfig(
            base_url="https://audio.local/v1",
            api_key="key-1",
            asr_model="x5-asr/x5-asr-large",
        )
    ).transcribe(_WAV_BYTES)
    verify = seen["verify"]
    try:
        import truststore  # noqa: F401

        assert isinstance(verify, ssl_mod.SSLContext)
    except ImportError:
        assert verify is True


# -- TtsSpeaker -------------------------------------------------------------------


class _FakeSynth:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.requests: list[str] = []

    def speech(self, text: str) -> bytes:
        self.requests.append(text)
        time.sleep(self.delay)
        return b"WAV:" + text.encode()


class _FakePlayer:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self._lock = threading.Lock()

    def __call__(self, data: bytes) -> None:
        with self._lock:
            self.played.append(data)


def _voice_settings(**tts: Any) -> ProjectSettings:
    return ProjectSettings(
        voice_api_url="https://audio.local/v1",
        voice_tts_model="x5-tts/tts-1",
        **tts,
    )


def test_speaker_off_is_silent() -> None:
    synth, player = _FakeSynth(), _FakePlayer()
    speaker = TtsSpeaker(
        ProjectSettings,  # no url → off
        client_factory=lambda c: synth,
        play=player,
    )
    speaker.speak("не должно прозвучать")
    time.sleep(0.1)
    assert synth.requests == []
    assert player.played == []


def test_speaker_respects_checkbox_via_caller() -> None:
    # The checkbox filtering happens at the notifier level; the speaker
    # itself only checks that a TTS model is configured.
    synth, player = _FakeSynth(), _FakePlayer()
    speaker = TtsSpeaker(
        lambda: _voice_settings(),
        client_factory=lambda c: synth,
        play=player,
    )
    speaker.speak("проверка")
    deadline = time.monotonic() + 5
    while len(player.played) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert player.played == [b"WAV:" + "проверка".encode()]


def test_speaker_serializes_and_caps_queue() -> None:
    synth, player = _FakeSynth(delay=0.15), _FakePlayer()
    speaker = TtsSpeaker(
        lambda: _voice_settings(),
        client_factory=lambda c: synth,
        play=player,
    )
    for i in range(10):
        speaker.speak(f"фраза {i}")
    # The burst never stacks up: at most the phrase being played plus
    # MAX_PENDING queued ones are ever synthesized.
    deadline = time.monotonic() + 3
    while not synth.requests and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(1.0)
    assert len(synth.requests) <= 1 + TtsSpeaker.MAX_PENDING


def test_speaker_error_is_reported_not_raised() -> None:
    errors: list[str] = []

    class _Boom:
        def speech(self, text: str) -> bytes:
            raise VoiceError("TTS failed: HTTP 500")

    speaker = TtsSpeaker(
        lambda: _voice_settings(),
        client_factory=lambda c: _Boom(),
        play=_FakePlayer(),
        on_error=errors.append,
    )
    speaker.speak("упс")
    deadline = time.monotonic() + 5
    while not errors and time.monotonic() < deadline:
        time.sleep(0.01)
    assert errors == ["TTS failed: HTTP 500"]


# -- Notifier voice hooks ------------------------------------------------------------


class _SpeakerStub:
    def __init__(self) -> None:
        self.phrases: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, text: str) -> None:
        with self._lock:
            self.phrases.append(text)


def _notifier(
    settings: ProjectSettings, speaker: Callable[[str], None] | None = None
) -> Notifier:
    return Notifier(
        lambda: settings,
        project="p",
        speaker=speaker,
    )


def _task(status: Any = None, **kwargs: Any) -> Any:
    from drcode.models import Task

    return Task(title="Починить кнопку", **{"status": status, **kwargs})


def test_notifier_completion_voiced() -> None:
    from drcode.models import TaskStatus, VerificationStatus

    settings = _voice_settings(voice_tts_completed=True)
    speaker = _SpeakerStub()
    notifier = _notifier(settings, speaker)
    task = _task(TaskStatus.COMPLETED, verification=VerificationStatus.PASSED)
    notifier.task_status(task, TaskStatus.TESTING, TaskStatus.COMPLETED)
    assert len(speaker.phrases) == 1
    assert "Починить кнопку" in speaker.phrases[0]
    assert "завершена" in speaker.phrases[0]


def test_notifier_failed_verdict_under_tester_failed_checkbox() -> None:
    from drcode.models import TaskStatus, VerificationStatus

    # Only the tester-failure checkbox is on: the FAILED completion voices
    # as a failure, not as a completion.
    settings = _voice_settings(voice_tts_tester_failed=True)
    speaker = _SpeakerStub()
    notifier = _notifier(settings, speaker)
    task = _task(TaskStatus.COMPLETED, verification=VerificationStatus.FAILED)
    notifier.task_status(task, TaskStatus.TESTING, TaskStatus.COMPLETED)
    assert len(speaker.phrases) == 1
    assert "с ошибкой" in speaker.phrases[0]


def test_notifier_requeue_from_testing_voiced() -> None:
    from drcode.models import TaskStatus

    settings = _voice_settings(voice_tts_tester_failed=True)
    speaker = _SpeakerStub()
    notifier = _notifier(settings, speaker)
    task = _task(TaskStatus.QUEUED)
    notifier.task_status(task, TaskStatus.TESTING, TaskStatus.QUEUED)
    assert len(speaker.phrases) == 1
    assert "возвращена в разработку" in speaker.phrases[0]


def test_notifier_user_origin_silent() -> None:
    from drcode.models import TaskStatus

    settings = _voice_settings(voice_tts_completed=True, voice_tts_tester_failed=True)
    speaker = _SpeakerStub()
    notifier = _notifier(settings, speaker)
    task = _task(TaskStatus.COMPLETED)
    notifier.task_status(task, TaskStatus.TESTING, TaskStatus.COMPLETED, origin="user")
    task2 = _task(TaskStatus.QUEUED)
    notifier.task_status(task2, TaskStatus.TESTING, TaskStatus.QUEUED, origin="user")
    assert speaker.phrases == []


def test_notifier_checkbox_off_silent() -> None:
    from drcode.models import TaskStatus

    settings = _voice_settings()  # models set, all checkboxes off
    speaker = _SpeakerStub()
    notifier = _notifier(settings, speaker)
    task = _task(TaskStatus.COMPLETED)
    notifier.task_status(task, TaskStatus.TESTING, TaskStatus.COMPLETED)
    notifier.pipeline_idle(blocked=False, detail="")
    notifier.bug_report_created(task, "system")
    assert speaker.phrases == []


def test_notifier_queue_empty_voiced_once() -> None:
    settings = _voice_settings(voice_tts_queue_empty=True)
    speaker = _SpeakerStub()
    notifier = _notifier(settings, speaker)
    notifier.pipeline_idle(blocked=False, detail="")
    notifier.pipeline_idle(blocked=False, detail="")  # dedup: no second phrase
    assert speaker.phrases == ["Очередь пуста — все задачи обработаны"]
    # The blocked state is never voiced.
    speaker.phrases.clear()
    notifier.pipeline_busy()
    notifier.pipeline_idle(blocked=True, detail="on hold: 1")
    assert speaker.phrases == []


def test_notifier_bug_report_created() -> None:
    from drcode.models import TaskKind, TaskStatus

    settings = _voice_settings(voice_tts_tester_failed=True)
    speaker = _SpeakerStub()
    notifier = _notifier(settings, speaker)
    bug = _task(TaskStatus.QUEUED, kind=TaskKind.BUG)
    notifier.bug_report_created(bug, "system")
    assert len(speaker.phrases) == 1
    assert "баг-репорт" in speaker.phrases[0]
    # A hand-created bug (origin "user") and a non-bug task stay silent.
    notifier.bug_report_created(bug, "user")
    feature = _task(TaskStatus.QUEUED, kind=TaskKind.FEATURE)
    notifier.bug_report_created(feature, "system")
    assert len(speaker.phrases) == 1


def test_notifier_broken_speaker_never_raises() -> None:
    from drcode.models import TaskStatus

    def boom(text: str) -> None:
        raise RuntimeError("no audio device")

    settings = _voice_settings(voice_tts_completed=True)
    notifier = _notifier(settings, boom)
    task = _task(TaskStatus.COMPLETED)
    notifier.task_status(task, TaskStatus.TESTING, TaskStatus.COMPLETED)  # no raise


# -- TUI integration (headless app) ----------------------------------------------


async def test_push_to_talk_binding_and_controller(project_root: Any) -> None:
    from drcode.tui.app import DrCodeApp

    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the first-start setup dialog
        assert app.voice_input is not None
        assert not app.voice_input.active
        # The new-task editor owns the ctrl+b binding (a TextArea stops
        # unhandled keys, so it cannot live on the app alone).
        assert app.new_task_input.has_focus
        bindings = app.screen.active_bindings
        assert "ctrl+b" in bindings
        # Without a voice config the hotkey stays a no-op: readable toast,
        # the controller stays idle (no microphone is ever opened).
        await pilot.press("ctrl+b")
        await pilot.pause()
        await pilot.pause()
        assert not app.voice_input.active


async def test_toggle_debounces_auto_repeat(project_root: Any) -> None:
    """Holding Ctrl+B sends a burst of synthetic presses (no repeat marker
    in Textual's Key event); the debounce must keep them from flipping the
    state machine — the first repeat used to stop the recording it just
    started."""
    from drcode.tui.app import DrCodeApp
    from drcode.tui.voice_input import _State

    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        controller = app.voice_input
        started: list[Any] = []
        controller._start = started.append  # type: ignore[method-assign]
        target = app.new_task_input
        controller.toggle(target)
        assert len(started) == 1
        for _ in range(10):
            controller.toggle(target)
        assert len(started) == 1
        assert controller._state is _State.IDLE


async def test_toggle_min_recording_guard(project_root: Any) -> None:
    """A stop arriving within the first beat of a recording (an initial
    auto-repeat delay longer than the debounce window) must not cut it to
    a fragment; a stop after a real recording transcribes."""
    from drcode.tui.app import DrCodeApp
    from drcode.tui.voice_input import _MIN_RECORDING, _State

    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        controller = app.voice_input
        target = app.new_task_input
        stopped: list[bool] = []
        controller._transcribe = lambda: stopped.append(True)  # type: ignore[method-assign]
        controller._state = _State.RECORDING
        controller._recording_since = time.monotonic()
        controller._last_toggle = 0.0
        controller.toggle(target)
        assert not stopped
        assert controller._state is _State.RECORDING
        controller._recording_since = time.monotonic() - (_MIN_RECORDING + 0.1)
        controller._last_toggle = 0.0
        controller.toggle(target)
        assert stopped == [True]
        assert controller._state is _State.TRANSCRIBING


async def test_header_shows_persistent_recording_indicator(project_root: Any) -> None:
    """The recording indicator lives in the header status line exactly as
    long as the dictation is busy — the toast auto-dismisses after a few
    seconds and must not be the only visible sign."""
    from drcode.tui.app import DrCodeApp
    from drcode.tui.voice_input import _State

    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        label = app.model_status
        controller = app.voice_input
        controller._state = _State.RECORDING
        app._render_model_status()
        assert "● запись…" in str(label.render())
        controller._state = _State.TRANSCRIBING
        app._render_model_status()
        assert "● распознаю…" in str(label.render())
        controller._state = _State.IDLE
        app._render_model_status()
        assert "● запись…" not in str(label.render())


async def test_voice_settings_dialog_saves(project_root: Any) -> None:
    from textual.widgets import Button, Input, Switch

    from drcode.settings import ProjectSettingsStore
    from drcode.tui.app import DrCodeApp
    from drcode.tui.voice_dialog import VoiceSettingsDialog

    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the first-start setup dialog
        app.action_change_voice()
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, VoiceSettingsDialog)
        assert dialog.query_one("#voice-url", Input).value == DEFAULT_VOICE_API_URL
        dialog.query_one("#voice-asr", Input).value = "x5-asr/x5-asr-large"
        dialog.query_one("#voice-tts-model", Input).value = "x5-tts/tts-1"
        dialog.query_one("#voice-tts-completed", Switch).value = True
        dialog.query_one("#voice-save", Button).press()
        await pilot.pause()
        assert app.settings.voice_asr_model == "x5-asr/x5-asr-large"
        assert app.settings.voice_api_key == ""  # untouched → catalog fallback
        stored = ProjectSettingsStore(project_root).load()
        assert stored.voice_asr_model == "x5-asr/x5-asr-large"
        assert stored.voice_tts_completed is True
        assert stored.voice_tts_tester_failed is False


async def test_voice_dialog_cancel_keeps_settings(project_root: Any) -> None:
    from drcode.tui.app import DrCodeApp
    from drcode.tui.voice_dialog import VoiceSettingsDialog

    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        app.action_change_voice()
        await pilot.pause()
        assert isinstance(app.screen, VoiceSettingsDialog)
        await pilot.press("escape")
        await pilot.pause()
        assert app.settings.voice_api_url == ""


# -- Dictation context safety ---------------------------------------------------------
#
# A dictation belongs to the context it was started in: a task switch, a
# new-task form enter/leave or a pushed modal must discard the in-flight
# text instead of inserting it into a stale field.


async def _running_app(project_root: Any) -> Any:
    from drcode.tui.app import DrCodeApp

    app = DrCodeApp(project_root)
    context = app.run_test()
    pilot = await context.__aenter__()
    await pilot.press("escape")  # skip the first-start setup dialog
    return app, pilot, context


def _arm_transcribing(app: Any, target: Any) -> None:
    """Fake the controller into TRANSCRIBING for `target` (no microphone,
    no network): the state `_insert` sees after a real ASR round-trip."""
    from drcode.tui.voice_input import _State

    vi = app.voice_input
    vi._state = _State.TRANSCRIBING
    vi._target = target
    vi._context = vi._snapshot_context(target)


async def test_dictation_discarded_after_task_switch(project_root: Any) -> None:
    app, pilot, context = await _running_app(project_root)
    try:
        t1 = app.manager.create("one", "m")
        t2 = app.manager.create("two", "m")
        app.view_task(t1.id)
        await pilot.pause()
        _arm_transcribing(app, app.chat_input)
        # The user switches to another task while ASR is in flight.
        app.view_task(t2.id)
        await pilot.pause()
        app.voice_input._insert(app.chat_input, "hello")
        await pilot.pause()
        assert app.chat_input.text == ""
        assert not app.voice_input.active
    finally:
        await context.__aexit__(None, None, None)


async def test_dictation_inserts_in_same_context(project_root: Any) -> None:
    app, pilot, context = await _running_app(project_root)
    try:
        task = app.manager.create("one", "m")
        app.view_task(task.id)
        await pilot.pause()
        _arm_transcribing(app, app.chat_input)
        app.voice_input._insert(app.chat_input, "hello")
        await pilot.pause()
        assert app.chat_input.text == "hello"
    finally:
        await context.__aexit__(None, None, None)


async def test_view_task_cancels_active_recording(project_root: Any) -> None:
    from drcode.tui.voice_input import _State
    from drcode.voice import MicrophoneRecorder

    app, pilot, context = await _running_app(project_root)
    try:
        t1 = app.manager.create("one", "m")
        t2 = app.manager.create("two", "m")
        app.view_task(t1.id)
        await pilot.pause()
        vi = app.voice_input
        vi._state = _State.RECORDING
        recorder = MicrophoneRecorder()
        vi._recorder = recorder
        vi._target = app.chat_input
        vi._context = vi._snapshot_context(app.chat_input)
        app.view_task(t2.id)
        await pilot.pause()
        assert not vi.active
        assert not recorder.recording
    finally:
        await context.__aexit__(None, None, None)


async def test_new_task_leave_discards_form_dictation(project_root: Any) -> None:
    app, pilot, context = await _running_app(project_root)
    try:
        app.action_new_task()
        await pilot.pause()
        _arm_transcribing(app, app.new_task_input)
        # Leaving the form hides the editor: the text must not land there.
        app.cancel_new_task()
        await pilot.pause()
        app.voice_input._insert(app.new_task_input, "hello")
        await pilot.pause()
        assert app.new_task_input.text == ""
        assert not app.voice_input.active
    finally:
        await context.__aexit__(None, None, None)


async def test_push_screen_cancels_active_recording(project_root: Any) -> None:
    from drcode.tui.voice_input import _State
    from drcode.voice import MicrophoneRecorder

    app, pilot, context = await _running_app(project_root)
    try:
        app.view_task(app.manager.create("one", "m").id)
        await pilot.pause()
        vi = app.voice_input
        vi._state = _State.RECORDING
        recorder = MicrophoneRecorder()
        vi._recorder = recorder
        vi._target = app.chat_input
        vi._context = vi._snapshot_context(app.chat_input)
        app.action_change_voice()  # pushes the voice settings dialog
        await pilot.pause()
        assert not vi.active
        assert not recorder.recording
    finally:
        await context.__aexit__(None, None, None)


# -- TaskManager creation listener -----------------------------------------------------


def test_creation_listener_fired_with_origin(manager: Any) -> None:
    seen: list[tuple[str, str]] = []
    manager.add_creation_listener(
        lambda task, origin: seen.append((task.kind.value, origin))
    )
    manager.create("t", "m")
    manager.create(
        "bug",
        "m",
        kind=__import__("drcode.models", fromlist=["TaskKind"]).TaskKind.BUG,
        origin="system",
    )
    assert seen == [("feature", "user"), ("bug", "system")]


def test_creation_listener_failure_not_fatal(manager: Any) -> None:
    def boom(task: Any, origin: str) -> None:
        raise RuntimeError("listener broke")

    manager.add_creation_listener(boom)
    task = manager.create("t", "m")  # must not raise
    assert task.id in manager.tasks


# -- MicrophoneRecorder WAV assembly (faked PortAudio stream) ----------------------------


class _FakeStream:
    """Just enough of sounddevice.RawInputStream for the recorder's stop/close."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


def test_recorder_wav_assembly() -> None:
    from array import array

    recorder = MicrophoneRecorder()
    # Pretend the PortAudio stream is open and feed the audio-thread
    # callback by hand (the real RawInputStream needs a microphone).
    # The callback receives a flat interleaved int16 buffer (cffi buffer
    # in production; array.array here) — mono, so one sample per frame.
    recorder._stream = _FakeStream()
    recorder._callback(array("h", [100, -200]), 2, None, None)
    recorder._callback(array("h", [300]), 1, None, None)
    assert recorder.recording

    wav = recorder.stop()

    with wave.open(io.BytesIO(wav)) as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == MicrophoneRecorder.SAMPLE_RATE
        frames = handle.readframes(handle.getnframes())
    samples = array("h", frames)
    assert samples.tolist() == [100, -200, 300]
    assert not recorder.recording


def test_recorder_callback_after_stop_is_ignored() -> None:
    from array import array

    recorder = MicrophoneRecorder()
    recorder._stream = _FakeStream()
    recorder._callback(array("h", [1]), 1, None, None)
    recorder.stop()
    recorder._callback(array("h", [2]), 1, None, None)
    # A late audio-thread chunk after the stop must not resurrect the
    # recording (stop() already returned; a new session calls start()).
    with wave.open(io.BytesIO(recorder.stop())) as handle:
        assert handle.getnframes() == 0


def test_recorder_discard_drops_audio() -> None:
    from array import array

    recorder = MicrophoneRecorder()
    recorder._stream = _FakeStream()
    recorder._callback(array("h", [7]), 1, None, None)
    recorder.discard()
    assert not recorder.recording
    # A discard leaves nothing to stop() — an empty valid WAV comes back.
    with wave.open(io.BytesIO(recorder.stop())) as handle:
        assert handle.getnframes() == 0


# -- bundled PortAudio shim (frozen Linux bundle) ---------------------------------


def test_bundled_portaudio_found_in_frozen_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import sys as _sys

    from drcode import voice

    monkeypatch.setattr(_sys, "platform", "linux")
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    monkeypatch.setattr(_sys, "_MEIPASS", str(tmp_path), raising=False)
    assert voice._bundled_portaudio() is None  # nothing staged yet
    lib = tmp_path / "libportaudio.so.2"
    lib.write_bytes(b"ELF")
    assert voice._bundled_portaudio() == str(lib)


def test_bundled_portaudio_ignored_outside_frozen_linux(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import sys as _sys

    from drcode import voice

    (tmp_path / "libportaudio.so.2").write_bytes(b"ELF")
    monkeypatch.setattr(_sys, "platform", "linux")
    monkeypatch.setattr(_sys, "_MEIPASS", str(tmp_path), raising=False)
    assert voice._bundled_portaudio() is None  # not frozen
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    monkeypatch.setattr(_sys, "platform", "darwin")
    assert voice._bundled_portaudio() is None  # not Linux


def test_availability_hint_frozen_vs_source(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys as _sys

    from drcode.voice import VOICE_UNAVAILABLE_HINT, availability_hint

    monkeypatch.delattr(_sys, "frozen", raising=False)
    assert availability_hint() == VOICE_UNAVAILABLE_HINT
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    assert "libportaudio2" in availability_hint()
