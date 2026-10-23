"""Voice I/O: ASR dictation and TTS announcements over an OpenAI-compatible
audio API (default gateway: ``https://api-copilot.x5.ru/v1``).

Three independent pieces live here:

- :class:`VoiceClient` — the HTTP layer: ``POST /audio/transcriptions``
  (multipart WAV → text), ``POST /audio/speech`` (text → WAV bytes) and a
  cheap ``GET /models`` availability probe. Everything is synchronous and
  meant to be called from worker threads, never from the UI event loop.
- :class:`MicrophoneRecorder` — push-to-talk recording via ``sounddevice``
  (PortAudio): 16 kHz mono PCM16 into an in-memory buffer, returned as a
  complete WAV file. Importing this module never fails when sounddevice is
  missing (it is an optional ``voice`` extra) — :data:`SOUNDDEVICE_AVAILABLE`
  says whether the recording/playback half is usable, and the callers show
  the "pip install drcode[voice]" hint instead.
- :class:`TtsSpeaker` — fire-and-forget announcer used by the notifier: a
  single daemon worker thread synthesizes and plays the queued phrases one
  at a time (at most :data:`TtsSpeaker.MAX_PENDING` phrases wait in the
  queue — a burst of pipeline events can never stack up minutes of speech).

Configuration comes from the project settings (``voice_*`` fields of
:class:`drcode.settings.ProjectSettings`): everything empty = the feature is
off. The API key may be left empty to reuse the token of the user-catalog
provider whose host matches the audio URL's host (no second token for the
common single-gateway case).

The ASR/TTS model ids (e.g. ``x5-asr/x5-asr-large``) contain ``/`` — the
separator of the agent-catalog ``<provider>/<model-id>`` reference format
(:func:`drcode.providers._reject_slash`) — so they are stored in the project
settings directly, never in the model catalog.
"""

from __future__ import annotations

import io
import queue
import ssl
import sys
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from .providers import UserModelsStore

if TYPE_CHECKING:
    from .settings import ProjectSettings

# Default audio-API base URL (offered as the dialog's prefill; the setting
# stays empty until the user saves the voice dialog — empty = voice off).
DEFAULT_VOICE_API_URL = "https://api-copilot.x5.ru/v1"

# Default speaker voice when voice_tts_voice is empty (OpenAI TTS names).
DEFAULT_TTS_VOICE = "alloy"

# Network timeouts: transcription of a long utterance can take a while;
# the calls run in worker threads, so a generous ceiling costs nothing.
ASR_TIMEOUT = 60.0
TTS_TIMEOUT = 60.0
CHECK_TIMEOUT = 10.0

try:  # pragma: no cover - exercised indirectly (availability flag)
    import sounddevice as _sd  # noqa: F401 - the import itself is the probe

    SOUNDDEVICE_AVAILABLE = True
    _SOUNDDEVICE_ERROR: str | None = None
except Exception as _e:  # noqa: BLE001 - ImportError, but also a missing PortAudio library
    SOUNDDEVICE_AVAILABLE = False
    _SOUNDDEVICE_ERROR = f"{type(_e).__name__}: {_e}"

# python-sounddevice is not thread-safe at module level (global default
# stream, cffi callback bookkeeping): every PortAudio lifecycle operation —
# opening/starting/stopping/closing a stream, the sd.play()/sd.wait()
# playback pair — must go through this one lock. Recording and TTS playback
# run on independent threads, so without it their interleaved
# open/close/stop calls corrupt the callback trampolines (observed as a
# SIGSEGV on macOS's CoreAudio IOThread client).
_PORTAUDIO_LOCK = threading.Lock()

VOICE_EXTRA_HINT = "pip install drcode[voice]"


class VoiceError(Exception):
    """A voice feature could not do its job (config missing, network or
    API error). The message is user-readable on purpose."""


@dataclass(frozen=True)
class VoiceConfig:
    """Resolved connection + model selection for the audio API."""

    base_url: str
    api_key: str
    asr_model: str = ""
    tts_model: str = ""
    tts_voice: str = DEFAULT_TTS_VOICE

    def _base(self) -> str:
        return self.base_url.rstrip("/")


def catalog_api_key_for_host(base_url: str) -> str:
    """The api_key of the user-catalog provider whose base_url host matches
    ``base_url``'s host ("" when nothing matches). Audio gateways that also
    serve the chat models reuse the same token — the user then never enters
    it twice."""
    from urllib.parse import urlsplit

    host = urlsplit(base_url).netloc.rsplit("@", 1)[-1].lower()
    host = host.rsplit(":", 1)[0] if host else ""
    if not host:
        return ""
    for provider in UserModelsStore().load().providers:
        provider_host = urlsplit(provider.base_url).netloc.rsplit("@", 1)[-1].lower()
        if provider_host.rsplit(":", 1)[0] == host and provider.api_key:
            return provider.api_key
    return ""


def resolve_api_key(settings: ProjectSettings, base_url: str) -> str:
    """The effective audio-API key: the explicit setting wins, the host
    match against the user catalog is the fallback."""
    key = settings.voice_api_key.strip()
    if key:
        return key
    return catalog_api_key_for_host(base_url)


def config_from_settings(settings: ProjectSettings) -> VoiceConfig | None:
    """The resolved config, or None when voice is off (no base URL).
    api_key resolution against the catalog happens here, once, so callers
    never re-implement the fallback."""
    base_url = settings.voice_api_url.strip()
    if not base_url:
        return None
    return VoiceConfig(
        base_url=base_url,
        api_key=resolve_api_key(settings, base_url),
        asr_model=settings.voice_asr_model.strip(),
        tts_model=settings.voice_tts_model.strip(),
        tts_voice=settings.voice_tts_voice.strip() or DEFAULT_TTS_VOICE,
    )


class VoiceClient:
    """Synchronous OpenAI-compatible audio-API client (worker threads only)."""

    def __init__(
        self, config: VoiceConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        # `transport` is a test seam (httpx.MockTransport): production code
        # always leaves it None and gets a fresh real connection per call.
        self.config = config
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _require(self, model: str, what: str) -> None:
        if not model:
            raise VoiceError(f"Voice is not configured: no {what} model set")

    def _verify(self) -> ssl.SSLContext | bool:
        """Trust-store-backed TLS verification, falling back to httpx
        defaults. Системное хранилище сертификатов: корпоративные CA
        отсутствуют в certifi, из-за чего httpx даёт
        CERTIFICATE_VERIFY_FAILED, который оборачивается в ConnectError
        (тот же паттерн, что в ``agent_model._ssl_verify``)."""
        try:
            import truststore

            return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except ImportError:
            return True

    def _post(self, url: str, **kwargs: object) -> httpx.Response:
        if self._transport is not None:
            with httpx.Client(transport=self._transport) as client:
                return client.post(url, **kwargs)  # type: ignore[arg-type]
        return httpx.post(url, verify=self._verify(), **kwargs)  # type: ignore[arg-type]

    def _get(self, url: str, **kwargs: object) -> httpx.Response:
        if self._transport is not None:
            with httpx.Client(transport=self._transport) as client:
                return client.get(url, **kwargs)  # type: ignore[arg-type]
        return httpx.get(url, verify=self._verify(), **kwargs)  # type: ignore[arg-type]

    def transcribe(self, wav: bytes) -> str:
        """Speech → text (``POST /audio/transcriptions``): multipart WAV
        upload, the JSON answer carries the recognized text with
        punctuation and capitalization already in place."""
        self._require(self.config.asr_model, "ASR")
        try:
            response = self._post(
                f"{self.config._base()}/audio/transcriptions",
                headers=self._headers(),
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={"model": self.config.asr_model},
                timeout=ASR_TIMEOUT,
            )
            response.raise_for_status()
            text = str(response.json().get("text") or "").strip()
        except httpx.HTTPStatusError as e:
            raise VoiceError(f"ASR failed: HTTP {e.response.status_code}") from e
        except (httpx.HTTPError, ValueError) as e:
            raise VoiceError(f"ASR failed: {type(e).__name__}: {e}") from e
        if not text:
            raise VoiceError("ASR returned no text")
        return text

    def speech(self, text: str) -> bytes:
        """Text → WAV bytes (``POST /audio/speech``)."""
        self._require(self.config.tts_model, "TTS")
        try:
            response = self._post(
                f"{self.config._base()}/audio/speech",
                headers=self._headers(),
                json={
                    "model": self.config.tts_model,
                    "input": text,
                    "voice": self.config.tts_voice,
                    "response_format": "wav",
                },
                timeout=TTS_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise VoiceError(f"TTS failed: HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise VoiceError(f"TTS failed: {type(e).__name__}: {e}") from e
        return response.content

    def check(self) -> str:
        """Cheap availability probe (``GET /models``): raises VoiceError
        with a readable reason, returns a short human summary on success."""
        try:
            response = self._get(
                f"{self.config._base()}/models",
                headers=self._headers(),
                timeout=CHECK_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as e:
            raise VoiceError(f"HTTP {e.response.status_code}") from e
        except (httpx.HTTPError, ValueError) as e:
            raise VoiceError(f"{type(e).__name__}: {e}") from e
        models = payload.get("data") if isinstance(payload, dict) else None
        count = len(models) if isinstance(models, list) else 0
        if count:
            return f"OK — {count} models available"
        return "OK — endpoint answered"


# -- recording (push-to-talk) -------------------------------------------------


class _AudioStream(Protocol):
    """The subset of sounddevice.InputStream the recorder touches (tests
    fake the whole stream)."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class MicrophoneRecorder:
    """Push-to-talk recorder: 16 kHz mono PCM16 from the default input
    device into memory, delivered as a complete WAV file on stop.

    The callback runs on PortAudio's audio thread — it only appends copies
    of the incoming chunks (list.append is atomic enough under the GIL);
    everything else happens on the caller's thread.
    """

    SAMPLE_RATE = 16_000

    def __init__(self) -> None:
        self._chunks: list[Any] = []
        self._stream: _AudioStream | None = None
        self._lock = threading.Lock()
        # Strong reference to the bound callback for as long as the stream
        # lives: sounddevice wraps it into a cffi trampoline, and letting the
        # wrapper be garbage-collected while PortAudio still calls it is a
        # crash (released only after the stream is fully closed).
        self._cb: Callable[..., object] | None = None

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        """Open the input stream. Raises VoiceError when the microphone
        cannot be opened (device busy/absent, PortAudio trouble)."""
        if not SOUNDDEVICE_AVAILABLE or self._stream is not None:
            return
        import sounddevice as sd

        self._cb = self._callback
        try:
            with _PORTAUDIO_LOCK:
                stream: _AudioStream = sd.InputStream(
                    samplerate=self.SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                    callback=self._cb,
                )
                stream.start()
        except Exception as e:
            self._cb = None
            raise VoiceError(f"Microphone unavailable: {e}") from e
        with self._lock:
            self._chunks = []
            self._stream = stream

    def _callback(
        self, indata: object, frames: int, time: object, status: object
    ) -> None:
        # Audio-thread callback: a plain append of a numpy copy; the import
        # guard above guarantees numpy exists whenever a stream is running.
        with self._lock:
            if self._stream is not None:
                self._chunks.append(indata.copy())  # type: ignore[attr-defined]

    def stop(self) -> bytes:
        """Close the stream and return the recording as WAV bytes (empty
        recording still yields a valid, silent WAV header)."""
        import numpy as np

        with self._lock:
            stream, self._stream = self._stream, None
            chunks = self._chunks
            self._chunks = []
        _close_stream(stream)
        self._cb = None
        if not chunks:
            pcm = np.zeros((0,), dtype=np.int16)
        else:
            pcm = np.concatenate(chunks).reshape(-1)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())
        return buffer.getvalue()

    def discard(self) -> None:
        """Drop the recording and close the stream (cancel path)."""
        with self._lock:
            stream, self._stream = self._stream, None
            self._chunks = []
        _close_stream(stream)
        self._cb = None


# -- playback ------------------------------------------------------------------


def _close_stream(stream: _AudioStream | None) -> None:
    """Stop and close a PortAudio stream under the module-level lock, from
    whatever thread owns the recorder at that moment (never the UI thread —
    the controller offloads cancel/stop to workers)."""
    if stream is None:
        return
    try:
        with _PORTAUDIO_LOCK:
            stream.stop()
            stream.close()
    except Exception as e:  # noqa: BLE001 - closing must never mask the result
        print(f"drcode voice: stream close failed: {e}", file=sys.stderr)


def play_wav(data: bytes) -> None:
    """Decode PCM WAV bytes and play them synchronously (blocks until the
    sound finishes). Worker threads only."""
    if not SOUNDDEVICE_AVAILABLE:
        raise VoiceError(f"sounddevice is not installed ({VOICE_EXTRA_HINT})")
    try:
        with wave.open(io.BytesIO(data)) as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as e:
        raise VoiceError(f"TTS answer is not a valid WAV: {e}") from e
    import numpy as np
    import sounddevice as sd

    pcm = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)
    if pcm.size:
        # sd.play()/sd.wait() drive sounddevice's global output stream —
        # the whole playback (including the wait) stays under the lock so a
        # concurrent recording open/close can never interleave with it.
        with _PORTAUDIO_LOCK:
            sd.play(pcm, samplerate=rate)
            sd.wait()


class _VoiceSynth(Protocol):
    """Structural subset of VoiceClient used by TtsSpeaker's factory seam
    (tests inject stubs with just this method)."""

    def speech(self, text: str) -> bytes: ...


class TtsSpeaker:
    """Fire-and-forget TTS announcer: one daemon worker thread synthesizes
    and plays the queued phrases strictly one at a time.

    ``speak`` never raises and never touches the network on the caller's
    thread; when more than MAX_PENDING phrases are already waiting, new ones
    are dropped (a burst of events must not stack up minutes of speech).
    A disabled/empty voice config in the project settings silences the
    speaker entirely.
    """

    MAX_PENDING = 2

    def __init__(
        self,
        settings_getter: Callable[[], ProjectSettings],
        *,
        on_error: Callable[[str], None] | None = None,
        client_factory: Callable[[VoiceConfig], _VoiceSynth] | None = None,
        play: Callable[[bytes], None] | None = None,
    ) -> None:
        # Test seams: `client_factory` builds the synthesizing client, `play`
        # consumes the WAV bytes. Defaults are the real VoiceClient + the
        # local sounddevice playback.
        self._settings_getter = settings_getter
        self._on_error = on_error
        self._client_factory = client_factory or VoiceClient
        self._play = play or play_wav
        self._queue: queue.Queue[tuple[str, VoiceConfig]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        """Queue a phrase for synthesis+playback. Safe from any thread;
        silently no-op when voice is off/unconfigured."""
        text = text.strip()
        if not text:
            return
        try:
            config = config_from_settings(self._settings_getter())
        except Exception:  # noqa: BLE001 - settings must never break the caller
            return
        if config is None or not config.tts_model:
            return
        with self._lock:
            # The worker is persistent (blocks on the queue forever, exits
            # with the process as a daemon): starting it lazily avoids a
            # "thread exited between the alive-check and the put" race.
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._worker, daemon=True, name="drcode-tts"
                )
                self._thread.start()
            if self._queue.qsize() >= self.MAX_PENDING:
                return
        self._queue.put((text, config))

    def _worker(self) -> None:
        while True:
            text, config = self._queue.get()
            try:
                self._play(self._client_factory(config).speech(text))
            except Exception as e:  # noqa: BLE001 - announcements never break anything
                self._report_error(str(e))

    def _report_error(self, message: str) -> None:
        if self._on_error is not None:
            try:
                self._on_error(message)
            except Exception as e:  # noqa: BLE001 - the callback is best-effort
                print(f"drcode voice: error callback failed: {e}", file=sys.stderr)
