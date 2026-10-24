"""Voice I/O: ASR dictation and TTS announcements over an OpenAI-compatible
audio API (default gateway: ``https://api-copilot.x5.ru/v1``).

Three independent pieces live here:

- :class:`VoiceClient` — the HTTP layer: ``POST /audio/transcriptions``
  (multipart WAV → text), ``POST /audio/speech`` (text → WAV bytes) and a
  cheap ``GET /models`` availability probe. Everything is synchronous and
  meant to be called from worker threads, never from the UI event loop.
- :class:`MicrophoneRecorder` — push-to-talk recording via ``sounddevice``
  (PortAudio): 16 kHz mono PCM16 into an in-memory buffer, returned as a
  complete WAV file. sounddevice is a core dependency, but the import is
  still guarded (a missing system PortAudio library raises the same
  ImportError from inside the wheel) — :data:`SOUNDDEVICE_AVAILABLE`
  says whether the recording/playback half is usable, and the callers
  show a readable toast instead of crashing.
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
import time
import wave
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

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


def _bundled_portaudio() -> str | None:
    """Absolute path of the ``libportaudio.so.2`` shipped inside the frozen
    Linux bundle (see drcode.spec and scripts/fetch-portaudio.sh), or None.

    sounddevice's POSIX code path locates PortAudio exclusively through
    ``ctypes.util.find_library`` (ldconfig / compiler default dirs) — its
    bundled ``_sounddevice_data`` fallback exists for macOS/Windows only.
    The Linux bundle therefore carries the library next to the executable,
    and the import guard below points the lookup at it before sounddevice
    is imported."""
    if not sys.platform.startswith("linux") or not getattr(sys, "frozen", False):
        return None
    base = getattr(sys, "_MEIPASS", None) or str(Path(sys.executable).parent)
    candidate = Path(base) / "libportaudio.so.2"
    return str(candidate) if candidate.is_file() else None


_BUNDLED_PORTAUDIO = _bundled_portaudio()

try:  # pragma: no cover - exercised indirectly (availability flag)
    if _BUNDLED_PORTAUDIO is not None:
        import ctypes.util

        _system_find_library = ctypes.util.find_library

        def _find_library(name: str) -> str | None:
            """sounddevice shim: resolve 'portaudio' to the copy bundled
            next to the frozen executable (the system search cannot see
            it), everything else to the system lookup."""
            if name == "portaudio":
                return _BUNDLED_PORTAUDIO
            return _system_find_library(name)

        ctypes.util.find_library = _find_library

    import sounddevice as _sd  # noqa: F401 - the import itself is the probe

    SOUNDDEVICE_AVAILABLE = True
    _SOUNDDEVICE_ERROR: str | None = None
except Exception as _e:  # noqa: BLE001 - ImportError, but also a missing PortAudio library
    SOUNDDEVICE_AVAILABLE = False
    _SOUNDDEVICE_ERROR = f"{type(_e).__name__}: {_e}"

# python-sounddevice is not thread-safe at module level (global default
# stream, cffi callback bookkeeping): every PortAudio lifecycle operation —
# opening/starting/stopping/closing a stream, the RawOutputStream
# playback (write + stop-drain) — must go through this one lock. Recording
# and TTS playback
# run on independent threads, so without it their interleaved
# open/close/stop calls corrupt the callback trampolines (observed as a
# SIGSEGV on macOS's CoreAudio IOThread client).
_PORTAUDIO_LOCK = threading.Lock()

# Shown when the guarded import above failed despite sounddevice being a
# core dependency (normally a missing system PortAudio library on Linux).
VOICE_UNAVAILABLE_HINT = (
    "the sounddevice/PortAudio audio backend is unavailable "
    "(on Linux install the libportaudio2 system package)"
)


def availability_hint() -> str:
    """Actionable advice for the unavailable-sounddevice toast/error: the
    generic backend hint in a normal installation, the system PortAudio
    package in a frozen bundle (where pip advice makes no sense)."""
    if getattr(sys, "frozen", False):
        return "install the system libportaudio2 package (Linux)"
    return VOICE_UNAVAILABLE_HINT


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

    # A zero-frame recording of a stream that ran at least this long is a
    # dead recording (the callback never delivered anything): stop() raises
    # instead of shipping a silent WAV to the ASR endpoint.
    MIN_AUDIBLE_DURATION = 0.5

    def __init__(self) -> None:
        self._chunks: list[array[int]] = []
        self._stream: _AudioStream | None = None
        self._lock = threading.Lock()
        # Failures of the audio-thread callback (sounddevice swallows
        # exceptions raised there — "Exception ignored from cffi callback"
        # — so they must be counted here to be visible at all).
        self._callback_errors = 0
        self._opened_at = time.monotonic()
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
                # RawInputStream (not InputStream): its callback hands over
                # a plain cffi buffer, so the audio path needs no numpy —
                # the regular InputStream hard-requires numpy to wrap the
                # callback data into ndarrays.
                stream: _AudioStream = sd.RawInputStream(
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
            self._callback_errors = 0
            self._opened_at = time.monotonic()
            self._stream = stream

    def _callback(
        self, indata: object, frames: int, time: object, status: object
    ) -> None:
        # Audio-thread callback: a plain append of a copy. RawInputStream
        # hands over a byte-typed cffi buffer, which array.array("h", ...)
        # rejects outright (TypeError) — convert through bytes() first.
        # Cheap and non-blocking under the GIL — all assembly happens
        # later, on the caller's thread in stop().
        try:
            chunk = array("h", bytes(indata))  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001 - a dead callback is invisible otherwise
            with self._lock:
                self._callback_errors += 1
            print(f"drcode voice: microphone callback failed: {e}", file=sys.stderr)
            return
        with self._lock:
            if self._stream is not None:
                self._chunks.append(chunk)

    def stop(self) -> bytes:
        """Close the stream and return the recording as WAV bytes.

        A recording that ended up with zero frames despite the stream
        having been open longer than a moment (or whose callback failed)
        raises a distinct VoiceError instead — shipping a silent WAV to
        the ASR endpoint only surfaces later as the misleading
        "ASR returned no text". A genuinely brief tap still yields a
        valid, silent WAV."""
        with self._lock:
            stream, self._stream = self._stream, None
            chunks = self._chunks
            self._chunks = []
        _close_stream(stream)
        self._cb = None
        pcm = array("h")
        for chunk in chunks:
            pcm.extend(chunk)
        if not pcm:
            with self._lock:
                errors = self._callback_errors
            if errors:
                raise VoiceError(
                    f"Microphone returned no audio: the recording callback "
                    f"failed {errors} time(s)"
                )
            duration = time.monotonic() - self._opened_at
            if stream is not None and duration >= self.MIN_AUDIBLE_DURATION:
                raise VoiceError(
                    f"Microphone returned no audio in {duration:.1f} s of recording"
                )
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
        raise VoiceError(f"sounddevice is not available ({VOICE_UNAVAILABLE_HINT})")
    try:
        with wave.open(io.BytesIO(data)) as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as e:
        raise VoiceError(f"TTS answer is not a valid WAV: {e}") from e
    if sampwidth != 2:
        # play_wav interprets the frames as native int16 (array("h")); a
        # 24-bit or float32 answer from the gateway would be played as
        # shifted garbage (crackle over the whole phrase) — fail readable
        # instead.
        raise VoiceError(
            f"TTS answer is {sampwidth * 8}-bit PCM; only 16-bit is supported"
        )
    import sounddevice as sd

    # int16 framing: building array.array("h") from the raw frames doubles
    # as the odd-byte guard — a truncated final sample raises ValueError
    # (surfaced as a readable VoiceError) instead of corrupting playback.
    # This replaces np.frombuffer; sd.play()/sd.wait() hard-require numpy,
    # which the voice extra deliberately avoids.
    try:
        pcm = array("h", frames)
    except ValueError as e:
        raise VoiceError(f"TTS answer has a truncated PCM frame: {e}") from e
    if not pcm:
        return
    raw = pcm.tobytes()
    total = len(raw)
    state = {"pos": 0, "underflowed": False}
    done = threading.Event()

    # Callback mode, not the blocking RawOutputStream.write() pair: with
    # blocking writes the ring buffer is empty when the stream starts, so
    # the first frames go out as zeros (an audible click at the head of
    # every phrase) and any writer-thread hiccup mid-phrase drops out.
    # (Priming via Pa_WriteStream before start() is NOT an option — on
    # current PortAudio it fails with paStreamIsStopped.) The callback
    # fills each block straight from memory on demand — nothing can run
    # dry. Replaces sd.play()/sd.wait(), which hard-require numpy.

    def _callback(outdata: object, frames: int, time: object, status: object) -> None:
        try:
            if status:
                state["underflowed"] = True
            view = memoryview(outdata).cast("B")  # type: ignore[arg-type]
            end = min(state["pos"] + len(view), total)
            block = end - state["pos"]
            view[:block] = raw[state["pos"] : end]
            if block < len(view):
                # The final partial block is padded with silence — the
                # phrase's natural tail.
                view[block:] = bytes(len(view) - block)
            state["pos"] = end
            if end >= total:
                done.set()
        except Exception as e:  # noqa: BLE001 - a dead callback would hang play_wav
            print(f"drcode voice: playback callback failed: {e}", file=sys.stderr)
            done.set()

    # The whole playback (including the drain in stop()) stays under the
    # lock so a concurrent recording open/close can never interleave with it.
    with _PORTAUDIO_LOCK:
        stream = sd.RawOutputStream(
            samplerate=rate,
            channels=channels,
            dtype="int16",
            latency="high",  # extra headroom against mid-phrase dropouts
            callback=_callback,
        )
        try:
            stream.start()
            done.wait()
        finally:
            stream.stop()  # drains the pending audio, like sd.wait()
            stream.close()
    if state["underflowed"]:
        print("drcode voice: playback buffer underflowed", file=sys.stderr)


class _VoiceSynth(Protocol):
    """Structural subset of VoiceClient used by TtsSpeaker's factory seam
    (tests inject stubs with just this method)."""

    def speech(self, text: str) -> bytes: ...


class TtsSpeaker:
    """Fire-and-forget TTS announcer: one daemon worker thread synthesizes
    and plays the queued phrases strictly one at a time.

    The phrase following the one being played is synthesized in parallel on
    a short-lived helper thread, so consecutive announcements follow each
    other without a network-synthesis gap between them (the playback itself
    stays strictly serialized).

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
        # Phrases currently held out of the queue by the prefetch helper
        # (0 or 1): they are already being synthesized but sit in no queue,
        # so speak()'s cap must count them on top of qsize.
        self._prefetching = 0

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
            if self._queue.qsize() + self._prefetching >= self.MAX_PENDING:
                return
        self._queue.put((text, config))

    def _worker(self) -> None:
        # ready = the next phrase, already synthesized by the prefetch
        # helper while the previous one was playing.
        ready: tuple[bytes | None, str | None] | None = None
        while True:
            if ready is None:
                text, config = self._queue.get()
                wav, error = self._synthesize(text, config)
            else:
                wav, error = ready
                ready = None
            if wav is None:
                self._report_error(error or "TTS synthesis failed")
                continue
            finish = self._start_prefetch()
            try:
                self._play(wav)
            except Exception as e:  # noqa: BLE001 - announcements never break anything
                self._report_error(str(e))
            finally:
                if finish is not None:
                    ready = finish()

    def _synthesize(
        self, text: str, config: VoiceConfig
    ) -> tuple[bytes | None, str | None]:
        """Network synthesis, never raising: (wav, None) on success,
        (None, readable message) on failure."""
        try:
            return self._client_factory(config).speech(text), None
        except Exception as e:  # noqa: BLE001 - announcements never break anything
            return None, str(e)

    def _synthesize_into(
        self,
        text: str,
        config: VoiceConfig,
        out: list[tuple[bytes | None, str | None]],
    ) -> None:
        """Helper-thread body: synthesize one phrase into ``out``."""
        out.append(self._synthesize(text, config))

    def _start_prefetch(self) -> Callable[[], tuple[bytes | None, str | None]] | None:
        """Pop the next queued phrase and start synthesizing it on a helper
        thread while the current one plays. Returns a callable that joins
        the helper and yields its result (None when the queue was empty)."""
        try:
            text, config = self._queue.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            self._prefetching += 1
        box: list[tuple[bytes | None, str | None]] = []
        thread = threading.Thread(
            target=self._synthesize_into,
            args=(text, config, box),
            daemon=True,
            name="drcode-tts-prefetch",
        )
        thread.start()

        def finish() -> tuple[bytes | None, str | None]:
            thread.join()
            with self._lock:
                self._prefetching -= 1
            return box[0]

        return finish

    def _report_error(self, message: str) -> None:
        if self._on_error is not None:
            try:
                self._on_error(message)
            except Exception as e:  # noqa: BLE001 - the callback is best-effort
                print(f"drcode voice: error callback failed: {e}", file=sys.stderr)
