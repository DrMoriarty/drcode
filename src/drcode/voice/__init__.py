"""Voice I/O: ASR dictation and TTS announcements over an OpenAI-compatible
audio API (default gateway: ``https://provider.ru/v1``).

The package is split by responsibility:

- :mod:`drcode.voice.config` — settings resolution: :class:`VoiceConfig`,
  the api-key fallback against the user catalog, the announcement-language
  selection and the spoken phrase tables.
- :mod:`drcode.voice.client` — :class:`VoiceClient`, the synchronous HTTP
  layer (``POST /audio/transcriptions``, ``POST /audio/speech``, a
  ``GET /models`` probe). Worker threads only, never the UI event loop.
- :mod:`drcode.voice.recorder` — :class:`MicrophoneRecorder`, push-to-talk
  recording via sounddevice (PortAudio): 16 kHz mono PCM16 into memory,
  returned as a complete WAV file.
- :mod:`drcode.voice.playback` — :func:`play_wav`, synchronous numpy-free
  WAV playback through a RawOutputStream callback.
- :mod:`drcode.voice.synth` — :class:`TtsSpeaker`, the fire-and-forget
  announcer used by the notifier: a single daemon worker thread synthesizes
  and plays the queued phrases one at a time (at most
  :data:`TtsSpeaker.MAX_PENDING` phrases wait in the queue — a burst of
  pipeline events can never stack up minutes of speech).
- :mod:`drcode.voice._sounddevice` — the guarded sounddevice import:
  sounddevice is a core dependency, but the import is still guarded (a
  missing system PortAudio library raises the same ImportError from inside
  the wheel) — :data:`SOUNDDEVICE_AVAILABLE` says whether the
  recording/playback half is usable, and the callers show a readable toast
  instead of crashing. Every PortAudio lifecycle operation goes through the
  module's :data:`_PORTAUDIO_LOCK` (python-sounddevice is not thread-safe
  at module level; without the lock, interleaved recording/playback
  open/close calls corrupt the callback trampolines — observed as a SIGSEGV
  on macOS's CoreAudio IOThread client).

Configuration comes from the project settings (``voice_*`` fields of
:class:`drcode.settings.ProjectSettings`): everything empty = the feature is
off. The API key may be left empty to reuse the token of the user-catalog
provider whose host matches the audio URL's host (no second token for the
common single-gateway case).

The ASR/TTS model ids (e.g. ``gigaam_rnnt_v3_e2e``) contain ``/`` — the
separator of the agent-catalog ``<provider>/<model-id>`` reference format
(:func:`drcode.providers._reject_slash`) — so they are stored in the project
settings directly, never in the model catalog.
"""

from __future__ import annotations

from ._sounddevice import (
    _SOUNDDEVICE_ERROR,
    SOUNDDEVICE_AVAILABLE,
    VOICE_UNAVAILABLE_HINT,
    _bundled_portaudio,
    availability_hint,
)
from .client import ASR_TIMEOUT, CHECK_TIMEOUT, TTS_TIMEOUT, VoiceClient
from .config import (
    DEFAULT_TTS_VOICE,
    DEFAULT_VOICE_API_URL,
    VOICE_LANGUAGE_EN,
    VOICE_LANGUAGE_RU,
    VOICE_PHRASES,
    VoiceConfig,
    VoiceError,
    announcement_phrase,
    catalog_api_key_for_host,
    config_from_settings,
    effective_voice_language,
    resolve_api_key,
    terminal_voice_language,
)
from .playback import play_wav
from .recorder import MicrophoneRecorder
from .synth import TtsSpeaker

__all__ = [
    "ASR_TIMEOUT",
    "CHECK_TIMEOUT",
    "DEFAULT_TTS_VOICE",
    "DEFAULT_VOICE_API_URL",
    "SOUNDDEVICE_AVAILABLE",
    "TTS_TIMEOUT",
    "VOICE_LANGUAGE_EN",
    "VOICE_LANGUAGE_RU",
    "VOICE_PHRASES",
    "VOICE_UNAVAILABLE_HINT",
    "_SOUNDDEVICE_ERROR",
    "MicrophoneRecorder",
    "TtsSpeaker",
    "VoiceClient",
    "VoiceConfig",
    "VoiceError",
    "_bundled_portaudio",
    "announcement_phrase",
    "availability_hint",
    "catalog_api_key_for_host",
    "config_from_settings",
    "effective_voice_language",
    "play_wav",
    "resolve_api_key",
    "terminal_voice_language",
]
