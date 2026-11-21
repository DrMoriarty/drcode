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
separator of the agent-catalog ``<provider>/<model-id>`` reference format —
so voice models never mix into the catalog.

COLD START: nothing is imported eagerly. The package `__getattr__` below
resolves every public name on first use by importing the owning submodule —
so the NATIVE PortAudio library (loaded by ``sounddevice`` at import time)
only loads when the recording/playback half is actually touched, and the
TUI cold start (which imports :class:`TtsSpeaker`) never pays for it.
"""

from __future__ import annotations

import importlib
from typing import Any

# public name -> submodule that defines it
_EXPORTS: dict[str, str] = {
    # _sounddevice
    "_SOUNDDEVICE_ERROR": "_sounddevice",
    "SOUNDDEVICE_AVAILABLE": "_sounddevice",
    "VOICE_UNAVAILABLE_HINT": "_sounddevice",
    "_bundled_portaudio": "_sounddevice",
    "availability_hint": "_sounddevice",
    # client
    "ASR_TIMEOUT": "client",
    "CHECK_TIMEOUT": "client",
    "TTS_TIMEOUT": "client",
    "VoiceClient": "client",
    # config
    "DEFAULT_TTS_VOICE": "config",
    "DEFAULT_VOICE_API_URL": "config",
    "VOICE_LANGUAGE_EN": "config",
    "VOICE_LANGUAGE_RU": "config",
    "VOICE_PHRASES": "config",
    "VoiceConfig": "config",
    "VoiceError": "config",
    "announcement_phrase": "config",
    "catalog_api_key_for_host": "config",
    "config_from_settings": "config",
    "effective_voice_language": "config",
    "resolve_api_key": "config",
    "terminal_voice_language": "config",
    # playback
    "play_wav": "playback",
    # recorder
    "MicrophoneRecorder": "recorder",
    # synth
    "TtsSpeaker": "synth",
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{target}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
