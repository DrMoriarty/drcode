"""The guarded sounddevice import and the PortAudio serialization lock.

sounddevice is a core dependency, but the import is still guarded (a
missing system PortAudio library raises the same ImportError from inside
the wheel) — :data:`SOUNDDEVICE_AVAILABLE` says whether the
recording/playback half is usable, and the callers show a readable toast
instead of crashing.

python-sounddevice is not thread-safe at module level (global default
stream, cffi callback bookkeeping): every PortAudio lifecycle operation —
opening/starting/stopping/closing a stream, the RawOutputStream playback
(write + stop-drain) — must go through :data:`_PORTAUDIO_LOCK`. Recording
and TTS playback run on independent threads, so without it their
interleaved open/close/stop calls corrupt the callback trampolines
(observed as a SIGSEGV on macOS's CoreAudio IOThread client).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path


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
