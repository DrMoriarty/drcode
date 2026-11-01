"""Optional kill-switch for textual's synchronized-output (DEC 2026) mode.

Background: on startup textual's Linux drivers send a DECRQM probe
(``\\x1b[?2026$p``) asking the terminal whether it supports the
synchronized-output protocol (DEC 2026). A terminal that answers "yes"
makes textual wrap every redraw into SYNC_START/SYNC_END pairs
(``App._sync_available``). iTerm2 3.7.x buffers those synchronized frames
long enough to show up as a constant 0.5–1.0 s keystroke lag, with no way
to disable DEC 2026 from the iTerm2 side.

``DRCODE_DISABLE_SYNC_OUTPUT=1`` makes drcode opt out: the DECRQM probe is
never sent and any "supported" answer is ignored, so frames are drawn
unwrapped. The default behavior (probe sent, sync mode used when the
terminal offers it) is unchanged when the variable is unset or falsy.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Final

ENV_VAR: Final = "DRCODE_DISABLE_SYNC_OUTPUT"

_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})

# The Linux drivers are the only ones that send the DECRQM probe
# (drivers/web_driver.py does too, but that path is textual-web only and
# never runs the desktop TUI). Imports are per-driver: a missing module
# (Windows has no termios) just means there is nothing to patch.
_PROBE_DRIVERS: Final[tuple[tuple[str, str], ...]] = (
    ("textual.drivers.linux_driver", "LinuxDriver"),
    ("textual.drivers.linux_inline_driver", "LinuxInlineDriver"),
)


def sync_output_disabled() -> bool:
    """Whether the user asked to keep synchronized output off."""
    return os.environ.get(ENV_VAR, "").strip().lower() in _TRUTHY


def apply_sync_output_override() -> None:
    """Stop textual from enabling synchronized output, when opted in.

    Idempotent and cheap: a no-op unless :data:`ENV_VAR` is set to a truthy
    value. Patches the drivers' DECRQM probe away (so the terminal is never
    asked and never answers); the app-level belt-and-braces — ignoring a
    ``TerminalSupportsSynchronizedOutput`` message should one arrive anyway —
    lives in :meth:`drcode.tui.app.DrCodeApp._on_terminal_supports_synchronized_output`.
    """
    if not sync_output_disabled():
        return

    def _skip_probe(self: Any) -> None:
        """No-op: never ask the terminal about DEC 2026 support."""

    for module_name, class_name in _PROBE_DRIVERS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # platform-dependent (Windows has no termios)
            continue
        driver_class = getattr(module, class_name, None)
        if driver_class is not None:
            driver_class._request_terminal_sync_mode_support = _skip_probe
