"""Kill-switch / opt-out for textual's synchronized-output (DEC 2026) mode.

Background: on startup textual's Linux drivers send a DECRQM probe
(``\\x1b[?2026$p``) asking the terminal whether it supports the
synchronized-output protocol (DEC 2026). A terminal that answers "yes"
makes textual wrap every redraw into SYNC_START/SYNC_END pairs
(``App._sync_available``). iTerm2 3.7.x buffers those synchronized frames
long enough to show up as a constant 0.5–1.0 s keystroke lag, with no way
to disable DEC 2026 from the iTerm2 side. Textual itself excludes iTerm2
from the adjacent 2048 mode for the same reason ("iTerm is buggy in one or
more of the protocols required here", ``textual/_xterm_parser.py``).

Priority matrix, strictest first:

1. ``DRCODE_DISABLE_SYNC_OUTPUT`` (truthy) — explicit opt-OUT: the DECRQM
   probe is never sent and any "supported" answer is ignored; frames are
   drawn unwrapped. Wins over everything, including the enable variable.
2. ``DRCODE_ENABLE_SYNC_OUTPUT`` (truthy) — explicit opt-IN: re-enables the
   probe + sync mode even on iTerm2 (but never overrides #1).
3. iTerm2 detected (``TERM_PROGRAM == "iTerm.app"`` or
   ``LC_TERMINAL == "iTerm2"``, same check as textual's ``IS_ITERM``) —
   sync output is off BY DEFAULT: no probe, no sync, no lag.
4. Anything else — stock textual behavior: the probe is sent and sync mode
   is used when the terminal offers it.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Final

ENV_VAR: Final = "DRCODE_DISABLE_SYNC_OUTPUT"
ENABLE_ENV_VAR: Final = "DRCODE_ENABLE_SYNC_OUTPUT"

_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})

# The Linux drivers are the only ones that send the DECRQM probe
# (drivers/web_driver.py does too, but that path is textual-web only and
# never runs the desktop TUI). Imports are per-driver: a missing module
# (Windows has no termios) just means there is nothing to patch.
_PROBE_DRIVERS: Final[tuple[tuple[str, str], ...]] = (
    ("textual.drivers.linux_driver", "LinuxDriver"),
    ("textual.drivers.linux_inline_driver", "LinuxInlineDriver"),
)


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def is_iterm2() -> bool:
    """Whether the session runs inside iTerm2.

    Same environment check as textual's module-level ``IS_ITERM``, but
    evaluated at call time (textual freezes it at import; tests and IDE
    launches need the live value).
    """
    return os.environ.get("TERM_PROGRAM", "") == "iTerm.app" or (
        os.environ.get("LC_TERMINAL", "") == "iTerm2"
    )


def sync_output_disabled() -> bool:
    """Whether the user explicitly asked to keep synchronized output off.

    Strictly the ``DRCODE_DISABLE_SYNC_OUTPUT`` opt-out — the strongest
    signal, it beats the enable variable and the iTerm2 default.
    """
    return _truthy(ENV_VAR)


def sync_output_should_be_off() -> bool:
    """Whether synchronized output must stay off in this session.

    True when (a) the explicit disable variable is set, OR (b) the terminal
    is iTerm2 and the explicit enable variable does not override the
    iTerm2-specific default-off.
    """
    if sync_output_disabled():
        return True
    if _truthy(ENABLE_ENV_VAR):
        return False
    return is_iterm2()


def apply_sync_output_override() -> None:
    """Stop textual from enabling synchronized output, when warranted.

    Idempotent and cheap: a no-op unless :func:`sync_output_should_be_off`
    is true (explicit opt-out, or iTerm2 without an explicit opt-in).
    Patches the drivers' DECRQM probe away (so the terminal is never asked
    and never answers); the app-level belt-and-braces — ignoring a
    ``TerminalSupportsSynchronizedOutput`` message should one arrive
    anyway — lives in
    :meth:`drcode.tui.app.DrCodeApp._on_terminal_supports_synchronized_output`.
    """
    if not sync_output_should_be_off():
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
