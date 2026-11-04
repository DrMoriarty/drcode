"""App-level fix for textual's parser dropping the Alt modifier on
"tuple-mapped" keys (textual 8.2.8).

Background: macOS Terminal.app (with "Use Option as Meta key") sends
Alt+Enter as the legacy two-byte sequence ESC+CR (``\\x1b\\r``). Textual's
``XTermParser`` buffers the ESC, times out, and reissues the sequence with
``alt=True`` — but the tuple branch of ``_sequence_to_key_events``
(``ANSI_SEQUENCES_KEYS['\\r']`` maps to ``(Keys.Enter,)`` — a tuple)
ignores the ``alt`` flag and yields a plain ``enter``. Alt+Enter therefore
reached drcode's ChatInput as Enter and SENT the draft instead of inserting
a newline (the single-character branch of the same method handles ESC+a →
``alt+a`` correctly; only the tuple branch is broken). Upstream defect —
8.2.8 is the newest release at the time of writing, so there is no upgrade
path; this module patches the parser the way ``sync_output.py`` patches the
DEC 2026 probe.

The fix is general, not Enter-specific: every tuple-mapped character that
the parser reissues after the escape timeout — the legacy Alt encoding is
always ESC + a single character — now reports ``alt+<key>`` (e.g. ESC+\x01
→ ``alt+ctrl+a``), matching what terminals with the kitty keyboard protocol
already report. Whole-sequence matches (ESC+arrows, ESC+Tab, …) and plain
keys take other branches and are untouched.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from textual import events
from textual._ansi_sequences import ANSI_SEQUENCES_KEYS
from textual._xterm_parser import XTermParser

_patched = False


def apply_alt_enter_fix() -> None:
    """Make ``XTermParser`` honor the Alt modifier for tuple-mapped keys.

    Idempotent; wraps (does not replace) the original
    ``_sequence_to_key_events`` so every other branch keeps its behavior.
    """
    global _patched
    if _patched:
        return
    _patched = True
    original: Any = XTermParser._sequence_to_key_events

    def _sequence_to_key_events(
        self: XTermParser, sequence: str, alt: bool = False
    ) -> Iterable[events.Key]:
        if alt:
            keys = ANSI_SEQUENCES_KEYS.get(sequence)
            if isinstance(keys, tuple):
                for key in keys:
                    # Same shape as the unpatched tuple branch, plus the
                    # alt prefix the single-character branch already adds.
                    name: str = key.value
                    yield events.Key(
                        f"alt+{name}",
                        sequence if len(sequence) == 1 else None,
                    )
                return
        yield from original(self, sequence, alt)

    _sequence_to_key_events.__name__ = original.__name__
    _sequence_to_key_events.__qualname__ = getattr(
        original, "__qualname__", original.__name__
    )
    XTermParser._sequence_to_key_events = _sequence_to_key_events  # type: ignore[method-assign]
