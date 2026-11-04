"""Alt+Enter decoding in Terminal.app (tui/key_compat.py).

macOS Terminal.app ("Use Option as Meta key") sends Alt+Enter as the legacy
ESC+CR sequence. Textual 8.2.8's ``XTermParser`` drops the Alt modifier when
the sequence maps to a tuple of ``Keys`` (``'\\r'`` → ``(Keys.Enter,)``),
so the key arrived as plain ``enter`` — ChatInput's submit binding SENT the
draft instead of inserting a newline. The parser patch must restore
``alt+enter`` without changing any unmodified key.
"""

from __future__ import annotations

import time
from typing import Any

from textual import events
from textual._xterm_parser import XTermParser

from drcode.tui.key_compat import apply_alt_enter_fix


def _parse_keys(data: str) -> list[str]:
    """Feed bytes to the parser, let the escape timeout elapse, tick, and
    return the decoded key names."""
    # No debug flag: a truthy second argument makes textual open keys.log
    # in the CWD and write a line per token.
    parser = XTermParser()
    tokens = list(parser.feed(data))
    time.sleep(0.2)  # past textual's ESCAPE_DELAY (100 ms default)
    tokens.extend(parser.tick())
    return [event.key for event in tokens if isinstance(event, events.Key)]


def test_esc_cr_decodes_as_alt_enter() -> None:
    apply_alt_enter_fix()
    assert _parse_keys("\x1b\r") == ["alt+enter"]


def test_plain_enter_unchanged() -> None:
    apply_alt_enter_fix()
    assert _parse_keys("\r") == ["enter"]


def test_esc_letter_still_alt_prefixed() -> None:
    # The single-character branch was never broken — must stay intact.
    apply_alt_enter_fix()
    assert _parse_keys("\x1ba") == ["alt+a"]


def test_kitty_alt_enter_unchanged() -> None:
    # Terminals with the kitty keyboard protocol already report alt+enter.
    apply_alt_enter_fix()
    assert _parse_keys("\x1b[13;3u") == ["alt+enter"]


def test_esc_control_char_gets_alt_prefix() -> None:
    # The fix covers every tuple-mapped character on the reissue path,
    # not just Enter (kitty encoding would report "alt+ctrl+a" too).
    apply_alt_enter_fix()
    assert _parse_keys("\x1b\x01") == ["alt+ctrl+a"]


def test_patch_is_idempotent(monkeypatch: Any) -> None:
    apply_alt_enter_fix()
    wrapped = XTermParser._sequence_to_key_events
    apply_alt_enter_fix()
    assert XTermParser._sequence_to_key_events is wrapped
