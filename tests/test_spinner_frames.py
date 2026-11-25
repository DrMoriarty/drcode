"""Terminal-capable spinner selection (core.blocks.spinner_frames).

The braille frames only render on a terminal whose character set carries
them; a single-byte session (C/POSIX, KOI8-*, ISO-8859-*, …) shows the
plain `/ - \\ |` wheel instead. The selection is driven by the session's
locale environment, with an explicit `DRCODE_SPINNER` escape hatch and
the stderr encoding as the last-resort signal (no locale, e.g. Windows).
"""

from __future__ import annotations

from drcode.core import blocks
from drcode.core.blocks import (
    ASCII_SPINNER_FRAMES,
    BRAILLE_SPINNER_FRAMES,
    spinner_frames,
)


def test_ascii_frame_set_is_pure_ascii() -> None:
    assert ASCII_SPINNER_FRAMES == "/-\\|"
    ASCII_SPINNER_FRAMES.encode("ascii")


def test_braille_frame_set_is_the_braille_pattern() -> None:
    assert BRAILLE_SPINNER_FRAMES == "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    for ch in BRAILLE_SPINNER_FRAMES:
        assert 0x2800 <= ord(ch) <= 0x28FF


def test_single_byte_locales_pick_the_ascii_wheel() -> None:
    for locale in ("C", "POSIX", "ru_RU.KOI8-R", "ru_RU", "C.KOI8", "en_US.ISO8859-1"):
        assert spinner_frames(environ={"LANG": locale}) == ASCII_SPINNER_FRAMES


def test_utf8_locales_pick_braille() -> None:
    for locale in ("en_US.UTF-8", "C.utf8", "de_DE@UTF-8", "en_US.utf8", "ru_RU.UTF-8"):
        assert spinner_frames(environ={"LANG": locale}) == BRAILLE_SPINNER_FRAMES


def test_locale_precedence_lc_all_beats_lang() -> None:
    assert (
        spinner_frames(environ={"LC_ALL": "C", "LANG": "en_US.UTF-8"})
        == ASCII_SPINNER_FRAMES
    )
    assert (
        spinner_frames(environ={"LC_ALL": "en_US.UTF-8", "LANG": "C"})
        == BRAILLE_SPINNER_FRAMES
    )
    # LC_CTYPE sits between the two.
    assert (
        spinner_frames(environ={"LC_CTYPE": "ru_RU.KOI8-R", "LANG": "en_US.UTF-8"})
        == ASCII_SPINNER_FRAMES
    )


def test_no_locale_falls_back_to_stderr_encoding(monkeypatch) -> None:
    class _Stream:
        def __init__(self, encoding: str) -> None:
            self.encoding = encoding

    for encoding, expected in (
        ("cp1251", ASCII_SPINNER_FRAMES),
        ("utf-8", BRAILLE_SPINNER_FRAMES),
        ("", ASCII_SPINNER_FRAMES),
    ):
        monkeypatch.setattr(blocks.sys, "__stderr__", _Stream(encoding))
        assert spinner_frames(environ={}) == expected


def test_explicit_override_wins(monkeypatch) -> None:
    # ...over a UTF-8 session (a terminal font without the braille block
    # would render empty boxes) — and over a single-byte one.
    assert (
        spinner_frames(environ={"LANG": "en_US.UTF-8", "DRCODE_SPINNER": "ascii"})
        == ASCII_SPINNER_FRAMES
    )
    assert (
        spinner_frames(environ={"LANG": "C", "DRCODE_SPINNER": "braille"})
        == BRAILLE_SPINNER_FRAMES
    )
    assert (
        spinner_frames(environ={"LANG": "en_US.UTF-8", "DRCODE_SPINNER": "BRATTLE"})
        == BRAILLE_SPINNER_FRAMES
    )  # a typo'ed value is ignored, not fatal


def test_module_constant_is_a_valid_frame_set() -> None:
    # Resolved for this process's own terminal at import time: one of the
    # two sets, and the ASCII one must encode as plain ascii by
    # construction (the whole point on a single-byte terminal).
    assert blocks.SPINNER_FRAMES in (ASCII_SPINNER_FRAMES, BRAILLE_SPINNER_FRAMES)
    if blocks.SPINNER_FRAMES == ASCII_SPINNER_FRAMES:
        ASCII_SPINNER_FRAMES.encode("ascii")
