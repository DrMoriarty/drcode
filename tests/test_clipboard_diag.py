"""Unit tests for drcode.tui.clipboard_diag: remote-session detection,
the tmux set-clipboard check (stubbed subprocess), OSC 52 reply
extraction from a raw input stream (incl. chunk-split replies), the
session verdict transitions and the toast wording.

Nothing here touches a real terminal or tmux binary.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from drcode.tui import clipboard_diag
from drcode.tui.clipboard_diag import (
    DiagState,
    _extract_osc52_replies,
    arm_probe,
    check_tmux_set_clipboard,
    detect_remote,
    disarm_probe,
    poll_probe_reply,
    toast_message,
)

# -- remote detection ---------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"TMUX": "/tmp/tmux-0/default,123,0"}, True),
        ({"SSH_CONNECTION": "1 2 3 4"}, True),
        ({"SSH_TTY": "/dev/pts/0"}, True),
        ({"TMUX": "/tmp/tmux-0/default,123,0", "SSH_CONNECTION": "1 2 3 4"}, True),
        # TERM_PROGRAM deliberately ignored (does not survive ssh).
        ({"TERM_PROGRAM": "iTerm.app"}, False),
        ({"SSH_CLIENT": ""}, False),
    ],
)
def test_detect_remote(env: dict[str, str], expected: bool) -> None:
    assert detect_remote(env) is expected


# -- tmux check ---------------------------------------------------------------


def _stub_run(monkeypatch: pytest.MonkeyPatch, result: Any) -> None:
    if isinstance(result, Exception):
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(result)
        )
    else:
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: result)


def test_tmux_check_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMUX", "/tmp/tmux-0/default,123,0")
    _stub_run(monkeypatch, SimpleNamespace(returncode=0, stdout="off\n", stderr=""))
    assert check_tmux_set_clipboard() == "off"


@pytest.mark.parametrize("value", ["on", "external"])
def test_tmux_check_pass_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("TMUX", "/tmp/tmux-0/default,123,0")
    _stub_run(
        monkeypatch, SimpleNamespace(returncode=0, stdout=f"{value}\n", stderr="")
    )
    assert check_tmux_set_clipboard() == value


def test_tmux_check_junk_value_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMUX", "/tmp/tmux-0/default,123,0")
    _stub_run(monkeypatch, SimpleNamespace(returncode=0, stdout="weird\n", stderr=""))
    assert check_tmux_set_clipboard() is None


def test_tmux_check_nonzero_exit_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMUX", "/tmp/tmux-0/default,123,0")
    _stub_run(monkeypatch, SimpleNamespace(returncode=1, stdout="", stderr="error"))
    assert check_tmux_set_clipboard() is None


def test_tmux_check_missing_binary_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TMUX", "/tmp/tmux-0/default,123,0")
    _stub_run(monkeypatch, FileNotFoundError("no tmux"))
    assert check_tmux_set_clipboard() is None


def test_tmux_check_timeout_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMUX", "/tmp/tmux-0/default,123,0")
    _stub_run(monkeypatch, subprocess.TimeoutExpired(cmd="tmux", timeout=2))
    assert check_tmux_set_clipboard() is None


def test_tmux_check_outside_tmux_skips_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TMUX", raising=False)

    def _explode(*a: Any, **kw: Any) -> None:
        raise AssertionError("tmux must not be called outside tmux")

    monkeypatch.setattr(subprocess, "run", _explode)
    assert check_tmux_set_clipboard() is None


# -- reply extraction ---------------------------------------------------------


@pytest.fixture(autouse=True)
def _armed():
    """Extraction only captures while a probe is armed (the hook is inert
    otherwise); arm for every test here."""
    arm_probe()
    yield
    disarm_probe()


REPLY = "\x1b]52;c;aGVsbG8=\x07"  # "hello" base64, BEL-terminated
REPLY_ST = "\x1b]52;c;aGVsbG8=\x1b\\"  # ST-terminated


def _hooked_feed(data: str) -> list[Any]:
    """Feed ``data`` through a real hooked XTermParser (the class the
    driver ends up with after install_parser_hook's factory), return the
    parsed tokens."""
    from textual._xterm_parser import XTermParser

    parser = clipboard_diag._make_hooked_parser(XTermParser)()
    return list(parser.feed(data))


def test_extract_complete_reply_bel() -> None:
    assert _extract_osc52_replies(f"abc{REPLY}def") == "abcdef"
    assert poll_probe_reply() == "aGVsbG8="
    assert poll_probe_reply() is None


def test_extract_complete_reply_st() -> None:
    assert _extract_osc52_replies(REPLY_ST) == ""
    assert poll_probe_reply() == "aGVsbG8="


def test_extract_empty_payload_counts_as_reply() -> None:
    # An empty payload still proves the terminal answers the query.
    assert _extract_osc52_replies("\x1b]52;c;\x07") == ""
    assert poll_probe_reply() == ""


def test_extract_reply_split_across_feeds() -> None:
    # Through the hooked parser: the split reply must never leak into the
    # key parser, and the two chunks must reassemble into one reply.
    tokens_first = _hooked_feed("junk\x1b]52;c;aGVs")
    assert not any(
        getattr(t, "key", "").startswith(("a", "G", "V", "s")) for t in tokens_first
    )
    assert poll_probe_reply() is None  # nothing queued until complete
    _hooked_feed("bG8=\x07tail")
    assert poll_probe_reply() == "aGVsbG8="


def test_extract_no_reply_leaves_data_intact() -> None:
    data = "\x1b[A\x1b[<0;1;1M hello"
    assert _extract_osc52_replies(data) == data
    assert poll_probe_reply() is None


# -- verdict transitions ------------------------------------------------------


def test_verdict_local_when_not_remote() -> None:
    state = DiagState(remote=False)
    assert state.verdict() == "local"


def test_verdict_tmux_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMUX", "/tmp/tmux-0/default,123,0")
    _stub_run(monkeypatch, SimpleNamespace(returncode=0, stdout="off\n", stderr=""))
    state = DiagState(remote=True)
    assert state.verdict() == "unknown"  # not checked yet
    state.run_tmux_check()
    # Known breakage outranks even a hypothetical probe confirmation.
    state.probe_ok = True
    assert state.verdict() == "tmux-off"
    assert state.tmux_off


def test_verdict_confirmed_by_probe() -> None:
    state = DiagState(remote=True)
    state.tmux_value = "external"
    state.tmux_checked = True
    state.probe_done = True
    assert state.verdict() == "unknown"  # silence is unknown, never breakage
    state.probe_ok = True
    assert state.verdict() == "confirmed"


def test_verdict_unknown_without_probe() -> None:
    state = DiagState(remote=True)
    state.tmux_value = "on"
    state.tmux_checked = True
    state.probe_done = True
    assert state.verdict() == "unknown"


# -- toast wording ------------------------------------------------------------


def test_toast_message_confident_when_local_or_confirmed() -> None:
    assert toast_message("local", "hi") == "✓ copied: hi"
    assert toast_message("confirmed", "hi") == "✓ copied: hi"


def test_toast_message_tmux_off_names_fix() -> None:
    msg = toast_message("tmux-off", "hi")
    assert "✓ copied" not in msg
    assert "tmux set-clipboard off" in msg
    assert "set -g set-clipboard on" in msg
    assert "~/.tmux.conf" in msg


def test_toast_message_unknown_does_not_promise_delivery() -> None:
    msg = toast_message("unknown", "hi")
    assert "hi" in msg
    assert "tmux set-clipboard" in msg
    assert "OSC 52" in msg
    # The honest wording keeps the "copied" verb but frames it as OSC 52.
    assert "OSC 52" in msg.splitlines()[0]


# -- probe window gating ------------------------------------------------------


def test_extraction_inert_when_not_armed() -> None:
    disarm_probe()
    # The hook is a passthrough outside the probe window: the reply-shaped
    # sequence flows to the (wrapped) parser untouched — nothing queued,
    # nothing held back.
    _hooked_feed(f"pre{REPLY}post")
    assert poll_probe_reply() is None
    arm_probe()
    assert poll_probe_reply() is None
    disarm_probe()
