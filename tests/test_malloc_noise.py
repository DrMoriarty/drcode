"""macOS malloc-stack-logging noise: the fd2 stderr filter installed at
startup (darwin, tty stderr) and the sanitizer for captured child output.

Diagnosis (2026-10-31, task 27ef7164): siagent/cpard toggle malloc stack
logging remotely in live processes; libmalloc then makes every fork()ed
child print `Name(pid) MallocStackLogging: ...` to the INHERITED stderr in
the fork→exec window — before subprocess dup2's its capture pipes. The
startup filter splits the TUI's fd2 through a byte-filter child; the
sanitizer strips the lines that land in captured outputs anyway (a child
toggled ON mid-run writes the report into its own captured stderr).

The filter must be byte-exact, not `grep -v`: Textual frames often carry no
trailing newline and would stall in grep's stdio buffer (up to seconds of
key-echo/repaint latency), so the filter loop is tested to pass newline-less
frames through while no further input arrives.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import pytest

import drcode
from drcode.tools import strip_malloc_stack_logging_noise

# -- sanitizer -----------------------------------------------------------------


def test_strip_removes_noise_lines() -> None:
    text = (
        "git: warning: something real\n"
        "Python(44765) MallocStackLogging: can't turn off malloc stack logging "
        "because it was not enabled.\n"
        "exit=0\n"
    )
    assert strip_malloc_stack_logging_noise(text) == (
        "git: warning: something real\nexit=0\n"
    )


def test_strip_keeps_last_line_without_newline() -> None:
    text = "real output\nPython(1) MallocStackLogging: turning off stack logging."
    assert strip_malloc_stack_logging_noise(text) == "real output\n"


def test_strip_keeps_quoted_word_in_other_context() -> None:
    # An agent legitimately grepping logs for the same strings must not have
    # its findings sanitized away: only the exact libmalloc report shape
    # (`Name(pid) MallocStackLogging: ...`) is dropped.
    text = (
        "MallocStackLogging is documented in the manual\n"
        "grep: MallocStackLogging: no such file\n"
        "user said: MallocStackLogging=1 helps\n"
    )
    assert strip_malloc_stack_logging_noise(text) == text


def test_strip_handles_pid_like_names_and_multiple_lines() -> None:
    text = (
        "uv(43486) MallocStackLogging: stack logs being written to "
        "/private/tmp/stack-logs.43486.10fcfc000.uv\n"
        "Python Helper(12) MallocStackLogging: could not tag MSL-related "
        "memory as no_footprint\n"
        "siagent(32197) MallocStackLogging: can't turn off malloc stack "
        "logging because it was not enabled.\n"
        "kept\n"
    )
    assert strip_malloc_stack_logging_noise(text) == "kept\n"


# -- startup filter gating -------------------------------------------------------


@pytest.fixture
def darwin_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(os, "isatty", lambda fd: fd == 2)


def test_filter_skipped_when_stderr_not_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(os, "isatty", lambda fd: False)
    assert not drcode._stderr_filter_needed()


def test_filter_skipped_outside_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "isatty", lambda fd: True)
    assert not drcode._stderr_filter_needed()


def test_filter_skipped_when_already_installed(
    monkeypatch: pytest.MonkeyPatch, darwin_tty: None
) -> None:
    monkeypatch.setenv(drcode._MSL_FILTER_ENV, "1")
    assert not drcode._stderr_filter_needed()


def test_filter_needed_on_darwin_tty(darwin_tty: None) -> None:
    assert drcode._stderr_filter_needed()


# -- the held-tail fuse ----------------------------------------------------------


def test_tail_esc_frame_never_held() -> None:
    # Textual frames always contain ESC — they must pass through instantly.
    assert not drcode._msl_tail_may_become_noise(b"\x1b[3;5Hcursor")
    assert not drcode._msl_tail_may_become_noise(b"\x1b[1;1H\x1b[2K")


def test_tail_potential_noise_prefixes_held() -> None:
    assert drcode._msl_tail_may_become_noise(b"Python(42")
    assert drcode._msl_tail_may_become_noise(b"git(7) ")
    assert drcode._msl_tail_may_become_noise(b"git(7) MallocStack")
    assert drcode._msl_tail_may_become_noise(
        b"git(7) MallocStackLogging: partial message"
    )
    assert drcode._msl_tail_may_become_noise(b"plain name still open")


def test_tail_innocent_or_oversized_fragments_flushed() -> None:
    assert not drcode._msl_tail_may_become_noise(b"Traceback (most")
    assert not drcode._msl_tail_may_become_noise(b"Python(42) Malformed output")
    assert not drcode._msl_tail_may_become_noise(b"Python(42) x")
    assert not drcode._msl_tail_may_become_noise(b"x" * 200)


# -- the byte filter loop --------------------------------------------------------


class _FilterFixture:
    """A live _msl_filter_loop in a thread, fed through pipes."""

    def __init__(self) -> None:
        self.in_r, self.in_w = os.pipe()
        self.out_r, self.out_w = os.pipe()
        os.set_blocking(self.out_r, False)
        self.thread = threading.Thread(
            target=drcode._msl_filter_loop, args=(self.in_r, self.out_w), daemon=True
        )
        self.thread.start()

    def feed(self, data: bytes) -> None:
        os.write(self.in_w, data)

    def drain(self, timeout: float = 5.0) -> bytes:
        """Read everything currently available, waiting up to `timeout`
        for the first byte (then only for a quiet gap)."""
        got = b""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                part = os.read(self.out_r, 65536)
            except BlockingIOError:
                if got:
                    break
                time.sleep(0.005)
                continue
            if not part:
                break
            got += part
        return got

    def close_input(self) -> None:
        os.close(self.in_w)
        self.thread.join(timeout=10)
        assert not self.thread.is_alive()

    def read_all(self) -> bytes:
        os.close(self.out_w)  # the loop thread is done — drop our copy for EOF
        os.set_blocking(self.out_r, True)
        chunks = []
        while True:
            part = os.read(self.out_r, 65536)
            if not part:
                break
            chunks.append(part)
        return b"".join(chunks)

    def cleanup(self) -> None:
        for fd in (self.in_r, self.out_r, self.out_w):
            try:
                os.close(fd)
            except OSError:
                pass


def _run_filter(chunks: list[bytes]) -> bytes:
    fx = _FilterFixture()
    try:
        for chunk in chunks:
            fx.feed(chunk)
        fx.close_input()
        return fx.read_all()
    finally:
        fx.cleanup()


def test_filter_drops_noise_keeps_rest() -> None:
    out = _run_filter(
        [
            b"err-line\n",
            (
                b"Python(4242) MallocStackLogging: can't turn off malloc stack "
                b"logging because it was not enabled.\n"
            ),
            (
                b"git(7) MallocStackLogging: stack logs being written to "
                b"/private/tmp/stack-logs.7\n"
            ),
        ]
    )
    assert out == b"err-line\n"


def test_filter_keeps_word_in_other_context() -> None:
    # Only the exact libmalloc report shape is dropped.
    text = (
        b"MallocStackLogging is documented in the manual\n"
        b"grep: MallocStackLogging: no such file\n"
        b"user said: MallocStackLogging=1 helps\n"
    )
    assert _run_filter([text]) == text


def test_filter_handles_pid_like_names_and_multiple_lines() -> None:
    noise = (
        b"uv(43486) MallocStackLogging: stack logs being written to "
        b"/private/tmp/stack-logs.43486.10fcfc000.uv\n"
        b"Python Helper(12) MallocStackLogging: could not tag MSL-related "
        b"memory as no_footprint\n"
        b"siagent(32197) MallocStackLogging: can't turn off malloc stack "
        b"logging because it was not enabled.\n"
    )
    assert _run_filter([noise, b"kept\n"]) == b"kept\n"


def test_filter_flushes_unterminated_tail_at_eof() -> None:
    # An incomplete last line can no longer grow at EOF — it is passed
    # through (it never became a noise *line*).
    out = _run_filter(
        [b"real output\nPython(1) MallocStackLogging: turning off stack logging."]
    )
    assert out == (
        b"real output\nPython(1) MallocStackLogging: turning off stack logging."
    )


def test_filter_passes_newline_less_frame_immediately() -> None:
    # THE regression: a Textual-like frame without trailing newline must
    # reach the terminal while NO further input arrives — under the grep
    # filter it stalled until the next newline-terminated write.
    fx = _FilterFixture()
    try:
        fx.feed(b"\x1b[3;5Hcursor")
        assert fx.drain(timeout=1.0) == b"\x1b[3;5Hcursor"
    finally:
        fx.close_input()
        fx.cleanup()


def test_filter_holds_possible_noise_start_until_line_completes() -> None:
    fx = _FilterFixture()
    try:
        fx.feed(b"Python(42")
        assert fx.drain(timeout=0.2) == b""  # held: could grow into a noise line
        fx.feed(b") MallocStackLogging: not enabled.\n")
        assert fx.drain(timeout=0.5) == b""  # completed line was noise — dropped
        fx.feed(b"\x1b[1;1Hok")
        assert fx.drain(timeout=1.0) == b"\x1b[1;1Hok"
    finally:
        fx.close_input()
        fx.cleanup()


def test_filter_flushes_held_noise_message_at_eof() -> None:
    out = _run_filter([b"Python(42) MallocStackLogging: unterminated message"])
    assert out == b"Python(42) MallocStackLogging: unterminated message"


def test_filter_on_captured_child_stderr() -> None:
    # End-to-end shape: what the sanitizer sees for a child toggled ON
    # mid-run, fed through the real filter loop.
    emitter = (
        "echo out-line; "
        "echo \"Python(4242) MallocStackLogging: can't turn off "
        'malloc stack logging because it was not enabled." >&2; '
        "echo err-line >&2"
    )
    proc = subprocess.run(
        ["/bin/sh", "-c", emitter],
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == b"out-line"
    assert _run_filter([proc.stderr]) == b"err-line\n"
