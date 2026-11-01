"""macOS malloc-stack-logging noise: the fd2 stderr filter installed at
startup (darwin, tty stderr) and the sanitizer for captured child output.

Diagnosis (2026-10-31, task 27ef7164): siagent/cpard toggle malloc stack
logging remotely in live processes; libmalloc then makes every fork()ed
child print `Name(pid) MallocStackLogging: ...` to the INHERITED stderr in
the fork→exec window — before subprocess dup2's its capture pipes. The
startup filter turns the TUI's fd2 into a `grep -v` stream (same mechanism
as the manual `uv run drcode 2> >(grep -v MallocStackLogging >&2)`); the
sanitizer strips the lines that land in captured outputs anyway (a child
toggled ON mid-run writes the report into its own captured stderr).
"""

from __future__ import annotations

import os
import subprocess
import sys

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


# -- startup filter argv --------------------------------------------------------


@pytest.fixture
def darwin_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(os, "isatty", lambda fd: fd == 2)


def test_filter_argv_skipped_when_stderr_not_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(os, "isatty", lambda fd: False)
    assert drcode._stderr_filter_argv() is None


def test_filter_argv_skipped_outside_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "isatty", lambda fd: True)
    assert drcode._stderr_filter_argv() is None


def test_filter_argv_skipped_when_already_installed(
    monkeypatch: pytest.MonkeyPatch, darwin_tty: None
) -> None:
    monkeypatch.setenv(drcode._MSL_FILTER_ENV, "1")
    assert drcode._stderr_filter_argv() is None


def test_filter_argv_skipped_without_bash(
    monkeypatch: pytest.MonkeyPatch, darwin_tty: None
) -> None:
    real_exists = os.path.exists
    monkeypatch.setattr(
        os.path, "exists", lambda p: False if p == "/bin/bash" else real_exists(p)
    )
    assert drcode._stderr_filter_argv() is None


def test_filter_argv_console_script(darwin_tty: None) -> None:
    monkey_argv = ["/Users/x/.local/bin/drcode", "--version"]
    original_argv, original_platform = sys.argv, sys.platform
    sys.argv = monkey_argv
    try:
        argv = drcode._stderr_filter_argv()
    finally:
        sys.argv, sys.platform = original_argv, original_platform
    assert argv is not None
    assert argv[:3] == ["/bin/bash", "-c", drcode._MSL_FILTER_SCRIPT]
    # $0 is a throwaway marker; the real command follows in $@
    assert argv[3] == drcode._MSL_FILTER_MARKER
    assert argv[4:] == monkey_argv


def test_filter_argv_python_m(darwin_tty: None) -> None:
    main_py = str(__import__("pathlib").Path(drcode.__file__).with_name("__main__.py"))
    original_argv = sys.argv
    sys.argv = [main_py, "-x"]
    try:
        argv = drcode._stderr_filter_argv()
    finally:
        sys.argv = original_argv
    assert argv is not None
    assert argv[4] == sys.executable
    assert argv[5] == main_py
    assert argv[6:] == ["-x"]


# -- the bash filter script itself ---------------------------------------------


@pytest.mark.skipif(not os.path.exists("/bin/bash"), reason="bash required")
def test_bash_filter_drops_noise_keeps_rest() -> None:
    emitter = (
        "echo out-line; "
        "echo \"Python(4242) MallocStackLogging: can't turn off "
        'malloc stack logging because it was not enabled." >&2; '
        'echo "git(7) MallocStackLogging: stack logs being written to '
        '/private/tmp/stack-logs.7" >&2; '
        "echo err-line >&2"
    )
    proc = subprocess.run(
        [
            "/bin/bash",
            "-c",
            drcode._MSL_FILTER_SCRIPT,
            drcode._MSL_FILTER_MARKER,
            "/bin/sh",
            "-c",
            emitter,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "out-line"
    # The real stderr survives; the noise does not.
    assert proc.stderr.strip() == "err-line"
