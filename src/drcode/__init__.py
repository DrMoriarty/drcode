from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import TaskKind


# -- macOS stderr malloc-stack-logging filter ------------------------------------
#
# Corporate monitoring daemons (siagent/SearchInform, Check Point cpard) toggle
# malloc stack logging remotely in live processes, and libmalloc then makes
# every fork()ed child print a `Name(pid) MallocStackLogging: ...` report line
# to the INHERITED stderr between fork() and exec() — before subprocess's
# capture pipes are dup2'd, so no spawn-side redirection can intercept it.
# The TUI forks constantly (git polls, run_bash, hooks), so an affected
# session floods the terminal with these lines. CPython 3.14 on macOS cannot
# avoid the fork path (close_fds defaults to True and
# os.POSIX_SPAWN_CLOSEFROM is missing), so the fix is to split the TUI's own
# fd2 through a filter process that drops exactly the noise lines and passes
# everything else (tracebacks, warnings) through.
#
# The filter is a BYTE stream, deliberately not `grep -v`: Textual writes every
# rendered frame to stderr, and most frames (cursor moves, partial CHOPS
# updates) contain no trailing newline. grep's stdio buffering holds bytes
# until a line completes, so newline-less frames stall in its buffer until the
# next newline-terminated write — measured 0.3–1.5 s of delivery latency for
# ~60% of all frames, i.e. the key-echo / cursor-blink / stale-bottom-line
# stalls. This filter writes everything through immediately; it only HOLDS a
# trailing fragment that could still grow into a noise line, and even that
# only while the fragment is short, newline-free and ESC-free (Textual frames
# always contain ESC, so real frames are never held).
#
# The filter child is a fork() of this very process — it already runs on the
# right interpreter. Launching one via `sys.executable -c` is not an option:
# in a frozen bundle sys.executable is the app binary, and /usr/bin/python3
# would pop the CLT install dialog. The fork happens in main() before any
# threads exist, and the child never returns into application code.

_MSL_FILTER_ENV = "DRCODE_MSL_STDERR_FILTER"
# One libmalloc report line: name (no parens/newline), `(pid) MallocStack
# Logging: `, message, newline. Applied to complete lines only.
_MSL_NOISE_RE = re.compile(rb"(?m)^[^(\n]{0,100}\(\d+\) MallocStackLogging: [^\n]*\n")
# After the last `(` of a held fragment: digits, optional `)`, then an
# optional PREFIX of the fixed keyword — anything else can never grow into
# a noise line.
_MSL_TAIL_PID_RE = re.compile(rb"\d{0,8}\)?")
_MSL_TAIL_KEYWORD = b" MallocStackLogging: "
# A longer newline-less fragment is flushed even if it may be a noise line:
# unbounded holding is worse than a rare half-filtered line.
_MSL_TAIL_LIMIT = 128


def _stderr_filter_needed() -> bool:
    """Whether the fd2 noise filter must be installed: only on macOS with
    stderr attached to the terminal. Redirected stderr (file/pipe) is the
    user's own filtering — a nested `2> >(grep …)` workaround stays
    untouched because a pipe is not a tty — and every child of a live TUI
    inherits the marker env var, so nested drcode instances never stack
    filters."""
    return (
        sys.platform == "darwin"
        and os.isatty(2)
        and not os.environ.get(_MSL_FILTER_ENV)
    )


def _msl_tail_may_become_noise(tail: bytes) -> bool:
    """Whether a trailing newline-less fragment could still be the beginning
    of a noise line and therefore must be held back until the line
    completes (or the fuse fires)."""
    if not tail or b"\n" in tail or b"\x1b" in tail or len(tail) > _MSL_TAIL_LIMIT:
        return False
    if b"MallocStackLogging" in tail:
        return True  # already inside the message part of a noise line
    i = tail.rfind(b"(")
    if i == -1:
        return True  # still inside the process-name part
    m = _MSL_TAIL_PID_RE.match(tail, i + 1)
    after = tail[m.end() :] if m else tail[i + 1 :]
    return not after or _MSL_TAIL_KEYWORD.startswith(after)


def _msl_filter_loop(read_fd: int, write_fd: int) -> None:
    """The filter child body: read raw stderr bytes, drop completed noise
    lines, pass everything else through immediately (unbuffered)."""
    buf = b""

    def emit(data: bytes) -> None:
        while data:
            data = data[os.write(write_fd, data) :]

    while True:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            break
        buf += chunk
        cut = buf.rfind(b"\n")
        if cut >= 0:
            emit(_MSL_NOISE_RE.sub(b"", buf[: cut + 1]))
            buf = buf[cut + 1 :]
        if buf and not _msl_tail_may_become_noise(buf):
            emit(buf)
            buf = b""
    if buf:  # EOF: an unterminated tail can no longer grow — flush it
        emit(buf)


def _install_stderr_filter() -> None:
    """Split fd2 through the byte-filter child (no-op unless needed — see
    the module comment). Must run before anything spawns children: every
    child born afterwards inherits the filtered fd2. The parent keeps its
    PID and exit status (no re-exec); the filter child waits on the pipe
    and dies of EOF when the TUI and all its descendants are gone."""
    if not _stderr_filter_needed():
        return
    pipe_r, pipe_w = os.pipe()
    tty = os.dup(2)
    try:
        pid = os.fork()
    except OSError:
        os.close(pipe_r)
        os.close(pipe_w)
        os.close(tty)
        return
    if pid == 0:
        # Filter child: the pipe becomes its stdin, the original stderr its
        # stdout. No exception may unwind it back into application code.
        try:
            os.close(pipe_w)
            os.dup2(pipe_r, 0)
            os.close(pipe_r)
            os.dup2(tty, 1)
            os.close(tty)
            _msl_filter_loop(0, 1)
        finally:
            os._exit(0)
    os.close(pipe_r)
    os.close(tty)
    os.dup2(pipe_w, 2)
    os.close(pipe_w)
    os.environ[_MSL_FILTER_ENV] = "1"


def _print_help() -> None:
    from .version import APP_VERSION

    print(f"drcode {APP_VERSION} — task-centric console AI coding agent")
    print()
    print("Usage: drcode [options]")
    print()
    print("Options:")
    print("  -h, --help            Show this help message and exit.")
    print("  -V, --version         Show the application version and exit.")
    print("      --git-hub-sync    Run one GitHub Issues sync cycle and exit.")
    print("      --bug-report DESC Create a bug-report task with the given")
    print("                        description and exit (no TUI).")
    print("      --feature-request DESC")
    print("                        Create a feature-request task with the given")
    print("                        description and exit (no TUI).")
    print()
    print("With no options the interactive TUI starts and operates")
    print("on the current directory.")


def _pop_option_value(args: list[str], flag: str) -> str | None:
    """Extract the value of a `--flag VALUE` / `--flag=VALUE` option from
    args, removing both tokens. Returns None when the option is absent;
    a missing value is a usage error."""
    for i, arg in enumerate(args):
        if arg == flag or arg.startswith(flag + "="):
            if "=" in arg:
                value = arg.split("=", 1)[1]
                del args[i]
            else:
                if i + 1 >= len(args):
                    print(
                        f"error: {flag} requires a description argument.",
                        file=sys.stderr,
                    )
                    raise SystemExit(2)
                value = args[i + 1]
                del args[i : i + 2]
            return value
    return None


def main() -> None:
    _install_stderr_filter()

    args = sys.argv[1:]

    # Cheap flags first: they must not pay for the heavy imports below.
    if "--help" in args or "-h" in args:
        _print_help()
        return

    if "--version" in args or "-V" in args:
        from .version import APP_VERSION

        print(APP_VERSION)
        return

    from .models import TaskKind

    if "--git-hub-sync" in args:
        from .cli import run_github_sync

        raise SystemExit(run_github_sync(Path.cwd()))

    # Headless task creation: `--bug-report DESC` / `--feature-request DESC`
    # (or `--bug-report=DESC`). The description may be arbitrarily long —
    # it is stored verbatim as the task's first message.
    kind_flags: list[tuple[str, TaskKind]] = [
        ("--bug-report", TaskKind.BUG),
        ("--feature-request", TaskKind.FEATURE),
    ]
    requested: list[tuple[TaskKind, str]] = []
    for flag, kind in kind_flags:
        value = _pop_option_value(args, flag)
        if value is not None:
            requested.append((kind, value))
    if requested:
        if len(requested) > 1:
            print(
                "error: pass only one of --bug-report / --feature-request "
                "per invocation.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        kind, description = requested[0]
        from .cli import create_task_from_cli

        raise SystemExit(create_task_from_cli(Path.cwd(), kind, description))

    from .tui.app import run_app

    run_app(Path.cwd())
