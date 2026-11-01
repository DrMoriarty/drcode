from __future__ import annotations

import os
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
# os.POSIX_SPAWN_CLOSEFROM is missing), so the only in-process fix is to make
# the TUI's own fd2 a filtered stream: re-exec through a bash process
# substitution that drops exactly the noise lines and passes everything else
# (tracebacks, warnings) through. Mirrors the manual
# `uv run drcode 2> >(grep -v MallocStackLogging >&2)` workaround.

_MSL_FILTER_ENV = "DRCODE_MSL_STDERR_FILTER"
_MSL_FILTER_SCRIPT = 'exec 2> >(grep -v MallocStackLogging >&2); exec "$@"'
_MSL_FILTER_MARKER = "drcode-stderr-filter"


def _stderr_filter_argv() -> list[str] | None:
    """The re-exec argv installing the fd2 noise filter, or None when it
    must not (or need not) be installed: not macOS, stderr is not the
    terminal (redirected to a file/pipe — the user manages their own
    filtering, and a nested `2> >(grep …)` workaround stays untouched
    because a pipe is not a tty), the filter is already installed, or bash
    is unavailable (non-macOS fallbacks, hardened systems)."""
    if sys.platform != "darwin" or not os.isatty(2):
        return None
    if os.environ.get(_MSL_FILTER_ENV):
        return None
    if not os.path.exists("/bin/bash"):
        return None
    argv0 = sys.argv[0]
    rest = sys.argv[1:]
    if argv0.endswith("__main__.py") and os.path.exists(argv0):
        # `python -m drcode`: exec the module file through the same
        # interpreter (preserving -u), a shebang-less file cannot be
        # exec'd directly.
        cmd: list[str] = [sys.executable]
        if not getattr(sys.flags, "buffered", 1):  # `-u`
            cmd.append("-u")
        cmd.append(argv0)
        return ["/bin/bash", "-c", _MSL_FILTER_SCRIPT, _MSL_FILTER_MARKER, *cmd, *rest]
    # Console script (`~/.local/bin/drcode`), frozen binary, `python
    # /path/to/script`: argv[0] is executable as-is; a bare name found via
    # PATH stays resolvable because bash's exec searches PATH too.
    return ["/bin/bash", "-c", _MSL_FILTER_SCRIPT, _MSL_FILTER_MARKER, argv0, *rest]


def _install_stderr_filter() -> None:
    """Re-exec self with a filtered stderr (no-op unless needed — see
    _stderr_filter_argv). Must run before anything spawns children."""
    argv = _stderr_filter_argv()
    if argv is None:
        return
    os.environ[_MSL_FILTER_ENV] = "1"
    os.execv(argv[0], argv)


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
