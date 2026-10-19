from __future__ import annotations

import sys
from pathlib import Path

from .models import TaskKind


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
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        _print_help()
        return

    if "--version" in args or "-V" in args:
        from .version import APP_VERSION

        print(APP_VERSION)
        return

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
