from __future__ import annotations

import sys
from pathlib import Path


def _print_help() -> None:
    from .version import APP_VERSION

    print(f"drcode {APP_VERSION} — task-centric console AI coding agent")
    print()
    print("Usage: drcode [options]")
    print()
    print("Options:")
    print("  -h, --help         Show this help message and exit.")
    print("  -V, --version      Show the application version and exit.")
    print("      --git-hub-sync Run one GitHub Issues sync cycle and exit.")
    print()
    print("With no options the interactive TUI starts and operates")
    print("on the current directory.")


def main() -> None:
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        _print_help()
        return

    if "--version" in sys.argv[1:] or "-V" in sys.argv[1:]:
        from .version import APP_VERSION

        print(APP_VERSION)
        return

    if "--git-hub-sync" in sys.argv[1:]:
        from .cli import run_github_sync

        raise SystemExit(run_github_sync(Path.cwd()))

    from .tui.app import run_app

    run_app(Path.cwd())
