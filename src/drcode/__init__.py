from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if "--version" in sys.argv[1:] or "-V" in sys.argv[1:]:
        from .version import APP_VERSION

        print(APP_VERSION)
        return

    if "--git-hub-sync" in sys.argv[1:]:
        from .cli import run_github_sync

        raise SystemExit(run_github_sync(Path.cwd()))

    from .tui.app import run_app

    run_app(Path.cwd())
