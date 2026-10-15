from __future__ import annotations

from pathlib import Path


def main() -> None:
    from .tui.app import run_app

    run_app(Path.cwd())
