"""Recent-projects storage for DrCode Desktop.

A tiny JSON file at ``~/.config/drcode-desktop.json`` (XDG_CONFIG_HOME
aware). This is the *desktop shell's* own config and is deliberately
separate from drcode's user config — the shell must not touch anything
the TUI manages.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

FILE_NAME = "drcode-desktop.json"
MAX_RECENTS = 10


def config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / FILE_NAME


def load_recents() -> list[dict[str, object]]:
    path = config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    recents = data.get("recents") if isinstance(data, dict) else None
    if not isinstance(recents, list):
        return []
    out: list[dict[str, object]] = []
    for item in recents:
        if not isinstance(item, dict):
            continue
        project = item.get("project")
        if not isinstance(project, str) or not project:
            continue
        entry: dict[str, object] = {
            "project": project,
            "last_used": item.get("last_used") or 0,
        }
        if isinstance(item.get("title"), str) and item["title"]:
            entry["title"] = item["title"]
        out.append(entry)
    return out


def add_recent(project: str, *, title: str | None = None) -> list[dict[str, object]]:
    """Record a launched project (most recent first, deduped, capped)."""
    recents = load_recents()
    recents = [r for r in recents if r.get("project") != project]
    entry: dict[str, object] = {"project": project, "last_used": int(time.time())}
    if title:
        entry["title"] = title
    recents.insert(0, entry)
    recents = recents[:MAX_RECENTS]
    _save(recents)
    return recents


def _save(recents: list[dict[str, object]]) -> None:
    path = config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"recents": recents}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError:
        pass  # storage is best-effort; never block launching a project


def project_title(project: str) -> str:
    name = Path(project).name or project
    return name
