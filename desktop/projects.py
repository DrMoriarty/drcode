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

# Terminal font bounds — mirrors the JS clamps in web/term.js so the
# stored value is sane no matter which side validates first.
MIN_FONT_SIZE = 6
MAX_FONT_SIZE = 40


def config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / FILE_NAME


def load_config() -> dict[str, object]:
    """Whole desktop config file; missing/corrupt file → empty dict."""
    path = config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_recents() -> list[dict[str, object]]:
    recents = load_config().get("recents")
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


def remove_recent(project: str) -> list[dict[str, object]]:
    """Drop one entry by its project path; returns the updated recents.

    Matching is on the stored ``project`` string (exactly the value the
    welcome screen got from ``load_recents`` and passes back). Removing
    an unknown path is a no-op; storage failures are silently ignored
    like everywhere else (best-effort file).
    """
    recents = [r for r in load_recents() if r.get("project") != project]
    _save(recents)
    return recents


def _save_config(data: dict[str, object]) -> None:
    """Atomic best-effort write of the whole config (recents included)."""
    path = config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except (OSError, ValueError, TypeError):
        pass  # storage is best-effort; never block launching a project


def _save(recents: list[dict[str, object]]) -> None:
    data = load_config()
    data["recents"] = recents
    _save_config(data)


def project_title(project: str) -> str:
    name = Path(project).name or project
    return name


def load_font_settings() -> dict[str, object]:
    """Terminal font settings; missing/invalid values → null (JS defaults).

    Returns ``{"font_family": str | None, "font_size": int | None}``.
    """
    data = load_config()
    out: dict[str, object] = {"font_family": None, "font_size": None}
    family = data.get("font_family")
    if isinstance(family, str) and family.strip():
        out["font_family"] = family.strip()
    size = data.get("font_size")
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        try:
            size = int(size)
        except (ValueError, OverflowError):
            size = None
        if size is not None and MIN_FONT_SIZE <= size <= MAX_FONT_SIZE:
            out["font_size"] = size
    return out


def save_font_settings(font_family: str | None, font_size: object) -> dict[str, object]:
    """Validate + persist the font settings; returns the stored values.

    The size is clamped to MIN_FONT_SIZE..MAX_FONT_SIZE here so the
    python side is the source of truth even if the JS clamp is bypassed.
    Recents and any other keys in the file are preserved.
    """
    family = font_family.strip() if isinstance(font_family, str) else ""
    size_val: int | None = None
    if font_size is not None and not isinstance(font_size, bool):
        try:
            size_val = int(font_size)  # type: ignore[arg-type]  # rejects junk strings
        except (TypeError, ValueError, OverflowError):
            size_val = None
    if size_val is not None:
        size_val = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, size_val))
    data = load_config()
    data["font_family"] = family or None
    data["font_size"] = size_val
    _save_config(data)
    return {"font_family": family or None, "font_size": size_val}
