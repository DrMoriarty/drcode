"""Shared fixtures: isolated project roots and a sandboxed user-config dir.

Every test works in a temporary directory; the user-level drcode config dir
(`~/.config/drcode` by default) is redirected to a tmp path so tests never
touch the developer's real configuration, and the repo's own `.env` /
`DRCODE_*` environment are neutralized so layering tests are deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import drcode.config as config_module
from drcode.config import Config


@pytest.fixture(autouse=True)
def _sandbox_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the user config dir (computed at import time) to tmp."""
    config_dir = tmp_path / "xdg" / "drcode"
    monkeypatch.setattr(config_module, "_USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_module, "USER_CONFIG_FILE", config_dir / "drcode.json")
    monkeypatch.setattr(
        config_module, "USER_CONFIG_EXAMPLE_FILE", config_dir / "drcode.json.example"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    # Disable the dotenv source: the repository's own .env (and any .env in
    # cwd) must not leak values into configuration tests.
    monkeypatch.setitem(Config.model_config, "env_file", ())
    # Keep the developer's real environment from leaking into assertions.
    for name in (
        "DRCODE_MODEL",
        "DRCODE_BASE_URL",
        "DRCODE_API_KEY",
        "DRCODE_MAX_CONTEXT_TOKENS",
        "DRCODE_COMPACTION_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def user_config_dir(tmp_path: Path) -> Path:
    """The redirected user config dir (created on demand)."""
    return config_module._USER_CONFIG_DIR


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """A fresh temporary project directory (the agent's writable root)."""
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def manager(project_root: Path):
    from drcode.task_manager import TaskManager

    return TaskManager(project_root)
