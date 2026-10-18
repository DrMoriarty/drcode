"""Tests for drcode.version: the single source of truth for the app version."""

from __future__ import annotations

from pathlib import Path

import drcode.version as version_module
from drcode.version import APP_VERSION, app_version


def test_app_version_nonempty() -> None:
    assert app_version()
    assert APP_VERSION == app_version()


def test_app_version_is_cached() -> None:
    assert app_version.cache_info().currsize == 1


def test_app_version_matches_version_file() -> None:
    # In a source checkout the version comes from the repo-root `version` file.
    repo_root = Path(version_module.__file__).resolve().parents[2]
    version_file = repo_root / "version"
    if version_file.is_file() and (repo_root / "pyproject.toml").is_file():
        assert app_version() == version_file.read_text().strip()


def test_app_version_looks_like_a_version() -> None:
    parts = app_version().split(".")
    assert len(parts) >= 2
    assert parts[0].isdigit()


def test_frozen_version_uses_meipass(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "version").write_text("9.9.9\n")

    class FakeSys:
        frozen = True
        _MEIPASS = str(tmp_path)
        executable = "/fake/drcode"

    monkeypatch.setattr(version_module, "sys", FakeSys)
    assert version_module._frozen_version() == "9.9.9"


def test_frozen_version_none_when_not_frozen() -> None:
    # Real sys module: frozen is not set.
    assert version_module._frozen_version() is None
