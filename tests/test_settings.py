"""Tests for drcode.settings: retention enum and the project settings store."""

from __future__ import annotations

import json
from pathlib import Path

from drcode.settings import (
    RETENTION_OPTIONS,
    CompletedRetention,
    ProjectSettings,
    ProjectSettingsStore,
)


# -- CompletedRetention -------------------------------------------------------


def test_retention_days() -> None:
    assert CompletedRetention.DAYS_7.days == 7
    assert CompletedRetention.DAYS_30.days == 30
    assert CompletedRetention.DAYS_90.days == 90
    assert CompletedRetention.NEVER.days is None


def test_retention_labels() -> None:
    assert CompletedRetention.DAYS_7.label == "after 7 days"
    assert CompletedRetention.DAYS_30.label == "after 30 days"
    assert CompletedRetention.DAYS_90.label == "after 90 days"
    assert CompletedRetention.NEVER.label == "never"


def test_retention_values_stable() -> None:
    # Persisted in settings.json — must not change silently.
    assert CompletedRetention.DAYS_7.value == "7"
    assert CompletedRetention.NEVER.value == "never"


def test_retention_options_order() -> None:
    assert RETENTION_OPTIONS == (
        CompletedRetention.DAYS_7,
        CompletedRetention.DAYS_30,
        CompletedRetention.DAYS_90,
        CompletedRetention.NEVER,
    )


# -- ProjectSettings ------------------------------------------------------------


def test_default_settings() -> None:
    settings = ProjectSettings()
    assert settings.completed_retention is CompletedRetention.DAYS_7


def test_store_roundtrip(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    settings = ProjectSettings(completed_retention=CompletedRetention.NEVER)
    store.save(settings)

    assert (project_root / ".drcode" / "settings.json").is_file()
    assert store.load() == settings


def test_store_creates_parent_dir(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root / "sub" / "dir")
    store.save(ProjectSettings())
    assert (project_root / "sub" / "dir" / "settings.json").is_file()


def test_store_missing_file_returns_defaults(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    assert store.load() == ProjectSettings()


def test_store_corrupted_json_returns_defaults(project_root: Path) -> None:
    path = project_root / ".drcode" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken")
    store = ProjectSettingsStore(project_root)
    assert store.load() == ProjectSettings()


def test_store_invalid_enum_returns_defaults(project_root: Path) -> None:
    path = project_root / ".drcode" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"completed_retention": "bogus"}))
    store = ProjectSettingsStore(project_root)
    assert store.load() == ProjectSettings()


def test_store_overwrite(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    store.save(ProjectSettings(completed_retention=CompletedRetention.DAYS_7))
    store.save(ProjectSettings(completed_retention=CompletedRetention.DAYS_90))
    assert store.load().completed_retention is CompletedRetention.DAYS_90


def test_store_save_leaves_no_tmp_files(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    store.save(ProjectSettings())
    assert list(store.path.parent.glob("*.tmp")) == []
