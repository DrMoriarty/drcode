"""Tests for drcode.settings: retention enum and the project settings store."""

from __future__ import annotations

import json
from pathlib import Path

from drcode.settings import (
    RETENTION_OPTIONS,
    STALL_NUDGE_OPTIONS,
    WORKTREE_REMOVAL_OPTIONS,
    CompletedRetention,
    ProjectSettings,
    ProjectSettingsStore,
    StallNudge,
    WorktreeRemoval,
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


# -- StallNudge ---------------------------------------------------------------


def test_stall_nudge_minutes() -> None:
    assert StallNudge.MIN_15.minutes == 15
    assert StallNudge.MIN_30.minutes == 30
    assert StallNudge.HOUR_1.minutes == 60
    assert StallNudge.HOUR_2.minutes == 120
    assert StallNudge.NEVER.minutes is None


def test_stall_nudge_labels() -> None:
    assert StallNudge.MIN_15.label == "15 min"
    assert StallNudge.MIN_30.label == "30 min"
    assert StallNudge.HOUR_1.label == "1 hour"
    assert StallNudge.HOUR_2.label == "2 hours"
    assert StallNudge.NEVER.label == "never"


def test_stall_nudge_values_stable() -> None:
    # Persisted in settings.json — must not change silently.
    assert StallNudge.MIN_30.value == "30"
    assert StallNudge.HOUR_1.value == "60"
    assert StallNudge.NEVER.value == "never"


def test_stall_nudge_options_order() -> None:
    assert STALL_NUDGE_OPTIONS == (
        StallNudge.MIN_15,
        StallNudge.MIN_30,
        StallNudge.HOUR_1,
        StallNudge.HOUR_2,
        StallNudge.NEVER,
    )


# -- WorktreeRemoval ------------------------------------------------------------


def test_worktree_removal_default() -> None:
    # The historical behaviour: the worktree lives as long as the task.
    assert ProjectSettings().worktree_removal is WorktreeRemoval.ON_DELETE


def test_worktree_removal_labels() -> None:
    assert WorktreeRemoval.ON_DELETE.label == "on task deletion"
    assert WorktreeRemoval.ON_COMPLETION.label == "on task completion"
    assert WorktreeRemoval.NEVER.label == "never"


def test_worktree_removal_values_stable() -> None:
    # Persisted in settings.json — must not change silently.
    assert WorktreeRemoval.ON_DELETE.value == "on_delete"
    assert WorktreeRemoval.ON_COMPLETION.value == "on_completion"
    assert WorktreeRemoval.NEVER.value == "never"


def test_worktree_removal_options_order() -> None:
    assert WORKTREE_REMOVAL_OPTIONS == (
        WorktreeRemoval.ON_DELETE,
        WorktreeRemoval.ON_COMPLETION,
        WorktreeRemoval.NEVER,
    )


def test_store_worktree_removal_roundtrip(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    store.save(ProjectSettings(worktree_removal=WorktreeRemoval.ON_COMPLETION))
    assert store.load().worktree_removal is WorktreeRemoval.ON_COMPLETION
    store.save(ProjectSettings(worktree_removal=WorktreeRemoval.NEVER))
    assert store.load().worktree_removal is WorktreeRemoval.NEVER


# -- ProjectSettings ------------------------------------------------------------


def test_default_settings() -> None:
    settings = ProjectSettings()
    assert settings.completed_retention is CompletedRetention.DAYS_7
    # A silent agent gets nudged after half an hour by default.
    assert settings.stall_nudge is StallNudge.MIN_30


def test_store_roundtrip(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    settings = ProjectSettings(completed_retention=CompletedRetention.NEVER)
    store.save(settings)

    assert (project_root / ".drcode" / "settings.json").is_file()
    assert store.load() == settings


def test_store_creates_parent_dir(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root / "sub" / "dir")
    store.save(ProjectSettings())
    # Settings live under <root>/.drcode/settings.json; the store must
    # create the whole chain itself.
    assert (project_root / "sub" / "dir" / ".drcode" / "settings.json").is_file()


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


def test_store_stall_nudge_roundtrip(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    store.save(ProjectSettings(stall_nudge=StallNudge.HOUR_2))
    assert store.load().stall_nudge is StallNudge.HOUR_2
    # A stale/unknown value must not break the app either.
    (project_root / ".drcode" / "settings.json").write_text(
        json.dumps({"stall_nudge": "3"})
    )
    assert store.load().stall_nudge is StallNudge.MIN_30


def test_store_overwrite(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    store.save(ProjectSettings(completed_retention=CompletedRetention.DAYS_7))
    store.save(ProjectSettings(completed_retention=CompletedRetention.DAYS_90))
    assert store.load().completed_retention is CompletedRetention.DAYS_90


def test_store_save_leaves_no_tmp_files(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    store.save(ProjectSettings())
    assert list(store.path.parent.glob("*.tmp")) == []
