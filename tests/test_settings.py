"""Tests for drcode.settings: retention enum and the project settings store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drcode.settings import (
    RETENTION_OPTIONS,
    STALL_NUDGE_OPTIONS,
    WORKTREE_REMOVAL_OPTIONS,
    WORKTREE_SNAPSHOT_OPTIONS,
    CompletedRetention,
    GitPolicy,
    GitPolicySetting,
    ProjectSettings,
    ProjectSettingsStore,
    StallNudge,
    Verification,
    WorktreeRemoval,
    WorktreeSnapshot,
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


# -- WorktreeSnapshot -------------------------------------------------------------


def test_worktree_snapshot_default() -> None:
    # The smart default: the snapshot is insurance only when the run made
    # no commits of its own.
    assert ProjectSettings().worktree_snapshot is WorktreeSnapshot.ONLY_IF_NO_COMMITS


def test_worktree_snapshot_labels() -> None:
    assert WorktreeSnapshot.ALWAYS.label == "always"
    assert WorktreeSnapshot.ONLY_IF_NO_COMMITS.label == "only if no commits"
    assert WorktreeSnapshot.NEVER.label == "never"


def test_worktree_snapshot_values_stable() -> None:
    # Persisted in settings.json — must not change silently.
    assert WorktreeSnapshot.ALWAYS.value == "always"
    assert WorktreeSnapshot.ONLY_IF_NO_COMMITS.value == "only-if-no-commits"
    assert WorktreeSnapshot.NEVER.value == "never"


def test_worktree_snapshot_options_order() -> None:
    assert WORKTREE_SNAPSHOT_OPTIONS == (
        WorktreeSnapshot.ALWAYS,
        WorktreeSnapshot.ONLY_IF_NO_COMMITS,
        WorktreeSnapshot.NEVER,
    )


def test_store_worktree_snapshot_roundtrip(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    store.save(ProjectSettings(worktree_snapshot=WorktreeSnapshot.ALWAYS))
    assert store.load().worktree_snapshot is WorktreeSnapshot.ALWAYS
    store.save(ProjectSettings(worktree_snapshot=WorktreeSnapshot.NEVER))
    assert store.load().worktree_snapshot is WorktreeSnapshot.NEVER


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


# -- Verification / GitPolicy ---------------------------------------------------


def test_default_verification_and_git_policy() -> None:
    settings = ProjectSettings()
    # The historical behavior: tester on, both roles commit + push.
    assert settings.verification is Verification.TESTER
    assert settings.git_policy.tester is GitPolicy.PUSH
    assert settings.git_policy.research is GitPolicy.PUSH


def test_verification_values_and_labels() -> None:
    assert Verification.TESTER.value == "tester"
    assert Verification.NONE.value == "none"
    assert Verification.TESTER.label == "tester"
    assert Verification.NONE.label == "none"


def test_git_policy_values_and_labels() -> None:
    # Persisted in settings.json — must not change silently.
    assert GitPolicy.OFF.value == "off"
    assert GitPolicy.LOCAL.value == "local"
    assert GitPolicy.PUSH.value == "push"
    assert GitPolicy.OFF.label == "off"
    assert GitPolicy.LOCAL.label == "commit locally"
    assert GitPolicy.PUSH.label == "commit + push"


def test_git_policy_commits_and_pushes() -> None:
    assert GitPolicy.OFF.commits is False
    assert GitPolicy.LOCAL.commits is True
    assert GitPolicy.PUSH.commits is True
    assert not GitPolicy.OFF.pushes and not GitPolicy.LOCAL.pushes
    assert GitPolicy.PUSH.pushes is True


def test_store_git_policy_roundtrip(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    settings = ProjectSettings(
        verification=Verification.NONE,
        git_policy=GitPolicySetting(tester=GitPolicy.LOCAL, research=GitPolicy.OFF),
    )
    store.save(settings)
    assert store.load() == settings
    # The nested shape lands in the file as an object.
    data = json.loads(store.path.read_text())
    assert data["git_policy"] == {"tester": "local", "research": "off"}
    assert data["verification"] == "none"
    assert "task_finalization" not in data


# -- Legacy task_finalization migration ------------------------------------------


def _write_raw_settings(project_root: Path, data: dict) -> ProjectSettingsStore:
    store = ProjectSettingsStore(project_root)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(json.dumps(data))
    return store


@pytest.mark.parametrize(
    ("legacy", "verification", "tester_policy"),
    [
        ("testing_commit", Verification.TESTER, GitPolicy.PUSH),
        ("testing_commit_no_push", Verification.TESTER, GitPolicy.LOCAL),
        ("testing", Verification.TESTER, GitPolicy.OFF),
        ("none", Verification.NONE, GitPolicy.PUSH),
    ],
)
def test_legacy_finalization_migration(
    project_root: Path,
    legacy: str,
    verification: Verification,
    tester_policy: GitPolicy,
) -> None:
    """Each legacy `task_finalization` value maps onto the orthogonal
    fields: the tester policy follows the legacy commit/push combination
    (NONE completes without verification) and the research policy keeps
    its default (push — the historical behavior)."""
    store = _write_raw_settings(project_root, {"task_finalization": legacy})
    settings = store.load()
    assert settings.verification is verification
    assert settings.git_policy.tester is tester_policy
    assert settings.git_policy.research is GitPolicy.PUSH
    # Saving writes only the new shape: the legacy key is gone.
    store.save(settings)
    data = json.loads(store.path.read_text())
    assert "task_finalization" not in data


def test_legacy_finalization_migration_unknown_value(
    project_root: Path,
) -> None:
    """An unrecognized legacy value must not corrupt the settings: the
    new fields keep their defaults."""
    store = _write_raw_settings(project_root, {"task_finalization": "bogus"})
    settings = store.load()
    assert settings == ProjectSettings()


def test_legacy_finalization_migration_new_keys_win(project_root: Path) -> None:
    """When the file already carries the new fields, the legacy key is
    dropped and the new values survive untouched."""
    store = _write_raw_settings(
        project_root,
        {
            "task_finalization": "testing",
            "verification": "tester",
            "git_policy": {"tester": "push", "research": "local"},
        },
    )
    settings = store.load()
    assert settings.verification is Verification.TESTER
    assert settings.git_policy.tester is GitPolicy.PUSH
    assert settings.git_policy.research is GitPolicy.LOCAL


def test_git_policy_stale_value_returns_defaults(project_root: Path) -> None:
    """A stale/unknown policy value must not break the app either."""
    store = _write_raw_settings(project_root, {"git_policy": {"tester": "rebase-all"}})
    assert store.load() == ProjectSettings()


# -- task_settings_hints -------------------------------------------------------


def test_hints_parallel_dormant_without_isolation() -> None:
    """parallel_tasks > 1 without worktree isolation is dormant, and its
    fix (a callable the user opts into) enables the isolation."""
    from drcode.settings import TaskIsolation, task_settings_hints

    draft = ProjectSettings(parallel_tasks=3)
    hints = task_settings_hints(draft, git_available=True)
    dormant = [
        h for h in hints if h.severity == "dormant" and "Parallel tasks" in h.text
    ]
    assert len(dormant) == 1
    hint = dormant[0]
    assert hint.fix is not None
    assert hint.fix_label
    hint.fix(draft)
    assert draft.task_isolation is TaskIsolation.WORKTREE


def test_hints_no_parallel_hint_under_isolation() -> None:
    from drcode.settings import TaskIsolation, task_settings_hints

    draft = ProjectSettings(parallel_tasks=3, task_isolation=TaskIsolation.WORKTREE)
    assert not [
        h
        for h in task_settings_hints(draft, git_available=True)
        if "Parallel tasks" in h.text
    ]


def test_hints_tester_git_policy_dormant_under_no_verification() -> None:
    """A non-off tester git policy with verification none is dormant; the
    fix sets the tester policy to off."""
    from drcode.settings import GitPolicy, task_settings_hints

    draft = ProjectSettings(
        verification=Verification.NONE,
        git_policy=GitPolicySetting(tester=GitPolicy.LOCAL),
    )
    hints = task_settings_hints(draft, git_available=True)
    dormant = [h for h in hints if h.severity == "dormant" and "Git (tester)" in h.text]
    assert len(dormant) == 1
    assert dormant[0].fix is not None
    dormant[0].fix(draft)
    assert draft.git_policy.tester is GitPolicy.OFF


def test_hints_tester_git_policy_fine_with_verification() -> None:
    from drcode.settings import task_settings_hints

    draft = ProjectSettings(verification=Verification.TESTER)
    assert task_settings_hints(draft, git_available=True) == []


def test_hints_parked_fields_under_isolation_off() -> None:
    """A configured start branch / non-default removal are inert without
    isolation (dormant); the defaults stay silent."""
    from drcode.settings import WorktreeRemoval, task_settings_hints

    base = ProjectSettings()
    assert task_settings_hints(base, git_available=True) == []

    parked = ProjectSettings(
        isolation_start_branch="main",
        worktree_removal=WorktreeRemoval.NEVER,
        worktree_snapshot=WorktreeSnapshot.NEVER,
    )
    texts = [h.text for h in task_settings_hints(parked, git_available=True)]
    assert any("main" in t for t in texts)
    assert any("Worktree removal" in t for t in texts)
    assert any("Worktree snapshot" in t for t in texts)


def test_hints_worktree_without_git_is_conflict() -> None:
    """Worktree isolation without a git repository is a conflict, not a
    dormant value (it cannot work here at all)."""
    from drcode.settings import TaskIsolation, task_settings_hints

    draft = ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    hints = task_settings_hints(draft, git_available=False)
    assert [h.severity for h in hints] == ["conflict"]
    # With git available the same draft is clean.
    assert task_settings_hints(draft, git_available=True) == []
