"""Tests for drcode.task_env: the optional per-task git-worktree isolation.

Lifecycle contract under test: worktree created on activation, preserved
across unfinished/re-queue, removed on cancel/delete, orphans swept at
startup — and the "worktree without a git repo" refusal (the setting is
still saved, the mode stays dormant).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from drcode.models import TaskStatus
from drcode.settings import (
    ISOLATION_OPTIONS,
    ProjectSettings,
    ProjectSettingsStore,
    TaskIsolation,
)
from drcode.task_env import TaskEnvironment
from drcode.task_manager import TaskManager

# -- fixtures --------------------------------------------------------------------


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


@pytest.fixture
def git_root(project_root: Path) -> Path:
    """A project root that is a real git repository with one commit."""
    _git(project_root, "init", "-q")
    _git(project_root, "config", "user.email", "test@example.com")
    _git(project_root, "config", "user.name", "Test")
    # drcode's private state is gitignored (as ensure_drcode_gitignored
    # would arrange); the README gives the repo its initial commit.
    (project_root / ".gitignore").write_text(
        ".drcode/tasks/\n.drcode/worktrees/\n.env\n"
    )
    (project_root / "README.md").write_text("hello")
    _git(project_root, "add", "-A")
    _git(project_root, "commit", "-q", "-m", "init")
    return project_root


@pytest.fixture
def isolated_settings() -> ProjectSettings:
    return ProjectSettings(task_isolation=TaskIsolation.WORKTREE)


@pytest.fixture
def env(git_root: Path, isolated_settings: ProjectSettings) -> TaskEnvironment:
    return TaskEnvironment(git_root, isolated_settings)


def branch_exists(root: Path, branch: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", branch],
            cwd=root,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


# -- setting persistence ----------------------------------------------------------


def test_isolation_off_by_default() -> None:
    assert ProjectSettings().task_isolation is TaskIsolation.OFF


def test_isolation_roundtrip(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    store.save(ProjectSettings(task_isolation=TaskIsolation.WORKTREE))
    assert store.load().task_isolation is TaskIsolation.WORKTREE


def test_isolation_values_stable() -> None:
    # Persisted in settings.json — must not change silently.
    assert TaskIsolation.OFF.value == "off"
    assert TaskIsolation.WORKTREE.value == "worktree"
    assert ISOLATION_OPTIONS == (TaskIsolation.OFF, TaskIsolation.WORKTREE)


# -- enabling without git: saved, but dormant -------------------------------------


def test_enabled_without_git_stays_dormant(
    project_root: Path, isolated_settings: ProjectSettings
) -> None:
    env = TaskEnvironment(project_root, isolated_settings)
    # The user's choice is kept (the app warns at startup instead of
    # silently rewriting it), but nothing is isolated without a repo.
    assert env.enabled
    assert not env.available
    assert not env.active
    root, note = env.ensure_root("t1")
    assert root == project_root
    assert note
    assert not (project_root / ".drcode" / "worktrees" / "t1").exists()


# -- isolation off: the project root, nothing created -----------------------------


def test_isolation_off_returns_project_root(git_root: Path) -> None:
    env = TaskEnvironment(git_root, ProjectSettings())  # task_isolation: off
    assert not env.enabled
    assert not env.active
    root, note = env.ensure_root("t1")
    assert root == git_root
    assert note == ""
    assert not (git_root / ".drcode" / "worktrees" / "t1").exists()
    assert not branch_exists(git_root, "drcode/t1")


# -- creation and reuse -------------------------------------------------------------


def test_worktree_created_on_activation(env: TaskEnvironment, git_root: Path) -> None:
    root, note = env.ensure_root("t1")
    assert note == ""
    wt = git_root / ".drcode" / "worktrees" / "t1"
    assert root == wt
    assert wt.is_dir()
    assert (wt / "README.md").is_file()  # checked out from HEAD
    assert branch_exists(git_root, "drcode/t1")
    # The main tree stays clean: worktrees are ignored, nothing else moved.
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=git_root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert status.strip() == ""


def test_worktree_reused_not_recreated(env: TaskEnvironment, git_root: Path) -> None:
    root, _ = env.ensure_root("t1")
    (root / "README.md").write_text("changed in the worktree")
    root2, _ = env.ensure_root("t1")
    assert root2 == root
    assert (root2 / "README.md").read_text() == "changed in the worktree"


def test_unfinished_task_keeps_its_worktree(
    env: TaskEnvironment, manager: TaskManager
) -> None:
    """The point of the feature: a re-queued (unfinished) task keeps its
    snapshot; the worktree survives until the task is finished for good."""
    task = manager.create("t", "do it")
    root, _ = env.ensure_root(task.id)
    (root / "uncommitted.txt").write_text("work in progress")
    # Simulate the pipeline's interrupt path: unfinished + back to queue.
    task.unfinished = True
    task.status = TaskStatus.QUEUED
    manager.storage.save(task)
    # Next activation — same task: its own dirty tree comes back.
    root2, _ = env.ensure_root(task.id)
    assert root2 == root
    assert (root2 / "uncommitted.txt").is_file()


def test_next_task_gets_clean_tree(env: TaskEnvironment, manager: TaskManager) -> None:
    dirty = manager.create("dirty", "x")
    root, _ = env.ensure_root(dirty.id)
    (root / "README.md").write_text("half-done work")
    fresh = manager.create("fresh", "y")
    root2, _ = env.ensure_root(fresh.id)
    assert root2 != root
    assert (root2 / "README.md").read_text() == "hello"


# -- untracked/ignored files ---------------------------------------------------------


def test_ignored_files_copied_into_worktree(
    env: TaskEnvironment, git_root: Path
) -> None:
    (git_root / ".env").write_text("SECRET=1")
    root, _ = env.ensure_root("t1")
    assert (root / ".env").read_text() == "SECRET=1"


def test_copy_skips_dependency_dirs(env: TaskEnvironment, git_root: Path) -> None:
    heavy = git_root / "node_modules" / "pkg"
    heavy.mkdir(parents=True)
    (heavy / "lib.js").write_text("x")
    root, _ = env.ensure_root("t1")
    assert not (root / "node_modules").exists()


# -- workflow hook payload -----------------------------------------------------------


def test_worktree_info_present_when_isolated(env: TaskEnvironment) -> None:
    env.ensure_root("t1")
    info = env.worktree_info("t1")
    assert info is not None
    assert info["branch"] == "drcode/t1"
    assert info["path"] == str(env.worktree_path("t1"))


def test_worktree_info_none_when_off(
    git_root: Path, isolated_settings: ProjectSettings
) -> None:
    off = TaskEnvironment(git_root, ProjectSettings())
    assert off.worktree_info("t1") is None


# -- removal (cancel / delete) --------------------------------------------------------


def test_remove_deletes_worktree_and_branch(
    env: TaskEnvironment, git_root: Path
) -> None:
    env.ensure_root("t1")
    assert env.worktree_path("t1").is_dir()
    env.remove("t1")
    assert not env.worktree_path("t1").exists()
    assert not branch_exists(git_root, "drcode/t1")


def test_cancel_tears_worktree_down(
    env: TaskEnvironment, manager: TaskManager, git_root: Path
) -> None:
    manager.add_removal_listener(env.remove)
    task = manager.create("t", "x")
    env.ensure_root(task.id)
    manager.cancel(task.id)
    assert not env.worktree_path(task.id).exists()
    assert not branch_exists(git_root, f"drcode/{task.id}")


def test_delete_tears_worktree_down(
    env: TaskEnvironment, manager: TaskManager, git_root: Path
) -> None:
    manager.add_removal_listener(env.remove)
    task = manager.create("t", "x")
    env.ensure_root(task.id)
    manager.set_hold(task.id, on=True)
    manager.delete(task.id)
    assert not env.worktree_path(task.id).exists()
    assert not branch_exists(git_root, f"drcode/{task.id}")


def test_completion_keeps_worktree(
    env: TaskEnvironment, manager: TaskManager, git_root: Path
) -> None:
    """An ordinary completion does NOT remove the worktree: the verified
    work stays on the branch drcode/<id> (merging is the team's
    on_accepted decision, see drcode.workflow)."""
    manager.add_removal_listener(env.remove)
    task = manager.create("t", "x")
    env.ensure_root(task.id)
    # A completion transition (tester PASSED / research done) never fires
    # the removal listeners — only cancel/delete do.
    task.status = TaskStatus.COMPLETED
    task.completed_at = task.updated_at
    manager.storage.save(task)
    assert env.worktree_path(task.id).is_dir()
    assert branch_exists(git_root, f"drcode/{task.id}")


# -- startup orphan sweep --------------------------------------------------------------


def test_cleanup_orphans_removes_dead_tasks_resources(
    env: TaskEnvironment, manager: TaskManager, git_root: Path
) -> None:
    live = manager.create("live", "x")
    dead = manager.create("dead", "x")
    env.ensure_root(live.id)
    env.ensure_root(dead.id)
    # The task vanished (retention cleanup ran in a previous session and
    # the worktree sweep never did).
    manager.tasks.pop(dead.id)
    manager.storage.delete(dead.id)
    removed = env.cleanup_orphans(set(manager.tasks))
    assert dead.id in removed
    assert live.id not in removed
    assert not env.worktree_path(dead.id).exists()
    assert not branch_exists(git_root, f"drcode/{dead.id}")
    assert env.worktree_path(live.id).is_dir()
    assert branch_exists(git_root, f"drcode/{live.id}")


def test_cleanup_orphans_skipped_when_never_used(
    project_root: Path, isolated_settings: ProjectSettings
) -> None:
    # Not a git repo at all: the sweep must not spawn git or touch anything.
    env = TaskEnvironment(project_root, isolated_settings)
    assert env.cleanup_orphans({"t1"}) == []


# -- settings roundtrip through the dormant-warning path ---------------------------------


def test_dormant_setting_survives_restart_without_git(
    project_root: Path,
) -> None:
    """Chosen behaviour: worktree without git is SAVED and re-warned at
    every startup — the value must survive a store roundtrip untouched."""
    store = ProjectSettingsStore(project_root)
    store.save(ProjectSettings(task_isolation=TaskIsolation.WORKTREE))
    loaded = store.load()
    assert loaded.task_isolation is TaskIsolation.WORKTREE
    env = TaskEnvironment(project_root, loaded)
    assert not env.active  # the startup warning path fires, mode dormant
