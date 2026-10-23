"""Tests for drcode.task_env: the optional per-task git-worktree isolation.

Lifecycle contract under test: worktree created on activation, preserved
across unfinished/re-queue, removed on cancel/delete, orphans swept at
startup — and the "worktree without a git repo" refusal (the setting is
still saved, the mode stays dormant). Plus the two no-loss rules:

1. a verdict never leaves the task with a dirty tree (`snapshot`);
2. a branch with unmerged commits is never deleted automatically.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
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


def unmerged_count(root: Path, branch: str) -> int:
    """Commits of `branch` not reachable from the main checkout's HEAD
    (the tests' base branch — the repo was committed on it)."""
    out = subprocess.run(
        ["git", "rev-list", "--count", f"HEAD..{branch}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return int(out)


def last_commit_subject(root: Path, branch: str) -> str:
    return subprocess.run(
        ["git", "log", "-1", "--format=%s", branch],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


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


# -- isolation off: no worktree, even in a git repo ----------------------------------


def test_isolation_off_never_creates_or_reuses_worktree(git_root: Path) -> None:
    """Default (off) in a git repo: every activation returns the project
    root — the worktree path is neither created nor silently reused as the
    task's root when a leftover worktree exists."""
    env = TaskEnvironment(git_root, ProjectSettings())
    assert not env.enabled
    assert env.available  # a repo, but the setting says off
    root, note = env.ensure_root("t1")
    assert root == git_root
    assert note == ""
    assert not (git_root / ".drcode" / "worktrees" / "t1").exists()
    assert not branch_exists(git_root, "drcode/t1")
    # A worktree left over from a session where the setting was on stays
    # out of the way: the task keeps working in the project directory.
    leftover = git_root / ".drcode" / "worktrees" / "t1"
    leftover.mkdir(parents=True)
    (leftover / "README.md").write_text("stale")
    root2, _ = env.ensure_root("t1")
    assert root2 == git_root
    assert (git_root / "README.md").read_text() == "hello"


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


# -- RULE 1: snapshot -----------------------------------------------------------------


def test_snapshot_commits_dirty_worktree(env: TaskEnvironment, git_root: Path) -> None:
    root, _ = env.ensure_root("t1")
    (root / "work.txt").write_text("precious uncommitted work")
    result = env.snapshot("t1", "tester verdict: Verification failed")
    assert result is not None
    assert result.startswith("snapshot committed on drcode/t1 @ ")
    sha = result.rsplit("@ ", 1)[1]
    # The tree is clean now (private .drcode state excluded).
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", ":(exclude).drcode"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert status.strip() == ""
    # The commit landed on the task's branch, with the documented message.
    assert unmerged_count(git_root, "drcode/t1") >= 1
    assert last_commit_subject(git_root, "drcode/t1") == (
        "drcode: snapshot of t1 (tester verdict: Verification failed)"
    )
    assert len(sha) > 0
    # The work is reachable: the file content is on the branch.
    content = subprocess.run(
        ["git", "show", "drcode/t1:work.txt"],
        cwd=git_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert content == "precious uncommitted work"


def test_snapshot_clean_tree_is_reported_not_committed(
    env: TaskEnvironment, git_root: Path
) -> None:
    env.ensure_root("t1")
    before = unmerged_count(git_root, "drcode/t1")
    assert env.snapshot("t1", "verdict") == "worktree clean, nothing to snapshot"
    assert unmerged_count(git_root, "drcode/t1") == before


def test_snapshot_noop_without_isolation(
    git_root: Path, isolated_settings: ProjectSettings
) -> None:
    off = TaskEnvironment(git_root, ProjectSettings())  # task_isolation: off
    (git_root / "README.md").write_text("dirty in the main tree")
    assert off.snapshot("t1", "verdict") is None


def test_snapshot_noop_without_git(
    project_root: Path, isolated_settings: ProjectSettings
) -> None:
    env = TaskEnvironment(project_root, isolated_settings)  # not a git repo
    assert env.snapshot("t1", "verdict") is None


def test_snapshot_noop_without_worktree(env: TaskEnvironment) -> None:
    assert env.snapshot("never-activated", "verdict") is None


def test_snapshot_excludes_private_drcode_state(
    env: TaskEnvironment, git_root: Path
) -> None:
    root, _ = env.ensure_root("t1")
    (root / "code.py").write_text("real work")
    (root / ".drcode").mkdir()
    (root / ".drcode" / "scratch.txt").write_text("private state")
    result = env.snapshot("t1", "before removal")
    assert result is not None and result.startswith("snapshot committed")
    names = subprocess.run(
        ["git", "show", "--name-only", "--format=", "drcode/t1"],
        cwd=git_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert "code.py" in names
    # The private state was NOT committed — it stays untracked in the tree.
    assert not any(name.startswith(".drcode") for name in names)
    assert (root / ".drcode" / "scratch.txt").is_file()


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


def test_remove_keeps_branch_with_unmerged_commits(
    env: TaskEnvironment, git_root: Path
) -> None:
    """RULE 2: a branch holding the task's work is never deleted — only
    the (disposable) worktree directory goes."""
    notices: list[tuple[str, str]] = []
    env.event_log = lambda task_id, text: notices.append((task_id, text))
    env.ensure_root("t1")
    root = env.worktree_path("t1")
    (root / "work.txt").write_text("the actual work")
    assert env.snapshot("t1", "before removal") is not None
    env.remove("t1")
    assert not env.worktree_path("t1").exists()
    assert branch_exists(git_root, "drcode/t1")
    assert unmerged_count(git_root, "drcode/t1") >= 1
    assert ("t1", "branch drcode/t1 kept (1 unmerged commits)") in notices


def test_remove_snapshots_dirty_tree_before_removal(
    env: TaskEnvironment, git_root: Path
) -> None:
    """Belt-and-suspenders: even a removal that never passed through a
    verdict transition (delete / retention purge) first anchors a dirty
    tree on the branch."""
    notices: list[tuple[str, str]] = []
    env.event_log = lambda task_id, text: notices.append((task_id, text))
    env.ensure_root("t1")
    (env.worktree_path("t1") / "late.txt").write_text("never got a verdict")
    env.remove("t1")
    assert not env.worktree_path("t1").exists()
    assert branch_exists(git_root, "drcode/t1")
    assert last_commit_subject(git_root, "drcode/t1") == (
        "drcode: snapshot of t1 (before removal)"
    )
    assert any("snapshot committed on drcode/t1" in t for _, t in notices)


def test_remove_snapshots_are_not_logged_when_not_taken(
    env: TaskEnvironment, git_root: Path
) -> None:
    """A clean tree produces no snapshot noise in the event log."""
    notices: list[tuple[str, str]] = []
    env.event_log = lambda task_id, text: notices.append((task_id, text))
    env.ensure_root("t1")
    env.remove("t1")
    assert not any("snapshot committed" in t for _, t in notices)
    assert not branch_exists(git_root, "drcode/t1")  # deleted as today


def test_cancel_preserves_dirty_work_on_branch(
    env: TaskEnvironment, manager: TaskManager, git_root: Path
) -> None:
    """Cancel of a dirty task: the work is snapshotted and kept on the
    branch, only the worktree directory is torn down."""
    notices: list[tuple[str, str]] = []
    env.event_log = lambda task_id, text: notices.append((task_id, text))
    manager.add_removal_listener(env.remove)
    task = manager.create("t", "x")
    root, _ = env.ensure_root(task.id)
    (root / "work.txt").write_text("cancelled mid-flight")
    manager.cancel(task.id)
    assert not env.worktree_path(task.id).exists()
    assert branch_exists(git_root, f"drcode/{task.id}")
    assert unmerged_count(git_root, f"drcode/{task.id}") >= 1
    assert any("snapshot committed" in t for _, t in notices)
    assert any("kept" in t for _, t in notices)


def test_retention_purge_keeps_unmerged_branch(
    env: TaskEnvironment, manager: TaskManager, git_root: Path
) -> None:
    """The startup retention cleanup goes through delete → removal
    listener: worktree removed, branch with unmerged commits kept."""
    notices: list[tuple[str, str]] = []
    env.event_log = lambda task_id, text: notices.append((task_id, text))
    manager.add_removal_listener(env.remove)
    task = manager.create("old", "x")
    root, _ = env.ensure_root(task.id)
    (root / "feature.txt").write_text("accepted-looking work")
    assert env.snapshot(task.id, "tester verdict: Verified successfully")
    task.status = TaskStatus.COMPLETED
    task.completed_at = datetime.now(UTC) - timedelta(days=10)
    manager.storage.save(task)
    purged = manager.purge_expired_completed(retention_days=7)
    assert [t.id for t in purged] == [task.id]
    assert not env.worktree_path(task.id).exists()
    assert branch_exists(git_root, f"drcode/{task.id}")
    assert any("kept" in t for _, t in notices)


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


def test_cleanup_orphans_snapshots_and_keeps_unmerged_branch(
    env: TaskEnvironment, manager: TaskManager, git_root: Path
) -> None:
    """RULE 1 + RULE 2 at startup: a dead task's dirty tree is snapshotted
    onto its branch, and a branch with unmerged commits survives the
    sweep (only the worktree directory goes)."""
    notices: list[tuple[str, str]] = []
    env.event_log = lambda task_id, text: notices.append((task_id, text))
    dead = manager.create("dead", "x")
    env.ensure_root(dead.id)
    (env.worktree_path(dead.id) / "crashed.txt").write_text("lost session work")
    assert env.snapshot(dead.id, "before orphan cleanup") is not None
    # Now make MORE dirt (the sweep itself must snapshot it) and drop the task.
    (env.worktree_path(dead.id) / "more.txt").write_text("even more work")
    manager.tasks.pop(dead.id)
    manager.storage.delete(dead.id)
    removed = env.cleanup_orphans(set(manager.tasks))
    assert dead.id in removed  # the worktree directory went away
    assert not env.worktree_path(dead.id).exists()
    assert branch_exists(git_root, f"drcode/{dead.id}")
    assert unmerged_count(git_root, f"drcode/{dead.id}") >= 2
    assert any("snapshot committed" in t for _, t in notices)
    assert any("kept" in t for _, t in notices)


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
