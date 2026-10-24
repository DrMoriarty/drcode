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

from drcode.models import TaskKind, TaskStatus
from drcode.settings import (
    ISOLATION_OPTIONS,
    ProjectSettings,
    ProjectSettingsStore,
    TaskIsolation,
    WorktreeRemoval,
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


# -- unmerged_ids (dependency-gate mergedness) ------------------------------------


def test_unmerged_ids_mixed(env: TaskEnvironment, git_root: Path) -> None:
    # Merged: a branch sitting exactly at the base HEAD.
    _git(git_root, "branch", "drcode/merged")
    # Unmerged: a branch with one extra commit, main checkout left behind.
    _git(git_root, "checkout", "-q", "-b", "drcode/work")
    (git_root / "work.txt").write_text("work")
    _git(git_root, "add", "-A")
    _git(git_root, "commit", "-q", "-m", "work")
    _git(
        git_root,
        "checkout",
        "-",
    )
    assert env.unmerged_ids(["merged", "work", "absent"]) == {"work"}


def test_unmerged_ids_isolation_off_or_no_repo(git_root: Path) -> None:
    # Setting off: there is nothing to merge — the gate degenerates to the
    # dependency's status, the git half always reports "satisfied".
    env = TaskEnvironment(git_root, ProjectSettings())
    _git(git_root, "checkout", "-q", "-B", "drcode/x")
    (git_root / "x.txt").write_text("x")
    _git(git_root, "add", "-A")
    _git(git_root, "commit", "-q", "-m", "x")
    _git(git_root, "checkout", "-")
    assert env.unmerged_ids(["x"]) == set()


def test_unmerged_ids_branch_outlives_task_file(
    env: TaskEnvironment, git_root: Path
) -> None:
    """RULE 2: a deleted dependency's branch survives when it holds
    unmerged commits — and still blocks its dependents (the gate checks
    git, not the task store)."""
    _git(git_root, "checkout", "-q", "-b", "drcode/gone")
    (git_root / "gone.txt").write_text("gone")
    _git(git_root, "add", "-A")
    _git(git_root, "commit", "-q", "-m", "gone")
    _git(git_root, "checkout", "-")
    # No task with id "gone" exists anywhere — the branch alone decides.
    assert env.unmerged_ids(["gone"]) == {"gone"}


def test_unmerged_ids_empty_branch_not_blocking(env: TaskEnvironment) -> None:
    # A branch that exists but adds nothing (created and never committed
    # onto) counts as merged.
    _git(env.project_root, "branch", "drcode/empty")
    assert env.unmerged_ids(["empty"]) == set()


# -- unmerged_ids batching + cache dynamics ---------------------------------------


def _commit_on_branch(root: Path, branch: str, name: str) -> None:
    """A commit on `branch` made in the main checkout; the tests' base
    branch (the one the repo was committed on) is restored afterwards."""
    _git(root, "checkout", "-q", "-B", branch)
    (root / f"{name}.txt").write_text(name)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", name)
    _git(root, "checkout", "-q", "-")


def test_unmerged_ids_batch_matches_per_branch_answers(
    env: TaskEnvironment, git_root: Path
) -> None:
    _commit_on_branch(git_root, "drcode/a", "a")
    _commit_on_branch(git_root, "drcode/b", "b")
    _git(git_root, "branch", "drcode/merged")
    ids = ["a", "b", "merged", "absent"]
    assert env.unmerged_ids(ids) == {"a", "b"}
    # The batched answer must equal the reference per-branch loop.
    assert env.unmerged_ids(ids) == env._unmerged_ids_slow(ids)


def test_unmerged_ids_repeated_calls_reuse_cache(
    env: TaskEnvironment, git_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steady state (refs unchanged): a poll is one cheap `show-ref`
    snapshot — no batch re-computation, the cached answers are reused."""
    _commit_on_branch(git_root, "drcode/work", "work")
    _git(git_root, "branch", "drcode/merged")
    ids = ["work", "merged", "absent"]
    assert env.unmerged_ids(ids) == {"work"}
    calls: list[tuple[str, ...]] = []
    original = env._git

    def spy(*args: str, **kwargs: object) -> tuple[int, str]:
        calls.append(args)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(env, "_git", spy)
    assert env.unmerged_ids(ids) == {"work"}
    assert calls == [("show-ref", "--head")]


def test_unmerged_ids_reflects_manual_merge_dynamically(
    env: TaskEnvironment, git_root: Path
) -> None:
    """A manual `git merge drcode/<id>` must unblock dependents on the
    next call — the fingerprint cache may not keep a stale 'unmerged'."""
    _commit_on_branch(git_root, "drcode/work", "work")
    assert env.unmerged_ids(["work"]) == {"work"}
    _git(git_root, "merge", "-q", "--no-ff", "drcode/work", "-m", "merge work")
    assert env.unmerged_ids(["work"]) == set()
    # ... and stays merged on the now-cached follow-up calls.
    assert env.unmerged_ids(["work"]) == set()


def test_unmerged_ids_reflects_new_commits_dynamically(
    env: TaskEnvironment, git_root: Path
) -> None:
    _git(git_root, "branch", "drcode/work")  # at the base tip: merged
    assert env.unmerged_ids(["work"]) == set()
    # New work landed on the branch (the tester committed, say) — the
    # moved branch tip must flip the answer.
    _commit_on_branch(git_root, "drcode/work", "work")
    assert env.unmerged_ids(["work"]) == {"work"}


def test_unmerged_ids_reflects_branch_removal_dynamically(
    env: TaskEnvironment, git_root: Path
) -> None:
    _commit_on_branch(git_root, "drcode/work", "work")
    assert env.unmerged_ids(["work"]) == {"work"}
    _git(git_root, "branch", "-D", "drcode/work")
    assert env.unmerged_ids(["work"]) == set()


def test_unmerged_ids_new_branch_appears_dynamically(
    env: TaskEnvironment, git_root: Path
) -> None:
    """A branch created after the first poll (fresh fingerprint) is
    picked up by the batch — no stale 'absent' from the first pass."""
    assert env.unmerged_ids(["work"]) == set()
    _commit_on_branch(git_root, "drcode/work", "work")
    assert env.unmerged_ids(["work"]) == {"work"}


# -- drcode_branch_state (existence + mergedness, one snapshot) --------------------


def test_drcode_branch_state_batches_existence_and_mergedness(
    env: TaskEnvironment, git_root: Path
) -> None:
    _commit_on_branch(git_root, "drcode/a", "a")
    _commit_on_branch(git_root, "drcode/b", "b")
    _git(git_root, "branch", "drcode/merged")
    existing, unmerged = env.drcode_branch_state()
    assert existing == {"drcode/a", "drcode/b", "drcode/merged"}
    assert unmerged == {"drcode/a", "drcode/b"}
    # The shorthand is exactly the unmerged half of the same snapshot.
    assert env.unmerged_drcode_branches() == unmerged


def test_drcode_branch_state_isolation_off(git_root: Path) -> None:
    env = TaskEnvironment(git_root, ProjectSettings())
    assert not env.active
    assert env.drcode_branch_state() == (set(), set())


def test_drcode_branch_state_reflects_merge_and_removal_dynamically(
    env: TaskEnvironment, git_root: Path
) -> None:
    """Same fingerprint cache as unmerged_ids: a manual merge and a
    later branch cleanup must both flip the answer on the next call."""
    _commit_on_branch(git_root, "drcode/work", "work")
    _git(git_root, "branch", "drcode/idle")
    existing, unmerged = env.drcode_branch_state()
    assert unmerged == {"drcode/work"}
    _git(git_root, "merge", "-q", "--no-ff", "drcode/work", "-m", "merge work")
    existing, unmerged = env.drcode_branch_state()
    assert existing == {"drcode/work", "drcode/idle"}
    assert unmerged == set()
    # The accepted-worktree-removal path deletes a merged branch — the
    # task must fall back to "no branch" (white ✓), not stay green.
    _git(git_root, "branch", "-D", "drcode/work")
    existing, unmerged = env.drcode_branch_state()
    assert "drcode/work" not in existing
    assert unmerged == set()


def test_drcode_branch_state_repeated_calls_reuse_cache(
    env: TaskEnvironment, git_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steady state (refs unchanged): a poll is one cheap `show-ref`
    snapshot — the warm cache answers both halves, no extra glob."""
    _commit_on_branch(git_root, "drcode/work", "work")
    _git(git_root, "branch", "drcode/merged")
    assert env.drcode_branch_state() == (
        {"drcode/work", "drcode/merged"},
        {"drcode/work"},
    )
    calls: list[tuple[str, ...]] = []
    original = env._git

    def spy(*args: str, **kwargs: object) -> tuple[int, str]:
        calls.append(args)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(env, "_git", spy)
    assert env.drcode_branch_state() == (
        {"drcode/work", "drcode/merged"},
        {"drcode/work"},
    )
    assert calls == [("show-ref", "--head")]

# -- worktree_removal: on_completion / never ---------------------------------------


def _removal_env(git_root: Path, removal: WorktreeRemoval) -> TaskEnvironment:
    return TaskEnvironment(
        git_root,
        ProjectSettings(
            task_isolation=TaskIsolation.WORKTREE,
            worktree_removal=removal,
        ),
    )


def test_completion_removal_research_tears_worktree_down(
    manager: TaskManager, git_root: Path
) -> None:
    """worktree_removal: on_completion — a research task is completed
    right in mark_done, and its worktree directory goes with it. The
    branch of a commit-less task is deleted (nothing unmerged)."""
    env = _removal_env(git_root, WorktreeRemoval.ON_COMPLETION)
    manager.add_completion_listener(env.on_completed)
    manager.add_removal_listener(env.remove)
    task = manager.create("research", "x", kind=TaskKind.RESEARCH)
    env.ensure_root(task.id)
    assert env.worktree_path(task.id).is_dir()
    manager.mark_done(task.id)
    assert task.status is TaskStatus.COMPLETED
    assert not env.worktree_path(task.id).exists()
    assert not branch_exists(git_root, f"drcode/{task.id}")


def test_completion_removal_dev_tears_worktree_down_on_verdict(
    manager: TaskManager, git_root: Path
) -> None:
    """worktree_removal: on_completion — a dev task's directory goes at
    the tester's verdict. RULE 1: the dirty tree is snapshotted first;
    RULE 2: the branch with the work survives."""
    env = _removal_env(git_root, WorktreeRemoval.ON_COMPLETION)
    manager.add_completion_listener(env.on_completed)
    manager.add_removal_listener(env.remove)
    task = manager.create("dev", "x")
    root, _ = env.ensure_root(task.id)
    (root / "feature.txt").write_text("the work")
    manager.mark_done(task.id)  # → TESTING, no removal yet
    assert env.worktree_path(task.id).is_dir()
    manager.set_verification(task.id, True)
    assert task.status is TaskStatus.COMPLETED
    assert not env.worktree_path(task.id).exists()
    assert branch_exists(git_root, f"drcode/{task.id}")
    assert unmerged_count(git_root, f"drcode/{task.id}") >= 1
    assert last_commit_subject(git_root, f"drcode/{task.id}") == (
        f"drcode: snapshot of {task.id} (before removal)"
    )


def test_completion_removal_then_requeue_recreates_worktree(
    manager: TaskManager, git_root: Path
) -> None:
    """Sending input to a COMPLETED task re-queues it — ensure_root
    recreates the removed worktree from the surviving branch, so the
    earlier committed work is back on disk."""
    env = _removal_env(git_root, WorktreeRemoval.ON_COMPLETION)
    manager.add_completion_listener(env.on_completed)
    task = manager.create("research", "x", kind=TaskKind.RESEARCH)
    root, _ = env.ensure_root(task.id)
    (root / "docs.md").write_text("the research result")
    manager.mark_done(task.id)
    assert not env.worktree_path(task.id).exists()
    # Re-queue: the user sent new input to the completed task.
    root2, _ = env.ensure_root(task.id)
    assert env.worktree_path(task.id).is_dir()
    assert (root2 / "docs.md").read_text() == "the research result"


def test_completion_removal_off_by_default(
    manager: TaskManager, git_root: Path
) -> None:
    """The default (on_delete): a completion never touches the worktree —
    the historical lifecycle is untouched."""
    env = _removal_env(git_root, WorktreeRemoval.ON_DELETE)
    manager.add_completion_listener(env.on_completed)
    task = manager.create("research", "x", kind=TaskKind.RESEARCH)
    env.ensure_root(task.id)
    manager.mark_done(task.id)
    assert env.worktree_path(task.id).is_dir()
    assert branch_exists(git_root, f"drcode/{task.id}")


def test_removal_never_keeps_worktree_directory(git_root: Path) -> None:
    """worktree_removal: never — remove() only anchors the dirty tree on
    the branch (RULE 1) and keeps the directory AND the branch."""
    notices: list[tuple[str, str]] = []
    env = _removal_env(git_root, WorktreeRemoval.NEVER)
    env.event_log = lambda task_id, text: notices.append((task_id, text))
    env.ensure_root("t1")
    (env.worktree_path("t1") / "keep.txt").write_text("still wanted")
    env.remove("t1")
    assert env.worktree_path("t1").is_dir()
    assert branch_exists(git_root, "drcode/t1")
    assert last_commit_subject(git_root, "drcode/t1") == (
        "drcode: snapshot of t1 (before removal)"
    )


def test_removal_never_cancel_keeps_worktree(
    manager: TaskManager, git_root: Path
) -> None:
    """worktree_removal: never — even a cancel (Ctrl+X) leaves the
    directory in place; the dirty work is still committed to the branch."""
    env = _removal_env(git_root, WorktreeRemoval.NEVER)
    manager.add_removal_listener(env.remove)
    task = manager.create("t", "x")
    root, _ = env.ensure_root(task.id)
    (root / "wip.txt").write_text("cancelled, but kept")
    manager.cancel(task.id)
    assert env.worktree_path(task.id).is_dir()
    assert unmerged_count(git_root, f"drcode/{task.id}") >= 1


def test_cleanup_orphans_ignores_removal_never(
    manager: TaskManager, git_root: Path
) -> None:
    """The startup sweep ignores worktree_removal: a dead task's
    directory is garbage from a deleted task, not state to preserve —
    otherwise `never` would grow .drcode/worktrees/ without bound. Live
    tasks (including their dirty trees) are never touched."""
    env = _removal_env(git_root, WorktreeRemoval.NEVER)
    live = manager.create("live", "x")
    dead = manager.create("dead", "x")
    env.ensure_root(live.id)
    env.ensure_root(dead.id)
    manager.tasks.pop(dead.id)
    manager.storage.delete(dead.id)
    removed = env.cleanup_orphans(set(manager.tasks))
    assert dead.id in removed
    assert not env.worktree_path(dead.id).exists()
    assert env.worktree_path(live.id).is_dir()

