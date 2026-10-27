"""Tests for the per-task isolation pin (`Task.isolation_pinned`).

The isolation mode is a property of a task's active work, not of the
moment a run boundary reads the setting: the mode observed at the task's
FIRST activation is frozen onto the task, so toggling `task_isolation`
mid-task (between the coder's run and the tester round, across a restart,
...) never moves the work between the worktree and the main checkout —
neither tree holds the other's changes, and a run in the wrong tree
verifies nothing (the false-FAILED bug this pin exists to prevent).

Contract under test:

- first activation stamps the live mode (None → True/False);
- `pinned True` keeps the worktree path even with the setting off
  (off→on toggle: the shared-mode task stays shared);
- `pinned False` keeps the project root even with the setting on
  (on→off toggle: the worktree task stays in its worktree);
- the pin is persisted with the task (survives restarts);
- the startup pre-stamp reconstructs the mode for tasks that were
  mid-work at shutdown from the surviving evidence (worktree directory,
  `drcode/<id>` branch, unfinished attempt);
- a task pinned to a worktree keeps snapshotting (RULE 1) after the
  setting was turned off.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from drcode.models import Task, TaskStatus
from drcode.settings import ProjectSettings, TaskIsolation
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
    (project_root / ".gitignore").write_text(".drcode/tasks/\n.drcode/worktrees/\n")
    (project_root / "README.md").write_text("hello")
    _git(project_root, "add", "-A")
    _git(project_root, "commit", "-q", "-m", "init")
    return project_root


@pytest.fixture
def env(git_root: Path) -> TaskEnvironment:
    """Isolation OFF (the default) — the tests toggle it mid-task."""
    return TaskEnvironment(git_root, ProjectSettings())


def turn_on(env: TaskEnvironment) -> None:
    env.settings.task_isolation = TaskIsolation.WORKTREE


def turn_off(env: TaskEnvironment) -> None:
    env.settings.task_isolation = TaskIsolation.OFF


def branch_exists(env: TaskEnvironment, task_id: str) -> bool:
    code, _ = env._git("rev-parse", "--verify", "--quiet", env.branch(task_id))
    return code == 0


# -- first activation stamps the live mode ----------------------------------------


def test_first_activation_with_isolation_off_stamps_false(env: TaskEnvironment) -> None:
    task = Task(title="t")
    root, note = env.ensure_root(task.id, task)
    assert root == env.project_root
    assert note == ""
    assert task.isolation_pinned is False


def test_first_activation_with_isolation_on_stamps_true(git_root: Path) -> None:
    env = TaskEnvironment(
        git_root, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    task = Task(title="t")
    root, _note = env.ensure_root(task.id, task)
    assert root == env.worktree_path(task.id)
    assert task.isolation_pinned is True
    assert branch_exists(env, task.id)


def test_ensure_root_without_task_keeps_live_behavior(env: TaskEnvironment) -> None:
    # No Task passed (internal/test callers): the historic live-setting
    # behavior, nothing stamped anywhere.
    turn_on(env)
    root, _ = env.ensure_root("t1")
    assert root == env.worktree_path("t1")
    turn_off(env)
    root, _ = env.ensure_root("t2")
    assert root == env.project_root


# -- off → on: the shared-mode task stays shared -----------------------------------


def test_toggle_off_then_on_keeps_task_in_project_root(env: TaskEnvironment) -> None:
    task = Task(title="t")
    root, _ = env.ensure_root(task.id, task)
    assert root == env.project_root
    assert task.isolation_pinned is False
    # The user enables worktree isolation (maybe with parallel slots) —
    # the task's next run boundary (tester round, unfinished resume)
    # must NOT jump into a fresh worktree forked from the branch tip:
    # the coder's uncommitted changes live in the main checkout.
    turn_on(env)
    root, note = env.ensure_root(task.id, task)
    assert root == env.project_root
    assert note == ""
    assert not env.worktree_path(task.id).exists()
    assert task.isolation_pinned is False


# -- on → off: the worktree task stays in its worktree -----------------------------


def test_toggle_on_then_off_keeps_task_in_worktree(git_root: Path) -> None:
    env = TaskEnvironment(
        git_root, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    task = Task(title="t")
    root, _ = env.ensure_root(task.id, task)
    assert root == env.worktree_path(task.id)
    # Uncommitted work in the worktree (the coder mid-run).
    (root / "work.txt").write_text("wip")
    # The user turns isolation off — the next run boundary must stay in
    # the task's worktree (the main checkout has none of the task's work).
    turn_off(env)
    root2, note = env.ensure_root(task.id, task)
    assert root2 == env.worktree_path(task.id)
    assert note == ""
    assert (root2 / "work.txt").read_text() == "wip"
    assert task.isolation_pinned is True


def test_pinned_worktree_recreated_after_manual_deletion(git_root: Path) -> None:
    # A pinned task's worktree deleted by hand is recreated from the
    # surviving branch — even with the setting off (the pin, not the
    # setting, drives the mode).
    env = TaskEnvironment(
        git_root, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    task = Task(title="t")
    root, _ = env.ensure_root(task.id, task)
    (root / "work.txt").write_text("wip")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "task work")
    turn_off(env)
    shutil.rmtree(root)
    root2, note = env.ensure_root(task.id, task)
    assert root2 == env.worktree_path(task.id)
    assert note == ""
    # The branch (with the committed work) was re-checked-out.
    assert (root2 / "work.txt").read_text() == "wip"
    _code, out = env._git("log", "-1", "--format=%s", cwd=root2)
    assert "task work" in out


def test_pinned_off_task_with_isolation_on_does_not_create_worktree(
    git_root: Path,
) -> None:
    # Mirror-image guard: a task pinned to the project directory never
    # gets a worktree, no matter how long the setting stays on.
    env = TaskEnvironment(
        git_root, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    task = Task(title="t")
    task.isolation_pinned = False
    root, _note = env.ensure_root(task.id, task)
    assert root == env.project_root
    assert not env.worktree_path(task.id).exists()
    assert not branch_exists(env, task.id)


# -- persistence --------------------------------------------------------------------


def test_pin_survives_storage_roundtrip(manager: TaskManager) -> None:
    created = manager.create("pin me", "do it")
    created.isolation_pinned = True
    manager.storage.save(created)
    path = manager.storage.root / ".drcode" / "tasks" / f"{created.id}.json"
    loaded = TaskManager(manager.storage.root).storage.load_one(path)
    assert loaded is not None
    assert loaded.isolation_pinned is True


# -- startup pre-stamp ---------------------------------------------------------------


def test_prestamp_true_from_existing_worktree(git_root: Path) -> None:
    env = TaskEnvironment(
        git_root, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    task = Task(title="t", status=TaskStatus.QUEUED, unfinished=True)
    env.ensure_root(task.id, task)  # worktree exists, pin stamped True
    task.isolation_pinned = None  # simulate the pre-pin-feature restart
    turn_off(env)  # the setting changed while the app was down
    stamped = env.prestamp_isolation_pins([task])
    assert stamped == [task]
    assert task.isolation_pinned is True
    # And the task really stays in its worktree.
    root, _ = env.ensure_root(task.id, task)
    assert root == env.worktree_path(task.id)


def test_prestamp_true_from_branch_without_worktree(git_root: Path) -> None:
    env = TaskEnvironment(
        git_root, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    task = Task(title="t", status=TaskStatus.QUEUED, unfinished=True)
    env.ensure_root(task.id, task)
    shutil.rmtree(env.worktree_path(task.id))  # branch survives the deletion
    task.isolation_pinned = None
    stamped = env.prestamp_isolation_pins([task])
    assert stamped == [task]
    assert task.isolation_pinned is True


def test_prestamp_false_for_unfinished_shared_task(git_root: Path) -> None:
    # A task that was mid-work in the SHARED checkout (no worktree, no
    # branch) at shutdown, with the setting turned on meanwhile: it must
    # be pinned to the project directory — a fresh worktree would NOT
    # contain its uncommitted changes.
    env = TaskEnvironment(git_root, ProjectSettings())
    task = Task(title="t", status=TaskStatus.QUEUED, unfinished=True)
    turn_on(env)
    stamped = env.prestamp_isolation_pins([task])
    assert stamped == [task]
    assert task.isolation_pinned is False
    root, _ = env.ensure_root(task.id, task)
    assert root == env.project_root


def test_prestamp_false_for_testing_task_without_worktree(git_root: Path) -> None:
    env = TaskEnvironment(git_root, ProjectSettings())
    task = Task(title="t", status=TaskStatus.TESTING)
    task.messages.append({"role": "user", "parts": []})
    turn_on(env)
    stamped = env.prestamp_isolation_pins([task])
    assert stamped == [task]
    assert task.isolation_pinned is False


def test_prestamp_skips_never_started_tasks(git_root: Path) -> None:
    # A queued task that never ran keeps None and stamps at its first
    # activation (no backfill).
    env = TaskEnvironment(git_root, ProjectSettings())
    queued = Task(title="t", status=TaskStatus.QUEUED)
    turn_on(env)
    assert env.prestamp_isolation_pins([queued]) == []
    assert queued.isolation_pinned is None


def test_prestamp_skips_already_pinned_tasks(git_root: Path) -> None:
    env = TaskEnvironment(git_root, ProjectSettings())
    task = Task(title="t", status=TaskStatus.TESTING, isolation_pinned=False)
    assert env.prestamp_isolation_pins([task]) == []
    assert task.isolation_pinned is False


# -- RULE 1 tolerance for pinned worktrees -------------------------------------------


def test_snapshot_covers_pinned_worktree_after_setting_off(git_root: Path) -> None:
    env = TaskEnvironment(
        git_root, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    task = Task(title="t")
    root, _ = env.ensure_root(task.id, task)
    (root / "wip.txt").write_text("precious")
    turn_off(env)
    # The verdict-time snapshot still anchors the dirty tree on the
    # branch even though the live setting is off now.
    result = env.snapshot(task.id, "verdict")
    assert result is not None and result.startswith("snapshot committed")
    code, out = env._git("show", f"{env.branch(task.id)}:wip.txt")
    assert code == 0 and out == "precious"
