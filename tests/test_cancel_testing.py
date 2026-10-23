"""Cancelling a task that is in TESTING (Ctrl+X → action_cancel_task).

A TESTING task must be cancellable like any other unfinished one: it
completes immediately, the tester never runs for it again (neither a live
run nor a pending retry backoff), stale tester pipeline state is dropped
and a verdict arriving after the cancel (the tester's `set_verification`
tool runs in a worker thread and may still be in flight) is ignored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drcode.models import TaskStatus, VerificationStatus
from drcode.task_manager import TaskManager
from drcode.tui.app import DrCodeApp


@pytest.fixture
def app(project_root: Path) -> DrCodeApp:
    return DrCodeApp(project_root)


def _to_testing(manager: TaskManager, task_id: str) -> None:
    """Push a task into TESTING without a model (tests have none)."""
    task = manager.get(task_id)
    task.status = TaskStatus.TESTING
    task.touch()
    manager.storage.save(task)


# -- manager-level -----------------------------------------------------------------


def test_cancel_testing_completes_and_clears_label(manager: TaskManager) -> None:
    task = manager.create("t", "prompt")
    _to_testing(manager, task.id)
    # A previous testing round failed: the stale label is on the task.
    task.verification = VerificationStatus.FAILED

    manager.cancel(task.id)
    assert task.status is TaskStatus.COMPLETED
    assert task.completed_at is not None
    # The task completes without verification: the stale failed label
    # from the earlier round must not survive the cancel.
    assert task.verification is None


def test_set_verification_ignores_cancelled_task(manager: TaskManager) -> None:
    task = manager.create("t", "prompt")
    _to_testing(manager, task.id)
    manager.cancel(task.id)

    # The tester's set_verification tool runs in a worker thread and may
    # still be in flight when the user cancels: its verdict must not stamp
    # a label onto (or re-complete) the cancelled task.
    manager.set_verification(task.id, True)
    assert task.status is TaskStatus.COMPLETED
    assert task.verification is None


def test_set_verification_ignores_requeued_task(manager: TaskManager) -> None:
    """Same race on the other escape from TESTING: new input re-queues the
    task while the verdict is in flight — it must not complete it."""
    task = manager.create("t", "prompt")
    _to_testing(manager, task.id)
    manager.send(task.id, "one more round")

    manager.set_verification(task.id, True)
    assert task.status is TaskStatus.QUEUED
    assert task.verification is None


def test_set_verification_still_works_on_testing(manager: TaskManager) -> None:
    task = manager.create("t", "prompt")
    _to_testing(manager, task.id)
    manager.set_verification(task.id, True)
    assert task.status is TaskStatus.COMPLETED
    assert task.verification is VerificationStatus.PASSED


# -- TUI-level ---------------------------------------------------------------------


async def test_action_cancel_testing_task(app: DrCodeApp) -> None:
    task = app.manager.create("t", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        _to_testing(app.manager, task.id)
        app.refresh_sidebar()
        await pilot.pause()

        app.action_cancel_task()
        await pilot.pause()
        assert task.status is TaskStatus.COMPLETED
        assert not app.tester_running


async def test_action_cancel_testing_drops_retry_state(app: DrCodeApp) -> None:
    """Tester failed (no model) → retry backoff pending. Ctrl+X must
    complete the task, drop the retry bookkeeping and the fired retry
    timer must NOT resurrect the task into a new testing round."""
    task = app.manager.create("t", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        _to_testing(app.manager, task.id)
        app.refresh_sidebar()
        await pilot.pause()
        # Simulate the failed-tester state: retry backoff pending.
        app._tester_fail_count[task.id] = 3
        app._schedule_tester_retry(task.id)
        assert task.id in app._tester_retry_at

        app.action_cancel_task()
        await pilot.pause()
        assert task.status is TaskStatus.COMPLETED
        assert task.id not in app._tester_fail_count
        assert task.id not in app._tester_retry_at


# -- fix loop: tester retry_here requeues the task into development -------------------


async def test_fix_requeue_returns_task_to_development(app: DrCodeApp) -> None:
    """A tester failure attributed to the task's own changes (retry_here)
    must NOT complete the task: it goes back to the queue with the report
    as its pending input and the pipeline promotes it again — the fix round
    resumes the same transcript in the same (dirty) worktree."""
    task = app.manager.create("t", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        _to_testing(app.manager, task.id)
        app.refresh_sidebar()
        await pilot.pause()

        app.manager.requeue_failed_fix(
            task.id,
            "your change broke tests",
            max_rounds=app.settings.max_dev_rounds,
        )
        # In the live pipeline run_tester's completion path calls _advance();
        # here the manager transition is applied directly.
        app._advance()
        await pilot.pause()
        await pilot.pause()

        assert task.status is TaskStatus.ACTIVE
        assert task.fix_rounds == 1
        assert task.verification is None
        # The tester's report is queued as the task's next input (behind
        # the never-activated original prompt).
        assert task.pending_input[-1].text == "your change broke tests"


async def test_fix_requeue_cap_completes_task(app: DrCodeApp) -> None:
    """After max_dev_rounds consecutive failed rounds the retry_here
    failure completes the task as FAILED (no requeue, no promotion)."""
    task = app.manager.create("t", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        _to_testing(app.manager, task.id)
        task.fix_rounds = app.settings.max_dev_rounds - 1
        app.manager.storage.save(task)
        app.refresh_sidebar()
        await pilot.pause()

        app.manager.requeue_failed_fix(
            task.id,
            "still broken",
            max_rounds=app.settings.max_dev_rounds,
        )
        await pilot.pause()
        await pilot.pause()

        assert task.status is TaskStatus.COMPLETED
        assert task.fix_rounds == app.settings.max_dev_rounds - 1
        # No report was queued: the capped failure completes the task.
        assert [e.text for e in task.pending_input] == ["prompt"]
        # The cap is visible in the task's service events.
        assert any("fix-round cap" in e.text for e in task.events)


def test_fix_requeue_preserves_worktree(project_root: Path) -> None:
    """With worktree isolation the re-queued task's worktree (and its
    uncommitted work) is preserved: ensure_root reuses an existing
    worktree untouched, so the fix round develops in the full context of
    its previous work."""
    import subprocess

    from drcode.settings import ProjectSettings, TaskIsolation
    from drcode.task_env import TaskEnvironment

    # A git repository with one commit: the worktree isolation needs it.
    subprocess.run(
        ["git", "init", "-q", str(project_root)], check=True, capture_output=True
    )
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=project_root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=project_root, check=True)
    (project_root / "base.txt").write_text("base")
    subprocess.run(
        ["git", "add", "-A"], cwd=project_root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-qm", "init"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )

    manager = TaskManager(project_root)
    task = manager.create("t", "x")
    task.pending_input.clear()
    env = TaskEnvironment(
        project_root, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    root, _ = env.ensure_root(task.id)
    assert root != project_root  # isolated worktree created
    # Uncommitted work left behind by the interrupted round:
    (root / "wip.txt").write_text("half-done feature work")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)

    task.status = TaskStatus.TESTING
    manager.storage.save(task)
    manager.requeue_failed_fix(task.id, "broke tests", max_rounds=3)

    root2, _ = env.ensure_root(task.id)
    assert root2 == root
    assert (root2 / "wip.txt").read_text() == "half-done feature work"
