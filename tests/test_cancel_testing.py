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
        assert task.id not in app._tester_net_count
        assert app._tester_task_id is None

        # Let the pending retry timer fire and the pipeline advance.
        await pilot.pause(0.5)
        assert task.status is TaskStatus.COMPLETED
        assert not app.tester_running
