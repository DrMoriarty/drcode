"""The depends_on promotion gate end-to-end: the pipeline's promote worker
(dispatched by _advance when any queued task carries dependencies), its
service events and the TUI surfaces (form field, sidebar marker).

No model is configured in tests: a promoted task stops at run_agent's
"model not configured" guard and stays ACTIVE with its prompt kept — which
is exactly the observable "was promoted" signal used here.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import cast

import pytest
from textual.widgets import Static, TextArea

from drcode.models import Task, TaskStatus
from drcode.tui.app import DrCodeApp
from drcode.tui.inputs import DepsInput
from drcode.tui.sidebar import TaskItem


async def _settle(pilot, app: DrCodeApp, predicate, timeout_s: float = 5.0) -> bool:
    """Pump the event loop until `predicate()` holds (the promote worker
    runs its git check via asyncio.to_thread and resumes asynchronously)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await pilot.pause(0.05)
        if predicate():
            return True
    return False


def _complete(app: DrCodeApp, task: Task) -> None:
    app.manager.mark_done(task.id)
    app.manager.set_verification(task.id, True)


@pytest.fixture
def app(project_root: Path) -> DrCodeApp:
    return DrCodeApp(project_root)


# -- pipeline: the promotion gate --------------------------------------------------


async def test_blocked_task_is_not_promoted_until_dependency_completes(
    app: DrCodeApp,
) -> None:
    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the first-start setup dialog
        a = app.manager.create("first", "do a")
        b = app.manager.create("second", "do b")
        app.manager.set_dependencies(b.id, [a.id])
        app._advance()
        assert await _settle(pilot, app, lambda: a.status is TaskStatus.ACTIVE), (
            "runnable task must be promoted"
        )
        assert b.status is TaskStatus.QUEUED
        assert b.id in app._dep_gate, "the gate must record why b waits"

        _complete(app, a)
        app._advance()
        assert await _settle(pilot, app, lambda: b.status is TaskStatus.ACTIVE), (
            "completing the dependency must unblock b"
        )
        assert "Dependencies satisfied" in " ".join(e.text for e in b.events)


async def test_block_service_event_emitted_once(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await pilot.press("escape")
        a = app.manager.create("first", "do a")
        b = app.manager.create("second", "do b")
        app.manager.set_dependencies(b.id, [a.id])
        for _ in range(3):
            app._advance()
            await pilot.pause(0.1)
        await _settle(pilot, app, lambda: a.status is TaskStatus.ACTIVE)
        texts = [e.text for e in b.events]
        assert sum("Task waits" in t for t in texts) == 1
        assert any("#" + a.id in t and "not completed" in t for t in texts)


async def test_advance_keeps_testing_priority_over_dependency_queue(
    app: DrCodeApp,
) -> None:
    async with app.run_test() as pilot:
        await pilot.press("escape")
        t = app.manager.create("being tested", "x")
        app.manager.mark_done(t.id)  # TESTING
        a = app.manager.create("runnable", "do a")
        b = app.manager.create("chained", "do b")
        app.manager.set_dependencies(b.id, [a.id])
        app._advance()
        await pilot.pause(0.2)
        # No new development while any testing is incomplete — even with
        # runnable dependency-chained tasks in the queue.
        assert a.status is TaskStatus.QUEUED
        assert b.status is TaskStatus.QUEUED
        assert t.status is TaskStatus.TESTING


async def test_settled_notification_reports_blocked_queue(
    app: DrCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    details: list[tuple[bool, str]] = []
    monkeypatch.setattr(
        type(app.notifier),
        "pipeline_idle",
        lambda self, *, blocked, detail, prime: details.append((blocked, detail)),
    )
    async with app.run_test() as pilot:
        await pilot.press("escape")
        a = app.manager.create("first", "do a")
        b = app.manager.create("second", "do b")
        app.manager.set_dependencies(b.id, [a.id])
        # Make a un-promotable too (no pending input, no unfinished
        # attempt): the queue settles with both tasks waiting.
        a.pending_input.clear()
        app._advance()
        await _settle(pilot, app, lambda: b.id in app._dep_gate)
        app._notifications_primed = True
        app._advance()
        await pilot.pause(0.2)
        assert any("blocked by dependencies: 1" in d for _, d in details)


async def test_promote_failed_queue_respects_blocked(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await pilot.press("escape")
        a = app.manager.create("failed", "do a")
        dep = app.manager.create("dep", "x")
        b = app.manager.create("also failed", "do b")
        app.manager.set_dependencies(b.id, [dep.id])
        # dep is parked on hold: the queue holds exactly the failed a and
        # the failed-and-blocked b — the deadlock-escape situation.
        app.manager.set_hold(dep.id, on=True)
        app._agent_failed.update({a.id, b.id})
        promoted = app._promote_failed_queue(blocked={b.id})
        # Only the genuinely failed task is rescued; the dependency-blocked
        # one is neither lifted nor promoted.
        assert promoted is not None and promoted.id == a.id
        assert b.status is TaskStatus.QUEUED
        assert b.id in app._agent_failed


# -- TUI: the new-task form field ---------------------------------------------------


async def test_form_creates_task_with_dependencies(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await pilot.press("escape")
        existing = app.manager.create("other", "x")
        app.action_new_task()
        await pilot.pause()
        app.query_one("#nt-input", TextArea).text = "chained task"
        app.query_one("#nt-deps", DepsInput).value = f" {existing.id} , "
        await pilot.press("ctrl+enter")
        await pilot.pause()
        tasks = app.manager.all_tasks()
        assert len(tasks) == 2
        assert tasks[1].depends_on == [existing.id]
        assert not app._new_task_mode


async def test_form_rejects_unknown_dependency_and_stays_open(
    app: DrCodeApp,
) -> None:
    async with app.run_test() as pilot:
        await pilot.press("escape")
        app.action_new_task()
        await pilot.pause()
        app.query_one("#nt-input", TextArea).text = "chained task"
        app.query_one("#nt-deps", DepsInput).value = "ffffffff"
        await pilot.press("ctrl+enter")
        await pilot.pause()
        # Nothing created, the form stays open with the error shown.
        assert len(app.manager.all_tasks()) == 0
        assert app._new_task_mode
        assert "unknown task id" in str(app.query_one("#nt-deps-note", Static).render())


async def test_form_rejects_dependency_cycle(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await pilot.press("escape")
        a = app.manager.create("a", "x")
        b = app.manager.create("b", "x")
        app.manager.set_dependencies(a.id, [b.id])
        # Edit the held copy of b... b is queued; hold it to edit deps.
        app.manager.set_hold(b.id, on=True)
        app.view_task(b.id)
        app.action_edit_task()
        await pilot.pause()
        app.query_one("#nt-input", TextArea).text = "b text"
        app.query_one("#nt-deps", DepsInput).value = a.id
        await pilot.press("ctrl+enter")
        await pilot.pause()
        assert app._new_task_mode, "the cycle error must keep the form open"
        assert "cycle" in str(app.query_one("#nt-deps-note", Static).render())
        assert b.depends_on == []


async def test_edit_held_task_loads_and_saves_dependencies(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await pilot.press("escape")
        a = app.manager.create("a", "x")
        b = app.manager.create("b", "x")
        app.manager.set_dependencies(b.id, [a.id])
        app.manager.set_hold(b.id, on=True)
        app.view_task(b.id)
        app.action_edit_task()
        await pilot.pause()
        assert app.query_one("#nt-deps", DepsInput).value == a.id
        app.query_one("#nt-input", TextArea).text = "rewritten b"
        await pilot.press("ctrl+enter")
        await pilot.pause()
        assert not app._new_task_mode
        assert app.manager.get(b.id).depends_on == [a.id]


# -- TUI: sidebar lock marker ---------------------------------------------------------


async def test_sidebar_shows_lock_marker_on_blocked_task(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await pilot.press("escape")
        a = app.manager.create("first", "do a")
        b = app.manager.create("second", "do b")
        app.manager.set_dependencies(b.id, [a.id])
        app.refresh_sidebar()
        items = cast(
            list[TaskItem],
            [w for w in app.sidebar.children if isinstance(w, TaskItem)],
        )
        b_items = [i for i in items if i.task_id == b.id]
        assert b_items and "🔒" in b_items[0].text and a.id in b_items[0].text
        a_items = [i for i in items if i.task_id == a.id]
        assert a_items and "🔒" not in a_items[0].text


# -- integration: mergedness under worktree isolation -------------------------------


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


async def test_mergedness_gate_with_real_git(project_root: Path) -> None:
    """Under worktree isolation a COMPLETED dependency with unmerged
    commits still blocks; a manual `git merge drcode/<dep>` unblocks the
    dependent without any drcode action (the dynamic gate)."""
    _git(project_root, "init", "-q")
    _git(project_root, "config", "user.email", "test@example.com")
    _git(project_root, "config", "user.name", "Test")
    (project_root / ".gitignore").write_text(".drcode/tasks/\n.drcode/worktrees/\n")
    (project_root / "README.md").write_text("hello")
    _git(project_root, "add", "-A")
    _git(project_root, "commit", "-q", "-m", "init")

    app = DrCodeApp(project_root)
    from drcode.settings import TaskIsolation

    app.settings.task_isolation = TaskIsolation.WORKTREE
    async with app.run_test() as pilot:
        await pilot.press("escape")
        a = app.manager.create("first", "do a")
        b = app.manager.create("second", "do b")
        app.manager.set_dependencies(b.id, [a.id])
        _complete(app, a)
        # The dependency is COMPLETED, but its work sits unmerged on the
        # task branch: b must stay queued.
        _git(project_root, "checkout", "-q", "-b", f"drcode/{a.id}")
        (project_root / "work.txt").write_text("work")
        _git(project_root, "add", "-A")
        _git(project_root, "commit", "-q", "-m", "work")
        _git(project_root, "checkout", "-")
        app._advance()
        assert await _settle(pilot, app, lambda: b.id in app._dep_gate), (
            "unmerged dependency work must block b"
        )
        assert b.status is TaskStatus.QUEUED
        assert any("not merged" in e.text for e in b.events), [e.text for e in b.events]

        # The user merges by hand: the next promotion attempt unblocks b.
        _git(project_root, "merge", "-q", "--no-ff", f"drcode/{a.id}")
        app._advance()
        assert await _settle(pilot, app, lambda: b.status is TaskStatus.ACTIVE), (
            "manual merge must unblock the dependent"
        )
