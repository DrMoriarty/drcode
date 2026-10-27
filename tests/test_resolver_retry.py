"""Regression tests for the conflict-resolver retry state (the a6a6df25
silence): a network failure schedules a retry ("retrying in N s"), but the
round's `finally` block wiped that state whenever `task.needs_human` was
set — including a STALE flag left over from a previous round (it survives
restarts and is only cleared by a successful resolution or the merge
screen). The retry timer then fired into an empty schedule and the resolver
went silent forever.

The scenario: a task with a parked conflict and needs_human=True from a
previous round; a fresh resolver round whose first model request raises a
retryable network error. The retry MUST survive the finally block and the
round must actually resume after the backoff.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent

from drcode.models import TaskKind
from drcode.settings import ProjectSettings, ProjectSettingsStore, TaskIsolation
from drcode.task_env import TaskEnvironment
from drcode.task_manager import TaskManager
from drcode.tui.app import DrCodeApp
from tests.network_mocks import (
    ScriptedState,
    api_error,
    event_texts,
    wait_until,
)


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


@pytest.fixture
def isolated(project_root: Path) -> Path:
    """A git repo with worktree isolation persisted."""
    _git(project_root, "init", "-q")
    _git(project_root, "config", "user.email", "test@example.com")
    _git(project_root, "config", "user.name", "Test")
    (project_root / ".gitignore").write_text(".drcode/tasks/\n.drcode/worktrees/\n")
    (project_root / "shared.txt").write_text("base")
    _git(project_root, "add", "-A")
    _git(project_root, "commit", "-q", "-m", "init")
    ProjectSettingsStore(project_root).save(
        ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    return project_root


@pytest.fixture
def conflicted_task(isolated: Path) -> Any:
    """A task whose branch diverged from the base on shared.txt — the
    engine's base-into-worktree merge (run at round start by prepare())
    parks a real conflict in the task's worktree."""
    manager = TaskManager(isolated)
    task = manager.create("Fix the conflict", "fix the conflict")
    task.kind = TaskKind.BUG
    manager.storage.save(task)
    env = TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    wt, _ = env.ensure_root(task.id)
    (wt / "shared.txt").write_text("task side")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "task work")
    (isolated / "shared.txt").write_text("base side")
    _git(isolated, "add", "-A")
    _git(isolated, "commit", "-q", "-m", "base work")
    return task


async def test_retry_survives_stale_needs_human(
    isolated: Path,
    conflicted_task: Any,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A network failure in a fresh round must schedule a retry that
    actually fires — even though needs_human is a stale True from a
    previous round."""
    task = conflicted_task
    task.needs_human = True  # stale flag from the pre-restart round
    TaskManager(isolated).storage.save(task)

    app = DrCodeApp(isolated)

    # Quiesce the background actors so only the test drives the resolver.
    async def _noop() -> None:
        return None

    app._unmerged_work_startup_scan = _noop  # type: ignore[method-assign]
    app._poll_git_indicator = _noop  # type: ignore[method-assign]
    app._poll_pending_installs = _noop  # type: ignore[method-assign]
    # The queued task must NOT be promoted into development by the
    # pipeline's `_advance()` (the resolver round's finally tail): the
    # coding agent is a different machinery — here it would only crash on
    # the missing '#chat-log' DOM node and panic the app mid-test.
    app.run_agent = lambda task_id: None  # type: ignore[method-assign]

    # The resolver model fails with a retryable transport error on every
    # request — the test asserts the retries happen, not the resolution.
    state = ScriptedState()
    state.steps = [api_error(), api_error()]

    def fake_build_agent(config: Any, root: Path | None = None) -> Agent[Any, str]:
        return Agent(state.model())

    import drcode.tui.pipeline_resolver as pr

    pr.build_resolver_agent = fake_build_agent
    # Shrink the 5 s backoff so the timer fires within the test.
    monkeypatch.setattr(pr, "_RETRY_BASE_DELAY_S", 0.2)

    async with app.run_test() as pilot:
        await pilot.pause()
        # The app loaded its own Task instance from storage — service
        # events land there, not on the pre-start object.
        live = app.manager.get(task.id)
        assert live.needs_human  # the stale flag survived the load
        app._maybe_start_resolver(task.id)
        # First attempt fails -> the "retrying" event AND a live retry
        # schedule + resume flag (the finally block must not wipe them).
        await wait_until(
            lambda: any("retrying in" in t for t in event_texts(live)),
            pilot,
            what="first retry announcement",
        )
        assert task.id in app._resolver_retry_at
        assert task.id in app._resolver_resume_round
        # The retry fires and the round resumes in place (a second model
        # request) — before the fix the timer fired into an empty schedule
        # and the resolver stayed silent forever.
        await wait_until(
            lambda: len(state.calls) >= 2,
            pilot,
            what="resumed second model request",
        )
        # The second attempt fails too -> another retry is scheduled and
        # the network-attempt counter keeps counting across the resume.
        await wait_until(
            lambda: len([t for t in event_texts(live) if "retrying in" in t]) >= 2,
            pilot,
            what="second retry announcement",
        )
        assert app._resolver_net_count.get(task.id) == 2
