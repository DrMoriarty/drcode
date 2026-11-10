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

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent

from drcode import agent_model
from drcode import merge_engine as me
from drcode.agent import build_resolver_agent, build_resolver_reviewer_agent
from drcode.models import TaskKind
from drcode.settings import ProjectSettings, ProjectSettingsStore, TaskIsolation
from drcode.task_env import TaskEnvironment
from drcode.task_manager import TaskManager
from drcode.tui.app import DrCodeApp
from tests.network_mocks import (
    ScriptedState,
    api_error,
    event_texts,
    text_response,
    tool_response,
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

    def fake_build_agent(
        config: Any, root: Path | None = None, task_id: str | None = None
    ) -> Agent[Any, str]:
        return Agent(state.model())

    import drcode.core.pipeline_resolver as pr

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


# -- the targeted-repair fork (a 'lost' intent-review verdict) -----------------------
# The old behavior on ANY failed funnel level: restore the pristine conflict —
# a 35-file resolution thrown away over a one-line fix the reviewer itself
# described. The repair fork feeds the review report back into the SAME
# resolver transcript and re-runs the FULL funnel; the budget is 2 per
# conflict, and everything that is not a concrete 'lost' finding still rolls
# back.

_LOST_REPORT = (
    "shared.txt: the base side's greeting line was lost — the merged "
    "content must keep the greeting the task branch removed."
)


def _resolve_steps(report: str = "wove both sides") -> list[Any]:
    return [
        tool_response("write_file", {"path": "shared.txt", "content": "merged\n"}),
        tool_response("set_resolution", {"resolved": True, "report": report}),
        text_response("done"),
    ]


def _repair_steps() -> list[Any]:
    # The repair continuation: a point fix in the SAME transcript.
    return [
        tool_response(
            "set_resolution", {"resolved": True, "report": "restored the greeting"}
        ),
        text_response("repaired"),
    ]


def _review_steps(verdict: str, report: str) -> list[Any]:
    return [
        tool_response("set_review", {"verdict": verdict, "report": report}),
        text_response("reviewed"),
    ]


def _repair_app(
    isolated: Path,
    task: Any,
    monkeypatch: pytest.MonkeyPatch,
    state: ScriptedState,
) -> DrCodeApp:
    """The scripted-model app for the repair tests: the real resolver and
    reviewer builders (both draw their model through agent_model.build_model),
    the background actors quiesced, the repair delay shrunk."""
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())
    import drcode.core.pipeline_resolver as pr

    monkeypatch.setattr(pr, "build_resolver_agent", build_resolver_agent)
    monkeypatch.setattr(
        pr, "build_resolver_reviewer_agent", build_resolver_reviewer_agent
    )
    monkeypatch.setattr(pr, "_REPAIR_RETRY_DELAY_S", 0.2)

    app = DrCodeApp(isolated)

    async def _noop() -> None:
        return None

    app._unmerged_work_startup_scan = _noop  # type: ignore[method-assign]
    app._poll_git_indicator = _noop  # type: ignore[method-assign]
    app._poll_pending_installs = _noop  # type: ignore[method-assign]
    app.run_agent = lambda task_id: None  # type: ignore[method-assign]
    return app


async def test_repair_round_fixes_lost_work_and_commits(
    isolated: Path,
    conflicted_task: Any,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 'lost' review verdict with a concrete finding schedules a repair
    round INSTEAD of the rollback: the resolution stays in the worktree,
    the review report continues the SAME transcript, the full funnel
    re-runs, and an 'intact' re-review commits the merged result."""
    task = conflicted_task
    state = ScriptedState()
    state.steps = (
        _resolve_steps()
        + _review_steps("lost", _LOST_REPORT)
        + _repair_steps()
        + _review_steps("intact", "both sides present")
    )
    app = _repair_app(isolated, task, monkeypatch, state)

    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        app._maybe_start_resolver(task.id)
        await wait_until(
            lambda: any("FAILED the intent review" in t for t in event_texts(live)),
            pilot,
            what="the repair-fork announcement",
        )
        assert "nothing is rolled back" in "\n".join(event_texts(live))
        assert not live.needs_human
        # The repair invocation continues the round — the conflict is NOT
        # redeployed over the resolution.
        await wait_until(
            lambda: any("targeted repair" in t for t in event_texts(live)),
            pilot,
            what="the repair continuation event",
        )
        await wait_until(
            lambda: any("verified and committed" in t for t in event_texts(live)),
            pilot,
            what="the committed resolution",
        )
        assert not live.needs_human
        # ONE round entry: the repair turn rides in task.resolver_messages[-1].
        assert len(live.resolver_messages) == 1
        blob = json.dumps(live.resolver_messages[-1])
        assert "intent review report" in blob
        assert "greeting" in blob  # the reviewer's finding reached the agent
        assert app._resolver_repair_count.get(task.id) == 1
        assert not app._resolver_repair_round
        # The merge is concluded in the worktree — a fast-forward for the hook.
        env = TaskEnvironment(
            isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
        )
        wt = env.worktree_path(task.id)
        assert not me.in_progress(wt)
        assert (wt / "shared.txt").read_text() == "merged\n"


async def test_repair_budget_exhaustion_rolls_back(
    isolated: Path,
    conflicted_task: Any,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the repair budget (2) is spent on a persistently 'lost'
    verdict, the behavior is the old one: the conflict is restored and a
    human takes over."""
    task = conflicted_task
    state = ScriptedState()
    state.steps = (
        _resolve_steps()
        + _review_steps("lost", _LOST_REPORT)
        + _repair_steps()  # repair 1
        + _review_steps("lost", _LOST_REPORT)
        + _repair_steps()  # repair 2
        + _review_steps("lost", _LOST_REPORT)
    )
    app = _repair_app(isolated, task, monkeypatch, state)

    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        app._maybe_start_resolver(task.id)
        await wait_until(
            lambda: any("human must take over" in t for t in event_texts(live)),
            pilot,
            what="the final human hand-off",
        )
        assert live.needs_human
        assert app._resolver_repair_count.get(task.id) == 2
        assert not app._resolver_repair_round
        texts = "\n".join(event_texts(live))
        assert "Repair round 1 of 2" in texts
        assert "Repair round 2 of 2" in texts
        # The pristine conflict is back.
        env = TaskEnvironment(
            isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
        )
        wt = env.worktree_path(task.id)
        assert "<<<<<<<" in (wt / "shared.txt").read_text()


async def test_unsure_verdict_never_repairs(
    isolated: Path,
    conflicted_task: Any,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'unsure' is not a concrete finding — no repair round, the old
    rollback runs immediately."""
    task = conflicted_task
    state = ScriptedState()
    state.steps = _resolve_steps() + _review_steps(
        "unsure", "something might be missing but I cannot name it"
    )
    app = _repair_app(isolated, task, monkeypatch, state)

    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        app._maybe_start_resolver(task.id)
        await wait_until(
            lambda: any("human must take over" in t for t in event_texts(live)),
            pilot,
            what="the human hand-off",
        )
        assert live.needs_human
        texts = "\n".join(event_texts(live))
        assert "targeted repair" not in texts
        assert app._resolver_repair_count.get(task.id) is None
        assert not app._resolver_repair_round
        env = TaskEnvironment(
            isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
        )
        wt = env.worktree_path(task.id)
        assert "<<<<<<<" in (wt / "shared.txt").read_text()


async def test_vague_lost_report_never_repairs(
    isolated: Path,
    conflicted_task: Any,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 'lost' verdict whose report names nothing concrete (no file, no
    path, too short to act on) keeps the rollback."""
    task = conflicted_task
    state = ScriptedState()
    state.steps = _resolve_steps() + _review_steps(
        "lost", "Work from one side of the merge appears to be missing."
    )
    app = _repair_app(isolated, task, monkeypatch, state)

    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        app._maybe_start_resolver(task.id)
        await wait_until(
            lambda: any("human must take over" in t for t in event_texts(live)),
            pilot,
            what="the human hand-off",
        )
        assert live.needs_human
        assert "targeted repair" not in "\n".join(event_texts(live))
        assert app._resolver_repair_count.get(task.id) is None
