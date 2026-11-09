"""Regression tests for the resolver scope guard and end-of-round hygiene
(the c3b17718 incident): a resolver round edited `src/drcode/tui/merge/
screen.py` OUTSIDE the triage report's set; the verification funnel
measured only the conflicted paths, the concluding commit staged only
them — the foreign edit stayed uncommitted in the worktree and blocked
the accept hook hours later ("the worktree has uncommitted changes"),
with the cause recoverable only from the transcript.

The fix, asserted here:

1. a scope guard BEFORE the verdict is applied: any worktree change the
   resolver made outside the triage set rejects the round outright —
   rollback + `needs_human`, nothing committed, the files named in the
   service events (the funnel never runs on a tree the report never
   covered);
2. end-of-round hygiene: EVERY round end (rejection, funnel failure,
   error/retry) leaves a service event naming uncommitted leftovers
   outside the conflicted set — the silent dirty tail is gone.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any, cast

import pytest

from drcode import agent_model
from drcode.agent import build_resolver_agent
from drcode.models import TaskKind, TaskStatus, VerificationStatus
from drcode.settings import ProjectSettings, ProjectSettingsStore, TaskIsolation
from drcode.task_env import TaskEnvironment
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


@pytest.fixture
def real_resolver_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the REAL builder (the scripted model arrives through
    agent_model.build_model). Guards against a leaked stub from another
    test file on the same xdist worker — a stub agent has no tools, so
    the round would die on the first write_file and the scope guard
    would see nothing to reject."""
    import drcode.core.pipeline_resolver as pr

    monkeypatch.setattr(pr, "build_resolver_agent", build_resolver_agent)


def _git(root: Any, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


@pytest.fixture
def isolated(project_root: Any) -> Any:
    """A git repo with worktree isolation persisted."""
    subprocess.run(["git", "init", "-q"], cwd=project_root, check=True)
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
def env(isolated: Any) -> TaskEnvironment:
    return TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )


def _finished_task(app: DrCodeApp) -> Any:
    task = app.manager.create("Fix the bug", "fix the bug")
    task.kind = TaskKind.FEATURE
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    app.manager.storage.save(task)
    return task


def _park_conflict(env: TaskEnvironment, isolated: Any, task_id: str) -> Any:
    """Diverge the task branch from the base on shared.txt so prepare()
    parks a real conflict in the worktree."""
    wt, _ = env.ensure_root(task_id)
    (wt / "shared.txt").write_text("task side\n")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "task work")
    (isolated / "shared.txt").write_text("base side\n")
    _git(isolated, "add", "-A")
    _git(isolated, "commit", "-q", "-m", "base work")
    assert env.unmerged_commits(task_id) == 1
    return wt


def _quiesce(app: DrCodeApp) -> None:
    async def _noop() -> None:
        return None

    app._unmerged_work_startup_scan = _noop  # type: ignore[method-assign]
    app._poll_git_indicator = _noop  # type: ignore[method-assign]
    app._poll_pending_installs = _noop  # type: ignore[method-assign]
    app.run_agent = lambda task_id: None  # type: ignore[method-assign]


async def test_scope_guard_rejects_out_of_scope_edit(
    isolated: Any,
    env: TaskEnvironment,
    configured_env: None,
    real_resolver_agent: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The incident scenario: the resolver resolves the conflicted file
    AND edits a foreign file. The round must be REJECTED — nothing
    committed, needs_human, the conflict redeployed, the foreign file
    named in the service events — and the funnel must not run."""
    state = ScriptedState()
    state.steps = [
        # The out-of-scope edit (the incident's foreign feature port).
        tool_response("write_file", {"path": "foreign.py", "content": "x\n"}),
        # The legitimate resolution of the conflicted file.
        tool_response("write_file", {"path": "shared.txt", "content": "merged\n"}),
        tool_response("set_resolution", {"resolved": True, "report": "wove both"}),
        text_response("done"),
    ]
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())

    app = DrCodeApp(isolated)
    task = _finished_task(app)
    wt = _park_conflict(env, isolated, task.id)

    _quiesce(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        app._maybe_start_resolver(task.id)
        await wait_until(
            lambda: any("REJECTED" in t for t in event_texts(live)),
            pilot,
            what="scope-guard rejection event",
        )
        texts = "\n".join(event_texts(live))
        assert "foreign.py" in texts
        assert "SCOPE EXPANSION" in texts
        assert live.needs_human
        # Nothing was committed — the conflict is back, the branch tip
        # gained no merge commit.
        assert env.unmerged_commits(task.id) == 1
        assert "<<<<<<<" in (wt / "shared.txt").read_text()
        # The foreign edit was NOT committed either (it survives in the
        # worktree — but now with a named trace in the events).
        assert (wt / "foreign.py").read_text() == "x\n"


async def test_error_round_leaves_named_trace(
    isolated: Any,
    env: TaskEnvironment,
    configured_env: None,
    real_resolver_agent: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A round interrupted mid-work (ModelAPIError — the incident's
    connection failure) schedules retries and finally hands over — the
    worktree tail it leaves behind is named in a service event, never
    silent."""
    state = ScriptedState()
    # First request does the out-of-scope edit, then the round dies with
    # the incident's connection error; the network-retry budget is burned
    # down to one retry so the round hands over quickly (each retry
    # resumes with history and consumes another step).
    state.steps = [
        tool_response("write_file", {"path": "foreign.py", "content": "x\n"}),
        *([api_error()] * 3),
    ]
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())

    app = DrCodeApp(isolated)
    task = _finished_task(app)
    _park_conflict(env, isolated, task.id)

    import drcode.core.pipeline_resolver as pr

    monkeypatch.setattr(pr, "_RETRY_BASE_DELAY_S", 0.05)
    monkeypatch.setattr(pr, "_MAX_RESOLVER_NETWORK_ATTEMPTS", 2)

    _quiesce(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        app._maybe_start_resolver(task.id)
        await wait_until(
            lambda: any("abandoned after repeated" in t for t in event_texts(live)),
            pilot,
            what="give-up event",
        )
        await wait_until(
            lambda: any(
                "uncommitted changes outside the conflicted set" in t
                for t in event_texts(live)
            ),
            pilot,
            what="leftover-trace event",
        )
        assert any(
            "foreign.py" in t for t in event_texts(live) if "uncommitted changes" in t
        )
        assert live.needs_human
        assert env.unmerged_commits(task.id) == 1


async def test_cancelled_round_leaves_named_trace(
    isolated: Any,
    env: TaskEnvironment,
    configured_env: None,
    real_resolver_agent: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A round aborted mid-work (ESC / the stall nudge): the worktree
    tail it leaves behind is still named in a service event — hygiene
    covers the cancellation path too."""
    state = ScriptedState()
    state.steps = [
        tool_response("write_file", {"path": "foreign.py", "content": "x\n"}),
        # asyncio.CancelledError is a BaseException, hence the cast — the
        # scripted queue only types ordinary Exception steps.
        cast(Any, asyncio.CancelledError()),
    ]
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())

    app = DrCodeApp(isolated)
    task = _finished_task(app)
    _park_conflict(env, isolated, task.id)

    _quiesce(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        app._maybe_start_resolver(task.id)
        await wait_until(
            lambda: any(
                "uncommitted changes outside the conflicted set" in t
                for t in event_texts(live)
            ),
            pilot,
            what="leftover-trace event after cancellation",
        )
        assert any(
            "foreign.py" in t for t in event_texts(live) if "uncommitted changes" in t
        )
        assert env.unmerged_commits(task.id) == 1
