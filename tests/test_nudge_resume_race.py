"""Regression for incident #88dd4aad: a Ctrl+R during an outage wait raced
the outage probe's own restart, the probe run was superseded before its
first model step, the generation guard skipped the CancelledError handler's
unfinished-restore — and the next run found no resume marker (pending_input
empty, unfinished=False), so the epilogue auto-completed the task with zero
development work (the tester then bounced it straight back).

Two guards close the window:

1. `_consume_turn_input` no longer clears `task.unfinished` when it starts
   a continuation turn — the flag survives until the safe turn boundary
   after the response lands (and every cancel/error handler keeps it).
2. `_nudge_run` marks the interrupted attempt itself (mirroring the ESC
   stop), so a worker cancelled before its first step (or inside its setup
   window) can no longer lose the marker.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from drcode import agent_model
from drcode.models import TaskKind, TaskStatus
from drcode.transcript import serialize_messages
from drcode.tui.app import DrCodeApp
from tests.network_mocks import (
    ScriptedState,
    last_user_text,
    text_response,
    wait_until,
)


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedState:
    state = ScriptedState()
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())
    return state


async def test_supersede_during_continuation_turn_keeps_unfinished(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel a run between `_consume_turn_input` (continuation branch) and
    the first model node — the exact supersede window of the incident. The
    unfinished marker must survive (it is NOT cleared at continuation
    start), the restart must RESUME the attempt (a second model call), and
    only then auto-complete."""
    app = DrCodeApp(project_root)
    # No description prompt: the task must have EMPTY pending_input, so the
    # run under test actually takes the continuation branch (the second
    # create() argument would queue a fresh prompt instead).
    task = app.manager.create("raced", "", kind=TaskKind.RESEARCH)
    # The post-outage state: ACTIVE, interrupted attempt, sanitized
    # transcript ending with the unanswered user request.
    task.status = TaskStatus.ACTIVE
    task.unfinished = True
    task.messages = serialize_messages(
        [ModelRequest(parts=[UserPromptPart(content="fix it")])]
    )
    app.manager.storage.save(task)

    entered = asyncio.Event()
    release = asyncio.Event()
    calls = {"n": 0}
    orig_stream = scripted.stream_fn

    async def gated_stream(
        messages: list[Any], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        if "fix it" in last_user_text(messages):
            calls["n"] += 1
            if calls["n"] == 1:
                # Hold the FIRST continuation call mid-request, so the
                # supersede lands between _consume_turn_input and the
                # model node's completion.
                entered.set()
                await release.wait()
        async for chunk in orig_stream(messages, info):
            yield chunk

    # One scripted response: consumed by the RESTART's continuation turn.
    scripted.steps = [text_response("resumed work")]
    monkeypatch.setattr(
        scripted,
        "model",
        lambda: FunctionModel(scripted.fn, stream_function=gated_stream),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        # The probe-style run: dispatch and let it enter the continuation
        # turn's first model request.
        app.run_agent(task.id)
        await wait_until(
            lambda: entered.is_set(),
            pilot,
            what="continuation turn's first model request in flight",
        )
        # THE FIX: the marker is still on mid-continuation (it used to be
        # cleared here, before the request even went out).
        assert task.unfinished

        # Supersede it the way the Ctrl+R restart does: a new run_agent
        # bumps the generation and the exclusive group cancels the old
        # worker — whose CancelledError handler then skips the restore
        # via the generation guard.
        app.run_agent(task.id)
        await wait_until(
            lambda: calls["n"] == 2,
            pilot,
            what="restart resumed the attempt with its own model call",
        )
        release.set()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED and not app.agent_running,
            pilot,
            what="task completed after the resumed attempt",
        )

    # The superseded run was cancelled before its model call (the gate
    # held it ahead of the stub) — the ONE recorded call is the restart's
    # continuation: the old bug auto-completed here with ZERO calls and no
    # resumed work.
    assert scripted.calls == ["fix it"]
    answers = [
        p.get("content")
        for m in task.messages
        for p in m.get("parts", [])
        if p.get("part_kind") == "text"
    ]
    assert any(c and "resumed work" in str(c) for c in answers)
    assert not task.unfinished


async def test_ctrl_r_marks_unfinished_when_worker_cancelled_pre_step(
    project_root: Path,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl+R on an ACTIVE task must mark the attempt unfinished itself —
    even when the cancelled worker never gets to run its CancelledError
    handler (cancelled before its first step / inside the setup window):
    without the marker the scheduled restart finds nothing to resume."""
    app = DrCodeApp(project_root)
    # No description prompt: nothing may auto-promote the task — the fake
    # registry entry below must be the ONLY "run" (a real promoted run's
    # CancelledError handler would mark the attempt itself and mask the
    # pre-step window under test).
    task = app.manager.create("nudged", "", kind=TaskKind.RESEARCH)
    task.status = TaskStatus.ACTIVE
    app.manager.storage.save(task)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        # A live-run registry entry WITHOUT a real worker behind it: the
        # "cancelled before its first step" window — no CancelledError
        # handler exists to mark the attempt (the ESC-matrix trick).
        app._agent_runs[task.id] = 1
        restarts: list[Any] = []
        monkeypatch.setattr(
            app,
            "_schedule_pipeline_timer",
            lambda delay, cb, *a, **kw: restarts.append(cb),
        )
        app.action_nudge_run()
        await pilot.pause()

        assert task.id not in app._agent_runs
        assert restarts, "the restart must be scheduled"
        assert task.unfinished
        # Persisted, not just in memory (the restart may run in a fresh
        # activation after an app restart).
        raw = json.loads(
            (project_root / ".drcode" / "tasks" / f"{task.id}.json").read_text()
        )
        assert raw["unfinished"] is True
