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
import time
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
    tool_response,
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
    monkeypatch.setattr(
        agent_model, "build_model", lambda cfg, announce_wait=None: state.model()
    )
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


async def test_empty_response_after_resume_keeps_unfinished(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
) -> None:
    """Regression for incident #0dc7fc65: a watchdog nudge resumed the
    interrupted attempt and the gateway answered with an EMPTY response
    (no text, no tool call). The run loop saw a closed exchange, the
    epilogue auto-completed the research task — and the completion
    listener tore down the worktree/branch with the partially-done work.

    The state seeded here is exactly what the epilogue saw after the turn
    boundary cleared the unfinished flag: ACTIVE, nothing pending, a
    transcript whose last message is a contentless model response. The
    epilogue must re-flag the attempt (unfinished) and NOT auto-complete;
    a real follow-up message then completes the task normally."""
    app = DrCodeApp(project_root)
    task = app.manager.create("stalled", "", kind=TaskKind.RESEARCH)
    task.status = TaskStatus.ACTIVE
    task.unfinished = False  # the turn boundary already cleared it
    task.messages = serialize_messages(
        [
            ModelRequest(parts=[UserPromptPart(content="fix it")]),
            text_response(""),  # the gateway's contentless answer
        ]
    )
    app.manager.storage.save(task)

    async with app.run_test() as pilot:
        await pilot.pause()
        await app._agent_run_epilogue(task, task.id)
        await pilot.pause()
        # THE FIX: no auto-completion on a contentless answer — the task
        # stays ACTIVE, paused on the unfinished marker.
        assert task.status is TaskStatus.ACTIVE
        assert task.unfinished
        # Persisted (an app restart's auto-resume keys off the flag).
        raw = json.loads(
            (project_root / ".drcode" / "tasks" / f"{task.id}.json").read_text()
        )
        assert raw["unfinished"] is True
        # The user sees why.
        assert any("empty response" in e.text for e in task.events), [
            e.text for e in task.events
        ]

        # A real follow-up message resumes and completes the task.
        scripted.steps = [text_response("resumed work")]
        app.manager.send(task.id, "continue")
        await pilot.pause()
        app.run_agent(task.id)
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED and not app.agent_running,
            pilot,
            what="task completed after a real follow-up answer",
        )
    assert any(
        p.get("content") == "resumed work"
        for m in task.messages
        for p in m.get("parts", [])
    )
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


async def test_tool_call_tail_epilogue_keeps_unfinished(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
) -> None:
    """Regression for incident #46c6314d (fix 1): a stale network-retry
    timer superseded the live run mid-tool; the fresh run dropped the
    unfinished flag on the response-tail and the epilogue — whose guard
    only recognized EMPTY responses — read the tail response with the
    unanswered tool call as a clean finish and sent the task to TESTING
    mid-development.

    Seeded state: ACTIVE, the flag already cleared, the transcript ending
    with a model response that carries a tool call no result ever arrived
    for. The epilogue must treat it like an interrupted attempt and NOT
    auto-complete; a follow-up then completes the task normally."""
    app = DrCodeApp(project_root)
    task = app.manager.create("midtool", "", kind=TaskKind.RESEARCH)
    task.status = TaskStatus.ACTIVE
    task.unfinished = False  # the turn boundary already cleared it
    task.messages = serialize_messages(
        [
            ModelRequest(parts=[UserPromptPart(content="fix it")]),
            # The superseded run's tail: tool call in flight, no result.
            tool_response("run_bash", {"command": "ninja -C build"}),
        ]
    )
    app.manager.storage.save(task)

    async with app.run_test() as pilot:
        await pilot.pause()
        await app._agent_run_epilogue(task, task.id)
        await pilot.pause()
        # THE FIX: no auto-completion on an open exchange — the task
        # stays ACTIVE, paused on the unfinished marker.
        assert task.status is TaskStatus.ACTIVE
        assert task.unfinished
        raw = json.loads(
            (project_root / ".drcode" / "tasks" / f"{task.id}.json").read_text()
        )
        assert raw["unfinished"] is True
        assert any("unfinished exchange" in e.text for e in task.events), [
            e.text for e in task.events
        ]

        # The resume finishes the development: the continuation turn
        # trims the broken tail, the model answers, the task completes.
        scripted.steps = [text_response("resumed work")]
        app.manager.send(task.id, "continue")
        await pilot.pause()
        app.run_agent(task.id)
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED and not app.agent_running,
            pilot,
            what="task completed after the resumed attempt",
        )
    assert scripted.calls, "the resume must consult the model"
    assert any(
        p.get("content") == "resumed work"
        for m in task.messages
        for p in m.get("parts", [])
    )
    assert not task.unfinished


async def test_unfinished_with_tool_call_tail_resumes_attempt(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
) -> None:
    """Regression for incident #46c6314d (fix 2): `_consume_turn_input`
    must not clear `task.unfinished` just because the transcript ends with
    a RESPONSE — a tail response with unanswered tool calls is an open
    exchange. The old code dropped the flag there (the run became
    "unresumable") and the run ended with zero work on the fresh attempt."""
    app = DrCodeApp(project_root)
    task = app.manager.create("superseded", "", kind=TaskKind.RESEARCH)
    task.status = TaskStatus.ACTIVE
    task.unfinished = True
    task.messages = serialize_messages(
        [
            ModelRequest(parts=[UserPromptPart(content="fix it")]),
            tool_response("run_bash", {"command": "ninja -C build"}),
        ]
    )
    app.manager.storage.save(task)

    scripted.steps = [text_response("resumed work")]
    async with app.run_test() as pilot:
        await pilot.pause()
        app.run_agent(task.id)
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED and not app.agent_running,
            pilot,
            what="task completed after the continuation turn",
        )
    # THE FIX: the continuation turn actually ran (the old code cleared the
    # flag, ended the run with zero model calls and left the task
    # ACTIVE-paused with a dead attempt).
    assert scripted.calls, "the continuation turn must consult the model"
    assert any(
        p.get("content") == "resumed work"
        for m in task.messages
        for p in m.get("parts", [])
    )
    assert not task.unfinished


async def test_nudge_defuses_armed_net_retry_timer(
    project_root: Path,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for incident #46c6314d (fix 3): Ctrl+R during a
    network-retry wait ("retry 5/5 in 60 s") must drop the armed retry
    timer along with the network-retry state — the orphaned timer used to
    fire minutes later, re-enter run_agent and have the exclusive worker
    group kill the fresh run mid-tool."""
    app = DrCodeApp(project_root)
    task = app.manager.create("retrying", "", kind=TaskKind.RESEARCH)
    task.status = TaskStatus.ACTIVE
    app.manager.storage.save(task)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Armed retry timer, the state _agent_turn_network_error leaves.
        retry_at = time.monotonic() + 60
        app._agent_net_retry_at[task.id] = retry_at
        app._agent_net_count[task.id] = 5
        app._agent_retry_resumes.add(task.id)
        fired: list[int] = []
        captured: list[Any] = []
        monkeypatch.setattr(
            app,
            "_schedule_pipeline_timer",
            lambda delay, cb, *a, **kw: captured.append(cb),
        )
        app._schedule_net_retry(task.id, retry_at, 60, lambda: fired.append(1))
        assert captured, "the guarded timer must be scheduled"
        # A matching stamp: the live timer fires.
        captured[-1]()
        assert fired == [1]

        # THE FIX: the nudge (Ctrl+R / stall watchdog) resets the
        # network-retry state...
        app._nudge_run(task)
        await pilot.pause()
        assert task.id not in app._agent_net_retry_at
        assert task.id not in app._agent_net_count
        assert task.id not in app._agent_retry_resumes
        # ...and the orphaned timer is defused: its stamp no longer
        # matches, so it never re-enters run_agent to supersede the
        # restarted run. (The nudge's own restart landed in `captured`
        # too — the stale net-retry callback is the one before it.)
        assert len(captured) >= 2
        captured[-2]()
        assert fired == [1]
