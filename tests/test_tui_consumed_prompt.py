"""Regression (task #17e64711): the chat must not lose the task's brief
between the run consuming the pending input and its first persist.

The incident: a task queued in a previous session auto-started on launch;
the original brief (still in pending_input) was rendered once — and then a
view_task() re-render in the first seconds of the run wiped it: the prompt
was already popped from pending_input but not persisted into the transcript
yet (the first node boundary does that), so the re-render found neither
copy and the screen showed only the agent's live work. Re-opening the task
later (after the persist) showed the brief again — from the transcript.

The fix records the consumed-but-unpersisted prompt in app state
(_consumed_prompts); _render_task_history draws it until the transcript
picks it up (or it is re-queued / the run ends).
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from drcode import agent_model
from drcode.models import TaskKind, TaskStatus
from drcode.task_manager import TaskManager
from drcode.transcript import all_user_prompts
from drcode.tui.app import DrCodeApp
from tests.network_mocks import (
    ScriptedState,
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
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())
    return state


async def test_reopen_during_consumption_window_keeps_prompt(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = TaskManager(project_root).create(
        "bug title", "исходное условие задачи", kind=TaskKind.BUG
    )
    app = DrCodeApp(project_root)
    # Hold the first model call, so the consumption happens but the prompt
    # is not persisted yet (the node boundary is what persists it).
    release = asyncio.Event()
    orig_stream = scripted.stream_fn

    async def gated_stream(
        messages: list[Any], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        for m in messages:
            if hasattr(m, "parts"):
                for p in m.parts:
                    if "исходное условие" in str(getattr(p, "content", "")):
                        await release.wait()
        async for chunk in orig_stream(messages, info):
            yield chunk

    monkeypatch.setattr(
        scripted,
        "model",
        lambda: FunctionModel(scripted.fn, stream_function=gated_stream),
    )
    scripted.steps = [
        tool_response("read_file", {"path": "x.py"}),
        text_response("готово"),
    ]
    # Emulate the production loop: Textual's real app installs
    # asyncio.eager_task_factory, the pilot env does not.
    asyncio.get_running_loop().set_task_factory(asyncio.eager_task_factory)
    async with app.run_test() as pilot:
        task = app.manager.get(task.id)
        await wait_until(
            lambda: app.agent_running and not task.pending_input,
            pilot,
            what="brief consumed",
        )
        # The consumed prompt is not in the transcript yet.
        assert task.messages == []

        def chat_text() -> str:
            return "\n".join(
                str(w.render()) for w in app.query_one("#chat-log").children
            )

        assert "исходное условие" in chat_text()
        # User re-opens the task mid-window: the brief must survive.
        app.view_task(task.id)
        for _ in range(20):
            await pilot.pause()
            if not app._chat_batching:
                break
        assert "исходное условие" in chat_text()

        # Once the transcript has the prompt, the record stops rendering
        # (no duplicate) and the fresh re-render shows the single copy.
        release.set()
        await wait_until(lambda: bool(task.messages), pilot, what="prompt persisted")
        app.view_task(task.id)
        for _ in range(20):
            await pilot.pause()
            if not app._chat_batching:
                break
        assert "исходное условие" in chat_text()
        assert not app._consumed_prompts or task.id not in app._consumed_prompts
        hits = sum(
            1
            for w in app.query_one("#chat-log").children
            if "исходное условие" in str(w.render())
        )
        assert hits == 1


async def test_cancel_during_consumption_window_keeps_prompt(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (task #0c519559): Ctrl+X during the consumption window must
    not swallow the brief.

    The run has already popped the prompt from pending_input but has not
    persisted it into the transcript yet; cancel() clears pending_input and
    completes the task, and the cancelled run's own persist skips a
    no-longer-ACTIVE task. The flush into the transcript is what keeps the
    user's input visible in the chat history."""
    task = TaskManager(project_root).create(
        "bug title", "исходное условие задачи", kind=TaskKind.BUG
    )
    app = DrCodeApp(project_root)
    # Hold the first model call: consumption happens, the persist doesn't.
    release = asyncio.Event()
    orig_stream = scripted.stream_fn

    async def gated_stream(
        messages: list[Any], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        for m in messages:
            if hasattr(m, "parts"):
                for p in m.parts:
                    if "исходное условие" in str(getattr(p, "content", "")):
                        await release.wait()
        async for chunk in orig_stream(messages, info):
            yield chunk

    monkeypatch.setattr(
        scripted,
        "model",
        lambda: FunctionModel(scripted.fn, stream_function=gated_stream),
    )
    scripted.steps = [text_response("готово")]
    asyncio.get_running_loop().set_task_factory(asyncio.eager_task_factory)
    async with app.run_test() as pilot:
        task = app.manager.get(task.id)
        await wait_until(
            lambda: app.agent_running and not task.pending_input,
            pilot,
            what="brief consumed",
        )
        assert task.messages == []

        app.action_cancel_task()
        await pilot.pause()

        assert task.status is TaskStatus.COMPLETED
        # The brief survived the cancel: it is in the transcript now.
        assert "исходное условие" in "\n".join(all_user_prompts(task.messages))
        # The in-flight record is consumed (no stale duplicate rendering).
        assert task.id not in app._consumed_prompts
        # Re-opening the cancelled task shows the brief from the transcript.
        app.view_task(task.id)
        for _ in range(20):
            await pilot.pause()
            if not app._chat_batching:
                break
        rendered = "\n".join(
            str(w.render()) for w in app.query_one("#chat-log").children
        )
        assert "исходное условие" in rendered
        assert "cancelled by the user" in rendered.lower()
    release.set()
