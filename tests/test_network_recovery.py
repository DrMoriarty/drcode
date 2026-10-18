"""Transcript-recovery semantics after a network failure mid-run.

Uses the real drcode agent (build_agent with its full tool set) driven by a
FunctionModel stub that raises on cue. Simulates what happens when the
endpoint dies between model requests: the run crashes, the transcript is
sanitized, and the next activation resumes the attempt without re-sending
the user's prompt.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.agent import AgentRun
from pydantic_ai.messages import ModelMessage

from drcode import agent as agent_mod
from drcode.agent import (
    AgentDeps,
    ends_with_open_request,
    extract_display,
    resumable_history,
    sanitized_messages,
    serialize_messages,
)
from drcode.config import Config
from drcode.models import TaskKind
from drcode.task_manager import TaskManager
from tests.network_mocks import (
    ScriptedState,
    api_error,
    http_error,
    text_response,
    tool_response,
)


@pytest.fixture
def scripted(monkeypatch) -> ScriptedState:
    state = ScriptedState()
    monkeypatch.setattr(agent_mod, "build_model", lambda cfg: state.model())
    return state


def build_doc_agent(root: Path):
    config = Config(project_root=root, model="mock-model", api_key="sk-test")
    return agent_mod.build_agent(config, kind=TaskKind.DOCUMENTATION, root=root)


async def _run(
    ag, prompt: str | None, history: list[ModelMessage] | None, deps: AgentDeps
) -> AgentRun:
    """Drive the agent: iter over the graph nodes (tool calls execute the
    same way as in the streamed TUI loop)."""
    async with ag.iter(prompt, message_history=history, deps=deps) as run:
        async for _node in run:
            pass
    return run


async def drive_expect_failure(
    ag, prompt: str | None, history: list[ModelMessage] | None, deps: AgentDeps
) -> tuple[Exception, AgentRun | None]:
    """Run the agent expecting a model error; returns (error, run) — `run`
    is the (closed) run context when the failure happened mid-run, None when
    it never entered (mirrors the app's `run` local)."""
    run: AgentRun | None = None
    try:
        async with ag.iter(prompt, message_history=history, deps=deps) as run:
            async for _node in run:  # type: ignore[union-attr]
                pass
    except Exception as e:  # noqa: BLE001 - the test asserts on the error type
        return e, run
    raise AssertionError("expected the run to fail")


def user_prompts(data: list[dict]) -> list[str]:
    return [text for _, role, text in extract_display(data) if role == "user"]


async def test_http_error_after_tool_call_leaves_resumable_transcript(
    project_root: Path, scripted: ScriptedState
) -> None:
    (project_root / "README.md").write_text("readme body")
    ag = build_doc_agent(project_root)
    mgr = TaskManager(project_root)
    mgr.create("docs", "the task")
    task = mgr.all_tasks()[0]
    deps = AgentDeps(root=project_root, manager=mgr, task_id=task.id)

    scripted.steps = [
        tool_response("read_file", {"path": "README.md"}),
        http_error(500, "gateway exploded"),
    ]
    error, run = await drive_expect_failure(ag, "the task", None, deps)
    assert type(error).__name__ == "ModelHTTPError"
    assert run is not None

    # What the app would store after the failure: the completed tool-call
    # round-trip is kept, only the crashed request is gone.
    stored = sanitized_messages(serialize_messages(run.all_messages()))
    kinds = [p["part_kind"] for m in stored for p in m["parts"]]
    assert "tool-call" in kinds
    assert "tool-return" in kinds
    # A continuation turn can pick the attempt up (prompt already recorded).
    assert ends_with_open_request(stored)

    # Resume: continuation runs with text=None and must not duplicate the
    # user's prompt.
    scripted.steps = [text_response("recovered answer")]
    run2 = await _run(ag, None, resumable_history(stored), deps)
    assert run2.result is not None
    assert run2.result.output == "recovered answer"

    final = serialize_messages(run2.result.all_messages())
    assert user_prompts(final) == ["the task"]
    final_kinds = [p["part_kind"] for m in final for p in m["parts"]]
    assert final_kinds.count("tool-return") == 1


async def test_timeout_before_first_response_is_fully_recoverable(
    project_root: Path, scripted: ScriptedState
) -> None:
    ag = build_doc_agent(project_root)
    mgr = TaskManager(project_root)
    mgr.create("t", "the task")
    task = mgr.all_tasks()[0]
    deps = AgentDeps(root=project_root, manager=mgr, task_id=task.id)

    scripted.steps = [api_error("Request timed out.")]
    error, run = await drive_expect_failure(ag, "the task", None, deps)
    assert type(error).__name__ == "ModelAPIError"
    assert run is not None

    stored = sanitized_messages(serialize_messages(run.all_messages()))
    assert len(stored) == 1  # only the user request, nothing of the response
    assert ends_with_open_request(stored)

    scripted.steps = [text_response("ok after retry")]
    run2 = await _run(ag, None, resumable_history(stored), deps)
    assert run2.result is not None
    assert run2.result.output == "ok after retry"
    final = serialize_messages(run2.result.all_messages())
    assert user_prompts(final) == ["the task"]


async def test_rate_limit_error_mid_run_does_not_corrupt_transcript(
    project_root: Path, scripted: ScriptedState
) -> None:
    """429 mid-run: the stored transcript must stay API-valid so the
    in-place retry can resume it."""
    ag = build_doc_agent(project_root)
    mgr = TaskManager(project_root)
    mgr.create("t", "the task")
    task = mgr.all_tasks()[0]
    deps = AgentDeps(root=project_root, manager=mgr, task_id=task.id)

    scripted.steps = [http_error(429)]
    error, run = await drive_expect_failure(ag, "the task", None, deps)
    assert type(error).__name__ == "ModelHTTPError"
    assert run is not None

    stored = sanitized_messages(serialize_messages(run.all_messages()))
    resumed = resumable_history(stored)
    assert len(resumed) == 1
    scripted.steps = [text_response("went through")]
    run2 = await _run(ag, None, resumed, deps)
    assert run2.result is not None
    assert run2.result.output == "went through"
