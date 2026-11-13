"""Model-switch housekeeping between a task's runs (persisted
`Task.last_model_key`, see core/pipeline_compaction.

When the model that factually STARTED the previous run differs from the
freshly resolved one — a settings edit, or a restart after one — the
transcript is cleaned for the new model BEFORE the first turn: the
previous model's thinking parts are stripped mechanically (always), a
sufficiently large transcript is additionally force-compacted, and a
synthetic takeover note is added when the run will make a model call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from drcode import agent_model
from drcode.agent import (
    MODEL_SWITCH_NOTE_PREFIX,
    all_user_prompts,
    drop_model_switch_notes,
    is_summary_message,
    serialize_messages,
    split_original_prompts,
    strip_thinking_parts,
)
from drcode.models import TaskKind, TaskStatus
from drcode.providers import AgentRole
from drcode.tui.app import DrCodeApp
from tests.network_mocks import (
    ScriptedState,
    event_texts,
    text_response,
    wait_until,
)

# -- fixtures (mirroring test_compaction_resume.py) -----------------------------


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


@pytest.fixture
def app(project_root: Path, configured_env: None) -> DrCodeApp:
    return DrCodeApp(project_root)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedState:
    state = ScriptedState()
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())
    return state


@pytest.fixture
def wide_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    real = DrCodeApp.resolved_config

    def resolved(self: DrCodeApp, role: AgentRole):  # type: ignore[no-untyped-def]
        cfg = real(self, role)
        if role is AgentRole.COMPACTION:
            return cfg.model_copy(update={"max_context_tokens": 200_000})
        return cfg

    monkeypatch.setattr(DrCodeApp, "resolved_config", resolved)


def _seed_task(
    app: DrCodeApp,
    fill_tokens: int,
    *,
    with_thinking: bool = True,
    kind: TaskKind = TaskKind.RESEARCH,
) -> Any:
    """A task with a persisted transcript (optionally carrying thinking
    parts) whose fill reading is `fill_tokens`."""
    response = ModelResponse(
        parts=(
            [ThinkingPart("deep thought of the previous model"), TextPart("work")]
            if with_thinking
            else [TextPart("work")]
        ),
        usage=RequestUsage(input_tokens=fill_tokens, output_tokens=8),
    )
    task = app.manager.create("docs", "do research", kind=kind)
    task.messages = serialize_messages(
        [ModelRequest(parts=[UserPromptPart("do research")]), response]
    )
    app.manager.storage.save(task)
    return task


# -- strip_thinking_parts --------------------------------------------------------


def _serialized_transcript() -> list[dict[str, Any]]:
    request = ModelRequest(
        parts=[
            UserPromptPart("go"),
            ToolReturnPart(
                tool_name="read_file",
                content="file body",
                tool_call_id="c1",
            ),
        ]
    )
    response = ModelResponse(
        parts=[
            ThinkingPart("reasoning one"),
            ThinkingPart("reasoning two"),
            ToolCallPart(tool_name="edit_file", args={}, tool_call_id="c2"),
        ]
    )
    thinking_only = ModelResponse(parts=[ThinkingPart("pure thought")])
    return serialize_messages([request, response, thinking_only])


def test_strip_thinking_parts_removes_only_thinking() -> None:
    cleaned, removed = strip_thinking_parts(_serialized_transcript())
    assert removed == 3
    kinds = [p.get("part_kind") for msg in cleaned for p in msg.get("parts", [])]
    assert "thinking" not in kinds
    # Everything else survives: the user prompt, the tool result, the call.
    assert kinds.count("user-prompt") == 1
    assert kinds.count("tool-return") == 1
    assert kinds.count("tool-call") == 1


def test_strip_thinking_parts_drops_thinking_only_response() -> None:
    cleaned, removed = strip_thinking_parts(_serialized_transcript())
    # The thinking-only response is dropped entirely (an empty ModelResponse
    # is not API-valid); its thinking counted toward `removed`.
    assert removed == 3
    assert len(cleaned) == 2


def test_strip_thinking_parts_idempotent_and_empty_safe() -> None:
    once, _removed = strip_thinking_parts(_serialized_transcript())
    twice, removed_twice = strip_thinking_parts(once)
    assert removed_twice == 0 and twice == once
    assert strip_thinking_parts([]) == ([], 0)


def test_drop_model_switch_notes_supersedes_old_notes() -> None:
    """A note left behind by an aborted run is replaced by the fresh one
    instead of piling up in the merged resume request."""
    messages = serialize_messages(
        [
            ModelRequest(parts=[UserPromptPart("do research")]),
            ModelRequest(parts=[UserPromptPart(f"{MODEL_SWITCH_NOTE_PREFIX} old")]),
            ModelResponse(parts=[TextPart("work")]),
            ModelRequest(
                parts=[
                    UserPromptPart("next"),
                    UserPromptPart(f"{MODEL_SWITCH_NOTE_PREFIX} stale"),
                ]
            ),
            ModelRequest(parts=[UserPromptPart(f"{MODEL_SWITCH_NOTE_PREFIX} only")]),
        ]
    )
    cleaned = drop_model_switch_notes(messages)
    texts = [
        str(p.get("content"))
        for msg in cleaned
        for p in msg.get("parts", [])
        if p.get("part_kind") == "user-prompt"
    ]
    assert texts == ["do research", "next"]
    # The request that held nothing but a note is dropped, not left empty.
    assert len(cleaned) == 3


# -- the note is compaction machinery ---------------------------------------------


def test_model_switch_note_is_not_a_user_prompt() -> None:
    messages = serialize_messages(
        [
            ModelRequest(parts=[UserPromptPart("original brief")]),
            ModelResponse(parts=[TextPart("done")]),
            ModelRequest(
                parts=[UserPromptPart(f"{MODEL_SWITCH_NOTE_PREFIX} briefing")]
            ),
        ]
    )
    assert all_user_prompts(messages) == ["original brief"]
    # And it is kept out of the verbatim-preserved prefix: it lives in
    # `rest` (summarized away on the next compaction), like a
    # compaction-continuation note.
    preserved, rest = split_original_prompts(messages)
    assert len(preserved) == 1
    assert len(rest) == 2


# -- housekeeping on run start ------------------------------------------------------


async def test_switch_strips_thinking_and_adds_note(
    app: DrCodeApp, scripted: ScriptedState
) -> None:
    """A key mismatch on run start (here: cross-restart — the session dict
    is empty, the persisted key differs): thinking parts are stripped, no
    compaction on a small transcript, the takeover note rides into the
    first request, the new key is stamped."""
    task = _seed_task(app, 100)
    task.last_model_key = "other-provider/other-model"
    app.manager.storage.save(task)
    app.manager.send(task.id, "continue")
    scripted.steps = [text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED, pilot, what="done"
        )

    kinds = [p.get("part_kind") for msg in task.messages for p in msg.get("parts", [])]
    assert "thinking" not in kinds
    assert any(
        p.get("part_kind") == "user-prompt"
        and str(p.get("content", "")).startswith(MODEL_SWITCH_NOTE_PREFIX)
        for msg in task.messages
        for p in msg.get("parts", [])
    )
    # The first (only) model call saw the prompt and the note — never the
    # foreign reasoning.
    assert len(scripted.calls) == 1
    assert "continue" in scripted.calls[0]
    assert MODEL_SWITCH_NOTE_PREFIX in scripted.calls[0]
    # The note names both models: the incoming one knows it inherited the
    # task from a named predecessor, not from itself.
    assert "other-provider/other-model" in scripted.calls[0]
    assert "mock-model" in scripted.calls[0]
    assert "deep thought" not in scripted.calls[0]
    assert task.last_model_key == "mock-model"
    texts = event_texts(task)
    assert any("Model switched: other-provider/other-model" in t for t in texts)
    assert any("Model-switch housekeeping" in t and "thinking part" in t for t in texts)


async def test_switch_with_large_transcript_compacts(
    app: DrCodeApp, scripted: ScriptedState, wide_compaction: None
) -> None:
    """Fill over MODEL_SWITCH_COMPACTION_MIN_TOKENS (20000): the summarizer
    runs before the agent's first turn — on top of the thinking strip."""
    task = _seed_task(app, 30_000)
    task.last_model_key = "other-provider/other-model"
    app.manager.storage.save(task)
    app.manager.send(task.id, "continue")
    scripted.steps = [
        text_response("THE SUMMARY BODY"),
        text_response("final answer"),
    ]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED, pilot, what="done"
        )

    assert len(scripted.calls) == 2
    assert any(is_summary_message(m) for m in task.messages)
    kinds = [p.get("part_kind") for msg in task.messages for p in msg.get("parts", [])]
    assert "thinking" not in kinds
    texts = event_texts(task)
    assert any(
        "Model-switch housekeeping" in t and "history compacted" in t for t in texts
    )


async def test_switch_replaces_stale_note_from_aborted_run(
    app: DrCodeApp, scripted: ScriptedState
) -> None:
    """A note written by a previous switch (its run aborted before the model
    answered) is superseded, not stacked: the incoming model gets exactly
    one takeover briefing, naming the current pair of models."""
    task = _seed_task(app, 100)
    stale = serialize_messages(
        [ModelRequest(parts=[UserPromptPart(f"{MODEL_SWITCH_NOTE_PREFIX} stale")])]
    )
    task.messages = task.messages + stale
    task.last_model_key = "third-provider/third-model"
    app.manager.storage.save(task)
    app.manager.send(task.id, "continue")
    scripted.steps = [text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED, pilot, what="done"
        )
    notes = [
        str(p.get("content"))
        for msg in task.messages
        for p in msg.get("parts", [])
        if p.get("part_kind") == "user-prompt"
        and str(p.get("content", "")).startswith(MODEL_SWITCH_NOTE_PREFIX)
    ]
    assert len(notes) == 1
    assert "third-provider/third-model" in notes[0]
    assert "stale" not in notes[0]


async def test_no_switch_is_a_noop(app: DrCodeApp, scripted: ScriptedState) -> None:
    """Same key as the previous run: nothing is stripped, no note, no
    housekeeping event — the run starts exactly as before."""
    task = _seed_task(app, 100)
    task.last_model_key = "mock-model"
    app.manager.storage.save(task)
    app.manager.send(task.id, "continue")
    scripted.steps = [text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED, pilot, what="done"
        )
    assert len(scripted.calls) == 1
    assert MODEL_SWITCH_NOTE_PREFIX not in scripted.calls[0]
    # Thinking parts untouched (no strip, no compaction on a key match).
    kinds = [p.get("part_kind") for msg in task.messages for p in msg.get("parts", [])]
    assert "thinking" in kinds
    assert not any("housekeeping" in t for t in event_texts(task))


async def test_first_run_stamps_key_silently(
    app: DrCodeApp, scripted: ScriptedState
) -> None:
    """No persisted key yet (first run / old task file): the key is stamped
    without any cleanup or events."""
    task = _seed_task(app, 100)
    assert task.last_model_key is None
    app.manager.send(task.id, "continue")
    scripted.steps = [text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED, pilot, what="done"
        )
    assert task.last_model_key == "mock-model"
    kinds = [p.get("part_kind") for msg in task.messages for p in msg.get("parts", [])]
    assert "thinking" in kinds
    assert not any("Model switched" in t for t in event_texts(task))


async def test_switch_on_interrupted_attempt_resumes_with_note(
    app: DrCodeApp, scripted: ScriptedState
) -> None:
    """The bug's actual shape: the interrupted attempt's request never got a
    response, the app restarted and the task auto-resumes on the NEW model.
    The continuation turn runs over the stripped transcript plus the
    takeover note (both requests merge into one)."""
    task = app.manager.create("docs", "do research", kind=TaskKind.RESEARCH)
    task.pending_input.clear()
    task.messages = serialize_messages(
        [ModelRequest(parts=[UserPromptPart("do research")])]
    )
    task.last_model_key = "other-provider/other-model"
    task.unfinished = True
    app.manager.storage.save(task)
    scripted.steps = [text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED, pilot, what="resumed"
        )
    assert len(scripted.calls) == 1
    assert MODEL_SWITCH_NOTE_PREFIX in scripted.calls[0]
    assert "do research" in scripted.calls[0]
    assert task.last_model_key == "mock-model"


async def test_note_not_appended_when_nothing_to_run(
    app: DrCodeApp, scripted: ScriptedState
) -> None:
    """An unfinished attempt whose exchange actually finished (and nothing
    queued): the run ends without a model call — the note must not turn
    that into a spurious continuation turn."""
    task = _seed_task(app, 100)
    task.pending_input.clear()  # nothing queued: promotion keys off unfinished only
    task.last_model_key = "other-provider/other-model"
    task.unfinished = True
    app.manager.storage.save(task)
    scripted.steps = []  # no model call expected
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(lambda: not app.agent_running, pilot, what="run finished")
    assert not scripted.calls
    assert not any(
        p.get("part_kind") == "user-prompt"
        and str(p.get("content", "")).startswith(MODEL_SWITCH_NOTE_PREFIX)
        for msg in task.messages
        for p in msg.get("parts", [])
    )
    # The key is stamped anyway — the cleanup ran, the run just had
    # nothing to resume.
    assert task.last_model_key == "mock-model"
