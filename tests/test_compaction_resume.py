"""Pre-run forced compaction of a cold-resume transcript (project setting
`compaction_resume_min_tokens`, see core/pipeline_compaction.

When a task is re-activated after its previous run ended ≥ 3 minutes ago
(the provider's prefix cache has burnt out) and its dev/research
transcript fill exceeds the token threshold, the transcript is
force-compacted BEFORE the first turn. Every gate below asserts the
negative branches too: the ordinary threshold gate behavior must be
untouched when any condition fails."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from drcode import agent_model
from drcode.agent import is_summary_message, serialize_messages
from drcode.models import TaskKind, TaskStatus
from drcode.providers import AgentRole
from drcode.tui.app import DrCodeApp
from tests.network_mocks import (
    ScriptedState,
    api_error,
    event_texts,
    function_model,
    text_response,
    wait_until,
)

# -- fixtures ---------------------------------------------------------------------


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured (but never contacted) model endpoint: the model layer
    is mocked — a real dial must never happen."""
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


@pytest.fixture
def app(project_root: Path, configured_env: None) -> DrCodeApp:
    return DrCodeApp(project_root)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedState:
    """Every agent drcode builds (coding, research, compaction alike) is a
    FunctionModel consuming `state.steps`."""
    state = ScriptedState()
    monkeypatch.setattr(
        agent_model, "build_model", lambda cfg, announce_wait=None: state.model()
    )
    return state


@pytest.fixture
def wide_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """The COMPACTION role resolves with a huge window: the rendered prompt
    fits, so these tests exercise the LLM summarizer path."""
    real = DrCodeApp.resolved_config

    def resolved(self: DrCodeApp, role: AgentRole):  # type: ignore[no-untyped-def]
        cfg = real(self, role)
        if role is AgentRole.COMPACTION:
            return cfg.model_copy(update={"max_context_tokens": 200_000})
        return cfg

    monkeypatch.setattr(DrCodeApp, "resolved_config", resolved)


def _seed_task(
    app: DrCodeApp, fill_tokens: int, kind: TaskKind = TaskKind.RESEARCH
) -> Any:
    """A task whose persisted transcript reports a context fill of
    `fill_tokens` (the usage numbers are arbitrary — the fill reading is
    taken from the last response's usage, no tokenizer involved)."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )
    from pydantic_ai.usage import RequestUsage

    task = app.manager.create("docs", "do research", kind=kind)
    response = ModelResponse(
        parts=[TextPart("partial work")],
        usage=RequestUsage(input_tokens=fill_tokens, output_tokens=8),
    )
    task.messages = serialize_messages(
        [ModelRequest(parts=[UserPromptPart("do research")]), response]
    )
    app.manager.storage.save(task)
    return task


def _idle(app: DrCodeApp, task: Any, seconds: float) -> None:
    """Backdate the previous run's end so the gate sees the idle gap."""
    task.last_run_end = datetime.now(UTC) - timedelta(seconds=seconds)
    app.manager.storage.save(task)


# -- the gate fires ----------------------------------------------------------------


async def test_resume_compaction_fires_before_first_turn(
    app: DrCodeApp, scripted: ScriptedState, wide_compaction: None
) -> None:
    """Fill above the threshold + idle ≥ 3 min: the summarizer runs BEFORE
    the agent's first turn, and the run then proceeds over the compacted
    transcript."""
    app.settings.compaction_resume_min_tokens = 25_000
    task = _seed_task(app, 30_000)
    _idle(app, task, 200)
    app.manager.send(task.id, "continue")
    scripted.steps = [
        text_response("THE SUMMARY BODY"),
        text_response("final answer"),
    ]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED,
            pilot,
            what="research task completed",
        )

    # First model call was the summarizer (never the agent turn), second
    # the agent's own prompt. `calls` records ALL user-prompt text of a
    # request (history + new prompt), so substring assertions.
    assert len(scripted.calls) == 2
    assert "continue" in scripted.calls[1]
    assert "continue" not in scripted.calls[0]
    assert "partial work" in scripted.calls[0]  # summarizer saw the transcript
    # The transcript the agent run resumed over contains the summary.
    assert any(is_summary_message(m) for m in task.messages)
    # Service events name the start and the outcome (with both fills).
    texts = event_texts(task)
    assert any("compacting history before the run" in t for t in texts)
    assert any("pre-run compaction: ≈" in t and "→" in t for t in texts)
    # The run's own end re-stamped the transcript-write stamp.
    assert task.last_run_end is not None


# -- the gate stays silent ----------------------------------------------------------


@pytest.mark.parametrize("threshold", [0, 19_999])
async def test_disabled_below_minimum_tokens(
    app: DrCodeApp,
    scripted: ScriptedState,
    monkeypatch: pytest.MonkeyPatch,
    threshold: int,
) -> None:
    """0 (the default) and anything below 20000 keep the feature off —
    even with a huge fill and a long idle."""
    app.settings.compaction_resume_min_tokens = threshold
    task = _seed_task(app, 30_000)
    task.last_run_end = datetime.now(UTC) - timedelta(seconds=200)
    app.manager.storage.save(task)
    app.manager.send(task.id, "continue")
    scripted.steps = [text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED,
            pilot,
            what=f"task completed (threshold={threshold})",
        )
    # Exactly one model call — the agent's own turn; no summary.
    assert len(scripted.calls) == 1 and scripted.calls[0].endswith("continue"), (
        threshold
    )
    assert not any(is_summary_message(m) for m in task.messages)
    assert not any("pre-run compaction" in t for t in event_texts(task))


async def test_no_compaction_below_threshold_fill(
    app: DrCodeApp, scripted: ScriptedState, wide_compaction: None
) -> None:
    """Feature on, idle long — but the fill is at/below the threshold: no
    pre-run compaction."""
    app.settings.compaction_resume_min_tokens = 25_000
    task = _seed_task(app, 20_000)
    _idle(app, task, 200)
    app.manager.send(task.id, "continue")
    scripted.steps = [text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED, pilot, what="done"
        )
    assert len(scripted.calls) == 1 and scripted.calls[0].endswith("continue")
    assert not any(is_summary_message(m) for m in task.messages)


async def test_no_compaction_on_short_idle(
    app: DrCodeApp, scripted: ScriptedState, wide_compaction: None
) -> None:
    """The pause is under 3 minutes: the cache may still be warm — no
    pre-run compaction, even with a huge fill."""
    app.settings.compaction_resume_min_tokens = 25_000
    task = _seed_task(app, 30_000)
    _idle(app, task, 60)
    app.manager.send(task.id, "continue")
    scripted.steps = [text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED, pilot, what="done"
        )
    assert len(scripted.calls) == 1 and scripted.calls[0].endswith("continue")
    assert not any(is_summary_message(m) for m in task.messages)


async def test_no_compaction_on_first_activation(
    app: DrCodeApp, scripted: ScriptedState, wide_compaction: None
) -> None:
    """No previous run ever wrote the transcript (`last_run_end` None):
    nothing to warm up — no pre-run compaction."""
    app.settings.compaction_resume_min_tokens = 25_000
    task = _seed_task(app, 30_000)
    assert task.last_run_end is None
    app.manager.send(task.id, "continue")
    scripted.steps = [text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED, pilot, what="done"
        )
    assert len(scripted.calls) == 1 and scripted.calls[0].endswith("continue")
    assert not any(is_summary_message(m) for m in task.messages)


# -- failure handling -----------------------------------------------------------------


async def test_summarizer_failure_degrades_to_uncompressed_run(
    app: DrCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken summarizer is never fatal: the gate logs an event and the
    transcript stays untouched; the NEXT activation does not retry the
    futile compaction (`_compaction_failed` latch)."""

    def failing(messages: list, info: Any) -> Any:
        raise api_error("Connection error.")

    monkeypatch.setattr(
        agent_model,
        "build_model",
        lambda cfg, announce_wait=None: function_model(failing),
    )
    app.settings.compaction_resume_min_tokens = 25_000
    task = _seed_task(app, 30_000)
    _idle(app, task, 200)

    async with app.run_test() as pilot:
        await pilot.pause()
        await app._maybe_compact_after_idle(task)
        await pilot.pause()

        texts = event_texts(task)
        assert any("Pre-run compaction failed" in t for t in texts)
        assert not any(is_summary_message(m) for m in task.messages)
        assert task.id in app._compaction_failed

        # The degraded latch: a second activation attempt is a no-op.
        events_before = len(task.events)
        await app._maybe_compact_after_idle(task)
        await pilot.pause()
        assert len(task.events) == events_before
        assert not any(is_summary_message(m) for m in task.messages)


# -- role resolution --------------------------------------------------------------------


async def test_research_task_compacts_with_research_role(
    app: DrCodeApp,
    scripted: ScriptedState,
    wide_compaction: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A research task (out of awaiting-clarification — the same run_agent
    path) compacts against the RESEARCH role's resolved config."""
    captured: dict[str, AgentRole] = {}
    real = DrCodeApp._compact_transcript

    async def spy(
        self: DrCodeApp,
        messages: list,
        *,
        summary_role: AgentRole,
        task: Any,
        what: str,
        **kwargs: Any,
    ):
        captured["role"] = summary_role
        return await real(
            self, messages, summary_role=summary_role, task=task, what=what, **kwargs
        )

    monkeypatch.setattr(DrCodeApp, "_compact_transcript", spy)
    app.settings.compaction_resume_min_tokens = 25_000
    task = _seed_task(app, 30_000, kind=TaskKind.RESEARCH)
    _idle(app, task, 200)
    app.manager.send(task.id, "continue")
    scripted.steps = [text_response("SUMMARY"), text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED, pilot, what="done"
        )

    assert captured.get("role") is AgentRole.RESEARCH
    assert any(is_summary_message(m) for m in task.messages)


# -- the run-end stamp --------------------------------------------------------------------


async def test_run_stamps_last_run_end(app: DrCodeApp, scripted: ScriptedState) -> None:
    """Every run exit path stamps `Task.last_run_end` in the worker's
    finally (here: the normal end) — the value the next activation's gate
    measures the idle gap against."""
    task = _seed_task(app, 100)
    assert task.last_run_end is None
    app.manager.send(task.id, "continue")
    scripted.steps = [text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED, pilot, what="done"
        )
    assert task.last_run_end is not None
    assert (datetime.now(UTC) - task.last_run_end).total_seconds() < 120


# -- settings plumbing ---------------------------------------------------------------------


def test_settings_validator_clamps_junk() -> None:
    """Junk in settings.json degrades to the disabled default instead of
    invalidating the whole file; negatives clamp to 0."""
    from drcode.settings import ProjectSettings

    assert ProjectSettings.model_validate({}).compaction_resume_min_tokens == 0
    assert (
        ProjectSettings.model_validate(
            {"compaction_resume_min_tokens": 25_000}
        ).compaction_resume_min_tokens
        == 25_000
    )
    assert (
        ProjectSettings.model_validate(
            {"compaction_resume_min_tokens": -5}
        ).compaction_resume_min_tokens
        == 0
    )
    assert (
        ProjectSettings.model_validate(
            {"compaction_resume_min_tokens": "garbage"}
        ).compaction_resume_min_tokens
        == 0
    )
