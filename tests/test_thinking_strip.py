"""The sliding-window thinking strip (the same-model half of the thinking
discipline — the model-switch strip is the other half).

Covers:
- the pure mechanics (drcode.transcript): the freshness window counts
  THINKING-BEARING exchanges only, the last `keep_recent_turns` keep
  their thinking verbatim, older ones lose it, a thinking-only stale
  response is dropped entirely, `keep_recent_turns <= 0` disables;
- the ≈token accounting (stale_thinking_tokens) the batch gate compares
  against ≈2/3 of the serving role's window (THINKING_PAUSE_FRACTION is
  the shared gate's divisor: 1.5 = 2/3);
- the pipeline wiring: the strip rides the batch eviction pass in
  _maybe_evict_marked (one pass, one event, one credit), fires standalone
  only above ≈2/3 of the window, and the `thinking_keep_recent_turns`
  project setting gates it all (default 0 = off).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

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

from drcode import agent
from drcode.models import TaskKind
from drcode.providers import AgentRole
from drcode.settings import ProjectSettings, ProjectSettingsStore
from drcode.storage import Storage
from drcode.transcript import (
    estimate_tokens,
    serialize_messages,
    stale_thinking_tokens,
    strip_stale_thinking,
)
from drcode.tui.app import DrCodeApp
from tests.network_mocks import event_texts

# -- helpers ---------------------------------------------------------------------


def user_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def thinking_response(thinking: str, text: str) -> ModelResponse:
    return ModelResponse(parts=[ThinkingPart(content=thinking), TextPart(text)])


def thinking_only_response(thinking: str) -> ModelResponse:
    return ModelResponse(parts=[ThinkingPart(content=thinking)])


def thinking_transcript(
    thinkings: list[str],
    *,
    text: str = "ok",
) -> list[dict[str, Any]]:
    """prompt → one response per entry, each carrying its thinking part."""
    messages: list[Any] = [user_msg("work on it")]
    for thinking in thinkings:
        messages.append(thinking_response(thinking, text))
    return serialize_messages(messages)


def thinking_parts_in(messages: list[dict[str, Any]]) -> list[str]:
    return [
        part["content"]
        for msg in messages
        for part in msg.get("parts", [])
        if isinstance(part, dict) and part.get("part_kind") == "thinking"
    ]


def tool_pair(call_id: str, content: str) -> list[Any]:
    """A tool call + its (big) return, as raw pydantic-ai messages."""
    return [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="read_file", args={"path": "a.py"}, tool_call_id=call_id
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="read_file", content=content, tool_call_id=call_id
                )
            ]
        ),
    ]


# -- pure mechanics ---------------------------------------------------------------


def test_strip_keeps_the_recent_window_verbatim() -> None:
    thinkings = [f"t{i}" * 10 for i in range(7)]
    messages = thinking_transcript(thinkings)
    cleaned, removed = strip_stale_thinking(messages, keep_recent_turns=5)
    # The two oldest exchanges fall out of the window of 5.
    assert removed == estimate_tokens(thinkings[0]) + estimate_tokens(thinkings[1])
    assert thinking_parts_in(cleaned) == thinkings[2:]
    # Text parts survive everywhere; the message count is unchanged.
    assert len(cleaned) == len(messages)
    texts = [
        part["content"]
        for msg in cleaned
        for part in msg.get("parts", [])
        if isinstance(part, dict) and part.get("part_kind") == "text"
    ]
    assert texts == ["ok"] * 7


def test_strip_disabled_when_keep_recent_turns_is_zero() -> None:
    thinkings = [f"t{i}" * 10 for i in range(9)]
    messages = thinking_transcript(thinkings)
    cleaned, removed = strip_stale_thinking(messages, keep_recent_turns=0)
    assert cleaned is messages
    assert removed == 0
    assert thinking_parts_in(cleaned) == thinkings


def test_strip_noop_when_everything_is_fresh() -> None:
    messages = thinking_transcript([f"t{i}" * 10 for i in range(3)])
    cleaned, removed = strip_stale_thinking(messages, keep_recent_turns=5)
    assert cleaned is messages
    assert removed == 0


def test_strip_drops_a_thinking_only_stale_response() -> None:
    messages: list[Any] = [user_msg("work on it")]
    # Six thinking-only exchanges (no text part): with a window of 5 the
    # oldest must be dropped ENTIRELY (an empty response is not API-valid),
    # never left behind as an empty part list.
    for i in range(6):
        messages.append(thinking_only_response(f"deep thought {i} " * 5))
    cleaned, removed = strip_stale_thinking(
        serialize_messages(messages), keep_recent_turns=5
    )
    assert len(cleaned) == len(messages) - 1
    assert removed == estimate_tokens("deep thought 0 " * 5)
    assert thinking_parts_in(cleaned) == [f"deep thought {i} " * 5 for i in range(1, 6)]


def test_freshness_counts_thinking_bearing_exchanges_only() -> None:
    messages: list[Any] = [user_msg("work on it")]
    for i in range(7):
        messages.append(thinking_response(f"think {i} " * 10, "working"))
        # A plain text exchange between: carries no thinking, consumes no
        # freshness — the window still spans 5 THINKING-BEARING exchanges.
        messages.append(ModelResponse(parts=[TextPart("still working")]))
    cleaned, _ = strip_stale_thinking(serialize_messages(messages), keep_recent_turns=5)
    kept = thinking_parts_in(cleaned)
    # The two oldest thinking exchanges are stale even though five plain
    # exchanges follow them.
    assert kept == [f"think {i} " * 10 for i in range(2, 7)]


def test_stale_thinking_tokens_matches_the_strip_result() -> None:
    thinkings = [f"t{i}" * 10 for i in range(7)]
    messages = thinking_transcript(thinkings)
    expected = estimate_tokens(thinkings[0]) + estimate_tokens(thinkings[1])
    assert stale_thinking_tokens(messages, keep_recent_turns=5) == expected
    assert stale_thinking_tokens(messages, keep_recent_turns=0) == 0


def test_stale_thinking_selection_is_idempotent() -> None:
    messages = thinking_transcript([f"t{i}" * 10 for i in range(7)])
    cleaned, _ = strip_stale_thinking(messages, keep_recent_turns=5)
    again, removed = strip_stale_thinking(cleaned, keep_recent_turns=5)
    assert again is cleaned
    assert removed == 0


def test_stale_thinking_selection_walks_responses_only() -> None:
    # A request-shaped message with a thinking-like part never enters the
    # window counting: the criterion walks `kind: "response"` messages only.
    messages = thinking_transcript(["x" * 800 for _ in range(7)])
    messages.append(
        {"kind": "request", "parts": [{"part_kind": "thinking", "content": "x" * 800}]}
    )
    assert stale_thinking_tokens(messages, keep_recent_turns=5) == (
        2 * estimate_tokens("x" * 800)
    )


def test_lazy_export_resolves() -> None:
    assert agent.strip_stale_thinking is strip_stale_thinking
    assert agent.stale_thinking_tokens is stale_thinking_tokens


# -- project setting --------------------------------------------------------------


def test_setting_defaults_to_zero() -> None:
    assert ProjectSettings().thinking_keep_recent_turns == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (5, 5),
        (0, 0),
        (-3, 0),
        (True, 0),  # bool is junk, not 1
        ("3", 0),  # a string is junk, not coerced
        (None, 0),
        (2.9, 2),
    ],
)
def test_setting_validator_coerces_junk(raw: object, expected: int) -> None:
    settings = ProjectSettings.model_validate({"thinking_keep_recent_turns": raw})
    assert settings.thinking_keep_recent_turns == expected


def test_setting_round_trips_through_the_store(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    store.save(ProjectSettings(thinking_keep_recent_turns=3))
    assert store.load().thinking_keep_recent_turns == 3
    store.save(ProjectSettings(thinking_keep_recent_turns=0))
    assert store.load().thinking_keep_recent_turns == 0


# -- pipeline wiring --------------------------------------------------------------


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


@pytest.fixture
def app_small_ctx(
    project_root: Path, configured_env: None, monkeypatch: pytest.MonkeyPatch
) -> DrCodeApp:
    # The strip is OFF by default (thinking_keep_recent_turns=0) — the
    # pipeline wiring tests opt in explicitly; the small window makes the
    # ≈2/3 pause threshold easy to cross.
    monkeypatch.setenv("DRCODE_MAX_CONTEXT_TOKENS", "1000")
    app = DrCodeApp(project_root)
    app.settings.thinking_keep_recent_turns = 5
    return app


def _seed_thinking_task(app: DrCodeApp, thinkings: list[str], *, usage: int = 0) -> Any:
    # No pending input (empty brief): the mount-time _advance() must leave
    # the task queued, so the gates are exercised by direct calls.
    task = app.manager.create("docs", "", kind=TaskKind.RESEARCH)
    messages: list[Any] = [user_msg("work on it")]
    for thinking in thinkings:
        messages.append(thinking_response(thinking, "ok"))
    if usage:
        messages.append(
            ModelResponse(
                parts=[TextPart("partial work")],
                usage=RequestUsage(input_tokens=usage, output_tokens=8),
            )
        )
    task.messages = serialize_messages(messages)
    app.manager.storage.save(task)
    return task


async def test_standalone_strip_fires_above_the_pause_threshold(
    app_small_ctx: DrCodeApp,
) -> None:
    # 7 thinking-bearing exchanges of ≈400 tokens each: the 2 stale ones
    # hold ≈800 tokens ≥ 2/3 of 1000 — the strip earns its own batch.
    thinkings = ["x" * 1600 for _ in range(7)]
    task = _seed_thinking_task(app_small_ctx, thinkings)
    assert app_small_ctx._stale_thinking_due(task)
    assert app_small_ctx._stale_thinking_tokens(task) == 2 * estimate_tokens("x" * 1600)

    async with app_small_ctx.run_test() as pilot:
        await pilot.pause()
        total = app_small_ctx._maybe_evict_marked(task)
        await pilot.pause()

    assert total == 2 * estimate_tokens("x" * 1600)
    assert thinking_parts_in(task.messages) == ["x" * 1600] * 5
    assert task.marked_unneeded == []
    assert any("stripped stale thinking" in t for t in event_texts(task))
    # The strip credit is deposited for the fill gates.
    assert app_small_ctx._evicted_ctx_tokens[task.id] == 2 * estimate_tokens("x" * 1600)
    # Persisted, not just in memory.
    reloaded = {t.id: t for t in Storage(app_small_ctx.project_root).load_all()}[
        task.id
    ]
    assert thinking_parts_in(reloaded.messages) == ["x" * 1600] * 5


async def test_strip_below_the_pause_threshold_is_noop(
    app_small_ctx: DrCodeApp,
) -> None:
    # 2 stale exchanges of ≈25 tokens: 50 * 1.5 < 1000 — no standalone batch.
    thinkings = ["y" * 100 for _ in range(7)]
    task = _seed_thinking_task(app_small_ctx, thinkings)
    assert not app_small_ctx._stale_thinking_due(task)
    assert app_small_ctx._maybe_evict_marked(task) == 0
    assert thinking_parts_in(task.messages) == ["y" * 100] * 7


async def test_strip_rides_the_eviction_batch(
    app_small_ctx: DrCodeApp,
) -> None:
    # A big marked tool return (the marks gate fires at ≈1/3 of the window)
    # PLUS stale thinking below the standalone threshold: the strip rides
    # the eviction batch for free — one pass, one event, one credit.
    task = _seed_thinking_task(app_small_ctx, ["y" * 100 for _ in range(7)])
    task.messages = task.messages + serialize_messages(tool_pair("c1", "z" * 6000))
    task.marked_unneeded = ["c1"]
    app_small_ctx.manager.storage.save(task)
    assert app_small_ctx._eviction_due(task)
    assert not app_small_ctx._stale_thinking_due(task)

    async with app_small_ctx.run_test() as pilot:
        await pilot.pause()
        total = app_small_ctx._maybe_evict_marked(task)
        await pilot.pause()

    expected = estimate_tokens("z" * 6000) + 2 * estimate_tokens("y" * 100)
    assert total == expected
    assert app_small_ctx._evicted_ctx_tokens[task.id] == expected
    assert thinking_parts_in(task.messages) == ["y" * 100] * 5
    event = " | ".join(event_texts(task))
    assert "collapsed 1 marked tool result(s) and stripped stale thinking" in event


async def test_force_strip_applies_everything_stale(
    app_small_ctx: DrCodeApp,
) -> None:
    # force=True (the pre-compaction eviction): the strip applies whatever
    # is stale, below the standalone threshold included.
    thinkings = ["y" * 100 for _ in range(7)]
    task = _seed_thinking_task(app_small_ctx, thinkings)
    assert not app_small_ctx._stale_thinking_due(task)

    async with app_small_ctx.run_test() as pilot:
        await pilot.pause()
        total = app_small_ctx._maybe_evict_marked(task, force=True)
        await pilot.pause()

    assert total == 2 * estimate_tokens("y" * 100)
    assert thinking_parts_in(task.messages) == ["y" * 100] * 5


async def test_setting_zero_disables_the_strip(
    project_root: Path, configured_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DRCODE_MAX_CONTEXT_TOKENS", "1000")
    ProjectSettingsStore(project_root).save(
        ProjectSettings(thinking_keep_recent_turns=0)
    )
    app = DrCodeApp(project_root)
    thinkings = ["x" * 800 for _ in range(7)]
    task = _seed_thinking_task(app, thinkings)
    assert app._stale_thinking_tokens(task) == 0
    assert not app._stale_thinking_due(task)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._maybe_evict_marked(task, force=True) == 0
        await pilot.pause()

    # Nothing but the model-switch housekeeping touches thinking then.
    assert thinking_parts_in(task.messages) == thinkings
    assert app._evicted_ctx_tokens.get(task.id, 0) == 0
    assert not any("stripped stale thinking" in t for t in event_texts(task))


def test_stale_thinking_due_needs_a_window(
    app_small_ctx: DrCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No window knowledge (max_context_tokens 0): the gate stays off
    # regardless of the stale volume.
    task = _seed_thinking_task(app_small_ctx, ["x" * 800 for _ in range(7)])
    real = DrCodeApp.resolved_config

    def resolved(self: DrCodeApp, role: AgentRole):  # type: ignore[no-untyped-def]
        return real(self, role).model_copy(update={"max_context_tokens": 0})

    monkeypatch.setattr(DrCodeApp, "resolved_config", resolved)
    assert app_small_ctx._stale_thinking_tokens(task) > 0
    assert not app_small_ctx._stale_thinking_due(task)


async def test_setting_reaches_the_batch_pass(
    app_small_ctx: DrCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A window of 1 strips everything but the freshest thinking-bearing
    # exchange — the setting flows into strip_stale_thinking, not just the
    # gate.
    task = _seed_thinking_task(app_small_ctx, ["x" * 800 for _ in range(7)])
    app_small_ctx.settings.thinking_keep_recent_turns = 1

    async with app_small_ctx.run_test() as pilot:
        await pilot.pause()
        total = app_small_ctx._maybe_evict_marked(task, force=True)
        await pilot.pause()

    assert total == 6 * estimate_tokens("x" * 800)
    assert thinking_parts_in(task.messages) == ["x" * 800]
    app_small_ctx.settings.thinking_keep_recent_turns = 5
    cast(Any, task).marked_unneeded = []
