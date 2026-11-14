"""Selective context eviction driven by the agent's mark_unneeded marks.

Covers the four layers of the mechanism:
- the pure eviction mechanics (drcode.eviction): stub replacement, the
  intact tool-call/tool-return pairs, ≈token accounting;
- the mark_unneeded tool body (selectors, ModelRetry on a bad target,
  persistence through the task metadata);
- the pipeline gates: the batch eviction at 1/3 of the window, the
  pre-compaction eviction (spares the summarizer or lightens its input),
  and the live-run pause→evict→resume cycle;
- the AUTOMATIC tool result clearing (the age criterion, no marks):
  `auto_evictable_ids` selection, the freshness anchor, and its wiring
  into the same batch the marks go through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from drcode import agent_model, eviction
from drcode.agent_tools import (
    _fmt_approx_tokens,
    _mark_unneeded,
    eviction_dynamic_instructions,
)
from drcode.compaction import (
    _atomic_groups,
    chunk_history_for_summary,
    mechanical_compaction_messages,
    render_history_blocks,
)
from drcode.models import TaskKind, TaskStatus
from drcode.providers import AgentRole
from drcode.storage import Storage
from drcode.transcript import (
    estimate_tokens,
    load_messages,
    resumable_history,
    sanitized_messages,
    serialize_messages,
)
from drcode.tui.app import DrCodeApp
from tests.network_mocks import (
    ScriptedState,
    event_texts,
    text_response,
    tool_response,
    wait_until,
)

# -- helpers ---------------------------------------------------------------------


def user_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def tool_call_msg(call_id: str, tool: str, args: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name=tool, args=args, tool_call_id=call_id)]
    )


def tool_return_msg(call_id: str, tool: str, content: Any) -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name=tool, content=content, tool_call_id=call_id)]
    )


def big_return_transcript(content: str, usage: int = 0) -> list[dict[str, Any]]:
    """prompt → tool call c1 → big tool return → (optionally) a usage-bearing
    response. The shape every eviction test seeds."""
    messages: list[Any] = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "src/big.py"}),
        tool_return_msg("c1", "read_file", content),
    ]
    if usage:
        messages.append(
            ModelResponse(
                parts=[TextPart("partial work")],
                usage=RequestUsage(input_tokens=usage, output_tokens=8),
            )
        )
    return serialize_messages(messages)


def stubs_in(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        part
        for msg in messages
        for part in msg.get("parts", [])
        if isinstance(part, dict) and eviction.is_evicted_stub(part.get("content"))
    ]


def returns_in(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        part["tool_call_id"]: part
        for msg in messages
        for part in msg.get("parts", [])
        if isinstance(part, dict) and part.get("part_kind") == "tool-return"
    }


# -- pure eviction mechanics -------------------------------------------------------


def test_stub_format_and_args_cap() -> None:
    stub = eviction.evicted_stub(
        "read_file", {"path": "src/" + "x" * 300 + ".py", "offset": 1}
    )
    assert stub.startswith("[tool result evicted: read_file(")
    assert "re-run the tool if needed]" in stub
    # The args summary is capped: the stub stays a short line.
    assert len(stub) < 300


def test_evict_marked_replaces_content_keeps_pair() -> None:
    data = big_return_transcript("y" * 6000)
    result = eviction.evict_marked(data, ["c1"])

    assert result.evicted == ["c1"]
    assert result.missing == []
    assert result.new_messages is not data  # input never mutated in place
    (stub_part,) = stubs_in(result.new_messages)
    assert stub_part["tool_call_id"] == "c1"
    assert stub_part["tool_name"] == "read_file"
    # The sibling tool-call part is untouched — the pair survives.
    calls = [
        part
        for msg in result.new_messages
        for part in msg.get("parts", [])
        if isinstance(part, dict) and part.get("part_kind") == "tool-call"
    ]
    assert len(calls) == 1
    assert calls[0]["tool_name"] == "read_file"
    assert calls[0]["args"] == {"path": "src/big.py"}
    assert calls[0]["tool_call_id"] == "c1"
    # The transcript stays API-valid and resumable.
    restored = load_messages(result.new_messages)
    assert resumable_history(result.new_messages) == restored


def test_evict_marked_missing_and_stub_are_noops() -> None:
    data = big_return_transcript("y" * 100)
    result = eviction.evict_marked(data, ["ghost"])
    assert result.evicted == []
    assert result.missing == ["ghost"]
    assert result.new_messages == data  # nothing to do — transcript unchanged

    once = eviction.evict_marked(data, ["c1"]).new_messages
    again = eviction.evict_marked(once, ["c1"])
    assert again.evicted == []
    assert again.new_messages == once  # an already-stubbed return is left alone


def test_marked_tokens_counts_pending_only() -> None:
    data = big_return_transcript("y" * 4000)
    assert eviction.marked_tokens(data, ["c1"]) == estimate_tokens("y" * 4000)
    assert eviction.marked_tokens(data, ["ghost"]) == 0
    stubbed = eviction.evict_marked(data, ["c1"]).new_messages
    assert eviction.marked_tokens(stubbed, ["c1"]) == 0


def test_evict_marked_matches_call_in_other_message() -> None:
    """The stub is built from the sibling tool-call's args even when the
    call and the return live in different messages (they always do)."""
    data = big_return_transcript("y" * 800)
    result = eviction.evict_marked(data, ["c1"])
    (stub_part,) = stubs_in(result.new_messages)
    assert "read_file" in stub_part["content"]
    assert "src/big.py" in stub_part["content"]


# -- compaction rendering works with stubs ------------------------------------------


def test_render_keeps_call_and_stub_adjacent() -> None:
    """The summarizer's chunker treats a call and its (stubbed) return as
    one atomic group — the pair is never separated in a map slice."""
    data = big_return_transcript("y" * 5000)
    stubbed = eviction.evict_marked(data, ["c1"]).new_messages
    blocks = render_history_blocks(stubbed)
    groups = _atomic_groups(blocks)
    # The user prompt may be glued to its answer's first group: find the
    # group holding the tool call and check its result stays attached.
    call_group = next(g for g in groups if any(k == "tool-call" for k, _ in g))
    kinds = [kind for kind, _ in call_group]
    call_idx = kinds.index("tool-call")
    assert kinds[call_idx + 1] == "tool-result"
    assert "tool result evicted" in call_group[call_idx + 1][1]
    # And the whole transcript renders with the stub in place of the body.
    rendered = "\n\n".join(text for _, text in blocks)
    assert "y" * 100 not in rendered
    assert "tool result evicted" in rendered


def test_chunking_and_mechanical_floor_survive_stubs() -> None:
    data = big_return_transcript("y" * 5000, usage=900)
    stubbed = eviction.evict_marked(data, ["c1"]).new_messages
    slices = chunk_history_for_summary(stubbed, budget_tokens=2000)
    assert slices
    # The mechanical floor keeps the pair: the stub return is not orphaned.
    trimmed = mechanical_compaction_messages(stubbed, 100_000)
    assert returns_in(trimmed)["c1"]["tool_call_id"] == "c1"
    assert sanitized_messages(stubbed) == stubbed


# -- mark_unneeded tool body ---------------------------------------------------------


class FakeDeps:
    def __init__(self, manager: Any, task_id: str) -> None:
        self.manager = manager
        self.task_id = task_id


class FakeCtx:
    """Just what _mark_unneeded reads: the run transcript and the deps."""

    def __init__(self, messages: list[Any], deps: Any) -> None:
        self.messages = messages
        self.deps = deps


def test_mark_unneeded_defaults_to_last_result(manager: Any) -> None:
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "a.py"}),
        tool_return_msg("c1", "read_file", "content a"),
        tool_call_msg("c2", "grep_search", {"pattern": "x"}),
        tool_return_msg("c2", "grep_search", ["hit"]),
        tool_call_msg("c3", "mark_unneeded", {}),
    ]
    result = _mark_unneeded(cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))))
    assert task.marked_unneeded == ["c2"]
    assert "no longer needed" in result
    assert "Marked:" in result
    assert "total pending" in result
    assert "Nothing is removed right now" in result
    assert "1/3 of the context window" in result


def test_mark_unneeded_reports_token_figures(manager: Any) -> None:
    """The answer carries the ≈estimate of what THIS call marked and the
    ≈total of ALL pending marks — chars/4, the same accounting the
    pipeline gate uses."""
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    big_a, big_b = "y" * 4000, "z" * 800
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "a.py"}),
        tool_return_msg("c1", "read_file", big_a),
        tool_call_msg("c2", "grep_search", {"pattern": "x"}),
        tool_return_msg("c2", "grep_search", big_b),
        tool_call_msg("c3", "mark_unneeded", {}),
    ]
    ctx = cast(Any, FakeCtx(messages, FakeDeps(manager, task.id)))
    result = _mark_unneeded(ctx)
    # The most recent result (c2) was marked; the total is just it so far.
    assert f"Marked: ~{_fmt_approx_tokens(estimate_tokens(big_b))} ≈tokens" in result
    assert f"total pending ≈{_fmt_approx_tokens(estimate_tokens(big_b))}" in result

    # A second call marks c1 too: the new figure covers only c1, the
    # total covers both pending marks.
    ctx.messages = [
        *messages,
        tool_return_msg("c3", "mark_unneeded", result),
        tool_call_msg("c4", "mark_unneeded", {}),
    ]
    result2 = _mark_unneeded(ctx, paths=["a.py"])
    total = estimate_tokens(big_a) + estimate_tokens(big_b)
    assert f"Marked: ~{_fmt_approx_tokens(estimate_tokens(big_a))} ≈tokens" in result2
    assert f"total pending ≈{_fmt_approx_tokens(total)}" in result2


def test_mark_unneeded_pending_figures_skip_stubs(manager: Any) -> None:
    """Already-evicted marks weigh zero in the total — the figure mirrors
    `eviction.marked_tokens`, not the raw length of the mark list."""
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    big = "y" * 4000
    stub = eviction.evicted_stub("read_file", {"path": "a.py"})
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "a.py"}),
        tool_return_msg("c1", "read_file", stub),  # already evicted earlier
        tool_call_msg("c2", "read_file", {"path": "b.py"}),
        tool_return_msg("c2", "read_file", big),
        tool_call_msg("c3", "mark_unneeded", {}),
    ]
    task.marked_unneeded = ["c1"]
    result = _mark_unneeded(cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))))
    assert task.marked_unneeded == ["c1", "c2"]
    assert f"Marked: ~{_fmt_approx_tokens(estimate_tokens(big))} ≈tokens" in result
    assert f"total pending ≈{_fmt_approx_tokens(estimate_tokens(big))}" in result


def test_fmt_approx_tokens() -> None:
    assert _fmt_approx_tokens(0) == "0"
    assert _fmt_approx_tokens(420) == "420"
    assert _fmt_approx_tokens(4200) == "4.2k"
    assert _fmt_approx_tokens(18000) == "18k"


def test_mark_unneeded_path_selector(manager: Any) -> None:
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "a.py"}),
        tool_return_msg("c1", "read_file", "content a"),
        tool_call_msg("c2", "read_file", {"path": "b.py", "offset": 5}),
        tool_return_msg("c2", "read_file", "content b"),
    ]
    _mark_unneeded(
        cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))), paths=["a.py"]
    )
    assert task.marked_unneeded == ["c1"]


def test_mark_unneeded_unknown_path_retries(manager: Any) -> None:
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "a.py"}),
        tool_return_msg("c1", "read_file", "content a"),
    ]
    with pytest.raises(ValueError, match="Nothing matched"):
        _mark_unneeded(
            cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))),
            paths=["zzz.py"],
        )
    # No completed tool result at all — nothing to mark.
    with pytest.raises(ValueError, match="no completed tool result"):
        _mark_unneeded(
            cast(Any, FakeCtx([user_msg("read things")], FakeDeps(manager, task.id)))
        )
    assert task.marked_unneeded == []


def test_mark_unneeded_never_marks_itself(manager: Any) -> None:
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "mark_unneeded", {}),
        tool_return_msg("c1", "mark_unneeded", "Marked: ..."),
        tool_call_msg("c2", "mark_unneeded", {}),
    ]
    with pytest.raises(ValueError):
        _mark_unneeded(cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))))


def test_mark_unneeded_survives_save_load(project_root: Path) -> None:
    from drcode.task_manager import TaskManager

    manager = TaskManager(project_root)
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "a.py"}),
        tool_return_msg("c1", "read_file", "content a"),
    ]
    _mark_unneeded(cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))))
    assert task.marked_unneeded == ["c1"]
    # A fresh store over the same project reads the mark back.
    tasks = {t.id: t for t in Storage(project_root).load_all()}
    assert tasks[task.id].marked_unneeded == ["c1"]


def test_mark_unneeded_batch_paths(manager: Any) -> None:
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "a.py"}),
        tool_return_msg("c1", "read_file", "content a"),
        tool_call_msg("c2", "read_file", {"path": "b.py"}),
        tool_return_msg("c2", "read_file", "content b"),
        tool_call_msg("c3", "read_file", {"path": "a.py"}),
        tool_return_msg("c3", "read_file", "content a again"),
    ]
    result = _mark_unneeded(
        cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))),
        paths=["a.py", "b.py"],
    )
    # The MOST RECENT completed call per path; insertion order preserved.
    assert task.marked_unneeded == ["c3", "c2"]
    assert "read_file(path='a.py')" in result
    assert "read_file(path='b.py')" in result


def test_mark_unneeded_batch_partial_miss_lists_errors(manager: Any) -> None:
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "a.py"}),
        tool_return_msg("c1", "read_file", "content a"),
    ]
    result = _mark_unneeded(
        cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))),
        paths=["a.py", "zzz.py"],
    )
    # The hit is kept, the miss is reported — the call does not fail.
    assert task.marked_unneeded == ["c1"]
    assert "read_file(path='a.py')" in result
    assert "Not marked" in result
    assert "zzz.py" in result


def test_mark_unneeded_batch_total_miss_raises(manager: Any) -> None:
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "a.py"}),
        tool_return_msg("c1", "read_file", "content a"),
    ]
    with pytest.raises(ValueError, match="zzz.py"):
        _mark_unneeded(
            cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))),
            paths=["zzz.py", "yyy.py"],
        )
    assert task.marked_unneeded == []


def test_mark_unneeded_tool_name_selector(manager: Any) -> None:
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "grep_search", {"pattern": "x"}),
        tool_return_msg("c1", "grep_search", ["hit1"]),
        tool_call_msg("c2", "read_file", {"path": "a.py"}),
        tool_return_msg("c2", "read_file", "content a"),
        tool_call_msg("c3", "grep_search", {"pattern": "y"}),
        tool_return_msg("c3", "grep_search", ["hit2"]),
        tool_call_msg("c4", "grep_search", {"pattern": "z"}),  # no return yet
    ]
    result = _mark_unneeded(
        cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))),
        tool_name="grep_search",
    )
    # Every COMPLETED grep result, in chronological order; the pending
    # c4 is untouched, the read_file result is untouched.
    assert task.marked_unneeded == ["c1", "c3"]
    assert "grep_search(pattern='x')" in result
    assert "grep_search(pattern='y')" in result


def test_mark_unneeded_tool_name_and_paths_combined(manager: Any) -> None:
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "grep_search", {"pattern": "x"}),
        tool_return_msg("c1", "grep_search", ["hit1"]),
        tool_call_msg("c2", "read_file", {"path": "a.py"}),
        tool_return_msg("c2", "read_file", "content a"),
        tool_call_msg("c3", "read_file", {"path": "b.py"}),
        tool_return_msg("c3", "read_file", "content b"),
    ]
    result = _mark_unneeded(
        cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))),
        paths=["a.py"],
        tool_name="grep_search",
    )
    # c2 via paths, c1 via tool_name (paths first); b.py (c3) stays
    # unmarked.
    assert task.marked_unneeded == ["c2", "c1"]
    assert "Not marked" not in result


def test_mark_unneeded_tool_name_never_marks_itself(manager: Any) -> None:
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "a.py"}),
        tool_return_msg("c1", "read_file", "content a"),
        tool_call_msg("c2", "mark_unneeded", {}),
        tool_return_msg("c2", "mark_unneeded", "Marked: ..."),
    ]
    # A total miss (the only requested target is mark_unneeded itself)
    # is a ModelRetry, like any other full miss.
    with pytest.raises(ValueError, match="never evictable"):
        _mark_unneeded(
            cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))),
            tool_name="mark_unneeded",
        )
    assert task.marked_unneeded == []


def test_mark_unneeded_batch_dedupes_overlapping_selectors(manager: Any) -> None:
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "a.py"}),
        tool_return_msg("c1", "read_file", "content a"),
    ]
    _mark_unneeded(
        cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))),
        paths=["a.py"],
        tool_name="read_file",
    )
    # The same call is reachable through both selectors — one mark.
    assert task.marked_unneeded == ["c1"]


def test_mark_unneeded_unknown_tool_name_reports_miss(manager: Any) -> None:
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "a.py"}),
        tool_return_msg("c1", "read_file", "content a"),
    ]
    with pytest.raises(ValueError, match="no_such_tool"):
        _mark_unneeded(
            cast(Any, FakeCtx(messages, FakeDeps(manager, task.id))),
            tool_name="no_such_tool",
        )
    assert task.marked_unneeded == []


# -- pipeline gates -------------------------------------------------------------------


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


@pytest.fixture
def app_small_ctx(
    project_root: Path, configured_env: None, monkeypatch: pytest.MonkeyPatch
) -> DrCodeApp:
    monkeypatch.setenv("DRCODE_MAX_CONTEXT_TOKENS", "1000")
    return DrCodeApp(project_root)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedState:
    state = ScriptedState()
    monkeypatch.setattr(
        agent_model, "build_model", lambda cfg, announce_wait=None: state.model()
    )
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


def _seed_marked_task(app: DrCodeApp, content: str, usage: int = 0) -> Any:
    # No pending input (empty brief): the mount-time _advance() must leave
    # the task queued, so the gates are exercised by direct calls, not by
    # an auto-promoted run racing the assertions.
    task = app.manager.create("docs", "", kind=TaskKind.RESEARCH)
    task.messages = big_return_transcript(content, usage=usage)
    task.marked_unneeded = ["c1"]
    app.manager.storage.save(task)
    return task


async def test_batch_eviction_gate_fires_at_third_of_window(
    app_small_ctx: DrCodeApp, scripted: ScriptedState
) -> None:
    task = _seed_marked_task(app_small_ctx, "y" * 6000)  # ≈1500 tokens ≥ 1000/3
    assert app_small_ctx._eviction_due(task)

    async with app_small_ctx.run_test() as pilot:
        await pilot.pause()
        evicted = app_small_ctx._maybe_evict_marked(task)
        await pilot.pause()

    assert evicted == estimate_tokens("y" * 6000)
    (stub_part,) = stubs_in(task.messages)
    assert stub_part["tool_call_id"] == "c1"
    assert task.marked_unneeded == []
    assert any("Context unloading" in t for t in event_texts(task))
    # The stub is persisted, not just in memory.
    reloaded = {t.id: t for t in Storage(app_small_ctx.project_root).load_all()}[
        task.id
    ]
    assert stubs_in(reloaded.messages)
    # The eviction credit is deposited for the fill gates.
    assert app_small_ctx._evicted_ctx_tokens[task.id] == estimate_tokens("y" * 6000)


async def test_batch_eviction_gate_below_threshold_is_noop(
    app_small_ctx: DrCodeApp, scripted: ScriptedState
) -> None:
    task = _seed_marked_task(app_small_ctx, "y" * 400)  # ≈100 tokens < 334
    assert not app_small_ctx._eviction_due(task)
    assert app_small_ctx._maybe_evict_marked(task) == 0
    assert stubs_in(task.messages) == []
    assert task.marked_unneeded == ["c1"]


async def test_pre_compaction_eviction_spares_the_summarizer(
    app_small_ctx: DrCodeApp, scripted: ScriptedState
) -> None:
    """Fill over the compaction threshold (900 ≥ 80% of 1000), marked
    result worth ≈1500 tokens: the forced pre-compaction eviction pulls
    the fill back under the threshold and the summarizer never runs."""
    task = _seed_marked_task(app_small_ctx, "y" * 6000, usage=900)
    assert app_small_ctx._compaction_due(task) == 908

    async with app_small_ctx.run_test() as pilot:
        await pilot.pause()
        await app_small_ctx._maybe_compact(task)
        await pilot.pause()

    assert scripted.calls == []  # the summarizer was never invoked
    assert not any("Context compacted" in t for t in event_texts(task))
    assert any("Context unloading" in t for t in event_texts(task))
    (stub_part,) = stubs_in(task.messages)
    assert stub_part["tool_call_id"] == "c1"
    assert task.marked_unneeded == []


async def test_pre_compaction_eviction_lightens_the_summarizer_input(
    app_small_ctx: DrCodeApp,
    scripted: ScriptedState,
    wide_compaction: None,
) -> None:
    """The eviction alone cannot get under the threshold (the mark is
    small next to the fill): the summarizer runs anyway — but on the
    lightened history, seeing the stub instead of the collapsed content."""
    task = _seed_marked_task(app_small_ctx, "y" * 400, usage=950)
    scripted.steps = [text_response("THE SUMMARY BODY")]

    async with app_small_ctx.run_test() as pilot:
        await pilot.pause()
        await app_small_ctx._maybe_compact(task)
        await pilot.pause()

    assert any("Context compacted" in t for t in event_texts(task))
    # The summarizer request carried the stub, not the collapsed body.
    assert "tool result evicted" in scripted.calls[-1]
    assert "y" * 100 not in scripted.calls[-1]


async def test_live_run_pauses_evicts_and_resumes(
    app_small_ctx: DrCodeApp,
    scripted: ScriptedState,
    project_root: Path,
) -> None:
    """Full cycle on a live run: read → mark → the tools-node gate pauses
    the run (marked results reached ≈1/3 of the window) → the next turn
    boundary evicts in one batch → the continuation turn resumes from the
    stubbed transcript and the task completes. The call/return pairs
    survive; the marks are consumed."""
    # ≈500 tokens: big enough to trip the 1/3-of-window gate (334), small
    # enough to pass read_file's own compaction-threshold guard (800).
    (project_root / "notes.txt").write_text("y" * 2000)
    task = app_small_ctx.manager.create(
        "Read stuff", "read notes.txt", kind=TaskKind.RESEARCH
    )
    scripted.steps = [
        tool_response("read_file", {"path": "notes.txt"}, call_id="c1"),
        tool_response("mark_unneeded", {"paths": ["notes.txt"]}, call_id="c2"),
        text_response("done reading"),
    ]

    async with app_small_ctx.run_test() as pilot:
        await pilot.pause()
        app_small_ctx._advance()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED,
            pilot,
            what="research task completed after the eviction cycle",
        )
        await pilot.pause()

    texts = event_texts(task)
    assert any("pausing the run to unload" in t for t in texts)
    assert any("Context unloading" in t for t in texts)
    # Three model requests: read, mark, the continuation turn.
    assert len(scripted.calls) == 3
    # The read result was collapsed to a stub; its tool call survives.
    (stub_part,) = stubs_in(task.messages)
    assert stub_part["tool_call_id"] == "c1"
    assert "read_file" in stub_part["content"]
    assert task.marked_unneeded == []
    load_messages(task.messages)  # API-valid


# -- automatic tool result clearing (the age criterion) -----------------------------


def auto_transcript_raw(content: str, n: int, last_usage: int = 0) -> list[Any]:
    """The unserialized form of auto_transcript (for tests that mutate
    single messages before serializing)."""
    messages: list[Any] = [user_msg("read things")]
    for i in range(n):
        messages.append(tool_call_msg(f"c{i}", "read_file", {"path": f"src/f{i}.py"}))
        messages.append(tool_return_msg(f"c{i}", "read_file", content))
    if last_usage:
        messages.append(
            ModelResponse(
                parts=[TextPart("final")],
                usage=RequestUsage(input_tokens=last_usage, output_tokens=8),
            )
        )
    return messages


def auto_transcript(content: str, n: int, last_usage: int = 0) -> list[Any]:
    """prompt → n rounds of (call-response ci → return-request ci), plus a
    final text response carrying `last_usage` when set.

    Exchange count is n without the final response, n + 1 with it.
    Return ci's age — exchanges AFTER the response carrying its tool
    call — is (n - 1 - i) without / (n - i) with the final response:
    with n=16 the eligible-by-default band (age ≥ 10) is c0..c5, and the
    freshness anchor covers c11..c15."""
    return serialize_messages(auto_transcript_raw(content, n, last_usage))


def test_auto_evictable_ids_age_and_anchor() -> None:
    """Age ≥ min_age_turns makes a result eligible; the last keep_recent
    live returns are anchored out of it however old."""
    messages = auto_transcript("y" * 100, n=5)  # ages c0..c4 = 4,3,2,1,0
    assert eviction.auto_evictable_ids(messages, min_age_turns=1, keep_recent=0) == {
        "c0",
        "c1",
        "c2",
        "c3",
    }
    assert eviction.auto_evictable_ids(messages, min_age_turns=1, keep_recent=3) == {
        "c0",
        "c1",
    }
    # Default knobs: nothing is 10 exchanges old here.
    assert eviction.auto_evictable_ids(messages) == set()


def test_auto_evictable_ids_exclusions() -> None:
    """mark_unneeded's own returns, already-collapsed stubs and returns
    without their sibling call are never selected."""
    messages = auto_transcript("y" * 100, n=12)  # ages: c0=11 … c11=0
    assert eviction.auto_evictable_ids(messages) == {"c0", "c1"}

    marked_own = auto_transcript_raw("y" * 100, n=12)
    marked_own[2] = tool_return_msg("c0", "mark_unneeded", "ok")
    assert eviction.auto_evictable_ids(serialize_messages(marked_own)) == {"c1"}

    stubbed = auto_transcript_raw("y" * 100, n=12)
    stubbed[2].parts[0].content = eviction.evicted_stub(
        "read_file", {"path": "src/f0.py"}
    )
    assert eviction.auto_evictable_ids(serialize_messages(stubbed)) == {"c1"}

    orphan = auto_transcript_raw("y" * 100, n=12)
    orphan[1] = ModelResponse(parts=[TextPart("no calls here")])  # c0's call gone
    assert eviction.auto_evictable_ids(serialize_messages(orphan)) == {"c1"}


async def test_auto_clearing_fires_between_turns_without_marks(
    app_small_ctx: DrCodeApp, scripted: ScriptedState
) -> None:
    """No marks anywhere: the batch eviction still collapses the six
    age-eligible results (≈1200 tokens) in ONE pass, keeps the anchor
    verbatim, deposits the credit and stays API-valid."""
    content = "y" * 800  # ≈200 tokens per result
    task = app_small_ctx.manager.create("docs", "", kind=TaskKind.RESEARCH)
    task.messages = auto_transcript(content, n=16)
    app_small_ctx.manager.storage.save(task)
    ids = app_small_ctx._auto_eviction_ids(task)
    assert ids == {f"c{i}" for i in range(6)}
    assert not task.marked_unneeded

    async with app_small_ctx.run_test() as pilot:
        await pilot.pause()
        evicted = app_small_ctx._maybe_evict_marked(task)
        await pilot.pause()

    assert evicted == estimate_tokens(content) * 6
    assert {p["tool_call_id"] for p in stubs_in(task.messages)} == ids
    # The anchor (c11..c15) and the not-yet-old middle stay verbatim.
    assert returns_in(task.messages)["c15"]["content"] == content
    assert returns_in(task.messages)["c10"]["content"] == content
    assert task.marked_unneeded == []
    assert app_small_ctx._evicted_ctx_tokens[task.id] == evicted
    assert any("auto-cleared 6 old tool result(s)" in t for t in event_texts(task))
    load_messages(task.messages)  # API-valid


async def test_pre_compaction_auto_clearing_spares_the_summarizer(
    app_small_ctx: DrCodeApp, scripted: ScriptedState
) -> None:
    """Fill over the threshold (908 ≥ 80% of 1000) with NO marks: the
    automatic clearing at the gate entry deposits its credit, the
    re-measured fill falls under the threshold and the summarizer never
    runs — the unloading alone did the compaction."""
    content = "y" * 800
    task = app_small_ctx.manager.create("docs", "", kind=TaskKind.RESEARCH)
    task.messages = auto_transcript(content, n=15, last_usage=900)
    app_small_ctx.manager.storage.save(task)

    async with app_small_ctx.run_test() as pilot:
        await pilot.pause()
        await app_small_ctx._maybe_compact(task)
        await pilot.pause()

    assert scripted.calls == []  # the summarizer was never invoked
    assert not any("Context compacted" in t for t in event_texts(task))
    assert any("Context unloading" in t for t in event_texts(task))
    assert len(stubs_in(task.messages)) == 6


async def test_marks_and_auto_clearing_share_one_batch(
    app_small_ctx: DrCodeApp, scripted: ScriptedState
) -> None:
    """A mark on a NOT-yet-old result rides the same batch as the
    automatic clearing — one pass, one prompt-cache invalidation, one
    event naming both origins."""
    content = "y" * 800
    task = app_small_ctx.manager.create("docs", "", kind=TaskKind.RESEARCH)
    task.messages = auto_transcript(content, n=16)
    task.marked_unneeded = ["c7"]  # age 8: below the age horizon
    app_small_ctx.manager.storage.save(task)

    async with app_small_ctx.run_test() as pilot:
        await pilot.pause()
        evicted = app_small_ctx._maybe_evict_marked(task)
        await pilot.pause()

    assert evicted == estimate_tokens(content) * 7
    assert {p["tool_call_id"] for p in stubs_in(task.messages)} == (
        {f"c{i}" for i in range(6)} | {"c7"}
    )
    assert task.marked_unneeded == []
    assert any(
        "collapsed 1 marked and 6 old tool result(s)" in t for t in event_texts(task)
    )


async def test_mid_run_pause_gate_auto_volume(
    app_small_ctx: DrCodeApp, scripted: ScriptedState
) -> None:
    """The live-run pause fires on the automatic clearing only when the
    auto-evictable ≈volume reached 1/6 of the window; between turns the
    pass fires on any eligible result regardless of volume."""
    big = app_small_ctx.manager.create("docs", "", kind=TaskKind.RESEARCH)
    big.messages = auto_transcript("y" * 800, n=16)  # 6 × ≈200 ≥ 1000/6
    assert app_small_ctx._auto_eviction_due(big)
    assert not app_small_ctx._eviction_due(big)

    small = app_small_ctx.manager.create("docs", "", kind=TaskKind.RESEARCH)
    small.messages = auto_transcript("y" * 100, n=16)  # 6 × ≈25 < 1000/6
    assert app_small_ctx._auto_eviction_ids(small) == {f"c{i}" for i in range(6)}
    assert not app_small_ctx._auto_eviction_due(small)

    async with app_small_ctx.run_test() as pilot:
        await pilot.pause()
        assert app_small_ctx._maybe_evict_marked(small) > 0
        await pilot.pause()
    assert len(stubs_in(small.messages)) == 6


# -- eviction nudge: dynamic-instructions hook --------------------------------------


@pytest.fixture
def fake_config() -> Any:
    from drcode.config import Config

    return Config(project_root=Path.cwd(), model="test-model", api_key="sk-test")


def hint_transcript(last_exchange_big: bool) -> list[Any]:
    """prompt → big read c1 → small grep c2 (freshest exchange), with a
    usage-bearing text response between them so the fill is ≈20k of a 30k
    window. `last_exchange_big` makes the FRESH c2 return heavy too."""
    big = "y" * 8000  # 2000 ≈tokens — exactly the candidate threshold
    return [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "src/big.py"}),
        tool_return_msg("c1", "read_file", big),
        ModelResponse(
            parts=[TextPart("noted")],
            usage=RequestUsage(input_tokens=20_000, output_tokens=8),
        ),
        tool_call_msg("c2", "grep_search", {"pattern": "x"}),
        tool_return_msg("c2", "grep_search", big if last_exchange_big else "hits"),
    ]


def test_eviction_hint_silent_below_half_window(manager: Any) -> None:
    """Condition (a) unmet — fill below ~1/2 of the window: no reminder,
    however heavy the unmarked old returns are."""
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = hint_transcript(last_exchange_big=False)
    messages[3] = ModelResponse(  # drop the fill to ≈10k of 30k
        parts=[TextPart("noted")],
        usage=RequestUsage(input_tokens=10_000, output_tokens=8),
    )
    ctx = cast(Any, FakeCtx(messages, FakeDeps(manager, task.id)))
    assert eviction_dynamic_instructions(ctx, max_context_tokens=30_000) is None


def test_eviction_hint_names_candidates_and_volume(manager: Any) -> None:
    """Both conditions met: the reminder carries the ≈total of the
    candidates, a per-result label (the same label format mark_unneeded's
    answer uses) and points at mark_unneeded's selectors and the exact
    numbers source (context_status)."""
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    big = "y" * 8000
    ctx = cast(
        Any,
        FakeCtx(
            hint_transcript(last_exchange_big=False),
            FakeDeps(manager, task.id),
        ),
    )
    hint = eviction_dynamic_instructions(ctx, max_context_tokens=30_000)
    assert hint is not None
    # Only c1 is a candidate: the fresh c2 return is small.
    assert f"≈{_fmt_approx_tokens(estimate_tokens(big))} ≈tokens" in hint
    assert "read_file(path='src/big.py')" in hint
    assert "mark_unneeded" in hint
    assert "paths=" in hint
    assert "tool_name=" in hint
    assert "context_status" in hint


def test_eviction_hint_skips_marked_and_fresh_returns(manager: Any) -> None:
    """Condition (b) unmet — nothing to point at: the heavy old return is
    already marked, and the fresh (latest-exchange) heavy return is not a
    candidate by design, so the hook stays silent despite a full context."""
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    task.marked_unneeded = ["c1"]
    ctx = cast(
        Any,
        FakeCtx(
            hint_transcript(last_exchange_big=True),
            FakeDeps(manager, task.id),
        ),
    )
    assert eviction_dynamic_instructions(ctx, max_context_tokens=30_000) is None


def test_eviction_hint_skips_latest_exchange_only(manager: Any) -> None:
    """A heavy return in the FRESHEST exchange alone never triggers the
    nudge — the model has not had a turn to act on it yet."""
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = [
        user_msg("read things"),
        tool_call_msg("c1", "read_file", {"path": "a.py"}),
        tool_return_msg("c1", "read_file", "small"),
        ModelResponse(
            parts=[TextPart("noted")],
            usage=RequestUsage(input_tokens=20_000, output_tokens=8),
        ),
        tool_call_msg("c2", "grep_search", {"pattern": "x"}),
        tool_return_msg("c2", "grep_search", "y" * 8000),
    ]
    ctx = cast(Any, FakeCtx(messages, FakeDeps(manager, task.id)))
    assert eviction_dynamic_instructions(ctx, max_context_tokens=30_000) is None


def test_eviction_hint_frozen_while_active(manager: Any) -> None:
    """Prefix-cache regression: while the hint is up, the SYSTEM-zone text
    must be byte-identical from request to request — the live fill figure
    and the candidate list are a frozen snapshot, NOT recomputed per
    request (a changing block invalidates the server-side prefix cache
    from the very top; on a local llama.cpp slot that is a full prompt
    reprocess per request). Here the fill grows and a candidate gets
    marked between the two requests — the text must not move."""
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    big = "y" * 8000
    messages = hint_transcript(last_exchange_big=False)
    # A second old heavy return, so marking c1 still leaves candidates.
    messages.extend(
        [
            ModelResponse(
                parts=[TextPart("again")],
                usage=RequestUsage(input_tokens=21_000, output_tokens=8),
            ),
            tool_call_msg("c3", "read_file", {"path": "src/more.py"}),
            tool_return_msg("c3", "read_file", big),
        ]
    )
    ctx = cast(Any, FakeCtx(messages, FakeDeps(manager, task.id)))
    first = eviction_dynamic_instructions(ctx, max_context_tokens=30_000)
    assert first is not None
    # Next request: more history, higher fill, c1 already marked (c3 stays).
    messages.append(
        ModelResponse(
            parts=[TextPart("more")],
            usage=RequestUsage(input_tokens=25_000, output_tokens=8),
        )
    )
    task.marked_unneeded = ["c1"]
    second = eviction_dynamic_instructions(ctx, max_context_tokens=30_000)
    assert second is not None and second == first
    assert "83%" not in second  # the fill snapshot is not re-taken (would be 83%)
    assert "70%" in second  # the frozen figure from the first active request
    # The frozen text replays from the manager cache (per task).
    assert manager._eviction_hints[task.id] == first


def test_eviction_hint_cleared_when_conditions_stop_holding(manager: Any) -> None:
    """The frozen text is dropped when the conditions stop holding: no
    candidates left (everything marked and collapsed) or the fill fell
    back below the threshold — after which a re-qualifying request
    generates a FRESH text (a new snapshot), not the stale frozen one."""
    task = manager.create("docs", "read things", kind=TaskKind.RESEARCH)
    messages = hint_transcript(last_exchange_big=False)
    ctx = cast(Any, FakeCtx(messages, FakeDeps(manager, task.id)))
    first = eviction_dynamic_instructions(ctx, max_context_tokens=30_000)
    assert first is not None
    # Everything marked (and collapsed to stubs): no candidates left.
    task.marked_unneeded = ["c1"]
    messages[2] = tool_return_msg(
        "c1", "read_file", eviction.evicted_stub("read_file", {"path": "a.py"})
    )
    assert eviction_dynamic_instructions(ctx, max_context_tokens=30_000) is None
    assert task.id not in manager._eviction_hints
    # A heavier candidate re-appears → a fresh text (a new snapshot).
    task.marked_unneeded = []
    messages[2] = tool_return_msg("c1", "read_file", "z" * 12000)
    second = eviction_dynamic_instructions(ctx, max_context_tokens=30_000)
    assert second is not None and second != first
    assert manager._eviction_hints[task.id] == second
    # Fill falls below the threshold → cleared again.
    messages.append(
        ModelResponse(
            parts=[TextPart("more")],
            usage=RequestUsage(input_tokens=10_000, output_tokens=8),
        )
    )
    assert eviction_dynamic_instructions(ctx, max_context_tokens=30_000) is None
    assert task.id not in manager._eviction_hints


def test_eviction_hint_registered_on_coding_agent(
    manager: Any, fake_config: Any
) -> None:
    """build_agent registers the hook as dynamic instructions with the
    eviction_hint setting ON; the tester agent gets none (no
    mark_unneeded tool there)."""
    from drcode import agent as agent_mod
    from drcode.agent import build_agent, build_tester_agent

    ag = build_agent(fake_config, kind=TaskKind.FEATURE, eviction_hint=True)
    assert any(
        part.dynamic
        and getattr(part.instruction, "__name__", "") == "_eviction_hint_instructions"
        for part in ag._instructions
    )
    tester = build_tester_agent(fake_config)
    assert not any(
        getattr(part.instruction, "__name__", "") == "_eviction_hint_instructions"
        for part in tester._instructions
    )
    assert agent_mod.eviction_dynamic_instructions is eviction_dynamic_instructions


def test_eviction_hint_absent_by_default(manager: Any, fake_config: Any) -> None:
    """The default (eviction_hint=False) registers no hint instruction at
    all: nothing rides the requests, nothing scans the transcript."""
    from drcode.agent import build_agent

    ag = build_agent(fake_config, kind=TaskKind.FEATURE)
    assert not any(
        getattr(part.instruction, "__name__", "") == "_eviction_hint_instructions"
        for part in ag._instructions
    )


def test_eviction_hint_defaults_off_in_settings(project_root: Any) -> None:
    """ProjectSettings.eviction_hint defaults to False (opt-in feature)."""
    from drcode.settings import ProjectSettings

    assert ProjectSettings().eviction_hint is False


def test_eviction_hint_setting_persisted(project_root: Any) -> None:
    """The flag round-trips through the project settings store, so a
    toggle applies to the next activation of each task."""
    from drcode.settings import ProjectSettings, ProjectSettingsStore

    store = ProjectSettingsStore(project_root)
    store.save(ProjectSettings(eviction_hint=True))
    assert ProjectSettingsStore(project_root).load().eviction_hint is True
    store.save(ProjectSettings(eviction_hint=False))
    assert ProjectSettingsStore(project_root).load().eviction_hint is False
