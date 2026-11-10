"""Tests for drcode.agent: transcript handling, compaction, display, tools."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from drcode import agent as agent_mod
from drcode.agent import (
    COMPACTION_RESUME_PREFIX,
    COMPACTION_SUMMARY_PREFIX,
    AgentDeps,
    all_user_prompts,
    build_compaction_prompt,
    compacted_messages,
    ends_with_open_request,
    extract_display,
    extract_previous_summary,
    first_user_prompt,
    is_pending_retry,
    is_summary_message,
    last_context_tokens,
    load_agents_md,
    load_messages,
    mechanical_compaction_messages,
    prompt_fits_context,
    render_history_for_summary,
    resumable_history,
    sanitized_messages,
    serialize_messages,
    split_original_prompts,
    unwrap_summary_envelope,
)
from drcode.config import Config
from drcode.models import TaskKind, TaskStatus, VerificationStatus
from drcode.settings import GitPolicy
from drcode.task_manager import TaskManager
from drcode.tools import PathViolation
from drcode.transcript import _message_time


def user_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def text_response(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def tool_call_response(call_id: str = "c1", tool: str = "run_bash") -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(tool_name=tool, args={"command": "ls"}, tool_call_id=call_id)
        ]
    )


def tool_return_msg(call_id: str = "c1", tool: str = "run_bash") -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name=tool, content="out", tool_call_id=call_id)]
    )


# -- serialization roundtrip ---------------------------------------------------


def test_serialize_load_roundtrip() -> None:
    messages = [
        user_msg("hello"),
        text_response("world"),
        tool_call_response(),
        tool_return_msg(),
    ]
    data = serialize_messages(messages)
    assert all(isinstance(m, dict) for m in data)
    restored = load_messages(data)
    assert restored == messages


def test_sanitized_messages_trims_invalid_tail() -> None:
    data = serialize_messages([user_msg("hi"), text_response("ok")])
    assert len(sanitized_messages(data)) == 2
    # Trailing tool call without a result is cut.
    data2 = serialize_messages([user_msg("hi"), tool_call_response()])
    assert len(sanitized_messages(data2)) == 1


# -- resumable_history -----------------------------------------------------------


def test_resumable_history_empty() -> None:
    assert resumable_history([]) == []


def test_resumable_history_keeps_completed_exchange() -> None:
    data = serialize_messages(
        [user_msg("q"), text_response("a"), user_msg("q2"), text_response("a2")]
    )
    assert len(resumable_history(data)) == 4


def test_resumable_history_cuts_open_tool_call() -> None:
    data = serialize_messages(
        [
            user_msg("q"),
            text_response("thinking"),
            tool_call_response(),
            tool_return_msg(),
        ]
    )
    # Full exchange with a result is valid.
    assert len(resumable_history(data)) == 4
    # Unanswered tool call at the tail is cut (with the text before it kept? no —
    # only up to the last safe point, which is the text response).
    data_bad = serialize_messages([user_msg("q"), tool_call_response()])
    kept = resumable_history(data_bad)
    assert len(kept) == 1


def test_resumable_history_tool_then_text_ok() -> None:
    data = serialize_messages(
        [user_msg("q"), tool_call_response(), tool_return_msg(), text_response("done")]
    )
    assert len(resumable_history(data)) == 4


# -- ends_with_open_request / is_pending_retry --------------------------------------


def test_ends_with_open_request() -> None:
    assert not ends_with_open_request([])
    data = serialize_messages([user_msg("q")])
    assert ends_with_open_request(data)
    data2 = serialize_messages([user_msg("q"), text_response("a")])
    assert not ends_with_open_request(data2)


def test_is_pending_retry() -> None:
    data = serialize_messages([user_msg("do it"), tool_call_response()])
    # Interrupted attempt: the prompt is in the transcript, no final answer yet.
    assert is_pending_retry(data, "do it")
    # Completed run: the prompt got a final text answer.
    data_done = serialize_messages([user_msg("do it"), text_response("finished")])
    assert not is_pending_retry(data_done, "do it")
    # Text never in the transcript.
    assert not is_pending_retry(data, "something else")


def test_first_user_prompt() -> None:
    data = serialize_messages([user_msg("  the task  "), text_response("x")])
    assert first_user_prompt(data) == "the task"
    assert first_user_prompt([]) == ""


def test_all_user_prompts() -> None:
    data = serialize_messages(
        [
            user_msg("  the task  "),
            text_response("x"),
            user_msg("actually use sqlite"),
            tool_call_response(),
            tool_return_msg(),
            text_response("ok"),
            user_msg("and add tests"),
        ]
    )
    assert all_user_prompts(data) == [
        "the task",
        "actually use sqlite",
        "and add tests",
    ]
    assert all_user_prompts([]) == []


def test_all_user_prompts_skips_compaction_summary() -> None:
    data = serialize_messages(
        [
            user_msg("the task"),
            user_msg(f"{COMPACTION_SUMMARY_PREFIX}\n\nsummary text"),
            text_response("done"),
        ]
    )
    assert all_user_prompts(data) == ["the task"]


def test_all_user_prompts_skips_compaction_resume_note() -> None:
    """The continuation note a round is resumed with after a successful
    compaction is pipeline machinery, not user input."""
    data = serialize_messages(
        [
            user_msg("the task"),
            user_msg(f"{COMPACTION_RESUME_PREFIX} continue the verification"),
        ]
    )
    assert all_user_prompts(data) == ["the task"]


# -- context tokens --------------------------------------------------------------


def test_last_context_tokens() -> None:
    data = [
        {
            "kind": "response",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 25,
            },
        }
    ]
    # Context fill = prompt + generated output (input_tokens already
    # includes cache reads/writes in pydantic-ai's normalized usage).
    assert last_context_tokens(data) == 125


def test_last_context_tokens_no_cache_double_count() -> None:
    # Regression: OpenAI-style usage where the provider prompt includes
    # cached tokens. pydantic-ai normalizes this to input_tokens=41030
    # (cache_read_tokens=40000 included), which must NOT be added again —
    # the old formula reported ~2x the real slot fill.
    data = [
        {
            "kind": "response",
            "usage": {
                "input_tokens": 41030,
                "cache_read_tokens": 40000,
                "cache_write_tokens": 0,
                "output_tokens": 30,
            },
        }
    ]
    assert last_context_tokens(data) == 41060


def test_last_context_tokens_zero_usage_skipped() -> None:
    data = [
        {"kind": "response", "usage": {"input_tokens": 0}},
        {"kind": "response", "usage": {"input_tokens": 42}},
    ]
    assert last_context_tokens(data) == 42


def test_last_context_tokens_empty() -> None:
    assert last_context_tokens([]) is None
    assert last_context_tokens([{"kind": "request", "parts": []}]) is None


# -- display extraction ------------------------------------------------------------


def test_extract_display_user_and_assistant() -> None:
    data = serialize_messages([user_msg("question"), text_response("answer")])
    out = extract_display(data)
    assert [(role, text) for _, role, text in out] == [
        ("user", "question"),
        ("assistant", "answer"),
    ]


def test_extract_display_resumed_request_uses_prompt_stamp() -> None:
    """A resumed request (network retry without new text) carries a fresh
    top-level timestamp from pydantic-ai's resume path while the
    UserPromptPart keeps the moment the prompt was typed — display order
    must follow the part stamp, or the prompt drops below the service
    events that happened between typing and the successful attempt
    (task 27ef7164)."""
    typed = datetime(2026, 10, 31, 6, 47, 20, tzinfo=UTC)
    resumed = datetime(2026, 10, 31, 6, 48, 9, tzinfo=UTC)
    request = ModelRequest(
        parts=[UserPromptPart(content="hi", timestamp=typed)], timestamp=resumed
    )
    out = extract_display(serialize_messages([request]))
    assert [(at, role) for at, role, _ in out] == [(typed, "user")]


def test_message_time_fallbacks() -> None:
    # No top-level timestamp (legacy transcripts): the latest part stamp wins.
    legacy = {
        "kind": "request",
        "parts": [
            {
                "part_kind": "user-prompt",
                "content": "hi",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    assert _message_time(legacy) == datetime(2026, 1, 1, tzinfo=UTC)
    # Non-prompt request (tool returns): top-level timestamp wins.
    tools = {
        "kind": "request",
        "timestamp": "2026-01-02T00:00:00+00:00",
        "parts": [
            {
                "part_kind": "tool-return",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    assert _message_time(tools) == datetime(2026, 1, 2, tzinfo=UTC)
    # Nothing recorded at all.
    assert _message_time({"kind": "request", "parts": []}) is None


def test_extract_display_groups_thinking_and_text() -> None:
    response = ModelResponse(
        parts=[
            ThinkingPart(content="hmm"),
            TextPart(content="answer text"),
        ]
    )
    data = serialize_messages([response])
    out = extract_display(data)
    assert [(role, text) for _, role, text in out] == [
        ("reasoning", "hmm"),
        ("assistant", "answer text"),
    ]


def test_extract_display_tool_call_with_args() -> None:
    data = serialize_messages([tool_call_response(call_id="c1")])
    out = extract_display(data)
    assert len(out) == 1
    _, role, text = out[0]
    assert role == "tool-call"
    assert text.startswith("run_bash(")
    assert "command" in text


def test_extract_display_summary_message() -> None:
    data = serialize_messages(
        [user_msg(f"{COMPACTION_SUMMARY_PREFIX}\n\nthe summary body")]
    )
    out = extract_display(data)
    assert [(role, text) for _, role, text in out] == [("summary", "the summary body")]


def test_extract_display_reasoning_flushed_before_tool_call() -> None:
    response = ModelResponse(
        parts=[
            ThinkingPart(content="think"),
            ToolCallPart(tool_name="t", args={}, tool_call_id="c"),
        ]
    )
    data = serialize_messages([response])
    out = extract_display(data)
    assert [role for _, role, _ in out] == ["reasoning", "tool-call"]


def test_extract_display_reasoning_flushed_at_message_end() -> None:
    response = ModelResponse(parts=[ThinkingPart(content="think")])
    data = serialize_messages([response])
    out = extract_display(data)
    assert [role for _, role, _ in out] == ["reasoning"]


# -- compaction -------------------------------------------------------------------


def test_is_summary_message() -> None:
    data = serialize_messages([user_msg(f"{COMPACTION_SUMMARY_PREFIX}\nbody")])
    assert is_summary_message(data[0])
    plain = serialize_messages([user_msg("regular")])
    assert not is_summary_message(plain[0])
    assert not is_summary_message({"kind": "response", "parts": []})


def test_split_original_prompts() -> None:
    original = serialize_messages([user_msg("task one"), user_msg("task two")])
    later = serialize_messages(
        [text_response("work"), user_msg(f"{COMPACTION_SUMMARY_PREFIX}\ns")]
    )
    preserved, rest = split_original_prompts(original + later)
    assert preserved == original
    assert rest == later


def test_split_original_prompts_summary_first() -> None:
    data = serialize_messages([user_msg(f"{COMPACTION_SUMMARY_PREFIX}\ns")])
    preserved, rest = split_original_prompts(data)
    assert preserved == []
    assert rest == data


def test_split_original_prompts_original_merged_with_summary() -> None:
    """pydantic-ai merges the summary request with the adjacent original
    prompt on resume (_merge_consecutive_messages): the original prompt
    parts of the merged carrier stay in `preserved` (verbatim prefix)."""
    merged = serialize_messages(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content="original task"),
                    UserPromptPart(content=f"{COMPACTION_SUMMARY_PREFIX}\n\nbody"),
                ]
            )
        ]
    )
    preserved, rest = split_original_prompts(merged)
    assert [[p["content"] for p in m["parts"]] for m in preserved] == [
        ["original task"]
    ]
    assert rest == merged


def test_split_original_prompts_resume_note_merged_with_summary() -> None:
    """A continuation note merged into the summary carrier is compaction
    machinery: it must NOT join the preserved verbatim prefix (it would be
    re-preserved on every compaction cycle) — it stays in `rest`, where the
    summarizer sees it as part of the conversation history."""
    merged = serialize_messages(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content=f"{COMPACTION_SUMMARY_PREFIX}\n\nbody"),
                    UserPromptPart(
                        content=f"{COMPACTION_RESUME_PREFIX} continue the checks"
                    ),
                ]
            )
        ]
    )
    preserved, rest = split_original_prompts(merged)
    assert preserved == []
    assert rest == merged


def test_split_original_prompts_original_summary_and_note_merged() -> None:
    """The three-way merge (original + summary + note in one carrier):
    only the original prompt is preserved; the note rides in `rest`."""
    merged = serialize_messages(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content="original task"),
                    UserPromptPart(content=f"{COMPACTION_SUMMARY_PREFIX}\n\nbody"),
                    UserPromptPart(
                        content=f"{COMPACTION_RESUME_PREFIX} continue the checks"
                    ),
                ]
            )
        ]
    )
    preserved, rest = split_original_prompts(merged)
    assert [[p["content"] for p in m["parts"]] for m in preserved] == [
        ["original task"]
    ]
    assert rest == merged


def test_render_history_includes_resume_note() -> None:
    """The resume note is summarized with the rest of the history (only the
    COMPACTION_SUMMARY_PREFIX parts are skipped as the separate anchor)."""
    data = serialize_messages(
        [user_msg(f"{COMPACTION_RESUME_PREFIX} continue the checks")]
    )
    assert COMPACTION_RESUME_PREFIX in render_history_for_summary(data)


def test_extract_previous_summary() -> None:
    data = serialize_messages(
        [
            user_msg("task"),
            text_response("w"),
            user_msg(f"{COMPACTION_SUMMARY_PREFIX}\nbody\n"),
        ]
    )
    assert extract_previous_summary(data) == "body"
    assert extract_previous_summary(serialize_messages([user_msg("task")])) is None


def test_extract_previous_summary_takes_latest() -> None:
    data = serialize_messages(
        [
            user_msg(f"{COMPACTION_SUMMARY_PREFIX}\nold"),
            text_response("x"),
            user_msg(f"{COMPACTION_SUMMARY_PREFIX}\nnew"),
        ]
    )
    assert extract_previous_summary(data) == "new"


def test_unwrap_summary_envelope_json_summary() -> None:
    # Model ignored the markdown template and wrapped the summary into a
    # JSON envelope with escaped newlines: unwrap to the markdown itself.
    text = json.dumps({"summary": "## Goal\n- Verify a\n- Verify b"})
    assert unwrap_summary_envelope(text) == "## Goal\n- Verify a\n- Verify b"


def test_unwrap_summary_envelope_inside_json_fence() -> None:
    text = '```json\n{"summary": "## Goal\\n- Verify x"}\n```'
    assert unwrap_summary_envelope(text) == "## Goal\n- Verify x"


def test_unwrap_summary_envelope_prefers_summary_key() -> None:
    text = json.dumps({"title": "t", "summary": "the body", "notes": "n"})
    assert unwrap_summary_envelope(text) == "the body"


def test_unwrap_summary_envelope_single_string_field() -> None:
    text = json.dumps({"markdown": "## Goal\n- done"})
    assert unwrap_summary_envelope(text) == "## Goal\n- done"


def test_unwrap_summary_envelope_strips_bare_fence() -> None:
    # A fence around plain markdown is also a formatting artifact: strip it.
    assert unwrap_summary_envelope("```\n## Goal\n- x\n```") == "## Goal\n- x"


def test_unwrap_summary_envelope_markdown_passthrough() -> None:
    text = "## Goal\n- Verify x\n```python\nprint(1)\n```\n## Notes\nok"
    assert unwrap_summary_envelope(text) == text


def test_unwrap_summary_envelope_non_dict_json_passthrough() -> None:
    assert unwrap_summary_envelope("42") == "42"
    assert unwrap_summary_envelope('["a", "b"]') == '["a", "b"]'


def test_unwrap_summary_envelope_json_without_string_field() -> None:
    text = json.dumps({"count": 3, "items": [1, 2]})
    assert unwrap_summary_envelope(text) == text


def test_unwrap_summary_envelope_json_without_summary_key() -> None:
    # Several string fields and no "summary": ambiguous, pass through.
    text = json.dumps({"title": "t", "body": "b"})
    assert unwrap_summary_envelope(text) == text


def test_unwrap_summary_envelope_unclosed_fence() -> None:
    # Truncated output: opening ```json fence without the closing one —
    # the leading fence line is still stripped.
    text = '```json\n{"summary": "## Goal\\n- Verify x"}'
    assert unwrap_summary_envelope(text) == "## Goal\n- Verify x"


def test_unwrap_summary_envelope_unclosed_fence_broken_json() -> None:
    # The #9c3236e8 case: unclosed fence AND invalid JSON — unescaped double
    # quotes inside the "summary" value. Soft shape-match recovers the text.
    raw = '```json\n{"summary": "## Goal\\n- Moved to "raw" bag\\n- Then "done" again"}'
    assert (
        unwrap_summary_envelope(raw)
        == '## Goal\n- Moved to "raw" bag\n- Then "done" again'
    )


def test_unwrap_summary_envelope_broken_json_no_fence() -> None:
    # Same broken envelope without any fence: still recovered.
    raw = '{"summary": "line one\\nquote "here" stays"}'
    assert unwrap_summary_envelope(raw) == 'line one\nquote "here" stays'


def test_unwrap_summary_envelope_closed_fence_still_unwrapped() -> None:
    # Regression guard: the closed-fence path keeps working after the
    # unclosed-fence change.
    text = '```json\n{"summary": "## Goal\\n- Verify x"}\n```'
    assert unwrap_summary_envelope(text) == "## Goal\n- Verify x"
    assert unwrap_summary_envelope("```\n## Goal\n- x\n```") == "## Goal\n- x"


def test_unwrap_summary_envelope_broken_envelope_non_summary_passthrough() -> None:
    # A broken (unparsable) object that is NOT the summary envelope shape
    # must not be silently mangled.
    raw = '{"title": "quoted "text" here"}'
    assert unwrap_summary_envelope(raw) == raw


def test_render_history_for_summary() -> None:
    data = serialize_messages(
        [
            user_msg("task"),
            text_response("partial answer"),
            tool_call_response(),
            tool_return_msg(),
        ]
    )
    text = render_history_for_summary(data)
    assert "[user]\ntask" in text
    assert "[assistant]\npartial answer" in text
    assert "[tool call] run_bash(" in text
    assert "[tool result for run_bash]\nout" in text


def test_render_history_skips_summary_message() -> None:
    data = serialize_messages([user_msg(f"{COMPACTION_SUMMARY_PREFIX}\nsecret")])
    assert "secret" not in render_history_for_summary(data)


def test_render_history_truncates_huge_tool_results() -> None:
    huge = "y" * 30_000
    # Manually craft an oversized tool return.
    data = serialize_messages(
        [
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="run_bash", content=huge, tool_call_id="c")
                ]
            )
        ]
    )
    text = render_history_for_summary(data)
    assert "…[truncated]" in text
    assert len(text) < 25_000


def test_build_compaction_prompt_structure() -> None:
    prompt = build_compaction_prompt("prev summary", "transcript", "TEMPLATE")
    assert "<conversation>\ntranscript\n</conversation>" in prompt
    assert "<previous-summary>\nprev summary\n</previous-summary>" in prompt
    assert prompt.rstrip().endswith("TEMPLATE")


def test_build_compaction_prompt_no_previous() -> None:
    prompt = build_compaction_prompt(None, "t", "T")
    assert "<previous-summary>\n(none)\n</previous-summary>" in prompt


# -- map-reduce chunking --------------------------------------------------------


def _long_transcript(turns: int, filler: str = "x" * 200) -> list[dict]:
    """A rendered-past-the-window transcript: each turn is a user prompt, a
    tool call and its (oversized) result plus an assistant remark."""
    messages: list[ModelRequest | ModelResponse] = []
    for i in range(turns):
        messages.append(user_msg(f"task {i}: {filler}"))
        messages.append(tool_call_response(call_id=f"c{i}"))
        messages.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="run_bash", content=filler, tool_call_id=f"c{i}"
                    )
                ]
            )
        )
        messages.append(text_response(f"did {i}"))
    return serialize_messages(messages)


def test_chunk_history_respects_budget() -> None:
    from drcode.agent import chunk_history_for_summary

    data = _long_transcript(turns=12)
    slices = chunk_history_for_summary(data, budget_tokens=600)
    assert len(slices) > 1
    for s in slices:
        assert len(s) <= 600 * 4


def test_chunk_history_never_separates_tool_call_from_result() -> None:
    from drcode.agent import chunk_history_for_summary

    data = _long_transcript(turns=10)
    for s in chunk_history_for_summary(data, budget_tokens=1_000):
        calls = s.count("[tool call]")
        results = s.count("[tool result for")
        assert calls == results


def test_chunk_history_never_splits_user_from_answer() -> None:
    from drcode.agent import chunk_history_for_summary

    data = _long_transcript(turns=8)
    for s in chunk_history_for_summary(data, budget_tokens=1_500):
        # A slice starting with a user prompt always carries at least one
        # following non-user block (its answer's beginning)...
        if s.startswith("[user]"):
            assert "\n[assistant]\n" in s or "[tool call]" in s


def test_chunk_history_prefers_turn_boundaries() -> None:
    from drcode.agent import chunk_history_for_summary

    filler = "x" * 200
    data = _long_transcript(turns=6, filler=filler)
    # One turn ≈ (8 + 200) + 40 + (17 + 200) + (9 + 6) chars — a budget of
    # two turns must make every interior slice start at a [user] block.
    slices = chunk_history_for_summary(data, budget_tokens=1_200)
    for s in slices[1:-1]:
        assert s.startswith("[user]")


def test_chunk_history_oversized_single_block_hard_split() -> None:
    from drcode.agent import chunk_history_for_summary

    data = serialize_messages([user_msg("huge " * 40_000)])
    slices = chunk_history_for_summary(data, budget_tokens=2_000)
    assert len(slices) > 1
    assert "".join(slc.replace("\n\n", " ") for slc in slices).startswith("[user]")


def test_chunk_history_empty_transcript() -> None:
    from drcode.agent import chunk_history_for_summary

    assert chunk_history_for_summary([], 1_000) == []


def test_build_reduce_prompt_structure() -> None:
    from drcode.agent import build_reduce_prompt

    prompt = build_reduce_prompt(["partial one", "partial two"], "TEMPLATE")
    assert "<partial index=1>\npartial one\n</partial>" in prompt
    assert "<partial index=2>\npartial two\n</partial>" in prompt
    assert prompt.rstrip().endswith("TEMPLATE")


def test_build_reduce_prompt_with_checklist() -> None:
    from drcode.agent import build_reduce_prompt

    prompt = build_reduce_prompt(["p"], "T", checklist="- [x] item")
    assert "<task-checklist>\n- [x] item\n</task-checklist>" in prompt


def test_slice_budget_tokens_leaves_headroom() -> None:
    from drcode.agent import slice_budget_tokens

    cfg = Config(max_context_tokens=100_000, max_tokens=8_192)
    budget = slice_budget_tokens(cfg, anchor="a" * 400, checklist=None)
    assert 0 < budget < 100_000
    # A bigger anchor eats into the budget.
    bigger = slice_budget_tokens(cfg, anchor="a" * 40_000, checklist=None)
    assert bigger < budget
    # A tiny window still yields the floor budget (chunker stays usable).
    tiny = slice_budget_tokens(
        Config(max_context_tokens=1_000, max_tokens=8_192), None, None
    )
    assert tiny >= 2_000


def test_compacted_messages_preserves_originals() -> None:
    data = serialize_messages(
        [
            user_msg("task one"),
            text_response("w"),
            user_msg(f"{COMPACTION_SUMMARY_PREFIX}\nold summary"),
        ]
    )
    result = compacted_messages(data, "new summary")
    assert len(result) == 2
    assert result[0] == data[0]
    assert is_summary_message(result[1])
    content = result[1]["parts"][0]["content"]
    assert content.startswith(COMPACTION_SUMMARY_PREFIX)
    assert "new summary" in content


# -- compaction of a merged request (pydantic-ai folds the summary into the
#    original-prompt request on resume, via _merge_consecutive_messages) ------


def merged_request() -> dict:
    """What the transcript looks like after a compaction + resume: one
    request whose user-prompt parts are [original, summary]."""
    return serialize_messages(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content="long task"),
                    UserPromptPart(content=f"{COMPACTION_SUMMARY_PREFIX}\nold summary"),
                ]
            )
        ]
    )[0]


def test_is_summary_message_merged_request() -> None:
    assert is_summary_message(merged_request())


def test_split_original_prompts_merged_request_keeps_original_only() -> None:
    merged = merged_request()
    preserved, rest = split_original_prompts([merged])
    assert len(preserved) == 1
    contents = [
        p["content"] for p in preserved[0]["parts"] if p["part_kind"] == "user-prompt"
    ]
    assert contents == ["long task"]
    assert not any(
        isinstance(p, dict) and "[context summary]" in str(p.get("content", ""))
        for p in preserved[0]["parts"]
    )
    # The merged request stays in rest (for the nothing-to-summarize guard),
    # but its summary must not survive into the next compaction.
    assert rest == [merged]


def test_extract_previous_summary_merged_request() -> None:
    assert extract_previous_summary([merged_request()]) == "old summary"


def test_render_history_skips_summary_part_of_merged_request() -> None:
    text = render_history_for_summary([merged_request()])
    assert "[user]\nlong task" in text
    assert "old summary" not in text


def test_compacted_messages_drops_stale_summary_of_merged_request() -> None:
    """Second compaction of the same transcript: the summary must be
    REPLACED, not accumulated (the compaction livelock regression)."""
    result = compacted_messages([merged_request()], "new summary")
    prompt_contents = [
        p["content"]
        for m in result
        for p in m["parts"]
        if p["part_kind"] == "user-prompt"
    ]
    assert prompt_contents[0] == "long task"
    summaries = [c for c in prompt_contents if c.startswith(COMPACTION_SUMMARY_PREFIX)]
    assert len(summaries) == 1
    assert "new summary" in summaries[0]
    assert "old summary" not in summaries[0]
    assert extract_previous_summary(result) == "new summary"


def test_compacted_messages_drops_request_reduced_to_summary() -> None:
    """A request holding nothing but summary parts vanishes instead of
    leaving an empty husk in the compacted transcript."""
    pure_summary = serialize_messages(
        [user_msg(f"{COMPACTION_SUMMARY_PREFIX}\nold summary")]
    )
    result = compacted_messages(pure_summary, "new summary")
    assert [is_summary_message(m) for m in result] == [True]
    assert "new summary" in result[0]["parts"][0]["content"]


# -- mechanical compaction floor (no LLM call) ---------------------------------


def _msg_tokens(msg: dict) -> int:
    """The same conservative wire-size estimate the mechanical floor uses."""
    return len(json.dumps(msg, ensure_ascii=False)) // 4


def _has_returns(msg: dict) -> bool:
    return any(
        isinstance(p, dict) and p.get("part_kind") == "tool-return"
        for p in msg.get("parts", [])
    )


def _has_calls(msg: dict) -> bool:
    return any(
        isinstance(p, dict) and p.get("part_kind") == "tool-call"
        for p in msg.get("parts", [])
    )


def test_prompt_fits_context() -> None:
    cfg = Config(project_root=None, max_context_tokens=1000, max_tokens=100)
    # 250 (prompt) + 1 (system) + 100 (output reserve) ≤ 1000.
    assert prompt_fits_context("x" * 1000, "s", cfg)
    assert not prompt_fits_context("x" * 100_000, "s", cfg)
    # An unknown window (≤ 0) counts as fitting — the LLM path is tried.
    unknown = Config(project_root=None, max_context_tokens=0)
    assert prompt_fits_context("x" * 10_000_000, "s", unknown)


def test_mechanical_compaction_noop_when_everything_fits() -> None:
    data = serialize_messages(
        [user_msg("task"), text_response("early work"), text_response("recent work")]
    )
    result = mechanical_compaction_messages(data, token_budget=10**9)
    assert [m["kind"] for m in result] == [
        "request",
        "response",
        "response",
        "request",
    ]
    assert result[0] == data[0]
    assert is_summary_message(result[-1])
    note = result[-1]["parts"][0]["content"]
    assert "dropped mechanically" in note
    # The note request closes the transcript: resumable as a continuation.
    assert ends_with_open_request(result)


def test_mechanical_compaction_trims_middle_within_budget() -> None:
    big = "x" * 40_000  # ≈10k tokens
    data = serialize_messages(
        [user_msg("task"), text_response(big), text_response("recent work")]
    )
    result = mechanical_compaction_messages(data, token_budget=6_000)
    # The original prompt and the LAST messages are kept, the heavy middle
    # response is gone, the note rides last.
    assert [m["kind"] for m in result] == ["request", "response", "request"]
    recent = [p["content"] for p in result[1]["parts"] if p["part_kind"] == "text"]
    assert recent == ["recent work"]
    note = result[-1]["parts"][0]["content"]
    assert big not in json.dumps(result)
    assert "dropped mechanically" in note


def test_mechanical_compaction_never_orphans_tool_returns() -> None:
    big_args = {"command": "q" * 4000}  # ≈1000 tokens as a call response
    big_result = "z" * 800  # ≈200 tokens as a return
    data = serialize_messages(
        [
            user_msg("task"),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name="run_bash", args=big_args, tool_call_id="c1")
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="run_bash",
                        content=big_result,
                        tool_call_id="c1",
                    )
                ]
            ),
        ]
    )
    preserved, rest = split_original_prompts(data)
    # Budget that fits exactly the last (return) request plus the stub, but
    # NOT the big call response before it.
    stub_tokens = _msg_tokens(mechanical_compaction_messages(data, 10**9)[-1])
    budget = (
        sum(_msg_tokens(m) for m in preserved)
        + stub_tokens
        + _msg_tokens(rest[-1])
        + 10
    )
    result = mechanical_compaction_messages(data, budget)
    # The orphan return (its call response was trimmed) is dropped with the
    # cut: no tool return ever appears without its call right before it.
    for i, m in enumerate(result):
        if _has_returns(m):
            assert i > 0 and _has_calls(result[i - 1])
    assert not any(_has_calls(m) or _has_returns(m) for m in result)
    assert [m["kind"] for m in result] == ["request", "request"]


def test_mechanical_compaction_keeps_intact_tool_pair() -> None:
    data = serialize_messages(
        [
            user_msg("task"),
            text_response("middle work"),
            tool_call_response("c1"),
            tool_return_msg("c1"),
        ]
    )
    result = mechanical_compaction_messages(data, token_budget=10**9)
    for i, m in enumerate(result):
        if _has_returns(m):
            assert i > 0 and _has_calls(result[i - 1])
    # The call→return pair survived the cut, the stub closes the transcript.
    assert _has_returns(result[-2]) and _has_calls(result[-3])
    assert is_summary_message(result[-1])


def test_mechanical_compaction_embeds_previous_summary() -> None:
    """The previous summary must survive verbatim: it is embedded in the
    note (the next compaction's previous summary), so the anchored chain
    never loses the condensed history."""
    data = serialize_messages(
        [
            user_msg(f"{COMPACTION_SUMMARY_PREFIX}\nold summary"),
            text_response("w"),
        ]
    )
    result = mechanical_compaction_messages(data, token_budget=10**9)
    note = result[-1]["parts"][0]["content"]
    assert "<previous-summary>\nold summary\n</previous-summary>" in note
    extracted = extract_previous_summary(result)
    assert extracted is not None
    assert extracted == note.split("\n", 1)[1].strip()
    assert "old summary" in extracted


# -- AGENTS.md loading ------------------------------------------------------------


def test_load_agents_md_present(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Rules\nbe nice")
    assert load_agents_md(tmp_path) == "# Rules\nbe nice"


def test_load_agents_md_missing(tmp_path: Path) -> None:
    assert load_agents_md(tmp_path) == ""


# -- tool error handling -------------------------------------------------------------


@pytest.fixture
def deps(project_root: Path) -> AgentDeps:
    manager = TaskManager(project_root)
    task = manager.create("t", "x")
    return AgentDeps(
        root=project_root,
        manager=manager,
        task_id=task.id,
        scratch=agent_mod.tools.scratch_dir(task.id),
    )


def test_log_violation_appends_event(deps: AgentDeps, project_root: Path) -> None:
    agent_mod._log_violation(deps, "write_file", "denied!")
    task = deps.manager.get(deps.task_id)
    assert len(task.events) == 1
    assert task.events[0].kind == "permission"
    assert task.events[0].text == "write_file: denied!"
    # Persisted.
    stored = agent_mod.TaskManager(project_root).get(task.id)
    assert stored.events[0].text == "write_file: denied!"


async def test_run_tool_logged_converts_violation(
    deps: AgentDeps,
) -> None:
    def violating_write(root: Path, path: str, content: str) -> str:
        raise PathViolation("denied")

    with pytest.raises(ModelRetry):
        await agent_mod._run_tool_logged(
            deps, "write_file", violating_write, deps.root, "/etc/x", "data"
        )
    task = deps.manager.get(deps.task_id)
    assert task.events[-1].text == "write_file: denied"


async def test_run_tool_converts_value_error_to_retry() -> None:
    def boom() -> None:
        raise ValueError("bad input")

    with pytest.raises(ModelRetry, match="bad input"):
        await agent_mod._run_tool(boom)


async def test_run_tool_passthrough_result() -> None:
    assert await agent_mod._run_tool(lambda: 42) == 42


async def test_run_bash_logged_converts_violation(deps: AgentDeps) -> None:
    with pytest.raises(ModelRetry):
        await agent_mod._run_bash_logged(deps, deps.root, "rm -rf /")
    task = deps.manager.get(deps.task_id)
    assert task.events[-1].kind == "permission"


# -- tester git push -----------------------------------------------------------------


def _git_init_repo(path: Path, *, bare: bool = False) -> None:
    import subprocess

    args = ["git", "init", "-q"]
    if bare:
        args.append("--bare")
    subprocess.run(args + [str(path)], check=True, capture_output=True)


def _git(repo: Path, *args: str) -> None:
    import subprocess

    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _git_out(repo: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def _tester_deps(root: Path) -> agent_mod.TesterDeps:
    manager = TaskManager(root)
    task = manager.create("t", "x")
    return agent_mod.TesterDeps(
        root=root,
        manager=manager,
        task_id=task.id,
        scratch=agent_mod.tools.scratch_dir(task.id),
    )


async def test_git_push_skips_without_remote(tmp_path: Path) -> None:
    _git_init_repo(tmp_path)
    deps = _tester_deps(tmp_path)
    result = await agent_mod._git_push(deps, tmp_path)
    assert "skipped" in result
    task = deps.manager.get(deps.task_id)
    assert [e.kind for e in task.events] == ["git"]
    assert "no remote" in task.events[0].text


async def test_git_push_failure_logs_event_but_never_fails(tmp_path: Path) -> None:
    _git_init_repo(tmp_path)
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "f.txt").write_text("x")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    # A remote that does not exist: the push must fail gracefully.
    _git(tmp_path, "remote", "add", "origin", str(tmp_path / "missing.git"))
    deps = _tester_deps(tmp_path)
    result = await agent_mod._git_push(deps, tmp_path)
    assert "Push failed" in result
    assert "not a verification failure" in result
    task = deps.manager.get(deps.task_id)
    assert len(task.events) == 1
    assert task.events[0].kind == "git"
    assert "fatal" in task.events[0].text


async def test_git_push_success_with_bare_remote(tmp_path: Path) -> None:
    _git_init_repo(tmp_path)
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "f.txt").write_text("x")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    origin = tmp_path / "origin.git"
    _git_init_repo(origin, bare=True)
    _git(tmp_path, "remote", "add", "origin", str(origin))
    deps = _tester_deps(tmp_path)
    result = await agent_mod._git_push(deps, tmp_path)
    assert "Pushed successfully" in result
    task = deps.manager.get(deps.task_id)
    assert task.events == []


def _remote_refs(origin: Path) -> str:
    return subprocess.run(
        ["git", "ls-remote", str(origin)], check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_commit_tool(ag) -> Any:
    """The registered git_commit tool of a built tester agent."""
    return next(
        t for t in ag._function_toolset.tools.values() if t.name == "git_commit"
    )


async def _run_git_commit(tool: Any, deps: Any, message: str) -> str:
    """Invoke the tool's raw function with a minimal stand-in RunContext
    (the tester's git_commit only touches `ctx.deps`)."""
    return cast(
        "str", await tool.function(cast("Any", SimpleNamespace(deps=deps)), message)
    )


async def test_git_commit_no_push_commits_locally(
    tmp_path: Path, fake_config: Config
) -> None:
    """The "Testing + Git commit" finalization: the tester's git_commit tool
    still creates the commit but never touches the remote — no push is
    attempted, no push events appear, and the remote stays empty."""
    _git_init_repo(tmp_path)
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "f.txt").write_text("x")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    origin = tmp_path / "origin.git"
    _git_init_repo(origin, bare=True)
    _git(tmp_path, "remote", "add", "origin", str(origin))

    deps = _tester_deps(tmp_path)
    (tmp_path / "g.txt").write_text("y")
    tester = agent_mod.build_tester_agent(fake_config, allow_push=False)
    tool = _git_commit_tool(tester)
    result = await _run_git_commit(tool, deps, "add g.txt")

    assert "Committed successfully" in result
    assert "never pushes" in result
    assert "Pushed" not in result
    # The commit landed locally…
    assert "add g.txt" in _git_log(tmp_path)
    # …but the remote branch was never created: no push happened.
    assert _remote_refs(origin) == ""
    task = deps.manager.get(deps.task_id)
    assert task.events == []
    # The no-push tool advertises its LOCAL-only behavior in its docstring.
    assert "LOCAL" in (tool.tool_def.description or "")
    assert "push the branch" not in (tool.tool_def.description or "")


async def test_git_commit_no_push_skips_push_even_with_remote(
    tmp_path: Path, fake_config: Config
) -> None:
    """Commit-only finalization on a repo without a remote: the commit still
    succeeds and no 'no remote' push note is logged (push is never tried)."""
    _git_init_repo(tmp_path)
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "f.txt").write_text("x")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")

    deps = _tester_deps(tmp_path)
    (tmp_path / "g.txt").write_text("y")
    tester = agent_mod.build_tester_agent(fake_config, allow_push=False)
    result = await _run_git_commit(_git_commit_tool(tester), deps, "add g.txt")
    assert "Committed successfully" in result
    assert "no remote" not in result
    task = deps.manager.get(deps.task_id)
    assert task.events == []


async def test_git_commit_push_variant_still_pushes(
    tmp_path: Path, fake_config: Config
) -> None:
    """Default finalization (commit + push): the same tool body pushes to the
    configured remote — the split must not have broken the default path."""
    _git_init_repo(tmp_path)
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "f.txt").write_text("x")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    origin = tmp_path / "origin.git"
    _git_init_repo(origin, bare=True)
    _git(tmp_path, "remote", "add", "origin", str(origin))

    deps = _tester_deps(tmp_path)
    (tmp_path / "g.txt").write_text("y")
    tester = agent_mod.build_tester_agent(fake_config)
    result = await _run_git_commit(_git_commit_tool(tester), deps, "add g.txt")
    assert "Committed successfully" in result
    assert "Pushed successfully" in result
    assert _remote_refs(origin) != ""


# -- tester set_verification(retry_here=True): the fix loop --------------------------


def _testing_deps(root: Path) -> agent_mod.TesterDeps:
    deps = _tester_deps(root)
    task = deps.manager.get(deps.task_id)
    task.pending_input.clear()  # the original prompt was consumed by round 1
    task.status = TaskStatus.TESTING
    deps.manager.storage.save(task)
    return deps


async def test_retry_failed_here_requeues_task(tmp_path: Path) -> None:
    deps = _testing_deps(tmp_path)
    task = deps.manager.get(deps.task_id)

    message = agent_mod._retry_failed_here(deps, "pytest fails: 1 broken")

    assert task.status is TaskStatus.QUEUED
    assert task.fix_rounds == 1
    assert task.verification is None
    assert task.completed_at is None
    assert [e.text for e in task.pending_input] == ["pytest fails: 1 broken"]
    assert "round 2/3" in message
    assert "do NOT call create_bug_report" in message
    # The re-queued task is promotable in its own (preserved) worktree.
    assert deps.manager.promote_next() is task
    # The event log carries the fix-loop note for the human.
    assert any("returned for another development round" in e.text for e in task.events)


async def test_retry_failed_here_caps_after_max_rounds(tmp_path: Path) -> None:
    deps = _testing_deps(tmp_path)
    task = deps.manager.get(deps.task_id)
    task.fix_rounds = 2
    deps.manager.storage.save(task)

    message = agent_mod._retry_failed_here(deps, "still broken")

    assert task.status is TaskStatus.COMPLETED
    assert task.verification is VerificationStatus.FAILED
    assert task.completed_at is not None
    # The counter is not incremented on the capped completion; the tester
    # must create the bug report now.
    assert task.fix_rounds == 2
    assert "create_bug_report" in message
    assert any("fix-round cap" in e.text for e in task.events)


async def test_retry_failed_here_respects_custom_cap(tmp_path: Path) -> None:
    deps = _testing_deps(tmp_path)
    deps.max_dev_rounds = 1
    task = deps.manager.get(deps.task_id)

    message = agent_mod._retry_failed_here(deps, "broken")

    assert task.status is TaskStatus.COMPLETED
    assert task.verification is VerificationStatus.FAILED
    assert "create_bug_report" in message


# -- tester create_bug_report: the report waits for the verified task ----------------


def _bug_report_tool(ag: Any) -> Any:
    """The registered create_bug_report tool of a built tester agent."""
    return next(
        t for t in ag._function_toolset.tools.values() if t.name == "create_bug_report"
    )


async def _run_bug_report(
    tool: Any, deps: Any, title: str, description: str, force: bool = False
) -> str:
    """Invoke the tool's raw function with a minimal stand-in RunContext
    (the tester's create_bug_report only touches `ctx.deps`). The tool is
    async (the duplicate guard runs the archive scan in a worker thread)."""
    return cast(
        "str",
        await tool.function(
            cast("Any", SimpleNamespace(deps=deps)), title, description, force
        ),
    )


async def test_tester_bug_report_depends_on_verified_task(
    tmp_path: Path, fake_config: Config
) -> None:
    """The tester's create_bug_report gates the new report on the verified
    task (the c8623e0c incident class): the report is not promoted until
    that task is COMPLETED and (under isolation) merged, so the fix agent's
    worktree reproduces the observed defect. The description records the
    origin; without isolation no branch is named."""
    deps = _tester_deps(tmp_path)
    tool = _bug_report_tool(agent_mod.build_tester_agent(fake_config))

    result = await _run_bug_report(
        tool, deps, "Broken X", "pytest fails: tests/test_x.py::test_y asserts 2 == 3"
    )

    report = next(t for t in deps.manager.tasks.values() if t.id != deps.task_id)
    assert report.depends_on == [deps.task_id]
    assert report.kind is TaskKind.BUG
    assert f"It waits for: #{deps.task_id}" in result
    prompt = report.pending_input[0].text
    assert f"Origin: filed by the tester of task #{deps.task_id}" in prompt
    assert "pytest fails" in prompt
    assert "branch" not in prompt


async def test_tester_bug_report_names_branch_under_isolation(
    tmp_path: Path, fake_config: Config
) -> None:
    """When the tester ran in the verified task's worktree, the origin note
    names that worktree and its drcode/<id> branch."""
    deps = _tester_deps(tmp_path)
    worktree = tmp_path / ".drcode" / "worktrees" / deps.task_id
    worktree.mkdir(parents=True)
    deps.root = worktree
    tool = _bug_report_tool(agent_mod.build_tester_agent(fake_config))

    await _run_bug_report(
        tool, deps, "Broken X", "pytest fails: tests/test_x.py::test_y asserts 2 == 3"
    )

    report = next(t for t in deps.manager.tasks.values() if t.id != deps.task_id)
    assert f"(its worktree, branch drcode/{deps.task_id})" in (
        report.pending_input[0].text
    )


# -- tester create_bug_report: mechanical stub-call validation ----------------


def _filed_reports(deps: Any) -> list[Any]:
    """Tasks in the manager other than the verified task itself."""
    return [t for t in deps.manager.tasks.values() if t.id != deps.task_id]


async def test_tester_bug_report_rejects_empty_description(
    tmp_path: Path, fake_config: Config
) -> None:
    """An empty/whitespace description is refused before anything is
    created; the ValueError reaches the model as a retry prompt."""
    deps = _tester_deps(tmp_path)
    tool = _bug_report_tool(agent_mod.build_tester_agent(fake_config))

    for description in ("", "   \n\t "):
        with pytest.raises(ModelRetry, match="description is empty"):
            await _run_bug_report(tool, deps, "Broken X", description)

    assert _filed_reports(deps) == []


async def test_tester_bug_report_rejects_short_description(
    tmp_path: Path, fake_config: Config
) -> None:
    """A description below the minimum length cannot say what is wrong,
    where and how it manifests — refused, nothing filed."""
    deps = _tester_deps(tmp_path)
    tool = _bug_report_tool(agent_mod.build_tester_agent(fake_config))

    with pytest.raises(ModelRetry, match="too short"):
        await _run_bug_report(tool, deps, "Broken X", "pytest fails")

    assert _filed_reports(deps) == []


async def test_tester_bug_report_rejects_stub_description(
    tmp_path: Path, fake_config: Config
) -> None:
    """Template-stub descriptions (normalized: case, whitespace, surrounding
    punctuation) are refused — the f668795c incident class: a degraded call
    with description='placeholder - will not file' must not file a task."""
    deps = _tester_deps(tmp_path)
    tool = _bug_report_tool(agent_mod.build_tester_agent(fake_config))

    for description in (
        "placeholder",
        "  PLACEHOLDER  ",
        "Placeholder - will not file.",
        "n/a",
        "TODO",
        "test",
        "bug report",
        "нет описания",
    ):
        with pytest.raises(ModelRetry, match="placeholder"):
            await _run_bug_report(tool, deps, "Broken X", description)

    assert _filed_reports(deps) == []


async def test_tester_bug_report_rejects_empty_title(
    tmp_path: Path, fake_config: Config
) -> None:
    """An empty title is refused too, nothing filed."""
    deps = _tester_deps(tmp_path)
    tool = _bug_report_tool(agent_mod.build_tester_agent(fake_config))

    with pytest.raises(ModelRetry, match="title is empty"):
        await _run_bug_report(
            tool,
            deps,
            "  ",
            "The CLI crashes with IndexError when run on an empty input file.",
        )

    assert _filed_reports(deps) == []


async def test_tester_bug_report_stub_title_with_real_description_passes(
    tmp_path: Path, fake_config: Config
) -> None:
    """The stub check applies to the description only: a title 'placeholder'
    over a real description is a legitimate report and is filed."""
    deps = _tester_deps(tmp_path)
    tool = _bug_report_tool(agent_mod.build_tester_agent(fake_config))
    description = (
        "The placeholder in the search dialog loses focus after the first "
        "keystroke, so the second character clears the query."
    )

    await _run_bug_report(tool, deps, "placeholder", description)

    reports = _filed_reports(deps)
    assert len(reports) == 1
    assert description in reports[0].pending_input[0].text


async def test_tester_bug_report_validation_is_mechanical_not_meaning(
    tmp_path: Path, fake_config: Config
) -> None:
    """The guard checks mechanical stub signs only: a short-looking but
    over-the-minimum description mentioning a stub word passes."""
    deps = _tester_deps(tmp_path)
    tool = _bug_report_tool(agent_mod.build_tester_agent(fake_config))

    await _run_bug_report(
        tool,
        deps,
        "Broken X",
        "TODO comments are not stripped from the built bundle (see spec).",
    )

    assert len(_filed_reports(deps)) == 1


# -- tester create_bug_report: the duplicate guard ----------------


async def test_tester_bug_report_blocks_on_similar_open_report(
    tmp_path: Path, fake_config: Config
) -> None:
    """The duplicate guard (the #6aa2a4ec / #927dde50 class): when the
    archive already holds an OPEN bug report sharing enough keywords, the
    tool creates NO task and returns the existing report's id/title/status
    with the force=true way out instead."""
    deps = _tester_deps(tmp_path)
    existing = deps.manager.create(
        "Fix ruff-format debt in six files", "ruff format --check .", kind=TaskKind.BUG
    )
    tool = _bug_report_tool(agent_mod.build_tester_agent(fake_config))

    result = await _run_bug_report(
        tool, deps, "Ruff format debt in six files", "the same ruff format debt"
    )

    tasks = [t for t in deps.manager.tasks.values() if t.id != deps.task_id]
    assert [t.id for t in tasks] == [existing.id]
    assert f"#{existing.id}" in result
    assert f"'{existing.title}'" in result
    assert "(status queued)" in result
    assert "force=true" in result


async def test_tester_bug_report_force_overrides_duplicate_guard(
    tmp_path: Path, fake_config: Config
) -> None:
    """force=true skips the guard: the report is created even when similar
    open reports exist (the model judged the match a false positive)."""
    deps = _tester_deps(tmp_path)
    deps.manager.create(
        "Fix ruff-format debt in six files", "ruff format --check .", kind=TaskKind.BUG
    )
    tool = _bug_report_tool(agent_mod.build_tester_agent(fake_config))

    result = await _run_bug_report(
        tool,
        deps,
        "Ruff format debt in six files",
        "a second batch of ruff format debt in the six files, separate defect",
        force=True,
    )

    reports = [t for t in deps.manager.tasks.values() if t.id != deps.task_id]
    assert len(reports) == 2  # the pre-existing one + the forced report
    assert "created at the head of the queue" in result


async def test_tester_bug_report_completed_report_is_no_duplicate(
    tmp_path: Path, fake_config: Config
) -> None:
    """A COMPLETED report is not a duplicate target — the defect was either
    fixed or dismissed, a fresh report must go through."""
    deps = _tester_deps(tmp_path)
    existing = deps.manager.create(
        "Fix ruff-format debt in six files", "ruff format --check .", kind=TaskKind.BUG
    )
    existing.status = TaskStatus.COMPLETED
    deps.manager.storage.save(existing)
    tool = _bug_report_tool(agent_mod.build_tester_agent(fake_config))

    result = await _run_bug_report(
        tool,
        deps,
        "Ruff format debt in six files",
        "the same ruff format debt re-observed after the earlier report closed",
    )

    reports = [t for t in deps.manager.tasks.values() if t.id != deps.task_id]
    assert [t.id for t in reports] == [existing.id, reports[-1].id]
    assert reports[-1].id != existing.id
    assert "created at the head of the queue" in result


async def test_retry_failed_here_ignores_non_testing_task(tmp_path: Path) -> None:
    deps = _tester_deps(tmp_path)
    task = deps.manager.get(deps.task_id)
    task.status = TaskStatus.ACTIVE
    deps.manager.storage.save(task)
    before = list(task.pending_input)

    message = agent_mod._retry_failed_here(deps, "late verdict")

    assert task.status is TaskStatus.ACTIVE
    assert task.fix_rounds == 0
    assert task.pending_input == before
    assert "no longer being tested" in message


async def test_retry_failed_here_ignores_redeveloping_task(tmp_path: Path) -> None:
    """A duplicate retry_here verdict arriving after the task already
    returned to development (incident 490d5652: a stale mid-round verdict
    was reported as "task completed" and spawned a false bug report) must
    be ignored — the tool result says so and never claims completion."""
    deps = _tester_deps(tmp_path)
    task = deps.manager.get(deps.task_id)
    task.pending_input.clear()
    task.fix_rounds = 1  # round 2/3 is already in development
    task.status = TaskStatus.QUEUED
    deps.manager.storage.save(task)
    before = list(task.pending_input)

    message = agent_mod._retry_failed_here(deps, "late duplicate verdict")

    assert task.status is TaskStatus.QUEUED
    assert task.fix_rounds == 1
    assert task.pending_input == before
    assert task.completed_at is None
    assert "no longer being tested" in message
    # The model must not read a completion (and start a bug report for a
    # fix-round cap that was never reached).
    assert "task completed" not in message


# -- research agent: tracked-file commit (git_commit_docs) ---------------------------


def _research_deps(root: Path) -> agent_mod.AgentDeps:
    manager = TaskManager(root)
    task = manager.create("docs", "write docs", kind=TaskKind.RESEARCH)
    return agent_mod.AgentDeps(
        root=root,
        manager=manager,
        task_id=task.id,
        scratch=agent_mod.tools.scratch_dir(task.id),
    )


def _committed_files(repo: Path) -> list[str]:
    import subprocess

    out = subprocess.run(
        ["git", "show", "--name-only", "--format="],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.split()


def _git_status(repo: Path, path: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", "status", "--porcelain", path],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _git_log(repo: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "log", "--oneline"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    ).stdout


async def test_git_commit_docs_commits_only_tracked_files(tmp_path: Path) -> None:
    _git_init_repo(tmp_path)
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    # A pre-existing tracked file the agent did NOT touch, left dirty, plus
    # a committed baseline so the dirty state is real.
    (tmp_path / "other.txt").write_text("original")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    (tmp_path / "other.txt").write_text("changed by someone else")
    # The agent created a doc and edited another one.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("guide")
    (tmp_path / "README.md").write_text("# readme")
    deps = _research_deps(tmp_path)
    deps.manager.get(deps.task_id).edited_files[:] = [
        "docs/guide.md",
        "README.md",
        "docs/guide.md",  # duplicate must collapse
    ]
    origin = tmp_path / "origin.git"
    _git_init_repo(origin, bare=True)
    _git(tmp_path, "remote", "add", "origin", str(origin))

    result = await agent_mod._commit_tracked_files(deps, tmp_path, "docs: add guide")

    assert "Committed successfully" in result
    assert "Pushed successfully" in result
    assert sorted(_committed_files(tmp_path)) == ["README.md", "docs/guide.md"]
    # The unrelated dirty change was NOT committed.
    assert " M other.txt" in _git_status(tmp_path, "other.txt")
    # Tracked list is cleared after a successful commit.
    assert deps.manager.get(deps.task_id).edited_files == []


async def test_git_commit_docs_local_policy_never_pushes(tmp_path: Path) -> None:
    """Research git policy local: the commit happens, the push never —
    even with a remote configured the result says the commit is local."""
    _git_init_repo(tmp_path)
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("# readme")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("guide")
    deps = _research_deps(tmp_path)
    deps.manager.get(deps.task_id).edited_files[:] = ["docs/guide.md"]
    origin = tmp_path / "origin.git"
    _git_init_repo(origin, bare=True)
    _git(tmp_path, "remote", "add", "origin", str(origin))
    _git(tmp_path, "push", "-q", "origin", "HEAD")
    branch = _git_out(tmp_path, "branch", "--show-current").strip()
    remote_before = _git_out(tmp_path, "ls-remote", "origin", branch).split()[0]

    result = await agent_mod._commit_tracked_files(
        deps, tmp_path, "docs: add guide", allow_push=False
    )

    assert "Committed successfully" in result
    assert "Commit is local — this project never pushes." in result
    assert "Pushed successfully" not in result
    # The commit landed locally, the remote branch did not move.
    local_head = _git_out(tmp_path, "rev-parse", "HEAD").strip()
    remote_after = _git_out(tmp_path, "ls-remote", "origin", branch).split()[0]
    assert local_head != remote_before
    assert remote_after == remote_before
    assert sorted(_committed_files(tmp_path)) == ["docs/guide.md"]


async def test_git_commit_docs_without_tracked_files(tmp_path: Path) -> None:
    _git_init_repo(tmp_path)
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "f.txt").write_text("x")
    deps = _research_deps(tmp_path)

    result = await agent_mod._commit_tracked_files(deps, tmp_path, "docs")

    assert "Nothing to commit" in result
    assert "not created or edited" in result
    # Nothing was committed.
    assert _git_log(tmp_path).strip() == ""


async def test_git_commit_docs_not_a_repo(tmp_path: Path) -> None:
    deps = _research_deps(tmp_path)
    deps.manager.get(deps.task_id).edited_files[:] = ["README.md"]
    result = await agent_mod._commit_tracked_files(deps, tmp_path, "docs")
    assert "Not a git repository" in result


async def test_git_commit_docs_empty_message(tmp_path: Path) -> None:
    deps = _research_deps(tmp_path)
    deps.manager.get(deps.task_id).edited_files[:] = ["README.md"]
    result = await agent_mod._commit_tracked_files(deps, tmp_path, "  ")
    assert "Empty commit message" in result


def test_track_edited_file(tmp_path: Path) -> None:
    deps = _research_deps(tmp_path)
    # Inside the project root: recorded project-relative, deduplicated.
    agent_mod._track_edited_file(deps, tmp_path / "docs" / "guide.md")
    agent_mod._track_edited_file(deps, tmp_path / "docs" / "guide.md")
    assert deps.manager.get(deps.task_id).edited_files == ["docs/guide.md"]
    # /tmp scratch file: never tracked.
    agent_mod._track_edited_file(deps, Path("/tmp/scratch/x.md"))
    assert deps.manager.get(deps.task_id).edited_files == ["docs/guide.md"]
    # Persisted with the task.
    reloaded = TaskManager(tmp_path).get(deps.task_id)
    assert reloaded is not None
    assert reloaded.edited_files == ["docs/guide.md"]


def test_track_edited_file_ignores_non_research(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    task = manager.create("f", "x", kind=TaskKind.FEATURE)
    deps = agent_mod.AgentDeps(
        root=tmp_path,
        manager=manager,
        task_id=task.id,
        scratch=agent_mod.tools.scratch_dir(task.id),
    )
    agent_mod._track_edited_file(deps, tmp_path / "README.md")
    assert manager.get(task.id).edited_files == []


def test_git_commit_docs_tool_only_for_research(fake_config: Config) -> None:
    from drcode.agent import build_agent

    research = build_agent(fake_config, kind=TaskKind.RESEARCH)
    assert "git_commit_docs" in _tool_names(research)
    for kind in (TaskKind.FEATURE, TaskKind.BUG):
        ag = build_agent(fake_config, kind=kind)
        assert "git_commit_docs" not in _tool_names(ag)


def test_git_commit_docs_tool_unregistered_under_off(fake_config: Config) -> None:
    """Research git policy off: the tool is not registered at all — the
    model cannot commit even if it tries (the prompt teaches no commit
    step either)."""
    from drcode.agent import build_agent

    research = build_agent(
        fake_config, kind=TaskKind.RESEARCH, research_git=GitPolicy.OFF
    )
    assert "git_commit_docs" not in _tool_names(research)
    assert "search_task_history" in _tool_names(research)  # others stay
    instructions = _instructions_text(research)
    assert "git_commit_docs" not in instructions


def test_git_commit_docs_tool_registered_under_local(fake_config: Config) -> None:
    """Research git policy local: the tool stays registered (the commit
    happens), only the push is dropped."""
    from drcode.agent import build_agent

    research = build_agent(
        fake_config, kind=TaskKind.RESEARCH, research_git=GitPolicy.LOCAL
    )
    assert "git_commit_docs" in _tool_names(research)


def test_search_task_history_tool_only_for_research(fake_config: Config) -> None:
    from drcode.agent import build_agent, build_tester_agent

    research = build_agent(fake_config, kind=TaskKind.RESEARCH)
    assert "search_task_history" in _tool_names(research)
    for kind in (TaskKind.FEATURE, TaskKind.BUG):
        ag = build_agent(fake_config, kind=kind)
        assert "search_task_history" not in _tool_names(ag)
    tester = build_tester_agent(fake_config)
    assert "search_task_history" not in _tool_names(tester)


def test_research_prompt_documents_task_search() -> None:
    from drcode.agent import RESEARCH_SYSTEM_PROMPT

    assert "search_task_history" in RESEARCH_SYSTEM_PROMPT


def test_research_prompt_documents_commit_flow() -> None:
    """The default research agent (git policy push) carries the push
    fragment teaching git_commit_docs; the OFF policy strips the whole
    git section from the prompt."""
    from drcode.agent import RESEARCH_SYSTEM_PROMPT, load_system_prompt

    assert "Documentation assignments" not in RESEARCH_SYSTEM_PROMPT
    push = load_system_prompt(TaskKind.RESEARCH, research_git=GitPolicy.PUSH)
    assert "git_commit_docs" in push
    assert "MUST finish the task by calling git_commit_docs" in push
    off = load_system_prompt(TaskKind.RESEARCH, research_git=GitPolicy.OFF)
    assert "git_commit_docs" not in off


# -- agent construction (no network calls) ------------------------------------------


@pytest.fixture
def fake_config() -> Config:
    return Config(project_root=Path.cwd(), model="test-model", api_key="sk-test")


def _tool_names(ag) -> set[str]:
    return {t.name for t in ag._function_toolset.tools.values()}


def _instructions_text(ag) -> str:
    # Dynamic instruction hooks are functions recomputed per request —
    # only the literal (static) parts carry prompt text.
    return "\n".join(
        si.instruction for si in ag._instructions if isinstance(si.instruction, str)
    )


def test_build_agent_instructions_by_kind(fake_config: Config) -> None:
    from drcode.agent import build_agent
    from drcode.prompts import load_system_prompt

    for kind in (TaskKind.FEATURE, TaskKind.BUG, TaskKind.RESEARCH):
        ag = build_agent(fake_config, kind=kind)
        # build_agent without a root substitutes the neutral marker, exactly
        # like load_system_prompt(kind) does.
        assert load_system_prompt(kind) in _instructions_text(ag)
    # All kinds share the same tool set.
    ag = build_agent(fake_config, kind=TaskKind.BUG)
    assert {
        "read_file",
        "write_file",
        "edit_file",
        "glob_files",
        "grep_search",
        "run_bash",
        "ask_clarifying_question",
    } <= _tool_names(ag)


def test_ask_clarifying_question_options_discipline(fake_config: Config) -> None:
    """The shared ask_clarifying_question tool description teaches the
    options-construction rule: `options` carries only what follows from the
    request, initiative goes through the separate `proposals` parameter
    (no textual markers — the separation is structural), and every
    option/proposal stays a short imperative label."""
    from drcode.agent import build_agent

    for kind in (TaskKind.FEATURE, TaskKind.BUG, TaskKind.RESEARCH):
        ag = build_agent(fake_config, kind=kind)
        tool = ag._function_toolset.tools["ask_clarifying_question"]
        # Whitespace-normalized: the docstring's line wrapping must not
        # decide where the assertions can look.
        desc = " ".join((tool.description or "").split())
        assert (
            "`proposals` is an optional list of options you propose on your OWN" in desc
        )
        assert "NEVER mix initiative into `options`" in desc
        # The old marker convention is dead: mentioned only as a ban.
        assert (
            'textual markers like "(agent\'s suggestion, not from your request)"'
            in desc
        )
        assert "the separation is structural" in desc
        # Brevity discipline.
        assert "SHORT imperative label" in desc
        # Positive trigger: choices between continuations — including the
        # agent's own initiative proposals — go through THIS tool.
        assert "choose between continuations" in desc
        assert "including your own initiative proposals" in desc
        assert "always better than prose" in desc


def _clarify_deps(project_root: Path) -> agent_mod.AgentDeps:
    manager = TaskManager(project_root)
    task = manager.create("research", "plan it", kind=TaskKind.RESEARCH)
    return agent_mod.AgentDeps(
        root=project_root,
        manager=manager,
        task_id=task.id,
        scratch=agent_mod.tools.scratch_dir(task.id),
    )


def _clarify_tool(fake_config: Config, deps: agent_mod.AgentDeps):
    from drcode.agent import build_agent

    ag = build_agent(fake_config, kind=TaskKind.RESEARCH, root=deps.root)
    return ag._function_toolset.tools["ask_clarifying_question"].function


def test_ask_clarifying_question_batch_saves_all_questions(
    fake_config: Config, project_root: Path
) -> None:
    """The batched call stores EVERY question in the task metadata — the
    old one-call-per-question flow overwrote task.clarification and only
    the last question survived."""
    from drcode.agent_build import ClarifyQuestion

    deps = _clarify_deps(project_root)
    ask = _clarify_tool(fake_config, deps)
    out = ask(
        cast("Any", SimpleNamespace(deps=deps)),
        questions=[
            ClarifyQuestion(question="Какая БД?", options=["Postgres", "SQLite"]),
            ClarifyQuestion(question="Завести задачу?", proposals=["Да", "Нет"]),
        ],
    )
    assert "end your turn" in out
    task = deps.manager.get(deps.task_id)
    assert task.status is TaskStatus.AWAITING_CLARIFICATION
    c = task.clarification
    assert c is not None and len(c.questions) == 2
    assert c.questions[0].question == "Какая БД?"
    assert c.questions[0].options == ["Postgres", "SQLite"]
    assert c.questions[1].agent_proposals == ["Да", "Нет"]
    # Persisted in the metadata file, not only in memory.
    raw = json.loads(
        (project_root / ".drcode" / "tasks" / f"{task.id}.json").read_text()
    )
    assert [q["question"] for q in raw["clarification"]["questions"]] == [
        "Какая БД?",
        "Завести задачу?",
    ]


def test_ask_clarifying_question_proxies_multiple_flag(
    fake_config: Config, project_root: Path
) -> None:
    """The `multiple` flag rides into the stored ClarificationItem; the
    default is False (old calls stay single-choice)."""
    from drcode.agent_build import ClarifyQuestion

    deps = _clarify_deps(project_root)
    ask = _clarify_tool(fake_config, deps)
    ask(
        cast("Any", SimpleNamespace(deps=deps)),
        questions=[
            ClarifyQuestion(question="q1?", options=["a", "b"]),
            ClarifyQuestion(
                question="q2?", options=["x", "y"], proposals=["z"], multiple=True
            ),
        ],
    )
    c = deps.manager.get(deps.task_id).clarification
    assert c is not None and len(c.questions) == 2
    assert c.questions[0].multiple is False
    assert c.questions[1].multiple is True


def test_ask_clarifying_question_refused_while_pending(
    fake_config: Config, project_root: Path
) -> None:
    """A second call while a clarification is already pending raises
    ModelRetry and leaves the pending package untouched (plus a service
    event for the chat log)."""
    from drcode.agent_build import ClarifyQuestion

    deps = _clarify_deps(project_root)
    ask = _clarify_tool(fake_config, deps)
    ctx = cast("Any", SimpleNamespace(deps=deps))
    ask(
        ctx,
        questions=[
            ClarifyQuestion(question="q1?", options=["a"]),
            ClarifyQuestion(question="q2?", proposals=["p"]),
        ],
    )
    before = deps.manager.get(deps.task_id).clarification
    with pytest.raises(ModelRetry, match="already pending"):
        ask(ctx, questions=[ClarifyQuestion(question="late?")])
    task = deps.manager.get(deps.task_id)
    assert task.clarification is before
    assert len(task.clarification.questions) == 2  # type: ignore[union-attr]
    assert task.clarification.questions[0].question == "q1?"  # type: ignore[union-attr]
    # The refusal is visible in the chat's service events.
    assert any("refused" in e.text and "already pending" in e.text for e in task.events)


def test_ask_clarifying_question_empty_questions_refused(
    fake_config: Config, project_root: Path
) -> None:
    """A call whose every question is blank gets a retry prompt instead of
    parking the task on an empty package."""
    from drcode.agent_build import ClarifyQuestion

    deps = _clarify_deps(project_root)
    ask = _clarify_tool(fake_config, deps)
    ctx = cast("Any", SimpleNamespace(deps=deps))
    with pytest.raises(ModelRetry, match="no questions"):
        ask(ctx, questions=[ClarifyQuestion(question="   ")])
    assert deps.manager.get(deps.task_id).clarification is None


def test_build_agent_appends_agents_md(fake_config: Config, project_root: Path) -> None:
    from drcode.agent import build_agent
    from drcode.prompts import load_system_prompt

    (project_root / "AGENTS.md").write_text("PROJECT SPECIFIC RULE")
    ag = build_agent(fake_config, kind=TaskKind.FEATURE, root=project_root)
    instructions = _instructions_text(ag)
    assert (
        load_system_prompt(TaskKind.FEATURE, project_root=project_root) in instructions
    )
    assert str(project_root) in instructions
    assert "Project instructions (AGENTS.md)" in instructions
    assert "PROJECT SPECIFIC RULE" in instructions


def test_build_agent_without_root_or_empty_agents_md(
    fake_config: Config, project_root: Path
) -> None:
    from drcode.agent import build_agent

    ag = build_agent(fake_config, kind=TaskKind.FEATURE)  # no root passed
    assert "Project instructions" not in _instructions_text(ag)
    (project_root / "AGENTS.md").write_text("   \n")  # whitespace only
    ag2 = build_agent(fake_config, kind=TaskKind.FEATURE, root=project_root)
    assert "Project instructions" not in _instructions_text(ag2)


def test_build_tester_agent_has_tester_tools(fake_config: Config) -> None:
    from drcode.agent import build_tester_agent

    ag = build_tester_agent(fake_config)
    tool_names = _tool_names(ag)
    assert {
        "read_file",
        "run_bash",
        "create_bug_report",
        "set_verification",
        "git_commit",
    } <= tool_names
    # Tester must not have write access: no write_file/edit_file tools.
    assert "write_file" not in tool_names
    assert "edit_file" not in tool_names


def test_build_tester_agent_registers_run_check(fake_config: Config) -> None:
    """The tester gets the project-environment check runner: bare PATH
    binaries report phantom failures (the 3ea596cb class), so Python
    tooling must go through run_check."""
    from drcode.agent import build_tester_agent

    ag = build_tester_agent(fake_config)
    assert "run_check" in _tool_names(ag)


def test_run_project_check_prefers_project_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_run_project_check` runs `python -m <module>` with the project's own
    .venv interpreter — never a PATH lookup — and reports the exit code.
    A venv python symlinked to the real interpreter proves the .venv plan
    was chosen (the output carries no uv run noise)."""
    from drcode.agent import _run_project_check

    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    out = _run_project_check(tmp_path, "pytest", ["--version"])
    assert out.startswith("exit=0")
    assert "pytest" in out


def test_run_project_check_pins_the_venv_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 9d29e63d failure class in the tester's run_check: the pip
    pyright package is a node wrapper that ignores the interpreter
    `-m pyright` ran under and probes VIRTUAL_ENV/PATH itself — when
    drcode is not launched via `uv run` that discovery lands on a foreign
    interpreter and reports phantom reportMissingImports. The env must be
    pinned to the plan's venv."""
    import drcode.agent_tools as at

    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("home = /fake\n")
    captured: dict[str, dict[str, str] | None] = {}

    def spy(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured["env"] = cast("dict[str, str] | None", kwargs.get("env"))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(at.subprocess, "run", spy)
    monkeypatch.setattr(
        at,
        "_project_python",
        lambda root, module: ([str(venv_bin / "python")], False),
    )
    out = at._run_project_check(tmp_path, "pyright", ["--version"])
    assert out.startswith("exit=0")
    env = captured["env"]
    assert env is not None
    assert env["VIRTUAL_ENV"] == str(tmp_path / ".venv")
    assert env["PATH"].startswith(str(venv_bin) + os.pathsep)


def test_run_project_check_missing_module_is_skipped(tmp_path: Path) -> None:
    """A module the project environment does not have is reported as
    unavailable (skip it) — no PATH fallback, no phantom-failure binary."""
    from drcode.agent import _run_project_check

    out = _run_project_check(tmp_path, "no_such_tool_xyz", ["--version"])
    assert "not available" in out
    assert "skipped" in out


def test_run_project_check_rejects_non_module() -> None:
    """run_check is for `python -m` only: a shell-ish name must raise the
    fixable ValueError (ModelRetry for the model), not reach a shell."""
    from drcode.agent import _run_project_check

    with pytest.raises(ValueError):
        _run_project_check(Path("/tmp"), "npx eslint", [])


def test_tester_prompt_teaches_retry_here() -> None:
    """Both tester templates must teach the retry_here criterion: a failure
    caused by the task's own changes returns the task to development
    (retry_here=True, no bug report); a foreign/environmental failure is
    filed as a separate bug report while the task itself PASSES
    (set_verification(success=True) + create_bug_report)."""
    from drcode.agent import (
        TESTER_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT_NO_COMMIT,
    )
    from drcode.prompts import TESTER_SYSTEM_PROMPT_COMMIT_NO_PUSH

    for prompt in (
        TESTER_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT_NO_COMMIT,
        TESTER_SYSTEM_PROMPT_COMMIT_NO_PUSH,
    ):
        assert "retry_here=True" in prompt
        assert "retry_here=False" in prompt
        assert "create_bug_report" in prompt
        # The cap escape must be taught too: after max failed rounds the
        # failure completes the task and the bug report is required again.
        assert "fix-round cap" in prompt
        # The 572b268d incident rule: an unrelated (pre-existing/foreign)
        # problem never fails the verified task — it passes and the problem
        # is filed separately.
        assert "must NOT be failed" in prompt


def test_build_tester_agent_substitutes_project_path(
    fake_config: Config, project_root: Path
) -> None:
    from drcode.agent import build_tester_agent
    from drcode.prompts import PROJECT_PATH_PLACEHOLDER

    ag = build_tester_agent(fake_config, root=project_root)
    instructions = _instructions_text(ag)
    assert str(project_root) in instructions
    assert PROJECT_PATH_PLACEHOLDER not in instructions
    # The commit-free variant carries the same header.
    ag2 = build_tester_agent(fake_config, allow_commit=False, root=project_root)
    assert str(project_root) in _instructions_text(ag2)


def test_build_tester_agent_appends_agents_md(
    fake_config: Config, project_root: Path
) -> None:
    from drcode.agent import build_tester_agent
    from drcode.prompts import load_tester_prompt

    (project_root / "AGENTS.md").write_text("PROJECT SPECIFIC RULE")
    ag = build_tester_agent(fake_config, root=project_root)
    instructions = _instructions_text(ag)
    assert load_tester_prompt(project_root=project_root) in instructions
    assert "Project instructions (AGENTS.md)" in instructions
    assert "PROJECT SPECIFIC RULE" in instructions
    # The commit-free variant gets the same rules.
    ag2 = build_tester_agent(fake_config, allow_commit=False, root=project_root)
    assert "PROJECT SPECIFIC RULE" in _instructions_text(ag2)


def test_build_tester_agent_without_root_or_empty_agents_md(
    fake_config: Config, project_root: Path
) -> None:
    from drcode.agent import build_tester_agent

    ag = build_tester_agent(fake_config)  # no root passed
    assert "Project instructions" not in _instructions_text(ag)
    (project_root / "AGENTS.md").write_text("   \n")  # whitespace only
    ag2 = build_tester_agent(fake_config, root=project_root)
    assert "Project instructions" not in _instructions_text(ag2)


def test_system_prompts_distinct() -> None:
    from drcode.agent import (
        BUG_SYSTEM_PROMPT,
        FEATURE_SYSTEM_PROMPT,
        RESEARCH_SYSTEM_PROMPT,
    )

    prompts = {BUG_SYSTEM_PROMPT, FEATURE_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT}
    assert len(prompts) == 3
    assert "fixing a BUG" in BUG_SYSTEM_PROMPT
    assert "implementing a FEATURE" in FEATURE_SYSTEM_PROMPT


def test_serialize_dump_is_json_safe() -> None:
    data = serialize_messages([user_msg("hi")])
    json.dumps(data)  # must not raise


# -- agent-created drafts (project setting agent_creates_drafts) ------------------


def test_create_report_task_full_by_default(project_root: Path) -> None:
    """Without the drafts setting the tools' shared body creates a full
    queued bug report with the historical tool-result text."""
    from drcode.agent_build import _create_report_task
    from drcode.models import TaskKind, TaskStatus

    manager = TaskManager(project_root)
    task = manager.create("t", "x")
    deps = agent_mod.AgentDeps(
        root=project_root,
        manager=manager,
        task_id=task.id,
        scratch=agent_mod.tools.scratch_dir(task.id),
    )
    created, text = _create_report_task(
        deps, "Broken thing", "it fails", kind=TaskKind.BUG, creator="tester"
    )
    assert created.status is TaskStatus.QUEUED
    assert created in manager.queued()
    assert created.events == []
    assert (
        f"Bug report #{created.id} '{created.title}' created at the head of the queue."
        == (text)
    )


def test_create_report_task_draft_with_setting(project_root: Path) -> None:
    """With agent_creates_drafts on, the shared creation body makes a
    DRAFT task: not in the queue, a service event tells the user to
    finish it, and the tool text says so."""
    from drcode.agent_build import _create_report_task
    from drcode.models import TaskKind, TaskStatus

    manager = TaskManager(project_root)
    task = manager.create("t", "x")
    deps = agent_mod.AgentDeps(
        root=project_root,
        manager=manager,
        task_id=task.id,
        scratch=agent_mod.tools.scratch_dir(task.id),
        agent_creates_drafts=True,
    )
    created, text = _create_report_task(
        deps, "Broken thing", "it fails", kind=TaskKind.BUG, creator="tester"
    )
    assert created.status is TaskStatus.DRAFT
    assert created not in manager.queued()
    promoted = manager.promote_next()
    assert promoted is not created
    assert created.status is TaskStatus.DRAFT
    assert [e.text for e in created.pending_input] == ["it fails"]
    # The draft carries a human-facing service event (never fed to the
    # model) and is persisted.
    assert any("draft" in e.text for e in created.events)
    stored = TaskManager(project_root).get(created.id)
    assert stored.status is TaskStatus.DRAFT
    assert stored.events
    assert "created as a draft" in text
    # The feature variant drafts the same way.
    feat, ftext = _create_report_task(
        deps, "New thing", "do it", kind=TaskKind.FEATURE, creator="research agent"
    )
    assert feat.status is TaskStatus.DRAFT
    assert "created as a draft" in ftext


def test_tester_deps_draft_flag(project_root: Path) -> None:
    """The tester deps carry the setting too (its create_bug_report uses
    the same shared body, including the fix-loop-end bug report)."""
    from drcode.agent_build import _create_report_task
    from drcode.models import TaskKind, TaskStatus

    manager = TaskManager(project_root)
    task = manager.create("t", "x")
    deps = agent_mod.TesterDeps(
        root=project_root,
        manager=manager,
        task_id=task.id,
        scratch=agent_mod.tools.scratch_dir(task.id),
        agent_creates_drafts=True,
    )
    created, text = _create_report_task(
        deps, "Env broken", "details", kind=TaskKind.BUG, creator="tester"
    )
    assert created.status is TaskStatus.DRAFT
    assert "created as a draft" in text


def test_cli_and_github_create_full_tasks(project_root: Path) -> None:
    """Regression: the headless CLI and the GitHub Issues import never
    create drafts — the drafts setting only governs the agents' tools."""
    from drcode.models import TaskKind, TaskStatus

    # The CLI path (manager.create without draft=).
    cli_manager = TaskManager(project_root)
    task = cli_manager.create("", "requested feature", kind=TaskKind.FEATURE)
    assert task.status is TaskStatus.QUEUED
    # The drafts setting itself defaults to off (and the CLI ignores it
    # even when on — the call site passes no draft flag at all).
    from drcode.settings import ProjectSettings, ProjectSettingsStore

    settings = ProjectSettingsStore(project_root)
    stored = ProjectSettings(agent_creates_drafts=True)
    settings.save(stored)
    assert settings.load().agent_creates_drafts is True
    task2 = cli_manager.create("", "another feature", kind=TaskKind.FEATURE)
    assert task2.status is TaskStatus.QUEUED


def test_set_task_dependencies_accepts_draft(project_root: Path) -> None:
    """A draft's dependency list stays editable (the same rule as for
    queued/held tasks): the guard in _set_task_dependencies allows it."""
    manager = TaskManager(project_root)
    dep = manager.create("dep task", "x")
    draft = manager.create("draft", "x", draft=True)
    deps = agent_mod.AgentDeps(
        root=project_root,
        manager=manager,
        task_id=draft.id,
        scratch=agent_mod.tools.scratch_dir(draft.id),
    )
    agent_mod._set_task_dependencies(deps, draft.id, [dep.id])
    assert draft.depends_on == [dep.id]
