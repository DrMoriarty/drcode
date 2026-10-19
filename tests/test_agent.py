"""Tests for drcode.agent: transcript handling, compaction, display, tools."""

from __future__ import annotations

import json
from pathlib import Path

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
    COMPACTION_SUMMARY_PREFIX,
    AgentDeps,
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
    render_history_for_summary,
    resumable_history,
    sanitized_messages,
    serialize_messages,
    split_original_prompts,
)
from drcode.config import Config
from drcode.models import TaskKind
from drcode.task_manager import TaskManager
from drcode.tools import PathViolation


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


# -- context tokens --------------------------------------------------------------


def test_last_context_tokens() -> None:
    data = [
        {
            "kind": "response",
            "usage": {
                "input_tokens": 100,
                "cache_read_tokens": 50,
                "cache_write_tokens": 25,
            },
        }
    ]
    assert last_context_tokens(data) == 175


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
    return AgentDeps(root=project_root, manager=manager, task_id=task.id)


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


# -- agent construction (no network calls) ------------------------------------------


@pytest.fixture
def fake_config() -> Config:
    return Config(project_root=Path.cwd(), model="test-model", api_key="sk-test")


def _tool_names(ag) -> set[str]:
    return {t.name for t in ag._function_toolset.tools.values()}


def _instructions_text(ag) -> str:
    return "\n".join(si.instruction for si in ag._instructions)


def test_build_agent_instructions_by_kind(fake_config: Config) -> None:
    from drcode.agent import (
        BUG_SYSTEM_PROMPT,
        FEATURE_SYSTEM_PROMPT,
        RESEARCH_SYSTEM_PROMPT,
        build_agent,
    )

    for kind, prompt in (
        (TaskKind.FEATURE, FEATURE_SYSTEM_PROMPT),
        (TaskKind.BUG, BUG_SYSTEM_PROMPT),
        (TaskKind.RESEARCH, RESEARCH_SYSTEM_PROMPT),
    ):
        ag = build_agent(fake_config, kind=kind)
        assert prompt in _instructions_text(ag)
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


def test_build_agent_appends_agents_md(fake_config: Config, project_root: Path) -> None:
    from drcode.agent import FEATURE_SYSTEM_PROMPT, build_agent

    (project_root / "AGENTS.md").write_text("PROJECT SPECIFIC RULE")
    ag = build_agent(fake_config, kind=TaskKind.FEATURE, root=project_root)
    instructions = _instructions_text(ag)
    assert FEATURE_SYSTEM_PROMPT in instructions
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
