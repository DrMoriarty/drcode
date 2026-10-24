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
    render_history_for_summary,
    resumable_history,
    sanitized_messages,
    serialize_messages,
    split_original_prompts,
)
from drcode.config import Config
from drcode.models import TaskKind, TaskStatus, VerificationStatus
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
    from drcode.agent import RESEARCH_SYSTEM_PROMPT

    assert "git_commit_docs" in RESEARCH_SYSTEM_PROMPT
    assert "MUST finish the task by calling git_commit_docs" in RESEARCH_SYSTEM_PROMPT


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


def test_tester_prompt_teaches_retry_here() -> None:
    """Both tester templates must teach the retry_here criterion: a failure
    caused by the task's own changes returns the task to development
    (retry_here=True, no bug report); a foreign/environmental failure
    completes it with a bug report (retry_here=False)."""
    from drcode.agent import (
        TESTER_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT_NO_COMMIT,
    )

    for prompt in (TESTER_SYSTEM_PROMPT, TESTER_SYSTEM_PROMPT_NO_COMMIT):
        assert "retry_here=True" in prompt
        assert "retry_here=False" in prompt
        assert "create_bug_report" in prompt
        # The cap escape must be taught too: after max failed rounds the
        # failure completes the task and the bug report is required again.
        assert "fix-round cap" in prompt


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
