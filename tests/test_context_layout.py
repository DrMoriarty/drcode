"""Tests for the context-layout estimation (context_layout.py) and the
context dialog it feeds (tui/ctx_dialog.py, opened by clicking the task
status bar's context counter)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic_ai import ModelRequest, ModelResponse
from pydantic_ai.messages import (
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from drcode.agent import serialize_messages
from drcode.config import Config
from drcode.context_layout import (
    AGENT_DEV,
    AGENT_RESOLVER,
    AGENT_TESTER,
    CHARS_PER_TOKEN,
    agent_transcript,
    analyze_transcript,
    build_report,
    coding_tool_schemas,
    cumulative_usage,
    estimate_tokens,
    last_agent,
    system_components_coding,
    system_components_resolver,
    system_components_tester,
    task_reports,
)
from drcode.context_layout import (
    resolver_tool_schemas as make_resolver_tools,
)
from drcode.context_layout import (
    tester_tool_schemas as make_tester_tools,
)
from drcode.models import TaskKind
from drcode.settings import GitPolicy
from drcode.tui.app import DrCodeApp
from drcode.tui.ctx_dialog import ContextDialog
from drcode.tui.pipeline import PipelineLayer


def _transcript() -> list[dict]:
    request = ModelRequest(
        parts=[
            UserPromptPart(content="fix the bug in main.py"),
        ]
    )
    response = ModelResponse(
        parts=[
            ThinkingPart(content="thinking about main.py"),
            TextPart(content="reading the file first"),
            ToolCallPart(
                tool_name="read_file",
                args=json.dumps({"path": "main.py", "offset": 1, "limit": 50}),
                tool_call_id="call-1",
            ),
        ],
        usage=RequestUsage(
            input_tokens=1000,
            output_tokens=100,
            cache_read_tokens=800,
            cache_write_tokens=200,
        ),
    )
    retry = ModelRequest(
        parts=[RetryPromptPart(content="path must exist", tool_call_id="call-1")]
    )
    tool_result = ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name="read_file",
                content="x" * 400,
                tool_call_id="call-1",
            )
        ]
    )
    return serialize_messages([request, response, retry, tool_result])


def test_estimate_tokens_is_chars_over_four() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("x" * 400) == 400 // CHARS_PER_TOKEN
    # Tiny texts still cost a token (rounding up).
    assert estimate_tokens("abcd") == 1


def test_system_components_split_base_agents_and_git_workflow(
    project_root: Path,
) -> None:
    (project_root / "AGENTS.md").write_text("# Project rules\n\nbe terse.\n")
    drcode_dir = project_root / ".drcode"
    drcode_dir.mkdir(exist_ok=True)
    (drcode_dir / "git-workflow.md").write_text("- never commit to main\n")
    parts = system_components_coding(TaskKind.FEATURE, project_root)
    labels = [label for label, _ in parts]
    assert labels == ["base template", "AGENTS.md", "git workflow"]
    base, agents_md, git_workflow = (text for _, text in parts)
    assert base is not None and "{project_path}" not in base
    assert "{project_path}" not in base
    assert agents_md is not None and "be terse." in agents_md
    assert git_workflow is not None and "never commit to main" in git_workflow
    # The bundled "Never commit to git" sentence is dropped when the
    # workflow exists (same rule as the real prompt assembly).
    assert "Never commit to" not in base

    # Without the files the components are absent (None) — the dialog
    # renders them as "absent".
    empty = system_components_coding(TaskKind.FEATURE, project_root.parent)
    assert empty[0][1] is not None
    assert empty[1][1] is None
    assert empty[2][1] is None


def test_system_components_tester_split_base_agents_and_git_workflow(
    project_root: Path,
) -> None:
    (project_root / "AGENTS.md").write_text("# Project rules\n\nbe terse.\n")
    drcode_dir = project_root / ".drcode"
    drcode_dir.mkdir(exist_ok=True)
    (drcode_dir / "git-workflow.md").write_text("- never commit to main\n")
    parts = system_components_tester(True, project_root)
    labels = [label for label, _ in parts]
    assert labels == ["base template", "AGENTS.md", "git workflow"]
    base, agents_md, git_workflow = (text for _, text in parts)
    assert base is not None and "{project_path}" not in base
    assert agents_md is not None and "be terse." in agents_md
    assert git_workflow is not None and "never commit to main" in git_workflow

    # Without the files the components are absent (None).
    empty = system_components_tester(True, project_root.parent)
    assert empty[0][1] is not None
    assert empty[1][1] is None
    assert empty[2][1] is None


def test_tool_schemas_follow_role_and_kind(project_root: Path) -> None:
    config = Config()
    feature = dict(coding_tool_schemas(config, TaskKind.FEATURE, root=project_root))
    bug = dict(coding_tool_schemas(config, TaskKind.BUG, root=project_root))
    research = dict(coding_tool_schemas(config, TaskKind.RESEARCH, root=project_root))
    # todo — FEATURE only; create_* — RESEARCH only.
    assert "todo_create" in feature and "todo_create" not in bug
    assert "create_bug_report" in research
    assert "create_bug_report" not in feature and "create_bug_report" not in bug
    # Common tools everywhere; each entry carries name + docstring + schema.
    for tools in (feature, bug, research):
        assert "read_file" in tools and "run_bash" in tools
        assert "read_file" in tools["read_file"]
    # Tester: git_commit only when commits are allowed.
    with_commit = dict(make_tester_tools(config, allow_commit=True, root=project_root))
    without_commit = dict(
        make_tester_tools(config, allow_commit=False, root=project_root)
    )
    assert "git_commit" in with_commit and "git_commit" not in without_commit
    assert "set_verification" in without_commit
    # Commit-without-push tester git policy: the tool stays registered
    # (its wire-text differs — LOCAL commit, no push promise).
    no_push = dict(
        make_tester_tools(
            config, allow_commit=True, allow_push=False, root=project_root
        )
    )
    assert "git_commit" in no_push
    assert "LOCAL" in no_push["git_commit"]
    assert "push the branch" not in no_push["git_commit"]
    # Research: git_commit_docs follows the research git policy — push by
    # default, still registered under local, gone under off.
    research_off = dict(
        coding_tool_schemas(
            config, TaskKind.RESEARCH, root=project_root, research_git=GitPolicy.OFF
        )
    )
    research_local = dict(
        coding_tool_schemas(
            config, TaskKind.RESEARCH, root=project_root, research_git=GitPolicy.LOCAL
        )
    )
    assert "git_commit_docs" not in research_off
    assert "git_commit_docs" in research_local
    assert "this project never pushes" in research_local["git_commit_docs"]


def test_system_components_research_git_policy(project_root: Path) -> None:
    """The context estimate's research prompt mirrors the policy wiring:
    the push fragment under push, the local fragment under local, neither
    under off."""
    push = system_components_coding(
        TaskKind.RESEARCH, project_root, research_git=GitPolicy.PUSH
    )
    local = system_components_coding(
        TaskKind.RESEARCH, project_root, research_git=GitPolicy.LOCAL
    )
    off = system_components_coding(
        TaskKind.RESEARCH, project_root, research_git=GitPolicy.OFF
    )
    push_base = push[0][1] or ""
    local_base = local[0][1] or ""
    off_base = off[0][1] or ""
    assert "pushes the branch when a remote is configured" in push_base
    assert "never push" in local_base
    assert "pushes the branch" not in local_base
    assert "git_commit_docs" not in off_base


def test_analyze_buckets_parts_and_finds_heaviest_results() -> None:

    stats = analyze_transcript(_transcript())
    assert stats.user_prompt_count == 1
    assert stats.user_prompts == estimate_tokens("fix the bug in main.py")
    # The ModelRetry prompt is counted, and NOT as a user message.
    assert stats.retry_prompt_count == 1
    assert stats.retry_prompts == estimate_tokens("path must exist")
    assert stats.generation == estimate_tokens("thinking about main.py") + (
        estimate_tokens("reading the file first")
    )
    assert stats.tool_calls == 1
    assert stats.tool_result_count == 1
    assert stats.tool_results == 400 // CHARS_PER_TOKEN
    assert not stats.compacted
    # Result label reconstructs the file path from the call args.
    assert stats.heaviest[0].label.startswith("read_file: main.py")
    # Exact usage of the single response, incl. the cache split.
    assert stats.usage.input_tokens == 1000
    assert stats.usage.output_tokens == 100
    assert stats.requests == 1
    assert stats.last_cache_read == 800
    assert stats.last_cache_write == 200


def test_analyze_detects_compaction_summary() -> None:
    from drcode.compaction import compacted_messages

    messages = compacted_messages(_transcript(), "the summary body")
    stats = analyze_transcript(messages)
    assert stats.compacted
    assert stats.summary == estimate_tokens("the summary body")
    # The original user prompt is preserved verbatim across compaction.
    assert stats.user_prompt_count == 1


def test_cumulative_usage_spans_transcripts() -> None:
    u, requests = cumulative_usage([_transcript(), _transcript()])
    assert u.input_tokens == 2000
    assert u.output_tokens == 200
    assert requests == 2


def test_build_report_sections_and_totals(project_root: Path) -> None:
    config = Config()
    report = build_report(
        "Development (CODING)",
        _transcript(),
        config,
        system_components=system_components_coding(TaskKind.FEATURE, project_root),
        tools=coding_tool_schemas(config, TaskKind.FEATURE, root=project_root),
        exact_tokens=1100,
    )
    titles = [s.title for s in report.sections]
    assert titles == [
        "system prompt",
        "tools (descriptions + schemas)",
        "transcript",
    ]
    assert report.exact_tokens == 1100
    assert report.exact_pct == round(1100 * 100 / config.max_context_tokens, 1)
    assert report.estimate_total == sum(s.total for s in report.sections)
    assert report.threshold_tokens == round(
        config.compaction_threshold * config.max_context_tokens / 100
    )
    assert report.headroom_tokens == (report.threshold_tokens or 0) - 1100
    # Every system-prompt component and tool is listed as its own line.
    system_labels = [l.label for l in report.sections[0].lines]
    assert system_labels == ["base template", "AGENTS.md", "git workflow"]
    assert any("todo_create" in l.label for l in report.sections[1].lines)
    # Retry prompts surface as their own transcript line.
    transcript_labels = [l.label for l in report.sections[2].lines]
    assert "retry prompts (ModelRetry)" in transcript_labels


def test_system_components_resolver_split_base_agents(project_root: Path) -> None:
    (project_root / "AGENTS.md").write_text("# Project rules\n\nbe terse.\n")
    parts = system_components_resolver(project_root)
    labels = [label for label, _ in parts]
    # The resolver prompt has no git-workflow fragment (see
    # build_resolver_agent) — base template + AGENTS.md only.
    assert labels == ["base template", "AGENTS.md"]
    base, agents_md = (text for _, text in parts)
    assert base is not None and "{project_path}" not in base
    assert agents_md is not None and "be terse." in agents_md

    empty = system_components_resolver(project_root.parent)
    assert empty[0][1] is not None
    assert empty[1][1] is None


def test_resolver_tool_schemas(project_root: Path) -> None:
    tools = dict(make_resolver_tools(Config(), root=project_root))
    # The resolver's read/inspect set plus its verdict tool.
    assert "set_resolution" in tools
    assert "read_file" in tools and "run_bash" in tools
    assert "resolved=True" in tools["set_resolution"]


def _round(stamp: datetime) -> list[dict]:
    """One minimal round transcript stamped with `stamp` (the resolver /
    tester rounds are ordered by the timestamps pydantic-ai serializes
    into the messages)."""
    return serialize_messages(
        [
            ModelRequest(
                parts=[UserPromptPart(content="go", timestamp=stamp)],
            ),
            ModelResponse(
                parts=[TextPart(content="done")],
                usage=RequestUsage(input_tokens=100, output_tokens=10),
                timestamp=stamp,
            ),
        ]
    )


def test_last_agent_heuristic() -> None:
    from drcode.models import Task

    t1 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    t2 = datetime(2024, 1, 1, 13, 0, tzinfo=UTC)
    t3 = datetime(2024, 1, 1, 14, 0, tzinfo=UTC)

    # Development only.
    task = Task(title="a task")
    task.messages = _transcript()
    assert last_agent(task) == AGENT_DEV

    # A tester round after development → tester.
    task.tester_messages = [_round(t1)]
    assert last_agent(task) == AGENT_TESTER

    # A resolver round after the tester round → resolver.
    task.resolver_messages = [_round(t2)]
    assert last_agent(task) == AGENT_RESOLVER

    # A NEW tester round after the resolver round → tester again.
    task.tester_messages = [_round(t1), _round(t3)]
    assert last_agent(task) == AGENT_TESTER

    # Resolver without rounds is never picked; dev with no rounds at all.
    empty = Task(title="fresh")
    assert last_agent(empty) == AGENT_DEV
    empty.resolver_messages = [_round(t1)]
    assert last_agent(empty) == AGENT_RESOLVER


def test_agent_transcript_picks_last_round() -> None:
    from drcode.models import Task

    task = Task(title="a task")
    task.messages = _transcript()
    late = _round(datetime(2024, 1, 1, tzinfo=UTC))
    early = _round(datetime(2023, 1, 1, tzinfo=UTC))
    task.tester_messages = [early, late]
    task.resolver_messages = [_round(datetime(2022, 1, 1, tzinfo=UTC))]
    assert agent_transcript(task, AGENT_DEV) is task.messages
    assert agent_transcript(task, AGENT_TESTER) == late
    assert agent_transcript(task, AGENT_RESOLVER) == task.resolver_messages[-1]
    empty = Task(title="fresh")
    assert agent_transcript(empty, AGENT_TESTER) == []


def test_task_reports_builds_both_transcripts(project_root: Path) -> None:
    from drcode.models import Task

    task = Task(title="a task")
    task.messages = _transcript()
    task.tester_messages = [_transcript()]
    dev_config = Config()
    tester_config = Config()
    resolver_config = Config()
    dev, tester, resolver, usage, requests = task_reports(
        task,
        dev_config,
        tester_config,
        resolver_config,
        "CODING",
        project_root,
        live_tokens=1200,
    )
    assert dev is not None and tester is not None
    assert dev.title == "Development (CODING)"
    assert dev.exact_tokens == 1200  # live reading wins over the transcript
    assert tester.title == "Testing (tester, round 1)"
    assert resolver is None
    assert usage.input_tokens == 2000 and requests == 2
    # Without any tester round there is no testing report.
    task.tester_messages = []
    _dev, tester2, _resolver2, _usage, _requests = task_reports(
        task, dev_config, tester_config, resolver_config, "CODING", project_root
    )
    assert tester2 is None


def test_task_reports_resolver_report_and_cumulative(project_root: Path) -> None:
    """The latest resolver round gets its own report against the RESOLVER
    role's config, and its transcript joins the whole-task cumulative
    usage."""
    from drcode.models import Task

    task = Task(title="a task")
    task.messages = _transcript()
    task.tester_messages = [_transcript()]
    task.resolver_messages = [_transcript()]
    dev, tester, resolver, usage, requests = task_reports(
        task, Config(), Config(), Config(), "CODING", project_root
    )
    assert dev is not None and tester is not None
    assert resolver is not None
    assert resolver.title == "Resolver (conflict resolution, round 1)"
    assert resolver.model == Config().model
    assert resolver.max_context_tokens == Config().max_context_tokens
    # The resolver toolset (its own section of the report) lists set_resolution.
    assert any(
        "set_resolution" in line.label
        for section in resolver.sections
        if section.title.startswith("tools")
        for line in section.lines
    )
    # Cumulative now spans all three transcripts.
    assert usage.input_tokens == 3000 and requests == 3
    # Without resolver rounds there is no resolver report and the
    # cumulative falls back to dev + tester.
    task.resolver_messages = []
    _dev, _tester, resolver2, usage2, requests2 = task_reports(
        task, Config(), Config(), Config(), "CODING", project_root
    )
    assert resolver2 is None
    assert usage2.input_tokens == 2000 and requests2 == 2


# -- dialog wiring ---------------------------------------------------------------


async def test_ctx_counter_opens_dialog_on_action(app: DrCodeApp) -> None:
    task = app.manager.create("contextful", "the original prompt")
    task.messages = _transcript()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Skip the first-start setup dialog (no model configured in tests).
        await pilot.press("escape")
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        right = app.query_one("#task-status-right")
        # The counter renders the 🧠 figure; the [@click=app.show_ctx_dialog]
        # markup that makes it a link is the same mechanism the task-id
        # mention links use (see test_blocks) and the dialog open below
        # exercises the action itself.
        assert "🧠" in str(right.render())
        app.action_show_ctx_dialog()
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, ContextDialog)
        body = dialog.query_one("#ctx-body")
        text = str(body.render())
        assert "Development (CODING)" in text
        assert "top-5 heaviest tool results" in text
        # ESC closes and returns to the main screen.
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ContextDialog)

        # A real mouse click on the counter opens the dialog too (the
        # [@click] markup fires the app action).
        await pilot.click("#task-status-right")
        await pilot.pause()
        assert isinstance(app.screen, ContextDialog)


async def test_ctx_dialog_without_context_notifies(app: DrCodeApp) -> None:
    app.manager.create("empty", "no messages yet")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_show_ctx_dialog()
        await pilot.pause()
        assert not isinstance(app.screen, ContextDialog)


async def test_action_show_ctx_dialog_serving_role_label(
    app: DrCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A research task's development report is labeled RESEARCH (the role
    serving its transcript)."""
    task = app.manager.create("docs question", "what does this repo do?")
    task.kind = TaskKind.RESEARCH
    task.messages = _transcript()
    seen: dict[str, str] = {}

    original = PipelineLayer._task_role  # plain function (staticmethod)

    def spy(task):
        role = original(task)
        seen["role"] = role.value
        return role

    monkeypatch.setattr(PipelineLayer, "_task_role", staticmethod(spy))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        app.action_show_ctx_dialog()
        await pilot.pause()
        assert seen["role"] == "research"
        assert isinstance(app.screen, ContextDialog)
        text = str(app.screen.query_one("#ctx-body").render())
        assert "Development (RESEARCH)" in text


async def test_ctx_dialog_tabs_per_agent(app: DrCodeApp) -> None:
    """With all three agents' transcripts present the dialog shows
    Development / Testing / Resolver tabs, opens on the last agent
    (resolver) and switching tabs swaps the report."""
    task = app.manager.create("conflicted", "the original prompt")
    task.messages = _transcript()
    task.tester_messages = [_round(datetime(2024, 1, 1, 12, tzinfo=UTC))]
    task.resolver_messages = [_round(datetime(2024, 1, 1, 13, tzinfo=UTC))]
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        app.action_show_ctx_dialog()
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, ContextDialog)
        from textual.widgets import Tabs

        tabs = dialog.query_one("#ctx-tabs", Tabs)
        # The last agent (the resolver round is the newest) is selected.
        assert tabs.active == "resolver"
        body = str(dialog.query_one("#ctx-body").render())
        assert "Resolver (conflict resolution, round 1)" in body
        assert "Development (CODING)" not in body
        # Switch to the development tab: the body swaps to the dev report.
        tabs.active = "dev"
        await pilot.pause()
        body = str(dialog.query_one("#ctx-body").render())
        assert "Development (CODING)" in body
        assert "Resolver (conflict resolution, round 1)" not in body
        # The whole-task cumulative spans all three transcripts.
        assert "3 request(s)" in body


async def test_ctx_dialog_single_report_has_no_tabs(app: DrCodeApp) -> None:
    task = app.manager.create("plain", "the original prompt")
    task.messages = _transcript()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        app.action_show_ctx_dialog()
        await pilot.pause()
        assert isinstance(app.screen, ContextDialog)
        assert not app.screen.query("#ctx-tabs")
        assert "Development (CODING)" in str(app.screen.query_one("#ctx-body").render())


async def test_ctx_counter_shows_active_tester(app: DrCodeApp) -> None:
    """A live tester reading flips the counter to the tester's figures:
    tokens from the live map, measured against the TESTER role's config —
    even though the dev transcript has its own (larger) reading."""
    task = app.manager.create("under test", "the original prompt")
    task.messages = _transcript()
    task.tester_messages = [_transcript()]
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        right = app.query_one("#task-status-right")
        # Static context: dev is the last agent.
        assert "🧠" in str(right.render())
        # A live tester reading: the counter switches to the tester round.
        app._live_ctx_tester[task.id] = 4000
        app.refresh_task_status()
        await pilot.pause()
        text = str(right.render())
        assert "4k" in text
        # The window shown is the tester role's config (200k default).
        assert "200k" in text
        # And the resolver's live reading wins over the tester's stale one.
        app._live_ctx_tester.pop(task.id)
        app._live_ctx_resolver[task.id] = 8000
        app.refresh_task_status()
        await pilot.pause()
        assert "8k" in str(right.render())


async def test_ctx_counter_shows_last_agent_transcript(app: DrCodeApp) -> None:
    """Without any live reading the counter falls back to the last agent's
    persisted transcript: after a tester round it shows the round's fill,
    not the dev transcript's."""
    task = app.manager.create("tested", "the original prompt")
    task.messages = _transcript()
    task.tester_messages = [_transcript()]
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        right = app.query_one("#task-status-right")
        text = str(right.render())
        # Both transcripts end with the same usage — the tester reading is
        # shown (the tester role's window, same 200k default here) and the
        # counter did not vanish.
        assert "🧠" in text and "200k" in text
        # A resolver round becomes the last agent: same figures (identical
        # transcripts), but no crash — the counter still renders.
        task.resolver_messages = [_transcript()]
        app.refresh_task_status()
        await pilot.pause()
        assert "🧠" in str(right.render())


@pytest.fixture
def app(project_root: Path) -> DrCodeApp:
    return DrCodeApp(project_root)
