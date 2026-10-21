"""Coding and QA tester agents wired to an OpenAI-compatible endpoint.

This module is a *thin orchestrator*: the model / network layer lives in
`agent_model`, service-event logging in `service_events`, tool runtime in
`tool_runtime`, the summarizer/compaction block in `compaction`, the
transcript / usage / chat-rendering block in `transcript`, and the bundled
system prompts in `prompts.system`. Everything that *wires* the agents to
their tools and deps is kept here. The names below are re-exported so the
test-suite and the TUI keep importing them from ``drcode.agent``; see
`__all__`.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent, RunContext

from . import agent_model, tools
from .agent_model import (
    EXTRA_HEADERS,
    AsyncOpenAI,
    ModelFetchError,
    fetch_models,
    probe_model,
)
from .compaction import (
    build_compaction_agent,
    build_compaction_prompt,
    compacted_messages,
    extract_previous_summary,
    is_summary_message,
    render_history_for_summary,
    split_original_prompts,
)
from .config import Config
from .models import (
    Clarification,
    TaskKind,
    TaskStatus,
)
from .prompts import (
    BUG_SYSTEM_PROMPT,
    FEATURE_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    TESTER_SYSTEM_PROMPT,
    TESTER_SYSTEM_PROMPT_NO_COMMIT,
    load_system_prompt,
    load_tester_prompt,
)
from .service_events import log_service_event
from .service_events import log_violation as _log_violation
from .task_manager import TaskManager
from .tool_runtime import (
    _run_bash_logged,
    _run_tool,
    _run_tool_logged,
)
from .transcript import (
    COMPACTION_SUMMARY_PREFIX,
    _context_status,
    all_user_prompts,
    backdate_user_prompt,
    ctx_context_tokens,
    ends_with_open_request,
    extract_display,
    first_user_prompt,
    is_pending_retry,
    last_context_tokens,
    last_response_usage,
    load_messages,
    resumable_history,
    run_context_tokens,
    sanitized_messages,
    serialize_messages,
)

__all__ = [
    "BUG_SYSTEM_PROMPT",
    "COMPACTION_SUMMARY_PREFIX",
    "FEATURE_SYSTEM_PROMPT",
    "RESEARCH_SYSTEM_PROMPT",
    "TESTER_SYSTEM_PROMPT",
    "TESTER_SYSTEM_PROMPT_NO_COMMIT",
    "AgentDeps",
    "AsyncOpenAI",
    "ModelFetchError",
    "TaskManager",
    "TesterDeps",
    "_context_status",
    "_git_push",
    "_log_violation",
    "_run_bash_logged",
    "_run_tool",
    "_run_tool_logged",
    "all_user_prompts",
    "backdate_user_prompt",
    "build_agent",
    "build_compaction_agent",
    "build_compaction_prompt",
    "build_tester_agent",
    "compacted_messages",
    "ends_with_open_request",
    "extract_display",
    "extract_previous_summary",
    "fetch_models",
    "first_user_prompt",
    "is_pending_retry",
    "is_summary_message",
    "last_context_tokens",
    "last_response_usage",
    "load_agents_md",
    "load_messages",
    "load_system_prompt",
    "load_tester_prompt",
    "log_service_event",
    "probe_model",
    "render_history_for_summary",
    "resumable_history",
    "run_context_tokens",
    "sanitized_messages",
    "serialize_messages",
    "split_original_prompts",
]


@dataclass
class AgentDeps:
    """Dependencies for the coding agent: project root, task manager, task id."""

    root: Path
    manager: TaskManager
    task_id: str


@dataclass
class TesterDeps:
    """Dependencies for the tester agent: project root, task manager, verified task."""

    root: Path
    manager: TaskManager
    task_id: str


def load_agents_md(root: Path) -> str:
    """Return the project's AGENTS.md content, or an empty string if absent."""
    try:
        return (root / "AGENTS.md").read_text(encoding="utf-8")
    except OSError:
        return ""


def build_agent(
    config: Config, kind: TaskKind = TaskKind.FEATURE, root: Path | None = None
) -> Agent[AgentDeps, str]:
    """Build the coding agent. Every task kind gets a dedicated system prompt:
    bug tasks instruct minimal changes, feature tasks encourage refactoring and
    new classes, research tasks answer questions / update docs. The research
    agent additionally gets the task-creation tools (bug reports and feature
    requests). When the project root contains AGENTS.md, its content is
    appended to the instructions."""
    model = agent_model.build_model(config)
    instructions = load_system_prompt(kind, project_root=root)
    if root is not None:
        agents_md = load_agents_md(root)
        if agents_md.strip():
            instructions = (
                f"{instructions}\n\n# Project instructions (AGENTS.md)\n\n{agents_md}"
            )
    agent = Agent(
        model,
        instructions=instructions,
        deps_type=AgentDeps,
        model_settings={
            "extra_headers": EXTRA_HEADERS,
            "max_tokens": config.max_tokens,
        },
        retries=10,
    )

    # All blocking tools run in a worker thread (or as asyncio subprocesses
    # for run_bash): called inline they would stall the event loop, which
    # also drives the Textual UI — long commands froze keyboard input.
    @agent.tool
    async def read_file(
        ctx: RunContext[AgentDeps], path: str, offset: int = 1, limit: int = 2000
    ) -> str:
        """Read a text file from anywhere on the filesystem (relative paths resolve
        against the project root). Lines are numbered, starting from `offset`
        (1-based), up to `limit` lines."""
        current_usage = ctx_context_tokens(ctx) or 0
        return await _run_tool_logged(
            ctx.deps,
            "read_file",
            tools.read_file,
            ctx.deps.root,
            path,
            offset,
            limit,
            current_usage=current_usage,
            max_context_tokens=config.max_context_tokens,
            compaction_threshold=config.compaction_threshold,
        )

    @agent.tool
    async def context_status(ctx: RunContext[AgentDeps]) -> str:
        """Report the current context fill: how many tokens the model's context
        slot holds at the latest request (prompt history + generated output) out
        of the configured maximum, the percent used and how many tokens remain
        before the compaction threshold. Use it when you are unsure how much
        room you have — it gives the exact current numbers so you can decide
        when to stop reading, without having to guess or fret over a vague
        "nearly full" signal."""
        return _context_status(
            ctx_context_tokens(ctx),
            config.max_context_tokens,
            config.compaction_threshold,
        )

    # Muting-file tools are barriers: pydantic-ai runs tool calls of one
    # model response in parallel, so two concurrent edits of the same file
    # would both read the original snapshot and the later write would
    # silently overwrite the earlier edit (lost update). `sequential=True`
    # makes each of these run alone, never overlapping other tool calls.
    @agent.tool(sequential=True)
    async def write_file(ctx: RunContext[AgentDeps], path: str, content: str) -> str:
        """Create or overwrite a file. Only the project root (with all subdirectories)
        and /tmp (with all subdirectories) are writable; use /tmp to test hypotheses."""
        return await _run_tool_logged(
            ctx.deps, "write_file", tools.write_file, ctx.deps.root, path, content
        )

    @agent.tool(sequential=True)
    async def edit_file(
        ctx: RunContext[AgentDeps], path: str, old_string: str, new_string: str
    ) -> str:
        """Replace an exact unique substring in a file. old_string must match exactly once.
        Only the project root (with all subdirectories) and /tmp (with all subdirectories)
        are writable; use /tmp to test hypotheses."""
        return await _run_tool_logged(
            ctx.deps,
            "edit_file",
            tools.edit_file,
            ctx.deps.root,
            path,
            old_string,
            new_string,
        )

    @agent.tool
    async def glob_files(ctx: RunContext[AgentDeps], pattern: str) -> list[str]:
        """Find files by glob pattern relative to project root, e.g. 'src/**/*.py'."""
        return await _run_tool(tools.glob_files, ctx.deps.root, pattern)

    @agent.tool
    async def grep_search(
        ctx: RunContext[AgentDeps], pattern: str, include: str = ""
    ) -> list[str]:
        """Regex search in file contents. `include` is an optional glob mask
        filtering files by their project-relative path, e.g. '*.py' or 'src/**/*.py'."""
        return await _run_tool(tools.grep_search, ctx.deps.root, pattern, include)

    @agent.tool
    async def run_bash(
        ctx: RunContext[AgentDeps], command: str, timeout: int | None = None
    ) -> str:
        """Run a shell command in the project root and return its output.
        The command is killed after `timeout` seconds (default 120); pass a
        larger timeout for long-running commands (builds, test suites) —
        values above 3600 (60 minutes) are clamped to 3600.
        rm/mv/cp are only allowed with paths inside the project root or /tmp."""
        return await _run_bash_logged(ctx.deps, ctx.deps.root, command, timeout)

    # Task-creation tools exist only for the research agent: coding
    # agents (feature/bug) work on their own task and must not spawn more;
    # the research agent can file bug reports and feature requests on
    # the user's behalf. Placement follows the ordinary queue rules via
    # TaskManager.create (bugs join the head of the bug block, everything
    # else the tail of the queue).
    if kind is TaskKind.RESEARCH:

        @agent.tool
        def create_bug_report(
            ctx: RunContext[AgentDeps], title: str, description: str
        ) -> str:
            """Create a bug report task in the backlog. Use ONLY when the user
            explicitly asked for it — if you found a problem on your own, ask
            via ask_clarifying_question whether a task should be filed.
            `title` is a short imperative summary; `description` carries the
            details: what is wrong, where (files/lines), and how it manifests."""
            bug = ctx.deps.manager.create(title, description, kind=TaskKind.BUG)
            return (
                f"Bug report #{bug.id} '{bug.title}' created at the head of the queue."
            )

        @agent.tool
        def create_feature_request(
            ctx: RunContext[AgentDeps], title: str, description: str
        ) -> str:
            """Create a feature request task in the backlog. Use ONLY when the
            user explicitly asked for it — if you spotted an improvement on
            your own, ask via ask_clarifying_question whether a task should be
            filed. `title` is a short summary of the desired change;
            `description` details what should be done and why."""
            feature = ctx.deps.manager.create(title, description, kind=TaskKind.FEATURE)
            return (
                f"Feature request #{feature.id} '{feature.title}' created at "
                "the tail of the queue."
            )

    @agent.tool
    def ask_clarifying_question(
        ctx: RunContext[AgentDeps], question: str, options: list[str] | None = None
    ) -> str:
        """Ask the user a clarifying question and pause the task until they answer.
        Use when the request is ambiguous or a required detail is missing.
        `options` is an optional list of suggested answers; pass it when a small
        set of choices covers the reasonable answers, otherwise omit it."""
        task = ctx.deps.manager.get(ctx.deps.task_id)
        task.clarification = Clarification(
            question=question.strip(),
            options=[o.strip() for o in (options or []) if o.strip()],
        )
        task.status = TaskStatus.AWAITING_CLARIFICATION
        task.touch()
        ctx.deps.manager.storage.save(task)
        return (
            "Question sent to the user. The task is paused awaiting clarification; "
            "end your turn now and do not call any more tools."
        )

    return agent


def build_tester_agent(
    config: Config, *, allow_commit: bool = True, root: Path | None = None
) -> Agent[TesterDeps, str]:
    """Build the QA tester agent.

    `allow_commit=False` (project setting "Task finalization: Testing")
    selects the commit-free system prompt and does not register the
    git_commit tool, so verified changes are left uncommitted. When `root`
    is given, its actual path is substituted into the system prompt header
    (the `{project_path}` placeholder).
    """
    model = agent_model.build_model(config)
    agent = Agent(
        model,
        instructions=load_tester_prompt(allow_commit, project_root=root),
        deps_type=TesterDeps,
        model_settings={
            "extra_headers": EXTRA_HEADERS,
            "max_tokens": config.max_tokens,
        },
        retries=10,
    )

    @agent.tool
    async def read_file(
        ctx: RunContext[TesterDeps], path: str, offset: int = 1, limit: int = 2000
    ) -> str:
        """Read a text file from anywhere on the filesystem (relative paths resolve
        against the project root). Lines are numbered, starting from `offset`
        (1-based), up to `limit` lines."""
        current_usage = ctx_context_tokens(ctx) or 0
        return await _run_tool_logged(
            ctx.deps,
            "read_file",
            tools.read_file,
            ctx.deps.root,
            path,
            offset,
            limit,
            current_usage=current_usage,
            max_context_tokens=config.max_context_tokens,
            compaction_threshold=config.compaction_threshold,
        )

    @agent.tool
    async def context_status(ctx: RunContext[TesterDeps]) -> str:
        """Report the current context fill: how many tokens the model's context
        slot holds at the latest request (prompt history + generated output) out
        of the configured maximum, the percent used and how many tokens remain
        before the compaction threshold. Use it when you are unsure how much
        room you have — it gives the exact current numbers so you can decide
        when to stop reading, without having to guess or fret over a vague
        "nearly full" signal."""
        return _context_status(
            ctx_context_tokens(ctx),
            config.max_context_tokens,
            config.compaction_threshold,
        )

    @agent.tool
    async def glob_files(ctx: RunContext[TesterDeps], pattern: str) -> list[str]:
        """Find files by glob pattern relative to project root, e.g. 'src/**/*.py'."""
        return await _run_tool(tools.glob_files, ctx.deps.root, pattern)

    @agent.tool
    async def grep_search(
        ctx: RunContext[TesterDeps], pattern: str, include: str = ""
    ) -> list[str]:
        """Regex search in file contents. `include` is an optional glob mask
        filtering files by their project-relative path, e.g. '*.py' or 'src/**/*.py'."""
        return await _run_tool(tools.grep_search, ctx.deps.root, pattern, include)

    @agent.tool
    async def run_bash(
        ctx: RunContext[TesterDeps], command: str, timeout: int | None = None
    ) -> str:
        """Run a shell command in the project root and return its output.
        The command is killed after `timeout` seconds (default 120); pass a
        larger timeout for long-running checks (full test suites, builds) —
        values above 3600 (60 minutes) are clamped to 3600.
        rm/mv/cp are only allowed with paths inside the project root or /tmp."""
        return await _run_bash_logged(ctx.deps, ctx.deps.root, command, timeout)

    @agent.tool
    def create_bug_report(
        ctx: RunContext[TesterDeps], title: str, description: str
    ) -> str:
        """Create a bug report task. It is placed at the head of the execution queue."""
        bug = ctx.deps.manager.create(title, description, kind=TaskKind.BUG)
        return f"Bug report #{bug.id} '{bug.title}' created at the head of the queue."

    # The commit tool exists only when the project setting allows it: with
    # "Task finalization: Testing" the model cannot commit even if it tries.
    if allow_commit:

        @agent.tool
        async def git_commit(ctx: RunContext[TesterDeps], message: str) -> str:
            """Stage all changed files in the project root and create a git commit
            with the given message, then push the branch to its remote. Use only
            after all checks passed, with a short message paraphrasing the
            original task. Skips gracefully when the project root is not a git
            repository or there is nothing to commit. A failed push (no network,
            diverged remote, no upstream, …) is NOT a failure: the push error is
            recorded in the task's event log and the commit stays safe locally —
            still proceed with set_verification(success=True)."""
            root = ctx.deps.root
            message = message.strip().strip("'\"")
            if not message:
                return "Empty commit message — nothing committed."
            inside = await tools.run_bash(root, "git rev-parse --is-inside-work-tree")
            if not inside.startswith("exit=0") or "true" not in inside:
                return "Not a git repository — nothing committed."
            await tools.run_bash(root, "git add -A -- . ':(exclude).drcode'")
            staged = await tools.run_bash(root, "git diff --cached --name-only")
            if (
                staged.startswith("exit=0")
                and not staged.removeprefix("exit=0").strip()
            ):
                return "Nothing to commit — the working tree is clean."
            commit = await tools.run_bash(root, f"git commit -m {shlex.quote(message)}")
            if not commit.startswith("exit=0"):
                return f"Commit failed:\n{commit}"
            head = await tools.run_bash(root, "git log -1 --oneline")
            result = f"Committed successfully.\n{head}"
            return result + await _git_push(ctx.deps, root)

    @agent.tool
    def set_verification(ctx: RunContext[TesterDeps], success: bool) -> str:
        """Set the verification label on the verified task (passed or failed)
        and finish its testing: the task becomes completed."""
        ctx.deps.manager.set_verification(ctx.deps.task_id, success)
        # The label is stamped by the manager; formatting it from `success`
        # (not from the optional `task.verification`) keeps pyright happy.
        verdict = "passed" if success else "failed"
        return f"Verification label set: {verdict}; task completed."

    return agent


async def _git_push(deps: TesterDeps, root: Path) -> str:
    """Best-effort `git push` after a successful commit.

    A push failure never fails verification: the error is written into the
    task's service-event log and reported back to the model as a note, so the
    tester still passes the task."""
    remotes = await tools.run_bash(root, "git remote")
    names = remotes.removeprefix("exit=0").strip()
    if not remotes.startswith("exit=0") or not names or names == "(no output)":
        log_service_event(deps, "git", "git push skipped: no remote configured.")
        return "\ngit push skipped: no remote configured."
    push = await tools.run_bash(root, "git push")
    if push.startswith("exit=0"):
        return "\nPushed successfully."
    if "no upstream" in push:
        # Fresh branch without tracking: retry once, setting the upstream.
        push = await tools.run_bash(root, "git push --set-upstream origin HEAD")
        if push.startswith("exit=0"):
            return "\nPushed successfully (set upstream to origin/HEAD)."
    detail = push.removeprefix("exit=0").strip() or "(no output)"
    log_service_event(deps, "git", f"git push failed:\n{detail}")
    return (
        "\nPush failed (the commit is safe locally):\n"
        f"{detail}\n"
        "This is not a verification failure — proceed with "
        "set_verification(success=True) and mention the push problem "
        "in your summary."
    )
