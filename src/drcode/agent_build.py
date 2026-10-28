"""Builders for the coding and QA tester agents (OpenAI-compatible endpoint).

This module *wires* the agents: it picks the system prompt (per task kind,
with the project's AGENTS.md appended), constructs the pydantic-ai `Agent`
and registers its tools — thin wrappers that delegate the actual work to
`tools` (filesystem/grep/glob/shell), `tool_runtime` (worker-thread
execution so the Textual UI never stalls) and `agent_tools` (todo /
dependency / commit / push helpers). The deps dataclasses the tools
receive live here too.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry

from . import agent_model, task_search, tools
from .agent_model import EXTRA_HEADERS
from .agent_tools import (
    _commit_tracked_files,
    _deps_suffix,
    _git_push,
    _retry_failed_here,
    _run_project_check,
    _set_task_dependencies,
    _todo_create,
    _todo_update,
    _track_edited_file,
    load_agents_md,
    todo_dynamic_instructions,
)
from .config import Config
from .merge_resolver import conflict_marker_hits, unresolved_conflict_paths
from .models import Clarification, TaskKind, TaskStatus
from .prompts import (
    load_resolver_prompt,
    load_resolver_reviewer_prompt,
    load_system_prompt,
    load_tester_prompt,
)
from .settings import GitPolicy
from .subagents import SubagentGate, run_delegate_tool
from .task_manager import TaskManager
from .tool_runtime import _run_bash_logged, _run_tool, _run_tool_logged
from .transcript import _context_status, ctx_context_tokens

__all__ = [
    "AgentDeps",
    "ResolutionVerdict",
    "ResolverDeps",
    "TesterDeps",
    "build_agent",
    "build_resolver_agent",
    "build_resolver_reviewer_agent",
    "build_tester_agent",
]


@dataclass
class AgentDeps:
    """Dependencies for the coding agent: project root, task manager, task id.

    `scratch` is the task's personal writable scratch directory
    (/tmp/drcode/<task-id>, see tools.scratch_dir): the only place outside
    the project root the tools may write to.

    `subagent_gate` carries the per-run delegate_research slot counter
    (None = delegation not configured — the tool is not even registered
    when the limit is 0); `activity`, when set, is called by the nested
    subagent run to keep the parent's stall-watchdog fed while the
    parent itself emits nothing during the delegation."""

    root: Path
    manager: TaskManager
    task_id: str
    scratch: Path
    subagent_gate: SubagentGate | None = None
    activity: Callable[[], None] | None = None


@dataclass
class ResolutionVerdict:
    """The outcome the resolver (or the reviewer) reported through its
    verdict tool. Stored on ResolverDeps so the pipeline — not the agent —
    owns the follow-up: the agent's `resolved=True` never commits by
    itself, the mandatory verification funnel decides. `label` keeps the
    exact verdict word ("resolved"/"refused" for the resolver,
    "intact"/"lost"/"unsure" for the reviewer)."""

    resolved: bool
    report: str = ""
    label: str = ""


@dataclass
class TesterDeps:
    """Dependencies for the tester agent: project root, task manager, verified task.

    `scratch` is the task's personal writable scratch directory
    (/tmp/drcode/<task-id>, see tools.scratch_dir).

    `review_gate` mirrors the project's external-review hook
    (`.drcode/workflow.py` defines on_review): a PASSED verdict then leaves
    the task in TESTING awaiting the external review instead of completing
    it (see TaskManager.set_verification).

    `max_dev_rounds` is the project setting capping the fix loop: how many
    consecutive development rounds the tester may fail with a
    "the bug is this task's own change" verdict (retry_here=True) before
    the failure completes the task as FAILED (see
    TaskManager.requeue_failed_fix)."""

    root: Path
    manager: TaskManager
    task_id: str
    scratch: Path
    review_gate: bool = False
    max_dev_rounds: int = 3


@dataclass
class ResolverDeps:
    """Dependencies for the merge-conflict resolver agent (and its
    reviewer pass): the RESOLVED worktree the conflicted merge is parked
    in (`root` — writes are jailed to it like to any other project root),
    plus the task bookkeeping the service-event logger needs.

    `verdicts` accumulates the agent's set_resolution / set_review
    outcomes; the pipeline reads the LAST one after the run ends.
    """

    root: Path
    manager: TaskManager
    task_id: str
    scratch: Path
    verdicts: list[ResolutionVerdict] = field(default_factory=list)


def build_agent(
    config: Config,
    kind: TaskKind = TaskKind.FEATURE,
    root: Path | None = None,
    research_git: GitPolicy = GitPolicy.PUSH,
    subagent_limit: int = 0,
) -> Agent[AgentDeps, str]:
    """Build the coding agent. Every task kind gets a dedicated system prompt:
    bug tasks instruct minimal changes, feature tasks encourage refactoring and
    new classes, research tasks answer questions / update docs. The research
    agent additionally gets the task-creation tools (bug reports and feature
    requests). When the project root contains AGENTS.md, its content is
    appended to the instructions.

    `research_git` (project git policy for the research role) applies to
    research tasks only: PUSH (the default) registers git_commit_docs with
    push; LOCAL keeps the tool but the commit stays local; OFF does not
    register the tool at all and selects the fragment-free prompt — file
    tracking (`task.edited_files`) stays on for the "edited files" report.
    Feature/bug agents ignore the policy: they never commit themselves —
    the tester commits their verified changes.

    `subagent_limit` (the project's subagents_coder / subagents_researcher
    setting, chosen by the pipeline per the serving role) registers the
    delegate_research tool when above 0; at 0 (the default) the tool is
    not registered at all — the agent never sees it. The limit is baked
    into the tool description so the model knows its actual budget.
    """
    model = agent_model.build_model(config)
    instructions = load_system_prompt(
        kind, project_root=root, research_git=research_git
    )
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
    # Dynamic instructions: the feature task's checklist state rides on
    # EVERY model request (recomputed from `task.todos` at request time),
    # so a list created or updated mid-activation is seen by the next
    # request of the same run — run-level `instructions=` freezes at
    # `agent.iter` start and would miss exactly that. For non-feature
    # tasks and empty lists it yields None (excluded from the request).
    agent.instructions(todo_dynamic_instructions)

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
        and your scratch directory /tmp/drcode/<task-id> are writable; use the scratch
        directory to test hypotheses."""
        result = await _run_tool_logged(
            ctx.deps,
            "write_file",
            tools.write_file,
            ctx.deps.root,
            path,
            content,
            extra_writable=ctx.deps.scratch,
        )
        _track_edited_file(
            ctx.deps,
            tools.resolve_edit_path(
                ctx.deps.root,
                path,
                "create/overwrite",
                extra_writable=ctx.deps.scratch,
            ),
        )
        return result

    @agent.tool(sequential=True)
    async def edit_file(
        ctx: RunContext[AgentDeps], path: str, old_string: str, new_string: str
    ) -> str:
        """Replace an exact unique substring in a file. old_string must match exactly once.
        Only the project root (with all subdirectories) and your scratch directory
        /tmp/drcode/<task-id> are writable; use the scratch directory to test hypotheses."""
        result = await _run_tool_logged(
            ctx.deps,
            "edit_file",
            tools.edit_file,
            ctx.deps.root,
            path,
            old_string,
            new_string,
            extra_writable=ctx.deps.scratch,
        )
        _track_edited_file(
            ctx.deps,
            tools.resolve_edit_path(
                ctx.deps.root, path, extra_writable=ctx.deps.scratch
            ),
        )
        return result

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
        rm/mv/cp are only allowed with paths inside the project root or your
        scratch directory /tmp/drcode/<task-id>."""
        return await _run_bash_logged(ctx.deps, ctx.deps.root, command, timeout)

    # Self-managed todo checklist: registered for FEATURE agents only.
    # Bug tasks (minimal-change mandate in the bug prompt), research tasks
    # and the tester get NO todo tools — this structurally prevents the
    # one-item-list-on-a-small-bugfix anti-pattern. The list lives on the
    # task (`task.todos`, persisted by storage) and is injected into the
    # model context on every request by the dynamic instructions hook
    # (`todo_dynamic_instructions`, registered above), so an abandoned
    # stale list cannot go unnoticed either. Both tools run through
    # _run_tool so the fixable ValueErrors below become ModelRetry
    # prompts for the model.
    if kind is TaskKind.FEATURE:

        @agent.tool(sequential=True)
        async def todo_create(ctx: RunContext[AgentDeps], items: list[str]) -> str:
            """Create (or replace) the todo checklist for THIS task: the plan
            of the work you are about to do, shown to the user as a panel.
            Pass the steps as short imperative strings, in execution order.
            Call it once, at the start, when the work has 3 or more meaningful
            steps; do NOT create a checklist for trivial work (fewer than 3
            steps — just work instead). Replacing an existing list is allowed
            when the plan genuinely changed; keep already-done items and mark
            them done with todo_update afterwards."""
            return await _run_tool(_todo_create, ctx.deps, items)

        @agent.tool(sequential=True)
        async def todo_update(
            ctx: RunContext[AgentDeps], targets: list[int | str], done: bool
        ) -> str:
            """Check (done=True) or uncheck (done=False) checklist items of
            this task. `targets` are 1-based item positions (as shown in the
            checklist) or item-text fragments; update the checklist
            IMMEDIATELY after finishing each item — before starting the next
            one."""
            return await _run_tool(_todo_update, ctx.deps, targets, done)

    # Task-creation tools exist only for the research agent: coding
    # agents (feature/bug) work on their own task and must not spawn more;
    # the research agent can file bug reports and feature requests on
    # the user's behalf. Placement follows the ordinary queue rules via
    # TaskManager.create (bugs join the head of the bug block, everything
    # else the tail of the queue).
    if kind is TaskKind.RESEARCH:

        @agent.tool
        def create_bug_report(
            ctx: RunContext[AgentDeps],
            title: str,
            description: str,
            depends_on: list[str] | None = None,
        ) -> str:
            """Create a bug report task in the backlog. Use ONLY when the user
            explicitly asked for it — if you found a problem on your own, ask
            via ask_clarifying_question whether a task should be filed.
            `title` is a short imperative summary; `description` carries the
            details: what is wrong, where (files/lines), and how it manifests.
            Optional `depends_on`: short task ids (as shown in the TUI and in
            search_task_history output) the new task must wait for — it is not
            promoted until every listed task is COMPLETED and, under worktree
            isolation, its drcode/<id> branch is merged into the base branch.
            The returned id lets you chain subsequent creations on it."""
            # origin="system": the tool (not the user) created the task —
            # creation listeners (e.g. the voice announcements) can tell
            # agent-created bug reports from hand-typed ones.
            bug = ctx.deps.manager.create(
                title,
                description,
                kind=TaskKind.BUG,
                origin="system",
                depends_on=depends_on,
            )
            return (
                f"Bug report #{bug.id} '{bug.title}' created at the head of the queue."
                + _deps_suffix(bug)
            )

        @agent.tool
        def create_feature_request(
            ctx: RunContext[AgentDeps],
            title: str,
            description: str,
            depends_on: list[str] | None = None,
        ) -> str:
            """Create a feature request task in the backlog. Use ONLY when the
            user explicitly asked for it — if you spotted an improvement on
            your own, ask via ask_clarifying_question whether a task should be
            filed. `title` is a short summary of the desired change;
            `description` details what should be done and why.
            Optional `depends_on`: short task ids (as shown in the TUI and in
            search_task_history output) the new task must wait for — it is not
            promoted until every listed task is COMPLETED and, under worktree
            isolation, its drcode/<id> branch is merged into the base branch.
            The returned id lets you chain subsequent creations on it."""
            feature = ctx.deps.manager.create(
                title, description, kind=TaskKind.FEATURE, depends_on=depends_on
            )
            return (
                f"Feature request #{feature.id} '{feature.title}' created at "
                "the tail of the queue." + _deps_suffix(feature)
            )

        @agent.tool(sequential=True)
        async def set_task_dependencies(
            ctx: RunContext[AgentDeps], task_id: str, deps: list[str]
        ) -> str:
            """Replace the dependency list of an existing QUEUED or HOLD task
            (only those can still be edited — a task under way or finished
            cannot). `task_id` is the short task id; `deps` is the FULL new
            list of short task ids it must wait for (pass [] to clear —
            dependencies are replaced, not merged). The task is not promoted
            until every listed task is COMPLETED and, under worktree
            isolation, its drcode/<id> branch is merged into the base branch.
            Use it to link a task to an older one that existed before this
            run; for a task you create yourself, prefer the depends_on
            parameter of create_bug_report/create_feature_request."""
            return await _run_tool(_set_task_dependencies, ctx.deps, task_id, deps)

        # The commit tool follows the project's research git policy:
        # OFF registers nothing (the model cannot commit even if it
        # tries); LOCAL keeps the tool but the shared body below commits
        # locally and returns, skipping `_git_push` (mirrors the tester's
        # allow_push wiring below). The docstring mirrors the policy too,
        # exactly like the tester's git_commit variants.
        if research_git is not GitPolicy.OFF:
            if research_git is GitPolicy.PUSH:

                @agent.tool(sequential=True)
                async def git_commit_docs(
                    ctx: RunContext[AgentDeps], message: str
                ) -> str:
                    """Commit and push the documentation changes you made in this task.
                    Stages EXACTLY the files you created or edited yourself (tracked
                    automatically by write_file/edit_file) — never any other change in
                    the working tree — commits them with the given message and pushes
                    the branch when a remote is configured. Call it once, when the
                    documentation assignment is complete; do NOT call it for pure
                    question-answering tasks where you changed no files. Skips
                    gracefully when the project root is not a git repository. A failed
                    push is NOT a failure: the commit stays safe locally, the error is
                    logged into the task's event log — mention it in your summary."""
                    return await _commit_tracked_files(
                        ctx.deps, ctx.deps.root, message, allow_push=True
                    )

            else:

                @agent.tool(sequential=True)
                async def git_commit_docs(
                    ctx: RunContext[AgentDeps], message: str
                ) -> str:
                    """Commit the documentation changes you made in this task — LOCALLY,
                    this project never pushes. Stages EXACTLY the files you created or
                    edited yourself (tracked automatically by write_file/edit_file) —
                    never any other change in the working tree — and commits them with
                    the given message. Do not push yourself and do not mention pushing.
                    Call it once, when the documentation assignment is complete; do NOT
                    call it for pure question-answering tasks where you changed no
                    files. Skips gracefully when the project root is not a git
                    repository."""
                    return await _commit_tracked_files(
                        ctx.deps, ctx.deps.root, message, allow_push=False
                    )

        @agent.tool
        async def search_task_history(
            ctx: RunContext[AgentDeps],
            keywords: list[str],
            status: str | None = None,
            kind: str | None = None,
        ) -> str:
            """Search the project's task archive (ALL statuses: queued, hold,
            active, testing, completed, clarification) by keywords. Case-
            insensitive substring match over each task's short id, title,
            user prompts, pending input, edited file paths, the agent's
            final summary response and its service events; a common
            trailing ending of a keyword is also stripped heuristically
            ("смерженных" matches "смерженный"), so prefer full natural
            wordforms. Searching by a task's short id (or a prefix of it)
            finds exactly that task — handy to locate a task you only know
            by its id from another tool's output. Several keywords
            are combined with AND; when nothing matches the AND, an OR pass
            runs and its results come back under an explicit "no exact AND
            match" preamble. Results are ranked by keyword occurrences
            (ties: newest first). Optional filters: `status` and `kind`
            (bug/feature/research). Output is one compact line per task —
            id, status/kind, verification label, an "unmerged" marker when
            the task's drcode/<id> branch still holds unmerged commits (only
            computed for short result lists), date, title and a short
            snippet around the most informative match — capped at 50
            results (a longer hit list is truncated with a "refine the
            query" tail). Use it to assess the state of development around
            a topic — including "was it done AND merged?" — or find past
            decisions/bug reports without reading raw task JSONs; for a
            specific task's details, read_file its file
            <project>/.drcode/tasks/<id>.json. Your own task is excluded
            from the results, but the output tells you when it matches too
            — its transcript is often exactly what you are looking for."""
            # The archive lives in the MAIN checkout's .drcode/tasks/ — under
            # worktree isolation ctx.deps.root is the task's worktree, whose
            # .drcode/ has no task files. The manager's storage root always
            # points at the main project root, so search against that.
            return await _run_tool(
                task_search.search,
                ctx.deps.manager.storage.root,
                keywords,
                exclude_task_id=ctx.deps.task_id,
                status=status,
                kind=kind,
            )

    @agent.tool
    def ask_clarifying_question(
        ctx: RunContext[AgentDeps], question: str, options: list[str] | None = None
    ) -> str:
        """Ask the user a clarifying question and pause the task until they answer.
        Use when the request is ambiguous or a required detail is missing.
        `options` is an optional list of suggested answers; pass it when a small
        set of choices covers the reasonable answers, otherwise omit it.
        When constructing `options`, never mix what the user requested with
        your own initiative: do NOT bundle an initiative item into a package
        option like "yes: do X + Y + Z, everything at once" — declining the
        bundle would then force declining the requested work too. Each
        initiative item is its own SEPARATE opt-in option, phrased as a
        proposal ("(agent's suggestion, not from your request): …"), so the
        user can accept or decline it independently of the requested scope."""
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

    # Read-only subagent delegation (project settings subagents_coder /
    # subagents_researcher): registered ONLY when the limit is above 0 —
    # at the default 0 the tool does not exist for the model at all. The
    # actual limit is baked into the description (the tool is rebuilt per
    # run, so a settings change reaches the very next activation). Not
    # sequential: several subagents may run in parallel within the limit.
    if subagent_limit > 0:

        async def delegate_research(ctx: RunContext[AgentDeps], task: str) -> str:
            return await run_delegate_tool(ctx.deps, config, task)

        # The docstring is assigned (not written as an f-string in the
        # def): pydantic-ai parses the tool description out of the SOURCE
        # and an f-string docstring yields an empty description.
        delegate_research.__doc__ = f"""Spawn a READ-ONLY research subagent and delegate a piece of THIS
            task to it. You may run up to {subagent_limit} subagent{"" if subagent_limit == 1 else "s"} in parallel; the
            call returns the subagent's final report (its intermediate tool
            traffic stays invisible to your context) plus a one-line usage
            summary. The subagent can only read: read_file, grep_search,
            glob_files, search_task_archive — no writing, no shell.
            `task` is a SELF-CONTAINED brief for the subagent: state the
            question or investigation goal precisely, name the starting
            points you already know (files, symbols, error text), and spell
            out what the report must contain (file paths, line numbers).
            The subagent cannot ask you questions — the brief must be enough
            on its own. Use delegation for wide but shallow reading (mapping
            call sites, surveying a subsystem, scanning many files for a
            pattern); do the targeted verification reads yourself."""
        agent.tool(delegate_research)

    return agent


def build_tester_agent(
    config: Config,
    *,
    allow_commit: bool = True,
    allow_push: bool = True,
    root: Path | None = None,
    review_gate: bool = False,
) -> Agent[TesterDeps, str]:
    """Build the QA tester agent.

    `allow_commit=False` (tester git policy: off)
    selects the commit-free system prompt and does not register the
    git_commit tool, so verified changes are left uncommitted.
    `allow_push=False` (tester git policy: local) keeps the git_commit
    tool but it commits locally without ever
    pushing — the commit-only system prompt is selected. When `root`
    is given, its actual path is substituted into the system prompt header
    (the `{project_path}` placeholder). When the project root contains
    AGENTS.md, its content is appended to the instructions — the tester
    writes/commits code and runs the project's checks in the real
    repository, so the project rules apply to it just like to the coding
    agent. With `review_gate=True` (an
    external review is defined in `.drcode/workflow.py`) a successful
    verdict only labels the task PASSED and leaves it TESTING — the
    external review decides the final completion.
    """
    model = agent_model.build_model(config)
    instructions = load_tester_prompt(
        allow_commit, project_root=root, allow_push=allow_push
    )
    if root is not None:
        agents_md = load_agents_md(root)
        if agents_md.strip():
            instructions = (
                f"{instructions}\n\n# Project instructions (AGENTS.md)\n\n{agents_md}"
            )
    agent = Agent(
        model,
        instructions=instructions,
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
        rm/mv/cp are only allowed with paths inside the project root or your
        scratch directory /tmp/drcode/<task-id>."""
        return await _run_bash_logged(ctx.deps, ctx.deps.root, command, timeout)

    @agent.tool
    async def run_check(
        ctx: RunContext[TesterDeps],
        module: str,
        args: list[str],
        timeout: int | None = None,
    ) -> str:
        """Run a Python check tool from the project's OWN environment:
        `python -m <module> <args>` with the interpreter of the project's
        .venv / uv environment. Use it for pytest, ruff, mypy, pyright and
        any other Python tooling instead of run_bash — a bare `ruff` or
        `pytest` from PATH can be a different version with a different
        default ruleset and report phantom failures the project's own gate
        never produces. Example: run_check("ruff", ["check", "."]). When the
        result says the module is not installed in the project environment,
        treat that check as skipped — NEVER fall back to a PATH binary.
        The command is killed after `timeout` seconds (default 120); values
        above 3600 are clamped."""
        return await _run_tool(_run_project_check, ctx.deps.root, module, args, timeout)

    @agent.tool
    def create_bug_report(
        ctx: RunContext[TesterDeps], title: str, description: str
    ) -> str:
        """Create a bug report task. It is placed at the head of the execution
        queue. Use it to file problems found during verification that are NOT
        caused by the verified task's own changes (pre-existing/foreign bugs,
        environment or infrastructure issues) — the verified task itself then
        passes (set_verification(success=True)); see set_verification."""
        # origin="system" — see the research variant of this tool above.
        bug = ctx.deps.manager.create(
            title, description, kind=TaskKind.BUG, origin="system"
        )
        return f"Bug report #{bug.id} '{bug.title}' created at the head of the queue."

    # The commit tool exists only when the project's tester git policy
    # allows it: with policy off the model cannot commit even if it
    # tries. Policy local keeps the tool but pushes nothing: the shared
    # body below commits locally and returns, skipping `_git_push`.
    if allow_commit:

        async def _tester_git_commit(ctx: RunContext[TesterDeps], message: str) -> str:
            root = ctx.deps.root
            message = message.strip().strip("'\"")
            if not message:
                return "Empty commit message — nothing committed."
            inside = await tools.run_bash(
                root,
                "git rev-parse --is-inside-work-tree",
                extra_writable=ctx.deps.scratch,
            )
            if not inside.startswith("exit=0") or "true" not in inside:
                return "Not a git repository — nothing committed."
            await tools.run_bash(
                root,
                "git add -A -- . ':(exclude).drcode'",
                extra_writable=ctx.deps.scratch,
            )
            staged = await tools.run_bash(
                root, "git diff --cached --name-only", extra_writable=ctx.deps.scratch
            )
            if (
                staged.startswith("exit=0")
                and not staged.removeprefix("exit=0").strip()
            ):
                return "Nothing to commit — the working tree is clean."
            commit = await tools.run_bash(
                root,
                f"git commit -m {shlex.quote(message)}",
                extra_writable=ctx.deps.scratch,
            )
            if not commit.startswith("exit=0"):
                return f"Commit failed:\n{commit}"
            head = await tools.run_bash(
                root, "git log -1 --oneline", extra_writable=ctx.deps.scratch
            )
            result = f"Committed successfully.\n{head}"
            if allow_push:
                return result + await _git_push(ctx.deps, root)
            return result + "\nCommit is local — this project never pushes."

        if allow_push:

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
                return await _tester_git_commit(ctx, message)

        else:

            @agent.tool
            async def git_commit(ctx: RunContext[TesterDeps], message: str) -> str:
                """Stage all changed files in the project root and create a LOCAL git
                commit with the given message — this project never pushes. Use only
                after all checks passed, with a short message paraphrasing the
                original task. Do not push yourself and do not mention pushing.
                Skips gracefully when the project root is not a git repository or
                there is nothing to commit. Still proceed with
                set_verification(success=True) afterwards."""
                return await _tester_git_commit(ctx, message)

    @agent.tool
    def set_verification(
        ctx: RunContext[TesterDeps],
        success: bool,
        retry_here: bool = True,
        report: str = "",
    ) -> str:
        """Set the verification label on the verified task (passed or failed).

        The verdict judges THIS task's own work only. Before failing the
        task, first decide who owns the bug:
        - retry_here=True: the failure is caused by THIS task's own changes
          (broken behaviour, tests this change should have kept green,
          regressions of previously working functionality). The task returns
          to development in its own worktree and `report` (what failed: the
          exact command and the error output) becomes its next input — do
          NOT call create_bug_report in this case.
        - retry_here=False: THIS task's own deliverable does not work and
          cannot be accepted, and another development round is pointless
          (the task built the wrong thing, its result is fundamentally
          broken or unbuildable). The task completes as failed.

        Problems NOT caused by this task's changes — pre-existing linter
        debt, a foreign/pre-existing bug, CI/network or infrastructure
        issues, anything that also fails without this task's changes — are
        NOT a failure of this task: its own work is done, so call
        set_verification(success=True) and then file the unrelated problem
        separately via create_bug_report (title + description with what
        failed, the exact command and the error output). Never complete the
        task as failed over a problem it did not introduce.

        If the fix-round cap is reached (the task has already failed the
        maximum number of consecutive development rounds), a retry_here=True
        verdict completes the task as failed too — the tool result tells you,
        and you must call create_bug_report then.

        When the project uses an external review, a passed task stays in
        testing until the external review delivers its verdict."""
        deps = ctx.deps
        task = deps.manager.get(deps.task_id)
        if not success and retry_here and task.status is TaskStatus.TESTING:
            return _retry_failed_here(deps, report)
        deps.manager.set_verification(
            deps.task_id, success, review_gate=deps.review_gate
        )
        # The label is stamped by the manager; formatting it from `success`
        # (not from the optional `task.verification`) keeps pyright happy.
        verdict = "passed" if success else "failed"
        if success and deps.review_gate:
            return (
                f"Verification label set: {verdict}; the task now awaits "
                "the external review, which decides the final completion."
            )
        return f"Verification label set: {verdict}; task completed."

    return agent


def _resolver_shared_tools(agent: Agent[ResolverDeps, str], config: Config) -> None:
    """The read/inspect toolset shared by the resolver and its reviewer:
    reading works anywhere, everything else resolves against the RESOLVED
    worktree (`deps.root`). No write tools here — the reviewer is
    read-only by design, and the resolver gets its edit tools registered
    separately (see build_resolver_agent)."""

    @agent.tool
    async def read_file(
        ctx: RunContext[ResolverDeps], path: str, offset: int = 1, limit: int = 2000
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
    async def context_status(ctx: RunContext[ResolverDeps]) -> str:
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
    async def glob_files(ctx: RunContext[ResolverDeps], pattern: str) -> list[str]:
        """Find files by glob pattern relative to project root, e.g. 'src/**/*.py'."""
        return await _run_tool(tools.glob_files, ctx.deps.root, pattern)

    @agent.tool
    async def grep_search(
        ctx: RunContext[ResolverDeps], pattern: str, include: str = ""
    ) -> list[str]:
        """Regex search in file contents. `include` is an optional glob mask
        filtering files by their project-relative path, e.g. '*.py' or 'src/**/*.py'."""
        return await _run_tool(tools.grep_search, ctx.deps.root, pattern, include)


def build_resolver_agent(
    config: Config,
    *,
    root: Path | None = None,
) -> Agent[ResolverDeps, str]:
    """Build the merge-conflict resolver agent.

    Runs a conflicted merge that the algorithmic triage could not clear:
    the worktree (passed as `root` — its actual path is substituted into
    the system prompt's ``{project_path}`` placeholder) already carries
    the zdiff3 markers, the agent edits the conflicted files and reports
    through set_resolution. NO commit tool: the pipeline commits only
    after the mandatory verification funnel (mechanics + tests +
    intent-preservation review) passes — the resolver's verdict alone can
    never finalize anything.
    """
    model = agent_model.build_model(config)
    instructions = load_resolver_prompt(root)
    if root is not None:
        agents_md = load_agents_md(root)
        if agents_md.strip():
            instructions = (
                f"{instructions}\n\n# Project instructions (AGENTS.md)\n\n{agents_md}"
            )
    agent = Agent(
        model,
        instructions=instructions,
        deps_type=ResolverDeps,
        model_settings={
            "extra_headers": EXTRA_HEADERS,
            "max_tokens": config.max_tokens,
        },
        retries=10,
    )
    _resolver_shared_tools(agent, config)

    @agent.tool(sequential=True)
    async def write_file(ctx: RunContext[ResolverDeps], path: str, content: str) -> str:
        """Create or overwrite a file. Only the resolved worktree (with all
        subdirectories) and your scratch directory /tmp/drcode/<task-id> are
        writable; use the scratch directory to test hypotheses."""
        return await _run_tool_logged(
            ctx.deps,
            "write_file",
            tools.write_file,
            ctx.deps.root,
            path,
            content,
            extra_writable=ctx.deps.scratch,
        )

    @agent.tool(sequential=True)
    async def edit_file(
        ctx: RunContext[ResolverDeps], path: str, old_string: str, new_string: str
    ) -> str:
        """Replace an exact unique substring in a file. old_string must match exactly once.
        Only the resolved worktree (with all subdirectories) and your scratch
        directory /tmp/drcode/<task-id> are writable; use the scratch directory
        to test hypotheses."""
        return await _run_tool_logged(
            ctx.deps,
            "edit_file",
            tools.edit_file,
            ctx.deps.root,
            path,
            old_string,
            new_string,
            extra_writable=ctx.deps.scratch,
        )

    @agent.tool
    async def run_bash(
        ctx: RunContext[ResolverDeps], command: str, timeout: int | None = None
    ) -> str:
        """Run a shell command in the resolved worktree and return its output.
        The command is killed after `timeout` seconds (default 120); pass a
        larger timeout for long-running commands — values above 3600 (60
        minutes) are clamped to 3600. Use it for quick read-only inspection
        (git log/diff, compiling a module) — NOT for committing, merging or
        the full verification suite (the pipeline runs those checks itself
        and its verdict decides).
        rm/mv/cp are only allowed with paths inside the worktree or your
        scratch directory /tmp/drcode/<task-id>."""
        return await _run_bash_logged(ctx.deps, ctx.deps.root, command, timeout)

    @agent.tool
    def set_resolution(
        ctx: RunContext[ResolverDeps], resolved: bool, report: str
    ) -> str:
        """Report the conflict-resolution outcome. resolved=True when every
        conflicted file from the assignment is resolved (no conflict markers
        remain) and `report` explains per file what was done and why each
        side's intent survives; resolved=False when you cannot resolve with
        confidence — `report` says what blocks you and what a human should
        decide. This call does NOT commit anything: the pipeline runs the
        mandatory verification (mechanics, tests, intent review) and only a
        full pass commits the resolution."""
        if resolved:
            # Cheap pre-check before the verdict is recorded (the
            # a6a6df25 incident class: the resolution was described in
            # prose but never written into the files). A ModelRetry here
            # feeds the marker list straight back into THIS round — the
            # agent still has the transcript and can apply the edits —
            # instead of burning the whole verification funnel on a
            # syntactically still-conflicted tree.
            hits = conflict_marker_hits(
                ctx.deps.root,
                unresolved_conflict_paths(
                    ctx.deps.manager.storage.root, ctx.deps.task_id, ctx.deps.root
                ),
            )
            if hits:
                raise ModelRetry(
                    "set_resolution(resolved=True) rejected: conflict markers "
                    "are still present — the resolution was never written to "
                    "the files. Markers found:\n"
                    + "\n".join(hits)
                    + "\nApply the resolution with edit_file/write_file (remove "
                    "every marker, keep both sides' intent), then call "
                    "set_resolution again."
                )
        ctx.deps.verdicts.append(
            ResolutionVerdict(
                resolved=bool(resolved),
                report=report.strip(),
                label="resolved" if resolved else "refused",
            )
        )
        if resolved:
            return (
                "Resolution recorded. The pipeline will now run the "
                "verification funnel (mechanics, tests, intent-preservation "
                "review); only a full pass commits it. End your turn."
            )
        return (
            "Refusal recorded: the conflict stays untouched and a human "
            "takes over. End your turn."
        )

    return agent


def build_resolver_reviewer_agent(
    config: Config,
    *,
    root: Path | None = None,
) -> Agent[ResolverDeps, str]:
    """Build the intent-preservation reviewer of a resolved merge.

    A single-purpose read-only pass: it receives diff(resolved, ours) and
    diff(resolved, theirs) in its prompt and answers via set_review
    whether a piece of either side was lost (the #6c065401 incident
    class). Reuses ResolverDeps (`root` = the resolved worktree); its
    verdict lands in the same `verdicts` list.
    """
    model = agent_model.build_model(config)
    agent = Agent(
        model,
        instructions=load_resolver_reviewer_prompt(root),
        deps_type=ResolverDeps,
        model_settings={
            "extra_headers": EXTRA_HEADERS,
            "max_tokens": config.max_tokens,
        },
        retries=10,
    )
    _resolver_shared_tools(agent, config)

    @agent.tool
    def set_review(ctx: RunContext[ResolverDeps], verdict: str, report: str) -> str:
        """Deliver the review verdict: "intact" (every change in both diffs
        is justified, each side's intent survives), "lost" (a piece of one
        side was dropped without a semantic reason — name the file and the
        lost piece in `report`) or "unsure" (cannot tell — blocks the
        auto-commit, a human reviews instead)."""
        normalized = verdict.strip().lower()
        if normalized not in ("intact", "lost", "unsure"):
            return (
                "Invalid verdict — call set_review again with exactly "
                '"intact", "lost" or "unsure".'
            )
        ctx.deps.verdicts.append(
            ResolutionVerdict(
                resolved=normalized == "intact",
                report=report.strip(),
                label=normalized,
            )
        )
        return (
            f"Review verdict '{normalized}' recorded. End your turn — the "
            "pipeline owns the follow-up."
        )

    return agent
