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

from . import agent_model, task_search, tools
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
    build_reduce_prompt,
    chunk_history_for_summary,
    compacted_messages,
    extract_previous_summary,
    is_summary_message,
    prompt_fits_window,
    render_history_for_summary,
    slice_budget_tokens,
    split_original_prompts,
)
from .config import Config
from .models import (
    Clarification,
    Task,
    TaskKind,
    TaskStatus,
    TodoItem,
    render_todos,
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
    "_commit_tracked_files",
    "_context_status",
    "_git_push",
    "_log_violation",
    "_run_bash_logged",
    "_run_tool",
    "_run_tool_logged",
    "_track_edited_file",
    "all_user_prompts",
    "backdate_user_prompt",
    "build_agent",
    "build_compaction_agent",
    "build_compaction_prompt",
    "build_reduce_prompt",
    "build_tester_agent",
    "chunk_history_for_summary",
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
    "prompt_fits_window",
    "render_history_for_summary",
    "resumable_history",
    "run_context_tokens",
    "sanitized_messages",
    "serialize_messages",
    "slice_budget_tokens",
    "split_original_prompts",
    "todo_dynamic_instructions",
    "todo_instructions",
]


@dataclass
class AgentDeps:
    """Dependencies for the coding agent: project root, task manager, task id.

    `scratch` is the task's personal writable scratch directory
    (/tmp/drcode/<task-id>, see tools.scratch_dir): the only place outside
    the project root the tools may write to."""

    root: Path
    manager: TaskManager
    task_id: str
    scratch: Path


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


# Minimum checklist size: a shorter list is the degenerate anti-pattern
# (OpenCode-style one-item todos on trivial work) — the guard rejects the
# call and the ModelRetry corrects the model in-session.
TODO_MIN_ITEMS = 3


def _todo_create(deps: AgentDeps, items: list[str]) -> str:
    """Create/replace the task's todo checklist (feature tasks only).

    Raises ValueError (→ ModelRetry via _run_tool) for a degenerate list:
    the agent must not maintain a checklist for single-step work."""
    cleaned = [item.strip() for item in items if item and item.strip()]
    if len(cleaned) < TODO_MIN_ITEMS:
        raise ValueError(
            f"todo_create needs at least {TODO_MIN_ITEMS} items, got "
            f"{len(cleaned)}: for single-step work do not use the checklist, "
            "just work."
        )
    task = deps.manager.get(deps.task_id)
    replaced = bool(task.todos)
    task.todos = [TodoItem(text=text) for text in cleaned]
    task.touch()
    deps.manager.storage.save(task)
    head = "Checklist replaced" if replaced else "Checklist created"
    return (
        f"{head} with {len(cleaned)} items:\n{render_todos(task.todos)}\n\n"
        "Keep the list accurate: call todo_update immediately after "
        "finishing each item, before starting the next one."
    )


def _todo_update(deps: AgentDeps, targets: list[int | str], done: bool) -> str:
    """Check/uncheck checklist items by 1-based index or text fragment.

    Raises ValueError (→ ModelRetry via _run_tool) when a target matches
    nothing — the model then corrects itself instead of silently losing an
    item."""
    task = deps.manager.get(deps.task_id)
    if not task.todos:
        raise ValueError("This task has no checklist — call todo_create first.")
    matched: set[int] = set()
    for target in targets:
        if isinstance(target, int):
            index = target
        elif isinstance(target, str) and target.strip().isdigit():
            index = int(target.strip())
        else:
            fragment = str(target).strip().lower()
            hits = [
                i for i, item in enumerate(task.todos) if fragment in item.text.lower()
            ]
            if not hits:
                raise ValueError(
                    f"No checklist item matches {target!r}; the current "
                    f"items are:\n{render_todos(task.todos)}"
                )
            matched.update(hits)
            continue
        if not 1 <= index <= len(task.todos):
            raise ValueError(
                f"todo index {index} is out of range 1..{len(task.todos)}."
            )
        matched.add(index - 1)
    for i in sorted(matched):
        task.todos[i].done = done
    task.touch()
    deps.manager.storage.save(task)
    return (
        f"Checklist updated ({len(matched)} item(s) marked "
        f"{'done' if done else 'not done'}):\n{render_todos(task.todos)}"
    )


def _set_task_dependencies(deps: AgentDeps, task_id: str, dep_ids: list[str]) -> str:
    """Replace a task's depends_on list (research agent only).

    Raises ValueError (→ ModelRetry via _run_tool) when the task does not
    exist, is already under way (only QUEUED/HOLD tasks are editable —
    the same rule as the command palette) or a dependency id is unknown /
    would form a cycle."""
    if task_id not in deps.manager.tasks:
        raise ValueError(f"unknown task id: {task_id}")
    task = deps.manager.get(task_id)
    if task.status not in (TaskStatus.QUEUED, TaskStatus.HOLD):
        raise ValueError(
            f"task #{task_id} is {task.status.value} — dependencies can only "
            "be set on queued or held tasks (before development starts)."
        )
    deps.manager.set_dependencies(task_id, dep_ids)
    if task.depends_on:
        return (
            f"Task #{task_id} now waits for: "
            + ", ".join(f"#{d}" for d in task.depends_on)
            + " — it is not promoted until every listed task is COMPLETED "
            "and (under worktree isolation) merged."
        )
    return f"Task #{task_id} has no dependencies — it is freely promotable."


def _deps_suffix(task: Task) -> str:
    """The dependency note appended to a create_* tool result."""
    if not task.depends_on:
        return ""
    return (
        " It waits for: "
        + ", ".join(f"#{d}" for d in task.depends_on)
        + " (not promoted until all are COMPLETED and merged)."
    )


def todo_instructions(task: Task) -> str | None:
    """The task's current checklist state as an instruction block (feature
    tasks with a non-empty list only — nothing else has todo tools or a
    checklist).

    This is the key anti-"forgot the list" measure. It is injected as
    DYNAMIC instructions (see `todo_dynamic_instructions`), so it is
    recomputed for EVERY model request of every activation: the model sees
    the live list state on the very first request after a `todo_create`
    call within the same uninterrupted run, not only after a restart or a
    compaction. A synthetic user message would pile up one stale copy per
    activation in the persisted transcript; instructions never touch the
    transcript. The list itself lives in `task.todos`; if it were only
    prompt material, an interrupted attempt or a compaction could drop it."""
    if task.kind is not TaskKind.FEATURE or not task.todos:
        return None
    unchecked = len(task.unchecked_todos)
    lines = [
        "# Task checklist (current state)",
        "",
        "Your todo checklist for this task, as it stands right now:",
        "",
        render_todos(task.todos),
        "",
        (
            f"{unchecked} of {len(task.todos)} items are unchecked. Keep the "
            "list accurate with todo_update: check an item off immediately "
            "after finishing it, before starting the next one. Do not finish "
            "the task while items remain unchecked unless the user "
            "explicitly dropped them."
        ),
    ]
    return "\n".join(lines)


def todo_dynamic_instructions(ctx: RunContext[AgentDeps]) -> str | None:
    """Dynamic instructions hook: recompute the checklist block from the
    task's CURRENT state for every model request. Registered on the coding
    agent in `build_agent` — pydantic-ai evaluates it per request, so a
    checklist created (or updated) mid-activation is visible to the next
    request of the same run, which run-level `instructions=` (frozen at
    `agent.iter` start) could never achieve."""
    return todo_instructions(ctx.deps.manager.get(ctx.deps.task_id))


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

        @agent.tool(sequential=True)
        async def git_commit_docs(ctx: RunContext[AgentDeps], message: str) -> str:
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
            return await _commit_tracked_files(ctx.deps, ctx.deps.root, message)

        @agent.tool
        async def search_task_history(
            ctx: RunContext[AgentDeps],
            keywords: list[str],
            status: str | None = None,
            kind: str | None = None,
        ) -> str:
            """Search the project's task archive (ALL statuses: queued, hold,
            active, testing, completed, clarification) by keywords. Case-
            insensitive substring match over each task's title, user prompts,
            pending input, edited file paths, the agent's final summary
            response and its service events; a common trailing ending of a
            keyword is also stripped heuristically ("смерженных" matches
            "смерженный"), so prefer full natural wordforms. Several keywords
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
    config: Config,
    *,
    allow_commit: bool = True,
    root: Path | None = None,
    review_gate: bool = False,
) -> Agent[TesterDeps, str]:
    """Build the QA tester agent.

    `allow_commit=False` (project setting "Task finalization: Testing")
    selects the commit-free system prompt and does not register the
    git_commit tool, so verified changes are left uncommitted. When `root`
    is given, its actual path is substituted into the system prompt header
    (the `{project_path}` placeholder). With `review_gate=True` (an
    external review is defined in `.drcode/workflow.py`) a successful
    verdict only labels the task PASSED and leaves it TESTING — the
    external review decides the final completion.
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
        rm/mv/cp are only allowed with paths inside the project root or your
        scratch directory /tmp/drcode/<task-id>."""
        return await _run_bash_logged(ctx.deps, ctx.deps.root, command, timeout)

    @agent.tool
    def create_bug_report(
        ctx: RunContext[TesterDeps], title: str, description: str
    ) -> str:
        """Create a bug report task. It is placed at the head of the execution queue."""
        # origin="system" — see the research variant of this tool above.
        bug = ctx.deps.manager.create(
            title, description, kind=TaskKind.BUG, origin="system"
        )
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
            return result + await _git_push(ctx.deps, root)

    @agent.tool
    def set_verification(
        ctx: RunContext[TesterDeps],
        success: bool,
        retry_here: bool = True,
        report: str = "",
    ) -> str:
        """Set the verification label on the verified task (passed or failed).

        On failure, first decide who owns the bug:
        - retry_here=True: the failure is caused by THIS task's own changes
          (broken behaviour, tests this change should have kept green,
          regressions of previously working functionality). The task returns
          to development in its own worktree and `report` (what failed: the
          exact command and the error output) becomes its next input — do
          NOT call create_bug_report in this case.
        - retry_here=False: the failure is unrelated to the task's changes
          or environmental (pre-existing linter debt, CI/network issues, a
          broken dependency). The task completes as failed — call
          create_bug_report right after.

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


def _retry_failed_here(deps: TesterDeps, report: str) -> str:
    """Apply the tester's `retry_here` failure verdict (set_verification with
    success=False, retry_here=True): requeue the task into development with
    the failure report as its next input — unless the fix-round cap
    (project setting `max_dev_rounds`) is reached, in which case the task
    completes as FAILED and the tester must create the bug report. Both
    outcomes are logged into the task's service-event history by the
    manager (requeue_failed_fix); returns the tool-result text telling the
    model which path was taken."""
    task = deps.manager.get(deps.task_id)
    rounds_done = task.fix_rounds + 1
    max_rounds = max(1, deps.max_dev_rounds)
    deps.manager.requeue_failed_fix(deps.task_id, report, max_rounds=max_rounds)
    task = deps.manager.get(deps.task_id)
    if task.status is TaskStatus.QUEUED and task.fix_rounds == rounds_done:
        return (
            "Verification label set: failed. The failure is caused by this "
            "task's own changes: the task returns to development in its own "
            f"worktree (round {task.fix_rounds + 1}/{max_rounds} of its fix "
            "loop). The report was queued as the task's next input — do NOT "
            "call create_bug_report."
        )
    if task.status is TaskStatus.COMPLETED:
        return (
            "Verification label set: failed; task completed. The bug "
            f"survived {max_rounds} dedicated development rounds — call "
            "create_bug_report now with a clear title and a description of "
            "what still fails (the exact command and the error output)."
        )
    # The task left TESTING while the verdict was in flight (the user
    # re-queued or cancelled it): the stale verdict changed nothing.
    return "Verification label ignored: the task is no longer being tested."


async def _git_push(deps: AgentDeps | TesterDeps, root: Path) -> str:
    """Best-effort `git push` after a successful commit.

    A push failure never fails the caller's flow: the error is written into
    the task's service-event log and reported back to the model as a note, so
    the tester still passes the task and the research agent still finishes
    its documentation assignment with the commit safe locally."""
    remotes = await tools.run_bash(root, "git remote", extra_writable=deps.scratch)
    names = remotes.removeprefix("exit=0").strip()
    if not remotes.startswith("exit=0") or not names or names == "(no output)":
        log_service_event(deps, "git", "git push skipped: no remote configured.")
        return "\ngit push skipped: no remote configured."
    push = await tools.run_bash(root, "git push", extra_writable=deps.scratch)
    if push.startswith("exit=0"):
        return "\nPushed successfully."
    if "no upstream" in push:
        # Fresh branch without tracking: retry once, setting the upstream.
        push = await tools.run_bash(
            root, "git push --set-upstream origin HEAD", extra_writable=deps.scratch
        )
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


def _track_edited_file(deps: AgentDeps, resolved: Path) -> None:
    """Record a file the research agent created/edited on the task.

    Only research tasks are tracked: the coding agent's feature/bug changes
    are committed by the tester, not by the agent itself. Paths outside the
    project root (scratch files under /tmp/drcode/<task-id>) are ignored —
    they can never become commit candidates. The list persists with the
    task, so tracking survives
    restarts and the multi-turn loop of one run (a new AgentDeps is built per
    `agent.iter`, the task field is the shared accumulator). Duplicates are
    collapsed, keeping the first-seen order.
    """
    task = deps.manager.get(deps.task_id)
    if task.kind is not TaskKind.RESEARCH:
        return
    try:
        rel = resolved.relative_to(deps.root.resolve())
    except ValueError:
        return
    rel_posix = rel.as_posix()
    if rel_posix not in task.edited_files:
        task.edited_files.append(rel_posix)
        task.touch()
        deps.manager.storage.save(task)


async def _commit_tracked_files(deps: AgentDeps, root: Path, message: str) -> str:
    """Commit EXACTLY the files the research agent created/edited itself.

    The stage list comes from `task.edited_files` (see `_track_edited_file`)
    — the model cannot widen it, so unrelated working-tree changes are never
    committed. Pushes via `_git_push` when a remote is configured. The
    tracked list is cleared after a successful commit so a later call does
    not re-stage already-committed files.
    """
    message = message.strip().strip("'\"")
    if not message:
        return "Empty commit message — nothing committed."
    task = deps.manager.get(deps.task_id)
    files = list(dict.fromkeys(task.edited_files))
    if not files:
        return (
            "Nothing to commit — you have not created or edited any files in "
            "this task yourself. Only files you changed are ever committed; "
            "other project changes are never staged."
        )
    inside = await tools.run_bash(
        root, "git rev-parse --is-inside-work-tree", extra_writable=deps.scratch
    )
    if not inside.startswith("exit=0") or "true" not in inside:
        return "Not a git repository — nothing committed."
    quoted = " ".join(shlex.quote(f) for f in files)
    add = await tools.run_bash(
        root, f"git add -A -- {quoted}", extra_writable=deps.scratch
    )
    if not add.startswith("exit=0"):
        return f"Staging failed:\n{add}"
    staged = await tools.run_bash(
        root, "git diff --cached --name-only", extra_writable=deps.scratch
    )
    if staged.startswith("exit=0") and not staged.removeprefix("exit=0").strip():
        return (
            "Nothing to commit — the files you edited carry no changes "
            "against HEAD (already committed or untouched)."
        )
    commit = await tools.run_bash(
        root, f"git commit -m {shlex.quote(message)}", extra_writable=deps.scratch
    )
    if not commit.startswith("exit=0"):
        return f"Commit failed:\n{commit}"
    head = await tools.run_bash(
        root, "git log -1 --oneline", extra_writable=deps.scratch
    )
    task.edited_files.clear()
    task.touch()
    deps.manager.storage.save(task)
    return f"Committed successfully.\n{head}" + await _git_push(deps, root)
