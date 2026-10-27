"""Read-only research subagents spawned by the coding/research agents.

The parent agent's `delegate_research` tool (registered by
`build_agent` only when the project's `subagents_coder` /
`subagents_researcher` setting is above 0) runs a SHORT-LIVED nested
pydantic-ai run whose toolset is deliberately READ-ONLY — read_file,
grep_search, glob_files and search_task_archive. No run_bash, no
writing tools: subagents must never touch the (possibly shared)
worktree, the git refs or the stash stack while the parent (or a
sibling slot) is working — that is the structural guarantee against
the #5c4c9912 write-race class. The path jail is the parent's own
(`tools.read_file` reads anywhere anyway; the brief decides what is
looked at).

The subagent's transcript is session-only (in memory, never persisted
into task.json): the parent receives ONLY the subagent's final answer
plus a one-line usage summary — that compactness is the whole point
(saving the parent's context from hundreds of tool results, fewer
compactions). Spawns/finishes land in the task's service-event log
for observability in the chat.

The gate counter (`SubagentGate`) is per parent run (generation): the
pipeline creates one per activation and hands it to the agent deps;
exceeding the limit returns a plain explanatory tool result — never a
crash of the parent run. A failing subagent (network, model refusal,
usage cap) is likewise converted into a plain tool result.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

from . import task_search, tools
from .agent_model import EXTRA_HEADERS
from .providers import AgentRole
from .service_events import log_service_event
from .tool_runtime import _run_tool

if TYPE_CHECKING:
    from .agent import AgentDeps
    from .config import Config
    from .settings import ProjectSettings

__all__ = [
    "SubagentDeps",
    "SubagentGate",
    "build_subagent_agent",
    "run_delegate_tool",
    "subagent_limit_for",
]


@dataclass
class SubagentGate:
    """Per-parent-run counter of concurrently running subagents.

    Plain int arithmetic on the event loop (tool bodies of one run all
    execute on it) is race-free; the pipeline creates one gate per run
    so a superseded run's gate dies with it."""

    limit: int
    active: int = 0


@dataclass
class SubagentDeps:
    """Dependencies of the read-only subagent: the tree it reads (the
    parent's root — the task's worktree under isolation), the MAIN
    checkout's task-storage root for the archive search, and the parent
    task id (excluded from the search results)."""

    root: Path
    storage_root: Path
    parent_task_id: str


def subagent_limit_for(settings: ProjectSettings, role: AgentRole) -> int:
    """The project's subagent cap for the role serving a task: RESEARCH
    uses `subagents_researcher`, everything else (CODING) uses
    `subagents_coder`. The tester/resolver never get subagents."""
    if role is AgentRole.RESEARCH:
        return settings.subagents_researcher
    return settings.subagents_coder


def _build_subagent_model(config: Config):
    """The subagent's model: the resolved config of the role serving the
    parent (CODING → the coding model, RESEARCH → the research model —
    `build_agent` receives exactly that config). Split into a helper so
    the test-suite can stub it without touching the parent's wiring."""
    from . import agent_model

    return agent_model.build_model(config)


def build_subagent_agent(config: Config, *, root: Path) -> Agent[SubagentDeps, str]:
    """Build the read-only research subagent: the bundled `subagent.md`
    system prompt (project-overridable) and exactly four tools, all
    reading — no run_bash, no write tools, no ask_clarifying_question
    (there is no user to answer a subagent)."""
    agent = Agent(
        _build_subagent_model(config),
        instructions=_subagent_instructions(root),
        deps_type=SubagentDeps,
        model_settings={
            "extra_headers": EXTRA_HEADERS,
            "max_tokens": config.max_tokens,
        },
        # Short runs: fewer retries than the parent agents keep.
        retries=3,
    )

    @agent.tool
    async def read_file(
        ctx: RunContext[SubagentDeps], path: str, offset: int = 1, limit: int = 2000
    ) -> str:
        """Read a text file from anywhere on the filesystem (relative paths resolve
        against the project root). Lines are numbered, starting from `offset`
        (1-based), up to `limit` lines."""
        return await _run_tool(tools.read_file, ctx.deps.root, path, offset, limit)

    @agent.tool
    async def glob_files(ctx: RunContext[SubagentDeps], pattern: str) -> list[str]:
        """Find files by glob pattern relative to project root, e.g. 'src/**/*.py'."""
        return await _run_tool(tools.glob_files, ctx.deps.root, pattern)

    @agent.tool
    async def grep_search(
        ctx: RunContext[SubagentDeps], pattern: str, include: str = ""
    ) -> list[str]:
        """Regex search in file contents. `include` is an optional glob mask
        filtering files by their project-relative path, e.g. '*.py' or 'src/**/*.py'."""
        return await _run_tool(tools.grep_search, ctx.deps.root, pattern, include)

    @agent.tool
    async def search_task_archive(
        ctx: RunContext[SubagentDeps], keywords: list[str]
    ) -> str:
        """Search the project's task archive (ALL statuses) by keywords —
        title, user prompts, edited files, the agents' final summaries.
        Several keywords are combined with AND (an OR pass runs when the
        AND finds nothing). Output: one compact line per task, capped at
        50. Use it when the brief concerns past work on this project."""
        return await _run_tool(
            task_search.search,
            ctx.deps.storage_root,
            keywords,
            exclude_task_id=ctx.deps.parent_task_id,
        )

    return agent


def _subagent_instructions(root: Path) -> str:
    from .prompts import load_subagent_prompt

    return load_subagent_prompt(root)


def _brief_label(brief: str) -> str:
    """One-line label of the delegation brief for the service-event log."""
    for line in brief.strip().splitlines():
        line = line.strip()
        if line:
            return line[:100]
    return "(no brief)"


async def run_delegate_tool(deps: AgentDeps, config: Config, brief: str) -> str:
    """The body of the parent's `delegate_research` tool.

    Enforces the per-run gate, runs the nested read-only subagent (the
    event loop stays free — the nested run is plain async), logs the
    spawn/finish into the task's service-event log and returns the
    subagent's final answer plus a one-line usage summary. Every failure
    mode — gate exhausted, model/network error, usage cap — comes back
    as a PLAIN tool result with an actionable explanation: the parent
    run must never crash because a subagent did.
    """
    gate = deps.subagent_gate
    if gate is None or gate.limit <= 0:
        return (
            "Delegation is not configured for this run (no subagent slots). "
            "Continue without it."
        )
    if gate.active >= gate.limit:
        return (
            f"Subagent limit reached ({gate.active} running of {gate.limit} "
            "allowed). Wait for a running subagent to finish or continue "
            "without delegation."
        )
    gate.active += 1
    label = _brief_label(brief)
    log_service_event(
        deps,
        "subagent",
        f"subagent spawned ({gate.active}/{gate.limit}): {label}",
    )
    try:
        agent = build_subagent_agent(config, root=deps.root)
        subagent_deps = SubagentDeps(
            root=deps.root,
            storage_root=deps.manager.storage.root,
            parent_task_id=deps.task_id,
        )
        try:
            async with agent.iter(
                brief,
                deps=subagent_deps,
                usage_limits=UsageLimits(request_limit=config.request_limit),
            ) as run:
                async for _node in run:
                    # Keep the parent's stall-watchdog fed while the
                    # subagent works (the parent itself emits nothing
                    # while this tool call is in flight).
                    if deps.activity is not None:
                        deps.activity()
                result = run.result
            if result is None:
                raise RuntimeError("subagent produced no result")
            output = (result.output or "").strip()
            usage = result.usage
            requests = getattr(usage, "requests", 0) or 0
            tokens = usage.total_tokens
            log_service_event(
                deps,
                "subagent",
                f"subagent finished ({requests} requests, "
                f"{tokens} total tokens): {label}",
            )
            summary = (
                f"[subagent usage: {requests} requests, {tokens} tokens]"
                if tokens or requests
                else ""
            )
            if not output:
                return "The subagent returned an empty answer." + (
                    f" {summary}" if summary else ""
                )
            return f"{output}\n\n{summary}".strip()
        except asyncio.CancelledError:
            # The parent run is being aborted (ESC / nudge / a newer run):
            # propagate — the finally below releases the slot.
            raise
        except Exception as e:  # noqa: BLE001 - a subagent failure never kills the parent
            log_service_event(
                deps,
                "subagent",
                f"subagent failed: {type(e).__name__}: {e}",
            )
            return (
                f"The subagent failed ({type(e).__name__}: {e}). You can "
                "retry the delegation (perhaps with a narrower brief) or "
                "continue without it."
            )
    finally:
        gate.active -= 1
