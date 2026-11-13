"""Read-only research subagents spawned by the coding/research agents.

The parent agent's `delegate_research` tool (registered by
`build_agent` only when the project's `subagents_coder` /
`subagents_researcher` setting is above 0) runs a SHORT-LIVED nested
pydantic-ai run whose toolset is deliberately READ-ONLY — read_file,
grep_search, glob_files, search_task_archive and a whitelist-gated
read-only git tool (`git_history`: log/blame/show/diff). No run_bash,
no writing tools: subagents must never touch the (possibly shared)
worktree, the git refs or the stash stack while the parent (or a
sibling slot) is working — that is the structural guarantee against
the #5c4c9912 write-race class. The git tool stays inside that
guarantee structurally: a fixed subcommand whitelist (history readers
only) plus GIT_OPTIONAL_LOCKS=0, so even the harmless-looking index
stat-refresh a plain `git diff` performs can never take index.lock.
The path jail is the parent's own (`tools.read_file` reads anywhere
anyway; the brief decides what is looked at).

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

PROMPTING is what makes the tool get called — the wiring alone does not.
A whole 119-task drcode-bench run with the slots enabled fired ZERO
delegations while the tool was merely present in the schema. Three
places carry the prompting, all of them deliberate:

- the "Read-only subagents: delegate_research" section of the coding
  system prompts (`prompts/system/{feature,bug,research}.md`) — imperative,
  with hard triggers (mapping call sites, recon over >3-5 files, surveying
  a subsystem before editing) and a worked three-part brief example;
- the tool description built in `agent_build` — the same triggers plus the
  economics (the parent's own reads are permanent context and move it
  toward a compaction; a subagent's reads are free);
- `agent_tools.subagent_notice` — one line put into the request TAIL on
  every request while the limit is above 0, right next to the live
  checklist (`agent_model.TailInjectionModel`). The tail, never leading
  dynamic instructions: the leading system zone is the server-side prefix
  cache.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

from . import agent_model, task_search, tools
from .providers import AgentRole
from .service_events import log_service_event
from .tool_runtime import _run_tool, attach_batch_guard

if TYPE_CHECKING:
    from .agent import AgentDeps
    from .config import Config
    from .settings import ProjectSettings

__all__ = [
    "SubagentDeps",
    "SubagentGate",
    "build_subagent_agent",
    "git_read",
    "run_delegate_tool",
    "subagent_limit_for",
]

# -- read-only git history tool --------------------------------------------------
#
# The subagent is blind to the code's history without git: "why is this code
# like this", "when did this change", "what did the function look like
# before" are unanswerable with read/grep/glob alone. A git tool would be the
# natural fix, but a shell opens the write-race door (#5c4c9912) — so this is
# NOT run_bash: a FIXED subcommand whitelist of pure history readers, run
# without a shell (argv list — `>`/`>>` redirection is impossible by
# construction), with GIT_OPTIONAL_LOCKS=0 so even the stat-cache refresh a
# plain `git diff` may attempt never takes index.lock on a shared checkout.

# Pure readers: they never write the worktree, the index, the refs or the
# stash. Anything else (`commit`, `checkout`, `stash`, `apply`, `reset`, …)
# is refused by the whitelist before git even runs.
GIT_READ_SUBCOMMANDS = ("log", "blame", "show", "diff")

# Argument-level refusals: options of the whitelisted subcommands that would
# still leave the read-only lane. `--output`/`-o` (git diff) writes a file;
# the global options that retarget the repository (`--git-dir`,
# `--work-tree`, `-C`) would read a foreign tree the brief did not sanction —
# refused along with the config-injecting global forms.
GIT_FORBIDDEN_ARG_PREFIXES = ("--output", "--git-dir=", "--work-tree=")
GIT_FORBIDDEN_ARGS = {"-o", "-C", "--git-dir", "--work-tree", "--namespace", "-c"}

GIT_OUTPUT_CAP = 24_000
GIT_TIMEOUT_SECONDS = 30


def git_read(root: Path, args: list[str]) -> str:
    """Run a whitelisted READ-ONLY git command in `root` and return its output.

    `args` is `[subcommand, *arguments]` with `subcommand` one of
    GIT_READ_SUBCOMMANDS (log/blame/show/diff). Anything else — and any
    argument that would leave the read-only lane (file-writing `--output`,
    repository-retargeting global options) — raises `ValueError`, which
    `_run_tool` converts into a ModelRetry. Git failures (bad ref, unknown
    revision) are plain ValueErrors too: never a crash of the nested run.

    Executed without a shell, with `--no-pager` and GIT_OPTIONAL_LOCKS=0;
    output is capped at GIT_OUTPUT_CAP chars."""
    if not args:
        raise ValueError(
            "git command required: ["
            + "|".join(GIT_READ_SUBCOMMANDS)
            + "] plus arguments"
        )
    subcommand, rest = args[0], args[1:]
    if subcommand not in GIT_READ_SUBCOMMANDS:
        raise ValueError(
            f"git subcommand {subcommand!r} is not allowed; the read-only "
            "tool accepts only: " + ", ".join(GIT_READ_SUBCOMMANDS)
        )
    for arg in rest:
        if arg in GIT_FORBIDDEN_ARGS or any(
            arg.startswith(prefix) for prefix in GIT_FORBIDDEN_ARG_PREFIXES
        ):
            raise ValueError(
                f"git argument {arg!r} is not allowed in the read-only tool "
                "(it writes a file or retargets the repository)"
            )
    command = ["git", "--no-pager", subcommand, *rest]
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        proc = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            env=env,
            check=False,
        )
    except FileNotFoundError as e:
        raise ValueError("git is not available on this machine") from e
    except subprocess.TimeoutExpired as e:
        raise ValueError(
            f"git {subcommand} timed out after {GIT_TIMEOUT_SECONDS}s"
        ) from e
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        raise ValueError(
            f"git {subcommand} failed (exit {proc.returncode}): "
            f"{stderr or 'no error output'}"
        )
    out = proc.stdout or ""
    if len(out) > GIT_OUTPUT_CAP:
        out = out[:GIT_OUTPUT_CAP] + f"\n... [truncated at {GIT_OUTPUT_CAP} chars]"
    elif stderr:
        out = f"{out}\n[git stderr]\n{stderr}" if out else stderr
    return out or "(empty output)"


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


def _build_subagent_model(
    config: Config, announce_wait: agent_model.WaitAnnouncer | None = None
):
    """The subagent's model: the resolved config of the role serving the
    parent (CODING → the coding model, RESEARCH → the research model —
    `build_agent` receives exactly that config). Split into a helper so
    the test-suite can stub it without touching the parent's wiring.

    `announce_wait` rides along like everywhere else: a subagent request
    queued behind the provider's `max_parallel_requests` cap reports the
    wait in the PARENT's chat (it is the parent run that looks stalled)."""
    from . import agent_model

    return agent_model.build_model(config, announce_wait)


def build_subagent_agent(
    config: Config,
    *,
    root: Path,
    announce_wait: agent_model.WaitAnnouncer | None = None,
) -> Agent[SubagentDeps, str]:
    """Build the read-only research subagent: the bundled `subagent.md`
    system prompt (project-overridable) and exactly five tools, all
    reading — no run_bash, no write tools, no ask_clarifying_question
    (there is no user to answer a subagent). The git tool
    (`git_history`) is read-only by whitelist (log/blame/show/diff),
    not by convention — see `git_read`.

    `announce_wait` (the parent run's slot-wait announcer) is forwarded to
    the model builder — see `_build_subagent_model`."""
    agent = Agent(
        _build_subagent_model(config, announce_wait),
        instructions=_subagent_instructions(root),
        deps_type=SubagentDeps,
        model_settings=agent_model.base_model_settings(config),
        # Short runs: fewer retries than the parent agents keep.
        retries=3,
    )
    # Degenerate-batch guard (dedup + per-response cap, see tool_runtime):
    # a subagent with five read tools can degenerate just like a parent.
    attach_batch_guard(agent, config.tool_batch_limit)

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

    @agent.tool
    async def git_history(ctx: RunContext[SubagentDeps], args: list[str]) -> str:
        """Read-only git history: answer "why is the code like this", "when did
        this change", "what did it look like before". `args` starts with a
        whitelisted subcommand — log, blame, show or diff — followed by its
        arguments, e.g.:
        - ["log", "--oneline", "-15", "--", "src/foo.py"] — recent commits of a file;
        - ["blame", "src/foo.py"] — last change per line;
        - ["show", "abc1234"] — a commit (use --stat or a path to narrow it);
        - ["diff", "HEAD~1", "--", "src/foo.py"] — what changed between revisions.
        Only reads: no shell, no writing options (--output etc. refused), other
        subcommands refused. Run from the project root."""
        return await _run_tool(git_read, ctx.deps.root, args)

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

    The parent's slot-wait announcer (`deps.wait_announcer`, when the
    pipeline wired one) is forwarded to the subagent's model, so a
    subagent request queued behind the provider's concurrency cap reports
    the wait in the parent task's chat.
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
        agent = build_subagent_agent(
            config, root=deps.root, announce_wait=deps.wait_announcer
        )
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
            # The subagent's tokens are billed the same as the parent's:
            # fold the whole nested run's usage into the per-day usage
            # store under the parent's role/model key (the recorder is
            # wired by the pipeline — see AgentDeps.record_usage). Best
            # effort: an accounting failure must not fail the delegation.
            if deps.record_usage is not None and usage.total_tokens:
                try:
                    deps.record_usage(usage)
                except Exception:  # noqa: BLE001, S110 - statistics never fatal
                    pass
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
