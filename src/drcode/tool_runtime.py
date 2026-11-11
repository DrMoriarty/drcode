"""Tool runtime: run blocking tools in a worker thread and translate refusals.

`_run_tool` runs a tool in a thread (stalling the event loop otherwise, which
also drives the Textual UI); tools that raise a fixable error are converted to
`ModelRetry`. `_run_tool_logged` / `_run_bash_logged` additionally turn a
`PathViolation` into a logged service event plus a retry prompt. The deps types
are imported under `TYPE_CHECKING` to keep `agent.py` free of a circular import.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset

from . import tools
from .service_events import log_service_event, log_violation
from .tools import PathViolation

if TYPE_CHECKING:
    from .agent import AgentDeps, ResolverDeps, TesterDeps

# Errors tools raise for situations the model can fix (bad path, missing
# old_string, broken regex). pydantic-ai 2.x converts only ModelRetry and
# arg-validation failures into retry prompts; any other exception crashes
# the whole run, so these must be re-raised as ModelRetry.
TOOL_RETRY_ERRORS = (ValueError, OSError, re.error)


async def _run_tool(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except TOOL_RETRY_ERRORS as e:
        raise ModelRetry(str(e) or type(e).__name__) from e


async def _run_tool_logged(
    deps: AgentDeps | TesterDeps | ResolverDeps,
    tool: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a tool and turn a PathViolation into a retry prompt for the model,
    logging the refusal into the task history first."""
    try:
        return await _run_tool(fn, *args, **kwargs)
    except PathViolation as e:
        log_violation(deps, tool, str(e))
        raise ModelRetry(str(e)) from e


async def _run_bash_logged(
    deps: AgentDeps | TesterDeps | ResolverDeps,
    root: Path,
    command: str,
    timeout: float | None = None,
) -> str:
    try:
        return await tools.run_bash(
            root, command, timeout, extra_writable=deps.scratch, task_id=deps.task_id
        )
    except PathViolation as e:
        log_violation(deps, "run_bash", str(e))
        raise ModelRetry(str(e)) from e


# -- degenerate tool-batch guard ------------------------------------------------
#
# Some models (observed on xiaomi/mimo-v2.6-flash via zveno.ai, task 1ad9c359)
# emit degenerate PARALLEL tool batches: up to 32 calls in one response,
# including exact duplicates — one response there carried 31 identical
# grep_search calls (every one honestly executed, ~2 minutes for one answer),
# and a third of the whole transcript was exact repeats of already-executed
# calls. Two defenses, both deliberately scoped to ONE model response (a
# "batch" = all tool calls between two model requests, i.e. one run_step):
#
# 1. DEDUP: identical (tool_name, args) calls inside one batch execute once;
#    the twins get the same result content, each under its own tool_call_id
#    (the protocol demands an answer per call). A repeat of the same call in
#    a LATER response is legitimate (re-read a file after an edit) and always
#    executes — the epoch resets on every run_step.
# 2. BATCH LIMIT: at most `limit` distinct calls per batch execute; the rest
#    get a plain tool result telling the model to re-issue them (a plain
#    result, NOT a ModelRetry — a retry would burn the per-tool retry budget
#    and can crash the whole run when a degenerate batch refuses 20 calls at
#    once against retries=10).
#
# 3. DEGRADATION DETECTION (passive — execution behavior is untouched): when
#    a batch crosses a pathology heuristic, a "Warning:" service event is
#    logged (shown in the chat, never fed to the model) so the user sees the
#    model degrading instead of just a long silence. Two triggers: a single
#    (tool_name, args) pair arriving DEGENERATE_IDENTICAL_THRESHOLD+ times in
#    one batch, and a total batch size over DEGENERATE_BATCH_SIZE. The
#    arrivals are counted at the TOP of call_tool, BEFORE the dedup/limit
#    branches — after the guards a 31-call twins storm would look like one
#    innocent call, and the pathology would be invisible.
#
# The guard is UI-neutral and hooks ONE seam: a WrapperToolset around the
# agent's own function toolset.

DEFAULT_TOOL_BATCH_LIMIT = 8

# Degradation-detection heuristics (see the section comment above): a single
# (tool_name, args) pair arriving this many times in ONE batch, or a total
# batch size over this line, means the model is almost certainly stuck in a
# degenerate parallel-batch loop (the observed mimo-v2.6-flash pathology ran
# up to 32 calls per response, the protocol cap).
DEGENERATE_IDENTICAL_THRESHOLD = 3
DEGENERATE_BATCH_SIZE = 16

# At most this many "N identical <tool> calls" clauses go into one event
# (sorted by count) — a pathology spanning many tools must not spam a
# wall of text into the chat.
_MAX_IDENTICAL_NOTES = 3

_BATCH_LIMIT_RESULT_PREFIX = "Batch limit exceeded"


def _args_fingerprint(args: dict[str, Any]) -> str:
    """Canonical text of validated tool args — the dedup key half. Validated
    args come from the model's JSON, so a sorted-key dump is exact; the repr
    fallback only exists to never crash on an exotic value."""
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(args)


def _limit_message(limit: int) -> str:
    return (
        f"{_BATCH_LIMIT_RESULT_PREFIX}: at most {limit} tool calls are executed "
        "per model response, and this call was NOT executed. Re-issue the "
        "remaining calls in your NEXT response — a few per response at most, "
        "or one at a time."
    )


class BatchGuard:
    """Per-agent state behind the tool-batch dedup and the parallel-call cap.

    One instance per agent build (an agent is built per activation, so per
    run). All bookkeeping happens on the event loop between awaits — plain
    attribute arithmetic is race-free (same reasoning as SubagentGate). The
    batch is keyed by `ctx.run_step` (pydantic-ai increments it on every
    model request; all tool calls of one response share it): a new run_step
    rolls the state over, which is what makes a repeated call in the NEXT
    response execute again.
    """

    def __init__(self, limit: int = DEFAULT_TOOL_BATCH_LIMIT) -> None:
        self.limit = max(1, limit)
        self._run_step: int | None = None
        # (tool_name, args fingerprint) → future of (ok, result-or-exception):
        # the primary registers the future and resolves it when done, twins
        # await it. The (ok, value) tuple (instead of set_exception) keeps a
        # failed primary from ever surfacing asyncio's "exception was never
        # retrieved" noise.
        self._futures: dict[tuple[str, str], asyncio.Future[tuple[bool, Any]]] = {}
        self._executed = 0
        self._duplicates = 0
        self._refused = 0
        # ORIGINAL-batch bookkeeping, counted at the top of call_tool before
        # the dedup/limit branches touch anything (the degradation event must
        # describe the model's actual output, not what survived the guards).
        self._arrivals: dict[tuple[str, str], int] = {}
        self._batch_size = 0

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: Any,
        wrapped: AbstractToolset[Any],
        tool: Any,
    ) -> Any:
        run_step = getattr(ctx, "run_step", 0) or 0
        if run_step != self._run_step:
            self._flush_batch(ctx)
            self._run_step = run_step
        key = (name, _args_fingerprint(tool_args))
        self._arrivals[key] = self._arrivals.get(key, 0) + 1
        self._batch_size += 1
        future = self._futures.get(key)
        if future is not None:
            # Exact twin of a call in this batch: same result content, own
            # tool_call_id (the framework builds the return part per call).
            self._duplicates += 1
            ok, value = await future
            if not ok:
                raise value
            return value
        if self._executed >= self.limit:
            self._refused += 1
            return _limit_message(self.limit)
        self._executed += 1
        future = asyncio.get_running_loop().create_future()
        self._futures[key] = future
        try:
            result = await wrapped.call_tool(name, tool_args, ctx, tool)
        except BaseException as e:  # twins must see the same failure
            future.set_result((False, e))
            raise
        future.set_result((True, result))
        return result

    def _degradation_notes(self) -> list[str]:
        """Pathology heuristics over the ORIGINAL batch (arrival counts, i.e.
        what the model actually issued — not what survived dedup/limiting).
        Offending keys are aggregated per tool name (max count): the observed
        pathology also repeats the same grep with permuted alternatives —
        several keys of one tool, reported as one clause."""
        notes: list[str] = []
        by_tool: dict[str, int] = {}
        for (name, _fingerprint), count in self._arrivals.items():
            if count >= DEGENERATE_IDENTICAL_THRESHOLD:
                by_tool[name] = max(by_tool.get(name, 0), count)
        offenders = sorted(
            ((count, name) for name, count in by_tool.items()),
            reverse=True,
        )
        for count, name in offenders[:_MAX_IDENTICAL_NOTES]:
            notes.append(f"model issued {count} identical {name} calls in one batch")
        if len(offenders) > _MAX_IDENTICAL_NOTES:
            notes.append(
                f"and {len(offenders) - _MAX_IDENTICAL_NOTES} more tools with "
                "identical-call storms"
            )
        if self._batch_size > DEGENERATE_BATCH_SIZE:
            notes.append(
                f"a single response carried {self._batch_size} tool calls "
                f"(sanity threshold {DEGENERATE_BATCH_SIZE})"
            )
        return notes

    def _flush_batch(self, ctx: Any) -> None:
        """The batch ended: flush its stats into the service-event log (best
        effort — deps without a manager, like the subagent's, are skipped)
        and reset the counters. Degenerate batches prepend a "Warning:" note
        computed from the ORIGINAL batch, so the pathology stays visible no
        matter how hard the execution guards collapsed it."""
        degenerate = self._degradation_notes()
        notes = list(degenerate)
        if self._duplicates:
            notes.append(
                f"{self._duplicates} duplicate tool call(s) in one response "
                "answered from cache without re-execution"
            )
        if self._refused:
            notes.append(
                f"{self._refused} tool call(s) refused: batch limit "
                f"({self.limit} per response)"
            )
        self._reset_batch()
        if not notes:
            return
        deps = getattr(ctx, "deps", None)
        if deps is not None and hasattr(deps, "manager") and hasattr(deps, "task_id"):
            text = "; ".join(notes)
            if degenerate:
                text = f"Warning: {text} — possible model degradation"
            try:
                log_service_event(deps, "tool-batch", text)
            except Exception:  # noqa: BLE001, S110 - stats never fatal
                pass

    def _reset_batch(self) -> None:
        self._futures.clear()
        self._executed = 0
        self._duplicates = 0
        self._refused = 0
        self._arrivals.clear()
        self._batch_size = 0


_AgentDepsT = TypeVar("_AgentDepsT")


@dataclass
class BatchGuardToolset(WrapperToolset[_AgentDepsT]):
    """A toolset wrapper applying the BatchGuard to every tool call of the
    wrapped toolset (dedup within a model response + the per-response cap).
    `dataclasses.replace` in the inherited `for_run`/`for_run_step` keeps the
    `guard` field, so per-run toolset transitions preserve the state."""

    guard: BatchGuard

    def __getattr__(self, name: str) -> Any:
        """Forward anything this wrapper does not define (e.g.
        `FunctionToolset.add_function`, used by `@agent.tool`) to the wrapped
        toolset — tools may be registered on the agent after the guard is
        attached, and they must land in the INNER toolset."""
        try:
            wrapped = object.__getattribute__(self, "wrapped")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(wrapped, name)

    async def call_tool(  # type: ignore[override]
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: Any,
        tool: Any,
    ) -> Any:
        return await self.guard.call_tool(name, tool_args, ctx, self.wrapped, tool)

    async def for_run_step(self, ctx: Any) -> Any:
        """Flush the finished batch at the step boundary. pydantic-ai calls
        `for_run_step` before EVERY model request — including the one that
        follows the run's LAST tool-call batch (a text-only final response
        triggers no further `call_tool`, so the call_tool-side rollover alone
        would silently drop that batch's stats)."""
        self.guard._flush_batch(ctx)
        return await super().for_run_step(ctx)


def attach_batch_guard(agent: Any, limit: int) -> None:
    """Replace the agent's own function-toolset with a batch-guarded wrapper.

    `limit <= 0` disables the guard entirely (toolset untouched). The agent's
    function toolset is a private pydantic-ai attribute — there is no public
    post-construction API for wrapping tools registered via `@agent.tool` —
    but it is re-read on every run step, so the replacement applies to all
    runs of the agent and stays invisible to the tool wiring itself."""
    if limit <= 0:
        return
    agent._function_toolset = BatchGuardToolset(  # single seam, see docstring
        wrapped=agent._function_toolset, guard=BatchGuard(limit)
    )
