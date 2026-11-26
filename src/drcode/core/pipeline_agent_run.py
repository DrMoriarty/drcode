"""The development agent's run loop.

Split out of the former monolithic `tui/pipeline.py`: `run_agent`
(the per-task exclusive worker dispatcher) and the worker itself —
turn preparation, streaming render, network-retry/outage handling,
failure demotion and the run epilogue (auto-completion, research
finalization, pipeline advance).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..agent import (
    backdate_user_prompt,
    build_title_prompt,
    clean_generated_title,
    ends_with_open_request,
    ends_with_truncated_response,
    is_pending_retry,
    last_response_usage,
    resumable_history,
    run_context_tokens,
    sanitized_messages,
    serialize_messages,
)
from ..core.usage import (
    NETWORK_RETRIES,
    _in_outage,
    _is_context_overflow_error,
    _is_deterministic_request_rejection,
    _is_retryable_network_error,
    _network_retry_delay,
)
from ..models import ImageAttachment, PendingInput, Task, TaskKind, TaskStatus
from ..providers import AgentRole, resolve_available, subagent_usage_key
from ..read_cache import ReadCache
from ..settings import WebFetchMode
from ..tools import scratch_dir
from ..web_fetch import WebFetchGate, web_fetch_policy_for
from .pipeline_stream import NodeStreamView

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.agent import AgentRun
    from pydantic_ai.messages import BinaryContent

    from ..agent_build import (
        AgentDeps,
        SubagentGate,  # noqa: TC004
        build_agent,  # noqa: TC004
        code_index_tools_for,  # noqa: TC004
    )
    from ..agent_tools import InputDeliveryHooks
    from ..subagents import subagent_limit_for  # noqa: TC004
from .state import CoreState

# -- lazy heavy-name seam --------------------------------------------------------
# The names below pull pydantic_ai / agent_build / subagents; they are NOT
# imported at module level so this module (on the TUI cold-start path) stays
# cheap. They resolve as module attributes on first access (PEP 562) and are
# cached as globals — so a test monkeypatching
# `pipeline_agent_run.build_agent` keeps working, provided the functions call
# `_ensure_agent_names()` before using them (a local import would bypass the
# monkeypatched attribute).
_HEAVY_AGENT_NAMES = (
    "SubagentGate",
    "build_agent",
    "code_index_tools_for",
    "subagent_limit_for",
)


def __getattr__(name: str) -> Any:
    if name in _HEAVY_AGENT_NAMES:
        from .. import agent

        value = getattr(agent, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _ensure_agent_names() -> None:
    for _name in _HEAVY_AGENT_NAMES:
        if _name not in globals():
            __getattr__(_name)


class PipelineAgentRunLayer(CoreState):
    def _input_delivery_hooks(self) -> InputDeliveryHooks:
        """Pipeline triggers for the agents' send_task_input tool: after a
        successful delivery the tool restarts the receiver's in-flight run
        (ACTIVE target) and schedules `_advance()` — exactly what the TUI
        does after a user send on an ACTIVE task. The tool body may run on
        a worker thread (tool_runtime), while `run_agent`/`_advance` belong
        to the app's event loop — hence the `call_soon_threadsafe`
        wrappers around the running loop (safe from any thread, including
        the loop thread itself, since async tools run on the loop)."""
        from ..agent_tools import InputDeliveryHooks

        loop = asyncio.get_running_loop()

        def advance() -> None:
            loop.call_soon_threadsafe(self._advance)

        def restart_run(task_id: str) -> None:
            loop.call_soon_threadsafe(self.run_agent, task_id)

        return InputDeliveryHooks(advance=advance, restart_run=restart_run)

    def run_agent(self, task_id: str) -> None:
        """(Re)start the development agent for one task, in that task's
        own exclusive worker group (`agent-<task-id>`): a newer run of
        the SAME task (a mid-flight comment, a nudge, a network-retry
        timer) supersedes the in-flight one exactly as the single global
        group always did — while every other task's run is untouched.
        This per-task independence is the base of the parallel
        development slots (see `parallel_limit`)."""
        if not self.host.host_active:
            # Teardown race: a network-retry set_timer fired while the app
            # was being dismantled — starting a worker now would only have
            # its body trip over the already-removed DOM. The task stays
            # ACTIVE with pending retry state; nothing is lost (the next
            # start resumes it).
            return
        # Register the run BEFORE the worker is spawned: between the
        # dispatch and the worker's first tick the task must already
        # count as owned — otherwise a same-tick _advance() (slot
        # accounting) or the stall watchdog's zombie sweep would see an
        # ACTIVE task with no run and misread it as a silent worker
        # death. The generation bump moved here from the worker body
        # for the same reason (the worker reads the pre-stamped
        # generation).
        generation = self._agent_generations.get(task_id, 0) + 1
        self._agent_generations[task_id] = generation
        self._agent_runs[task_id] = generation
        # A fresh registration starts with clean stall clocks: the stamp
        # left by the PREVIOUS run (e.g. the ask_clarifying_question one,
        # which may have ended tens of minutes ago) would otherwise make
        # the watchdog abort this run on its very next tick — the worker
        # only touches the clock itself inside _prepare_agent_run, well
        # after the dispatch. The zombie sweep already treats a missing
        # stamp as "clock starts on first sight", so a reset here keeps
        # both branches consistent.
        self._touch_activity(task_id, bookkeeping=True)
        self.host.host_spawn_worker(
            partial(self._run_agent_worker, task_id),
            name=f"agent run {task_id}",
            group=f"agent-{task_id}",
            exclusive=True,
        )

    async def _maybe_generate_title(self, task: Task, role: AgentRole) -> None:
        # Deferred heavy imports: pydantic_ai / agent_build stay off the
        # TUI cold-start path (this module loads before the first frame).
        from ..title_gen import build_title_agent

        """Generate the task's title from its description (project setting
        ``generate_task_titles``).

        The call profile is identical to the compaction summarizer's — a
        tool-less one-shot mini-agent with a bounded prompt and a one-line
        output — so it is served by the COMPACTION role's model (usually a
        cheap one) instead of loading the task's coding/research model with
        a cosmetic call before the main run even starts (see goal c6ee24e4:
        token economy, plus the title no longer adds to the activation
        latency of the expensive model). COMPACTION has NO fallback of its
        own in ``providers.resolve_effective``, so an unset compaction slot
        (``is_ready == False``) keeps the historical behavior: the model
        serving the task.

        Only DERIVED titles are rewritten: a title that arrived explicitly
        at creation (``task.title_explicit`` — the agents' report tools,
        the GitHub import, a goal's title) is skipped — generating over it
        would replace a deliberate name with a paraphrase of the
        description (and spend a model call on it).

        Exactly-once semantics for derived titles come from
        ``task.title_generated``: set only on SUCCESS, so a re-activation
        never re-generates a generated title and a failed one keeps the
        first-line title while the NEXT activation may retry — one bounded
        attempt per activation, never a loop within a run, and never fatal
        for the run itself. A task with nothing queued (an unfinished
        resume) has no description to name it from and is skipped
        silently. Tasks that never reach this worker (hold / draft /
        queued) keep the title derived at creation/edit.
        """
        if (
            not self.settings.generate_task_titles
            or task.title_explicit
            or task.title_generated
        ):
            return
        description = self.manager.editable_condition(task.id)
        if not description:
            return
        title_role = AgentRole.COMPACTION
        title_config = self.resolved_config(AgentRole.COMPACTION)
        if not title_config.is_ready:
            # No compaction model selected anywhere: fall back to the
            # task's serving model (the pre-change behavior).
            title_role = role
            title_config = self.resolved_config(role)
        try:
            agent = build_title_agent(
                title_config,
                # Provider slot-wait visibility (see _wait_announcer); the
                # wait wording stays on the task's slot role — it is shown
                # in the task's chat, whatever model ends up naming it.
                announce_wait=self._wait_announcer(task.id, role),
            )
            result = await agent.run(build_title_prompt(description))
            # The title call counts toward the session usage like every
            # other model call (under the role that actually served it —
            # compaction/<model> normally, the task's role on fallback).
            self._record_session_usage(
                result.usage,
                title_role,
                self.resolved_model_key(title_role),
                task_id=task.id,
            )
            title = clean_generated_title(result.output)
        except Exception as e:  # noqa: BLE001 - a failed title must not kill the run
            self._show_event_once(
                task,
                "Title generation failed "
                f"({type(e).__name__}: {e}); keeping the old title.",
            )
            return
        if title is None:
            self._show_event_once(
                task,
                "Title generation returned an empty name; keeping the old title.",
            )
            return
        old_title = task.title
        task.title = title
        task.title_generated = True
        task.touch()
        self.manager.storage.save(task)
        self._show_event_once(task, f"Task title generated: {old_title} → {title}.")
        self.refresh_sidebar()
        self.refresh_status()

    async def _run_agent_worker(self, task_id: str) -> None:
        # The generation was stamped and the run registered by run_agent()
        # right before the spawn — read it, don't bump again: a second
        # bump would orphan the registration (the finally below pops the
        # registry entry keyed by THIS generation). The fallback keeps a
        # direct invocation working the old way.
        generation = self._agent_generations.get(task_id, 0) or 1
        self._agent_generations[task_id] = generation
        self._agent_runs[task_id] = generation
        # Key for the tool-spinner registry: superseded runs clean up only
        # their own entries.
        run_key: tuple[str, str, int] = ("agent", task_id, generation)
        task = self.manager.get(task_id)
        # Bound only once `async with agent.iter(...)` is entered; the
        # except handlers below must not touch a run that never started.
        # Initialized before the try: build_agent() itself may raise.
        run: AgentRun[AgentDeps, str] | None = None
        cancelled = False
        # The role serving this task's transcript: research tasks run on the
        # research model (falling back to the coding one when unset), the
        # other kinds on the coding model. Everything below — the readiness
        # check, the agent build and the compaction gates — uses it.
        role = self._task_role(task)
        from pydantic_ai import Agent
        from pydantic_ai.exceptions import UsageLimitExceeded
        from pydantic_ai.usage import UsageLimits

        from ..agent_build import AgentDeps

        try:
            if not self.resolved_config(role).is_ready:
                self._show_event(
                    task,
                    "Model is not configured. Restart drcode to open the setup "
                    "dialog, or pick a model for the coding role in Settings "
                    "(or set DRCODE_MODEL / DRCODE_BASE_URL env vars); the "
                    "message is kept and will be sent once the model is "
                    "available.",
                )
                return
            # Availability-aware model choice for THIS run (probed once per
            # run start, never mid-run): the candidate chain (project slot,
            # then the user-level primary + fallbacks) is probed in
            # parallel and the first reachable provider wins. The factual
            # choice is pinned over the resolution caches so every
            # consumer below (usage accounting, compaction gates, events)
            # sees the model the run actually uses.
            resolution = await resolve_available(self.config, role)
            if resolution.unavailable and not resolution.available:
                # Every candidate provider is unreachable: degrade —
                # announce and start on the theoretical primary anyway;
                # the run's own network-retry/outage layer handles the
                # dead endpoints (a silent pause here would look like a
                # zombie run to the watchdog).
                self._show_event(
                    task,
                    "Provider(s) unreachable: "
                    + ", ".join(resolution.unavailable)
                    + f" — starting on {resolution.model_key or 'the configured model'} anyway.",
                )
            else:
                self._pin_resolution(role, resolution)
                if resolution.unavailable:
                    # A fallback (or a cross-provider match) is serving the
                    # run: name the unavailable providers once.
                    self._show_event(
                        task,
                        "Provider(s) unavailable: "
                        + ", ".join(resolution.unavailable)
                        + f" — using {resolution.model_key} for this run.",
                    )
            # Service event naming the model this run is served by (and a
            # model switch when the previous attempt started on a
            # different one — see _emit_run_model_event). Fires on EVERY
            # run start, including the automatic resume of a demoted or
            # unfinished attempt.
            self._emit_run_model_event(task, role)
            # Model-switch housekeeping (persisted `task.last_model_key` vs
            # the freshly resolved key): when the model changed BETWEEN
            # runs — a settings edit, or a restart after one — the
            # transcript is cleaned for the new model before the first
            # turn: the previous model's thinking parts are stripped
            # mechanically (a cross-provider resume would otherwise map
            # the foreign reasoning into the assistant content), a large
            # history is force-compacted and a takeover note is added.
            # Never fatal; a no-op when the key is unchanged.
            await self._prepare_model_switch_transcript(task, role)
            # Agent-generated task title (project setting
            # generate_task_titles): on the FIRST activation only, before
            # the run loop consumes the pending input — so the user sees
            # the new name on the active task from the very start. Never
            # fatal: the helper swallows its own failures.
            await self._maybe_generate_title(task, role)
            (
                root,
                agent,
                workflow_instructions,
                subagent_gate,
                web_fetch_gate,
                code_index,
                read_cache,
            ) = await self._prepare_agent_run(task, task_id, role, generation)
            # Pre-run forced compaction (project setting
            # compaction_resume_min_tokens): when the task waited long
            # enough for the provider's prefix cache to burn out and its
            # transcript is over the token threshold, compact it BEFORE
            # the first turn's history is built — the cold re-read then
            # happens over a compacted transcript. All other conditions
            # (first activation, short idle, small fill) skip silently
            # and the ordinary threshold gate below keeps working as
            # before. Never fatal: the helper swallows its own failures.
            await self._maybe_compact_after_idle(task)
            # A held-summary mark with no background job behind it (the
            # held state is session-only — a restart discards it): clean
            # it BEFORE the turn loop, or the stale mark would end the
            # transcript as an open request and a resumed attempt would
            # burn a spurious continuation turn on it.
            self._cleanup_stale_mark(task)
            # Chronological message flow: the transcript is the single
            # chronological record — the original prompt lives in it exactly
            # once and is NEVER re-queued/re-sent. Each activation first
            # consumes every pending user message in arrival order — merged
            # into ONE prompt, so a brief queued up before the run (original
            # message + comments) is solved as a whole, not turn-by-turn —
            # then, only if the last attempt was interrupted, runs one
            # continuation turn without new text to pick the interrupted
            # work back up.
            while task.status is TaskStatus.ACTIVE:
                run = None
                turn = await self._consume_turn_input(
                    task, task_id, workflow_instructions
                )
                if turn is None:
                    # Nothing queued and no unfinished attempt: the run is
                    # over (the epilogue below decides what the task
                    # becomes).
                    break
                content, consumed, workflow_instructions = turn
                try:
                    # Context compaction gate: before the history for the
                    # next turn is loaded, the fill is checked against the
                    # threshold; when exceeded, the transcript is summarized
                    # in place (history is computed AFTER compaction).
                    # Checked here (not only mid-run) to avoid one-turn lag
                    # in detection.
                    await self._maybe_compact(task)
                    history = resumable_history(task.messages)
                    async with agent.iter(
                        content,
                        message_history=history,
                        deps=AgentDeps(
                            root=root,
                            manager=self.manager,
                            task_id=task_id,
                            scratch=scratch_dir(task_id),
                            subagent_gate=subagent_gate,
                            # The nested subagent runs touch this while
                            # the parent itself emits nothing during the
                            # delegation tool call (see drcode.subagents).
                            activity=partial(self._touch_activity, task_id),
                            # create_bug_report/create_feature_request/
                            # create_research_task
                            # create DRAFT tasks when the setting is on
                            # (the user finishes them in the editor).
                            agent_creates_drafts=(self.settings.agent_creates_drafts),
                            # delegate_research subagents run on the
                            # project's `models["subagent"]` slot when one
                            # is set, else on THIS role's model: their
                            # usage folds into the per-day store under the
                            # subagent's OWN key in the first case, the
                            # parent's key in the second (see
                            # providers.subagent_usage_key and
                            # _record_session_usage).
                            record_usage=(
                                lambda u, _role=role, _tid=task_id: (
                                    self._record_session_usage(
                                        u,
                                        *subagent_usage_key(
                                            self.config,
                                            _role,
                                            self.resolved_model_key(_role),
                                        ),
                                        task_id=_tid,
                                    )
                                )
                            ),
                            # The research agent's web_fetch budget/policy
                            # for THIS run (None = the tool is not
                            # registered, see _prepare_agent_run).
                            web_fetch_gate=web_fetch_gate,
                            # The project's `unrestricted` web-fetch mode
                            # lifts the run_bash network guard for THIS run
                            # (any role — a project-level decision).
                            network_unrestricted=(
                                self.settings.web_fetch_mode
                                is WebFetchMode.UNRESTRICTED
                            ),
                            # The code-index decision for THIS run: the
                            # nested delegate_research subagents inherit
                            # it (they are the index's main
                            # beneficiaries).
                            code_index=code_index,
                            # Provider slot-wait visibility for the nested
                            # delegate_research runs (see _wait_announcer).
                            wait_announcer=self._wait_announcer(task_id, role),
                            # Pipeline triggers for the send_task_input
                            # tool (deliver extra input into a task this
                            # run created): restart the receiver's
                            # in-flight run and schedule _advance — the
                            # same two steps the TUI performs after a user
                            # send on an ACTIVE task.
                            input_hooks=self._input_delivery_hooks(),
                            # Per-run read_file dedup (created in
                            # _prepare_agent_run next to the gates).
                            read_cache=read_cache,
                        ),
                        usage_limits=UsageLimits(
                            request_limit=self.config.request_limit,
                            # Per-run ≈token budget (project setting
                            # `run_token_limit`): a cumulative cap on the
                            # run's total (input + output) tokens — the
                            # money safety net against a runaway run.
                            # Exhaustion raises UsageLimitExceeded, handled
                            # as a graceful stop below (see
                            # _agent_turn_budget_exhausted).
                            total_tokens_limit=self.config.run_token_limit,
                        ),
                    ) as run:
                        # Sync markers for the per-node re-render (below):
                        # view_dirty — content was produced while the task
                        # was off-screen, so the open chat view (if any) is
                        # missing it; _chat_epoch mismatch — the user
                        # switched views mid-node, and the switch-back
                        # render covered only the persisted transcript.
                        stream = NodeStreamView(
                            self,
                            task_id,
                            record_tps=True,
                            track_dirty=True,
                        )
                        synced_epoch = self._chat_epoch
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                stream.begin_node()
                                async with node.stream(run.ctx) as node_stream:
                                    try:
                                        async for event in node_stream:
                                            stream.feed(event)
                                    finally:
                                        # A mid-stream death (network error,
                                        # abort) propagates out of the run
                                        # loop, skipping the normal node
                                        # end: stop the blocks' spinners
                                        # here, or they keep animating
                                        # through the retry backoff (and
                                        # the outage wait).
                                        stream.finish_node()
                                self._agent_model_request_done(
                                    task, task_id, run, consumed, content, role
                                )
                            elif Agent.is_call_tools_node(node):
                                if await self._agent_tools_node(
                                    task, task_id, run_key, node, run, role, stream
                                ):
                                    # Mid-run compaction pause: the attempt
                                    # is marked unfinished and resumed via a
                                    # continuation turn (see the helper).
                                    break
                            # Keep the stored history current while the
                            # turn runs: view_task() renders from
                            # task.messages, so without a per-node snapshot
                            # a switch away+back mid-turn shows the pre-turn
                            # transcript and everything streamed before the
                            # switch is lost from the history.
                            self._persist_task_messages(task, run)
                    if run.result is not None:
                        task.messages = serialize_messages(run.result.all_messages())
                        self.manager.storage.save(task)
                    if self.viewed_id == task_id and (
                        stream.view_dirty or self._chat_epoch != synced_epoch
                    ):
                        # The streamed reasoning/assistant blocks are already
                        # in the chat: re-rendering from the persisted
                        # transcript after every node would remount the whole
                        # (growing) history each time and freeze the UI for
                        # seconds on fat transcripts. Do it only when the open
                        # view is actually out of sync: content arrived while
                        # the task was off-screen (view_dirty), or the user
                        # switched views mid-node — the switch-back render
                        # covered only the transcript persisted at the
                        # previous node boundary. The rebuild itself goes
                        # through the delta path: the streamed blocks are
                        # reconciled against the (now persisted) timeline and
                        # only the missing tail is appended.
                        self.view_task(task_id)
                        stream.view_dirty = False
                        synced_epoch = self._chat_epoch
                    # The turn went through: the network-retry budget for
                    # this task starts over (and the failed-queue rescue
                    # backoff with it — see the model-request branch above).
                    # A count past the budget = the run was in outage mode:
                    # announce the recovery before the reset.
                    self._agent_turn_succeeded(task, task_id)
                except asyncio.CancelledError:
                    # Interrupted (abort, nudge, or a newer run taking over).
                    # The consumed prompt is NOT re-queued — it stays in the
                    # transcript at its chronological position;
                    # task.unfinished marks the possibly-undone attempt and
                    # the next activation resumes it without re-sending text.
                    cancelled = True
                    self._agent_turn_cancelled(
                        task, task_id, run, consumed, content, generation
                    )
                    raise
                except UsageLimitExceeded as e:
                    # The run's budget (request_limit, or the per-run token
                    # budget `run_token_limit`) is spent: a graceful stop —
                    # the attempt is kept unfinished like after an abort
                    # (no requeue, no failed-queue exclusion), a service
                    # event reports what the run spent.
                    self._agent_turn_budget_exhausted(
                        task, task_id, run, consumed, content, generation, e
                    )
                    break
                except Exception as e:  # noqa: BLE001 - show agent errors in the chat
                    # A transient network error (transport failure like a
                    # connection refused/timeout, or HTTP 429 rate limiting)
                    # retries the run in place after a backoff (see the
                    # helper); anything else demotes the task to the queue.
                    if not _is_context_overflow_error(
                        e
                    ) and _is_retryable_network_error(e):
                        self._agent_turn_network_error(
                            task, task_id, run, consumed, content, generation, role, e
                        )
                        return
                    self._agent_turn_failed(
                        task, task_id, run, consumed, content, generation, e
                    )
                    break
        except asyncio.CancelledError:
            # Cancelled BEFORE the turn loop's own handler (worktree setup,
            # the task-start hook, the pre-turn compaction gate): the
            # pending prompt was never consumed. Mark the interrupted
            # attempt exactly like a mid-stream abort does — otherwise the
            # ESC stop leaves the task "active, nothing running" with no
            # resume marker (Ctrl+R / the restart-time auto-resume both key
            # off task.unfinished), and an abort racing the run's slow
            # setup window looks like the run silently ignores it.
            cancelled = True
            if self._agent_generations.get(task_id) == generation and (
                task.status is TaskStatus.ACTIVE
            ):
                task.unfinished = True
                self.manager.storage.save(task)
            raise
        except Exception as e:  # noqa: BLE001 - show preparation errors in the chat
            # A failure BEFORE the turn loop's own handler (worktree setup
            # incl. the bounded activation-lock waits, the task-start hook,
            # build_agent, the consumed-prompt assembly): without this the
            # worker would die silently — the task stays ACTIVE with no
            # runner and no event (the mute "active, nothing happening"
            # state, task #a2dc8424). Route it through the ordinary failure
            # path: a visible service event + honest requeue. `run` is None
            # here (nothing was consumed yet — pending_input is intact and
            # `_persist_task_messages` no-ops).
            self._agent_turn_failed(task, task_id, run, [], None, generation, e)
        finally:
            # Stop any spinner whose result event was lost to an abort or a
            # run error (normally all of them are already stopped).
            self._stop_run_tools(run_key)
            if self._agent_generations.get(task_id) == generation:
                # This is still the task's newest run: release its slot.
                # (Guarded: a superseded run must not steal a newer run's
                # entry — parallel slots make the other tasks' runs
                # untouchable by definition.)
                self._agent_runs.pop(task_id, None)
                # The run's factual model choice no longer serves
                # anything: drop the role's pin unless another live run
                # of the role still keeps it (parallel slots) — between
                # runs the resolution must reflect the current config
                # (see `invalidate_resolved_configs`).
                self._drop_run_pin_if_idle(role)
                self.refresh_status()
                # The run is over: the consumed prompt is either persisted in
                # the transcript or back in pending_input — the in-flight
                # record has served its purpose. (Guarded: a superseded run
                # must not drop a newer run's record.)
                self._consumed_prompts.pop(task_id, None)
                # Stamp the moment the run stopped writing to the
                # transcript (every exit path lands here: normal end,
                # abort, failure, awaiting-clarification). The pre-run
                # compaction gate measures its idle gap against this
                # stamp — a superseded run must not re-stamp (it would
                # shorten the pause the newer run's gate will see; the
                # newer run re-stamps at its own end anyway).
                task.last_run_end = datetime.now(UTC)
                self.manager.storage.save(task)
            # A cancelled run (user abort, or a newer run taking over) must
            # not restart the pipeline: the abort stops everything until the
            # user acts, and the newer run advances on its own completion.
            if not cancelled and self._agent_generations.get(task_id) == generation:
                await self._agent_run_epilogue(task, task_id)
                self._guard_zombie_exit(task, task_id)

    # -- run-loop helpers -------------------------------------------------------
    # (the decomposed pieces of _run_agent_worker; the worker above keeps the
    # turn loop's control flow readable, each concern lives below)

    async def _prepare_agent_run(
        self,
        task: Task,
        task_id: str,
        role: AgentRole,
        generation: int,
    ) -> tuple[
        Path,
        Agent[AgentDeps, str],
        str | None,
        SubagentGate | None,
        WebFetchGate | None,
        bool,
        ReadCache,
    ]:
        """Register the run and assemble everything the turn loop needs:
        the (worktree) root the tools run in, the built agent, the
        on_task_start hook's instructions for the first prompt, the run's
        subagent gate (None when delegation is off), the run's web-fetch
        gate (None unless this is a RESEARCH task with web fetch enabled)
        and the run's read_file dedup cache."""
        # Deferred heavy imports (pydantic_ai / agent_build / subagents):
        # keep them off the TUI cold-start path. `build_agent` resolves
        # through the module global — a seam tests monkeypatch (see the
        # module `__getattr__` below).
        _ensure_agent_names()
        self._agent_runs[task_id] = generation
        self._agent_failed.discard(task_id)
        self._agent_dead.discard(task_id)
        self._agent_net_retry_at.pop(task_id, None)
        self._current_tool[task_id] = None
        self._tps_samples.pop(task_id, None)
        self._touch_activity(task_id, bookkeeping=True)
        self.refresh_status()
        # Per-task worktree isolation (project setting
        # `task_isolation: worktree`, see drcode.task_env): the whole
        # run — tools, prompts, tester commit — happens in the task's
        # own worktree, so an interrupted (unfinished) task never
        # hands its dirty tree to the next one in the queue. The
        # worktree is created when missing and reused as-is (an
        # unfinished task resumes its own snapshot). Passing the task
        # resolves the run against its isolation PIN (stamped on first
        # activation) — a mid-task toggle of the setting must not move
        # the work between the worktree and the main checkout.
        pinned_before = task.isolation_pinned is not None
        root, iso_note = await asyncio.to_thread(
            self.task_env.ensure_root, task_id, task
        )
        self._isolation_pin_note(task, save=not pinned_before)
        if root != self.project_root:
            message = f"Task isolated in worktree {root} (branch drcode/{task_id})."
            if iso_note:
                message += f" ({iso_note})"
            # A network retry re-enters run_agent whole and reuses the
            # worktree created by the first attempt — announce it once,
            # not once per retry (see _show_event_once).
            self._show_event_once(task, message)
        elif iso_note:
            # The only remaining non-empty note is the benign
            # "no git repository" mode: a FAILED `worktree add` raises
            # WorktreeUnavailable now (no silent fallback into this
            # tree — the cb4937bb/540a500d incident) and is routed
            # through the preparation-failure handler below.
            self._show_event_once(task, f"Task isolation unavailable: {iso_note}")
        # Custom workflow (`.drcode/workflow.py`): the on_task_start
        # hook runs before the first activation (branch checkout etc.);
        # its instructions are prepended to the first consumed prompt.
        # With isolation on the hook runs INSIDE the worktree (see
        # _hook_env).
        workflow_instructions = await self._run_task_start_hook(task)
        # Read-only subagent delegation (project settings): the cap follows
        # the role serving this task (subagents_researcher for research,
        # subagents_coder for coding); 0 = the tool is not registered at
        # all and the gate stays None.
        subagent_limit = subagent_limit_for(self.settings, role)
        # The run's delegation budget, created BEFORE the agent build so
        # the SAME object does both jobs: it enforces the total slot count
        # in run_delegate_tool (deps carry it) and it makes the
        # delegation tail notice state-aware (build_agent closes a
        # supplier over it — the reminder names the remaining slots and
        # switches to the "slots used" text once the budget is spent).
        # The gate OBJECT is per run, but the BUDGET is per activation:
        # it is re-seeded from the task's mirrored counters, so a
        # non-user restart (network retry, comment takeover — run_agent
        # re-enters whole and lands here again) continues the spent
        # budget instead of returning the slots; the counters themselves
        # reset at the activation boundaries (promote_next of a
        # non-unfinished task, _nudge_run).
        subagent_gate = None
        if subagent_limit > 0:
            subagent_gate = SubagentGate(
                subagent_limit,
                used=task.subagent_slots_used,
                refunds=task.subagent_slot_refunds,
            )

            def _record_subagent_budget() -> None:
                task.subagent_slots_used = subagent_gate.used
                task.subagent_slot_refunds = subagent_gate.refunds
                self.manager.storage.save(task)

            subagent_gate.on_change = _record_subagent_budget

            # The decay window of the delegation tail notice counts the
            # TASK's cumulative model requests, not the build's: the
            # supplier below is handed to build_agent and called exactly
            # once per outgoing request (TailInjectionModel reads each
            # supplier once per request), incrementing the persisted
            # `task.model_request_count`. A re-entry into the run of the
            # same task — network retry, mid-run compaction/eviction
            # pause, watchdog restart, clarification answer — rebuilds the
            # agent and lands here again, but the counter lives on the
            # task, so the decay window CONTINUES instead of restarting
            # (#8c746feb: two activations of 9 requests each produced ~16
            # reminders instead of one window). Persisted metadata-only
            # (the transcripts are in memory during a run — a full `save`
            # would rewrite the megabyte sidecar per request). Reset only
            # when the user sends new input into a COMPLETED/TESTING task
            # (TaskManager.send's re-queue branch — a full new run).
            def _next_notice_turn() -> int:
                task.model_request_count += 1
                self.manager.storage.save_metadata(task)
                return task.model_request_count

            request_turns: Callable[[], int] | None = _next_notice_turn
        else:
            request_turns = None
        # Per-run read_file dedup (see read_cache.ReadCache): ONE object for
        # the whole run — shared across the per-turn deps rebuilds and a
        # mid-run compaction pause (created outside the turn loop for the
        # same reason the gates are), dying with the run closure so no
        # orphan map outlives the transcript it must stay consistent with.
        # A run restart (network retry, watchdog) starts a fresh cache: the
        # first re-read after a restart costs a full text once, then dedups
        # again — a safe degradation, never a wrong one.
        read_cache = ReadCache()
        # Managed web reading (project setting web_fetch_mode): RESEARCH
        # only — web_fetch_policy_for returns None for every other role, so
        # the tool is not registered there at all. Read at activation, so a
        # settings change reaches the next run of each task.
        web_fetch_policy = web_fetch_policy_for(self.settings, role)
        # Symbolic code index (project setting code_index_mode, §8.2's
        # role gating): coding/tester/resolver AND research agents get
        # the tools when ON, on the same terms; the decision rides the
        # deps so the delegate_research subagents inherit the parent
        # run's decision. Read at activation, so a settings change
        # reaches the next run of each task.
        code_index = code_index_tools_for(self.settings, role)
        agent = build_agent(
            self.resolved_config(role),
            kind=task.kind,
            root=root,
            research_git=self.settings.git_policy.research,
            subagent_limit=subagent_limit,
            # The run's delegation gate (created above, BEFORE the build):
            # makes the delegation tail notice state-aware and enforces
            # the budget inside run_delegate_tool.
            subagent_gate=subagent_gate,
            # The delegation notice's decay window keyed on the task's
            # cumulative request count (survives agent rebuilds); None
            # without a gate.
            request_turns=request_turns,
            # The research agent's web_fetch tool (SSRF-guarded, capped,
            # framed as untrusted content): None when the mode is off.
            research_web_fetch=web_fetch_policy,
            # The code-index tools (code_symbols/code_outline): None
            # effect at the OFF default — nothing is registered.
            code_index=code_index,
            # The project's `unrestricted` web-fetch mode lifts the
            # run_bash network guard for THIS run — run_bash's
            # description must say so (see build_agent).
            network_unrestricted=(
                self.settings.web_fetch_mode is WebFetchMode.UNRESTRICTED
            ),
            # The real scratch path is baked into the write tools'
            # descriptions (no literal <task-id> placeholder for the model
            # to copy verbatim into a path argument).
            task_id=task_id,
            # The eviction-hint nudge is opt-in (the project's
            # eviction_hint setting): read at activation, so a settings
            # toggle applies to the next activation of each task.
            eviction_hint=self.settings.eviction_hint,
            # The live checklist tail (agent_model.TailInjectionModel): the
            # supplier re-reads the task per REQUEST, so a todo_create /
            # todo_update executed mid-run is visible to that run's next
            # model request — the same liveness the old dynamic-instructions
            # hook provided, but injected at the request TAIL so a checklist
            # mutation never invalidates the cached leading prefix (the
            # #a0a55d3d production incident). Non-feature tasks and empty
            # lists yield None (pass-through). Opt-in via the project's
            # todo_tail setting: when off the model is not wrapped at all —
            # no request modification whatsoever (the checklist still lives
            # in task.todos and stays visible in the TUI). Read at
            # activation, so a toggle applies to the next activation.
            todo_supplier=(
                partial(self._todo_tail_block, task_id)
                if self.settings.todo_tail
                else None
            ),
            # Provider slot-wait visibility: when the resolved provider's
            # max_parallel_requests cap is saturated, a request queued for
            # longer than agent_model.THROTTLE_WAIT_ANNOUNCE_SECONDS says so
            # in the task's chat instead of looking like a hung run.
            announce_wait=self._wait_announcer(task_id, role),
            # External MCP servers (docs/mcp-client-design.md §2): read at
            # activation from the user + project mcp-servers.json layers,
            # gated by the roles matrix (empty = none, the safe default).
            # Note the MAIN project root — under worktree isolation the
            # project config lives in the main checkout's .drcode/.
            mcp_toolsets=self._mcp_toolsets(task, role),
        )
        # One gate per activation: it carries the run's policy AND the
        # fetch budget, so a superseded run's gate dies with it (the same
        # reasoning as the subagent gate). The BUDGET is per activation:
        # seeded from the task's mirrored counter, so a non-user restart
        # (network retry, demote-on-failure — run_agent re-enters whole
        # and lands here again) continues the spent budget instead of
        # returning the fetches; `on_change` mirrors every spend/refund
        # back onto the task. The counter resets at the activation
        # boundaries (promote_next of a non-unfinished task, _nudge_run).
        web_fetch_gate = None
        if web_fetch_policy is not None:
            fetch_gate = WebFetchGate(web_fetch_policy, used=task.web_fetch_used)
            web_fetch_gate = fetch_gate

            def _record_web_fetch_budget() -> None:
                task.web_fetch_used = fetch_gate.used
                self.manager.storage.save(task)

            fetch_gate.on_change = _record_web_fetch_budget
        return (
            root,
            agent,
            workflow_instructions,
            subagent_gate,
            web_fetch_gate,
            code_index,
            read_cache,
        )

    async def _consume_turn_input(
        self,
        task: Task,
        task_id: str,
        workflow_instructions: str | None,
    ) -> (
        tuple[str | list[str | BinaryContent] | None, list[PendingInput], str | None]
        | None
    ):
        """Consume the next turn's input (chronological message flow: the
        transcript is the single chronological record — the original
        prompt lives in it exactly once and is NEVER re-queued/re-sent;
        each activation first consumes every pending user message in
        arrival order — merged into ONE prompt, so a brief queued up
        before the run (original message + comments) is solved as a
        whole, not turn-by-turn — then, only if the last attempt was
        interrupted, runs one continuation turn without new text to pick
        the interrupted work back up).

        Returns (prompt content, consumed entries, possibly-injected
        workflow instructions) or None when there is nothing to run (no
        queued input, no unfinished attempt) — the caller ends the run.
        The content is a plain str for a text-only batch; when any
        consumed entry carries image attachments it is a multimodal
        content list (text element first, then one BinaryContent per
        image — pydantic-ai accepts it as UserContent directly). The
        chat-display text of the batch stays a plain str regardless
        (see _consumed_prompts below and the `_write_history_line`
        loop, which render `entry.text`).

        Async because of the interrupted-but-finished branch below: the
        mid-run gate can pause the run on the TERMINAL empty tools node
        pydantic-ai emits right after the final text response (the fill
        crossed the threshold exactly at that boundary) — the exchange is
        already complete, but the pause announced a compaction, and this
        is the last safe point to honor it: the turn-boundary gate would
        never run again (the loop ends here), and the task would finish
        over the threshold with the promise unkept."""
        if task.pending_input:
            # Batch consumption: everything queued before this turn
            # (the original brief plus comments typed while the task
            # sat in the queue or while another turn ran) is sent as
            # a single prompt — the agent must see the whole picture
            # at once and decide from all of it.
            consumed = list(task.pending_input)
            task.pending_input.clear()
            text = "\n\n".join(entry.text for entry in consumed)
            if workflow_instructions:
                text = (
                    "# Project workflow instructions\n\n"
                    f"{workflow_instructions}\n\n{text}"
                )
                workflow_instructions = None
            content: str | list[str | BinaryContent] = text
            attachments = [a for entry in consumed for a in entry.attachments]
            if attachments:
                content = self._build_multimodal_prompt(
                    task, task_id, text, attachments
                )
            # Remember the consumed batch until the transcript picks
            # it up (the first node-boundary persist): a view_task()
            # re-render in between finds neither copy and wipes the
            # prompt from the chat — the record lets
            # _render_task_history draw it anyway. Dropped when the
            # run ends or the batch is re-queued (below). The record
            # keeps the plain TEXT (never the multimodal content list):
            # it is a chat-display projection, not model context — the
            # base64 payloads must not ride through the render path.
            self._consumed_prompts[task_id] = (text, list(consumed))
            if self.viewed_id == task_id:
                for entry in consumed:
                    if entry.text in self._pending_rendered:
                        # Already shown as a queued block; don't repeat it.
                        self._pending_rendered.remove(entry.text)
                    elif not is_pending_retry(task.messages, entry.text):
                        # Not already visible from the transcript either
                        # (an unfinished attempt's prompt renders from
                        # there) — draw the user block once.
                        self._write_history_line("user", entry.text)
            return content, consumed, workflow_instructions
        if task.unfinished:
            if not ends_with_open_request(
                task.messages
            ) and not ends_with_truncated_response(task.messages):
                # The interrupted attempt actually finished its
                # exchange — nothing to resume. But the pause that set
                # the flag may have announced a compaction (the gate
                # fires on every tools node, including the terminal
                # empty one after the final response): run the
                # turn-boundary gate here, before ending the run — the
                # announced compaction must actually happen (no-op when
                # the fill is under the threshold, or the livelock
                # guard / a failed summarizer keeps it silent).
                await self._maybe_compact(task)
                task.unfinished = False
                self.manager.storage.save(task)
                return None
            # A tail response that still carries unanswered tool calls (or
            # no content at all — `ends_with_truncated_response`) is NOT a
            # finished exchange: the flag stays ON and the continuation
            # turn below resumes the attempt (`resumable_history` trims
            # the invalid tail and the model re-generates). Clearing the
            # flag on a mere response-tail made the run "unresumable" and
            # the epilogue read the tool-call tail as a clean finish
            # (incident #46c6314d: a stale network-retry timer superseded
            # the live run mid-tool, the fresh run saw a response-tail,
            # dropped the flag and auto-completed the task).
            # The unfinished flag stays ON through the continuation turn:
            # clearing it here (before the model request even went out)
            # lost the resume marker when the run was cancelled in that
            # window — a superseded run's _agent_turn_cancelled skips the
            # restore via its generation guard, and the epilogue then
            # auto-completed the task with zero work (incident 88dd4aad:
            # Ctrl-R during an outage wait raced the outage probe's own
            # restart). The flag is cleared at the SAFE moment — the turn
            # boundary after the response lands (the
            # `not ends_with_open_request` branch above), while the
            # cancel / network-error / failure handlers all (re)set it.
            # The unfinished flag stays ON through the continuation turn:
            # clearing it here (before the model request even went out)
            # lost the resume marker when the run was cancelled in that
            # window — a superseded run's _agent_turn_cancelled skips the
            # restore via its generation guard, and the epilogue then
            # auto-completed the task with zero work (incident 88dd4aad:
            # Ctrl-R during an outage wait raced the outage probe's own
            # restart). The flag is cleared at the SAFE moment — the turn
            # boundary after the response lands (the
            # `not ends_with_open_request` branch above), while the
            # cancel / network-error / failure handlers all (re)set it.
            return None, [], workflow_instructions
        return None

    # Caps of ONE consumed prompt's image attachments: the multimodal
    # content is serialized into the transcript as base64 and re-sent to
    # the API on every request of the run — a pile of screenshots must not
    # blow the context window in a single turn.
    MAX_PROMPT_IMAGES = 6
    MAX_IMAGE_BYTES = 5 * 1024 * 1024

    def _build_multimodal_prompt(
        self,
        task: Task,
        task_id: str,
        text: str,
        attachments: list[ImageAttachment],
    ) -> list[str | BinaryContent]:
        from pydantic_ai.messages import BinaryContent

        """The multimodal content of the consumed prompt: the text element
        first, then one `BinaryContent` per attached image.

        Limits: at most MAX_PROMPT_IMAGES images per prompt, at most
        MAX_IMAGE_BYTES bytes per image — anything over is skipped with a
        service event, never silently. A missing file (a cleanup raced the
        consumption, or the metadata outlived a manual deletion) degrades
        to a placeholder text item: the run must never die because an
        attachment vanished."""
        items: list[str | BinaryContent] = [text] if text else []
        taken = 0
        dropped: list[str] = []
        for att in attachments:
            if taken >= self.MAX_PROMPT_IMAGES:
                dropped.append(
                    f"{att.filename} (over the {self.MAX_PROMPT_IMAGES}-image prompt limit)"
                )
                continue
            data = self.manager.storage.read_attachment(task_id, att)
            if data is None:
                items.append(f"[image {att.media_type} — file missing]")
                continue
            if len(data) > self.MAX_IMAGE_BYTES:
                dropped.append(
                    f"{att.filename} ({len(data)} bytes, over the "
                    f"{self.MAX_IMAGE_BYTES}-byte per-image limit)"
                )
                continue
            items.append(BinaryContent(data=data, media_type=att.media_type))
            taken += 1
        if dropped:
            self._show_event(
                task,
                "Skipped attached image(s): " + "; ".join(dropped) + ".",
            )
        return items

    def _agent_model_request_done(
        self,
        task: Task,
        task_id: str,
        run: AgentRun[AgentDeps, str],
        consumed: list[PendingInput],
        content: str | list[Any] | None,
        role: AgentRole,
    ) -> None:
        """Bookkeeping after one completed model request node: backdate
        the just-recorded prompt, refresh the live context counter, fold
        the usage into the session totals and reset the network-retry /
        failed-queue-rescue budgets."""
        # The consumed prompt is now in the transcript (the
        # model-request node just appended it). Backdate it to the moment
        # the user typed it, so the chat thread stays chronological even
        # when the prompt sat in pending_input for a while before this
        # run picked it up.
        if consumed and content is not None:
            backdate_user_prompt(
                run.all_messages(),
                content,
                consumed[0].at,
            )
        # The stream is closed: this request's usage is final. Refresh
        # the context counter in the task status bar for the running
        # task.
        ctx_tokens = run_context_tokens(run)
        if ctx_tokens is not None:
            self._live_ctx_tokens[task_id] = ctx_tokens
            # A fresh usage reading postdates any batch eviction: the
            # eviction credit (which corrected the pre-eviction readings)
            # is consumed — the real numbers speak for themselves now.
            self._evicted_ctx_tokens.pop(task_id, None)
            if self.viewed_id == task_id:
                self.refresh_task_status()
        # Exact per-request usage of the completed model call, into the
        # session totals and the persistent per-day usage store.
        usage = last_response_usage(run)
        model_key = self.resolved_model_key(role)
        self._record_session_usage(usage, role, model_key, task_id=task_id)
        # The same request feeds the cache hit-rate watchdog: 5 consecutive
        # low-hit requests on the same task + model key drop one deduped
        # service event (see core.usage.CacheHitRateMonitor) — the
        # prefix-cache invalidations (#a0a55d3d) are otherwise invisible
        # until the bill arrives.
        self._cache_hit_monitor.record(task_id, f"{role.value}/{model_key}", usage)
        # A model call went through: the network-retry budget for this
        # task starts over, so the NEXT network error counts from one
        # again — even mid-run, where a single turn makes many model
        # requests. A count past the budget means the run was in outage
        # mode: this success ends it (announce, then pop).
        self._note_network_recovered(task, self._agent_net_count)
        self._agent_net_count.pop(task_id, None)
        self._agent_net_retry_at.pop(task_id, None)
        # The model works again: the failed-queue rescue backoff (see
        # _promote_failed_queue) starts over, so a later single failure
        # is rescued immediately, not after a stale backoff window.
        self._failed_queue_rounds = 0
        self._failed_queue_retry_at = 0.0

    def _backdate_consumed_prompt(
        self,
        run: AgentRun[AgentDeps, str] | None,
        consumed: list[PendingInput],
        content: str | list[Any] | None,
    ) -> None:
        """Backdate the consumed prompt the run's transcript already holds.

        The exception handlers below run when a request node died BEFORE
        its completion bookkeeping (`_agent_model_request_done`, the only
        normal backdating point) — the prompt sits in the transcript with
        the CONSUMPTION stamp, and persisting it as is freezes that stamp:
        the resumed attempt is a continuation turn with no consumed input,
        so nothing ever corrects it and the prompt renders BELOW the
        service events that happened between typing and consumption. This
        applies the same backdating the node completion would have — a
        no-op when the prompt never reached the transcript (the caller
        re-queues it with its original stamps then)."""
        if run is None or not consumed or content is None:
            return
        try:
            messages = run.all_messages()
        except Exception:  # noqa: BLE001 - state may be mid-teardown after cancel
            return
        backdate_user_prompt(messages, content, consumed[0].at)

    async def _agent_tools_node(
        self,
        task: Task,
        task_id: str,
        run_key: tuple[str, str, int],
        node: Any,
        run: AgentRun[AgentDeps, str],
        role: AgentRole,
        stream: NodeStreamView,
    ) -> bool:
        """Render one call-tools node (tool-call blocks + spinners) and
        run the mid-run compaction gate at the tool boundary.

        Returns True when the gate paused the run (the caller breaks out
        of the node loop; the attempt is marked unfinished and resumed
        via a continuation turn).

        The gate exists because a single turn can run for many model
        requests, and the turn-boundary gate would only fire when the
        whole run ends — the fill can grow far past the threshold inside
        one turn. Stopping here: the tool results of the round just
        executed are the last thing the model saw, and a break leaves
        them uncommitted in run.all_messages() (the transcript ends with
        unanswered tool calls — sanitized away, the redo of that one
        round is the price of the pause). Skipped when compaction keeps
        failing at the THRESHOLD level, otherwise a persistently broken
        summarizer would loop break→continuation→break — but the
        HARD-LIMIT level fires regardless: growth past it means a
        deterministic death of the run, so the gate must bypass
        `_compaction_failed` (unless even the hard-level escalation
        proved unable to shrink the transcript —
        `_hard_limit_exhausted`)."""
        from pydantic_ai.messages import (
            FunctionToolCallEvent,
            FunctionToolResultEvent,
        )

        async with node.stream(run.ctx) as node_stream:
            async for event in node_stream:
                if isinstance(event, FunctionToolCallEvent):
                    self._touch_activity(task_id, event.part.tool_name)
                elif isinstance(event, FunctionToolResultEvent):
                    # Result stops the spinner even when the task is
                    # off-screen.
                    self._tool_call_finished(event.part.tool_call_id)
                    self._touch_activity(task_id, done=True)
                    # A todo tool just mutated the checklist: refresh the
                    # panel of the open chat view (the 1s poll would catch
                    # it too — this is the instant path).
                    if (event.part.tool_name or "").startswith("todo_"):
                        self.host.host_call_after_refresh(self.refresh_todo_panel)
                if self.viewed_id != task_id:
                    # The tool-call block is skipped (it will come from the
                    # next re-render): the open view is missing it.
                    stream.view_dirty = True
                    continue
                if isinstance(event, FunctionToolCallEvent):
                    args = event.part.args
                    args_str = (
                        args
                        if isinstance(args, str)
                        else json.dumps(args or {}, ensure_ascii=False)
                    )
                    call_text = f"{event.part.tool_name}({args_str})"
                    block = self.host.host_new_tool_call_block(call_text)
                    self._tool_call_started(
                        run_key,
                        task_id,
                        event.part.tool_call_id,
                        call_text,
                        block,
                    )
                    block.start_spinner()
                    self._write_chat(block, note=("tool-call", call_text, None))
        tokens = self._compaction_due(task)
        # Under the threshold: credit progress toward the progress-gated
        # reset of the threshold-livelock counter (a single observation
        # proves nothing — the first boundary of a resumed segment reads
        # the compacted transcript's lagging live usage, which predates
        # the tool result that re-crosses the threshold — see
        # _threshold_observation). Over it: the streak breaks.
        self._threshold_observation(task, tokens, self._threshold_compactions)
        hard_due = tokens is not None and self._hard_limit_due(task)
        # A background summary maturing (or ready to land at the next turn
        # boundary) suppresses the THRESHOLD-level pause exactly like it
        # suppresses the turn-boundary gate; the HARD-LIMIT level stays
        # active — it is the crash-guard, not an ordinary compaction.
        bg_held = task.id in self._background_compactions
        if tokens is not None and (
            (not bg_held and task.id not in self._compaction_failed)
            or (hard_due and task.id not in self._hard_limit_exhausted)
        ):
            pct = round(tokens * 100 / self.resolved_config(role).max_context_tokens)
            self._show_event(
                task,
                f"Context fill {pct}% — pausing the run to compact history…",
            )
            self._persist_task_messages(task, run)
            task.messages = sanitized_messages(task.messages)
            task.unfinished = True
            self.manager.storage.save(task)
            return True
        # Selective-eviction gate (mark_unneeded marks + the automatic
        # tool result clearing): the threshold is NOT reached, but the
        # agent-marked results have piled up to a third of the window, or
        # the auto-evictable old results to AUTO_EVICTION_PAUSE_FRACTION-th
        # of it — pause the run so the batch eviction can collapse them
        # between turns (the actual eviction happens in _maybe_compact at
        # the next turn boundary, exactly like a mid-run compaction
        # pause).
        if self._eviction_due(task):
            pause_event = (
                "Marked tool results reached ≈1/3 of the context window — "
                "pausing the run to unload them…"
            )
        elif self._auto_eviction_due(task):
            pause_event = (
                "Old tool results (automatic clearing) reached ≈1/6 of the "
                "context window — pausing the run to unload them…"
            )
        elif self._stale_thinking_due(task):
            pause_event = (
                "Stale thinking parts (sliding-window strip) reached ≈2/3 "
                "of the context window — pausing the run to strip them…"
            )
        else:
            return False
        self._show_event(task, pause_event)
        self._persist_task_messages(task, run)
        task.messages = sanitized_messages(task.messages)
        task.unfinished = True
        self.manager.storage.save(task)
        return True

    def _agent_turn_succeeded(self, task: Task, task_id: str) -> None:
        """The turn went through: the network-retry budget for this task
        starts over (and the failed-queue rescue backoff with it — see
        _agent_model_request_done). A count past the budget = the run
        was in outage mode: announce the recovery before the reset."""
        self._note_network_recovered(task, self._agent_net_count)
        self._agent_net_count.pop(task_id, None)
        self._agent_net_retry_at.pop(task_id, None)
        self._failed_queue_rounds = 0
        self._failed_queue_retry_at = 0.0

    def _agent_turn_cancelled(
        self,
        task: Task,
        task_id: str,
        run: AgentRun[AgentDeps, str] | None,
        consumed: list[PendingInput],
        content: str | list[Any] | None,
        generation: int,
    ) -> None:
        """Interrupted (abort, nudge, or a newer run taking over): keep
        whatever the interrupted turn managed to do, sanitized. The
        consumed prompt is NOT re-queued — it stays in the transcript at
        its chronological position (unless it never reached it — then it
        goes back to pending_input); task.unfinished marks the
        possibly-undone attempt and the next activation resumes it
        without re-sending text."""
        if self._agent_generations.get(task_id) == generation and (
            task.status is TaskStatus.ACTIVE
        ):
            # Persist whatever the interrupted turn managed to do, then
            # sanitize: drop the API-invalid tail (tool calls without
            # results, the empty interrupted request) so later messages
            # can append in any order.
            self._backdate_consumed_prompt(run, consumed, content)
            self._persist_task_messages(task, run)
            if run is None and consumed:
                # Cancelled before the run context was even entered (e.g.
                # during the compaction gate): the prompts were popped
                # from pending_input but never reached the transcript —
                # put them back, or the nudge/abort would silently eat
                # them.
                task.pending_input[:0] = consumed
                # Back to pending: the pending-input render owns the
                # bubbles again, drop the in-flight record.
                self._consumed_prompts.pop(task_id, None)
                if self.viewed_id == task_id:
                    for entry in consumed:
                        if entry.text not in self._pending_rendered:
                            self._pending_rendered.append(entry.text)
            task.messages = sanitized_messages(task.messages)
            task.unfinished = True
            self.manager.storage.save(task)
        # Drop our streaming preview; a newer run (if any) owns the label
        # now and must not be touched.
        self._drop_stream_preview(task_id, generation)

    def _agent_turn_budget_exhausted(
        self,
        task: Task,
        task_id: str,
        run: AgentRun[AgentDeps, str] | None,
        consumed: list[PendingInput],
        content: str | list[Any] | None,
        generation: int,
        error: Exception,
    ) -> None:
        """The run hit its usage budget (pydantic-ai's UsageLimits — the
        per-run token budget `run_token_limit`, or `request_limit`):
        stop GRACEFULLY instead of failing. Exactly the abort path's
        transcript care (persist + sanitize + `task.unfinished`), but the
        task KEEPS its ACTIVE status — it stays paused on the unfinished
        flag, and a new message (or a restart's auto-resume) continues the
        attempt; the queue is not touched and the failed-queue exclusion
        does not fire. A service event reports the spent amount and points
        at the context dialog (🧠 in the task status bar) and the knob that
        lifts the budget."""
        spent = 0
        if run is not None:
            try:
                # `usage` is a property (RunUsage) — the cumulative spend of
                # the run so far, exact API numbers (never the chars/4 ≈).
                spent = run.usage.total_tokens or 0
            except Exception:  # noqa: BLE001, S110 - reporting never fatal
                pass
        limit = self.config.run_token_limit
        limit_txt = f"{limit:,}" if limit else "the configured"
        self._show_event(
            task,
            f"Run budget exhausted ({spent:,} ≈tokens spent of {limit_txt}): "
            "the attempt is kept unfinished — send a message (or restart "
            "drcode) to resume it. The 🧠 counter in the task status bar "
            "opens the context dialog with the per-agent token layout; "
            "raise `run_token_limit` in drcode.json (env: "
            "DRCODE_RUN_TOKEN_LIMIT) to lift the budget.",
        )
        if self._agent_generations.get(task_id) == generation and (
            task.status is TaskStatus.ACTIVE
        ):
            # The same transcript care an abort takes: persist whatever the
            # interrupted turn managed to do, then sanitize the API-invalid
            # tail so later messages can append in any order.
            self._backdate_consumed_prompt(run, consumed, content)
            self._persist_task_messages(task, run)
            task.messages = sanitized_messages(task.messages)
            task.unfinished = True
            self.manager.storage.save(task)
        if self._agent_generations.get(task_id) == generation:
            self._drop_stream_preview(task_id, generation)

    def _agent_turn_network_error(
        self,
        task: Task,
        task_id: str,
        run: AgentRun[AgentDeps, str] | None,
        consumed: list[PendingInput],
        content: str | list[Any] | None,
        generation: int,
        role: AgentRole,
        error: Exception,
    ) -> None:
        """Retry the run in place after a backoff delay (5s doubling up
        to 60s), at most NETWORK_RETRIES times, instead of requeueing the
        task right away. Past the budget (consecutive failures with no
        successful model call — a sustained outage, e.g. the machine
        slept through the gateway's downtime) the run does NOT fall
        through to the ordinary failure path: it enters outage mode (see
        usage.py) — the task keeps its status and keeps retrying in
        place on a longer backoff until the connection returns. The
        caller returns right after (the task stays ACTIVE — no requeue,
        no pipeline advance; the delayed re-run picks it up)."""
        attempt = self._agent_net_count.get(task_id, 0) + 1
        self._agent_net_count[task_id] = attempt
        self._net_retries_total += 1
        outage = _in_outage(attempt)
        delay = _network_retry_delay(attempt, outage)
        if outage:
            if attempt == NETWORK_RETRIES + 1:
                self._show_event(
                    task,
                    f"Network outage: the model gateway has been "
                    f"unreachable for {NETWORK_RETRIES} retries "
                    f"({type(error).__name__}) — the task keeps its "
                    f"status and waits out the outage, retrying "
                    f"in place until the connection returns.",
                )
            else:
                self._show_event(
                    task,
                    f"Network outage: still unreachable "
                    f"({type(error).__name__}) — retry {attempt} "
                    f"in {int(delay)} s",
                )
        else:
            self._show_event(
                task,
                f"Network error ({type(error).__name__}): retry {attempt}/"
                f"{NETWORK_RETRIES} in {int(delay)} s",
            )
        if self._agent_generations.get(task_id) == generation:
            self._drop_stream_preview(task_id, generation)
        # Whatever re-enters run_agent from here (the retry timer, or the
        # outage probe ticks once the gateway answers TCP) is an IN-PLACE
        # resume: the task keeps its ACTIVE status, so the resumed run
        # must not repeat the "Model:" line (see _emit_run_model_event).
        self._agent_retry_resumes.add(task_id)
        # The request node died mid-flight: the consumed prompt is already
        # in the transcript with the consumption stamp (the completion
        # bookkeeping that backdates it was skipped) — correct it before
        # the persist freezes the wrong stamp.
        self._backdate_consumed_prompt(run, consumed, content)
        self._persist_task_messages(task, run)
        task.messages = sanitized_messages(task.messages)
        if run is None and consumed:
            # The request never started (pydantic-ai appends the user
            # message before calling the model, and with `run` None
            # nothing was persisted): put the consumed prompts back at the
            # head of the pending queue so the retry re-sends them
            # verbatim (with their original "typed at" stamps).
            task.pending_input[:0] = consumed
            # Back to pending: the pending-input render owns the bubbles
            # again, drop the in-flight record.
            self._consumed_prompts.pop(task_id, None)
            # They were already rendered as user blocks when they were
            # popped: the retry must not draw them twice.
            if self.viewed_id == task_id:
                for entry in consumed:
                    if entry.text not in self._pending_rendered:
                        self._pending_rendered.append(entry.text)
        if ends_with_open_request(task.messages):
            # The request is recorded in the transcript: the retry
            # resumes the attempt without new text.
            task.unfinished = True
        retry_at = time.monotonic() + delay
        self._agent_net_retry_at[task_id] = retry_at
        self.manager.storage.save(task)
        if outage:
            # Outage mode: the re-run is gated behind a cheap TCP probe of
            # the gateway host — while the host is unreachable the tick
            # only re-probes (see _outage_retry_tick) instead of firing
            # full model requests at a dead gateway.
            self._schedule_net_retry(
                task_id,
                retry_at,
                delay,
                partial(self._outage_retry_tick, task_id, role),
            )
        else:
            self._schedule_net_retry(
                task_id, retry_at, delay, partial(self.run_agent, task_id)
            )

    def _agent_turn_failed(
        self,
        task: Task,
        task_id: str,
        run: AgentRun[AgentDeps, str] | None,
        consumed: list[PendingInput],
        content: str | list[Any] | None,
        generation: int,
        error: Exception,
    ) -> None:
        """A non-network turn failure: show it, persist the sanitized
        transcript, mark the attempt unfinished and demote the task back
        to the queue (excluded from promotion for the session via
        `_agent_failed`, so a dead gateway can't loop the queue). A
        deterministic failure (context overflow, or another known request
        rejection) additionally marks the task dead (the failed-queue
        rescue must never lift it back)."""
        # Persist the failure even for a superseded run: it happened, and
        # the human must be able to see it.
        self._show_event(task, f"{type(error).__name__}: {error}")
        if self._agent_generations.get(task_id) == generation:
            # A mid-stream failure leaves a half-streamed orange preview
            # above the input: clear it, otherwise the stale fragments
            # hang in the dialog (and would be shown for whatever the
            # user does next).
            self._drop_stream_preview(task_id, generation)
        self._backdate_consumed_prompt(run, consumed, content)
        self._persist_task_messages(task, run)
        task.messages = sanitized_messages(task.messages)
        # Not lost: unfinished marks the attempt; promotion resumes it
        # (this session it is excluded via _agent_failed so a dead
        # gateway can't loop the queue).
        task.unfinished = True
        # Demote honestly: an ACTIVE task with no runner is a state
        # nothing ever picks up again (promote_next skips it) — the task
        # goes back to the queue instead, and the queue continues with
        # the next task.
        task.status = TaskStatus.QUEUED
        self._agent_failed.add(task_id)
        if _is_context_overflow_error(error):
            # Deterministic failure: the transcript itself is over the
            # model's context window, so every retry of the same history
            # is guaranteed the same 400. Mark the task dead — the
            # failed-queue rescue must never lift it back into
            # development — and tell the user what manual fix is needed.
            self._agent_dead.add(task_id)
            self._show_event(
                task,
                "The transcript exceeds the model's context "
                "window: the model rejects it deterministically "
                "on every attempt, retrying cannot help. The "
                "task needs manual intervention — delete it, or "
                "shrink its transcript (e.g. mechanical "
                "compaction).",
            )
            self.notifier.task_context_overflow(task.title)
        elif _is_deterministic_request_rejection(error):
            # Deterministic failure of a different kind: the gateway
            # rejects the request itself (e.g. system-message placement
            # rules — the #83431cb3 carousel), so every retry of the same
            # history is guaranteed the same 400. Mark the task dead —
            # the failed-queue rescue must never lift it back into
            # development — and tell the user what manual fix is needed.
            self._agent_dead.add(task_id)
            self._show_event(
                task,
                "The model deterministically rejects this "
                "request (HTTP 400): the same request is "
                "refused on every attempt, retrying cannot "
                "help. The task needs manual intervention — "
                "fix the model/gateway configuration for "
                "this project (e.g. switch the model), or "
                "delete the task.",
            )
            self.notifier.task_request_rejected(task.title)
        self.manager.storage.save(task)
        # The transition happened outside the manager — report it to the
        # status observers (notifications) explicitly.
        self.manager.fire_status_change(task, TaskStatus.ACTIVE, "system")

    async def _agent_run_epilogue(self, task: Task, task_id: str) -> None:
        """The end-of-run pipeline continuation (a non-cancelled run that
        is still the task's newest): an answered clarification is
        re-queued, a finished task is auto-completed (research tasks with
        their own finalization), a queue-demoted task keeps the pipeline
        going with the next one."""
        # A background summary that ripened while the last turn ran: this
        # is the last stitch point the run offers (the loop breaks before
        # the gate ticks again). An in-flight one is left alone — the
        # worker stitches on completion, the transcript is stable now.
        self._finish_background_compaction(task)
        if (
            task.status is TaskStatus.ACTIVE
            and not task.pending_input
            and not task.unfinished
            and ends_with_truncated_response(task.messages)
        ):
            # A "clean" run end whose final exchange is actually OPEN: the
            # tail response either carries no usable content at all (no
            # text, no tool call) or carries a tool call whose result
            # never arrived (a tool result is a separate request message
            # — the same invalid tail `resumable_history` trims). The
            # model did not finish, so this is not a completed answer.
            # Auto-completing here would mark the task done with zero
            # content and tear down its worktree/branch (incident
            # #0dc7fc65: a watchdog nudge resumed an interrupted attempt,
            # the gateway answered with an empty response, the run loop
            # saw a closed exchange and this epilogue completed the
            # research task — the partially-done work was deleted with
            # the worktree; incident #46c6314d: the tail response still
            # held an unanswered tool call, the narrower empty-response
            # guard passed it, and the task went to TESTING mid-work).
            # Treat it like an interrupted attempt instead: keep the task
            # ACTIVE paused on the unfinished flag — a new message or a
            # restart (auto-resume) picks the attempt back up.
            task.unfinished = True
            self.manager.storage.save(task)
            self._show_event(
                task,
                "The run ended on an unfinished exchange (a response with "
                "tool calls no result arrived for, or an empty response) "
                "— the attempt is kept unfinished instead of being "
                "completed; send a message to resume it.",
            )
            self.refresh_sidebar()
            return
        if task.status is TaskStatus.AWAITING_CLARIFICATION:
            # The user may have answered while the run was still finishing
            # (the answer sits in pending_input and send()'s clarification
            # transition never fired): don't wait forever for an answer
            # that has already arrived.
            self.manager.requeue_answered_clarification(task.id)
            # The agent paused the task on a clarifying question: show it
            # and let the pipeline continue while the user answers. Skip
            # when the end-of-turn view_task() re-render above already
            # drew it — otherwise the question shows twice while the
            # dialog stays open (a switch away+back renders it once, via
            # view_task alone).
            if (
                self.viewed_id == task_id
                and task.clarification
                and task.clarification is not self._rendered_clarification
            ):
                self._write_clarification(task.clarification)
                self._rendered_clarification = task.clarification
            self.refresh_sidebar()
            self._advance()
        elif (
            task.status is TaskStatus.ACTIVE
            and not task.pending_input
            and not task.unfinished
        ):
            # The agent answered and nothing is pending or left to resume:
            # the task is done automatically (as if the user pressed
            # done). An unfinished (interrupted) task stays ACTIVE paused
            # instead — the poller surfaces it as "active, nothing
            # running" until the user acts or the app restarts (promotion
            # then resumes the attempt). Custom workflow:
            # on_development_done (push / open a PR) runs BEFORE mark_done
            # — between the hook and the TESTING transition there is no
            # await point, so no concurrent _advance() can start the
            # tester round without the hook's note.
            await self._run_development_done_hook(task)
            # Completion soft gate: unchecked checklist items are
            # announced, not blocking (see _todo_completion_warning).
            self._todo_completion_warning(task)
            self.manager.mark_done(task.id)
            self.refresh_sidebar()
            # No view_task() here: the transcript is fully on screen
            # already (streamed live, or re-rendered by the end-of-turn
            # dirty-check above) and mark_done changes only the status — a
            # full wipe-and-remount would flash the whole chat for nothing
            # (service events draw themselves, see _task_env_event /
            # _show_event).
            if task.kind is TaskKind.RESEARCH:
                # Research is complete (no tester): anchor the snapshot on
                # the branch, then the research accept hook (merge the
                # docs branch) or the unmerged-work reminder — the
                # research counterpart of the PASSED-finalization steps
                # the tester path runs.
                await self._finalize_research_completion(task)
                self.refresh_sidebar()
            # A test that can start right now (this task or an older one
            # in the backlog) always goes before any new development; under
            # parallel slots the rest of the queue flows while the tester
            # is busy.
            self._advance()
        elif task.status is TaskStatus.QUEUED:
            # The run failed and demoted the task back to the queue: keep
            # the pipeline going with the next task (failed ones are
            # excluded, so a dead gateway can't loop the queue).
            self._advance()

    def _guard_zombie_exit(self, task: Task, task_id: str) -> None:
        """Last-resort guard at the very end of a run. By this point the
        epilogue must have resolved the task: completed it, re-queued a
        failure, or left it deliberately paused on an unfinished attempt
        (abort / mid-run compaction pause — those handlers leave their
        own traces) or behind a scheduled network retry. Reaching this
        with the task still ACTIVE, owned by nobody and paused by
        nothing is the silent-worker-exit state (incident #a2dc8424):
        the worker is gone, nothing will ever pick an ACTIVE task up
        (promote_next skips non-queued ones) and it blocks a development
        slot forever. Leave the trace the vanished path did not write
        and hand the task back to the pipeline: re-queue it, excluded
        from automatic promotion for the rest of the session exactly
        like an ordinary failure (the failed-queue rescue lifts the
        exclusion when nothing error-free remains)."""
        if task.status is not TaskStatus.ACTIVE:
            return
        if task.id in self._agent_runs or task.id in self._agent_net_retry_at:
            # A newer run owns it / a retry is already scheduled.
            return
        if task.unfinished:
            # A deliberate pause (abort, compaction): not a zombie — its
            # own handlers left the traces and the resume markers.
            return
        if not self.resolved_config(self._task_role(task)).is_ready:
            # The model-not-configured pause: the early return above the
            # turn loop left its own event ("message is kept and will be
            # sent once the model is available") — re-queueing here would
            # only churn the queue while nothing can run at all.
            return
        if task.pending_input:
            detail = "ended without consuming the queued message"
        else:
            detail = "ended without completing or re-queueing the task"
        self._show_event(
            task,
            f"The agent run {detail} — re-queueing it so the pipeline keeps flowing.",
        )
        # The resume marker: a re-queued task is promotable only with
        # pending input or an unfinished attempt — without one it would
        # sit in the queue forever.
        task.unfinished = True
        task.status = TaskStatus.QUEUED
        self._agent_failed.add(task_id)
        # No live run means no retry can be in flight — drop any stale
        # network bookkeeping or the outage latch would hold the session
        # (and the retry-resume marker would silence the model line of
        # the task's next genuine start).
        self._agent_net_count.pop(task_id, None)
        self._agent_net_retry_at.pop(task_id, None)
        self._agent_retry_resumes.discard(task_id)
        self.manager.storage.save(task)
        # The transition happened outside the manager — report it to the
        # status observers (notifications) explicitly.
        self.manager.fire_status_change(task, TaskStatus.ACTIVE, "system")
        self.refresh_sidebar()
        self._advance()
