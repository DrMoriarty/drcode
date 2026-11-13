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
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRun
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)
from pydantic_ai.usage import UsageLimits

from ..agent import (
    AgentDeps,
    SubagentGate,
    backdate_user_prompt,
    build_agent,
    build_title_agent,
    build_title_prompt,
    clean_generated_title,
    ends_with_open_request,
    is_pending_retry,
    last_response_usage,
    resumable_history,
    run_context_tokens,
    sanitized_messages,
    serialize_messages,
    subagent_limit_for,
)
from ..core.usage import (
    NETWORK_RETRIES,
    _in_outage,
    _is_context_overflow_error,
    _is_deterministic_request_rejection,
    _is_retryable_network_error,
    _network_retry_delay,
)
from ..models import PendingInput, Task, TaskKind, TaskStatus
from ..providers import AgentRole, resolve_available
from ..tools import scratch_dir
from ..web_fetch import WebFetchGate, web_fetch_policy_for
from .pipeline_stream import NodeStreamView
from .state import CoreState


class PipelineAgentRunLayer(CoreState):
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
        self.host.host_spawn_worker(
            partial(self._run_agent_worker, task_id),
            name=f"agent run {task_id}",
            group=f"agent-{task_id}",
            exclusive=True,
        )

    async def _maybe_generate_title(self, task: Task, role: AgentRole) -> None:
        """Generate the task's title from its description with the model
        serving the task (project setting ``generate_task_titles``).

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
        try:
            agent = build_title_agent(
                self.resolved_config(role),
                # Provider slot-wait visibility (see _wait_announcer).
                announce_wait=self._wait_announcer(task.id, role),
            )
            result = await agent.run(build_title_prompt(description))
            # The title call counts toward the session usage like every
            # other model call (under the task's serving role).
            self._record_session_usage(
                result.usage, role, self.resolved_model_key(role)
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
                text, consumed, workflow_instructions = turn
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
                        text,
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
                            # delegate_research subagents run on THIS
                            # role's model: their usage folds into the
                            # per-day store under the parent's key (see
                            # _record_session_usage).
                            record_usage=(
                                lambda u, _role=role: self._record_session_usage(
                                    u, _role, self.resolved_model_key(_role)
                                )
                            ),
                            # The research agent's web_fetch budget/policy
                            # for THIS run (None = the tool is not
                            # registered, see _prepare_agent_run).
                            web_fetch_gate=web_fetch_gate,
                            # Provider slot-wait visibility for the nested
                            # delegate_research runs (see _wait_announcer).
                            wait_announcer=self._wait_announcer(task_id, role),
                        ),
                        usage_limits=UsageLimits(
                            request_limit=self.config.request_limit
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
                                    task, task_id, run, consumed, text, role
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
                        task, task_id, run, consumed, text, generation
                    )
                    raise
                except Exception as e:  # noqa: BLE001 - show agent errors in the chat
                    # A transient network error (transport failure like a
                    # connection refused/timeout, or HTTP 429 rate limiting)
                    # retries the run in place after a backoff (see the
                    # helper); anything else demotes the task to the queue.
                    if not _is_context_overflow_error(
                        e
                    ) and _is_retryable_network_error(e):
                        self._agent_turn_network_error(
                            task, task_id, run, consumed, text, generation, role, e
                        )
                        return
                    self._agent_turn_failed(
                        task, task_id, run, consumed, text, generation, e
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
    ]:
        """Register the run and assemble everything the turn loop needs:
        the (worktree) root the tools run in, the built agent, the
        on_task_start hook's instructions for the first prompt, the run's
        subagent gate (None when delegation is off) and the run's web-fetch
        gate (None unless this is a RESEARCH task with web fetch enabled)."""
        self._agent_runs[task_id] = generation
        self._agent_failed.discard(task_id)
        self._agent_dead.discard(task_id)
        self._agent_net_retry_at.pop(task_id, None)
        self._current_tool[task_id] = None
        self._tps_samples.pop(task_id, None)
        self._touch_activity(task_id)
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
        # Managed web reading (project setting web_fetch_mode): RESEARCH
        # only — web_fetch_policy_for returns None for every other role, so
        # the tool is not registered there at all. Read at activation, so a
        # settings change reaches the next run of each task.
        web_fetch_policy = web_fetch_policy_for(self.settings, role)
        agent = build_agent(
            self.resolved_config(role),
            kind=task.kind,
            root=root,
            research_git=self.settings.git_policy.research,
            subagent_limit=subagent_limit,
            # The research agent's web_fetch tool (SSRF-guarded, capped,
            # framed as untrusted content): None when the mode is off.
            research_web_fetch=web_fetch_policy,
            # The real scratch path is baked into the write tools'
            # descriptions (no literal <task-id> placeholder for the model
            # to copy verbatim into a path argument).
            task_id=task_id,
            # The eviction-hint nudge is opt-in (the project's
            # eviction_hint setting): read at activation, so a settings
            # toggle applies to the next activation of each task.
            eviction_hint=self.settings.eviction_hint,
            # The live checklist tail (agent_model.TodoTailModel): the
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
        )
        subagent_gate = SubagentGate(subagent_limit) if subagent_limit > 0 else None
        # One gate per activation: it carries the run's policy AND the
        # fetch budget, so a superseded run's gate dies with it (the same
        # reasoning as the subagent gate).
        web_fetch_gate = (
            WebFetchGate(web_fetch_policy) if web_fetch_policy is not None else None
        )
        return root, agent, workflow_instructions, subagent_gate, web_fetch_gate

    async def _consume_turn_input(
        self,
        task: Task,
        task_id: str,
        workflow_instructions: str | None,
    ) -> tuple[str | None, list[PendingInput], str | None] | None:
        """Consume the next turn's input (chronological message flow: the
        transcript is the single chronological record — the original
        prompt lives in it exactly once and is NEVER re-queued/re-sent;
        each activation first consumes every pending user message in
        arrival order — merged into ONE prompt, so a brief queued up
        before the run (original message + comments) is solved as a
        whole, not turn-by-turn — then, only if the last attempt was
        interrupted, runs one continuation turn without new text to pick
        the interrupted work back up).

        Returns (text, consumed entries, possibly-injected workflow
        instructions) or None when there is nothing to run (no queued
        input, no unfinished attempt) — the caller ends the run.

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
            # Remember the consumed batch until the transcript picks
            # it up (the first node-boundary persist): a view_task()
            # re-render in between finds neither copy and wipes the
            # prompt from the chat — the record lets
            # _render_task_history draw it anyway. Dropped when the
            # run ends or the batch is re-queued (below).
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
            return text, consumed, workflow_instructions
        if task.unfinished:
            if not ends_with_open_request(task.messages):
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

    def _agent_model_request_done(
        self,
        task: Task,
        task_id: str,
        run: AgentRun[AgentDeps, str],
        consumed: list[PendingInput],
        text: str | None,
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
        if consumed and text is not None:
            backdate_user_prompt(
                run.all_messages(),
                text,
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
        self._record_session_usage(
            last_response_usage(run), role, self.resolved_model_key(role)
        )
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
        text: str | None,
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
        if run is None or not consumed or text is None:
            return
        try:
            messages = run.all_messages()
        except Exception:  # noqa: BLE001 - state may be mid-teardown after cancel
            return
        backdate_user_prompt(messages, text, consumed[0].at)

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
        if tokens is not None and (
            task.id not in self._compaction_failed
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
        # Selective-eviction gate (mark_unneeded marks): the threshold is
        # NOT reached, but the agent-marked results have piled up to a
        # third of the window — pause the run so the batch eviction can
        # collapse them between turns (the actual eviction happens in
        # _maybe_compact at the next turn boundary, exactly like a
        # mid-run compaction pause).
        if self._eviction_due(task):
            self._show_event(
                task,
                "Marked tool results reached ≈1/3 of the context window — "
                "pausing the run to unload them…",
            )
            self._persist_task_messages(task, run)
            task.messages = sanitized_messages(task.messages)
            task.unfinished = True
            self.manager.storage.save(task)
            return True
        return False

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
        text: str | None,
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
            self._backdate_consumed_prompt(run, consumed, text)
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

    def _agent_turn_network_error(
        self,
        task: Task,
        task_id: str,
        run: AgentRun[AgentDeps, str] | None,
        consumed: list[PendingInput],
        text: str | None,
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
        self._backdate_consumed_prompt(run, consumed, text)
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
        self._agent_net_retry_at[task_id] = time.monotonic() + delay
        self.manager.storage.save(task)
        if outage:
            # Outage mode: the re-run is gated behind a cheap TCP probe of
            # the gateway host — while the host is unreachable the tick
            # only re-probes (see _outage_retry_tick) instead of firing
            # full model requests at a dead gateway.
            self._schedule_pipeline_timer(
                delay,
                partial(self._outage_retry_tick, task_id, role),
            )
        else:
            self._schedule_pipeline_timer(delay, partial(self.run_agent, task_id))

    def _agent_turn_failed(
        self,
        task: Task,
        task_id: str,
        run: AgentRun[AgentDeps, str] | None,
        consumed: list[PendingInput],
        text: str | None,
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
        self._backdate_consumed_prompt(run, consumed, text)
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
