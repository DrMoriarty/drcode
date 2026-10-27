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
    backdate_user_prompt,
    build_agent,
    ends_with_open_request,
    is_pending_retry,
    last_response_usage,
    resumable_history,
    run_context_tokens,
    sanitized_messages,
    serialize_messages,
)
from ..models import PendingInput, Task, TaskKind, TaskStatus
from ..providers import AgentRole
from ..tools import scratch_dir
from .blocks import ToolCallBlock
from .pipeline_stream import NodeStreamView
from .state import DrCodeState
from .usage import (
    NETWORK_RETRIES,
    _in_outage,
    _is_context_overflow_error,
    _is_retryable_network_error,
    _network_retry_delay,
)


class PipelineAgentRunLayer(DrCodeState):
    def run_agent(self, task_id: str) -> None:
        """(Re)start the development agent for one task, in that task's
        own exclusive worker group (`agent-<task-id>`): a newer run of
        the SAME task (a mid-flight comment, a nudge, a network-retry
        timer) supersedes the in-flight one exactly as the single global
        group always did — while every other task's run is untouched.
        This per-task independence is the base of the parallel
        development slots (see `parallel_limit`)."""
        if self._exit or not self.is_running:
            # Teardown race: a network-retry set_timer fired while the app
            # was being dismantled — starting a worker now would only have
            # its body trip over the already-removed DOM. The task stays
            # ACTIVE with pending retry state; nothing is lost (the next
            # start resumes it).
            return
        self.run_worker(
            partial(self._run_agent_worker, task_id),
            name=f"agent run {task_id}",
            group=f"agent-{task_id}",
            exclusive=True,
        )

    async def _run_agent_worker(self, task_id: str) -> None:
        generation = self._agent_generations.get(task_id, 0) + 1
        self._agent_generations[task_id] = generation
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
            root, agent, workflow_instructions = await self._prepare_agent_run(
                task, task_id, role, generation
            )
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
                            follow=True,
                            track_dirty=True,
                        )
                        synced_epoch = self._chat_epoch
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                stream.begin_node()
                                async with node.stream(run.ctx) as node_stream:
                                    async for event in node_stream:
                                        stream.feed(event)
                                stream.finish_node()
                                self._agent_model_request_done(
                                    task, task_id, run, consumed, text
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
                    self._agent_turn_cancelled(task, task_id, run, consumed, generation)
                    raise
                except Exception as e:  # noqa: BLE001 - show agent errors in the chat
                    # A transient network error (transport failure like a
                    # connection refused/timeout, or HTTP 429 rate limiting)
                    # retries the run in place after a backoff (see the
                    # helper); anything else demotes the task to the queue.
                    if _is_retryable_network_error(e):
                        self._agent_turn_network_error(
                            task, task_id, run, consumed, generation, role, e
                        )
                        return
                    self._agent_turn_failed(task, task_id, run, generation, e)
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
                self.refresh_status()
                # The run is over: the consumed prompt is either persisted in
                # the transcript or back in pending_input — the in-flight
                # record has served its purpose. (Guarded: a superseded run
                # must not drop a newer run's record.)
                self._consumed_prompts.pop(task_id, None)
            # A cancelled run (user abort, or a newer run taking over) must
            # not restart the pipeline: the abort stops everything until the
            # user acts, and the newer run advances on its own completion.
            if not cancelled and self._agent_generations.get(task_id) == generation:
                await self._agent_run_epilogue(task, task_id)

    # -- run-loop helpers -------------------------------------------------------
    # (the decomposed pieces of _run_agent_worker; the worker above keeps the
    # turn loop's control flow readable, each concern lives below)

    async def _prepare_agent_run(
        self,
        task: Task,
        task_id: str,
        role: AgentRole,
        generation: int,
    ) -> tuple[Path, Agent[AgentDeps, str], str | None]:
        """Register the run and assemble everything the turn loop needs:
        the (worktree) root the tools run in, the built agent and the
        on_task_start hook's instructions for the first prompt."""
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
            self._show_event(task, message)
        elif iso_note:
            # Isolation is on but the worktree could not be created:
            # the task falls back to the project directory — say so.
            self._show_event(task, f"Task isolation unavailable: {iso_note}")
        # Custom workflow (`.drcode/workflow.py`): the on_task_start
        # hook runs before the first activation (branch checkout etc.);
        # its instructions are prepended to the first consumed prompt.
        # With isolation on the hook runs INSIDE the worktree (see
        # _hook_env).
        workflow_instructions = await self._run_task_start_hook(task)
        agent = build_agent(
            self.resolved_config(role),
            kind=task.kind,
            root=root,
            research_git=self.settings.git_policy.research,
        )
        return root, agent, workflow_instructions

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
            task.unfinished = False
            self.manager.storage.save(task)
            return None, [], workflow_instructions
        return None

    def _agent_model_request_done(
        self,
        task: Task,
        task_id: str,
        run: AgentRun[AgentDeps, str],
        consumed: list[PendingInput],
        text: str | None,
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
            if self.viewed_id == task_id:
                self.refresh_task_status()
        # Exact per-request usage of the completed model call, into the
        # session totals.
        self._record_session_usage(last_response_usage(run))
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
                        self.call_after_refresh(self.refresh_todo_panel)
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
                    block = ToolCallBlock(call_text)
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
        generation: int,
        error: Exception,
    ) -> None:
        """A non-network turn failure: show it, persist the sanitized
        transcript, mark the attempt unfinished and demote the task back
        to the queue (excluded from promotion for the session via
        `_agent_failed`, so a dead gateway can't loop the queue). A
        deterministic context-overflow additionally marks the task dead
        (the failed-queue rescue must never lift it back)."""
        # Persist the failure even for a superseded run: it happened, and
        # the human must be able to see it.
        self._show_event(task, f"{type(error).__name__}: {error}")
        if self._agent_generations.get(task_id) == generation:
            # A mid-stream failure leaves a half-streamed orange preview
            # above the input: clear it, otherwise the stale fragments
            # hang in the dialog (and would be shown for whatever the
            # user does next).
            self._drop_stream_preview(task_id, generation)
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
