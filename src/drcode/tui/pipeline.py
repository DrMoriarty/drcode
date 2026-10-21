from __future__ import annotations

import asyncio
import json
import time
from functools import partial
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRun
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.usage import UsageLimits
from textual import work

from ..agent import (
    AgentDeps,
    TesterDeps,
    backdate_user_prompt,
    build_agent,
    build_compaction_agent,
    build_compaction_prompt,
    build_tester_agent,
    compacted_messages,
    ends_with_open_request,
    extract_previous_summary,
    first_user_prompt,
    is_summary_message,
    last_context_tokens,
    last_response_usage,
    probe_model,
    render_history_for_summary,
    resumable_history,
    run_context_tokens,
    sanitized_messages,
    serialize_messages,
    split_original_prompts,
)
from ..github import (
    issue_commit_hint,
)
from ..models import (
    PendingInput,
    Task,
    TaskStatus,
)
from ..prompts import load_compaction_prompts
from ..settings import (
    TaskFinalization,
)
from .blocks import (
    AssistantBlock,
    ReasoningBlock,
    ToolCallBlock,
)
from .state import DrCodeState
from .usage import (
    NETWORK_RETRIES,
    _is_retryable_network_error,
    _network_retry_delay,
)


class PipelineLayer(DrCodeState):

    async def _check_model(self) -> None:
        label = self.model_status
        was_error = label.has_class("error")
        if not self.config.is_ready:
            self._conn_label = "○ model not configured"
            self._last_probe_ms = None
            label.remove_class("ok", "error")
            label.add_class("warn")
            self._render_model_status()
            return
        model = self.config.model
        started = time.monotonic()
        # The cached health_check_url is probed first (cheaper, no generation);
        # the checked_url itself is discarded here — the header only needs the
        # ok/detail status.
        ok, detail, _checked_url = await probe_model(
            model,
            self.config.base_url,
            self.config.api_key,
            health_check_url=self.config.health_check_url,
        )
        # Round-trip of the probe itself, shown on the expanded header line.
        self._last_probe_ms = int((time.monotonic() - started) * 1000)
        if not ok:
            self._probe_fail_streak += 1
            self._conn_label = f"✗ {model} — {detail}"
            label.remove_class("ok", "warn")
            label.add_class("error")
            self._render_model_status()
            return
        self._probe_fail_streak = 0
        self._conn_label = f"● {model}"
        label.remove_class("warn", "error")
        label.add_class("ok")
        self._render_model_status()
        if was_error:
            self._recover_from_outage()

    # -- live output rate (header tok/s) --------------------------------------

    # Window over which the streamed output rate is averaged, and the minimum
    # span of samples before a rate is shown at all (the very first second of
    # a stream would otherwise produce a wildly spiky figure).
    _TPS_WINDOW = 6.0
    _TPS_MIN_SPAN = 1.5


    def _record_stream_tokens(self, text: str) -> None:
        """Record a streamed delta for the live tok/s estimate.

        Streaming APIs emit deltas roughly token-sized, so the character
        count divided by the conventional ~4 chars/token is a good enough
        approximation — the point is the visual "is it producing" signal,
        not billing-grade accounting.
        """
        now = time.monotonic()
        self._tps_samples.append((now, max(1, len(text) // 4)))
        self._prune_tps_samples(now)


    def _prune_tps_samples(self, now: float) -> None:
        cutoff = now - self._TPS_WINDOW
        samples = self._tps_samples
        while samples and samples[0][0] < cutoff:
            samples.popleft()


    def _tokens_per_second(self) -> float | None:
        """Output rate over the sliding window (None = nothing measurable:
        no recent stream output, or too little of it to average yet)."""
        now = time.monotonic()
        self._prune_tps_samples(now)
        samples = self._tps_samples
        if not samples:
            return None
        span = now - samples[0][0]
        if span < self._TPS_MIN_SPAN:
            return None
        return sum(n for _, n in samples) / span


    def _next_testing(self) -> Task | None:
        """Oldest task in TESTING (crash/abort mid-test) whose retry backoff
        has expired. Tasks whose tester run failed recently are skipped
        until their scheduled retry time."""
        now = time.monotonic()
        return next(
            (
                t
                for t in self.manager.all_tasks()
                if t.status is TaskStatus.TESTING
                and self._tester_retry_at.get(t.id, 0.0) <= now
            ),
            None,
        )


    def _advance(self) -> None:
        """The single pipeline gate: testing always has absolute priority.

        Every path that used to promote the queue goes through here, so a
        new development run can never start while any task is awaiting
        verification. While a tester run is in flight this is a no-op —
        the tester's own completion path advances the pipeline.
        """
        if self.tester_running:
            return
        if self.settings.task_finalization is TaskFinalization.NONE:
            # No tester at all: every task that reached TESTING (this tick or
            # left over from a run started under another setting) is completed
            # straight away, and the queue continues below.
            finalized = [
                t for t in self.manager.all_tasks() if t.status is TaskStatus.TESTING
            ]
            for t in finalized:
                self.manager.complete_without_verification(t.id)
            if finalized:
                self.refresh_sidebar()
        testing = self._next_testing()
        if testing is not None:
            # Set synchronously: two _advance() calls within one event-loop
            # tick must not start two tester workers (the exclusive group
            # would cancel the first before it does anything).
            self.tester_running = True
            self.run_tester(testing.id)
            return
        if any(t.status is TaskStatus.TESTING for t in self.manager.all_tasks()):
            # Every TESTING task is within its retry backoff window: the
            # rule "no new development while any testing is incomplete"
            # holds even then — the retry timer will call _advance() again.
            return
        promoted = self.manager.promote_next(exclude=self._agent_failed)
        if promoted is None:
            # Deadlock escape: the queue is not empty, but every task in it
            # failed earlier this session (see _promote_failed_queue).
            promoted = self._promote_failed_queue()
        if promoted:
            self.notifier.pipeline_busy()
            self.run_agent(promoted.id)
        else:
            self._notify_pipeline_settled()
        self.refresh_status()

    # Backoff schedule of the failed-queue deadlock escape (see
    # _promote_failed_queue): rescue rounds of a queue that keeps failing
    # grow from this base up to the cap, so a persistently broken model
    # cannot hot-loop the whole backlog (an instantly failing task would
    # otherwise cycle error → promote next → error forever).
    _FAILED_QUEUE_RETRY_BASE = 60.0
    _FAILED_QUEUE_RETRY_CAP = 600.0

    def _promote_failed_queue(self) -> Task | None:
        """Deadlock escape for a queue of session-failed tasks.

        A task whose run crashed (network outage, model error) is demoted
        back to the queue and excluded from promotion via _agent_failed for
        the rest of the session, so one broken task can't loop the queue.
        That exclusion is session-only (not persisted), which used to make
        the situation where EVERY queued task is failed a dead end: nothing
        can be promoted until the app is restarted (or the connection probe
        recovers and _recover_from_outage fires).

        Now, when nothing is active, nothing is being tested and every
        queued task carries the error flag, the flag is forcibly lifted
        from all of them and the next task is promoted by the ordinary
        priority rules — the queue keeps working instead of stalling. A
        notification reports how many tasks were rescued this way.

        Returns the promoted task, or None when the escape does not apply
        (an active task exists, the queue holds a non-failed task, or the
        next rescue round is still within its backoff window — a timer
        re-enters _advance() when the window expires).
        """
        if self.manager.active is not None:
            return None
        queued = self.manager.queued()
        if not queued or any(t.id not in self._agent_failed for t in queued):
            return None
        now = time.monotonic()
        if now < self._failed_queue_retry_at:
            # This rescue round must still wait out its backoff: settle for
            # now — the timer scheduled below re-calls _advance() in time.
            return None
        self._failed_queue_rounds += 1
        delay = min(
            self._FAILED_QUEUE_RETRY_BASE * 2 ** (self._failed_queue_rounds - 1),
            self._FAILED_QUEUE_RETRY_CAP,
        )
        self._failed_queue_retry_at = now + delay
        self.set_timer(delay, self._advance)
        for t in queued:
            self._agent_failed.discard(t.id)
            self._show_event(
                t,
                "Задача завершалась ошибкой и была исключена из разработки "
                "до конца сессии; в очереди не осталось задач без ошибок — "
                "исключение снято, задача снова возвращена в разработку.",
            )
        self.notifier.failed_tasks_requeued(len(queued))
        return self.manager.promote_next()

    def _notify_pipeline_settled(self) -> None:
        """Notifications: the gate settled without starting anything.

        Only meaningful when the pipeline is really idle — nothing is being
        developed or tested and no task is active. Then either the queue is
        empty (everything processed) or tasks remain that cannot be taken
        into development right now (all held / awaiting clarification /
        failed this session / queued without any input). Both states are
        reported once per transition — the notifier deduplicates; the very
        first evaluation after app startup only primes the state (a cold
        start on an empty queue must not announce "queue is empty").
        """
        if self.tester_running or self.agent_running:
            return
        if self.manager.active is not None:
            return
        if any(t.status is TaskStatus.TESTING for t in self.manager.all_tasks()):
            return
        remaining = [
            t
            for t in self.manager.all_tasks()
            if t.status is not TaskStatus.COMPLETED
        ]
        if not remaining:
            self.notifier.pipeline_idle(
                blocked=False,
                detail="",
                prime=not self._notifications_primed,
            )
            self._notifications_primed = True
            return
        # Tasks remain, but promote_next() found nothing to activate:
        # explain why none of them can be taken into development.
        reasons: list[str] = []
        held = sum(1 for t in remaining if t.status is TaskStatus.HOLD)
        if held:
            reasons.append(f"на удержании (hold): {held}")
        awaiting = sum(
            1 for t in remaining if t.status is TaskStatus.AWAITING_CLARIFICATION
        )
        if awaiting:
            reasons.append(f"ждут уточняющего вопроса: {awaiting}")
        failed = sum(1 for t in remaining if t.id in self._agent_failed)
        if failed:
            reasons.append(f"завершились ошибкой в этой сессии: {failed}")
        idle = sum(
            1
            for t in remaining
            if t.status is TaskStatus.QUEUED
            and t.id not in self._agent_failed
            and not t.pending_input
            and not t.unfinished
        )
        if idle:
            reasons.append(f"в очереди без текста задачи: {idle}")
        self.notifier.pipeline_idle(
            blocked=True,
            detail=("Причины: " + ", ".join(reasons) + ".") if reasons else "",
            prime=not self._notifications_primed,
        )
        self._notifications_primed = True


    def _persist_task_messages(
        self, task: Task, run: AgentRun[AgentDeps, str] | None
    ) -> None:
        """Snapshot the transcript produced so far into the task.

        One agent turn can run for minutes (many model requests and tool
        calls). Saving only on turn completion leaves the stored history
        stale while the agent works: switching to the task then shows
        nothing new (for a fresh task — nothing at all), and an interrupted
        turn (abort, a follow-up comment cancelling the exclusive worker)
        lost everything it had done. Persisting after every completed node
        keeps both the on-disk history and `view_task` current.

        The snapshot may end with an incomplete exchange (tool calls
        without results), which the model API would reject on resume —
        `resumable_history()` trims that tail when the transcript is next
        loaded as history.

        `run` is None when the failure happened before the run context was
        entered (e.g. `agent.iter()` or `resumable_history()` raised); there
        is nothing new to persist then — the pending input is re-queued by
        the caller and retried on the next activation.
        """
        if run is None:
            return
        try:
            messages = run.all_messages()
        except Exception:  # noqa: BLE001 - state may be mid-teardown after cancel
            return
        task.messages = serialize_messages(messages)
        self.manager.storage.save(task)


    def _persist_tester_messages(
        self, task: Task, run: AgentRun[TesterDeps, str] | None
    ) -> None:
        """Snapshot the tester transcript into the task's last testing round.

        Mirrors _persist_task_messages for the tester run: without it the
        tester's reasoning, tool calls and answer would exist only in the
        chat widgets and vanish from the history on the next view_task().
        The round entry is created up front by run_tester; `run` is None
        when the failure happened before the run context was entered.
        """
        if run is None:
            return
        try:
            messages = run.all_messages()
        except Exception:  # noqa: BLE001 - state may be mid-teardown after cancel
            return
        if not task.tester_messages:
            task.tester_messages.append([])
        task.tester_messages[-1] = serialize_messages(messages)
        self.manager.storage.save(task)


    def _fill_over_threshold(
        self,
        live_tokens: int | None,
        messages: list[dict[str, Any]],
    ) -> int | None:
        """Context fill (tokens) when the compaction threshold is reached
        (percent of max_context_tokens), None otherwise. Prefers the live
        reading of the running turn over the persisted transcript, whose
        usage lags behind by the not-yet-persisted requests."""
        max_tokens = self.config.max_context_tokens
        if max_tokens <= 0:
            return None
        threshold = min(max(self.config.compaction_threshold, 1), 100)
        tokens = live_tokens
        if tokens is None:
            tokens = last_context_tokens(messages)
        if not tokens or tokens * 100 < threshold * max_tokens:
            return None
        return tokens


    def _compaction_due(self, task: Task) -> int | None:
        """Compaction threshold check for the main agent transcript."""
        return self._fill_over_threshold(
            self._live_ctx_tokens.get(task.id), task.messages
        )


    async def _maybe_compact(self, task: Task) -> None:
        """Context compaction gate, called between agent turns (a turn
        boundary is the only safe point: the transcript never ends with
        unanswered tool calls there).

        When the context fill reaches the configured threshold (percent of
        max_context_tokens), the main agent's work is paused, a tool-less
        summarizer agent rewrites the transcript into a structured summary
        (anchored to the previous one, if any), and the transcript is
        replaced by the original user prompt(s) plus that summary. A
        compaction failure is never fatal: it is logged into the service
        events and the run continues with the uncompressed history.
        """
        tokens = self._compaction_due(task)
        if tokens is None:
            return
        pct = round(tokens * 100 / self.config.max_context_tokens)
        _, rest = split_original_prompts(task.messages)
        if not any(not is_summary_message(m) for m in rest):
            # Nothing to summarize (only the original prompt(s), possibly
            # plus a previous summary): compacting again would be a no-op.
            return
        self._show_event(
            task,
            f"Context fill {pct}% — pausing the agent and compacting history…",
        )
        try:
            agent = build_compaction_agent(self.config)
            prompt = build_compaction_prompt(
                extract_previous_summary(task.messages),
                render_history_for_summary(task.messages),
                load_compaction_prompts()[1],
            )
            result = await agent.run(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - compaction must not kill the run
            self._compaction_failed.add(task.id)
            self._show_event(task, f"compaction failed: {type(e).__name__}: {e}")
            return
        self._compaction_failed.discard(task.id)
        # The summarizer's model calls count toward the session usage too.
        self._record_session_usage(result.usage)
        task.messages = compacted_messages(task.messages, result.output)
        task.touch()
        self.manager.storage.save(task)
        # The live token reading is stale now (it belongs to the old
        # transcript); drop it so the counter falls back to the compacted
        # transcript until the next model request reports real usage.
        self._live_ctx_tokens.pop(task.id, None)
        self._show_event(
            task,
            "Context compacted: history summarized, original task "
            "prompt(s) kept verbatim.",
        )
        if self.viewed_id == task.id:
            self.view_task(task.id)


    async def _maybe_compact_tester_round(
        self, task: Task, live_tokens: int | None
    ) -> bool:
        """Compaction gate for the current testing round, called after
        run_tester paused the round at a tool boundary.

        Mirrors _maybe_compact for the tester, with one crucial difference:
        the summarizer rewrites the ROUND transcript
        (`task.tester_messages[-1]`) in place — `task.messages` (the
        developer transcript) is never touched, the two histories must not
        mix (tester rounds are persisted separately precisely so they never
        leak into the coding agent's model history).

        Returns True when the round was compacted: run_tester must resume
        it as a continuation turn from the compacted transcript (and drop
        the stale live token reading). Returns False when compaction was
        not possible — the threshold is not reached, there is nothing to
        summarize (a no-op), or the summarizer failed — in which case the
        round is resumed uncompressed and the caller adds the task to
        `_compaction_failed`, so the mid-round gate stays off (a
        persistently broken summarizer must not loop pause→resume
        forever); turn-boundary compaction of the main agent still retries
        on later turns.
        """
        round_messages = task.tester_messages[-1] if task.tester_messages else []
        tokens = self._fill_over_threshold(live_tokens, round_messages)
        if tokens is None:
            return False
        pct = round(tokens * 100 / self.config.max_context_tokens)
        _, rest = split_original_prompts(round_messages)
        if not any(not is_summary_message(m) for m in rest):
            # Nothing to summarize (only the original prompt(s), possibly
            # plus a previous summary): compacting again would be a no-op —
            # resuming the round as-is would hit the gate again.
            self._compaction_failed.add(task.id)
            return False
        self._show_event(
            task,
            f"Context fill {pct}% — compacting the testing round…",
        )
        try:
            agent = build_compaction_agent(self.config)
            prompt = build_compaction_prompt(
                extract_previous_summary(round_messages),
                render_history_for_summary(round_messages),
                load_compaction_prompts()[1],
            )
            result = await agent.run(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - compaction must not kill the round
            self._compaction_failed.add(task.id)
            self._show_event(task, f"compaction failed: {type(e).__name__}: {e}")
            return False
        self._compaction_failed.discard(task.id)
        # The summarizer's model calls count toward the session usage too.
        self._record_session_usage(result.usage)
        task.tester_messages[-1] = compacted_messages(round_messages, result.output)
        task.touch()
        self.manager.storage.save(task)
        self._show_event(
            task,
            "Context compacted: testing round summarized, original "
            "prompt kept verbatim.",
        )
        if self.viewed_id == task.id:
            self.view_task(task.id)
        return True


    @work(exclusive=True, group="agent")
    async def run_agent(self, task_id: str) -> None:
        self._agent_generation += 1
        generation = self._agent_generation
        # Key for the tool-spinner registry: superseded runs clean up only
        # their own entries.
        run_key: tuple[str, int] = ("agent", generation)
        task = self.manager.get(task_id)
        # Bound only once `async with agent.iter(...)` is entered; the
        # except handlers below must not touch a run that never started.
        # Initialized before the try: build_agent() itself may raise.
        run: AgentRun[AgentDeps, str] | None = None
        cancelled = False
        try:
            if not self.config.is_ready:
                self._show_event(
                    task,
                    "Model is not configured. Restart drcode to open the setup "
                    "dialog, or set model/base_url in .drcode/drcode.json "
                    "(project) / "
                    "~/.config/drcode/drcode.json / DRCODE_* env vars; the "
                    "message is kept and will be sent once the model is "
                    "available.",
                )
                return
            self.agent_running = True
            self._agent_failed.discard(task_id)
            self._agent_net_retry_at.pop(task_id, None)
            self._current_tool = None
            self._tps_samples.clear()
            self._touch_activity()
            self.refresh_status()
            agent = build_agent(self.config, kind=task.kind, root=self.project_root)
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
                # The consumed pending entries (empty for a continuation
                # turn): kept around so a transient failure can re-queue
                # them with their original "typed at" stamps intact.
                consumed: list[PendingInput] = []
                if task.pending_input:
                    # Batch consumption: everything queued before this turn
                    # (the original brief plus comments typed while the task
                    # sat in the queue or while another turn ran) is sent as
                    # a single prompt — the agent must see the whole picture
                    # at once and decide from all of it.
                    consumed = list(task.pending_input)
                    task.pending_input.clear()
                    text = "\n\n".join(entry.text for entry in consumed)
                    if self.viewed_id == task_id:
                        for entry in consumed:
                            if entry.text in self._pending_rendered:
                                # Already shown as a queued block; don't repeat it.
                                self._pending_rendered.remove(entry.text)
                            else:
                                self._write_history_line("user", entry.text)
                elif task.unfinished:
                    if not ends_with_open_request(task.messages):
                        # The interrupted attempt actually finished its
                        # exchange — nothing to resume; clear the flag so
                        # the task can complete normally below.
                        task.unfinished = False
                        self.manager.storage.save(task)
                        break
                    task.unfinished = False
                    self.manager.storage.save(task)
                    text = None
                else:
                    break
                try:
                    # Context compaction gate: before the history for the
                    # next turn is loaded, the fill is checked against the
                    # threshold; when exceeded, the transcript is summarized
                    # in place (history is computed AFTER compaction).
                    # Check compaction early to avoid one-turn lag in detection
                    compaction_due = self._compaction_due(task)
                    if compaction_due is not None:
                        # Compaction is due, trigger it now to avoid delay
                        await self._maybe_compact(task)
                    else:
                        # Normal execution path
                        await self._maybe_compact(task)
                    history = resumable_history(task.messages)
                    async with agent.iter(
                        text,
                        message_history=history,
                        deps=AgentDeps(
                            root=self.project_root,
                            manager=self.manager,
                            task_id=task_id,
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
                        view_dirty = False
                        synced_epoch = self._chat_epoch
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                reasoning_block: ReasoningBlock | None = None
                                assistant_block: AssistantBlock | None = None
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        self._touch_activity()
                                        if isinstance(event, PartStartEvent):
                                            # The first chunk of a new text/thinking
                                            # part arrives as PartStartEvent carrying
                                            # the content itself (only subsequent
                                            # chunks come as deltas); dropping it
                                            # loses the beginning of the message
                                            # from the live view.
                                            part = event.part
                                            if isinstance(part, ThinkingPart):
                                                delta: (
                                                    ThinkingPartDelta | TextPartDelta
                                                ) = ThinkingPartDelta(
                                                    content_delta=part.content
                                                )
                                            elif isinstance(part, TextPart):
                                                delta = TextPartDelta(
                                                    content_delta=part.content
                                                )
                                            else:
                                                continue
                                        elif isinstance(event, PartDeltaEvent):
                                            if isinstance(
                                                event.delta,
                                                ThinkingPartDelta | TextPartDelta,
                                            ):
                                                delta = event.delta
                                            else:
                                                continue
                                        else:
                                            continue
                                        if delta.content_delta:
                                            self._record_stream_tokens(
                                                delta.content_delta
                                            )
                                        if (
                                            isinstance(delta, ThinkingPartDelta)
                                            and delta.content_delta
                                        ):
                                            if (
                                                reasoning_block is None
                                                or reasoning_block.stream_epoch
                                                != self._chat_epoch
                                            ):
                                                old, reasoning_block = (
                                                    reasoning_block,
                                                    ReasoningBlock(
                                                        reasoning_block.at
                                                        if reasoning_block
                                                        else None
                                                    ),
                                                )
                                                reasoning_block.stream_epoch = (
                                                    self._chat_epoch
                                                )
                                                if old is not None:
                                                    # A mid-node view_task()
                                                    # re-render orphaned the
                                                    # old block: carry the
                                                    # already-streamed text
                                                    # over, or the rest of
                                                    # this response would
                                                    # never become visible.
                                                    reasoning_block.set_content(
                                                        old._content
                                                    )
                                                reasoning_block.begin()
                                                if self.viewed_id == task_id:
                                                    self._write_chat(reasoning_block)
                                                else:
                                                    view_dirty = True
                                                reasoning_block.append(
                                                    delta.content_delta
                                                )
                                            else:
                                                # Streaming grows the block
                                                # without a new mount: keep
                                                # following only if the tail
                                                # was visible before the
                                                # delta arrived. Deltas are
                                                # appended even when the task
                                                # is off-screen: they must
                                                # survive a later re-render.
                                                reasoning_block.thinking(True)
                                                if self.viewed_id != task_id:
                                                    view_dirty = True
                                                follow = (
                                                    self._chat_at_bottom()
                                                    if self.viewed_id == task_id
                                                    else False
                                                )
                                                reasoning_block.append(
                                                    delta.content_delta
                                                )
                                                if follow:
                                                    self._stream_follow()
                                        elif (
                                            isinstance(delta, TextPartDelta)
                                            and delta.content_delta
                                        ):
                                            # Stream the answer into the chat
                                            # log as it is produced: the block
                                            # grows in place, so the text shows
                                            # up between the reasoning and
                                            # tool-call blocks (no separate
                                            # preview area under the log).
                                            # Thinking is over for now — stop
                                            # the reasoning block's spinner.
                                            if reasoning_block is not None:
                                                reasoning_block.thinking(False)
                                            if (
                                                assistant_block is None
                                                or assistant_block.stream_epoch
                                                != self._chat_epoch
                                            ):
                                                old, assistant_block = (
                                                    assistant_block,
                                                    AssistantBlock(),
                                                )
                                                assistant_block.stream_epoch = (
                                                    self._chat_epoch
                                                )
                                                if old is not None:
                                                    # See the reasoning branch:
                                                    # re-mount after a mid-node
                                                    # view_task() wipe, keeping
                                                    # the streamed so far.
                                                    assistant_block.set_content(
                                                        old._content
                                                    )
                                                if self.viewed_id == task_id:
                                                    self._write_chat(assistant_block)
                                                else:
                                                    view_dirty = True
                                                assistant_block.append(
                                                    delta.content_delta
                                                )
                                            else:
                                                if self.viewed_id != task_id:
                                                    view_dirty = True
                                                follow = (
                                                    self._chat_at_bottom()
                                                    if self.viewed_id == task_id
                                                    else False
                                                )
                                                assistant_block.append(
                                                    delta.content_delta
                                                )
                                                if follow:
                                                    self._stream_follow()
                                if reasoning_block is not None:
                                    reasoning_block.finish()
                                if assistant_block is not None:
                                    assistant_block.finish()
                                # The consumed prompt is now in the transcript
                                # (the model-request node just appended it).
                                # Backdate it to the moment the user typed it,
                                # so the chat thread stays chronological even
                                # when the prompt sat in pending_input for a
                                # while before this run picked it up.
                                if consumed and text is not None:
                                    backdate_user_prompt(
                                        run.all_messages(),
                                        text,
                                        consumed[0].at,
                                    )
                                # The stream is closed: this request's usage
                                # is final. Refresh the context counter in
                                # the task status bar for the running task.
                                ctx_tokens = run_context_tokens(run)
                                if ctx_tokens is not None:
                                    self._live_ctx_tokens[task_id] = ctx_tokens
                                    if self.viewed_id == task_id:
                                        self.refresh_task_status()
                                # Exact per-request usage of the completed
                                # model call, into the session totals.
                                self._record_session_usage(last_response_usage(run))
                                # A model call went through: the network-retry
                                # budget for this task starts over, so the
                                # NEXT network error counts from one again —
                                # even mid-run, where a single turn makes
                                # many model requests.
                                self._agent_net_count.pop(task_id, None)
                                self._agent_net_retry_at.pop(task_id, None)
                                # The model works again: the failed-queue
                                # rescue backoff (see _promote_failed_queue)
                                # starts over, so a later single failure is
                                # rescued immediately, not after a stale
                                # backoff window.
                                self._failed_queue_rounds = 0
                                self._failed_queue_retry_at = 0.0
                            elif Agent.is_call_tools_node(node):
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        if isinstance(event, FunctionToolCallEvent):
                                            self._touch_activity(event.part.tool_name)
                                        elif isinstance(event, FunctionToolResultEvent):
                                            # Result stops the spinner even
                                            # when the task is off-screen.
                                            self._tool_call_finished(
                                                event.part.tool_call_id
                                            )
                                            self._touch_activity(done=True)
                                        if self.viewed_id != task_id:
                                            # The tool-call block is skipped
                                            # (it will come from the next
                                            # re-render): the open view is
                                            # missing it.
                                            view_dirty = True
                                            continue
                                        if isinstance(event, FunctionToolCallEvent):
                                            args = event.part.args
                                            args_str = (
                                                args
                                                if isinstance(args, str)
                                                else json.dumps(
                                                    args or {}, ensure_ascii=False
                                                )
                                            )
                                            call_text = (
                                                f"{event.part.tool_name}({args_str})"
                                            )
                                            block = ToolCallBlock(call_text)
                                            self._tool_call_started(
                                                run_key,
                                                task_id,
                                                event.part.tool_call_id,
                                                call_text,
                                                block,
                                            )
                                            block.start_spinner()
                                            self._write_chat(block)
                                # Mid-run compaction gate: a single turn can
                                # run for many model requests, and the
                                # turn-boundary gate above would only fire
                                # when the whole run ends — the fill can grow
                                # far past the threshold inside one turn.
                                # Stop the run here: the tool results of the
                                # round just executed are the last thing the
                                # model saw, and a break leaves them
                                # uncommitted in run.all_messages() (the
                                # transcript ends with unanswered tool calls
                                # — sanitized away below, the redo of that
                                # one round is the price of the pause). The
                                # attempt is marked unfinished, so the
                                # while-loop's next iteration compact (the
                                # gate above) and resume via a continuation
                                # turn. Skipped when compaction keeps
                                # failing, otherwise a persistently broken
                                # summarizer would loop
                                # break→continuation→break.
                                tokens = self._compaction_due(task)
                                if tokens is not None and (
                                    task.id not in self._compaction_failed
                                ):
                                    pct = round(
                                        tokens * 100 / self.config.max_context_tokens
                                    )
                                    self._show_event(
                                        task,
                                        f"Context fill {pct}% — pausing the "
                                        "run to compact history…",
                                    )
                                    self._persist_task_messages(task, run)
                                    task.messages = sanitized_messages(task.messages)
                                    task.unfinished = True
                                    self.manager.storage.save(task)
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
                        view_dirty or self._chat_epoch != synced_epoch
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
                        # previous node boundary.
                        self.view_task(task_id)
                        view_dirty = False
                        synced_epoch = self._chat_epoch
                    # The turn went through: the network-retry budget for
                    # this task starts over (and the failed-queue rescue
                    # backoff with it — see the model-request branch above).
                    self._agent_net_count.pop(task_id, None)
                    self._agent_net_retry_at.pop(task_id, None)
                    self._failed_queue_rounds = 0
                    self._failed_queue_retry_at = 0.0
                except asyncio.CancelledError:
                    # Interrupted (abort, nudge, or a newer run taking over).
                    # The consumed prompt is NOT re-queued — it stays in the
                    # transcript at its chronological position;
                    # task.unfinished marks the possibly-undone attempt and
                    # the next activation resumes it without re-sending text.
                    cancelled = True
                    if generation == self._agent_generation and (
                        task.status is TaskStatus.ACTIVE
                    ):
                        # Persist whatever the interrupted turn managed to
                        # do, then sanitize: drop the API-invalid tail (tool
                        # calls without results, the empty interrupted
                        # request) so later messages can append in any order.
                        self._persist_task_messages(task, run)
                        if run is None and consumed:
                            # Cancelled before the run context was even
                            # entered (e.g. during the compaction gate): the
                            # prompts were popped from pending_input but
                            # never reached the transcript — put them back,
                            # or the nudge/abort would silently eat them.
                            task.pending_input[:0] = consumed
                            if self.viewed_id == task_id:
                                for entry in consumed:
                                    if entry.text not in self._pending_rendered:
                                        self._pending_rendered.append(entry.text)
                        task.messages = sanitized_messages(task.messages)
                        task.unfinished = True
                        self.manager.storage.save(task)
                    # Drop our streaming preview; a newer run (if any) owns
                    # the label now and must not be touched.
                    self._drop_stream_preview(generation)
                    raise
                except Exception as e:  # noqa: BLE001 - show agent errors in the chat
                    # A transient network error (transport failure like a
                    # connection refused/timeout, or HTTP 429 rate limiting)
                    # is transient: instead of requeueing the task right
                    # away, retry the run in place after a backoff delay
                    # (5s doubling up to 60s), at most NETWORK_RETRIES times.
                    # Only when the budget is exhausted does the ordinary
                    # failure path below apply.
                    if (
                        _is_retryable_network_error(e)
                        and self._agent_net_count.get(task_id, 0) < NETWORK_RETRIES
                    ):
                        attempt = self._agent_net_count.get(task_id, 0) + 1
                        self._agent_net_count[task_id] = attempt
                        self._net_retries_total += 1
                        delay = _network_retry_delay(attempt)
                        self._show_event(
                            task,
                            f"Network error ({type(e).__name__}): retry {attempt}/"
                            f"{NETWORK_RETRIES} in {int(delay)} s",
                        )
                        if generation == self._agent_generation:
                            self._drop_stream_preview(generation)
                        self._persist_task_messages(task, run)
                        task.messages = sanitized_messages(task.messages)
                        if run is None and consumed:
                            # The request never started (pydantic-ai appends
                            # the user message before calling the model, and
                            # with `run` None nothing was persisted): put the
                            # consumed prompts back at the head of the pending
                            # queue so the retry re-sends them verbatim (with
                            # their original "typed at" stamps).
                            task.pending_input[:0] = consumed
                            # They were already rendered as user blocks when
                            # they were popped: the retry must not draw them
                            # twice.
                            if self.viewed_id == task_id:
                                for entry in consumed:
                                    if entry.text not in self._pending_rendered:
                                        self._pending_rendered.append(entry.text)
                        if ends_with_open_request(task.messages):
                            # The request is recorded in the transcript: the
                            # retry resumes the attempt without new text.
                            task.unfinished = True
                        self._agent_net_retry_at[task_id] = time.monotonic() + delay
                        # The task stays ACTIVE — no requeue, no pipeline
                        # advance; the delayed re-run below picks it up.
                        self.manager.storage.save(task)
                        self.set_timer(delay, partial(self.run_agent, task_id))
                        return
                    # Persist the failure even for a superseded run: it
                    # happened, and the human must be able to see it.
                    self._show_event(task, f"{type(e).__name__}: {e}")
                    if generation == self._agent_generation:
                        # A mid-stream failure leaves a half-streamed orange
                        # preview above the input: clear it, otherwise the
                        # stale fragments hang in the dialog (and would be
                        # shown for whatever the user does next).
                        self._drop_stream_preview(generation)
                    self._persist_task_messages(task, run)
                    task.messages = sanitized_messages(task.messages)
                    # Not lost: unfinished marks the attempt; promotion
                    # resumes it (this session it is excluded via
                    # _agent_failed so a dead gateway can't loop the queue).
                    task.unfinished = True
                    # Demote honestly: an ACTIVE task with no runner is a
                    # state nothing ever picks up again (promote_next skips
                    # it) — the task goes back to the queue instead, and the
                    # queue continues with the next task.
                    task.status = TaskStatus.QUEUED
                    self._agent_failed.add(task_id)
                    self.manager.storage.save(task)
                    # The transition happened outside the manager — report it
                    # to the status observers (notifications) explicitly.
                    self.manager.fire_status_change(task, TaskStatus.ACTIVE, "system")
                    break
        finally:
            # Stop any spinner whose result event was lost to an abort or a
            # run error (normally all of them are already stopped).
            self._stop_run_tools(run_key)
            if generation == self._agent_generation:
                self.agent_running = False
                self.refresh_status()
            # A cancelled run (user abort, or a newer run taking over) must
            # not restart the pipeline: the abort stops everything until the
            # user acts, and the newer run advances on its own completion.
            if not cancelled and generation == self._agent_generation:
                if task.status is TaskStatus.AWAITING_CLARIFICATION:
                    # The user may have answered while the run was still
                    # finishing (the answer sits in pending_input and send()'s
                    # clarification transition never fired): don't wait forever
                    # for an answer that has already arrived.
                    self.manager.requeue_answered_clarification(task.id)
                    # The agent paused the task on a clarifying question: show it
                    # and let the pipeline continue while the user answers.
                    # Skip when the end-of-turn view_task() re-render above
                    # already drew it — otherwise the question shows twice
                    # while the dialog stays open (a switch away+back renders
                    # it once, via view_task alone).
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
                    # The agent answered and nothing is pending or left to
                    # resume: the task is done automatically (as if the user
                    # pressed done). An unfinished (interrupted) task stays
                    # ACTIVE paused instead — the poller surfaces it as
                    # "active, nothing running" until the user acts or the
                    # app restarts (promotion then resumes the attempt).
                    self.manager.mark_done(task.id)
                    self.refresh_sidebar()
                    if self.viewed_id == task_id:
                        self.view_task(task.id)
                    # Testing (this task or an older one in the backlog)
                    # always goes before any new development.
                    self._advance()
                elif task.status is TaskStatus.QUEUED:
                    # The run failed and demoted the task back to the queue:
                    # keep the pipeline going with the next task (failed ones
                    # are excluded, so a dead gateway can't loop the queue).
                    self._advance()

    # -- tester run ------------------------------------------------------------------


    def _schedule_tester_retry(self, task_id: str) -> None:
        """Backoff before the next tester attempt for a failed task.

        TESTING tasks block the whole pipeline, so a failed tester run is
        retried (with capped exponential backoff) instead of being dropped
        from the backlog until restart."""
        count = self._tester_fail_count.get(task_id, 1)
        delay = min(5 * 2 ** (count - 1), 300)
        self._tester_retry_at[task_id] = time.monotonic() + delay
        self.set_timer(delay, self._advance)


    @work(exclusive=True, group="tester")
    async def run_tester(self, task_id: str) -> None:
        """After any task completes, verify the project and label the task."""
        self._tester_task_id = task_id
        self._tester_generation += 1
        run_key: tuple[str, int] = ("tester", self._tester_generation)
        task = self.manager.get(task_id)
        # A network-retry continues the current testing round in place
        # instead of starting a new one (the flag is set by the network
        # handler below).
        # Ignored when the task left TESTING in the meantime: the round
        # would test stale state.
        resume_round = task_id in self._tester_resume_round and (
            task.status is TaskStatus.TESTING
        )
        self._tester_resume_round.discard(task_id)
        # Initialized before the try: build_tester_agent() itself may raise,
        # and the handlers below reference `run`.
        run: AgentRun[TesterDeps, str] | None = None
        cancelled = False
        try:
            if not self.config.is_ready:
                self._tester_fail_count[task_id] = (
                    self._tester_fail_count.get(task_id, 0) + 1
                )
                self._schedule_tester_retry(task_id)
                self._show_event(
                    task,
                    "Tester skipped: model is not configured. Set DRCODE_MODEL, "
                    "DRCODE_BASE_URL and DRCODE_API_KEY (env vars or .env).",
                )
                return
            # Idempotent: _advance() sets this before starting the worker so
            # two _advance() calls in one tick cannot double-start the tester.
            self.tester_running = True
            self._current_tool = None
            self._tps_samples.clear()
            self._touch_activity()
            self.refresh_status()
            # New testing round: created up front, filled by
            # _persist_tester_messages as the run progresses. An empty round
            # (crash before any snapshot) is skipped when rendering.
            if not resume_round:
                task.tester_messages.append([])
                if self.viewed_id == task_id:
                    self._write_history_line("testing-sep", "")
            tester = build_tester_agent(
                self.config,
                allow_commit=(
                    self.settings.task_finalization is TaskFinalization.TESTING_COMMIT
                ),
            )
            deps = TesterDeps(
                root=self.project_root, manager=self.manager, task_id=task_id
            )
            try:
                prompt = f"Verify completed task #{task.id}: {task.title}"
                # The tester needs the original assignment to paraphrase it
                # into a commit message (see the git_commit step in its prompt).
                original = first_user_prompt(task.messages)
                if original and original != task.title:
                    prompt += f"\n\nOriginal task:\n{original}"
                # A GitHub-imported task must link its commit back to the
                # originating issue ("Fixes #N" keyword in the message).
                if task.github_issue is not None:
                    prompt += f"\n\n{issue_commit_hint(task.github_issue)}"
                # Round continuation loop: a testing round is a single
                # `tester.iter()` run, but it can make many model requests
                # (each `read_file`/`run_bash` result grows the context).
                # When the fill crosses the compaction threshold at a tool
                # boundary, the run is stopped, the ROUND transcript is
                # compacted in place (task.messages is never touched) and
                # the round resumes as a continuation turn from the
                # compacted transcript — otherwise a pathological tool
                # result could blow the model window before the tester
                # ever calls set_verification, looping the round in
                # fruitless retries.
                # A resumed round continues from the persisted round
                # transcript (no new prompt — the while-loop below builds
                # the continuation history from it); an empty round (the
                # failed request was never recorded) re-sends the
                # prompt as usual.
                text: str | None = (
                    None
                    if resume_round
                    and task.tester_messages
                    and task.tester_messages[-1]
                    else prompt
                )
                # Live context reading of the current round; the shared
                # _live_ctx_tokens map belongs to the agent transcript and
                # must not be polluted with the tester's.
                live_tokens: int | None = None
                while True:
                    # Reset per attempt: if tester.iter() raises before the
                    # `as run` binding, the except handlers must not
                    # re-persist the PREVIOUS attempt's transcript over the
                    # (possibly compacted) round.
                    run = None
                    history: list[Any] | None = None
                    if text is None:
                        # Continuation of a paused round: resume from the
                        # stored round transcript (compacted, or — when the
                        # summarizer failed — the original one with its
                        # invalid tail trimmed).
                        history = resumable_history(task.tester_messages[-1])
                    paused = False
                    async with tester.iter(
                        text,
                        message_history=history,
                        deps=deps,
                        usage_limits=UsageLimits(
                            request_limit=self.config.request_limit
                        ),
                    ) as run:  # noqa: F811 - rebinding per attempt is intended
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                # Drain the stream (nothing to render — the
                                # answer comes from the end-of-round
                                # re-render) and take the round's freshest
                                # context reading once the request is final.
                                # Every event is a sign of life: without the
                                # touch a long (minutes) report generation
                                # reads as "stalled" — and the stall watchdog
                                # aborts a perfectly healthy run.
                                async with node.stream(run.ctx) as stream:
                                    async for _event in stream:
                                        self._touch_activity()
                                live_tokens = run_context_tokens(run)
                                # Exact per-request usage of the completed
                                # model call, into the session totals.
                                self._record_session_usage(last_response_usage(run))
                                # A model call went through: the network-retry
                                # budget for this testing round starts over,
                                # so the NEXT network error counts from one
                                # again (mirrors the agent's per-request
                                # reset).
                                self._tester_net_count.pop(task_id, None)
                            elif Agent.is_call_tools_node(node):
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        if isinstance(event, FunctionToolCallEvent):
                                            self._touch_activity(event.part.tool_name)
                                        elif isinstance(event, FunctionToolResultEvent):
                                            # Result stops the spinner even when
                                            # the task is off-screen.
                                            self._tool_call_finished(
                                                event.part.tool_call_id
                                            )
                                            self._touch_activity(done=True)
                                        if self.viewed_id == task_id and isinstance(
                                            event, FunctionToolCallEvent
                                        ):
                                            args = event.part.args
                                            args_str = (
                                                args
                                                if isinstance(args, str)
                                                else json.dumps(
                                                    args or {}, ensure_ascii=False
                                                )
                                            )
                                            call_text = (
                                                f"tester: {event.part.tool_name}"
                                                f"({args_str})"
                                            )
                                            block = ToolCallBlock(call_text)
                                            self._tool_call_started(
                                                run_key,
                                                task_id,
                                                event.part.tool_call_id,
                                                call_text,
                                                block,
                                            )
                                            block.start_spinner()
                                            self._write_chat(block)
                                # Mid-round compaction gate: a tool boundary
                                # is the only safe stopping point (same as
                                # the agent's mid-run gate). Skipped when a
                                # previous compaction attempt failed, or a
                                # persistently broken summarizer would loop
                                # pause→resume forever.
                                tokens = self._fill_over_threshold(
                                    live_tokens, task.tester_messages[-1]
                                )
                                if tokens is not None and (
                                    task.id not in self._compaction_failed
                                ):
                                    pct = round(
                                        tokens * 100 / self.config.max_context_tokens
                                    )
                                    self._show_event(
                                        task,
                                        f"Context fill {pct}% — pausing the "
                                        "tester round to compact history…",
                                    )
                                    self._persist_tester_messages(task, run)
                                    paused = True
                                    break
                            # Keep the stored history current while the test
                            # runs: switching to the task mid-test shows the
                            # tester's work done so far.
                            self._persist_tester_messages(task, run)
                    if paused:
                        if await self._maybe_compact_tester_round(task, live_tokens):
                            # The round transcript was replaced by the
                            # original prompt plus the summary: the live
                            # token reading belongs to the old transcript —
                            # drop it so the counter falls back to the
                            # compacted one (mirrors _maybe_compact).
                            live_tokens = None
                        # Either way (compacted, or the summarizer failed
                        # and the gate is now disabled for this task) the
                        # round resumes as a continuation turn: no new
                        # text, history from the round transcript.
                        text = None
                        continue
                    if run.result is not None and self.viewed_id == task_id:
                        # Re-render from the persisted transcript: the streamed
                        # tool-call blocks above are replaced by the full history
                        # view (separator + reasoning + tool calls + answer).
                        self.view_task(task_id)
                    break
            except asyncio.CancelledError:
                # Aborted (or superseded): keep whatever the tester did so
                # far, then re-raise so the worker teardown stays clean.
                cancelled = True
                self._persist_tester_messages(task, run)
                if self.viewed_id == task_id:
                    # Watchdog abort (or esc) of a hung run: re-render from
                    # the persisted transcript, so the open view shows what
                    # was actually done instead of stale blocks (e.g. a tool
                    # call that never returned) until the user switches views.
                    # Sync only — no awaits inside a cancelled coroutine.
                    self.view_task(task_id)
                raise
            except Exception as e:  # noqa: BLE001 - show tester errors in the chat
                # A transient network error (transport failure like a
                # connection refused/timeout, or HTTP 429 rate limiting)
                # is transient: instead of restarting the whole testing
                # round, retry it in place (the task stays TESTING, the
                # round transcript is kept and the retry resumes it as a
                # continuation turn) after a backoff delay, at most
                # NETWORK_RETRIES times — only when the budget is exhausted
                # does the ordinary failure path below apply (mirrors
                # run_agent's network handling).
                if (
                    _is_retryable_network_error(e)
                    and self._tester_net_count.get(task_id, 0) < NETWORK_RETRIES
                ):
                    attempt = self._tester_net_count.get(task_id, 0) + 1
                    self._tester_net_count[task_id] = attempt
                    self._net_retries_total += 1
                    delay = _network_retry_delay(attempt)
                    self._show_event(
                        task,
                        f"Network error ({type(e).__name__}): tester retry "
                        f"{attempt}/{NETWORK_RETRIES} in {int(delay)} s",
                    )
                    self._persist_tester_messages(task, run)
                    self._tester_resume_round.add(task_id)
                    self._tester_retry_at[task_id] = time.monotonic() + delay
                    self.set_timer(delay, self._advance)
                    return
                self._persist_tester_messages(task, run)
                self._tester_resume_round.discard(task_id)
                self._tester_fail_count[task_id] = (
                    self._tester_fail_count.get(task_id, 0) + 1
                )
                self._schedule_tester_retry(task_id)
                self._show_event(task, f"tester failed: {type(e).__name__}: {e}")
        finally:
            # Stop any spinner whose result event was lost to an abort or a
            # tester error (normally all of them are already stopped).
            self._stop_run_tools(run_key)
            self.tester_running = False
            self._tester_task_id = None
            self.refresh_status()
            self.refresh_sidebar()
            if task.status is TaskStatus.COMPLETED:
                # Success: give the next testing round of this task a fresh
                # backoff and network-retry budget.
                self._tester_fail_count.pop(task_id, None)
                self._tester_net_count.pop(task_id, None)
                self._tester_resume_round.discard(task_id)
                self._tester_retry_at.pop(task_id, None)
            # A cancelled run (user abort, or a newer tester taking over)
            # must not restart the pipeline.
            if not cancelled:
                self._advance()

    # -- actions ---------------------------------------------------------------------

    # -- new task mode ---------------------------------------------------------------

