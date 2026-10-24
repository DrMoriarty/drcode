from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRun
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    UserPromptPart,
)
from pydantic_ai.usage import UsageLimits
from textual import work

from ..agent import (
    MECHANICAL_HEADROOM_TOKENS,
    AgentDeps,
    TesterDeps,
    all_user_prompts,
    backdate_user_prompt,
    build_agent,
    build_compaction_agent,
    build_compaction_prompt,
    build_tester_agent,
    compacted_messages,
    ends_with_open_request,
    extract_previous_summary,
    is_pending_retry,
    is_summary_message,
    last_context_tokens,
    last_response_usage,
    mechanical_compaction_messages,
    probe_model,
    prompt_fits_context,
    render_history_for_summary,
    resumable_history,
    run_context_tokens,
    sanitized_messages,
    serialize_messages,
    split_original_prompts,
    todo_instructions,
)
from ..config import Config
from ..github import (
    issue_commit_hint,
)
from ..models import (
    PendingInput,
    Task,
    TaskKind,
    TaskStatus,
    VerificationStatus,
    render_todos,
)
from ..prompts import load_compaction_prompts
from ..providers import AgentRole
from ..settings import (
    TaskFinalization,
    TaskIsolation,
)
from ..task_env import branch_for
from ..tools import scratch_dir
from ..workflow import MAX_REVIEW_REJECTIONS
from .blocks import (
    AssistantBlock,
    ReasoningBlock,
    ToolCallBlock,
)
from .state import DrCodeState
from .usage import (
    NETWORK_RETRIES,
    _is_context_overflow_error,
    _is_retryable_network_error,
    _network_retry_delay,
)


class PipelineLayer(DrCodeState):
    @property
    def parallel_limit(self) -> int:
        """Effective number of parallel development slots: the
        `parallel_tasks` project setting, but only under worktree
        isolation — each parallel task needs its own worktree, so
        anything else behaves as 1 (the persisted value stays, warned
        about at startup)."""
        if (
            self.settings.task_isolation is TaskIsolation.WORKTREE
            and self.task_env.active
        ):
            return max(1, self.settings.parallel_tasks)
        return 1

    @property
    def agent_running(self) -> bool:
        """Whether ANY development agent run is in flight (parallel
        slots: one flag over the per-task run registry — the tester
        keeps its own singular `tester_running`)."""
        return bool(self._agent_runs)

    def _coding_config(self) -> Config:
        """Effective model connection of the coding agent (resolved against
        the project's per-role selection and the user-level catalog)."""
        return self.resolved_config(AgentRole.CODING)

    @staticmethod
    def _task_role(task: Task) -> AgentRole:
        """The agent role serving the task's main transcript: research
        tasks run on the research role (which falls back to the coding
        model when nothing is selected for it — see
        `providers.resolve_effective`), everything else on the coding role."""
        return (
            AgentRole.RESEARCH if task.kind is TaskKind.RESEARCH else AgentRole.CODING
        )

    @staticmethod
    def _todo_instructions(task: Task) -> str | None:
        """Delegating alias for `agent.todo_instructions` — the checklist
        block injected into the model context.

        The injection itself is DYNAMIC now: `build_agent` registers
        `agent.todo_dynamic_instructions`, which pydantic-ai recomputes
        from the task's current state for EVERY model request. That fixed
        the #fbb0d9fe class of failures: run-level `instructions=` was
        frozen at `agent.iter` start, so a checklist created (or updated)
        mid-activation was invisible to the remaining requests of that
        run. The alias stays for tests and the state protocol; the
        pipeline itself no longer passes run-level instructions."""
        return todo_instructions(task)

    def _todo_completion_warning(self, task: Task) -> None:
        """Completion soft gate: when a task leaves development with
        unchecked checklist items, the human sees a service event listing
        them (never fatal — the agent may have consciously skipped items,
        or the user pressed Done by hand). The checklist state is also
        re-injected on the next activation, so a re-queued task's agent
        sees what is still open."""
        unchecked = task.unchecked_todos
        if not unchecked:
            return
        items = "; ".join(f"({i}) {t.text}" for i, t in enumerate(unchecked, 1))
        self._show_event(
            task,
            f"Task completed with {len(unchecked)} unchecked todo item(s): {items}",
        )

    async def _check_model(self) -> None:
        label = self.model_status
        was_error = label.has_class("error")
        # The header shows the CODING model's connection: the role the
        # pipeline spends most of its life in.
        config = self._coding_config()
        if not config.is_ready:
            self._conn_label = "○ model not configured"
            self._last_probe_ms = None
            label.remove_class("ok", "error")
            label.add_class("warn")
            self._render_model_status()
            return
        model = config.model
        started = time.monotonic()
        # The cached health_check_url is probed first (cheaper, no generation);
        # the checked_url itself is discarded here — the header only needs the
        # ok/detail status.
        ok, detail, _checked_url = await probe_model(
            model,
            config.base_url,
            config.api_key,
            health_check_url=config.health_check_url,
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

    def _record_stream_tokens(self, task_id: str, text: str) -> None:
        """Record a streamed delta of `task_id`'s run for the live tok/s
        estimate.

        Streaming APIs emit deltas roughly token-sized, so the character
        count divided by the conventional ~4 chars/token is a good enough
        approximation — the point is the visual "is it producing" signal,
        not billing-grade accounting. With parallel slots the samples of
        all running tasks are summed (one aggregate "is the model
        producing" figure).
        """
        now = time.monotonic()
        samples = self._tps_samples.setdefault(task_id, deque())
        samples.append((now, max(1, len(text) // 4)))
        self._prune_tps_samples(now)

    def _prune_tps_samples(self, now: float) -> None:
        cutoff = now - self._TPS_WINDOW
        for task_id in list(self._tps_samples):
            samples = self._tps_samples[task_id]
            while samples and samples[0][0] < cutoff:
                samples.popleft()
            if not samples:
                del self._tps_samples[task_id]

    def _tokens_per_second(self) -> float | None:
        """Aggregate output rate over the sliding window across every
        run's samples (None = nothing measurable: no recent stream
        output, or too little of it to average yet)."""
        now = time.monotonic()
        self._prune_tps_samples(now)
        all_samples: list[tuple[float, int]] = []
        for samples in self._tps_samples.values():
            all_samples.extend(samples)
        if not all_samples:
            return None
        span = now - min(t for t, _ in all_samples)
        if span < self._TPS_MIN_SPAN:
            return None
        return sum(n for _, n in all_samples) / span

    def _next_testing(self) -> Task | None:
        """Oldest task in TESTING (crash/abort mid-test) whose retry backoff
        has expired. Tasks whose tester run failed recently are skipped
        until their scheduled retry time. A TESTING task the tester already
        PASSED is not "testing" — it awaits the external review verdict
        (see _next_review) and is never re-tested."""
        now = time.monotonic()
        return next(
            (
                t
                for t in self.manager.all_tasks()
                if t.status is TaskStatus.TESTING
                and t.verification is not VerificationStatus.PASSED
                and self._tester_retry_at.get(t.id, 0.0) <= now
            ),
            None,
        )

    # Review retry backoff (a failed/pending external review is retried
    # like a failed tester round: capped exponential, from _REVIEW_RETRY_BASE).
    _REVIEW_RETRY_BASE = 30.0
    _REVIEW_RETRY_CAP = 600.0
    # Poll interval for the async review mode: how often the completed
    # tasks with an open PR (review_pending) ask the on_review hook again.
    _REVIEW_PENDING_POLL = 60.0

    def _schedule_pipeline_timer(
        self, delay: float, callback: Callable[[], Any]
    ) -> None:
        """Schedule a deferred pipeline tick (retry backoffs, the
        failed-queue rescue, the nudge restart) — unless the app is being
        dismantled. Every deferred tick eventually re-enters the pipeline
        (`_advance` / `run_agent`): scheduling one into a closing event
        loop either raises outright (`create_task` → `call_soon` on a
        closed loop) or fires the callback into a dead DOM — the same
        teardown race the `_advance`/`run_agent` entry guards (see
        32abee3) catch one step later, at the source here."""
        if self._exit or not self.is_running:
            return
        self.set_timer(delay, callback)

    def _next_review(self) -> Task | None:
        """A task the tester passed (TESTING + PASSED) whose external
        review may run now: the sync verdict call after a tester round, or
        the retry of a failed/unfinished one. Only meaningful when the
        project defines on_review in `.drcode/workflow.py`."""
        if self.workflow is None or not self.workflow.has("on_review"):
            return None
        now = time.monotonic()
        return next(
            (
                t
                for t in self.manager.all_tasks()
                if t.status is TaskStatus.TESTING
                and t.verification is VerificationStatus.PASSED
                and self._review_retry_at.get(t.id, 0.0) <= now
            ),
            None,
        )

    def _advance(self) -> None:
        """The single pipeline gate: a test that can start right now has
        absolute priority over development.

        Every path that used to promote the queue goes through here. The
        historical rule "a new development run can never start while any
        task is awaiting verification" holds in full for a single
        development slot (parallel_limit == 1). With parallel slots the
        block is narrowed: development is held back only while the tester
        is FREE and a TESTING task is ready to be tested right now (that
        test is dispatched first and the gate returns). A tester run
        already in flight — or a TESTING task waiting for the busy
        tester's single slot or sitting in its retry backoff — does not
        stop free development slots from being filled; the tester reaches
        the queued TESTING tasks as each of its rounds ends (bug reports
        land at the queue head, so `_dispatch_promotions` picks them up
        first). The same holds for the external review
        (`.drcode/workflow.py` on_review): a task the tester passed but
        the reviewer has not judged yet keeps the pipeline in the
        "development finished, verdict outstanding" state (it stays
        TESTING), and the review worker advances the pipeline on its own
        completion.
        """
        if self._exit or not self.is_running:
            # Teardown race: deferred set_timer callbacks (retry backoffs,
            # the failed-queue rescue, the dependency poll tick) can fire
            # while the app is being dismantled — the DOM is already gone,
            # so anything below (chat_log access via refresh paths) would
            # raise NoMatches. Skip the tick (same pattern as _poll_activity).
            return
        if not self.tester_running:
            # Tester-side duties — any of them firing returns: while a
            # test can start right now, no development is promoted.
            if self.settings.task_finalization is TaskFinalization.NONE:
                # No tester at all: every task that reached TESTING (this
                # tick or left over from a run started under another
                # setting) is completed straight away, and the queue
                # continues below.
                finalized = [
                    t
                    for t in self.manager.all_tasks()
                    if t.status is TaskStatus.TESTING
                ]
                for t in finalized:
                    self.manager.complete_without_verification(t.id)
                if finalized:
                    self.refresh_sidebar()
            if self._review_running:
                # A review worker is in flight; its completion path advances
                # the pipeline (a second dispatch here would cancel it).
                return
            review = self._next_review()
            if review is not None:
                # Set synchronously: two _advance() calls within one tick must
                # not start two review workers (mirrors the tester_running flag).
                self._review_running = True
                self._review_task_id = review.id
                self.refresh_status()
                self.run_external_review(review.id)
                return
            if not (self.workflow is not None and self.workflow.has("on_review")):
                # No external review hook: a TESTING task the tester already
                # passed (left over when the workflow file was removed
                # mid-flight) would block the pipeline forever — finalize it
                # the way the review's accept would have.
                stale = next(
                    (
                        t
                        for t in self.manager.all_tasks()
                        if t.status is TaskStatus.TESTING
                        and t.verification is VerificationStatus.PASSED
                    ),
                    None,
                )
                if stale is not None:
                    self.manager.finalize_verified(stale.id)
                    self.refresh_sidebar()
            testing = self._next_testing()
            if testing is not None:
                # Set synchronously: two _advance() calls within one event-loop
                # tick must not start two tester workers (the exclusive group
                # would cancel the first before it does anything).
                self.tester_running = True
                self.run_tester(testing.id)
                return
        # -- promotion tail (reached with the tester busy, too) ------------
        if self.parallel_limit == 1 and (
            self.tester_running
            or any(t.status is TaskStatus.TESTING for t in self.manager.all_tasks())
        ):
            # Single development slot: testing keeps its absolute priority —
            # no development starts while a tester run is in flight or any
            # task awaits verification (also covers a TESTING task inside
            # its retry backoff window — the retry timer will call
            # _advance() again — and the window where a comment re-queued
            # the tested task while its tester round is still running).
            return
        if any(t.depends_on for t in self.manager.queued()):
            # The dependency gate is active: mergedness of the depends_on
            # branches must be checked with git OFF the event loop — hand
            # the whole promotion tail to the worker (exclusive group: a
            # re-entrant _advance() cancels the pending one instead of
            # promoting twice).
            self._promote_with_dependencies()
            return
        if self._dispatch_promotions(blocked=None):
            self.refresh_status()
        else:
            self._notify_pipeline_settled()
            self.refresh_status()

    def _dispatch_promotions(self, blocked: set[str] | None) -> bool:
        """Fill the free development slots from the queue: promote tasks
        one by one (promote_next) while slots remain, dispatching each
        promoted task's run_agent in its own worker group. `blocked` is
        the dependency gate's set (None when there is no gate), computed
        ONCE per pass by the caller — promotion itself never changes the
        mergedness of a dependency branch.

        The failed-queue deadlock escape (see _promote_failed_queue) runs
        inside the loop: a queue where every remaining task failed this
        session gets rescued into the free slots.

        Returns True when at least one task was promoted (the caller
        skips the settled notification then)."""
        if self._exit or not self.is_running:
            # Teardown race: the dependency-gate worker calls this directly
            # (not through _advance's guard) after its off-loop git check —
            # the app can be dismantled while that check runs.
            return False
        promoted_any = False
        while len(self.manager.active_tasks()) < self.parallel_limit:
            promoted = self.manager.promote_next(
                exclude=self._agent_failed,
                blocked=blocked,
                max_active=self.parallel_limit,
            )
            if promoted is None:
                # Deadlock escape: the queue is not empty, but every
                # promotable task in it failed earlier this session.
                promoted = self._promote_failed_queue(blocked=blocked)
            if promoted is None:
                break
            self.notifier.pipeline_busy()
            self.run_agent(promoted.id)
            promoted_any = True
        return promoted_any

    # Backoff schedule of the failed-queue deadlock escape (see
    # _promote_failed_queue): rescue rounds of a queue that keeps failing
    # grow from this base up to the cap, so a persistently broken model
    # cannot hot-loop the whole backlog (an instantly failing task would
    # otherwise cycle error → promote next → error forever).
    _FAILED_QUEUE_RETRY_BASE = 60.0
    _FAILED_QUEUE_RETRY_CAP = 600.0

    def _promote_failed_queue(self, blocked: set[str] | None = None) -> Task | None:
        """Deadlock escape for a queue of session-failed tasks.

        A task whose run crashed (network outage, model error) is demoted
        back to the queue and excluded from promotion via _agent_failed for
        the rest of the session, so one broken task can't loop the queue.
        That exclusion is session-only (not persisted), which used to make
        the situation where EVERY queued task is failed a dead end: nothing
        can be promoted until the app is restarted (or the connection probe
        recovers and _recover_from_outage fires).

        Now, when no development slot is free (nothing is active under
        parallelism 1; all slots taken otherwise) and every queued task
        carries the error flag, the flag is forcibly lifted from all of
        them and the next task is promoted by the ordinary priority rules
        — the queue keeps working instead of stalling. A notification
        reports how many tasks were rescued this way.

        `blocked` (the dependency gate's set, when the caller computed one)
        counts as "cannot be rescued": dependency-blocked tasks are neither
        lifted nor promoted, and they don't keep the rescue from firing for
        the genuinely failed ones.

        Returns the promoted task, or None when the escape does not apply
        (no free slot, the queue holds a non-failed task, or the next
        rescue round is still within its backoff window — a timer
        re-enters _advance() when the window expires).
        """
        if len(self.manager.active_tasks()) >= self.parallel_limit:
            return None
        queued = self.manager.queued()
        if not queued or any(
            t.id not in self._agent_failed and (blocked is None or t.id not in blocked)
            for t in queued
        ):
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
        self._schedule_pipeline_timer(delay, self._advance)
        rescued = [t for t in queued if blocked is None or t.id not in blocked]
        for t in rescued:
            self._agent_failed.discard(t.id)
            self._show_event(
                t,
                "Task failed and was excluded from development "
                "for the rest of the session; no error-free tasks remain "
                "in the queue — exclusion lifted, task back in development.",
            )
        self.notifier.failed_tasks_requeued(len(rescued))
        return self.manager.promote_next(
            blocked=blocked, max_active=self.parallel_limit
        )

    # -- dependency gate (depends_on promotion condition) ---------------------

    def _dep_gate_sync(self, dep_ids: list[str]) -> tuple[set[str], str]:
        """The blocking part of the dependency gate, run in a worker thread
        by _promote_with_dependencies: which dependency ids still hold
        unmerged commits on their `drcode/<id>` branch, plus the base
        branch name for the service-event texts (empty when isolation is
        off — mergedness is not checkable then)."""
        unmerged = self.task_env.unmerged_ids(dep_ids)
        base = self.task_env.base_branch() if self.task_env.active else ""
        return unmerged, base

    @work(exclusive=True, group="promote")
    async def _promote_with_dependencies(self) -> None:
        """Promotion with the dependency gate active: the mirror of
        _advance()'s tail, plus a git check of every queued task's
        depends_on ids off the event loop.

        A dependency is satisfied iff its task is COMPLETED and — under
        per-task worktree isolation — its `drcode/<dep>` branch is gone or
        fully merged into the base branch (a missing dep task is judged by
        its branch alone). The set of unsatisfied tasks is passed to
        promote_next as `blocked`, which skips them exactly like held
        ones. The gate is computed ONCE per pass (promotion never changes
        the mergedness of a dependency branch) and then the free
        development slots are filled in a loop (parallel slots — see
        _dispatch_promotions). Tasks without dependencies never enter
        this path — _advance() keeps its synchronous zero-overhead route.
        """
        queued = [t for t in self.manager.queued() if t.depends_on]
        dep_ids = sorted({d for t in queued for d in t.depends_on})
        unmerged, base = await asyncio.to_thread(self._dep_gate_sync, dep_ids)
        if self._exit or not self.is_running:
            # The git check ran off the loop; the app may have been
            # dismantled in the meantime — nothing below may touch the
            # DOM (service events, refreshes) or the pipeline.
            return
        # The git calls ran in a thread — the pipeline may have moved on
        # (a tester round started/ended, a review is in flight): re-check
        # the gates _advance() enforces synchronously before dispatching
        # here. A test that can start right now (the tester is free and a
        # TESTING task is ready) outranks promotion; a tester run already
        # in flight does not — under parallel slots the free development
        # slots are filled anyway (mirrors _advance()'s narrowed block).
        if self._review_running:
            return
        if not self.tester_running and self._next_testing() is not None:
            return
        if self.parallel_limit == 1 and (
            self.tester_running
            or any(t.status is TaskStatus.TESTING for t in self.manager.all_tasks())
        ):
            return
        blocked: set[str] = set()
        gate: dict[str, dict[str, str]] = {}
        for t in queued:
            reasons: dict[str, str] = {}
            for dep in t.depends_on:
                dep_task = self.manager.tasks.get(dep)
                if dep_task is not None and dep_task.status is not TaskStatus.COMPLETED:
                    reasons[dep] = "not completed"
                elif dep in unmerged:
                    reasons[dep] = "unmerged"
            if reasons:
                blocked.add(t.id)
                gate[t.id] = reasons
        self._dep_gate = gate
        # Fill the slots first (the git state cannot change mid-loop:
        # promotion never merges anything), then announce the gate and
        # refresh the UI once for the whole pass.
        promoted_any = self._dispatch_promotions(blocked=blocked)
        self._announce_dependency_gate(queued, gate, base)
        if not promoted_any:
            self._notify_pipeline_settled()
        self.refresh_status()
        self.refresh_sidebar()

    def _poll_dependency_unblock(self) -> None:
        """The 30s unblock tick (registered in on_mount): the dependency
        gate's mergedness half is dynamic, but it is only re-evaluated
        when something calls _advance() — a manual `git merge
        drcode/<dep>` (or a foreign merge) would otherwise sit unnoticed
        while the pipeline idles. When there are queued tasks with
        dependencies, free development slots and no development/review
        run in flight, this re-fires the gate so a hand-merge unblocks
        the dependents by itself. A busy tester does not suppress the
        tick — _advance() applies its own (narrowed) promotion gates.
        Works for parallelism 1 too (it fixes the pre-existing
        "manual merge not noticed while idle" gap)."""
        if self.agent_running or self._review_running:
            return
        if len(self.manager.active_tasks()) >= self.parallel_limit:
            return
        if not any(t.depends_on for t in self.manager.queued()):
            return
        self._advance()

    def _announce_dependency_gate(
        self,
        queued: list[Task],
        gate: dict[str, dict[str, str]],
        base: str,
    ) -> None:
        """Service events of the dependency gate, deduplicated per task and
        session: the first time a task enters the blocked set it gets one
        "Task waits: …" line listing every unsatisfied dependency (with
        title, or a note that the dep's task file is gone while its branch
        still holds work); when it leaves the set — one "eligible" line.
        Re-evaluations that don't change a task's blockedness stay silent."""
        for t in queued:
            reasons = gate.get(t.id)
            if reasons:
                if t.id in self._dep_block_notified:
                    continue
                self._dep_block_notified.add(t.id)
                parts: list[str] = []
                for dep, kind in reasons.items():
                    dep_task = self.manager.tasks.get(dep)
                    title = f" ({dep_task.title_display})" if dep_task else ""
                    if kind == "unmerged":
                        what = f"not merged into {base}" if base else "not merged"
                        if dep_task is None:
                            what += " (task file gone, branch survives)"
                    else:
                        what = "not completed"
                    parts.append(f"#{dep}{title} — {what}")
                self._show_event(t, "Task waits: dependency " + "; ".join(parts) + ".")
            elif t.id in self._dep_block_notified:
                self._dep_block_notified.discard(t.id)
                self._show_event(
                    t, "Dependencies satisfied — task eligible for development."
                )

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
        if self.manager.active_tasks():
            return
        if any(t.status is TaskStatus.TESTING for t in self.manager.all_tasks()):
            return
        remaining = [
            t for t in self.manager.all_tasks() if t.status is not TaskStatus.COMPLETED
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
            reasons.append(f"on hold: {held}")
        awaiting = sum(
            1 for t in remaining if t.status is TaskStatus.AWAITING_CLARIFICATION
        )
        if awaiting:
            reasons.append(f"awaiting a clarifying question: {awaiting}")
        failed = sum(1 for t in remaining if t.id in self._agent_failed)
        if failed:
            reasons.append(f"failed this session: {failed}")
        idle = sum(
            1
            for t in remaining
            if t.status is TaskStatus.QUEUED
            and t.id not in self._agent_failed
            and not t.pending_input
            and not t.unfinished
        )
        if idle:
            reasons.append(f"queued without task text: {idle}")
        # Tasks waiting on the dependency gate (last computed set — the
        # git check runs only in the promote worker): reported separately
        # from the "queue empty" idle state, so a backlog of chained tasks
        # never reads as "everything processed".
        dep_blocked = sum(
            1
            for t in remaining
            if t.status is TaskStatus.QUEUED and t.id in self._dep_gate
        )
        if dep_blocked:
            reasons.append(f"blocked by dependencies: {dep_blocked}")
        self.notifier.pipeline_idle(
            blocked=True,
            detail=("Reasons: " + ", ".join(reasons) + ".") if reasons else "",
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

    def _flush_in_flight_prompt(self, task: Task) -> None:
        """Persist a consumed-but-not-yet-transcribed prompt of `task`.

        run_agent pops the queued prompts from pending_input as soon as the
        run starts and records them in `_consumed_prompts`; the transcript
        picks them up only at the first node boundary. A cancel arriving in
        between drops the pending_input copy AND skips the run's own persist
        (its CancelledError handler only acts on an ACTIVE task) — without
        this flush the user's brief would vanish from the chat history and
        only the "cancelled" service event would remain.
        """
        in_flight = self._consumed_prompts.pop(task.id, None)
        if in_flight is None or in_flight[0] in all_user_prompts(task.messages):
            return
        request = ModelRequest(parts=[UserPromptPart(content=in_flight[0])])
        # The prompt was typed when it was queued, not when this run
        # consumed it — keep the chat thread chronological (same backdating
        # the node-boundary persist path does).
        backdate_user_prompt([request], in_flight[0], in_flight[1][0].at)
        task.messages.extend(serialize_messages([request]))

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
        role: AgentRole = AgentRole.CODING,
    ) -> int | None:
        """Context fill (tokens) when the compaction threshold is reached
        (percent of max_context_tokens), None otherwise. Prefers the live
        reading of the running turn over the persisted transcript, whose
        usage lags behind by the not-yet-persisted requests.

        Both the context size and the threshold come from the resolved
        config of `role` — the model actually serving the transcript — so
        a per-model compaction limit from the catalog (models.json) is
        honored instead of the flat config's value."""
        config = self.resolved_config(role)
        max_tokens = config.max_context_tokens
        if max_tokens <= 0:
            return None
        threshold = min(max(config.compaction_threshold, 1), 100)
        tokens = live_tokens
        if tokens is None:
            tokens = last_context_tokens(messages)
        if not tokens or tokens * 100 < threshold * max_tokens:
            return None
        return tokens

    def _compaction_due(self, task: Task) -> int | None:
        """Compaction threshold check for the main agent transcript,
        measured against the model actually serving it (research tasks use
        the research role's resolved config, everything else the coding
        one)."""
        return self._fill_over_threshold(
            self._live_ctx_tokens.get(task.id), task.messages, self._task_role(task)
        )

    def _mechanical_token_budget(self, role: AgentRole) -> int:
        """Token budget of the mechanical compaction floor for `role`: the
        serving model's window under its compaction threshold, minus a
        headroom reserve (system prompt, tool schemas, the next turn's
        output — MECHANICAL_HEADROOM_TOKENS, capped at a quarter of the
        window). An unknown serving window falls back to the compaction
        model's (smaller) one; both unknown → 0 (only the original prompts
        and the note survive)."""
        cfg = self.resolved_config(role)
        window = cfg.max_context_tokens
        if window <= 0:
            window = self.resolved_config(AgentRole.COMPACTION).max_context_tokens
        if window <= 0:
            return 0
        threshold = min(max(cfg.compaction_threshold, 1), 100)
        headroom = min(MECHANICAL_HEADROOM_TOKENS, window // 4)
        return max(window * threshold // 100 - headroom, 0)

    def _mechanical_compact(
        self, messages: list[dict[str, Any]], summary_role: AgentRole, what: str
    ) -> tuple[list[dict[str, Any]], str]:
        """The LLM-free floor: trim `messages` at message boundaries into a
        transcript that fits `summary_role`'s window, plus the success event
        text announcing it."""
        trimmed = mechanical_compaction_messages(
            messages, self._mechanical_token_budget(summary_role)
        )
        event = (
            "Context compacted mechanically: the transcript did not fit the "
            f"summarizer model's context window, {what} was trimmed at "
            "message boundaries without a summarizer call (original task "
            "prompt(s), the previous summary and the most recent messages "
            "kept)."
        )
        return trimmed, event

    async def _compact_transcript(
        self,
        messages: list[dict[str, Any]],
        summary_role: AgentRole,
        task: Task,
        what: str,
        checklist: str | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Summarize `messages` into a compacted transcript; returns it
        together with the success event text.

        LLM path first — but only after a pre-flight size check: when the
        rendered prompt plausibly does not fit the COMPACTION role's context
        window (chars/4 estimate), the summarizer request would be rejected
        deterministically (HTTP 400) and compaction would fail with the run
        continuing uncompressed past the hard limit. In that case the LLM is
        skipped and the transcript is trimmed mechanically
        (`mechanical_compaction_messages`) — a guaranteed floor that never
        calls the model. The same floor engages as a fallback when the
        gateway rejects the summary request anyway with a deterministic
        context-overflow error (see `_is_context_overflow_error`) — the
        chars/4 estimate can under-shoot. Any OTHER summarizer failure is
        re-raised: a transient outage should retry later, not throw away
        history.
        """
        compaction_config = self.resolved_config(AgentRole.COMPACTION)
        system_prompt, template = load_compaction_prompts()
        prompt = build_compaction_prompt(
            extract_previous_summary(messages),
            render_history_for_summary(messages),
            template,
            checklist=checklist,
        )
        if not prompt_fits_context(prompt, system_prompt, compaction_config):
            return self._mechanical_compact(messages, summary_role, what)
        agent = build_compaction_agent(compaction_config)
        try:
            result = await agent.run(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not _is_context_overflow_error(e):
                raise
            # The estimate under-shot and the gateway rejected the summary
            # request: fall back to the mechanical floor instead of failing
            # the whole compaction.
            self._show_event(
                task,
                "The summarizer model rejected the transcript "
                f"({type(e).__name__}: context overflow) — trimming "
                "mechanically instead.",
            )
            return self._mechanical_compact(messages, summary_role, what)
        # The summarizer's model calls count toward the session usage too.
        self._record_session_usage(result.usage)
        event = (
            f"Context compacted: {what} summarized, original prompt(s) kept verbatim."
        )
        return compacted_messages(messages, result.output), event

    async def _maybe_compact(self, task: Task) -> None:
        """Context compaction gate, called between agent turns (a turn
        boundary is the only safe point: the transcript never ends with
        unanswered tool calls there).

        When the context fill reaches the configured threshold (percent of
        max_context_tokens), the main agent's work is paused and the
        transcript is compacted: a tool-less summarizer agent rewrites it
        into a structured summary (anchored to the previous one, if any) —
        or, when the summarizer itself cannot be run (the rendered prompt
        does not fit the compaction model's window, or the gateway rejects
        the summary request with a deterministic context-overflow error),
        the transcript is trimmed mechanically without any LLM call. Either
        way the transcript is replaced by the original user prompt(s) plus
        the summary/note, so compaction always terminates with a history
        that fits the serving model's context. A compaction failure is
        never fatal: it is logged into the service events and the run
        continues with the uncompressed history.
        """
        tokens = self._compaction_due(task)
        if tokens is None:
            return
        role_config = self.resolved_config(self._task_role(task))
        pct = round(tokens * 100 / role_config.max_context_tokens)
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
            new_messages, event = await self._compact_transcript(
                task.messages,
                summary_role=self._task_role(task),
                task=task,
                what="history",
                checklist=(
                    render_todos(task.todos)
                    if task.kind is TaskKind.FEATURE and task.todos
                    else None
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - compaction must not kill the run
            self._compaction_failed.add(task.id)
            self._show_event(task, f"compaction failed: {type(e).__name__}: {e}")
            return
        self._compaction_failed.discard(task.id)
        task.messages = new_messages
        task.touch()
        self.manager.storage.save(task)
        # The live token reading is stale now (it belongs to the old
        # transcript); drop it so the counter falls back to the compacted
        # transcript until the next model request reports real usage.
        self._live_ctx_tokens.pop(task.id, None)
        self._show_event(task, event)
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
        leak into the coding agent's model history). The same pre-flight /
        mechanical-floor guarantees apply: when the summarizer cannot be
        run, the round is trimmed mechanically instead of looping through a
        guaranteed failure.

        Returns True when the round was compacted: run_tester must resume
        it as a continuation turn from the compacted transcript (and drop
        the stale live token reading). Returns False when compaction was
        not possible — the threshold is not reached, there is nothing to
        summarize (a no-op), or the summarizer failed with a non-overflow
        error — in which case the round is resumed uncompressed and the
        caller adds the task to `_compaction_failed`, so the mid-round gate
        stays off (a persistently broken summarizer must not loop
        pause→resume forever); turn-boundary compaction of the main agent
        still retries on later turns.
        """
        round_messages = task.tester_messages[-1] if task.tester_messages else []
        tokens = self._fill_over_threshold(
            live_tokens, round_messages, AgentRole.TESTER
        )
        if tokens is None:
            return False
        pct = round(
            tokens * 100 / self.resolved_config(AgentRole.TESTER).max_context_tokens
        )
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
            new_messages, event = await self._compact_transcript(
                round_messages,
                summary_role=AgentRole.TESTER,
                task=task,
                what="testing round",
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - compaction must not kill the round
            self._compaction_failed.add(task.id)
            self._show_event(task, f"compaction failed: {type(e).__name__}: {e}")
            return False
        self._compaction_failed.discard(task.id)
        task.tester_messages[-1] = new_messages
        task.touch()
        self.manager.storage.save(task)
        self._show_event(task, event)
        if self.viewed_id == task.id:
            self.view_task(task.id)
        return True

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
            self._agent_runs[task_id] = generation
            self._agent_failed.discard(task_id)
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
            # unfinished task resumes its own snapshot).
            root, iso_note = await asyncio.to_thread(self.task_env.ensure_root, task_id)
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
                        view_dirty = False
                        synced_epoch = self._chat_epoch
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                reasoning_block: ReasoningBlock | None = None
                                assistant_block: AssistantBlock | None = None
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        self._touch_activity(task_id)
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
                                                task_id, delta.content_delta
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
                                                    # Noted: the block mirrors
                                                    # the timeline "reasoning"
                                                    # entry the persisted
                                                    # transcript gains at the
                                                    # node boundary — the
                                                    # end-of-turn view_task
                                                    # then appends instead of
                                                    # wiping.
                                                    self._write_chat(
                                                        reasoning_block,
                                                        note=(
                                                            "reasoning",
                                                            "",
                                                            reasoning_block,
                                                        ),
                                                    )
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
                                                    # Noted: mirrors the
                                                    # timeline "assistant"
                                                    # entry (see the reasoning
                                                    # branch above).
                                                    self._write_chat(
                                                        assistant_block,
                                                        note=(
                                                            "assistant",
                                                            "",
                                                            assistant_block,
                                                        ),
                                                    )
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
                                            self._touch_activity(
                                                task_id, event.part.tool_name
                                            )
                                        elif isinstance(event, FunctionToolResultEvent):
                                            # Result stops the spinner even
                                            # when the task is off-screen.
                                            self._tool_call_finished(
                                                event.part.tool_call_id
                                            )
                                            self._touch_activity(task_id, done=True)
                                            # A todo tool just mutated the
                                            # checklist: refresh the panel of
                                            # the open chat view (the 1s poll
                                            # would catch it too — this is the
                                            # instant path).
                                            if (event.part.tool_name or "").startswith(
                                                "todo_"
                                            ):
                                                self.call_after_refresh(
                                                    self.refresh_todo_panel
                                                )
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
                                            self._write_chat(
                                                block,
                                                note=("tool-call", call_text, None),
                                            )
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
                                        tokens
                                        * 100
                                        / self.resolved_config(role).max_context_tokens
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
                        # previous node boundary. The rebuild itself goes
                        # through the delta path: the streamed blocks are
                        # reconciled against the (now persisted) timeline and
                        # only the missing tail is appended.
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
                    if self._agent_generations.get(task_id) == generation and (
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
                            # Back to pending: the pending-input render owns
                            # the bubbles again, drop the in-flight record.
                            self._consumed_prompts.pop(task_id, None)
                            if self.viewed_id == task_id:
                                for entry in consumed:
                                    if entry.text not in self._pending_rendered:
                                        self._pending_rendered.append(entry.text)
                        task.messages = sanitized_messages(task.messages)
                        task.unfinished = True
                        self.manager.storage.save(task)
                    # Drop our streaming preview; a newer run (if any) owns
                    # the label now and must not be touched.
                    self._drop_stream_preview(task_id, generation)
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
                        if self._agent_generations.get(task_id) == generation:
                            self._drop_stream_preview(task_id, generation)
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
                            # Back to pending: the pending-input render owns
                            # the bubbles again, drop the in-flight record.
                            self._consumed_prompts.pop(task_id, None)
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
                        self._schedule_pipeline_timer(
                            delay, partial(self.run_agent, task_id)
                        )
                        return
                    # Persist the failure even for a superseded run: it
                    # happened, and the human must be able to see it.
                    self._show_event(task, f"{type(e).__name__}: {e}")
                    if self._agent_generations.get(task_id) == generation:
                        # A mid-stream failure leaves a half-streamed orange
                        # preview above the input: clear it, otherwise the
                        # stale fragments hang in the dialog (and would be
                        # shown for whatever the user does next).
                        self._drop_stream_preview(task_id, generation)
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
                    # Custom workflow: on_development_done (push / open a PR)
                    # runs BEFORE mark_done — between the hook and the
                    # TESTING transition there is no await point, so no
                    # concurrent _advance() can start the tester round
                    # without the hook's note.
                    await self._run_development_done_hook(task)
                    # Completion soft gate: unchecked checklist items are
                    # announced, not blocking (see _todo_completion_warning).
                    self._todo_completion_warning(task)
                    self.manager.mark_done(task.id)
                    self.refresh_sidebar()
                    # No view_task() here: the transcript is fully on screen
                    # already (streamed live, or re-rendered by the end-of-turn
                    # dirty-check above) and mark_done changes only the status —
                    # a full wipe-and-remount would flash the whole chat for
                    # nothing (service events draw themselves, see
                    # _task_env_event / _show_event).
                    if task.kind is TaskKind.RESEARCH:
                        # Research is complete (no tester): anchor the
                        # snapshot on the branch, then the research accept
                        # hook (merge the docs branch) or the unmerged-work
                        # reminder — the research counterpart of the
                        # PASSED-finalization steps the tester path runs.
                        await self._finalize_research_completion(task)
                        self.refresh_sidebar()
                    # A test that can start right now (this task or an
                    # older one in the backlog) always goes before any
                    # new development; under parallel slots the rest of
                    # the queue flows while the tester is busy.
                    self._advance()
                elif task.status is TaskStatus.QUEUED:
                    # The run failed and demoted the task back to the queue:
                    # keep the pipeline going with the next task (failed ones
                    # are excluded, so a dead gateway can't loop the queue).
                    self._advance()

    # -- tester run ------------------------------------------------------------------

    def _schedule_tester_retry(self, task_id: str) -> None:
        """Backoff before the next tester attempt for a failed task.

        A TESTING task must not be dropped from the backlog until restart,
        so a failed tester run is retried with capped exponential backoff
        (the timer re-fires _advance(); under parallel slots the queue
        keeps flowing while the task waits out its window)."""
        count = self._tester_fail_count.get(task_id, 1)
        delay = min(5 * 2 ** (count - 1), 300)
        self._tester_retry_at[task_id] = time.monotonic() + delay
        self._schedule_pipeline_timer(delay, self._advance)

    @work(exclusive=True, group="tester")
    async def run_tester(self, task_id: str) -> None:
        """After any task completes, verify the project and label the task."""
        self._tester_task_id = task_id
        self._tester_generation += 1
        run_key: tuple[str, str, int] = ("tester", task_id, self._tester_generation)
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
            if not self.resolved_config(AgentRole.TESTER).is_ready:
                self._tester_fail_count[task_id] = (
                    self._tester_fail_count.get(task_id, 0) + 1
                )
                self._schedule_tester_retry(task_id)
                self._show_event(
                    task,
                    "Tester skipped: model is not configured. Pick a model "
                    "for the tester role in Settings (or set DRCODE_MODEL / "
                    "DRCODE_BASE_URL env vars).",
                )
                return
            # Idempotent: _advance() sets this before starting the worker so
            # two _advance() calls in one tick cannot double-start the tester.
            self.tester_running = True
            self._current_tool[task_id] = None
            self._tps_samples.pop(task_id, None)
            self._touch_activity(task_id)
            self.refresh_status()
            # New testing round: created up front, filled by
            # _persist_tester_messages as the run progresses. An empty round
            # (crash before any snapshot) is skipped when rendering.
            if not resume_round:
                task.tester_messages.append([])
                if self.viewed_id == task_id:
                    self._write_history_line("testing-sep", "")
            # The tester verifies the task in ITS worktree (per-task
            # worktree isolation, see run_agent): the round runs against
            # exactly the tree the development agent worked in, and the
            # TESTING_COMMIT finalization commits onto the task's branch.
            # A worktree lost to a manual deletion is recreated here (the
            # branch with the task's commits is re-checked-out).
            root, iso_note = await asyncio.to_thread(self.task_env.ensure_root, task_id)
            if iso_note:
                self._show_event(task, f"Task isolation unavailable: {iso_note}")
            tester = build_tester_agent(
                self.resolved_config(AgentRole.TESTER),
                allow_commit=(
                    self.settings.task_finalization is TaskFinalization.TESTING_COMMIT
                ),
                root=root,
                # The external review (`.drcode/workflow.py` on_review)
                # splits "the tester passed" from "the task is accepted":
                # a PASSED verdict then leaves the task TESTING.
                review_gate=(
                    self.workflow is not None and self.workflow.has("on_review")
                ),
            )
            deps = TesterDeps(
                root=root,
                manager=self.manager,
                task_id=task_id,
                scratch=scratch_dir(task_id),
                review_gate=(
                    self.workflow is not None and self.workflow.has("on_review")
                ),
                # The fix-loop cap: how many consecutive failed development
                # rounds (retry_here verdicts) the task gets before its
                # failure completes it as FAILED (and the tester creates
                # the bug report).
                max_dev_rounds=self.settings.max_dev_rounds,
            )
            try:
                prompt = f"Verify completed task #{task.id}: {task.title}"
                # The tester needs the original assignment (to paraphrase it
                # into a commit message — see the git_commit step in its
                # prompt) plus every clarification the user sent while the
                # task was being worked on, so the verification targets the
                # CURRENT intent, not a stale initial one.
                user_prompts = [
                    p for p in all_user_prompts(task.messages) if p != task.title
                ]
                if user_prompts:
                    prompt += f"\n\nOriginal task:\n{user_prompts[0]}"
                if len(user_prompts) > 1:
                    prompt += (
                        "\n\nUser clarifications sent during development "
                        "(in order, latest last):\n"
                        + "\n\n".join(f"- {c}" for c in user_prompts[1:])
                    )
                # A GitHub-imported task must link its commit back to the
                # originating issue ("Fixes #N" keyword in the message).
                if task.github_issue is not None:
                    prompt += f"\n\n{issue_commit_hint(task.github_issue)}"
                # Custom workflow: context returned by on_development_done
                # (the opened PR, the pushed branch, CI links, …).
                dev_note = self._dev_done_notes.pop(task_id, None)
                if dev_note:
                    prompt += (
                        "\n\nProject workflow notes (from the on_development_done "
                        f"step):\n{dev_note}"
                    )
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
                                # Stream the round's reasoning/answer into
                                # the open chat view (mirrors run_agent's
                                # streaming branch): everything the tester
                                # produces must already be on screen when
                                # the end-of-round re-render runs, so that
                                # re-render can go through the delta path
                                # (append the missing tail) instead of
                                # wiping and remounting the whole history.
                                # Every event is a sign of life: without the
                                # touch a long (minutes) report generation
                                # reads as "stalled" — and the stall watchdog
                                # aborts a perfectly healthy run.
                                reasoning_block: ReasoningBlock | None = None
                                assistant_block: AssistantBlock | None = None
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        self._touch_activity(task_id)
                                        if isinstance(event, PartStartEvent):
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
                                        if not delta.content_delta:
                                            continue
                                        if isinstance(delta, ThinkingPartDelta):
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
                                                    # over (see run_agent).
                                                    reasoning_block.set_content(
                                                        old._content
                                                    )
                                                reasoning_block.begin()
                                                if self.viewed_id == task_id:
                                                    self._write_chat(
                                                        reasoning_block,
                                                        note=(
                                                            "reasoning",
                                                            "",
                                                            reasoning_block,
                                                        ),
                                                    )
                                                reasoning_block.append(
                                                    delta.content_delta
                                                )
                                            else:
                                                reasoning_block.thinking(True)
                                                reasoning_block.append(
                                                    delta.content_delta
                                                )
                                        else:
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
                                                    # See the reasoning branch.
                                                    assistant_block.set_content(
                                                        old._content
                                                    )
                                                if self.viewed_id == task_id:
                                                    self._write_chat(
                                                        assistant_block,
                                                        note=(
                                                            "assistant",
                                                            "",
                                                            assistant_block,
                                                        ),
                                                    )
                                                assistant_block.append(
                                                    delta.content_delta
                                                )
                                            else:
                                                assistant_block.append(
                                                    delta.content_delta
                                                )
                                if reasoning_block is not None:
                                    reasoning_block.finish()
                                if assistant_block is not None:
                                    assistant_block.finish()
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
                                            self._touch_activity(
                                                task_id, event.part.tool_name
                                            )
                                        elif isinstance(event, FunctionToolResultEvent):
                                            # Result stops the spinner even when
                                            # the task is off-screen.
                                            self._tool_call_finished(
                                                event.part.tool_call_id
                                            )
                                            self._touch_activity(task_id, done=True)
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
                                            self._write_chat(
                                                block,
                                                note=("tool-call", call_text, None),
                                            )
                                # Mid-round compaction gate: a tool boundary
                                # is the only safe stopping point (same as
                                # the agent's mid-run gate). Skipped when a
                                # previous compaction attempt failed, or a
                                # persistently broken summarizer would loop
                                # pause→resume forever.
                                tokens = self._fill_over_threshold(
                                    live_tokens,
                                    task.tester_messages[-1],
                                    AgentRole.TESTER,
                                )
                                if tokens is not None and (
                                    task.id not in self._compaction_failed
                                ):
                                    pct = round(
                                        tokens
                                        * 100
                                        / self.resolved_config(
                                            AgentRole.TESTER
                                        ).max_context_tokens
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
                        # Re-render from the persisted transcript: everything
                        # the round streamed live (separator, tool calls,
                        # reasoning, answer) is on screen and mirrored by the
                        # delta ledger, so this appends only the entries the
                        # live view is missing (usually nothing) instead of
                        # wiping and remounting the whole history.
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
                    self._schedule_pipeline_timer(delay, self._advance)
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
            # Custom workflow: on_tester_passed after a PASSED finalization
            # (no external review — with the review gate the task is still
            # TESTING here and the hook fires from the review's accept path
            # instead). State-checked, so it also fires when the run was
            # aborted after set_verification had already finalized the task
            # (the stall nudge), and never for FAILED labels, research
            # tasks, cancels, or the "finalization: none" path (no label).
            if (
                task.status is TaskStatus.COMPLETED
                and task.verification is VerificationStatus.PASSED
                and not task.review_pending
            ):
                await self._run_tester_passed_hook(task)
                await self._remind_unmerged_work(task)
            # A cancelled run (user abort, or a newer tester taking over)
            # must not restart the pipeline.
            if not cancelled:
                self._advance()

    # -- external review (.drcode/workflow.py hooks) -----------------------------------

    def _hook_env(self, task: Task) -> tuple[Path, dict[str, str] | None]:
        """Where a workflow hook for `task` runs: with per-task worktree
        isolation on — inside the task's worktree (the payload describes
        that tree in `git.*` and carries the `worktree` path/branch entry);
        without it — the project directory as before."""
        info = self.task_env.worktree_info(task.id)
        if info is not None:
            return Path(info["path"]), info
        return self.project_root, None

    async def _run_task_start_hook(self, task: Task) -> str | None:
        """`on_task_start` before the first activation: branch checkout and
        friends. Returns the hook's "instructions" text (injected into the
        coding agent's first prompt), None when there is nothing to inject.
        Never fatal: a failing hook is logged and the run proceeds. Fires
        only when the transcript is still empty — the hook must not run
        again on every resume; it is the script's job to be idempotent
        beyond that."""
        workflow = self.workflow
        if workflow is None or not workflow.has("on_task_start"):
            return None
        if task.messages:
            return None  # not the first activation
        cwd, wt = self._hook_env(task)
        result = await workflow.call("on_task_start", task, cwd=cwd, worktree=wt)
        if not result.ok:
            self._show_event(
                task,
                f"Workflow on_task_start failed (continuing without it): "
                f"{result.error}",
                kind="workflow",
            )
            return None
        instructions = str(result.output.get("instructions") or "").strip()
        if instructions:
            self._show_event(
                task,
                "Workflow on_task_start: project instructions injected "
                "into the task prompt.",
                kind="workflow",
            )
            return instructions
        return None

    async def _run_development_done_hook(self, task: Task) -> None:
        """`on_development_done` when a development round finishes (push,
        open a PR, …) and before the tester round starts. The returned text
        is kept for the tester's prompt; without a synchronous on_review
        hook this switches the task into the ASYNC review mode — it will
        complete normally and await the verdict via the review poll."""
        workflow = self.workflow
        if workflow is None or not workflow.has("on_development_done"):
            return
        if task.kind is TaskKind.RESEARCH:
            return  # research tasks never go through testing/finalization
        cwd, wt = self._hook_env(task)
        result = await workflow.call("on_development_done", task, cwd=cwd, worktree=wt)
        if not result.ok:
            self._show_event(
                task,
                f"Workflow on_development_done failed (continuing without "
                f"it): {result.error}",
                kind="workflow",
            )
            return
        text = str(result.output.get("text") or "").strip()
        if text:
            self._dev_done_notes[task.id] = text

    async def _run_tester_passed_hook(self, task: Task) -> None:
        """`on_tester_passed` right after the tester stamped the task PASSED
        and the task is finalized (install, deploy, …). Fires on every path
        to COMPLETED-after-PASSED: without an external review directly from
        the tester round's completion, and on a review-mode task when the
        review finally accepts it — an install step must run no matter
        which path accepted the task. Fail-open like `on_accepted` (an
        install step must not park the task the way the fail-closed
        `on_review` does); its text goes to the service-event log."""
        workflow = self.workflow
        if workflow is None or not workflow.has("on_tester_passed"):
            return
        if task.kind is TaskKind.RESEARCH:
            return  # research tasks never go through testing/finalization
        cwd, wt = self._hook_env(task)
        result = await workflow.call("on_tester_passed", task, cwd=cwd, worktree=wt)
        if not result.ok:
            self._show_event(
                task,
                f"Workflow on_tester_passed failed (non-fatal): {result.error}",
                kind="workflow",
            )
            return
        text = str(result.output.get("text") or "").strip()
        if text:
            self._show_event(task, f"on_tester_passed:\n{text}", kind="workflow")

    # Hook-merge failure markers: the workflow template's hooks report a
    # failed `git merge` as free text ("merge of drcode/<id> into main
    # FAILED — CONFLICT …", "NOT merged — the checkout has uncommitted
    # changes"). A crashed/nonzero-exit hook arrives as `not result.ok`
    # instead — covered separately in _record_hook_merge_outcome.
    _HOOK_MERGE_FAILURE_MARKERS = ("FAILED", "NOT merged")

    def _record_hook_merge_outcome(self, task: Task, result: Any) -> None:
        """Stamp `task.hook_merge_failed` with the accept hook's merge
        outcome and persist it. The accept hooks are fail-open (the task
        stays COMPLETED), but a FAILED/NOT merged merge must not vanish:
        the flag re-admits the task into the unmerged-work tracking (the
        header's `⎇ N` counter and the startup reminder) so a human
        resolves it — see DrCodeApp._unmerged_completed_ids. A successful
        merge (or a later successful re-run) clears the flag again."""
        failed = not result.ok or any(
            marker in str(result.output.get("text") or "")
            for marker in self._HOOK_MERGE_FAILURE_MARKERS
        )
        if task.hook_merge_failed is failed:
            return
        task.hook_merge_failed = failed
        task.touch()
        self.manager.storage.save(task)

    async def _run_accepted_hook(self, task: Task) -> None:
        """`on_accepted` after an accept verdict (merge the PR, delete the
        branch, …). Its text goes to the service-event log. Never fatal.

        Serialized through `_accept_hook_lock`: with parallel slots two
        tasks can be accepted around the same moment, and both hooks
        (typically `git merge` into the shared main checkout) must not
        interleave — git's index.lock and conflicting merges would break
        one of them. Fail-open semantics unchanged: a failed hook-merge
        only stamps `task.hook_merge_failed` (unmerged-work tracking picks
        the task up), the task itself stays COMPLETED."""
        workflow = self.workflow
        if workflow is None or not workflow.has("on_accepted"):
            return
        async with self._accept_hook_lock:
            cwd, wt = self._hook_env(task)
            result = await workflow.call("on_accepted", task, cwd=cwd, worktree=wt)
            self._record_hook_merge_outcome(task, result)
            if not result.ok:
                self._show_event(
                    task,
                    f"Workflow on_accepted failed (non-fatal): {result.error}",
                    kind="workflow",
                )
                return
            text = str(result.output.get("text") or "").strip()
            if text:
                self._show_event(task, f"on_accepted:\n{text}", kind="workflow")

    async def _run_research_accepted_hook(self, task: Task) -> None:
        """`on_research_accepted` right after a RESEARCH task completes —
        the research counterpart of `on_accepted` (research skips testing,
        so the review-accept path never runs for it): the team's script
        merges the docs branch `drcode/<task-id>` back, pushes, opens a
        PR, … Its text goes to the service-event log. Never fatal. The
        payload carries `project` (the user's checkout path and base
        branch) so a hook running inside the task's worktree can still
        merge into the main tree (`git -C <project.path> merge <branch>`).

        Serialized through `_accept_hook_lock`, like `on_accepted` (see
        there): parallel finalizations merge one at a time. A failed
        hook-merge stamps `task.hook_merge_failed` (unmerged-work tracking
        picks the task up) — fail-open, the task stays COMPLETED."""
        workflow = self.workflow
        if workflow is None or not workflow.has("on_research_accepted"):
            return
        async with self._accept_hook_lock:
            project: dict[str, Any] = {"path": str(self.project_root)}
            if self.task_env.available:
                project["base_branch"] = await asyncio.to_thread(
                    self.task_env.base_branch
                )
            cwd, wt = self._hook_env(task)
            result = await workflow.call(
                "on_research_accepted",
                task,
                cwd=cwd,
                worktree=wt,
                extra={"project": project},
            )
            self._record_hook_merge_outcome(task, result)
            if not result.ok:
                self._show_event(
                    task,
                    f"Workflow on_research_accepted failed (non-fatal): {result.error}",
                    kind="workflow",
                )
                return
            text = str(result.output.get("text") or "").strip()
            if text:
                self._show_event(
                    task, f"on_research_accepted:\n{text}", kind="workflow"
                )

    async def _anchor_research_snapshot(self, task: Task) -> None:
        """Make sure the completion snapshot has landed on the task's
        branch before the research hook (or the reminder) looks at it:
        `_snapshot_on_settled` commits the worktree in a background job —
        a hook firing concurrently could merge a branch still being
        written to. Waits for the pending snapshot jobs, then snapshots
        once more synchronously (a no-op when the listener already
        committed everything)."""
        if not self.task_env.active:
            return
        if self._snapshot_jobs:
            await asyncio.gather(*list(self._snapshot_jobs), return_exceptions=True)
        msg = await asyncio.to_thread(
            self.task_env.snapshot, task.id, "research completed"
        )
        if msg is not None and msg.startswith("snapshot committed"):
            self._show_event(task, msg)

    async def _finalize_research_completion(self, task: Task) -> None:
        """Post-completion steps of a RESEARCH task (research skips
        testing, so none of the PASSED-finalization steps run for it):
        anchor the completion snapshot on the branch, then the
        on_research_accepted hook (merge the docs branch) — or, without
        the hook, the unmerged-work reminder. Every step is non-fatal."""
        await self._anchor_research_snapshot(task)
        await self._run_research_accepted_hook(task)
        await self._remind_unmerged_work(task)

    async def _remind_unmerged_work(self, task: Task) -> None:
        """After a finalization (PASSED for development tasks, plain
        completion for research): with per-task worktree isolation
        the task's work lives on its own branch `drcode/<task-id>` —
        drcode deliberately never merges it back (merging is a team
        decision, see `drcode.task_env`); it is left to the project's
        `on_accepted` hook or to the user. When the branch still holds
        commits the main checkout's current branch does not have, say so
        explicitly — into the task's service-event log and as a toast —
        instead of leaving a PASSED label imply that the user's checkout
        already has the fix (for a research task, that its docs commit
        reached the checkout; without isolation the work is in the shared
        tree already).

        With the task kind's accept hook defined (`on_accepted` for
        development tasks, `on_research_accepted` for research) the merge
        is (supposed to be) automated by that hook, so the toast is
        skipped there — unless the hook-merge itself failed
        (`task.hook_merge_failed`): then the branch is stuck on a conflict
        and the user must hear about it. The service event above is kept
        either way, as the durable record."""
        env = self.task_env
        if not env.active:
            return  # no isolation — nothing ever lives on a task branch
        unmerged = await asyncio.to_thread(env.unmerged_commits, task.id)
        if unmerged <= 0:
            return  # merged (or never committed) — nothing to remind about
        branch = branch_for(task.id)
        base = await asyncio.to_thread(env.base_branch)
        plural = "" if unmerged == 1 else "s"
        research = task.kind is TaskKind.RESEARCH
        if research:
            text = (
                f"Documentation committed on branch {branch} and not "
                f"merged into {base} yet ({unmerged} commit{plural}). "
                f"Merging is up to you (or the on_research_accepted "
                f"hook): git merge {branch}"
            )
        else:
            text = (
                f"Work finished on branch {branch} and not merged into "
                f"{base} yet ({unmerged} commit{plural}). Merging is up to "
                f"you (or the on_accepted hook): git merge {branch}"
            )
        self._show_event(task, text, kind="workflow")
        hook = "on_research_accepted" if research else "on_accepted"
        if not (
            self.workflow is not None
            and self.workflow.has(hook)
            and not task.hook_merge_failed
        ):
            # The chat may be open on another task — also raise a toast.
            self.notify(text, title="Unmerged work", severity="warning", timeout=12)

    def _apply_review_accept(self, task: Task) -> None:
        """Finish the task after an accept verdict: a sync-mode task is
        still TESTING (finalize_verified completes it with the PASSED
        label); an async-mode task is already COMPLETED (just clear the
        review-awaiting marker and the rejection count). A task the user
        re-queued while the review ran (QUEUED/HOLD) is left alone."""
        if task.status is TaskStatus.TESTING:
            self.manager.finalize_verified(task.id)
        elif task.status is TaskStatus.COMPLETED:
            task.review_pending = False
            task.review_rejections = 0
            task.touch()
            self.manager.storage.save(task)

    @work(exclusive=True, group="review")
    async def run_external_review(self, task_id: str) -> None:
        """One external review attempt for a passed task (sync mode after
        the tester round, or the retry/async poll). A reject re-queues the
        task with the reviewer's comment as new input (the coding agent
        reworks it in the same transcript); an accept completes it and
        runs on_accepted; a failure or a pending verdict keeps the task
        waiting (fail-closed) and schedules a retry."""
        task = self.manager.get(task_id)
        workflow = self.workflow
        cwd, wt = self._hook_env(task)
        try:
            if workflow is None:
                return
            result = await workflow.call("on_review", task, cwd=cwd, worktree=wt)
        finally:
            self._review_running = False
            self._review_task_id = None
            self.refresh_status()
        if not result.ok:
            # Fail-closed: an unreachable reviewer must not wave the change
            # through — the task stays TESTING+PASSED and is retried with
            # backoff (like a failed tester round).
            count = self._review_fail_count.get(task_id, 0) + 1
            self._review_fail_count[task_id] = count
            delay = min(
                self._REVIEW_RETRY_BASE * 2 ** (count - 1), self._REVIEW_RETRY_CAP
            )
            self._review_retry_at[task_id] = time.monotonic() + delay
            self._show_event(
                task,
                f"External review (on_review) failed — the task waits for "
                f"the next attempt ({count}, in {int(delay)} s): {result.error}",
                kind="workflow",
            )
            self._schedule_pipeline_timer(delay, self._advance)
            return
        self._review_fail_count.pop(task_id, None)
        verdict = str(result.output.get("verdict") or "").strip().lower()
        if verdict == "accept":
            self._show_event(task, "External review: accepted.", kind="workflow")
            if task.status not in (TaskStatus.TESTING, TaskStatus.COMPLETED):
                # The user took the task back into development (send()) or
                # parked it while the review ran — no finalization.
                self._advance()
                return
            self._apply_review_accept(task)
            self.refresh_sidebar()
            # on_tester_passed fires on every accept path too (a review-mode
            # task reaches COMPLETED-after-PASSED only here), before
            # on_accepted — lifecycle order.
            await self._run_tester_passed_hook(task)
            await self._run_accepted_hook(task)
            await self._remind_unmerged_work(task)
            self._advance()
        elif verdict == "reject":
            comment = (
                str(result.output.get("comment") or "").strip()
                or "The external review rejected the change — rework the "
                "task according to the review notes."
            )
            self._show_event(
                task,
                f"External review: rejected — the task goes back to "
                f"development.\n\n{comment}",
                kind="workflow",
            )
            self.manager.requeue_review_rejection(task.id, comment)
            self.refresh_sidebar()
            if task.review_rejections >= MAX_REVIEW_REJECTIONS:
                # Loop protection: park the task for a human instead of
                # ping-ponging agent ⇄ reviewer forever (same idea as the
                # tester backoff).
                self.manager.set_hold(task.id, on=True)
                self._show_event(
                    task,
                    f"Rejected {task.review_rejections} times in a row by "
                    "the external review — the task is put on hold and "
                    "requires human attention (Ctrl+P resumes it).",
                    kind="workflow",
                )
            self._advance()
        else:
            # No final verdict yet (CI still running, a human review in
            # progress): release the pipeline — the task completes (with
            # its PASSED label) and the periodic poll re-asks on_review
            # until a final verdict arrives. "Awaiting review" = COMPLETED
            # with an open PR; such tasks are excluded from the startup
            # auto-cleanup, so a verdict days later still finds them.
            self._show_event(
                task,
                f"External review has no final verdict yet "
                f"({verdict or 'none reported'}) — the task is released "
                "and the review verdict will be polled.",
                kind="workflow",
            )
            if task.status is TaskStatus.TESTING:
                self.manager.finalize_verified(task.id)
            if task.status is TaskStatus.COMPLETED:
                task.review_pending = True
                task.touch()
                self.manager.storage.save(task)
            self._review_poll_at = time.monotonic() + self._REVIEW_PENDING_POLL

    def _poll_workflow_review(self) -> None:
        """Async review poll (every 30s tick, rate-limited to one attempt
        per _REVIEW_PENDING_POLL): ask on_review for the verdict of every
        COMPLETED task awaiting its external review (review_pending — an
        open PR). The script checks the PR state itself (merged/approved →
        accept, "Changes requested" → reject, still open → pending)."""
        if self.workflow is None or not self.workflow.has("on_review"):
            return
        if self._exit or not self.is_running:
            return
        if self._review_running:
            return
        if time.monotonic() < self._review_poll_at:
            return
        pending = next(
            (
                t
                for t in self.manager.all_tasks()
                if t.status is TaskStatus.COMPLETED and t.review_pending
            ),
            None,
        )
        if pending is None:
            return
        self._review_poll_at = time.monotonic() + self._REVIEW_PENDING_POLL
        self._review_running = True
        self._review_task_id = pending.id
        self.refresh_status()
        self.run_external_review(pending.id)

    # -- actions ---------------------------------------------------------------------

    # -- new task mode ---------------------------------------------------------------
