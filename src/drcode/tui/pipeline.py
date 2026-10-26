"""The pipeline core: the single `_advance` gate and its satellites.

The role-specific layers live in the sibling `pipeline_*.py`
modules and are composed into `PipelineLayer` below — the one name
`drcode.tui.pipeline` has always exported (the app's MRO plugs it in
between `ActionsLayer` and `App`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from pydantic_ai.agent import AgentRun
from pydantic_ai.messages import ModelRequest, UserPromptPart

from ..agent import (
    AgentDeps,
    TesterDeps,
    all_user_prompts,
    backdate_user_prompt,
    serialize_messages,
    todo_instructions,
)
from ..config import Config
from ..keepawake import statuses_hold_awake
from ..models import Task, TaskKind, TaskStatus, VerificationStatus
from ..providers import AgentRole
from ..settings import TaskIsolation, Verification
from .pipeline_agent_run import PipelineAgentRunLayer
from .pipeline_compaction import PipelineCompactionLayer
from .pipeline_deps import PipelineDepsLayer
from .pipeline_hooks import PipelineHooksLayer
from .pipeline_net import PipelineNetLayer
from .pipeline_resolver import PipelineResolverLayer
from .pipeline_tester import PipelineTesterLayer
from .state import DrCodeState


class PipelineCoreLayer(DrCodeState):
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
        """The single pipeline gate: every state change re-enters here.
        Thin wrapper so the keep-awake assertion is re-evaluated on EVERY
        path — including the early returns of the gate itself (a tester
        dispatch, an external review) that would otherwise skip a tail
        hook. See _update_keepawake below."""
        self._advance_gate()
        self._update_keepawake()

    def _keepawake_busy(self) -> bool:
        """Whether the machine must be kept awake right now: a run in
        flight (agent, tester, external review) or any task in a status
        the pipeline may still act on (QUEUED/ACTIVE/TESTING — see
        drcode.keepawake.BUSY_STATUSES; dependency-blocked QUEUED counts,
        a manual `git merge drcode/<dep>` can unblock it at any time).
        HOLD, AWAITING_CLARIFICATION and COMPLETED do not hold it."""
        if (
            self.agent_running
            or self.tester_running
            or self._review_running
            or self._resolver_running
        ):
            return True
        return statuses_hold_awake(t.status for t in self.manager.all_tasks())

    def _update_keepawake(self) -> None:
        """Recompute the sleep assertion from the busy state and the
        project setting. Called from the _advance wrapper (every pipeline
        transition) and the periodic tick in on_mount (which doubles as
        the caffeinate respawn check). Teardown always releases. When the
        held/unheld state flips, the header status is re-rendered so the
        `· no-sleep` marker never goes stale."""
        was_active = self.keepawake.active
        if self._exit or not self.is_running:
            self.keepawake.stop()
            return
        if not self.settings.keep_awake or not self._keepawake_busy():
            self.keepawake.stop()
        else:
            self.keepawake.start("drcode: task in flight")
        if self.keepawake.active != was_active:
            self.refresh_status()

    def _advance_gate(self) -> None:
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
            if self.settings.verification is Verification.NONE:
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

        The lift applies only to plausibly-transient failures: tasks whose
        failure is deterministic (`_agent_dead` — e.g. a transcript over
        the model's context window, rejected with the same 400 on every
        attempt) are never lifted, otherwise they would loop
        error → rescue → error forever. When every failed task is dead,
        the escape does not fire — the settled-pipeline notification
        reports those tasks as needing manual intervention.

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
        if self._outage_active():
            # A network outage is in progress (some task sits in the
            # outage-mode wait): rescue rounds would only fire full model
            # requests at a gateway the outage probe just proved
            # unreachable. The outage ticks re-advance the pipeline, and
            # the first successful model call resets this escape's backoff
            # — so the rescue happens right after the recovery, not
            # against the dead network.
            return None
        queued = self.manager.queued()
        # Only plausibly-transient failures are rescuable: a deterministically
        # dead task (_agent_dead — e.g. a transcript over the model's context
        # window) keeps its exclusion forever, lifting it would just re-run
        # the same guaranteed error.
        rescuable = [
            t
            for t in queued
            if t.id in self._agent_failed
            and t.id not in self._agent_dead
            and (blocked is None or t.id not in blocked)
        ]
        if not rescuable or any(
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
        rescued = rescuable
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
        dead = sum(1 for t in remaining if t.id in self._agent_dead)
        if dead:
            reasons.append(
                f"transcript over the model's context window "
                f"(needs manual intervention): {dead}"
            )
        failed = sum(
            1
            for t in remaining
            if t.id in self._agent_failed and t.id not in self._agent_dead
        )
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


class PipelineLayer(
    PipelineDepsLayer,
    PipelineHooksLayer,
    PipelineTesterLayer,
    PipelineResolverLayer,
    PipelineAgentRunLayer,
    PipelineCompactionLayer,
    PipelineNetLayer,
    PipelineCoreLayer,
):
    """The full pipeline: every sub-layer composed over `DrCodeState`.

    The split keeps the app's class line (`DrCodeApp(ChatViewLayer,
    ActionsLayer, PipelineLayer, App[None])`) and the test import
    (`from drcode.tui.pipeline import PipelineLayer`) untouched — this
    class resolves every method the monolith used to define.
    """
