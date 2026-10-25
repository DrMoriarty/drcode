"""The dependency gate and the external review.

Split out of the former monolithic `tui/pipeline.py`: promotion with
`depends_on` (mergedness checked with git off the event loop), the
30s unblock poll and the `.drcode/workflow.py` on_review worker with
its retry/async-poll schedule.
"""

from __future__ import annotations

import asyncio
import time

from textual import work

from ..models import Task, TaskStatus
from ..workflow import MAX_REVIEW_REJECTIONS
from .state import DrCodeState


class PipelineDepsLayer(DrCodeState):
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

    # Review retry backoff (a failed/pending external review is retried
    # like a failed tester round: capped exponential, from _REVIEW_RETRY_BASE).
    _REVIEW_RETRY_BASE = 30.0
    _REVIEW_RETRY_CAP = 600.0
    # Poll interval for the async review mode: how often the completed
    # tasks with an open PR (review_pending) ask the on_review hook again.
    _REVIEW_PENDING_POLL = 60.0

    @work(exclusive=True, group="review")
    async def run_external_review(self, task_id: str) -> None:
        """One external review attempt for a passed task (sync mode after
        the tester round, or the retry/async poll). A reject re-queues the
        task with the reviewer's comment as new input (the coding agent
        reworks it in the same transcript); an accept completes it and
        runs on_accepted (the merge) followed by on_tester_passed (the
        install/deploy — it must see the merged checkout); a failure or a
        pending verdict keeps the task waiting (fail-closed) and schedules
        a retry."""
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
            # task reaches COMPLETED-after-PASSED only here) — but AFTER
            # on_accepted: an install/deploy step must see the merged main
            # checkout, not the pre-merge state (otherwise it always lags
            # one merge behind; only a version bump would mask it).
            await self._run_accepted_hook(task)
            await self._run_tester_passed_hook(task)
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
