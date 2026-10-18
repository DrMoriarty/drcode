"""Task state machine: creation, queue, activation, re-activation."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from .models import PendingInput, Task, TaskKind, TaskStatus, _now
from .storage import Storage


class TaskManager:
    def __init__(self, project_root: Path) -> None:
        self.storage = Storage(project_root)
        self.tasks: dict[str, Task] = {t.id: t for t in self.storage.load_all()}
        # Monotonic queue-position counter: bumped when a task is explicitly
        # re-positioned in the queue (kind change to bug → head, to feature
        # → tail). Older tasks without a stamp get one in created_at order,
        # so the stamp reproduces the historical ordering exactly.
        self._queue_seq = max(
            (t.queue_seq for t in self.tasks.values() if t.queue_seq is not None),
            default=0,
        )
        for t in sorted(self.tasks.values(), key=lambda t: t.created_at):
            if t.queue_seq is None:
                t.queue_seq = self._next_queue_seq()
                self.storage.save(t)
        # On startup at most one task may have been active when the app closed.
        for t in self.tasks.values():
            if t.status is TaskStatus.ACTIVE:
                t.status = TaskStatus.QUEUED
                self.storage.save(t)

    def _next_queue_seq(self) -> int:
        self._queue_seq += 1
        return self._queue_seq

    # -- queries ------------------------------------------------------------

    def all_tasks(self) -> list[Task]:
        return sorted(self.tasks.values(), key=lambda t: t.created_at)

    def get(self, task_id: str) -> Task:
        return self.tasks[task_id]

    @property
    def active(self) -> Task | None:
        for t in self.tasks.values():
            if t.status is TaskStatus.ACTIVE:
                return t
        return None

    def queued(self) -> list[Task]:
        tasks = [t for t in self.all_tasks() if t.status is TaskStatus.QUEUED]

        # Bug reports always come first — newest at the very head, so a bug
        # report the tester has just created is picked up immediately after
        # the testing round finishes. Everything else follows: answered
        # clarifications (queue_head) before the rest, then the explicit
        # queue stamp (creation order, bumped when a task is re-positioned,
        # e.g. by a kind change).
        def _queue_key(t: Task) -> tuple:
            if t.kind is TaskKind.BUG:
                return (0, 0, -(t.queue_seq if t.queue_seq is not None else 0))
            return (1, not t.queue_head, t.queue_seq if t.queue_seq is not None else 0)

        tasks.sort(key=_queue_key)
        return tasks

    # -- mutations ------------------------------------------------------------

    def create(
        self, title: str, first_message: str, kind: TaskKind = TaskKind.FEATURE
    ) -> Task:
        first_line = (
            first_message.splitlines()[0][:60] if first_message.splitlines() else ""
        )
        task = Task(
            title=title or first_line or "New task",
            kind=kind,
            queue_seq=self._next_queue_seq(),
        )
        if first_message.strip():
            task.pending_input.append(PendingInput(text=first_message.strip()))
        self.tasks[task.id] = task
        self.storage.save(task)
        return task

    def send(self, task_id: str, text: str) -> Task:
        """Add user input to a task (appended chronologically). A completed
        (or being-tested) task re-enters the pipeline; an answered
        clarification goes to the head of the queue. Promotion (activation)
        is up to the caller: the app advances the pipeline through
        _advance(), which never starts new development while any testing is
        incomplete."""
        task = self.get(task_id)
        task.pending_input.append(PendingInput(text=text.strip()))
        if task.status in (TaskStatus.COMPLETED, TaskStatus.TESTING):
            task.status = TaskStatus.QUEUED
            # Old verification label no longer applies: the task will be
            # tested again after its next development round.
            task.verification = None
            # Fresh pipeline run: the completion timestamp (auto-cleanup
            # anchor) restarts with the eventual new completion.
            task.completed_at = None
            task.touch()
            self.storage.save(task)
        elif task.status is TaskStatus.AWAITING_CLARIFICATION:
            task.status = TaskStatus.QUEUED
            task.queue_head = True
            task.clarification = None
            task.touch()
            self.storage.save(task)
        else:
            # QUEUED/ACTIVE: just accumulate input (also when the model is not
            # configured — the agent picks it up on the next activation).
            task.touch()
            self.storage.save(task)
        return task

    def set_kind(self, task_id: str, kind: TaskKind) -> Task:
        """Change the kind of a not-yet-started (QUEUED) task and re-position
        it in the queue: bug → head of the queue, feature → tail. Other kinds
        (documentation) keep the current position. The caller is responsible
        for only calling this on queued tasks."""
        task = self.get(task_id)
        if task.kind is kind:
            return task
        task.kind = kind
        if kind is TaskKind.BUG:
            task.queue_head = True
        elif kind is TaskKind.FEATURE:
            task.queue_head = False
            task.queue_seq = self._next_queue_seq()
        task.touch()
        self.storage.save(task)
        return task

    def delete(self, task_id: str) -> None:
        """Remove the task entirely: from memory and from disk storage.
        Intended for parked (held) tasks; the caller decides which statuses
        may be deleted."""
        self.tasks.pop(task_id, None)
        self.storage.delete(task_id)

    def purge_expired_completed(self, retention_days: int | None) -> list[Task]:
        """Delete COMPLETED tasks older than the retention period.

        Runs once at app startup. `retention_days` None (setting "never")
        disables the cleanup. Age anchor: `completed_at` (set on every
        transition to COMPLETED); for tasks completed by an older build
        without the timestamp, `updated_at` is used as a fallback. Returns
        the removed tasks so the caller can report the cleanup.
        """
        if retention_days is None:
            return []
        cutoff = _now() - timedelta(days=retention_days)
        expired = [
            t
            for t in self.tasks.values()
            if t.status is TaskStatus.COMPLETED
            and (t.completed_at or t.updated_at) < cutoff
        ]
        for task in expired:
            self.delete(task.id)
        return expired

    def set_hold(self, task_id: str, *, on: bool) -> Task:
        """Park a not-yet-started task out of the backlog (hold) or return it
        to where it was. Held tasks keep their pending input, queue stamp and
        the pending clarifying question, but are never promoted while held —
        queued() skips them, so promote_next() can't pick them up.

        Holdable statuses: QUEUED and AWAITING_CLARIFICATION (a question the
        user is not ready to answer can be shelved without losing it). On
        resume the task returns to AWAITING_CLARIFICATION if its question is
        still unanswered, otherwise to the queue — and if the user answered
        while the task was held, the answer counts as an answered
        clarification (head of the queue), mirroring send()'s transition."""
        task = self.get(task_id)
        if on and task.status in (
            TaskStatus.QUEUED,
            TaskStatus.AWAITING_CLARIFICATION,
        ):
            task.status = TaskStatus.HOLD
        elif not on and task.status is TaskStatus.HOLD:
            if task.clarification is not None:
                if task.pending_input:
                    # Answered while held: re-queue at the head, like an
                    # ordinary answered clarification.
                    task.status = TaskStatus.QUEUED
                    task.queue_head = True
                    task.clarification = None
                else:
                    # The question is still open: back to awaiting an answer.
                    task.status = TaskStatus.AWAITING_CLARIFICATION
            else:
                task.status = TaskStatus.QUEUED
        task.touch()
        self.storage.save(task)
        return task

    def mark_done(self, task_id: str) -> Task:
        """Development is finished: move the task to TESTING. Only the tester
        promotes it to COMPLETED (via set_verification). Promotion of the next
        task is up to the caller (the app runs the tester first, then
        promotes). If the app dies while TESTING, the task is re-tested on
        the next startup before any queued task starts.

        Documentation tasks skip the tester entirely: they are completed
        right away."""
        task = self.get(task_id)
        task.status = (
            TaskStatus.COMPLETED
            if task.kind is TaskKind.DOCUMENTATION
            else TaskStatus.TESTING
        )
        if task.status is TaskStatus.COMPLETED:
            task.completed_at = _now()
        task.touch()
        self.storage.save(task)
        return task

    def promote_next(self, exclude: set[str] | None = None) -> Task | None:
        """If no task is active, activate the first queued task that has
        pending input or an interrupted attempt to resume.
        `exclude` skips tasks that already failed this session (bounded
        retries: a failing task must not stall the whole queue in a loop)."""
        if self.active is not None:
            return None
        for t in self.queued():
            if (t.pending_input or t.unfinished) and (
                exclude is None or t.id not in exclude
            ):
                t.status = TaskStatus.ACTIVE
                t.queue_head = False
                t.touch()
                self.storage.save(t)
                return t
        return None

    def complete_without_verification(self, task_id: str) -> Task:
        """Finish the task without a tester round (project setting
        "Task finalization: None"): a task that reached TESTING is completed
        right away, with no verification label."""
        task = self.get(task_id)
        task.status = TaskStatus.COMPLETED
        task.completed_at = _now()
        task.unfinished = False
        task.touch()
        self.storage.save(task)
        return task

    def cancel(self, task_id: str) -> Task:
        """User cancelled the task (not relevant anymore): drop all queued
        input, clear the resume flag and complete it immediately — the
        tester is skipped everywhere a finished task would run it."""
        task = self.get(task_id)
        task.pending_input.clear()
        task.unfinished = False
        task.clarification = None
        task.queue_head = False
        task.status = TaskStatus.COMPLETED
        task.completed_at = _now()
        task.touch()
        self.storage.save(task)
        return task

    def requeue_answered_clarification(self, task_id: str) -> None:
        """A clarifying question was asked, but the user's answer already sits
        in pending_input (it was typed while the agent run was still finishing,
        so send()'s clarification transition never fired). Re-queue the task at
        the head instead of waiting forever for an answer that has arrived."""
        task = self.get(task_id)
        if task.status is not TaskStatus.AWAITING_CLARIFICATION:
            return
        if not task.pending_input:
            return
        task.status = TaskStatus.QUEUED
        task.queue_head = True
        task.clarification = None
        task.touch()
        self.storage.save(task)
