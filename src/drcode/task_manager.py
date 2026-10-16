"""Task state machine: creation, queue, activation, re-activation."""

from __future__ import annotations

from pathlib import Path

from .models import Task, TaskKind, TaskStatus
from .storage import Storage


class TaskManager:
    def __init__(self, project_root: Path) -> None:
        self.storage = Storage(project_root)
        self.tasks: dict[str, Task] = {t.id: t for t in self.storage.load_all()}
        # On startup at most one task may have been active when the app closed.
        for t in self.tasks.values():
            if t.status is TaskStatus.ACTIVE:
                t.status = TaskStatus.QUEUED
                self.storage.save(t)

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
        # Answered clarifications jump the queue first, then bug reports
        # (both oldest first), then features.
        tasks.sort(
            key=lambda t: (not t.queue_head, t.kind is not TaskKind.BUG, t.created_at)
        )
        return tasks

    # -- mutations ------------------------------------------------------------

    def create(
        self, title: str, first_message: str, kind: TaskKind = TaskKind.FEATURE
    ) -> Task:
        task = Task(
            title=title or first_message.splitlines()[0][:60] or "New task",
            kind=kind,
        )
        if first_message.strip():
            task.pending_input.append(first_message.strip())
        self.tasks[task.id] = task
        self.storage.save(task)
        return task

    def send(self, task_id: str, text: str) -> Task:
        """Add user input to a task. A completed (or being-tested) task
        re-enters the pipeline; an answered clarification goes to the head
        of the queue."""
        task = self.get(task_id)
        task.pending_input.append(text.strip())
        if task.status in (TaskStatus.COMPLETED, TaskStatus.TESTING):
            task.status = TaskStatus.QUEUED
            # Old verification label no longer applies: the task will be
            # tested again after its next development round.
            task.verification = None
            task.touch()
            self.storage.save(task)
            self.promote_next()
        elif task.status is TaskStatus.AWAITING_CLARIFICATION:
            task.status = TaskStatus.QUEUED
            task.queue_head = True
            task.clarification = None
            task.touch()
            self.storage.save(task)
            self.promote_next()
        else:
            # QUEUED/ACTIVE: just accumulate input (also when the model is not
            # configured — the agent picks it up on the next activation).
            task.touch()
            self.storage.save(task)
        return task

    def mark_done(self, task_id: str) -> Task:
        """Development is finished: move the task to TESTING. Only the tester
        promotes it to COMPLETED (via set_verification). Promotion of the next
        task is up to the caller (the app runs the tester first, then
        promotes). If the app dies while TESTING, the task is re-tested on
        the next startup before any queued task starts."""
        task = self.get(task_id)
        task.status = TaskStatus.TESTING
        task.touch()
        self.storage.save(task)
        return task

    def promote_next(self) -> Task | None:
        """If no task is active, activate the first queued task that has input."""
        if self.active is not None:
            return None
        for t in self.queued():
            if t.pending_input:
                t.status = TaskStatus.ACTIVE
                t.queue_head = False
                t.touch()
                self.storage.save(t)
                return t
        return None

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
