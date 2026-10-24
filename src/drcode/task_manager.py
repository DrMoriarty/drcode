"""Task state machine: creation, queue, activation, re-activation."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from datetime import timedelta
from pathlib import Path

from .models import (
    PendingInput,
    Task,
    TaskEvent,
    TaskKind,
    TaskStatus,
    VerificationStatus,
    _now,
)
from .storage import Storage
from .tools import remove_scratch
from .transcript import is_pending_retry
from .workflow import ensure_workflow_examples

# Status-change observer: (task, old status, new status, origin). Origin is
# "user" when the transition was directly initiated by the user sitting at
# the computer (typing a message, cancelling, holding, pressing Done) and
# "system" when the pipeline changed it on its own (auto-completion,
# promotion, the tester's verdict, an agent error). Listeners must never
# raise: a broken observer must not break the state machine.
StatusListener = Callable[[Task, TaskStatus, TaskStatus, str], None]

# Removal observer: (task id) — fired AFTER the task is gone (cancelled or
# deleted, including the retention auto-cleanup), so per-task resources
# (the optional worktree, see drcode.task_env) can be torn down. Sync,
# like the status listeners: must be fast and non-raising.
RemovalListener = Callable[[str], None]


def validate_dependencies(
    tasks: Mapping[str, Task], task_id: str | None, deps: Iterable[str]
) -> list[str]:
    """Normalize and validate a `depends_on` list for `task_id`.

    Strips whitespace, drops empties and duplicates; every id must exist in
    `tasks`, and the task must not depend on itself. When `task_id` is not
    None (i.e. the task already exists — creation-time validation passes
    None, self-reference is impossible then), the new edges are also
    checked for cycles over the full depends_on graph.

    Returns the cleaned id list; raises ValueError on any violation —
    callers (the new-task form, the command palette) surface the message.
    Deliberately tolerant at LOAD time: a bad id in a stored task file must
    never break reading it, validation happens on write only.
    """
    cleaned: list[str] = []
    for raw in deps:
        dep = raw.strip()
        if not dep:
            continue
        if dep == task_id:
            raise ValueError("a task cannot depend on itself")
        if dep not in tasks:
            raise ValueError(f"unknown task id: {dep}")
        if dep not in cleaned:
            cleaned.append(dep)
    if task_id is not None and cleaned:
        graph = {tid: list(t.depends_on) for tid, t in tasks.items()}
        graph[task_id] = cleaned
        # A cycle can only involve the edge(s) just added (existing lists
        # were validated the same way on write): the new dependency chain
        # must not lead back to the task itself.
        seen: set[str] = set()
        stack = list(cleaned)
        while stack:
            node = stack.pop()
            if node == task_id:
                raise ValueError("dependency cycle detected")
            if node in seen:
                continue
            seen.add(node)
            stack.extend(graph.get(node, ()))
    return cleaned


class TaskManager:
    def __init__(self, project_root: Path) -> None:
        self.storage = Storage(project_root)
        # Scaffold the workflow templates (.drcode/workflow.py.example,
        # .drcode/git-workflow.md.example) — idempotent, never overwrites.
        ensure_workflow_examples(project_root)
        self._status_listeners: list[StatusListener] = []
        # Observers of task removal (cancel/delete/purge): the optional
        # worktree environment cleans its per-task resources here.
        self._removal_listeners: list[RemovalListener] = []
        # Observers of task creation (create()) — e.g. the voice
        # announcements hook for tester-created bug reports.
        self._creation_listeners: list[Callable[[Task, str], None]] = []
        # Serializes reload_new_tasks(): it runs both on the app thread
        # (the periodic disk poll) and on worker threads (promote_next in
        # the run's finally-blocks) — two concurrent scans must not load
        # the same new file twice.
        self._disk_lock = threading.Lock()
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

    # -- status-change observers ---------------------------------------------

    def add_status_listener(self, listener: StatusListener) -> None:
        """Register an observer of status transitions (used by the
        notifications feature). Listeners are called synchronously — possibly
        from a worker thread (the tester's tools run via asyncio.to_thread) —
        and must be fast and non-raising."""
        self._status_listeners.append(listener)

    def add_removal_listener(self, listener: RemovalListener) -> None:
        """Register an observer of task removal (cancel/delete/purge) —
        the worktree environment tears the task's branch and tree down
        here. Same contract as the status listeners: sync, fast,
        non-raising."""
        self._removal_listeners.append(listener)

    def add_creation_listener(self, listener: Callable[[Task, str], None]) -> None:
        """Register an observer of task creation (create()): the listener
        gets the new task and the origin ("user" — typed by the person at
        the computer, "system" — created by an agent tool, e.g. the
        tester's bug report). Same contract as the status listeners: sync,
        fast, non-raising."""
        self._creation_listeners.append(listener)

    def _fire_removed(self, task_id: str) -> None:
        for listener in list(self._removal_listeners):
            try:
                listener(task_id)
            except Exception as e:  # noqa: BLE001 - an observer must not break the pipeline
                print(
                    f"drcode: removal listener failed: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )

    def fire_status_change(self, task: Task, old: TaskStatus, origin: str) -> None:
        """Public hook for status mutations that happen outside the manager
        (e.g. the pipeline demoting a failed run's task directly). Call it
        AFTER the new status has been assigned to `task`."""
        if task.status is old:
            return
        for listener in list(self._status_listeners):
            try:
                listener(task, old, task.status, origin)
            except Exception as e:  # noqa: BLE001 - an observer must not break the pipeline
                print(
                    f"drcode: status listener failed: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )

    def _save_changed(self, task: Task, old: TaskStatus, origin: str) -> None:
        """Persist a status mutation and notify the observers."""
        self.storage.save(task)
        self.fire_status_change(task, old, origin)

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

    def active_tasks(self) -> list[Task]:
        """ALL currently ACTIVE tasks (parallel development: under
        worktree isolation with `parallel_tasks > 1` more than one task
        can be in development at once). Creation order, like all_tasks."""
        return [t for t in self.all_tasks() if t.status is TaskStatus.ACTIVE]

    def queued(self) -> list[Task]:
        tasks = [t for t in self.all_tasks() if t.status is TaskStatus.QUEUED]

        # Bug reports always come first — newest at the very head, so a bug
        # report the tester has just created is picked up immediately after
        # the testing round finishes. Everything else follows: answered
        # clarifications (queue_head) before the rest, then the explicit
        # queue stamp (creation order, bumped when a task is re-positioned,
        # e.g. by a kind change). A manual move (move_in_queue) re-stamps
        # the whole queue so this same key reproduces the user's order.
        # Top tier: tasks blocked by dependencies (STATUS part only — a dep
        # that exists and is not COMPLETED yet) sink below every runnable
        # one; the git mergedness half of the gate is checked by the
        # pipeline's promotion worker, not here (no git calls in a sort).
        def _queue_key(t: Task) -> tuple:
            blocked = 1 if self._status_blocked(t) else 0
            if t.kind is TaskKind.BUG:
                return (blocked, 0, 0, -(t.queue_seq if t.queue_seq is not None else 0))
            return (
                blocked,
                1,
                not t.queue_head,
                t.queue_seq if t.queue_seq is not None else 0,
            )

        tasks.sort(key=_queue_key)
        return tasks

    def _status_blocked(self, task: Task) -> bool:
        """The STATUS half of the dependency gate: some dependency exists
        and is not COMPLETED yet. Pure in-memory check (no git): the
        mergedness half lives in drcode.task_env / the pipeline."""
        return any(
            (dep_task := self.tasks.get(dep)) is not None
            and dep_task.status is not TaskStatus.COMPLETED
            for dep in task.depends_on
        )

    def reload_new_tasks(self) -> list[Task]:
        """Pick up task files that appeared on disk since the last check.

        Another process (`drcode --feature-request DESC` in a second
        terminal, a one-shot GitHub sync) writes tasks straight into
        `<project>/.drcode/tasks/` — the running app used to notice them
        only after a restart. Called whenever the pipeline searches the
        queue for the next task to promote (promote_next) and by the
        app's periodic disk poll, so external tasks join the backlog
        without a restart.

        Loaded tasks are stamped into the queue like fresh creations
        (queue_seq in created_at order) and a stale ACTIVE status — left
        by an instance that died or a crashed peer — is demoted to QUEUED,
        mirroring the startup rule (a local idle pipeline must not wait
        for a foreign active task that will never finish). Files already
        known are not re-read, so in-memory state (transcripts consumed
        mid-run) is never overwritten from disk. Returns the tasks that
        were added.
        """
        with self._disk_lock:
            known = set(self.tasks)
            new: list[Task] = []
            for path in sorted(self.storage.tasks_dir.glob("*.json")):
                if path.stem in known:
                    continue
                task = self.storage.load_one(path)
                if task is None:
                    continue
                known.add(task.id)
                new.append(task)
            ordered = sorted(new, key=lambda t: t.created_at)
            for t in ordered:
                changed = False
                if t.queue_seq is None:
                    t.queue_seq = self._next_queue_seq()
                    changed = True
                if t.status is TaskStatus.ACTIVE:
                    t.status = TaskStatus.QUEUED
                    changed = True
                self.tasks[t.id] = t
                if changed:
                    self.storage.save(t)
            return ordered

    # -- mutations ------------------------------------------------------------

    def create(
        self,
        title: str,
        first_message: str,
        kind: TaskKind = TaskKind.FEATURE,
        *,
        hold: bool = False,
        origin: str = "user",
        depends_on: list[str] | None = None,
    ) -> Task:
        # Creation-time validation passes task_id=None (self-reference is
        # impossible — the id does not exist yet, so a dep naming it comes
        # back as "unknown task id"), only the existence of the ids needs
        # checking. Raises ValueError BEFORE anything is stored or any
        # listener fires — the caller surfaces the message.
        cleaned_deps = validate_dependencies(self.tasks, None, depends_on or [])
        first_line = (
            first_message.splitlines()[0][:60] if first_message.splitlines() else ""
        )
        task = Task(
            title=title or first_line or "New task",
            kind=kind,
            queue_seq=self._next_queue_seq(),
        )
        task.depends_on = cleaned_deps
        if first_message.strip():
            task.pending_input.append(PendingInput(text=first_message.strip()))
        if hold:
            # Parked right away: the prompt stays in pending_input (the same
            # state a queued-then-held task is in) and the pipeline never
            # picks the task up until it is resumed. The status is assigned
            # directly — no observers yet, and a "created → held" toast
            # would only be noise.
            task.status = TaskStatus.HOLD
        self.tasks[task.id] = task
        self.storage.save(task)
        for listener in list(self._creation_listeners):
            try:
                listener(task, origin)
            except Exception as e:  # noqa: BLE001 - an observer must not break the pipeline
                print(
                    f"drcode: creation listener failed: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
        return task

    def editable_condition(self, task_id: str) -> str:
        """The task's editable condition: every user text still queued in
        `pending_input` (the original prompt plus each additional comment)
        merged into one block, separated by blank lines.

        Texts already consumed into the transcript (a started attempt) are
        not part of it — they cannot be edited without breaking the message
        history."""
        task = self.get(task_id)
        return "\n\n".join(entry.text for entry in task.pending_input).strip()

    def edit_task(self, task_id: str, text: str, kind: TaskKind | None = None) -> Task:
        """Rewrite the condition of a held task.

        The merged editable text (see editable_condition) is replaced by a
        single new pending input — the previously separate additional
        comments are gone, their content now lives in the main text. The
        title is re-derived from the first line, and the kind can be changed
        like on creation (bug → head of the queue, feature → tail). An open
        clarifying question is dropped together with the texts it belonged
        to: the rewritten condition supersedes it.

        Empty text is a no-op (the caller keeps the form open instead).
        """
        task = self.get(task_id)
        stripped = text.strip()
        if not stripped:
            return task
        task.pending_input.clear()
        task.pending_input.append(PendingInput(text=stripped))
        task.clarification = None
        task.title = stripped.splitlines()[0][:60]
        # Reuses the queue repositioning rules of set_kind (which does
        # not check the status: held tasks are not started either).
        if kind is not None and task.kind is not kind:
            self.set_kind(task_id, kind)
        task.touch()
        self.storage.save(task)
        return task

    def send(self, task_id: str, text: str) -> Task:
        """Add user input to a task (appended chronologically). A completed
        (or being-tested) task re-enters the pipeline; an answered
        clarification goes to the head of the queue. Promotion (activation)
        is up to the caller: the app advances the pipeline through
        _advance(), which never starts new development while any testing is
        incomplete.

        A re-send of a text that is still queued in `pending_input` — or
        already sitting in the transcript as the prompt of an unfinished
        attempt — is ignored: the user re-typing a comment that the busy
        pipeline has not visibly reacted to yet must not duplicate the
        message (it would show twice in the chat history and reach the
        model twice)."""
        task = self.get(task_id)
        stripped = text.strip()
        if any(e.text == stripped for e in task.pending_input) or is_pending_retry(
            task.messages, stripped
        ):
            task.touch()
            self.storage.save(task)
            return task
        task.pending_input.append(PendingInput(text=stripped))
        if task.status in (TaskStatus.COMPLETED, TaskStatus.TESTING):
            old = task.status
            task.status = TaskStatus.QUEUED
            # Old verification label no longer applies: the task will be
            # tested again after its next development round.
            task.verification = None
            # Fresh pipeline run: the completion timestamp (auto-cleanup
            # anchor) restarts with the eventual new completion.
            task.completed_at = None
            # Any pending asynchronous review verdict is moot now: the user
            # took the task back into development by hand. The consecutive
            # rejection count restarts too — human intervention breaks the
            # review ping-pong loop.
            task.review_pending = False
            task.review_rejections = 0
            # Human intervention breaks the fix loop too: the consecutive
            # failed development rounds restart.
            task.fix_rounds = 0
            task.touch()
            self._save_changed(task, old, "user")
        elif task.status is TaskStatus.AWAITING_CLARIFICATION:
            task.status = TaskStatus.QUEUED
            task.queue_head = True
            task.clarification = None
            task.touch()
            self._save_changed(task, TaskStatus.AWAITING_CLARIFICATION, "user")
        else:
            # QUEUED/ACTIVE: just accumulate input (also when the model is not
            # configured — the agent picks it up on the next activation).
            task.touch()
            self.storage.save(task)
        return task

    def set_kind(self, task_id: str, kind: TaskKind) -> Task:
        """Change the kind of a not-yet-started (QUEUED or HOLD) task and
        re-position it in the queue: bug → head of the queue, feature →
        tail. Other kinds (research) keep the current position. The caller
        is responsible for only calling this on queued/held tasks."""
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

    def set_dependencies(self, task_id: str, deps: list[str]) -> Task:
        """Replace the task's `depends_on` list (the promotion gate "all
        dependencies COMPLETED and merged into the base branch").

        Validation (see validate_dependencies): no self-reference, every id
        must exist, no cycles — a violation raises ValueError for the
        caller to show. Git-agnostic on purpose: mergedness is checked
        dynamically by the pipeline (task_env.unmerged_ids) at every
        promotion attempt, never stored."""
        task = self.get(task_id)
        task.depends_on = validate_dependencies(self.tasks, task_id, deps)
        task.touch()
        self.storage.save(task)
        return task

    def blocking_dependencies(self, task_id: str) -> list[str]:
        """Dependency ids whose STATUS still blocks the task: the dep task
        exists and is not COMPLETED yet. Missing dep ids (deleted task /
        retention purge) are NOT reported here — their gate is the git
        branch, checked by the pipeline via task_env.unmerged_ids."""
        task = self.get(task_id)
        return [
            dep
            for dep in task.depends_on
            if (dep_task := self.tasks.get(dep)) is not None
            and dep_task.status is not TaskStatus.COMPLETED
        ]

    def _apply_manual_order(self, order: list[Task]) -> None:
        """Persist a manually adjusted queue order (index 0 = next to run)
        into the queue stamps, so the standard sort key in queued()
        reproduces it — including after an app restart.

        The stamps are reassigned above the current counter (no collisions
        with stamps held by non-queued tasks): non-bug tasks ascending along
        the order (tasks created later still append at the tail), bugs
        descending (a freshly created bug still lands at the very head of
        the bug block). The queue_head markers are folded into the stamps —
        after a manual move the explicit positions are the source of truth,
        while a future answered clarification (queue_head=True + a fresh
        stamp) still jumps ahead of the non-bug block.
        """
        bugs = [t for t in order if t.kind is TaskKind.BUG]
        rest = [t for t in order if t.kind is not TaskKind.BUG]
        base = self._queue_seq
        for i, t in enumerate(bugs):
            # Descending: the closer to execution, the larger the stamp.
            t.queue_seq = base + len(bugs) - i
        seq = base + len(bugs)
        for t in rest:
            seq += 1
            t.queue_seq = seq
            t.queue_head = False
        for t in bugs:
            t.queue_head = False
        self._queue_seq = seq
        for t in order:
            self.storage.save(t)

    def move_in_queue(self, task_id: str, *, up: bool) -> Task | None:
        """Manually move a QUEUED task one step in the backlog.

        The sidebar runs top → bottom from the farthest task to the one the
        pipeline picks next; `up=True` moves the task one step away from
        execution (up the list), `up=False` — one step closer (down the
        list). The new position is persisted (see _apply_manual_order) and
        survives restarts; tasks created afterwards still enter by the
        default algorithms (bugs at the very head, the rest at the tail).

        Returns the moved task, or None when the move is impossible: the
        task is not queued, it is already at the corresponding edge of the
        queue, or the swap would cross the bug block (bug reports always
        run first — hold a bug with set_hold if it must wait).
        """
        task = self.tasks.get(task_id)
        if task is None or task.status is not TaskStatus.QUEUED:
            return None
        order = self.queued()
        i = order.index(task)
        j = i + 1 if up else i - 1
        if not 0 <= j < len(order):
            return None
        if (order[j].kind is TaskKind.BUG) != (task.kind is TaskKind.BUG):
            return None
        order[i], order[j] = order[j], order[i]
        task.touch()
        self._apply_manual_order(order)
        return task

    def delete(self, task_id: str) -> None:
        """Remove the task entirely: from memory and from disk storage.
        Intended for parked (held) tasks; the caller decides which statuses
        may be deleted."""
        self.tasks.pop(task_id, None)
        self.storage.delete(task_id)
        remove_scratch(task_id)
        self._fire_removed(task_id)

    def purge_expired_completed(self, retention_days: int | None) -> list[Task]:
        """Delete COMPLETED tasks older than the retention period.

        Runs once at app startup. `retention_days` None (setting "never")
        disables the cleanup. Age anchor: `completed_at` (set on every
        transition to COMPLETED); for tasks completed by an older build
        without the timestamp, `updated_at` is used as a fallback. Tasks
        still awaiting an asynchronous external review verdict
        (`review_pending` — a COMPLETED task with an open PR) are skipped:
        a verdict arriving after N days must not resurrect a deleted task.
        Returns the removed tasks so the caller can report the cleanup.
        """
        if retention_days is None:
            return []
        cutoff = _now() - timedelta(days=retention_days)
        expired = [
            t
            for t in self.tasks.values()
            if t.status is TaskStatus.COMPLETED
            and not t.review_pending
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
        old = task.status
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
        self._save_changed(task, old, "user")
        return task

    def mark_done(self, task_id: str, origin: str = "system") -> Task:
        """Development is finished: move the task to TESTING. Only the tester
        promotes it to COMPLETED (via set_verification). Promotion of the next
        task is up to the caller (the app runs the tester first, then
        promotes). If the app dies while TESTING, the task is re-tested on
        the next startup before any queued task starts.

        Research tasks skip the tester entirely: they are completed
        right away.

        `origin` tells observers who initiated the transition: "user"
        (the Ctrl+D shortcut) or "system" (the automatic end-of-run
        completion in the pipeline)."""
        task = self.get(task_id)
        old = task.status
        task.status = (
            TaskStatus.COMPLETED
            if task.kind is TaskKind.RESEARCH
            else TaskStatus.TESTING
        )
        if task.status is TaskStatus.COMPLETED:
            task.completed_at = _now()
            remove_scratch(task_id)
        task.touch()
        self._save_changed(task, old, origin)
        return task

    def promote_next(
        self,
        exclude: set[str] | None = None,
        blocked: set[str] | None = None,
        *,
        max_active: int = 1,
    ) -> Task | None:
        """If a development slot is free (fewer than `max_active` tasks
        are ACTIVE — parallelism lives with the pipeline's promotion loop,
        this check only keeps the count honest), activate the first queued
        task that has pending input or an interrupted attempt to resume.
        `exclude` skips tasks that already failed this session (bounded
        retries: a failing task must not stall the whole queue in a loop).
        `blocked` skips tasks whose dependencies are not satisfied yet
        (the pipeline's dynamic gate: not COMPLETED, or work not merged
        into the base branch — re-evaluated by the caller at every
        attempt, exactly like the hold skip: a manual merge unblocks the
        dependents without any drcode action)."""
        # New tasks may have appeared on disk since the last look (created
        # by another process, e.g. drcode --feature-request): pick them up
        # before searching the queue, so the promotion acts on the full
        # backlog, not just on what this instance loaded earlier.
        self.reload_new_tasks()
        if len(self.active_tasks()) >= max_active:
            return None
        for t in self.queued():
            if (
                (t.pending_input or t.unfinished)
                and (exclude is None or t.id not in exclude)
                and (blocked is None or t.id not in blocked)
            ):
                t.status = TaskStatus.ACTIVE
                t.queue_head = False
                # Stamp the activation moment: the expanded-header session
                # statistics measure activation → completion from it.
                t.activated_at = _now()
                t.touch()
                self._save_changed(t, TaskStatus.QUEUED, "system")
                return t
        return None

    def complete_without_verification(self, task_id: str) -> Task:
        """Finish the task without a tester round (project setting
        "Task finalization: None"): a task that reached TESTING is completed
        right away, with no verification label."""
        task = self.get(task_id)
        old = task.status
        task.status = TaskStatus.COMPLETED
        task.completed_at = _now()
        task.unfinished = False
        remove_scratch(task_id)
        task.touch()
        self._save_changed(task, old, "system")
        return task

    def cancel(self, task_id: str) -> Task:
        """User cancelled the task (not relevant anymore): drop all queued
        input, clear the resume flag and complete it immediately — the
        tester is skipped everywhere a finished task would run it.

        Unlike an ordinary completion, the task's worktree environment is
        torn down here (a cancelled task's work is not wanted): the
        removal listeners run after the transition."""
        task = self.get(task_id)
        old = task.status
        task.pending_input.clear()
        task.unfinished = False
        task.clarification = None
        task.queue_head = False
        # Cancelling from TESTING (or a re-queued round): a stale passed/
        # failed label from an earlier testing round no longer applies —
        # the task completes without verification.
        task.verification = None
        task.status = TaskStatus.COMPLETED
        task.completed_at = _now()
        task.review_pending = False
        task.review_rejections = 0
        remove_scratch(task_id)
        self._fire_removed(task_id)
        task.touch()
        self._save_changed(task, old, "user")
        return task

    def set_verification(
        self, task_id: str, success: bool, *, review_gate: bool = False
    ) -> Task:
        """The tester's verdict: stamp the verification label and complete
        the task. Used by the tester's `set_verification` tool (which runs
        in a worker thread — see tool_runtime) so the transition goes
        through the same observer mechanism as every other status change.

        With ``review_gate=True`` (the project defines an external review
        in `.drcode/workflow.py`) a PASSED verdict does NOT complete the
        task: "the tester is done" and "the task is finally accepted" are
        split, and the task stays TESTING until the external review's
        verdict arrives (accept → finalize_verified, reject →
        requeue_review_rejection). A FAILED verdict behaves as without the
        gate: the task completes with the failure label.

        A no-op when the task is no longer TESTING: while the tool call was
        in flight the user may have cancelled the task (Ctrl+X) or sent new
        input to it (which re-queues it) — a stale verdict must neither
        complete the cancelled task nor swallow the re-queued one."""
        task = self.get(task_id)
        if task.status is not TaskStatus.TESTING:
            return task
        old = task.status
        task.verification = (
            VerificationStatus.PASSED if success else VerificationStatus.FAILED
        )
        if success:
            # A passed testing round ends the fix loop: the consecutive
            # failed-development-rounds counter starts over.
            task.fix_rounds = 0
        if success and review_gate:
            # Passed, but the final completion waits for the external
            # review: the TESTING status keeps the pipeline blocked (no new
            # development starts while a verdict is outstanding), exactly
            # like an in-flight tester round.
            task.touch()
            self.storage.save(task)
            return task
        task.status = TaskStatus.COMPLETED
        task.completed_at = _now()
        remove_scratch(task_id)
        task.touch()
        self._save_changed(task, old, "system")
        return task

    def finalize_verified(self, task_id: str, origin: str = "system") -> Task:
        """The external review accepted the task (accept verdict): finish
        what set_verification(review_gate=True) held back — the task leaves
        TESTING as COMPLETED with its PASSED label intact. The consecutive
        review-rejection count resets: the loop is over."""
        task = self.get(task_id)
        if task.status is not TaskStatus.TESTING:
            return task
        old = task.status
        task.status = TaskStatus.COMPLETED
        task.completed_at = _now()
        task.unfinished = False
        task.review_rejections = 0
        task.fix_rounds = 0
        remove_scratch(task_id)
        task.touch()
        self._save_changed(task, old, origin)
        return task

    def requeue_failed_fix(self, task_id: str, report: str, *, max_rounds: int) -> Task:
        """The tester failed the task but attributed the failure to the
        task's OWN changes (the ``retry_here=True`` verdict of its
        `set_verification` tool): instead of completing the task with a bug
        report, send it back to development — the tester's failure report
        becomes the task's next input and the coding agent fixes the bug in
        the same transcript and the same (worktree) environment where the
        work happened. With per-task worktree isolation `ensure_root`
        reuses the existing worktree untouched, so the fix round sees the
        full context of its previous work (a foreign bug report, in
        contrast, branches from clean HEAD).

        Mirrors requeue_review_rejection. The consecutive fix-round count
        grows (`task.fix_rounds`); once `max_rounds` development rounds have
        failed in a row (project setting `max_dev_rounds`), the failure
        completes the task as FAILED instead of re-queueing it — the bug
        survived its dedicated fix rounds, so the tester's bug report
        legitimately branches from clean HEAD. A no-op when the task is no
        longer TESTING (the user re-queued or cancelled it while the verdict
        was in flight)."""
        task = self.get(task_id)
        if task.status is not TaskStatus.TESTING:
            return task
        rounds_limit = max(1, max_rounds)
        if task.fix_rounds + 1 >= rounds_limit:
            # Cap reached: an ordinary FAILED completion — the tester
            # creates the bug report separately.
            completed = self.set_verification(task_id, False)
            completed.events.append(
                TaskEvent(
                    kind="testing",
                    text=(
                        f"testing failed {task.fix_rounds + 1} development "
                        f"rounds in a row — the fix-round cap "
                        f"({rounds_limit}) is reached; the task completed "
                        "as FAILED."
                    ),
                )
            )
            completed.touch()
            self.storage.save(completed)
            return completed
        stripped = report.strip()
        if stripped:
            task.pending_input.append(PendingInput(text=stripped))
        old = task.status
        task.status = TaskStatus.QUEUED
        task.verification = None
        task.completed_at = None
        task.fix_rounds += 1
        task.events.append(
            TaskEvent(
                kind="testing",
                text=(
                    "testing failed — returned for another development "
                    f"round (round {task.fix_rounds + 1}/{rounds_limit})"
                ),
            )
        )
        task.touch()
        self._save_changed(task, old, "system")
        return task

    def requeue_review_rejection(self, task_id: str, comment: str) -> Task:
        """The external review rejected the task: send the reviewer's
        comment to it as new user input and send it back to development —
        the same path an ordinary user message to a TESTING/COMPLETED task
        takes (send()'s re-queue branch), so the coding agent receives the
        remarks in its existing transcript on the next promotion.

        The consecutive rejection count grows (the pipeline parks the task
        on hold after MAX_REVIEW_REJECTIONS in a row); a no-op when the
        task is no longer TESTING/COMPLETED (the user got there first)."""
        task = self.get(task_id)
        if task.status not in (TaskStatus.TESTING, TaskStatus.COMPLETED):
            return task
        stripped = comment.strip()
        if stripped:
            task.pending_input.append(PendingInput(text=stripped))
        old = task.status
        task.status = TaskStatus.QUEUED
        task.verification = None
        task.completed_at = None
        task.review_pending = False
        task.review_rejections += 1
        task.touch()
        self._save_changed(task, old, "system")
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
        self._save_changed(task, TaskStatus.AWAITING_CLARIFICATION, "system")
