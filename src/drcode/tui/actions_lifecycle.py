"""Task-lifecycle actions layer — user commands over tasks.

The keyboard/palette commands that change what the pipeline does with a
task: done, cancel, the ESC run-stop (with the double-press arming
machinery), the stall nudge, the kind cycle, queue reordering,
dependencies, hold/resume/delete — plus submitting user input to a task
(chat editor, clarification options, task-id mentions) and the sidebar
selection handler.

Split out of :mod:`drcode.tui.actions` — see the composition in
``ActionsLayer``.
"""

from __future__ import annotations

import time
from typing import ClassVar

from textual.widgets import ListView

from ..agent import (
    is_pending_retry,
)
from ..models import (
    Task,
    TaskKind,
    TaskStatus,
)
from ..settings import (
    EscAbort,
)
from .sidebar import TaskItem
from .state import DrCodeState


class TaskLifecycleLayer(DrCodeState):
    # Armed-ESC confirmation (project setting `esc_abort: double_press`):
    # how long the armed state waits for the confirming second press, and
    # the minimum gap between two ESC presses for the second one to count
    # as deliberate (a faster repeat is an accidental double-keystroke).
    _ESC_ARM_SECONDS: ClassVar[float] = 2.0
    _ESC_DEBOUNCE_SECONDS: ClassVar[float] = 0.25

    def action_done_task(self) -> None:
        task = self.viewed
        if task is None or task.status is TaskStatus.COMPLETED:
            return
        # Stop only the VIEWED task's agent run (parallel slots: other
        # tasks' runs must keep going).
        if task.id in self._agent_runs:
            self.workers.cancel_group(self, f"agent-{task.id}")
            self._agent_runs.pop(task.id, None)
            self._current_tool[task.id] = None
        # The user pressed Done right here: the status observers
        # (notifications) must not report this transition.
        # Completion soft gate (same as the pipeline's automatic path):
        # unchecked checklist items are announced, not blocking.
        self._todo_completion_warning(task)
        self.manager.mark_done(task.id, origin="user")
        self.refresh_sidebar()
        self.view_task(task.id)
        if task.kind is TaskKind.RESEARCH and task.status is TaskStatus.COMPLETED:
            # Research done by hand (Ctrl+D): same post-completion steps as
            # the automatic end-of-run completion — snapshot, then the
            # on_research_accepted hook (or the unmerged-work reminder).
            # A worker: the hook subprocess runs up to HOOK_TIMEOUT.
            self.run_worker(
                self._finalize_research_completion(task),
                exclusive=True,
                group="research-accept",
            )
        # Testing backlog first (possibly an older TESTING task), then queue.
        self._advance()

    def action_cancel_task(self) -> None:
        """User cancels the viewed task (not relevant anymore): drop all
        queued input and complete it immediately — the tester never runs."""
        task = self.viewed
        if task is None or task.status is TaskStatus.COMPLETED:
            return
        # Stop only the target task's runs (parallel slots: other tasks'
        # agent runs are independent and keep running).
        if task.id in self._agent_runs:
            self.workers.cancel_group(self, f"agent-{task.id}")
            self._agent_runs.pop(task.id, None)
            self._current_tool[task.id] = None
        if self.tester_running and self._tester_task_id == task.id:
            self.workers.cancel_group(self, "tester")
            self.tester_running = False
            self._current_tool[task.id] = None
        # A queued comment (if any) leaves pending_input with the cancel:
        # its rendered "user-pending" block must come off the screen, which
        # only a full re-render does. Captured before cancel() clears it.
        had_pending = bool(task.pending_input)
        # A running agent may have already consumed the queued prompts
        # (popped from pending_input, not yet persisted into the transcript):
        # cancel() is about to clear pending_input and complete the task,
        # and the cancelled run's own persist skips a no-longer-ACTIVE task —
        # flush the in-flight copy into the transcript, or the user's brief
        # would disappear from the chat history.
        self._flush_in_flight_prompt(task)
        self.manager.cancel(task.id)
        self._tester_resume_round.discard(task.id)
        # The task was possibly in TESTING: drop its tester pipeline state.
        # With a live tester run this is done by the cancelled worker's
        # `finally` (idempotent); without one (retry backoff, failed round)
        # nobody else would clear it — and the stale backoff/retry entries
        # and the stale `_tester_task_id` would linger until restart.
        self._tester_fail_count.pop(task.id, None)
        self._tester_retry_at.pop(task.id, None)
        self._tester_net_count.pop(task.id, None)
        if self._tester_task_id == task.id and not self.tester_running:
            self._tester_task_id = None
        # Сервисное событие: "Задача отменена пользователем — снята с конвейера без тестирования".
        self._show_event(
            task,
            "Task cancelled by the user — removed from the pipeline without testing.",
        )
        self.refresh_sidebar()
        self.refresh_status()
        # No full re-render when nothing but the event changed: the event
        # line above is already on screen (one appended block), and a
        # wipe-and-remount of the whole history here is the 1–2s flash.
        if self.viewed_id == task.id and had_pending:
            self.view_task(task.id)
        # The pipeline continues: testing backlog first, then the queue.
        self._advance()

    def _run_alive_signature(
        self,
    ) -> tuple[frozenset[str], bool, str | None, str | None]:
        """Everything about the current run state that the armed-ESC
        confirmation cares about: which agent runs are live, whether the
        tester runs (and on what), and which task is viewed. A change in
        any of these between the arming and the confirming ESC invalidates
        the arming — a stale "armed" state must never fire against runs
        the user has not just seen."""
        return (
            frozenset(self._agent_runs),
            self.tester_running,
            self._tester_task_id,
            self.viewed_id,
        )

    def _esc_targets_running(self) -> bool:
        """Whether the task(s) THIS ESC press would stop are actually
        running: with a single slot it is the historical global check,
        with parallel slots only the viewed task's runs count."""
        if self.parallel_limit <= 1:
            return self.agent_running or self.tester_running
        task = self.viewed
        if task is None:
            return False
        return task.id in self._agent_runs or (
            self.tester_running and self._tester_task_id == task.id
        )

    def _esc_disarm(self) -> None:
        """Drop the armed-ESC confirmation state (window expired, runs
        changed, or the stop actually fired)."""
        self._esc_armed_signature = None
        if self._esc_disarm_timer is not None:
            self._esc_disarm_timer.stop()
            self._esc_disarm_timer = None

    def _esc_arm_expired(self) -> None:
        """The confirmation window closed without a second ESC: disarm
        quietly, so the NEXT ESC arms afresh."""
        self._esc_disarm_timer = None
        self._esc_armed_signature = None
        self.refresh_status()

    def _esc_arm(self) -> None:
        """Arm the stop: announce it and start the confirmation window.
        The actual abort still requires a second ESC within the window."""
        self._esc_armed_signature = self._run_alive_signature()
        self._esc_armed_at = time.monotonic()
        if self._esc_disarm_timer is not None:
            self._esc_disarm_timer.stop()
        self._esc_disarm_timer = self.set_timer(
            self._ESC_ARM_SECONDS, self._esc_arm_expired
        )
        self.notify(
            "Press ESC again to stop the run (2s)",
            title="Stop run",
            timeout=self._ESC_ARM_SECONDS + 0.5,
        )
        self.refresh_status()

    def _abort_active_runs(self) -> None:
        """The actual stop, shared by both ESC modes.

        With a single development slot (the default, or parallelism
        without worktree isolation) this is the historical global stop:
        it halts whatever run is alive. With parallel slots (>= 2) it
        stops ONLY the currently viewed task's run — the other tasks'
        runs are independent work the user did not ask to stop. Every
        aborted run leaves a service event in its task's chat ("press
        Ctrl+R to resume") and a toast, so the stop is never silent."""
        stopped_tasks: list[Task] = []
        if self.parallel_limit <= 1:
            if not (self.agent_running or self.tester_running):
                return
            stopped_tasks = [
                self.manager.get(tid)
                for tid in list(self._agent_runs)
                if tid in self.manager.tasks
            ]
            for group in [f"agent-{tid}" for tid in list(self._agent_runs)]:
                self.workers.cancel_group(self, group)
            if self.tester_running:
                self.workers.cancel_group(self, "tester")
                if (
                    self._tester_task_id is not None
                    and self._tester_task_id in self.manager.tasks
                ):
                    stopped_tasks.append(self.manager.get(self._tester_task_id))
            self._agent_runs.clear()
            self.tester_running = False
            self._current_tool.clear()
            # Hide AND empty: the aborted run's partial text must not be
            # resurrected by a later add_class("visible").
            self._clear_stream_label()
            self.refresh_status()
        else:
            task = self.viewed
            if task is None:
                return
            stopped = False
            if task.id in self._agent_runs:
                self.workers.cancel_group(self, f"agent-{task.id}")
                self._agent_runs.pop(task.id, None)
                self._current_tool[task.id] = None
                stopped = True
                stopped_tasks.append(task)
            if self.tester_running and self._tester_task_id == task.id:
                self.workers.cancel_group(self, "tester")
                self.tester_running = False
                self._current_tool[task.id] = None
                stopped = True
                stopped_tasks.append(task)
            if stopped:
                self._clear_stream_label()
                self.refresh_status()
            else:
                self.notify(
                    "Task is not running — open an active task to stop one",
                    title="Stop run",
                    timeout=5,
                )
                return
        # Mitigation: make the stop explicit and show the way back
        # (Ctrl+R re-runs the interrupted attempt — action_nudge_run).
        for stopped in stopped_tasks:
            if stopped.status is TaskStatus.ACTIVE and not stopped.unfinished:
                # Mark the interrupted attempt HERE, not only in the run's
                # CancelledError handler: the worker may be cancelled
                # before its first step (a just-dispatched run under load)
                # or inside its setup window (worktree git, the task-start
                # hook) — the handler then never runs, the marker is lost,
                # and the stopped task looks like a run that silently
                # ignored the stop (or auto-restarts itself out of the
                # queue). Idempotent with the handler's own marking.
                stopped.unfinished = True
                stopped.touch()
                self.manager.storage.save(stopped)
            self._show_event(
                stopped,
                "Run aborted by user (ESC) — press Ctrl+R to resume.",
            )
        self.notify(
            "Run stopped · Ctrl+R to resume",
            title="Stop run",
            timeout=5,
        )

    def action_abort_run(self) -> None:
        """ESC: stop the running agent/tester.

        Two modes (project setting `esc_abort`). DOUBLE_PRESS (the
        default): the first ESC only ARMS the stop — a toast asks for a
        confirmation press — and a second ESC within the window performs
        the stop. A press faster than the debounce interval (an
        accidental double-keystroke, exactly the inertia scenario this
        guards against) is ignored; an ESC after the window expired
        re-arms instead of firing. INSTANT: the historical single-press
        stop. In both modes the footer binding stays "Stop agent"."""
        if self.settings.esc_abort is EscAbort.INSTANT:
            self._abort_active_runs()
            return
        now = time.monotonic()
        # Debounce: a second ESC within the debounce interval of the last
        # ACCEPTED press is an accidental fast repeat — ignore it without
        # consuming the armed window (and without re-stamping the press
        # time, so a deliberate third press still confirms).
        if now - self._esc_last_press < self._ESC_DEBOUNCE_SECONDS:
            return
        self._esc_last_press = now
        if not self._esc_targets_running():
            # Nothing to stop: the historical no-op / toast. Any stale
            # arming is dropped — it must not wait for runs that are gone.
            self._esc_disarm()
            if self.parallel_limit > 1:
                self.notify(
                    "Task is not running — open an active task to stop one",
                    title="Stop run",
                    timeout=5,
                )
            return
        signature = self._run_alive_signature()
        armed = self._esc_armed_signature is not None and (
            now - self._esc_armed_at <= self._ESC_ARM_SECONDS
        )
        if armed and self._esc_armed_signature == signature:
            # Second press within the window and the runs are the same
            # ones the user saw when arming: fire the actual stop.
            self._esc_disarm()
            self._abort_active_runs()
            return
        # First press, re-arm after an expired window, or the run state
        # changed since arming (a run finished/started on its own, the
        # viewed task switched): (re-)arm against the CURRENT state.
        self._esc_arm()

    def action_nudge_run(self) -> None:
        """Ctrl+R on the viewed ACTIVE/TESTING task: abort its hung or
        stalled run and restart it — the same move the stall watchdog makes
        automatically, and the one-key resume for a task left "active,
        nothing running" by a previous abort."""
        task = self.viewed
        if task is None:
            return
        self._nudge_run(task)

    def action_cycle_task_kind(self) -> None:
        """Cycle the type of the viewed task (feature → bug → research).
        Only a queued (not yet started) task can be re-typed; bug moves it
        to the head of the queue, feature — to the tail."""
        task = self.viewed
        if task is None or task.status is not TaskStatus.QUEUED:
            return
        idx = self._KIND_CYCLE.index(task.kind)
        new_kind = self._KIND_CYCLE[(idx + 1) % len(self._KIND_CYCLE)]
        self.manager.set_kind(task.id, new_kind)
        self.refresh_sidebar()
        self.refresh_status()
        if self.viewed_id == task.id:
            self.refresh_task_status()

    def action_move_task_up(self) -> None:
        """Move the viewed queued task one step up the backlog — away from
        execution (the list runs top = farthest, bottom = next to run)."""
        self._move_viewed_task(up=True)

    def action_move_task_down(self) -> None:
        """Move the viewed queued task one step down the backlog — closer
        to execution (swaps it with the task the pipeline would pick
        before it)."""
        self._move_viewed_task(up=False)

    def _move_viewed_task(self, *, up: bool) -> None:
        task = self.viewed
        if task is None or task.status is not TaskStatus.QUEUED:
            return
        if self.manager.move_in_queue(task.id, up=up) is None:
            return
        self.refresh_sidebar()
        self._refocus_sidebar_task(task.id)
        # The Move up/Move down footer buttons hide at the queue edges and
        # at the bug-block boundary — re-evaluate the visible bindings.
        self.refresh_bindings()

    # -- dependencies (depends_on gate) ----------------------------------------

    def _can_edit_dependencies(self, task: Task | None) -> bool:
        """Dependencies only make sense before development starts: queued
        and held tasks (the same statuses the kind cycle accepts)."""
        return task is not None and task.status in (
            TaskStatus.QUEUED,
            TaskStatus.HOLD,
        )

    def action_add_dependency(self, dep_id: str) -> None:
        """Command-palette entry: make the viewed task wait for `dep_id`
        (COMPLETED + merged) before it can be taken into development."""
        task = self.viewed
        if not self._can_edit_dependencies(task) or task is None:
            return
        try:
            self.manager.set_dependencies(task.id, [*task.depends_on, dep_id])
        except ValueError as e:
            self.notify(str(e), title="Dependencies", severity="warning", timeout=6)
            return
        self._show_event(task, f"Dependency #{dep_id} added — task waits for it.")
        self.refresh_sidebar()
        self.refresh_status()
        if self.viewed_id == task.id:
            self.refresh_task_status()

    def action_remove_dependency(self, dep_id: str) -> None:
        """Command-palette entry: drop one dependency from the viewed task.

        Doubles as the force-start: removing the blocking dependency makes
        the task eligible immediately (the gate is dynamic — nothing else
        to reset). The manual override is logged into the task's service
        events."""
        task = self.viewed
        if not self._can_edit_dependencies(task) or task is None:
            return
        if dep_id not in task.depends_on:
            return
        self.manager.set_dependencies(
            task.id, [d for d in task.depends_on if d != dep_id]
        )
        self._dep_block_notified.discard(task.id)
        self._dep_gate.pop(task.id, None)
        self._show_event(
            task,
            f"Manual override: dependency #{dep_id} removed — "
            "task eligible for development.",
        )
        self.refresh_sidebar()
        self.refresh_status()
        if self.viewed_id == task.id:
            self.refresh_task_status()
        self._advance()

    def action_hold_task(self) -> None:
        """Park the viewed task out of the backlog (hold). Works for queued
        tasks and for tasks awaiting a clarifying answer (the question is
        preserved and shown again on resume). Held tasks are never taken
        into work until explicitly resumed; typed input keeps accumulating."""
        task = self.viewed
        if task is None or task.status not in (
            TaskStatus.QUEUED,
            TaskStatus.AWAITING_CLARIFICATION,
        ):
            return
        self.manager.set_hold(task.id, on=True)
        self.refresh_sidebar()
        self.refresh_status()
        if self.viewed_id == task.id:
            self.refresh_task_status()

    def action_resume_task(self) -> None:
        """Return a held task to where it was (queued or awaiting
        clarification, whichever it was parked from)."""
        task = self.viewed
        if task is None or task.status is not TaskStatus.HOLD:
            return
        self.manager.set_hold(task.id, on=False)
        self._advance()
        self.refresh_sidebar()
        self.refresh_status()
        if self.viewed_id == task.id:
            self.refresh_task_status()

    def action_delete_task(self) -> None:
        """Fully remove the viewed task — from the sidebar and from disk
        (.drcode/tasks/<id>.json). Deletable statuses: HOLD (nothing is
        running for a parked task, so no worker state is left behind) and
        COMPLETED (the work is done; deletion only takes effect earlier
        than the retention purge would). A COMPLETED task still awaiting
        its external review verdict (`review_pending` — an open PR) is
        NOT deletable: a verdict arriving later must not find the task
        gone. Everything else (QUEUED/ACTIVE/TESTING/AWAITING_CLARIFICATION)
        is refused. Resume from hold is Ctrl+H; deletion is Ctrl+Delete."""
        task = self.viewed
        if task is None or task.status not in (
            TaskStatus.HOLD,
            TaskStatus.COMPLETED,
        ):
            return
        if task.status is TaskStatus.COMPLETED and task.review_pending:
            text = (
                "Task is not deleted: it still awaits an external review "
                "verdict (an open PR). Delete it after the verdict arrives "
                "(or cancel the review)."
            )
            self._show_event(task, text)
            self.notify(text, title="Delete task", severity="warning", timeout=8)
            return
        # Sidebar order as the user sees it: the flattened non-empty
        # section list. Computed BEFORE the delete — the neighbour choice
        # needs the deleted row's position (all_tasks() creation order
        # would jump the view to the oldest archived task, visually far
        # from the queue tail the user was working with).
        flat = [t for _name, tasks in self._sidebar_sections() for t in tasks]
        pos = next((i for i, t in enumerate(flat) if t.id == task.id), None)
        self.manager.delete(task.id)
        # The chat pane shows the deleted task: switch to another one (or
        # an empty pane when the backlog is gone entirely).
        self.viewed_id = None
        self.chat_log.remove_children()
        # Invalidate any pending history drain: without the epoch bump its
        # remaining chunks would mount into this wiped log as zombies.
        self._chat_epoch += 1
        self._chat_has_content = False
        self._pending_rendered = []
        self._invalidate_chat_delta()
        remaining = [t for t in flat if t.id != task.id]
        if remaining and pos is not None and pos < len(remaining):
            # The task that FOLLOWED the deleted one in the sidebar (the
            # rows before it kept their positions, so remaining[pos] is
            # exactly the next row as displayed).
            self.view_task(remaining[pos].id)
        elif remaining:
            # Deleted task was the last row: fall back to the one that
            # preceded it.
            self.view_task(remaining[-1].id)
        else:
            self.refresh_sidebar()
            self._sync_sidebar_cursor(None)
            self.refresh_status()
            self.refresh_task_status()

    # -- events -------------------------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, TaskItem):
            if item.section is not None:
                # Enter (or a mouse click) on a section header toggles the
                # category; focus stays in the sidebar for further toggling.
                event.stop()
                self.toggle_sidebar_section(item.section)
                self.sidebar.focus()
                return
            if item.task_id:
                self.view_task(item.task_id)
                self.chat_input.focus()

    def submit_chat_input(self) -> None:
        """Send the chat editor's content (Enter in the chat input)."""
        text = self.chat_input.text.strip()
        if not text:
            return
        task = self.viewed
        if task is None:
            return
        self.chat_input.clear()
        self._submit_user_text(task, text)

    def _submit_user_text(self, task: Task, text: str) -> None:
        # A re-send of a text that is still queued (an identical entry sits
        # in pending_input) or already in flight (the transcript holds it as
        # the prompt of the unfinished attempt) is a no-op: send() ignores
        # it, and drawing/promoting it again would show the comment twice.
        stripped = text.strip()
        if any(e.text == stripped for e in task.pending_input) or is_pending_retry(
            task.messages, stripped
        ):
            return
        # Capture the pre-run state: _advance() promotes the task to ACTIVE
        # (promote_next) and flips its status, so the render decision below
        # must use the ORIGINAL status, not the post-promotion one.
        was_active = task.status is TaskStatus.ACTIVE
        self.manager.send(task.id, text)
        if not was_active and self.viewed_id == task.id:
            # Draw the queued message BEFORE advancing the pipeline. The run
            # worker started by _advance() executes eagerly (Textual installs
            # asyncio.eager_task_factory): its pending-input loop can run
            # synchronously inside _advance() and would draw the "user" block
            # itself — at that point the message is neither in
            # _pending_rendered nor in the persisted transcript yet, so the
            # run cannot know it is already on screen. Drawing here first
            # registers the text in _pending_rendered; the consuming run then
            # skips it and the bubble shows exactly once. For an active task
            # run_agent() renders the message itself (see the was_active
            # branch below), so no pre-draw there.
            self._write_history_line("user-pending", text)
        if was_active:
            # A comment to the running task interrupts it: the exclusive
            # "agent" group cancels the in-flight run (its message goes back
            # to pending_input) and the new run resumes with the comment.
            self._clear_stream_label()
            self.run_agent(task.id)
        else:
            # Queued/clarification/... input: advance the pipeline — testing
            # backlog first, then the queue.
            self._advance()
        self.refresh_sidebar()
        self.refresh_status()

    def action_pick_option(self, index: int) -> None:
        """User clicked a clarifying-question option: submit it as their answer."""
        task = self.viewed
        if task is None or task.clarification is None:
            return
        if not 0 <= index < len(task.clarification.options):
            return
        self._submit_user_text(task, task.clarification.options[index])

    def action_open_task(self, task_id: str) -> None:
        """User clicked a task-id mention (#69c8bd30) in the chat or the
        status surfaces: open the mentioned task."""
        task_id = task_id.lower()
        # A drag-selection that ends on a mention still dispatches the
        # @click action (App._broker_event runs before any widget on_click
        # and cannot be stopped by the block): an active selection means
        # the user is copying text, not navigating.
        if self.screen.selections:
            return
        if task_id not in self.manager.tasks:
            self.notify(
                f"Task #{task_id} not found",
                title="Tasks",
                severity="warning",
                timeout=6,
            )
            return
        if task_id == self.viewed_id:
            # Already on screen: skip the full chat wipe + scroll jump of
            # view_task().
            return
        self.view_task(task_id)
        self.chat_input.focus()
