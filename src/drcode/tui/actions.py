from __future__ import annotations

import time
from typing import ClassVar

from textual.command import (
    CommandPalette,
)
from textual.containers import Vertical
from textual.widgets import (
    ListView,
    Static,
)

from ..agent import (
    is_pending_retry,
)
from ..github import (
    GitHubClient,
    GitHubError,
    sync_issues,
)
from ..models import (
    Task,
    TaskKind,
    TaskStatus,
)
from ..prompts import load_init_repo_prompt
from ..settings import (
    CompletedRetention,
    StallNudge,
    TaskFinalization,
)
from .agents_dialog import AgentsMdDialog
from .command_providers import (
    FinalizationOptionsProvider,
    RetentionOptionsProvider,
    SettingsPalette,
    StallNudgeOptionsProvider,
    ThemeOptionsProvider,
)
from .github_dialog import GitHubSettingsDialog, GitHubSetup
from .inputs import NewTaskInput
from .setup_dialog import ModelSetup, ModelSetupDialog
from .sidebar import TaskItem
from .state import DrCodeState


class ActionsLayer(DrCodeState):

    def _model_setup_done(self, setup: ModelSetup | None) -> None:
        """Result of the first-start setup dialog.

        None — the user skipped it (ESC): keep running without a model.
        ModelSetup — the connection was probed successfully: apply the
        values to the live config, persist them into
        <project>/drcode.json and start the pipeline.
        """
        if setup is None:
            return
        self.config.model = setup.model
        self.config.base_url = setup.base_url
        self.config.api_key = setup.api_key
        self.config.health_check_url = setup.health_check_url
        if setup.max_context_tokens is not None:
            self.config.max_context_tokens = setup.max_context_tokens
        if setup.max_tokens is not None:
            self.config.max_tokens = setup.max_tokens
        try:
            saved = self.config.save_to_project()
        except OSError as e:
            self.notify(
                f"Could not write drcode.json: {e}",
                title="Model setup",
                severity="error",
                timeout=8,
            )
        else:
            self.notify(
                f"Model saved to {saved.name}",
                title="Model setup",
                timeout=4,
            )
        self.refresh_connection()
        self.refresh_status()
        self._advance()
        # The settings were just created — the "no AGENTS.md" offer now
        # applies as well.
        self._maybe_offer_agents_md()


    def _maybe_offer_agents_md(self) -> None:
        """Startup offer: model configured, project root has no AGENTS.md.

        Asks the user once per session; on agreement creates the same
        pre-filled research task as the Settings palette's
        "Update/create AGENTS.md" entry. Skipped silently when the model
        is not configured (the pipeline is idle anyway), when AGENTS.md
        already exists, or when such a research task is already in the
        backlog.
        """
        if not self.config.is_ready:
            return
        if (self.project_root / "AGENTS.md").exists():
            return
        if any(
            t.kind is TaskKind.RESEARCH and "AGENTS.md" in t.title
            for t in self.manager.tasks.values()
        ):
            return
        self.push_screen(AgentsMdDialog(), self._agents_md_answered)


    def _agents_md_answered(self, create: bool | None) -> None:
        # Textual's dismissal type includes None (ESC without a result);
        # the dialog always answers with True/False, treat None as "no".
        if create:
            self._create_agents_task()

    # -- widgets helpers ------------------------------------------------------


    @property
    def new_task_input(self) -> NewTaskInput:
        return self.query_one("#nt-input", NewTaskInput)


    def _clear_stream_label(self) -> None:
        """Legacy hook of the removed streaming-preview widget.

        The preview was replaced by in-place assistant blocks; nothing to
        clear — kept as a no-op so abort/new-task paths stay simple.
        """


    def _drop_stream_preview(self, generation: int) -> None:
        """Clear the streaming preview after an agent run ends, but only if
        no newer run has (already) taken over the label."""
        if generation == self._agent_generation:
            self._clear_stream_label()


    def action_new_task(self) -> None:
        # Every entry into the new-task screen starts from a clean editor.
        self.new_task_input.clear()
        self._new_task_mode = True
        # The form is not tied to any task: drop the viewed highlight so the
        # sidebar doesn't single out a task while the user composes a new one.
        self._new_task_prev_id = self.viewed_id
        self.viewed_id = None
        self.refresh_sidebar()
        # The streaming preview belongs to the chat view, not the form: a
        # run in progress must not leak its orange text over the form.
        self._clear_stream_label()
        self.chat_log.add_class("hidden")
        self.chat_input.add_class("hidden")
        self.query_one("#new-task", Vertical).remove_class("hidden")
        self._update_kind_label()
        self.refresh_task_status()
        self.new_task_input.focus()


    def _leave_new_task_mode(self) -> None:
        self._new_task_mode = False
        self.query_one("#new-task", Vertical).add_class("hidden")
        self.chat_log.remove_class("hidden")
        # Content may be stale (a run streamed while the form was open, or
        # finished in the background): don't resurrect it.
        self._clear_stream_label()
        self.chat_input.remove_class("hidden")
        self.refresh_task_status()


    def cancel_new_task(self) -> None:
        self._leave_new_task_mode()
        # The chat log still shows the task that was open before the form:
        # restore its view so the status bar matches the visible content
        # (action_new_task() cleared viewed_id, leaving the bar empty).
        if self.viewed_id is None:
            prev = self._new_task_prev_id
            task = self.manager.tasks.get(prev) if prev else None
            if task is None:
                task = self._sidebar_top_task()
            if task is not None:
                self.view_task(task.id)
        self.chat_input.focus()

    # Tab cycles the new-task type through all three kinds.
    _KIND_CYCLE: ClassVar = (TaskKind.FEATURE, TaskKind.BUG, TaskKind.RESEARCH)

    # One-line summary per kind, shown at the bottom of the new-task form;
    # the currently selected kind is highlighted (markup colors mirror the
    # #nt-kind-label classes above).
    _KIND_DESCRIPTIONS: ClassVar[dict[TaskKind, str]] = {
        TaskKind.BUG: "fix broken functionality with minimal, targeted changes",
        TaskKind.FEATURE: "build new functionality and refactor the codebase",
        TaskKind.RESEARCH: (
            "explore the project, answer questions and file follow-up tasks"
        ),
    }
    _KIND_COLOR: ClassVar[dict[TaskKind, str]] = {
        TaskKind.BUG: "red",
        TaskKind.FEATURE: "green",
        TaskKind.RESEARCH: "yellow",
    }


    def toggle_new_task_kind(self) -> None:
        idx = self._KIND_CYCLE.index(self.new_task_kind)
        self.new_task_kind = self._KIND_CYCLE[(idx + 1) % len(self._KIND_CYCLE)]
        self._update_kind_label()


    def _update_kind_label(self) -> None:
        label = self.query_one("#nt-kind-label", Static)
        label.remove_class("kind-feature", "kind-bug", "kind-research")
        if self.new_task_kind is TaskKind.BUG:
            label.update("type: bug")
            label.add_class("kind-bug")
        elif self.new_task_kind is TaskKind.RESEARCH:
            label.update("type: research")
            label.add_class("kind-research")
        else:
            label.update("type: feature")
            label.add_class("kind-feature")
        self._update_kind_descriptions()


    def _update_kind_descriptions(self) -> None:
        lines = []
        for kind in self._KIND_CYCLE:
            active = kind is self.new_task_kind
            name = kind.value.ljust(8)
            if active:
                name = f"[{self._KIND_COLOR[kind]}][b]{name}[/][/]"
            marker = "▸" if active else " "
            lines.append(f"{marker} {name} — {self._KIND_DESCRIPTIONS[kind]}")
        self.query_one("#nt-kinds", Static).update("\n".join(lines))


    def submit_new_task(self) -> None:
        text = self.new_task_input.text.strip()
        if not text:
            return
        task = self.manager.create("", text, kind=self.new_task_kind)
        self._leave_new_task_mode()
        self.refresh_sidebar()
        self.view_task(task.id)
        self.chat_input.focus()
        # Starts immediately unless testing is pending or the queue is busy.
        self._advance()


    def action_done_task(self) -> None:
        task = self.viewed
        if task is None or task.status is TaskStatus.COMPLETED:
            return
        self.workers.cancel_group(self, "agent")
        self.agent_running = False
        self.manager.mark_done(task.id)
        self.refresh_sidebar()
        self.view_task(task.id)
        # Testing backlog first (possibly an older TESTING task), then queue.
        self._advance()


    def action_cancel_task(self) -> None:
        """User cancels the viewed task (not relevant anymore): drop all
        queued input and complete it immediately — the tester never runs."""
        task = self.viewed
        if task is None or task.status is TaskStatus.COMPLETED:
            return
        active = self.manager.active
        if self.agent_running and active is not None and active.id == task.id:
            self.workers.cancel_group(self, "agent")
            self.agent_running = False
            self._current_tool = None
        if self.tester_running and self._tester_task_id == task.id:
            self.workers.cancel_group(self, "tester")
            self.tester_running = False
            self._current_tool = None
        self.manager.cancel(task.id)
        self._tester_resume_round.discard(task.id)
        self._show_event(
            task,
            "Задача отменена пользователем — снята с конвейера без тестирования.",
        )
        self.refresh_sidebar()
        self.refresh_status()
        if self.viewed_id == task.id:
            self.view_task(task.id)
        # The pipeline continues: testing backlog first, then the queue.
        self._advance()


    def action_abort_run(self) -> None:
        if self.agent_running or self.tester_running:
            self.workers.cancel_group(self, "agent")
            self.workers.cancel_group(self, "tester")
            self.agent_running = False
            self.tester_running = False
            self._current_tool = None
            # Hide AND empty: the aborted run's partial text must not be
            # resurrected by a later add_class("visible").
            self._clear_stream_label()
            self.refresh_status()


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
        """Fully remove the viewed held task — from the sidebar and from disk
        (.drcode/tasks/<id>.json). Only parked tasks can be deleted: nothing
        is running for them, so no worker state is left behind. Resume from
        hold is Ctrl+H; deletion is Ctrl+Delete."""
        task = self.viewed
        if task is None or task.status is not TaskStatus.HOLD:
            return
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
        remaining = self.manager.all_tasks()
        if remaining:
            self.view_task(remaining[0].id)
        else:
            self.refresh_sidebar()
            self.refresh_status()
            self.refresh_task_status()


    def action_focus_sidebar(self) -> None:
        self.sidebar.focus()


    def action_focus_input(self) -> None:
        # The new-task form replaces the chat input while it is open: the
        # hidden chat editor must not steal focus (a focused display:none
        # widget swallows typing into nowhere).
        if self._new_task_mode:
            self.new_task_input.focus()
        else:
            self.chat_input.focus()


    def action_settings(self) -> None:
        """Open the Settings modal (the command palette, Ctrl+S)."""
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(SettingsPalette(id="--command-palette"))


    def action_open_model_setup(self) -> None:
        """Re-open the model setup dialog on demand (a Settings entry).

        Pre-filled with the current model/base_url/api_key/context-size, so
        the user can edit the connection anytime, not just on first start.
        The result is handled by `_model_setup_done`, which applies the new
        values to the live config immediately and persists them.
        """
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            ModelSetupDialog(
                model=self.config.model,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                max_context_tokens=self.config.max_context_tokens,
                max_tokens=self.config.max_tokens,
            ),
            self._model_setup_done,
        )


    def action_set_completed_retention(self, retention: CompletedRetention) -> None:
        """Settings command: after how long completed tasks are deleted at
        startup. Persisted per project (.drcode/settings.json); the cleanup
        itself runs once per app start."""
        self.settings.completed_retention = retention
        self.settings_store.save(self.settings)
        self.notify(
            f"Auto-delete completed tasks: {retention.label}",
            title="Project settings",
            timeout=4,
        )


    def action_change_completed_retention(self) -> None:
        """Completed-tasks auto-cleanup picker, opened from the single
        Settings entry ("Auto-delete completed tasks: …").

        Pushes the narrowed SettingsPalette with the retention options so the
        selection window has the same width as the Settings modal; picking an
        option applies it, Escape dismisses without changing anything.
        """
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--retention-palette",
                providers=[RetentionOptionsProvider],
                placeholder="Search for options…",
            )
        )


    def action_set_task_finalization(self, finalization: TaskFinalization) -> None:
        """Settings command: what happens after a development round finishes
        (tester + commit / tester / nothing). Persisted per project
        (.drcode/settings.json); applies to every task finalized from now on.
        Switching to "None" lets the pipeline complete any tasks currently
        waiting for testing on the next _advance()."""
        self.settings.task_finalization = finalization
        self.settings_store.save(self.settings)
        self.notify(
            f"Task finalization: {finalization.label}",
            title="Project settings",
            timeout=4,
        )
        self._advance()


    def action_change_task_finalization(self) -> None:
        """Task-finalization picker, opened from the Settings entry
        ("Task finalization: …"). Same UX as the retention picker: the
        narrowed SettingsPalette with the three options; picking one applies
        it, Escape dismisses without changing anything.
        """
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--finalization-palette",
                providers=[FinalizationOptionsProvider],
                placeholder="Search for options…",
            )
        )


    def action_set_stall_nudge(self, nudge: StallNudge) -> None:
        """Settings command: after how long without any activity a hung
        agent/tester run is aborted and restarted. Persisted per project
        (.drcode/settings.json); the watchdog picks the new value up on its
        next 1-second tick ("never" disables it)."""
        self.settings.stall_nudge = nudge
        self.settings_store.save(self.settings)
        self.notify(
            f"Nudge stalled agent after: {nudge.label}",
            title="Project settings",
            timeout=4,
        )


    def action_change_stall_nudge(self) -> None:
        """Stall-nudge picker, opened from the Settings entry ("Nudge
        stalled agent after: …"). Same UX as the retention picker: the
        narrowed SettingsPalette with the five options; picking one applies
        it, Escape dismisses without changing anything.
        """
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--stall-nudge-palette",
                providers=[StallNudgeOptionsProvider],
                placeholder="Search for options…",
            )
        )


    def action_change_github(self) -> None:
        """GitHub-integration dialog, opened from the Settings entry
        ("GitHub integration: on/off"). Collects the token, repository,
        sync interval and the enable switch; Save persists the values into
        the project settings regardless of the connection status shown in
        the dialog."""
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            GitHubSettingsDialog(
                enabled=self.settings.github_enabled,
                token=self.settings.github_token,
                repo=self.settings.github_repo,
                interval=self.settings.github_sync_interval,
            ),
            self._github_settings_done,
        )


    def _github_settings_done(self, setup: GitHubSetup | None) -> None:
        if setup is None:
            return
        self.settings.github_enabled = setup.enabled
        self.settings.github_token = setup.token
        self.settings.github_repo = setup.repo
        self.settings.github_sync_interval = setup.interval
        self.settings_store.save(self.settings)
        self.notify(
            f"GitHub integration: {'on' if setup.enabled else 'off'}"
            + (
                f" ({setup.repo}, every {setup.interval.label})"
                if setup.enabled and setup.repo
                else ""
            ),
            title="Project settings",
            timeout=4,
        )
        if setup.enabled:
            # A fresh save (or just turning it on) triggers a sync cycle
            # right away; the schedule then continues by the interval.
            self._github_next_sync = time.monotonic()
            self._poll_github_sync()
        self.refresh_status()

    # -- GitHub Issues sync ----------------------------------------------------


    def _poll_github_sync(self) -> None:
        """Scheduler tick (every 30s): when the integration is enabled and
        the configured interval has elapsed, run one sync cycle. The next
        slot is stamped at attempt time, so a failing cycle (bad token,
        unreachable GitHub) retries on the next interval instead of
        hammering the API."""
        if not (
            self.settings.github_enabled
            and self.settings.github_token
            and self.settings.github_repo
        ):
            return
        if time.monotonic() < self._github_next_sync:
            return
        self._github_next_sync = (
            time.monotonic() + self.settings.github_sync_interval.hours * 3600
        )
        self.run_worker(self._github_sync(), exclusive=True, group="github-sync")


    async def _github_sync(self) -> None:
        client = GitHubClient(self.settings.github_token, self.settings.github_repo)
        self._github_syncing = True
        try:
            try:
                imported, skipped = await sync_issues(
                    self.manager, client, self._github_state
                )
            except GitHubError as e:
                self._github_last_error = str(e)
                self.notify(
                    f"GitHub sync failed: {e}",
                    title="GitHub integration",
                    severity="warning",
                    timeout=8,
                )
                return
            self._github_last_error = None
            self._github_state_store.save(self._github_state)
            if imported or skipped:
                self.notify(
                    f"GitHub sync: {imported} new task(s) imported, {skipped} skipped",
                    title="GitHub integration",
                    timeout=6,
                )
                if imported:
                    self.refresh_sidebar()
                    # New tasks enter the pipeline (testing backlog first).
                    self._advance()
        finally:
            self._github_syncing = False


    def action_update_agents_md(self) -> None:
        """Create a research task that writes/refreshes AGENTS.md.

        The task is pre-filled with the bundled init-repo prompt (its final
        paragraph names the actual project directory) and runs with the
        `research` kind: the tester is skipped and mark_done() completes
        it right away."""
        self._create_agents_task()


    def _create_agents_task(self) -> None:
        """Shared body of the Settings action and the startup AGENTS.md
        offer: queue the pre-filled research task and open it in chat."""
        task = self.manager.create(
            "Update/create AGENTS.md",
            load_init_repo_prompt(self.project_root),
            kind=TaskKind.RESEARCH,
        )
        self.refresh_sidebar()
        self.view_task(task.id)
        self.chat_input.focus()
        # Starts immediately unless testing is pending or the queue is busy.
        self._advance()


    def action_change_theme(self) -> None:
        """Theme picker, opened from the Settings palette ("Theme" command).

        The stock App.action_change_theme pushes a full-width CommandPalette;
        reuse the narrowed SettingsPalette instead so the theme list has the
        same width as the Settings modal.
        """
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--theme-palette",
                providers=[ThemeOptionsProvider],
                placeholder="Search for themes…",
                classes="theme-palette",
            )
        )

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
        # Capture the pre-run state: _advance() promotes the task to ACTIVE
        # (promote_next) and flips its status, so the render check below must
        # compare the ORIGINAL status, not the post-promotion one.
        was_active = task.status is TaskStatus.ACTIVE
        self.manager.send(task.id, text)
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
        # For an active task run_agent() renders the message itself. A
        # non-active task keeps the queued comment visible even after the
        # promotion flips it to ACTIVE — so use the captured original state.
        if (
            not was_active
            and self.viewed_id == task.id
            and not is_pending_retry(task.messages, text)
        ):
            self._write_history_line("user-pending", text)


    def action_pick_option(self, index: int) -> None:
        """User clicked a clarifying-question option: submit it as their answer."""
        task = self.viewed
        if task is None or task.clarification is None:
            return
        if not 0 <= index < len(task.clarification.options):
            return
        self._submit_user_text(task, task.clarification.options[index])


