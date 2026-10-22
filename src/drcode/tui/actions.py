from __future__ import annotations

import time
from functools import partial
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
from ..providers import (
    AgentRole,
    CustomModelEntry,
    ModelRef,
    UserModelsStore,
    load_project_custom_models,
    load_project_model_slots,
    set_project_custom_active,
    set_project_custom_model,
    set_project_model,
)
from ..settings import (
    CompletedRetention,
    StallNudge,
    TaskFinalization,
    TaskIsolation,
)
from ..workflow import describe_project_workflow, load_git_workflow
from .agents_dialog import AgentsMdDialog
from .command_providers import (
    DevRoundsOptionsProvider,
    FinalizationOptionsProvider,
    IsolationOptionsProvider,
    ModelCatalogCommandProvider,
    ModelSettingsPalette,
    RetentionOptionsProvider,
    RoleModelsCommandsProvider,
    SettingsPalette,
    StallNudgeOptionsProvider,
    ThemeOptionsProvider,
    role_model_options_provider,
)
from .github_dialog import GitHubSettingsDialog, GitHubSetup
from .inputs import NewTaskInput
from .model_setup_app import ModelCatalogScreen
from .notifications_dialog import NotificationsDialog, NotificationsSetup
from .setup_dialog import ModelSetup, ModelSetupDialog
from .sidebar import TaskItem
from .state import DrCodeState


class ActionsLayer(DrCodeState):
    @staticmethod
    def _custom_entry_from_setup(setup: ModelSetup) -> CustomModelEntry:
        """The dialog result as a per-role custom model entry (a None cap
        from the dialog means "keep the default"/"no cap")."""
        return CustomModelEntry(
            model=setup.model,
            base_url=setup.base_url,
            api_key=setup.api_key,
            health_check_url=setup.health_check_url,
            max_context_tokens=setup.max_context_tokens or 200_000,
            max_tokens=setup.max_tokens,
        )

    def _model_setup_done(self, setup: ModelSetup | None) -> None:
        """Result of the first-start setup dialog.

        None — the user skipped it (ESC): keep running without a model.
        ModelSetup — the connection was probed successfully: it is saved
        as a CUSTOM model for each of the coding / tester / compaction
        roles (independent copies in .drcode/drcode.json — each editable
        separately later) and applied to the live config. The research
        role is deliberately left unpinned: it inherits the coding model
        via the `resolve_effective` fallback, so improving the coding
        model later also improves research until a dedicated research
        model is picked.
        """
        if setup is None:
            return
        # Apply to the flat config as well: it is the live container the
        # env layer writes into (the project slots resolve on top of it).
        self.config.model = setup.model
        # The flat connection changed: per-role resolutions are stale.
        self.invalidate_resolved_configs()
        self.config.base_url = setup.base_url
        self.config.api_key = setup.api_key
        self.config.health_check_url = setup.health_check_url
        if setup.max_context_tokens is not None:
            self.config.max_context_tokens = setup.max_context_tokens
        if setup.max_tokens is not None:
            self.config.max_tokens = setup.max_tokens
        entry = self._custom_entry_from_setup(setup)
        try:
            for role in AgentRole:
                if role is AgentRole.RESEARCH:
                    # Unpinned on purpose: research falls back to the
                    # coding model (see providers.resolve_effective).
                    continue
                set_project_custom_model(self.project_root, role, entry)
        except OSError as e:
            self.notify(
                f"Could not write drcode.json: {e}",
                title="Model setup",
                severity="error",
                timeout=8,
            )
        else:
            self.notify(
                "Custom model saved for coding, tester and compaction "
                "(research follows the coding model)",
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
        if not self.resolved_config(AgentRole.CODING).is_ready:
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
        self._editing_task_id = None
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
        self._new_task_chrome()
        self._update_kind_label()
        self.refresh_task_status()
        self.new_task_input.focus()

    # -- editing a held task ---------------------------------------------------

    def _new_task_chrome(self) -> None:
        """Note + hint line of the new-task form, per mode: fresh creation
        vs editing a held task (the same editor, different wording)."""
        note = self.query_one("#nt-note", Static)
        hint = self.query_one("#nt-hint", Static)
        if self._editing_task_id is not None:
            note.update(
                "Editing a held task — its original text and queued comments "
                "are merged below; save it back on hold or into the queue."
            )
            hint.update(
                "tab — switch type   ctrl+enter — save & queue   "
                "ctrl+h — save on hold   esc — cancel"
            )
        else:
            note.update(
                "This is not a live chat with the agent — "
                "you are composing a task to be executed later."
            )
            hint.update(
                "tab — switch type   ctrl+enter — create   "
                "ctrl+h — create on hold   esc — cancel"
            )

    def action_edit_task(self) -> None:
        """Edit the condition of the viewed held task: the original text and
        every additional comment still queued in pending_input are merged
        into the new-task editor. On save the merged text replaces them (the
        separate comments are gone), the type can be changed like on
        creation, and the task either stays on hold or re-enters the queue."""
        task = self.viewed
        if task is None or task.status is not TaskStatus.HOLD:
            return
        self._editing_task_id = task.id
        self.new_task_input.load_text(self.manager.editable_condition(task.id))
        self.new_task_kind = task.kind
        self._new_task_mode = True
        self._new_task_prev_id = self.viewed_id
        self.viewed_id = None
        self.refresh_sidebar()
        self._clear_stream_label()
        self.chat_log.add_class("hidden")
        self.chat_input.add_class("hidden")
        self.query_one("#new-task", Vertical).remove_class("hidden")
        self._new_task_chrome()
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
        self._editing_task_id = None
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
        """Ctrl+Enter in the form: create a task (or save the edited held
        one) and put it into the queue."""
        self._submit_new_form(hold=False)

    def action_submit_new_task_hold(self) -> None:
        """Ctrl+H in the form: create the task directly on hold — or, while
        editing a held task, save it back on hold (still out of the
        pipeline). Same behaviour as save_task_edit_hold; the two actions
        exist so the footer can label the key per mode."""
        self._submit_new_form(hold=True)

    def action_save_task_edit_hold(self) -> None:
        """Ctrl+H while editing a held task: save it back on hold."""
        self._submit_new_form(hold=True)

    def _submit_new_form(self, *, hold: bool) -> None:
        if self._editing_task_id is not None:
            self._finish_task_edit(queue=not hold)
            return
        text = self.new_task_input.text.strip()
        if not text:
            return
        task = self.manager.create("", text, kind=self.new_task_kind, hold=hold)
        self._leave_new_task_mode()
        self.refresh_sidebar()
        self.view_task(task.id)
        self.chat_input.focus()
        # Starts immediately unless testing is pending, the queue is busy —
        # or the task was created on hold.
        self._advance()

    def _finish_task_edit(self, *, queue: bool) -> None:
        """Save the edited held task: the merged text replaces the old
        condition and comments; `queue` decides between re-entering the
        pipeline and staying parked on hold."""
        task_id = self._editing_task_id
        task = self.manager.tasks.get(task_id) if task_id else None
        if task is None:
            # The task vanished while the form was open (deleted elsewhere):
            # just close the editor.
            self._editing_task_id = None
            self.cancel_new_task()
            return
        text = self.new_task_input.text.strip()
        if not text:
            return
        self.manager.edit_task(task.id, text, kind=self.new_task_kind)
        self._editing_task_id = None
        self._leave_new_task_mode()
        if queue:
            # Back to where the task was parked from (its clarification, if
            # any, was dropped by edit_task — so this lands in the queue).
            self.manager.set_hold(task.id, on=False)
            self._advance()
        self.refresh_sidebar()
        self.view_task(task.id)
        self.chat_input.focus()

    def action_done_task(self) -> None:
        task = self.viewed
        if task is None or task.status is TaskStatus.COMPLETED:
            return
        self.workers.cancel_group(self, "agent")
        self.agent_running = False
        # The user pressed Done right here: the status observers
        # (notifications) must not report this transition.
        self.manager.mark_done(task.id, origin="user")
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

    def action_open_model_settings(self) -> None:
        """Open the "Model settings…" sub-menu: the provider catalog editor
        and one model picker per agent role, in one palette. Runs as the
        callback of a palette entry — the palette has already dismissed
        itself by the time this executes, so pushing the sub-palette is
        safe."""
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            ModelSettingsPalette(
                id="--model-settings-palette",
                providers=[ModelCatalogCommandProvider, RoleModelsCommandsProvider],
                placeholder="Model settings…",
            )
        )

    # -- per-role model selection ---------------------------------------------

    def action_open_model_catalog(self) -> None:
        """Open the provider/model catalog editor (user-level models.json)
        as a modal screen. Shared with the standalone `drcode-models`
        utility; when no catalog exists yet, the screen pre-fills itself
        from this project's custom models (saved only on explicit Save).
        On close the per-role resolutions are recomputed."""
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            ModelCatalogScreen(project_root=self.project_root),
            self._model_catalog_done,
        )

    def _model_catalog_done(self, _result: object) -> None:
        self.invalidate_resolved_configs()
        self.refresh_connection()

    def role_model_label(self, role: AgentRole) -> str:
        """Human-readable effective model of a role, for the Settings
        palette entries: the project slot (catalog reference or the role's
        custom model id), else the user-level default, else "not set".
        The research role falls back to the coding model when unpinned —
        the label says so instead of the bare "not set"."""
        slot = load_project_model_slots(self.project_root).get(role)
        if isinstance(slot, ModelRef):
            return f"{slot.provider}/{slot.model}"
        if slot == "custom":
            custom = load_project_custom_models(self.project_root).get(role)
            return custom.model if custom is not None else "not set"
        ref = UserModelsStore().load().defaults.get(role)
        if ref is not None:
            return f"{ref.provider}/{ref.model}"
        if role is AgentRole.RESEARCH:
            return f"{self.role_model_label(AgentRole.CODING)} (inherits coding)"
        return "not set"

    def action_change_coding_model(self) -> None:
        self._open_role_model_picker(AgentRole.CODING)

    def action_change_research_model(self) -> None:
        self._open_role_model_picker(AgentRole.RESEARCH)

    def action_change_tester_model(self) -> None:
        self._open_role_model_picker(AgentRole.TESTER)

    def action_change_compaction_model(self) -> None:
        self._open_role_model_picker(AgentRole.COMPACTION)

    def _open_role_model_picker(self, role: AgentRole) -> None:
        """Selection palette with the catalog models for a role (grouped by
        provider) plus the "Custom…" free-form entry."""
        if isinstance(self.screen, CommandPalette):
            return
        catalog = UserModelsStore().load()
        if not any(p.models for p in catalog.providers):
            # No catalog models to choose from: skip the (empty) list and
            # open the custom-model dialog right away. "Provider catalog…"
            # stays available from the main Settings menu.
            self.action_custom_role_model(role)
            return
        self.push_screen(
            SettingsPalette(
                id=f"--role-model-{role.value}",
                providers=[role_model_options_provider(role)],
                placeholder=f"Pick the {role.value} model…",
            )
        )

    def action_set_role_model(self, role: AgentRole, ref: ModelRef | None) -> None:
        """Persist one role's model choice into the project's
        drcode.json ("models" key) and apply it live."""
        try:
            set_project_model(self.project_root, role, ref)
        except OSError as e:
            self.notify(
                f"Could not write drcode.json: {e}",
                title="Model selection",
                severity="error",
                timeout=8,
            )
            return
        self.invalidate_resolved_configs()
        self.refresh_connection()
        label = f"{ref.provider}/{ref.model}" if ref else "user default"
        self.notify(
            f"{role.value} model: {label}",
            title="Model selection",
            timeout=4,
        )

    def action_activate_role_custom(self, role: AgentRole) -> None:
        """Re-activate the role's saved custom model (picker item
        "Custom: <model-id>"): only the `models[role]` slot flips back to
        "custom" — the stored custom model is reused as-is, no re-entry."""
        customs = load_project_custom_models(self.project_root)
        if role not in customs:
            # Broken/missing custom_models[role] (a slot pointing at it is
            # treated as "nothing selected"): never silently rewritten —
            # tell the user instead.
            self.notify(
                f"No valid custom model saved for the {role.value} role — "
                "use “Custom…” to configure one.",
                title="Model selection",
                severity="warning",
                timeout=8,
            )
            return
        try:
            set_project_custom_active(self.project_root, role)
        except OSError as e:
            self.notify(
                f"Could not write drcode.json: {e}",
                title="Model selection",
                severity="error",
                timeout=8,
            )
            return
        self.invalidate_resolved_configs()
        self.refresh_connection()
        self.notify(
            f"{role.value} model: {customs[role].model}",
            title="Model selection",
            timeout=4,
        )

    def action_custom_role_model(self, role: AgentRole) -> None:
        """Free-form model for a role: the ModelSetupDialog, pre-filled
        from the role's saved custom model (if any). The result is stored
        under `custom_models[role]` and activated (`models[role] =
        "custom"`) — custom models of different roles never mix."""
        if isinstance(self.screen, CommandPalette):
            return
        custom = load_project_custom_models(self.project_root).get(role)
        self.push_screen(
            ModelSetupDialog(
                model=custom.model if custom else "",
                base_url=custom.base_url if custom else "",
                api_key=custom.api_key if custom else "",
                max_context_tokens=(custom.max_context_tokens if custom else 200_000),
                max_tokens=(
                    custom.max_tokens if custom and custom.max_tokens else 8192
                ),
            ),
            partial(self._custom_role_model_done, role),
        )

    def _custom_role_model_done(
        self, role: AgentRole, setup: ModelSetup | None
    ) -> None:
        if setup is None:
            return
        try:
            set_project_custom_model(
                self.project_root, role, self._custom_entry_from_setup(setup)
            )
        except OSError as e:
            self.notify(
                f"Could not write drcode.json: {e}",
                title="Model selection",
                severity="error",
                timeout=8,
            )
            return
        self.invalidate_resolved_configs()
        self.refresh_connection()
        self.refresh_status()
        self._advance()
        self.notify(
            f"{role.value} model: {setup.model}",
            title="Model selection",
            timeout=4,
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

    def action_set_max_dev_rounds(self, rounds: int) -> None:
        """Settings command: how many consecutive failed development rounds
        (tester failures attributed to the task's own changes) a task gets
        before the failure completes it as failed and the tester creates a
        bug report instead. Persisted per project (.drcode/settings.json);
        applies from the next testing round on."""
        self.settings.max_dev_rounds = max(1, rounds)
        self.settings_store.save(self.settings)
        self.notify(
            f"Max development rounds: {self.settings.max_dev_rounds}",
            title="Project settings",
            timeout=4,
        )

    def action_change_max_dev_rounds(self) -> None:
        """Max-development-rounds picker, opened from the Settings entry
        ("Max development rounds: …"). Same UX as the retention picker: the
        narrowed SettingsPalette with the options; picking one applies it,
        Escape dismisses without changing anything.
        """
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--dev-rounds-palette",
                providers=[DevRoundsOptionsProvider],
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

    def _task_isolation_label(self) -> str:
        """The Settings-entry label of the task-isolation setting: the
        option's own label, plus the dormant marker when worktree is
        selected but the project has no git repository."""
        label = self.settings.task_isolation.label
        if (
            self.settings.task_isolation is TaskIsolation.WORKTREE
            and not self.task_env.available
        ):
            label += " — no git repo, inactive"
        return label

    def action_set_task_isolation(self, isolation: TaskIsolation) -> None:
        """Settings command: whether each task develops in its own git
        worktree (`drcode.task_env`) or directly in the project directory.
        Persisted per project (.drcode/settings.json); applies from the
        next task activation on (running tasks keep their root).

        A worktree value is saved even when the project has no git
        repository — the user's choice is never silently rewritten — but
        the mode stays dormant: tasks keep running in the project
        directory and a warning is shown here AND at every app startup
        until either a repository appears or the setting is turned off.
        """
        self.settings.task_isolation = isolation
        self.settings_store.save(self.settings)
        if isolation is TaskIsolation.WORKTREE and not self.task_env.available:
            self.notify(
                "Task isolation: worktree — saved, but this project has no "
                "git repository, so the mode will NOT work: tasks keep "
                "running in the project directory. Init git (git init) or "
                "switch the setting back to off.",
                title="Project settings",
                severity="warning",
                timeout=10,
            )
        else:
            self.notify(
                f"Task isolation: {isolation.label}",
                title="Project settings",
                timeout=4,
            )

    def action_change_task_isolation(self) -> None:
        """Task-isolation picker, opened from the Settings entry ("Task
        isolation: …"). Same UX as the retention picker: the narrowed
        SettingsPalette with the two options; picking one applies it,
        Escape dismisses without changing anything.
        """
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--task-isolation-palette",
                providers=[IsolationOptionsProvider],
                placeholder="Search for options…",
            )
        )

    def workflow_status_label(self) -> str:
        """One-line status of the custom team workflow for the Settings
        entry: "default" without the workflow files, otherwise what is
        active (from .drcode/git-workflow.md and .drcode/workflow.py)."""
        return describe_project_workflow(self.project_root)

    def action_workflow_info(self) -> None:
        """Settings entry ("Workflow: …") — informational: explains which
        workflow layers are active and where their files live. The workflow
        is enabled by file presence alone; there is nothing to toggle."""
        parts = [
            (
                "Git rules: .drcode/git-workflow.md — "
                f"{'present' if load_git_workflow(self.project_root) else 'not present'}"
            )
        ]
        hooks = self.workflow
        if hooks is None:
            parts.append("Hooks: .drcode/workflow.py — not present")
        elif hooks.load_error:
            parts.append(
                f"Hooks: .drcode/workflow.py — unparseable ({hooks.load_error})"
            )
        else:
            defined = ", ".join(hooks.defined_hooks()) or "no hook functions defined"
            parts.append(f"Hooks: .drcode/workflow.py — {defined}")
        parts.append(
            "Security: .drcode/workflow.py is arbitrary code running with "
            "your user permissions — never trust it from a repository you "
            "did not author."
        )
        self.notify(
            "\n".join(parts),
            title="Custom workflow",
            timeout=10,
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

    # -- notifications (ntfy.sh) -----------------------------------------------

    def action_change_notifications(self) -> None:
        """Notifications dialog, opened from the Settings entry
        ("Notifications: on/off"). Collects the enable switch and the
        ntfy.sh topic; Save persists the values into the project settings
        and fires a test notification when the feature is turned on, so the
        user can immediately check the subscription on the phone."""
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            NotificationsDialog(
                enabled=self.settings.notifications_enabled,
                topic=self.settings.ntfy_topic,
            ),
            self._notifications_settings_done,
        )

    def _notifications_settings_done(self, setup: NotificationsSetup | None) -> None:
        if setup is None:
            return
        self.settings.notifications_enabled = setup.enabled
        self.settings.ntfy_topic = setup.topic
        self.settings_store.save(self.settings)
        self.notify(
            f"Notifications: {'on' if setup.enabled else 'off'}"
            + (f" (ntfy.sh/{setup.topic})" if setup.enabled and setup.topic else ""),
            title="Project settings",
            timeout=4,
        )
        if setup.enabled and setup.topic:
            # A test push right away: the user sees the channel works (or
            # that the topic is wrong) without waiting for a real event.
            self.notifier.notify(
                "DrCode: notifications enabled",
                f'Project "{self.project_root.name}": test '
                "notification — the channel works.",
                tags=["bell"],
            )

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
            # Every successful cycle is reported (with the counts of what
            # was taken into the local task store), not only the cycles
            # that imported something: a silent "nothing new" sync is also
            # an event the user asked to be told about.
            self.notifier.github_sync(imported, skipped)
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
