"""Task-form actions layer — the new-task / edit-task editor.

Everything the new-task form does: entering and leaving the form, the
kind (feature/bug/research) cycling with its label and descriptions, the
"Depends on" field, submitting a fresh task or saving an edited held one.
Also the first-start model-setup result handling and the AGENTS.md
startup offer (both create/queue tasks through the same form machinery).

Split out of :mod:`drcode.tui.actions` — see the composition in
``ActionsLayer``.
"""

from __future__ import annotations

from typing import ClassVar

from rich.markup import escape
from textual.containers import Vertical
from textual.widgets import Static

from ..models import TaskKind, TaskStatus
from ..prompts import load_init_repo_prompt
from ..providers import AgentRole, set_project_custom_model
from ..task_manager import validate_dependencies
from .actions_models import custom_entry_from_setup
from .agents_dialog import AgentsMdDialog
from .inputs import DepsInput, NewTaskInput
from .setup_dialog import ModelSetup
from .state import DrCodeState


class TaskFormLayer(DrCodeState):
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
        entry = custom_entry_from_setup(setup)
        try:
            for role in AgentRole:
                if role in (AgentRole.RESEARCH, AgentRole.RESOLVER):
                    # Unpinned on purpose: research and the merge-conflict
                    # resolver fall back to the coding model (see
                    # providers.resolve_effective).
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
                "(research and the conflict resolver follow the coding model)",
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

    def _drop_stream_preview(self, task_id: str, generation: int) -> None:
        """Clear the streaming preview after an agent run ends, but only if
        no newer run of the SAME task has (already) taken over the label."""
        if self._agent_generations.get(task_id) == generation:
            self._clear_stream_label()

    def action_new_task(self) -> None:
        # Every entry into the new-task screen starts from a clean editor.
        if not self._new_task_mode:
            # Entering the form is a context change: a dictation started
            # in the chat input must not outlive it.
            self.voice_input.cancel(silent=True)
        self._editing_task_id = None
        self.new_task_input.clear()
        self._clear_deps_field()
        # A fresh form has no preset goal (the GoalScreen's T sets it
        # AFTER entering the form).
        self._new_task_goal_id = None
        self.query_one("#nt-goal", Static).update("")
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
        self.refresh_todo_panel()
        self.new_task_input.focus()
        # After the focus move (the cursor may only be kept while the
        # sidebar itself is focused): drop the ListView cursor that was
        # parked on the previously viewed row — with viewed_id cleared it
        # would light that row in the cursor colour while composing.
        self._sync_sidebar_cursor(None)

    # Goal capture lives in tui.actions_goals (GoalsActionsLayer): the
    # plain Ctrl+Y dialog grew a parent preset and a task-link context
    # shared with the GoalScreen.

    # -- editing a held task ---------------------------------------------------

    def _render_chat_hint(self) -> None:
        """Dim key-hint line under the chat editor (#chat-hint): Send and
        Voice live here instead of the footer — a key with a permanent
        affordance next to its field does not take a footer slot. The
        voice part appears only while dictation is configured (everything
        empty = voice off, see voice_dictation_configured)."""
        text = "enter — send · shift+enter — newline"
        if self.voice_dictation_configured:
            text += " · ctrl+b — voice"
        self.query_one("#chat-hint", Static).update(text)

    def _new_task_chrome(self) -> None:
        """Note + hint line of the new-task form, per mode: fresh creation,
        editing a held task or finishing an agent-created draft (the same
        editor, different wording)."""
        note = self.query_one("#nt-note", Static)
        hint = self.query_one("#nt-hint", Static)
        # Push-to-talk is bound in the form's editor too; advertised only
        # while it can actually record (see _render_chat_hint).
        voice = "   ctrl+b — voice" if self.voice_dictation_configured else ""
        editing = (
            self.manager.tasks.get(self._editing_task_id)
            if self._editing_task_id
            else None
        )
        if editing is not None and editing.status is TaskStatus.DRAFT:
            note.update(
                "Finishing an agent-created draft — its text is loaded below; "
                "send it into the queue or park it on hold. Esc keeps the draft."
            )
            hint.update(
                "tab — switch type   ctrl+g — depends on   "
                "ctrl+enter — finish & queue   ctrl+h — finish on hold   "
                "esc — keep draft" + voice
            )
        elif self._editing_task_id is not None:
            note.update(
                "Editing a held task — its original text and queued comments "
                "are merged below; save it back on hold or into the queue."
            )
            hint.update(
                "tab — switch type   ctrl+g — depends on   "
                "ctrl+enter — save & queue   ctrl+h — save on hold   esc — cancel"
                + voice
            )
        else:
            note.update(
                "This is not a live chat with the agent — "
                "you are composing a task to be executed later."
            )
            hint.update(
                "tab — switch type   ctrl+g — depends on   ctrl+enter — create   "
                "ctrl+h — create on hold   ctrl+d — create as draft   esc — cancel"
                + voice
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
        # Opening the editor is a context change: drop any dictation
        # started in the chat input (the editor reuses the new-task form).
        self.voice_input.cancel(silent=True)
        self._editing_task_id = task.id
        self.new_task_input.load_text(self.manager.editable_condition(task.id))
        self._load_deps_field(task.depends_on)
        # The edit path never re-links the task: show its current goal as
        # read-only info (the Link/Unlink palette commands own the link).
        self._new_task_goal_id = None
        goal_label = self.query_one("#nt-goal", Static)
        if task.goal_id and task.goal_id in self.manager.goals:
            goal = self.manager.goals.get(task.goal_id)
            goal_label.update(
                f"goal: [magenta]◆[/] {escape(goal.title_display)} (kept)"
            )
        else:
            goal_label.update("")
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
        self.refresh_todo_panel()
        self.new_task_input.focus()
        # See action_new_task: drop the cursor parked on the edited task's
        # row — nothing is viewed while the editor is open.
        self._sync_sidebar_cursor(None)

    def action_open_draft(self, task_id: str | None = None) -> None:
        """Open an agent-created DRAFT task in the new-task editor (the
        same form action_edit_task reuses) instead of the chat view: a
        draft has no conversation yet — the user is meant to finish its
        composition here. On save the draft enters the queue (Ctrl+Enter)
        or is parked on hold (Ctrl+H); ESC leaves it a draft.

        Called by view_task's draft redirect (sidebar clicks, mentions,
        the startup view pick) and the Ctrl+E binding."""
        task = self.manager.tasks.get(task_id) if task_id else self.viewed
        if task is None or task.status is not TaskStatus.DRAFT:
            return
        # Opening the editor is a context change: drop any dictation
        # started in the chat input (the editor reuses the new-task form).
        self.voice_input.cancel(silent=True)
        self._editing_task_id = task.id
        self.new_task_input.load_text(self.manager.editable_condition(task.id))
        self._load_deps_field(task.depends_on)
        # The draft path never re-links the task: show its current goal as
        # read-only info (the Link/Unlink palette commands own the link).
        self._new_task_goal_id = None
        goal_label = self.query_one("#nt-goal", Static)
        if task.goal_id and task.goal_id in self.manager.goals:
            goal = self.manager.goals.get(task.goal_id)
            goal_label.update(
                f"goal: [magenta]◆[/] {escape(goal.title_display)} (kept)"
            )
        else:
            goal_label.update("")
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
        self.refresh_todo_panel()
        self.new_task_input.focus()
        # See action_new_task: drop the cursor parked on the draft's row —
        # nothing is viewed while the editor is open.
        self._sync_sidebar_cursor(None)

    def _leave_new_task_mode(self) -> None:
        # Leaving the form (cancel, submit, sidebar click) is a context
        # change: a dictation started in the form's editors is discarded.
        self.voice_input.cancel(silent=True)
        self._new_task_mode = False
        self.query_one("#new-task", Vertical).add_class("hidden")
        self.chat_log.remove_class("hidden")
        # The preset goal link (the GoalScreen's T) belongs to the form
        # session: consumed by the submit, dropped by a cancel.
        self._new_task_goal_id = None
        self.query_one("#nt-goal", Static).update("")
        # Content may be stale (a run streamed while the form was open, or
        # finished in the background): don't resurrect it.
        self._clear_stream_label()
        self.chat_input.remove_class("hidden")
        self.refresh_task_status()
        self.refresh_todo_panel()

    def cancel_new_task(self) -> None:
        self._editing_task_id = None
        self._leave_new_task_mode()
        # The chat log still shows the task that was open before the form:
        # restore its view so the status bar matches the visible content
        # (action_new_task() cleared viewed_id, leaving the bar empty).
        if self.viewed_id is None:
            prev = self._new_task_prev_id
            task = self.manager.tasks.get(prev) if prev else None
            if task is not None and task.status is TaskStatus.DRAFT:
                # A draft re-opens its editor (view_task's redirect) — ESC
                # would trap the user in the form forever. Never restore
                # to a draft; prefer a non-draft task or an empty pane.
                task = None
            if task is None:
                task = self._sidebar_top_task(skip_drafts=True)
            if task is not None:
                self.view_task(task.id)
            else:
                self._sync_sidebar_cursor(None)
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

    # -- the "Depends on" field of the new-task form ---------------------------

    def _clear_deps_field(self) -> None:
        self.query_one("#nt-deps", DepsInput).value = ""
        self.query_one("#nt-deps-note", Static).update("")

    def _load_deps_field(self, deps: list[str]) -> None:
        self.query_one("#nt-deps", DepsInput).value = ", ".join(deps)
        self.query_one("#nt-deps-note", Static).update("")

    def _deps_from_form(self) -> list[str] | None:
        """Parse + validate the form's Depends-on field.

        Returns the cleaned id list, or None on a violation — the error is
        shown in the form (#nt-deps-note) so the user can fix it without
        losing the composed text. Validation mirrors
        TaskManager.set_dependencies (existence, no self-reference, no
        cycles); for a fresh creation the task id is still unknown (None —
        self-reference and cycles are impossible then)."""
        raw = self.query_one("#nt-deps", DepsInput).value
        deps = [part for part in raw.replace(",", " ").split()]
        if not deps:
            self.query_one("#nt-deps-note", Static).update("")
            return []
        try:
            cleaned = validate_dependencies(
                self.manager.tasks, self._editing_task_id, deps
            )
        except ValueError as e:
            self.query_one("#nt-deps-note", Static).update(escape(str(e)))
            return None
        self.query_one("#nt-deps-note", Static).update("")
        return cleaned

    def action_submit_new_task_hold(self) -> None:
        """Ctrl+H in the form: create the task directly on hold — or, while
        editing a held task, save it back on hold (still out of the
        pipeline). Same behaviour as save_task_edit_hold; the two actions
        exist so the footer can label the key per mode."""
        self._submit_new_form(hold=True)

    def action_save_task_edit_hold(self) -> None:
        """Ctrl+H while editing a held task: save it back on hold."""
        self._submit_new_form(hold=True)

    def action_submit_new_task_draft(self) -> None:
        """Ctrl+D in the form: create the task as a DRAFT — parked in the
        sidebar's DRAFTS block, never picked up by the pipeline until the
        user opens it (Ctrl+E) and finishes it. Fresh creation only: a
        held task cannot become a draft, and a draft being finished is
        already one (ESC keeps it)."""
        self._submit_new_form(hold=False, draft=True)

    def _submit_new_form(self, *, hold: bool, draft: bool = False) -> None:
        if self._editing_task_id is not None:
            self._finish_task_edit(queue=not hold)
            return
        text = self.new_task_input.text.strip()
        if not text:
            return
        # Validate the dependencies BEFORE creating: an invalid list keeps
        # the form open with the error shown, nothing is created.
        deps = self._deps_from_form()
        if deps is None:
            return
        task = self.manager.create(
            "",
            text,
            kind=self.new_task_kind,
            hold=hold,
            draft=draft,
            goal_id=self._new_task_goal_id,
        )
        if deps:
            self.manager.set_dependencies(task.id, deps)
        if draft:
            # A draft never enters the pipeline and has no chat view:
            # leave the form and restore the pre-form view (cancel_new_task
            # skips drafts — no redirect trap), the draft waits in the
            # sidebar's DRAFTS block.
            self.cancel_new_task()
            self.refresh_sidebar()
            self.notify(
                "Draft created — open it (ctrl+e) to finish and queue it",
                title="New task",
                timeout=6,
            )
            return
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
        deps = self._deps_from_form()
        if deps is None:
            return
        was_draft = task.status is TaskStatus.DRAFT
        self.manager.edit_task(task.id, text, kind=self.new_task_kind)
        self.manager.set_dependencies(task.id, deps)
        self._editing_task_id = None
        self._leave_new_task_mode()
        if was_draft:
            # An agent-created draft: the edit finished its composition —
            # Ctrl+Enter sends it into the queue, Ctrl+H parks it on hold
            # (ESC never reaches here: it cancels and keeps the draft).
            self.manager.finish_draft(task.id, hold=not queue)
            if queue:
                self._advance()
        elif queue:
            # Back to where the task was parked from (its clarification, if
            # any, was dropped by edit_task — so this lands in the queue).
            self.manager.set_hold(task.id, on=False)
            self._advance()
        self.refresh_sidebar()
        self.view_task(task.id)
        self.chat_input.focus()

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
