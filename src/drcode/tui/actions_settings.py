"""Settings actions layer — navigation and project preferences.

The navigation actions (focus sidebar/input, archive search), the
Settings palette and every preference it edits (retention, verification,
the per-role git policies, the workflow presets, worktree removal, dev
rounds, stall nudge, keep-awake,
task isolation, parallel tasks, ESC-stop behavior), the custom-workflow
status, the GitHub/notifications/voice dialogs plus the GitHub sync
scheduler, and the theme picker.

Split out of :mod:`drcode.tui.actions` — see the composition in
``ActionsLayer``.
"""

from __future__ import annotations

import time

from textual.command import (
    CommandPalette,
)

from ..github import (
    GitHubClient,
    GitHubError,
    sync_issues,
)
from ..settings import (
    CompletedRetention,
    EscAbort,
    GitPolicy,
    ProjectSettings,
    StallNudge,
    TaskIsolation,
    Verification,
    WorktreeRemoval,
    parallel_tasks_label,
)
from ..workflow import describe_project_workflow, load_git_workflow
from .command_providers import (
    DevRoundsOptionsProvider,
    EscAbortOptionsProvider,
    GitResearchOptionsProvider,
    GitTesterOptionsProvider,
    IsolationOptionsProvider,
    ParallelTasksOptionsProvider,
    RetentionOptionsProvider,
    SettingsPalette,
    StallNudgeOptionsProvider,
    StartBranchOptionsProvider,
    ThemeOptionsProvider,
    VerificationOptionsProvider,
    WorkflowPreset,
    WorkflowPresetOptionsProvider,
    WorktreeRemovalOptionsProvider,
    workflow_preset_label,
)
from .github_dialog import GitHubSettingsDialog, GitHubSetup
from .notifications_dialog import NotificationsDialog, NotificationsSetup
from .state import DrCodeState
from .voice_dialog import VoiceSettingsDialog, VoiceSetup


class SettingsLayer(DrCodeState):
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

    def action_search_tasks(self) -> None:
        """Open the task-archive search dialog (Ctrl+F).

        The dialog receives the MAIN checkout's task storage root
        (`manager.storage.root`) — under worktree isolation the task files
        live in the main checkout's `.drcode/tasks/`, never in a task
        worktree (see the note in drcode.task_search's docstring)."""
        from .search_dialog import SearchDialog

        self.push_screen(SearchDialog(self.manager.storage.root))

    def action_settings(self) -> None:
        """Open the Settings modal (the command palette, Ctrl+S)."""
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(SettingsPalette(id="--command-palette"))

    def _apply_settings_side_effects(self) -> None:
        """The shared tail of EVERY task-settings mutation — the
        per-field ``action_set_*`` actions and the atomic
        Task-settings dialog alike (``action_apply_task_settings``):
        re-run the pipeline gate (verification / preset changes can
        complete tasks waiting for testing, a raised parallel limit can
        start waiting tasks) and re-evaluate the keep-awake assertion.
        One helper so the quick palettes and the dialog can never drift
        apart."""
        self._advance()
        self._update_keepawake()

    def action_apply_task_settings(self, draft: ProjectSettings) -> None:
        """Apply the Task-settings dialog's accepted draft ATOMICALLY:
        persist once, then replay the same side effects the per-field
        actions run (see `_apply_settings_side_effects`). The draft is a
        deep copy made by the dialog; it is merged into the LIVE settings
        object in place (TaskEnvironment holds the same instance)."""
        from .task_settings_dialog import copy_task_settings

        copy_task_settings(self.settings, draft)
        self.settings_store.save(self.settings)
        self.notify(
            "Task settings applied",
            title="Project settings",
            timeout=4,
        )
        self._apply_settings_side_effects()

    def _task_settings_done(self, result: ProjectSettings | None) -> None:
        if result is None:
            return  # ESC / Cancel — nothing persisted, nothing replayed.
        self.action_apply_task_settings(result)

    def action_open_task_settings(self) -> None:
        """Open the "Task settings…" dialog: all task-management project
        preferences in one modal form (workflow preset on top, live
        coupling hints at the bottom, atomic Apply). Runs as the callback
        of a palette entry — the palette has already dismissed itself by
        the time this executes, so pushing the dialog is safe."""
        if isinstance(self.screen, CommandPalette):
            return
        from .task_settings_dialog import TaskSettingsDialog

        self.push_screen(
            TaskSettingsDialog(self.settings),
            self._task_settings_done,
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

    def workflow_preset_label(self) -> str:
        """The Settings-entry label of the workflow preset: the name of the
        preset the settings currently match exactly, or "Custom" when any
        individual setting was changed away from every preset."""
        return workflow_preset_label(self.settings)

    def action_set_workflow_preset(self, preset: WorkflowPreset) -> None:
        """Settings command: apply a workflow preset — one named
        combination of verification, the per-role git policies, task
        isolation and worktree removal. The preset is an applier, not a
        persisted entity: only the fields it declares are written (the
        rest keep their current values) and the file keeps storing the
        individual settings."""
        preset.apply(self.settings)
        self.settings_store.save(self.settings)
        self.notify(
            f"Workflow preset: {preset.label}",
            title="Project settings",
            timeout=4,
        )
        # A preset that switches verification off lets the pipeline
        # complete any tasks waiting for testing on the next gate run.
        self._apply_settings_side_effects()

    def action_workflow_preset_custom(self) -> None:
        """Picked "Custom" in the preset picker: a no-op (Custom is the
        display state of settings matching no preset, not a choice)."""
        self.notify(
            "Workflow preset: Custom — change the individual settings "
            "below (isolation, verification, git policies, …).",
            title="Project settings",
            timeout=6,
        )

    def action_change_workflow_preset(self) -> None:
        """Workflow-preset picker, opened from the Settings entry
        ("Workflow preset: …"). Same UX as the retention picker: the
        narrowed SettingsPalette with the presets plus the Custom marker;
        picking a preset applies it, Escape dismisses without changing
        anything."""
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--workflow-preset-palette",
                providers=[WorkflowPresetOptionsProvider],
                placeholder="Search for options…",
            )
        )

    def action_set_verification(self, verification: Verification) -> None:
        """Settings command: whether the QA tester round runs at all.
        Persisted per project (.drcode/settings.json); switching to "none"
        lets the pipeline complete any tasks currently waiting for testing
        on the next _advance(). The tester's git policy stays persisted
        but dormant while no tester runs (a startup warning says so)."""
        self.settings.verification = verification
        self.settings_store.save(self.settings)
        if (
            verification is Verification.NONE
            and self.settings.git_policy.tester is not GitPolicy.OFF
        ):
            self.notify(
                "Verification: none — the tester's Git policy "
                f"({self.settings.git_policy.tester.label}) is now "
                "dormant: no tester runs to commit anything. Set Git "
                "(tester): off to silence this warning.",
                title="Project settings",
                severity="warning",
                timeout=10,
            )
        else:
            self.notify(
                f"Verification: {verification.label}",
                title="Project settings",
                timeout=4,
            )
        self._apply_settings_side_effects()

    def action_change_verification(self) -> None:
        """Verification picker, opened from the Settings entry
        ("Verification: …"). Same UX as the retention picker: the narrowed
        SettingsPalette with the two options; picking one applies it,
        Escape dismisses without changing anything.
        """
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--verification-palette",
                providers=[VerificationOptionsProvider],
                placeholder="Search for options…",
            )
        )

    def action_set_git_tester(self, policy: GitPolicy) -> None:
        """Settings command: what the tester may do to git after a passed
        verification (off / local commit / commit + push). Persisted per
        project (.drcode/settings.json); applies to testing rounds from
        now on. Dormant while Verification is none."""
        self.settings.git_policy.tester = policy
        self.settings_store.save(self.settings)
        self.notify(
            f"Git (tester): {policy.label}",
            title="Project settings",
            timeout=4,
        )

    def action_change_git_tester(self) -> None:
        """Tester git-policy picker, opened from the Settings entry
        ("Git (tester): …"). Same UX as the retention picker: the narrowed
        SettingsPalette with the three policies; picking one applies it,
        Escape dismisses without changing anything.
        """
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--git-tester-palette",
                providers=[GitTesterOptionsProvider],
                placeholder="Search for options…",
            )
        )

    def action_set_git_research(self, policy: GitPolicy) -> None:
        """Settings command: what the research agent's git_commit_docs may
        do (off / local commit / commit + push). Persisted per project
        (.drcode/settings.json); applies to research tasks from now on.
        Under "off" the tool is not registered and the prompt promises no
        commit step; edited files stay tracked for the report."""
        self.settings.git_policy.research = policy
        self.settings_store.save(self.settings)
        self.notify(
            f"Git (research): {policy.label}",
            title="Project settings",
            timeout=4,
        )

    def action_change_git_research(self) -> None:
        """Research git-policy picker, opened from the Settings entry
        ("Git (research): …"). Same UX as the retention picker: the
        narrowed SettingsPalette with the three policies; picking one
        applies it, Escape dismisses without changing anything.
        """
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--git-research-palette",
                providers=[GitResearchOptionsProvider],
                placeholder="Search for options…",
            )
        )

    def action_set_worktree_removal(self, removal: WorktreeRemoval) -> None:
        """Settings command: when the per-task worktree directory
        (.drcode/worktrees/<id>) is removed automatically — with the task
        (cancel / delete / retention purge), as soon as the task reaches
        COMPLETED, or never. Persisted per project
        (.drcode/settings.json); applies to transitions from now on.

        Only the DIRECTORY is governed: the branch `drcode/<id>` with its
        committed work survives every completion, a branch holding
        unmerged commits is never deleted (see `drcode.task_env`), and a
        re-activated completed task gets its worktree recreated from the
        branch."""
        self.settings.worktree_removal = removal
        self.settings_store.save(self.settings)
        self.notify(
            f"Worktree removal: {removal.label}",
            title="Project settings",
            timeout=4,
        )

    def action_change_worktree_removal(self) -> None:
        """Worktree-removal picker, opened from the Settings entry
        ("Worktree removal: …"). Same UX as the retention picker: the
        narrowed SettingsPalette with the three options; picking one
        applies it, Escape dismisses without changing anything.
        """
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--worktree-removal-palette",
                providers=[WorktreeRemovalOptionsProvider],
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

    def action_toggle_keep_awake(self) -> None:
        """Settings command: whether the computer is kept awake (a
        system-sleep assertion) while the pipeline has work — any task
        QUEUED/ACTIVE/TESTING or a run in flight. Persisted per project
        (.drcode/settings.json); the change applies immediately (the next
        _update_keepawake tick releases the assertion when switched off)."""
        self.settings.keep_awake = not self.settings.keep_awake
        self.settings_store.save(self.settings)
        self._apply_settings_side_effects()
        state = "on" if self.settings.keep_awake else "off"
        self.notify(
            f"Keep computer awake while working: {state}",
            title="Project settings",
            timeout=4,
        )

    def _task_isolation_label(self) -> str:
        """The Settings-entry label of the task-isolation setting: the
        option's own label, plus the dormant marker when worktree is
        selected but the project has no git repository, plus the
        parallel-slot count when more than one is configured."""
        label = self.settings.task_isolation.label
        if (
            self.settings.task_isolation is TaskIsolation.WORKTREE
            and not self.task_env.available
        ):
            label += " — no git repo, inactive"
        if self.settings.parallel_tasks > 1:
            label += f" — parallel: {self.settings.parallel_tasks}"
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

    def _isolation_start_branch_label(self) -> str:
        """The Settings-entry label of the isolation-start-branch setting:
        "auto (HEAD branch)" when unset, else the configured branch name."""
        return self.settings.isolation_start_branch or "auto (HEAD branch)"

    def action_set_isolation_start_branch(self, branch: str) -> None:
        """Settings command: the branch NEW task branches are forked from
        under worktree isolation (empty = the main checkout's current
        branch, with a detached HEAD refusing the isolation instead of
        forking blindly). Persisted per project (.drcode/settings.json);
        applies to the next task activation. A branch that does not exist
        is persisted anyway (the user's choice is never silently
        rewritten) with a warning — each activation would then fall back
        to the auto mode and say so in the task's service-event log."""
        branch = branch.strip()
        self.settings.isolation_start_branch = branch
        self.settings_store.save(self.settings)
        label = branch or "auto (HEAD branch)"
        if (
            branch
            and self.task_env.available
            and not self.task_env._branch_exists(branch)
        ):
            self.notify(
                f"Isolation start branch: {label} — saved, but no such "
                "branch exists in this repository: task branches keep "
                "forking from the main checkout's current branch (with "
                "the detached-HEAD refusal) until it does.",
                title="Project settings",
                severity="warning",
                timeout=10,
            )
            return
        self.notify(
            f"Isolation start branch: {label}",
            title="Project settings",
            timeout=4,
        )

    def action_change_isolation_start_branch(self) -> None:
        """Isolation-start-branch picker, opened from the "Task settings…"
        entry ("Isolation start branch: …"). Lists the auto mode plus the
        repository's local branches (discovered by git, off the event
        loop); picking one applies it, Escape keeps the current value."""
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--isolation-start-branch-palette",
                providers=[StartBranchOptionsProvider],
                placeholder="Search for options…",
            )
        )

    def action_set_parallel_tasks(self, count: int) -> None:
        """Settings command: how many tasks may develop in parallel
        (development slots). Persisted per project (.drcode/
        settings.json); applies from the next promotion on. Only
        meaningful under worktree isolation — every parallel task needs
        its own worktree, so without a usable isolation the picker
        refuses values above 1 (the persisted value stays; the effective
        limit is 1 and the app warns at startup)."""
        count = max(1, count)
        if count > 1 and not self.task_env.active:
            self.notify(
                "Parallel tasks needs Task isolation: worktree — each "
                "parallel task runs in its own git worktree, and this "
                "project has none. Set Task isolation first; the limit "
                "stays 1.",
                title="Project settings",
                severity="warning",
                timeout=10,
            )
            return
        self.settings.parallel_tasks = count
        self.settings_store.save(self.settings)
        self.notify(
            f"Parallel tasks: {parallel_tasks_label(count)}",
            title="Project settings",
            timeout=4,
        )
        # A raised limit can start waiting tasks right away.
        self._advance()

    def action_change_parallel_tasks(self) -> None:
        """Parallel-tasks picker, opened from the Settings entry
        ("Parallel tasks: …"). Same UX as the retention picker: the
        narrowed SettingsPalette with the options; picking one applies
        it, Escape dismisses without changing anything.
        """
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--parallel-tasks-palette",
                providers=[ParallelTasksOptionsProvider],
                placeholder="Search for options…",
            )
        )

    def action_set_esc_abort(self, mode: EscAbort) -> None:
        """Settings command: how ESC stops a running agent/tester
        (double-press confirmation vs the historical instant stop).
        Persisted per project (.drcode/settings.json); applies to the
        very next ESC press."""
        self.settings.esc_abort = mode
        self.settings_store.save(self.settings)
        self.notify(
            f"ESC stop behavior: {mode.label}",
            title="Project settings",
            timeout=4,
        )

    def action_change_esc_abort(self) -> None:
        """ESC-stop-behavior picker, opened from the Settings entry
        ("ESC stop behavior: …"). Same UX as the retention picker: the
        narrowed SettingsPalette with the options; picking one applies
        it, Escape dismisses without changing anything.
        """
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            SettingsPalette(
                id="--esc-abort-palette",
                providers=[EscAbortOptionsProvider],
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

    # -- voice (ASR dictation + TTS announcements) ------------------------------

    def voice_status_label(self) -> str:
        """Short status of the voice features for the Settings entry:
        which halves are configured (dictation needs an ASR model,
        announcements a TTS model; an empty base URL switches everything
        off)."""
        s = self.settings
        if not s.voice_api_url.strip():
            return "off"
        parts: list[str] = []
        if s.voice_asr_model.strip():
            parts.append("dictation")
        if s.voice_tts_model.strip():
            parts.append("announcements")
        return ", ".join(parts) if parts else "not configured"

    def action_change_voice(self) -> None:
        """Voice settings dialog, opened from the Settings entry
        ("Voice…"). Collects the audio-API connection, the ASR/TTS models
        and the announcement checkboxes; Save persists the values into the
        project settings. The API key field is pre-filled from the
        user-catalog provider whose host matches the audio URL."""
        if isinstance(self.screen, CommandPalette):
            return
        from ..voice import catalog_api_key_for_host

        settings = self.settings
        api_url = settings.voice_api_url
        self.push_screen(
            VoiceSettingsDialog(
                api_url=api_url,
                api_key=settings.voice_api_key,
                # Prefill the key only while the user has not stored their
                # own: after that the explicit value wins on every open.
                catalog_api_key=(
                    "" if settings.voice_api_key else catalog_api_key_for_host(api_url)
                ),
                asr_model=settings.voice_asr_model,
                tts_model=settings.voice_tts_model,
                tts_voice=settings.voice_tts_voice,
                tts_completed=settings.voice_tts_completed,
                tts_tester_failed=settings.voice_tts_tester_failed,
                tts_queue_empty=settings.voice_tts_queue_empty,
                language=settings.voice_language,
            ),
            self._voice_settings_done,
        )

    def _voice_settings_done(self, setup: VoiceSetup | None) -> None:
        if setup is None:
            return
        self.settings.voice_api_url = setup.api_url
        self.settings.voice_api_key = setup.api_key
        self.settings.voice_asr_model = setup.asr_model
        self.settings.voice_tts_model = setup.tts_model
        self.settings.voice_tts_voice = setup.tts_voice
        self.settings.voice_tts_completed = setup.tts_completed
        self.settings.voice_tts_tester_failed = setup.tts_tester_failed
        self.settings.voice_tts_queue_empty = setup.tts_queue_empty
        self.settings.voice_language = setup.language
        self.settings_store.save(self.settings)
        enabled = bool(setup.api_url and (setup.asr_model or setup.tts_model))
        self.notify(
            f"Voice: {'on' if enabled else 'off'}",
            title="Project settings",
            timeout=4,
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
