"""Settings palette, its sub-menus and the providers behind them.

Everything reachable from the "Settings" palette: the main palette
(``SettingsPalette``) with its narrowed layout and deterministic entry
order, the "Model settings…" and "Task settings…" sub-menu palettes, the
providers that supply the main-menu entries and the per-option selection
palettes for each project preference.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, ClassVar, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.command import (
    CommandInput,
    CommandPalette,
    DiscoveryHit,
    Hit,
    SearchIcon,
)
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.widgets import Button, LoadingIndicator

from ...settings import (
    DEV_ROUNDS_OPTIONS,
    ESC_ABORT_OPTIONS,
    GIT_POLICY_OPTIONS,
    ISOLATION_OPTIONS,
    PARALLEL_TASKS_OPTIONS,
    RETENTION_OPTIONS,
    STALL_NUDGE_OPTIONS,
    SUBAGENTS_OPTIONS,
    VERIFICATION_OPTIONS,
    WORKTREE_REMOVAL_OPTIONS,
    WORKTREE_SNAPSHOT_OPTIONS,
    GitPolicy,
    ProjectSettings,
    TaskIsolation,
    Verification,
    WorktreeRemoval,
    dev_rounds_label,
    parallel_tasks_label,
    subagents_label,
)
from .base import CommandEntry, CommandListProvider, SpacedCommandList

if TYPE_CHECKING:
    from ..app import DrCodeApp


def _marked_option(
    label: str, current: bool, callback: Callable[[], object]
) -> CommandEntry:
    """One enum-option palette entry; the active option is marked with
    "(current)" (picking it is a no-op re-apply, Escape keeps it)."""
    return CommandEntry(label + (" (current)" if current else ""), callback)


# -- Workflow presets ----------------------------------------------------------
#
# A preset is a multi-field applier over the project settings, NOT a
# persisted entity: the file keeps storing the individual fields, and any
# manual change of one of them flips the palette display to "Custom"
# (`workflow_preset_for` finds no fully-matching preset). A preset touches
# ONLY the fields it declares (None = leave the current value as-is).


@dataclass(frozen=True)
class WorkflowPreset:
    """One named combination of the workflow-related project settings.

    Fields left as None are not touched when the preset is applied and
    are ignored by the match check."""

    key: str
    label: str
    description: str
    verification: Verification | None = None
    git_tester: GitPolicy | None = None
    git_research: GitPolicy | None = None
    isolation: TaskIsolation | None = None
    worktree_removal: WorktreeRemoval | None = None

    def matches(self, settings: ProjectSettings) -> bool:
        """Whether the settings currently equal every field this preset
        declares (fields it does not declare are irrelevant)."""
        return (
            (self.verification is None or settings.verification is self.verification)
            and (
                self.git_tester is None or settings.git_policy.tester is self.git_tester
            )
            and (
                self.git_research is None
                or settings.git_policy.research is self.git_research
            )
            and (self.isolation is None or settings.task_isolation is self.isolation)
            and (
                self.worktree_removal is None
                or settings.worktree_removal is self.worktree_removal
            )
        )

    def apply(self, settings: ProjectSettings) -> None:
        """Write the declared fields into the settings (the caller saves)."""
        if self.verification is not None:
            settings.verification = self.verification
        if self.git_tester is not None:
            settings.git_policy.tester = self.git_tester
        if self.git_research is not None:
            settings.git_policy.research = self.git_research
        if self.isolation is not None:
            settings.task_isolation = self.isolation
        if self.worktree_removal is not None:
            settings.worktree_removal = self.worktree_removal


WORKFLOW_PRESETS: tuple[WorkflowPreset, ...] = (
    WorkflowPreset(
        key="local_sandbox",
        label="Local sandbox",
        description=(
            "Tester on; everything commits locally and never pushes; "
            "each task in its own worktree, removed with the task"
        ),
        verification=Verification.TESTER,
        git_tester=GitPolicy.LOCAL,
        git_research=GitPolicy.LOCAL,
        isolation=TaskIsolation.WORKTREE,
        worktree_removal=WorktreeRemoval.ON_DELETE,
    ),
    WorkflowPreset(
        key="team_pr_flow",
        label="Team / PR flow",
        description=(
            "Tester commits and pushes both agents' branches to the "
            "remote (pull-request flow); each task in its own worktree"
        ),
        git_tester=GitPolicy.PUSH,
        git_research=GitPolicy.PUSH,
        isolation=TaskIsolation.WORKTREE,
    ),
    WorkflowPreset(
        key="minimal",
        label="Minimal",
        description=(
            "No tester, no git at all: tasks complete right after "
            "development, changes stay uncommitted in the project "
            "directory (research answers questions / edits docs "
            "without a commit step)"
        ),
        verification=Verification.NONE,
        git_tester=GitPolicy.OFF,
        git_research=GitPolicy.OFF,
    ),
)


def workflow_preset_for(settings: ProjectSettings) -> WorkflowPreset | None:
    """The preset the current settings exactly match, or None ("Custom")."""
    return next((p for p in WORKFLOW_PRESETS if p.matches(settings)), None)


def workflow_preset_label(settings: ProjectSettings) -> str:
    """The "Workflow preset:" palette-entry value for the settings."""
    preset = workflow_preset_for(settings)
    return preset.label if preset is not None else "Custom"


class SettingsPalette(CommandPalette):
    """Command palette ("Settings") modal.

    The stock CommandPalette stretches over the full screen width; this
    subclass narrows its container to about half of the screen, but never
    below 60 columns (on a narrow terminal 60 wins and may overflow). It
    also swaps in a SpacedCommandList so palette entries are separated by a
    blank line.

    Entry order is fixed, not alphabetical and not provider-dependent:
    Textual streams the hits of all providers through one shared queue in
    completion order, and `DrCodeApp.COMMANDS` is a set — so without the
    re-sort below the menu order would differ from run to run. Instead the
    aggregated hits are ordered by `ENTRY_ORDER` (see its comment); in
    search mode Textual's relevance sort (`_refresh_command_list`) still
    wins, this only fixes the discovery (empty-query) listing.
    """

    DEFAULT_CSS = """
    SettingsPalette #--container {
        width: 50%;
        min-width: 60;
    }
    """

    # Logical display order of the Settings entries — the single source of
    # truth for the menu layout. Entries are matched top-to-bottom by name
    # *prefix* (dynamic values like "…: on" are appended after the colon, so
    # prefixes must include it where present). Reordering lines here — or
    # adding prefixes for new entries — rearranges the menu arbitrarily;
    # anything not matching a prefix slots in right before the last entry
    # ("Quit"). Head block per the current UX convention: Keys, Screenshot,
    # Theme, then the "Find task…" action; Quit is pinned to the very end.
    # The model-related entries
    # (provider catalog + per-role picks) live in the "Model settings…"
    # sub-menu (ModelSettingsPalette), and the six task-management
    # preferences live in the "Task settings…" sub-menu
    # (TaskSettingsPalette) — not here.
    ENTRY_ORDER: ClassVar[tuple[str, ...]] = (
        "Keys",
        "Screenshot",
        "Theme",
        "Find task…",
        "Update/create AGENTS.md",
        "Model settings",
        "ESC stop behavior:",
        "Task settings",
        "Workflow:",
        "GitHub integration:",
        "Notifications:",
        "Voice:",
        "Maximize",
        "Minimize",
        "Quit",
    )

    def _entry_rank(self, name: str) -> int:
        """Position of an entry in the menu: index of the first matching
        `ENTRY_ORDER` prefix, or just before the last entry for anything
        unknown."""
        for rank, prefix in enumerate(self.ENTRY_ORDER):
            if name.startswith(prefix):
                return rank
        return max(0, len(self.ENTRY_ORDER) - 1)

    @staticmethod
    def _hit_name(hit: DiscoveryHit | Hit) -> str:
        """Plain-text name of a hit (the prompt can be Text/Content)."""
        display = hit.prompt
        if isinstance(display, str):
            return display
        if isinstance(display, Text | Content):
            return display.plain
        return str(display)

    async def _search_for(
        self, search_value: str
    ) -> AsyncGenerator[DiscoveryHit | Hit, bool]:
        # Buffer every hit from all providers, then re-emit them in
        # ENTRY_ORDER. The parent streams hits from concurrently running
        # provider tasks in completion order; buffering costs nothing here
        # (all providers are local and instant) and makes the listing
        # deterministic. Cancellation still works: the gather loop checks
        # `worker.is_cancelled` itself, and closing this generator closes
        # the parent one via `aclose`.
        hits: list[DiscoveryHit | Hit] = []
        async for hit in super()._search_for(search_value):
            hits.append(hit)
        hits.sort(key=lambda hit: self._entry_rank(self._hit_name(hit)))
        for hit in hits:
            yield hit

    def compose(self) -> ComposeResult:
        # Same as CommandPalette.compose, but with the spaced command list.
        with Vertical(id="--container"):
            with Horizontal(id="--input"):
                yield SearchIcon()
                yield CommandInput(placeholder=self._placeholder, select_on_focus=False)
                if not self.run_on_select:
                    yield Button("▶")
            with Vertical(id="--results"):
                yield SpacedCommandList()
                yield LoadingIndicator()


class ModelSettingsPalette(SettingsPalette):
    """Sub-menu of the Settings palette with the model-related entries.

    Opened from the single "Model settings…" entry of the main Settings
    menu: the user-level provider catalog editor plus one model picker per
    agent role. Same narrowed layout and spaced list as the parent; the
    entry order is its own (model entries only), so the shared prefix-sort
    in `SettingsPalette._entry_rank` orders this menu independently.
    """

    # Same prefix-matching contract as SettingsPalette.ENTRY_ORDER, but
    # scoped to this sub-menu: the provider-catalog entry first, then the
    # four role pickers in pipeline order.
    ENTRY_ORDER: ClassVar[tuple[str, ...]] = (
        "Provider catalog…",
        "Coding model:",
        "Research model:",
        "Tester model:",
        "Compaction model:",
    )


class TaskSettingsPalette(SettingsPalette):
    """Sub-menu of the Settings palette with the task-management entries.

    Opened from the single "Task settings…" entry of the main Settings
    menu: the project preferences that govern how tasks behave, grouped
    into three labeled sections — Workflow (preset, isolation, parallel
    tasks, worktree removal), Testing (verification, dev rounds, stall
    nudge) and Git (tester/research policies, retention). Same narrowed
    layout and spaced list as the parent; the entry order is its own
    (task entries only), so the shared prefix-sort in
    `SettingsPalette._entry_rank` orders this menu independently.
    """

    # Same prefix-matching contract as SettingsPalette.ENTRY_ORDER, but
    # scoped to this sub-menu: the task-management preferences in their
    # fixed logical order, grouped by section (the section labels live
    # only in the docstrings/help texts — the palette is a flat list).
    ENTRY_ORDER: ClassVar[tuple[str, ...]] = (
        # -- Workflow --
        "Workflow preset:",
        "Task isolation:",
        "Isolation start branch:",
        "Parallel tasks:",
        "Worktree removal:",
        "Subagents (coder):",
        "Subagents (researcher):",
        # -- Testing --
        "Verification:",
        "Max development rounds:",
        "Nudge stalled agent after:",
        # -- Git --
        "Git (tester):",
        "Git (research):",
        "Auto-delete completed tasks:",
        "Keep computer awake:",
    )


class ServiceCommandsProvider(CommandListProvider):
    """App-specific commands in the Settings palette.

    Registered alongside the system commands via `DrCodeApp.COMMANDS`;
    every hit is an app action — most create a pre-configured service
    task, the ESC-stop entry toggles a project preference.
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        return [
            CommandEntry(
                "Find task…",
                app.action_search_tasks,
                "Search the task archive by keywords — title, prompt, "
                "edited files, the agent's final summary (the same "
                "dialog as Ctrl+F)",
            ),
            CommandEntry(
                "Update/create AGENTS.md",
                app.action_update_agents_md,
                "research task: agent creates or refreshes AGENTS.md",
            ),
            CommandEntry(
                f"ESC stop behavior: {app.settings.esc_abort.label}",
                app.action_change_esc_abort,
                "How ESC stops a running agent/tester: double-press "
                "(the first ESC only arms the stop, the second "
                "confirms) or instant — the historical single-press "
                "stop (project setting)",
            ),
        ]


class ModelCatalogCommandProvider(CommandListProvider):
    """Provider-catalog entry in the "Model settings…" sub-menu.

    Always visible (independent of any configuration state): opens the
    user-level provider/model catalog editor
    (`~/.config/drcode/models.json`).
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        return [
            CommandEntry(
                "Provider catalog…",
                app.action_open_model_catalog,
                "Edit the user-level catalog of OpenAI-compatible providers "
                "and their models (auto-discovery, context sizes, per-role "
                "defaults) — stored in ~/.config/drcode/models.json",
            ),
        ]


class ModelSettingsCommandProvider(CommandListProvider):
    """The single "Model settings…" entry in the main Settings palette.

    Opens the ModelSettingsPalette sub-menu with the provider catalog
    editor and the per-role model pickers.
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        return [
            CommandEntry(
                "Model settings…",
                app.action_open_model_settings,
                "Provider catalog and per-agent model selection "
                "(coding / research / tester / compaction)",
            ),
        ]


class TaskSettingsCommandProvider(CommandListProvider):
    """The single "Task settings…" entry in the main Settings palette.

    Opens the TaskSettingsPalette sub-menu with the seven project
    preferences that govern task behavior (retention, finalization,
    worktree removal, dev rounds, stall nudge, isolation, parallelism).
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        return [
            CommandEntry(
                "Task settings…",
                app.action_open_task_settings,
                "Task-management preferences: auto-delete, "
                "finalization, worktree removal, dev rounds, stall "
                "nudge, isolation, parallelism",
            ),
        ]


class ProjectSettingsCommandsProvider(CommandListProvider):
    """Project-preference commands in the Settings palette.

    One entry per project preference; its current value is shown after the
    ":". Picking an entry opens the option-selection palette (same UX as the
    theme picker); choosing an option there persists it into
    `<project>/.drcode/settings.json`, so the choice is scoped to the
    project, not the user. Escape dismisses the selection window and keeps
    the current value.

    The six task-management preferences (retention, finalization, dev
    rounds, stall nudge, isolation, parallelism) are not here — they live
    in the "Task settings…" sub-menu (TaskSettingsPalette), whose entries
    are provided by TaskSettingsEntriesProvider.
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        return [
            CommandEntry(
                f"Workflow: {app.workflow_status_label()}",
                app.action_workflow_info,
                "Custom branching/review rules: .drcode/git-workflow.md "
                "(prose rules for the agents) and .drcode/workflow.py "
                "(lifecycle hooks, incl. external code review) — press "
                "for details",
            ),
            CommandEntry(
                f"GitHub integration: {'on' if app.settings.github_enabled else 'off'}",
                app.action_change_github,
                "Periodically import open GitHub issues as tasks: bug → "
                "bug report, enhancement → feature request (project "
                "setting)",
            ),
            CommandEntry(
                "Notifications: "
                f"{'on' if app.settings.notifications_enabled else 'off'}",
                app.action_change_notifications,
                "Push notifications about important pipeline events "
                "(task status changes, GitHub sync, an idle pipeline) "
                "via ntfy.sh (project setting)",
            ),
            CommandEntry(
                f"Voice: {app.voice_status_label()}",
                app.action_change_voice,
                "Speech recognition (push-to-talk dictation into the "
                "input fields) and spoken announcements, via an "
                "OpenAI-compatible audio API (project setting)",
            ),
        ]


class TaskSettingsEntriesProvider(CommandListProvider):
    """The task-management entries of the "Task settings…" sub-menu,
    grouped into three sections: Workflow (preset, isolation, parallel
    tasks, worktree removal), Testing (verification, dev rounds, stall
    nudge) and Git (tester/research policies, retention).

    One entry per project preference; its current value is shown after the
    ":". Picking an entry opens the option-selection palette (same UX as the
    theme picker); choosing an option there persists it into
    `<project>/.drcode/settings.json`, so the choice is scoped to the
    project, not the user. Escape dismisses the selection window and keeps
    the current value.
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        return [
            CommandEntry(
                f"Workflow preset: {app.workflow_preset_label()}",
                app.action_change_workflow_preset,
                "Apply a named combination of the workflow settings "
                "(verification, per-role git policies, isolation, "
                "worktree removal) in one pick — Local sandbox, "
                "Team / PR flow or Minimal; any manual change of an "
                "individual setting flips this display to Custom",
            ),
            # -- Workflow --
            CommandEntry(
                f"Task isolation: {app._task_isolation_label()}",
                app.action_change_task_isolation,
                "Give each task its own git worktree (.drcode/"
                "worktrees/<id>, branch drcode/<id>) so an interrupted "
                "task never leaves its changes to the next one — needs "
                "a git repository, otherwise tasks run in the shared "
                "project directory (project setting)",
            ),
            CommandEntry(
                f"Isolation start branch: {app._isolation_start_branch_label()}",
                app.action_change_isolation_start_branch,
                "Which commit NEW task branches are forked from under "
                "worktree isolation: auto (the main checkout's current "
                "branch — a detached HEAD then refuses the isolation "
                "instead of forking from an unknown commit) or a named "
                "branch such as main, used even when the checkout is "
                "detached (project setting)",
            ),
            CommandEntry(
                f"Parallel tasks: {parallel_tasks_label(app.settings.parallel_tasks)}",
                app.action_change_parallel_tasks,
                "How many tasks may develop in parallel, each in its "
                "own git worktree — needs Task isolation: worktree, "
                "otherwise tasks run one at a time (project setting; "
                "the tester stays single)",
            ),
            CommandEntry(
                f"Worktree removal: {app.settings.worktree_removal.label}",
                app.action_change_worktree_removal,
                "When the per-task worktree directory (.drcode/"
                "worktrees/<id>) is deleted automatically: with the "
                "task, on task completion, or never — the branch "
                "drcode/<id> with the committed work always survives "
                "(project setting)",
            ),
            CommandEntry(
                f"Worktree snapshot: {app.settings.worktree_snapshot.label}",
                app.action_change_worktree_snapshot,
                "When a settling task's worktree gets the automatic "
                "snapshot commit on drcode/<id>: always, only when the "
                "run made no commits of its own (a deliberate uncommitted "
                "tail is kept — and any removal keeps the worktree with "
                "it), or never (project setting)",
            ),
            CommandEntry(
                f"Subagents (coder): {subagents_label(app.settings.subagents_coder)}",
                app.action_change_subagents_coder,
                "How many read-only research subagents the CODING agent "
                "may run in parallel via its delegate_research tool — "
                "each returns only its final report, saving the agent's "
                "context; off removes the tool entirely (project "
                "setting)",
            ),
            CommandEntry(
                f"Subagents (researcher): "
                f"{subagents_label(app.settings.subagents_researcher)}",
                app.action_change_subagents_researcher,
                "The same delegation cap for the RESEARCH agent's runs "
                "(its resolved model serves the subagents; the counter "
                "is per run) — off removes the delegate_research tool "
                "(project setting)",
            ),
            # -- Testing --
            CommandEntry(
                f"Verification: {app.settings.verification.label}",
                app.action_change_verification,
                "Whether the QA tester round runs after development: "
                "tester (verify before completion) or none (tasks "
                "complete right away — project setting)",
            ),
            CommandEntry(
                f"Max development rounds: {app.settings.max_dev_rounds}",
                app.action_change_max_dev_rounds,
                "How many consecutive failed development rounds a task "
                "gets when the tester attributes the failure to the "
                "task's own changes, before the failure completes the "
                "task as failed and a bug report is created instead "
                "(project setting)",
            ),
            CommandEntry(
                f"Nudge stalled agent after: {app.settings.stall_nudge.label}",
                app.action_change_stall_nudge,
                "Abort and restart an agent/tester run that has shown no "
                "activity for this long (project setting)",
            ),
            # -- Git --
            CommandEntry(
                f"Git (tester): {app.settings.git_policy.tester.label}",
                app.action_change_git_tester,
                "What the QA tester may do to git after a passed "
                "verification: commit + push, local commit only, or "
                "off (verify without committing — project setting; "
                "dormant while Verification is none)",
            ),
            CommandEntry(
                f"Git (research): {app.settings.git_policy.research.label}",
                app.action_change_git_research,
                "What the research agent's git_commit_docs may do: "
                "commit + push, local commit only, or off (no commit "
                "step at all — edited files are still tracked for the "
                "report; project setting)",
            ),
            CommandEntry(
                f"Auto-delete completed tasks: "
                f"{app.settings.completed_retention.label}",
                app.action_change_completed_retention,
                "Delete completed tasks older than the retention period at app "
                "startup (project setting)",
            ),
            CommandEntry(
                f"Keep computer awake: {'on' if app.settings.keep_awake else 'off'}",
                app.action_toggle_keep_awake,
                "Hold a system-sleep assertion while the pipeline has "
                "work (any task queued/active/testing or a run in "
                "flight) — a sleeping machine tears the network down "
                "and kills the runs (project setting; does not stop "
                "the lid-closed clamshell sleep)",
            ),
        ]


class RetentionOptionsProvider(CommandListProvider):
    """Options of the completed-tasks auto-cleanup setting.

    Shown in the selection palette opened from the single Settings entry;
    picking an option applies it immediately, Escape dismisses the palette
    without changing anything. The currently active option is marked.
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.completed_retention
        return [
            _marked_option(
                option.label,
                option is current,
                partial(app.action_set_completed_retention, option),
            )
            for option in RETENTION_OPTIONS
        ]


class VerificationOptionsProvider(CommandListProvider):
    """Options of the "Verification" setting (tester / none).

    Same selection-palette UX as RetentionOptionsProvider: pick an option to
    apply it immediately, Escape keeps the current one. The active option is
    marked with "(current)".
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.verification
        return [
            _marked_option(
                option.label,
                option is current,
                partial(app.action_set_verification, option),
            )
            for option in VERIFICATION_OPTIONS
        ]


class GitTesterOptionsProvider(CommandListProvider):
    """Options of the "Git (tester)" setting (off / local / push).

    Same selection-palette UX as RetentionOptionsProvider: pick an option to
    apply it immediately, Escape keeps the current one. The active option is
    marked with "(current)".
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.git_policy.tester
        return [
            _marked_option(
                option.label,
                option is current,
                partial(app.action_set_git_tester, option),
            )
            for option in GIT_POLICY_OPTIONS
        ]


class GitResearchOptionsProvider(CommandListProvider):
    """Options of the "Git (research)" setting (off / local / push).

    Same selection-palette UX as RetentionOptionsProvider: pick an option to
    apply it immediately, Escape keeps the current one. The active option is
    marked with "(current)".
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.git_policy.research
        return [
            _marked_option(
                option.label,
                option is current,
                partial(app.action_set_git_research, option),
            )
            for option in GIT_POLICY_OPTIONS
        ]


class WorkflowPresetOptionsProvider(CommandListProvider):
    """Options of the "Workflow preset" applier (Local sandbox / Team /
    PR flow / Minimal / Custom).

    Picking a preset applies its declared fields immediately (the current
    value elsewhere is kept); "Custom" is a no-op marker for settings that
    match no preset. Escape keeps everything as-is.
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        entries = [
            CommandEntry(preset.label, partial(app.action_set_workflow_preset, preset))
            for preset in WORKFLOW_PRESETS
        ]
        entries.append(CommandEntry("Custom", app.action_workflow_preset_custom))
        return entries


class DevRoundsOptionsProvider(CommandListProvider):
    """Options of the "Max development rounds" setting (the fix-loop cap).

    Same selection-palette UX as RetentionOptionsProvider: pick an option to
    apply it immediately, Escape keeps the current one. The active option is
    marked with "(current)".
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.max_dev_rounds
        return [
            _marked_option(
                dev_rounds_label(rounds),
                rounds == current,
                partial(app.action_set_max_dev_rounds, rounds),
            )
            for rounds in DEV_ROUNDS_OPTIONS
        ]


class IsolationOptionsProvider(CommandListProvider):
    """Options of the "Task isolation" setting (off / git worktree).

    Same selection-palette UX as RetentionOptionsProvider: pick an option
    to apply it immediately, Escape keeps the current one. The active
    option is marked with "(current)".
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.task_isolation
        return [
            _marked_option(
                option.label,
                option is current,
                partial(app.action_set_task_isolation, option),
            )
            for option in ISOLATION_OPTIONS
        ]


class ParallelTasksOptionsProvider(CommandListProvider):
    """Options of the "Parallel tasks" setting (the development-slot
    limit, 1–4).

    Same selection-palette UX as RetentionOptionsProvider: pick an option
    to apply it immediately, Escape keeps the current one. The active
    option is marked with "(current)".
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.parallel_tasks
        return [
            _marked_option(
                parallel_tasks_label(count),
                count == current,
                partial(app.action_set_parallel_tasks, count),
            )
            for count in PARALLEL_TASKS_OPTIONS
        ]


class WorktreeRemovalOptionsProvider(CommandListProvider):
    """Options of the "Worktree removal" setting (with the task / on
    completion / never).

    Same selection-palette UX as RetentionOptionsProvider: pick an option
    to apply it immediately, Escape keeps the current one. The active
    option is marked with "(current)".
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.worktree_removal
        return [
            _marked_option(
                option.label,
                option is current,
                partial(app.action_set_worktree_removal, option),
            )
            for option in WORKTREE_REMOVAL_OPTIONS
        ]


class WorktreeSnapshotOptionsProvider(CommandListProvider):
    """Options of the "Worktree snapshot" setting (always / only if no
    commits / never).

    Same selection-palette UX as WorktreeRemovalOptionsProvider: pick an
    option to apply it immediately, Escape keeps the current one. The
    active option is marked with "(current)".
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.worktree_snapshot
        return [
            _marked_option(
                option.label,
                option is current,
                partial(app.action_set_worktree_snapshot, option),
            )
            for option in WORKTREE_SNAPSHOT_OPTIONS
        ]


class SubagentsCoderOptionsProvider(CommandListProvider):
    """Options of the "Subagents (coder)" setting (off / 1–4 concurrent
    read-only subagents the coding agent may spawn via
    delegate_research).

    Same selection-palette UX as RetentionOptionsProvider: pick an option
    to apply it immediately, Escape keeps the current one. The active
    option is marked with "(current)".
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.subagents_coder
        return [
            _marked_option(
                subagents_label(count),
                count == current,
                partial(app.action_set_subagents_coder, count),
            )
            for count in SUBAGENTS_OPTIONS
        ]


class SubagentsResearcherOptionsProvider(CommandListProvider):
    """Options of the "Subagents (researcher)" setting (off / 1–4
    concurrent read-only subagents the research agent may spawn via
    delegate_research).

    Same selection-palette UX as RetentionOptionsProvider: pick an option
    to apply it immediately, Escape keeps the current one. The active
    option is marked with "(current)".
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.subagents_researcher
        return [
            _marked_option(
                subagents_label(count),
                count == current,
                partial(app.action_set_subagents_researcher, count),
            )
            for count in SUBAGENTS_OPTIONS
        ]


class StartBranchOptionsProvider(CommandListProvider):
    """Options of the "Isolation start branch" setting: the auto mode
    ("auto (HEAD branch)" — fork new task branches from the main
    checkout's current branch, refusing a detached HEAD) plus every local
    branch git knows about (fork from THAT branch's tip always, even with
    the main checkout detached mid-rebase).

    Unlike the enum-valued pickers the options are discovered by git at
    palette-open time (off the event loop); picking an option applies it
    immediately, Escape keeps the current value."""

    AUTO_LABEL = "auto (HEAD branch)"

    async def _entries(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.isolation_start_branch
        branches: list[str] = []
        if app.task_env.available:
            branches = await asyncio.to_thread(app.task_env.local_branches)
        entries = [
            CommandEntry(
                self.AUTO_LABEL + (" (current)" if not current else ""),
                partial(app.action_set_isolation_start_branch, ""),
            )
        ]
        entries.extend(
            CommandEntry(
                branch + (" (current)" if branch == current else ""),
                partial(app.action_set_isolation_start_branch, branch),
            )
            for branch in branches
        )
        return entries


class EscAbortOptionsProvider(CommandListProvider):
    """Options of the "ESC stop behavior" setting (double-press confirm /
    instant stop).

    Same selection-palette UX as RetentionOptionsProvider: pick an option
    to apply it immediately, Escape keeps the current one. The active
    option is marked with "(current)".
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.esc_abort
        return [
            _marked_option(
                option.label,
                option is current,
                partial(app.action_set_esc_abort, option),
            )
            for option in ESC_ABORT_OPTIONS
        ]


class StallNudgeOptionsProvider(CommandListProvider):
    """Options of the "Nudge stalled agent" setting.

    Same selection-palette UX as RetentionOptionsProvider: pick an option to
    apply it immediately, Escape keeps the current one. The active option is
    marked with "(current)".
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.stall_nudge
        return [
            _marked_option(
                option.label,
                option is current,
                partial(app.action_set_stall_nudge, option),
            )
            for option in STALL_NUDGE_OPTIONS
        ]
