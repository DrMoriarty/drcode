"""Settings palette, its sub-menus and the providers behind them.

Everything reachable from the "Settings" palette: the main palette
(``SettingsPalette``) with its narrowed layout and deterministic entry
order, the "Model settings…" and "Task settings…" sub-menu palettes, the
providers that supply the main-menu entries and the per-option selection
palettes for each project preference.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from functools import partial
from typing import TYPE_CHECKING, ClassVar, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.command import (
    CommandInput,
    CommandPalette,
    DiscoveryHit,
    Hit,
    Hits,
    Provider,
    SearchIcon,
)
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.widgets import Button, LoadingIndicator

from ...settings import (
    DEV_ROUNDS_OPTIONS,
    ESC_ABORT_OPTIONS,
    FINALIZATION_OPTIONS,
    ISOLATION_OPTIONS,
    PARALLEL_TASKS_OPTIONS,
    RETENTION_OPTIONS,
    STALL_NUDGE_OPTIONS,
    WORKTREE_REMOVAL_OPTIONS,
    dev_rounds_label,
    parallel_tasks_label,
)
from .base import SpacedCommandList

if TYPE_CHECKING:
    from ..app import DrCodeApp


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
    menu: the seven project preferences that govern how tasks behave
    (retention, finalization, worktree removal, dev rounds, stall nudge,
    isolation, parallelism). Same narrowed layout and spaced list as the
    parent; the entry order is its own (task entries only), so the shared
    prefix-sort in `SettingsPalette._entry_rank` orders this menu
    independently.
    """

    # Same prefix-matching contract as SettingsPalette.ENTRY_ORDER, but
    # scoped to this sub-menu: the seven task-management preferences in
    # their fixed logical order.
    ENTRY_ORDER: ClassVar[tuple[str, ...]] = (
        "Auto-delete completed tasks:",
        "Task finalization:",
        "Worktree removal:",
        "Max development rounds:",
        "Nudge stalled agent after:",
        "Task isolation:",
        "Parallel tasks:",
    )


class ServiceCommandsProvider(Provider):
    """App-specific commands in the Settings palette.

    Registered alongside the system commands via `DrCodeApp.COMMANDS`;
    every hit is an app action — most create a pre-configured service
    task, the ESC-stop entry toggles a project preference.
    """

    def _commands(self) -> list[tuple[str, str, Callable[[], object]]]:
        app = cast("DrCodeApp", self.app)
        return [
            (
                "Find task…",
                (
                    "Search the task archive by keywords — title, prompt, "
                    "edited files, the agent's final summary (the same "
                    "dialog as Ctrl+F)"
                ),
                app.action_search_tasks,
            ),
            (
                "Update/create AGENTS.md",
                "research task: agent creates or refreshes AGENTS.md",
                app.action_update_agents_md,
            ),
            (
                f"ESC stop behavior: {app.settings.esc_abort.label}",
                (
                    "How ESC stops a running agent/tester: double-press "
                    "(the first ESC only arms the stop, the second "
                    "confirms) or instant — the historical single-press "
                    "stop (project setting)"
                ),
                app.action_change_esc_abort,
            ),
        ]

    async def discover(self) -> Hits:
        for name, help_text, callback in self._commands():
            yield DiscoveryHit(name, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, help_text, callback in self._commands():
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name), callback, help=help_text)


class ModelCatalogCommandProvider(Provider):
    """Provider-catalog entry in the "Model settings…" sub-menu.

    Always visible (independent of any configuration state): opens the
    user-level provider/model catalog editor
    (`~/.config/drcode/models.json`).
    """

    def _commands(self) -> list[tuple[str, str, Callable[[], object]]]:
        app = cast("DrCodeApp", self.app)
        return [
            (
                "Provider catalog…",
                (
                    "Edit the user-level catalog of OpenAI-compatible providers "
                    "and their models (auto-discovery, context sizes, per-role "
                    "defaults) — stored in ~/.config/drcode/models.json"
                ),
                app.action_open_model_catalog,
            ),
        ]

    async def discover(self) -> Hits:
        for name, help_text, callback in self._commands():
            yield DiscoveryHit(name, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, help_text, callback in self._commands():
            if (match := matcher.match(name)) > 0:
                yield Hit(match, matcher.highlight(name), callback, help=help_text)


class ModelSettingsCommandProvider(Provider):
    """The single "Model settings…" entry in the main Settings palette.

    Opens the ModelSettingsPalette sub-menu with the provider catalog
    editor and the per-role model pickers.
    """

    def _commands(self) -> list[tuple[str, str, Callable[[], object]]]:
        app = cast("DrCodeApp", self.app)
        return [
            (
                "Model settings…",
                (
                    "Provider catalog and per-agent model selection "
                    "(coding / research / tester / compaction)"
                ),
                app.action_open_model_settings,
            ),
        ]

    async def discover(self) -> Hits:
        for name, help_text, callback in self._commands():
            yield DiscoveryHit(name, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, help_text, callback in self._commands():
            if (match := matcher.match(name)) > 0:
                yield Hit(match, matcher.highlight(name), callback, help=help_text)


class TaskSettingsCommandProvider(Provider):
    """The single "Task settings…" entry in the main Settings palette.

    Opens the TaskSettingsPalette sub-menu with the seven project
    preferences that govern task behavior (retention, finalization,
    worktree removal, dev rounds, stall nudge, isolation, parallelism).
    """

    def _commands(self) -> list[tuple[str, str, Callable[[], object]]]:
        app = cast("DrCodeApp", self.app)
        return [
            (
                "Task settings…",
                (
                    "Task-management preferences: auto-delete, "
                    "finalization, worktree removal, dev rounds, stall "
                    "nudge, isolation, parallelism"
                ),
                app.action_open_task_settings,
            ),
        ]

    async def discover(self) -> Hits:
        for name, help_text, callback in self._commands():
            yield DiscoveryHit(name, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, help_text, callback in self._commands():
            if (match := matcher.match(name)) > 0:
                yield Hit(match, matcher.highlight(name), callback, help=help_text)


class ProjectSettingsCommandsProvider(Provider):
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

    def _commands(self) -> list[tuple[str, str, Callable[[], object]]]:
        app = cast("DrCodeApp", self.app)
        return [
            (
                f"Workflow: {app.workflow_status_label()}",
                (
                    "Custom branching/review rules: .drcode/git-workflow.md "
                    "(prose rules for the agents) and .drcode/workflow.py "
                    "(lifecycle hooks, incl. external code review) — press "
                    "for details"
                ),
                app.action_workflow_info,
            ),
            (
                (
                    "GitHub integration: "
                    f"{'on' if app.settings.github_enabled else 'off'}"
                ),
                (
                    "Periodically import open GitHub issues as tasks: bug → "
                    "bug report, enhancement → feature request (project "
                    "setting)"
                ),
                app.action_change_github,
            ),
            (
                (
                    "Notifications: "
                    f"{'on' if app.settings.notifications_enabled else 'off'}"
                ),
                (
                    "Push notifications about important pipeline events "
                    "(task status changes, GitHub sync, an idle pipeline) "
                    "via ntfy.sh (project setting)"
                ),
                app.action_change_notifications,
            ),
            (
                f"Voice: {app.voice_status_label()}",
                (
                    "Speech recognition (push-to-talk dictation into the "
                    "input fields) and spoken announcements, via an "
                    "OpenAI-compatible audio API (project setting)"
                ),
                app.action_change_voice,
            ),
        ]

    async def discover(self) -> Hits:
        for name, help_text, callback in self._commands():
            yield DiscoveryHit(name, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, help_text, callback in self._commands():
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name), callback, help=help_text)


class TaskSettingsEntriesProvider(Provider):
    """The seven task-management entries of the "Task settings…" sub-menu.

    One entry per project preference; its current value is shown after the
    ":". Picking an entry opens the option-selection palette (same UX as the
    theme picker); choosing an option there persists it into
    `<project>/.drcode/settings.json`, so the choice is scoped to the
    project, not the user. Escape dismisses the selection window and keeps
    the current value.
    """

    def _commands(self) -> list[tuple[str, str, Callable[[], object]]]:
        app = cast("DrCodeApp", self.app)
        return [
            (
                f"Auto-delete completed tasks: {app.settings.completed_retention.label}",
                (
                    "Delete completed tasks older than the retention period at app "
                    "startup (project setting)"
                ),
                app.action_change_completed_retention,
            ),
            (
                f"Task finalization: {app.settings.task_finalization.label}",
                (
                    "What runs after development: tester + git commit, tester "
                    "without commit, or nothing (project setting)"
                ),
                app.action_change_task_finalization,
            ),
            (
                f"Worktree removal: {app.settings.worktree_removal.label}",
                (
                    "When the per-task worktree directory (.drcode/"
                    "worktrees/<id>) is deleted automatically: with the "
                    "task, on task completion, or never — the branch "
                    "drcode/<id> with the committed work always survives "
                    "(project setting)"
                ),
                app.action_change_worktree_removal,
            ),
            (
                f"Max development rounds: {app.settings.max_dev_rounds}",
                (
                    "How many consecutive failed development rounds a task "
                    "gets when the tester attributes the failure to the "
                    "task's own changes, before the failure completes the "
                    "task as failed and a bug report is created instead "
                    "(project setting)"
                ),
                app.action_change_max_dev_rounds,
            ),
            (
                f"Nudge stalled agent after: {app.settings.stall_nudge.label}",
                (
                    "Abort and restart an agent/tester run that has shown no "
                    "activity for this long (project setting)"
                ),
                app.action_change_stall_nudge,
            ),
            (
                f"Task isolation: {app._task_isolation_label()}",
                (
                    "Give each task its own git worktree (.drcode/"
                    "worktrees/<id>, branch drcode/<id>) so an interrupted "
                    "task never leaves its changes to the next one — needs "
                    "a git repository, otherwise tasks run in the shared "
                    "project directory (project setting)"
                ),
                app.action_change_task_isolation,
            ),
            (
                f"Parallel tasks: {parallel_tasks_label(app.settings.parallel_tasks)}",
                (
                    "How many tasks may develop in parallel, each in its "
                    "own git worktree — needs Task isolation: worktree, "
                    "otherwise tasks run one at a time (project setting; "
                    "the tester stays single)"
                ),
                app.action_change_parallel_tasks,
            ),
            (
                f"Keep computer awake: {'on' if app.settings.keep_awake else 'off'}",
                (
                    "Hold a system-sleep assertion while the pipeline has "
                    "work (any task queued/active/testing or a run in "
                    "flight) — a sleeping machine tears the network down "
                    "and kills the runs (project setting; does not stop "
                    "the lid-closed clamshell sleep)"
                ),
                app.action_toggle_keep_awake,
            ),
        ]

    async def discover(self) -> Hits:
        for name, help_text, callback in self._commands():
            yield DiscoveryHit(name, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, help_text, callback in self._commands():
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name), callback, help=help_text)


class RetentionOptionsProvider(Provider):
    """Options of the completed-tasks auto-cleanup setting.

    Shown in the selection palette opened from the single Settings entry;
    picking an option applies it immediately, Escape dismisses the palette
    without changing anything. The currently active option is marked.
    """

    def _commands(self) -> list[tuple[str, Callable[[], None]]]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.completed_retention
        return [
            (
                option.label + (" (current)" if option is current else ""),
                partial(app.action_set_completed_retention, option),
            )
            for option in RETENTION_OPTIONS
        ]

    async def discover(self) -> Hits:
        for name, callback in self._commands():
            yield DiscoveryHit(name, callback)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, callback in self._commands():
            if (match := matcher.match(name)) > 0:
                yield Hit(match, matcher.highlight(name), callback)


class FinalizationOptionsProvider(Provider):
    """Options of the "Task finalization" setting.

    Same selection-palette UX as RetentionOptionsProvider: pick an option to
    apply it immediately, Escape keeps the current one. The active option is
    marked with "(current)".
    """

    def _commands(self) -> list[tuple[str, Callable[[], None]]]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.task_finalization
        return [
            (
                option.label + (" (current)" if option is current else ""),
                partial(app.action_set_task_finalization, option),
            )
            for option in FINALIZATION_OPTIONS
        ]

    async def discover(self) -> Hits:
        for name, callback in self._commands():
            yield DiscoveryHit(name, callback)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, callback in self._commands():
            if (match := matcher.match(name)) > 0:
                yield Hit(match, matcher.highlight(name), callback)


class DevRoundsOptionsProvider(Provider):
    """Options of the "Max development rounds" setting (the fix-loop cap).

    Same selection-palette UX as RetentionOptionsProvider: pick an option to
    apply it immediately, Escape keeps the current one. The active option is
    marked with "(current)".
    """

    def _commands(self) -> list[tuple[str, Callable[[], None]]]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.max_dev_rounds
        return [
            (
                dev_rounds_label(rounds) + (" (current)" if rounds == current else ""),
                partial(app.action_set_max_dev_rounds, rounds),
            )
            for rounds in DEV_ROUNDS_OPTIONS
        ]

    async def discover(self) -> Hits:
        for name, callback in self._commands():
            yield DiscoveryHit(name, callback)

    async def search(self, query: str) -> Hits:
        for name, callback in self._commands():
            if (match := self.matcher(query).match(name)) > 0:
                yield Hit(match, self.matcher(query).highlight(name), callback)


class IsolationOptionsProvider(Provider):
    """Options of the "Task isolation" setting (off / git worktree).

    Same selection-palette UX as RetentionOptionsProvider: pick an option
    to apply it immediately, Escape keeps the current one. The active
    option is marked with "(current)".
    """

    def _commands(self) -> list[tuple[str, Callable[[], None]]]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.task_isolation
        return [
            (
                option.label + (" (current)" if option is current else ""),
                partial(app.action_set_task_isolation, option),
            )
            for option in ISOLATION_OPTIONS
        ]

    async def discover(self) -> Hits:
        for name, callback in self._commands():
            yield DiscoveryHit(name, callback)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, callback in self._commands():
            if (match := matcher.match(name)) > 0:
                yield Hit(match, matcher.highlight(name), callback)


class ParallelTasksOptionsProvider(Provider):
    """Options of the "Parallel tasks" setting (the development-slot
    limit, 1–4).

    Same selection-palette UX as RetentionOptionsProvider: pick an option
    to apply it immediately, Escape keeps the current one. The active
    option is marked with "(current)".
    """

    def _commands(self) -> list[tuple[str, Callable[[], None]]]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.parallel_tasks
        return [
            (
                parallel_tasks_label(count)
                + (" (current)" if count == current else ""),
                partial(app.action_set_parallel_tasks, count),
            )
            for count in PARALLEL_TASKS_OPTIONS
        ]

    async def discover(self) -> Hits:
        for name, callback in self._commands():
            yield DiscoveryHit(name, callback)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, callback in self._commands():
            if (match := matcher.match(name)) > 0:
                yield Hit(match, matcher.highlight(name), callback)


class WorktreeRemovalOptionsProvider(Provider):
    """Options of the "Worktree removal" setting (with the task / on
    completion / never).

    Same selection-palette UX as RetentionOptionsProvider: pick an option
    to apply it immediately, Escape keeps the current one. The active
    option is marked with "(current)".
    """

    def _commands(self) -> list[tuple[str, Callable[[], None]]]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.worktree_removal
        return [
            (
                option.label + (" (current)" if option is current else ""),
                partial(app.action_set_worktree_removal, option),
            )
            for option in WORKTREE_REMOVAL_OPTIONS
        ]

    async def discover(self) -> Hits:
        for name, callback in self._commands():
            yield DiscoveryHit(name, callback)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, callback in self._commands():
            if (match := matcher.match(name)) > 0:
                yield Hit(match, matcher.highlight(name), callback)


class EscAbortOptionsProvider(Provider):
    """Options of the "ESC stop behavior" setting (double-press confirm /
    instant stop).

    Same selection-palette UX as RetentionOptionsProvider: pick an option
    to apply it immediately, Escape keeps the current one. The active
    option is marked with "(current)".
    """

    def _commands(self) -> list[tuple[str, Callable[[], None]]]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.esc_abort
        return [
            (
                option.label + (" (current)" if option is current else ""),
                partial(app.action_set_esc_abort, option),
            )
            for option in ESC_ABORT_OPTIONS
        ]

    async def discover(self) -> Hits:
        for name, callback in self._commands():
            yield DiscoveryHit(name, callback)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, callback in self._commands():
            if (match := matcher.match(name)) > 0:
                yield Hit(match, matcher.highlight(name), callback)


class StallNudgeOptionsProvider(Provider):
    """Options of the "Nudge stalled agent" setting.

    Same selection-palette UX as RetentionOptionsProvider: pick an option to
    apply it immediately, Escape keeps the current one. The active option is
    marked with "(current)".
    """

    def _commands(self) -> list[tuple[str, Callable[[], None]]]:
        app = cast("DrCodeApp", self.app)
        current = app.settings.stall_nudge
        return [
            (
                option.label + (" (current)" if option is current else ""),
                partial(app.action_set_stall_nudge, option),
            )
            for option in STALL_NUDGE_OPTIONS
        ]

    async def discover(self) -> Hits:
        for name, callback in self._commands():
            yield DiscoveryHit(name, callback)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, callback in self._commands():
            if (match := matcher.match(name)) > 0:
                yield Hit(match, matcher.highlight(name), callback)
