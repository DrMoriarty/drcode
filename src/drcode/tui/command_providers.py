from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from functools import partial
from operator import attrgetter
from typing import TYPE_CHECKING, ClassVar, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.color import Color
from textual.command import (
    CommandInput,
    CommandList,
    CommandPalette,
    DiscoveryHit,
    Hit,
    Hits,
    Provider,
    SearchIcon,
)
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.css.styles import RulesMap
from textual.geometry import Size
from textual.strip import Strip
from textual.theme import Theme
from textual.widgets import Button, LoadingIndicator

from ..providers import (
    AgentRole,
    ModelRef,
    UserModelsStore,
    load_project_custom_models,
    load_project_model_slots,
)
from ..settings import (
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

if TYPE_CHECKING:
    from .app import DrCodeApp


class SpacedCommandList(CommandList):
    """CommandList that renders one blank line between adjacent options.

    Textual's OptionList lays options out back-to-back with no vertical
    spacing, and its virtual layout ignores component-class padding, so the
    gap is implemented at the layout level: every option gets a trailing
    spacer row (the one after the last option is cropped from the virtual
    size; no option meta, so clicks/hover on spacers are ignored). Overrides
    the private layout internals of OptionList (`_update_lines`,
    `get_content_height`, `render_line`); revisit when upgrading textual.
    """

    _SPACER: ClassVar[int] = -1
    """Line-offset sentinel marking a blank separator row."""

    def _update_lines(self) -> None:
        if not self.scrollable_content_region:
            return
        line_cache = self._line_cache
        lines = line_cache.lines
        next_index = lines[-1][0] + 1 if lines else 0
        get_visual = self._get_visual
        width = self.scrollable_content_region.width - self._get_left_gutter_width()

        if next_index < len(self.options):
            padding = self.get_component_styles("option-list--option").padding
            for index, option in enumerate(self.options[next_index:], next_index):
                line_cache.index_to_line[index] = len(line_cache.lines)
                # Cast: get_height() wants RulesMap, but the base class
                # passes its RenderStyles here too (same runtime lookup).
                line_count = (
                    get_visual(option).get_height(
                        cast("RulesMap", self.styles), width - padding.width
                    )
                    + option._divider
                )
                line_cache.heights[index] = line_count
                line_cache.lines.extend(
                    [(index, line_no) for line_no in range(line_count)]
                )
                # A spacer after every option; the trailing one after the
                # last option is cropped from the virtual size below, and
                # option batches appended later keep their gaps.
                line_cache.lines.append((index, self._SPACER))

        last_divider = self.options and self.options[-1]._divider
        virtual_size = Size(
            width,
            len(lines) - (1 if last_divider else 0) - (1 if lines else 0),
        )
        if virtual_size != self.virtual_size:
            self.virtual_size = virtual_size
            self._scroll_update(virtual_size)

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        # One extra row for each gap between adjacent options.
        return super().get_content_height(container, viewport, width) + max(
            0, len(self.options) - 1
        )

    def render_line(self, y: int) -> Strip:
        # y is viewport-relative; the line cache is indexed by absolute
        # (scroll-adjusted) line number — same adjustment the base class
        # makes in OptionList.render_line. Without it the spacer check
        # below inspects the wrong row once the list is scrolled.
        try:
            _, line_offset = self._lines[self.scroll_offset.y + y]
        except IndexError:
            return Strip.blank(
                self.scrollable_content_region.width,
                self.get_visual_style("option-list--option").rich_style,
            )
        if line_offset == self._SPACER:
            return Strip.blank(
                self.scrollable_content_region.width,
                self.get_visual_style("option-list--option").rich_style,
            )
        return super().render_line(y)


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
    # Theme; Quit is pinned to the very end. The model-related entries
    # (provider catalog + per-role picks) live in the "Model settings…"
    # sub-menu (ModelSettingsPalette), and the six task-management
    # preferences live in the "Task settings…" sub-menu
    # (TaskSettingsPalette) — not here.
    ENTRY_ORDER: ClassVar[tuple[str, ...]] = (
        "Keys",
        "Screenshot",
        "Theme",
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


class RoleModelsCommandsProvider(Provider):
    """Per-role model pickers of the "Model settings…" sub-menu.

    One entry per agent role; its current effective model is shown after
    the ":". Picking an entry opens the role's selection palette (same UX
    as the option pickers); Escape dismisses it and keeps the current
    model.
    """

    def _commands(self) -> list[tuple[str, str, Callable[[], object]]]:
        app = cast("DrCodeApp", self.app)
        return [
            (
                f"Coding model: {app.role_model_label(AgentRole.CODING)}",
                (
                    "Model the development agent runs on; pick from the "
                    "user-level provider catalog (saved to the project's "
                    "drcode.json)"
                ),
                app.action_change_coding_model,
            ),
            (
                f"Research model: {app.role_model_label(AgentRole.RESEARCH)}",
                (
                    "Model research tasks run on (analysis/docs profile: "
                    "reads a lot, writes little — a cheap-input "
                    "large-context model works well); falls back to the "
                    "coding model when not set; pick from the user-level "
                    "provider catalog (saved to the project's drcode.json)"
                ),
                app.action_change_research_model,
            ),
            (
                f"Tester model: {app.role_model_label(AgentRole.TESTER)}",
                (
                    "Model the QA tester runs on; pick from the user-level "
                    "provider catalog (saved to the project's drcode.json)"
                ),
                app.action_change_tester_model,
            ),
            (
                f"Compaction model: {app.role_model_label(AgentRole.COMPACTION)}",
                (
                    "Model that summarizes the history when the context "
                    "fill hits the threshold; pick from the user-level "
                    "provider catalog (saved to the project's drcode.json)"
                ),
                app.action_change_compaction_model,
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


class DependencyCommandsProvider(Provider):
    """Dependency-gate commands for the viewed task in the Settings palette.

    One "Remove dependency: #<id>" entry per current dependency (the
    force-start: dropping a blocking dependency makes the task eligible
    immediately and logs the manual override) and one "Add dependency:
    #<id> <title>" entry per other task. Only offered while a queued or
    held task is open — dependencies of a running/completed task make no
    sense (the gate is evaluated at promotion time only)."""

    def _commands(self) -> list[tuple[str, str, Callable[[], object]]]:
        from ..models import TaskStatus

        app = cast("DrCodeApp", self.app)
        task = app.viewed
        if task is None or task.status not in (
            TaskStatus.QUEUED,
            TaskStatus.HOLD,
        ):
            return []
        commands: list[tuple[str, str, Callable[[], object]]] = []
        for dep in task.depends_on:
            dep_task = app.manager.tasks.get(dep)
            title = f" {dep_task.title_display}" if dep_task else ""
            commands.append(
                (
                    f"Remove dependency: #{dep}{title}",
                    (
                        "Force start: drop this dependency — the task stops "
                        "waiting for it (logged as a manual override into "
                        "the task's service events)"
                    ),
                    partial(app.action_remove_dependency, dep),
                )
            )
        for other in app.manager.all_tasks():
            if other.id == task.id or other.id in task.depends_on:
                continue
            commands.append(
                (
                    f"Add dependency: #{other.id} {other.title_display}",
                    (
                        "Make this task wait for it: the dependency must be "
                        "COMPLETED and (under worktree isolation) merged "
                        "into the base branch before this task is taken "
                        "into development"
                    ),
                    partial(app.action_add_dependency, other.id),
                )
            )
        return commands

    async def discover(self) -> Hits:
        for name, help_text, callback in self._commands():
            yield DiscoveryHit(name, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, help_text, callback in self._commands():
            if (match := matcher.match(name)) > 0:
                yield Hit(match, matcher.highlight(name), callback, help=help_text)


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


def _color_spec(color: Color) -> str:
    """Color as a textual style spec: hex for ordinary colors, the ANSI
    name ("ansi_blue", "ansi_default") for ANSI-theme colors — both parse
    as textual styles."""
    return color.hex


def _theme_preview(
    theme: Theme, *, current: bool, name: Content | None = None
) -> Content:
    """One-line preview of a theme for the picker: the theme's name followed
    by sample words ("text", "primary", "secondary", "accent") rendered in
    the theme's own colors on the theme's background, so it is immediately
    obvious whether the combination is e.g. pink-on-gray or green-on-black.
    """
    variables = theme.to_color_system().generate()

    def spec(key: str) -> str:
        return _color_spec(Color.parse(variables[key]))

    words = (
        ("text", "foreground"),
        ("primary", "primary"),
        ("secondary", "secondary"),
        ("accent", "accent"),
    )
    content = name if name is not None else Content(theme.name)
    if current:
        content = content.append_text(" (current)", "dim")
    content = content.append_text("  ")
    block_start = content.cell_length
    content = content.append_text("".join(f" {label} " for label, _ in words))
    # Word colors overlay the background so each sample word shows its
    # theme role ("primary" in $primary, …) on the theme's own background.
    content = content.stylize(f"on {spec('background')}", block_start)
    offset = block_start
    for label, key in words:
        content = content.stylize(spec(key), offset + 1, offset + 1 + len(label))
        offset += len(label) + 2
    return content


class ThemeOptionsProvider(Provider):
    """Options of the theme picker.

    Same selection-palette UX as the other option providers: pick a theme to
    apply it immediately, Escape keeps the current one. The currently active
    theme is marked with "(current)". Every entry carries a one-line color
    preview so themes can be told apart without applying them.
    """

    def _themes(self) -> list[tuple[Theme, bool, Callable[[], None]]]:
        themes = self.app.available_themes

        def set_app_theme(name: str) -> None:
            self.app.theme = name

        return [
            (
                theme,
                theme.name == self.app.theme,
                partial(set_app_theme, theme.name),
            )
            for theme in sorted(themes.values(), key=attrgetter("name"))
        ]

    async def discover(self) -> Hits:
        for theme, current, callback in self._themes():
            yield DiscoveryHit(_theme_preview(theme, current=current), callback)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for theme, current, callback in self._themes():
            if (match := matcher.match(theme.name)) > 0:
                yield Hit(
                    match,
                    _theme_preview(
                        theme,
                        current=current,
                        name=matcher.highlight(theme.name),
                    ),
                    callback,
                )


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


def role_model_options_provider(role: AgentRole) -> type[Provider]:
    """Selection palette listing the catalog models available for one agent
    role ("provider/model" entries; the user-level default is marked
    "(default)", the effective one "(current)"), the saved custom model of
    the role as "Custom: <model-id>" (selecting it re-activates it without
    re-entering anything), "(default — user catalog)" to drop the project
    override (only when a user default exists) and "Custom…" to enter a
    free-form model via ModelSetupDialog."""

    class RoleModelOptions(Provider):
        def _commands(self) -> list[tuple[str, Callable[[], object]]]:
            app = cast("DrCodeApp", self.app)
            catalog = UserModelsStore().load()
            slots = load_project_model_slots(app.project_root)
            slot = slots.get(role)
            current_ref = slot if isinstance(slot, ModelRef) else None
            current_custom = slot == "custom"
            user_default = catalog.defaults.get(role)
            commands: list[tuple[str, Callable[[], object]]] = []
            for provider in catalog.providers:
                for entry in provider.models:
                    ref = ModelRef(provider=provider.name, model=entry.id)
                    label = f"{provider.name}/{entry.id}"
                    if ref == user_default:
                        label += " (default)"
                    if ref == current_ref:
                        label += " (current)"
                    commands.append(
                        (label, partial(app.action_set_role_model, role, ref))
                    )
            customs = load_project_custom_models(app.project_root)
            if role in customs:
                label = f"Custom: {customs[role].model}"
                if current_custom:
                    label += " (current)"
                commands.append((label, partial(app.action_activate_role_custom, role)))
            if user_default is not None:
                commands.append(
                    (
                        "(default — user catalog)",
                        partial(app.action_set_role_model, role, None),
                    )
                )
            commands.append(("Custom…", partial(app.action_custom_role_model, role)))
            return commands

        async def discover(self) -> Hits:
            for name, callback in self._commands():
                yield DiscoveryHit(name, callback)

        async def search(self, query: str) -> Hits:
            matcher = self.matcher(query)
            for name, callback in self._commands():
                if (match := matcher.match(name)) > 0:
                    yield Hit(match, matcher.highlight(name), callback)

    return RoleModelOptions
