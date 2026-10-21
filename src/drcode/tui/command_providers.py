from __future__ import annotations

from collections.abc import Callable
from functools import partial
from operator import attrgetter
from typing import TYPE_CHECKING, ClassVar, cast

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

from ..settings import (
    FINALIZATION_OPTIONS,
    RETENTION_OPTIONS,
    STALL_NUDGE_OPTIONS,
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
    """

    DEFAULT_CSS = """
    SettingsPalette #--container {
        width: 50%;
        min-width: 60;
    }
    """

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


class ServiceCommandsProvider(Provider):
    """App-specific commands in the Settings palette.

    Registered alongside the system commands via `DrCodeApp.COMMANDS`;
    every hit is an app action that creates a pre-configured service task.
    """

    def _commands(self) -> list[tuple[str, str, Callable[[], object]]]:
        app = cast("DrCodeApp", self.app)
        return [
            (
                "Update/create AGENTS.md",
                "research task: agent creates or refreshes AGENTS.md",
                app.action_update_agents_md,
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


class ModelConfigCommandProvider(Provider):
    """Model-connection entry in the Settings palette.

    Re-opens the first-start model setup dialog, pre-filled with the
    current model/base_url/api_key/context-size, so the user can change the
    connection on demand (not only on first start). Save applies the new
    values live and persists them (see `_model_setup_done`); Escape dismisses
    without changing anything. The current status ("configured" /
    "not configured") is shown after the colon.
    """

    def _commands(self) -> list[tuple[str, str, Callable[[], object]]]:
        app = cast("DrCodeApp", self.app)
        status = "configured" if app.config.is_ready else "not configured"
        return [
            (
                f"Model settings: {status}",
                (
                    "Change the model id, base URL, API key and context-window "
                    "size, then apply them immediately (user setting)"
                ),
                app.action_open_model_setup,
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
                f"Nudge stalled agent after: {app.settings.stall_nudge.label}",
                (
                    "Abort and restart an agent/tester run that has shown no "
                    "activity for this long (project setting)"
                ),
                app.action_change_stall_nudge,
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

                yield Hit(match, matcher.highlight(name), callback)

