from __future__ import annotations

import asyncio
import io
import json
import math
import shutil
import ssl
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import ClassVar, cast

import httpx
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRun
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.usage import UsageLimits
from rich.cells import cell_len
from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.markup import escape
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult, get_system_commands_provider
from textual.binding import Binding
from textual.command import (
    CommandInput,
    CommandList,
    CommandPalette,
    DiscoveryHit,
    Hit,
    Hits,
    LoadingIndicator,
    Provider,
    SearchIcon,
)
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.geometry import Size
from textual.strip import Strip
from textual.theme import ThemeProvider
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Label,
    ListItem,
    ListView,
    Static,
    TextArea,
)

from ..agent import (
    USER_AGENT,
    AgentDeps,
    TesterDeps,
    build_agent,
    build_compaction_agent,
    build_compaction_prompt,
    build_tester_agent,
    compacted_messages,
    ends_with_open_request,
    extract_display,
    extract_previous_summary,
    first_user_prompt,
    is_pending_retry,
    is_summary_message,
    last_context_tokens,
    render_history_for_summary,
    resumable_history,
    run_context_tokens,
    sanitized_messages,
    serialize_messages,
    split_original_prompts,
)
from ..config import Config
from ..models import (
    Clarification,
    Task,
    TaskEvent,
    TaskKind,
    TaskStatus,
    VerificationStatus,
)
from ..prompts import load_compaction_prompts, load_init_repo_prompt
from ..settings import (
    RETENTION_OPTIONS,
    CompletedRetention,
    ProjectSettings,
    ProjectSettingsStore,
)
from ..task_manager import TaskManager
from ..version import APP_VERSION
from .header import ModelStatusHeader

# Sort anchor for history entries without a recorded timestamp (old
# transcripts predating per-message stamps): they keep their relative
# order and sort before everything that has a real time.
_MIN_TIME = datetime.min.replace(tzinfo=UTC)

# Braille spinner frames for the "work in progress" animation shown in a
# reasoning block while the model is thinking and in a tool-call block while
# the tool is executing.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_INTERVAL = 0.12


class TaskItem(ListItem):
    def __init__(
        self, text: str, task_id: str | None = None, *, header: bool = False
    ) -> None:
        super().__init__(
            Label(text, classes="section-header" if header else "task-item")
        )
        self.task_id = task_id


class WordNavTextArea(TextArea):
    """TextArea with macOS-style Option+Arrow word navigation.

    Terminals report Option+Left/Right as `alt+left`/`alt+right` (kitty
    keyboard protocol or CSI 1;3x sequences), but stock TextArea only binds
    `ctrl+left`/`ctrl+right` — so Option+Arrow arrives unbound. Some
    terminals (legacy esc+b / esc+f meta encoding) surface as `alt+b` /
    `alt+f`, so those aliases are bound too. Option+Shift+Arrow extends the
    selection by word, matching native macOS text fields.
    """

    BINDINGS: ClassVar = [
        Binding("alt+left,alt+b", "cursor_word_left", "Cursor word left", show=False),
        Binding(
            "alt+right,alt+f", "cursor_word_right", "Cursor word right", show=False
        ),
        Binding(
            "alt+shift+left,alt+shift+b",
            "cursor_word_left(True)",
            "Cursor word left select",
            show=False,
        ),
        Binding(
            "alt+shift+right,alt+shift+f",
            "cursor_word_right(True)",
            "Cursor word right select",
            show=False,
        ),
    ]


class NewTaskInput(WordNavTextArea):
    """Multi-line editor for a new task; Tab toggles the task type."""

    BINDINGS: ClassVar = [
        Binding("tab", "toggle_kind", "Toggle type", priority=True),
        Binding("ctrl+enter", "submit_new_task", "Create", priority=True),
        Binding("escape", "cancel_new_task", "Cancel", priority=True),
    ]

    def _app(self) -> DrCodeApp:
        app = self.app
        assert isinstance(app, DrCodeApp)
        return app

    def action_toggle_kind(self) -> None:
        self._app().toggle_new_task_kind()

    def action_submit_new_task(self) -> None:
        self._app().submit_new_task()

    def action_cancel_new_task(self) -> None:
        self._app().cancel_new_task()


class ChatInput(WordNavTextArea):
    """Multi-line chat editor that grows with the text.

    Enter sends the message; Shift+Enter / Alt+Enter insert a newline.
    Soft wrapping breaks long lines at the right edge, and the widget's
    height follows the wrapped text (up to a cap) so the whole draft
    stays visible while typing; beyond the cap it scrolls internally.
    """

    MAX_LINES: ClassVar = 10

    BINDINGS: ClassVar = [
        Binding("enter", "submit", "Send", priority=True),
    ]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            soft_wrap=True,
            show_line_numbers=False,
            **kwargs,  # type: ignore[arg-type]
        )

    def _app(self) -> DrCodeApp:
        app = self.app
        assert isinstance(app, DrCodeApp)
        return app

    def action_submit(self) -> None:
        self._app().submit_chat_input()

    async def _on_key(self, event: events.Key) -> None:
        # A "vertical" Enter variant inserts a newline; plain Enter is
        # bound to submit above. Terminals without the kitty keyboard
        # protocol report Shift+Enter as plain Enter — Alt+Enter is the
        # portable fallback for a newline.
        if event.key in ("shift+enter", "alt+enter"):
            event.stop()
            event.prevent_default()
            if not self.read_only:
                self.insert("\n")
                self._update_height()
            return
        await super()._on_key(event)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._update_height()

    def on_resize(self) -> None:
        self._update_height()

    def _wrapped_line_count(self) -> int:
        """Number of visual lines the draft occupies at the current width
        (soft-wrapped)."""
        # content_size excludes the 1+1 horizontal padding; before the first
        # layout it may be 0 — fall back to the widget/screen width so the
        # estimate stays sane.
        width = (
            self.content_size.width
            or (self.size.width - 2)
            or (self.screen.size.width - 2)
            or 80
        )
        width = max(width, 1)
        return sum(
            max(1, math.ceil(cell_len(line) / width))
            for line in (self.text.splitlines() or [""])
        )

    def _update_height(self) -> None:
        self.styles.height = max(2, min(self._wrapped_line_count(), self.MAX_LINES))


class UserBlock(Static):
    """User message: full-width background with an accent bar on the left.

    Plain Text content (not a rich Table) so selection extraction stays clean.
    """

    DEFAULT_CSS = """
    UserBlock {
        background: $panel;
        border-left: thick $accent;
        padding: 1 2;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(Text(text), expand=True)


class ReasoningBlock(Static):
    """Collapsible model-reasoning block; collapsed by default, click to toggle.

    While the model is thinking, `append()` streams text into the block and
    an animated spinner runs in the header; `finish()` (the request stream
    ended) stops it. `thinking(False)` pauses the spinner mid-stream when the
    model moved on from thinking to the answer text.
    """

    def __init__(self) -> None:
        super().__init__(classes="reasoning collapsed")
        self._content = ""
        self._streaming = False
        self._flush_timer: Timer | None = None
        self._spin_timer: Timer | None = None
        self._spin_frame = 0
        # Chat-view epoch this block belongs to; a mid-node view_task()
        # wipes the log and bumps the app's epoch, orphaning the block.
        self.stream_epoch = -1

    def begin(self) -> None:
        self._streaming = True
        self._set_thinking(True)
        self._redraw()

    def thinking(self, on: bool) -> None:
        """Spinner on/off while the stream is still running: True while a
        thinking part is arriving, False once text parts take over."""
        if self._streaming:
            self._set_thinking(on)

    def _set_thinking(self, on: bool) -> None:
        if on and self._spin_timer is None:
            self._spin_timer = self.set_interval(_SPINNER_INTERVAL, self._spin)
        elif not on and self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None

    def _spin(self) -> None:
        self._spin_frame = (self._spin_frame + 1) % len(_SPINNER_FRAMES)
        if self.is_attached:
            self._redraw()

    def append(self, delta: str) -> None:
        # Content is accumulated immediately (a mid-node view_task() carries
        # it over), but the redraw is throttled: _redraw rebuilds the full
        # Text from the whole accumulated content — doing that per delta
        # saturates the event loop (which also drives the UI) and makes
        # keyboard/mouse input lag while the agent runs.
        self._content += delta
        if self._flush_timer is None:
            self._flush_timer = self.set_timer(0.1, self._flush)

    def _flush(self) -> None:
        self._flush_timer = None
        if self.is_attached:
            self._redraw()

    def set_content(self, text: str) -> None:
        """Fill from history (not streaming)."""
        self._content = text
        self._redraw()

    def finish(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        self._streaming = False
        self._set_thinking(False)
        self._redraw()

    def on_unmount(self) -> None:
        # A view switch wipes the block mid-stream: stop the animation, so
        # an orphaned widget doesn't keep ticking until the run ends.
        self._set_thinking(False)

    def on_click(self) -> None:
        self.toggle_class("collapsed")
        self._redraw()

    def _redraw(self) -> None:
        # Body is passed as plain Text, never as a markup string: the model's
        # reasoning regularly contains bracket sequences ("[Binding(...") that
        # Textual's markup parser accepts as tags (unlike Rich's escape()).
        collapsed = self.has_class("collapsed")
        arrow = "▸" if collapsed else "▾"
        status = (
            f" {_SPINNER_FRAMES[self._spin_frame]}"
            if self._spin_timer is not None
            else ""
        )
        text = Text.assemble((f"{arrow} Reasoning{status}", "bold dim"))
        if not collapsed:
            text.append("\n")
            if self._content:
                text.append(self._content)
            else:
                text.append("(empty)", "dim italic")
        self.update(text)


def _markdown_text(text: str, width: int) -> Text:
    """Render markdown to a styled Text at a fixed width.

    A rich Markdown renderable can't take part in Textual's mouse selection:
    RichVisual strips carry no "offset" meta, so a drag inside the widget
    degrades to a select-all of the whole block, and no selection highlight
    is ever painted. A Text is promoted to Content, which supports both.
    """
    console = Console(width=width, legacy_windows=False, file=io.StringIO())
    options = console.options.update(width=width, highlight=False)
    result = Text()
    for segment in console.render(Markdown(text, code_theme="ansi_dark"), options):
        result.append(segment.text, segment.style)
    return result


class AssistantBlock(Static):
    """Live assistant answer, streamed directly into the chat log.

    Grows in place like ReasoningBlock, so the text appears between the
    reasoning and tool-call blocks while the model is still producing it
    (instead of a preview panel under the chat log).
    """

    def __init__(self, text: str = "") -> None:
        super().__init__(classes="assistant")
        self._content = text
        self._text: Text | None = None
        self._width = 0
        self._streaming = False
        self._flush_timer: Timer | None = None
        # Chat-view epoch this block belongs to; see ReasoningBlock.
        self.stream_epoch = -1

    def append(self, delta: str) -> None:
        # Streaming mode: raw text, throttled repaint. Rendering the full
        # markdown (Rich parse + highlighting of the whole accumulated
        # answer) on every delta starves the event loop that also drives
        # the UI — the markdown is rendered once when the stream finishes.
        self._streaming = True
        self._content += delta
        if self._flush_timer is None:
            self._flush_timer = self.set_timer(0.1, self._flush)

    def _flush(self) -> None:
        self._flush_timer = None
        if self.is_attached:
            self._text = None
            self.refresh()

    def finish(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        self._streaming = False
        self._text = None
        self.refresh()

    def set_content(self, text: str) -> None:
        """Fill or replace the markdown (also used for history rendering)."""
        self._content = text
        self._streaming = False
        self._text = None
        self.refresh()

    def on_resize(self) -> None:
        self._text = None
        self.refresh()

    def render(self) -> Text:
        width = self.content_size.width or self.size.width or 80
        if self._text is None or self._width != width:
            if self._streaming:
                self._text = Text(self._content)
            else:
                self._text = _markdown_text(self._content, width)
            self._width = width
        return self._text


class ToolCallBlock(Static):
    """Collapsible tool-call block; collapsed by default, click to toggle.

    Header shows the tool name plus the value of its primary argument
    ("⚙ edit_file src/drcode/agent.py") in gray; expanding reveals the
    full arguments, also in gray.
    """

    # Which parameter carries the "main" value when a tool takes several
    # (checked in order; the first present wins). Single-argument tools
    # always show that one argument's value.
    _MAIN_PARAM_KEYS = ("path", "command", "pattern", "question")
    _MAX_ARG_LEN = 60

    def __init__(self, call: str) -> None:
        super().__init__(classes="tool-call collapsed")
        self._header, self._args = self._split(call)
        self._header += self._primary_arg_suffix(self._args)
        self._spin_timer: Timer | None = None
        self._spin_frame = 0
        self._redraw()

    def start_spinner(self) -> None:
        """Tool execution in progress: animate the header marker until
        stop_spinner() (the result event) — or unmount (the view was
        switched away) — stops it."""
        if self._spin_timer is None:
            self._spin_timer = self.set_interval(_SPINNER_INTERVAL, self._spin)

    def stop_spinner(self) -> None:
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None
        if self.is_attached:
            self._redraw()

    def _spin(self) -> None:
        self._spin_frame = (self._spin_frame + 1) % len(_SPINNER_FRAMES)
        if self.is_attached:
            self._redraw()

    def on_unmount(self) -> None:
        # A view switch wipes the block while the tool is still running: the
        # animation must not keep ticking on an orphaned widget (the fresh
        # block created on re-render gets its own spinner).
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None

    @staticmethod
    def _split(call: str) -> tuple[str, str]:
        """Split "name(args)" (possibly with a prefix like "tester: ") into
        (header, args). Falls back to (call, "") if the shape is unexpected."""
        open_idx = call.find("(")
        if open_idx != -1 and call.endswith(")"):
            return call[:open_idx].rstrip(), call[open_idx + 1 : -1]
        return call, ""

    @classmethod
    def _primary_arg_suffix(cls, args: str) -> str:
        """Return " value" for the tool's primary argument, "" if none."""
        try:
            parsed = json.loads(args) if args else None
        except ValueError:
            return ""
        if not isinstance(parsed, dict) or not parsed:
            return ""
        if len(parsed) == 1:
            value = next(iter(parsed.values()))
        else:
            value = next((parsed[k] for k in cls._MAIN_PARAM_KEYS if k in parsed), None)
        if value is None:
            return ""
        text = " ".join(str(value).split())
        if not text:
            return ""
        if len(text) > cls._MAX_ARG_LEN:
            text = text[: cls._MAX_ARG_LEN - 1] + "…"
        return f" {text}"

    def on_click(self) -> None:
        self.toggle_class("collapsed")
        self._redraw()

    def _redraw(self) -> None:
        # Plain Text, not a markup string — tool arguments are model JSON and
        # may contain bracket sequences Textual's markup parser takes for tags.
        arrow = "▸" if self.has_class("collapsed") else "▾"
        marker = (
            _SPINNER_FRAMES[self._spin_frame] if self._spin_timer is not None else "⚙"
        )
        text = Text(f"{arrow} {marker} {self._header}")
        if not self.has_class("collapsed") and self._args:
            text.append("\n").append(self._args)
        self.update(text)


class SpacedCommandList(CommandList):
    """CommandList that renders one blank line between adjacent options.

    Textual's OptionList lays options out back-to-back with no vertical
    spacing, and its virtual layout ignores component-class padding, so the
    gap is implemented at the layout level: every option except the last
    gets a trailing spacer row (no option meta, so clicks/hover on it are
    ignored). Overrides the private layout internals of OptionList
    (`_update_lines`, `get_content_height`, `render_line`); revisit when
    upgrading textual.
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
            last_index = len(self.options) - 1
            for index, option in enumerate(self.options[next_index:], next_index):
                line_cache.index_to_line[index] = len(line_cache.lines)
                line_count = (
                    get_visual(option).get_height(self.styles, width - padding.width)
                    + option._divider
                )
                line_cache.heights[index] = line_count
                line_cache.lines.extend(
                    [(index, line_no) for line_no in range(line_count)]
                )
                if index < last_index:
                    line_cache.lines.append((index, self._SPACER))

        last_divider = self.options and self.options[-1]._divider
        virtual_size = Size(width, len(lines) - (1 if last_divider else 0))
        if virtual_size != self.virtual_size:
            self.virtual_size = virtual_size
            self._scroll_update(virtual_size)

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        # One extra row for each gap between adjacent options.
        return super().get_content_height(container, viewport, width) + max(
            0, len(self.options) - 1
        )

    def render_line(self, y: int) -> Strip:
        try:
            _, line_offset = self._lines[y]
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
                "documentation task: agent creates or refreshes AGENTS.md",
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


class ProjectSettingsCommandsProvider(Provider):
    """Project-preference commands in the Settings palette.

    A single entry for the completed-tasks auto-cleanup setting; its current
    value is shown after the ":". Picking it opens the option-selection
    palette (same UX as the theme picker); choosing an option there persists
    it into `<project>/.drcode/settings.json`, so the choice is scoped to
    the project, not the user. Escape dismisses the selection window and
    keeps the current value.
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
            )
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


class DrCodeApp(App[None]):
    # The stock command-palette binding (ctrl+p, shown as "palette" in the
    # footer) is replaced by our own "Settings" binding (ctrl+s) below;
    # disabling the built-in also stops it from shadowing the app's own
    # settings binding.
    ENABLE_COMMAND_PALETTE: ClassVar[bool] = False

    # Providers for the Settings palette (it passes no explicit providers,
    # so it reads this set): the system commands (theme, quit, …) plus our
    # own service commands (AGENTS.md refresh and friends).
    COMMANDS: ClassVar = {
        get_system_commands_provider,
        ServiceCommandsProvider,
        ProjectSettingsCommandsProvider,
    }

    CSS = """
    Screen { layers: base toast; }
    #main { height: 1fr; }
    #sidebar-col { width: 36; border-right: solid $primary; }
    #sidebar { height: 1fr; border: none; }
    # Sidebar bottom: project-info then activity-status, plain flow (no
    # dock) so they always stack in compose order under the list.
    #project-info { height: auto; padding: 0 1; }
    #project-info .project-dir { color: $text-muted; }
    #project-info .app-title { color: $accent; text-style: bold; }
    # Connection status lives in the header (see tui/header.py), docked
    # left between the icon and the title; the ok/warn/error classes are
    # set by _check_model().
    #model-status { width: auto; padding: 0 1; }
    #model-status.ok { color: $success; }
    #model-status.warn { color: $warning; }
    #model-status.error { color: $error; }
    #activity-status {
        height: auto; padding: 0 1; margin-top: 1;
        color: $text-disabled;
    }
    #activity-status.fresh { color: $success; }
    #activity-status.stale { color: $warning; }
    #chat-col { height: 1fr; }
    #task-statusbar { dock: top; height: 1; background: $panel; padding: 0 1; }
    #task-status-left { width: 1fr; color: $text; }
    #task-status-right { width: auto; color: $text; }
    # Context counter: red from the compaction threshold, yellow from
    # twice the remaining headroom (threshold 80 → yellow from 60%).
    #task-status-right.ctx-warn { color: $warning; }
    #task-status-right.ctx-crit { color: $error; }
    # 1-column empty gutter on both sides: no text sticks to the window edge.
    #chat-log { height: 1fr; border: none; background: $surface; padding: 0 1; }
    #chat-log > * { margin-bottom: 1; }
    # Multi-line chat editor: height is driven from Python (wrapped line
    # count, capped); auto here is just the pre-measure fallback.
    #chat-input {
        height: auto;
        min-height: 2;
        padding: 0 1;
        border: none;
        background: $surface;
    }
    #chat-input:focus {
        background: $surface-lighten-1;
    }
    .reasoning { padding: 0 4; color: $text-muted; }
    .reasoning:hover { background: $panel; }
    .tool-call { padding: 0 4; color: $text-muted; }
    .tool-call:hover { background: $panel; }
    # Fixed width: the markdown is rendered to a Text at the content width,
    # so an auto-width widget would resize (and rewrap) as lines arrive.
    .assistant { width: 100%; }
    #statusbar { dock: bottom; height: 1; background: $panel; color: $text; padding: 0 1; }
    .section-header { color: $accent; text-style: bold; padding: 0 1; }
    .task-item { padding: 0 1; }
    # Highlight for the task currently open in the chat pane.
    TaskItem.viewed { background: $panel; }
    TaskItem.viewed > .task-item { text-style: bold; }
    #copy-toast {
        layer: toast;
        dock: bottom;
        width: auto;
        height: auto;
        margin: 0 0 3 1;
        padding: 0 2;
        background: $panel;
        border: solid $success;
        color: $text;
    }
    .hidden { display: none; }
    #new-task {
        height: 1fr; width: 1fr;
        align: center middle;
    }
    #new-task.hidden { display: none; }
    #nt-note {
        width: 80%;
        color: $text-disabled;
        text-align: center;
        margin-bottom: 1;
    }
    #nt-input {
        width: 80%; height: 10;
        border: solid $primary;
    }
    # Labels must stay width: auto: a full-width (1fr) sibling disables
    # horizontal centering for the whole #new-task container, leaving the
    # input glued to the left edge.
    #nt-kind-label {
        width: auto;
        margin-top: 1;
        text-style: bold;
    }
    #nt-kind-label.kind-feature { color: $success; }
    #nt-kind-label.kind-bug { color: $error; }
    #nt-kind-label.kind-documentation { color: yellow; }
    #nt-hint { width: auto; color: $text-disabled; }
    """
    BINDINGS: ClassVar = [
        ("ctrl+n", "new_task", "New task"),
        ("ctrl+d", "done_task", "Done"),
        ("ctrl+x", "cancel_task", "Cancel task"),
        ("ctrl+o", "focus_sidebar", "Tasks"),
        ("ctrl+i", "focus_input", "Input"),
        ("ctrl+t", "cycle_task_kind", "Change type"),
        ("ctrl+s", "settings", "Settings"),
        ("ctrl+h", "toggle_hold", "Hold/Resume"),
        ("ctrl+delete", "delete_task", "Delete held"),
        ("escape", "abort_run", "Stop agent"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        # Layered config: env > project (.env / drcode.json) >
        # ~/.config/drcode/drcode.json > defaults. The user config dir is
        # scaffolded (with drcode.json.example) on first start inside Config.
        self.config = Config(project_root=project_root)
        self.manager = TaskManager(project_root)
        # Project-scoped preferences (<project>/.drcode/settings.json), then
        # the startup cleanup: completed tasks older than the retention
        # setting are removed before anything is shown or resumed.
        self.settings_store = ProjectSettingsStore(project_root)
        self.settings: ProjectSettings = self.settings_store.load()
        self._purged_tasks = self.manager.purge_expired_completed(
            self.settings.completed_retention.days
        )
        self.viewed_id: str | None = None
        self.agent_running = False
        self.tester_running = False
        # Task id the tester worker is currently verifying (for Ctrl+X).
        self._tester_task_id: str | None = None
        self.new_task_kind: TaskKind = TaskKind.FEATURE
        # Incremented on every run_agent() start; the finally block of a
        # superseded (cancelled) run must not touch shared state.
        self._agent_generation = 0
        # Pending messages already drawn in the current chat view, so the
        # agent run doesn't re-render them when it consumes them.
        self._pending_rendered: list[str] = []
        # Whether the chat log currently holds at least one block.
        self._chat_has_content = False
        # Auto-hide timer for the "copied" toast.
        self._copy_toast_timer: Timer | None = None
        # Per-task tester failure tracking: consecutive failure count and
        # the earliest monotonic time the task may be retried. A failed
        # tester run is retried with backoff, not dropped: tasks in TESTING
        # block the whole pipeline, so they must eventually get their round.
        self._tester_fail_count: dict[str, int] = {}
        self._tester_retry_at: dict[str, float] = {}
        # Same idea for agent runs: a task whose run crashed (e.g. the model
        # gateway is unreachable) is demoted to QUEUED and skipped by
        # promote_next for the rest of the session, so one broken task (or a
        # dead gateway) doesn't loop the whole queue through the error.
        self._agent_failed: set[str] = set()
        # True while the new-task form covers the chat view. Must exist from
        # the start: the streaming loop reads it on every text delta, and an
        # unset attribute would crash the run (it did — the AttributeError
        # surfaced as a red "error ›" line and killed the first streamed run
        # after a startup resume).
        self._new_task_mode = False
        # Snapshot of the task state the sidebar was last rendered from; the
        # poller compares against it to redraw only on actual change.
        self._sidebar_snapshot: tuple = ()
        # Last moment an agent/tester event (stream delta, tool call/result)
        # was seen; the activity label compares it against now. Zero means
        # the current run has not emitted anything yet.
        self._last_activity = 0.0
        # Tool the run is currently waiting on (name at call start, cleared
        # on the result event) — lets the label distinguish "tool busy" from
        # "total silence".
        self._current_tool: str | None = None
        # Tool calls currently executing, per run (the run key is
        # ("agent"|"tester", generation)): tool_call_id ->
        # (task_id, rendered call text, live block with the spinner). The
        # text lets a mid-run view re-render (view_task wipes the chat log)
        # re-attach the running spinner to the freshly rendered block.
        self._active_tools: dict[
            tuple[str, int], dict[str, tuple[str, str, ToolCallBlock]]
        ] = {}
        # Monotonic stamp distinguishing successive tester runs.
        self._tester_generation = 0
        # Context fill (tokens) seen by the latest completed model request,
        # per task, updated live while the run streams. The stored transcript
        # is only refreshed at persist points, so the status bar reads the
        # live value first and falls back to the persisted one (the final
        # value of a finished task — it stays in the history via the saved
        # per-response usage).
        self._live_ctx_tokens: dict[str, int] = {}
        # Monotonic stamp of the last coalesced follow-the-tail scroll while
        # streaming (see _stream_follow).
        self._last_follow = 0.0
        # Bumped by every view_task(): the streaming loop compares it with
        # each block's stream_epoch to detect that a mid-node re-render
        # orphaned the live blocks and they must be re-mounted.
        self._chat_epoch = 0

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield ModelStatusHeader(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="sidebar-col"):
                yield ListView(id="sidebar")
                yield Static("", id="project-info")
                yield Static("", id="activity-status")
            with Vertical(id="chat-col"):
                with Horizontal(id="task-statusbar"):
                    yield Static("", id="task-status-left")
                    yield Static("", id="task-status-right")
                yield VerticalScroll(id="chat-log")
                yield ChatInput(placeholder="Type a message…", id="chat-input")
                with Vertical(id="new-task", classes="hidden"):
                    yield Static(
                        "This is not a live chat with the agent — "
                        "you are composing a task to be executed later.",
                        id="nt-note",
                    )
                    yield NewTaskInput(placeholder="Describe the task…", id="nt-input")
                    yield Static("type: feature", id="nt-kind-label")
                    yield Static(
                        "tab — switch type   ctrl+enter — create   esc — cancel",
                        id="nt-hint",
                    )
        yield Static("", id="statusbar")
        yield Static("", id="copy-toast", classes="hidden")
        yield Footer()

    def on_mount(self) -> None:
        if self._purged_tasks:
            # The startup cleanup removed something — say so, so deleted
            # history never "silently vanishes".
            self.notify(
                f"Удалено завершённых задач: {len(self._purged_tasks)} "
                f"({self.settings.completed_retention.label})",
                title="Автоочистка",
                timeout=6,
            )
        self.refresh_project_info()
        self.refresh_sidebar()
        tasks = self.manager.all_tasks()
        active = self.manager.active
        initial = active or (tasks[0] if tasks else None)
        if initial:
            self.view_task(initial.id)
        else:
            self.action_new_task()
        self.refresh_status()
        self.refresh_connection()
        self.set_interval(30, self.refresh_connection)
        # The sidebar must follow task-status changes on its own: statuses
        # also mutate mid-run (agent tools, promote_next in the worker
        # finally-blocks) where no one calls refresh_sidebar().
        self.set_interval(1.0, self._poll_sidebar)
        self.set_interval(1.0, self._poll_activity)
        # Resume after restart. _advance() re-tests TESTING tasks (the app
        # died during testing) FIRST — the startup demotion (active ->
        # queued) leaves pending_input stranded unless the queue is
        # promoted here.
        self._advance()

    # -- widgets helpers ------------------------------------------------------

    @property
    def chat_log(self) -> VerticalScroll:
        return self.query_one("#chat-log", VerticalScroll)

    @property
    def chat_input(self) -> ChatInput:
        return self.query_one("#chat-input", ChatInput)

    @property
    def sidebar(self) -> ListView:
        return self.query_one("#sidebar", ListView)

    @property
    def project_info(self) -> Static:
        return self.query_one("#project-info", Static)

    @property
    def model_status(self) -> Static:
        return self.query_one("#model-status", Static)

    @property
    def activity_status(self) -> Static:
        return self.query_one("#activity-status", Static)

    @property
    def viewed(self) -> Task | None:
        return self.manager.tasks.get(self.viewed_id) if self.viewed_id else None

    # -- copy on select -------------------------------------------------------

    def _selected_text(self) -> str | None:
        """Extract the currently selected text from rendered lines.

        Done at the app level instead of Widget.get_selection, which works on
        logical (unwrapped) text and returns garbled results for wrapped lines.
        """
        parts: list[str] = []
        for widget, selection in self.screen.selections.items():
            if not widget.is_attached:
                continue
            lines: list[str] = []
            for y in range(widget.size.height):
                span = selection.get_span(y)
                if span is None:
                    continue
                start, end = span
                line = widget.render_line(y).text
                lines.append((line[start:] if end == -1 else line[start:end]).rstrip())
            if lines:
                parts.append("\n".join(lines).rstrip())
        if not parts:
            return None
        return "\n".join(parts)

    def _copy_with_fallback(self, text: str) -> None:
        # copy_to_clipboard uses OSC52, which macOS Terminal.app ignores;
        # pbcopy always works there.
        self.copy_to_clipboard(text)
        if sys.platform == "darwin" and shutil.which("pbcopy"):
            subprocess.run(["pbcopy"], input=text.encode(), check=False)

    def _show_copy_toast(self, text: str) -> None:
        preview = escape(text if len(text) <= 40 else text[:39] + "…")
        toast = self.query_one("#copy-toast", Static)
        toast.update(f"✓ copied: {preview}")
        toast.remove_class("hidden")
        if self._copy_toast_timer is not None:
            self._copy_toast_timer.stop()
        self._copy_toast_timer = self.set_timer(2.0, lambda: toast.add_class("hidden"))

    def on_text_selected(self, event: events.TextSelected) -> None:
        text = self._selected_text()
        if not text:
            return
        self._copy_with_fallback(text)
        self._show_copy_toast(text)

    # -- sidebar ---------------------------------------------------------------

    def _display_dir(self) -> str:
        """Working directory with the home prefix collapsed to `~`."""
        path = self.project_root
        try:
            rel = path.relative_to(Path.home())
        except ValueError:
            return str(path)
        if str(rel) == ".":
            return "~"
        return f"~/{rel}"

    def _git_branch(self) -> str | None:
        """Current branch of the git repo in the working directory.

        Parsed straight from `.git` (also handles worktrees, where `.git`
        is a file with a `gitdir:` pointer) so no `git` binary is needed.
        """
        dot_git = self.project_root / ".git"
        try:
            if dot_git.is_dir():
                git_dir: Path | None = dot_git
            elif dot_git.is_file():
                line = dot_git.read_text(encoding="utf-8").strip()
                if not line.startswith("gitdir:"):
                    return None
                git_dir = Path(line.split(":", 1)[1].strip())
                if not git_dir.is_absolute():
                    git_dir = self.project_root / git_dir
            else:
                git_dir = None
        except OSError:
            return None
        if git_dir is None:
            return None
        try:
            content = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if content.startswith("ref: refs/heads/"):
            return content[len("ref: refs/heads/") :]
        if content.startswith("ref: "):
            return content[len("ref: ") :]
        return content[:7]  # detached HEAD

    def refresh_project_info(self) -> None:
        """Top of the sidebar, above the model status: working directory
        with the git branch (`~/pr/drcode:main`), then name and version."""
        branch = self._git_branch()
        dir_text = escape(self._display_dir())
        where = f"{dir_text}:{branch}" if branch else dir_text
        self.project_info.update(
            f"[dim]{where}[/]\n[bold]DrCode[/] [dim]{APP_VERSION}[/]"
        )

    def _sidebar_fingerprint(self) -> tuple:
        """Lightweight snapshot of everything the sidebar renders: status,
        verification label, title and the viewed-marker per task."""
        return tuple(
            (
                t.id,
                t.status.value,
                t.kind.value,
                t.verification,
                t.title_display,
                t.id == self.viewed_id,
            )
            for t in self.manager.all_tasks()
        )

    def _poll_sidebar(self) -> None:
        """Redraw the sidebar only when the underlying task state changed
        since the last render (status transitions, new bug reports from the
        tester, promote_next() activations, …)."""
        if self._sidebar_fingerprint() != self._sidebar_snapshot:
            self.refresh_sidebar()

    def refresh_sidebar(self) -> None:
        sidebar = self.sidebar
        sidebar.clear()
        # Group tasks by status; each group keeps the created_at order of
        # all_tasks() unless sorted explicitly below.
        by_status: dict[TaskStatus, list[Task]] = {}
        for t in self.manager.all_tasks():
            by_status.setdefault(t.status, []).append(t)

        # Sidebar mirrors the pipeline a task flows through, top to bottom:
        # queue → clarifications → active → testing → completed.
        queue = list(reversed(self.manager.queued()))
        clarification = sorted(
            by_status.get(TaskStatus.AWAITING_CLARIFICATION, []),
            key=lambda t: t.updated_at,
            reverse=True,
        )
        active = by_status.get(TaskStatus.ACTIVE, [])
        testing = by_status.get(TaskStatus.TESTING, [])
        # Normally one task is being tested; if several are stuck in
        # TESTING, the one the tester is working on goes to the bottom of
        # the section (closest to the pipeline's exit into COMPLETED).
        testing_active_id = self._tester_task_id
        testing = [t for t in testing if t.id != testing_active_id] + [
            t for t in testing if t.id == testing_active_id
        ]
        completed = sorted(
            by_status.get(TaskStatus.COMPLETED, []),
            key=lambda t: t.updated_at,
            reverse=True,
        )
        hold = by_status.get(TaskStatus.HOLD, [])

        # Empty sections are not rendered at all: the sidebar shows only the
        # stages the pipeline currently has tasks in.
        sections: list[tuple[str, list[Task]]] = [
            ("QUEUE", queue),
            ("NEEDS CLARIFICATION", clarification),
            ("ACTIVE", active),
            ("TESTING", testing),
            ("COMPLETED", completed),
            ("ON HOLD", hold),
        ]
        for name, tasks in sections:
            if not tasks:
                continue
            sidebar.append(TaskItem(f"{name} ({len(tasks)})", header=True))
            for t in tasks:
                marker = {
                    "queued": "⏳",
                    "hold": "⏸",
                    "active": "▶",
                    "testing": "🧪",
                    "completed": "✓",
                    "clarification": "❓",
                }.get(t.status.value, "⏳")
                kind_letter = {
                    TaskKind.BUG: "[red]B[/]",
                    TaskKind.DOCUMENTATION: "[yellow]D[/]",
                }.get(t.kind, "[green]F[/]")
                if t.verification is VerificationStatus.PASSED:
                    suffix = " ✅"
                elif t.verification is VerificationStatus.FAILED:
                    suffix = " ❌"
                else:
                    suffix = ""
                viewed = " ●" if t.id == self.viewed_id else ""
                item = TaskItem(
                    f"{marker} {kind_letter} #{t.id} "
                    f"{escape(t.title_display)}{suffix}{viewed}",
                    t.id,
                )
                if t.id == self.viewed_id:
                    item.add_class("viewed")
                sidebar.append(item)
        sidebar.refresh()
        self._sidebar_snapshot = self._sidebar_fingerprint()

    # -- model connection status ---------------------------------------------

    def _recover_from_outage(self) -> None:
        """Error → connected transition: the outage is over, so un-block the
        tasks whose runs died with it. They were excluded from promotion via
        _agent_failed and would stay stranded until restart otherwise. If
        nothing is running, the pipeline picks the next task immediately."""
        self._agent_failed.clear()
        self._advance()

    def refresh_connection(self) -> None:
        self.run_worker(self._check_model(), exclusive=True, group="model-check")

    async def _check_model(self) -> None:
        label = self.model_status
        was_error = label.has_class("error")
        if not self.config.is_ready:
            label.remove_class("ok", "error")
            label.add_class("warn")
            label.update("○ model not configured")
            return
        model = self.config.model
        # Системное хранилище сертификатов: корпоративные CA (как у
        # api-copilot.x5.ru) отсутствуют в certifi, из-за чего httpx даёт
        # CERTIFICATE_VERIFY_FAILED, который оборачивается в ConnectError.
        try:
            import truststore

            verify: ssl.SSLContext | bool = truststore.SSLContext(
                ssl.PROTOCOL_TLS_CLIENT
            )
        except ImportError:
            verify = True
        try:
            async with httpx.AsyncClient(timeout=5, verify=verify) as client:
                # Зонд через chat/completions, а не /models: этот гейтвей не
                # отдаёт /models (503). Пустой messages даёт 400 от бэкенда —
                # значит TLS, ключ, UA и имя модели приняты.
                resp = await client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "User-Agent": USER_AGENT,
                    },
                    json={"model": model, "messages": []},
                )
        except httpx.HTTPError as e:
            label.remove_class("ok", "warn")
            label.add_class("error")
            label.update(f"✗ {model} — unavailable ({type(e).__name__})")
            return
        label.remove_class("warn", "error")
        if resp.status_code in (401, 403):
            label.add_class("error")
            label.update(f"✗ {model} — auth failed (HTTP {resp.status_code})")
        elif resp.status_code < 500:
            label.add_class("ok")
            label.update(f"● {model}")
            if was_error:
                self._recover_from_outage()
        else:
            label.add_class("error")
            label.update(f"✗ {model} — unavailable (HTTP {resp.status_code})")

    # -- run activity indicator ----------------------------------------------

    def _touch_activity(self, tool: str | None = None, *, done: bool = False) -> None:
        """Record a sign of life from the agent/tester run. `tool` marks a
        tool call start; `done` clears the current-tool marker (result
        arrived)."""
        self._last_activity = time.monotonic()
        if done:
            self._current_tool = None
        elif tool is not None:
            self._current_tool = tool

    @staticmethod
    def _fmt_age(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds}s"
        return f"{seconds // 60}m{seconds % 60:02d}s"

    def _poll_activity(self) -> None:
        """Every second, show in the sidebar whether the run makes progress
        or has gone silent (hung model request, long tool, deadlock)."""
        label = self.activity_status
        if not (self.agent_running or self.tester_running):
            active = self.manager.active
            if active is not None:
                # Task claims to be active but no worker owns it (e.g. the
                # user aborted the run, or an older build stranded it):
                # surface the inconsistency instead of showing a fake idle.
                label.remove_class("fresh")
                label.add_class("stale")
                label.update(
                    f"⚠ #{active.id} active, nothing running — "
                    "send a message to (re)start"
                )
                return
            label.remove_class("fresh", "stale")
            label.update("○ idle")
            return
        who = "agent" if self.agent_running else "tester"
        age = int(time.monotonic() - self._last_activity)
        if self._current_tool:
            detail = f"tool {self._current_tool} running {self._fmt_age(age)}"
        else:
            detail = f"last event {self._fmt_age(age)} ago"
        if age < 30:
            label.remove_class("stale")
            label.add_class("fresh")
            label.update(f"▶ {who}: working · {detail}")
        else:
            label.remove_class("fresh")
            label.add_class("stale")
            label.update(f"⚠ {who}: stalled? · {detail} (esc to stop)")

    # -- chat pane ---------------------------------------------------------------

    def _log_event(self, task: Task, text: str, *, kind: str = "system") -> None:
        """Persist a service event into the task's event log.

        These lines are shown in the chat but must never reach the model:
        they live in `task.events`, outside the transcripts that
        resumable_history() reads, so a resumed agent never sees them.
        """
        task.events.append(TaskEvent(kind=kind, text=text))
        task.touch()
        self.manager.storage.save(task)

    @staticmethod
    def _format_event(event: TaskEvent) -> str:
        ts = event.at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        return f"[dim]{ts}[/] {escape(event.text)}"

    def _show_event(self, task: Task, text: str, *, kind: str = "system") -> None:
        """Persist an event and, if the task is open, render it live."""
        self._log_event(task, text, kind=kind)
        if self.viewed_id == task.id:
            self._write_history_line("event", self._format_event(task.events[-1]))

    def _next_testing(self) -> Task | None:
        """Oldest task in TESTING (crash/abort mid-test) whose retry backoff
        has expired. Tasks whose tester run failed recently are skipped
        until their scheduled retry time."""
        now = time.monotonic()
        return next(
            (
                t
                for t in self.manager.all_tasks()
                if t.status is TaskStatus.TESTING
                and self._tester_retry_at.get(t.id, 0.0) <= now
            ),
            None,
        )

    def _advance(self) -> None:
        """The single pipeline gate: testing always has absolute priority.

        Every path that used to promote the queue goes through here, so a
        new development run can never start while any task is awaiting
        verification. While a tester run is in flight this is a no-op —
        the tester's own completion path advances the pipeline.
        """
        if self.tester_running:
            return
        testing = self._next_testing()
        if testing is not None:
            # Set synchronously: two _advance() calls within one event-loop
            # tick must not start two tester workers (the exclusive group
            # would cancel the first before it does anything).
            self.tester_running = True
            self.run_tester(testing.id)
            return
        if any(t.status is TaskStatus.TESTING for t in self.manager.all_tasks()):
            # Every TESTING task is within its retry backoff window: the
            # rule "no new development while any testing is incomplete"
            # holds even then — the retry timer will call _advance() again.
            return
        promoted = self.manager.promote_next(exclude=self._agent_failed)
        if promoted:
            self.run_agent(promoted.id)
        self.refresh_status()

    def view_task(self, task_id: str) -> None:
        # Opening a task cancels a pending new-task creation: the click on a
        # sidebar item is an explicit "leave the form" from the user.
        if self._new_task_mode:
            self._leave_new_task_mode()
        self.viewed_id = task_id
        task = self.manager.get(task_id)
        # Wipe marker for the streaming loop: any block created before this
        # point belongs to a dead view and must be re-mounted on the next
        # delta (its accumulated content is carried over, so the live text
        # never loses the tail of an in-flight response).
        self._chat_epoch += 1
        self.chat_log.remove_children()
        self._chat_has_content = False
        self._pending_rendered = []
        self._clear_stream_label()
        # Rebuild the full history as ONE chronological timeline. Both the
        # serialized transcripts (per-message timestamps) and the service
        # events (TaskEvent.at) carry wall-clock stamps, so a stable sort
        # by timestamp reproduces exactly the order in which the entries
        # were rendered live — the same order on every (re)open, instead of
        # clumping all events at the tail.
        timeline: list[tuple[datetime, int, str, str]] = []
        seq = 0

        def _add(at: datetime | None, role: str, text: str) -> None:
            nonlocal seq
            timeline.append((at or _MIN_TIME, seq, role, text))
            seq += 1

        for at, role, text in extract_display(task.messages):
            _add(at, role, text)
        # Tester rounds: the synthetic prompt ("Verify completed task #…")
        # is not shown; each round is announced by its own "testing"
        # separator anchored at the round's first message.
        for round_messages in task.tester_messages:
            if not round_messages:
                continue
            round_items = extract_display(round_messages)
            round_at = next((at for at, _, _ in round_items if at is not None), None)
            _add(round_at, "testing-sep", "")
            for at, role, text in round_items:
                if role == "user":
                    continue
                if role == "tool-call":
                    text = f"tester: {text}"
                _add(at, role, text)
        # Service events (failures, skips) — human-only, with timestamps;
        # interleaved at the position they occurred at.
        for event in task.events:
            _add(event.at, "event", self._format_event(event))
        timeline.sort(key=lambda item: item[:2])
        for _, _, role, text in timeline:
            block = self._write_history_line(role, text)
            if role == "tool-call" and isinstance(block, ToolCallBlock):
                # A tool still executing (run in flight) gets its spinner
                # back on the freshly rendered block.
                self._reattach_tool_spinner(block, text, task_id)
        for pending in task.pending_input:
            if is_pending_retry(task.messages, pending):
                # The failed/interrupted run left this prompt in the
                # transcript: it is already rendered from task.messages
                # above — writing the pending copy again duplicates it.
                continue
            self._write_history_line("user-pending", pending)
        if task.status is TaskStatus.AWAITING_CLARIFICATION and task.clarification:
            self._write_clarification(task.clarification)
        # Opening a task always lands on the latest entry (previously this
        # happened implicitly because every mount scrolled to the end).
        self.chat_log.scroll_end(animate=False)
        self.refresh_sidebar()
        self.refresh_status()

    def _chat_at_bottom(self) -> bool:
        """True when the last line of the chat log is in the viewport.

        Tolerance of 1 line absorbs rounding of fractional scroll offsets.
        """
        log = self.chat_log
        return log.scroll_offset.y >= log.max_scroll_y - 1

    def _stream_follow(self) -> None:
        """Keep the chat tail visible while streaming, coalesced to ~10 Hz.

        Called per streamed delta; each scroll_end schedules a refresh+scroll
        pass over the (growing) chat log — doing that per delta saturates the
        event loop and makes keyboard/mouse input unresponsive.
        """
        now = time.monotonic()
        if now - self._last_follow >= 0.1:
            self._last_follow = now
            self.chat_log.call_after_refresh(self.chat_log.scroll_end, animate=False)

    # -- tool-call spinners ------------------------------------------------------

    def _tool_call_started(
        self,
        run_key: tuple[str, int],
        task_id: str,
        call_id: str,
        call_text: str,
        block: ToolCallBlock,
    ) -> None:
        """Register a tool call whose execution just began (spinner on)."""
        self._active_tools.setdefault(run_key, {})[call_id] = (
            task_id,
            call_text,
            block,
        )

    def _tool_call_finished(self, call_id: str) -> None:
        """Tool result arrived: stop the spinner of its block (in whichever
        run registered it — agent and tester never overlap, but the lookup
        is over all runs anyway)."""
        for per_run in self._active_tools.values():
            entry = per_run.pop(call_id, None)
            if entry is not None:
                entry[2].stop_spinner()

    def _stop_run_tools(self, run_key: tuple[str, int]) -> None:
        """End-of-run cleanup: anything still marked running for this run
        (a result event lost to an abort/error) loses its spinner."""
        for _task_id, _text, block in self._active_tools.pop(run_key, {}).values():
            block.stop_spinner()

    def _reattach_tool_spinner(
        self, block: ToolCallBlock, call_text: str, task_id: str
    ) -> None:
        """After a mid-run view re-render, hand a still-running call's
        spinner over to the freshly rendered block (matched by the exact
        rendered call text — same tool with the same arguments)."""
        for per_run in self._active_tools.values():
            for call_id, (tid, text, _old) in per_run.items():
                if tid == task_id and text == call_text:
                    per_run[call_id] = (tid, text, block)
                    block.start_spinner()
                    return

    def _write_chat(
        self, content: RenderableType | Widget, *, expand: bool = False
    ) -> None:
        """Mount one block to the chat log.

        Scrolls to the end only when the user is already looking at the
        bottom: a scrolled-up reader must not be yanked down by new tool
        calls / reasoning blocks arriving while the agent runs.
        """
        follow = self._chat_at_bottom()
        if isinstance(content, Widget):
            content.add_class("chat-block")
        else:
            content = Static(content, expand=expand, classes="chat-block")
        self.chat_log.mount(content)
        if follow:
            self.chat_log.scroll_end(animate=False)
        self._chat_has_content = True

    def _write_history_line(self, role: str, text: str) -> Widget | None:
        """Render one history entry into the chat log.

        Returns the mounted block widget (None for markup-string lines) so
        callers can tweak it afterwards — view_task re-attaches the running
        spinner to tool-call blocks this way.
        """
        block: Widget | None = None
        if role == "user":
            block = UserBlock(text)
        elif role == "user-pending":
            self._pending_rendered.append(text)
            block = UserBlock(text)
        elif role == "reasoning":
            block = ReasoningBlock()
            block.set_content(text)
        elif role == "tool-call":
            block = ToolCallBlock(text)
        elif role == "assistant":
            block = AssistantBlock(text)
        elif role == "tool":
            self._write_chat(f"[dim]{escape(text)}[/]")
        elif role == "tester":
            self._write_chat(f"[bold magenta]tester ›[/] {text}")
        elif role == "testing-sep":
            label = f"testing: {text}" if text else "testing"
            self._write_chat(f"[bold magenta]──── {label} ────[/]")
        elif role == "summary":
            # Compaction summary: a marker separator, then the summary
            # itself rendered as markdown (it is the model's output the
            # agent continues from, not a user message).
            self._write_chat("[bold magenta]──── context compacted ────[/]")
            block = AssistantBlock(text)
        elif role == "event":
            self._write_chat(f"[bold red]event ›[/] {text}")
        elif role == "system":
            self._write_chat(f"[bold red]error ›[/] {text}")
        if block is not None:
            self._write_chat(block)
        return block

    def _write_clarification(self, clarification: Clarification) -> None:
        lines = [f"[bold blue]drcode ❓[/] {escape(clarification.question)}"]
        for i, option in enumerate(clarification.options):
            lines.append(
                f"  ▸ [@click=app.pick_option({i})]"
                f"[underline cyan]{escape(option)}[/][/]"
            )
        lines.append("[dim]Click an option or answer in the input below.[/]")
        self._write_chat("\n".join(lines))

    def refresh_status(self) -> None:
        active = self.manager.active
        queue_len = len(self.manager.queued())
        testing_len = len(
            [t for t in self.manager.all_tasks() if t.status is TaskStatus.TESTING]
        )
        state = "running" if self.agent_running else "idle"
        tester = "running" if self.tester_running else "idle"
        who = f"#{active.id} {active.title_display}" if active else "none"
        model = self.config.model or "(model not configured)"
        self.query_one("#statusbar", Static).update(
            f" active: {who}   queue: {queue_len}   testing: {testing_len}   "
            f"agent: {state}   tester: {tester}   model: {model} "
        )
        self.refresh_task_status()

    def refresh_task_status(self) -> None:
        left = self.query_one("#task-status-left", Static)
        right = self.query_one("#task-status-right", Static)
        # The new-task form is not tied to any task: keep the bar empty so
        # the previously viewed task doesn't keep updating behind the form.
        if self._new_task_mode:
            left.update("")
            right.update("")
            return
        task = self.viewed
        if task is None:
            left.update("")
            right.update("")
            return
        created = task.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
        kind = {
            TaskKind.BUG: "[bold red]bug[/]",
            TaskKind.DOCUMENTATION: "[bold yellow]documentation[/]",
        }.get(task.kind, "[bold green]feature[/]")
        status_colors = {
            TaskStatus.QUEUED: "yellow",
            TaskStatus.HOLD: "blue",
            TaskStatus.ACTIVE: "cyan",
            TaskStatus.TESTING: "magenta",
            TaskStatus.COMPLETED: "green",
            TaskStatus.AWAITING_CLARIFICATION: "yellow",
        }
        status = f"[{status_colors[task.status]}]{task.status.value}[/]"
        parts = [
            kind,
            f"#{task.id} {escape(task.title_display)}",
            f"[dim]{created}[/]",
            status,
        ]
        if task.verification is VerificationStatus.PASSED:
            parts.append("[green]✓ testing[/]")
        elif task.verification is VerificationStatus.FAILED:
            parts.append("[red]✗ testing[/]")
        left.update("   ".join(parts))
        self._update_ctx_counter(right, task)

    @staticmethod
    def _fmt_ctx_tokens(n: int) -> str:
        if n >= 1000:
            return f"{n / 1000:.1f}".rstrip("0").rstrip(".") + "k"
        return str(n)

    def _update_ctx_counter(self, right: Static, task: Task) -> None:
        """Right corner of the task status bar: current context fill in
        tokens and percent of the allowed maximum.

        Source of truth: the latest model request usage. For a run in
        progress — the live value; otherwise (finished task, history view)
        — the value computed from the persisted transcript, i.e. the final
        value is saved in the task history together with the per-response
        usage and survives restarts.
        """
        right.remove_class("ctx-warn", "ctx-crit")
        tokens = self._live_ctx_tokens.get(task.id)
        if tokens is None:
            tokens = last_context_tokens(task.messages)
        if not tokens:
            right.update("")
            return
        max_tokens = self.config.max_context_tokens
        if max_tokens > 0:
            pct = round(tokens * 100 / max_tokens)
            text = (
                f"🧠 {self._fmt_ctx_tokens(tokens)} / "
                f"{self._fmt_ctx_tokens(max_tokens)} · {pct}%"
            )
            # Red from the compaction threshold, yellow when the remaining
            # headroom is twice the red-zone headroom: threshold 80 →
            # yellow from 60, threshold 90 → yellow from 80.
            threshold = min(max(self.config.compaction_threshold, 1), 100)
            warn_pct = 2 * threshold - 100
            if pct >= threshold:
                right.add_class("ctx-crit")
            elif warn_pct > 0 and pct >= warn_pct:
                right.add_class("ctx-warn")
        else:
            text = f"🧠 {self._fmt_ctx_tokens(tokens)}"
        right.update(text)

    # -- agent run -----------------------------------------------------------------

    def _persist_task_messages(
        self, task: Task, run: AgentRun[AgentDeps, str] | None
    ) -> None:
        """Snapshot the transcript produced so far into the task.

        One agent turn can run for minutes (many model requests and tool
        calls). Saving only on turn completion leaves the stored history
        stale while the agent works: switching to the task then shows
        nothing new (for a fresh task — nothing at all), and an interrupted
        turn (abort, a follow-up comment cancelling the exclusive worker)
        lost everything it had done. Persisting after every completed node
        keeps both the on-disk history and `view_task` current.

        The snapshot may end with an incomplete exchange (tool calls
        without results), which the model API would reject on resume —
        `resumable_history()` trims that tail when the transcript is next
        loaded as history.

        `run` is None when the failure happened before the run context was
        entered (e.g. `agent.iter()` or `resumable_history()` raised); there
        is nothing new to persist then — the pending input is re-queued by
        the caller and retried on the next activation.
        """
        if run is None:
            return
        try:
            messages = run.all_messages()
        except Exception:  # noqa: BLE001 - state may be mid-teardown after cancel
            return
        task.messages = serialize_messages(messages)
        self.manager.storage.save(task)

    def _persist_tester_messages(
        self, task: Task, run: AgentRun[TesterDeps, str] | None
    ) -> None:
        """Snapshot the tester transcript into the task's last testing round.

        Mirrors _persist_task_messages for the tester run: without it the
        tester's reasoning, tool calls and answer would exist only in the
        chat widgets and vanish from the history on the next view_task().
        The round entry is created up front by run_tester; `run` is None
        when the failure happened before the run context was entered.
        """
        if run is None:
            return
        try:
            messages = run.all_messages()
        except Exception:  # noqa: BLE001 - state may be mid-teardown after cancel
            return
        if not task.tester_messages:
            task.tester_messages.append([])
        task.tester_messages[-1] = serialize_messages(messages)
        self.manager.storage.save(task)

    async def _maybe_compact(self, task: Task) -> None:
        """Context compaction gate, called between agent turns (a turn
        boundary is the only safe point: the transcript never ends with
        unanswered tool calls there).

        When the context fill reaches the configured threshold (percent of
        max_context_tokens), the main agent's work is paused, a tool-less
        summarizer agent rewrites the transcript into a structured summary
        (anchored to the previous one, if any), and the transcript is
        replaced by the original user prompt(s) plus that summary. A
        compaction failure is never fatal: it is logged into the service
        events and the run continues with the uncompressed history.
        """
        max_tokens = self.config.max_context_tokens
        if max_tokens <= 0:
            return
        threshold = min(max(self.config.compaction_threshold, 1), 100)
        tokens = self._live_ctx_tokens.get(task.id)
        if tokens is None:
            tokens = last_context_tokens(task.messages)
        if not tokens or tokens * 100 < threshold * max_tokens:
            return
        pct = round(tokens * 100 / max_tokens)
        _, rest = split_original_prompts(task.messages)
        if not any(not is_summary_message(m) for m in rest):
            # Nothing to summarize (only the original prompt(s), possibly
            # plus a previous summary): compacting again would be a no-op.
            return
        self._show_event(
            task,
            f"Context fill {pct}% — pausing the agent and compacting history…",
        )
        try:
            agent = build_compaction_agent(self.config)
            prompt = build_compaction_prompt(
                extract_previous_summary(task.messages),
                render_history_for_summary(task.messages),
                load_compaction_prompts()[1],
            )
            result = await agent.run(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - compaction must not kill the run
            self._show_event(task, f"compaction failed: {type(e).__name__}: {e}")
            return
        task.messages = compacted_messages(task.messages, result.output)
        task.touch()
        self.manager.storage.save(task)
        # The live token reading is stale now (it belongs to the old
        # transcript); drop it so the counter falls back to the compacted
        # transcript until the next model request reports real usage.
        self._live_ctx_tokens.pop(task.id, None)
        self._show_event(
            task,
            "Context compacted: history summarized, original task "
            "prompt(s) kept verbatim.",
        )
        if self.viewed_id == task.id:
            self.view_task(task.id)

    @work(exclusive=True, group="agent")
    async def run_agent(self, task_id: str) -> None:
        self._agent_generation += 1
        generation = self._agent_generation
        # Key for the tool-spinner registry: superseded runs clean up only
        # their own entries.
        run_key: tuple[str, int] = ("agent", generation)
        task = self.manager.get(task_id)
        # Bound only once `async with agent.iter(...)` is entered; the
        # except handlers below must not touch a run that never started.
        # Initialized before the try: build_agent() itself may raise.
        run: AgentRun[AgentDeps, str] | None = None
        cancelled = False
        try:
            if not self.config.is_ready:
                self._show_event(
                    task,
                    "Model is not configured. Set DRCODE_MODEL, DRCODE_BASE_URL "
                    "and DRCODE_API_KEY (env vars or .env); the message is kept "
                    "and will be sent once the model is available.",
                )
                return
            self.agent_running = True
            self._agent_failed.discard(task_id)
            self._current_tool = None
            self._touch_activity()
            self.refresh_status()
            agent = build_agent(self.config, kind=task.kind, root=self.project_root)
            # Chronological message flow: the transcript is the single
            # chronological record — the original prompt lives in it exactly
            # once and is NEVER re-queued/re-sent. Each activation first
            # consumes every pending user message in arrival order, then —
            # only if the last attempt was interrupted — runs one
            # continuation turn without new text to pick the interrupted
            # work back up.
            while task.status is TaskStatus.ACTIVE:
                run = None
                if task.pending_input:
                    text = task.pending_input.pop(0)
                    if self.viewed_id == task_id:
                        if text in self._pending_rendered:
                            # Already shown as a queued block; don't repeat it.
                            self._pending_rendered.remove(text)
                        else:
                            self._write_history_line("user", text)
                elif task.unfinished:
                    if not ends_with_open_request(task.messages):
                        # The interrupted attempt actually finished its
                        # exchange — nothing to resume; clear the flag so
                        # the task can complete normally below.
                        task.unfinished = False
                        self.manager.storage.save(task)
                        break
                    task.unfinished = False
                    self.manager.storage.save(task)
                    text = None
                else:
                    break
                try:
                    # Context compaction gate: before the history for the
                    # next turn is loaded, the fill is checked against the
                    # threshold; when exceeded, the transcript is summarized
                    # in place (history is computed AFTER compaction).
                    await self._maybe_compact(task)
                    history = resumable_history(task.messages)
                    async with agent.iter(
                        text,
                        message_history=history,
                        deps=AgentDeps(
                            root=self.project_root,
                            manager=self.manager,
                            task_id=task_id,
                        ),
                        usage_limits=UsageLimits(
                            request_limit=self.config.request_limit
                        ),
                    ) as run:
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                reasoning_block: ReasoningBlock | None = None
                                assistant_block: AssistantBlock | None = None
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        self._touch_activity()
                                        if isinstance(event, PartStartEvent):
                                            # The first chunk of a new text/thinking
                                            # part arrives as PartStartEvent carrying
                                            # the content itself (only subsequent
                                            # chunks come as deltas); dropping it
                                            # loses the beginning of the message
                                            # from the live view.
                                            part = event.part
                                            if isinstance(part, ThinkingPart):
                                                delta: (
                                                    ThinkingPartDelta | TextPartDelta
                                                ) = ThinkingPartDelta(
                                                    content_delta=part.content
                                                )
                                            elif isinstance(part, TextPart):
                                                delta = TextPartDelta(
                                                    content_delta=part.content
                                                )
                                            else:
                                                continue
                                        elif isinstance(event, PartDeltaEvent):
                                            if isinstance(
                                                event.delta,
                                                ThinkingPartDelta | TextPartDelta,
                                            ):
                                                delta = event.delta
                                            else:
                                                continue
                                        else:
                                            continue
                                        if (
                                            isinstance(delta, ThinkingPartDelta)
                                            and delta.content_delta
                                        ):
                                            if (
                                                reasoning_block is None
                                                or reasoning_block.stream_epoch
                                                != self._chat_epoch
                                            ):
                                                old, reasoning_block = (
                                                    reasoning_block,
                                                    ReasoningBlock(),
                                                )
                                                reasoning_block.stream_epoch = (
                                                    self._chat_epoch
                                                )
                                                if old is not None:
                                                    # A mid-node view_task()
                                                    # re-render orphaned the
                                                    # old block: carry the
                                                    # already-streamed text
                                                    # over, or the rest of
                                                    # this response would
                                                    # never become visible.
                                                    reasoning_block.set_content(
                                                        old._content
                                                    )
                                                reasoning_block.begin()
                                                if self.viewed_id == task_id:
                                                    self._write_chat(reasoning_block)
                                                reasoning_block.append(
                                                    delta.content_delta
                                                )
                                            else:
                                                # Streaming grows the block
                                                # without a new mount: keep
                                                # following only if the tail
                                                # was visible before the
                                                # delta arrived. Deltas are
                                                # appended even when the task
                                                # is off-screen: they must
                                                # survive a later re-render.
                                                reasoning_block.thinking(True)
                                                follow = (
                                                    self._chat_at_bottom()
                                                    if self.viewed_id == task_id
                                                    else False
                                                )
                                                reasoning_block.append(
                                                    delta.content_delta
                                                )
                                                if follow:
                                                    self._stream_follow()
                                        elif (
                                            isinstance(delta, TextPartDelta)
                                            and delta.content_delta
                                        ):
                                            # Stream the answer into the chat
                                            # log as it is produced: the block
                                            # grows in place, so the text shows
                                            # up between the reasoning and
                                            # tool-call blocks (no separate
                                            # preview area under the log).
                                            # Thinking is over for now — stop
                                            # the reasoning block's spinner.
                                            if reasoning_block is not None:
                                                reasoning_block.thinking(False)
                                            if (
                                                assistant_block is None
                                                or assistant_block.stream_epoch
                                                != self._chat_epoch
                                            ):
                                                old, assistant_block = (
                                                    assistant_block,
                                                    AssistantBlock(),
                                                )
                                                assistant_block.stream_epoch = (
                                                    self._chat_epoch
                                                )
                                                if old is not None:
                                                    # See the reasoning branch:
                                                    # re-mount after a mid-node
                                                    # view_task() wipe, keeping
                                                    # the streamed so far.
                                                    assistant_block.set_content(
                                                        old._content
                                                    )
                                                if self.viewed_id == task_id:
                                                    self._write_chat(assistant_block)
                                                assistant_block.append(
                                                    delta.content_delta
                                                )
                                            else:
                                                follow = (
                                                    self._chat_at_bottom()
                                                    if self.viewed_id == task_id
                                                    else False
                                                )
                                                assistant_block.append(
                                                    delta.content_delta
                                                )
                                                if follow:
                                                    self._stream_follow()
                                if reasoning_block is not None:
                                    reasoning_block.finish()
                                if assistant_block is not None:
                                    assistant_block.finish()
                                # The stream is closed: this request's usage
                                # is final. Refresh the context counter in
                                # the task status bar for the running task.
                                ctx_tokens = run_context_tokens(run)
                                if ctx_tokens is not None:
                                    self._live_ctx_tokens[task_id] = ctx_tokens
                                    if self.viewed_id == task_id:
                                        self.refresh_task_status()
                            elif Agent.is_call_tools_node(node):
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        if isinstance(event, FunctionToolCallEvent):
                                            self._touch_activity(event.part.tool_name)
                                        elif isinstance(event, FunctionToolResultEvent):
                                            # Result stops the spinner even
                                            # when the task is off-screen.
                                            self._tool_call_finished(
                                                event.part.tool_call_id
                                            )
                                            self._touch_activity(done=True)
                                        if self.viewed_id != task_id:
                                            continue
                                        if isinstance(event, FunctionToolCallEvent):
                                            args = event.part.args
                                            args_str = (
                                                args
                                                if isinstance(args, str)
                                                else json.dumps(
                                                    args or {}, ensure_ascii=False
                                                )
                                            )
                                            call_text = (
                                                f"{event.part.tool_name}({args_str})"
                                            )
                                            block = ToolCallBlock(call_text)
                                            self._tool_call_started(
                                                run_key,
                                                task_id,
                                                event.part.tool_call_id,
                                                call_text,
                                                block,
                                            )
                                            block.start_spinner()
                                            self._write_chat(block)
                            # Keep the stored history current while the
                            # turn runs: view_task() renders from
                            # task.messages, so without a per-node snapshot
                            # a switch away+back mid-turn shows the pre-turn
                            # transcript and everything streamed before the
                            # switch is lost from the history.
                            self._persist_task_messages(task, run)
                    if run.result is not None:
                        task.messages = serialize_messages(run.result.all_messages())
                        self.manager.storage.save(task)
                    if self.viewed_id == task_id:
                        # The streamed reasoning/assistant blocks are already
                        # in the chat. Re-render from the persisted transcript
                        # only to cover the case where the view was switched
                        # away and back mid-run (the streamed blocks were wiped
                        # by view_task and must be restored from history).
                        self.view_task(task_id)
                except asyncio.CancelledError:
                    # Interrupted (abort, or a newer run taking over). The
                    # consumed prompt is NOT re-queued — it stays in the
                    # transcript at its chronological position;
                    # task.unfinished marks the possibly-undone attempt and
                    # the next activation resumes it without re-sending text.
                    cancelled = True
                    if generation == self._agent_generation and (
                        task.status is TaskStatus.ACTIVE
                    ):
                        # Persist whatever the interrupted turn managed to
                        # do, then sanitize: drop the API-invalid tail (tool
                        # calls without results, the empty interrupted
                        # request) so later messages can append in any order.
                        self._persist_task_messages(task, run)
                        task.messages = sanitized_messages(task.messages)
                        task.unfinished = True
                        self.manager.storage.save(task)
                    # Drop our streaming preview; a newer run (if any) owns
                    # the label now and must not be touched.
                    self._drop_stream_preview(generation)
                    raise
                except Exception as e:  # noqa: BLE001 - show agent errors in the chat
                    # Persist the failure even for a superseded run: it
                    # happened, and the human must be able to see it.
                    self._show_event(task, f"{type(e).__name__}: {e}")
                    if generation == self._agent_generation:
                        # A mid-stream failure leaves a half-streamed orange
                        # preview above the input: clear it, otherwise the
                        # stale fragments hang in the dialog (and would be
                        # shown for whatever the user does next).
                        self._drop_stream_preview(generation)
                    self._persist_task_messages(task, run)
                    task.messages = sanitized_messages(task.messages)
                    # Not lost: unfinished marks the attempt; promotion
                    # resumes it (this session it is excluded via
                    # _agent_failed so a dead gateway can't loop the queue).
                    task.unfinished = True
                    # Demote honestly: an ACTIVE task with no runner is a
                    # state nothing ever picks up again (promote_next skips
                    # it) — the task goes back to the queue instead, and the
                    # queue continues with the next task.
                    task.status = TaskStatus.QUEUED
                    self._agent_failed.add(task_id)
                    self.manager.storage.save(task)
                    break
        finally:
            # Stop any spinner whose result event was lost to an abort or a
            # run error (normally all of them are already stopped).
            self._stop_run_tools(run_key)
            if generation == self._agent_generation:
                self.agent_running = False
                self.refresh_status()
            # A cancelled run (user abort, or a newer run taking over) must
            # not restart the pipeline: the abort stops everything until the
            # user acts, and the newer run advances on its own completion.
            if not cancelled and generation == self._agent_generation:
                if task.status is TaskStatus.AWAITING_CLARIFICATION:
                    # The user may have answered while the run was still
                    # finishing (the answer sits in pending_input and send()'s
                    # clarification transition never fired): don't wait forever
                    # for an answer that has already arrived.
                    self.manager.requeue_answered_clarification(task.id)
                    # The agent paused the task on a clarifying question: show it
                    # and let the pipeline continue while the user answers.
                    if self.viewed_id == task_id and task.clarification:
                        self._write_clarification(task.clarification)
                    self.refresh_sidebar()
                    self._advance()
                elif (
                    task.status is TaskStatus.ACTIVE
                    and not task.pending_input
                    and not task.unfinished
                ):
                    # The agent answered and nothing is pending or left to
                    # resume: the task is done automatically (as if the user
                    # pressed done). An unfinished (interrupted) task stays
                    # ACTIVE paused instead — the poller surfaces it as
                    # "active, nothing running" until the user acts or the
                    # app restarts (promotion then resumes the attempt).
                    self.manager.mark_done(task.id)
                    self.refresh_sidebar()
                    if self.viewed_id == task_id:
                        self.view_task(task.id)
                    # Testing (this task or an older one in the backlog)
                    # always goes before any new development.
                    self._advance()
                elif task.status is TaskStatus.QUEUED:
                    # The run failed and demoted the task back to the queue:
                    # keep the pipeline going with the next task (failed ones
                    # are excluded, so a dead gateway can't loop the queue).
                    self._advance()

    # -- tester run ------------------------------------------------------------------

    def _schedule_tester_retry(self, task_id: str) -> None:
        """Backoff before the next tester attempt for a failed task.

        TESTING tasks block the whole pipeline, so a failed tester run is
        retried (with capped exponential backoff) instead of being dropped
        from the backlog until restart."""
        count = self._tester_fail_count.get(task_id, 1)
        delay = min(5 * 2 ** (count - 1), 300)
        self._tester_retry_at[task_id] = time.monotonic() + delay
        self.set_timer(delay, self._advance)

    @work(exclusive=True, group="tester")
    async def run_tester(self, task_id: str) -> None:
        """After any task completes, verify the project and label the task."""
        self._tester_task_id = task_id
        self._tester_generation += 1
        run_key: tuple[str, int] = ("tester", self._tester_generation)
        task = self.manager.get(task_id)
        # Initialized before the try: build_tester_agent() itself may raise,
        # and the handlers below reference `run`.
        run: AgentRun[TesterDeps, str] | None = None
        cancelled = False
        try:
            if not self.config.is_ready:
                self._tester_fail_count[task_id] = (
                    self._tester_fail_count.get(task_id, 0) + 1
                )
                self._schedule_tester_retry(task_id)
                self._show_event(
                    task,
                    "Tester skipped: model is not configured. Set DRCODE_MODEL, "
                    "DRCODE_BASE_URL and DRCODE_API_KEY (env vars or .env).",
                )
                return
            # Idempotent: _advance() sets this before starting the worker so
            # two _advance() calls in one tick cannot double-start the tester.
            self.tester_running = True
            self._current_tool = None
            self._touch_activity()
            self.refresh_status()
            # New testing round: created up front, filled by
            # _persist_tester_messages as the run progresses. An empty round
            # (crash before any snapshot) is skipped when rendering.
            task.tester_messages.append([])
            if self.viewed_id == task_id:
                self._write_history_line("testing-sep", "")
            tester = build_tester_agent(self.config)
            deps = TesterDeps(
                root=self.project_root, manager=self.manager, task_id=task_id
            )
            try:
                prompt = f"Verify completed task #{task.id}: {task.title}"
                # The tester needs the original assignment to paraphrase it
                # into a commit message (see the git_commit step in its prompt).
                original = first_user_prompt(task.messages)
                if original and original != task.title:
                    prompt += f"\n\nOriginal task:\n{original}"
                async with tester.iter(
                    prompt,
                    deps=deps,
                    usage_limits=UsageLimits(request_limit=self.config.request_limit),
                ) as run:
                    async for node in run:
                        if Agent.is_call_tools_node(node):
                            async with node.stream(run.ctx) as stream:
                                async for event in stream:
                                    if isinstance(event, FunctionToolCallEvent):
                                        self._touch_activity(event.part.tool_name)
                                    elif isinstance(event, FunctionToolResultEvent):
                                        # Result stops the spinner even when
                                        # the task is off-screen.
                                        self._tool_call_finished(
                                            event.part.tool_call_id
                                        )
                                        self._touch_activity(done=True)
                                    if self.viewed_id == task_id and isinstance(
                                        event, FunctionToolCallEvent
                                    ):
                                        args = event.part.args
                                        args_str = (
                                            args
                                            if isinstance(args, str)
                                            else json.dumps(
                                                args or {}, ensure_ascii=False
                                            )
                                        )
                                        call_text = (
                                            f"tester: {event.part.tool_name}"
                                            f"({args_str})"
                                        )
                                        block = ToolCallBlock(call_text)
                                        self._tool_call_started(
                                            run_key,
                                            task_id,
                                            event.part.tool_call_id,
                                            call_text,
                                            block,
                                        )
                                        block.start_spinner()
                                        self._write_chat(block)
                        # Keep the stored history current while the test
                        # runs: switching to the task mid-test shows the
                        # tester's work done so far.
                        self._persist_tester_messages(task, run)
                if run.result is not None and self.viewed_id == task_id:
                    # Re-render from the persisted transcript: the streamed
                    # tool-call blocks above are replaced by the full history
                    # view (separator + reasoning + tool calls + answer).
                    self.view_task(task_id)
            except asyncio.CancelledError:
                # Aborted (or superseded): keep whatever the tester did so
                # far, then re-raise so the worker teardown stays clean.
                cancelled = True
                self._persist_tester_messages(task, run)
                raise
            except Exception as e:  # noqa: BLE001 - show tester errors in the chat
                self._persist_tester_messages(task, run)
                self._tester_fail_count[task_id] = (
                    self._tester_fail_count.get(task_id, 0) + 1
                )
                self._schedule_tester_retry(task_id)
                self._show_event(task, f"tester failed: {type(e).__name__}: {e}")
        finally:
            # Stop any spinner whose result event was lost to an abort or a
            # tester error (normally all of them are already stopped).
            self._stop_run_tools(run_key)
            self.tester_running = False
            self._tester_task_id = None
            self.refresh_status()
            self.refresh_sidebar()
            if task.status is TaskStatus.COMPLETED:
                # Success: give the next testing round of this task a fresh
                # backoff budget.
                self._tester_fail_count.pop(task_id, None)
                self._tester_retry_at.pop(task_id, None)
            # A cancelled run (user abort, or a newer tester taking over)
            # must not restart the pipeline.
            if not cancelled:
                self._advance()

    # -- actions ---------------------------------------------------------------------

    # -- new task mode ---------------------------------------------------------------

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
        self.chat_input.focus()

    # Tab cycles the new-task type through all three kinds.
    _KIND_CYCLE: ClassVar = (TaskKind.FEATURE, TaskKind.BUG, TaskKind.DOCUMENTATION)

    def toggle_new_task_kind(self) -> None:
        idx = self._KIND_CYCLE.index(self.new_task_kind)
        self.new_task_kind = self._KIND_CYCLE[(idx + 1) % len(self._KIND_CYCLE)]
        self._update_kind_label()

    def _update_kind_label(self) -> None:
        label = self.query_one("#nt-kind-label", Static)
        label.remove_class("kind-feature", "kind-bug", "kind-documentation")
        if self.new_task_kind is TaskKind.BUG:
            label.update("type: bug")
            label.add_class("kind-bug")
        elif self.new_task_kind is TaskKind.DOCUMENTATION:
            label.update("type: documentation")
            label.add_class("kind-documentation")
        else:
            label.update("type: feature")
            label.add_class("kind-feature")

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

    def action_cycle_task_kind(self) -> None:
        """Cycle the type of the viewed task (feature → bug → documentation).
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

    def action_toggle_hold(self) -> None:
        """Park the viewed task out of the backlog (hold) or return a held
        task to where it was. Works for queued tasks and for tasks awaiting
        a clarifying answer (the question is preserved and shown again on
        resume). Held tasks are never taken into work until explicitly
        resumed; typed input keeps accumulating."""
        task = self.viewed
        if task is None:
            return
        if task.status in (TaskStatus.QUEUED, TaskStatus.AWAITING_CLARIFICATION):
            self.manager.set_hold(task.id, on=True)
        elif task.status is TaskStatus.HOLD:
            self.manager.set_hold(task.id, on=False)
        else:
            return
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
        self.chat_input.focus()

    def action_settings(self) -> None:
        """Open the Settings modal (the command palette, Ctrl+S)."""
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(SettingsPalette(id="--command-palette"))

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

    def action_update_agents_md(self) -> None:
        """Create a documentation task that writes/refreshes AGENTS.md.

        The task is pre-filled with the bundled init-repo prompt (its final
        paragraph names the actual project directory) and runs with the
        `documentation` kind: the tester is skipped and mark_done() completes
        it right away."""
        task = self.manager.create(
            "Update/create AGENTS.md",
            load_init_repo_prompt(self.project_root),
            kind=TaskKind.DOCUMENTATION,
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
                providers=[ThemeProvider],
                placeholder="Search for themes…",
            )
        )

    # -- events -------------------------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, TaskItem) and item.task_id:
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
        self.manager.send(task.id, text)
        if task.status is TaskStatus.ACTIVE:
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
        # For an active task run_agent() renders the message itself.
        if (
            task.status is not TaskStatus.ACTIVE
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


def run_app(project_root: Path) -> None:
    DrCodeApp(project_root).run()
