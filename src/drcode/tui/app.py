from __future__ import annotations

import asyncio
import io
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from functools import partial
from operator import attrgetter
from pathlib import Path
from typing import Any, ClassVar, cast

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRun
from pydantic_ai.exceptions import ModelHTTPError
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
from textual.app import (
    App,
    ComposeResult,
    SystemCommand,
    get_system_commands_provider,
)
from textual.binding import Binding
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
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.css.styles import RulesMap
from textual.geometry import Size
from textual.screen import Screen
from textual.strip import Strip
from textual.theme import Theme
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Label,
    ListItem,
    ListView,
    LoadingIndicator,
    Static,
    TextArea,
)

from ..agent import (
    AgentDeps,
    TesterDeps,
    backdate_user_prompt,
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
    probe_model,
    render_history_for_summary,
    resumable_history,
    run_context_tokens,
    sanitized_messages,
    serialize_messages,
    split_original_prompts,
)
from ..config import Config
from ..github import (
    GitHubClient,
    GitHubError,
    GitHubSyncState,
    GitHubSyncStateStore,
    issue_commit_hint,
    sync_issues,
)
from ..models import (
    Clarification,
    PendingInput,
    Task,
    TaskEvent,
    TaskKind,
    TaskStatus,
    VerificationStatus,
)
from ..prompts import load_compaction_prompts, load_init_repo_prompt
from ..settings import (
    FINALIZATION_OPTIONS,
    RETENTION_OPTIONS,
    STALL_NUDGE_OPTIONS,
    CompletedRetention,
    ProjectSettings,
    ProjectSettingsStore,
    StallNudge,
    TaskFinalization,
)
from ..task_manager import TaskManager
from ..version import APP_VERSION
from .agents_dialog import AgentsMdDialog
from .github_dialog import GitHubSettingsDialog, GitHubSetup
from .header import HeaderConnInfo, HeaderSessionInfo, ModelStatusHeader
from .setup_dialog import ModelSetup, ModelSetupDialog

# Sort anchor for history entries without a recorded timestamp (old
# transcripts predating per-message stamps): they keep their relative
# order and sort before everything that has a real time.
_MIN_TIME = datetime.min.replace(tzinfo=UTC)

# Braille spinner frames for the "work in progress" animation shown in a
# reasoning block while the model is thinking and in a tool-call block while
# the tool is executing.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_INTERVAL = 0.12

# Rate-limit (HTTP 429) handling for agent runs: a rate-limited run is
# retried in place (the task stays ACTIVE) after a backoff delay, at most
# RATE_LIMIT_RETRIES times — only when the budget is exhausted does the
# run fall back to the ordinary failure path (requeue + session exclusion).
RATE_LIMIT_RETRIES = 5
# Backoff: 5s doubling per failure, capped here (5, 10, 20, 40, 60).
RATE_LIMIT_DELAY_MAX = 60.0

# How many chat blocks one step of the batched history drain mounts before
# yielding to the event loop (see _flush_chat_batch). ~1-2ms per block means
# a step stays around a 60ms frame even on the slowest widgets.
_CHAT_MOUNT_CHUNK = 16


class TaskItem(ListItem):
    def __init__(
        self,
        text: str,
        task_id: str | None = None,
        *,
        header: bool = False,
        section: str | None = None,
    ) -> None:
        super().__init__(
            Label(text, classes="section-header" if header else "task-item")
        )
        self.task_id = task_id
        self.text = text
        # For section headers: the collapsible category they belong to
        # (None on task rows). Enter/space/click on a header toggles it.
        self.section = section


class Sidebar(ListView):
    """Task list with collapsible pipeline sections.

    Space on a highlighted section header toggles it (Enter and mouse
    clicks are handled via the ListView.Selected event in the app).
    """

    BINDINGS: ClassVar = [
        Binding("space", "toggle_section", "Expand/collapse", show=False),
    ]

    def action_toggle_section(self) -> None:
        item = self.highlighted_child
        if isinstance(item, TaskItem) and item.section is not None:
            app = cast("DrCodeApp", self.app)
            app.toggle_sidebar_section(item.section)

    def _on_list_item__child_clicked(self, event: ListItem._ChildClicked) -> None:
        """Ignore clicks on items removed by a sidebar rebuild.

        refresh_sidebar() clears and re-appends the items, and
        ListView.clear() removes the old widgets asynchronously (it returns
        an AwaitRemove) — so a click can land on an item that is still in
        the DOM but already pruned from the ListView's node list. The stock
        handler does an unchecked index lookup and crashes on it.

        Textual dispatches this handler for every class in the MRO, so the
        ListView base implementation runs after this one — prevented here
        via prevent_default(); no super() call (it would double-run both).
        """
        if event.item not in self.children:
            event.stop()
            event.prevent_default()


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


def _header_line_width(widget: Static) -> int:
    """Usable width of a block's header line (0 before layout → fallback)."""
    return widget.content_size.width or widget.size.width or 80


def _stamp_label(at: datetime | None) -> str:
    """Right-side header stamp for a block, "" when no time is known."""
    if at is None:
        return ""
    return at.astimezone().strftime("%H:%M:%S")


def _append_stamp(header: Text, stamp: str, width: int) -> None:
    """Push `stamp` to the right edge of a header line `width` cells wide.

    Just enough padding is inserted between the header text and the stamp;
    when the terminal is too narrow for both, the stamp still renders,
    separated by the minimum gap (it may wrap/clip, which beats vanishing).
    """
    if not stamp:
        return
    pad = width - cell_len(header.plain) - len(stamp) - 1
    header.append(" " * max(pad, 1), "dim")
    header.append(stamp, "dim italic")


class ReasoningBlock(Static):
    """Collapsible model-reasoning block; collapsed by default, click to toggle.

    While the model is thinking, `append()` streams text into the block and
    an animated spinner runs in the header; `finish()` (the request stream
    ended) stops it. `thinking(False)` pauses the spinner mid-stream when the
    model moved on from thinking to the answer text.
    """

    def __init__(self, at: datetime | None = None) -> None:
        super().__init__(classes="reasoning collapsed")
        self.at = at or datetime.now()
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

    def on_resize(self) -> None:
        # The stamp is pushed to the right edge by padding: re-pad on width
        # changes (terminal resize).
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
        _append_stamp(text, _stamp_label(self.at), _header_line_width(self))
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
            # layout=True: the block's height is auto (computed at layout
            # time from the renderable) and the content just grew — a
            # repaint-only refresh would paint into the stale region and
            # clip the message to the old height until some other event
            # forces a re-layout.
            self.refresh(layout=True)

    def finish(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        self._streaming = False
        self._text = None
        self.refresh(layout=True)

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

    def __init__(self, call: str, at: datetime | None = None) -> None:
        super().__init__(classes="tool-call collapsed")
        self._header, self._args = self._split(call)
        self._header += self._primary_arg_suffix(self._args)
        self.at = at or datetime.now()
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

    def on_resize(self) -> None:
        # The stamp is pushed to the right edge by padding: re-pad on width
        # changes (terminal resize).
        self._redraw()

    def _redraw(self) -> None:
        # Plain Text, not a markup string — tool arguments are model JSON and
        # may contain bracket sequences Textual's markup parser takes for tags.
        arrow = "▸" if self.has_class("collapsed") else "▾"
        marker = (
            _SPINNER_FRAMES[self._spin_frame] if self._spin_timer is not None else "⚙"
        )
        text = Text(f"{arrow} {marker} {self._header}")
        _append_stamp(text, _stamp_label(self.at), _header_line_width(self))
        if not self.has_class("collapsed") and self._args:
            text.append("\n").append(self._args)
        self.update(text)


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


class DrCodeApp(App[None]):
    # The stock command-palette binding (ctrl+p, shown as "palette" in the
    # footer) is replaced by our own "Settings" binding (ctrl+s) below;
    # disabling the built-in also stops it from shadowing the app's own
    # settings binding.
    ENABLE_COMMAND_PALETTE: ClassVar[bool] = False

    # Header title: the stock Textual default is the class name (DrCodeApp).
    # Unannotated on purpose — Textual declares TITLE as an instance
    # variable, so a ClassVar override trips pyright.
    TITLE = "DrCode"

    # Terminal tab title (iTerm2 etc.): Textual never emits an OSC title
    # sequence itself, so the tab falls back to the process name ("Python").
    # We write OSC 0 on mount (see _set_terminal_title) and clear it on
    # unmount so the title doesn't linger after the app exits.
    TERMINAL_TITLE: ClassVar[str] = "DrCode"

    # Providers for the Settings palette (it passes no explicit providers,
    # so it reads this set): the system commands (theme, quit, …) plus our
    # own service commands (AGENTS.md refresh and friends).
    COMMANDS: ClassVar = {
        get_system_commands_provider,
        ServiceCommandsProvider,
        ProjectSettingsCommandsProvider,
    }

    # Drop the built-in "Maximize the focused widget" system command from the
    # Settings palette: maximizing a single widget is of no use here (and the
    # paired "Minimize" command never appears anyway, since nothing can get
    # maximized without it).
    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        for command in super().get_system_commands(screen):
            if command.title == "Maximize":
                continue
            yield command

    CSS = """
    Screen { layers: base toast; }
    #main { height: 1fr; }
    #sidebar-col { width: 36; border-right: solid $primary; }
    #sidebar { height: 1fr; border: none; }
    # Sidebar bottom: project-info then activity-status, plain flow (no
    # dock) so they always stack in compose order under the list.
    # project-info is the directory line plus one blank line; the activity
    # line (agent status left, app name+version right) and the detail line
    # below it are rendered by _render_activity().
    #project-info { height: auto; padding: 0 1; text-align: right; }
    #project-info .project-dir { color: $text-muted; }
    # Connection status lives in the header (see tui/header.py), docked
    # left between the icon and the title; the ok/warn/error classes are
    # set by _check_model().
    #model-status { width: auto; padding: 0 1; }
    #model-status.ok { color: $success; }
    #model-status.warn { color: $warning; }
    #model-status.error { color: $error; }
    #activity-status {
        height: auto; padding: 0 1;
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
    # Literal colors: they must match the markup colors used for the kind
    # descriptions (#nt-kinds, _KIND_COLOR) and the task status bar — the
    # theme's $success/$error are different shades and would not line up.
    #nt-kind-label.kind-feature { color: green; }
    #nt-kind-label.kind-bug { color: red; }
    #nt-kind-label.kind-research { color: yellow; }
    #nt-hint { width: auto; color: $text-disabled; }
    #nt-kinds {
        width: auto;
        margin-top: 1;
        color: $text-disabled;
    }
    """
    BINDINGS: ClassVar = [
        ("ctrl+n", "new_task", "New task"),
        ("ctrl+d", "done_task", "Done"),
        ("ctrl+x", "cancel_task", "Cancel task"),
        ("ctrl+o", "focus_sidebar", "Tasks"),
        ("ctrl+i", "focus_input", "Input"),
        ("ctrl+t", "cycle_task_kind", "Change type"),
        ("ctrl+up", "move_task_up", "Move up"),
        ("ctrl+down", "move_task_down", "Move down"),
        ("ctrl+s", "settings", "Settings"),
        # Two bindings on one key: the footer (and the key dispatch) picks
        # the first one whose check_action allows it — so exactly one of
        # Hold/Resume is shown, depending on the viewed task's status.
        ("ctrl+h", "hold_task", "Hold"),
        ("ctrl+h", "resume_task", "Resume"),
        ("ctrl+delete", "delete_task", "Delete held"),
        ("escape", "abort_run", "Stop agent"),
        ("ctrl+r", "nudge_run", "Nudge agent"),
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
        # GitHub Issues sync: persisted bookkeeping (last sync, seen issue
        # numbers) and the monotonic time of the next scheduled cycle. The
        # first cycle runs shortly after startup.
        self._github_state_store = GitHubSyncStateStore(project_root)
        self._github_state: GitHubSyncState = self._github_state_store.load()
        self._github_next_sync = time.monotonic() + 15.0
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
        # Rate-limit (HTTP 429) bookkeeping for agent runs, per task:
        # consecutive 429 failures this session (retry budget) and the
        # earliest monotonic time the delayed re-run fires at. Unlike a
        # generic failure (immediate requeue), a rate-limited run keeps
        # the task ACTIVE and is retried after a backoff delay.
        self._agent_429_count: dict[str, int] = {}
        self._agent_429_retry_at: dict[str, float] = {}
        # Total 429 backoffs this session (the per-task dict above holds
        # only the current retry budgets); shown in the expanded header.
        self._agent_429_total = 0
        # Session statistics for the expanded header (never persisted):
        # tasks present at startup don't count as "created this session",
        # tasks already COMPLETED at startup don't count as "processed".
        self._session_known_ids: set[str] = set(self.manager.tasks)
        self._session_completed_ids: set[str] = {
            t.id
            for t in self.manager.tasks.values()
            if t.status is TaskStatus.COMPLETED
        }
        self._session_created: dict[TaskKind, int] = {kind: 0 for kind in TaskKind}
        self._session_completed = 0
        # Activation → completion spans (seconds) of the tasks completed
        # this session; feeds the average/median figures.
        self._session_durations: list[float] = []
        # True while the new-task form covers the chat view. Must exist from
        # the start: the streaming loop reads it on every text delta, and an
        # unset attribute would crash the run (it did — the AttributeError
        # surfaced as a red "error ›" line and killed the first streamed run
        # after a startup resume).
        self._new_task_mode = False
        # Task that was open in the chat when the new-task form was entered;
        # cancel (ESC) restores its view (action_new_task() clears
        # viewed_id, but the chat log keeps showing the previous task).
        self._new_task_prev_id: str | None = None
        # Snapshot of the task state the sidebar was last rendered from; the
        # poller compares against it to redraw only on actual change.
        self._sidebar_snapshot: tuple = ()
        # Sidebar sections currently collapsed (Enter/space/click on a
        # section header toggles it); session-only, not persisted.
        self._collapsed_sections: set[str] = set()
        # Last moment an agent/tester event (stream delta, tool call/result)
        # was seen; the activity label compares it against now. Zero means
        # the current run has not emitted anything yet.
        self._last_activity = 0.0
        # Tool the run is currently waiting on (name at call start, cleared
        # on the result event) — lets the label distinguish "tool busy" from
        # "total silence".
        self._current_tool: str | None = None
        # Sliding window of streamed output ((monotonic stamp, approximated
        # token count) samples) — feeds the live tok/s figure in the header
        # connection status. See _record_stream_tokens/_tokens_per_second.
        self._tps_samples: deque[tuple[float, int]] = deque()
        # Base text of the header connection label ("● model" / "✗ …"),
        # written by _check_model; _render_model_status appends the live
        # activity suffix (tok/s / waiting / testing) on top of it.
        self._conn_label = "○ checking…"
        # Connection probe bookkeeping for the expanded header: last
        # round-trip of probe_model in ms (None = no probe finished yet)
        # and the streak of consecutive failed probes (reset on success).
        self._last_probe_ms: int | None = None
        self._probe_fail_streak = 0
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
        # Tasks whose latest compaction attempt failed: the mid-run
        # compaction gate must not stop the run for them (a persistently
        # failing summarizer would otherwise loop break→continuation→break
        # forever); turn-boundary compaction still retries each turn.
        self._compaction_failed: set[str] = set()
        # Monotonic stamp of the last coalesced follow-the-tail scroll while
        # streaming (see _stream_follow).
        self._last_follow = 0.0
        # Bumped by every view_task(): the streaming loop compares it with
        # each block's stream_epoch to detect that a mid-node re-render
        # orphaned the live blocks and they must be re-mounted.
        self._chat_epoch = 0
        # While True, _write_chat() collects blocks into _chat_batch instead
        # of mounting them one by one (view_task's history rebuild mounts the
        # whole timeline chunked + one final scroll — per-block mounting of
        # a fat transcript forces a layout/scroll pass per block and freezes
        # the event loop for seconds).
        self._chat_batching = False
        self._chat_batch: list[Widget] = []
        # _chat_epoch value the running drain belongs to: a view switch
        # mid-drain bumps the epoch and the remaining stale chunks are
        # dropped instead of mounting into the new view's log.
        self._chat_batch_epoch = 0
        # Clarification object currently rendered in the open chat view.
        # Compared by identity: a newly asked question is a fresh instance,
        # so the agent run's end-of-run handler never re-draws a question
        # that a view_task() re-render has already put on screen (which
        # used to show the question twice while the dialog stayed open).
        self._rendered_clarification: Clarification | None = None

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield ModelStatusHeader(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="sidebar-col"):
                yield Sidebar(id="sidebar")
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
                    yield Static("", id="nt-kinds")
        yield Static("", id="statusbar")
        yield Static("", id="copy-toast", classes="hidden")
        yield Footer()

    def _set_terminal_title(self, title: str) -> None:
        """Write an OSC 0 sequence so the terminal shows `title` in its tab.

        iTerm2 (and other xterm-like terminals) title a tab from the last
        title escape received, falling back to the job name — which is why
        an untitled TUI shows up as "Python".
        """
        driver = self._driver
        if driver is not None:
            driver.write(f"\x1b]0;{title}\x07")
            driver.flush()

    def on_unmount(self) -> None:
        # Clear the tab title so "DrCode — <project>" doesn't stick around
        # in the terminal after the app has exited.
        self._set_terminal_title("")

    def on_mount(self) -> None:
        self._set_terminal_title(f"{self.TERMINAL_TITLE} — {self.project_root.name}")
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
        # Open the chat on the task at the top of the first non-empty
        # sidebar section (same pick the sidebar renders first), not on the
        # active task / newest entry.
        initial = self._sidebar_top_task()
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
        # Stall watchdog: auto-nudge a run that stopped showing signs of
        # life for the configured period (see _poll_stall).
        self.set_interval(1.0, self._poll_stall)
        # GitHub Issues sync scheduler: every 30s, checks whether the
        # configured interval has elapsed (see _poll_github_sync).
        self.set_interval(30, self._poll_github_sync)
        # Resume after restart. _advance() re-tests TESTING tasks (the app
        # died during testing) FIRST — the startup demotion (active ->
        # queued) leaves pending_input stranded unless the queue is
        # promoted here. With no model configured anywhere, the setup
        # dialog takes precedence: the pipeline stays idle until the
        # dialog is saved (or skipped with ESC — tasks can still be
        # created and browsed, they just never start).
        if self.config.is_ready:
            self._advance()
            # Model settings exist — offer to create AGENTS.md if the
            # project has none (the setup-dialog path offers it too,
            # right after the settings are saved).
            self._maybe_offer_agents_md()
        else:
            self.push_screen(
                ModelSetupDialog(
                    model=self.config.model,
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    max_context_tokens=self.config.max_context_tokens,
                ),
                self._model_setup_done,
            )

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
        if setup.max_context_tokens is not None:
            self.config.max_context_tokens = setup.max_context_tokens
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

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Gate the app-level bindings on the current state: a binding that
        can't do anything right now (no viewed task, wrong status, nothing
        running) is hidden from the footer instead of failing silently.

        True = enabled+visible, False = disabled+hidden, None = disabled
        but grayed out in the footer.
        """
        task = self.viewed
        if action == "done_task":
            return task is not None and task.status is not TaskStatus.COMPLETED
        if action == "cancel_task":
            return task is not None and task.status is not TaskStatus.COMPLETED
        if action == "cycle_task_kind":
            # Only a queued (not yet started) task can be re-typed.
            return task is not None and task.status is TaskStatus.QUEUED
        if action in ("move_task_up", "move_task_down"):
            # Manual reordering works on queued tasks only; the binding is
            # hidden while the task sits at the corresponding edge of the
            # queue or at the bug-block boundary (bugs always run first).
            if task is None or task.status is not TaskStatus.QUEUED:
                return False
            order = self.manager.queued()
            if len(order) < 2 or task not in order:
                return False
            i = order.index(task)
            j = i - 1 if action == "move_task_down" else i + 1
            if not 0 <= j < len(order):
                return False
            return (order[j].kind is TaskKind.BUG) == (task.kind is TaskKind.BUG)
        if action == "hold_task":
            return task is not None and task.status in (
                TaskStatus.QUEUED,
                TaskStatus.AWAITING_CLARIFICATION,
            )
        if action == "resume_task":
            return task is not None and task.status is TaskStatus.HOLD
        if action == "delete_task":
            return task is not None and task.status is TaskStatus.HOLD
        if action == "abort_run":
            return self.agent_running or self.tester_running
        if action == "nudge_run":
            # A nudge makes sense on a task the pipeline is (or was) working
            # on: abort the hung run and restart it. Also the one-key resume
            # for "active, nothing running" after a previous abort.
            return task is not None and task.status in (
                TaskStatus.ACTIVE,
                TaskStatus.TESTING,
            )
        return True

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
        with the git branch (`~/pr/drcode:main`), then a blank line. The
        app name/version moved onto the activity-status line below."""
        branch = self._git_branch()
        dir_text = escape(self._display_dir())
        where = f"{dir_text}:{branch}" if branch else dir_text
        self.project_info.update(f"[dim]{where}[/]\n")

    def _sidebar_fingerprint(self) -> tuple:
        """Lightweight snapshot of everything the sidebar renders: status,
        verification label, title, the viewed-marker and the queue stamp
        (the QUEUE section is order-sensitive) per task."""
        return tuple(
            (
                t.id,
                t.status.value,
                t.kind.value,
                t.verification,
                t.title_display,
                t.id == self.viewed_id,
                t.queue_seq,
                t.queue_head,
            )
            for t in self.manager.all_tasks()
        )

    def _poll_sidebar(self) -> None:
        """Redraw the sidebar only when the underlying task state changed
        since the last render (status transitions, new bug reports from the
        tester, promote_next() activations, …)."""
        if self._sidebar_fingerprint() != self._sidebar_snapshot:
            self.refresh_sidebar()
            # The task status header (and the bottom statusbar) have no
            # poller of their own: they are refreshed explicitly, and mid-run
            # state changes (set_verification flips the viewed task to
            # COMPLETED while its tester run is still finishing) would leave
            # them stale until the run ends — or a view switch.
            self.refresh_status()

    def _sidebar_sections(self) -> list[tuple[str, list[Task]]]:
        """Sidebar content: pipeline sections top to bottom, empty ones
        dropped (the sidebar shows only the stages that have tasks).

        Shared by the sidebar rendering and the startup view pick (the
        opened task must be the one at the top of the sidebar).
        """
        # Group tasks by status; each group keeps the created_at order of
        # all_tasks() unless sorted explicitly below.
        by_status: dict[TaskStatus, list[Task]] = {}
        for t in self.manager.all_tasks():
            by_status.setdefault(t.status, []).append(t)

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

        sections: list[tuple[str, list[Task]]] = [
            ("QUEUE", queue),
            ("NEEDS CLARIFICATION", clarification),
            ("ACTIVE", active),
            ("TESTING", testing),
            ("COMPLETED", completed),
            ("ON HOLD", hold),
        ]
        return [(name, tasks) for name, tasks in sections if tasks]

    def _sidebar_top_task(self) -> Task | None:
        """First task of the first non-empty sidebar section — what the
        chat pane opens on startup."""
        for _name, tasks in self._sidebar_sections():
            return tasks[0]
        return None

    def refresh_sidebar(self) -> None:
        sidebar = self.sidebar
        sidebar.clear()
        for name, tasks in self._sidebar_sections():
            # A collapsed section keeps only its header line: the "+" shows
            # there is content hidden underneath, "-" means fully expanded.
            collapsed = name in self._collapsed_sections
            marker = "+" if collapsed else "-"
            sidebar.append(
                TaskItem(f"{marker} {name} ({len(tasks)})", header=True, section=name)
            )
            if collapsed:
                continue
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
                    TaskKind.RESEARCH: "[yellow]R[/]",
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

    def toggle_sidebar_section(self, section: str) -> None:
        """Collapse/expand one sidebar category (its header row).

        The rebuild resets the ListView cursor, so it is put back onto the
        toggled header — the next space/enter keeps working on it.
        """
        if section in self._collapsed_sections:
            self._collapsed_sections.discard(section)
        else:
            self._collapsed_sections.add(section)
        self.refresh_sidebar()
        sidebar = self.sidebar
        for index, item in enumerate(sidebar.children):
            if isinstance(item, TaskItem) and item.section == section:
                sidebar.index = index
                break

    def _refocus_sidebar_task(self, task_id: str) -> None:
        """Put the ListView cursor back onto a task's row after a sidebar
        rebuild (the rebuild resets it), so a series of reorder/hold keys
        keeps working on the same task without re-selecting it."""
        sidebar = self.sidebar
        for index, item in enumerate(sidebar.children):
            if isinstance(item, TaskItem) and item.task_id == task_id:
                sidebar.index = index
                break

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
            self._conn_label = "○ model not configured"
            self._last_probe_ms = None
            label.remove_class("ok", "error")
            label.add_class("warn")
            self._render_model_status()
            return
        model = self.config.model
        started = time.monotonic()
        ok, detail = await probe_model(model, self.config.base_url, self.config.api_key)
        # Round-trip of the probe itself, shown on the expanded header line.
        self._last_probe_ms = int((time.monotonic() - started) * 1000)
        if not ok:
            self._probe_fail_streak += 1
            self._conn_label = f"✗ {model} — {detail}"
            label.remove_class("ok", "warn")
            label.add_class("error")
            self._render_model_status()
            return
        self._probe_fail_streak = 0
        self._conn_label = f"● {model}"
        label.remove_class("warn", "error")
        label.add_class("ok")
        self._render_model_status()
        if was_error:
            self._recover_from_outage()

    # -- live output rate (header tok/s) --------------------------------------

    # Window over which the streamed output rate is averaged, and the minimum
    # span of samples before a rate is shown at all (the very first second of
    # a stream would otherwise produce a wildly spiky figure).
    _TPS_WINDOW = 6.0
    _TPS_MIN_SPAN = 1.5

    def _record_stream_tokens(self, text: str) -> None:
        """Record a streamed delta for the live tok/s estimate.

        Streaming APIs emit deltas roughly token-sized, so the character
        count divided by the conventional ~4 chars/token is a good enough
        approximation — the point is the visual "is it producing" signal,
        not billing-grade accounting.
        """
        now = time.monotonic()
        self._tps_samples.append((now, max(1, len(text) // 4)))
        self._prune_tps_samples(now)

    def _prune_tps_samples(self, now: float) -> None:
        cutoff = now - self._TPS_WINDOW
        samples = self._tps_samples
        while samples and samples[0][0] < cutoff:
            samples.popleft()

    def _tokens_per_second(self) -> float | None:
        """Output rate over the sliding window (None = nothing measurable:
        no recent stream output, or too little of it to average yet)."""
        now = time.monotonic()
        self._prune_tps_samples(now)
        samples = self._tps_samples
        if not samples:
            return None
        span = now - samples[0][0]
        if span < self._TPS_MIN_SPAN:
            return None
        return sum(n for _, n in samples) / span

    def _render_model_status(self) -> None:
        """Header connection label + live activity suffix.

        The suffix is the global "is the model producing" signal: a tok/s
        figure while tokens stream in (regardless of which task is open in
        the chat), "waiting" while a run is alive but the model is silent
        (a tool executing, a non-streamed request in flight), "testing"
        while the tester works, and just the plain connection label when
        nothing runs.
        """
        label = self.model_status
        tps = self._tokens_per_second()
        if tps is not None:
            label.update(f"{self._conn_label} · {tps:.0f} tok/s")
        elif self.agent_running:
            label.update(f"{self._conn_label} · waiting")
        elif self.tester_running:
            label.update(f"{self._conn_label} · testing")
        else:
            label.update(self._conn_label)

    # -- expanded (tall) header: connection details + session statistics ----

    @property
    def header_conn_info(self) -> Static:
        return self.query_one("#header-conn", HeaderConnInfo)

    @property
    def header_session_info(self) -> Static:
        return self.query_one("#header-sessions", HeaderSessionInfo)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """Compact human duration for the header stats line (e.g. 45s, 6m12s)."""
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds}s"
        return f"{seconds // 60}m{seconds % 60:02d}s"

    def _collect_session_stats(self) -> None:
        """Fold task-tree changes into the session counters.

        Runs on the 1 s poll: tasks that appeared since the previous tick
        count as created this session (by kind), tasks that reached
        COMPLETED count as processed, with the activation → completion
        span (when both timestamps exist — a task cancelled straight from
        the queue has no activation stamp and is counted without a
        duration).
        """
        tasks = self.manager.tasks
        for task_id in set(tasks) - self._session_known_ids:
            self._session_known_ids.add(task_id)
            self._session_created[tasks[task_id].kind] += 1
        for task in tasks.values():
            if (
                task.status is TaskStatus.COMPLETED
                and task.id not in self._session_completed_ids
            ):
                self._session_completed_ids.add(task.id)
                self._session_completed += 1
                if task.activated_at and task.completed_at:
                    self._session_durations.append(
                        (task.completed_at - task.activated_at).total_seconds()
                    )

    def _header_conn_text(self) -> str:
        """The expanded header's connection line: the connection label
        (dot + model, error detail on failures), URL, last probe
        round-trip, consecutive probe failures and 429 retry counters."""
        if not self.config.is_ready:
            return "model not configured"
        url = self.config.base_url
        probe = (
            f"probe {self._last_probe_ms} ms"
            if self._last_probe_ms is not None
            else "probe —"
        )
        fails = f"probe fails: {self._probe_fail_streak}"
        open_429 = sum(self._agent_429_count.values())
        retries = f"429: {self._agent_429_total} total"
        if open_429:
            retries += f" / {open_429} active"
        return f"{self._conn_label} · {url} · {probe} · {fails} · {retries}"

    def _header_stats_text(self) -> str:
        """The expanded header's session-statistics line: tasks completed
        this session, average/median activation→completion time and the
        bugs:features ratio of the tasks created this session (research
        tasks are excluded from the ratio: they are neither bug fixes nor
        implemented features)."""
        durations = self._session_durations
        if durations:
            avg = statistics.fmean(durations)
            median = statistics.median(durations)
            times = (
                f"avg {self._fmt_duration(avg)} · median {self._fmt_duration(median)}"
            )
        else:
            times = "avg — · median —"
        bugs = self._session_created[TaskKind.BUG]
        features = self._session_created[TaskKind.FEATURE]
        return (
            f"tasks done: {self._session_completed} · {times} · "
            f"bugs:features {bugs}:{features}"
        )

    def _render_header_tall(self) -> None:
        """Refresh the two expanded-header lines (no-op while collapsed:
        the widgets are display:none and their content is refreshed on
        toggle / by the next poll tick anyway)."""
        header = self.query_one(ModelStatusHeader)
        if not header.has_class("-tall"):
            return
        self.header_conn_info.update(self._header_conn_text())
        self.header_session_info.update(self._header_stats_text())

    def on_model_status_header_tall_toggled(
        self, event: ModelStatusHeader.TallToggled
    ) -> None:
        """The header was clicked: fill the expanded lines immediately (the
        1 s poll keeps them live afterwards)."""
        if event.tall:
            self._render_header_tall()

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
        try:
            self._render_model_status()
            # Expanded header: fold task-tree changes into the session
            # counters and refresh the two tall-only lines (no-op while
            # the header is collapsed).
            self._collect_session_stats()
            self._render_header_tall()
            # Touch the widget so the teardown race (the 1s timer ticking
            # while the app is being dismantled) raises NoMatches here.
            _ = self.activity_status
        except NoMatches:
            # Teardown race: the 1s timer can tick while the app is being
            # dismantled and its status widgets are already gone.
            return
        if not (self.agent_running or self.tester_running):
            active = self.manager.active
            if active is not None:
                # Rate-limited run waiting for its delayed retry: the task
                # is fine, the re-run timer just has not fired yet.
                retry_at = self._agent_429_retry_at.get(active.id)
                if retry_at is not None and retry_at > time.monotonic():
                    left = max(1, int(retry_at - time.monotonic()))
                    attempt = self._agent_429_count.get(active.id, 0)
                    self._render_activity(
                        "⏳ agent: 429 retry",
                        f"#{active.id} attempt {attempt}/{RATE_LIMIT_RETRIES}"
                        f" in {left}s",
                    )
                    return
                # Task claims to be active but no worker owns it (e.g. the
                # user aborted the run, or an older build stranded it):
                # surface the inconsistency instead of showing a fake idle.
                self._render_activity(
                    "⚠ agent: idle",
                    f"#{active.id} active, nothing running — "
                    "send a message or ctrl+r to (re)start",
                    "stale",
                )
                return
            self._render_activity("○ idle", "")
            return
        who = "agent" if self.agent_running else "tester"
        age = int(time.monotonic() - self._last_activity)
        if self._current_tool:
            detail = f"tool {self._current_tool} running {self._fmt_age(age)}"
        else:
            detail = f"last event {self._fmt_age(age)} ago"
        if age < 30:
            self._render_activity(f"▶ {who}: working", detail, "fresh")
        else:
            limit = self.settings.stall_nudge.minutes
            if limit is not None:
                left = max(1, limit * 60 - age)
                detail += f" · auto-nudge in {self._fmt_age(int(left))}"
            else:
                detail += " (esc to stop, ctrl+r to nudge)"
            self._render_activity(f"⚠ {who}: stalled", detail, "stale")

    def _render_activity(
        self, status: str, detail: str, state: str | None = None
    ) -> None:
        """Sidebar bottom: agent/tester status on the left of one line, the
        app name and version on the right; the live detail (current tool,
        retry countdown, …) sits on the line below with no blank line in
        between. `state` ("fresh"/"stale"/None) picks the status color.
        """
        label = self.activity_status
        label.remove_class("fresh", "stale")
        if state is not None:
            label.add_class(state)
        version = f"DrCode {APP_VERSION}"
        # Content width (the widget's padding is already excluded); falls
        # back to the composed sidebar width minus padding.
        width = label.container_size.width or 34
        pad = max(1, width - cell_len(status) - cell_len(version))
        label.update(
            f"{status}{' ' * pad}[bold $accent]DrCode[/] [white]{APP_VERSION}[/]\n"
            f"{detail}"
        )

    # -- stall watchdog ---------------------------------------------------------

    # Delay between aborting a nudged run and starting its replacement: the
    # cancelled worker's handlers (persist + sanitize the transcript) must
    # finish before the resume run reads the task state.
    _NUDGE_RESTART_DELAY = 1.0

    def _poll_stall(self) -> None:
        """Watchdog: when the running agent/tester has emitted nothing (no
        stream deltas, no tool events) for the configured stall limit, abort
        the hung run and restart it. A stalled run blocks its own task — and
        through the pipeline gate, the whole backlog — so this keeps an
        overnight run going without human intervention."""
        limit = self.settings.stall_nudge.minutes
        if limit is None:
            return
        if not (self.agent_running or self.tester_running):
            return
        stalled_for = time.monotonic() - self._last_activity
        if stalled_for < limit * 60:
            return
        if self.agent_running:
            task = self.manager.active
        else:
            task = (
                self.manager.get(self._tester_task_id) if self._tester_task_id else None
            )
        if task is None:
            return
        who = "agent" if self.agent_running else "tester"
        self._show_event(
            task,
            f"{who} stalled for {self._fmt_age(int(stalled_for))} without any "
            f"activity — aborting the run (stall limit: "
            f"{self.settings.stall_nudge.label})",
        )
        self._nudge_run(task)

    def _nudge_run(self, task: Task) -> None:
        """Abort the hung/stalled run for `task` (if one is in flight) and
        schedule its restart: the agent resumes via a continuation turn (no
        text is re-sent — `task.unfinished` marks the interrupted attempt),
        the tester gets a fresh verification round via the pipeline gate
        (`_advance` re-tests the still-TESTING task)."""
        if task.status is TaskStatus.ACTIVE:
            if self.agent_running:
                self.workers.cancel_group(self, "agent")
                self.agent_running = False
                self._current_tool = None
                self._clear_stream_label()
                self.refresh_status()
            self.set_timer(self._NUDGE_RESTART_DELAY, partial(self.run_agent, task.id))
        elif task.status is TaskStatus.TESTING:
            if self.tester_running:
                self.workers.cancel_group(self, "tester")
                self.tester_running = False
                self._current_tool = None
                self.refresh_status()
            self.set_timer(self._NUDGE_RESTART_DELAY, self._advance)
        elif self.tester_running and self._tester_task_id == task.id:
            # The run hung AFTER its useful work: set_verification already
            # flipped the task to COMPLETED mid-run. Nothing to restart —
            # but the hung worker keeps tester_running True, which blocks
            # the whole pipeline (_advance is a no-op while testing runs),
            # so it must be aborted and the pipeline re-advanced.
            self.workers.cancel_group(self, "tester")
            self.tester_running = False
            self._current_tool = None
            self.refresh_status()
            self.set_timer(self._NUDGE_RESTART_DELAY, self._advance)

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
        if self.settings.task_finalization is TaskFinalization.NONE:
            # No tester at all: every task that reached TESTING (this tick or
            # left over from a run started under another setting) is completed
            # straight away, and the queue continues below.
            finalized = [
                t for t in self.manager.all_tasks() if t.status is TaskStatus.TESTING
            ]
            for t in finalized:
                self.manager.complete_without_verification(t.id)
            if finalized:
                self.refresh_sidebar()
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
        # Drop any un-drained chunks of a previous view: their drain is
        # stale now (the epoch guard below would drop them anyway, but the
        # new _render_task_history appends into the same list).
        self._chat_batch.clear()
        self._chat_batching = True
        try:
            self._render_task_history(task, task_id)
        finally:
            # Opening a task always lands on the latest entry (previously
            # this happened implicitly because every mount scrolled to the
            # end). The finally keeps a mid-render exception from leaving
            # batching on (later live writes would be swallowed).
            self._flush_chat_batch()
        self.refresh_sidebar()
        self.refresh_status()

    def _render_task_history(self, task: Task, task_id: str) -> None:
        """Write the task's full history into the (batched) chat log."""
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
        # Queued user input — interleaved by the moment it was TYPED, not
        # pinned to the tail: comments sent while the task was queued or
        # running belong between the messages that were on screen then.
        for entry in task.pending_input:
            if is_pending_retry(task.messages, entry.text):
                # The failed/interrupted run left this prompt in the
                # transcript: it is already rendered from task.messages
                # above — writing the pending copy again duplicates it.
                continue
            _add(entry.at, "user-pending", entry.text)
        timeline.sort(key=lambda item: item[:2])
        for at, _, role, text in timeline:
            block = self._write_history_line(
                role, text, None if at == _MIN_TIME else at
            )
            if role == "tool-call" and isinstance(block, ToolCallBlock):
                # A tool still executing (run in flight) gets its spinner
                # back on the freshly rendered block.
                self._reattach_tool_spinner(block, text, task_id)
        if task.status is TaskStatus.AWAITING_CLARIFICATION and task.clarification:
            self._write_clarification(task.clarification)
            self._rendered_clarification = task.clarification

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

        While batching (view_task's history rebuild and its chunked drain),
        the block is only collected — _flush_chat_batch() mounts the
        timeline in chunks afterwards.
        """
        if isinstance(content, Widget):
            content.add_class("chat-block")
        else:
            content = Static(content, expand=expand, classes="chat-block")
        if self._chat_batching:
            self._chat_batch.append(content)
            self._chat_has_content = True
            return
        follow = self._chat_at_bottom()
        self.chat_log.mount(content)
        if follow:
            self.chat_log.scroll_end(animate=False)
        self._chat_has_content = True

    def _flush_chat_batch(self) -> None:
        """Start mounting the blocks collected by a batched history rebuild.

        Mounts in chunks, yielding to the event loop between them: one
        mounted block costs a few ms (textual re-applies styles over all
        mounted widgets per mount), so a fat 600-block history would freeze
        keyboard/mouse input for ~2s if mounted in a single synchronous
        pass. Late live writes (a stream starting while the drain is still
        going) keep appending to _chat_batch — batching stays on until the
        last chunk — and ride along with the remaining chunks.
        """
        self._chat_batching = True
        self._chat_batch_epoch = self._chat_epoch
        self._drain_chat_batch()

    def _drain_chat_batch(self) -> None:
        """Mount one chunk of the pending batch, reschedule for the rest."""
        if self._chat_batch_epoch != self._chat_epoch:
            # The view switched mid-drain: the new view_task wiped the log
            # and owns it now — drop the stale tail.
            self._chat_batching = False
            self._chat_batch.clear()
            return
        chunk = self._chat_batch[:_CHAT_MOUNT_CHUNK]
        del self._chat_batch[:_CHAT_MOUNT_CHUNK]
        try:
            if chunk:
                self.chat_log.mount(*chunk)
            if self._chat_batch:
                self.call_after_refresh(self._drain_chat_batch)
            else:
                self._chat_batching = False
                # Opening a task always lands on the latest entry (previously
                # this happened implicitly because every mount scrolled to
                # the end).
                self.chat_log.scroll_end(animate=False)
        except Exception:
            # Never leave batching stuck on: later live writes would be
            # silently swallowed.
            self._chat_batching = False
            self._chat_batch.clear()
            raise

    def _write_history_line(
        self, role: str, text: str, at: datetime | None = None
    ) -> Widget | None:
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
            block = ReasoningBlock(at)
            block.set_content(text)
        elif role == "tool-call":
            block = ToolCallBlock(text, at)
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
        # The footer mirrors the state check_action() gates on (viewed task,
        # its status, running flags): re-evaluate the visible bindings.
        self.refresh_bindings()
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
            TaskKind.RESEARCH: "[bold yellow]research[/]",
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
                f"🧠 {self._fmt_ctx_tokens(tokens)}/"
                f"{self._fmt_ctx_tokens(max_tokens)}·{pct}%"
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

    def _fill_over_threshold(
        self,
        live_tokens: int | None,
        messages: list[dict[str, Any]],
    ) -> int | None:
        """Context fill (tokens) when the compaction threshold is reached
        (percent of max_context_tokens), None otherwise. Prefers the live
        reading of the running turn over the persisted transcript, whose
        usage lags behind by the not-yet-persisted requests."""
        max_tokens = self.config.max_context_tokens
        if max_tokens <= 0:
            return None
        threshold = min(max(self.config.compaction_threshold, 1), 100)
        tokens = live_tokens
        if tokens is None:
            tokens = last_context_tokens(messages)
        if not tokens or tokens * 100 < threshold * max_tokens:
            return None
        return tokens

    def _compaction_due(self, task: Task) -> int | None:
        """Compaction threshold check for the main agent transcript."""
        return self._fill_over_threshold(
            self._live_ctx_tokens.get(task.id), task.messages
        )

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
        tokens = self._compaction_due(task)
        if tokens is None:
            return
        pct = round(tokens * 100 / self.config.max_context_tokens)
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
            self._compaction_failed.add(task.id)
            self._show_event(task, f"compaction failed: {type(e).__name__}: {e}")
            return
        self._compaction_failed.discard(task.id)
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

    async def _maybe_compact_tester_round(
        self, task: Task, live_tokens: int | None
    ) -> bool:
        """Compaction gate for the current testing round, called after
        run_tester paused the round at a tool boundary.

        Mirrors _maybe_compact for the tester, with one crucial difference:
        the summarizer rewrites the ROUND transcript
        (`task.tester_messages[-1]`) in place — `task.messages` (the
        developer transcript) is never touched, the two histories must not
        mix (tester rounds are persisted separately precisely so they never
        leak into the coding agent's model history).

        Returns True when the round was compacted: run_tester must resume
        it as a continuation turn from the compacted transcript (and drop
        the stale live token reading). Returns False when compaction was
        not possible — the threshold is not reached, there is nothing to
        summarize (a no-op), or the summarizer failed — in which case the
        round is resumed uncompressed and the caller adds the task to
        `_compaction_failed`, so the mid-round gate stays off (a
        persistently broken summarizer must not loop pause→resume
        forever); turn-boundary compaction of the main agent still retries
        on later turns.
        """
        round_messages = task.tester_messages[-1] if task.tester_messages else []
        tokens = self._fill_over_threshold(live_tokens, round_messages)
        if tokens is None:
            return False
        pct = round(tokens * 100 / self.config.max_context_tokens)
        _, rest = split_original_prompts(round_messages)
        if not any(not is_summary_message(m) for m in rest):
            # Nothing to summarize (only the original prompt(s), possibly
            # plus a previous summary): compacting again would be a no-op —
            # resuming the round as-is would hit the gate again.
            self._compaction_failed.add(task.id)
            return False
        self._show_event(
            task,
            f"Context fill {pct}% — compacting the testing round…",
        )
        try:
            agent = build_compaction_agent(self.config)
            prompt = build_compaction_prompt(
                extract_previous_summary(round_messages),
                render_history_for_summary(round_messages),
                load_compaction_prompts()[1],
            )
            result = await agent.run(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - compaction must not kill the round
            self._compaction_failed.add(task.id)
            self._show_event(task, f"compaction failed: {type(e).__name__}: {e}")
            return False
        self._compaction_failed.discard(task.id)
        task.tester_messages[-1] = compacted_messages(round_messages, result.output)
        task.touch()
        self.manager.storage.save(task)
        self._show_event(
            task,
            "Context compacted: testing round summarized, original "
            "prompt kept verbatim.",
        )
        if self.viewed_id == task.id:
            self.view_task(task.id)
        return True

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
                    "Model is not configured. Restart drcode to open the setup "
                    "dialog, or set model/base_url in .drcode/drcode.json "
                    "(project) / "
                    "~/.config/drcode/drcode.json / DRCODE_* env vars; the "
                    "message is kept and will be sent once the model is "
                    "available.",
                )
                return
            self.agent_running = True
            self._agent_failed.discard(task_id)
            self._agent_429_retry_at.pop(task_id, None)
            self._current_tool = None
            self._tps_samples.clear()
            self._touch_activity()
            self.refresh_status()
            agent = build_agent(self.config, kind=task.kind, root=self.project_root)
            # Chronological message flow: the transcript is the single
            # chronological record — the original prompt lives in it exactly
            # once and is NEVER re-queued/re-sent. Each activation first
            # consumes every pending user message in arrival order — merged
            # into ONE prompt, so a brief queued up before the run (original
            # message + comments) is solved as a whole, not turn-by-turn —
            # then, only if the last attempt was interrupted, runs one
            # continuation turn without new text to pick the interrupted
            # work back up.
            while task.status is TaskStatus.ACTIVE:
                run = None
                # The consumed pending entries (empty for a continuation
                # turn): kept around so a transient failure can re-queue
                # them with their original "typed at" stamps intact.
                consumed: list[PendingInput] = []
                if task.pending_input:
                    # Batch consumption: everything queued before this turn
                    # (the original brief plus comments typed while the task
                    # sat in the queue or while another turn ran) is sent as
                    # a single prompt — the agent must see the whole picture
                    # at once and decide from all of it.
                    consumed = list(task.pending_input)
                    task.pending_input.clear()
                    text = "\n\n".join(entry.text for entry in consumed)
                    if self.viewed_id == task_id:
                        for entry in consumed:
                            if entry.text in self._pending_rendered:
                                # Already shown as a queued block; don't repeat it.
                                self._pending_rendered.remove(entry.text)
                            else:
                                self._write_history_line("user", entry.text)
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
                        # Sync markers for the per-node re-render (below):
                        # view_dirty — content was produced while the task
                        # was off-screen, so the open chat view (if any) is
                        # missing it; _chat_epoch mismatch — the user
                        # switched views mid-node, and the switch-back
                        # render covered only the persisted transcript.
                        view_dirty = False
                        synced_epoch = self._chat_epoch
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
                                        if delta.content_delta:
                                            self._record_stream_tokens(
                                                delta.content_delta
                                            )
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
                                                    ReasoningBlock(
                                                        reasoning_block.at
                                                        if reasoning_block
                                                        else None
                                                    ),
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
                                                else:
                                                    view_dirty = True
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
                                                if self.viewed_id != task_id:
                                                    view_dirty = True
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
                                                else:
                                                    view_dirty = True
                                                assistant_block.append(
                                                    delta.content_delta
                                                )
                                            else:
                                                if self.viewed_id != task_id:
                                                    view_dirty = True
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
                                # The consumed prompt is now in the transcript
                                # (the model-request node just appended it).
                                # Backdate it to the moment the user typed it,
                                # so the chat thread stays chronological even
                                # when the prompt sat in pending_input for a
                                # while before this run picked it up.
                                if consumed and text is not None:
                                    backdate_user_prompt(
                                        run.all_messages(),
                                        text,
                                        consumed[0].at,
                                    )
                                # The stream is closed: this request's usage
                                # is final. Refresh the context counter in
                                # the task status bar for the running task.
                                ctx_tokens = run_context_tokens(run)
                                if ctx_tokens is not None:
                                    self._live_ctx_tokens[task_id] = ctx_tokens
                                    if self.viewed_id == task_id:
                                        self.refresh_task_status()
                                # A model call went through: the rate-limit
                                # retry budget for this task starts over, so
                                # the NEXT 429 counts from one again — even
                                # mid-run, where a single turn makes many
                                # model requests.
                                self._agent_429_count.pop(task_id, None)
                                self._agent_429_retry_at.pop(task_id, None)
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
                                            # The tool-call block is skipped
                                            # (it will come from the next
                                            # re-render): the open view is
                                            # missing it.
                                            view_dirty = True
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
                                # Mid-run compaction gate: a single turn can
                                # run for many model requests, and the
                                # turn-boundary gate above would only fire
                                # when the whole run ends — the fill can grow
                                # far past the threshold inside one turn.
                                # Stop the run here: the tool results of the
                                # round just executed are the last thing the
                                # model saw, and a break leaves them
                                # uncommitted in run.all_messages() (the
                                # transcript ends with unanswered tool calls
                                # — sanitized away below, the redo of that
                                # one round is the price of the pause). The
                                # attempt is marked unfinished, so the
                                # while-loop's next iteration compact (the
                                # gate above) and resume via a continuation
                                # turn. Skipped when compaction keeps
                                # failing, otherwise a persistently broken
                                # summarizer would loop
                                # break→continuation→break.
                                tokens = self._compaction_due(task)
                                if tokens is not None and (
                                    task.id not in self._compaction_failed
                                ):
                                    pct = round(
                                        tokens * 100 / self.config.max_context_tokens
                                    )
                                    self._show_event(
                                        task,
                                        f"Context fill {pct}% — pausing the "
                                        "run to compact history…",
                                    )
                                    self._persist_task_messages(task, run)
                                    task.messages = sanitized_messages(task.messages)
                                    task.unfinished = True
                                    self.manager.storage.save(task)
                                    break
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
                    if self.viewed_id == task_id and (
                        view_dirty or self._chat_epoch != synced_epoch
                    ):
                        # The streamed reasoning/assistant blocks are already
                        # in the chat: re-rendering from the persisted
                        # transcript after every node would remount the whole
                        # (growing) history each time and freeze the UI for
                        # seconds on fat transcripts. Do it only when the open
                        # view is actually out of sync: content arrived while
                        # the task was off-screen (view_dirty), or the user
                        # switched views mid-node — the switch-back render
                        # covered only the transcript persisted at the
                        # previous node boundary.
                        self.view_task(task_id)
                        view_dirty = False
                        synced_epoch = self._chat_epoch
                    # The turn went through: the rate-limit retry budget for
                    # this task starts over.
                    self._agent_429_count.pop(task_id, None)
                    self._agent_429_retry_at.pop(task_id, None)
                except asyncio.CancelledError:
                    # Interrupted (abort, nudge, or a newer run taking over).
                    # The consumed prompt is NOT re-queued — it stays in the
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
                        if run is None and consumed:
                            # Cancelled before the run context was even
                            # entered (e.g. during the compaction gate): the
                            # prompts were popped from pending_input but
                            # never reached the transcript — put them back,
                            # or the nudge/abort would silently eat them.
                            task.pending_input[:0] = consumed
                            if self.viewed_id == task_id:
                                for entry in consumed:
                                    if entry.text not in self._pending_rendered:
                                        self._pending_rendered.append(entry.text)
                        task.messages = sanitized_messages(task.messages)
                        task.unfinished = True
                        self.manager.storage.save(task)
                    # Drop our streaming preview; a newer run (if any) owns
                    # the label now and must not be touched.
                    self._drop_stream_preview(generation)
                    raise
                except Exception as e:  # noqa: BLE001 - show agent errors in the chat
                    # HTTP 429 (rate limit) is transient: instead of
                    # requeueing the task right away, retry the run in place
                    # after a backoff delay (5s doubling up to 60s), at most
                    # RATE_LIMIT_RETRIES times. Only when the budget is
                    # exhausted does the ordinary failure path below apply.
                    if (
                        isinstance(e, ModelHTTPError)
                        and e.status_code == 429
                        and self._agent_429_count.get(task_id, 0) < RATE_LIMIT_RETRIES
                    ):
                        attempt = self._agent_429_count.get(task_id, 0) + 1
                        self._agent_429_count[task_id] = attempt
                        self._agent_429_total += 1
                        delay = min(5 * 2 ** (attempt - 1), RATE_LIMIT_DELAY_MAX)
                        self._show_event(
                            task,
                            f"HTTP 429 (rate limited): retry {attempt}/"
                            f"{RATE_LIMIT_RETRIES} in {int(delay)} s",
                        )
                        if generation == self._agent_generation:
                            self._drop_stream_preview(generation)
                        self._persist_task_messages(task, run)
                        task.messages = sanitized_messages(task.messages)
                        if run is None and consumed:
                            # The request never started (pydantic-ai appends
                            # the user message before calling the model, and
                            # with `run` None nothing was persisted): put the
                            # consumed prompts back at the head of the pending
                            # queue so the retry re-sends them verbatim (with
                            # their original "typed at" stamps).
                            task.pending_input[:0] = consumed
                            # They were already rendered as user blocks when
                            # they were popped: the retry must not draw them
                            # twice.
                            if self.viewed_id == task_id:
                                for entry in consumed:
                                    if entry.text not in self._pending_rendered:
                                        self._pending_rendered.append(entry.text)
                        if ends_with_open_request(task.messages):
                            # The request is recorded in the transcript: the
                            # retry resumes the attempt without new text.
                            task.unfinished = True
                        self._agent_429_retry_at[task_id] = time.monotonic() + delay
                        # The task stays ACTIVE — no requeue, no pipeline
                        # advance; the delayed re-run below picks it up.
                        self.manager.storage.save(task)
                        self.set_timer(delay, partial(self.run_agent, task_id))
                        return
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
                    # Skip when the end-of-turn view_task() re-render above
                    # already drew it — otherwise the question shows twice
                    # while the dialog stays open (a switch away+back renders
                    # it once, via view_task alone).
                    if (
                        self.viewed_id == task_id
                        and task.clarification
                        and task.clarification is not self._rendered_clarification
                    ):
                        self._write_clarification(task.clarification)
                        self._rendered_clarification = task.clarification
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
            self._tps_samples.clear()
            self._touch_activity()
            self.refresh_status()
            # New testing round: created up front, filled by
            # _persist_tester_messages as the run progresses. An empty round
            # (crash before any snapshot) is skipped when rendering.
            task.tester_messages.append([])
            if self.viewed_id == task_id:
                self._write_history_line("testing-sep", "")
            tester = build_tester_agent(
                self.config,
                allow_commit=(
                    self.settings.task_finalization is TaskFinalization.TESTING_COMMIT
                ),
            )
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
                # A GitHub-imported task must link its commit back to the
                # originating issue ("Fixes #N" keyword in the message).
                if task.github_issue is not None:
                    prompt += f"\n\n{issue_commit_hint(task.github_issue)}"
                # Round continuation loop: a testing round is a single
                # `tester.iter()` run, but it can make many model requests
                # (each `read_file`/`run_bash` result grows the context).
                # When the fill crosses the compaction threshold at a tool
                # boundary, the run is stopped, the ROUND transcript is
                # compacted in place (task.messages is never touched) and
                # the round resumes as a continuation turn from the
                # compacted transcript — otherwise a pathological tool
                # result could blow the model window before the tester
                # ever calls set_verification, looping the round in
                # fruitless retries.
                text: str | None = prompt
                # Live context reading of the current round; the shared
                # _live_ctx_tokens map belongs to the agent transcript and
                # must not be polluted with the tester's.
                live_tokens: int | None = None
                while True:
                    # Reset per attempt: if tester.iter() raises before the
                    # `as run` binding, the except handlers must not
                    # re-persist the PREVIOUS attempt's transcript over the
                    # (possibly compacted) round.
                    run = None
                    history: list[Any] | None = None
                    if text is None:
                        # Continuation of a paused round: resume from the
                        # stored round transcript (compacted, or — when the
                        # summarizer failed — the original one with its
                        # invalid tail trimmed).
                        history = resumable_history(task.tester_messages[-1])
                    paused = False
                    async with tester.iter(
                        text,
                        message_history=history,
                        deps=deps,
                        usage_limits=UsageLimits(
                            request_limit=self.config.request_limit
                        ),
                    ) as run:  # noqa: F811 - rebinding per attempt is intended
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                # Drain the stream (nothing to render — the
                                # answer comes from the end-of-round
                                # re-render) and take the round's freshest
                                # context reading once the request is final.
                                # Every event is a sign of life: without the
                                # touch a long (minutes) report generation
                                # reads as "stalled" — and the stall watchdog
                                # aborts a perfectly healthy run.
                                async with node.stream(run.ctx) as stream:
                                    async for _event in stream:
                                        self._touch_activity()
                                live_tokens = run_context_tokens(run)
                            elif Agent.is_call_tools_node(node):
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
                                # Mid-round compaction gate: a tool boundary
                                # is the only safe stopping point (same as
                                # the agent's mid-run gate). Skipped when a
                                # previous compaction attempt failed, or a
                                # persistently broken summarizer would loop
                                # pause→resume forever.
                                tokens = self._fill_over_threshold(
                                    live_tokens, task.tester_messages[-1]
                                )
                                if tokens is not None and (
                                    task.id not in self._compaction_failed
                                ):
                                    pct = round(
                                        tokens * 100 / self.config.max_context_tokens
                                    )
                                    self._show_event(
                                        task,
                                        f"Context fill {pct}% — pausing the "
                                        "tester round to compact history…",
                                    )
                                    self._persist_tester_messages(task, run)
                                    paused = True
                                    break
                            # Keep the stored history current while the test
                            # runs: switching to the task mid-test shows the
                            # tester's work done so far.
                            self._persist_tester_messages(task, run)
                    if paused:
                        if await self._maybe_compact_tester_round(task, live_tokens):
                            # The round transcript was replaced by the
                            # original prompt plus the summary: the live
                            # token reading belongs to the old transcript —
                            # drop it so the counter falls back to the
                            # compacted one (mirrors _maybe_compact).
                            live_tokens = None
                        # Either way (compacted, or the summarizer failed
                        # and the gate is now disabled for this task) the
                        # round resumes as a continuation turn: no new
                        # text, history from the round transcript.
                        text = None
                        continue
                    if run.result is not None and self.viewed_id == task_id:
                        # Re-render from the persisted transcript: the streamed
                        # tool-call blocks above are replaced by the full history
                        # view (separator + reasoning + tool calls + answer).
                        self.view_task(task_id)
                    break
            except asyncio.CancelledError:
                # Aborted (or superseded): keep whatever the tester did so
                # far, then re-raise so the worker teardown stays clean.
                cancelled = True
                self._persist_tester_messages(task, run)
                if self.viewed_id == task_id:
                    # Watchdog abort (or esc) of a hung run: re-render from
                    # the persisted transcript, so the open view shows what
                    # was actually done instead of stale blocks (e.g. a tool
                    # call that never returned) until the user switches views.
                    # Sync only — no awaits inside a cancelled coroutine.
                    self.view_task(task_id)
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
        try:
            imported, skipped = await sync_issues(
                self.manager, client, self._github_state
            )
        except GitHubError as e:
            self.notify(
                f"GitHub sync failed: {e}",
                title="GitHub integration",
                severity="warning",
                timeout=8,
            )
            return
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
