from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any

from rich.cells import cell_len
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
from textual.timer import Timer
from textual.widgets import Static

# Braille spinner frames for the "work in progress" animation shown in a
# reasoning block while the model is thinking and in a tool-call block while
# the tool is executing.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_INTERVAL = 0.12

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
        # Markdown render of the body, cached and invalidated on content
        # change (see _render_body): a full markdown parse on every throttled
        # repaint would starve the event loop that drives the UI.
        self._body_key: tuple[Any, ...] = ()
        self._body_text: Text | None = None
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
        # A mouse-up that lands on the block after a drag-selection still
        # fires `Click`, but collapsing here re-renders the widget and
        # invalidates the in-progress selection on the parent scrollable,
        # so the selected text is dropped instead of copied. Honour the
        # selection instead of toggling when one is active.
        if self.screen.selections:
            return
        self.toggle_class("collapsed")
        self._redraw()

    def on_resize(self) -> None:
        # The stamp is pushed to the right edge by padding: re-pad on width
        # changes (terminal resize).
        self._redraw()

    def _redraw(self) -> None:
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
            body = self._render_body()
            if body is not None:
                text.append(body)
        self.update(text)

    def _render_body(self) -> Text | None:
        """Markdown render of the accumulated reasoning content.

        Live-highlight the already-received complete fragments while the model
        still streams; render the finished block as full markdown. The cache
        key (streaming flag, content, rendered width) re-renders only when the
        body actually changed — the animated spinner lives in the header, not
        in the body.
        """
        if not self._content:
            return Text("(empty)", "dim italic")
        width = self.content_size.width or self.size.width or 80
        if self._body_key == (self._streaming, self._content, width):
            return self._body_text
        renderer = _streamable_markdown if self._streaming else _markdown_text
        self._body_key = (self._streaming, self._content, width)
        self._body_text = renderer(self._content, width)
        return self._body_text


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


def _streamable_markdown(text: str, width: int) -> Text:
    """Render markdown on the longest complete prefix of *text*.

    Rich parses markdown as a whole document, so it can't be streamed token
    by token. Instead, format only the portion of *text* that ends at a
    newline and sits outside any *unclosed* fenced code block (```/~~~), and
    leave the partial remainder as raw text. Already-received complete
    fragments get live highlighting; the still-producing tail stays plain —
    an acceptable trade-off while the model is still streaming.
    """
    # A fenced code block is tracked via its opening/closing fences (runs of
    # three or more backticks or tildes): while the stack is non-empty we are
    # inside an open block, so a newline there is not a safe formatting
    # boundary. Inline code spans (1-2 backticks) are ignored here and left
    # to Rich (an unterminated one renders as literal text).
    fences: list[str] = []
    i, n = 0, len(text)
    last_safe = -1
    while i < n:
        c = text[i]
        if c in ("`", "~"):
            run = 1
            while i + run < n and text[i + run] == c:
                run += 1
            if run >= 3:
                if fences and fences[0] == c:
                    fences.pop()
                elif not fences:
                    fences.append(c)
                # A mismatched-char fence inside a block is literal code.
        elif c == "\n":
            if not fences:
                last_safe = i
        i += 1

    if last_safe == -1:
        return Text(text)

    result = _markdown_text(text[: last_safe + 1], width)
    result.append(text[last_safe + 1 :])
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
                # Format the already-received complete lines live; the
                # still-producing tail streams in as raw text.
                self._text = _streamable_markdown(self._content, width)
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

