"""Line-API scroll views of the conflict screen.

Both widgets render no-wrap text through their own ``render_line``
(conflict files and check output are code — a soft-wrapped line would
shift every following hunk row and break the cursor/scroll mapping;
horizontal overflow scrolls instead), with the CHOPS last-line
workaround shared with ``task_settings_dialog.DialogScroll``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from rich.segment import Segment
from rich.style import Style
from textual.binding import Binding
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.strip import Strip

from ...conflict import Hunk

# The gutter in front of every content line (cursor marker + space).
_GUTTER = "  "

# Section styles of the zdiff3 view: ours green, the common ancestor dim
# italic, theirs blue, the marker lines dim.
_SECTION_STYLES = {
    "head": "dim",
    "tail": "dim",
    "ours": "$success",
    "base": "italic dim",
    "theirs": "$accent",
}


class ConflictView(ScrollView, can_focus=True):
    """The conflict-file viewer: no-wrap line rendering with a hunk
    cursor.

    A line-API widget (own ``render_line``) instead of a Static child:
    conflict files are code — lines must never soft-wrap (a wrapped line
    would shift every following hunk row and break the cursor/scroll
    mapping), horizontal overflow scrolls instead."""

    BINDINGS: ClassVar = [
        Binding("up,k", "cursor_up", "Previous hunk", show=False),
        Binding("down,j", "cursor_down", "Next hunk", show=False),
    ]

    DEFAULT_CSS = """
    ConflictView {
        background: $surface;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lines: list[tuple[str, str | None]] = []
        self._hunk_lines: dict[int, int] = {}  # content line -> hunk index
        self._hunks: list[Hunk] = []
        self.cursor: int = -1

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if round(old_value) != round(new_value):
            # CHOPS misses the viewport's last line in its dirty regions
            # during wheel-event scrolling — a full refresh repaints it.
            self.refresh()

    # -- content -----------------------------------------------------------------

    def set_file(self, text: str, hunks: list[Hunk]) -> None:
        """Render a file's content with the conflict sections colored and
        reset the cursor to the first hunk."""
        self._hunks = hunks
        self._hunk_lines = {}
        for idx, hunk in enumerate(hunks):
            for line_no in range(hunk.start, hunk.end):
                self._hunk_lines[line_no] = idx
        self._lines = self._styled_lines(text.split("\n"), hunks)
        self.cursor = 0 if hunks else -1
        self.virtual_size = Size(
            max((len(line) for line in self._lines), default=1) + 2,
            len(self._lines),
        )
        self.scroll_to(x=0, y=0, animate=False)
        self.refresh()

    @staticmethod
    def _styled_lines(
        lines: list[str], hunks: list[Hunk]
    ) -> list[tuple[str, str | None]]:
        """Content lines as rich Texts, colored per zdiff3 section."""
        section_of: dict[int, str] = {}
        for hunk in hunks:
            for line_no in range(hunk.start, hunk.end):
                if line_no == hunk.start:
                    section_of[line_no] = "head"
                elif line_no == hunk.end - 1:
                    section_of[line_no] = "tail"
                elif line_no <= hunk.start + len(hunk.ours):
                    section_of[line_no] = "ours"
                elif hunk.base and line_no <= (
                    hunk.start + len(hunk.ours) + 1 + len(hunk.base)
                ):
                    section_of[line_no] = "base"
                else:
                    section_of[line_no] = "theirs"
        result: list[tuple[str, str | None]] = []
        for line_no, line in enumerate(lines):
            section = section_of.get(line_no)
            result.append((line, _SECTION_STYLES.get(section) if section else None))
        return result

    # -- cursor ------------------------------------------------------------------

    def _set_cursor(self, index: int) -> None:
        if not self._hunks:
            return
        old = self.cursor
        self.cursor = max(0, min(index, len(self._hunks) - 1))
        if old != self.cursor:
            for line_no in range(self._hunks[old].start, self._hunks[old].end):
                self.refresh_line(line_no)
        hunk = self._hunks[self.cursor]
        for line_no in range(hunk.start, hunk.end):
            self.refresh_line(line_no)
        self.scroll_to(y=max(0, hunk.start - 2), animate=False)

    def action_cursor_up(self) -> None:
        self._set_cursor(self.cursor - 1)

    def action_cursor_down(self) -> None:
        self._set_cursor(self.cursor + 1)

    def current_hunk(self) -> Hunk | None:
        if 0 <= self.cursor < len(self._hunks):
            return self._hunks[self.cursor]
        return None

    # -- rendering -----------------------------------------------------------------

    def _rich_style(self, spec: str | None) -> Style | None:
        """A rich Style from a spec that may carry Textual CSS variables
        (`$success`) — rich cannot parse those, resolve them against the
        app's active theme first."""
        if not spec:
            return None
        variables = self.app.get_css_variables()
        resolved = " ".join(
            variables.get(word[1:], word) if word.startswith("$") else word
            for word in spec.split()
        )
        return Style.parse(resolved)

    def render_line(self, y: int) -> Strip:
        line_no = self.scroll_offset.y + y
        if 0 <= line_no < len(self._lines):
            text, style_spec = self._lines[line_no]
            hunk_idx = self._hunk_lines.get(line_no)
            gutter = _GUTTER
            style = self._rich_style(style_spec)
            if hunk_idx is not None and hunk_idx == self.cursor:
                gutter = "❯ "
                style = (style or Style()) + Style(bold=True)
            padded = (gutter + text).ljust(
                max(self.virtual_size.width, self.size.width) - 2
            )
            return Strip([Segment(padded, style)])
        return Strip.blank(self.size.width)


class CheckOutputView(ScrollView, can_focus=True):
    """The raw output of one verification check: no-wrap line rendering
    with virtual sizing computed from the longest line — pyright's lines
    are wide, horizontal overflow scrolls instead of soft-wrapping (the
    same line-API approach as `ConflictView`)."""

    DEFAULT_CSS = """
    CheckOutputView {
        background: $surface;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lines: list[str] = []

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if round(old_value) != round(new_value):
            # Same CHOPS last-line workaround as ConflictView above.
            self.refresh()

    def set_output(self, text: str) -> None:
        """Replace the shown output and scroll back to the top-left."""
        self._lines = text.split("\n") if text else ["(no output)"]
        self.virtual_size = Size(
            max((len(line) for line in self._lines), default=1) + 2,
            len(self._lines),
        )
        self.scroll_to(x=0, y=0, animate=False)
        self.refresh()

    def render_line(self, y: int) -> Strip:
        line_no = self.scroll_offset.y + y
        if 0 <= line_no < len(self._lines):
            padded = ("  " + self._lines[line_no]).ljust(
                max(self.virtual_size.width, self.size.width) - 2
            )
            return Strip([Segment(padded, None)])
        return Strip.blank(self.size.width)
