"""Shared widgets for the command-palette providers."""

from __future__ import annotations

from typing import ClassVar, cast

from textual.command import CommandList
from textual.css.styles import RulesMap
from textual.geometry import Size
from textual.strip import Strip


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
