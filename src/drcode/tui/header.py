"""App header with the model-connection status in the top-left corner.

The stock Textual `Header` renders icon → title → clock. `ModelStatusHeader`
inserts a `HeaderModelStatus` widget right after the icon, so the indicator
sits between the ⭘ button and the centered title.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static
from textual.widgets._header import (
    Header,
    HeaderClock,
    HeaderClockSpace,
    HeaderIcon,
    HeaderTitle,
)


class HeaderModelStatus(Static):
    """Connection indicator docked left, directly after the header icon."""

    DEFAULT_CSS = """
    HeaderModelStatus {
        dock: left;
        width: auto;
        padding: 0 1;
        text-opacity: 85%;
    }
    """


class ModelStatusHeader(Header):
    """Header showing the model-connection status after the icon."""

    def compose(self) -> ComposeResult:
        yield HeaderIcon().data_bind(Header.icon)
        yield HeaderModelStatus(id="model-status")
        yield HeaderTitle()
        yield (
            HeaderClock().data_bind(Header.time_format)
            if self._show_clock
            else HeaderClockSpace()
        )
