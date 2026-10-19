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

    DEFAULT_CSS = """
    ModelStatusHeader {
        layers: title-layer _default;
    }
    ModelStatusHeader HeaderTitle {
        /* The docked icon/status/clock shrink the flow layout, so an
        in-flow, width:100% title would drift as the status text (and the
        tok/s counter) changes width. Pin the title to its own layer
        spanning the full header width to keep it centered. */
        position: absolute;
        layer: title-layer;
        width: 100%;
        offset: 0 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield HeaderIcon().data_bind(Header.icon)
        yield HeaderModelStatus(id="model-status")
        yield HeaderTitle()
        yield (
            HeaderClock().data_bind(Header.time_format)
            if self._show_clock
            else HeaderClockSpace()
        )
