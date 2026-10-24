"""App header with the model-connection status in the top-left corner.

The stock Textual `Header` renders icon → title → clock. `ModelStatusHeader`
inserts a `HeaderModelStatus` widget right after the icon, so the indicator
sits between the ⭘ button and the centered title. The icon itself is hidden
via CSS: the app replaces the stock command palette with its own Settings
palette (Ctrl+S), which leaves the stock icon disabled — it would still show
a hover highlight while ignoring clicks.

Clicking the header toggles the stock "tall" mode (1 → 3 lines). In the
expanded state two extra lines appear (see `HeaderTallInfo`): the
model-connection details on the top line and the per-session task
statistics on the bottom line. Both are filled by the app's 1 s activity
poller (`DrCodeApp._render_header_tall`); the centered title keeps its own
absolute layer and is never shifted by the added content.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.events import Click
from textual.message import Message
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


class HeaderGitIndicator(Static):
    """Unmerged-completed-tasks counter, docked right, left of the clock.

    Shows `⎇ N` when N completed tasks still sit unmerged on their
    `drcode/<id>` branches (see DrCodeState._poll_git_indicator); hidden
    entirely (display:none) when every task's work is merged. The amber
    bold styling is deliberate — the indicator must stand out from the
    muted header furniture so an unmerged fix is hard to miss.
    """

    DEFAULT_CSS = """
    HeaderGitIndicator {
        dock: right;
        width: auto;
        padding: 0 1;
        color: $warning;
        text-style: bold;
        display: none;
    }
    HeaderGitIndicator.-unmerged {
        display: block;
    }
    """


class HeaderTallInfo(Static):
    """Base for the extra lines shown only in the expanded (tall) header.

    Hidden in the collapsed state so the single-line header keeps its
    stock layout; docked top/bottom by the subclasses. The widgets are
    composed after the left/right-docked icon and clock, so their 100%
    width already excludes those columns — the text can't slide under
    them and is ellipsized instead.
    """

    DEFAULT_CSS = """
    HeaderTallInfo {
        height: 1;
        width: 100%;
        padding: 0 1;
        color: $text-muted;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    """


class HeaderConnInfo(HeaderTallInfo):
    """Expanded-header top line: connection URL, model, probe latency,
    error counters (consecutive probe failures, network retries) and the
    exact cumulative session token usage."""

    DEFAULT_CSS = """
    HeaderConnInfo {
        dock: top;
    }
    """


class HeaderSessionInfo(HeaderTallInfo):
    """Expanded-header bottom line: per-session task statistics (tasks
    completed, average/median activation→completion time, bugs:features
    ratio of the tasks created this session)."""

    DEFAULT_CSS = """
    HeaderSessionInfo {
        dock: bottom;
    }
    """


class ModelStatusHeader(Header):
    """Header showing the model-connection status after the icon."""

    DEFAULT_CSS = """
    ModelStatusHeader {
        layers: title-layer _default;
    }
    /* The app disables the command palette (Settings on Ctrl+S instead), so
    the stock HeaderIcon is inert: Textual marks it disabled and drops its
    click events, but still paints the :hover highlight — a dead-looking
    button. Hide it (and free its docked 8-cell column); the status label
    takes its place at the left edge. Scoped to this header only. */
    ModelStatusHeader HeaderIcon {
        display: none;
    }
    /* The extra tall-only lines are hidden in the collapsed state; the
    display rules live here (not in HeaderTallInfo's own stylesheet) so
    the -tall override reliably wins the cascade. */
    ModelStatusHeader HeaderTallInfo {
        display: none;
    }
    ModelStatusHeader.-tall HeaderTallInfo {
        display: block;
    }
    /* Tall mode: the absolute title must span all three rows so its
    center-middle alignment keeps "DrCode" on the middle line (its own
    layer is untouched — the tall lines dock in the default layer). */
    ModelStatusHeader.-tall HeaderTitle {
        height: 100%;
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

    class TallToggled(Message):
        """The user clicked the header: tall mode is now `tall`."""

        def __init__(self, tall: bool) -> None:
            self.tall = tall
            super().__init__()

    def compose(self) -> ComposeResult:
        yield HeaderIcon().data_bind(Header.icon)
        yield HeaderModelStatus(id="model-status")
        yield (
            HeaderClock().data_bind(Header.time_format)
            if self._show_clock
            else HeaderClockSpace()
        )
        # Docked right AFTER the clock: right-docks stack away from the
        # edge in compose order, so the indicator lands just left of it.
        yield HeaderGitIndicator(id="header-git")
        # The tall-only lines dock around the icon/clock columns (they are
        # composed after those docks, see HeaderTallInfo).
        yield HeaderConnInfo(id="header-conn")
        yield HeaderSessionInfo(id="header-sessions")
        yield HeaderTitle()

    @on(Click)
    def _toggle_tall_on_click(self, event: Click) -> None:
        # A handler named `_on_click` here won't work: Textual dispatches
        # `_on_click` for EVERY class in the MRO (see
        # MessagePump._get_dispatch_methods), so the stock Header._on_click
        # would run after ours and toggle -tall a second time — the two
        # toggles cancel out and the click appears to do nothing. Use a
        # decorated handler instead and prevent the remaining handlers
        # (the stock toggle plus Widget's text selection), doing the stock
        # toggle here, plus a notification so the app can fill the
        # expanded lines immediately instead of waiting for the next 1 s
        # poll tick.
        event.prevent_default()
        self.toggle_class("-tall")
        self.post_message(self.TallToggled(self.has_class("-tall")))
