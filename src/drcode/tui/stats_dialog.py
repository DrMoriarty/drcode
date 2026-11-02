"""Project statistics dialog (Ctrl+G) — day buckets of the task funnel.

A read-only modal over the main screen: a DataTable with the last 14/30
UTC days (created / completed / cancelled / failed + the average
dev / testing / resolver time per finished stay) and two sparklines over
the same window (created and completed per day).

The data never touches the event loop's hot path: the whole read (a fresh
:class:`drcode.stats.StatsStore` load of ``.drcode/stats.json`` + the
window snapshot) runs once per dialog session in a worker thread — the
same treatment as the Ctrl+F archive load. The store is mutated on the
pipeline side at event time; the dialog is a pure reader (a throwaway
store instance, no writes).

"p" toggles the window between 14 and 30 days; ESC closes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import DataTable, Sparkline, Static, Tab, Tabs

from ..stats import DayStats, StatsStore, format_seconds

# The two window sizes the "p" toggle flips between.
_PERIODS: tuple[int, ...] = (14, 30)


class StatsDialog(ModalScreen[None]):
    """Modal project statistics: day table + created/completed sparklines."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Close", priority=True),
        Binding("p", "toggle_period", "14/30 дней"),
    ]

    DEFAULT_CSS = """
    StatsDialog {
        align: center middle;
    }
    #stats-dialog {
        width: 100;
        max-width: 95%;
        height: 34;
        max-height: 90%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #stats-title { color: $text; text-style: bold; }
    #stats-tabs { margin-top: 1; }
    #stats-scroll { height: 1fr; margin-top: 1; }
    #stats-table { height: auto; max-height: 1fr; }
    #stats-sparklines {
        height: auto;
        margin-top: 1;
        display: none;
    }
    #stats-sparklines.shown { display: block; }
    .stats-spark-row { height: auto; }
    .stats-spark-row Static { color: $text-muted; }
    .stats-spark-row Sparkline { height: 3; width: 1fr; }
    #stats-hint {
        margin-top: 1;
        color: $text-disabled;
        text-align: right;
    }
    """

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        # The MAIN checkout's root — the stats live in the main checkout's
        # .drcode/ (the same storage root the manager's store uses).
        self._project_root = Path(project_root)
        self._period = _PERIODS[0]
        self._days: list[tuple[str, DayStats]] | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="stats-dialog"):
            yield Static("Статистика", id="stats-title")
            yield Tabs(Tab("Проект", id="project"), id="stats-tabs")
            with VerticalScroll(id="stats-scroll"):
                yield DataTable(id="stats-table")
                with Vertical(id="stats-sparklines"):
                    with Vertical(classes="stats-spark-row"):
                        yield Static("Создано")
                        yield Sparkline(id="stats-spark-created")
                    with Vertical(classes="stats-spark-row"):
                        yield Static("Выполнено")
                        yield Sparkline(id="stats-spark-completed")
            yield Static("p — окно 14/30 дней · esc — close", id="stats-hint")

    def on_mount(self) -> None:
        table = self.query_one("#stats-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        self._build_columns()
        self._load_days()

    # -- loading (worker thread) -------------------------------------------------

    @work(exclusive=True, group="stats-load")
    async def _load_days(self) -> None:
        """Read the store off the disk in a worker thread: constructing
        the throwaway StatsStore does the (blocking) file read; the
        snapshot needs no lock of its own — the store instance is local
        to this read."""

        def read() -> list[tuple[str, DayStats]]:
            return StatsStore(self._project_root).snapshot(self._period)

        self._days = await asyncio.to_thread(read)
        self._render_days()

    # -- rendering ---------------------------------------------------------------

    def _build_columns(self) -> None:
        table = self.query_one("#stats-table", DataTable)
        table.clear(columns=True)
        table.add_columns(
            "Дата",
            "Создано",
            "Выполнено",
            "Отменено",
            "Провалено",
            "Разработка",
            "Тесты",
            "Резолв",
        )

    @staticmethod
    def _average(day: DayStats, kind: str) -> str:
        """The on-the-fly average for one kind ("dev"/"testing"/"resolver"):
        seconds / finished stays, human-formatted — "—" when nothing
        finished that day."""
        seconds = getattr(day, f"{kind}_seconds")
        rounds = getattr(day, f"{kind}_rounds")
        if rounds <= 0:
            return "—"
        return format_seconds(seconds / rounds)

    def _render_days(self) -> None:
        if self._days is None:
            return
        try:
            table = self.query_one("#stats-table", DataTable)
            spark_box = self.query_one("#stats-sparklines", Vertical)
            created_spark = self.query_one("#stats-spark-created", Sparkline)
            completed_spark = self.query_one("#stats-spark-completed", Sparkline)
        except NoMatches:
            # The dialog was dismissed while the load worker was suspended
            # in to_thread — DOM is gone, drop the render.
            return
        table.clear()
        for day_key, day in reversed(self._days):  # newest day on top
            table.add_row(
                day_key,
                str(day.created),
                str(day.completed),
                str(day.cancelled),
                str(day.failed),
                self._average(day, "dev"),
                self._average(day, "testing"),
                self._average(day, "resolver"),
            )
        # Sparklines over the same window, oldest → newest (left → right).
        created_spark.data = [float(day.created) for _, day in self._days]
        completed_spark.data = [float(day.completed) for _, day in self._days]
        spark_box.add_class("shown")

    # -- actions -----------------------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_toggle_period(self) -> None:
        idx = _PERIODS.index(self._period)
        self._period = _PERIODS[(idx + 1) % len(_PERIODS)]
        self._build_columns()
        # A different window needs a fresh (wider) snapshot from disk.
        self._days = None
        self._load_days()
