"""Combined statistics dialog (Ctrl+G, Settings → "Statistics…").

A single modal with two tabs over the main screen:

- **Проект** — day buckets of the task funnel: a DataTable with the last
  14/30 UTC days (created / completed / cancelled / failed + the average
  dev / testing / resolver time per finished stay) and two sparklines
  over the same window (created and completed per day).
- **Токены** — the persistent per-day, per-model token statistics
  collected by `drcode.usage_store` (`~/.config/drcode/usage.json`): one
  table row per (day, model key) with the four per-response counters
  pydantic-ai reports plus the request count, and a period-total row at
  the bottom. The keys are "<role>/<model>" — model id in the catalog's
  "<provider>/<model-id>" format where the provider is known — so the
  same model serving different roles shows as separate rows.

No data ever touches the event loop's hot path: both reads (the
:class:`drcode.stats.StatsStore` window snapshot and the usage file
parse) run in worker threads. The project snapshot loads on mount; the
usage tab loads lazily on its first activation (an empty-history read is
cheap, but there is no reason to pay it for a user who never leaves the
project tab). The stores are mutated on the pipeline side at event time;
the dialog is a pure reader (throwaway store instances, no writes).

"p" toggles the project window between 14 and 30 days (project tab
only); ESC closes.
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
from ..usage_store import FIELDS, UsageStore, usage_file

# The two window sizes the "p" toggle flips between.
_PERIODS: tuple[int, ...] = (14, 30)

# Tab ids (Tabs widget ids) and their hint lines.
_PROJECT_TAB = "project"
_USAGE_TAB = "usage"

_HINTS: dict[str, str] = {
    _PROJECT_TAB: "p — окно 14/30 дней · esc — close",
    _USAGE_TAB: "esc — close",
}


def _fmt(n: int) -> str:
    """Thousands-separated counter for the usage table cells."""
    return f"{n:,}"


class StatsDialog(ModalScreen[None]):
    """Modal statistics: project day table + sparklines, and the
    per-day/per-model token usage — one tab each."""

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
    #stats-project-pane { height: 1fr; }
    #stats-project-pane.hidden { display: none; }
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
    #stats-usage-pane {
        height: 1fr;
        display: none;
    }
    #stats-usage-pane.shown { display: block; }
    #usage-table { height: auto; max-height: 1fr; }
    #usage-status {
        height: auto;
        min-height: 1;
        margin-top: 1;
        color: $text-muted;
    }
    #stats-hint {
        margin-top: 1;
        color: $text-disabled;
        text-align: right;
    }
    """

    def __init__(
        self, project_root: Path, usage_store: UsageStore | None = None
    ) -> None:
        super().__init__()
        # The MAIN checkout's root — the stats live in the main checkout's
        # .drcode/ (the same storage root the manager's store uses).
        self._project_root = Path(project_root)
        # Explicit resolved path by default (incident ebe6ba15
        # discipline): read-only, but a bare UsageStore() would be
        # guard-locked forever under an inherited DRCODE_TASK_ID.
        self._usage_store = usage_store or UsageStore(path=usage_file())
        self._period = _PERIODS[0]
        self._days: list[tuple[str, DayStats]] | None = None
        self._usage_loaded = False

    def compose(self) -> ComposeResult:
        with Vertical(id="stats-dialog"):
            yield Static("Статистика", id="stats-title")
            yield Tabs(
                Tab("Проект", id=_PROJECT_TAB),
                Tab("Токены", id=_USAGE_TAB),
                id="stats-tabs",
            )
            with Vertical(id="stats-project-pane"), VerticalScroll(id="stats-scroll"):
                yield DataTable(id="stats-table")
                with Vertical(id="stats-sparklines"):
                    with Vertical(classes="stats-spark-row"):
                        yield Static("Создано")
                        yield Sparkline(id="stats-spark-created")
                    with Vertical(classes="stats-spark-row"):
                        yield Static("Выполнено")
                        yield Sparkline(id="stats-spark-completed")
            with Vertical(id="stats-usage-pane"):
                yield DataTable(id="usage-table", cursor_type="row")
                yield Static("Loading…", id="usage-status")
            yield Static(_HINTS[_PROJECT_TAB], id="stats-hint")

    def on_mount(self) -> None:
        table = self.query_one("#stats-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        self._build_columns()
        usage_table = self.query_one("#usage-table", DataTable)
        usage_table.add_columns(
            "День",
            "Модель",
            "Запросы",
            "Input",
            "Output",
            "Cache read",
            "Cache write",
        )
        self._load_days()

    # -- tab switching ------------------------------------------------------------

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Show the activated tab's pane, hide the other one, swap the
        hint line; the usage tab loads its store lazily on the FIRST
        activation (later switches just re-show the filled table)."""
        tab_id = str(event.tab.id) if event.tab.id else _PROJECT_TAB
        self._show_pane(tab_id)
        if tab_id == _USAGE_TAB and not self._usage_loaded:
            self._usage_loaded = True
            self._load_usage()

    def _show_pane(self, tab_id: str) -> None:
        project_pane = self.query_one("#stats-project-pane", Vertical)
        usage_pane = self.query_one("#stats-usage-pane", Vertical)
        project_pane.set_class(tab_id != _PROJECT_TAB, "hidden")
        usage_pane.set_class(tab_id == _USAGE_TAB, "shown")
        self.query_one("#stats-hint", Static).update(_HINTS.get(tab_id, ""))

    # -- project tab: loading (worker thread) --------------------------------------

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

    # -- project tab: rendering ------------------------------------------------------

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

    # -- usage tab: loading (worker thread) ------------------------------------------

    @work(exclusive=True, group="usage-load")
    async def _load_usage(self) -> None:
        """Read the usage file in a worker thread (the store may hold a
        long history; the parse itself is part of the read)."""

        def read() -> dict:
            return self._usage_store.load()

        try:
            days = await asyncio.to_thread(read)
            status = self.query_one("#usage-status", Static)
            table = self.query_one("#usage-table", DataTable)
            self._fill_usage_table(table, days)
            status.update(self._usage_summary(days))
        except Exception:  # noqa: BLE001, S110 - the dialog may be mid-teardown
            # DOM pruning of a popped screen is gradual; dropping the
            # render is the same pattern as the search dialog's.
            pass

    @staticmethod
    def _fill_usage_table(table: DataTable, days: dict) -> None:
        """One row per (day, model), newest day first, then a period
        total. Clearing first makes a future reload idempotent."""
        table.clear()
        totals = dict.fromkeys(FIELDS, 0)
        rows = 0
        for day in sorted(days, reverse=True):
            for key in sorted(days[day]):
                counters = days[day][key]
                for field in FIELDS:
                    totals[field] += counters.get(field, 0)
                rows += 1
                table.add_row(
                    day,
                    key,
                    *(_fmt(counters.get(field, 0)) for field in FIELDS),
                )
        if rows == 0:
            return
        table.add_row(
            "Σ период",
            "",
            *(_fmt(totals[field]) for field in FIELDS),
        )

    @staticmethod
    def _usage_summary(days: dict) -> str:
        """One-line status under the table: the period covered and the
        row count — or the empty-archive hint."""
        if not days:
            return "Записей об использовании пока нет."
        span = f"{min(days)} … {max(days)}"
        models = {key for entry in days.values() for key in entry}
        return f"{len(days)} дн., {len(models)} ключей моделей: {span}"

    # -- actions -----------------------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_toggle_period(self) -> None:
        tabs = self.query_one("#stats-tabs", Tabs)
        if tabs.active != _PROJECT_TAB:
            # "p" is a project-tab concern; on the usage tab it is a no-op.
            return
        idx = _PERIODS.index(self._period)
        self._period = _PERIODS[(idx + 1) % len(_PERIODS)]
        self._build_columns()
        # A different window needs a fresh (wider) snapshot from disk.
        self._days = None
        self._load_days()
