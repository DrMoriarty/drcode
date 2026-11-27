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
  the bottom. The days are not a column — each day opens with a styled
  date header row its model rows group under (rows painted per agent
  role). The keys are "<role>/<model>" — model id in the catalog's
  "<provider>/<model-id>" format where the provider is known — so the
  same model serving different roles shows as separate rows. When the
  model's catalog entry carries per-token prices (auto-discovered from
  the gateway's pricing block or entered by hand in models.json), a
  final "Стоимость" column shows the row's USD cost; rows of unpriced
  models show an empty cell. Below it, a second table shows the
  per-day average cost of one session of each agent (role), computed
  from the FROZEN costs recorded at request time (`drcode.cost` —
  price edits never re-price history, unlike the recomputed column
  above); days without priced sessions of a role are not shown.

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

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import DataTable, Sparkline, Static, Tab, Tabs

from ..cost import Prices, fmt_cost, price_index, row_cost
from ..stats import DayStats, StatsStore, format_seconds
from ..usage_store import (
    FIELDS,
    ROLLING_WINDOW,
    UsageStore,
    daily_hit_rate,
    hit_rate,
    rolling_hit_rate,
    usage_file,
)

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


# The counter fields the usage table renders, in display order — the
# canonical FIELDS list of the store minus cache_write (a write-amplification
# detail; the store keeps accounting it, the table just doesn't show it).
_USAGE_FIELDS: tuple[str, ...] = tuple(f for f in FIELDS if f != "cache_write")

# Header row of the usage table: the model key first, then one column per
# rendered field, then hit-rate and cost. The day is NOT a column — each
# day's models are preceded by a full-width date header row instead.
_USAGE_COLUMNS: tuple[str, ...] = (
    "Модель",
    "Запросы",
    "Input",
    "Output",
    "Cache read",
    "Hit-rate",
    "Стоимость",
)

# Rich style of the per-day header rows (and the Σ period row): a muted
# group separator the model rows are visually grouped under.
_DAY_STYLE = "bold dim"

# Row color per agent role (the "<role>/" prefix of the usage key): the
# whole model row is painted so rows of different roles stay distinguishable
# even when the Модель column has scrolled out of view. Unknown roles fall
# back to the base style; "coding" IS the base style.
_ROLE_STYLES: dict[str, str] = {
    "coding": "",
    "tester": "yellow",
    "resolver": "magenta",
    "compaction": "blue",
    "subagent": "dim",
    "research": "cyan",
}


# Per-model prices of one usage key: (prompt, completion, cache_read) per
# token in USD; cache_read may be None (unknown) while the other two are
# always present — the index only holds entries whose cost CAN be computed
# (with the caveat that a nonzero cache_read still needs price_cache_read).
# The pricing rules themselves (the index build, the cost formula, the
# formatting) live in `drcode.cost` — shared with the per-request frozen
# recording in `_record_session_usage`; the historical module-level names
# below are kept as thin aliases for the existing import sites/tests.
_PriceTriple = Prices

# The blocking catalog read happens in the usage-load worker thread,
# never on the event loop (the request-time caller stat-caches it — see
# `drcode.cost`).
_price_index = price_index
_row_cost = row_cost
_fmt_cost = fmt_cost


def _fmt_rate(rate: float | None) -> str:
    """Hit-rate cell: a percent figure, or an empty cell when the row has
    no input tokens (an undefined ratio must not read as "0%")."""
    if rate is None:
        return ""
    return f"{round(rate * 100)}%"


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
    #stats-table { height: 1fr; }
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
    #cost-heading {
        margin-top: 1;
        color: $text-muted;
    }
    #cost-table { height: auto; max-height: 1fr; }
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
            with Vertical(id="stats-project-pane"):
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
                yield Static(
                    "Средняя стоимость сессии по агентам (цены на момент запроса)",
                    id="cost-heading",
                )
                yield DataTable(id="cost-table")
            yield Static(_HINTS[_PROJECT_TAB], id="stats-hint")

    def on_mount(self) -> None:
        table = self.query_one("#stats-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        self._build_columns()
        usage_table = self.query_one("#usage-table", DataTable)
        usage_table.add_columns(*_USAGE_COLUMNS)
        cost_table = self.query_one("#cost-table", DataTable)
        cost_table.add_columns(
            "День",
            "Агент",
            "Сессий",
            "Средняя",
            "Всего",
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
        """Read the usage file (and the price index) in a worker thread
        (the store may hold a long history; the parse itself is part of
        the read, and the catalog read belongs off the event loop too)."""

        def read() -> tuple[dict, dict[str, _PriceTriple], list[dict], dict[str, dict]]:
            return (
                self._usage_store.load(),
                _price_index(),
                self._usage_store.load_recent(ROLLING_WINDOW),
                self._usage_store.load_costs(),
            )

        try:
            days, prices, recent, costs = await asyncio.to_thread(read)
            status = self.query_one("#usage-status", Static)
            table = self.query_one("#usage-table", DataTable)
            cost_table = self.query_one("#cost-table", DataTable)
            self._fill_usage_table(table, days, prices)
            self._fill_cost_table(cost_table, costs)
            status.update(self._usage_summary(days, recent))
        except Exception:  # noqa: BLE001, S110 - the dialog may be mid-teardown
            # DOM pruning of a popped screen is gradual; dropping the
            # render is the same pattern as the search dialog's.
            pass

    @staticmethod
    def _fill_usage_table(
        table: DataTable, days: dict, prices: dict[str, _PriceTriple] | None = None
    ) -> None:
        """One row per (day, model), newest day first, then a period
        total. Clearing first makes a future reload idempotent.

        The day is not a column: each day's model rows are preceded by a
        full-width date header row (the date in the first cell, styled
        `_DAY_STYLE`, every other cell empty) — the model rows of the day
        follow immediately, without a day cell of their own. The period
        total ("Σ период") stays the last row, its format unchanged.

        Every model-row cell is painted in the row's role style
        (`_ROLE_STYLES`) so rows of different roles stay distinguishable
        with the Модель column scrolled out of view; the date header rows
        keep the neutral group style.

        The cost column shows the USD amount of the row per the model's
        CURRENT catalog prices (the historical behavior of this table —
        deliberately untouched; the FROZEN-at-record-time figures live
        in the agent table below and in `drcode.cost`'s contract); a
        model whose prices are unknown (or missing the cache-read price
        while the row has cache reads) shows an empty cell — never 0.
        The period total sums the computable rows only. The cache_write
        counter is accounted in the totals but never rendered.
        The hit-rate column is the row's daily cache_read/input share —
        the day window of the metric (the rolling last-N-requests window
        lives in the status line under the table)."""
        table.clear()
        totals = dict.fromkeys(FIELDS, 0)
        total_cost = 0.0
        rows = 0
        for day in sorted(days, reverse=True):
            if not days[day]:
                continue
            table.add_row(
                Text(day, style=_DAY_STYLE),
                *([""] * (len(_USAGE_COLUMNS) - 1)),
            )
            for key in sorted(days[day]):
                counters = days[day][key]
                for field in FIELDS:
                    totals[field] += counters.get(field, 0)
                rows += 1
                # The usage key is "<role>/<model>"; the price index is
                # keyed by the model part ("<provider>/<model-id>" or the
                # bare id) — strip the role prefix (first slash only: the
                # model id itself may contain slashes).
                role, _, _model = key.partition("/")
                row_style = _ROLE_STYLES.get(role, "")
                row_prices = (prices or {}).get(_model)
                cost = _row_cost(counters, row_prices) if row_prices else None
                if cost is not None:
                    total_cost += cost
                table.add_row(
                    *(
                        Text(cell, style=row_style)
                        for cell in (
                            key,
                            *(_fmt(counters.get(field, 0)) for field in _USAGE_FIELDS),
                            _fmt_rate(daily_hit_rate(counters)),
                            _fmt_cost(cost) if cost is not None else "",
                        )
                    )
                )
        if rows == 0:
            return
        table.add_row(
            Text("Σ период", style=_DAY_STYLE),
            *(_fmt(totals[field]) for field in _USAGE_FIELDS),
            _fmt_rate(hit_rate(totals["cache_read"], totals["input"])),
            _fmt_cost(total_cost) if total_cost else "",
        )

    @staticmethod
    def _fill_cost_table(table: DataTable, costs: dict[str, dict[str, dict]]) -> None:
        """The frozen-cost averages: one row per (day, agent role) that
        HAS priced sessions — the sum of the sessions' frozen costs over
        their count. Roles (the "<role>/" prefix of the usage key) are
        aggregated ACROSS their models: "the average cost of one session
        of an agent" is a per-agent figure, not a per-model one. Days
        (or roles) without priced sessions simply do not appear — a
        free model's sessions are absent from the ledger by design, not
        zero-cost rows. Clearing first makes a reload idempotent."""
        table.clear()
        aggregated: dict[tuple[str, str], list[float]] = {}
        for day, entry in costs.items():
            for key, cell in entry.items():
                role = key.partition("/")[0]
                acc = aggregated.setdefault((day, role), [0.0, 0.0])
                acc[0] += float(cell.get("sessions", 0))
                acc[1] += float(cell.get("cost", 0.0))
        for day, role in sorted(aggregated, reverse=True):  # newest day on top
            sessions, total = aggregated[(day, role)]
            if sessions <= 0:
                continue
            table.add_row(
                day,
                role,
                f"{int(sessions):,}",
                _fmt_cost(total / sessions),
                _fmt_cost(total),
            )

    @staticmethod
    def _usage_summary(days: dict, recent: list[dict] | None = None) -> str:
        """One-line status under the table: the period covered, the row
        count and the ROLLING hit-rate per key over the last
        `ROLLING_WINDOW` requests — or the empty-archive hint. Keys are
        ordered by their most recent request (the freshest first), and a
        key whose rolling window holds no input tokens shows nothing."""
        if not days:
            return "Записей об использовании пока нет."
        span = f"{min(days)} … {max(days)}"
        models = {key for entry in days.values() for key in entry}
        summary = f"{len(days)} дн., {len(models)} ключей моделей: {span}"
        if not recent:
            return summary
        rolling_keys: list[str] = []
        for record in reversed(recent):
            key = record.get("key")
            if isinstance(key, str) and key not in rolling_keys:
                rolling_keys.append(key)
        rates = []
        for key in rolling_keys[:8]:
            rate = rolling_hit_rate(recent, key=key, n=ROLLING_WINDOW)
            if rate is not None:
                rates.append(f"{key} {round(rate * 100)}%")
        if rates:
            summary += f" · hit-rate (последние {ROLLING_WINDOW} запр.): " + ", ".join(
                rates
            )
        return summary

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
