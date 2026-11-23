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
  same model serving different roles shows as separate rows. When the
  model's catalog entry carries per-token prices (auto-discovered from
  the gateway's pricing block or entered by hand in models.json), a
  final "Стоимость" column shows the row's USD cost; rows of unpriced
  models show an empty cell.

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

from ..providers import UserModelsStore, user_models_file
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


# Per-model prices of one usage key: (prompt, completion, cache_read) per
# token in USD; cache_read may be None (unknown) while the other two are
# always present — the index only holds entries whose cost CAN be computed
# (with the caveat that a nonzero cache_read still needs price_cache_read).
_PriceTriple = tuple[float, float, float | None]


def _price_index() -> dict[str, _PriceTriple]:
    """Model reference → per-token prices, from the user-level catalog.

    The usage keys are "<role>/<model>" where <model> is the catalog's
    "<provider>/<model-id>" reference (or the bare model id for custom /
    env models), so the lookup mirrors `providers._match_ref`: the full
    "<provider>/<model-id>" form first, then the bare model id across the
    catalog. Only entries with BOTH the prompt and the completion price
    are indexed — without either the cost is never computable; the
    cache-read price stays optional (a row with zero cache reads doesn't
    need it). The blocking catalog read happens in the usage-load worker
    thread, never on the event loop.
    """
    catalog = UserModelsStore(path=user_models_file()).load()
    index: dict[str, _PriceTriple] = {}
    for provider in catalog.providers:
        for entry in provider.models:
            if entry.price_prompt is None or entry.price_completion is None:
                continue
            prices: _PriceTriple = (
                entry.price_prompt,
                entry.price_completion,
                entry.price_cache_read,
            )
            index.setdefault(f"{provider.name}/{entry.id}", prices)
    for provider in catalog.providers:
        for entry in provider.models:
            if entry.price_prompt is None or entry.price_completion is None:
                continue
            index.setdefault(
                entry.id,
                (entry.price_prompt, entry.price_completion, entry.price_cache_read),
            )
    return index


def _row_cost(counters: dict[str, int], prices: _PriceTriple) -> float | None:
    """The USD cost of one (day, model) counter row.

    `input` already INCLUDES cache reads/writes (pydantic-ai normalizes
    usage that way — see `transcript._usage_context_tokens`), so the
    cache-read tokens are priced separately and subtracted from the
    prompt-priced remainder; cache writes stay inside the prompt term
    (no separate per-token price exists for them). None = not computable:
    a needed price is missing (the cache-read price only once the row
    actually has cache reads)."""
    price_prompt, price_completion, price_cache_read = prices
    cache_read = counters.get("cache_read", 0)
    if cache_read and price_cache_read is None:
        return None
    plain_input = max(0, counters.get("input", 0) - cache_read)
    return (
        plain_input * price_prompt
        + cache_read * (price_cache_read or 0.0)
        + counters.get("output", 0) * price_completion
    )


def _fmt_cost(cost: float) -> str:
    """Human-readable USD amount: costs are usually cents or less, so
    the precision scales down with the value; trailing zeros are trimmed
    ("$0.023", "$0.000735")."""
    if cost >= 1:
        text = f"{cost:,.2f}"
    elif cost >= 0.01:
        text = f"{cost:.4f}"
    else:
        text = f"{cost:.6f}"
    text = text.rstrip("0").rstrip(".")
    return f"${text or '0'}"


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
            "Hit-rate",
            "Стоимость",
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

        def read() -> tuple[dict, dict[str, _PriceTriple], list[dict]]:
            return (
                self._usage_store.load(),
                _price_index(),
                self._usage_store.load_recent(ROLLING_WINDOW),
            )

        try:
            days, prices, recent = await asyncio.to_thread(read)
            status = self.query_one("#usage-status", Static)
            table = self.query_one("#usage-table", DataTable)
            self._fill_usage_table(table, days, prices)
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

        The cost column shows the USD amount of the row per the model's
        catalog prices; a model whose prices are unknown (or missing the
        cache-read price while the row has cache reads) shows an empty
        cell — never 0. The period total sums the computable rows only.
        The hit-rate column is the row's daily cache_read/input share —
        the day window of the metric (the rolling last-N-requests window
        lives in the status line under the table)."""
        table.clear()
        totals = dict.fromkeys(FIELDS, 0)
        total_cost = 0.0
        rows = 0
        for day in sorted(days, reverse=True):
            for key in sorted(days[day]):
                counters = days[day][key]
                for field in FIELDS:
                    totals[field] += counters.get(field, 0)
                rows += 1
                # The usage key is "<role>/<model>"; the price index is
                # keyed by the model part ("<provider>/<model-id>" or the
                # bare id) — strip the role prefix (first slash only: the
                # model id itself may contain slashes).
                row_prices = (prices or {}).get(key.partition("/")[2])
                cost = _row_cost(counters, row_prices) if row_prices else None
                if cost is not None:
                    total_cost += cost
                table.add_row(
                    day,
                    key,
                    *(_fmt(counters.get(field, 0)) for field in FIELDS),
                    _fmt_rate(daily_hit_rate(counters)),
                    _fmt_cost(cost) if cost is not None else "",
                )
        if rows == 0:
            return
        table.add_row(
            "Σ период",
            "",
            *(_fmt(totals[field]) for field in FIELDS),
            _fmt_rate(hit_rate(totals["cache_read"], totals["input"])),
            _fmt_cost(total_cost) if total_cost else "",
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
