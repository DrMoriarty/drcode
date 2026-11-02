"""User-level token-usage dialog ("Token usage…" in the Settings palette).

Shows the persistent per-day, per-model token statistics collected by
`drcode.usage_store` (`~/.config/drcode/usage.json`): one table row per
(day, model key) with the four per-response counters pydantic-ai reports
plus the request count, and a period-total row at the bottom. The keys
are "<role>/<model>" — model id in the catalog's
"<provider>/<model-id>" format where the provider is known — so the same
model serving different roles shows as separate rows.

The file is read in a worker thread (asyncio.to_thread): a large usage
history must never stall the UI. Read-only: the dialog never writes.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Static

from ..usage_store import FIELDS, UsageStore, usage_file


def _fmt(n: int) -> str:
    """Thousands-separated counter for the table cells."""
    return f"{n:,}"


class UsageDialog(ModalScreen[None]):
    """Modal table: days × models with input/output/cache breakdown."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Close", priority=True),
    ]

    DEFAULT_CSS = """
    UsageDialog {
        align: center middle;
    }
    #usage-dialog {
        width: 96;
        max-width: 95%;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #usage-title { color: $text; text-style: bold; margin-bottom: 1; }
    #usage-table { height: auto; max-height: 24; }
    #usage-status {
        height: auto;
        min-height: 1;
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(self, store: UsageStore | None = None) -> None:
        super().__init__()
        # Explicit resolved path by default (incident ebe6ba15
        # discipline): read-only today, but a bare UsageStore() would be
        # guard-locked forever under an inherited DRCODE_TASK_ID.
        self._store = store or UsageStore(path=usage_file())

    def compose(self) -> ComposeResult:
        with Vertical(id="usage-dialog"):
            yield Static("Token usage — per day and model", id="usage-title")
            yield DataTable(id="usage-table", cursor_type="row")
            yield Static("Loading…", id="usage-status")

    def on_mount(self) -> None:
        table = self.query_one("#usage-table", DataTable)
        table.add_columns(
            "Day",
            "Model",
            "Requests",
            "Input",
            "Output",
            "Cache read",
            "Cache write",
        )
        self.run_worker(self._load(), exclusive=True, group="usage-load")

    async def _load(self) -> None:
        # Blocking disk read off the event loop (the store may hold a
        # long history); the parse itself is part of the read.
        days = await asyncio.to_thread(self._store.load)
        status = self.query_one("#usage-status", Static)
        try:
            table = self.query_one("#usage-table", DataTable)
            self._fill_table(table, days)
            status.update(self._summary(days))
        except Exception:  # noqa: BLE001, S110 - the dialog may be mid-teardown
            # DOM pruning of a popped screen is gradual; dropping the
            # render is the same pattern as the search dialog's.
            pass

    @staticmethod
    def _fill_table(table: DataTable, days: dict) -> None:
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
            "Σ period",
            "",
            *(_fmt(totals[field]) for field in FIELDS),
        )

    @staticmethod
    def _summary(days: dict) -> str:
        """One-line status under the table: the period covered and the
        row count — or the empty-archive hint."""
        if not days:
            return "No usage recorded yet."
        span = f"{min(days)} … {max(days)}"
        models = {key for entry in days.values() for key in entry}
        return f"{len(days)} day(s), {len(models)} model key(s): {span}"

    def action_cancel(self) -> None:
        self.dismiss(None)
