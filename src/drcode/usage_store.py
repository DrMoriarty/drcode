"""User-level token-usage statistics, persisted in
`~/.config/drcode/usage.json` (next to the model catalog, XDG-aware and
honoring `DRCODE_USER_CONFIG_DIR` — see `config._resolve_user_config_dir`).

Every model call funnels through the pipeline's single
`_record_session_usage` hook (TUI), which folds the exact per-request
usage into BOTH the session totals (header line) and this persistent
per-day, per-model store: ``days[date][<role>/<model>] → counters``.
Writing happens per event (one atomic replace per record), so a crash
loses at most the one in-flight model call, never the whole day.

Deliberately decoupled from pydantic-ai beyond the `RunUsage` shape:
the store folds plain counters, the TUI layer extracts them.

File format (version 2)::

    {
      "version": 2,
      "days": {
        "2026-11-04": {
          "coding/openai/gpt-5": {"requests": 40, "input": 120000,
                                   "output": 8000, "cache_read": 90000,
                                   "cache_write": 3000}
        }
      },
      "costs": {
        "2026-11-04": {
          "coding/openai/gpt-5": {"cost": 0.023, "sessions": 12}
        }
      },
      "recent": [
        {"at": "2026-11-04T12:03:44", "key": "coding/openai/gpt-5",
         "input": 3100, "cache_read": 2800, "cache_write": 0, "output": 210,
         "cost": 0.0007, "task": "e5449310"}
      ]
    }

Version 2 added money on top of version 1's token accounting:

- ``costs`` — per (day, model key) aggregates of the FROZEN USD cost
  (see `drcode.cost`): the sum of the request costs computed at
  response time from the catalog prices AS THEY WERE THEN, and the
  number of priced sessions (requests) behind that sum. Requests of
  unpriced models (custom / env / locally free endpoints) carry no
  cost and enter neither the sum nor the session count — a session
  without a price is NOT a zero-cost session (`drcode.cost` for the
  full contract).
- ``recent`` records gained the optional per-request ``cost`` and the
  ``task`` id the request was served for (audit trail; the durable
  task attribution lives in the Task metadata itself).

Migration: version-1 files stay fully readable — their ``days`` load
as-is, the money fields are simply absent (an empty ``costs``), and the
next record rewrites the file in the version-2 shape. Any OTHER version
(older or newer) is still rejected wholesale: a newer format would be
misread here — starting fresh beats corrupting it.

The optional ``recent`` ring holds the last `RECENT_LIMIT` per-request
records (oldest first) — the rolling cache hit-rate over the last N
requests is computed from it (`rolling_hit_rate`, `load_recent`); a file
without it (or with a malformed ring) simply has no rolling history.

The key is ``<role>/<model>`` where ``<model>`` is the resolved model in
the catalog's "<provider>/<model-id>" reference format (the bare model
id for custom models and the DRCODE_* env override, which carry no
provider name) — calls of different roles therefore always land under
different keys. The day is the UTC date of `_now()`.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage


from . import config as _config
from .models import _now
from .providers import AgentRole

USAGE_FILENAME = "usage.json"

# The version this module reads and writes; data carrying any other
# version is treated as unreadable (a newer format would be misread
# here — starting fresh beats corrupting it). Version 1 files are the
# one accepted exception: the pre-cost format loads as-is (its money
# fields are simply absent) and is rewritten in the v2 shape on the
# next record — see the module docstring's migration note.
VERSION = 2

# Counter fields of one day/model entry, in display order.
FIELDS: tuple[str, ...] = (
    "requests",
    "input",
    "output",
    "cache_read",
    "cache_write",
)

# Type of one day's model→counters map.
DayEntry = dict[str, dict[str, int]]

# Type of one day's model→money map: the frozen USD cost of the day's
# priced sessions of the key and their count. Unpriced sessions never
# enter it — "no price" is not "zero price" (see `drcode.cost`).
CostCell = dict[str, float]  # {"cost": float, "sessions": int}
CostEntry = dict[str, dict[str, CostCell]]

# How many most-recent per-request records the store keeps alongside the
# day aggregates (the "recent" ring in the file). The aggregates cannot
# answer "how did the hit-rate move over the LAST N requests" — that needs
# per-request granularity, which only this ring has.
RECENT_LIMIT = 500

# Default window (requests) of the rolling hit-rate shown in the stats
# dialog: recent enough to expose a fresh prefix invalidation, wide enough
# to smooth the per-request noise (the first request of a run never hits).
ROLLING_WINDOW = 50


def usage_file() -> Path:
    """Path of the user-level usage statistics (XDG_CONFIG_HOME-aware).

    Resolved dynamically through the config module (never imported by
    value): the conftest sandbox and the per-task run_bash redirect both
    repoint `config._USER_CONFIG_DIR`, and a binding captured at import
    time would pin the store to whichever env was live first — the exact
    incident-ebe6ba15 trap this module exists to avoid.
    """
    return _config._USER_CONFIG_DIR / USAGE_FILENAME


def _empty_entry() -> dict[str, int]:
    return dict.fromkeys(FIELDS, 0)


def hit_rate(cache_read: int, input_tokens: int) -> float | None:
    """The cache hit-rate of an aggregate: ``cache_read / input`` (0..1).

    pydantic-ai's ``input_tokens`` already INCLUDES cache reads and writes
    (see `transcript._usage_context_tokens`), so this ratio never exceeds 1
    and directly reads as "the share of billed prompt tokens served from
    cache". None when there is no input to speak of — an undefined ratio,
    not a zero: a day with no requests must not display as "0% hits".
    """
    if input_tokens <= 0:
        return None
    return cache_read / input_tokens


def daily_hit_rate(counters: dict[str, int]) -> float | None:
    """Hit-rate of one (day, model) counter row of the days mapping."""
    return hit_rate(counters.get("cache_read", 0), counters.get("input", 0))


def rolling_hit_rate(
    records: list[dict[str, Any]],
    key: str | None = None,
    n: int | None = None,
) -> float | None:
    """Hit-rate over the last `n` per-request records (optionally filtered
    to one ``<role>/<model>`` key), newest-last order as `load_recent`
    returns them. The ratio is summed reads over summed input — volume-
    weighted, so one giant cached request outweighs many small misses —
    and None when the window holds no input tokens at all."""
    if key is not None:
        records = [r for r in records if r.get("key") == key]
    if n is not None:
        records = records[-n:]
    input_total = sum(r.get("input", 0) for r in records)
    read_total = sum(r.get("cache_read", 0) for r in records)
    return hit_rate(read_total, input_total)


def _clean_record(raw: Any) -> dict[str, Any] | None:
    """Shape-check one per-request record (tolerant, like `_validated`):
    a record without a string key is junk and dropped, a bad counter
    becomes 0 — one malformed line never invalidates the history. The
    v2 money fields are optional: `cost` passes through only as a real
    number (a null/absent/insane one degrades to None — the record was
    unpriced), `task` only as a string."""
    if not isinstance(raw, dict) or not isinstance(raw.get("key"), str):
        return None
    at = raw.get("at")
    record: dict[str, Any] = {
        "at": at if isinstance(at, str) else "",
        "key": raw["key"],
    }
    for field in ("input", "cache_read", "cache_write", "output"):
        value = raw.get(field, 0)
        record[field] = (
            value if isinstance(value, int) and not isinstance(value, bool) else 0
        )
    cost = raw.get("cost")
    record["cost"] = (
        cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None
    )
    task = raw.get("task")
    record["task"] = task if isinstance(task, str) else ""
    return record


def _clean_records(data: Any) -> list[dict[str, Any]]:
    """The parsed file's per-request ring, shape-checked. Read failures
    degrade to an empty history — the ring is a metric, losing it in a
    corrupt file is never worth a failed usage record."""
    raw = data.get("recent") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    records: list[dict[str, Any]] = []
    for item in raw:
        record = _clean_record(item)
        if record is not None:
            records.append(record)
    return records


class UsageStore:
    """Loads/records the per-day, per-model token usage.

    Mirrors `UserModelsStore`'s guard rails: the constructor accepts an
    explicit `path=` (incident ebe6ba15 — a default-constructed store
    inside a drcode agent's verification snippet must never resolve to
    the real user-level file), and `record` refuses to write the real
    file when `DRCODE_TASK_ID` is set without an explicit path.
    """

    def __init__(self, path: Path | None = None) -> None:
        # Whether the path was passed explicitly matters for the
        # agent-guard in `record` (incident ebe6ba15) — see
        # UserModelsStore.__init__ for the full story.
        self._explicit_path = path is not None
        self.path = path or usage_file()

    # -- reading -------------------------------------------------------------

    def load(self) -> dict[str, DayEntry]:
        """The persisted days mapping (UTC date → model key → counters).

        A missing, corrupted or foreign-version file is skipped, never
        fatal: statistics are a sidecar, an unreadable file must not
        crash the pipeline (or the usage dialog) — the next record
        simply starts a fresh file. Version-1 files load exactly like
        version-2 ones (see the migration note in the module docstring).
        """
        return self._read_all()[0]

    def load_costs(self) -> CostEntry:
        """The persisted per-day frozen-cost aggregates (UTC date →
        model key → {"cost": USD sum, "sessions": priced count}).

        Same tolerance rules as `load`; a version-1 file (or any file
        without a well-shaped ``costs`` section) yields an empty map —
        old days genuinely have no recorded cost, which must not read
        as zero."""
        return self._read_all()[1]

    @staticmethod
    def _accepted_version(data: Any) -> bool:
        """The version gate: 1 (pre-cost, money fields absent) and the
        current VERSION are readable; anything else is foreign."""
        return isinstance(data, dict) and data.get("version") in (1, VERSION)

    @staticmethod
    def _validated(data: Any) -> dict[str, DayEntry]:
        """Shape-check the parsed json: dict of date-strings → dict of
        model keys → counter dicts (ints only). Anything else (or a
        wrong `version`) is rejected wholesale."""
        if not UsageStore._accepted_version(data):
            return {}
        days = data.get("days")
        if not isinstance(days, dict):
            return {}
        out: dict[str, DayEntry] = {}
        for day, entry in days.items():
            if not isinstance(day, str) or not isinstance(entry, dict):
                continue
            clean: DayEntry = {}
            for key, counters in entry.items():
                if not isinstance(key, str) or not isinstance(counters, dict):
                    continue
                row = _empty_entry()
                for field in FIELDS:
                    value = counters.get(field, 0)
                    if isinstance(value, int) and not isinstance(value, bool):
                        row[field] = value
                clean[key] = row
            out[day] = clean
        return out

    @staticmethod
    def _validated_costs(data: Any) -> CostEntry:
        """Shape-check the ``costs`` section the same tolerant way:
        day → key → {"cost": positive-sum float, "sessions": positive
        int}. Junk cells are dropped, never fatal — a malformed money
        line must not take the token history down with it."""
        if not isinstance(data, dict) or not UsageStore._accepted_version(data):
            return {}
        raw = data.get("costs")
        if not isinstance(raw, dict):
            return {}
        out: CostEntry = {}
        for day, entry in raw.items():
            if not isinstance(day, str) or not isinstance(entry, dict):
                continue
            clean: dict[str, CostCell] = {}
            for key, cell in entry.items():
                if not isinstance(key, str) or not isinstance(cell, dict):
                    continue
                cost = cell.get("cost")
                sessions = cell.get("sessions")
                if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                    continue
                if (
                    isinstance(sessions, bool)
                    or not isinstance(sessions, int)
                    or sessions <= 0
                ):
                    continue
                clean[key] = {"cost": float(cost), "sessions": sessions}
            out[day] = clean
        return out

    def _read_all(self) -> tuple[dict[str, DayEntry], CostEntry, list[dict[str, Any]]]:
        """One parse of the file → (days, costs, recent ring). All three
        readers share it so a record never pays three reads, and a
        read failure degrades every section to empty together — the
        whole file is unreadable or none of it is."""
        try:
            data: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}, {}, []
        return self._validated(data), self._validated_costs(data), _clean_records(data)

    # -- writing -------------------------------------------------------------

    def record(
        self,
        role: AgentRole,
        model_id: str,
        usage: RunUsage,
        cost: float | None = None,
        task_id: str | None = None,
    ) -> None:
        """Fold one model call's exact usage into today's (UTC) entry of
        the calling role's model key and atomically rewrite the file.

        `cost` is the request's FROZEN USD cost (see `drcode.cost`):
        computed by the caller at response time from the then-current
        catalog prices, or None when the model has no computable price.
        None contributes NOTHING to the ``costs`` section — neither to
        the sum nor to the priced-session count (a session without a
        price is not a zero-cost session) — while the token counters
        fold exactly as before. `task_id` rides the recent ring's
        record for the audit trail.

        The role is part of the key, so calls made by different roles are
        always accounted separately, even when they resolve to the same
        model. Write failures (disk full, read-only dir, a concurrent
        replace) never raise — statistics must not break a model run; the
        next call retries with the full accumulated state re-read from
        disk. The ONE deliberate exception is the agent-guard ValueError
        below: a default-constructed store inside an agent's run_bash
        child must fail loudly so the agent can fix its snippet.

        The counters accept None the same way `SessionTokenUsage.add`
        does: providers report what they report.
        """
        if os.environ.get("DRCODE_TASK_ID") and not self._explicit_path:
            # Defense in depth (incident ebe6ba15): a default-constructed
            # store inside a run_bash child resolves to the REAL
            # ~/.config/drcode/usage.json. The error reaches the agent as
            # the snippet's output and the agent fixes itself by passing
            # an explicit path (a tmp directory).
            raise ValueError(
                "UsageStore: running under a drcode agent (DRCODE_TASK_ID "
                "is set); writing to the real user-level usage file is "
                "forbidden — pass an explicit path (a tmp directory) to "
                "UsageStore()."
            )
        day = _now().date().isoformat()
        key = f"{role.value}/{model_id}" if model_id else role.value
        try:
            days, costs, recent = self._read_all()
            entry = days.setdefault(day, {}).setdefault(key, _empty_entry())
            entry["requests"] += usage.requests or 1
            entry["input"] += usage.input_tokens or 0
            entry["output"] += usage.output_tokens or 0
            entry["cache_read"] += usage.cache_read_tokens or 0
            entry["cache_write"] += usage.cache_write_tokens or 0
            # The frozen money aggregate: only priced sessions move it.
            if cost is not None:
                cell = costs.setdefault(day, {}).setdefault(
                    key, {"cost": 0.0, "sessions": 0}
                )
                cell["cost"] = cell.get("cost", 0.0) + cost
                cell["sessions"] = int(cell.get("sessions", 0)) + 1
            # Per-request granularity for the rolling hit-rate (the day
            # aggregates alone cannot answer "the last N requests"). The
            # ring is trimmed to RECENT_LIMIT — the file must stay small.
            recent.append(
                {
                    "at": _now().isoformat(timespec="seconds"),
                    "key": key,
                    "input": usage.input_tokens or 0,
                    "cache_read": usage.cache_read_tokens or 0,
                    "cache_write": usage.cache_write_tokens or 0,
                    "output": usage.output_tokens or 0,
                    "cost": cost,
                    "task": task_id or "",
                }
            )
            self._write(
                {
                    "version": VERSION,
                    "days": days,
                    "costs": costs,
                    "recent": recent[-RECENT_LIMIT:],
                }
            )
        except ValueError:
            raise
        except (OSError, TypeError):
            # Disk full, read-only config dir, a concurrent replace —
            # usage bookkeeping is best-effort by design.
            pass

    # -- per-request history (rolling hit-rate window) -----------------------

    def load_recent(self, limit: int | None = None) -> list[dict[str, Any]]:
        """The last `limit` per-request records, oldest → newest (an empty
        list when the file has no history yet). Read failures are never
        fatal — same discipline as `load`."""
        records = self._read_all()[2]
        return records if limit is None else records[-limit:]

    def _write(self, data: dict[str, Any]) -> None:
        """Atomic persist (tempfile + os.replace, mirroring the other
        stores): a crash mid-write never truncates the previous file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
