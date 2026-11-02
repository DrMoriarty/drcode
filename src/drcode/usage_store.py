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

File format (version 1)::

    {
      "version": 1,
      "days": {
        "2026-11-04": {
          "coding/openai/gpt-5": {"requests": 40, "input": 120000,
                                   "output": 8000, "cache_read": 90000,
                                   "cache_write": 3000}
        }
      }
    }

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
from typing import Any

from pydantic_ai.usage import RunUsage

from . import config as _config
from .models import _now
from .providers import AgentRole

USAGE_FILENAME = "usage.json"

# The version this module reads and writes; data carrying any other
# version is treated as unreadable (a newer format would be misread
# here — starting fresh beats corrupting it).
VERSION = 1

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
        simply starts a fresh file.
        """
        try:
            data: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return self._validated(data)

    @staticmethod
    def _validated(data: Any) -> dict[str, DayEntry]:
        """Shape-check the parsed json: dict of date-strings → dict of
        model keys → counter dicts (ints only). Anything else (or a
        wrong `version`) is rejected wholesale."""
        if not isinstance(data, dict) or data.get("version") != VERSION:
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

    # -- writing -------------------------------------------------------------

    def record(self, role: AgentRole, model_id: str, usage: RunUsage) -> None:
        """Fold one model call's exact usage into today's (UTC) entry of
        the calling role's model key and atomically rewrite the file.

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
            days = self.load()
            entry = days.setdefault(day, {}).setdefault(key, _empty_entry())
            entry["requests"] += usage.requests or 1
            entry["input"] += usage.input_tokens or 0
            entry["output"] += usage.output_tokens or 0
            entry["cache_read"] += usage.cache_read_tokens or 0
            entry["cache_write"] += usage.cache_write_tokens or 0
            self._write({"version": VERSION, "days": days})
        except ValueError:
            raise
        except (OSError, TypeError):
            # Disk full, read-only config dir, a concurrent replace —
            # usage bookkeeping is best-effort by design.
            pass

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
