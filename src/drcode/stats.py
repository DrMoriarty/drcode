"""Project-level task statistics, persisted in ``.drcode/stats.json``.

A tiny JSON store in the spirit of ``search_index.json`` /
``pending_notifications.json``: read once at startup, every mutation
rewrites the whole file atomically (tempfile + ``os.replace`` in the same
directory), a corrupted file is moved aside (``.bak``) and the store starts
fresh — never fatal.

What is counted and when:

- ``created`` — a task entered the system (``TaskManager.create``, which
  covers the GitHub Issues import too: it goes through the same method).
- ``completed`` / ``cancelled`` / ``failed`` — every transition INTO
  COMPLETED in ``TaskManager.fire_status_change``, distinguished by
  ``task.cancelled`` (a user cancel skips the tester and carries no
  verdict) and ``task.verification == FAILED``. A re-queued COMPLETED task
  that completes again counts again ("how many completions happened that
  day", not "how many unique tasks").
- ``dev_seconds`` / ``dev_rounds`` — cumulative time the task spent in
  ACTIVE across all its development rounds; each finished ACTIVE stay adds
  its dwell and one round. The dwell is computed in ``fire_status_change``
  from ``task.status_changed_at`` and accumulated on the task itself
  (``task.time_in_status``) — both fields persist, so the counters survive
  restarts (a stale ACTIVE task is closed out on the next startup by the
  manager's demotion loop).
- ``testing_seconds`` / ``testing_rounds`` — same, for TESTING stays.
- ``resolver_seconds`` / ``resolver_rounds`` — conflict-resolution is NOT
  expressible as a task status (the resolver runs while the task is
  parked), so it is measured directly around the resolver round in
  ``core/pipeline_resolver.py`` and accumulated on ``task.resolver_seconds``.
- ``subagent_spawns`` / ``subagent_failures`` — observability of the
  read-only subagents (``delegate_research``): a SUCCESSFUL nested spawn
  counts one ``subagent_spawns`` (recorded next to the "subagent finished"
  service event in ``subagents.py``), a failed one counts only
  ``subagent_failures`` (next to "subagent failed"). Separated from the
  user preferences (``settings.json``) on purpose — this is usage telemetry,
  not a preference. The task-settings hints read the totals to report
  "subagents enabled but never used".

The aggregates are folded into the day buckets AT THE MOMENT of the event
(a transition, a completion, a round end), never recomputed from the task
archive — so the retention auto-cleanup of old completed tasks cannot
disturb them. A chunk of time that crosses midnight is attributed entirely
to the UTC day of the chunk's END (a deliberate v1 simplification: the
dwell is only known when the status is left). Averages are computed on the
fly (seconds / rounds), never stored.

All mutations take a lock (the status listeners fire from worker threads
too) and never raise out of the caller: a stats problem must never break
the state machine.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ValidationError

# Name of the stats file inside `<project>/.drcode/`.
STATS_FILENAME = "stats.json"

# Schema version of the file (bumped on a breaking change; older files are
# then to be migrated or discarded on load).
VERSION = 1

# The time-average kinds (field prefix in DayStats). The UI iterates these.
TIME_KINDS: tuple[str, ...] = ("dev", "testing", "resolver")


def _utc_today() -> str:
    """The UTC date key of "now" (module-level for test monkeypatching)."""
    return datetime.now(UTC).date().isoformat()


class DayStats(BaseModel):
    """One UTC day's aggregates. Seconds are floats (monotonic-clock
    style dwell); rounds count the finished stays of the corresponding
    kind, so the on-the-fly average is ``*_seconds / *_rounds``."""

    created: int = 0
    completed: int = 0
    cancelled: int = 0
    failed: int = 0
    dev_seconds: float = 0.0
    dev_rounds: int = 0
    testing_seconds: float = 0.0
    testing_rounds: int = 0
    resolver_seconds: float = 0.0
    resolver_rounds: int = 0
    subagent_spawns: int = 0
    subagent_failures: int = 0


class StatsStore:
    """Read-modify-write owner of ``<project>/.drcode/stats.json``.

    One instance per TaskManager (``manager.stats``); the TUI's statistics
    dialog builds a THROWAWAY instance in a worker thread to read the file
    fresh off the disk without touching the event loop.
    """

    def __init__(self, project_root: Path) -> None:
        self.path = Path(project_root) / ".drcode" / STATS_FILENAME
        self._lock = threading.Lock()
        self._days: dict[str, DayStats] = self._load()

    # -- loading / saving ------------------------------------------------------

    def _load(self) -> dict[str, DayStats]:
        """Read the day buckets from disk. Missing file = empty store; a
        corrupted/unexpected file is moved to ``.bak`` (the notifications
        store's pattern) and the store starts fresh — never fatal."""
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as e:
            self._report(f"stats file unreadable: {e}")
            return {}
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise TypeError("expected a JSON object")
            days = data.get("days")
            if days is None:
                days = {}
            if not isinstance(days, dict):
                raise TypeError("'days' must be an object")
            return {key: DayStats.model_validate(val) for key, val in days.items()}
        except (json.JSONDecodeError, TypeError, ValidationError) as e:
            backup = self.path.with_name(self.path.name + ".bak")
            try:
                self.path.replace(backup)
            except OSError:
                pass
            self._report(f"stats file corrupted, moved to {backup.name}: {e}")
            return {}

    def _save_locked(self) -> None:
        """Write the whole store atomically. Caller holds ``_lock``.
        Failures are reported, never raised — a read-only or vanished
        ``.drcode/`` must not break the state machine mid-transition."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": VERSION,
                "days": {
                    day: stats.model_dump() for day, stats in sorted(self._days.items())
                },
            }
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except OSError as e:
            self._report(f"stats not saved: {e}")

    @staticmethod
    def _report(text: str) -> None:
        print(f"drcode stats: {text}", file=sys.stderr)

    # -- mutations (one per event kind) ------------------------------------------

    def _bucket(self) -> DayStats:
        """Today's bucket (UTC), created on demand. Caller holds ``_lock``."""
        return self._days.setdefault(_utc_today(), DayStats())

    def record_created(self) -> None:
        """A task entered the system (user creation, agent bug report,
        GitHub import)."""
        with self._lock:
            self._bucket().created += 1
            self._save_locked()

    def record_completion(self, *, cancelled: bool, failed: bool) -> None:
        """A transition INTO COMPLETED. A cancel outranks a failed verdict
        (a cancelled task may still carry an old FAILED label from an
        earlier round — it did not finish as one)."""
        with self._lock:
            day = self._bucket()
            if cancelled:
                day.cancelled += 1
            elif failed:
                day.failed += 1
            else:
                day.completed += 1
            self._save_locked()

    def add_time(self, kind: str, seconds: float, *, rounds: int = 0) -> None:
        """Add a finished stay of `kind` ("dev" / "testing" / "resolver"):
        `seconds` of dwell and `rounds` finished stays (0 for a partial
        addition, e.g. a resolver invocation that scheduled a retry and
        will continue the same logical round). Attributed to TODAY (the
        day the stay ENDED — a midnight-crossing chunk is not split)."""
        if kind not in TIME_KINDS:
            raise ValueError(f"unknown stats time kind: {kind}")
        if seconds <= 0 and rounds <= 0:
            return
        with self._lock:
            day = self._bucket()
            if seconds > 0:
                setattr(
                    day, f"{kind}_seconds", getattr(day, f"{kind}_seconds") + seconds
                )
            if rounds:
                setattr(day, f"{kind}_rounds", getattr(day, f"{kind}_rounds") + rounds)
            self._save_locked()

    def record_subagent_spawn(self, *, failed: bool = False) -> None:
        """A delegate_research delegation finished: a SUCCESSFUL spawn
        counts one ``subagent_spawns``, a failed one counts only
        ``subagent_failures`` (a cancelled run — the parent aborted —
        counts neither). Attributed to TODAY (UTC)."""
        with self._lock:
            day = self._bucket()
            if failed:
                day.subagent_failures += 1
            else:
                day.subagent_spawns += 1
            self._save_locked()

    # -- reads ---------------------------------------------------------------------

    def subagent_usage(self, recent_days: int = 14) -> tuple[int, int]:
        """Successful delegate_research spawns as ``(total, recent)`` —
        the whole store and the last `recent_days` UTC days (today
        included). The "enabled but never used" hint in the task-settings
        dialog reads exactly this pair; failures are deliberately NOT
        folded in (a failed delegation still proves the parent tried)."""
        recent_days = max(recent_days, 1)
        today = date.fromisoformat(_utc_today())
        oldest = today - timedelta(days=recent_days - 1)
        with self._lock:
            total = 0
            recent = 0
            for key, day in self._days.items():
                total += day.subagent_spawns
                try:
                    day_date = date.fromisoformat(key)
                except ValueError:
                    continue
                if day_date >= oldest:
                    recent += day.subagent_spawns
            return total, recent

    def snapshot(self, days: int = 14) -> list[tuple[str, DayStats]]:
        """The last `days` UTC days, oldest first, zero-filled — the
        direct shape the statistics dialog renders (rows and sparkline
        series never need gap-filling logic of their own)."""
        with self._lock:
            today = date.fromisoformat(_utc_today())
            oldest = today - timedelta(days=max(days, 1) - 1)
            return [
                (
                    (oldest + timedelta(days=offset)).isoformat(),
                    self._days.get(
                        (oldest + timedelta(days=offset)).isoformat(), DayStats()
                    ),
                )
                for offset in range(max(days, 1))
            ]


def format_seconds(seconds: float) -> str:
    """Human one-liner for a dwell/average: "45 с", "3 мин 20 с",
    "1 ч 10 мин". Sub-second values collapse to "0 с" (the stats are not
    that precise anyway)."""
    total = int(seconds)
    if total < 60:
        return f"{total} с"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} мин {sec} с" if sec else f"{minutes} мин"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes} мин" if minutes else f"{hours} ч"
