"""Stage arithmetic of the conflict screen's progress panel.

The screen renders its long-running work (the OPENING prepare phase and
the ``v`` verification funnel) as one line per stage with a braille
spinner while a stage runs, ok/FAILED/skip marks when it is done, and a
percent/ETA footer estimated from the rolling per-stage durations
persisted across runs next to the verify report. This module holds the
stage dataclasses and the pure ETA/percent arithmetic over them —
deliberately widget-free and unit-testable without an app.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rich.cells import cell_len

from ...merge_resolver import CheckResult


def _fmt_seconds(seconds: float | None) -> str:
    """Compact human duration for the progress panel ("42s", "1m20s")."""
    if seconds is None:
        return "?"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _elide_middle(path: str, max_cells: int) -> str:
    """Middle-elide a path to at most ``max_cells`` cells: a head prefix
    and the file-name tail (the last ``/`` component, or its END when
    even that alone is too long — the extension survives) joined by
    "…". Paths that already fit pass through unchanged. A budget below
    4 cells also leaves the path untouched — a degenerate elision would
    show less than a plain crop."""
    if max_cells < 4 or cell_len(path) <= max_cells:
        return path
    ell = "…"
    keep = max_cells - cell_len(ell)
    if keep <= 0:
        return ell
    head, _sep, tail = path.rpartition("/")
    if tail and cell_len(tail) <= keep:
        # The name fits whole: fill the rest of the budget with head.
        head_budget = keep - cell_len(tail)
        while head and cell_len(head) > head_budget:
            head = head[:-1]
        return f"{head}{ell}{tail}" if head else f"{ell}{tail}"
    # The file name itself does not fit: keep its end.
    return f"{ell}{tail[-keep:]}" if tail else f"{ell}{path[-keep:]}"


@dataclass
class _CheckStage:
    """One stage of the interactive funnel's live progress: `key` names
    the persisted-timing slot, `fn_name` the `merge_resolver` function to
    resolve at CALL time (never captured — tests monkeypatch the module
    attribute), `status` is pending / running / done."""

    key: str
    fn_name: str
    label: str
    status: str = "pending"
    result: CheckResult | None = None
    started: float | None = None
    duration: float | None = None


@dataclass
class _PrepareStage:
    """One step of the screen's prepare phase (the engine's reversed
    merge + triage + safe auto-resolvers) rendered in the same progress
    panel as the funnel: `key` names the persisted-timing slot
    (`<task-id>-prepare-timings.json`), `status` is pending / running /
    done. The steps advance from `merge_resolver.prepare`'s progress
    callback, applied on the event loop."""

    key: str
    label: str
    status: str = "pending"
    started: float | None = None
    duration: float | None = None


def _stages_estimate(
    stage: _CheckStage | _PrepareStage,
    averages: dict[str, float],
    defaults: dict[str, float],
) -> float:
    """The historical duration of `stage`: the persisted rolling average
    when there is one, else the built-in default."""
    return averages.get(stage.key) or defaults.get(stage.key, 30.0)


def _stages_eta(
    stages: list[_CheckStage] | list[_PrepareStage],
    averages: dict[str, float],
    defaults: dict[str, float],
) -> float:
    """ETA over the not-yet-done stages: each remaining stage's
    historical duration, the running one credited with
    `historical − elapsed` (floored at 10% of it)."""
    now = time.monotonic()
    total = 0.0
    for stage in stages:
        if stage.status == "done":
            continue
        estimate = _stages_estimate(stage, averages, defaults)
        if stage.status == "running" and stage.started is not None:
            total += max(estimate - (now - stage.started), estimate * 0.1)
        else:
            total += estimate
    return total


def _stages_percent(
    stages: list[_CheckStage] | list[_PrepareStage],
    averages: dict[str, float],
    defaults: dict[str, float],
) -> int:
    """Completed / total stages, the running one credited proportionally
    to its historical duration (capped at 95% — it never claims to be
    done while it is still running)."""
    if not stages:
        return 0
    now = time.monotonic()
    done = 0.0
    for stage in stages:
        if stage.status == "done":
            done += 1
        elif stage.status == "running" and stage.started is not None:
            done += min(
                (now - stage.started) / _stages_estimate(stage, averages, defaults),
                0.95,
            )
    return int(done / len(stages) * 100)
