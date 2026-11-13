"""Project statistics: the stats.json store, the funnel-side accumulation
(fire_status_change dwell + completion counters, the startup demotion of
a stale ACTIVE task) and the day-bucket snapshot."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from drcode.models import TaskKind, TaskStatus
from drcode.stats import StatsStore, format_seconds
from drcode.task_manager import TaskManager

DAY = "2026-11-04"


@pytest.fixture
def manager(project_root: Path) -> TaskManager:
    return TaskManager(project_root)


@pytest.fixture
def fixed_day(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr("drcode.stats._utc_today", lambda: DAY)
    return DAY


def _past(seconds: float) -> datetime:
    return datetime.now(UTC) - timedelta(seconds=seconds)


def _bucket(path: Path, day: str) -> dict:
    return json.loads((path / ".drcode" / "stats.json").read_text())["days"][day]


# -- store: persistence, atomicity, corruption tolerance ----------------------


def test_store_persists_and_reloads(project_root: Path, fixed_day: str) -> None:
    store = StatsStore(project_root)
    store.record_created()
    store.record_completion(cancelled=False, failed=True)
    store.add_time("dev", 50.0, rounds=2)

    reloaded = StatsStore(project_root)
    day = dict(reloaded.snapshot(1))[fixed_day]
    assert day.created == 1
    assert day.failed == 1
    assert day.completed == 0
    assert day.dev_seconds == 50.0
    assert day.dev_rounds == 2


def test_store_writes_atomically_no_tmp_leftovers(
    project_root: Path, fixed_day: str
) -> None:
    store = StatsStore(project_root)
    store.record_created()
    store.record_completion(cancelled=True, failed=False)
    stats_dir = project_root / ".drcode"
    # The atomic write leaves exactly the payload — no .tmp sibling, valid
    # JSON with the version field.
    assert [p.name for p in stats_dir.iterdir() if p.suffix == ".tmp"] == []
    data = json.loads((stats_dir / "stats.json").read_text())
    assert data["version"] == 1
    assert data["days"][fixed_day]["created"] == 1
    assert data["days"][fixed_day]["cancelled"] == 1


def test_store_corrupted_json_skipped_and_backed_up(
    project_root: Path, fixed_day: str
) -> None:
    stats_path = project_root / ".drcode" / "stats.json"
    stats_path.parent.mkdir(parents=True)
    stats_path.write_text("{not json", encoding="utf-8")

    store = StatsStore(project_root)  # must not raise
    # The store starts empty: today's bucket is zeroed.
    assert store.snapshot(1)[0][0] == fixed_day
    assert store.snapshot(1)[0][1].created == 0
    # Corrupted file moved aside, never deleted.
    assert stats_path.with_name("stats.json.bak").read_text() == "{not json"
    # The store still works: the next mutation writes a fresh file.
    store.record_created()
    assert _bucket(project_root, fixed_day)["created"] == 1


def test_store_unexpected_shape_skipped(project_root: Path, fixed_day: str) -> None:
    stats_path = project_root / ".drcode" / "stats.json"
    stats_path.parent.mkdir(parents=True)
    stats_path.write_text(json.dumps({"version": 1, "days": ["nope"]}))
    store = StatsStore(project_root)
    store.record_created()
    assert _bucket(project_root, fixed_day)["created"] == 1


def test_store_unwritable_path_never_raises(project_root: Path) -> None:
    # stats.json as a DIRECTORY: os.replace onto it fails — the store
    # reports and keeps working in memory, the caller never sees it.
    (project_root / ".drcode" / "stats.json").mkdir(parents=True)
    store = StatsStore(project_root)
    store.record_created()  # must not raise
    store.add_time("resolver", 5.0, rounds=1)  # must not raise
    day = store.snapshot(1)[0][1]
    assert day.created == 1
    assert day.resolver_rounds == 1


def test_store_add_time_validates(project_root: Path) -> None:
    store = StatsStore(project_root)
    with pytest.raises(ValueError):
        store.add_time("bogus", 1.0)
    # A no-op addition does not create the file.
    store.add_time("dev", 0.0)
    assert not (project_root / ".drcode" / "stats.json").exists()


def test_subagent_spawn_counters_and_usage_window(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful spawn counts subagent_spawns, a failed one only
    subagent_failures; `subagent_usage` splits the successful totals into
    (ever, last-N-days) for the "enabled but never used" hint."""
    today = "2026-11-04"
    old = "2026-10-10"
    monkeypatch.setattr("drcode.stats._utc_today", lambda: today)
    store = StatsStore(project_root)

    store.record_subagent_spawn()
    store.record_subagent_spawn()
    store.record_subagent_spawn(failed=True)
    # A failure is a failure, never a spawn.
    day = store.snapshot(1)[0][1]
    assert day.subagent_spawns == 2
    assert day.subagent_failures == 1
    assert store.subagent_usage() == (2, 2)
    assert store.subagent_usage(recent_days=14) == (2, 2)

    # An older successful spawn lands outside the recent window but stays
    # in the "ever" total; the persisted file survives a reload.
    monkeypatch.setattr("drcode.stats._utc_today", lambda: old)
    StatsStore(project_root).record_subagent_spawn()
    monkeypatch.setattr("drcode.stats._utc_today", lambda: today)

    reloaded = StatsStore(project_root)
    assert reloaded.subagent_usage() == (3, 2)
    assert reloaded.subagent_usage(recent_days=1) == (3, 2)


def test_record_subagent_spawn_unwritable_path_never_raises(
    project_root: Path,
) -> None:
    (project_root / ".drcode" / "stats.json").mkdir(parents=True)
    store = StatsStore(project_root)
    store.record_subagent_spawn()  # must not raise
    store.record_subagent_spawn(failed=True)  # must not raise
    assert store.snapshot(1)[0][1].subagent_spawns == 1
    assert store.snapshot(1)[0][1].subagent_failures == 1


def test_snapshot_zero_fills_and_orders(project_root: Path, fixed_day: str) -> None:
    store = StatsStore(project_root)
    store.record_created()
    days = store.snapshot(14)
    assert len(days) == 14
    assert days[-1][0] == fixed_day
    assert days[-1][1].created == 1
    assert all(day.created == 0 for _, day in days[:-1])


# -- funnel accumulation ---------------------------------------------------------


def test_created_counted_on_create_and_bug_reports(
    manager: TaskManager, fixed_day: str
) -> None:
    manager.create("a", "input")
    manager.create("b", "input", kind=TaskKind.BUG)
    assert _bucket(manager.storage.root, fixed_day)["created"] == 2


def test_active_and_testing_dwell_accumulate(
    manager: TaskManager, fixed_day: str
) -> None:
    task = manager.create("t", "input")
    # Enter ACTIVE, then backdate the stamp: the stay is simulated to have
    # lasted 50 s when the task leaves ACTIVE.
    manager.promote_next()
    assert task.status is TaskStatus.ACTIVE
    task.status_changed_at = _past(50)
    manager.mark_done(task.id)
    assert task.status is TaskStatus.TESTING
    assert task.time_in_status["active"] == pytest.approx(50, abs=2)

    task.status_changed_at = _past(30)
    manager.set_verification(task.id, True)
    assert task.status is TaskStatus.COMPLETED
    assert task.time_in_status["testing"] == pytest.approx(30, abs=2)

    day = _bucket(manager.storage.root, fixed_day)
    assert day["dev_seconds"] == pytest.approx(50, abs=2)
    assert day["dev_rounds"] == 1
    assert day["testing_seconds"] == pytest.approx(30, abs=2)
    assert day["testing_rounds"] == 1
    assert day["completed"] == 1


def test_completion_counters_distinguish_cancel_and_failed(
    manager: TaskManager, fixed_day: str
) -> None:
    cancelled = manager.create("c", "input")
    manager.promote_next()
    cancelled.status_changed_at = _past(10)
    manager.cancel(cancelled.id)
    assert cancelled.status is TaskStatus.COMPLETED

    failed = manager.create("f", "input")
    manager.promote_next()
    manager.mark_done(failed.id)
    manager.set_verification(failed.id, False)
    assert failed.verification is not None and failed.verification.value == (
        "Verification failed"
    )

    passed = manager.create("p", "input")
    manager.promote_next()
    manager.mark_done(passed.id)
    manager.set_verification(passed.id, True)

    day = _bucket(manager.storage.root, fixed_day)
    assert day["cancelled"] == 1
    assert day["failed"] == 1
    assert day["completed"] == 1
    # Every finished ACTIVE stay is a dev round (all three tasks were
    # activated); the cancel's 10 s stay is part of the honest dev time.
    assert day["dev_rounds"] == 3
    assert day["dev_seconds"] >= 10


def test_requeued_completion_counts_again(manager: TaskManager, fixed_day: str) -> None:
    task = manager.create("t", "input")
    manager.promote_next()
    manager.mark_done(task.id)
    manager.set_verification(task.id, True)
    # Send input to the COMPLETED task: re-queued, completes again —
    # "how many completions that day", not "how many unique tasks".
    manager.send(task.id, "one more round")
    manager.promote_next()
    manager.mark_done(task.id)
    manager.set_verification(task.id, True)
    assert _bucket(manager.storage.root, fixed_day)["completed"] == 2


def test_time_in_status_survives_restart_via_reload(
    project_root: Path, fixed_day: str
) -> None:
    """The dwell accumulators persist with the task: a restart (a fresh
    TaskManager over the same root) finds the stale ACTIVE task, closes
    its interrupted stay into time_in_status + the dev stats and demotes
    it to QUEUED."""
    manager = TaskManager(project_root)
    task = manager.create("t", "input")
    manager.promote_next()
    task.status_changed_at = _past(120)
    manager.storage.save(task)

    reloaded = TaskManager(project_root)
    kept = reloaded.tasks[task.id]
    assert kept.status is TaskStatus.QUEUED  # stale ACTIVE demoted
    assert kept.time_in_status["active"] == pytest.approx(120, abs=5)

    day = _bucket(project_root, fixed_day)
    assert day["dev_rounds"] == 1
    assert day["dev_seconds"] == pytest.approx(120, abs=5)


def test_testing_stay_survives_restart_untouched(
    project_root: Path, fixed_day: str
) -> None:
    """A task stuck in TESTING is re-tested on startup — its TESTING stay
    keeps accruing (no demotion, no re-stamp), nothing is double-counted."""
    manager = TaskManager(project_root)
    task = manager.create("t", "input")
    manager.promote_next()
    manager.mark_done(task.id)
    stamp = task.status_changed_at
    assert stamp is not None

    reloaded = TaskManager(project_root)
    kept = reloaded.tasks[task.id]
    assert kept.status is TaskStatus.TESTING
    assert kept.status_changed_at == stamp
    day = _bucket(project_root, fixed_day)
    assert day["testing_rounds"] == 0  # the stay has not ended yet


# -- formatting --------------------------------------------------------------------


def test_format_seconds() -> None:
    assert format_seconds(0.4) == "0 с"
    assert format_seconds(45.9) == "45 с"
    assert format_seconds(80) == "1 мин 20 с"
    assert format_seconds(120) == "2 мин"
    assert format_seconds(3600) == "1 ч"
    assert format_seconds(4200) == "1 ч 10 мин"


# -- dialog smoke ---------------------------------------------------------------------


async def test_stats_dialog_renders_table_and_sparklines(
    project_root: Path, fixed_day: str
) -> None:
    """The dialog loads the store off the disk in its worker thread and
    renders one row per day + the sparkline data; "p" re-loads with the
    wider window."""
    from textual.app import App, ComposeResult

    from drcode.tui.stats_dialog import StatsDialog

    store = StatsStore(project_root)
    store.record_created()
    store.record_completion(cancelled=False, failed=False)
    store.add_time("dev", 90.0, rounds=1)

    class Host(App[None]):
        def compose(self) -> ComposeResult:
            yield from ()

    app = Host()
    async with app.run_test() as pilot:
        dialog = StatsDialog(project_root)
        await app.push_screen(dialog)
        await pilot.pause()
        await pilot.pause(0.3)  # let the load worker finish
        from textual.widgets import DataTable, Sparkline

        table = dialog.query_one("#stats-table", DataTable)
        assert table.row_count == dialog._period
        created = dialog.query_one("#stats-spark-created", Sparkline)
        completed = dialog.query_one("#stats-spark-completed", Sparkline)
        assert created.data is not None and created.data[-1] == 1.0
        assert completed.data is not None and completed.data[-1] == 1.0
        # "p" flips 14 → 30 days and re-loads.
        await pilot.press("p")
        await pilot.pause(0.3)
        assert dialog._period == 30
        assert table.row_count == 30
