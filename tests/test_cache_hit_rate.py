"""Cache hit-rate metric and runtime alert: the hit-rate helpers and the
per-request ring of `drcode.usage_store`, the consecutive-miss monitor of
`drcode.core.usage` and its pipeline wiring (`_cache_hit_alert`).

Cached prompt tokens cost a fraction of uncached ones and a prefix-cache
invalidation is otherwise invisible until the bill arrives (incident
#a0a55d3d): the tests pin the volume-weighted ratio definition
(cache_read / input, pydantic-ai's input already includes cache tokens),
the once-per-episode alert discipline and the small-request floor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.usage import RunUsage

from drcode.core.usage import (
    CACHE_ALERT_MIN_INPUT,
    CACHE_ALERT_STREAK,
    CacheHitRateMonitor,
)
from drcode.providers import AgentRole
from drcode.usage_store import (
    RECENT_LIMIT,
    UsageStore,
    daily_hit_rate,
    hit_rate,
    rolling_hit_rate,
)

# -- the ratio ----------------------------------------------------------------------


def test_hit_rate_is_cache_read_over_input() -> None:
    # input already INCLUDES cache reads/writes (pydantic-ai normalization),
    # so the ratio is bounded by 1 and reads as "share of prompt from cache".
    assert hit_rate(80, 100) == pytest.approx(0.8)
    assert hit_rate(0, 100) == 0.0
    assert hit_rate(120, 100) == pytest.approx(1.2)  # foreign data; not clamped


def test_hit_rate_is_none_without_input() -> None:
    # Undefined, not zero: an empty window must not display as "0% hits".
    assert hit_rate(0, 0) is None
    assert daily_hit_rate({"requests": 3, "input": 0, "cache_read": 0}) is None


def test_daily_hit_rate_reads_a_counter_row() -> None:
    row = {"requests": 4, "input": 10_000, "cache_read": 9_000}
    assert daily_hit_rate(row) == pytest.approx(0.9)


def test_rolling_hit_rate_is_volume_weighted_and_windowed() -> None:
    records = [
        {"key": "coding/m", "input": 10_000, "cache_read": 0},
        {"key": "coding/m", "input": 1_000, "cache_read": 900},
        {"key": "tester/m", "input": 2_000, "cache_read": 0},
        {"key": "coding/m", "input": 4_000, "cache_read": 3_000},
    ]
    # Sums, not a mean of per-request rates: 3900 / 15000.
    assert rolling_hit_rate(records, key="coding/m") == pytest.approx(0.26)
    # Without the key filter every record counts.
    assert rolling_hit_rate(records) == pytest.approx(3900 / 17_000)
    # The window is the LAST n records of the (filtered) list.
    assert rolling_hit_rate(records, key="coding/m", n=1) == pytest.approx(0.75)
    # Unknown key / no input → None.
    assert rolling_hit_rate(records, key="nope") is None
    assert rolling_hit_rate([{"key": "m", "input": 0}]) is None


# -- the per-request ring of the store ----------------------------------------------


def _usage(**kw: int) -> RunUsage:
    return RunUsage(
        input_tokens=kw.get("input", 0),
        output_tokens=kw.get("output", 0),
        cache_read_tokens=kw.get("cache_read", 0),
        cache_write_tokens=kw.get("cache_write", 0),
        requests=1,
    )


def test_record_appends_per_request_history(tmp_path: Path) -> None:
    store = UsageStore(path=tmp_path / "usage.json")
    store.record(
        AgentRole.CODING, "openai/gpt-5", _usage(input=3_000, cache_read=2_500)
    )
    store.record(AgentRole.TESTER, "openai/gpt-5", _usage(input=4_000, cache_read=0))
    recent = store.load_recent()
    assert [r["key"] for r in recent] == [
        "coding/openai/gpt-5",
        "tester/openai/gpt-5",
    ]
    assert recent[0]["input"] == 3_000
    assert recent[0]["cache_read"] == 2_500
    # The day aggregates are untouched by the ring.
    days = store.load()
    assert next(iter(days.values()))["coding/openai/gpt-5"]["input"] == 3_000
    assert next(iter(days.values()))["tester/openai/gpt-5"]["input"] == 4_000


def test_recent_ring_is_trimmed_and_limit_loadable(tmp_path: Path) -> None:
    store = UsageStore(path=tmp_path / "usage.json")
    for i in range(RECENT_LIMIT + 10):
        store.record(AgentRole.CODING, "m", _usage(input=1_000 + i))
    recent = store.load_recent()
    assert len(recent) == RECENT_LIMIT
    # The oldest records were dropped, the newest survive in order.
    assert recent[-1]["input"] == 1_000 + RECENT_LIMIT + 9
    assert store.load_recent(limit=5)[-1]["input"] == 1_000 + RECENT_LIMIT + 9


def test_recent_history_is_tolerant_to_junk(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    data: dict[str, Any] = {
        "version": 1,
        "days": {},
        "recent": [
            "junk",
            {"key": "coding/m", "input": "bad", "cache_read": 5},
            {"input": 10},  # no key → dropped
            {"key": "coding/m", "input": 100, "cache_read": 90},
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    store = UsageStore(path=path)
    recent = store.load_recent()
    assert len(recent) == 2
    assert recent[0]["input"] == 0  # the string counter became 0
    assert recent[1]["cache_read"] == 90
    # A file WITHOUT the ring (an older writer) reads as no history.
    path.write_text(json.dumps({"version": 1, "days": {}}), encoding="utf-8")
    assert store.load_recent() == []


# -- the runtime monitor ------------------------------------------------------------


class AlertLog:
    def __init__(self) -> None:
        self.alerts: list[tuple[str, str, float]] = []

    def __call__(self, task_id: str, key: str, rate: float) -> None:
        self.alerts.append((task_id, key, rate))


def _monitor() -> tuple[CacheHitRateMonitor, AlertLog]:
    log = AlertLog()
    return CacheHitRateMonitor(log), log


def test_monitor_alerts_after_consecutive_misses() -> None:
    monitor, log = _monitor()
    big = _usage(input=CACHE_ALERT_MIN_INPUT, cache_read=0)
    for _ in range(CACHE_ALERT_STREAK - 1):
        monitor.record("t1", "coding/m", big)
    assert log.alerts == []
    monitor.record("t1", "coding/m", big)
    assert len(log.alerts) == 1
    task_id, key, rate = log.alerts[0]
    assert (task_id, key) == ("t1", "coding/m")
    assert rate == 0.0


def test_monitor_alerts_once_per_episode() -> None:
    monitor, log = _monitor()
    miss = _usage(input=CACHE_ALERT_MIN_INPUT, cache_read=0)
    for _ in range(CACHE_ALERT_STREAK + 5):
        monitor.record("t1", "coding/m", miss)
    assert len(log.alerts) == 1
    # A healthy request ends the episode — and re-arms the alert.
    hit = _usage(input=CACHE_ALERT_MIN_INPUT, cache_read=CACHE_ALERT_MIN_INPUT)
    monitor.record("t1", "coding/m", hit)
    for _ in range(CACHE_ALERT_STREAK):
        monitor.record("t1", "coding/m", miss)
    assert len(log.alerts) == 2


def test_monitor_streak_is_per_task_and_key() -> None:
    monitor, log = _monitor()
    miss = _usage(input=CACHE_ALERT_MIN_INPUT, cache_read=0)
    # The misses split across two keys (and two tasks) never reach the
    # streak threshold on either.
    for i in range(CACHE_ALERT_STREAK - 1):
        monitor.record("t1", "coding/m" if i % 2 else "tester/m", miss)
        monitor.record("t2", "coding/m", miss)
    assert log.alerts == []


def test_monitor_skips_small_requests_and_none_usage() -> None:
    monitor, log = _monitor()
    small = _usage(input=CACHE_ALERT_MIN_INPUT - 1, cache_read=0)
    for _ in range(CACHE_ALERT_STREAK * 3):
        monitor.record("t1", "coding/m", small)
        monitor.record("t1", "coding/m", None)
    assert log.alerts == []
    # A small request after misses does not RESET the streak either —
    # it is simply invisible to the monitor.
    miss = _usage(input=CACHE_ALERT_MIN_INPUT, cache_read=0)
    for _ in range(CACHE_ALERT_STREAK - 1):
        monitor.record("t1", "coding/m", miss)
    monitor.record("t1", "coding/m", small)
    monitor.record("t1", "coding/m", miss)
    assert len(log.alerts) == 1


def test_monitor_alert_reports_volume_weighted_rate() -> None:
    monitor, log = _monitor()
    monitor.record("t1", "coding/m", _usage(input=10_000, cache_read=1_000))
    for _ in range(CACHE_ALERT_STREAK - 1):
        monitor.record("t1", "coding/m", _usage(input=2_000, cache_read=0))
    _, _, rate = log.alerts[0]
    # 1000 / 18000 over the whole episode, not the last request's 0%.
    assert rate == pytest.approx(1_000 / 18_000)


def test_monitor_forget_drops_a_task_state() -> None:
    monitor, log = _monitor()
    miss = _usage(input=CACHE_ALERT_MIN_INPUT, cache_read=0)
    for _ in range(CACHE_ALERT_STREAK - 1):
        monitor.record("t1", "coding/m", miss)
    monitor.forget("t1")
    for _ in range(CACHE_ALERT_STREAK - 1):
        monitor.record("t1", "coding/m", miss)
    assert log.alerts == []


# -- the pipeline wiring ------------------------------------------------------------


@pytest.fixture
def app_factory(project_root, monkeypatch: pytest.MonkeyPatch):
    """Build a TUI app over the temp project (the pipeline methods are
    exercised without a running app — same harness as the throttle tests)."""
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")

    def make() -> Any:
        from drcode.tui.app import DrCodeApp

        return DrCodeApp(project_root)

    return make


def test_cache_hit_alert_logs_a_service_event(app_factory: Any) -> None:
    from tests.network_mocks import event_texts

    app = app_factory()
    task = app.manager.create("docs", "hit the cache")
    key = f"{AgentRole.CODING.value}/openai/gpt-5"

    app._cache_hit_alert(task.id, key, 0.12)

    texts = event_texts(task)
    assert any("Cache hit-rate упал до 12%" in t for t in texts)
    assert any(key in t for t in texts)
    # The event is persisted and never enters the model transcript.
    assert app.manager.get(task.id).messages == []


def test_cache_hit_alert_is_deduplicated(app_factory: Any) -> None:
    from tests.network_mocks import event_texts

    app = app_factory()
    task = app.manager.create("docs", "hit the cache")
    key = f"{AgentRole.CODING.value}/openai/gpt-5"

    app._cache_hit_alert(task.id, key, 0.12)
    app._cache_hit_alert(task.id, key, 0.12)

    assert len([t for t in event_texts(task) if "hit-rate" in t]) == 1


def test_cache_hit_alert_survives_a_deleted_task(app_factory: Any) -> None:
    app = app_factory()
    task = app.manager.create("docs", "hit the cache")
    app.manager.delete(task.id)
    app._cache_hit_alert(task.id, "coding/m", 0.1)  # no raise
