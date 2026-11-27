"""Unit tests for the user-level token-usage store
(drcode.usage_store): atomic writes, corrupt-file tolerance, the explicit
path / agent guard (incident ebe6ba15 discipline), the version field and
the per-role model keys of the TUI usage accounting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.usage import RunUsage
from rich.text import Text
from textual.widgets import DataTable, Tabs

import drcode.config as config_module
from drcode.providers import AgentRole, UserModelsStore
from drcode.usage_store import FIELDS, VERSION, UsageStore


def _usage(**kw: int) -> RunUsage:
    return RunUsage(
        input_tokens=kw.get("input", 0),
        output_tokens=kw.get("output", 0),
        cache_read_tokens=kw.get("cache_read", 0),
        cache_write_tokens=kw.get("cache_write", 0),
        requests=kw.get("requests", 1),
    )


def _plain(cell: object) -> str:
    """Plain text of a usage-table cell (Rich Text on the styled rows,
    a bare string on the padding cells)."""
    return cell.plain if isinstance(cell, Text) else str(cell)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# -- basics -----------------------------------------------------------------------


def test_explicit_path_is_honored(tmp_path: Path) -> None:
    target = tmp_path / "usage.json"
    store = UsageStore(path=target)
    assert store.path == target
    store.record(AgentRole.CODING, "openai/gpt-5", _usage(input=100, output=5))
    assert target.exists()
    data = _read(target)
    assert data["version"] == VERSION
    day = next(iter(data["days"]))
    assert data["days"][day] == {
        "coding/openai/gpt-5": {
            "requests": 1,
            "input": 100,
            "output": 5,
            "cache_read": 0,
            "cache_write": 0,
        }
    }


def test_default_path_sits_in_the_user_config_dir() -> None:
    # The conftest sandbox redirects config._USER_CONFIG_DIR; the default
    # store must resolve next to models.json THERE (never the real
    # ~/.config/drcode).
    store = UsageStore()
    assert store.path == config_module._USER_CONFIG_DIR / "usage.json"


def test_record_accumulates_per_day_and_model(tmp_path: Path) -> None:
    target = tmp_path / "usage.json"
    store = UsageStore(path=target)
    store.record(AgentRole.CODING, "openai/gpt-5", _usage(input=100, output=5))
    store.record(
        AgentRole.CODING, "openai/gpt-5", _usage(input=50, output=7, requests=2)
    )
    data = _read(target)
    day = next(iter(data["days"]))
    entry = data["days"][day]["coding/openai/gpt-5"]
    assert entry["requests"] == 3
    assert entry["input"] == 150
    assert entry["output"] == 12


def test_version_field_is_written(tmp_path: Path) -> None:
    target = tmp_path / "usage.json"
    UsageStore(path=target).record(AgentRole.TESTER, "m", _usage(input=1))
    assert _read(target)["version"] == VERSION


def test_load_of_missing_file_is_empty(tmp_path: Path) -> None:
    assert UsageStore(path=tmp_path / "absent.json").load() == {}


def test_no_tmp_files_left_behind(tmp_path: Path) -> None:
    target = tmp_path / "usage.json"
    store = UsageStore(path=target)
    for _ in range(5):
        store.record(AgentRole.CODING, "m", _usage(input=1))
    assert list(tmp_path.iterdir()) == [target]


# -- different roles land under different keys -------------------------------------


def test_roles_get_separate_keys(tmp_path: Path) -> None:
    """The gate: calls made by different roles are accounted separately,
    even when they resolve to the SAME model id."""
    target = tmp_path / "usage.json"
    store = UsageStore(path=target)
    store.record(AgentRole.CODING, "openai/gpt-5", _usage(input=100))
    store.record(AgentRole.TESTER, "openai/gpt-5", _usage(input=200))
    day = next(iter(store.load()))
    days = store.load()[day]
    assert set(days) == {"coding/openai/gpt-5", "tester/openai/gpt-5"}
    assert days["coding/openai/gpt-5"]["input"] == 100
    assert days["tester/openai/gpt-5"]["input"] == 200


def test_role_prefix_survives_empty_model_id(tmp_path: Path) -> None:
    target = tmp_path / "usage.json"
    store = UsageStore(path=target)
    store.record(AgentRole.CODING, "", _usage(input=1))
    assert set(next(iter(store.load().values()))) == {"coding"}


def test_composite_model_id_key_roundtrip(tmp_path: Path) -> None:
    """A composite model id ("provider/vendor/model") makes the usage key
    carry multiple slashes — the key is stored verbatim and never parsed
    back: record + load keep it intact."""
    target = tmp_path / "usage.json"
    store = UsageStore(path=target)
    store.record(AgentRole.CODING, "prov/deepseek/deepseek-v4-0731", _usage(input=10))
    day = next(iter(store.load()))
    days = store.load()[day]
    assert set(days) == {"coding/prov/deepseek/deepseek-v4-0731"}
    assert days["coding/prov/deepseek/deepseek-v4-0731"]["input"] == 10


# -- corrupt / foreign data is skipped, never fatal --------------------------------


def test_corrupt_json_is_skipped_and_record_starts_fresh(tmp_path: Path) -> None:
    target = tmp_path / "usage.json"
    target.write_text("{not json at all", encoding="utf-8")
    store = UsageStore(path=target)
    assert store.load() == {}
    # Recording over a corrupt file starts a fresh valid one.
    store.record(AgentRole.CODING, "m", _usage(input=3))
    data = store.load()
    day = next(iter(data))
    assert data[day]["coding/m"]["input"] == 3


def test_wrong_shape_is_skipped(tmp_path: Path) -> None:
    target = tmp_path / "usage.json"
    target.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert UsageStore(path=target).load() == {}
    target.write_text(json.dumps({"version": 1, "days": "nope"}), encoding="utf-8")
    assert UsageStore(path=target).load() == {}
    target.write_text(
        json.dumps({"version": 1, "days": {"2026-01-01": {"m": {"input": "x"}}}}),
        encoding="utf-8",
    )
    days = UsageStore(path=target).load()
    # Non-int counters are dropped (zeroed), the entry survives.
    assert days["2026-01-01"]["m"] == dict.fromkeys(FIELDS, 0)


def test_foreign_version_is_ignored(tmp_path: Path) -> None:
    target = tmp_path / "usage.json"
    target.write_text(
        json.dumps({"version": 999, "days": {"2026-01-01": {}}}), encoding="utf-8"
    )
    assert UsageStore(path=target).load() == {}


# -- frozen-cost section (version 2) -------------------------------------------------


def test_record_with_cost_accumulates_money(tmp_path: Path) -> None:
    """A priced request folds its FROZEN cost into the day's ``costs``
    cell (sum + priced-session count); an unpriced request in the same
    day contributes NOTHING to the money cell but keeps its tokens."""
    target = tmp_path / "usage.json"
    store = UsageStore(path=target)
    store.record(AgentRole.CODING, "openai/gpt-5", _usage(input=100), cost=0.001)
    store.record(
        AgentRole.CODING, "openai/gpt-5", _usage(input=50), cost=0.002, task_id="t1"
    )
    store.record(AgentRole.CODING, "openai/gpt-5", _usage(input=70))
    day = next(iter(store.load()))
    costs = store.load_costs()
    cell = costs[day]["coding/openai/gpt-5"]
    assert cell["cost"] == pytest.approx(0.003)
    assert cell["sessions"] == 2  # the unpriced request never entered
    # Token accounting is untouched by the money layer.
    assert store.load()[day]["coding/openai/gpt-5"]["input"] == 220
    # The recent ring carries the per-request audit trail.
    recent = store.load_recent()
    assert [r.get("cost") for r in recent] == [0.001, 0.002, None]
    assert [r.get("task") for r in recent] == ["", "t1", ""]


def test_record_without_cost_writes_no_money_cells(tmp_path: Path) -> None:
    target = tmp_path / "usage.json"
    store = UsageStore(path=target)
    store.record(AgentRole.TESTER, "m", _usage(input=1))
    assert store.load_costs() == {}
    data = _read(target)
    assert data["version"] == VERSION
    assert data["costs"] == {}


def test_v1_file_loads_and_upgrades_in_place(tmp_path: Path) -> None:
    """A version-1 file (pre-cost era) stays fully readable: its days
    load as-is, money fields are simply absent — and the next record
    rewrites it in the v2 shape WITHOUT losing the old days."""
    target = tmp_path / "usage.json"
    target.write_text(
        json.dumps(
            {
                "version": 1,
                "days": {
                    "2026-01-01": {
                        "coding/openai/gpt-5": {
                            "requests": 4,
                            "input": 1000,
                            "output": 100,
                            "cache_read": 0,
                            "cache_write": 0,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    store = UsageStore(path=target)
    days = store.load()
    assert days["2026-01-01"]["coding/openai/gpt-5"]["input"] == 1000
    assert store.load_costs() == {}  # no recorded cost — absent, not zero
    store.record(AgentRole.CODING, "openai/gpt-5", _usage(input=10), cost=0.5)
    data = _read(target)
    assert data["version"] == VERSION
    assert data["days"]["2026-01-01"]["coding/openai/gpt-5"]["input"] == 1000
    cost_day = next(iter(data["costs"]))  # today's day, distinct from the v1 one
    assert data["costs"][cost_day]["coding/openai/gpt-5"] == {
        "cost": 0.5,
        "sessions": 1,
    }


def test_malformed_cost_cells_are_dropped(tmp_path: Path) -> None:
    """One insane money cell (string cost, zero/negative sessions) is
    dropped, never fatal — and never takes the days mapping down."""
    target = tmp_path / "usage.json"
    target.write_text(
        json.dumps(
            {
                "version": VERSION,
                "days": {"2026-01-01": {"m": dict.fromkeys(FIELDS, 1)}},
                "costs": {
                    "2026-01-01": {
                        "m": {"cost": "rich", "sessions": 2},
                        "m2": {"cost": 1.0, "sessions": 0},
                        "m3": {"cost": 1.5, "sessions": 3},
                    },
                    "oops": "not a dict",
                },
            }
        ),
        encoding="utf-8",
    )
    store = UsageStore(path=target)
    assert store.load()["2026-01-01"]["m"]["input"] == 1  # days unaffected
    costs = store.load_costs()
    assert set(costs["2026-01-01"]) == {"m3"}
    assert costs["2026-01-01"]["m3"] == {"cost": 1.5, "sessions": 3}


# -- the ebe6ba15 agent guard -------------------------------------------------------


def test_record_under_agent_without_explicit_path_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DRCODE_TASK_ID", "t1")
    with pytest.raises(ValueError, match="explicit path"):
        UsageStore().record(AgentRole.CODING, "m", _usage(input=1))


def test_record_under_agent_with_explicit_path_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DRCODE_TASK_ID", "t1")
    target = tmp_path / "usage.json"
    UsageStore(path=target).record(AgentRole.CODING, "m", _usage(input=1))
    assert UsageStore(path=target).load()


# -- TUI wiring ---------------------------------------------------------------------


async def test_app_records_usage_per_role_and_model(
    project_root: Path,
) -> None:
    """The TUI hook folds into the persistent store under the exact
    "<role>/<model>" keys its call sites pass."""
    from drcode.tui.app import DrCodeApp

    app = DrCodeApp(project_root)
    async with app.run_test():
        app._record_session_usage(
            _usage(input=100, output=10), AgentRole.CODING, "openai/gpt-5"
        )
        app._record_session_usage(
            _usage(input=40), AgentRole.TESTER, "anthropic/claude-x"
        )
    days = app._usage_store.load()
    assert len(days) == 1
    day = next(iter(days))
    assert set(days[day]) == {
        "coding/openai/gpt-5",
        "tester/anthropic/claude-x",
    }
    assert days[day]["coding/openai/gpt-5"]["input"] == 100
    assert days[day]["tester/anthropic/claude-x"]["input"] == 40
    # The session header totals still fold the same calls.
    assert app._session_usage.input_tokens == 140
    assert app._session_usage.output_tokens == 10


async def test_app_usage_hook_survives_drcode_task_id(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (dogfood child TUI): a dev agent exercising
    `uv run drcode` via run_bash inherits DRCODE_TASK_ID, and the app's
    store must therefore NOT trip the ebe6ba15 agent-guard — the app
    passes the resolved path explicitly, so the guard keeps firing only
    for bare UsageStore() snippets. The conftest strips the variable for
    every other test; here it is set back exactly like run_bash does.
    The resolved path lands in the conftest-sandboxed user config dir,
    never the real ~/.config/drcode."""
    from drcode.tui.app import DrCodeApp

    monkeypatch.setenv("DRCODE_TASK_ID", "dogfood-t1")
    app = DrCodeApp(project_root)
    # The guard is path-based: the app's explicit path must have been
    # resolved through the ordinary layering (the sandbox here).
    assert app._usage_store._explicit_path
    assert app._usage_store.path == (config_module._USER_CONFIG_DIR / "usage.json")
    async with app.run_test():
        app._record_session_usage(_usage(input=100), AgentRole.CODING, "openai/gpt-5")
    days = app._usage_store.load()
    assert next(iter(days.values()))["coding/openai/gpt-5"]["input"] == 100


async def test_app_usage_hook_is_best_effort(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing store write must never break the model run the hook
    sits in the hot path of: the session totals still fold, the
    exception is swallowed (the run's own except handlers would demote
    the task for a statistics error — far too high a price)."""
    from drcode.tui.app import DrCodeApp

    app = DrCodeApp(project_root)
    async with app.run_test():

        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("disk exploded")

        monkeypatch.setattr(app._usage_store, "record", boom)
        app._record_session_usage(
            _usage(input=100, output=10), AgentRole.CODING, "openai/gpt-5"
        )
        assert app._session_usage.input_tokens == 100
        assert app._session_usage.output_tokens == 10


async def test_app_usage_dialog_renders_the_store(
    project_root: Path,
) -> None:
    """The token-usage tab of the combined statistics dialog renders the
    store: activating the tab triggers the lazy load, and the table shows
    one data row + the period total."""
    from drcode.tui.app import DrCodeApp
    from drcode.tui.stats_dialog import StatsDialog

    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        app._record_session_usage(
            _usage(input=100, output=10, cache_read=50, requests=2),
            AgentRole.CODING,
            "openai/gpt-5",
        )
        dialog = StatsDialog(project_root, usage_store=app._usage_store)
        await app.push_screen(dialog)
        await pilot.pause()
        # Switch to the usage tab — the load starts on the FIRST
        # activation.
        dialog.query_one("#stats-tabs", Tabs).active = "usage"
        await app.workers.wait_for_complete()
        await pilot.pause()
        table = dialog.query_one("#usage-table", DataTable)
        # One date header row + one data row + the period total.
        assert table.row_count == 3
        header, model_row, total = (table.get_row_at(i) for i in range(3))
        # The day is a styled group header row, not a cell: date in the
        # first cell, everything else empty; the model row carries the
        # key first.
        assert _plain(header[0]).count("-") == 2  # a YYYY-MM-DD date
        assert all(_plain(cell) == "" for cell in header[1:])
        assert _plain(model_row[0]) == "coding/openai/gpt-5"
        assert _plain(total[0]) == "Σ период"
        status = str(dialog.query_one("#usage-status").render())
        assert "1 дн." in status


async def test_app_usage_dialog_cost_column(
    project_root: Path, user_config_dir: Path
) -> None:
    """The cost column prices each (day, model) row from the model's
    catalog prices: input minus cache reads at the prompt price, cache
    reads at the cache price, output at the completion price. Rows of
    unpriced models show an EMPTY cell (never 0); the period total sums
    the computable rows."""
    from drcode.providers import ModelEntry, Provider, UserModelsConfig
    from drcode.tui.app import DrCodeApp
    from drcode.tui.stats_dialog import StatsDialog

    config_module._USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    UserModelsStore().save(
        UserModelsConfig(
            providers=[
                Provider(
                    name="zveno",
                    base_url="https://zveno.ai/v1",
                    models=[
                        ModelEntry(
                            id="m1",
                            price_prompt=0.000002,
                            price_completion=0.000008,
                            price_cache_read=0.0000005,
                        ),
                        ModelEntry(id="unpriced"),
                    ],
                )
            ]
        )
    )

    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        # cost = (1000-500)*0.000002 + 500*0.0000005 + 100*0.000008
        #      = 0.001 + 0.00025 + 0.0008 = 0.00205
        app._record_session_usage(
            _usage(input=1000, output=100, cache_read=500, requests=1),
            AgentRole.CODING,
            "zveno/m1",
        )
        app._record_session_usage(
            _usage(input=100, output=10, requests=1),
            AgentRole.CODING,
            "zveno/unpriced",
        )
        dialog = StatsDialog(project_root, usage_store=app._usage_store)
        await app.push_screen(dialog)
        await pilot.pause()
        dialog.query_one("#stats-tabs", Tabs).active = "usage"
        await app.workers.wait_for_complete()
        await pilot.pause()
        table = dialog.query_one("#usage-table", DataTable)
        # Date header + two data rows + the period total.
        assert table.row_count == 4
        # Rows are sorted by (day desc, key): the date header, then
        # "coding/zveno/m1", then "coding/zveno/unpriced", then the Σ row.
        first, second, total = (table.get_row_at(i) for i in range(1, 4))
        assert _plain(first[-1]) == "$0.00205"
        assert _plain(second[-1]) == ""  # unknown price — empty, not zero
        assert _plain(total[-1]) == "$0.00205"


async def test_app_usage_dialog_cost_without_cache_read_price(
    project_root: Path, user_config_dir: Path
) -> None:
    """A model with prompt/completion prices but no cache-read price is
    still costed while it has zero cache reads; once a row HAS cache
    reads, its cell goes empty (the cache term is unknowable)."""
    from drcode.providers import ModelEntry, Provider, UserModelsConfig
    from drcode.tui.app import DrCodeApp
    from drcode.tui.stats_dialog import StatsDialog

    config_module._USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    UserModelsStore().save(
        UserModelsConfig(
            providers=[
                Provider(
                    name="zveno",
                    base_url="https://zveno.ai/v1",
                    models=[
                        ModelEntry(
                            id="m1", price_prompt=0.000002, price_completion=0.000008
                        )
                    ],
                )
            ]
        )
    )

    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        app._record_session_usage(
            _usage(input=1000, output=100, requests=1),
            AgentRole.CODING,
            "zveno/m1",
        )
        app._record_session_usage(
            _usage(input=1000, output=100, cache_read=500, requests=1),
            AgentRole.RESEARCH,
            "zveno/m1",
        )
        dialog = StatsDialog(project_root, usage_store=app._usage_store)
        await app.push_screen(dialog)
        await pilot.pause()
        dialog.query_one("#stats-tabs", Tabs).active = "usage"
        await app.workers.wait_for_complete()
        await pilot.pause()
        table = dialog.query_one("#usage-table", DataTable)
        rows = [table.get_row_at(i) for i in range(table.row_count)]
        # Model rows are keyed by the FIRST cell now (the day header rows
        # carry the date there); a model key always contains a slash.
        by_key = {
            _plain(row[0]): _plain(row[-1]) for row in rows if "/" in _plain(row[0])
        }
        coding = by_key["coding/zveno/m1"]
        research = by_key["research/zveno/m1"]
        # cost = 1000*0.000002 + 100*0.000008 = 0.0028
        assert coding == "$0.0028"
        assert research == ""  # cache reads present, cache price unknown


# -- the frozen task-cost ledger (TUI hook) -------------------------------------------


def _priced_catalog() -> None:
    """A tiny user catalog: one priced model, one unpriced."""
    from drcode.providers import ModelEntry, Provider, UserModelsConfig

    config_module._USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    UserModelsStore().save(
        UserModelsConfig(
            providers=[
                Provider(
                    name="zveno",
                    base_url="https://zveno.ai/v1",
                    models=[
                        ModelEntry(
                            id="m1",
                            price_prompt=0.000002,
                            price_completion=0.000008,
                        ),
                        ModelEntry(id="free-local"),
                    ],
                )
            ]
        )
    )


async def test_app_task_cost_freezes_per_request(
    project_root: Path, user_config_dir: Path
) -> None:
    """The hook folds each priced request's cost into the task's
    cumulative `cost_usd` (frozen at record time) AND persists it into
    the metadata file — compaction cannot destroy it later. An unpriced
    model's session contributes NOTHING (not zero, nothing)."""
    from drcode.tui.app import DrCodeApp

    _priced_catalog()
    app = DrCodeApp(project_root)
    async with app.run_test():
        task = app.manager.create("paid", "work the bill")
        # cost = 1000*0.000002 + 100*0.000008 = 0.0028
        app._record_session_usage(
            _usage(input=1000, output=100, requests=1),
            AgentRole.CODING,
            "zveno/m1",
            task_id=task.id,
        )
        app._record_session_usage(
            _usage(input=5000, output=10, requests=1),
            AgentRole.COMPACTION,
            "zveno/free-local",  # unpriced: no cost, no ledger entry
            task_id=task.id,
        )
        app._record_session_usage(
            _usage(input=2000, output=50, requests=1),
            AgentRole.TESTER,
            "zveno/m1",
            task_id=task.id,
        )
        assert task.cost_usd == pytest.approx(0.0028 + 0.0044)
        # Survives a reload from disk (metadata, not just memory).
        meta = json.loads(
            (project_root / ".drcode" / "tasks" / f"{task.id}.json").read_text(
                encoding="utf-8"
            )
        )
        assert meta["cost_usd"] == pytest.approx(task.cost_usd)
        # The store's money section saw exactly the two priced sessions.
        costs = app._usage_store.load_costs()
        cell = costs[next(iter(costs))]["coding/zveno/m1"]
        assert cell["sessions"] == 1
        tester_cell = costs[next(iter(costs))]["tester/zveno/m1"]
        assert tester_cell["sessions"] == 1


async def test_app_task_cost_survives_a_failing_store(
    project_root: Path, user_config_dir: Path
) -> None:
    """The task-side ledger is independent of the usage file's fate:
    a store write failure must not lose the task's own figure (and the
    header session totals fold regardless)."""
    from drcode.tui.app import DrCodeApp

    _priced_catalog()
    app = DrCodeApp(project_root)
    async with app.run_test():
        task = app.manager.create("paid", "work the bill")

        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("disk exploded")

        original = app._usage_store.record
        object.__setattr__(app._usage_store, "record", boom)
        try:
            app._record_session_usage(
                _usage(input=1000, output=100, requests=1),
                AgentRole.CODING,
                "zveno/m1",
                task_id=task.id,
            )
        finally:
            object.__setattr__(app._usage_store, "record", original)
        assert task.cost_usd == pytest.approx(0.0028)
        assert app._session_usage.input_tokens == 1000
        assert app._usage_store.load_costs() == {}


async def test_app_header_shows_live_task_cost(
    project_root: Path, user_config_dir: Path
) -> None:
    """The expanded header's second line carries the viewed task's
    cumulative cost: a dash while no priced session exists (never a
    bare "$0"), the frozen sum once one lands, and no segment at all
    when no task is viewed."""
    from drcode.tui.app import DrCodeApp

    _priced_catalog()
    app = DrCodeApp(project_root)
    async with app.run_test():
        # No viewed task: no segment.
        assert "Задача" not in app._header_stats_text()
        task = app.manager.create("paid", "work the bill")
        app.viewed_id = task.id
        assert app._header_stats_text().startswith("Задача: —")
        app._record_session_usage(
            _usage(input=1000, output=100, requests=1),
            AgentRole.CODING,
            "zveno/m1",
            task_id=task.id,
        )
        assert app._header_stats_text().startswith("Задача: $0.0028")
        # A different (unpriced) viewed task shows the dash again.
        other = app.manager.create("free", "work the other bill")
        app.viewed_id = other.id
        assert app._header_stats_text().startswith("Задача: —")


async def test_app_usage_dialog_shows_agent_cost_averages(
    project_root: Path, user_config_dir: Path
) -> None:
    """The frozen-cost averages table: one row per (day, agent) with
    priced sessions — the average over the PRICED session count, free
    models invisible; the day×model table above keeps its rows."""
    from textual.widgets import DataTable

    from drcode.tui.app import DrCodeApp
    from drcode.tui.stats_dialog import StatsDialog

    _priced_catalog()
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        # coding: 0.0028 and 0.0056 → avg 0.0042 over 2 sessions.
        app._record_session_usage(
            _usage(input=1000, output=100, requests=1),
            AgentRole.CODING,
            "zveno/m1",
        )
        app._record_session_usage(
            _usage(input=2000, output=200, requests=1),
            AgentRole.CODING,
            "zveno/m1",
        )
        # tester: one priced session, 0.0044.
        app._record_session_usage(
            _usage(input=2000, output=50, requests=1),
            AgentRole.TESTER,
            "zveno/m1",
        )
        # compaction on the free model: no money anywhere.
        app._record_session_usage(
            _usage(input=999, output=1, requests=1),
            AgentRole.COMPACTION,
            "zveno/free-local",
        )
        dialog = StatsDialog(project_root, usage_store=app._usage_store)
        await app.push_screen(dialog)
        await pilot.pause()
        dialog.query_one("#stats-tabs", Tabs).active = "usage"
        await app.workers.wait_for_complete()
        await pilot.pause()
        cost_table = dialog.query_one("#cost-table", DataTable)
        rows = [cost_table.get_row_at(i) for i in range(cost_table.row_count)]
        # Newest-first row order between days is irrelevant here (one
        # day): sorted by (day, role) desc → tester first, coding second.
        assert rows[0][1] == "tester"
        assert rows[0][2] == "1"
        assert rows[0][3] == "$0.0044"
        assert rows[0][4] == "$0.0044"
        assert rows[1][1] == "coding"
        assert rows[1][2] == "2"
        assert rows[1][3] == "$0.0042"
        assert rows[1][4] == "$0.0084"
        assert cost_table.row_count == 2  # compaction never appears
        # The day×model table above is intact: date header + 3 data rows + Σ.
        usage_table = dialog.query_one("#usage-table", DataTable)
        assert usage_table.row_count == 5
