"""Unit tests for the user-level token-usage store
(drcode.usage_store): atomic writes, corrupt-file tolerance, the explicit
path / agent guard (incident ebe6ba15 discipline), the version field and
the per-role model keys of the TUI usage accounting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.usage import RunUsage
from textual.widgets import DataTable, Tabs

import drcode.config as config_module
from drcode.providers import AgentRole
from drcode.usage_store import FIELDS, VERSION, UsageStore


def _usage(**kw: int) -> RunUsage:
    return RunUsage(
        input_tokens=kw.get("input", 0),
        output_tokens=kw.get("output", 0),
        cache_read_tokens=kw.get("cache_read", 0),
        cache_write_tokens=kw.get("cache_write", 0),
        requests=kw.get("requests", 1),
    )


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
    assert _read(target)["version"] == 1


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

        def boom(role: AgentRole, model_id: str, usage: RunUsage) -> None:
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
        # One data row + the period total.
        assert table.row_count == 2
        status = str(dialog.query_one("#usage-status").render())
        assert "1 дн." in status
