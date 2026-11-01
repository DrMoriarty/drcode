"""Round counters (`rounds: dev … · test … · resolve …`) in the task
statusbar's second row: hidden while all three counters are zero, zero
values dropped inside the fragment otherwise."""

import pytest
from textual.widgets import Static

from drcode.models import Task
from drcode.tui.app import DrCodeApp


@pytest.fixture
def app(project_root) -> DrCodeApp:
    return DrCodeApp(project_root)


def _prompt(text: str) -> dict:
    return {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": text}]}


async def test_rounds_fragment_hidden_on_fresh_task(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        task = app.manager.create("do it", "do it")
        app.view_task(task.id)
        await pilot.pause()
        assert app.viewed is task
        app.refresh_task_status()
        assert "rounds:" not in str(app.query_one("#task-status-extra", Static).content)


async def test_rounds_fragment_counts_and_drops_zeros(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        task: Task = app.manager.create("do it", "do it")
        # dev: initial prompt + one follow-up (e.g. a tester bug-report
        # round) = 2; test: one finished round; resolver: never ran.
        task.messages = [_prompt("do it"), _prompt("fix the flake")]
        task.tester_messages = [[_prompt("verify")]]
        task.resolver_messages = []
        app.view_task(task.id)
        await pilot.pause()
        app.refresh_task_status()
        extra = str(app.query_one("#task-status-extra", Static).content)
        assert "rounds: dev 2 · test 1" in extra
        assert "resolve" not in extra.replace("rounds: dev 2 · test 1", "")

        # A resolver round appears and is counted.
        task.resolver_messages = [[_prompt("resolve")]]
        app.refresh_task_status()
        extra = str(app.query_one("#task-status-extra", Static).content)
        assert "rounds: dev 2 · test 1 · resolve 1" in extra
