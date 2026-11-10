"""Round counters (`rounds: dev … · test … · resolve …`) in the task
statusbar's second row: hidden while all three counters are zero, zero
values dropped inside the fragment otherwise."""

import pytest
from textual.content import Content
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


def _styles_at(content: "Content", index: int) -> set[str]:
    """Style words of every span of a rendered Content covering `index`."""
    return {
        word
        for span in content.spans
        if span.start <= index < span.end
        for word in str(span.style).split()
    }


async def test_rounds_stays_dim_with_goal_assigned(app: DrCodeApp) -> None:
    """Regression: the goal fragment opened [magenta] and [@click=…] but
    closed only the inner tag, so [magenta] leaked onto every following
    fragment — `rounds:` rendered magenta+dim instead of plain dim (and
    the branch fragment's [deepskyblue] leaked the same way)."""
    async with app.run_test() as pilot:
        await pilot.pause()
        goal = app.manager.goals.create("Async core")
        task: Task = app.manager.create("do it", "do it", goal_id=goal.id)
        task.messages = [_prompt("do it")]
        task.tester_messages = [[_prompt("verify")]]
        task.isolation_pinned = True
        app.view_task(task.id)
        await pilot.pause()
        app.refresh_task_status()
        # Static.content is the raw str here (VisualType); render() applies
        # the markup and returns a Content — narrowed for pyright.
        content = app.query_one("#task-status-extra", Static).render()
        assert isinstance(content, Content)
        assert "rounds:" in content.plain
        # The rounds fragment is dim only — no leaked magenta from the goal.
        assert "magenta" not in _styles_at(content, content.plain.index("rounds:"))
        assert "dim" in _styles_at(content, content.plain.index("rounds:"))
        # The branch fragment carries deepskyblue inside its own bounds…
        branch = f"drcode/{task.id}"
        bstart = content.plain.index(f"branch: {branch}")
        assert "deepskyblue" in _styles_at(content, bstart)
        # …and stops at its end (no leak onto the rounds fragment).
        bend = bstart + len(f"branch: {branch}") - 1
        assert "deepskyblue" not in _styles_at(content, bend + 1)
