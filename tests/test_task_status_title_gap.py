"""Regression: the task-statusbar header keeps at least one blank cell
between the (possibly truncated) title and the following cluster
(creation date / status / context counter).

The title-room calculation must reserve the separator cell — a title
truncated to the full room used to fill the 1fr left widget exactly, so
the mid cluster started flush against the ellipsis with no space."""

import pytest
from rich.cells import cell_len
from textual.content import Content
from textual.widgets import Static

from drcode.tui.app import DrCodeApp


@pytest.fixture
def app(project_root) -> DrCodeApp:
    return DrCodeApp(project_root)


async def test_gap_between_truncated_title_and_date(app: DrCodeApp) -> None:
    task = app.manager.create("x" * 200, "the original prompt")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        app.refresh_task_status()
        await pilot.pause()
        left = app.query_one("#task-status-left", Static)
        mid = app.query_one("#task-status-mid", Static)
        left_render = left.render()
        mid_render = mid.render()
        assert isinstance(left_render, Content)
        assert isinstance(mid_render, Content)
        # The preconditions: the cluster is rendered and the long title
        # really went through the truncation path (the ellipsis).
        assert mid_render.plain.strip()
        assert left_render.plain.endswith("…")
        gap = mid.region.x - (left.region.x + cell_len(left_render.plain))
        assert gap >= 1


async def test_gap_with_short_title(app: DrCodeApp) -> None:
    """A title that fits leaves the gap for free — assert it stays."""
    task = app.manager.create("short", "the original prompt")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        app.refresh_task_status()
        await pilot.pause()
        left = app.query_one("#task-status-left", Static)
        mid = app.query_one("#task-status-mid", Static)
        left_render = left.render()
        mid_render = mid.render()
        assert isinstance(left_render, Content)
        assert isinstance(mid_render, Content)
        assert mid_render.plain.strip()
        gap = mid.region.x - (left.region.x + cell_len(left_render.plain))
        assert gap >= 1
