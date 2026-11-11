"""UI-level behavior of the manual queue reordering (Move up/Move down).

Two contracts:

- a legal move visibly reorders the sidebar's QUEUE section (the
  snap-back regression: a legal move around a dependency-blocked task
  used to be undone by the blocked tier of the queue sort);
- an impossible move (queue edge, bug-block boundary, blocked tier) must
  never die silently: check_action keeps the dispatch alive and the
  action toasts the reason — a falsy check_action suppresses the key
  dispatch entirely (App.run_action), which read as "the sorting is
  broken".
"""

from __future__ import annotations

import pytest

# FooterKey is not re-exported from textual.widgets — only the private
# module defines it; importing it here also lets query(FooterKey) narrow
# the result type for pyright (no cast needed).
from textual.widgets._footer import FooterKey

from drcode.models import TaskKind
from drcode.tui.app import DrCodeApp


@pytest.fixture
def app(project_root) -> DrCodeApp:
    return DrCodeApp(project_root)


async def _started(app: DrCodeApp, pilot) -> None:
    # Skip the first-start setup dialog (no model configured in tests).
    await pilot.press("escape")
    await pilot.pause()


def _footer_move_keys(app: DrCodeApp) -> list[str]:
    return sorted(
        k.key for k in app.query(FooterKey) if k.description in ("Move up", "Move down")
    )


async def test_queued_task_keeps_move_bindings_dispatchable(
    app: DrCodeApp,
) -> None:
    """A queued task adjacent to the bug block still owns BOTH footer
    entries: the impossible direction must stay dispatchable so the key
    press can explain itself (instead of being silently swallowed)."""
    async with app.run_test(size=(220, 50)) as pilot:
        await _started(app, pilot)
        app.manager.create("bug", "b", kind=TaskKind.BUG)
        a = app.manager.create("a", "a")
        app.manager.create("b", "b")
        app.view_task(a.id)  # mid-queue: up legal, down crosses the bug block
        await pilot.pause()
        assert app.check_action("move_task_up", ()) is True
        assert app.check_action("move_task_down", ()) is True
        assert _footer_move_keys(app) == ["ctrl+down", "ctrl+up"]


async def test_impossible_move_toasts_instead_of_dying_silently(
    app: DrCodeApp,
) -> None:
    async with app.run_test(size=(220, 50)) as pilot:
        await _started(app, pilot)
        app.manager.create("bug", "b", kind=TaskKind.BUG)
        a = app.manager.create("a", "a")
        app.manager.create("b", "b")
        # queued(): [bug, a, b]
        app.view_task(a.id)
        await pilot.pause()
        toasts: list[tuple[str, str]] = []
        app.notify = lambda message, *, title="", **_: toasts.append((title, message))  # type: ignore[method-assign]

        # Ctrl+Down crosses the bug block: order must not change and the
        # user must get the reason — not silence.
        await pilot.press("ctrl+down")
        await pilot.pause()
        assert [t.title for t in app.manager.queued()] == ["bug", "a", "b"]
        assert len(toasts) == 1
        assert "bug" in toasts[0][1]

        # Ctrl+Up: legal — reorders.
        toasts.clear()
        await pilot.press("ctrl+up")
        await pilot.pause()
        assert [t.title for t in app.manager.queued()] == ["bug", "b", "a"]
        assert toasts == []


async def test_move_updates_sidebar_rows(app: DrCodeApp) -> None:
    """The visible QUEUE section follows the move (no snap-back)."""
    from drcode.tui.sidebar import TaskItem

    async with app.run_test(size=(220, 50)) as pilot:
        await _started(app, pilot)
        app.manager.create("a", "a")
        b = app.manager.create("b", "b")
        app.manager.create("c", "c")
        dep = app.manager.create("dep", "d")
        blocked = app.manager.create("blocked", "bl")
        blocked.depends_on = [dep.id]
        app.view_task(b.id)  # queued order: [a, b, c, dep, blocked]
        await pilot.pause()

        def rows() -> list[str]:
            return [
                w.task_id
                for w in app.sidebar.children
                if isinstance(w, TaskItem) and w.task_id is not None
            ]

        before = rows()
        await pilot.press("ctrl+up")  # b moves up the sidebar (away from run)
        await pilot.pause()
        after = rows()
        assert after != before
        # The moved task's new neighbours are exactly the swapped pair.
        i_before = before.index(b.id)
        assert after.index(b.id) == i_before - 1
