"""Incremental sidebar refresh (chat_view.refresh_sidebar).

The sidebar reconciles mounted TaskItems instead of clear()+append of the
whole list: unchanged rows keep their widget instances, changed rows are
updated in place (label text / viewed class), new rows are mounted and
stale rows removed. These tests pin the widget-identity contract — the
whole point of the change — plus the ListView cursor behaviour and the
collapse/expand transitions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from textual.widgets import ListView

from drcode.models import TaskKind, TaskStatus, VerificationStatus
from drcode.tui.app import DrCodeApp
from drcode.tui.sidebar import TaskItem


@pytest.fixture
def app(project_root: Path) -> DrCodeApp:
    return DrCodeApp(project_root)


def task_rows(app: DrCodeApp) -> list[TaskItem]:
    """Live task rows (in DOM order), headers excluded."""
    return [
        w
        for w in app.sidebar.children
        if isinstance(w, TaskItem) and w.task_id is not None
    ]


def row_of(app: DrCodeApp, task_id: str) -> TaskItem:
    return next(row for row in task_rows(app) if row.task_id == task_id)


def header_of(app: DrCodeApp, section: str) -> TaskItem:
    return next(
        w
        for w in app.sidebar.children
        if isinstance(w, TaskItem) and w.section == section
    )


@pytest.fixture
def two_tasks(app: DrCodeApp):
    a = app.manager.create("first", "prompt a")
    b = app.manager.create("second", "prompt b")
    return a, b


async def _started(app: DrCodeApp, pilot) -> None:
    # Skip the first-start setup dialog (no model configured in tests).
    await pilot.press("escape")
    await pilot.pause()


async def test_status_change_reuses_row_widget(app: DrCodeApp, two_tasks) -> None:
    a, _b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        row = row_of(app, a.id)
        assert "⏳" in row.text

        a.status = TaskStatus.ACTIVE
        a.touch()
        app.manager.storage.save(a)
        app.refresh_sidebar()
        await pilot.pause()

        # Same widget instance, new label: a status flip must not remount.
        assert [r for r in task_rows(app) if r.task_id == a.id] == [row]
        assert "▶" in row.text


async def test_verification_change_updates_row_in_place(
    app: DrCodeApp, two_tasks
) -> None:
    a, _b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        row = row_of(app, a.id)
        assert "✅" not in row.text

        a.verification = VerificationStatus.PASSED
        a.touch()
        app.manager.storage.save(a)
        app.refresh_sidebar()
        await pilot.pause()

        assert row_of(app, a.id) is row
        assert "✅" in row.text


async def test_viewed_marker_moves_between_rows(app: DrCodeApp, two_tasks) -> None:
    a, b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        # Startup opened the top QUEUE row (newest first → b).
        assert app.viewed_id == b.id
        row_a, row_b = row_of(app, a.id), row_of(app, b.id)
        assert row_b.has_class("viewed") and not row_a.has_class("viewed")

        app.view_task(a.id)  # refreshes the sidebar itself
        await pilot.pause()

        # Both rows survive as the same widgets; only marker/class moved.
        assert row_of(app, a.id) is row_a
        assert row_of(app, b.id) is row_b
        assert row_a.has_class("viewed") and not row_b.has_class("viewed")
        assert "●" in row_a.text and "●" not in row_b.text


async def test_new_task_appends_row_old_rows_reused(app: DrCodeApp, two_tasks) -> None:
    a, b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        row_a, row_b = row_of(app, a.id), row_of(app, b.id)

        c = app.manager.create("buggy", "prompt c", kind=TaskKind.BUG)
        app.refresh_sidebar()
        await pilot.pause()

        # The QUEUE section flows downward into ACTIVE (the next task to
        # be promoted is the bottom row), so a fresh bug joins at the
        # section's end; the old rows keep identity.
        rows = task_rows(app)
        assert [r.task_id for r in rows] == [b.id, a.id, c.id]
        assert row_of(app, a.id) is row_a
        assert row_of(app, b.id) is row_b


async def test_promoted_tasks_land_below_their_section_header(
    app: DrCodeApp, two_tasks
) -> None:
    """Two tasks promoted at once must appear UNDER the "ACTIVE (n)" header.

    The QUEUE section displays newest-first (b above a); ACTIVE keeps the
    created order (a above b). The reconcile used to anchor the freshly
    mounted header and the reordered rows against a stale DOM snapshot,
    which left the header BELOW both task rows."""
    a, b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        assert [r.task_id for r in task_rows(app)] == [b.id, a.id]

        for t in (a, b):
            t.status = TaskStatus.ACTIVE
            t.touch()
            app.manager.storage.save(t)
        app.refresh_sidebar()
        await pilot.pause()

        children = [
            w.section if w.task_id is None else w.task_id
            for w in app.sidebar.children
            if isinstance(w, TaskItem)
        ]
        assert children == ["ACTIVE", a.id, b.id]


async def test_removed_task_drops_its_row(app: DrCodeApp, two_tasks) -> None:
    a, b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        row_a = row_of(app, a.id)

        app.manager.set_hold(b.id, on=True)
        app.manager.delete(b.id)
        app.refresh_sidebar()
        await pilot.pause()

        rows = task_rows(app)
        assert [r.task_id for r in rows] == [a.id]
        assert row_of(app, a.id) is row_a
        assert ("task", b.id) not in app._sidebar_items


async def test_collapse_reuses_header_and_restores_rows(
    app: DrCodeApp, two_tasks
) -> None:
    _a, _b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        header = header_of(app, "QUEUE")
        rows_before = task_rows(app)
        assert header.text.lstrip().startswith("-")

        app.toggle_sidebar_section("QUEUE")
        await pilot.pause()

        assert header_of(app, "QUEUE") is header
        assert header.text.lstrip().startswith("+")
        assert task_rows(app) == []

        app.toggle_sidebar_section("QUEUE")
        await pilot.pause()

        assert header_of(app, "QUEUE") is header
        assert header.text.lstrip().startswith("-")
        assert [r.task_id for r in task_rows(app)] == [r.task_id for r in rows_before]


async def test_cursor_follows_row_across_refresh(app: DrCodeApp, two_tasks) -> None:
    _a, b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        sidebar = app.sidebar
        sidebar.focus()
        await pilot.pause()
        row_b = row_of(app, b.id)
        sidebar.index = sidebar._nodes.index(row_b)
        await pilot.pause()
        assert sidebar.highlighted_child is row_b

        # A refresh triggered by another task's state change must leave
        # the cursor on b's row: a new task inserting above it shifts
        # every positional index, and the incremental refresh re-derives
        # the index from the widget instead of leaving it stale.
        app.manager.create("third", "prompt c")
        app.refresh_sidebar()
        await pilot.pause()

        assert sidebar.highlighted_child is row_b


async def test_cursor_reset_when_row_leaves_sidebar(app: DrCodeApp, two_tasks) -> None:
    _a, b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        sidebar = app.sidebar
        sidebar.focus()
        await pilot.pause()
        row_b = row_of(app, b.id)
        sidebar.index = sidebar._nodes.index(row_b)
        await pilot.pause()

        app.manager.set_hold(b.id, on=True)
        app.manager.delete(b.id)
        app.refresh_sidebar()
        await pilot.pause()

        # Old behaviour: a rebuild reset the cursor to None. The
        # incremental path matches it when the highlighted row is gone.
        assert sidebar.index is None


async def test_new_task_form_clears_stuck_cursor(app: DrCodeApp, two_tasks) -> None:
    """Ctrl+N / creating a task must not leave the previously viewed row
    lit in the ListView cursor colour.

    The cursor is parked on the clicked row (viewed A), the form clears
    viewed_id — the row then lost only its .viewed class and kept the
    bright -highlight cursor background, both while composing and after
    the new task B was created and viewed."""
    a, _b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        sidebar = app.sidebar
        sidebar.focus()
        await pilot.pause()
        # Click on A's row: cursor and viewed task both land on it.
        app.view_task(a.id)
        app._refocus_sidebar_task(a.id)
        await pilot.pause()
        row_a = row_of(app, a.id)
        assert sidebar.highlighted_child is row_a

        # Ctrl+N: viewed_id drops to None, focus moves to the form — the
        # cursor must go with it, not leave row A lit.
        app.action_new_task()
        await pilot.pause()
        assert app.viewed_id is None
        assert sidebar.highlighted_child is None
        assert not any(
            w.highlighted for w in sidebar.children if isinstance(w, TaskItem)
        )

        # Ctrl+Enter: task B created and viewed — row A stays unlit, only
        # row B carries the viewed marker.
        b = app.manager.create("third", "prompt c")
        app.view_task(b.id)
        app.refresh_sidebar()
        await pilot.pause()
        assert row_of(app, b.id).has_class("viewed")
        assert not row_a.highlighted
        assert sidebar.highlighted_child is None


async def test_refocus_sidebar_task_contract(app: DrCodeApp, two_tasks) -> None:
    a, _b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        sidebar = app.sidebar
        sidebar.focus()
        await pilot.pause()
        app._refocus_sidebar_task(a.id)
        await pilot.pause()
        assert sidebar.highlighted_child is row_of(app, a.id)


async def test_stale_highlight_cleared_after_prune_shifts_rows(
    app: DrCodeApp, two_tasks
) -> None:
    """An async prune must not leave the painted highlight stuck on a row.

    ListView clears the old highlight by *position* (_nodes[old_index]),
    not by widget. When a removed row's prune lands between two refreshes,
    the node list shrinks under the unchanged index and the widget the
    ListView un-highlights is not the one that was painted — the painted
    row keeps its -highlight class forever (a second, stuck highlight that
    matches neither the cursor nor the viewed task)."""
    a, b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        sidebar = app.sidebar
        sidebar.focus()
        await pilot.pause()
        row_a = row_of(app, a.id)
        sidebar.index = sidebar._nodes.index(row_a)
        await pilot.pause()
        assert row_a.highlighted

        # Deleting b pops its row from the tracking dict and issues an
        # async remove(): the prune completes after this refresh, shifting
        # row_a's position under the unchanged index.
        app.manager.set_hold(b.id, on=True)
        app.manager.delete(b.id)
        app.refresh_sidebar()
        await pilot.pause()

        # Desynced state (pre-fix): row_a is still painted highlighted
        # while the index points past the end of the node list.
        assert row_a.highlighted

        # The next refresh must repair the cursor instead of leaving the
        # stale -highlight in place.
        app.refresh_sidebar()
        await pilot.pause()

        assert sidebar.highlighted_child is row_a
        assert row_a.highlighted
        painted = [
            w for w in sidebar.children if isinstance(w, TaskItem) and w.highlighted
        ]
        assert painted == [row_a]


async def test_extra_highlight_reactives_are_cleared_on_refresh(
    app: DrCodeApp, two_tasks
) -> None:
    """refresh_sidebar syncs the highlight reactives to the cursor: only
    the row the ListView index points at may stay lit."""
    a, b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        sidebar = app.sidebar
        sidebar.focus()
        await pilot.pause()
        row_a, row_b = row_of(app, a.id), row_of(app, b.id)
        # Simulate the stuck state: painted highlights on rows the index
        # does not point at (what the prune/move desync used to produce).
        row_a.highlighted = True
        row_b.highlighted = True
        await pilot.pause()

        app.refresh_sidebar()
        await pilot.pause()

        cursor = sidebar.highlighted_child
        assert cursor is not None
        painted = [
            w for w in sidebar.children if isinstance(w, TaskItem) and w.highlighted
        ]
        assert painted == [cursor]


async def test_poll_sidebar_uses_reconcile_and_updates_snapshot(
    app: DrCodeApp, two_tasks
) -> None:
    a, _b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        row = row_of(app, a.id)
        a.status = TaskStatus.ACTIVE
        a.touch()
        app.manager.storage.save(a)
        app._poll_sidebar()
        await pilot.pause()

        assert row_of(app, a.id) is row
        assert "▶" in row.text
        # Snapshot updated: the next poll tick does not re-render.
        assert app._sidebar_snapshot == app._sidebar_fingerprint()


async def test_second_refresh_same_tick_is_stable(app: DrCodeApp, two_tasks) -> None:
    """Two refreshes in one tick (direct calls do not coalesce) must not
    duplicate or lose rows while the previous prune is still pending."""
    a, b = two_tasks
    async with app.run_test() as pilot:
        await _started(app, pilot)
        c = app.manager.create("third", "prompt c")
        app.refresh_sidebar()
        app.refresh_sidebar()
        await pilot.pause()

        rows = task_rows(app)
        assert [r.task_id for r in rows] == [c.id, b.id, a.id]
        # Every sidebar child is a tracked TaskItem: no duplicate or
        # half-pruned leftovers (header + 3 task rows).
        assert len(app.sidebar.children) == 4
        assert all(isinstance(w, TaskItem) for w in app.sidebar.children)
        assert len(app._sidebar_items) == 4
        assert isinstance(app.sidebar, ListView)


# -- post-delete selection ---------------------------------------------------------------
#
# Deleting a task must open the deleted row's NEIGHBOUR in the sidebar as
# displayed (_sidebar_sections flattened) — first the row that followed,
# then the one that preceded — never all_tasks()[0] (creation order, i.e.
# the oldest archived task).


def _complete_with_stamps(app: DrCodeApp, *names: str) -> list:
    """Create tasks and stamp distinct updated_at values, then mark them
    COMPLETED. The COMPLETED section renders newest-first, so the sidebar
    shows them in reverse creation order deterministically."""
    tasks = [app.manager.create(name, "prompt") for name in names]
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    for i, t in enumerate(tasks):
        t.updated_at = base + timedelta(seconds=i)
        t.status = TaskStatus.COMPLETED
        app.manager.storage.save(t)
    return tasks


async def test_delete_completed_selects_following_sidebar_row(
    app: DrCodeApp,
) -> None:
    """Deleting a middle COMPLETED row opens the row BELOW it (the next
    one the user sees), not the oldest archived task."""
    old, mid, new = _complete_with_stamps(app, "old", "mid", "new")
    async with app.run_test() as pilot:
        await _started(app, pilot)
        # COMPLETED renders newest-first: new, mid, old.
        assert [r.task_id for r in task_rows(app)] == [new.id, mid.id, old.id]

        app.view_task(mid.id)
        await pilot.pause()
        app.action_delete_task()
        await pilot.pause()

        assert mid.id not in app.manager.tasks
        assert app.viewed_id == old.id
        assert row_of(app, old.id).has_class("viewed")


async def test_delete_completed_top_row_selects_the_next_one(
    app: DrCodeApp,
) -> None:
    _old, mid, new = _complete_with_stamps(app, "old", "mid", "new")
    async with app.run_test() as pilot:
        await _started(app, pilot)
        app.view_task(new.id)  # top row
        await pilot.pause()
        app.action_delete_task()
        await pilot.pause()

        assert app.viewed_id == mid.id


async def test_delete_last_sidebar_row_selects_preceding(
    app: DrCodeApp,
) -> None:
    """The deleted row was the last one displayed: fall back to the row
    that preceded it."""
    old, mid, _new = _complete_with_stamps(app, "old", "mid", "new")
    async with app.run_test() as pilot:
        await _started(app, pilot)
        app.view_task(old.id)  # bottom row
        await pilot.pause()
        app.action_delete_task()
        await pilot.pause()

        assert app.viewed_id == mid.id


async def test_delete_held_task_selects_sidebar_neighbor(
    app: DrCodeApp,
) -> None:
    """HOLD deletion picks the neighbour within the ON HOLD section (its
    creation order), with the QUEUE rows above it untouched."""
    a = app.manager.create("queued", "prompt")
    held_first = app.manager.create("held 1", "prompt")
    held_second = app.manager.create("held 2", "prompt")
    for t in (held_first, held_second):
        app.manager.set_hold(t.id, on=True)
    async with app.run_test() as pilot:
        await _started(app, pilot)
        # QUEUE first (a), then ON HOLD in creation order.
        assert [r.task_id for r in task_rows(app)] == [
            a.id,
            held_first.id,
            held_second.id,
        ]

        app.view_task(held_first.id)
        await pilot.pause()
        app.action_delete_task()
        await pilot.pause()

        # The row that FOLLOWED held_first in the sidebar is held_second.
        assert app.viewed_id == held_second.id
        assert row_of(app, held_second.id).has_class("viewed")

        app.view_task(held_second.id)
        await pilot.pause()
        app.action_delete_task()
        await pilot.pause()

        # Last row: fall back to the preceding QUEUE row.
        assert app.viewed_id == a.id
