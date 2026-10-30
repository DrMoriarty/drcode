"""The goals interface layer: sidebar GOALS section, the full-screen
GoalScreen and the app-level goal actions.

Covers: the sidebar's ◆ title n/m rows (top-level counters survive a
collapsed section, a milestone sums its subtree, the + add goal row),
the ◆<goal-id> link marker on task rows, the GoalScreen tree (filter,
Enter-expandable details, link marker of the viewed task) and its R/T
actions, the Ctrl+Y capture context (parent preset + task link), the
palette command set and the GoalStore ordering primitives (move/reparent).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Checkbox, Input, ListView, Static, TextArea

from drcode.models import GoalType, TaskKind, TaskStatus
from drcode.tui.app import DrCodeApp
from drcode.tui.command_providers import GoalCommandsProvider
from drcode.tui.goal_screen import GoalCaptureResult, GoalEditScreen, GoalScreen
from drcode.tui.sidebar import TaskItem


@pytest.fixture
def app(project_root: Path) -> DrCodeApp:
    return DrCodeApp(project_root)


async def _started(app: DrCodeApp, pilot) -> None:
    # Skip the first-start setup dialog (no model configured in tests).
    await pilot.press("escape")
    await pilot.pause()


def goal_rows(app: DrCodeApp) -> dict[str, TaskItem]:
    """Live goal rows of the sidebar by goal id (headers/add-row excluded)."""
    return {
        w.goal_id: w
        for w in app.sidebar.children
        if isinstance(w, TaskItem) and w.goal_id and not w.goal_add
    }


def task_row(app: DrCodeApp, task_id: str) -> TaskItem:
    return next(
        w
        for w in app.sidebar.children
        if isinstance(w, TaskItem) and w.task_id == task_id
    )


def add_goal_row(app: DrCodeApp) -> TaskItem:
    return next(
        w for w in app.sidebar.children if isinstance(w, TaskItem) and w.goal_add
    )


# -- sidebar GOALS section ------------------------------------------------------


async def test_sidebar_goals_section_rows_and_counters(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        milestone = app.manager.goals.create("Release 1.0")
        app.manager.goals.update(milestone.id, type="milestone")
        child = app.manager.goals.create("Async core", parent_id=milestone.id)
        idea = app.manager.goals.create("wild idea")
        # One research task linked to the sub-goal, done immediately.
        task = app.manager.create(
            "migrate", "x", kind=TaskKind.RESEARCH, goal_id=child.id
        )
        app.manager.mark_done(task.id)
        app.refresh_sidebar()
        rows = goal_rows(app)
        assert set(rows) == {milestone.id, child.id, idea.id}
        # The milestone sums its subtree: the sub-goal's task is done → 1/1.
        assert "1/1" in rows[milestone.id].text
        assert "1/1" in rows[child.id].text
        # An idea without tasks shows the "—" placeholder.
        assert "—" in rows[idea.id].text
        # The sub-goal row is indented under its parent.
        assert rows[child.id].text.startswith("  ")
        # The capture shortcut row is always there.
        assert add_goal_row(app) is not None


async def test_sidebar_goal_marker_on_task_row(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        goal = app.manager.goals.create("Async core")
        task = app.manager.create(
            "migrate", "x", kind=TaskKind.RESEARCH, goal_id=goal.id
        )
        orphan = app.manager.create("no goal", "y", kind=TaskKind.RESEARCH)
        app.refresh_sidebar()
        linked = task_row(app, task.id)
        assert "◆" in linked.text and goal.id in linked.text
        plain = task_row(app, orphan.id)
        assert "◆" not in plain.text


async def test_sidebar_collapsed_goals_keeps_top_level_counters(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        milestone = app.manager.goals.create("Release 1.0")
        child = app.manager.goals.create("Async core", parent_id=milestone.id)
        app.refresh_sidebar()
        app.toggle_sidebar_section("GOALS")
        await pilot.pause()  # the removed row's prune is an async round-trip
        rows = goal_rows(app)
        # Collapsed folds only the sub-goal tree away: the top-level row
        # (with its counter) and the header stay visible.
        assert milestone.id in rows
        assert child.id not in rows
        app.toggle_sidebar_section("GOALS")
        await pilot.pause()
        assert set(goal_rows(app)) == {milestone.id, child.id}


async def test_sidebar_marker_tracks_link_changes(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        goal = app.manager.goals.create("Async core")
        task = app.manager.create("migrate", "x", kind=TaskKind.RESEARCH)
        app.refresh_sidebar()
        assert "◆" not in task_row(app, task.id).text
        app.view_task(task.id)
        app.link_viewed_goal(goal.id)
        app.refresh_sidebar()
        assert goal.id in task_row(app, task.id).text
        app.unlink_viewed_goal()
        app.refresh_sidebar()
        assert "◆" not in task_row(app, task.id).text


# -- GoalScreen -------------------------------------------------------------------


def _open_screen(app: DrCodeApp) -> GoalScreen:
    app.action_open_goals()
    screen = app.screen
    assert isinstance(screen, GoalScreen)
    return screen


async def test_goal_screen_tree_and_details(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        parent = app.manager.goals.create(
            "Release 1.0", description="the release umbrella"
        )
        child = app.manager.goals.create("Async core", parent_id=parent.id)
        app.manager.goals.update(
            parent.id,
            notes="deep context the agent reads via read_goal only",
        )
        task = app.manager.create(
            "migrate", "x", kind=TaskKind.RESEARCH, goal_id=child.id
        )
        app.view_task(task.id)
        screen = _open_screen(app)
        await pilot.pause()
        listing = screen.query_one("#goal-tree", ListView)
        texts = [w.text for w in listing.children if isinstance(w, TaskItem)]
        # The tree shows both goals with counters and the viewed task's link.
        joined = "\n".join(texts)
        assert "Release 1.0" in joined and "Async core" in joined
        assert f"#{task.id}" in joined and "◆" in joined
        # Nothing expanded yet: the notes stay out of the rendered rows.
        assert "read_goal" not in joined
        # Enter on the first row (the parent) expands its details.
        listing.index = 0
        await pilot.press("enter")
        await pilot.pause()
        texts = [w.text for w in listing.children if isinstance(w, TaskItem)]
        joined = "\n".join(texts)
        assert "the release umbrella" in joined
        # The notes preview is UI-only and rendered expanded.
        assert any("notes:" in t for t in texts)


async def test_goal_screen_filter_cycle(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        active = app.manager.goals.create("live goal")
        done = app.manager.goals.create("old goal")
        app.manager.goals.update(done.id, status="done")
        archived = app.manager.goals.create("buried")
        app.manager.goals.update(archived.id, status="archived")
        screen = _open_screen(app)
        await pilot.pause()
        listing = screen.query_one("#goal-tree", ListView)

        def shown() -> set[str]:
            return {
                w.goal_id
                for w in listing.children
                if isinstance(w, TaskItem) and w.goal_id
            }

        # Default filter "active": only the live goal.
        assert shown() == {active.id}
        await pilot.press("f")  # active → ideas: every unclosed IDEA
        # (all three goals here were created with the default IDEA type).
        assert shown() == {active.id, done.id}
        await pilot.press("f")  # ideas → done
        assert shown() == {done.id}
        await pilot.press("f")  # done → archived
        assert shown() == {archived.id}
        await pilot.press("f")  # archived → all
        assert shown() == {active.id, done.id, archived.id}
        await pilot.press("f")  # all → active (cycle wraps)
        assert shown() == {active.id}


async def test_goal_screen_research_task(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        goal = app.manager.goals.create("Async core", description="the direction")
        screen = _open_screen(app)
        await pilot.pause()
        screen.action_research_task()
        await pilot.pause()
        # The screen closes; an ordinary research task is created with the
        # goal reference and the chat input holds the prompt stub.
        assert not isinstance(app.screen, GoalScreen)
        research = [
            t for t in app.manager.tasks.values() if t.kind is TaskKind.RESEARCH
        ]
        assert len(research) == 1
        assert research[0].goal_id == goal.id
        assert research[0].status is TaskStatus.QUEUED
        assert app.chat_input.text.startswith("Цель: Async core")
        assert "the direction" in app.chat_input.text
        assert app.chat_input.text.rstrip().endswith("Задание:")
        # Created WITHOUT pending input: the pipeline must not start it
        # before the user finishes and sends the «Задание:» text.
        assert not research[0].pending_input


async def test_goal_screen_task_from_goal(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        goal = app.manager.goals.create("Async core")
        screen = _open_screen(app)
        await pilot.pause()
        screen.action_task_from_goal()
        await pilot.pause()
        # The standard new-task form is open with the goal line preset.
        assert app._new_task_mode
        label = str(app.query_one("#nt-goal", Static).content)
        assert "Async core" in label
        app.new_task_input.load_text("migrate the storage layer")
        app._submit_new_form(hold=False)
        created = [
            t
            for t in app.manager.tasks.values()
            if t.title == "migrate the storage layer"
        ]
        assert len(created) == 1
        assert created[0].goal_id == goal.id
        # The link is consumed with the form.
        assert app._new_task_goal_id is None
        assert app.query_one("#nt-goal", Static).content == ""


async def test_goal_capture_with_parent_and_link(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        parent = app.manager.goals.create("Release 1.0")
        task = app.manager.create("migrate", "x", kind=TaskKind.RESEARCH)
        app.view_task(task.id)
        app.open_new_goal_dialog(parent_id=parent.id, link_task_id=task.id)
        # Simulate the dialog result (the form itself is covered by the
        # capture tests of the plain path).
        app._goal_captured(
            GoalCaptureResult(
                title="Async core",
                description="short",
                notes="",
                goal_type=GoalType.GOAL,
                link_task=True,
            )
        )
        await pilot.pause()
        goal = next(
            g for g in app.manager.goals.goals.values() if g.title == "Async core"
        )
        assert goal.parent_id == parent.id
        assert task.goal_id == goal.id
        assert goal.type.value == "goal"


async def test_goal_palette_commands(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        goal = app.manager.goals.create("Async core")
        task = app.manager.create("migrate", "x", kind=TaskKind.RESEARCH)
        app.view_task(task.id)
        commands = GoalCommandsProvider(app.screen)._commands()  # pyright: ignore[reportAttributeAccessIssue]
        names = [c.name for c in commands]
        assert any(name.startswith("Link goal:") for name in names)
        assert not any(name.startswith("Unlink goal") for name in names)
        assert "Create goal…" in names and "New goal" not in names
        app.link_viewed_goal(goal.id)
        commands = GoalCommandsProvider(app.screen)._commands()  # pyright: ignore[reportAttributeAccessIssue]
        names = [c.name for c in commands]
        assert any(name.startswith("Unlink goal") for name in names)
        assert not any(name.startswith("Link goal:") for name in names)
        # Fire the unlink command.
        next(c for c in commands if c.name.startswith("Unlink goal")).callback()
        assert task.goal_id is None


# -- the unified GoalEditScreen (create mode: the dialog itself) ------------------


def _find_goal(app: DrCodeApp, title: str):
    return next(g for g in app.manager.goals.goals.values() if g.title == title)


async def test_goal_create_dialog_links_viewed_task(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        task = app.manager.create("migrate", "x", kind=TaskKind.RESEARCH)
        app.view_task(task.id)
        app.open_new_goal_dialog(link_task_id=task.id)
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, GoalEditScreen)
        # The link checkbox exists over an open task and is pre-enabled.
        checkbox = screen.query_one("#goal-link-task", Checkbox)
        assert checkbox.value is True
        screen.query_one("#goal-title", Input).value = "Async core"
        screen.query_one("#goal-edit-desc", TextArea).text = "short"
        screen.query_one("#goal-edit-notes", TextArea).text = "n1"
        screen._save()
        await pilot.pause()
        goal = _find_goal(app, "Async core")
        assert task.goal_id == goal.id
        assert goal.description == "short"
        assert goal.notes == "n1"
        assert goal.type.value == "idea"  # the create-mode default


async def test_goal_create_dialog_link_can_be_declined(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        task = app.manager.create("migrate", "x", kind=TaskKind.RESEARCH)
        app.view_task(task.id)
        app.open_new_goal_dialog(link_task_id=task.id)
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, GoalEditScreen)
        screen.query_one("#goal-link-task", Checkbox).value = False
        screen.query_one("#goal-title", Input).value = "Loose goal"
        screen._save()
        await pilot.pause()
        assert _find_goal(app, "Loose goal") is not None
        assert task.goal_id is None
        # The plain Ctrl+Y path carries no task id: no checkbox at all.
        app.action_new_goal()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, GoalEditScreen)
        assert not screen.query("#goal-link-task")
        screen.action_cancel()
        await pilot.pause()


async def test_goal_create_dialog_requires_title(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        app.action_new_goal()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, GoalEditScreen)
        screen._save()
        await pilot.pause()
        # Still open: the empty title was rejected with the error line.
        assert isinstance(app.screen, GoalEditScreen)
        assert "required" in str(screen.query_one("#goal-error", Static).content)
        screen.action_cancel()


async def test_goal_screen_subgoal_capture(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        parent = app.manager.goals.create("Release 1.0")
        screen = _open_screen(app)
        await pilot.pause()
        screen.action_new_goal()
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, GoalEditScreen)
        dialog.query_one("#goal-title", Input).value = "Async core"
        dialog._save()
        await pilot.pause()
        child = _find_goal(app, "Async core")
        assert child.parent_id == parent.id


async def test_goal_edit_dialog_description_and_notes(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await _started(app, pilot)
        goal = app.manager.goals.create("Async core", description="old desc")
        app.manager.goals.update(goal.id, notes="old notes")
        screen = _open_screen(app)
        await pilot.pause()
        screen.action_edit_goal()
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, GoalEditScreen)
        # Edit mode: the title is a fixed line — no title input, no
        # type line, no link checkbox.
        assert not dialog.query("#goal-title")
        assert not dialog.query("#goal-type-line")
        assert not dialog.query("#goal-link-task")
        dialog.query_one("#goal-edit-desc", TextArea).text = "new desc"
        dialog.query_one("#goal-edit-notes", TextArea).text = "new notes"
        dialog._save()
        await pilot.pause()
        refreshed = app.manager.goals.get(goal.id)
        assert refreshed.title == "Async core"
        assert refreshed.description == "new desc"
        assert refreshed.notes == "new notes"


# -- GoalStore ordering (the GoalScreen's Ctrl+Up/Down and M) ---------------------


def test_goal_move_swaps_sibling_order(project_root: Path) -> None:
    from drcode.task_manager import TaskManager

    manager = TaskManager(project_root)
    a = manager.goals.create("a")
    b = manager.goals.create("b")
    c = manager.goals.create("c")
    manager.goals.move(b.id, up=True)
    assert [g.id for g in manager.goals.top_level()] == [b.id, a.id, c.id]
    manager.goals.move(a.id, up=False)
    assert [g.id for g in manager.goals.top_level()] == [b.id, c.id, a.id]
    # Moving past the end is a no-op.
    manager.goals.move(a.id, up=False)
    assert [g.id for g in manager.goals.top_level()] == [b.id, c.id, a.id]


def test_goal_reparent_validates_and_orders(project_root: Path) -> None:
    from drcode.task_manager import TaskManager

    manager = TaskManager(project_root)
    a = manager.goals.create("a")
    b = manager.goals.create("b")
    child = manager.goals.create("child", parent_id=a.id)
    # Moving under an existing parent re-stamps the order at the end of
    # the new sibling group.
    manager.goals.reparent(child.id, b.id)
    assert child.parent_id == b.id
    assert [g.id for g in manager.goals.display_children(b.id)] == [child.id]
    # A goal cannot move under its own descendant (tree, not cycle).
    manager.goals.reparent(b.id, a.id)
    with pytest.raises(ValueError):
        manager.goals.reparent(a.id, b.id)
    with pytest.raises(ValueError):
        manager.goals.reparent(child.id, "nope1234")
    # Make top-level works.
    manager.goals.reparent(child.id, None)
    assert child.parent_id is None
