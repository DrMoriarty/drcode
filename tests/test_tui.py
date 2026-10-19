"""Smoke tests for the TUI (drcode.tui.app) using textual's pilot harness.

These run the real app headlessly: mount, task creation through the
new-task form, sidebar/status rendering, service events and the task
actions (cancel, done, hold, delete, kind cycle). No model is configured
in tests, so agent runs stop at the "Model is not configured" guard —
no network access happens.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import ModelRequest, ModelResponse
from pydantic_ai.messages import TextPart, UserPromptPart
from textual.widgets import ListView, TextArea

from drcode.agent import serialize_messages
from drcode.models import TaskKind, TaskStatus
from drcode.storage import Storage
from drcode.tui.app import DrCodeApp, TaskItem


@pytest.fixture
def app(project_root: Path) -> DrCodeApp:
    return DrCodeApp(project_root)


# -- mounting ------------------------------------------------------------------


async def test_app_mounts_with_new_task_form(app: DrCodeApp) -> None:
    async with app.run_test():
        # No tasks yet: the new-task form covers the chat view.
        assert app._new_task_mode
        assert app.query_one("#nt-input")
        assert app.query_one("#chat-input")
        assert app.query_one("#sidebar", ListView) is not None


async def test_app_mounts_sidebar_with_existing_tasks(project_root: Path) -> None:
    app = DrCodeApp(project_root)
    app.manager.create("existing task", "first prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.manager.all_tasks()) == 1
        assert app.viewed_id is not None


# -- new-task form ----------------------------------------------------------------


async def test_create_task_through_ui(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        # Skip the first-start setup dialog (no model configured in tests).
        await pilot.press("escape")
        await pilot.pause()
        nt_input = app.query_one("#nt-input", TextArea)
        nt_input.text = "Build the thing\nwith details"
        await pilot.press("ctrl+enter")
        await pilot.pause()

        tasks = app.manager.all_tasks()
        assert len(tasks) == 1
        task = tasks[0]
        assert task.kind is TaskKind.FEATURE
        assert [e.text for e in task.pending_input] == ["Build the thing\nwith details"]
        # No model configured: the run stops at the guard and the task
        # stays active with its prompt kept (it will be sent once a model
        # is available).
        assert task.status is TaskStatus.ACTIVE
        # The form is left after creation.
        assert not app._new_task_mode


async def test_new_task_form_kind_switch_and_cancel(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        # Skip the first-start setup dialog (no model configured in tests).
        await pilot.press("escape")
        await pilot.pause()
        nt_input = app.query_one("#nt-input", TextArea)
        nt_input.text = "a bug report"
        await pilot.press("tab")
        assert app.new_task_kind is TaskKind.BUG

        await pilot.press("tab")
        assert app.new_task_kind is TaskKind.DOCUMENTATION

        # esc cancels the form without creating anything.
        await pilot.press("escape")
        await pilot.pause()
        assert app.manager.all_tasks() == []
        assert app._new_task_mode is False


async def test_submit_empty_form_creates_nothing(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await pilot.press("ctrl+enter")
        await pilot.pause()
        assert app.manager.all_tasks() == []


# -- chat rendering ----------------------------------------------------------------


def _fat_transcript(turns: int) -> list[dict]:
    """Serialized user/assistant alternation — two blocks per turn."""
    messages: list[ModelRequest | ModelResponse] = []
    for i in range(turns):
        messages.append(ModelRequest(parts=[UserPromptPart(content=f"question {i}")]))
        messages.append(ModelResponse(parts=[TextPart(content=f"answer {i}")]))
    return serialize_messages(messages)


async def test_view_task_renders_pending_prompt(app: DrCodeApp) -> None:
    task = app.manager.create("shown task", "the original prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        # on_mount opened the first task automatically.
        assert app.viewed_id == task.id
        rendered = "\n".join(
            str(w.render()) for w in app.query_one("#chat-log").children
        )
        assert "the original prompt" in rendered


async def test_view_task_drains_fat_history_in_chunks(app: DrCodeApp) -> None:
    """A fat history mounts in chunks (input stays responsive); the drain
    must still end with the full timeline on screen and batching off."""
    task = app.manager.create("fat task", "prompt")
    task.messages = _fat_transcript(80)
    task.status = TaskStatus.COMPLETED
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        for _ in range(30):  # let the chunked drain run to completion
            await pilot.pause()
            if not app._chat_batching:
                break
        assert not app._chat_batching
        assert app._chat_batch == []
        log = app.query_one("#chat-log")
        # 2 blocks per turn + the still-pending original prompt.
        assert len(log.children) == 161
        rendered = "\n".join(str(w.render()) for w in log.children)
        assert "question 0" in rendered
        assert "answer 79" in rendered


async def test_view_switch_mid_drain_drops_stale_chunks(app: DrCodeApp) -> None:
    """Opening another task while a fat drain is still pending must not
    leak the old view's blocks into the new view's log."""
    fat = app.manager.create("fat task", "prompt")
    fat.messages = _fat_transcript(80)
    fat.status = TaskStatus.COMPLETED
    other = app.manager.create("other task", "other prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(fat.id)
        # Switch away before any drain chunk beyond the first could run.
        app.view_task(other.id)
        for _ in range(30):
            await pilot.pause()
            if not app._chat_batching:
                break
        assert not app._chat_batching
        rendered = "\n".join(
            str(w.render()) for w in app.query_one("#chat-log").children
        )
        assert "question 0" not in rendered
        assert "other prompt" in rendered


async def test_service_event_persisted_and_shown(app: DrCodeApp) -> None:
    task = app.manager.create("eventful", "prompt")
    async with app.run_test() as pilot:
        app._show_event(task, "something happened")
        await pilot.pause()
        assert task.events[-1].text == "something happened"
        # Persisted on disk as well.
        stored = Storage(app.project_root).load_all()
        assert stored[0].events[-1].text == "something happened"
        # And rendered in the chat (the task is open in the view).
        rendered = "\n".join(
            str(w.render()) for w in app.query_one("#chat-log").children
        )
        assert "something happened" in rendered


# -- task actions ------------------------------------------------------------------


async def test_action_cancel_task(app: DrCodeApp) -> None:
    task = app.manager.create("doomed", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()  # on_mount opened the task
        app.action_cancel_task()
        await pilot.pause()
        assert task.status is TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.pending_input == []


async def test_action_done_task_sends_to_testing(app: DrCodeApp) -> None:
    task = app.manager.create("feature", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_done_task()
        await pilot.pause()
        assert task.status is TaskStatus.TESTING


async def test_action_done_task_documentation_completes(app: DrCodeApp) -> None:
    task = app.manager.create("docs", "prompt", kind=TaskKind.DOCUMENTATION)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_done_task()
        await pilot.pause()
        assert task.status is TaskStatus.COMPLETED


async def test_action_toggle_hold_and_delete(app: DrCodeApp) -> None:
    task = app.manager.create("parked", "")  # no input: stays queued
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_hold_task()
        assert task.status is TaskStatus.HOLD

        app.action_resume_task()
        assert task.status is TaskStatus.QUEUED

        # Only held tasks can be deleted.
        app.action_delete_task()
        assert task.id in app.manager.tasks

        app.action_hold_task()
        app.action_delete_task()
        assert app.manager.all_tasks() == []
        assert not (app.project_root / ".drcode" / "tasks" / f"{task.id}.json").exists()


async def test_action_cycle_task_kind(app: DrCodeApp) -> None:
    task = app.manager.create("typed", "")  # no input: stays queued
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_cycle_task_kind()  # feature → bug
        assert task.kind is TaskKind.BUG
        app.action_cycle_task_kind()  # bug → documentation
        assert task.kind is TaskKind.DOCUMENTATION
        app.action_cycle_task_kind()  # documentation → feature
        assert task.kind is TaskKind.FEATURE


async def test_action_new_task_reopens_form(app: DrCodeApp) -> None:
    app.manager.create("t", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app._new_task_mode
        app.action_new_task()
        await pilot.pause()
        assert app._new_task_mode


# -- sidebar / status rendering -------------------------------------------------------


async def test_sidebar_shows_task_statuses(app: DrCodeApp) -> None:
    app.manager.create("visible", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        items = app.query("TaskItem")
        assert items
        app.action_hold_task()
        await pilot.pause()
        # Sidebar snapshot changed → the poller redraws it.
        app._poll_sidebar()
        items = app.query("TaskItem")
        assert items


# -- sidebar collapsible sections -------------------------------------------------


async def test_sidebar_header_enter_and_space_collapse_section(app: DrCodeApp) -> None:
    # No pending input: the tasks stay QUEUED (and in the QUEUE section)
    # for the whole test.
    app.manager.create("task one", "")
    app.manager.create("task two", "")
    async with app.run_test() as pilot:
        await pilot.pause()
        # Skip the first-start setup dialog (no model configured in tests).
        await pilot.press("escape")
        await pilot.pause()
        sidebar = app.sidebar
        sidebar.focus()
        await pilot.pause()
        await pilot.press("down")
        header = sidebar.highlighted_child
        assert isinstance(header, TaskItem)
        assert header.section is not None
        section = header.section
        row_count = len(sidebar.children)
        assert row_count > 1

        # Enter on the header collapses the category: only the header row
        # (with the "+" marker) remains.
        await pilot.press("enter")
        await pilot.pause()
        assert section in app._collapsed_sections
        assert len(sidebar.children) == 1
        header = sidebar.highlighted_child
        assert isinstance(header, TaskItem)
        assert header.text.lstrip().startswith("+")

        # Space expands it back ("-" marker, all rows visible again).
        await pilot.press("space")
        await pilot.pause()
        assert section not in app._collapsed_sections
        assert len(sidebar.children) == row_count
        header = sidebar.highlighted_child
        assert isinstance(header, TaskItem)
        assert header.text.lstrip().startswith("-")


async def test_sidebar_header_click_toggles_section(app: DrCodeApp) -> None:
    app.manager.create("task one", "")
    async with app.run_test() as pilot:
        await pilot.pause()
        # Skip the first-start setup dialog (no model configured in tests).
        await pilot.press("escape")
        await pilot.pause()
        sidebar = app.sidebar
        section = next(
            item.section
            for item in sidebar.children
            if isinstance(item, TaskItem) and item.section is not None
        )
        row_count = len(sidebar.children)
        header = next(
            item
            for item in sidebar.children
            if isinstance(item, TaskItem) and item.section
        )
        # Click the header row: collapses.
        await pilot.click(header)
        await pilot.pause()
        assert section in app._collapsed_sections
        assert len(sidebar.children) == 1
        # Click again: expands.
        header = sidebar.children[0]
        await pilot.click(header)
        await pilot.pause()
        assert section not in app._collapsed_sections
        assert len(sidebar.children) == row_count


async def test_sidebar_task_row_click_still_opens_task(app: DrCodeApp) -> None:
    task = app.manager.create("clickable", "")
    async with app.run_test() as pilot:
        await pilot.pause()
        sidebar = app.sidebar
        row = next(
            item
            for item in sidebar.children
            if isinstance(item, TaskItem) and item.task_id == task.id
        )
        await pilot.click(row)
        await pilot.pause()
        assert app.viewed_id == task.id
