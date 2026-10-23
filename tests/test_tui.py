"""Smoke tests for the TUI (drcode.tui.app) using textual's pilot harness.

These run the real app headlessly: mount, task creation through the
new-task form, sidebar/status rendering, service events and the task
actions (cancel, done, hold, delete, kind cycle). No model is configured
in tests, so agent runs stop at the "Model is not configured" guard —
no network access happens.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import ModelRequest, ModelResponse
from pydantic_ai.messages import TextPart, UserPromptPart
from textual.command import Command, CommandList
from textual.geometry import Offset
from textual.widgets import ListView, Static, TextArea
from textual.widgets._list_item import ListItem
from textual.widgets.text_area import Selection

from drcode.agent import serialize_messages
from drcode.models import TaskEvent, TaskKind, TaskStatus
from drcode.providers import AgentRole
from drcode.storage import Storage
from drcode.tui.app import DrCodeApp
from drcode.tui.command_providers import (
    SettingsPalette,
    role_model_options_provider,
)
from drcode.tui.setup_dialog import ModelSetupDialog
from drcode.tui.sidebar import TaskItem


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
        assert app.new_task_kind is TaskKind.RESEARCH

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


async def test_drain_survives_dropped_call_after_refresh(app: DrCodeApp) -> None:
    """Regression (task #79099479): a dropped `call_after_refresh` schedule
    (it returns False when the message pump is closed/closing and never
    queues the InvokeLater) used to kill the drain chain after the first
    chunk — the history stayed truncated at the chunk boundary with
    batching stuck on, swallowing every later live write (the whole tester
    round). The fallback schedule must still drain everything."""
    task = app.manager.create("fat task", "prompt")
    task.messages = _fat_transcript(80)
    task.status = TaskStatus.COMPLETED
    async with app.run_test() as pilot:
        await pilot.pause()
        calls = {"n": 0}
        original = app.call_after_refresh

        def drop_first_schedule(callback, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return False  # pump "closed": InvokeLater never queued
            return original(callback, *args, **kwargs)

        app.call_after_refresh = drop_first_schedule  # type: ignore[method-assign]
        app.view_task(task.id)
        for _ in range(60):
            await pilot.pause()
            if not app._chat_batching:
                break
        assert calls["n"] >= 1  # the drop actually happened
        assert not app._chat_batching
        assert app._chat_batch == []
        log = app.query_one("#chat-log")
        assert len(log.children) == 161
        rendered = "\n".join(str(w.render()) for w in log.children)
        assert "answer 79" in rendered


async def test_watchdog_resumes_dead_drain(app: DrCodeApp) -> None:
    """A drain chain that died outright (schedule dropped AND starved) is
    resurrected by the 1 s poll watchdog: pending blocks still mount and
    batching turns off."""
    task = app.manager.create("fat task", "prompt")
    task.messages = _fat_transcript(80)
    task.status = TaskStatus.COMPLETED
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        for _ in range(30):
            await pilot.pause()
            if not app._chat_batching:
                break
        assert not app._chat_batching
        # Simulate a new fat batch whose chain died right after the first
        # chunk: batching on, chunks pending, no progress for ages.
        app._chat_batching = True
        app._chat_batch_epoch = app._chat_epoch
        app._chat_batch = [
            Static(f"late block {i}", classes="chat-block") for i in range(40)
        ]
        app._chat_drain_at = time.monotonic() - 10
        app._poll_chat_drain()  # one 1 s poll tick
        for _ in range(60):
            await pilot.pause()
            if not app._chat_batching:
                break
        assert not app._chat_batching
        assert app._chat_batch == []
        rendered = "\n".join(
            str(w.render()) for w in app.query_one("#chat-log").children
        )
        assert "late block 39" in rendered


async def test_stale_drain_chain_leaves_new_view_alone(app: DrCodeApp) -> None:
    """A stale chain (scheduled under an older _chat_epoch, detected via
    the epoch captured per callback) must neither mount its chunks into
    the new view's log nor wipe the newer flush's state."""
    fat = app.manager.create("fat task", "prompt")
    fat.messages = _fat_transcript(80)
    fat.status = TaskStatus.COMPLETED
    other = app.manager.create("other task", "other prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(fat.id)
        stale_epoch = app._chat_epoch
        # Switch away before any drain chunk beyond the first could run:
        # the fat view's chain is now stale, the other view owns the log.
        app.view_task(other.id)
        app._drain_chat_batch(stale_epoch)  # the stale chain's callback fires
        await pilot.pause()  # let deferred DOM removal / queued callbacks settle
        # The stale callback neither mounted fat-view chunks…
        rendered = "\n".join(
            str(w.render()) for w in app.query_one("#chat-log").children
        )
        assert "question 0" not in rendered
        assert "other prompt" in rendered
        # …nor touched the new view's (already settled) drain state.
        assert not app._chat_batching
        assert app._chat_batch == []


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


async def test_task_env_event_from_worker_thread_draws_live(app: DrCodeApp) -> None:
    """A worktree event logged from a worker thread (not the app thread)
    must be drawn live — one appended line via call_from_thread — instead
    of waiting for the next full view_task() rebuild (_on_app_thread bug:
    _thread_id was never assigned, the live branch was dead)."""
    import asyncio

    task = app.manager.create("isolated", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        results: list[bool] = []

        def _worker() -> None:
            results.append(app._on_app_thread())
            app._task_env_event(task.id, "worktree clean, nothing to snapshot")

        # A plain thread.join() here would block the event loop and
        # deadlock the worker's call_from_thread — await an executor
        # thread instead so the loop keeps serving the hop.
        await asyncio.get_running_loop().run_in_executor(None, _worker)
        await pilot.pause()
        assert results == [False]  # the worker really is off the app thread
        assert app._on_app_thread()  # ...while the app thread is recognized
        rendered = "\n".join(
            str(w.render()) for w in app.query_one("#chat-log").children
        )
        assert "worktree clean, nothing to snapshot" in rendered


async def test_on_app_thread_true_on_app_thread(app: DrCodeApp) -> None:
    async with app.run_test():
        assert app._on_app_thread()


async def test_view_task_same_task_delta_appends_without_wipe(app: DrCodeApp) -> None:
    """Re-opening the SAME task appends only the new timeline tail — the
    already-mounted blocks keep their identity (no wipe-and-remount, no
    multi-second flash on fat histories)."""
    task = app.manager.create("delta task", "prompt")
    task.status = TaskStatus.COMPLETED
    task.pending_input.clear()
    task.messages = _fat_transcript(3)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        log = app.query_one("#chat-log")
        before = [id(w) for w in log.children]
        # A new service event lands in task.events without a live draw.
        task.events.append(TaskEvent(text="snapshot committed"))
        app.view_task(task.id)
        await pilot.pause()
        after = [id(w) for w in log.children]
        assert after[: len(before)] == before  # prefix kept, not remounted
        assert len(after) == len(before) + 1
        assert "snapshot committed" in str(log.children[-1].render())


async def test_view_task_delta_skips_rebuild_when_nothing_new(app: DrCodeApp) -> None:
    task = app.manager.create("quiet task", "prompt")
    task.status = TaskStatus.COMPLETED
    task.pending_input.clear()
    task.messages = _fat_transcript(3)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        log = app.query_one("#chat-log")
        before = [id(w) for w in log.children]
        assert app.view_task(task.id) is None
        assert [id(w) for w in log.children] == before


async def test_view_task_delta_falls_back_to_full_rebuild(app: DrCodeApp) -> None:
    """Delta state is fragile by design: a live mount (streaming block,
    event line) or a rewritten (compacted) transcript must force the full
    wipe-and-remount path again."""
    task = app.manager.create("live task", "prompt")
    task.status = TaskStatus.COMPLETED
    task.pending_input.clear()
    task.messages = _fat_transcript(3)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        log = app.query_one("#chat-log")
        before = [id(w) for w in log.children]

        # A live block mounted outside the timeline invalidates the delta.
        app._write_chat("[dim]live block[/]")
        task.events.append(TaskEvent(text="later event"))
        app.view_task(task.id)
        await pilot.pause()
        after = [id(w) for w in log.children]
        assert after != before  # full rebuild: blocks were remounted
        rendered = "\n".join(str(w.render()) for w in log.children)
        assert "later event" in rendered

        # A shrunken (compacted) transcript must rebuild too, never append
        # onto a longer rendered prefix.
        before = [id(w) for w in log.children]
        task.messages = _fat_transcript(2)
        app.view_task(task.id)
        await pilot.pause()
        after = [id(w) for w in log.children]
        assert after != before
        rendered = "\n".join(str(w.render()) for w in log.children)
        assert "question 1" in rendered


async def test_cancel_without_pending_skips_full_rebuild(app: DrCodeApp) -> None:
    """Ctrl+X draws its event line live; without queued input no full
    chat rebuild may follow (the mounted blocks keep their identity)."""
    task = app.manager.create("cancelled", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()  # on_mount opened the task
        task.pending_input.clear()  # no queued comment: rebuild must be skipped
        app.view_task(task.id)
        await pilot.pause()
        log = app.query_one("#chat-log")
        before = [id(w) for w in log.children]
        app.action_cancel_task()
        await pilot.pause()
        assert task.status is TaskStatus.COMPLETED
        after = [id(w) for w in log.children]
        assert after[: len(before)] == before
        assert len(after) == len(before) + 1  # only the event line appended
        assert "cancelled by the user" in str(log.children[-1].render()).lower()


async def test_cancel_with_pending_input_rebuilds(app: DrCodeApp) -> None:
    """A queued comment leaves pending_input with the cancel: its rendered
    "user-pending" block must come off the screen (full re-render)."""
    task = app.manager.create("cancelled", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        task.status = TaskStatus.HOLD
        app.manager.send(task.id, "a queued comment")
        app.view_task(task.id)
        await pilot.pause()
        log = app.query_one("#chat-log")
        assert "a queued comment" in "\n".join(str(w.render()) for w in log.children)
        before_count = len(log.children)
        app.action_cancel_task()
        await pilot.pause()
        rendered = "\n".join(str(w.render()) for w in log.children)
        assert "a queued comment" not in rendered
        assert len(log.children) < before_count + 2


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


async def test_task_action_keys_work_while_chat_input_focused(app: DrCodeApp) -> None:
    """ctrl+x / ctrl+d must reach the app actions even when the chat input
    (a TextArea subclass) is focused: its built-in `cut`/`delete_right`
    bindings must not shadow the app-level task actions."""
    async with app.run_test() as pilot:
        # Skip the first-start setup dialog (no model configured in tests).
        await pilot.press("escape")
        await pilot.pause()
        nt_input = app.query_one("#nt-input", TextArea)
        nt_input.text = "active task"
        await pilot.press("ctrl+enter")
        await pilot.pause()
        task = app.manager.all_tasks()[0]
        # No model configured: the run stops at the guard and the task
        # stays active with its prompt kept.
        assert task.status is TaskStatus.ACTIVE

        app.chat_input.focus()
        await pilot.pause()
        assert app.focused is app.chat_input

        # The footer resolves both keys to the app actions, not to the
        # TextArea's hidden `cut`/`delete_right` bindings.
        bindings = app.screen.active_bindings
        assert bindings["ctrl+x"].binding.action == "app.cancel_task"
        assert bindings["ctrl+d"].binding.action == "app.done_task"

        await pilot.press("ctrl+x")
        await pilot.pause()
        assert task.status is TaskStatus.COMPLETED
        assert app.chat_input.text == ""  # draft was not cut into clipboard


async def test_done_key_while_chat_input_focused_sends_to_testing(
    app: DrCodeApp,
) -> None:
    task = app.manager.create("feature", "prompt")
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
        app.view_task(task.id)
        app.chat_input.focus()
        await pilot.pause()
        await pilot.press("ctrl+d")
        await pilot.pause()
        assert task.status is TaskStatus.TESTING


async def test_done_task_sends_to_testing(app: DrCodeApp) -> None:
    task = app.manager.create("feature", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_done_task()
        await pilot.pause()
        assert task.status is TaskStatus.TESTING


async def test_action_done_task_research_completes(app: DrCodeApp) -> None:
    task = app.manager.create("docs", "prompt", kind=TaskKind.RESEARCH)
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
        app.action_cycle_task_kind()  # bug → research
        assert task.kind is TaskKind.RESEARCH
        app.action_cycle_task_kind()  # research → feature
        assert task.kind is TaskKind.FEATURE


async def test_action_new_task_reopens_form(app: DrCodeApp) -> None:
    app.manager.create("t", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app._new_task_mode
        app.action_new_task()
        await pilot.pause()
        assert app._new_task_mode


# -- create on hold / edit held task ----------------------------------------------


async def test_create_task_on_hold_through_ui(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the first-start setup dialog
        await pilot.pause()
        nt_input = app.query_one("#nt-input", TextArea)
        nt_input.text = "parked work\nwith details"
        await pilot.press("ctrl+h")
        await pilot.pause()

        tasks = app.manager.all_tasks()
        assert len(tasks) == 1
        task = tasks[0]
        assert task.status is TaskStatus.HOLD
        assert [e.text for e in task.pending_input] == ["parked work\nwith details"]
        assert not app._new_task_mode
        # The created task is opened in the chat pane.
        assert app.viewed_id == task.id


async def test_edit_held_task_merges_and_queues(app: DrCodeApp) -> None:
    task = app.manager.create("held", "original condition")
    app.manager.send(task.id, "extra comment")
    app.manager.set_hold(task.id, on=True)
    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the first-start setup dialog
        await pilot.pause()
        # on_mount opened the only task (the held one).
        assert app.viewed_id == task.id
        app.action_edit_task()
        await pilot.pause()
        assert app._new_task_mode
        assert app._editing_task_id == task.id
        # The condition and the comment are merged into the editor.
        assert app.new_task_input.text == "original condition\n\nextra comment"
        assert app.new_task_kind is task.kind

        # The user rewrites everything and switches the type, then queues.
        app.new_task_input.text = "rewritten condition"
        await pilot.press("tab")  # feature → bug
        assert app.new_task_kind is TaskKind.BUG
        await pilot.press("ctrl+enter")
        await pilot.pause()

        assert not app._new_task_mode
        # Out of hold and into the pipeline (with no model configured the
        # promotion leaves it active with the prompt kept).
        assert task.status is not TaskStatus.HOLD
        assert task.kind is TaskKind.BUG
        # The separate comment is gone — folded into the merged condition.
        assert [e.text for e in task.pending_input] == ["rewritten condition"]
        assert task.title == "rewritten condition"


async def test_edit_held_task_save_back_on_hold(app: DrCodeApp) -> None:
    task = app.manager.create("held", "condition")
    app.manager.set_hold(task.id, on=True)
    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the first-start setup dialog
        await pilot.pause()
        app.action_edit_task()
        await pilot.pause()
        app.new_task_input.text = "edited condition"
        await pilot.press("ctrl+h")
        await pilot.pause()

        assert not app._new_task_mode
        assert task.status is TaskStatus.HOLD
        assert [e.text for e in task.pending_input] == ["edited condition"]
        # The task stays out of the pipeline.
        assert app.manager.promote_next() is None


async def test_edit_form_cancel_keeps_task_untouched(app: DrCodeApp) -> None:
    task = app.manager.create("held", "condition")
    app.manager.set_hold(task.id, on=True)
    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the first-start setup dialog
        await pilot.pause()
        app.action_edit_task()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        assert not app._new_task_mode
        assert app._editing_task_id is None
        assert task.status is TaskStatus.HOLD
        assert [e.text for e in task.pending_input] == ["condition"]


async def test_edit_binding_only_for_held_tasks(app: DrCodeApp) -> None:
    queued = app.manager.create("queued", "prompt")
    held = app.manager.create("held", "prompt", hold=True)
    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the first-start setup dialog
        await pilot.pause()
        app.view_task(queued.id)
        await pilot.pause()
        assert app.check_action("edit_task", ()) is False
        app.view_task(held.id)
        await pilot.pause()
        assert app.check_action("edit_task", ()) is True
        binding = app.screen.active_bindings["ctrl+e"]
        assert binding.binding is not None
        assert binding.binding.action in ("edit_task", "app.edit_task")


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


async def test_sidebar_click_opens_task_and_cancels_new_task(app: DrCodeApp) -> None:
    """A plain click on a sidebar task opens it, leaving the new-task form."""
    task = app.manager.create("clickable", "")
    async with app.run_test() as pilot:
        # Skip the first-start setup dialog (no model configured in tests).
        await pilot.press("escape")
        await pilot.pause()
        assert app.viewed_id == task.id  # on_mount opened the only task
        app.action_new_task()
        await pilot.pause()
        assert app._new_task_mode

        item = next(i for i in app.query(TaskItem) if i.task_id == task.id)
        await pilot.click(item)
        await pilot.pause()
        assert not app._new_task_mode
        assert app.viewed_id == task.id


async def test_sidebar_drag_selection_does_not_navigate(app: DrCodeApp) -> None:
    """Selecting text in a sidebar row (press-and-drag) must not open the task.

    Textual synthesizes a Click at the release point even for a drag, as
    long as press and release land on the same widget — the sidebar tells
    the two apart and suppresses the synthetic click.
    """
    task = app.manager.create("selectable", "")
    async with app.run_test() as pilot:
        # Skip the first-start setup dialog (no model configured in tests).
        await pilot.press("escape")
        await pilot.pause()
        app.action_new_task()
        await pilot.pause()
        assert app._new_task_mode

        item = next(i for i in app.query(TaskItem) if i.task_id == task.id)
        # Press on the row…
        await pilot.mouse_down(item, offset=(1, 0))
        await pilot.pause()
        sidebar = app.sidebar
        assert sidebar._mouse_down_offset is not None
        # …and release a few cells to the right (same widget → synthesized
        # Click): the mouse moved between press and release, so it was a
        # selection, not a click.
        app.mouse_position = sidebar._mouse_down_offset + Offset(3, 0)
        sidebar.post_message(ListItem._ChildClicked(item))
        await pilot.pause()
        # Still in the new-task form — no navigation happened.
        assert app._new_task_mode


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


# -- notifications ---------------------------------------------------------------


async def test_notifications_dialog_saves_project_settings(
    app: DrCodeApp, project_root: Path
) -> None:
    from textual.widgets import Input, Switch

    from drcode.settings import ProjectSettingsStore
    from drcode.tui.notifications_dialog import NotificationsDialog

    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the setup dialog
        await pilot.pause()
        app.action_change_notifications()
        await pilot.pause()
        assert isinstance(app.screen, NotificationsDialog)
        app.screen.query_one("#ntf-enabled", Switch).value = True
        app.screen.query_one("#ntf-topic", Input).value = "my-secret-topic"
        await pilot.pause()
        await pilot.click("#ntf-save")
        await pilot.pause()
        settings = ProjectSettingsStore(project_root).load()
        assert settings.notifications_enabled is True
        assert settings.ntfy_topic == "my-secret-topic"


async def test_notifications_dialog_escape_keeps_settings(
    app: DrCodeApp, project_root: Path
) -> None:
    from textual.widgets import Input, Switch

    from drcode.settings import ProjectSettingsStore
    from drcode.tui.notifications_dialog import NotificationsDialog

    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the setup dialog
        await pilot.pause()
        # Pre-existing saved values: the dialog opens pre-filled with them.
        app.settings.notifications_enabled = True
        app.settings.ntfy_topic = "topic-a"
        app.settings_store.save(app.settings)
        app.action_change_notifications()
        await pilot.pause()
        assert isinstance(app.screen, NotificationsDialog)
        app.screen.query_one("#ntf-enabled", Switch).value = False
        app.screen.query_one("#ntf-topic", Input).value = "changed"
        await pilot.press("escape")
        await pilot.pause()
        settings = ProjectSettingsStore(project_root).load()
        assert settings.notifications_enabled is True
        assert settings.ntfy_topic == "topic-a"


async def test_pipeline_settled_notification_primes_then_sends(
    app: DrCodeApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A held task blocks the pipeline (blocked state, primed at startup —
    nothing sent); after the user finishes it off the empty queue is
    announced once."""
    from drcode.models import Task

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app.notifier,
        "notify",
        lambda title, message, **kw: sent.append((title, message)),
    )

    held = Task(title="parked", status=TaskStatus.HOLD)
    app.manager.tasks[held.id] = held
    app.manager.storage.save(held)

    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the setup dialog
        await pilot.pause()
        # First _advance(): primes the "blocked" settled state, nothing sent.
        app._advance()
        await pilot.pause()
        assert sent == []
        # User finishes the parked task: the queue becomes empty — the
        # settled-state transition is announced.
        app.manager.cancel(held.id)
        app._advance()
        await pilot.pause()
        assert len(sent) == 1
        assert "empty" in sent[0][0]
        # Another _advance() with the same (empty) state: deduplicated.
        app._advance()
        await pilot.pause()
        assert len(sent) == 1


# -- settings palette -----------------------------------------------------------


async def test_settings_palette_entry_order(app: DrCodeApp) -> None:
    """The Settings menu lists its entries in the fixed logical order
    (Keys, Screenshot, Theme, …, Quit) — never alphabetically and never in
    the provider-completion order, which would differ from run to run."""
    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the first-start setup dialog
        app.action_settings()
        await pilot.pause()
        palette = app.screen
        assert isinstance(palette, SettingsPalette)
        command_list = palette.query_one(CommandList)
        # Wait for the palette's gather worker to finish (it adds "-ready").
        for _ in range(100):
            if "-ready" in palette.classes and command_list.option_count:
                break
            await pilot.pause()
        await pilot.pause()
        names = [
            cast(Command, command).hit.text or "" for command in command_list._options
        ]

        order = SettingsPalette.ENTRY_ORDER
        ranks = [
            next(i for i, prefix in enumerate(order) if n.startswith(prefix))
            for n in names
        ]
        assert ranks == sorted(ranks)
        # Head block and pinned tail, exactly as the UX specifies.
        assert names[0] == "Keys"
        assert names[1] == "Screenshot"
        assert names[2] == "Theme"
        assert names[-1] == "Quit"


# -- terminal tab title ---------------------------------------------------------


async def test_terminal_title_tracks_working_agent(
    app: DrCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tab title reflects which agent is working: F/B/R for an active
    development task (by kind), T while the tester runs, and no badge at all
    when the pipeline is idle. The title is refreshed on status changes."""
    written: list[str] = []
    monkeypatch.setattr(
        DrCodeApp, "_set_terminal_title", lambda self, title: written.append(title)
    )
    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the first-start setup dialog
        await pilot.pause()
        base = f"DrCode — {app.project_root.name}"
        assert written[-1] == base  # idle: no agent badge

        # Feature goes active → development agent F.
        feature = app.manager.create("feature task", "build it")
        app.manager.send(feature.id, "go")
        app.manager.promote_next()
        await pilot.pause()
        assert written[-1] == f"DrCode [F] — {app.project_root.name}"

        # Development finished → tester T.
        app.manager.mark_done(feature.id)
        await pilot.pause()
        assert written[-1] == f"DrCode [T] — {app.project_root.name}"

        # Tester passed → completed, pipeline idle again.
        app.manager.set_verification(feature.id, success=True)
        await pilot.pause()
        assert written[-1] == base

        # Bug task active → B (mark_done → tester T → passed → idle).
        bug = app.manager.create("bug task", "fix it", kind=TaskKind.BUG)
        app.manager.send(bug.id, "go")
        app.manager.promote_next()
        await pilot.pause()
        assert written[-1] == f"DrCode [B] — {app.project_root.name}"
        app.manager.mark_done(bug.id)
        await pilot.pause()
        assert written[-1] == f"DrCode [T] — {app.project_root.name}"
        app.manager.set_verification(bug.id, success=True)
        await pilot.pause()
        assert written[-1] == base

        # Research task active → R; it completes right away (no tester).
        research = app.manager.create("research task", "do it", kind=TaskKind.RESEARCH)
        app.manager.send(research.id, "go")
        app.manager.promote_next()
        await pilot.pause()
        assert written[-1] == f"DrCode [R] — {app.project_root.name}"
        app.manager.mark_done(research.id)
        await pilot.pause()
        assert written[-1] == base


# -- new-task input: cursor scrolling & clipboard ---------------------------------


def _wrapped_row(input: TextArea) -> int:
    """Visual (soft-wrapped) row of the cursor inside the input."""
    return input.wrapped_document.location_to_offset(input.cursor_location).y


async def test_new_task_input_keeps_cursor_visible_after_edit(app: DrCodeApp) -> None:
    """Pressing Enter past the bottom of the input must scroll immediately.

    The stock TextArea scrolls the cursor into view before the layout
    refreshes virtual_size, so the first scroll is clamped to the old
    size and the cursor stays below the fold until the next keystroke.
    """
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("escape")  # skip the first-start setup dialog
        await pilot.pause()
        nt_input = app.query_one("#nt-input", TextArea)
        nt_input.focus()
        await pilot.pause()
        # Six logical lines, each soft-wrapping to two visual lines: more
        # than the 10-row (8 inner) input can show at once.
        nt_input.insert("\n".join("word " * 16 for _ in range(6)))
        await pilot.pause()
        inner_h = nt_input.scrollable_content_region.height
        nt_input.move_cursor((0, 0))
        nt_input.scroll_to(0, 0, animate=False)
        await pilot.pause()

        # Arrow down through every visual row: the cursor must always stay
        # inside the visible window.
        for _ in range(14):
            await pilot.press("down")
            await pilot.pause()
            row_in_view = _wrapped_row(nt_input) - nt_input.scroll_offset.y
            assert 0 <= row_in_view < inner_h

        # Enter at the very end appends empty lines below the fold; each
        # one must immediately bring the cursor into view.
        for _ in range(4):
            await pilot.press("enter")
            await pilot.pause()
            row_in_view = _wrapped_row(nt_input) - nt_input.scroll_offset.y
            assert 0 <= row_in_view < inner_h


async def test_new_task_input_copy_uses_clipboard_fallback(
    app: DrCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl+C in the input copies the selection and shows the toast.

    The stock TextArea copy only emits OSC 52, which macOS Terminal.app
    and iTerm2 ignore — the input must route through the app's fallback.
    """
    # Neutralize pbcopy so the test is deterministic on macOS too (the
    # OSC 52 write into app._clipboard is what we assert on).
    monkeypatch.setattr("shutil.which", lambda name: None)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("escape")  # skip the first-start setup dialog
        await pilot.pause()
        nt_input = app.query_one("#nt-input", TextArea)
        nt_input.focus()
        nt_input.insert("hello copyable world")
        nt_input.selection = Selection((0, 0), (0, 5))
        await pilot.pause()

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app._clipboard == "hello"
        toast = app.query_one("#copy-toast")
        assert not toast.has_class("hidden")


# -- per-role model selection (slots, custom models, picker) ---------------------


def _picker_names(app: DrCodeApp, role: AgentRole) -> list[str]:
    """Discovery entries of the role's model picker (fresh instance)."""
    provider_cls = role_model_options_provider(role)
    provider = provider_cls(app.screen)
    commands = provider._commands()  # pyright: ignore[reportAttributeAccessIssue]
    return [name for name, _callback in commands]


def _write_user_catalog(
    role: AgentRole = AgentRole.CODING,
    model_id: str = "qwen3-coder-30b",
    with_default: bool = True,
) -> None:
    from drcode.providers import (
        ModelEntry,
        ModelRef,
        Provider,
        UserModelsConfig,
        UserModelsStore,
    )

    catalog = UserModelsConfig(
        providers=[
            Provider(
                name="local-swap",
                base_url="http://localhost:8080/v1",
                models=[ModelEntry(id=model_id), ModelEntry(id="small")],
            )
        ]
    )
    if with_default:
        catalog.defaults[role] = ModelRef(provider="local-swap", model=model_id)
    UserModelsStore().save(catalog)


def _project_json(project_root: Path) -> dict:
    return json.loads((project_root / ".drcode" / "drcode.json").read_text())


async def test_no_setup_dialog_when_user_catalog_has_providers(
    project_root: Path,
) -> None:
    from drcode.tui.app import DrCodeApp

    _write_user_catalog()
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not isinstance(app.screen, ModelSetupDialog)


async def test_no_setup_dialog_when_project_has_custom_models(
    project_root: Path,
) -> None:
    from drcode.providers import CustomModelEntry, set_project_custom_model
    from drcode.tui.app import DrCodeApp

    set_project_custom_model(
        project_root,
        AgentRole.CODING,
        CustomModelEntry(model="m", base_url="http://x/v1"),
    )
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not isinstance(app.screen, ModelSetupDialog)


async def test_setup_dialog_condition_skips_dialog_with_env(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drcode.tui.app import DrCodeApp

    monkeypatch.setenv("DRCODE_MODEL", "env-m")
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not isinstance(app.screen, ModelSetupDialog)


async def test_setup_dialog_saves_three_custom_models_research_inherits(
    app: DrCodeApp, project_root: Path
) -> None:
    from drcode.providers import load_project_custom_models
    from drcode.tui.setup_dialog import ModelSetup

    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the dialog, call the callback directly
        await pilot.pause()
        app._model_setup_done(
            ModelSetup(
                model="first-m",
                base_url="http://x/v1",
                api_key="k",
                max_context_tokens=123456,
                max_tokens=2048,
                health_check_url="http://x/health",
            )
        )
        await pilot.pause()
    data = _project_json(project_root)
    # Three "custom" slots (coding/tester/compaction), three separate
    # copies, no flat model keys. The research role stays unpinned: it
    # inherits the coding model via the resolve_effective fallback.
    assert data["models"] == {
        "coding": "custom",
        "tester": "custom",
        "compaction": "custom",
    }
    customs = load_project_custom_models(project_root)
    assert len(customs) == 3
    for entry in customs.values():
        assert entry.model == "first-m"
        assert entry.base_url == "http://x/v1"
        assert entry.max_context_tokens == 123456
    for key in ("model", "base_url", "api_key", "max_context_tokens", "max_tokens"):
        assert key not in data
    # Independence: the entries are distinct objects.
    values = list(customs.values())
    assert values[0] is not values[1]
    # The coding model is live: the header/statusbar show it; research
    # transparently runs on the same connection.
    assert app.resolved_config(AgentRole.CODING).model == "first-m"
    assert app.resolved_config(AgentRole.RESEARCH).model == "first-m"


async def test_research_role_resolves_with_coding_fallback(
    app: DrCodeApp, project_root: Path
) -> None:
    """The app-level resolution: with no research selection, the research
    role transparently resolves to the coding model; a dedicated research
    custom model overrides it while coding keeps its own."""
    _write_user_catalog()
    app.invalidate_resolved_configs()
    assert app.resolved_config(AgentRole.CODING).model == "qwen3-coder-30b"
    assert app.resolved_config(AgentRole.RESEARCH).model == "qwen3-coder-30b"
    from drcode.providers import CustomModelEntry, set_project_custom_model

    set_project_custom_model(
        project_root,
        AgentRole.RESEARCH,
        CustomModelEntry(
            model="research-m",
            base_url="http://research/v1",
            max_context_tokens=400_000,
        ),
    )
    app.invalidate_resolved_configs()
    assert app.resolved_config(AgentRole.RESEARCH).model == "research-m"
    assert app.resolved_config(AgentRole.RESEARCH).max_context_tokens == 400_000
    # Coding is untouched by the research entry.
    assert app.resolved_config(AgentRole.CODING).model == "qwen3-coder-30b"


async def test_task_role_follows_kind_and_compaction_uses_serving_role(
    app: DrCodeApp, project_root: Path
) -> None:
    """Research tasks map to the RESEARCH role (`_task_role`), and the
    compaction gate measures their transcript against the research model's
    window — not the coding model's."""
    from drcode.providers import CustomModelEntry, set_project_custom_model

    _write_user_catalog()
    set_project_custom_model(
        project_root,
        AgentRole.RESEARCH,
        CustomModelEntry(
            model="research-m",
            base_url="http://research/v1",
            max_context_tokens=1000,
        ),
    )
    app.invalidate_resolved_configs()
    research = app.manager.create("r", "prompt", kind=TaskKind.RESEARCH)
    feature = app.manager.create("f", "prompt", kind=TaskKind.FEATURE)
    assert app._task_role(research) is AgentRole.RESEARCH
    assert app._task_role(feature) is AgentRole.CODING
    # A transcript whose last reading is 900 tokens: under the coding
    # model's 200k window the gate stays off for the feature task, but
    # the research task (1000-token window, default 80% threshold) is due.
    msg = [
        {"kind": "request", "role": "user", "parts": []},
        {"kind": "response", "usage": {"input_tokens": 850, "output_tokens": 50}},
    ]
    research.messages = msg
    feature.messages = msg
    assert app._compaction_due(research) == 900
    assert app._compaction_due(feature) is None


async def test_role_picker_marks_default_and_clears_slot(
    app: DrCodeApp, project_root: Path
) -> None:
    from drcode.providers import ModelRef, load_project_model_slots, set_project_model

    _write_user_catalog()
    ref = ModelRef(provider="local-swap", model="qwen3-coder-30b")
    set_project_model(project_root, AgentRole.CODING, ref)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
        names = _picker_names(app, AgentRole.CODING)
        assert "local-swap/qwen3-coder-30b (default) (current)" in names
        assert "local-swap/small" in names
        assert "(default — user catalog)" in names
        assert "Custom…" in names
        # Selecting the "(default — user catalog)" entry removes the
        # project override; the user default becomes effective again.
        provider = role_model_options_provider(AgentRole.CODING)(app.screen)
        commands = provider._commands()  # pyright: ignore[reportAttributeAccessIssue]
        callback = next(
            cb for name, cb in commands if name == "(default — user catalog)"
        )
        callback()
        await pilot.pause()
        assert load_project_model_slots(project_root) == {}
        assert app.resolved_config(AgentRole.CODING).model == "qwen3-coder-30b"


async def test_role_picker_custom_toggle_roundtrip(
    app: DrCodeApp, project_root: Path
) -> None:
    from drcode.providers import (
        CustomModelEntry,
        ModelRef,
        load_project_custom_models,
        set_project_custom_model,
        set_project_model,
    )

    entry = CustomModelEntry(
        model="custom-m", base_url="http://custom/v1", api_key="ck"
    )
    set_project_custom_model(project_root, AgentRole.TESTER, entry)
    set_project_model(
        project_root, AgentRole.TESTER, ModelRef(provider="local-swap", model="small")
    )
    _write_user_catalog(role=AgentRole.TESTER, model_id="small")
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
        names = _picker_names(app, AgentRole.TESTER)
        # The saved custom model is listed although a catalog ref is active.
        assert "Custom: custom-m" in names
        # "small" is both the user default for the role and the effective
        # model (the project slot points at it): both marks.
        assert "local-swap/small (default) (current)" in names
        provider = role_model_options_provider(AgentRole.TESTER)(app.screen)
        commands = dict(
            provider._commands()  # pyright: ignore[reportAttributeAccessIssue]
        )
        # Switch back to the custom model: slot flips to "custom", the
        # entry is reused without re-entering anything.
        commands["Custom: custom-m"]()
        await pilot.pause()
        assert app.resolved_config(AgentRole.TESTER).model == "custom-m"
        assert _project_json(project_root)["models"]["tester"] == "custom"
        # And to the catalog again: the custom entry survives ("parked").
        provider = role_model_options_provider(AgentRole.TESTER)(app.screen)
        names = _picker_names(app, AgentRole.TESTER)
        assert "Custom: custom-m (current)" in names
        commands = provider._commands()  # pyright: ignore[reportAttributeAccessIssue]
        catalog_switch = next(
            cb for name, cb in commands if name.startswith("local-swap/small")
        )
        catalog_switch()
        await pilot.pause()
        assert load_project_custom_models(project_root)[AgentRole.TESTER] == entry
        assert app.resolved_config(AgentRole.TESTER).model == "small"


async def test_role_picker_activate_custom_with_broken_entry_warns(
    app: DrCodeApp, project_root: Path
) -> None:
    """models[role]="custom" pointing at a broken entry: activation warns
    and the file is not silently rewritten."""
    cfg_file = project_root / ".drcode" / "drcode.json"
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(
        json.dumps({"models": {"tester": "custom"}, "custom_models": {"tester": {}}})
    )
    notified: list[str] = []
    app.notify = lambda message, **kw: notified.append(str(message))  # type: ignore[method-assign]
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
        app.action_activate_role_custom(AgentRole.TESTER)
        await pilot.pause()
    assert any("No valid custom model" in n for n in notified)
    data = json.loads(cfg_file.read_text())
    assert data["custom_models"] == {"tester": {}}


async def test_role_picker_empty_catalog_opens_custom_dialog(
    app: DrCodeApp,
) -> None:
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
        app._open_role_model_picker(AgentRole.COMPACTION)
        await pilot.pause()
        assert isinstance(app.screen, ModelSetupDialog)


async def test_role_model_label_variants(app: DrCodeApp, project_root: Path) -> None:
    from drcode.providers import CustomModelEntry, set_project_custom_model

    assert app.role_model_label(AgentRole.CODING) == "not set"
    _write_user_catalog()
    assert app.role_model_label(AgentRole.CODING) == "local-swap/qwen3-coder-30b"
    set_project_custom_model(
        project_root,
        AgentRole.TESTER,
        CustomModelEntry(model="custom-m", base_url="http://c/v1"),
    )
    assert app.role_model_label(AgentRole.TESTER) == "custom-m"


async def test_settings_menu_model_settings_entry(app: DrCodeApp) -> None:
    """The main Settings menu collapses all model-related entries into one
    "Model settings…" item, which opens the sub-menu with the provider
    catalog editor and the per-role model pickers."""
    from textual.command import CommandList

    from drcode.tui.command_providers import (
        ModelCatalogCommandProvider,
        ModelSettingsCommandProvider,
        ModelSettingsPalette,
        RoleModelsCommandsProvider,
        SettingsPalette,
    )

    # The main menu has no direct model entries anymore.
    assert "Model settings" in SettingsPalette.ENTRY_ORDER
    assert "Provider catalog…" not in SettingsPalette.ENTRY_ORDER
    assert "Coding model:" not in SettingsPalette.ENTRY_ORDER

    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
        provider = ModelSettingsCommandProvider(app.screen)
        commands = provider._commands()  # pyright: ignore[reportAttributeAccessIssue]
        assert [(name, cb) for name, _help, cb in commands] == [
            ("Model settings…", app.action_open_model_settings)
        ]
        app.action_open_model_settings()
        await pilot.pause()
        palette = app.screen
        assert isinstance(palette, ModelSettingsPalette)
        # Wait for the palette's gather worker to finish.
        command_list = palette.query_one(CommandList)
        for _ in range(100):
            if "-ready" in palette.classes and command_list.option_count:
                break
            await pilot.pause()
        await pilot.pause()
        names = [
            cast(Command, command).hit.text or "" for command in command_list._options
        ]
        assert names == [
            "Provider catalog…",
            f"Coding model: {app.role_model_label(AgentRole.CODING)}",
            f"Research model: {app.role_model_label(AgentRole.RESEARCH)}",
            f"Tester model: {app.role_model_label(AgentRole.TESTER)}",
            f"Compaction model: {app.role_model_label(AgentRole.COMPACTION)}",
        ]
        # Sub-menu providers: the catalog editor plus the role pickers,
        # nothing else (project preferences stay in the main menu).
        commands = RoleModelsCommandsProvider(palette)._commands()  # pyright: ignore[reportAttributeAccessIssue]
        assert next(name for name, _help, _cb in commands) == (
            f"Coding model: {app.role_model_label(AgentRole.CODING)}"
        )
        catalog = ModelCatalogCommandProvider(palette)._commands()  # pyright: ignore[reportAttributeAccessIssue]
        assert [name for name, _help, _cb in catalog] == ["Provider catalog…"]


async def test_custom_role_model_dialog_prefills_and_saves(
    app: DrCodeApp, project_root: Path
) -> None:
    """The per-role "Custom…" dialog: pre-filled from the saved custom
    model; saving overwrites the entry and keeps the role's slot custom."""
    from textual.widgets import Input

    from drcode.providers import (
        CustomModelEntry,
        load_project_custom_models,
        set_project_custom_model,
    )
    from drcode.tui.setup_dialog import ModelSetup

    set_project_custom_model(
        project_root,
        AgentRole.COMPACTION,
        CustomModelEntry(
            model="old-m",
            base_url="http://old/v1",
            api_key="ok",
            max_context_tokens=32000,
        ),
    )
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
        app.action_custom_role_model(AgentRole.COMPACTION)
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, ModelSetupDialog)
        assert dialog.query_one("#sd-model", Input).value == "old-m"
        assert dialog.query_one("#sd-url", Input).value == "http://old/v1"
        assert dialog.query_one("#sd-key", Input).value == "ok"
        assert dialog.query_one("#sd-ctx", Input).value == "32000"
        # A fresh role starts with an empty form.
        app.action_custom_role_model(AgentRole.CODING)
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, ModelSetupDialog)
        assert dialog.query_one("#sd-model", Input).value == ""
        # Save (the callback is invoked directly — the probe is bypassed).
        dialog.dismiss(
            ModelSetup(
                model="new-m",
                base_url="http://new/v1",
                api_key="nk",
                max_context_tokens=64000,
                max_tokens=None,
            )
        )
        await pilot.pause()
    customs = load_project_custom_models(project_root)
    assert customs[AgentRole.CODING].model == "new-m"
    assert customs[AgentRole.CODING].max_context_tokens == 64000
    # Untouched by the coding edit: the compaction entry stays as it was —
    # custom models of different roles never mix.
    assert customs[AgentRole.COMPACTION].model == "old-m"
    assert customs[AgentRole.COMPACTION].max_context_tokens == 32000
    assert _project_json(project_root)["models"]["coding"] == "custom"


# -- copy on select -----------------------------------------------------------------


async def test_copy_on_select_matches_visual_selection(
    app: DrCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mouse-selecting a code fragment inside a wrapped user prompt copies
    exactly that fragment: the selection offsets are content coordinates,
    so the extraction must not index them against display rows (padding
    and soft-wrapped rows above the selection would shift the fragment)."""
    long_prompt = (
        "The custom workflow layer has no hook for a post-verification step. "
        "The closest hook, `on_accepted`, is only reachable through the "
        "external-review gate:\n"
        "\n"
        "- `_run_accepted_hook` is called from exactly one place — inside "
        "`run_external_review` (`src/drcode/tui/pipeline.py:1749`);\n"
        "- the external review only starts when the script defines `on_review`.\n"
        "\n"
        'So a user who just wants "after the tester PASSED the task, run '
        '`uv tool install . --force`" must define a fake review gate:\n'
        "\n"
        "```python\n"
        "def on_review(task):\n"
        '    return {"verdict": "accept"}   # dummy, only to unlock on_accepted\n'
        "\n"
        "def on_accepted(task):\n"
        '    subprocess.run(["uv", "tool", "install", ".", "--force"], check=True)\n'
        "```\n"
    )
    task = app.manager.create("hook task", long_prompt)
    copied: list[str] = []
    monkeypatch.setattr(app, "_copy_with_fallback", copied.append)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")  # skip the first-start setup dialog
        await pilot.pause()
        assert app.viewed_id == task.id
        # The rendered user block: locate the display rows of the code
        # fragment the user would drag over.
        block = next(
            w
            for w in app.query_one("#chat-log").children
            if "```python" in str(w.render())
        )
        fence_row = close_row = None
        for y in range(block.size.height):
            line = block.render_line(y).text
            if "```python" in line:
                fence_row = y
            if line.rstrip() == "```":
                close_row = y
        assert fence_row is not None and close_row is not None
        content = block.content_region
        await pilot.mouse_down(offset=(content.x, content.y + fence_row))
        # Drag to just past the closing fence (as a user selecting the whole
        # displayed fragment would).
        await pilot.mouse_up(offset=(content.x + 4, content.y + close_row))
        await pilot.pause()
    assert copied == [
        (
            "```python\n"
            "def on_review(task):\n"
            '    return {"verdict": "accept"}   # dummy, only to unlock on_accepted\n'
            "\n"
            "def on_accepted(task):\n"
            '    subprocess.run(["uv", "tool", "install", ".", "--force"], check=True)\n'
            "```"
        )
    ]
