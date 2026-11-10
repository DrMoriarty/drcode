"""ChatLog tail-following via Textual's Widget.anchor mechanism.

The chat log is anchored by construction: the compositor pins it to the
bottom on every arrange while the anchor is held, any user scroll
releases the anchor and scrolling back to the bottom restores it. These
tests pin the contract the removed positional heuristic
(`_chat_at_bottom` + `watch_virtual_size`) used to provide — see
blocks.ChatLog and chat_timeline.view_task.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from textual.events import MouseScrollDown, MouseScrollUp

from drcode.agent import serialize_messages
from drcode.models import TaskEvent, TaskStatus
from drcode.tui.app import DrCodeApp
from drcode.tui.blocks import ChatLog


@pytest.fixture
def app(project_root: Path) -> DrCodeApp:
    return DrCodeApp(project_root)


def _transcript(turns: int) -> list[dict]:
    """Serialized user/assistant alternation — two blocks per turn."""
    messages: list[ModelRequest | ModelResponse] = []
    for i in range(turns):
        messages.append(ModelRequest(parts=[UserPromptPart(content=f"question {i}")]))
        messages.append(ModelResponse(parts=[TextPart(content=f"answer {i}")]))
    return serialize_messages(messages)


def _wheel(
    log: ChatLog, cls: type[MouseScrollDown | MouseScrollUp], delta_y: int
) -> None:
    """Post a synthetic wheel event straight to the log (the same handler
    path `_on_mouse_scroll_down/_up` take for real pointer scrolls)."""
    log.post_message(cls(log, 0, 0, 0, delta_y, 0, False, False, False))


def _at_bottom(log: ChatLog) -> bool:
    """The tail is visible: either scrolled to the very bottom, or the
    content is shorter than the viewport (the compositor's anchor pin may
    park a negative scroll_y then — its set_reactive write bypasses the
    clamp; short content trivially shows its tail)."""
    return log.max_scroll_y == 0 or log.scroll_offset.y >= log.max_scroll_y


async def _drain(app: DrCodeApp, pilot) -> None:
    for _ in range(60):
        await pilot.pause()
        if not app._chat_batching:
            break
    assert not app._chat_batching
    await pilot.pause()


async def test_chat_log_anchored_on_mount(app: DrCodeApp) -> None:
    """The freshly mounted #chat-log is anchored (following armed) and a
    rebuilt history lands on the latest entry."""
    task = app.manager.create("history", "prompt")
    task.messages = _transcript(40)
    task.status = TaskStatus.COMPLETED
    async with app.run_test() as pilot:
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        assert log.is_anchored
        assert not log._anchor_released
        await _drain(app, pilot)
        assert _at_bottom(log)


async def test_anchor_keeps_tail_on_growth(app: DrCodeApp) -> None:
    """While the anchor is held, mounting new blocks (plain live writes)
    keeps the bottom line visible — without any manual scrolling."""
    app.manager.create("live", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        await _drain(app, pilot)
        for i in range(5):
            app._write_chat(f"[dim]live line {i}[/]")
            await pilot.pause()
        assert _at_bottom(log)
        assert "live line 4" in str(log.children[-1].render())


async def test_user_scroll_up_releases_anchor(app: DrCodeApp) -> None:
    """A user scroll (the same _scroll_to path wheel/keyboard/scrollbar
    take) instantly releases the anchor; subsequent growth must not yank
    the reader back down."""
    task = app.manager.create("reading", "prompt")
    task.messages = _transcript(40)
    task.status = TaskStatus.COMPLETED
    async with app.run_test() as pilot:
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        await _drain(app, pilot)
        log.scroll_to(y=0, animate=False)
        await pilot.pause()
        assert log._anchor_released
        assert log.scroll_offset.y < log.max_scroll_y
        # New blocks arrive (an agent streaming while the user reads):
        # the reader stays where they are.
        app._write_chat("[dim]new tool result[/]")
        await pilot.pause()
        await pilot.pause()
        assert log.scroll_offset.y == 0
        assert log._anchor_released


async def test_scroll_back_to_bottom_restores_anchor(app: DrCodeApp) -> None:
    """Scrolling back to the bottom re-arms following (watch_scroll_y →
    _check_anchor)."""
    task = app.manager.create("return", "prompt")
    task.messages = _transcript(40)
    task.status = TaskStatus.COMPLETED
    async with app.run_test() as pilot:
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        await _drain(app, pilot)
        log.scroll_to(y=0, animate=False)
        await pilot.pause()
        assert log._anchor_released
        log.scroll_to(y=log.max_scroll_y, animate=False)
        await pilot.pause()
        assert not log._anchor_released
        # And following works again: growth keeps the tail on screen.
        app._write_chat("[dim]appended after return[/]")
        await pilot.pause()
        await pilot.pause()
        assert _at_bottom(log)


async def test_wheel_down_at_bottom_keeps_following(app: DrCodeApp) -> None:
    """The killer edge case of the anchor semantics: a wheel-down when
    already at the bottom scrolls nowhere — `_scroll_to` releases the
    anchor but `watch_scroll_y` never fires. ChatLog's wheel handlers
    re-run the restore check, so following survives the gesture."""
    app.manager.create("wheel", "prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        await _drain(app, pilot)
        _wheel(log, MouseScrollDown, 2)
        await pilot.pause()
        await pilot.pause()
        assert log.is_anchored and not log._anchor_released
        # Following still alive: new content keeps the tail visible.
        app._write_chat("[dim]tail after wheel[/]")
        await pilot.pause()
        await pilot.pause()
        assert _at_bottom(log)


async def test_wheel_up_releases_anchor(app: DrCodeApp) -> None:
    """A real wheel-up (via the ChatLog handler path) releases the anchor."""
    task = app.manager.create("wheel up", "prompt")
    task.messages = _transcript(40)
    task.status = TaskStatus.COMPLETED
    async with app.run_test() as pilot:
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        await _drain(app, pilot)
        _wheel(log, MouseScrollUp, 2)
        await pilot.pause()
        await pilot.pause()
        assert log._anchor_released
        assert log.scroll_offset.y < log.max_scroll_y


async def test_rebuild_lands_on_last_entry(app: DrCodeApp) -> None:
    """The 'opening a task always lands on the latest entry' contract: a
    full rebuild (view_task after the delta state was invalidated) re-arms
    the anchor at the wipe and the drained history ends at the bottom —
    even if the reader was scrolled up before the rebuild."""
    task = app.manager.create("rebuild", "prompt")
    task.messages = _transcript(40)
    task.status = TaskStatus.COMPLETED
    async with app.run_test() as pilot:
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        await _drain(app, pilot)
        log.scroll_to(y=0, animate=False)
        await pilot.pause()
        task.events.append(TaskEvent(text="forces a rebuild"))
        app._invalidate_chat_delta()
        app.view_task(task.id)
        await _drain(app, pilot)
        assert _at_bottom(log)
        assert log.is_anchored and not log._anchor_released


async def test_click_scroll_visible_releases_anchor(app: DrCodeApp) -> None:
    """A click that scrolls a block into view (scroll_visible on focus)
    counts as user interaction: it releases the anchor — accepted
    behaviour, pinned here so a regression would be a conscious one."""
    task = app.manager.create("click", "prompt")
    task.messages = _transcript(40)
    task.status = TaskStatus.COMPLETED
    async with app.run_test() as pilot:
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        await _drain(app, pilot)
        # Simulate the focus-click path: scroll a mid-history block into
        # view (screen.scroll_to_widget → scroll_to → releases).
        log.children[len(log.children) // 2].scroll_visible(animate=False)
        await pilot.pause()
        await pilot.pause()
        assert log._anchor_released
