from pathlib import Path

import pytest

from drcode.agent import is_pending_retry
from drcode.models import TaskStatus
from drcode.tui.app import DrCodeApp


@pytest.fixture
def app(project_root: Path) -> DrCodeApp:
    return DrCodeApp(project_root)


async def test_pending_comment_survives_view_switch(app: DrCodeApp) -> None:
    a = app.manager.create("task A", "original prompt")
    a.status = TaskStatus.TESTING
    b = app.manager.create("task B", "other prompt")

    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(a.id)
        text = "please also fix the sidebar bug"
        app.chat_input.text = text
        app.submit_chat_input()
        await pilot.pause()

        print("\nSTATUS after submit:", a.status)
        print("viewed_id:", app.viewed_id, "task.id:", a.id)
        print("is_pending_retry:", is_pending_retry(a.messages, text))
        print("batching:", app._chat_batching, "batch:", app._chat_batch)
        print("pending_rendered:", app._pending_rendered)

        rendered_a = "\n".join(str(w.render()) for w in app.query_one("#chat-log").children)
        print("\n=== CHAT AFTER SUBMIT ===")
        print(repr(rendered_a))

        app.view_task(b.id)
        for _ in range(20):
            await pilot.pause()
            if not app._chat_batching:
                break
        app.view_task(a.id)
        for _ in range(20):
            await pilot.pause()
            if not app._chat_batching:
                break

        rendered_a2 = "\n".join(str(w.render()) for w in app.query_one("#chat-log").children)
        print("\n=== CHAT AFTER SWITCH AWAY & BACK TO A ===")
        print(repr(rendered_a2))
        print("PENDING_INPUT:", [e.text for e in a.pending_input])
