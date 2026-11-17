"""Regression: a user comment must never appear twice in the task's chat
history — neither live nor after a re-render.

The reported incident (task #fd067403): a comment sent to a task that was
being tested showed up twice. The doubling happens when the same text is
submitted again while the first copy is still queued or still being
processed by a run: the pending bubble plus the consuming run's "user"
block (or the second copy persisted into the transcript) draw the comment
twice. The fix deduplicates identical submissions in `TaskManager.send()`
and skips the re-send in `_submit_user_text`.

A second doubling shape (task #671d6541): a message that re-queues a
COMPLETED task (or answers a clarifying question). Textual installs
asyncio.eager_task_factory, so the run worker started by `_advance()`
executes synchronously down to its first await — including the
pending-input consumption and its "user" draw — BEFORE `_submit_user_text`
gets to draw its own "user-pending" bubble (whose is_pending_retry guard
is still False: the transcript is not persisted yet). The fix draws the
bubble before advancing the pipeline, so the consuming run sees the text
in `_pending_rendered` and skips its own draw.

A third doubling shape: a message sent WITH staged images. The pre-drawn
"user-pending" bubble carries the "[image <media>]" placeholder lines,
but the consuming run's dedup matches the PLAIN prompt text in
`_pending_rendered` — with images the two diverged, the dedup missed and
the run drew the message a second time (the doubled bubble right after
the "Model:" event line). The fix registers the plain text alongside the
drawn bubble.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from drcode import agent_model
from drcode.models import TaskKind, TaskStatus
from drcode.tui.app import DrCodeApp
from tests.network_mocks import (
    ScriptedState,
    text_response,
    tool_response,
    wait_until,
)


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedState:
    state = ScriptedState()
    monkeypatch.setattr(
        agent_model, "build_model", lambda cfg, announce_wait=None: state.model()
    )
    return state


def _comment_hits(app: DrCodeApp, comment: str) -> int:
    return sum(
        1 for w in app.query_one("#chat-log").children if comment in str(w.render())
    )


async def test_repeated_comment_rendered_once(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The incident sequence: the comment arrives while a tester round is
    mid-run, the user nudges (Ctrl+R) and — seeing no reaction yet —
    re-sends the same text. The comment must show exactly once, live and
    in the final history."""
    app = DrCodeApp(project_root)
    task = app.manager.create("buggy", "fix it", kind=TaskKind.BUG)
    comment = "no real testing please, standard checks only"

    # Hold the tester's first model call, so the comment is typed while the
    # tester is genuinely hung mid-run (like the real incident).
    release = asyncio.Event()
    orig_stream = scripted.stream_fn

    async def gated_stream(
        messages: list[Any], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        for m in messages:
            if hasattr(m, "parts"):
                for p in m.parts:
                    if "Verify completed task" in str(getattr(p, "content", "")):
                        await release.wait()
        async for chunk in orig_stream(messages, info):
            yield chunk

    monkeypatch.setattr(
        scripted,
        "model",
        lambda: FunctionModel(scripted.fn, stream_function=gated_stream),
    )

    # Track every live draw of the comment; view_task wipes the log, so the
    # counter resets there — a value > 1 between wipes means duplication.
    onscreen = {"n": 0, "max": 0}
    orig_line = DrCodeApp._write_history_line

    def spy_line(self: DrCodeApp, role: str, text: str, at: Any = None) -> Any:
        if comment in str(text):
            onscreen["n"] += 1
            onscreen["max"] = max(onscreen["max"], onscreen["n"])
        return orig_line(self, role, text, at)

    monkeypatch.setattr(DrCodeApp, "_write_history_line", spy_line)

    orig_view = DrCodeApp.view_task

    def spy_view(self: DrCodeApp, task_id: str) -> None:
        if task_id == task.id:
            onscreen["n"] = 0
        return orig_view(self, task_id)

    monkeypatch.setattr(DrCodeApp, "view_task", spy_view)

    scripted.steps = [
        text_response("fixed it"),  # dev round 1
        tool_response("set_verification", {"success": True}),  # tester round 1
        text_response("all good"),
    ]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: app.tester_running and task.status is TaskStatus.TESTING,
            pilot,
            what="tester running",
        )
        # 1) the comment arrives while the tester is hung mid-run
        app._submit_user_text(task, comment)
        await pilot.pause()
        assert _comment_hits(app, comment) == 1
        # 2) Ctrl+R: abort the hung tester, schedule the pipeline advance
        app.action_nudge_run()
        # 3) still no visible reaction: the user re-sends the same text
        app._submit_user_text(task, comment)
        await pilot.pause()
        assert _comment_hits(app, comment) == 1
        assert [e.text for e in task.pending_input].count(comment) == 1
        # Let the pipeline run: the developer consumes the comment, the task
        # is re-tested and completes.
        release.set()
        scripted.steps.extend(
            [
                text_response("dev again"),
                tool_response("set_verification", {"success": True}),
                text_response("done again"),
            ]
        )
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and not app.tester_running
                and not app.agent_running
            ),
            pilot,
            what="settled",
        )
        await pilot.pause()
        assert onscreen["max"] <= 1, "the comment was drawn twice live"

        # Fresh re-render (what reopening the task shows): exactly once.
        app.view_task(task.id)
        for _ in range(60):
            await pilot.pause()
            if not app._chat_batching:
                break
        assert _comment_hits(app, comment) == 1


async def test_retype_while_active_reaches_model_once(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-sending the same text while a dev run processes it must not
    duplicate the comment: the model sees it once (the superseded run's
    in-flight request is replaced by the new run's single prompt)."""
    app = DrCodeApp(project_root)
    task = app.manager.create("t", "fix it", kind=TaskKind.RESEARCH)
    release = asyncio.Event()
    orig_stream = scripted.stream_fn

    async def gated_stream(
        messages: list[Any], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        for m in messages:
            if hasattr(m, "parts"):
                for p in m.parts:
                    content = str(getattr(p, "content", ""))
                    if "fix it" in content or "extra note" in content:
                        await release.wait()
        async for chunk in orig_stream(messages, info):
            yield chunk

    monkeypatch.setattr(
        scripted,
        "model",
        lambda: FunctionModel(scripted.fn, stream_function=gated_stream),
    )
    scripted.steps = [text_response("done"), text_response("done 2")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(lambda: app.agent_running, pilot, what="agent running")
        app._submit_user_text(task, "extra note")
        await pilot.pause()
        # Still no reaction (the run is held): the user re-sends.
        app._submit_user_text(task, "extra note")
        await pilot.pause()
        release.set()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED and not app.agent_running,
            pilot,
            what="completed",
        )
    # The superseded run's held request never fired; the retry run sent
    # the comment exactly once.
    assert scripted.calls == ["extra note"]


async def test_message_with_images_rendered_once_with_eager_worker(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A message sent WITH staged images (a clipboard screenshot + a file
    path) must show exactly once.

    Production sequence that doubled the message (verified on a baseline
    without the fix — this test fails there with "the message was drawn
    twice live"):

    1. `_submit_user_text` pre-draws the "user-pending" bubble BEFORE
       `_advance()`, because Textual's eager task factory can run the
       consuming worker synchronously inside `_advance()`. The drawn line
       is the DECORATED text: prompt + "[image <media>]" placeholder lines
       per attachment. `_write_history_line("user-pending", …)` registers
       exactly that decorated string in `_pending_rendered`.
    2. The consuming run (core/pipeline_agent_run.py) dedups per pending
       entry against the PLAIN `entry.text`. With images the two strings
       diverge, so the `_pending_rendered` check misses; the transcript is
       not persisted yet (first node-boundary persist happens later), so
       the `is_pending_retry` guard is False too — and the run draws the
       plain "user" block: the message appears twice, right after the
       run-start "Model: …" event line.
    3. The fix registers the PLAIN prompt text in `_pending_rendered`
       alongside the decorated bubble (when refs are present), so the
       consuming run's check hits and its own draw is skipped.

    The first assertion below pins the invariant directly: right after the
    submit, the plain prompt text is registered in `_pending_rendered`.
    The rest asserts the end-to-end outcome: exactly one bubble, live and
    after a fresh re-render.
    """
    from drcode.tui.attachments import StagedImage

    app = DrCodeApp(project_root)
    text = "вот два скриншота, разберись"

    onscreen = {"n": 0, "max": 0}
    orig_line = DrCodeApp._write_history_line

    def spy_line(self: DrCodeApp, role: str, line_text: str, at: Any = None) -> Any:
        if text in str(line_text):
            onscreen["n"] += 1
            onscreen["max"] = max(onscreen["max"], onscreen["n"])
        return orig_line(self, role, line_text, at)

    monkeypatch.setattr(DrCodeApp, "_write_history_line", spy_line)

    staged = [
        StagedImage(
            filename="Clipboard.png", media_type="image/png", data=b"\x89PNG-clip"
        ),
        StagedImage(
            filename="dropped.png", media_type="image/png", data=b"\x89PNG-file"
        ),
    ]
    scripted.steps = [text_response("Разобрался по скриншотам.")]
    async with app.run_test() as pilot:
        # Emulate the production loop: Textual's real app installs
        # asyncio.eager_task_factory, the pilot env does not.
        asyncio.get_running_loop().set_task_factory(asyncio.eager_task_factory)
        await pilot.pause()
        # Create the task AFTER the startup advance: a queued task existing
        # at mount is promoted right away and would consume the scripted
        # step meant for this message's run.
        task = app.manager.create(
            "research", "посмотри скриншоты", kind=TaskKind.RESEARCH
        )
        orig_view = DrCodeApp.view_task

        def spy_view(self: DrCodeApp, task_id: str) -> None:
            if task_id == task.id:
                onscreen["n"] = 0
            return orig_view(self, task_id)

        monkeypatch.setattr(DrCodeApp, "view_task", spy_view)

        # Direct invariant snapshot: _advance() is the point where the eager
        # consuming worker may already read `_pending_rendered`, so capture
        # the registration state right before it (after the submit's
        # pre-draw + plain-text registration, before any consumption can
        # remove the entry).
        at_advance = {"plain_registered": False}
        orig_advance = DrCodeApp._advance

        def spy_advance(self: DrCodeApp) -> Any:
            at_advance["plain_registered"] = text in self._pending_rendered
            return orig_advance(self)

        monkeypatch.setattr(DrCodeApp, "_advance", spy_advance)

        app.view_task(task.id)
        app._submit_user_text(task, text, attachments=staged)
        assert at_advance["plain_registered"], (
            "the plain prompt text must be registered in _pending_rendered "
            "before the pipeline advance — the consuming run dedups against "
            "the PLAIN text, and only this registration keeps it from "
            "drawing the message a second time"
        )
        for _ in range(5):
            await pilot.pause()
            assert onscreen["max"] <= 1, "the message was drawn twice live"
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED and not app.agent_running,
            pilot,
            what="research completed",
        )
        await pilot.pause()

        # Fresh re-render (what reopening the task shows): exactly once.
        app.view_task(task.id)
        for _ in range(60):
            await pilot.pause()
            if not app._chat_batching:
                break
        hits = sum(
            1 for w in app.query_one("#chat-log").children if text in str(w.render())
        )
        assert hits == 1


async def test_requeue_message_rendered_once_with_eager_worker(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A message that re-queues a COMPLETED task must show exactly once.

    The pilot test loop does not install Textual's eager task factory, so it
    is set explicitly: with it, the run worker started by `_advance()`
    consumes the pending input synchronously inside `_submit_user_text` —
    the exact production timing in which the pre-fix code drew the message
    twice (the run's "user" block, then the submit's "user-pending" bubble).
    """
    app = DrCodeApp(project_root)
    task = app.manager.create("research", "изучи проблему", kind=TaskKind.RESEARCH)
    answer = "заведи отдельную задачу на реализацию пункта 1"

    onscreen = {"n": 0, "max": 0}
    orig_line = DrCodeApp._write_history_line

    def spy_line(self: DrCodeApp, role: str, text: str, at: Any = None) -> Any:
        if answer in str(text):
            onscreen["n"] += 1
            onscreen["max"] = max(onscreen["max"], onscreen["n"])
        return orig_line(self, role, text, at)

    monkeypatch.setattr(DrCodeApp, "_write_history_line", spy_line)

    orig_view = DrCodeApp.view_task

    def spy_view(self: DrCodeApp, task_id: str) -> None:
        if task_id == task.id:
            onscreen["n"] = 0
        return orig_view(self, task_id)

    monkeypatch.setattr(DrCodeApp, "view_task", spy_view)

    scripted.steps = [
        text_response("Разобрался. Вот что я нашёл — жду решения."),
    ]
    async with app.run_test() as pilot:
        # Emulate the production loop: Textual's real app installs
        # asyncio.eager_task_factory, the pilot env does not.
        asyncio.get_running_loop().set_task_factory(asyncio.eager_task_factory)
        await pilot.pause()
        app.view_task(task.id)
        app._advance()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED and not app.agent_running,
            pilot,
            what="research completed",
        )
        await pilot.pause()
        assert onscreen["n"] == 0

        # The message re-queues the completed task (research: straight back
        # into development). The run streams long enough for the double to
        # be visible — assert on the live draw count between view wipes.
        scripted.steps.append(text_response("Создаю задачи."))
        app._submit_user_text(task, answer)
        for _ in range(5):
            await pilot.pause()
            assert onscreen["max"] <= 1, "the message was drawn twice live"
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED and not app.agent_running,
            pilot,
            what="answered run completed",
        )
        await pilot.pause()

        # Fresh re-render (what reopening the task shows): exactly once.
        app.view_task(task.id)
        for _ in range(60):
            await pilot.pause()
            if not app._chat_batching:
                break
        hits = sum(
            1 for w in app.query_one("#chat-log").children if answer in str(w.render())
        )
        assert hits == 1
