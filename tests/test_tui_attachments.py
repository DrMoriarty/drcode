"""The image-attachment UI layer (screenshots): staging chips, the paste
path shortcut, the file dialog and the submit wiring into manager.send /
manager.create — strictly on top of the FR #407e1959 contract (the bytes
land in `.drcode/tasks/<id>/attachments/` via Storage.save_attachment, the
references ride PendingInput.attachments)."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Static

from drcode.models import TaskStatus
from drcode.storage import Storage
from drcode.tui.app import DrCodeApp
from drcode.tui.attach_dialog import AttachPathScreen
from drcode.tui.attachments import StagedImage
from drcode.tui.inputs import ChatInput
from drcode.tui.state import DrCodeState

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-image-data-for-tests"
JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg-data"


@pytest.fixture
def app(project_root: Path) -> DrCodeApp:
    return DrCodeApp(project_root)


def _staged(name: str = "shot.png", data: bytes = PNG_BYTES) -> StagedImage:
    media = "image/png" if name.endswith(".png") else "image/jpeg"
    return StagedImage(filename=name, media_type=media, data=data)


# -- staging + chips -------------------------------------------------------------


async def test_stage_in_form_mode_updates_chip(app: DrCodeApp) -> None:
    # No tasks: the app starts in the new-task form, so staging lands in
    # the form's bucket.
    async with app.run_test():
        app.stage_attachment(_staged())
        assert len(app._form_attachments) == 1
        assert app._chat_attachments == []
        chip = app.query_one("#nt-attachments", Static)
        assert not chip.has_class("hidden")
        assert "shot.png" in str(chip.render())
        # The same bytes staged twice are ignored (no double attach).
        app.stage_attachment(_staged())
        assert len(app._form_attachments) == 1


async def test_chat_mode_stage_updates_chat_chip(app: DrCodeApp) -> None:
    app.manager.create("T", "hello")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app._new_task_mode
        app.stage_attachment(_staged())
        assert len(app._chat_attachments) == 1
        chip = app.query_one("#chat-attachments", Static)
        assert not chip.has_class("hidden")


# -- chat submit path --------------------------------------------------------------


async def test_chat_submit_saves_attachments(
    app: DrCodeApp, project_root: Path
) -> None:
    task = app.manager.create("T", "hello")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.stage_attachment(_staged())
        app.chat_input.insert("look at this")
        app.submit_chat_input()
        await pilot.pause()
        entry = task.pending_input[-1]
        assert entry.text == "look at this"
        assert len(entry.attachments) == 1
        att = entry.attachments[0]
        # The bytes live under the task's attachment directory; the
        # reference in the metadata only names the file.
        path = Storage(project_root).attachment_root(task.id) / att.filename
        assert path.read_bytes() == PNG_BYTES
        assert app._chat_attachments == []
        assert app.query_one("#chat-attachments", Static).has_class("hidden")


async def test_image_only_message_is_sendable(app: DrCodeApp) -> None:
    task = app.manager.create("T", "hello")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.stage_attachment(_staged())
        app.submit_chat_input()  # empty text, one staged image
        await pilot.pause()
        entry = task.pending_input[-1]
        assert entry.text == ""
        assert len(entry.attachments) == 1


async def test_duplicate_text_without_images_still_dropped(app: DrCodeApp) -> None:
    task = app.manager.create("T", "hello")
    app.manager.send(task.id, "hello")
    async with app.run_test() as pilot:
        await pilot.pause()
        before = len(task.pending_input)
        app.chat_input.insert("hello")
        app.submit_chat_input()
        await pilot.pause()
        assert len(task.pending_input) == before


async def test_duplicate_text_with_new_image_merges(
    app: DrCodeApp, project_root: Path
) -> None:
    task = app.manager.create("T", "hello")
    app.manager.send(task.id, "hello")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.stage_attachment(_staged())
        app.chat_input.insert("hello")
        app.submit_chat_input()
        await pilot.pause()
        # No second entry — the reference is merged into the queued one.
        assert len(task.pending_input) == 1
        assert len(task.pending_input[0].attachments) == 1
        root = Storage(project_root).attachment_root(task.id)
        assert any(
            (root / a.filename).exists() for a in task.pending_input[0].attachments
        )


# -- new-task form path ---------------------------------------------------------------


async def test_form_create_carries_attachments(
    app: DrCodeApp, project_root: Path
) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._new_task_mode
        app.stage_attachment(_staged("shot.png"))
        app.stage_attachment(_staged("photo.jpg", JPEG_BYTES))
        app.new_task_input.insert("do things")
        app.submit_new_task()
        await pilot.pause()
        task = app.manager.all_tasks()[0]
        assert task.status is not TaskStatus.DRAFT
        entry = task.pending_input[0]
        assert entry.text == "do things"
        assert [a.media_type for a in entry.attachments] == [
            "image/png",
            "image/jpeg",
        ]
        root = Storage(project_root).attachment_root(task.id)
        for att in entry.attachments:
            assert (root / att.filename).read_bytes() in (PNG_BYTES, JPEG_BYTES)
        # The form session's staged images are consumed.
        assert app._form_attachments == []


async def test_form_cancel_drops_staged(app: DrCodeApp) -> None:
    async with app.run_test() as pilot:
        await pilot.pause()
        app.stage_attachment(_staged())
        app.cancel_new_task()
        await pilot.pause()
        assert app._form_attachments == []
        assert app.query_one("#nt-attachments", Static).has_class("hidden")


# -- paste / file dialog entries --------------------------------------------------------


async def test_paste_of_image_path_stages_not_inserts(
    app: DrCodeApp, project_root: Path
) -> None:
    image = project_root / "dropped.png"
    image.write_bytes(PNG_BYTES)
    app.manager.create("T", "hello")
    async with app.run_test() as pilot:
        await pilot.pause()
        from textual import events

        await app.chat_input._on_paste(events.Paste(text=str(image)))
        await pilot.pause()
        assert len(app._chat_attachments) == 1
        # The path text never lands in the editor.
        assert app.chat_input.text == ""


async def test_paste_of_plain_text_stays_text(app: DrCodeApp) -> None:
    app.manager.create("T", "hello")
    async with app.run_test() as pilot:
        await pilot.pause()
        from textual import events

        await app.chat_input._on_paste(events.Paste(text="just words here"))
        await pilot.pause()
        assert app._chat_attachments == []
        assert app.chat_input.text == "just words here"


async def test_attach_path_dialog_result_stages(
    app: DrCodeApp, project_root: Path
) -> None:
    image = project_root / "picked.jpg"
    image.write_bytes(JPEG_BYTES)
    app.manager.create("T", "hello")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._on_attach_path(str(image))
        assert len(app._chat_attachments) == 1
        assert app._chat_attachments[0].media_type == "image/jpeg"
        # A bad path is refused with a toast, nothing staged.
        app._on_attach_path(str(project_root / "nope.png"))
        assert len(app._chat_attachments) == 1


async def test_attach_file_dialog_opens(app: DrCodeApp) -> None:
    app.manager.create("T", "hello")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.attach_file_dialog()
        await pilot.pause()
        assert isinstance(app.screen, AttachPathScreen)


# -- protocol surface ------------------------------------------------------------------


async def test_editors_expose_attach_actions(app: DrCodeApp) -> None:
    app.manager.create("T", "hello")
    async with app.run_test() as pilot:
        await pilot.pause()
        # The actions exist on the state protocol both editors target.
        state: DrCodeState = app
        assert callable(state.stage_attachment)
        assert callable(state.attach_clipboard_image)
        assert callable(state.attach_file_dialog)
        chat = app.query_one("#chat-input", ChatInput)
        keys = {b[0] if isinstance(b, tuple) else b.key for b in chat.BINDINGS}
        assert "ctrl+v" in keys and "ctrl+j" in keys
