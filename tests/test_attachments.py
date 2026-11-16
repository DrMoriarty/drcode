"""Image attachments (screenshots) on PendingInput: storage references,
multimodal consumption helpers and the silent-drop fixes across the
transcript / compaction / context-layout readers.

The image BYTES never live in the task metadata — `ImageAttachment` is a
reference to a file under `.drcode/tasks/<id>/attachments/` (see
drcode.storage); the consumer turns it into a pydantic-ai `BinaryContent`
item of a multimodal user prompt. Old task JSONs (no attachments field)
load unchanged — the field is optional.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from drcode.compaction import (
    _user_prompt_parts,
    compacted_messages,
    render_history_blocks,
    split_original_prompts,
)
from drcode.context_layout import IMAGE_URL_ESTIMATE_TOKENS, analyze_transcript
from drcode.models import ImageAttachment, PendingInput, Task
from drcode.storage import Storage
from drcode.task_manager import TaskManager
from drcode.transcript import (
    all_user_prompts,
    backdate_user_prompt,
    extract_display,
    first_user_prompt,
    is_pending_retry,
    prompt_image_placeholders,
    prompt_text_element,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-image-data-for-tests"


def _binary_item(media_type: str = "image/png", data: str = "aGVsbG8=") -> dict:
    """Serialized shape of a pydantic-ai BinaryContent inside a
    UserPromptPart content list (data is base64 by val_json_bytes)."""
    return {"kind": "binary", "media_type": media_type, "data": data}


def _multimodal_message(text: str, *items: dict, at: str | None = None) -> dict:
    part: dict = {"part_kind": "user-prompt", "content": [text, *items]}
    if at:
        part["timestamp"] = at
    return {
        "kind": "request",
        "parts": [part],
        "timestamp": at or "2025-01-01T00:00:00+00:00",
    }


# -- storage -------------------------------------------------------------------


def test_attachment_roundtrip(project_root: Path) -> None:
    storage = Storage(project_root)
    att = storage.save_attachment("t1", PNG_BYTES, "image/png")
    assert att.filename.endswith(".png")
    assert att.media_type == "image/png"
    path = storage.attachment_root("t1") / att.filename
    assert path.read_bytes() == PNG_BYTES
    assert storage.read_attachment("t1", att) == PNG_BYTES


def test_read_attachment_missing_is_none(project_root: Path) -> None:
    storage = Storage(project_root)
    att = ImageAttachment(filename="gone.png", media_type="image/png")
    assert storage.read_attachment("t1", att) is None


def test_delete_attachments_and_task_delete(project_root: Path) -> None:
    storage = Storage(project_root)
    att = storage.save_attachment("t1", PNG_BYTES, "image/png")
    assert storage.attachment_root("t1").exists()
    storage.delete_attachments("t1")
    assert not storage.attachment_root("t1").exists()
    assert storage.read_attachment("t1", att) is None

    # Task metadata + the whole per-task directory go together.
    task = Task(title="t2")
    att2 = storage.save_attachment(task.id, PNG_BYTES, "image/png")
    task.pending_input.append(PendingInput(text="hi", attachments=[att2]))
    storage.save(task)
    storage.delete(task.id)
    assert not (storage.tasks_dir / task.id).exists()


# -- models: persistence shape -------------------------------------------------


def test_pending_input_roundtrip_with_attachments() -> None:
    att = ImageAttachment(filename="shot.png", media_type="image/png")
    entry = PendingInput(
        text="look", at=datetime(2025, 1, 1, tzinfo=UTC), attachments=[att]
    )
    raw = json.loads(entry.model_dump_json())
    restored = PendingInput.model_validate(raw)
    assert restored.attachments == [att]
    # No base64 anywhere in the metadata shape.
    assert "aGVsbG8" not in raw["text"] and "data" not in raw["attachments"][0]


def test_pending_input_legacy_shape_loads() -> None:
    legacy = {"text": "plain", "at": "2025-01-01T00:00:00Z"}
    entry = PendingInput.model_validate(legacy)
    assert entry.text == "plain"
    assert entry.attachments == []


# -- task manager: send() threading ---------------------------------------------


def test_send_carries_attachments(manager: TaskManager, project_root: Path) -> None:
    storage = Storage(project_root)
    task = manager.create("T", "first")
    # save_attachment is caller-owned: the file must already live under the
    # TASK's attachment directory before send() is called.
    att2 = storage.save_attachment(task.id, PNG_BYTES, "image/png")

    manager.send(task.id, "look at this", [att2])
    entry = task.pending_input[-1]
    assert entry.text == "look at this"
    assert entry.attachments == [att2]
    assert (storage.attachment_root(task.id) / att2.filename).exists()


def test_send_duplicate_text_dropped_without_attachments(manager: TaskManager) -> None:
    task = manager.create("T", "first")
    manager.send(task.id, "first")
    before = len(task.pending_input)
    # Same text re-send: dropped — a plain re-send of the same package.
    manager.send(task.id, "first")
    assert len(task.pending_input) == before


def test_send_duplicate_text_merges_new_attachments(manager: TaskManager) -> None:
    task = manager.create("T", "first")
    att = ImageAttachment(filename="a.png", media_type="image/png")
    manager.send(task.id, "first", [att])
    assert len(task.pending_input) == 1
    # Same text WITH a new image reference is not a pure re-send: the
    # reference is merged into the still-queued entry (the bytes are
    # caller-saved on disk before the call).
    att2 = ImageAttachment(filename="b.png", media_type="image/png")
    manager.send(task.id, "first", [att2])
    assert len(task.pending_input) == 1
    assert task.pending_input[0].attachments == [att, att2]
    # Re-merging an already-present filename is a no-op (no duplicates).
    manager.send(task.id, "first", [ImageAttachment(filename="b.png")])
    assert task.pending_input[0].attachments == [att, att2]


def test_send_retry_prompt_match_drops_attachments(manager: TaskManager) -> None:
    # A text that sits in the transcript as the prompt of an unfinished
    # attempt (a request with no final response after it) cannot take
    # attachments: the prompt is consumed already — the duplicate branch
    # drops the references, nothing is merged.
    task = manager.create("T", "first")
    task.messages = [
        {
            "kind": "request",
            "parts": [{"part_kind": "user-prompt", "content": "retry me"}],
        }
    ]
    manager.storage.save(task)
    att = ImageAttachment(filename="c.png", media_type="image/png")
    before = len(task.pending_input)
    manager.send(task.id, "retry me", [att])
    assert len(task.pending_input) == before
    assert all(e.attachments == [] for e in task.pending_input)


# -- transcript helpers ---------------------------------------------------------


def test_prompt_text_element_and_placeholders() -> None:
    assert prompt_text_element("plain") == "plain"
    content = [
        "look",
        _binary_item("image/png"),
        {"kind": "image-url", "url": "http://x/y.png", "media_type": "image/png"},
    ]
    assert prompt_text_element(content) == "look"
    assert prompt_image_placeholders(content) == [
        "[image image/png]",
        "[image image/png: http://x/y.png]",
    ]
    assert prompt_text_element([_binary_item()]) is None
    assert prompt_image_placeholders("plain") == []


def test_extract_display_renders_placeholder_not_base64() -> None:
    msg = _multimodal_message("fix this", _binary_item("image/png", "T0" * 5000))
    out = extract_display([msg])
    user_blocks = [text for _, role, text in out if role == "user"]
    assert len(user_blocks) == 1
    assert "fix this" in user_blocks[0]
    assert "[image image/png]" in user_blocks[0]
    assert "T0" * 10 not in user_blocks[0]  # base64 never leaks into the chat


def test_extract_display_image_only_prompt() -> None:
    msg = (
        _multimodal_message("", _binary_item())
        if False
        else {
            "kind": "request",
            "parts": [{"part_kind": "user-prompt", "content": [_binary_item()]}],
        }
    )
    out = extract_display([msg])
    assert [text for _, role, text in out if role == "user"] == ["[image image/png]"]


def test_first_and_all_user_prompts_multimodal() -> None:
    messages = [
        _multimodal_message("original", _binary_item("image/png")),
        _multimodal_message("follow-up", _binary_item("image/jpeg")),
    ]
    assert first_user_prompt(messages) == "original"
    assert all_user_prompts(messages) == [
        "original\n\n[image image/png]",
        "follow-up\n\n[image image/jpeg]",
    ]


def test_backdate_user_prompt_matches_multimodal() -> None:
    from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart

    from drcode.transcript import load_messages, serialize_messages

    request = ModelRequest(
        parts=[
            UserPromptPart(
                content=["fix this", BinaryContent(data=b"x", media_type="image/png")]
            )
        ]
    )
    loaded: list = load_messages(serialize_messages([request]))
    at = datetime(2025, 3, 4, 5, 6, 7, tzinfo=UTC)
    backdate_user_prompt(loaded, "fix this", at)
    req = next(m for m in loaded if isinstance(m, ModelRequest))
    assert req.parts[0].timestamp == at  # type: ignore[attr-defined]
    assert req.timestamp == at


def test_is_pending_retry_matches_multimodal() -> None:
    messages = [_multimodal_message("redo", _binary_item())]
    assert is_pending_retry(messages, "redo")


# -- compaction -----------------------------------------------------------------


def test_user_prompt_parts_recognizes_multimodal() -> None:
    msg = _multimodal_message("text", _binary_item())
    parts = _user_prompt_parts(msg)
    assert len(parts) == 1


def test_split_original_prompts_strips_images_keeps_text() -> None:
    messages = [
        _multimodal_message("original", _binary_item("image/png", "QQ" * 1000)),
        {"kind": "response", "parts": [{"part_kind": "text", "content": "ok"}]},
    ]
    preserved, rest = split_original_prompts(messages)
    assert len(preserved) == 1
    part = preserved[0]["parts"][0]
    assert part["content"] == ["original"]  # image item stripped, text verbatim
    assert len(rest) == 1  # the response stays in `rest` for the summarizer


def test_split_original_prompts_drops_image_only_prompt() -> None:
    messages = [
        {
            "kind": "request",
            "parts": [{"part_kind": "user-prompt", "content": [_binary_item()]}],
        },
        {"kind": "response", "parts": [{"part_kind": "text", "content": "ok"}]},
    ]
    preserved, rest = split_original_prompts(messages)
    assert preserved == []
    # The image-only request leaves nothing at all: dropped whole (an
    # empty ModelRequest is not API-valid), the response stays in `rest`.
    assert len(rest) == 1


def test_compacted_messages_carry_no_base64() -> None:
    messages = [
        _multimodal_message("original", _binary_item("image/png", "QQ" * 20000)),
        {"kind": "response", "parts": [{"part_kind": "text", "content": "ok"}]},
    ]
    out = compacted_messages(messages, "summary body")
    assert "QQ" * 100 not in json.dumps(out)


def test_render_history_blocks_summarizer_placeholders() -> None:
    messages = [_multimodal_message("look", _binary_item("image/png", "QQ" * 5000))]
    blocks = render_history_blocks(messages)
    assert blocks == [("user", "[user]\nlook\n[image image/png]")]


# -- context layout --------------------------------------------------------------


def test_analyze_transcript_counts_image_payload() -> None:
    from drcode.context_layout import estimate_tokens

    payload = "QQ" * 8000  # 16k chars of base64
    messages = [_multimodal_message("look", _binary_item("image/png", payload))]
    stats = analyze_transcript(messages)
    assert stats.user_prompt_count == 1
    assert stats.user_prompts == estimate_tokens("look") + estimate_tokens(payload)


def test_analyze_transcript_url_image_stand_in() -> None:
    messages = [
        _multimodal_message(
            "look",
            {"kind": "image-url", "url": "http://x/y.png", "media_type": "image/png"},
        )
    ]
    stats = analyze_transcript(messages)
    assert stats.user_prompt_count == 1
    assert stats.user_prompts >= IMAGE_URL_ESTIMATE_TOKENS
