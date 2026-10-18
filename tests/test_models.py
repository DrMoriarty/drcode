"""Tests for drcode.models: Task fields, helpers and round-tripping."""

from __future__ import annotations

import json

from drcode.models import (
    Clarification,
    PendingInput,
    Task,
    TaskEvent,
    TaskKind,
    TaskStatus,
    VerificationStatus,
)


def test_task_defaults() -> None:
    task = Task(title="Fix thing")
    assert task.id and len(task.id) == 8
    assert task.kind is TaskKind.FEATURE
    assert task.status is TaskStatus.QUEUED
    assert task.verification is None
    assert task.messages == []
    assert task.tester_messages == []
    assert task.pending_input == []
    assert task.completed_at is None
    assert task.unfinished is False
    assert task.clarification is None
    assert task.events == []
    assert task.queue_head is False
    assert task.queue_seq is None
    assert task.created_at is not None
    assert task.updated_at is not None


def test_task_touch_updates_updated_at() -> None:
    task = Task(title="t")
    before = task.updated_at
    task.touch()
    assert task.updated_at >= before


def test_title_display_short() -> None:
    assert Task(title="short").title_display == "short"


def test_title_display_long_truncated_with_ellipsis() -> None:
    title = "x" * 50
    displayed = Task(title=title).title_display
    assert len(displayed) == 40
    assert displayed.endswith("…")
    assert displayed.startswith("x" * 39)


def test_task_json_roundtrip() -> None:
    task = Task(
        title="roundtrip",
        kind=TaskKind.BUG,
        status=TaskStatus.TESTING,
        verification=VerificationStatus.FAILED,
        pending_input=[PendingInput(text="hello")],
        unfinished=True,
        clarification=Clarification(question="Which one?", options=["a", "b"]),
        events=[TaskEvent(kind="permission", text="denied")],
        queue_head=True,
        queue_seq=7,
    )
    data = json.loads(task.model_dump_json())
    restored = Task.model_validate(data)
    assert restored == task


def test_status_and_kind_values() -> None:
    # These string values are persisted on disk — must stay stable.
    assert TaskStatus.QUEUED.value == "queued"
    assert TaskStatus.HOLD.value == "hold"
    assert TaskStatus.ACTIVE.value == "active"
    assert TaskStatus.TESTING.value == "testing"
    assert TaskStatus.COMPLETED.value == "completed"
    assert TaskStatus.AWAITING_CLARIFICATION.value == "clarification"
    assert TaskKind.BUG.value == "bug"
    assert TaskKind.FEATURE.value == "feature"
    assert TaskKind.DOCUMENTATION.value == "documentation"


def test_verification_labels() -> None:
    assert VerificationStatus.PASSED.value == "Проверено успешно"
    assert VerificationStatus.FAILED.value == "Проверено с ошибками"


def test_task_event_defaults() -> None:
    event = TaskEvent(text="boom")
    assert event.kind == "system"
    assert event.at is not None


def test_clarification_options_default_empty() -> None:
    assert Clarification(question="q?").options == []
