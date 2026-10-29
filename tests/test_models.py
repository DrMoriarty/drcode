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


def test_title_display_collapses_control_characters() -> None:
    # A literal tab in a stored title made Textual expand it to the next
    # tab stop in the sidebar, wrapping the row onto two lines.
    raw = "проверь, почему упал s9_research_count\tdeepseek-v4-baseline"
    task = Task(title=raw)
    collapsed = " ".join(raw.split())
    assert "\t" not in task.title_display
    assert task.title_display == collapsed[:39] + "…"
    # Whitespace collapsing happens before the 40-char cut, so the result
    # never exceeds the single-row budget.
    assert len(Task(title="a\t" + "x" * 50).title_display) == 40


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
    assert TaskKind.RESEARCH.value == "research"


def test_verification_labels() -> None:
    assert VerificationStatus.PASSED.value == "Verified successfully"
    assert VerificationStatus.FAILED.value == "Verification failed"


def test_task_event_defaults() -> None:
    event = TaskEvent(text="boom")
    assert event.kind == "system"
    assert event.at is not None


def test_clarification_options_default_empty() -> None:
    assert Clarification(question="q?").options == []


def test_clarification_agent_proposals_default_empty() -> None:
    """Old task JSONs without agent_proposals validate as an empty list —
    no migration needed."""
    assert Clarification(question="q?").agent_proposals == []
    legacy = Clarification.model_validate({"question": "q?", "options": ["a"]})
    assert legacy.agent_proposals == []
