"""Tests for drcode.task_manager: the full task state machine."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from pydantic_ai import ModelRequest, ModelResponse
from pydantic_ai.messages import TextPart, UserPromptPart

from drcode.agent import serialize_messages
from drcode.models import (
    PendingInput,
    Task,
    TaskKind,
    TaskStatus,
    VerificationStatus,
    _now,
)
from drcode.storage import Storage
from drcode.task_manager import TaskManager

# -- creation -----------------------------------------------------------------


def test_create_basic(manager: TaskManager, project_root: Path) -> None:
    task = manager.create("My title", "Do the thing")
    assert task.title == "My title"
    assert task.status is TaskStatus.QUEUED
    assert [e.text for e in task.pending_input] == ["Do the thing"]
    assert task.queue_seq == 1
    assert (project_root / ".drcode" / "tasks" / f"{task.id}.json").is_file()


def test_create_title_from_first_message(manager: TaskManager) -> None:
    task = manager.create("", "First line of the task\nsecond line")
    assert task.title == "First line of the task"[:60]


def test_create_title_fallback(manager: TaskManager) -> None:
    task = manager.create("", "")
    assert task.title == "New task"
    assert task.pending_input == []


def test_create_bug_kind(manager: TaskManager) -> None:
    task = manager.create("Bug", "desc", kind=TaskKind.BUG)
    assert task.kind is TaskKind.BUG


# -- queue ordering -------------------------------------------------------------


def test_queued_fifo_by_default(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    c = manager.create("c", "x")
    assert manager.queued() == [a, b, c]


def test_bugs_go_to_head_newest_first(manager: TaskManager) -> None:
    a = manager.create("feature1", "x")
    b = manager.create("feature2", "x")
    bug1 = manager.create("bug1", "x", kind=TaskKind.BUG)
    bug2 = manager.create("bug2", "x", kind=TaskKind.BUG)
    assert manager.queued() == [bug2, bug1, a, b]


def test_answered_clarification_goes_before_regular(manager: TaskManager) -> None:
    from drcode.models import Clarification

    a = manager.create("a", "x")
    b = manager.create("b", "x")
    # b asked a question and the user answers it → head of non-bug tasks.
    b.status = TaskStatus.AWAITING_CLARIFICATION
    b.clarification = Clarification(question="q?")
    manager.send(b.id, "the answer")
    assert manager.queued() == [b, a]


def test_held_tasks_skipped_in_queue(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    manager.create("b", "x")
    manager.set_hold(a.id, on=True)
    titles = [t.title for t in manager.queued()]
    assert "a" not in titles


# -- send() -------------------------------------------------------------------


def test_send_to_queued_accumulates(manager: TaskManager) -> None:
    task = manager.create("t", "first")
    manager.send(task.id, "second")
    assert [e.text for e in task.pending_input] == ["first", "second"]
    assert task.status is TaskStatus.QUEUED


def test_send_to_completed_requeues_and_clears_verification(
    manager: TaskManager,
) -> None:
    task = manager.create("t", "x")
    manager.send(task.id, "more")
    manager.promote_next()
    task.verification = VerificationStatus.PASSED
    task.status = TaskStatus.COMPLETED
    task.completed_at = _now()

    manager.send(task.id, "again")
    assert task.status is TaskStatus.QUEUED
    assert task.verification is None
    assert task.completed_at is None
    assert [e.text for e in task.pending_input] == ["x", "more", "again"]


def test_send_to_testing_requeues(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    manager.mark_done(task.id)
    assert task.status is TaskStatus.TESTING
    manager.send(task.id, "rework")
    assert task.status is TaskStatus.QUEUED


def test_send_to_awaiting_clarification_sets_queue_head(
    manager: TaskManager,
) -> None:
    task = manager.create("t", "x")
    task.status = TaskStatus.AWAITING_CLARIFICATION
    from drcode.models import Clarification

    task.clarification = Clarification(question="q?")

    manager.send(task.id, "my answer")
    assert task.status is TaskStatus.QUEUED
    assert task.queue_head is True
    assert task.clarification is None
    assert [e.text for e in task.pending_input] == ["x", "my answer"]


def test_send_strips_whitespace(manager: TaskManager) -> None:
    task = manager.create("t", "  padded  ")
    assert [e.text for e in task.pending_input] == ["padded"]


def test_send_ignores_duplicate_of_queued_text(manager: TaskManager) -> None:
    """A re-send of a text that is still sitting in pending_input is a
    no-op: the user re-typing a comment the busy pipeline has not visibly
    reacted to must not queue (and later render/send) the message twice."""
    task = manager.create("t", "x")
    manager.send(task.id, "comment")
    manager.send(task.id, "comment")
    assert [e.text for e in task.pending_input] == ["x", "comment"]


def test_send_ignores_duplicate_of_in_flight_prompt(manager: TaskManager) -> None:
    """A re-send while the identical text is already the transcript's
    unfinished prompt (a run is processing it) is a no-op: the prompt
    must not reach the model twice."""
    task = manager.create("t", "x")
    manager.promote_next()
    task.messages = serialize_messages(
        [ModelRequest(parts=[UserPromptPart(content="comment")])]
    )
    task.pending_input.clear()
    manager.send(task.id, "comment")
    assert task.pending_input == []
    assert task.status is TaskStatus.ACTIVE


def test_send_repeat_after_completion_requeues(manager: TaskManager) -> None:
    """A text whose transcript attempt is FINISHED (answered long ago) is
    not a duplicate: re-sending it after completion re-queues the task —
    only the still-unfinished/in-flight copies are deduplicated."""
    task = manager.create("t", "x")
    # The run consumed the original brief long ago.
    task.pending_input.clear()
    task.messages = serialize_messages(
        [
            ModelRequest(parts=[UserPromptPart(content="comment")]),
            ModelResponse(parts=[TextPart(content="done")]),
        ]
    )
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    task.completed_at = _now()

    manager.send(task.id, "comment")
    assert task.status is TaskStatus.QUEUED
    assert task.verification is None
    assert [e.text for e in task.pending_input] == ["comment"]


# -- set_kind -----------------------------------------------------------------


def test_set_kind_bug_moves_to_head(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    manager.set_kind(b.id, TaskKind.BUG)
    assert manager.queued() == [b, a]
    assert b.queue_head is True


def test_set_kind_feature_moves_to_tail(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    manager.set_kind(a.id, TaskKind.BUG)  # head
    c = manager.create("c", "x")
    manager.set_kind(a.id, TaskKind.FEATURE)  # back to tail, after c
    assert manager.queued() == [b, c, a]


def test_set_kind_research_keeps_position(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    manager.set_kind(b.id, TaskKind.RESEARCH)
    assert manager.queued() == [a, b]


def test_set_kind_same_kind_is_noop(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    seq_before = b.queue_seq
    manager.set_kind(b.id, TaskKind.FEATURE)
    assert manager.queued() == [a, b]
    assert b.queue_seq == seq_before


# -- move_in_queue --------------------------------------------------------------


def test_move_down_swaps_with_execution_neighbor(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    c = manager.create("c", "x")
    # Down the sidebar = closer to execution = one step toward the queue head.
    assert manager.move_in_queue(c.id, up=False) is c
    assert manager.queued() == [a, c, b]


def test_move_up_swaps_away_from_execution(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    c = manager.create("c", "x")
    assert manager.move_in_queue(a.id, up=True) is a
    assert manager.queued() == [b, a, c]


def test_move_at_edges_is_noop(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    assert manager.move_in_queue(a.id, up=False) is None  # already next to run
    assert manager.move_in_queue(b.id, up=True) is None  # already the last
    assert manager.queued() == [a, b]


def test_move_non_queued_is_noop(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    c = manager.create("c", "x")
    a.status = TaskStatus.ACTIVE
    assert manager.move_in_queue(a.id, up=True) is None
    assert manager.move_in_queue(b.id, up=True) is b  # unaffected
    assert manager.queued() == [c, b]


def test_manual_order_survives_restart(project_root: Path) -> None:
    manager = TaskManager(project_root)
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    c = manager.create("c", "x")
    manager.move_in_queue(c.id, up=False)  # [a, c, b]

    reloaded = TaskManager(project_root)
    assert reloaded.queued() == [a, c, b]


def test_new_tasks_follow_default_algorithms_after_manual_move(
    manager: TaskManager,
) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    manager.move_in_queue(b.id, up=False)  # [b, a]
    # A new feature still appends at the tail (farthest from execution).
    c = manager.create("c", "x")
    assert manager.queued() == [b, a, c]
    # A new bug still jumps to the very head (closest to execution).
    bug = manager.create("bug", "x", kind=TaskKind.BUG)
    assert manager.queued() == [bug, b, a, c]


def test_move_cannot_cross_bug_block(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    bug = manager.create("bug", "x", kind=TaskKind.BUG)
    # Moving the feature down would put it ahead of the bug — not allowed.
    assert manager.move_in_queue(a.id, up=False) is None
    assert manager.queued() == [bug, a]
    # Moving the bug up would demote it below the feature — not allowed either.
    assert manager.move_in_queue(bug.id, up=True) is None
    assert manager.queued() == [bug, a]
    # But within the bug block (and within the rest) moves work.
    bug2 = manager.create("bug2", "x", kind=TaskKind.BUG)
    assert manager.queued() == [bug2, bug, a]
    assert manager.move_in_queue(bug2.id, up=True) is bug2
    assert manager.queued() == [bug, bug2, a]


def test_move_repositions_answered_clarification_marker(
    manager: TaskManager,
) -> None:
    from drcode.models import Clarification

    a = manager.create("a", "x")
    b = manager.create("b", "x")
    b.status = TaskStatus.AWAITING_CLARIFICATION
    b.clarification = Clarification(question="q?")
    manager.send(b.id, "answer")  # [b, a] — answered clarification at the head
    manager.move_in_queue(b.id, up=True)  # [a, b]
    assert manager.queued() == [a, b]
    # A later answered clarification still lands ahead of the non-bug block.
    a.status = TaskStatus.AWAITING_CLARIFICATION
    a.clarification = Clarification(question="q2?")
    manager.send(a.id, "answer 2")
    assert manager.queued() == [a, b]


# -- hold / delete ------------------------------------------------------------


def test_hold_and_resume(manager: TaskManager) -> None:
    task = manager.create("t", "input text")
    manager.set_hold(task.id, on=True)
    assert task.status is TaskStatus.HOLD
    assert [e.text for e in task.pending_input] == ["input text"]

    manager.set_hold(task.id, on=False)
    assert task.status is TaskStatus.QUEUED
    assert [e.text for e in task.pending_input] == ["input text"]


def test_hold_awaiting_clarification_and_resume_back(manager: TaskManager) -> None:
    from drcode.models import Clarification

    task = manager.create("t", "")  # no pending input: the question is still open
    task.status = TaskStatus.AWAITING_CLARIFICATION
    task.clarification = Clarification(question="q?")

    manager.set_hold(task.id, on=True)
    assert task.status is TaskStatus.HOLD

    manager.set_hold(task.id, on=False)
    assert task.status is TaskStatus.AWAITING_CLARIFICATION
    assert task.clarification is not None


def test_resume_held_task_answered_while_held(manager: TaskManager) -> None:
    from drcode.models import Clarification

    task = manager.create("t", "x")
    task.status = TaskStatus.AWAITING_CLARIFICATION
    task.clarification = Clarification(question="q?")
    manager.set_hold(task.id, on=True)

    # The user answers while the task is parked.
    manager.send(task.id, "the answer")

    manager.set_hold(task.id, on=False)
    assert task.status is TaskStatus.QUEUED
    assert task.queue_head is True
    assert task.clarification is None


def test_hold_non_holdable_status_ignored(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    manager.mark_done(task.id)  # TESTING
    manager.set_hold(task.id, on=True)
    assert task.status is TaskStatus.TESTING


def test_resume_not_held_is_noop(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    manager.set_hold(task.id, on=False)
    assert task.status is TaskStatus.QUEUED


def test_delete_removes_from_memory_and_disk(
    manager: TaskManager, project_root: Path
) -> None:
    task = manager.create("t", "x")
    path = project_root / ".drcode" / "tasks" / f"{task.id}.json"
    manager.set_hold(task.id, on=True)
    manager.delete(task.id)
    assert task.id not in manager.tasks
    assert not path.exists()


# -- mark_done / verification ---------------------------------------------------


def test_mark_done_feature_goes_to_testing(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    manager.mark_done(task.id)
    assert task.status is TaskStatus.TESTING
    assert task.completed_at is None


def test_mark_done_research_completes(manager: TaskManager) -> None:
    task = manager.create("t", "x", kind=TaskKind.RESEARCH)
    manager.mark_done(task.id)
    assert task.status is TaskStatus.COMPLETED
    assert task.completed_at is not None


# -- promote_next ----------------------------------------------------------------


def test_promote_next_activates_first_with_input(manager: TaskManager) -> None:
    a = manager.create("a", "input")
    b = manager.create("b", "input")
    promoted = manager.promote_next()
    assert promoted is a
    assert a.status is TaskStatus.ACTIVE
    assert b.status is TaskStatus.QUEUED


def test_promote_next_skips_tasks_without_work(manager: TaskManager) -> None:
    empty = manager.create("empty", "")
    empty.pending_input.clear()
    worker = manager.create("worker", "input")
    assert manager.promote_next() is worker


def test_promote_next_resumes_unfinished(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    task.pending_input.clear()
    task.unfinished = True
    assert manager.promote_next() is task


def test_promote_next_noop_when_active_exists(manager: TaskManager) -> None:
    manager.create("a", "input")
    manager.promote_next()
    manager.create("b", "input")
    assert manager.promote_next() is None


def test_promote_next_none_when_queue_empty(manager: TaskManager) -> None:
    assert manager.promote_next() is None


def test_promote_next_exclude_set(manager: TaskManager) -> None:
    a = manager.create("a", "input")
    b = manager.create("b", "input")
    assert manager.promote_next(exclude={a.id}) is b


def test_promote_next_clears_queue_head(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    task.queue_head = True
    manager.promote_next()
    assert task.queue_head is False


def test_promote_next_bugs_first_even_without_queue_head(manager: TaskManager) -> None:
    manager.create("feature", "input")
    bug = manager.create("bug", "input", kind=TaskKind.BUG)
    assert manager.promote_next() is bug


# -- cancel --------------------------------------------------------------------


def test_cancel_completes_and_drops_input(manager: TaskManager) -> None:
    task = manager.create("t", "do stuff")
    task.unfinished = True
    from drcode.models import Clarification

    task.clarification = Clarification(question="q?")
    task.queue_head = True

    manager.cancel(task.id)
    assert task.status is TaskStatus.COMPLETED
    assert task.completed_at is not None
    assert task.pending_input == []
    assert task.unfinished is False
    assert task.clarification is None
    assert task.queue_head is False


# -- requeue_answered_clarification ----------------------------------------------


def test_requeue_answered_clarification(manager: TaskManager) -> None:
    from drcode.models import Clarification

    task = manager.create("t", "x")
    # State the helper targets: awaiting clarification, but the answer has
    # already arrived in pending_input (typed while the run was finishing).
    task.status = TaskStatus.AWAITING_CLARIFICATION
    task.clarification = Clarification(question="q?")
    task.pending_input.append(PendingInput(text="late answer"))
    manager.requeue_answered_clarification(task.id)
    assert task.status is TaskStatus.QUEUED
    assert task.queue_head is True
    assert task.clarification is None


def test_requeue_answered_clarification_noops(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    # Wrong status.
    manager.requeue_answered_clarification(task.id)
    assert task.status is TaskStatus.QUEUED
    # Right status but no pending answer.
    task.pending_input.clear()
    task.status = TaskStatus.AWAITING_CLARIFICATION
    manager.requeue_answered_clarification(task.id)
    assert task.status is TaskStatus.AWAITING_CLARIFICATION


# -- purge_expired_completed ------------------------------------------------------


def test_purge_disabled_when_retention_none(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    manager.cancel(task.id)
    task.completed_at = _now() - timedelta(days=365)
    assert manager.purge_expired_completed(None) == []
    assert task.id in manager.tasks


def test_purge_removes_old_completed(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    manager.cancel(task.id)
    task.completed_at = _now() - timedelta(days=30)
    removed = manager.purge_expired_completed(7)
    assert [t.id for t in removed] == [task.id]
    assert task.id not in manager.tasks


def test_purge_keeps_recent_completed(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    manager.cancel(task.id)
    task.completed_at = _now() - timedelta(days=1)
    assert manager.purge_expired_completed(7) == []
    assert task.id in manager.tasks


def test_purge_falls_back_to_updated_at(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    manager.cancel(task.id)
    task.completed_at = None  # legacy task without the stamp
    task.updated_at = _now() - timedelta(days=100)
    removed = manager.purge_expired_completed(7)
    assert [t.id for t in removed] == [task.id]


def test_purge_keeps_non_completed(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    task.created_at = _now() - timedelta(days=100)
    task.updated_at = _now() - timedelta(days=100)
    assert manager.purge_expired_completed(7) == []
    assert task.id in manager.tasks


def test_purge_removal_persisted(manager: TaskManager, project_root: Path) -> None:
    task = manager.create("t", "x")
    manager.cancel(task.id)
    task.completed_at = _now() - timedelta(days=30)
    manager.purge_expired_completed(7)
    assert not (project_root / ".drcode" / "tasks" / f"{task.id}.json").exists()
    assert Storage(project_root).load_all() == []


# -- restart behaviour ---------------------------------------------------------


def test_startup_demotes_stale_active(project_root: Path) -> None:
    storage = Storage(project_root)
    task = Task(title="stale", status=TaskStatus.ACTIVE)
    storage.save(task)

    manager = TaskManager(project_root)
    assert manager.get(task.id).status is TaskStatus.QUEUED
    # And it's persisted.
    assert Storage(project_root).load_all()[0].status is TaskStatus.QUEUED


def test_startup_keeps_testing_for_retest(project_root: Path) -> None:
    storage = Storage(project_root)
    task = Task(title="in testing", status=TaskStatus.TESTING)
    storage.save(task)

    manager = TaskManager(project_root)
    assert manager.get(task.id).status is TaskStatus.TESTING


def test_startup_assigns_queue_seq_in_created_order(project_root: Path) -> None:
    storage = Storage(project_root)
    first = Task(title="first")
    second = Task(title="second")
    second.created_at = first.created_at + timedelta(seconds=5)
    storage.save(second)
    storage.save(first)

    manager = TaskManager(project_root)
    assert manager.get(first.id).queue_seq == 1
    assert manager.get(second.id).queue_seq == 2


def test_startup_keeps_max_queue_seq(project_root: Path) -> None:
    storage = Storage(project_root)
    task = Task(title="t", queue_seq=42)
    storage.save(task)

    manager = TaskManager(project_root)
    new = manager.create("new", "x")
    assert new.queue_seq == 43


# -- queries ------------------------------------------------------------------


def test_all_tasks_sorted_by_created_at(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    assert [t.id for t in manager.all_tasks()] == [a.id, b.id]


def test_get_unknown_raises(manager: TaskManager) -> None:
    with pytest.raises(KeyError):
        manager.get("nope")


def test_active_returns_none_when_idle(manager: TaskManager) -> None:
    manager.create("t", "x")
    assert manager.active is None


def test_active_returns_the_active_task(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    manager.promote_next()
    assert manager.active is task
