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


# -- create on hold / edit condition --------------------------------------------


def test_create_on_hold(manager: TaskManager) -> None:
    task = manager.create("parked", "the prompt", hold=True)
    assert task.status is TaskStatus.HOLD
    # The prompt stays queued in pending_input, like a held-after-the-fact
    # task; it must never be promoted while held.
    assert [e.text for e in task.pending_input] == ["the prompt"]
    assert task not in manager.queued()
    assert manager.promote_next() is None

    manager.set_hold(task.id, on=False)
    assert task.status is TaskStatus.QUEUED


def test_editable_condition_merges_comments(manager: TaskManager) -> None:
    task = manager.create("t", "original condition")
    manager.send(task.id, "extra comment 1")
    manager.send(task.id, "extra comment 2")
    assert manager.editable_condition(task.id) == (
        "original condition\n\nextra comment 1\n\nextra comment 2"
    )


def test_edit_task_replaces_condition_and_comments(manager: TaskManager) -> None:
    task = manager.create("t", "original condition")
    manager.send(task.id, "extra comment")
    manager.set_hold(task.id, on=True)

    manager.edit_task(task.id, "rewritten condition\nwith details")
    assert task.status is TaskStatus.HOLD
    # The separate comment is gone — its content is expected to be folded
    # into the new condition by the user.
    assert [e.text for e in task.pending_input] == ["rewritten condition\nwith details"]
    # The title is re-derived from the first line, like on creation.
    assert task.title == "rewritten condition"

    # Back into the queue after the edit.
    manager.set_hold(task.id, on=False)
    assert task.status is TaskStatus.QUEUED
    assert manager.queued() == [task]


def test_edit_task_drops_answered_clarification(manager: TaskManager) -> None:
    from drcode.models import Clarification

    task = manager.create("t", "condition")
    task.status = TaskStatus.AWAITING_CLARIFICATION
    task.clarification = Clarification(question="q?")
    manager.set_hold(task.id, on=True)
    manager.send(task.id, "partial answer")

    manager.edit_task(task.id, "full rewritten condition")
    # The old question and its partial answer were folded into the edited
    # text: the stale question must not resurface on resume.
    assert task.clarification is None
    manager.set_hold(task.id, on=False)
    assert task.status is TaskStatus.QUEUED


def test_edit_task_changes_kind_with_repositioning(manager: TaskManager) -> None:
    old = manager.create("old", "x")
    task = manager.create("held feature", "condition", hold=True)

    manager.edit_task(task.id, "new condition", kind=TaskKind.BUG)
    assert task.kind is TaskKind.BUG
    assert task.queue_head is True

    # Same kind: the edit keeps the task where it is.
    seq = task.queue_seq
    manager.edit_task(task.id, "another rewrite")
    assert task.kind is TaskKind.BUG
    assert task.queue_seq == seq

    # Feature after bug → the tail of the queue (behind the older task).
    manager.edit_task(task.id, "third rewrite", kind=TaskKind.FEATURE)
    assert task.kind is TaskKind.FEATURE
    manager.set_hold(task.id, on=False)
    assert manager.queued() == [old, task]


def test_edit_task_empty_text_is_noop(manager: TaskManager) -> None:
    task = manager.create("t", "condition", hold=True)
    manager.edit_task(task.id, "   ")
    assert [e.text for e in task.pending_input] == ["condition"]
    assert task.status is TaskStatus.HOLD


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


# -- external task pickup (reload_new_tasks) ------------------------------------


def _external_create(project_root: Path, title: str) -> Task:
    """Simulate `drcode --feature-request DESC` from another process: a
    fresh TaskManager (like the CLI builds) writes a new task file."""
    return TaskManager(project_root).create(title, f"body of {title}")


def test_reload_new_tasks_picks_up_external_task(
    manager: TaskManager, project_root: Path
) -> None:
    manager.create("own", "x")
    external = _external_create(project_root, "external")
    assert external.id not in manager.tasks

    added = manager.reload_new_tasks()
    assert [t.id for t in added] == [external.id]
    # The loaded copy is a fresh object (parsed from the peer's file), not
    # the peer's in-memory instance.
    picked_up = manager.get(external.id)
    assert picked_up is not external
    assert picked_up.status is TaskStatus.QUEUED
    assert [e.text for e in picked_up.pending_input] == ["body of external"]
    # Stamped into the queue like a fresh creation: sorts after the
    # existing non-bug tasks.
    assert manager.queued()[-1].id == external.id


def test_reload_is_idempotent(manager: TaskManager, project_root: Path) -> None:
    external = _external_create(project_root, "external")
    assert len(manager.reload_new_tasks()) == 1
    assert manager.reload_new_tasks() == []
    assert manager.queued() == [external]


def test_reload_does_not_overwrite_in_memory_state(
    manager: TaskManager, project_root: Path
) -> None:
    """Known files are never re-read: a task whose pending input was
    consumed into the transcript mid-run (persisted later, by save) must
    not get its input resurrected by a disk scan."""
    task = manager.create("own", "to consume")
    manager.promote_next()
    task.pending_input.clear()
    task.status = TaskStatus.ACTIVE
    _external_create(project_root, "external")

    assert manager.reload_new_tasks() != []
    reloaded = manager.get(task.id)
    assert reloaded is task
    assert reloaded.pending_input == []
    assert reloaded.status is TaskStatus.ACTIVE


def test_reload_demotes_foreign_active_task(
    manager: TaskManager, project_root: Path
) -> None:
    """A stale ACTIVE file (a crashed peer instance) must not block the
    local pipeline: on pickup it is demoted to QUEUED, like at startup."""
    peer = TaskManager(project_root)
    stale = peer.create("stale", "x")
    stale.status = TaskStatus.ACTIVE
    peer.storage.save(stale)

    (added,) = manager.reload_new_tasks()
    assert added is manager.get(stale.id)
    assert added.status is TaskStatus.QUEUED
    # The demotion is persisted, not just in memory.
    fresh = TaskManager(project_root)
    assert fresh.get(stale.id).status is TaskStatus.QUEUED


def test_promote_next_picks_up_external_task(
    manager: TaskManager, project_root: Path
) -> None:
    """The reported bug: an idle pipeline promotes a task created by
    another process without a restart — the disk scan happens inside the
    promotion search itself."""
    _external_create(project_root, "external")
    promoted = manager.promote_next()
    assert promoted is not None
    assert promoted.title == "external"
    assert promoted.status is TaskStatus.ACTIVE


def test_reload_survives_corrupted_file(
    manager: TaskManager, project_root: Path
) -> None:
    tasks_dir = project_root / ".drcode" / "tasks"
    (tasks_dir / "broken.json").write_text("{not json")
    external = _external_create(project_root, "external")

    added = manager.reload_new_tasks()
    assert [t.id for t in added] == [external.id]
    assert (tasks_dir / "broken.json.bak").is_file()
    assert not (tasks_dir / "broken.json").exists()


def test_reload_orders_multiple_external_tasks_by_creation(
    manager: TaskManager, project_root: Path
) -> None:
    first = _external_create(project_root, "first")
    second = _external_create(project_root, "second")
    added = manager.reload_new_tasks()
    assert [t.id for t in added] == [first.id, second.id]
    assert manager.queued() == [first, second]


# -- scratch-directory lifecycle ---------------------------------------------------


@pytest.fixture
def scratch_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the /tmp/drcode scratch base into the test's tmp dir."""
    base = tmp_path / "drcode-scratch"
    monkeypatch.setattr("drcode.tools.SCRATCH_ROOT", base)
    return base


def _touch_scratch(base: Path, task_id: str) -> Path:
    scratch = base / task_id
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "f.txt").write_text("scratch")
    return scratch


def test_completion_removes_scratch(manager: TaskManager, scratch_base: Path) -> None:
    task = manager.create("t", "x", kind=TaskKind.RESEARCH)
    scratch = _touch_scratch(scratch_base, task.id)
    manager.mark_done(task.id)  # research: straight to COMPLETED
    assert not scratch.exists()


def test_mark_done_to_testing_keeps_scratch(
    manager: TaskManager, scratch_base: Path
) -> None:
    # A feature task goes to TESTING first; the tester may still need the
    # scratch files (and an unfinished/re-queued task must keep them too).
    task = manager.create("t", "x")
    scratch = _touch_scratch(scratch_base, task.id)
    manager.mark_done(task.id)
    assert scratch.is_dir()
    manager.set_verification(task.id, success=True)
    assert not scratch.exists()


def test_cancel_removes_scratch_but_requeue_keeps_it(
    manager: TaskManager, scratch_base: Path
) -> None:
    task = manager.create("t", "x")
    scratch = _touch_scratch(scratch_base, task.id)
    # Re-queue (user message on a completed task): the task may resume work
    # with its scratch files — nothing is removed.
    manager.send(task.id, "one more round")
    assert scratch.is_dir()
    manager.cancel(task.id)
    assert not scratch.exists()


def test_delete_removes_scratch(manager: TaskManager, scratch_base: Path) -> None:
    task = manager.create("t", "x")
    manager.set_hold(task.id, on=True)
    scratch = _touch_scratch(scratch_base, task.id)
    manager.delete(task.id)
    assert not scratch.exists()


# -- fix loop: requeue_failed_fix (tester retry_here) ---------------------------------


def _to_testing(manager: TaskManager, task_id: str) -> Task:
    task = manager.get(task_id)
    task.status = TaskStatus.TESTING
    task.touch()
    manager.storage.save(task)
    return task


def test_requeue_failed_fix_returns_task_to_development(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    task.pending_input.clear()  # the original prompt was consumed by round 1
    _to_testing(manager, task.id)
    task.verification = VerificationStatus.FAILED
    manager.storage.save(task)

    manager.requeue_failed_fix(task.id, "your change broke pytest", max_rounds=3)

    assert task.status is TaskStatus.QUEUED
    assert task.verification is None
    assert task.completed_at is None
    assert task.fix_rounds == 1
    assert [e.text for e in task.pending_input] == ["your change broke pytest"]
    # The re-queued task is promotable: the coding agent receives the
    # tester's report in the same transcript/worktree.
    assert manager.promote_next() is task


def test_requeue_failed_fix_rounds_accumulate(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    task.pending_input.clear()  # consumed by round 1
    _to_testing(manager, task.id)
    manager.requeue_failed_fix(task.id, "broken 1", max_rounds=3)
    _to_testing(manager, task.id)
    manager.requeue_failed_fix(task.id, "broken 2", max_rounds=3)
    assert task.fix_rounds == 2
    assert [e.text for e in task.pending_input] == ["broken 1", "broken 2"]


def test_requeue_failed_fix_caps_at_max_rounds(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    task.pending_input.clear()
    _to_testing(manager, task.id)
    task.fix_rounds = 2
    manager.storage.save(task)

    manager.requeue_failed_fix(task.id, "still broken", max_rounds=3)

    # Cap reached: ordinary FAILED completion (the bug report is created by
    # the tester separately, from clean HEAD).
    assert task.status is TaskStatus.COMPLETED
    assert task.verification is VerificationStatus.FAILED
    assert task.completed_at is not None
    assert task.fix_rounds == 2
    assert task.pending_input == []


def test_requeue_failed_fix_ignores_non_testing_task(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    before = list(task.pending_input)
    manager.requeue_failed_fix(task.id, "late", max_rounds=3)
    assert task.status is TaskStatus.QUEUED  # unchanged (was QUEUED already)
    assert task.fix_rounds == 0
    assert task.pending_input == before


def test_passed_verdict_resets_fix_rounds(manager: TaskManager) -> None:
    task = _to_testing(manager, manager.create("t", "x").id)
    task.fix_rounds = 2
    manager.storage.save(task)
    manager.set_verification(task.id, True)
    assert task.status is TaskStatus.COMPLETED
    assert task.fix_rounds == 0


def test_user_input_to_testing_task_resets_fix_rounds(manager: TaskManager) -> None:
    task = _to_testing(manager, manager.create("t", "x").id)
    task.fix_rounds = 2
    manager.storage.save(task)
    manager.send(task.id, "try again differently")
    assert task.status is TaskStatus.QUEUED
    assert task.fix_rounds == 0


def test_user_input_to_completed_task_resets_fix_rounds(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    task.status = TaskStatus.COMPLETED
    task.fix_rounds = 2
    task.completed_at = _now()
    manager.storage.save(task)
    manager.send(task.id, "one more round")
    assert task.status is TaskStatus.QUEUED
    assert task.fix_rounds == 0


def test_requeue_failed_fix_persists_across_restart(
    manager: TaskManager, project_root: Path
) -> None:
    task = manager.create("t", "x")
    task.pending_input.clear()
    _to_testing(manager, task.id)
    manager.requeue_failed_fix(task.id, "broken", max_rounds=3)
    reloaded = TaskManager(project_root)
    task2 = reloaded.get(task.id)
    assert task2.fix_rounds == 1
    assert [e.text for e in task2.pending_input] == ["broken"]


# -- dependencies (depends_on promotion gate) -------------------------------------


def _complete(manager: TaskManager, task_id: str) -> None:
    """Drive a task to COMPLETED the way the pipeline does (tester
    verdict path; the fix-loop caps are irrelevant here)."""
    manager.mark_done(task_id)
    manager.set_verification(task_id, True)


def test_depends_on_defaults_empty(manager: TaskManager) -> None:
    task = manager.create("t", "x")
    assert task.depends_on == []


def test_set_dependencies_stores_and_persists(
    manager: TaskManager, project_root: Path
) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    manager.set_dependencies(b.id, [a.id, "  "])  # blank entries dropped
    assert b.depends_on == [a.id]
    reloaded = TaskManager(project_root)
    assert reloaded.get(b.id).depends_on == [a.id]


def test_set_dependencies_deduplicates(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    manager.set_dependencies(a.id, [b.id, b.id])
    assert a.depends_on == [b.id]


def test_set_dependencies_rejects_self(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    with pytest.raises(ValueError, match="itself"):
        manager.set_dependencies(a.id, [a.id])


def test_set_dependencies_rejects_unknown_id(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    with pytest.raises(ValueError, match="unknown task id"):
        manager.set_dependencies(a.id, ["ffffffff"])


def test_set_dependencies_rejects_cycle(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    c = manager.create("c", "x")
    manager.set_dependencies(a.id, [b.id])  # a waits for b
    manager.set_dependencies(b.id, [c.id])  # b waits for c
    with pytest.raises(ValueError, match="cycle"):
        manager.set_dependencies(c.id, [a.id])  # would close the ring
    with pytest.raises(ValueError, match="cycle"):
        manager.set_dependencies(b.id, [a.id])  # two-node ring
    # Clearing the middle edge opens the graph again.
    manager.set_dependencies(b.id, [])
    manager.set_dependencies(c.id, [a.id])
    assert c.depends_on == [a.id]


def test_blocking_dependencies_status_only(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    r = manager.create("r", "x", kind=TaskKind.RESEARCH)
    manager.set_dependencies(a.id, [b.id, r.id])
    # A stale stored id (set_dependencies would reject it — load time must
    # not): simulate a dep whose task file was deleted/purged.
    a.depends_on.append("ghost")
    manager.storage.save(a)
    assert manager.blocking_dependencies(a.id) == [b.id, r.id]
    _complete(manager, b.id)
    _complete(manager, r.id)  # research: mark_done completes it directly
    assert manager.blocking_dependencies(a.id) == []


def test_promote_next_skips_blocked(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    manager.set_dependencies(b.id, [a.id])
    assert manager.promote_next() is a
    assert b.status is TaskStatus.QUEUED
    _complete(manager, a.id)
    assert manager.promote_next() is b
    assert b.status is TaskStatus.ACTIVE


def test_promote_next_skips_blocked_even_with_unfinished(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    manager.set_dependencies(b.id, [a.id])
    b.unfinished = True  # would normally auto-resume
    assert manager.promote_next() is a
    assert b.status is TaskStatus.QUEUED


def test_missing_dep_task_does_not_block_promotion(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    manager.set_dependencies(b.id, [a.id])
    manager.delete(a.id)
    assert manager.promote_next() is b


def test_blocked_tasks_sink_below_runnable_in_queue(manager: TaskManager) -> None:
    c = manager.create("c", "x")  # runnable, oldest
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    manager.set_dependencies(a.id, [c.id])
    assert manager.queued() == [c, b, a]


def test_requeued_dependency_reblocks(manager: TaskManager) -> None:
    a = manager.create("a", "x")
    b = manager.create("b", "x")
    manager.set_dependencies(b.id, [a.id])
    _complete(manager, a.id)
    assert manager.blocking_dependencies(b.id) == []
    # The user sends new input to the completed dependency: it re-enters
    # the pipeline and its dependents are blocked again until its next
    # completion (the gate is re-evaluated dynamically).
    manager.send(a.id, "one more round")
    assert a.status is TaskStatus.QUEUED
    assert manager.blocking_dependencies(b.id) == [a.id]
    assert manager.promote_next() is a
    assert b.status is TaskStatus.QUEUED
