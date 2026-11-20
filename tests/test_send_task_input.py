"""The send_task_input tool: cross-task user-input delivery with a
persisted ownership check (the #e5449310 incident regression suite).

The research/tester agents must deliver extra input into tasks THEY
created through the manager (TaskManager.send) — never by editing
another task's JSON behind the live manager, which the running task's
next in-memory save silently wipes. The tests cover the ownership gate
(own / foreign / unknown / self), the persisted provenance (a fresh
TaskManager — a restarted TUI — still recognizes the creator), the
status-dependent delivery (QUEUED pending_input, ACTIVE run
interruption via the hooks, COMPLETED re-queue), dedup through
TaskManager.send, the both-side service-event audit trail and the
create_* tool-result hint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from drcode import agent_build
from drcode.agent_build import AgentDeps, _create_report_task, _send_task_input
from drcode.agent_tools import InputDeliveryHooks
from drcode.models import Task, TaskKind, TaskStatus
from drcode.task_manager import TaskManager


def _agent_deps(
    project_root: Path, creator: Task, manager: TaskManager | None = None
) -> AgentDeps:
    return AgentDeps(
        root=project_root,
        manager=manager if manager is not None else TaskManager(project_root),
        task_id=creator.id,
        scratch=project_root / "scratch-unused",
    )


def _creator_with_report(project_root: Path) -> tuple[AgentDeps, Task]:
    """A creator task (held, so it never gets promoted by the tests that
    activate the queue) plus the report task its agent created through
    the shared creation body — so the provenance stamp is set the real
    way, not hand-assigned."""
    manager = TaskManager(project_root)
    creator = manager.create("Creator task", "the run delivering input", hold=True)
    deps = _agent_deps(project_root, creator)
    created, _text = _create_report_task(
        deps,
        "Follow-up feature",
        "the task the input will be appended to",
        kind=TaskKind.FEATURE,
        creator="research agent",
    )
    return deps, created


# -- ownership -----------------------------------------------------------------


def test_delivers_into_own_queued_task(project_root: Path) -> None:
    """A task the run created is owned: QUEUED target gets the text as
    pending user input (exactly the field the pipeline consumes)."""
    deps, created = _creator_with_report(project_root)
    result = _send_task_input(deps, created.id, "also support dark mode")
    assert created.pending_input[-1].text == "also support dark mode"
    assert created.status is TaskStatus.QUEUED
    assert "queued as pending user input" in result


def test_refuses_foreign_task_and_sends_nothing(project_root: Path) -> None:
    """A task NOT created by this run (user-made: no provenance stamp) is
    refused BEFORE any mutation — no pending input, no service events."""
    deps, _created = _creator_with_report(project_root)
    foreign = deps.manager.create("User task", "hand-typed")
    with pytest.raises(ValueError, match="out of your reach"):
        _send_task_input(deps, foreign.id, "try to sneak in")
    assert foreign.pending_input[-1].text == "hand-typed"  # unchanged by the tool
    assert foreign.events == []
    # Nothing was audited as "delivered" anywhere.
    assert not any(
        "delivered" in e.text for t in deps.manager.tasks.values() for e in t.events
    )


def test_refuses_unknown_task_id(project_root: Path) -> None:
    deps, _created = _creator_with_report(project_root)
    with pytest.raises(ValueError, match="no task with id"):
        _send_task_input(deps, "zzzzzzzz", "nobody home")


def test_refuses_own_task_and_points_at_clarification(project_root: Path) -> None:
    """The run's OWN task was created by the user (no provenance) — the
    correct channel for extra input into it is ask_clarifying_question."""
    deps, _created = _creator_with_report(project_root)
    own = deps.manager.get(deps.task_id)
    with pytest.raises(ValueError, match="ask_clarifying_question"):
        _send_task_input(deps, own.id, "self-delivery attempt")


def test_refuses_empty_text(project_root: Path) -> None:
    deps, created = _creator_with_report(project_root)
    with pytest.raises(ValueError, match="empty"):
        _send_task_input(deps, created.id, "   ")
    assert not any(e.text.strip() == "" for e in created.pending_input)


def test_provenance_survives_restart(project_root: Path) -> None:
    """The ownership check reads the PERSISTED field: a brand-new
    TaskManager (the TUI restarted between the creation and the
    follow-up input) still recognizes the creator."""
    deps, created = _creator_with_report(project_root)
    reloaded_deps = _agent_deps(project_root, deps.manager.get(deps.task_id))
    reloaded = reloaded_deps.manager.get(created.id)
    assert reloaded.created_by_task_id == deps.task_id
    result = _send_task_input(reloaded_deps, created.id, "input after restart")
    assert reloaded.pending_input[-1].text == "input after restart"
    assert "queued as pending user input" in result


def test_provenance_distinguishes_two_creators(project_root: Path) -> None:
    """Two runs each created their own report: neither may append into
    the other's (the stamps name the concrete creator run, not just
    "some agent made it")."""
    deps_a, created_a = _creator_with_report(project_root)
    shared = deps_a.manager
    creator_b = shared.create("Other creator", "another run", hold=True)
    deps_b = _agent_deps(project_root, creator_b, manager=shared)
    created_b, _text = _create_report_task(
        deps_b,
        "Other run's report",
        "belongs to the other run",
        kind=TaskKind.BUG,
        creator="research agent",
    )
    # The refusal names the CONCRETE creator run, so the agent can tell
    # whose report it just tried to touch.
    with pytest.raises(ValueError, match=f"created by task #{creator_b.id}"):
        _send_task_input(deps_a, created_b.id, "wrong sender")
    with pytest.raises(ValueError, match=f"created by task #{deps_a.task_id}"):
        _send_task_input(deps_b, created_a.id, "wrong sender")


# -- delivery semantics ---------------------------------------------------------


def test_active_delivery_interrupts_run_via_hooks(project_root: Path) -> None:
    """ACTIVE receiver: the hooks fire — restart_task_run for the
    receiver (its in-flight run is superseded, like the TUI does) and
    schedule_advance for the pipeline gate."""
    deps, created = _creator_with_report(project_root)
    manager = deps.manager
    manager.promote_next()  # the only promotable task is the report → ACTIVE
    assert created.status is TaskStatus.ACTIVE
    restarted: list[str] = []
    advanced: list[Any] = []
    deps.input_hooks = InputDeliveryHooks(
        advance=lambda: advanced.append(1),
        restart_run=restarted.append,
    )
    result = _send_task_input(deps, created.id, "mid-run addition")
    assert restarted == [created.id]
    assert advanced == [1]
    assert "restarts with your text" in result
    # The text rides in pending_input — the fresh run consumes it.
    assert created.pending_input[-1].text == "mid-run addition"


def test_active_delivery_without_hooks_still_delivers(project_root: Path) -> None:
    """No hooks (probe builds, test stand-ins): delivery still works, the
    hooks degrade to no-ops — the text waits in pending_input for the
    ordinary pipeline events to pick it up."""
    deps, created = _creator_with_report(project_root)
    deps.manager.promote_next()
    deps.input_hooks = None
    result = _send_task_input(deps, created.id, "no hooks here")
    assert created.pending_input[-1].text == "no hooks here"
    assert "restarts with your text" in result


def test_hook_failure_after_delivery_never_raises(project_root: Path) -> None:
    """A broken hook is a post-step: the delivery already happened, the
    failure is swallowed (logged) — never propagated into the tool result."""
    deps, created = _creator_with_report(project_root)

    def boom(*args: object) -> None:
        raise RuntimeError("pipeline gone")

    deps.input_hooks = InputDeliveryHooks(advance=boom, restart_run=boom)
    result = _send_task_input(deps, created.id, "delivered anyway")
    assert created.pending_input[-1].text == "delivered anyway"
    assert "delivered" in result


def test_completed_task_requeued(project_root: Path) -> None:
    """COMPLETED target: TaskManager.send's re-queue semantics — the
    task returns to QUEUED, the verification label and completion stamp
    are reset, ready for a fresh development round."""
    deps, created = _creator_with_report(project_root)
    manager = deps.manager
    manager.cancel(created.id)
    assert created.status is TaskStatus.COMPLETED
    result = _send_task_input(deps, created.id, "one more thing")
    assert created.status is TaskStatus.QUEUED
    assert created.verification is None
    assert created.completed_at is None
    assert "re-queued for a fresh development round" in result


def test_duplicate_send_not_duplicated(project_root: Path) -> None:
    """TaskManager.send's dedup is intact through the tool: a repeated
    identical text lands once in pending_input (the pipeline just did not
    visibly react yet)."""
    deps, created = _creator_with_report(project_root)
    first = _send_task_input(deps, created.id, "same note")
    second = _send_task_input(deps, created.id, "same note")
    assert [e.text for e in created.pending_input].count("same note") == 1
    assert "delivered" in first and "delivered" in second


# -- audit trail ----------------------------------------------------------------


def test_service_events_on_both_tasks(project_root: Path) -> None:
    """Every delivery is audited twice: the receiver's event log names
    the sender run and the appended text; the sender's own log records
    what was delivered where (who and what)."""
    deps, created = _creator_with_report(project_root)
    _send_task_input(deps, created.id, "audit me", "research agent")
    receiver_events = [e.text for e in created.events]
    sender_events = [e.text for e in deps.manager.get(deps.task_id).events]
    assert any(
        f"research agent of task #{deps.task_id}" in t and "audit me" in t
        for t in receiver_events
    )
    assert any(
        f"task #{created.id}" in t and f"'{created.title}'" in t and "audit me" in t
        for t in sender_events
    )
    # Both sides persisted.
    reloaded = TaskManager(project_root)
    assert any("audit me" in e.text for e in reloaded.get(created.id).events)
    assert any("audit me" in e.text for e in reloaded.get(deps.task_id).events)


# -- hint in the create_* results ----------------------------------------------


def test_create_result_carries_send_task_input_hint(project_root: Path) -> None:
    """The creation tool results tell the agent about send_task_input —
    the safe channel for later additions (one line, every kind)."""
    manager = TaskManager(project_root)
    creator = manager.create("Creator task", "the run", hold=True)
    deps = _agent_deps(project_root, creator)
    for kind in (TaskKind.BUG, TaskKind.FEATURE, TaskKind.RESEARCH):
        _task, text = _create_report_task(
            deps,
            f"Hint probe {kind.value}",
            "checking the suffix",
            kind=kind,
            creator="research agent",
        )
        assert "send_task_input" in text
        assert "created yourself" in text


def test_tester_creation_carries_hint_too(project_root: Path) -> None:
    """The tester's create_bug_report goes through the same shared body —
    its result carries the hint as well."""
    manager = TaskManager(project_root)
    creator = manager.create("Creator task", "the run", hold=True)
    tester_deps = agent_build.TesterDeps(
        root=project_root,
        manager=manager,
        task_id=creator.id,
        scratch=project_root / "scratch-unused",
    )
    _task, text = _create_report_task(
        tester_deps,
        "Tester bug",
        "details of the bug",
        kind=TaskKind.BUG,
        creator="tester",
    )
    assert "send_task_input" in text
