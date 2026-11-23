"""Tests for the per-run token budget (`Config.run_token_limit`).

Covers the graceful-stop path: when pydantic-ai raises UsageLimitExceeded
(the run's budget is spent), the attempt must be kept unfinished like
after an abort — the task stays ACTIVE (never requeued), the transcript
is sanitized, and a service event reports the spent amount.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic_ai.exceptions import UsageLimitExceeded

from drcode.config import Config
from drcode.models import TaskKind, TaskStatus
from drcode.tui.app import DrCodeApp


def _make_host(mgr, project_root: Path):
    """A minimal pipeline host: only the attributes the graceful-stop
    helper touches (the same stub pattern as the todo-completion tests)."""

    class Host:
        viewed_id = None  # the task is not open in the chat
        manager = mgr
        config = Config(project_root=project_root, run_token_limit=50_000)
        _agent_generations: ClassVar[dict[str, int]] = {}

        _show_event = DrCodeApp._show_event
        _log_event = DrCodeApp._log_event
        _format_event = DrCodeApp._format_event
        _backdate_consumed_prompt = DrCodeApp._backdate_consumed_prompt
        _persist_task_messages = DrCodeApp._persist_task_messages

        @staticmethod
        def _drop_stream_preview(task_id: str, generation: int) -> None:
            return None

    return Host()


def test_budget_exhausted_keeps_task_unfinished(manager, project_root) -> None:
    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    task.status = TaskStatus.ACTIVE
    host = _make_host(manager, project_root)
    host._agent_generations[task.id] = 1

    DrCodeApp._agent_turn_budget_exhausted(
        host,  # type: ignore[arg-type]
        task,
        task.id,
        run=None,
        consumed=[],
        content=None,
        generation=1,
        error=UsageLimitExceeded("Exceeded the total_tokens_limit of 50000 (0)"),
    )

    fresh = manager.get(task.id)
    # Graceful stop, NOT a failure: the task keeps its ACTIVE status and
    # pauses on the unfinished flag (like an abort) — no requeue, no
    # failed-queue exclusion.
    assert fresh.status is TaskStatus.ACTIVE
    assert fresh.unfinished is True


def test_budget_exhausted_reports_spent_tokens(manager, project_root) -> None:
    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    task.status = TaskStatus.ACTIVE
    host = _make_host(manager, project_root)
    host._agent_generations[task.id] = 1

    DrCodeApp._agent_turn_budget_exhausted(
        host,  # type: ignore[arg-type]
        task,
        task.id,
        run=None,
        consumed=[],
        content=None,
        generation=1,
        error=UsageLimitExceeded("Exceeded the total_tokens_limit of 50000 (0)"),
    )

    texts = [e.text for e in manager.get(task.id).events]
    assert any("Run budget exhausted" in t for t in texts)
    assert any("50,000" in t for t in texts)


def test_budget_exhausted_ignores_superseded_run(manager, project_root) -> None:
    """A superseded run (a newer generation took over) must not mark the
    attempt unfinished — the newer run owns the task."""
    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    task.status = TaskStatus.ACTIVE
    host = _make_host(manager, project_root)
    host._agent_generations[task.id] = 2  # this run's generation (1) is stale

    DrCodeApp._agent_turn_budget_exhausted(
        host,  # type: ignore[arg-type]
        task,
        task.id,
        run=None,
        consumed=[],
        content=None,
        generation=1,
        error=UsageLimitExceeded("Exceeded the total_tokens_limit of 50000 (0)"),
    )

    fresh = manager.get(task.id)
    assert fresh.unfinished is False
    # The event is still logged: it happened, the human must see it.
    assert any("Run budget exhausted" in e.text for e in manager.get(task.id).events)
