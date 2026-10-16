from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


class TaskStatus(str, enum.Enum):
    QUEUED = "queued"
    ACTIVE = "active"
    # Development finished, the tester has not labelled the task yet. If the
    # app dies here, the task is re-tested first on the next startup (before
    # any queued task is taken into development).
    TESTING = "testing"
    COMPLETED = "completed"
    AWAITING_CLARIFICATION = "clarification"


class TaskKind(str, enum.Enum):
    BUG = "bug"
    FEATURE = "feature"
    # Documentation: agent studies code/docs to answer questions or update
    # docs; never goes through the tester — mark_done() completes it.
    DOCUMENTATION = "documentation"


class VerificationStatus(str, enum.Enum):
    PASSED = "Проверено успешно"
    FAILED = "Проверено с ошибками"


class Clarification(BaseModel):
    """A question the agent asked the user and is waiting an answer for."""

    question: str
    options: list[str] = Field(default_factory=list)


class Task(BaseModel):
    id: str = Field(default_factory=_short_id)
    title: str
    kind: TaskKind = TaskKind.FEATURE
    status: TaskStatus = TaskStatus.QUEUED
    verification: VerificationStatus | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    # Serialized pydantic-ai ModelMessages (full transcript incl. tool calls).
    messages: list[dict[str, Any]] = Field(default_factory=list)
    # Serialized transcripts of the tester runs, one entry per testing round.
    # Kept separate from `messages` so the coding agent never resumes with
    # tester-only tool calls (set_verification, create_bug_report) in its
    # model history.
    tester_messages: list[list[dict[str, Any]]] = Field(default_factory=list)
    # User messages typed while the task is queued/not yet processed.
    pending_input: list[str] = Field(default_factory=list)
    # Set while the agent is waiting for the user to answer a question.
    clarification: Clarification | None = None
    # Answered clarification tasks jump to the head of the queue.
    queue_head: bool = False

    def touch(self) -> None:
        self.updated_at = _now()

    @property
    def title_display(self) -> str:
        return self.title if len(self.title) <= 40 else self.title[:39] + "…"
