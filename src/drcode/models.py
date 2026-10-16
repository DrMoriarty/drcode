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


class TaskEvent(BaseModel):
    """A service/UI event (errors, skips, aborts) shown in the chat.

    Kept in a dedicated field on the task, outside `messages` and
    `tester_messages`, so resuming an agent never feeds these lines into
    the model context — they are for the human only.
    """

    at: datetime = Field(default_factory=_now)
    kind: str = "system"
    text: str


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
    # True when the last agent attempt was interrupted (abort, error) and
    # may have left work undone. Replaces the old "re-queue the consumed
    # prompt" retry marker: the prompt stays in the transcript at its
    # chronological position and is NEVER re-sent as a new message. The
    # agent resumes the attempt without new user text on the next
    # activation (run_agent's continuation turn).
    unfinished: bool = False
    # Set while the agent is waiting for the user to answer a question.
    clarification: Clarification | None = None
    # Service events (tester/agent failures, skips) with timestamps. Human
    # only: never serialized into the model context on resume.
    events: list[TaskEvent] = Field(default_factory=list)
    # Answered clarification tasks jump to the head of the queue.
    queue_head: bool = False
    # Explicit position stamp within the queue (assigned in created_at order
    # on load, bumped when a task is re-positioned, e.g. by a kind change).
    # Together with queue_head and kind it fully determines queue order.
    queue_seq: int | None = None

    def touch(self) -> None:
        self.updated_at = _now()

    @property
    def title_display(self) -> str:
        return self.title if len(self.title) <= 40 else self.title[:39] + "…"
