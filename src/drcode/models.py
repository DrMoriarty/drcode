from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _now() -> datetime:
    return datetime.now(UTC)


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


class TaskStatus(str, enum.Enum):
    QUEUED = "queued"
    # Parked by the user out of the backlog: never promoted while held
    # (input keeps accumulating; Ctrl+P resumes it back into the queue, or
    # into NEEDS CLARIFICATION if its question is still unanswered).
    HOLD = "hold"
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
    # Research: agent studies code/docs to answer questions or update docs;
    # never goes through the tester — mark_done() completes it.
    RESEARCH = "research"


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


class PendingInput(BaseModel):
    """A user message queued before the agent consumes it.

    `at` is when the user typed the message — the chat thread places it by
    this stamp (not by the consumption order), so comments sent while the
    task was queued or running keep their chronological place in the thread.
    """

    text: str
    at: datetime = Field(default_factory=_now)


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
    # User messages typed while the task is queued/not yet processed,
    # stamped with the moment they were typed (see PendingInput).
    pending_input: list[PendingInput] = Field(default_factory=list)
    # When the task reached COMPLETED (tester verdict, research done or
    # user cancel). Drives the startup auto-cleanup of old completed tasks
    # (see ProjectSettings.completed_retention); None while not completed.
    completed_at: datetime | None = None
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
    # Answered clarification tasks jump ahead of other non-bug queued tasks
    # (bug reports always sort before them).
    queue_head: bool = False
    # Explicit position stamp within the queue (assigned in created_at order
    # on load, bumped when a task is re-positioned, e.g. by a kind change).
    # Together with queue_head and kind it fully determines queue order.
    queue_seq: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: Any) -> Any:
        """Migrate legacy task files:
        - pending_input was list[str]: wrap plain strings into PendingInput,
          all stamped with the same wall-clock anchor (updated_at, i.e. the
          last queued comment) so the stable timeline sort keeps their
          original relative order.
        - kind "documentation" was renamed to "research" (same behaviour:
          no tester, mark_done() completes the task); remap the stored
          value so old task files load instead of failing validation."""
        if isinstance(data, dict):
            if data.get("kind") == "documentation":
                data["kind"] = "research"
            pending = data.get("pending_input")
            if isinstance(pending, list) and any(
                isinstance(item, str) for item in pending
            ):
                stamp = data.get("updated_at") or data.get("created_at")
                migrated: list[Any] = []
                for item in pending:
                    if isinstance(item, str):
                        entry: dict[str, Any] = {"text": item}
                        if stamp is not None:
                            entry["at"] = stamp
                        migrated.append(entry)
                    else:
                        migrated.append(item)
                data["pending_input"] = migrated
        return data

    def touch(self) -> None:
        self.updated_at = _now()

    @property
    def title_display(self) -> str:
        return self.title if len(self.title) <= 40 else self.title[:39] + "…"
