from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


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
    # Метка, которую тестировщик ставит задаче: "Проверено успешно" / "Проверено с ошибками".
    PASSED = "Verified successfully"
    FAILED = "Verification failed"


# Values the enum carried before the Russian→English translation (c7d474e).
# Tasks persisted with them must keep loading; mapped back in Task below.
_LEGACY_VERIFICATION_VALUES = {
    "Проверено успешно": VerificationStatus.PASSED,
    "Проверено с ошибками": VerificationStatus.FAILED,
}


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


class TodoItem(BaseModel):
    """One entry of a feature task's self-managed todo checklist.

    Plain pydantic model with a plain default — old task JSONs (and task
    JSONs written before this field existed) load without any migration,
    same pattern as `depends_on`."""

    text: str
    done: bool = False


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

    @field_validator("verification", mode="before")
    @classmethod
    def _migrate_legacy_verification(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _LEGACY_VERIFICATION_VALUES.get(v, v)
        return v

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    # Serialized pydantic-ai ModelMessages (full transcript incl. tool calls).
    messages: list[dict[str, Any]] = Field(default_factory=list)
    # Serialized transcripts of the tester runs, one entry per testing round.
    # Kept separate from `messages` so the coding agent never resumes with
    # tester-only tool calls (set_verification, create_bug_report) in its
    # model history.
    tester_messages: list[list[dict[str, Any]]] = Field(default_factory=list)
    # Serialized transcripts of the automatic merge-conflict resolver
    # rounds (one entry per round), isolated from `messages` and
    # `tester_messages` for the same reason: the resolver's tool calls
    # (set_resolution, git plumbing) must never leak into the coding
    # agent's or the tester's model history.
    resolver_messages: list[list[dict[str, Any]]] = Field(default_factory=list)
    # True when an automatic conflict-resolution attempt FAILED its
    # verification (mechanics/tests/intent review, or the resolver gave
    # up): the conflicted merge is left untouched in the task's worktree
    # and a human must take over (the visual screen of the follow-up
    # task). Cleared when a later resolver round succeeds.
    needs_human: bool = False
    # User messages typed while the task is queued/not yet processed,
    # stamped with the moment they were typed (see PendingInput).
    pending_input: list[PendingInput] = Field(default_factory=list)
    # Project-relative paths the RESEARCH agent created/edited itself via
    # write_file/edit_file (tracked in agent._track_edited_file). The
    # research-only git_commit_docs tool stages exactly these paths — never
    # any other change in the working tree — and clears the list after a
    # successful commit. Scratch files under /tmp/drcode/<task-id> are never
    # recorded here.
    edited_files: list[str] = Field(default_factory=list)
    # When the task reached COMPLETED (tester verdict, research done or
    # user cancel). Drives the startup auto-cleanup of old completed tasks
    # (see ProjectSettings.completed_retention); None while not completed.
    completed_at: datetime | None = None
    # When the task last entered ACTIVE (stamped by promote_next; None
    # while the task has never been activated). Session statistics measure
    # the activation → completion span from this (see the expanded header);
    # a re-queued completed task gets a fresh stamp on its next activation.
    activated_at: datetime | None = None
    # When the task last ENTERED its current status (stamped by
    # TaskManager.fire_status_change, the single funnel for every status
    # transition; None for tasks created before the field existed). Drives
    # the per-status duration counter in the task-statusbar header — every
    # transition (including back to QUEUED) re-stamps it, so the counter
    # always measures the current continuous stay, never a cumulative one.
    status_changed_at: datetime | None = None
    # True when the last agent attempt was interrupted (abort, error) and
    # may have left work undone. Replaces the old "re-queue the consumed
    # prompt" retry marker: the prompt stays in the transcript at its
    # chronological position and is NEVER re-sent as a new message. The
    # agent resumes the attempt without new user text on the next
    # activation (run_agent's continuation turn).
    unfinished: bool = False
    # The worktree-isolation mode this task is PINNED to, stamped at its
    # FIRST activation (None = never activated — the live `task_isolation`
    # setting decides). True/False freeze the mode for the task's whole
    # lifetime, including re-queueing a COMPLETED task with new input: the
    # user may toggle the setting mid-task, and a run boundary (unfinished
    # resume, tester round, conflict resolver) that re-read the live value
    # would silently move the work — a tester would verify an empty worktree
    # forked from the branch tip while the coder's uncommitted changes sit
    # in the main checkout, or the mirror image (a run jumping from the
    # task's worktree to a main checkout that has none of its work).
    # Persisted with the task, so the pin survives restarts; the pin dies
    # with the task. Old task files load with None and stamp on their next
    # activation (no migration).
    isolation_pinned: bool | None = None
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
    # Number of the GitHub issue this task was imported from (GitHub Issues
    # integration); None for locally created tasks. The tester appends the
    # "Fixes #N" keyword to its commit message so GitHub links the commit
    # back to the issue.
    github_issue: int | None = None
    # External code-review gate (.drcode/workflow.py hooks): how many times
    # in a row the external review rejected this task. Reset when the review
    # finally accepts (or when the user intervenes by sending input /
    # cancelling) — after MAX_REVIEW_REJECTIONS consecutive rejects the
    # pipeline parks the task on hold instead of looping forever.
    review_rejections: int = 0
    # How many development rounds in a row the tester has failed with a
    # verdict "the bug is caused by THIS task's own changes" (the
    # retry_here=True path of set_verification): the task returns to
    # development in its own worktree instead of completing with a bug
    # report. Reset by a PASSED verdict or by the user taking a
    # COMPLETED/TESTING task back (send) — after `max_dev_rounds`
    # consecutive failed rounds (project setting) the tester's failure
    # completes the task as FAILED and the bug report is created as usual.
    fix_rounds: int = 0
    # True while a COMPLETED task awaits an ASYNCHRONOUS external review
    # verdict (workflow with on_development_done but no on_review: the task
    # completes with an open PR and the verdict arrives via the periodic
    # review poll). Such tasks are excluded from the startup auto-cleanup,
    # so a verdict arriving days later still finds its task; cleared when
    # the verdict is applied or the user re-queues/cancels the task.
    review_pending: bool = False
    # The task kind's accept hook (on_accepted for development tasks,
    # on_research_accepted for research) normally owns the merge of the
    # task's `drcode/<id>` branch back into the base branch — and while it
    # does, the unmerged-work indicator/startup reminder stay silent (the
    # merge is automated). Set when that hook-merge FAILED: a git conflict,
    # a dirty checkout ("NOT merged …") or a crashed hook. A failed merge
    # needs a human, so the task re-enters the unmerged-work tracking (the
    # header's `⎇ N` counter and the startup reminder) exactly as if no
    # hook existed — a manual `git merge drcode/<id>` (or a later
    # successful hook re-run) clears the signal again. Persisted like every
    # other Task field; False for old task files (plain default).
    hook_merge_failed: bool = False
    # True when the on_tester_passed hook (install/deploy) was SKIPPED by
    # the pipeline because the main checkout sat in a half-done merge
    # (MERGE_HEAD set — installing would ship conflict markers). The
    # pipeline re-runs the deferred hook once the blocker is gone (the
    # task's branch merged / the checkout clean again) and clears the flag
    # after the hook actually executed — it never runs while MERGE_HEAD
    # lives in the main checkout, that invariant is the pipeline's, not
    # the script's. Persisted like every other Task field; False for old
    # task files (plain default).
    install_pending: bool = False
    # Ids of tasks that must be COMPLETED *and* merged into the base branch
    # before this task may be taken into development (the promotion gate is
    # dynamic: git state is re-evaluated at every promotion attempt, so a
    # manual `git merge drcode/<dep>` unblocks dependents without any
    # drcode action). Mergedness is only checkable under per-task worktree
    # isolation (drcode.task_env); without isolation — and for missing dep
    # tasks with no branch left — the gate degenerates to "COMPLETED".
    # Written only through TaskManager.set_dependencies (validated: no
    # self-reference, existing ids, acyclic); a plain default keeps old
    # task JSONs loading without migration.
    depends_on: list[str] = Field(default_factory=list)
    # Self-managed todo checklist (feature tasks only — only those agents
    # get the todo_create/todo_update tools). Persisted with the task like
    # every other field, so the list survives restarts, compaction and
    # interrupted attempts; it never lives only in the transcript. The
    # agent owns the list (the TUI panel is read-only); its current state
    # is re-injected into the model context on every activation and after
    # every compaction (see tui/pipeline.py), so the agent cannot "forget"
    # it. Rendered by render_todos below.
    todos: list[TodoItem] = Field(default_factory=list)

    @property
    def unchecked_todos(self) -> list[TodoItem]:
        """Checklist items not marked done yet (the completion soft gate
        in the pipeline warns about these at mark_done time)."""
        return [t for t in self.todos if not t.done]

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
        # Collapse control characters (tabs/newlines) and repeated spaces
        # first: a literal tab survives `escape()` in the TUI and makes
        # Textual expand the row to the next tab stop, pushing the tail of
        # the truncated title onto a second sidebar line.
        title = " ".join(self.title.split())
        return title if len(title) <= 40 else title[:39] + "…"


def render_todos(todos: list[TodoItem]) -> str:
    """Plain-text rendering of a checklist ("1. [x] text"), numbered by its
    1-based position — the same numbers todo_update accepts. Shared by the
    agent context injection (tui/pipeline.py), the compaction prompt and
    the TUI panel; the panel wraps the lines in rich markup itself."""
    return "\n".join(
        f"{i}. [{'x' if t.done else ' '}] {t.text}" for i, t in enumerate(todos, 1)
    )
