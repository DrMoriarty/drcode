from __future__ import annotations

import enum
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator


def _now() -> datetime:
    return datetime.now(UTC)


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


class TaskStatus(str, enum.Enum):
    QUEUED = "queued"
    # Created by an AGENT as a draft (project setting "agent creates
    # drafts"): the prompt sits in pending_input like on any queued task,
    # but the pipeline never touches the status — the user must open the
    # task in the new-task editor and finish it first (finish_draft sends
    # it to the queue or parks it on hold). ESC in the editor keeps it a
    # draft; delete/cancel work like on a held task.
    DRAFT = "draft"
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


class GoalType(str, enum.Enum):
    """Classification of a long-lived product goal (NOT a TaskKind):
    an idea is a raw captured thought, a goal is an approved direction of
    development, a milestone is a decomposable checkpoint with sub-goals
    (parent_id) underneath it. Goals live for months, never pass through
    the task state machine and are never verified by the tester."""

    IDEA = "idea"
    GOAL = "goal"
    MILESTONE = "milestone"


class GoalStatus(str, enum.Enum):
    ACTIVE = "active"
    DONE = "done"
    ARCHIVED = "archived"


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


class ClarificationItem(BaseModel):
    """One question inside a clarification package (see Clarification).

    `agent_proposals` holds options the agent added on its OWN initiative,
    outside the user's request — kept structurally separate from `options`
    so no textual marker ("(agent's suggestion, …)") is needed. Rendered
    after `options` with a ★; clickable indices continue after them.

    `multiple` marks a question where ANY SUBSET of the combined options is
    a valid answer: the UI accumulates clicks into a set (toggle) instead
    of submitting/replacing on the first click. Default False — old task
    JSONs without the field load unchanged."""

    question: str
    options: list[str] = Field(default_factory=list)
    agent_proposals: list[str] = Field(default_factory=list)
    multiple: bool = False

    @property
    def combined_options(self) -> list[str]:
        """options + agent_proposals — the clickable list in render order."""
        return self.options + self.agent_proposals


_LEGACY_CLARIFICATION_KEYS = ("question", "options", "agent_proposals")


class Clarification(BaseModel):
    """A package of questions the agent asked in ONE ask_clarifying_question
    call and is waiting answers for.

    Historically a clarification held exactly one question (flat
    `question`/`options`/`agent_proposals` fields); those JSONs keep
    loading — the before-validator wraps the legacy shape into a
    single-item `questions` list (same idempotent-migration pattern as
    storage.py's legacy monolithic files). The agent is prompted to ask
    ALL its open questions in one batched call, and a re-call while a
    package is pending is refused (ModelRetry) instead of overwriting it
    — the old one-question-per-call flow silently lost every earlier
    question but the last."""

    questions: list[ClarificationItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_single_question(cls, data: Any) -> Any:
        if isinstance(data, dict) and "questions" not in data and "question" in data:
            item = {k: data[k] for k in _LEGACY_CLARIFICATION_KEYS if k in data}
            return {"questions": [item]}
        return data

    @classmethod
    def single(
        cls,
        question: str,
        options: list[str] | None = None,
        agent_proposals: list[str] | None = None,
    ) -> Clarification:
        """Build a one-question package (tests, legacy callers)."""
        return cls(
            questions=[
                ClarificationItem(
                    question=question,
                    options=options or [],
                    agent_proposals=agent_proposals or [],
                )
            ]
        )

    def __bool__(self) -> bool:
        return bool(self.questions)


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


# The three serialized transcripts a task carries (the coding agent's full
# history and the isolated tester / conflict-resolver round lists). They are
# NOT pydantic fields: they live in a per-task sidecar file
# (`<id>.transcript.json`, see drcode.storage) and are loaded lazily through
# the properties below, so `load_all` over a large archive never parses the
# bulky transcripts at startup.
TRANSCRIPT_KEYS = ("messages", "tester_messages", "resolver_messages")


class TranscriptState:
    """Mutable per-task transcript holder behind Task's lazy properties.

    A task loaded from disk starts UNLOADED (`loaded=False`, empty lists,
    `loader` installed by Storage): the first attribute access on any of
    the three transcripts reads the sidecar file once and flips to loaded.
    A task constructed in memory (fresh or in tests) starts loaded with
    empty lists. `loader` stays installed for the whole lifetime —
    `unload_transcripts()` (LRU eviction of idle tasks' transcripts) just
    flips back to unloaded, and the next access re-reads the (already
    flushed) sidecar file.
    """

    __slots__ = (
        "loaded",
        "loader",
        "messages",
        "resolver_messages",
        "tester_messages",
    )

    def __init__(
        self,
        messages: list[dict[str, Any]] | None = None,
        tester_messages: list[list[dict[str, Any]]] | None = None,
        resolver_messages: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self.loaded: bool = True
        self.messages: list[dict[str, Any]] = (
            list(messages) if messages is not None else []
        )
        self.tester_messages: list[list[dict[str, Any]]] = (
            list(tester_messages) if tester_messages is not None else []
        )
        self.resolver_messages: list[list[dict[str, Any]]] = (
            list(resolver_messages) if resolver_messages is not None else []
        )
        # Callable returning {"messages": [...], "tester_messages": [...],
        # "resolver_messages": [...]} — installed by Storage for disk-backed
        # tasks; None for tasks that have never been persisted.
        self.loader: Callable[[], dict[str, Any]] | None = None

    def __eq__(self, other: object) -> bool:
        # Transient state is deliberately NOT part of Task equality: two
        # tasks are equal when their persisted (metadata) fields match,
        # no matter whether the bulky transcripts happen to sit in memory
        # or on disk right now.
        return isinstance(other, TranscriptState)


class Task(BaseModel):
    id: str = Field(default_factory=_short_id)
    title: str
    # True when the title was already generated by the agent from the
    # description (project setting `generate_task_titles`, see
    # drcode.title_gen). The stamp makes generation exactly-once per task
    # lifetime: set ONLY on success, so a failed generation leaves it
    # False and the next activation may retry; a task whose title was
    # generated never gets it re-derived (TaskManager.create / edit_task
    # derive titles for tasks the agent has not seen yet anyway).
    title_generated: bool = False
    # True when the title arrived EXPLICITLY at creation (a non-empty title
    # argument: the agents' create_bug_report / create_feature_request /
    # create_research_task
    # tools, the GitHub Issues import, a goal's title) and must never be
    # overwritten by the title generator. False for the derived first-line
    # fallback (TaskManager.create / edit_task) — those remain eligible for
    # generation. Tasks created before the flag existed default to False:
    # their exactly-once generation already ran via `title_generated`.
    title_explicit: bool = False
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
    # The three full transcripts (dev / tester rounds / resolver rounds) do
    # NOT live here: they are persisted in a sidecar file
    # (`<id>.transcript.json`) and exposed through the `messages`,
    # `tester_messages` and `resolver_messages` properties below, loading
    # lazily from disk on first access. This metadata file carries only
    # the skeleton: the cheap, high-signal projection of the dev transcript
    # (user prompts + final response) the archive search indexes.
    skeleton: dict[str, Any] = Field(default_factory=dict)
    # Private lazy-transcript holder (see TranscriptState).
    _transcripts: TranscriptState = PrivateAttr(default_factory=TranscriptState)
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
    # True when the user CANCELLED the task (Ctrl+X): the status is
    # COMPLETED (cancel completes immediately, skipping the tester) but the
    # task's work was never honestly finished — the goal auto-lifecycle
    # (TaskManager._goal_auto_done) must not count it as a done task.
    # Cleared when the task is re-queued with new input (send()).
    cancelled: bool = False
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
    # When the agent's last run STOPPED writing to this task's transcript
    # (stamped in the run worker's finally: normal end, abort, failure,
    # awaiting-clarification — any exit). The pre-run compaction gate
    # (`compaction_resume_min_tokens`, see drcode.settings /
    # core/pipeline_compaction._maybe_compact_after_idle) measures the
    # idle gap against it: a run resuming ≥ 3 minutes later re-reads the
    # transcript with a cold provider prefix cache, and a transcript over
    # the threshold is compacted BEFORE that cold read. None = the task
    # has never been run (first activation — nothing to compact yet);
    # None for old task files (no migration needed). Superseded runs
    # (a newer run of the same task took over) never stamp — the newer
    # run re-stamps at its own end.
    last_run_end: datetime | None = None
    # The model key ("<provider>/<model-id>"; bare id for custom models and
    # the env override) that factually STARTED this task's last agent run —
    # stamped on every run start (see
    # core/pipeline_compaction._prepare_model_switch_transcript). Persisted
    # with the task, so a model change ACROSS a drcode restart is detected:
    # the next run compares the stored key against the freshly resolved one
    # and, on a mismatch, cleans the transcript for the new model (strips
    # the previous model's thinking parts — a cross-provider resume maps
    # foreign reasoning into the assistant content otherwise — and
    # compacts a large history). None for tasks never run and for old task
    # files (no migration; the first run after the upgrade just stamps).
    last_model_key: str | None = None
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
    # Cumulative seconds the task has spent in each status, keyed by the
    # status VALUE ("active", "testing", ...). Accumulated at every status
    # transition in TaskManager.fire_status_change (dwell = now −
    # status_changed_at, added BEFORE the stamp is refreshed) — both this
    # and status_changed_at persist, so the counters survive restarts (the
    # startup demotion of a stale ACTIVE task closes out its last stay).
    # Feeds the project statistics' dev/testing time averages; idle time
    # inside a stay (nudges, waiting) is NOT subtracted — a deliberate v1
    # simplification.
    time_in_status: dict[str, float] = Field(default_factory=dict)
    # Cumulative seconds the conflict resolver spent on this task (measured
    # around each resolver round in core/pipeline_resolver.py — resolving is
    # not expressible as a status, the task stays parked while the resolver
    # runs). Persisted like every other Task field; 0.0 for old task files.
    resolver_seconds: float = 0.0
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
    # Tool-call ids whose results the agent marked as no longer needed (the
    # mark_unneeded tool). The mark only RECORDS the decision — nothing is
    # removed at mark time: the pipeline collapses all marked tool-return
    # contents in ONE batch (one prompt-cache invalidation), either when
    # their ≈size (chars/4) reaches a third of the serving role's context
    # window, or right before a full LLM compaction. Each collapsed return
    # is replaced by a short stub naming the tool and its arguments; the
    # tool-call/tool-return pair itself always survives (the transcript must
    # stay API-valid). Ids are pruned by the eviction gate (stubbed) and by
    # a full compaction (the summarized-away history takes the marks with
    # it). Kept in the metadata — NOT in the serialized message parts — so
    # ModelMessagesTypeAdapter never sees foreign fields.
    marked_unneeded: list[str] = Field(default_factory=list)
    # The long-lived product goal this task serves (see drcode.models.Goal
    # and the goals registry under .drcode/goals/). None for tasks created
    # without a goal reference (old task files load with the plain default,
    # no migration). Written at creation (TaskManager.create goal_id / the
    # agent's create_* goal_id parameter) or by the user's explicit
    # header assign/unlink links (TaskManager.link_goal/unlink_goal)
    # — nothing else moves a task between goals. The goal's completion
    # service event ("цель Y: n/m задач завершено") counts over the tasks
    # whose goal_id points at the goal (the link lives ONLY here — the
    # goal side is derived dynamically, see drcode.models.Goal); a deleted
    # goal id is tolerated at load time (the event writer just skips it),
    # like a deleted depends_on id.
    goal_id: str | None = None

    @property
    def unchecked_todos(self) -> list[TodoItem]:
        """Checklist items not marked done yet (the completion soft gate
        in the pipeline warns about these at mark_done time)."""
        return [t for t in self.todos if not t.done]

    # -- lazy transcripts (messages / tester_messages / resolver_messages) ----

    def _ensure_transcripts(self, key: str) -> Any:
        """Transcript accessor: loads the sidecar file once, on the first
        access of any of the three transcripts (a disk-backed task starts
        unloaded so `load_all` over hundreds of tasks never parses the
        bulky histories). The load populates ALL three lists — they share
        one sidecar file."""
        ts = self._transcripts
        if not ts.loaded:
            data = ts.loader() if ts.loader is not None else {}
            ts.messages = list(data.get("messages") or [])
            ts.tester_messages = list(data.get("tester_messages") or [])
            ts.resolver_messages = list(data.get("resolver_messages") or [])
            ts.loaded = True
        return getattr(ts, key)

    def _set_transcript(self, key: str, value: Any) -> None:
        """Transcript assignment: the given list becomes the authoritative
        in-memory state (a task built or mutated in code is loaded by
        definition; Storage.save flushes it to the sidecar file)."""
        ts = self._transcripts
        ts.loaded = True
        setattr(ts, key, value if isinstance(value, list) else [])

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Serialized pydantic-ai ModelMessages (full transcript incl. tool
        calls). Lazy: the first access loads `<id>.transcript.json` from
        disk when the task came from storage and its transcript is not in
        memory yet."""
        return self._ensure_transcripts("messages")

    @messages.setter
    def messages(self, value: list[dict[str, Any]]) -> None:
        self._set_transcript("messages", value)

    @property
    def tester_messages(self) -> list[list[dict[str, Any]]]:
        """Serialized transcripts of the tester runs, one entry per testing
        round. Kept separate from `messages` so the coding agent never
        resumes with tester-only tool calls (set_verification,
        create_bug_report) in its model history. Same lazy sidecar load."""
        return self._ensure_transcripts("tester_messages")

    @tester_messages.setter
    def tester_messages(self, value: list[list[dict[str, Any]]]) -> None:
        self._set_transcript("tester_messages", value)

    @property
    def resolver_messages(self) -> list[list[dict[str, Any]]]:
        """Serialized transcripts of the automatic merge-conflict resolver
        rounds (one entry per round), isolated from `messages` and
        `tester_messages` for the same reason: the resolver's tool calls
        (set_resolution, git plumbing) must never leak into the coding
        agent's or the tester's model history. Same lazy sidecar load."""
        return self._ensure_transcripts("resolver_messages")

    @resolver_messages.setter
    def resolver_messages(self, value: list[list[dict[str, Any]]]) -> None:
        self._set_transcript("resolver_messages", value)

    @property
    def transcripts_loaded(self) -> bool:
        """True while the full transcripts sit in memory (a task constructed
        in code, or one whose sidecar has already been read this
        session). False for disk-backed tasks not touched yet — Storage.save
        then writes metadata only and never rewrites the sidecar file."""
        return self._transcripts.loaded

    def unload_transcripts(self) -> None:
        """Drop the in-memory transcripts (LRU eviction of idle tasks).

        The sidecar file MUST be current when this runs (Storage.save
        flushed it); the next attribute access lazily re-reads it through
        the loader installed at load time. A task without a loader
        (never persisted) keeps an empty transcript after this — callers
        only unload persisted tasks."""
        self._transcripts.loaded = False
        self._transcripts.messages = []
        self._transcripts.tester_messages = []
        self._transcripts.resolver_messages = []

    @model_validator(mode="wrap")
    @classmethod
    def _capture_transcript_kwargs(cls, values: Any, handler: Any) -> Task:
        """Route `Task(messages=..., tester_messages=..., ...)` constructor
        and `model_validate` kwargs into the private transcript state.

        The three transcripts are NOT fields (they are persisted in the
        sidecar file, not in the metadata dict), so pydantic would silently
        IGNORE them as unknown keys (extra="ignore" keeps legacy task files
        loadable). This wrap validator pops them out of the raw input and
        attaches them as the in-memory (loaded) transcript state after the
        model is built — the pre-split behaviour of constructing a Task
        with its transcripts keeps working."""
        transcripts: dict[str, Any] = {}
        if isinstance(values, dict):
            for key in TRANSCRIPT_KEYS:
                if key in values:
                    transcripts[key] = values.pop(key)
        instance = handler(values)
        if transcripts:
            state = TranscriptState(
                **transcripts  # type: ignore[arg-type]
            )
            instance._transcripts = state
        return instance

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


class Goal(BaseModel):
    """A long-lived product-development goal (stage A, the passive registry).

    Deliberately NOT a TaskKind: goals live for months, never enter the
    task state machine and are never verified by the tester — they are the
    roadmap layer the research agent plans against. Persisted as one JSON
    file per goal under `<project>/.drcode/goals/` (see drcode.goals), so
    they stay local like the tasks. Plain defaults everywhere keep forward
    compatibility: old tooling reading task JSONs ignores unknown fields,
    and a goal written by a newer build still loads here."""

    id: str = Field(default_factory=_short_id)
    title: str
    # Two text fields, deliberately separate (see the GoalScreen's E):
    # `description` is the short line the research agent's prompt injection
    # carries (and the tree rows show); `notes` is the long free-form
    # markdown the agent reads ONLY via its read_goal tool — it NEVER
    # enters a system prompt (fixed in the goal-tools task).
    description: str = ""
    notes: str = ""
    type: GoalType = GoalType.IDEA
    status: GoalStatus = GoalStatus.ACTIVE
    # Decomposition: a milestone may have sub-goals pointing at it. No
    # validation beyond "the parent must exist" at write time (see
    # GoalStore.create/update) — a deleted parent id is tolerated at load
    # time, exactly like depends_on of tasks.
    parent_id: str | None = None
    # The link to tasks is deliberately NOT persisted here: it is derived
    # dynamically from Task.goal_id (the single source of truth, scanned by
    # TaskManager.goal_progress / _goal_progress_event / the read_goal
    # tool). A persisted back-list made the goal file mutate on every
    # task create/link/unlink — and the goal files live under git-tracked
    # territory in team projects, so the lifecycle state had to leave the
    # file (see GoalStore's sidecar below). A re-queued completed task
    # keeps its link (the field is never cleared on completion); a task
    # moves between goals only through the user's explicit Link/Unlink
    # palette commands (the agent's create_* tools set the reference at
    # creation only).
    # Sibling order stamp (like Task.queue_seq for the queue): the position
    # among goals sharing the same parent_id. Assigned on creation
    # (GoalStore.create), bumped by the GoalScreen's Ctrl+Up/Down
    # (GoalStore.move); None = a legacy goal written before the field
    # existed — it sorts after ordered siblings, by created_at. None of the
    # ordering is semantic: it only fixes the sidebar/tree display order.
    order: int | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def touch(self) -> None:
        self.updated_at = _now()

    @property
    def title_display(self) -> str:
        title = " ".join(self.title.split())
        return title if len(title) <= 40 else title[:39] + "…"

    def summary_line(self, description_cap: int = 0) -> str:
        """One compact line for agent-facing listings: `[<id>] (<type>)
        <title>` plus, when `description_cap > 0`, the first cap characters
        of the (whitespace-collapsed) description. The context-discipline
        primitive: the system prompt injection lists goals with a small
        cap, the full description stays behind read_goal."""
        parts = [f"[{self.id}]", f"({self.type.value})", self.title_display]
        if description_cap > 0:
            text = " ".join(self.description.split())
            if text:
                parts.append(
                    "— "
                    + (
                        text
                        if len(text) <= description_cap
                        else text[: description_cap - 1] + "…"
                    )
                )
        return " ".join(parts)


def render_todos(todos: list[TodoItem]) -> str:
    """Plain-text rendering of a checklist ("1. [x] text"), numbered by its
    1-based position — the same numbers todo_update accepts. Shared by the
    agent context injection (tui/pipeline.py), the compaction prompt and
    the TUI panel; the panel wraps the lines in rich markup itself."""
    return "\n".join(
        f"{i}. [{'x' if t.done else ' '}] {t.text}" for i, t in enumerate(todos, 1)
    )
