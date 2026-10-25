"""Project-scoped settings, persisted in `<project>/.drcode/settings.json`.

Deliberately separate from `config.Config` (env-driven, model/API settings):
these are user preferences of the current project, so they live next to its
tasks and survive restarts regardless of the environment.
"""

from __future__ import annotations

import enum
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ValidationError


class CompletedRetention(str, enum.Enum):
    """How long successfully completed tasks are kept before automatic
    removal at startup."""

    DAYS_7 = "7"
    DAYS_30 = "30"
    DAYS_90 = "90"
    NEVER = "never"

    @property
    def days(self) -> int | None:
        """Retention period in days, or None when tasks are never removed."""
        if self is CompletedRetention.NEVER:
            return None
        return int(self.value)

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is CompletedRetention.NEVER:
            return "never"
        return f"after {self.value} days"


RETENTION_OPTIONS: tuple[CompletedRetention, ...] = (
    CompletedRetention.DAYS_7,
    CompletedRetention.DAYS_30,
    CompletedRetention.DAYS_90,
    CompletedRetention.NEVER,
)


class TaskFinalization(str, enum.Enum):
    """What happens to a task after its development round finishes.

    TESTING_COMMIT: the tester verifies the project, commits the verified
    changes and pushes them to the remote (push failure never fails the
    task — it is only logged). TESTING_COMMIT_NO_PUSH: the tester commits
    the verified changes but never pushes — the branch stays local.
    TESTING: the tester verifies but never commits. NONE: no tester at
    all — tasks go straight to COMPLETED.
    """

    TESTING_COMMIT = "testing_commit"
    TESTING_COMMIT_NO_PUSH = "testing_commit_no_push"
    TESTING = "testing"
    NONE = "none"

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is TaskFinalization.TESTING_COMMIT:
            return "Testing + Git commit + push"
        if self is TaskFinalization.TESTING_COMMIT_NO_PUSH:
            return "Testing + Git commit"
        if self is TaskFinalization.TESTING:
            return "Testing"
        return "None"

    @property
    def commits(self) -> bool:
        """Whether the tester is allowed to commit verified changes."""
        return self in (
            TaskFinalization.TESTING_COMMIT,
            TaskFinalization.TESTING_COMMIT_NO_PUSH,
        )

    @property
    def pushes(self) -> bool:
        """Whether the tester pushes the branch after committing."""
        return self is TaskFinalization.TESTING_COMMIT


FINALIZATION_OPTIONS: tuple[TaskFinalization, ...] = (
    TaskFinalization.TESTING_COMMIT,
    TaskFinalization.TESTING_COMMIT_NO_PUSH,
    TaskFinalization.TESTING,
    TaskFinalization.NONE,
)


# Options of the `max_dev_rounds` setting (the fix-loop cap), offered in the
# Settings picker.
DEV_ROUNDS_OPTIONS: tuple[int, ...] = (1, 2, 3, 5)


def dev_rounds_label(rounds: int) -> str:
    """Human-readable option label for the Settings UI."""
    return "1 round" if rounds == 1 else f"{rounds} rounds"


# Options of the `parallel_tasks` setting (the development-slot limit),
# offered in the Settings picker. Parallelism is only meaningful with
# worktree isolation: without it the effective limit stays 1.
PARALLEL_TASKS_OPTIONS: tuple[int, ...] = (1, 2, 3, 4)


def parallel_tasks_label(count: int) -> str:
    """Human-readable option label for the Settings UI."""
    return "1 task" if count == 1 else f"{count} tasks"


class EscAbort(str, enum.Enum):
    """How ESC stops a running agent/tester (project setting).

    DOUBLE_PRESS (the default): the first ESC only ARMS the stop — a toast
    asks for a confirmation press within a short window — and only the
    second ESC actually aborts. This guards against the accidental
    "closed a modal dialog with ESC, pressed ESC again out of inertia"
    scenario that silently stopped a working agent. INSTANT: the
    historical behaviour — a single ESC stops the run immediately.
    """

    DOUBLE_PRESS = "double_press"
    INSTANT = "instant"

    @property
    def label(self) -> str:
        """Human-readable option label for the Settings UI."""
        if self is EscAbort.INSTANT:
            return "instant (one ESC)"
        return "double-press (confirm)"

    @property
    def description(self) -> str:
        """One-line explanation shown in the Settings palette."""
        if self is EscAbort.INSTANT:
            return "A single ESC stops the running agent/tester immediately."
        return (
            "The first ESC only arms the stop (a toast asks for a second "
            "press within 2 seconds); the second ESC stops the run."
        )


ESC_ABORT_OPTIONS: tuple[EscAbort, ...] = (
    EscAbort.DOUBLE_PRESS,
    EscAbort.INSTANT,
)


class TaskIsolation(str, enum.Enum):
    """Where a task's development happens.

    OFF (the default): every task works directly in the project directory
    (the historical "everything in one tree" behaviour — an interrupted
    task leaves its changes behind for the next one). WORKTREE: each task
    gets its own git worktree under `.drcode/worktrees/<task-id>` on a
    dedicated branch `drcode/<task-id>` (see `drcode.task_env`), so an
    interrupted (unfinished) task never hands a dirty tree to the next
    task in the queue.

    WORKTREE needs a git repository in the project. The setting is still
    saved when the project has none — the app warns at every startup (and
    when the value is picked) that the mode stays dormant until a
    repository appears (or the setting is turned off again); without git
    tasks simply keep running in the project directory.
    """

    OFF = "off"
    WORKTREE = "worktree"

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is TaskIsolation.WORKTREE:
            return "worktree (git)"
        return "off (shared directory)"


ISOLATION_OPTIONS: tuple[TaskIsolation, ...] = (
    TaskIsolation.OFF,
    TaskIsolation.WORKTREE,
)


class WorktreeRemoval(str, enum.Enum):
    """When a task's worktree directory `.drcode/worktrees/<task-id>`
    (see `drcode.task_env`) is removed automatically (project setting).

    ON_DELETE (the default, the historical behaviour): the directory
    lives as long as the task — it is torn down on cancel (Ctrl+X),
    delete (Ctrl+Delete) and the retention cleanup. ON_COMPLETION: the
    directory is removed as soon as the task reaches COMPLETED
    (research right in `mark_done`, dev tasks at the tester's
    `set_verification` verdict — and the no-tester / review-accept
    completions alike); the branch `drcode/<task-id>` with its
    committed work always survives (RULE 2), and re-activating a
    completed task (new input) recreates the worktree from the branch.
    NEVER: the directory is never removed automatically — `remove()`
    then only snapshots a dirty tree onto the branch (RULE 1); dead
    tasks' leftover directories are still swept at startup
    (`cleanup_orphans` ignores this setting), otherwise the mode would
    grow the worktrees directory without bound.

    The setting governs ONLY the worktree directory — never the branch:
    a branch holding unmerged commits is never deleted automatically
    (see the `drcode.task_env` module docstring).
    """

    ON_DELETE = "on_delete"
    ON_COMPLETION = "on_completion"
    NEVER = "never"

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is WorktreeRemoval.ON_DELETE:
            return "on task deletion"
        if self is WorktreeRemoval.ON_COMPLETION:
            return "on task completion"
        return "never"


WORKTREE_REMOVAL_OPTIONS: tuple[WorktreeRemoval, ...] = (
    WorktreeRemoval.ON_DELETE,
    WorktreeRemoval.ON_COMPLETION,
    WorktreeRemoval.NEVER,
)


class StallNudge(str, enum.Enum):
    """How long an agent/tester run may stay without any sign of life
    (no stream deltas, no tool events) before it is automatically aborted
    and restarted ("nudged"). A stalled run blocks its task — and via the
    pipeline gate, often the whole backlog — so the watchdog keeps an
    overnight run going when the model silently hangs."""

    MIN_15 = "15"
    MIN_30 = "30"
    HOUR_1 = "60"
    HOUR_2 = "120"
    NEVER = "never"

    @property
    def minutes(self) -> int | None:
        """Stall limit in minutes, or None when runs are never nudged."""
        if self is StallNudge.NEVER:
            return None
        return int(self.value)

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is StallNudge.NEVER:
            return "never"
        if self is StallNudge.HOUR_1:
            return "1 hour"
        if self is StallNudge.HOUR_2:
            return "2 hours"
        return f"{self.value} min"


STALL_NUDGE_OPTIONS: tuple[StallNudge, ...] = (
    StallNudge.MIN_15,
    StallNudge.MIN_30,
    StallNudge.HOUR_1,
    StallNudge.HOUR_2,
    StallNudge.NEVER,
)


class GitHubSyncInterval(str, enum.Enum):
    """How often the GitHub Issues sync runs (project setting)."""

    HOUR_1 = "1"
    HOUR_2 = "2"
    HOUR_4 = "4"

    @property
    def hours(self) -> int:
        return int(self.value)

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        return f"{self.value} hour" + ("s" if self.hours > 1 else "")


GITHUB_SYNC_INTERVAL_OPTIONS: tuple[GitHubSyncInterval, ...] = (
    GitHubSyncInterval.HOUR_1,
    GitHubSyncInterval.HOUR_2,
    GitHubSyncInterval.HOUR_4,
)


class ProjectSettings(BaseModel):
    completed_retention: CompletedRetention = CompletedRetention.DAYS_7
    task_finalization: TaskFinalization = TaskFinalization.TESTING_COMMIT
    # Fix-loop cap: how many consecutive development rounds the tester may
    # fail with a "the bug is this task's own change" verdict (retry_here)
    # before the failure completes the task as FAILED and the bug report is
    # created (branching from clean HEAD). Reset by a PASSED verdict or by
    # the user taking the task back by hand.
    max_dev_rounds: int = 3
    stall_nudge: StallNudge = StallNudge.MIN_30
    # Per-task worktree isolation (TaskIsolation docstring): a value that
    # the project cannot currently honour (worktree without a git repo) is
    # still persisted — the app warns about the dormant mode at every
    # startup instead of silently rewriting the user's choice.
    task_isolation: TaskIsolation = TaskIsolation.OFF
    # How many tasks may develop in parallel (development slots). Only
    # effective under worktree isolation — each parallel task needs its
    # own worktree, so without it the effective limit is 1 (the value is
    # still persisted; the app warns and behaves as 1). The tester stays
    # single: testing never occupies a development slot and never runs
    # concurrently with another tester round.
    parallel_tasks: int = 1
    # When the per-task worktree directory (.drcode/worktrees/<id>) is
    # removed automatically (WorktreeRemoval docstring). Only the
    # directory is governed: the branch `drcode/<id>` lives by RULE 2,
    # whatever this says.
    worktree_removal: WorktreeRemoval = WorktreeRemoval.ON_DELETE
    # How ESC stops a running agent/tester (EscAbort docstring): the safe
    # double-press default guards the "closed a dialog with ESC, pressed
    # ESC once more by inertia" accident; "instant" restores the
    # historical one-press stop.
    esc_abort: EscAbort = EscAbort.DOUBLE_PRESS
    # Keep the computer awake (a system-sleep assertion) while the pipeline
    # has work: any task QUEUED/ACTIVE/TESTING or a run in flight. macOS
    # runs `caffeinate -i -s`, Windows holds SetThreadExecutionState, Linux
    # is a no-op. A sleeping machine tears the network down and kills every
    # concurrent run with ModelConnectionError — the retry budget does not
    # survive a full sleep. Does NOT stop the clamshell (lid closed on
    # battery) sleep mode.
    keep_awake: bool = True
    # -- GitHub Issues integration ------------------------------------------
    # Enabled may be switched on even with no token/repo set (or when the
    # connection check fails): the sync simply logs an error each cycle
    # until the values are fixed.
    github_enabled: bool = False
    # Personal access token (fine-grained or classic, needs repo read access).
    github_token: str = ""
    # Full repository name "owner/repo" (a pasted URL is normalized on save).
    github_repo: str = ""
    github_sync_interval: GitHubSyncInterval = GitHubSyncInterval.HOUR_1
    # -- ntfy.sh push notifications -------------------------------------------
    # Enabled may be switched on with no topic set: the notifier simply
    # stays silent (and the settings dialog warns about the missing topic)
    # until a topic is provided.
    notifications_enabled: bool = False
    # Public ntfy.sh topic name to publish to (see https://ntfy.sh). Anyone
    # who knows the topic can read its messages — pick an unguessable one.
    ntfy_topic: str = ""
    # -- Voice I/O (ASR dictation + TTS announcements) -----------------------
    # OpenAI-compatible audio API endpoint (WITH the /v1 suffix; the audio
    # routes are /audio/transcriptions and /audio/speech below it).
    # Deliberately NOT the chat provider's base_url: gateways commonly serve
    # audio on a different route than the chat models. Empty (the default)
    # = every voice feature is off.
    voice_api_url: str = ""
    # Bearer token for the audio API. Empty = reuse the api_key of the
    # user-catalog provider whose base_url host matches voice_api_url's
    # host (no second token to manage for the common single-gateway case).
    voice_api_key: str = ""
    # Full ASR model id, e.g. "gigaam_rnnt_v3_e2e". Model ids may contain
    # "/" (unlike the agent-catalog ids), so they are stored here directly
    # instead of the model catalog.
    voice_asr_model: str = ""
    # TTS model id (e.g. "higgs-tts-3-4b") and the speaker voice (e.g.
    # "alloy"); both required for the announcements.
    voice_tts_model: str = ""
    voice_tts_voice: str = ""
    # Which events the TTS announces (each independent):
    voice_tts_completed: bool = False
    voice_tts_tester_failed: bool = False
    voice_tts_queue_empty: bool = False


class ProjectSettingsStore:
    """Loads/saves project settings from `<project>/.drcode/settings.json`."""

    def __init__(self, project_root: Path) -> None:
        self.path = project_root / ".drcode" / "settings.json"

    def load(self) -> ProjectSettings:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return ProjectSettings()
        except (OSError, json.JSONDecodeError):
            # Corrupt settings must never keep the app from starting.
            return ProjectSettings()
        try:
            return ProjectSettings.model_validate(data)
        except ValidationError:
            return ProjectSettings()

    def save(self, settings: ProjectSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(settings.model_dump_json(indent=2))
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
