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
    task — it is only logged). TESTING: the tester verifies but never
    commits. NONE: no tester at all — tasks go straight to COMPLETED.
    """

    TESTING_COMMIT = "testing_commit"
    TESTING = "testing"
    NONE = "none"

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is TaskFinalization.TESTING_COMMIT:
            return "Testing + Git commit + push"
        if self is TaskFinalization.TESTING:
            return "Testing"
        return "None"


FINALIZATION_OPTIONS: tuple[TaskFinalization, ...] = (
    TaskFinalization.TESTING_COMMIT,
    TaskFinalization.TESTING,
    TaskFinalization.NONE,
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
    stall_nudge: StallNudge = StallNudge.MIN_30
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
