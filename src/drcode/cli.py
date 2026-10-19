"""Headless CLI entry points: one-shot commands that run without the TUI.

Currently hosts `--git-hub-sync`: a single GitHub Issues sync cycle
executed against the project's configured integration, after which the
process exits without ever opening the user interface. Imported tasks
are queued like they would be in the app; the pipeline itself (testing
backlog, promotion) is left to the next interactive start.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from .github import GitHubClient, GitHubError, GitHubSyncStateStore, sync_issues
from .models import TaskKind
from .settings import ProjectSettingsStore
from .task_manager import TaskManager


def run_github_sync(project_root: Path) -> int:
    """One-shot GitHub Issues sync; returns the process exit code.

    Requires the integration to be configured in the project settings
    (`.drcode/settings.json`): enabled, with a token and a repository
    name. On success prints the cycle summary and returns 0; a missing
    integration returns 1, a failed sync (network/API error) returns 2.
    """
    settings = ProjectSettingsStore(project_root).load()
    if not (settings.github_enabled and settings.github_token and settings.github_repo):
        print(
            "GitHub integration is not configured for this project "
            f"({project_root}): set it up in the TUI "
            "(Settings → GitHub integration) and try again.",
            file=sys.stderr,
        )
        return 1

    manager = TaskManager(project_root)
    state_store = GitHubSyncStateStore(project_root)
    state = state_store.load()
    client = GitHubClient(settings.github_token, settings.github_repo)
    try:
        imported, skipped = asyncio.run(sync_issues(manager, client, state))
    except GitHubError as e:
        print(f"GitHub sync failed: {e}", file=sys.stderr)
        return 2
    # The caller is responsible for persisting the sync state (github.py):
    # stamp the cycle so the seen issues are never reconsidered.
    state_store.save(state)
    print(
        f"GitHub sync ({settings.github_repo}): "
        f"{imported} new task(s) imported, {skipped} skipped."
    )
    return 0


def create_task_from_cli(project_root: Path, kind: TaskKind, description: str) -> int:
    """One-shot task creation without the TUI; returns the exit code.

    Stores the description as the task's first queued message (the title
    is derived from its first line, like in the app) and prints the id of
    the created task. The pipeline itself is left to the next interactive
    start — no agent run is started here.
    """
    text = description.strip()
    if not text:
        print(
            "error: the task description is empty — pass a non-empty text "
            "after the option.",
            file=sys.stderr,
        )
        return 1

    manager = TaskManager(project_root)
    task = manager.create("", text, kind=kind)
    print(f"Task {task.id} ({kind.value}) created successfully.")
    return 0
