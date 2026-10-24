"""Tests for the research accept path: `on_research_accepted` — the
workflow hook fired when a RESEARCH task completes (the research
counterpart of `on_accepted`, which research tasks never reach because
they skip testing) — and the unmerged-work reminder extended to research
tasks when the hook is not defined. Hook scripts run as real
subprocesses; the model layer is stubbed (no network anywhere) — same
style as tests/test_workflow.py / tests/test_unmerged_reminder.py."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from drcode import agent_model
from drcode.models import TaskKind, TaskStatus
from drcode.settings import (
    ProjectSettings,
    ProjectSettingsStore,
    TaskIsolation,
)
from drcode.task_env import TaskEnvironment, branch_for
from drcode.task_manager import TaskManager
from drcode.tui.app import DrCodeApp
from drcode.workflow import load_hooks
from tests.network_mocks import (
    ScriptedState,
    event_texts,
    text_response,
    wait_until,
)

# -- shared fixtures (mirroring tests/test_workflow.py / test_unmerged_reminder.py) --


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured (but never contacted) model endpoint — suppresses the
    first-start ModelSetupDialog so the app mounts straight into the
    pipeline state under test."""
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedState:
    """Every agent drcode builds is a FunctionModel consuming `steps`."""
    state = ScriptedState()
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())
    return state


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


@pytest.fixture
def git_root(project_root: Path) -> Path:
    """A project root that is a real git repository with one commit."""
    _git(project_root, "init", "-q")
    _git(project_root, "config", "user.email", "test@example.com")
    _git(project_root, "config", "user.name", "Test")
    (project_root / ".gitignore").write_text(".drcode/tasks/\n.drcode/worktrees/\n")
    (project_root / "README.md").write_text("hello")
    _git(project_root, "add", "-A")
    _git(project_root, "commit", "-q", "-m", "init")
    return project_root


@pytest.fixture
def isolated(git_root: Path) -> Path:
    """The git repo with worktree isolation persisted (an app picks it up)."""
    ProjectSettingsStore(git_root).save(
        ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    return git_root


def current_branch(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def make_script(root: Path, source: str) -> None:
    drcode = root / ".drcode"
    drcode.mkdir(parents=True, exist_ok=True)
    (drcode / "workflow.py").write_text(source)


def track_toasts(app: DrCodeApp) -> list[tuple[str, str]]:
    toasts: list[tuple[str, str]] = []

    def notify(message: str = "", *, title: str = "", **_: Any) -> None:
        toasts.append((title, message))

    app.notify = notify  # type: ignore[method-assign]
    return toasts


# -- hook scripts -----------------------------------------------------------------


def research_hook_events(task: Any) -> list[str]:
    return [t for t in event_texts(task) if "on_research_accepted" in t]


ECHO_HOOK_SCRIPT = """
import json

def on_research_accepted(task):
    with open("payload.json", "w") as f:
        json.dump(task, f)
    return {"text": "hook ran"}
"""

# The whole point of the hook: with isolation on it runs INSIDE the
# task's worktree, yet must be able to merge the docs branch into the
# user's main checkout — via the payload's `project` entry.
MERGE_HOOK_SCRIPT = """
import subprocess

def on_research_accepted(task):
    wt = task["worktree"]["path"]
    p = task["project"]
    with open(wt + "/docs.md", "w") as f:
        f.write("docs")
    subprocess.run(["git", "-C", wt, "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", wt, "commit", "-m", "docs"], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", p["path"], "merge", task["worktree"]["branch"], "-m", "merge docs"],
        check=True,
        capture_output=True,
    )
    return {"text": "docs merged"}
"""

CRASH_HOOK_SCRIPT = """
def on_research_accepted(task):
    raise RuntimeError("merge exploded")
"""


# -- hook discovery -------------------------------------------------------------------


def test_research_hook_discovered(project_root: Path) -> None:
    make_script(project_root, "def on_research_accepted(task):\n    return None\n")
    hooks = load_hooks(project_root)
    assert hooks is not None
    assert hooks.has("on_research_accepted")
    assert hooks.defined_hooks() == ["on_research_accepted"]
    assert "on_research_accepted" in hooks.describe()


# -- pipeline: the hook fires when a research task completes --------------------------


async def test_hook_fires_once_with_payload_without_isolation(
    git_root: Path, configured_env: None, scripted: ScriptedState
) -> None:
    """No isolation: the hook runs in the project directory, the payload
    carries the `project` entry (checkout path + base branch) and no
    `worktree` key — and fires exactly once."""
    make_script(git_root, ECHO_HOOK_SCRIPT)
    app = DrCodeApp(git_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        task = app.manager.create(
            "Write docs", "write the docs", kind=TaskKind.RESEARCH
        )
        scripted.steps = [text_response("docs done")]
        app._advance()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and any(t.endswith("hook ran") for t in research_hook_events(task))
            ),
            pilot,
            what="research task completed with the hook event",
        )
        await pilot.pause()
    payload = json.loads((git_root / "payload.json").read_text())
    assert payload["hook"] == "on_research_accepted"
    assert payload["task"]["id"] == task.id
    assert payload["task"]["kind"] == "research"
    assert payload["project"]["path"] == str(git_root)
    assert payload["project"]["base_branch"] == current_branch(git_root)
    assert "worktree" not in payload
    # Exactly once per completion.
    assert [t for t in research_hook_events(task) if t.endswith("hook ran")] == [
        "on_research_accepted:\nhook ran"
    ]


async def test_hook_fires_inside_worktree_with_isolation(
    isolated: Path, configured_env: None, scripted: ScriptedState
) -> None:
    """With worktree isolation the hook's cwd is the task's worktree (the
    payload.json lands there, `git.branch` describes the task branch) and
    the payload carries the `worktree` path/branch entry."""
    make_script(isolated, ECHO_HOOK_SCRIPT)
    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await pilot.pause()
        task = app.manager.create(
            "Write docs", "write the docs", kind=TaskKind.RESEARCH
        )
        scripted.steps = [text_response("docs done")]
        app._advance()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and any(t.endswith("hook ran") for t in research_hook_events(task))
            ),
            pilot,
            what="research task completed with the hook event",
        )
        await pilot.pause()
    wt = isolated / ".drcode" / "worktrees" / task.id
    payload = json.loads((wt / "payload.json").read_text())
    branch = f"drcode/{task.id}"
    assert payload["worktree"] == {"path": str(wt), "branch": branch}
    assert payload["git"]["branch"] == branch
    assert payload["project"]["path"] == str(isolated)
    assert payload["project"]["base_branch"] == current_branch(isolated)


async def test_hook_can_merge_docs_branch_into_main_checkout(
    isolated: Path, configured_env: None, scripted: ScriptedState
) -> None:
    """The hook's whole point: run inside the worktree, commit the docs
    there and merge the branch into the user's main checkout via the
    payload's `project` entry — afterwards nothing is unmerged."""
    make_script(isolated, MERGE_HOOK_SCRIPT)
    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await pilot.pause()
        task = app.manager.create(
            "Write docs", "write the docs", kind=TaskKind.RESEARCH
        )
        scripted.steps = [text_response("docs done")]
        app._advance()
        await wait_until(
            lambda: (isolated / "docs.md").exists(),
            pilot,
            what="docs.md merged into the main checkout",
        )
        # The merge lands inside the hook subprocess; the "docs merged"
        # service event is recorded by the pipeline right after the hook
        # call returns — wait for it too (a bare pause raced the
        # continuation and lost the event on app teardown).
        await wait_until(
            lambda: any(
                t.endswith("docs merged") for t in research_hook_events(task)
            ),
            pilot,
            what="docs-merged hook event",
        )
    assert any(t.endswith("docs merged") for t in research_hook_events(task))
    env = TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    assert env.unmerged_commits(task.id) == 0


async def test_hook_failure_is_non_fatal(
    git_root: Path, configured_env: None, scripted: ScriptedState
) -> None:
    """A crashing hook is logged into the service-event log; the task
    stays completed and the pipeline moves on."""
    make_script(git_root, CRASH_HOOK_SCRIPT)
    app = DrCodeApp(git_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        task = app.manager.create(
            "Write docs", "write the docs", kind=TaskKind.RESEARCH
        )
        scripted.steps = [text_response("docs done")]
        app._advance()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and any(
                    "on_research_accepted failed (non-fatal)" in t
                    and "merge exploded" in t
                    for t in research_hook_events(task)
                )
            ),
            pilot,
            what="hook failure logged",
        )
        await pilot.pause()
    assert task.status is TaskStatus.COMPLETED
    assert not app.agent_running


# -- startup scan includes research tasks ---------------------------------------------


async def test_startup_scan_toasts_unmerged_research_branch(
    isolated: Path, configured_env: None
) -> None:
    """A completed research task whose docs branch was never merged is
    announced at startup (without the hook, nothing else would ever nag)."""
    manager = TaskManager(isolated)
    task = manager.create("Write docs", "docs", kind=TaskKind.RESEARCH)
    task.status = TaskStatus.COMPLETED
    manager.storage.save(task)
    env = TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    wt, _ = env.ensure_root(task.id)
    (wt / "docs.md").write_text("docs")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "docs")

    app = DrCodeApp(isolated)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: any(title == "Unmerged work" for title, _ in toasts),
            pilot,
            what="startup unmerged-work toast",
        )
    unmerged = [msg for title, msg in toasts if title == "Unmerged work"]
    assert len(unmerged) == 1
    assert f"drcode/{task.id}" in unmerged[0]


async def test_startup_scan_research_not_suppressed_by_on_accepted(
    isolated: Path, configured_env: None
) -> None:
    """`on_accepted` automates development-task merges only — a research
    task's unmerged docs branch still produces the startup toast."""
    manager = TaskManager(isolated)
    task = manager.create("Write docs", "docs", kind=TaskKind.RESEARCH)
    task.status = TaskStatus.COMPLETED
    manager.storage.save(task)
    env = TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    wt, _ = env.ensure_root(task.id)
    (wt / "docs.md").write_text("docs")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "docs")

    make_script(isolated, "def on_accepted(task):\n    return None\n")

    app = DrCodeApp(isolated)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: any(title == "Unmerged work" for title, _ in toasts),
            pilot,
            what="startup unmerged-work toast",
        )
    unmerged = [msg for title, msg in toasts if title == "Unmerged work"]
    assert len(unmerged) == 1
    assert f"drcode/{task.id}" in unmerged[0]


async def test_startup_scan_research_suppressed_by_research_hook(
    isolated: Path, configured_env: None
) -> None:
    manager = TaskManager(isolated)
    task = manager.create("Write docs", "docs", kind=TaskKind.RESEARCH)
    task.status = TaskStatus.COMPLETED
    manager.storage.save(task)
    env = TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    wt, _ = env.ensure_root(task.id)
    (wt / "docs.md").write_text("docs")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "docs")
    # The hook's claim must hold: the docs merge really landed (the
    # tracking audits git, not just the hook's presence).
    _git(
        isolated,
        "merge",
        "-q",
        "--no-ff",
        branch_for(task.id),
        "-m",
        "merge docs",
    )

    make_script(isolated, "def on_research_accepted(task):\n    return None\n")

    app = DrCodeApp(isolated)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._unmerged_work_startup_scan()
        await pilot.pause()
    assert not any(title == "Unmerged work" for title, _ in toasts)
