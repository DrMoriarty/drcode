"""Regression tests for the resolver racing a merge (#94033d3f): the
resolver's triggers (the hook-merge audit re-stamp, the startup scan)
act on a BATCHED mergedness snapshot — between that snapshot and the
round's start the hook (or a hand) can merge the branch into the base.
The round then had nothing to resolve yet still ran: `ensure_root`
re-created a worktree removed at completion and `prepare` parked a
pointless "Merge branch 'main' into drcode/<id>" commit, formally
re-flagging the branch as unmerged.

The fix: `run_resolver` re-reads mergedness FRESH (no batch cache)
before touching anything and no-ops on a fully reachable branch; the
audit re-stamp (`_hook_merge_still_owns`) fresh-checks before stamping
`hook_merge_failed` / starting the resolver.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent

from drcode.models import TaskKind, TaskStatus, VerificationStatus
from drcode.settings import ProjectSettings, ProjectSettingsStore, TaskIsolation
from drcode.task_env import TaskEnvironment
from drcode.tui.app import DrCodeApp
from tests.network_mocks import (
    ScriptedState,
    api_error,
    event_texts,
    wait_until,
)


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def _rev_parse(root: Path, ref: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def isolated(project_root: Path) -> Path:
    """A git repo with worktree isolation persisted."""
    _git(project_root, "init", "-q")
    _git(project_root, "config", "user.email", "test@example.com")
    _git(project_root, "config", "user.name", "Test")
    (project_root / ".gitignore").write_text(".drcode/tasks/\n.drcode/worktrees/\n")
    (project_root / "shared.txt").write_text("base")
    _git(project_root, "add", "-A")
    _git(project_root, "commit", "-q", "-m", "init")
    ProjectSettingsStore(project_root).save(
        ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    return project_root


@pytest.fixture
def env(isolated: Path) -> TaskEnvironment:
    return TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )


def _finished_task(app: DrCodeApp, kind: TaskKind = TaskKind.FEATURE) -> Any:
    task = app.manager.create("Fix the bug", "fix the bug")
    task.kind = kind
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    # Past the hook-merge audit grace so the audit path is live.
    task.completed_at = datetime.now(UTC) - timedelta(hours=1)
    app.manager.storage.save(task)
    return task


def _commit_on_branch(
    env: TaskEnvironment, task_id: str, name: str = "fix.txt"
) -> None:
    """One commit on `drcode/<task-id>` via the task's worktree."""
    wt, _ = env.ensure_root(task_id)
    (wt / name).write_text(name)
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", f"fix for {task_id}")


def _quiesce(app: DrCodeApp) -> None:
    async def _noop() -> None:
        return None

    app._unmerged_work_startup_scan = _noop  # type: ignore[method-assign]
    app._poll_git_indicator = _noop  # type: ignore[method-assign]
    app._poll_pending_installs = _noop  # type: ignore[method-assign]
    app.run_agent = lambda task_id: None  # type: ignore[method-assign]


async def test_resolver_skips_fully_merged_branch(
    isolated: Path, env: TaskEnvironment, configured_env: None
) -> None:
    """A branch fully reachable from the base (merged between the
    trigger's snapshot and the round): the round is a no-op — no
    worktree re-creation, no merge commit, the stale
    `hook_merge_failed` flag cleared."""
    app = DrCodeApp(isolated)
    task = _finished_task(app)
    _commit_on_branch(env, task.id)
    branch = env.branch(task.id)
    # The hook (or a hand) merged the branch into the base — fast-forward,
    # the branch tip is fully reachable from it.
    _git(isolated, "merge", "--ff-only", branch)
    tip_before = _rev_parse(isolated, branch)
    # worktree_removal: on_completion — the directory was removed at
    # completion, the branch kept.
    wt = env.worktree_path(task.id)
    _git(isolated, "worktree", "remove", "--force", str(wt))
    assert not wt.is_dir()
    # The stale audit stamp from the racing snapshot.
    task.hook_merge_failed = True
    app.manager.storage.save(task)

    _quiesce(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        app._maybe_start_resolver(task.id)
        await wait_until(
            lambda: any("already merged" in t for t in event_texts(live)),
            pilot,
            what="resolver no-op event",
        )
        # The worktree was NOT re-created and the branch got no merge
        # commit (the tip is unchanged — still exactly reachable).
        assert not wt.is_dir()
        assert _rev_parse(isolated, branch) == tip_before
        assert env.unmerged_commits(task.id) == 0
        # The stale flag is cleared, not left for the toasts/reminders.
        assert not live.hook_merge_failed


async def test_resolver_runs_for_unmerged_branch(
    isolated: Path,
    env: TaskEnvironment,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely unmerged branch: the round works exactly as before —
    ensure_root runs, prepare parks the conflict, the agent round
    starts."""
    app = DrCodeApp(isolated)
    task = _finished_task(app)
    # The branch diverged from the base on shared.txt — prepare()'s
    # base-into-worktree merge parks a real conflict.
    wt, _ = env.ensure_root(task.id)
    (wt / "shared.txt").write_text("task side")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "task work")
    (isolated / "shared.txt").write_text("base side")
    _git(isolated, "add", "-A")
    _git(isolated, "commit", "-q", "-m", "base work")
    assert env.unmerged_commits(task.id) == 1

    _quiesce(app)
    # The model endpoint fails with a retryable error — the test asserts
    # the round STARTED (the fresh check passed), not its conclusion.
    import drcode.core.pipeline_resolver as pr

    state = ScriptedState()
    state.steps = [api_error()]

    def fake_build_agent(
        config: Any, root: Path | None = None, task_id: str | None = None
    ) -> Agent[Any, str]:
        return Agent(state.model())

    # Restored on teardown — a leaked stub (an agent with NO tools) would
    # poison every later resolver test scheduled on the same xdist worker.
    monkeypatch.setattr(pr, "build_resolver_agent", fake_build_agent)
    # Burn the retries fast — the test is over once the round started.
    monkeypatch.setattr(pr, "_RETRY_BASE_DELAY_S", 0.05)

    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        app._maybe_start_resolver(task.id)
        await wait_until(
            lambda: any(
                "starting an automatic resolution round" in t for t in event_texts(live)
            ),
            pilot,
            what="resolver round start",
        )
        assert wt.is_dir()  # the worktree is in place, as before


async def test_hook_audit_fresh_check_before_stamping(
    isolated: Path, env: TaskEnvironment, configured_env: None
) -> None:
    """`_hook_merge_still_owns` with a STALE batched snapshot: a branch
    merged since the snapshot must NOT be back-stamped
    hook_merge_failed nor handed to the resolver; a genuinely unmerged
    branch still is."""
    app = DrCodeApp(isolated)
    task = _finished_task(app)
    _commit_on_branch(env, task.id)
    branch = env.branch(task.id)
    started: list[str] = []
    app._maybe_start_resolver = started.append  # type: ignore[method-assign]
    _quiesce(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        # Stale snapshot says unmerged, but the branch is in fact fully
        # reachable — the fresh re-read must win: no stamp, no resolver.
        _git(isolated, "merge", "--ff-only", branch)
        assert env.unmerged_commits(task.id) == 0
        assert app._hook_merge_still_owns(live, "on_accepted", {branch}) is True
        assert not live.hook_merge_failed
        assert started == []
        # A genuinely unmerged branch (a second commit): the audit still
        # stamps the failure and starts the resolver round. Drop the
        # audited-merged cache the first call installed — it is keyed per
        # (id, completed_at) and would short-circuit the second audit.
        app._hook_merge_audit_ok.clear()
        _commit_on_branch(env, task.id, name="fix2.txt")
        assert env.unmerged_commits(task.id) == 1
        assert app._hook_merge_still_owns(live, "on_accepted", {branch}) is False
        assert live.hook_merge_failed
        assert started == [live.id]
