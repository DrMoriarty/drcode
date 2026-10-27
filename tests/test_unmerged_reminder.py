"""Tests for the unmerged-work reminder: after a tester PASSED finalizes
a task (worktree isolation on), drcode checks whether the task's
`drcode/<task-id>` branch holds commits the main checkout's current
branch does not have, and reminds the user explicitly — merging is a
team decision (drcode never merges), but it must not be a silent one.
Covers `TaskEnvironment.unmerged_commits` / `base_branch`, the pipeline
reminder on the finalization paths (toast + service event, softened by
an `on_accepted` hook — research tasks are covered by the same reminder
with `on_research_accepted` wording) and the startup sweep for old stuck
fixes."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from drcode.models import TaskKind, TaskStatus, VerificationStatus
from drcode.settings import (
    ProjectSettings,
    ProjectSettingsStore,
    TaskIsolation,
)
from drcode.task_env import TaskEnvironment, branch_for
from drcode.task_manager import TaskManager
from drcode.tui.app import DrCodeApp
from drcode.workflow import load_hooks
from tests.network_mocks import event_texts, wait_until

# -- shared fixtures (mirroring tests/test_task_env.py / test_workflow.py) --------


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured (but never contacted) model endpoint — suppresses the
    first-start ModelSetupDialog so the app mounts straight into the
    pipeline state under test."""
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


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


@pytest.fixture
def env(git_root: Path) -> TaskEnvironment:
    return TaskEnvironment(
        git_root, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )


def commit_on_branch(env: TaskEnvironment, task_id: str, name: str = "fix.txt") -> None:
    """Give the task's branch one commit the current branch lacks — via
    the task's worktree, the way a real finished task leaves its work."""
    wt, _ = env.ensure_root(task_id)
    (wt / name).write_text(name)
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", f"fix for {task_id}")


def current_branch(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# -- TaskEnvironment.unmerged_commits ----------------------------------------------


def test_unmerged_commits_counts_task_branch_work(env: TaskEnvironment) -> None:
    root = env.project_root
    env.ensure_root("t1")
    commit_on_branch(env, "t1")
    assert env.unmerged_commits("t1") == 1
    assert env.base_branch() == current_branch(root)


def test_unmerged_commits_zero_after_merge(env: TaskEnvironment) -> None:
    root = env.project_root
    env.ensure_root("t1")
    commit_on_branch(env, "t1")
    _git(root, "merge", "-q", "--no-ff", branch_for("t1"), "-m", "merge t1")
    assert env.unmerged_commits("t1") == 0


def test_unmerged_commits_zero_without_branch(env: TaskEnvironment) -> None:
    env.ensure_root("t1")  # worktree exists, but nothing was committed
    assert env.unmerged_commits("t1") == 0
    # No worktree at all (task never activated / already cleaned up).
    assert env.unmerged_commits("t2") == 0


def test_unmerged_commits_zero_without_git_repo(project_root: Path) -> None:
    env = TaskEnvironment(
        project_root, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    assert not env.available
    assert env.unmerged_commits("t1") == 0


def test_unmerged_commits_zero_without_isolation(git_root: Path) -> None:
    env = TaskEnvironment(git_root, ProjectSettings(task_isolation=TaskIsolation.OFF))
    assert not env.active
    assert env.unmerged_commits("t1") == 0


# -- pipeline reminder on the PASSED finalization paths -----------------------------


def quiet_background_tracking(app: DrCodeApp) -> None:
    """Neutralize the mount-time background unmerged-tracking actors (the
    startup-scan worker, the git-indicator poll and the deferred-install
    poll — all three run `_unmerged_completed_ids` in worker threads, see
    DrCodeApp.on_mount). The tests below drive that tracking synchronously
    and assert on its state (flags, `_merge_state`, toasts) immediately
    after their own mutations: a background pass delayed under load would
    interleave with them — auditing fresh tasks (grace is skipped for
    completed_at=None), back-stamping `hook_merge_failed`, starting the
    conflict resolver (whose engine merge reshapes the task branch behind
    the git-cherry guard) or clearing a freshly stamped flag via the
    stale-flag branch. Quiescing the actors makes the synchronous asserts
    deterministic instead of racing a scheduler."""

    async def _noop() -> None:
        return None

    app._unmerged_work_startup_scan = _noop  # type: ignore[method-assign]
    app._poll_git_indicator = _noop  # type: ignore[method-assign]
    app._poll_pending_installs = _noop  # type: ignore[method-assign]
    app._maybe_start_resolver = lambda task_id: None  # type: ignore[method-assign]


@pytest.fixture
def app_env(isolated: Path, configured_env: None) -> DrCodeApp:
    app = DrCodeApp(isolated)
    quiet_background_tracking(app)
    return app


def finished_task(app: DrCodeApp, kind: TaskKind = TaskKind.FEATURE) -> Any:
    task = app.manager.create("Fix the bug", "fix the bug")
    task.kind = kind
    task.status = TaskStatus.COMPLETED
    # mark_done always stamps this; without it the hook-merge audit treats
    # the task as ancient (the grace check skips completed_at=None) and a
    # delayed background pass back-stamps hook_merge_failed and starts the
    # conflict resolver mid-test (#d1f3fa59).
    task.completed_at = datetime.now(UTC)
    if kind is not TaskKind.RESEARCH:
        task.verification = VerificationStatus.PASSED
    # A real completion stamps completed_at (mark_done). Without it the
    # app's startup audit (_hook_merge_still_owns) treats the task as
    # ancient: past HOOK_MERGE_AUDIT_GRACE it back-stamps
    # hook_merge_failed and starts the conflict resolver, whose
    # base-into-branch merge then races the hook under test — it can land
    # before the wrong-fork guard's `git cherry` and hide the patch-id
    # duplicates (the branch now contains everything the base has), so
    # the guard passes the merge through and the tests flake. A fresh
    # completed_at keeps the task inside the grace period: the audit
    # stands down, no resolver round.
    task.completed_at = datetime.now(UTC)
    app.manager.storage.save(task)
    return task


def track_toasts(app: DrCodeApp) -> list[tuple[str, str]]:
    toasts: list[tuple[str, str]] = []

    def notify(message: str = "", *, title: str = "", **_: Any) -> None:
        toasts.append((title, message))

    app.notify = notify  # type: ignore[method-assign]
    return toasts


async def test_reminder_fires_for_unmerged_task(app_env: DrCodeApp) -> None:
    """The task's branch holds work the current branch lacks: the
    finalization reminder lands in the service-event log AND as a toast."""
    app = app_env
    task = finished_task(app)
    commit_on_branch(app.task_env, task.id)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._remind_unmerged_work(task)
        await pilot.pause()
    base = current_branch(app.project_root)
    events = "\n".join(event_texts(task))
    assert f"drcode/{task.id}" in events
    assert f"not merged into {base}" in events
    assert f"git merge drcode/{task.id}" in events
    assert any(title == "Unmerged work" for title, _ in toasts)


async def test_reminder_silent_when_merged_or_absent(app_env: DrCodeApp) -> None:
    app = app_env
    task = finished_task(app)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        # No task branch at all (isolation just turned on / cancelled task).
        await app._remind_unmerged_work(task)
        # A branch fully merged into the current one.
        commit_on_branch(app.task_env, task.id)
        _git(
            app.project_root,
            "merge",
            "-q",
            "--no-ff",
            branch_for(task.id),
            "-m",
            "merge it",
        )
        await app._remind_unmerged_work(task)
        await pilot.pause()
    # The only event is the branch-creation record (the worktree
    # environment logs every drcode/<id> creation now); no REMINDER of
    # unmerged work — the branch is merged.
    assert not [t for t in event_texts(task) if "not merged" in t]
    assert not toasts


async def test_reminder_fires_for_research_task(app_env: DrCodeApp) -> None:
    """Research tasks complete without the tester — the reminder covers
    them too, with research wording (docs, on_research_accepted)."""
    app = app_env
    task = finished_task(app, kind=TaskKind.RESEARCH)
    commit_on_branch(app.task_env, task.id)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._remind_unmerged_work(task)
        await pilot.pause()
    base = current_branch(app.project_root)
    events = "\n".join(event_texts(task))
    assert "Documentation committed on branch" in events
    assert f"not merged into {base}" in events
    assert "on_research_accepted" in events
    assert f"git merge drcode/{task.id}" in events
    assert any(title == "Unmerged work" for title, _ in toasts)


async def test_research_reminder_toast_softened_by_research_hook(
    app_env: DrCodeApp,
) -> None:
    """With an on_research_accepted hook defined the docs merge is
    automated — the service event stays, the toast is skipped (an
    on_accepted hook alone does not suppress it)."""
    app = app_env
    drcode_dir = app.project_root / ".drcode"
    drcode_dir.mkdir(exist_ok=True)
    (drcode_dir / "workflow.py").write_text(
        "def on_research_accepted(task):\n    return {'text': 'merged'}\n"
    )
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app, kind=TaskKind.RESEARCH)
    # The hook has just reported "merged" — within the audit grace its
    # claim is trusted (the self-healing audit itself is covered below).
    task.completed_at = datetime.now(UTC)
    app.manager.storage.save(task)
    commit_on_branch(app.task_env, task.id)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._remind_unmerged_work(task)
        await pilot.pause()
    assert any("Documentation committed" in text for text in event_texts(task))
    assert not toasts


async def test_reminder_toast_softened_by_on_accepted_hook(
    app_env: DrCodeApp,
) -> None:
    """With an on_accepted hook defined the merge is automated — the
    service event stays (the durable record), the toast is skipped."""
    app = app_env
    drcode_dir = app.project_root / ".drcode"
    drcode_dir.mkdir(exist_ok=True)
    (drcode_dir / "workflow.py").write_text(
        "def on_accepted(task):\n    return {'text': 'merged'}\n"
    )
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    # The hook has just reported "merged" — within the audit grace its
    # claim is trusted (the self-healing audit itself is covered below).
    task.completed_at = datetime.now(UTC)
    app.manager.storage.save(task)
    commit_on_branch(app.task_env, task.id)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._remind_unmerged_work(task)
        await pilot.pause()
    assert any(f"drcode/{task.id}" in text for text in event_texts(task))
    assert not toasts


# -- startup sweep: old stuck fixes are surfaced once per session --------------------


async def test_startup_scan_toasts_stuck_fixes(
    isolated: Path, configured_env: None
) -> None:
    """A completed task whose branch never got merged is announced at
    startup; merged/absent branches are not."""
    manager = TaskManager(isolated)
    stuck = manager.create("Fix the bug", "fix the bug")
    stuck.status = TaskStatus.COMPLETED
    stuck.verification = VerificationStatus.PASSED
    manager.storage.save(stuck)
    merged = manager.create("Already merged", "merged work")
    merged.status = TaskStatus.COMPLETED
    merged.verification = VerificationStatus.PASSED
    manager.storage.save(merged)
    env = TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    commit_on_branch(env, stuck.id)

    app = DrCodeApp(isolated)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: any(title == "Unmerged work" for title, _ in toasts),
            pilot,
            what="startup unmerged-work toast",
        )
    unmerged_toasts = [msg for title, msg in toasts if title == "Unmerged work"]
    assert len(unmerged_toasts) == 1
    assert f"drcode/{stuck.id}" in unmerged_toasts[0]
    assert f"drcode/{merged.id}" not in unmerged_toasts[0]


async def test_startup_scan_silent_without_isolation(
    git_root: Path, configured_env: None
) -> None:
    """Without worktree isolation nothing ever lives on a task branch —
    the startup scan does not run at all."""
    manager = TaskManager(git_root)
    task = manager.create("Fix the bug", "fix the bug")
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    manager.storage.save(task)
    env = TaskEnvironment(
        git_root, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    commit_on_branch(env, task.id)  # branch exists anyway

    app = DrCodeApp(git_root)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._unmerged_work_startup_scan()
        await pilot.pause()
    assert not toasts


async def test_startup_scan_silent_with_on_accepted_hook(
    isolated: Path, configured_env: None
) -> None:
    manager = TaskManager(isolated)
    task = manager.create("Fix the bug", "fix the bug")
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    manager.storage.save(task)
    env = TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    commit_on_branch(env, task.id)
    # The hook's claim must hold: the merge really landed (the tracking
    # audits git, not just the hook's presence).
    _git(
        isolated,
        "merge",
        "-q",
        "--no-ff",
        branch_for(task.id),
        "-m",
        "merge it",
    )

    drcode_dir = isolated / ".drcode"
    drcode_dir.mkdir(exist_ok=True)
    (drcode_dir / "workflow.py").write_text(
        "def on_accepted(task):\n    return {'text': 'merged'}\n"
    )

    app = DrCodeApp(isolated)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._unmerged_work_startup_scan()
        await pilot.pause()
    assert not any(title == "Unmerged work" for title, _ in toasts)


# -- failed hook merges re-enter the unmerged-work tracking --------------------------


def _workflow_file(isolated: Path, source: str) -> None:
    drcode_dir = isolated / ".drcode"
    drcode_dir.mkdir(exist_ok=True)
    (drcode_dir / "workflow.py").write_text(source)


async def test_crashed_hook_merge_sets_hook_merge_failed(app_env: DrCodeApp) -> None:
    """A crashed (nonzero-exit) on_accepted hook — typically a failed
    `git merge` — stamps `task.hook_merge_failed` and persists it."""
    app = app_env
    _workflow_file(app.project_root, "def on_accepted(task):\n    raise RuntimeError\n")
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    commit_on_branch(app.task_env, task.id)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._run_accepted_hook(task)
        await pilot.pause()
    assert task.hook_merge_failed
    reloaded = TaskManager(app.project_root).get(task.id)
    assert reloaded.hook_merge_failed


async def test_failed_hook_merge_text_sets_hook_merge_failed(
    app_env: DrCodeApp,
) -> None:
    """The workflow template reports a conflicted merge as free text
    ("… FAILED — CONFLICT", "NOT merged …") with a zero exit — the text
    marks the flag too."""
    app = app_env
    _workflow_file(
        app.project_root,
        "def on_accepted(task):\n"
        "    return {'text': 'merge of drcode/<id> into main "
        "FAILED — CONFLICT (content)'}\n",
    )
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._run_accepted_hook(task)
        await pilot.pause()
    assert task.hook_merge_failed


async def test_successful_hook_merge_clears_hook_merge_failed(
    app_env: DrCodeApp,
) -> None:
    """A successful hook-merge clears the flag (a later successful re-run
    after a manual fix does too)."""
    app = app_env
    _workflow_file(
        app.project_root, "def on_accepted(task):\n    return {'text': 'merged'}\n"
    )
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    task.hook_merge_failed = True
    app.manager.storage.save(task)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._run_accepted_hook(task)
        await pilot.pause()
    assert not task.hook_merge_failed
    reloaded = TaskManager(app.project_root).get(task.id)
    assert not reloaded.hook_merge_failed


async def test_failed_hook_merge_counts_in_unmerged_completed_ids(
    app_env: DrCodeApp,
) -> None:
    """With an on_accepted hook defined, a completed task is normally
    invisible to the unmerged-work tracking — but only while its branch
    is really merged (the tracking audits git, not just the flag);
    research tasks get the same treatment under their own hook."""
    app = app_env
    _workflow_file(app.project_root, "def on_accepted(task):\n    return None\n")
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    research = finished_task(app, kind=TaskKind.RESEARCH)
    commit_on_branch(app.task_env, task.id)
    commit_on_branch(app.task_env, research.id)
    _git(
        app.project_root,
        "merge",
        "-q",
        "--no-ff",
        branch_for(task.id),
        "-m",
        "merge it",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        # Merged branch, no flag: the development task is hidden
        # (on_accepted owns its merge — and the audit sees it landed);
        # the research task counts anyway — on_accepted does not
        # automate research merges, and its branch is unmerged.
        assert app._unmerged_completed_ids() == [research.id]
        # A failed hook-merge — the development task counts again too.
        commit_on_branch(app.task_env, task.id, name="fix2.txt")
        task.hook_merge_failed = True
        assert sorted(app._unmerged_completed_ids()) == sorted([task.id, research.id])
        # The research counterpart: on_research_accepted defined instead —
        # it hides the research task while its merge really landed.
        (app.project_root / ".drcode" / "workflow.py").write_text(
            "def on_research_accepted(task):\n    return None\n"
        )
        app.workflow = load_hooks(app.project_root)
        commit_on_branch(app.task_env, research.id, name="docs2.txt")
        _git(
            app.project_root,
            "merge",
            "-q",
            "--no-ff",
            branch_for(research.id),
            "-m",
            "merge docs",
        )
        assert app._unmerged_completed_ids() == [task.id]
        research.hook_merge_failed = True
        commit_on_branch(app.task_env, research.id, name="docs3.txt")
        assert sorted(app._unmerged_completed_ids()) == sorted([task.id, research.id])
        await pilot.pause()


async def test_startup_scan_toasts_failed_hook_merge(
    isolated: Path, configured_env: None
) -> None:
    """The counterpart of the suppressed case: with an on_accepted hook
    defined, a completed task whose hook-merge FAILED still produces the
    startup toast (a stuck conflict must not vanish silently)."""
    manager = TaskManager(isolated)
    task = manager.create("Fix the bug", "fix the bug")
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    task.hook_merge_failed = True
    manager.storage.save(task)
    env = TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    commit_on_branch(env, task.id)

    _workflow_file(isolated, "def on_accepted(task):\n    return None\n")
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


async def test_reminder_toast_not_softened_after_hook_merge_failure(
    app_env: DrCodeApp,
) -> None:
    """The finalization reminder's toast is suppressed while the accept
    hook owns the merge — but after a failed hook-merge the user must
    hear about the stuck branch again."""
    app = app_env
    _workflow_file(app.project_root, "def on_accepted(task):\n    return None\n")
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    task.hook_merge_failed = True
    commit_on_branch(app.task_env, task.id)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._remind_unmerged_work(task)
        await pilot.pause()
    assert any(f"drcode/{task.id}" in text for text in event_texts(task))
    assert any(title == "Unmerged work" for title, _ in toasts)


# -- self-healing audit: the flag is stamped by the live pipeline, but a
# long-lived session on older code (or a crash mid-stamp) never stamps it —
# the tracking must not take the flag's word for a successful merge ----------


async def test_startup_scan_self_heals_unstamped_hook_merge(
    isolated: Path, configured_env: None
) -> None:
    """Version-skew reproduction (#8322696b): a process without the
    stamping code finalized the task — only the FAILED event exists,
    `hook_merge_failed` stayed False and the counter stayed hidden. The
    startup scan audits the branch past the grace period, back-stamps
    the flag (persisted, so `_remind_unmerged_work` sees it too) and
    counts the task."""
    manager = TaskManager(isolated)
    task = manager.create("Fix the bug", "fix the bug")
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    task.completed_at = datetime.now(UTC) - timedelta(minutes=10)
    manager.storage.save(task)
    env = TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    commit_on_branch(env, task.id)
    _workflow_file(isolated, "def on_accepted(task):\n    return None\n")

    app = DrCodeApp(isolated)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: any(title == "Unmerged work" for title, _ in toasts),
            pilot,
            what="self-healed unmerged-work toast",
        )
    unmerged = [msg for title, msg in toasts if title == "Unmerged work"]
    assert len(unmerged) == 1
    assert f"drcode/{task.id}" in unmerged[0]
    reloaded = TaskManager(isolated).get(task.id)
    assert reloaded.hook_merge_failed


async def test_hook_merge_audit_respects_grace(app_env: DrCodeApp) -> None:
    """A task finalized moments ago may still have its hook-merge in
    flight — within the grace period the hook is trusted (no count, no
    back-stamp); once the grace passes, an unmerged branch counts and is
    back-stamped."""
    app = app_env
    _workflow_file(app.project_root, "def on_accepted(task):\n    return None\n")
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    task.completed_at = datetime.now(UTC)
    app.manager.storage.save(task)
    commit_on_branch(app.task_env, task.id)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._unmerged_completed_ids() == []
        assert not task.hook_merge_failed
        task.completed_at = datetime.now(UTC) - timedelta(minutes=10)
        app.manager.storage.save(task)
        assert app._unmerged_completed_ids() == [task.id]
        assert task.hook_merge_failed
        reloaded = TaskManager(app.project_root).get(task.id)
        assert reloaded.hook_merge_failed
        await pilot.pause()


async def test_stale_hook_merge_failed_cleared_after_manual_merge(
    app_env: DrCodeApp,
) -> None:
    """A manual `git merge drcode/<id>` merges the branch — the stale
    `hook_merge_failed` flag is dropped instead of surviving forever
    (the dynamic check hides the task either way, but the flag would
    leak into toasts/reminders)."""
    app = app_env
    _workflow_file(app.project_root, "def on_accepted(task):\n    return None\n")
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    task.hook_merge_failed = True
    app.manager.storage.save(task)
    commit_on_branch(app.task_env, task.id)
    _git(
        app.project_root,
        "merge",
        "-q",
        "--no-ff",
        branch_for(task.id),
        "-m",
        "manual merge",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._unmerged_completed_ids() == []
        assert not task.hook_merge_failed
        reloaded = TaskManager(app.project_root).get(task.id)
        assert not reloaded.hook_merge_failed
        await pilot.pause()


# -- sidebar merge-state colors (the ✓ marker of COMPLETED tasks) --------------------


async def test_merge_state_tracks_branch_lifecycle(app_env: DrCodeApp) -> None:
    """_unmerged_completed_ids (the 10 s poll's worker) refreshes
    `_merge_state` from one batched snapshot: no branch → "none", unmerged
    commits → "unmerged", merged branch → "merged", branch deleted after
    the merge → back to "none" (the accepted cleanup limitation)."""
    app = app_env
    task = finished_task(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        # No worktree/branch yet: white.
        assert app._unmerged_completed_ids() == []
        assert app._merge_state == {task.id: "none"}
        commit_on_branch(app.task_env, task.id)
        assert app._unmerged_completed_ids() == [task.id]
        assert app._merge_state == {task.id: "unmerged"}
        _git(
            app.project_root,
            "merge",
            "-q",
            "--no-ff",
            branch_for(task.id),
            "-m",
            "merge it",
        )
        assert app._unmerged_completed_ids() == []
        assert app._merge_state == {task.id: "merged"}
        # Worktree removal / retention cleanup deletes the merged branch
        # (TaskEnvironment.remove: worktree first, then `branch -D`).
        app.task_env.remove(task.id)
        assert app._unmerged_completed_ids() == []
        assert app._merge_state == {task.id: "none"}
        await pilot.pause()


async def test_merge_state_hook_merge_failed_is_unmerged(app_env: DrCodeApp) -> None:
    """A failed hook-merge renders orange: the task is re-admitted to the
    unmerged-work tracking, and its branch holds the unmerged commits."""
    app = app_env
    _workflow_file(app.project_root, "def on_accepted(task):\n    return None\n")
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    task.hook_merge_failed = True
    app.manager.storage.save(task)
    commit_on_branch(app.task_env, task.id)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._unmerged_completed_ids() == [task.id]
        assert app._merge_state == {task.id: "unmerged"}
        await pilot.pause()


async def test_merge_state_research_task_colored_by_branch(
    app_env: DrCodeApp,
) -> None:
    """Research tasks have no verification — the same git criteria apply:
    an unmerged research branch colors its ✓ orange."""
    app = app_env
    task = finished_task(app, kind=TaskKind.RESEARCH)
    commit_on_branch(app.task_env, task.id)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._unmerged_completed_ids() == [task.id]
        assert app._merge_state == {task.id: "unmerged"}
        await pilot.pause()


def test_merge_state_empty_without_isolation(git_root: Path) -> None:
    """Isolation off: no branch ever exists — no colored markers, the
    state dict stays empty (white ✓ for every completed task)."""
    manager = TaskManager(git_root)
    task = manager.create("Fix the bug", "fix the bug")
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    manager.storage.save(task)
    app = DrCodeApp(git_root)
    assert app._unmerged_completed_ids() == []
    assert app._merge_state == {}


async def test_sidebar_row_marker_follows_merge_state(app_env: DrCodeApp) -> None:
    """The rendered ✓ marker: plain (white) without a branch, orange
    while unmerged, green once merged — and the sidebar fingerprint
    changes with the color, so the incremental refresh actually redraws
    a row whose only change is the marker color."""
    app = app_env
    task = finished_task(app)
    commit_on_branch(app.task_env, task.id)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._unmerged_completed_ids()
        assert "[orange]✓[/]" in app._task_row_text(task)
        fp_unmerged = app._sidebar_fingerprint()
        _git(
            app.project_root,
            "merge",
            "-q",
            "--no-ff",
            branch_for(task.id),
            "-m",
            "merge it",
        )
        app._unmerged_completed_ids()
        assert "[green]✓[/]" in app._task_row_text(task)
        assert app._sidebar_fingerprint() != fp_unmerged
        app.task_env.remove(task.id)
        app._unmerged_completed_ids()
        plain = app._task_row_text(task)
        assert "✓" in plain
        assert "[orange]" not in plain and "[green]✓" not in plain
        await pilot.pause()


# -- the wrong-fork guard before the accept-hook merge (git cherry) --------------------


def _wrong_fork_branch(app: DrCodeApp, task: Any) -> None:
    """Give the task's branch the incident shape: a patch-id TWIN of a
    commit already on the base (the branch was forked before the base got
    that change) plus one real commit. The two twin commits get distinct
    timestamps — same-second commits with the same parent/tree collapse
    into one SHA and would hide the shape."""
    import os

    root = app.project_root
    wt, _ = app.task_env.ensure_root(task.id)
    dates = {
        "GIT_AUTHOR_DATE": "2024-01-01T00:00:00 +0000",
        "GIT_COMMITTER_DATE": "2024-01-01T00:00:00 +0000",
    }
    twin_dates = {
        "GIT_AUTHOR_DATE": "2024-01-02T00:00:00 +0000",
        "GIT_COMMITTER_DATE": "2024-01-02T00:00:00 +0000",
    }

    def commit(cwd: Path, message: str, when: dict[str, str]) -> None:
        subprocess.run(
            ["git", "commit", "-q", "-m", message],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, **when},
        )

    # Stage exactly x.txt: the main tree carries untracked-but-not-ignored
    # .drcode/ files (workflow.py, settings.json) that `add -A` would pull
    # into the base commit and break the patch-id match with the twin.
    (root / "x.txt").write_text("x")
    _git(root, "add", "x.txt")
    commit(root, "add x", dates)
    (wt / "x.txt").write_text("x")
    _git(wt, "add", "x.txt")
    commit(wt, "add x (task side)", twin_dates)
    (wt / "y.txt").write_text("y")
    _git(wt, "add", "-A")
    commit(wt, "real task work", twin_dates)


async def test_wrong_fork_branch_blocks_accept_hook(app_env: DrCodeApp) -> None:
    """A task branch carrying patch-id duplicates of merged work (forked
    from a wrong HEAD) is never auto-merged: the on_accepted hook is
    skipped, the situation is explained in the service events and
    `task.hook_merge_failed` parks the branch for manual analysis."""
    app = app_env
    _workflow_file(
        app.project_root, "def on_accepted(task):\n    return {'text': 'merged'}\n"
    )
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    _wrong_fork_branch(app, task)
    toasts = track_toasts(app)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._run_accepted_hook(task)
        await pilot.pause()
    assert task.hook_merge_failed
    reloaded = TaskManager(app.project_root).get(task.id)
    assert reloaded.hook_merge_failed
    events = "\n".join(event_texts(task))
    assert "on_accepted SKIPPED" in events
    assert "already present" in events
    assert "on_accepted:\nmerged" not in events  # the hook never ran
    assert any(title == "Suspicious task branch" for title, _ in toasts)


async def test_wrong_fork_branch_blocks_research_hook(
    app_env: DrCodeApp,
) -> None:
    """Same guard on the research completion path
    (on_research_accepted)."""
    app = app_env
    _workflow_file(
        app.project_root,
        "def on_research_accepted(task):\n    return {'text': 'merged'}\n",
    )
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app, kind=TaskKind.RESEARCH)
    _wrong_fork_branch(app, task)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._run_research_accepted_hook(task)
        await pilot.pause()
    assert task.hook_merge_failed
    events = "\n".join(event_texts(task))
    assert "on_research_accepted SKIPPED" in events
    assert "on_research_accepted:\nmerged" not in events


async def test_normal_branch_still_merges_through_hook(
    app_env: DrCodeApp,
) -> None:
    """The legitimate one-commit-ahead branch passes the git-cherry guard
    and the accept hook runs as before (the dependency-gate scenario is
    untouched: the guard only looks at patch-id duplicates)."""
    app = app_env
    _workflow_file(
        app.project_root, "def on_accepted(task):\n    return {'text': 'merged'}\n"
    )
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    commit_on_branch(app.task_env, task.id)
    assert app.task_env.duplicate_commits(task.id) == (0, 1)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._run_accepted_hook(task)
        await pilot.pause()
    assert not task.hook_merge_failed
    assert "on_accepted:\nmerged" in "\n".join(event_texts(task))
