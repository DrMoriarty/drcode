"""Tests for the unmerged-branches list screen (``tui.unmerged_screen``)
and the user-triggerable resolver launch spread across the merge UI:

- Ctrl+M with MORE than one ready candidate opens the list (exactly one
  still opens the ConflictScreen directly);
- the rows carry the per-task merge state (conflict / ff-ready);
- the list's actions: ``enter`` opens the per-task ConflictScreen, ``m``
  lands an ff-able branch (prepare + ``--ff-only`` merge into the base)
  and refuses to commit behind a parked conflict, ``a`` starts the
  automatic resolver through the app's guarded `launch_resolver`;
- the ConflictScreen's ``A`` key launches the same resolver round and
  closes the screen on success (inert in ff-ready, refused while a round
  is already running).

Everything runs against real throwaway git repositories; the resolver
round itself is stubbed at `run_resolver` — the guard semantics are
LLM-free, the round machinery has its own tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from drcode import merge_engine as me
from drcode.models import TaskStatus, VerificationStatus
from drcode.settings import ProjectSettings, ProjectSettingsStore, TaskIsolation
from drcode.task_env import TaskEnvironment, branch_for
from drcode.task_manager import TaskManager
from drcode.tui.app import DrCodeApp
from drcode.tui.merge_screen import ConflictScreen
from drcode.tui.unmerged_screen import UnmergedListScreen
from tests.network_mocks import wait_until


def _git(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if check:
        assert proc.returncode == 0, f"git {args}: {proc.stderr}"
    return proc.stdout


@pytest.fixture
def repo(project_root: Path) -> Path:
    """A git repository on `main` with one commit (the merge base)."""
    _git(project_root, "init", "-q", "-b", "main")
    _git(project_root, "config", "user.email", "test@example.com")
    _git(project_root, "config", "user.name", "Test")
    (project_root / "app.py").write_text("a\nb\nc\n")
    # An AGENTS.md on disk suppresses the startup offer dialog (it would
    # sit on top of the screen stack and eat the key presses).
    (project_root / "AGENTS.md").write_text("# AGENTS\n")
    _git(project_root, "add", "-A")
    _git(project_root, "commit", "-qm", "base")
    return project_root


@pytest.fixture
def isolated(repo: Path) -> Path:
    """The git repo with worktree isolation persisted (an app picks it up)."""
    ProjectSettingsStore(repo).save(
        ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    return repo


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured (but never contacted) model endpoint — suppresses the
    first-start ModelSetupDialog so the app mounts straight into the
    pipeline state under test."""
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


def _completed_task(manager: TaskManager, title: str) -> str:
    """A COMPLETED PASSED task (the unmerged-work tracking's candidate)."""
    task = manager.create(title, "do it")
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    task.touch()
    manager.storage.save(task)
    return task.id


def _unmerged_branch(repo: Path, task_id: str, marker: str) -> Path:
    """A `drcode/<id>` branch holding one commit beyond main (via the
    task's worktree — the way a real task leaves it)."""
    env = TaskEnvironment(repo, ProjectSettings(task_isolation=TaskIsolation.WORKTREE))
    wt, _note = env.ensure_root(task_id)
    (wt / "app.py").write_text(f"a\n{marker}\nc\n")
    _git(wt, "commit", "-qam", f"{task_id} edit")
    return wt


def _ancestor(repo: Path, branch: str, base: str) -> bool:
    """Whether `branch`'s tip is an ancestor of `base` (the branch is
    fully merged)."""
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", branch, base],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _parked_conflict(repo: Path, task_id: str) -> Path:
    """A real conflicted merge parked in the task's worktree the way the
    engine leaves it (zdiff3 markers, task commit beyond main)."""
    wt = repo / ".drcode" / "worktrees" / task_id
    _git(repo, "worktree", "add", str(wt), "-b", branch_for(task_id), "HEAD")
    (wt / "app.py").write_text("a\nTASK\nc\n")
    _git(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("a\nMAIN\nc\n")
    _git(repo, "commit", "-qam", "main edit")
    report = me.merge_base_into_worktree(repo, task_id, wt, "main", save=True)
    assert report.ok and report.conflicted
    return wt


# -- the Ctrl+M entry ----------------------------------------------------------


async def test_ctrl_m_with_two_candidates_opens_list(
    isolated: Path,
    configured_env: None,
) -> None:
    """More than one ready candidate: the list screen instead of the
    per-task ConflictScreen — one row per task with its branch and state."""
    manager = TaskManager(isolated)
    t1 = _completed_task(manager, "Fix one")
    t2 = _completed_task(manager, "Fix two")
    _unmerged_branch(isolated, t1, "ONE")
    _unmerged_branch(isolated, t2, "TWO")
    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(t1)
        await pilot.pause()
        await pilot.press("ctrl+m")
        await wait_until(
            lambda: isinstance(app.screen, UnmergedListScreen),
            pilot=pilot,
            what="unmerged list open",
        )
        screen = app.screen
        assert isinstance(screen, UnmergedListScreen)
        await wait_until(
            lambda: len(screen._entries) == 2,
            pilot=pilot,
            what="both rows computed",
        )
        assert sorted(e.task_id for e in screen._entries) == sorted([t1, t2])
        # Plain unmerged branches with nothing parked are ff-ready.
        assert all(e.kind == "ff" for e in screen._entries)


async def test_ctrl_m_with_one_candidate_opens_conflict_screen(
    isolated: Path,
    configured_env: None,
) -> None:
    """Exactly one candidate: straight into the ConflictScreen — the list
    would be a pointless extra click (the historic behavior)."""
    manager = TaskManager(isolated)
    task_id = _completed_task(manager, "Fix one")
    _unmerged_branch(isolated, task_id, "ONE")
    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task_id)
        await pilot.press("ctrl+m")
        await wait_until(
            lambda: isinstance(app.screen, ConflictScreen),
            pilot=pilot,
            what="conflict screen open",
        )
        assert not isinstance(app.screen, UnmergedListScreen)


# -- the list's actions ----------------------------------------------------------


async def test_enter_opens_conflict_screen_for_selected_task(
    isolated: Path,
    configured_env: None,
) -> None:
    """enter on the SECOND row opens the ConflictScreen for THAT task —
    the user picks a concrete task, not just the first one."""
    manager = TaskManager(isolated)
    t1 = _completed_task(manager, "Fix one")
    t2 = _completed_task(manager, "Fix two")
    _unmerged_branch(isolated, t1, "ONE")
    _unmerged_branch(isolated, t2, "TWO")
    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(t1)
        await pilot.press("ctrl+m")
        await wait_until(
            lambda: (
                isinstance(app.screen, UnmergedListScreen)
                and len(app.screen._entries) == 2
            ),
            pilot=pilot,
            what="unmerged list open",
        )
        screen = app.screen
        assert isinstance(screen, UnmergedListScreen)
        # Move the selection to the second row, then open its screen.
        await pilot.press("down")
        await pilot.pause()
        assert screen._selected() is not None
        selected = screen._selected()
        assert selected is not None and selected.task_id == t2
        await pilot.press("enter")
        await wait_until(
            lambda: isinstance(app.screen, ConflictScreen),
            pilot=pilot,
            what="conflict screen open",
        )
        opened = app.screen
        assert isinstance(opened, ConflictScreen)
        assert opened.task_id == t2
        # ESC returns to the list, which refreshes itself.
        await pilot.press("escape")
        await wait_until(
            lambda: isinstance(app.screen, UnmergedListScreen),
            pilot=pilot,
            what="back on the list",
        )


async def test_land_ff_merges_the_selected_branch(
    isolated: Path,
    configured_env: None,
) -> None:
    """m on an ff-able row: prepare runs the clean reversed merge, then
    the `--ff-only` merge lands the branch in the main checkout — the
    row drops on the refresh, the OTHER unmerged branch is untouched."""
    manager = TaskManager(isolated)
    t1 = _completed_task(manager, "Fix one")
    t2 = _completed_task(manager, "Fix two")
    _unmerged_branch(isolated, t1, "ONE")
    _unmerged_branch(isolated, t2, "TWO")
    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(t1)
        await pilot.press("ctrl+m")
        await wait_until(
            lambda: (
                isinstance(app.screen, UnmergedListScreen)
                and len(app.screen._entries) == 2
            ),
            pilot=pilot,
            what="unmerged list open",
        )
        screen = app.screen
        assert isinstance(screen, UnmergedListScreen)
        await pilot.press("m")
        await wait_until(
            lambda: _ancestor(isolated, branch_for(t1), "main"),
            pilot=pilot,
            what="branch ff-merged into main",
        )
        # The second branch stays unmerged and the list refreshes to it.
        assert not _ancestor(isolated, branch_for(t2), "main")
        await wait_until(
            lambda: (
                isinstance(app.screen, UnmergedListScreen)
                and len(app.screen._entries) == 1
                and app.screen._entries[0].task_id == t2
            ),
            pilot=pilot,
            what="list refreshed to the remaining branch",
        )


async def test_land_refuses_to_commit_a_parked_conflict(
    isolated: Path,
    configured_env: None,
) -> None:
    """m on a conflicted row: nothing is committed behind the user's
    back — the conflict stays parked and the merge is untouched (the
    toast points at enter/a)."""
    manager = TaskManager(isolated)
    t1 = _completed_task(manager, "Fix one")
    t2 = _completed_task(manager, "Fix two")
    wt1 = _parked_conflict(isolated, t1)
    _unmerged_branch(isolated, t2, "TWO")
    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(t1)
        await pilot.press("ctrl+m")
        await wait_until(
            lambda: (
                isinstance(app.screen, UnmergedListScreen)
                and len(app.screen._entries) == 2
            ),
            pilot=pilot,
            what="unmerged list open",
        )
        screen = app.screen
        assert isinstance(screen, UnmergedListScreen)
        await wait_until(
            lambda: any(
                e.task_id == t1 and e.kind == "conflict" for e in screen._entries
            ),
            pilot=pilot,
            what="conflict row computed",
        )
        await pilot.press("m")
        await pilot.pause()
        await pilot.pause()
        assert me.in_progress(wt1)
        assert "<<<<<<<" in (wt1 / "app.py").read_text()
        # The screen stays open — the user picks enter or a instead.
        assert isinstance(app.screen, UnmergedListScreen)


async def test_auto_resolve_starts_round_and_closes_list(
    isolated: Path,
    configured_env: None,
) -> None:
    """a on the selected row goes through the app's guarded launcher:
    the recorded round starts for THAT task and the list closes (the
    round's progress lives in the task's chat)."""
    manager = TaskManager(isolated)
    t1 = _completed_task(manager, "Fix one")
    t2 = _completed_task(manager, "Fix two")
    _parked_conflict(isolated, t1)
    _unmerged_branch(isolated, t2, "TWO")
    app = DrCodeApp(isolated)
    started: list[str] = []
    app._resolver_start_refusal = lambda _tid: None  # type: ignore[method-assign]
    app.run_resolver = started.append  # type: ignore[method-assign]
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(t1)
        await pilot.press("ctrl+m")
        await wait_until(
            lambda: (
                isinstance(app.screen, UnmergedListScreen)
                and len(app.screen._entries) == 2
            ),
            pilot=pilot,
            what="unmerged list open",
        )
        await pilot.press("a")
        await wait_until(
            lambda: started == [t1],
            pilot=pilot,
            what="resolver round started for the selected task",
        )
        assert not isinstance(app.screen, UnmergedListScreen)  # dismissed


# -- the ConflictScreen's A key ----------------------------------------------------


async def test_conflict_screen_auto_resolve_key_starts_round(
    isolated: Path,
    configured_env: None,
) -> None:
    """A on a parked conflict: the round starts through `launch_resolver`
    and the screen closes itself — after a successful round Ctrl+M
    reopens the branch ff-ready."""
    manager = TaskManager(isolated)
    task_id = _completed_task(manager, "Fix one")
    _parked_conflict(isolated, task_id)
    app = DrCodeApp(isolated)
    started: list[str] = []
    app._resolver_start_refusal = lambda _tid: None  # type: ignore[method-assign]
    app.run_resolver = started.append  # type: ignore[method-assign]
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task_id)
        await pilot.press("ctrl+m")
        await wait_until(
            lambda: isinstance(app.screen, ConflictScreen),
            pilot=pilot,
            what="conflict screen open",
        )
        screen = app.screen
        assert isinstance(screen, ConflictScreen)
        await wait_until(
            lambda: screen.wt is not None and screen._conflicted,
            pilot=pilot,
            what="parked conflict prepared",
        )
        await pilot.press("A")
        await wait_until(
            lambda: started == [task_id],
            pilot=pilot,
            what="resolver round started",
        )
        assert not isinstance(app.screen, ConflictScreen)  # dismissed


async def test_conflict_screen_auto_resolve_refused_while_round_running(
    isolated: Path,
    configured_env: None,
) -> None:
    """The guard holds through the UI: with a resolver round already
    running, A keeps the screen open and starts nothing (the refusal is
    a toast, never a second prepare/verify on the same worktree)."""
    manager = TaskManager(isolated)
    task_id = _completed_task(manager, "Fix one")
    _parked_conflict(isolated, task_id)
    app = DrCodeApp(isolated)
    started: list[str] = []
    app.run_resolver = started.append  # type: ignore[method-assign]
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task_id)
        await pilot.press("ctrl+m")
        await wait_until(
            lambda: (
                isinstance(app.screen, ConflictScreen) and app.screen.wt is not None
            ),
            pilot=pilot,
            what="conflict screen prepared",
        )
        screen = app.screen
        assert isinstance(screen, ConflictScreen)
        await wait_until(
            lambda: screen._conflicted,
            pilot=pilot,
            what="parked conflict prepared",
        )
        app._resolver_running = True
        await pilot.press("A")
        await pilot.pause()
        await pilot.pause()
        assert started == []  # nothing started
        assert isinstance(app.screen, ConflictScreen)  # the screen stays


async def test_conflict_screen_auto_resolve_disabled_when_ff_ready(
    isolated: Path,
    configured_env: None,
) -> None:
    """ff-ready means nothing to resolve: check_action disables A (the
    footer names it among the unavailable keys) and the action stays
    inert even when a round could be started."""
    manager = TaskManager(isolated)
    task_id = _completed_task(manager, "Fix one")
    # A clean unmerged branch: prepare concludes with nothing to resolve.
    _unmerged_branch(isolated, task_id, "ONE")
    app = DrCodeApp(isolated)
    started: list[str] = []
    app._resolver_start_refusal = lambda _tid: None  # type: ignore[method-assign]
    app.run_resolver = started.append  # type: ignore[method-assign]
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task_id)
        await pilot.press("ctrl+m")
        await wait_until(
            lambda: (
                isinstance(app.screen, ConflictScreen) and app.screen.wt is not None
            ),
            pilot=pilot,
            what="conflict screen prepared",
        )
        screen = app.screen
        assert isinstance(screen, ConflictScreen)
        await wait_until(
            lambda: screen._ff_ready,
            pilot=pilot,
            what="ff-ready state reached",
        )
        assert screen.check_action("auto_resolve", ()) is False
        await pilot.press("A")
        await pilot.pause()
        await pilot.pause()
        assert started == []


# -- the guarded launcher ------------------------------------------------------------


async def test_launch_resolver_refuses_without_a_model(isolated: Path) -> None:
    """No resolver model configured (and no env fallback in this
    sandbox): the launcher refuses with False — the human path stays
    the only one, exactly like the silent automatic marker."""
    manager = TaskManager(isolated)
    task_id = _completed_task(manager, "Fix one")
    app = DrCodeApp(isolated)
    started: list[str] = []
    app.run_resolver = started.append  # type: ignore[method-assign]
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.launch_resolver(task_id) is False
        assert started == []
