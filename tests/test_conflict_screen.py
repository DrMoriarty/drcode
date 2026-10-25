"""Tests for the manual conflict-resolution layer: the zdiff3 hunk model
(`drcode.conflict` — parsing, per-hunk ours/theirs/both, whole-file side
taking for the markerless kinds, the engine-backed commit/abort) and the
TUI ConflictScreen flow (Ctrl+M entry, hunk resolution, armed commit,
refusals, the needs_human fallback markers).

Everything runs against real throwaway git repositories — no network,
no models (the conflict layer is deliberately LLM-free)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from drcode import merge_engine as me
from drcode.conflict import (
    ConflictWorktree,
    parse_hunks,
)
from drcode.models import TaskStatus
from drcode.settings import ProjectSettings, ProjectSettingsStore, TaskIsolation
from drcode.task_env import TaskEnvironment, branch_for
from drcode.task_manager import TaskManager
from drcode.tui.app import DrCodeApp
from drcode.tui.merge_screen import ConflictScreen
from tests.network_mocks import wait_until

# -- shared fixtures (mirroring tests/test_merge_engine.py / test_unmerged_reminder.py)


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


def _parked_content_conflict(repo: Path, task_id: str) -> tuple[TaskEnvironment, Path]:
    """A real conflicted merge parked in the task's worktree the way the
    engine leaves it (zdiff3 markers, rerere/identical resolvers on)."""
    env = TaskEnvironment(repo, ProjectSettings(task_isolation=TaskIsolation.WORKTREE))
    wt = repo / ".drcode" / "worktrees" / task_id
    _git(repo, "worktree", "add", str(wt), "-b", branch_for(task_id), "HEAD")
    (wt / "app.py").write_text("a\nTASK\nc\n")
    _git(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("a\nMAIN\nc\n")
    _git(repo, "commit", "-qam", "main edit")
    report = me.merge_base_into_worktree(repo, task_id, wt, "main", save=True)
    assert report.ok and report.conflicted
    return env, wt


# -- the hunk model ---------------------------------------------------------------


def test_parse_hunks_zdiff3_sections() -> None:
    text = (
        "before\n"
        "<<<<<<< ours: app.py\n"
        "TASK\n"
        "||||||| merge base: app.py\n"
        "OLD\n"
        "=======\n"
        "MAIN\n"
        ">>>>>>> main\n"
        "after\n"
    )
    hunks = parse_hunks(text)
    assert len(hunks) == 1
    hunk = hunks[0]
    assert hunk.ours == ["TASK"]
    assert hunk.base == ["OLD"]
    assert hunk.theirs == ["MAIN"]
    assert hunk.head_label == "ours: app.py"
    assert hunk.tail_label == "main"
    assert hunk.start == 1 and hunk.end == 8  # the full marker block


def test_parse_hunks_plain_style_and_unterminated_tail() -> None:
    text = "<<<<<<< ours\nA\n=======\nB\n>>>>>>> theirs\n<<<<<<< dangling"
    hunks = parse_hunks(text)
    assert len(hunks) == 1  # the unterminated tail is garbage, not a hunk
    assert hunks[0].ours == ["A"]
    assert hunks[0].base == []
    assert hunks[0].theirs == ["B"]


def test_parse_hunks_no_markers() -> None:
    assert parse_hunks("just\ncode\n") == []


# -- ConflictWorktree: hunk ops -----------------------------------------------------


def test_apply_hunk_choices_stage_when_marker_free(repo: Path) -> None:
    _env, wt = _parked_content_conflict(repo, "t1")
    w = ConflictWorktree(repo, "t1", wt)
    states = w.file_states()
    assert [s.path for s in states] == ["app.py"]
    state = states[0]
    assert state.unmerged and state.hunks == 1

    content = w.read_file("app.py")
    assert content is not None
    hunk = parse_hunks(content)[0]

    ok, _ = w.apply_hunk("app.py", hunk, "ours")
    assert ok
    assert (wt / "app.py").read_text() == "a\nTASK\nc\n"
    # Marker-free: staged — the index holds no unmerged entry anymore.
    assert _git(wt, "ls-files", "-u").strip() == ""

    # Start over for theirs/both: re-apply on a fresh parse each time.
    for choice, expected in (
        ("theirs", "a\nMAIN\nc\n"),
        ("both", "a\nTASK\nMAIN\nc\n"),
    ):
        me.abort(wt)
        me.merge_base_into_worktree(repo, "t1", wt, "main")
        content = w.read_file("app.py")
        assert content is not None
        ok, _ = w.apply_hunk("app.py", parse_hunks(content)[0], choice)
        assert ok
        assert (wt / "app.py").read_text() == expected


def test_apply_hunk_refuses_stale_hunk(repo: Path) -> None:
    _env, wt = _parked_content_conflict(repo, "t1")
    w = ConflictWorktree(repo, "t1", wt)
    # A hunk parsed from the WRONG file content must not hit wrong lines.
    bogus = parse_hunks("<<<<<<< x\nA\n=======\nB\n>>>>>>> y\n")[0]
    ok, message = w.apply_hunk("app.py", bogus, "ours")
    assert not ok
    assert "changed on disk" in message


def test_take_side_materializes_stages_for_modify_delete(repo: Path) -> None:
    """A modify-delete conflict (no markers): the side is materialized
    from the triage report's stage blobs — or recorded as a deletion."""
    wt = repo / ".drcode" / "worktrees" / "t2"
    _git(repo, "worktree", "add", str(wt), "-b", branch_for("t2"), "HEAD")
    (wt / "app.py").write_text("a\nTASK\nc\n")
    _git(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("a\nMAIN\nc\n")
    _git(repo, "commit", "-qam", "main edits app.py")
    _git(wt, "rm", "app.py")
    _git(wt, "commit", "-qam", "task deletes app.py")
    report = me.merge_base_into_worktree(repo, "t2", wt, "main")
    assert report.ok and report.conflicted
    entry = report.remaining[0]
    assert entry.kind == me.MODIFY_DELETE

    w = ConflictWorktree(repo, "t2", wt, report=report)
    # take theirs (the base side kept the file): materialized from the blob.
    ok, _ = w.take_side("app.py", "theirs")
    assert ok
    assert (wt / "app.py").read_text() == "a\nMAIN\nc\n"
    assert _git(wt, "ls-files", "-u").strip() == ""
    assert all(s.resolved for s in w.file_states())

    # And the delete side: `git rm` records it, the path resolves too.
    me.abort(wt)
    report2 = me.merge_base_into_worktree(repo, "t2", wt, "main")
    assert report2.conflicted
    w.report = report2
    ok, _ = w.take_side("app.py", "ours")
    assert ok
    assert not (wt / "app.py").exists()
    assert all(s.resolved for s in w.file_states())

    with pytest.raises(ValueError):
        w.take_side("app.py", "both")


def test_commit_refuses_then_succeeds(repo: Path) -> None:
    _env, wt = _parked_content_conflict(repo, "t3")
    w = ConflictWorktree(repo, "t3", wt)
    # The engine guard: nothing commits while unmerged paths remain.
    ok, message = w.commit("should not land")
    assert not ok
    assert "remain" in message
    assert me.in_progress(wt)

    content = w.read_file("app.py")
    assert content is not None
    ok, _ = w.apply_hunk("app.py", parse_hunks(content)[0], "theirs")
    assert ok
    ok, _ = w.commit("Merge main into drcode/t3: resolve conflicts (manual)")
    assert ok
    assert not me.in_progress(wt)
    # The resolution landed on the task branch and the merge into the
    # base is a fast-forward — the contract holds.
    assert me.is_fast_forward(repo, branch_for("t3"), "main")
    assert "MAIN" in _git(wt, "show", "HEAD:app.py")


def test_abort_rolls_back(repo: Path) -> None:
    _env, wt = _parked_content_conflict(repo, "t4")
    w = ConflictWorktree(repo, "t4", wt)
    ok, _ = w.abort()
    assert ok
    assert not me.in_progress(wt)
    assert (wt / "app.py").read_text() == "a\nTASK\nc\n"


# -- the TUI screen -----------------------------------------------------------------


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured (but never contacted) model endpoint — suppresses the
    first-start ModelSetupDialog so the app mounts straight into the
    pipeline state under test."""
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


@pytest.fixture
def isolated(repo: Path) -> Path:
    """The git repo with worktree isolation persisted (an app picks it up)."""
    ProjectSettingsStore(repo).save(
        ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    return repo


async def test_screen_resolves_hunk_and_commits(
    isolated: Path, configured_env: None
) -> None:
    """Ctrl+M on the viewed task opens the screen on the parked conflict;
    `o` resolves the hunk (staged, marker-free), and the double-press
    `s` commits the resolution on the task branch — after which the
    merge into the base is a fast-forward."""
    manager = TaskManager(isolated)
    task = manager.create("Fix app", "fix the app")
    # Hold: the pipeline must never develop this task during the test
    # (a fresh task carries its prompt as pending input).
    manager.set_hold(task.id, on=True)
    _env, wt = _parked_content_conflict(isolated, task.id)
    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()

        await pilot.press("ctrl+m")
        await wait_until(
            lambda: isinstance(app.screen, ConflictScreen),
            pilot=pilot,
            what="conflict screen open",
        )
        screen = app.screen
        assert isinstance(screen, ConflictScreen)
        await wait_until(
            lambda: screen.wt is not None
            and bool(screen.files_list.children)
            and screen.view.current_hunk() is not None,
            pilot=pilot,
            what="conflict screen prepared",
        )
        assert screen.view.current_hunk() is not None

        await pilot.press("o")
        await wait_until(
            lambda: _git(wt, "ls-files", "-u").strip() == "",
            pilot=pilot,
            what="hunk resolved and staged",
        )
        assert (wt / "app.py").read_text() == "a\nTASK\nc\n"

        # Armed commit: the first press only arms, the second commits.
        await pilot.press("s")
        await pilot.pause()
        assert me.in_progress(wt)
        await pilot.press("s")
        await wait_until(
            lambda: not me.in_progress(wt),
            pilot=pilot,
            what="resolution committed",
        )
        assert "Merge branch" in _git(wt, "log", "--oneline", "-1") or "Merge main" in (
            _git(wt, "log", "--oneline", "-1")
        )
        assert me.is_fast_forward(isolated, branch_for(task.id), "main")
        reloaded = TaskManager(isolated).get(task.id)
        assert reloaded.needs_human is False


async def test_commit_refused_while_conflicts_remain(
    isolated: Path, configured_env: None
) -> None:
    """The engine guard holds through the UI: with an untouched hunk the
    armed commit is refused and the merge stays parked."""
    manager = TaskManager(isolated)
    task = manager.create("Fix app", "fix the app")
    manager.set_hold(task.id, on=True)
    _env, wt = _parked_content_conflict(isolated, task.id)
    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.press("ctrl+m")
        await wait_until(
            lambda: isinstance(app.screen, ConflictScreen),
            pilot=pilot,
            what="conflict screen open",
        )
        screen = app.screen
        assert isinstance(screen, ConflictScreen)
        await wait_until(
            lambda: screen.wt is not None and bool(screen.files_list.children),
            pilot=pilot,
            what="conflict screen prepared",
        )
        await pilot.press("s")
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()
        assert me.in_progress(wt)
        # The file is untouched — nothing was committed behind the
        # user's back.
        assert "<<<<<<<" in (wt / "app.py").read_text()


async def test_needs_human_task_opens_screen_without_parked_merge(
    isolated: Path, configured_env: None
) -> None:
    """The fallback flag is an entry of its own: a `needs_human` task
    whose merge is not parked (a crash after the resolver gave up)
    still opens the screen — the prepare worker deploys the conflict."""
    manager = TaskManager(isolated)
    task = manager.create("Fix app", "fix the app")
    manager.set_hold(task.id, on=True)
    env = TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    wt, _note = env.ensure_root(task.id)
    (wt / "app.py").write_text("a\nTASK\nc\n")
    _git(wt, "commit", "-qam", "task edit")
    (isolated / "app.py").write_text("a\nMAIN\nc\n")
    _git(isolated, "commit", "-qam", "main edit")
    task.needs_human = True
    task.touch()
    manager.storage.save(task)

    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await pilot.pause()
        # The sidebar carries the persistent ✋ fallback marker.
        assert "✋" in app._task_row_text(manager.get(task.id))
        app.view_task(task.id)
        await pilot.press("ctrl+m")
        await wait_until(
            lambda: isinstance(app.screen, ConflictScreen),
            pilot=pilot,
            what="conflict screen open",
        )
        screen = app.screen
        assert isinstance(screen, ConflictScreen)
        await wait_until(
            lambda: screen.wt is not None and me.in_progress(screen.wt.worktree),
            pilot=pilot,
            what="conflict deployed by prepare",
        )


async def test_ctrl_m_without_conflicts_is_a_noop_toast(
    isolated: Path, configured_env: None
) -> None:
    """A healthy task: Ctrl+M does not open the screen (nothing to
    resolve) — the app stays on the main screen."""
    manager = TaskManager(isolated)
    task = manager.create("Nothing to merge", "plain task")
    manager.set_hold(task.id, on=True)
    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+m")
        await pilot.pause()
        assert not isinstance(app.screen, ConflictScreen)


async def test_screen_close(isolated: Path, configured_env: None) -> None:
    """ESC closes the screen; the parked merge stays exactly as it was."""
    manager = TaskManager(isolated)
    task = manager.create("Fix app", "fix the app")
    manager.set_hold(task.id, on=True)
    _env, wt = _parked_content_conflict(isolated, task.id)
    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.press("ctrl+m")
        await wait_until(
            lambda: isinstance(app.screen, ConflictScreen),
            pilot=pilot,
            what="conflict screen open",
        )
        screen = app.screen
        assert isinstance(screen, ConflictScreen)
        await wait_until(
            lambda: screen.wt is not None, pilot=pilot, what="screen prepared"
        )
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ConflictScreen)
        assert me.in_progress(wt)
        assert task.status is TaskStatus.HOLD  # the screen touches no status
