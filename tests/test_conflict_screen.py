"""Tests for the manual conflict-resolution layer: the zdiff3 hunk model
(`drcode.conflict` — parsing, per-hunk ours/theirs/both, whole-file side
taking for the markerless kinds, the engine-backed commit/abort) and the
TUI ConflictScreen flow (Ctrl+M entry, hunk resolution, armed commit,
refusals, the needs_human fallback markers).

Everything runs against real throwaway git repositories — no network,
no models (the conflict layer is deliberately LLM-free)."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest
from textual.widgets import Static

from drcode import merge_engine as me
from drcode import merge_resolver
from drcode.conflict import (
    ConflictWorktree,
    parse_hunks,
)
from drcode.merge_resolver import CheckResult
from drcode.models import TaskStatus, VerificationStatus
from drcode.settings import ProjectSettings, ProjectSettingsStore, TaskIsolation
from drcode.task_env import TaskEnvironment, branch_for
from drcode.task_manager import TaskManager
from drcode.tui.app import DrCodeApp
from drcode.tui.merge_screen import CheckDetailsScreen, ConflictScreen
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


# -- TaskEnvironment.ff_merge_branch (the user-triggered ff-merge) -------------------


def _branch_with_work(repo: Path, task_id: str) -> tuple[TaskEnvironment, Path]:
    """A task worktree on `drcode/<id>` holding one commit the main
    checkout lacks (no conflict — the branch is simply ahead)."""
    env = TaskEnvironment(repo, ProjectSettings(task_isolation=TaskIsolation.WORKTREE))
    wt, _note = env.ensure_root(task_id)
    (wt / "fix.txt").write_text("fixed\n")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-qm", f"fix for {task_id}")
    return env, wt


def test_ff_merge_branch_fast_forwards_main_checkout(repo: Path) -> None:
    env, _wt = _branch_with_work(repo, "t5")
    assert env.unmerged_commits("t5") == 1
    ok, _out = env.ff_merge_branch("t5")
    assert ok
    assert (repo / "fix.txt").read_text() == "fixed\n"
    assert env.unmerged_commits("t5") == 0


def test_ff_merge_branch_refuses_when_base_moved(repo: Path) -> None:
    env, _wt = _branch_with_work(repo, "t5")
    # The base moves ahead while the task branch sits unmerged: a ff is
    # no longer possible — the refusal must leave the checkout untouched
    # (no merge commit, no conflict to resolve in the user's tree).
    (repo / "app.py").write_text("a\nMAIN NEWER\nc\n")
    _git(repo, "commit", "-qam", "main moves on")
    main_head = _git(repo, "rev-parse", "HEAD").strip()
    ok, out = env.ff_merge_branch("t5")
    assert not ok
    assert out  # git names the refusal for the error toast
    assert not (repo / "fix.txt").exists()
    assert _git(repo, "rev-parse", "HEAD").strip() == main_head
    assert env.unmerged_commits("t5") == 1


def test_ff_merge_branch_without_branch_or_isolation(repo: Path) -> None:
    env = TaskEnvironment(repo, ProjectSettings(task_isolation=TaskIsolation.WORKTREE))
    ok, out = env.ff_merge_branch("ghost")
    assert not ok
    assert "does not exist" in out
    off = TaskEnvironment(repo, ProjectSettings(task_isolation=TaskIsolation.OFF))
    ok, out = off.ff_merge_branch("t5")
    assert not ok
    assert "isolation" in out


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
            lambda: (
                screen.wt is not None
                and bool(screen.files_list.children)
                and screen.view.current_hunk() is not None
            ),
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


# -- the verification report (merge_resolver helpers) --------------------------------


def _canned_results() -> tuple[list[CheckResult], CheckResult]:
    """One funnel run: compile ok, ruff FAILED, pyright skipped (with
    its explanatory output), the test suite FAILED."""
    checks = [
        CheckResult(
            name="compile (py_compile of tracked .py files)", ok=True, output=""
        ),
        CheckResult(name="ruff check", ok=False, output="app.py:1:1: F401 unused"),
        CheckResult(name="pyright", ok=True, skipped=True, output="no node / npx"),
    ]
    tests = CheckResult(
        name="test suite (pytest -q -x)",
        ok=False,
        output="FAILED tests/test_app.py::test_x - assert False",
    )
    return checks, tests


def test_failed_names_lists_real_failures_only() -> None:
    checks, tests = _canned_results()
    assert merge_resolver.failed_names([*checks, tests]) == [
        "ruff check",
        "test suite (pytest -q -x)",
    ]


def test_format_verify_report_carries_full_output() -> None:
    checks, tests = _canned_results()
    text = merge_resolver.format_verify_report(checks, tests)
    assert "Conflict resolution verification: FAILED" in text
    assert "failed checks: ruff check, test suite (pytest -q -x)" in text
    assert "app.py:1:1: F401 unused" in text  # the full ruff output
    assert "assert False" in text  # the full pytest output
    assert "no node / npx" in text  # skipped checks keep their explanation
    # A clean (or fully skipped) run reads PASSED and names nothing.
    ok_text = merge_resolver.format_verify_report(
        [CheckResult(name="ruff check", ok=True, skipped=True, output="ruff missing")],
        CheckResult(name="test suite (pytest -q -x)", ok=True),
    )
    assert "PASSED" in ok_text
    assert "failed checks" not in ok_text


def test_save_verify_report_overwrites_per_run(tmp_path: Path) -> None:
    checks, tests = _canned_results()
    path = merge_resolver.save_verify_report(tmp_path, "t1", checks, tests)
    assert path == tmp_path / ".drcode" / "merge_reports" / "t1-verify.txt"
    assert path is not None
    assert "F401" in path.read_text()
    # The next `v` run rewrites the report in place — with a clean run.
    merge_resolver.save_verify_report(
        tmp_path,
        "t1",
        [
            CheckResult(
                name="compile (py_compile of tracked .py files)", ok=True, output=""
            ),
            CheckResult(name="ruff check", ok=True),
        ],
        CheckResult(name="test suite (pytest -q -x)", ok=True),
    )
    rewritten = path.read_text()
    assert "assert False" not in rewritten
    assert "PASSED" in rewritten


def test_check_timings_persist_and_roll(tmp_path: Path) -> None:
    """The per-check rolling averages persist next to the verify report:
    the first save plants them, the next merges at 0.7 old / 0.3 new, a
    corrupted file is skipped (empty history, never fatal)."""
    assert merge_resolver.load_check_timings(tmp_path, "t1") == {}
    merge_resolver.save_check_timings(tmp_path, "t1", {"pytest": 100.0, "ruff": 2.0})
    assert merge_resolver.load_check_timings(tmp_path, "t1") == {
        "pytest": 100.0,
        "ruff": 2.0,
    }
    merge_resolver.save_check_timings(tmp_path, "t1", {"pytest": 200.0})
    timings = merge_resolver.load_check_timings(tmp_path, "t1")
    assert timings["pytest"] == pytest.approx(0.7 * 100.0 + 0.3 * 200.0)
    assert timings["ruff"] == 2.0
    path = merge_resolver.check_timings_path(tmp_path, "t1")
    assert path.name == "t1-verify-timings.json"
    path.write_text("{not json", encoding="utf-8")
    assert merge_resolver.load_check_timings(tmp_path, "t1") == {}


# -- the screen: failure toast, report file, details viewer ---------------------------


async def test_verify_failure_names_checks_writes_report_and_opens_details(
    isolated: Path, configured_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`v` with failing checks: the toast names them, the full funnel
    report is persisted next to the triage report, and `d` opens the
    details viewer on the first FAILED check — esc returns to the
    resolution state untouched."""
    manager = TaskManager(isolated)
    task = manager.create("Fix app", "fix the app")
    manager.set_hold(task.id, on=True)
    _env, wt = _parked_content_conflict(isolated, task.id)
    checks, tests = _canned_results()
    compile_c, ruff_c, pyright_c = checks
    monkeypatch.setattr(merge_resolver, "compile_check", lambda _wt: compile_c)
    monkeypatch.setattr(merge_resolver, "ruff_check", lambda _wt: ruff_c)
    monkeypatch.setattr(merge_resolver, "pyright_check", lambda _wt: pyright_c)
    monkeypatch.setattr(merge_resolver, "run_test_suite", lambda _wt: tests)
    app = DrCodeApp(isolated)

    def toasts() -> list[str]:
        return [n.message for n in app._notifications._notifications.values()]

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

        await pilot.press("v")
        report = merge_resolver.verify_report_path(isolated, task.id)
        await wait_until(
            lambda: report.exists(), pilot=pilot, what="verify report written"
        )
        assert "F401" in report.read_text()
        assert "assert False" in report.read_text()
        # The toast names the failing checks by name (plus the report path).
        await wait_until(
            lambda: any(
                "Checks FAILED: ruff check, test suite (pytest -q -x)" in message
                and f"{task.id}-verify.txt" in message
                for message in toasts()
            ),
            pilot=pilot,
            what="named-failure toast",
        )
        # The panel stays a compact status list, but mentions the report.
        panel_text = str(screen.query_one("#merge-checks", Static).content)
        assert "ruff check: FAILED" in panel_text
        assert f"Full report: {report}" in panel_text

        # d opens the details viewer, defaulting to the first FAILED check.
        await pilot.press("d")
        await wait_until(
            lambda: isinstance(app.screen, CheckDetailsScreen),
            pilot=pilot,
            what="details viewer open",
        )
        details = app.screen
        assert isinstance(details, CheckDetailsScreen)
        assert details._checks[details._index].name == "ruff check"
        # [ / ] step through ALL checks, skipped included.
        await pilot.press("]")
        assert details._checks[details._index].name == "pyright"
        await pilot.press("]")
        assert details._checks[details._index].name == "test suite (pytest -q -x)"
        await pilot.press("]")
        assert details._checks[details._index].name.startswith("compile")
        # esc returns to the conflict screen — resolution state intact.
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen
        assert me.in_progress(wt)
        assert "<<<<<<<" in (wt / "app.py").read_text()


async def test_verify_shows_live_progress_and_persists_timings(
    isolated: Path, configured_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While the funnel runs the #merge-checks panel shows the four
    stages live — a spinner line for the running check, pending markers
    for the rest, a percent/ETA footer — and after the run the measured
    durations are persisted as the rolling-average history (skipped
    checks excluded)."""
    manager = TaskManager(isolated)
    task = manager.create("Fix app", "fix the app")
    manager.set_hold(task.id, on=True)
    _env, _wt = _parked_content_conflict(isolated, task.id)
    checks, tests = _canned_results()
    compile_c, ruff_c, pyright_c = checks
    started = threading.Event()
    release = threading.Event()

    def slow_compile(_wt: Path) -> CheckResult:
        started.set()
        release.wait(10)
        return compile_c

    monkeypatch.setattr(merge_resolver, "compile_check", slow_compile)
    monkeypatch.setattr(merge_resolver, "ruff_check", lambda _wt: ruff_c)
    monkeypatch.setattr(merge_resolver, "pyright_check", lambda _wt: pyright_c)
    monkeypatch.setattr(merge_resolver, "run_test_suite", lambda _wt: tests)
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
            lambda: screen.wt is not None,
            pilot=pilot,
            what="screen prepared",
        )

        await pilot.press("v")
        # Mid-run: compile is blocked inside its thread, everything else
        # pending — the panel animates independently of the worker.
        await wait_until(
            lambda: started.is_set(), pilot=pilot, what="compile check running"
        )
        await pilot.pause()
        panel_text = str(screen.query_one("#merge-checks", Static).content)
        assert "Verification funnel:" in panel_text
        assert "] compile…" in panel_text  # the spinner line (any frame)
        assert "[ ] ruff check" in panel_text
        assert "[ ] pyright" in panel_text
        assert "[ ] test suite" in panel_text
        assert "ETA ~" in panel_text
        assert "%" in panel_text
        assert screen._checks_timer is not None  # the animation timer runs

        release.set()
        report = merge_resolver.verify_report_path(isolated, task.id)
        await wait_until(
            lambda: report.exists(), pilot=pilot, what="verify report written"
        )
        await wait_until(
            lambda: screen._checks_timer is None, pilot=pilot, what="timer stopped"
        )
        # The final panel is the static status list again…
        final_text = str(screen.query_one("#merge-checks", Static).content)
        assert "ruff check: FAILED" in final_text
        # …and the measured durations (skipped pyright excluded) landed
        # in the persisted rolling averages.
        timings = merge_resolver.load_check_timings(isolated, task.id)
        assert set(timings) == {"compile", "ruff", "pytest"}
        assert all(value > 0 for value in timings.values())


async def test_details_key_without_checks_and_report_key_without_file(
    isolated: Path, configured_env: None
) -> None:
    """`d` before any `v` run and `E` without a persisted report are
    polite no-op toasts — never crashes, never a half-open viewer."""
    manager = TaskManager(isolated)
    task = manager.create("Fix app", "fix the app")
    manager.set_hold(task.id, on=True)
    _env, _wt = _parked_content_conflict(isolated, task.id)
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
        await pilot.press("d")
        await pilot.pause()
        assert isinstance(app.screen, ConflictScreen)
        assert any(
            "No checks yet" in m
            for m in (n.message for n in app._notifications._notifications.values())
        )
        await pilot.press("E")
        await pilot.pause()
        assert isinstance(app.screen, ConflictScreen)
        assert any(
            "No report yet" in m
            for m in (n.message for n in app._notifications._notifications.values())
        )
        assert not isinstance(app.screen, CheckDetailsScreen)


# -- the ff-ready mode (nothing to resolve; f merges into the base) -------------------


def _finished_task_with_branch(repo: Path) -> tuple[str, Path]:
    """A completed PASSED task whose `drcode/<id>` branch holds one clean
    commit the main checkout lacks — the ff-ready screen scenario."""
    manager = TaskManager(repo)
    # TaskManager scaffolds an untracked .gitignore — track it BEFORE the
    # task branch diverges, or the ff-merge below refuses over it (in a
    # real repo the file is already committed).
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "gitignore")
    task = manager.create("Fix app", "fix the app")
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    task.touch()
    manager.storage.save(task)
    env = TaskEnvironment(repo, ProjectSettings(task_isolation=TaskIsolation.WORKTREE))
    wt, _note = env.ensure_root(task.id)
    (wt / "fix.txt").write_text("fixed\n")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-qm", "fix app")
    return task.id, wt


def _toasts(app: DrCodeApp) -> list[str]:
    return [n.message for n in app._notifications._notifications.values()]


async def test_ff_ready_screen_merges_into_base_and_clears_counter(
    isolated: Path, configured_env: None
) -> None:
    """The bug-example scenario: an unmerged task branch with nothing to
    resolve opens the screen in ff-ready mode; `f` fast-forwards the
    MAIN checkout to the task branch, the mergedness snapshot flips to
    "merged" (the header's ⎇ counter drains) and a further `f` is a
    no-op (check_action gates it off)."""
    task_id, _wt = _finished_task_with_branch(isolated)
    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task_id)
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
            lambda: screen.wt is not None and screen._ff_ready,
            pilot=pilot,
            what="ff-ready state reached",
        )
        # Nothing to resolve: the resolution keys are gated off, the f
        # key is live, the footer names both facts.
        assert screen.check_action("resolve", ("ours",)) is False
        assert screen.check_action("commit", ()) is False
        assert screen.check_action("abort_merge", ()) is False
        assert screen.check_action("ff_merge", ()) is True
        footer = str(screen.query_one("#merge-footer", Static).content)
        assert "f merge into main (ff-only)" in footer
        assert "disabled — nothing to resolve" in footer

        await pilot.press("f")
        await wait_until(
            lambda: screen._ff_done,
            pilot=pilot,
            what="main checkout fast-forwarded",
        )
        assert (isolated / "fix.txt").exists()
        assert screen.check_action("ff_merge", ()) is False
        # A successful merge is the logical end of work: the screen
        # closes itself (the outcome toast outlives it).
        await wait_until(
            lambda: not isinstance(app.screen, ConflictScreen),
            pilot=pilot,
            what="merge screen auto-closed",
        )
        # Mergedness snapshot + header counter refreshed by the same pass.
        await wait_until(
            lambda: app._merge_state.get(task_id) == "merged",
            pilot=pilot,
            what="mergedness snapshot refreshed",
        )
        assert app._unmerged_completed_ids() == []
        # The service event records the merge.
        assert any(
            "ff-only" in event.text
            for event in TaskManager(isolated).get(task_id).events
        )


async def test_ff_merge_refused_when_base_moved(
    isolated: Path, configured_env: None
) -> None:
    """`f` when the base moved ahead: --ff-only refuses, the error toast
    names it, the main checkout stays untouched and the screen stays in
    the (still ff-ready) pre-merge state."""
    task_id, _wt = _finished_task_with_branch(isolated)
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
        screen = app.screen
        assert isinstance(screen, ConflictScreen)
        await wait_until(
            lambda: screen.wt is not None and screen._ff_ready,
            pilot=pilot,
            what="ff-ready state reached",
        )
        # The base moves on while the screen is open.
        (isolated / "app.py").write_text("a\nMAIN NEWER\nc\n")
        _git(isolated, "commit", "-qam", "main moves on")
        main_head = _git(isolated, "rev-parse", "HEAD").strip()

        await pilot.press("f")
        await pilot.pause()
        await pilot.pause()
        await wait_until(
            lambda: any("refused" in m for m in _toasts(app)),
            pilot=pilot,
            what="refusal toast",
        )
        # No merge happened in the user's checkout, no ff-done state.
        assert not (isolated / "fix.txt").exists()
        assert _git(isolated, "rev-parse", "HEAD").strip() == main_head
        assert not screen._ff_done


async def test_staged_unconcluded_merge_is_not_ff_ready(
    isolated: Path, configured_env: None
) -> None:
    """A merge resolved and staged but never committed (the screen was
    closed mid-resolution) reopens WITHOUT the ff-ready mode: f is gated
    off — no merge commit exists yet, --ff-only would refuse — and the
    honest title points at `s`, which then concludes the merge."""
    manager = TaskManager(isolated)
    task = manager.create("Fix app", "fix the app")
    manager.set_hold(task.id, on=True)
    _env, wt = _parked_content_conflict(isolated, task.id)
    # Resolve and stage by hand, do NOT commit — the resuming scenario.
    (wt / "app.py").write_text("resolved\nb\n")
    _git(wt, "add", "app.py")
    head_before = _git(wt, "rev-parse", "HEAD").strip()

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
            lambda: screen.wt is not None,
            pilot=pilot,
            what="screen prepared",
        )
        await pilot.pause()
        # Not ff-ready: f is inert, the resolution set (s among it) stays.
        assert screen._ff_ready is False
        assert screen.check_action("ff_merge", ()) is False
        assert screen.check_action("commit", ()) is True
        title = str(screen.query_one("#merge-title", Static).content)
        assert "NOT committed" in title and "s to commit" in title

        # The offered next step actually works: armed s commits the merge.
        await pilot.press("s")
        await pilot.pause()
        await pilot.press("s")
        await wait_until(
            lambda: not me.in_progress(wt),
            pilot=pilot,
            what="resolution committed",
        )
        # A successful commit is the logical end of work: the screen
        # closes itself (the outcome toast outlives it).
        await wait_until(
            lambda: not isinstance(app.screen, ConflictScreen),
            pilot=pilot,
            what="merge screen auto-closed",
        )
        # The toast is posted app-level and pumped separately from the
        # screen dismissal — wait for it instead of asserting directly.
        await wait_until(
            lambda: any("Committed on" in m for m in _toasts(app)),
            pilot=pilot,
            what="commit toast shown",
        )
        assert _git(wt, "rev-parse", "HEAD").strip() != head_before
        assert me.is_fast_forward(isolated, branch_for(task.id), "main")


async def test_conflicted_mode_keeps_ff_out_and_resolution_in(
    isolated: Path, configured_env: None
) -> None:
    """Regression: a genuinely conflicted merge keeps the classic key
    set — f is gated OFF, the resolution keys stay live, and the footer
    documents the working set (s/commit still completes the parked
    merge as before)."""
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
            lambda: (
                screen.wt is not None
                and bool(screen.files_list.children)
                and screen.view.current_hunk() is not None
            ),
            pilot=pilot,
            what="conflict screen prepared",
        )
        assert not screen._ff_ready
        assert screen.check_action("ff_merge", ()) is False
        assert screen.check_action("resolve", ("ours",)) is True
        assert screen.check_action("commit", ()) is True
        footer = str(screen.query_one("#merge-footer", Static).content)
        assert "s commit" in footer and "f merge" not in footer

        # The classic flow still works end to end: resolve, armed commit.
        await pilot.press("o")
        await wait_until(
            lambda: _git(wt, "ls-files", "-u").strip() == "",
            pilot=pilot,
            what="hunk resolved and staged",
        )
        await pilot.press("s")
        await pilot.press("s")
        await wait_until(
            lambda: not me.in_progress(wt),
            pilot=pilot,
            what="resolution committed",
        )
        # Auto-close on a successful commit here as well.
        await wait_until(
            lambda: not isinstance(app.screen, ConflictScreen),
            pilot=pilot,
            what="merge screen auto-closed",
        )
        assert me.is_fast_forward(isolated, branch_for(task.id), "main")
