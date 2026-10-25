"""The on_tester_passed MERGE_HEAD guard and the deferred install re-run.

An install/deploy hook reads "whatever lies in the checkout right now" —
and the main checkout is a mutating shared resource whose state after
on_accepted is not guaranteed (the #9d29e63d incident: a failed hook-merge
left MERGE_HEAD + conflict markers on disk, and the immediately-following
`uv tool install .` built a wheel OUT OF THE CONFLICTED TREE). Three
mechanisms under test:

1. the pipeline guard: the hook never runs while the main checkout has
   MERGE_HEAD set — it is skipped, `task.install_pending` is stamped
   (persisted) and a loud service event is written;
2. the pinned payload: `git.commit` carries the VERIFIED commit (the task
   branch tip with isolation, the shared checkout's HEAD without it) so
   a well-written hook never needed the cwd at all;
3. the deferred re-run: once the branch is merged and the checkout is
   clean, the unmerged-work tracking (10 s poll / startup scan) re-runs
   the hook exactly once and clears the flag."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import pytest

from drcode.models import TaskStatus, VerificationStatus
from drcode.settings import (
    ProjectSettings,
    ProjectSettingsStore,
    TaskIsolation,
)
from drcode.task_env import TaskEnvironment, branch_for
from drcode.task_manager import TaskManager
from drcode.tui.app import DrCodeApp
from drcode.workflow import WORKFLOW_EXAMPLE_CONTENT, load_hooks
from tests.network_mocks import event_texts, wait_until

# -- fixtures (mirroring tests/test_unmerged_reminder.py) ----------------------------


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured (but never contacted) model endpoint — suppresses the
    first-start ModelSetupDialog so the app mounts straight into the
    pipeline state under test."""
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=root, check=check, capture_output=True, text=True
    )


@pytest.fixture
def git_root(project_root: Path) -> Path:
    """A project root that is a real git repository with one commit."""
    _git(project_root, "init", "-q")
    _git(project_root, "config", "user.email", "test@example.com")
    _git(project_root, "config", "user.name", "Test")
    (project_root / ".gitignore").write_text(
        ".drcode/tasks/\n.drcode/settings.json\n.drcode/github_state.json\n"
        ".drcode/worktrees/\n.drcode/merge_reports/\n.drcode/pending_notifications.json\n"
    )
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
def app_env(isolated: Path, configured_env: None) -> DrCodeApp:
    return DrCodeApp(isolated)


def finished_task(app: DrCodeApp) -> Any:
    task = app.manager.create("Fix the bug", "fix the bug")
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    app.manager.storage.save(task)
    return task


def commit_on_branch(env: TaskEnvironment, task_id: str) -> None:
    """Give the task's branch one commit the current branch lacks — via
    the task's worktree, the way a real finished task leaves its work."""
    wt, _ = env.ensure_root(task_id)
    (wt / "fix.txt").write_text("fix")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", f"fix for {task_id}")


def make_hook_script(root: Path, marker: Path) -> None:
    """An on_tester_passed hook that appends its payload to a marker file —
    every invocation leaves one JSON line, so the line count IS the call
    count."""
    source = (
        "import json\n"
        "\n"
        "def on_tester_passed(task):\n"
        f"    with open({str(marker)!r}, 'a') as f:\n"
        "        f.write(json.dumps(task) + '\\n')\n"
        "    return {'text': 'installed'}\n"
    )
    drcode = root / ".drcode"
    drcode.mkdir(parents=True, exist_ok=True)
    (drcode / "workflow.py").write_text(source)


def make_mid_merge(root: Path, task_id: str) -> None:
    """Leave a half-done conflicted merge of `drcode/<task_id>` in the MAIN
    checkout — MERGE_HEAD set, conflict markers on disk (the incident's
    state). Both sides change README.md so the merge actually conflicts."""
    env = TaskEnvironment(root, ProjectSettings(task_isolation=TaskIsolation.WORKTREE))
    wt, _ = env.ensure_root(task_id)
    (wt / "README.md").write_text("task side")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "task side")
    (root / "README.md").write_text("main side")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "main side")
    # check=False: the merge is SUPPOSED to fail with a conflict.
    _git(root, "merge", "--no-edit", branch_for(task_id), check=False)


# -- 1. the MERGE_HEAD guard ----------------------------------------------------------


async def test_guard_skips_hook_on_mid_merge_checkout(app_env: DrCodeApp) -> None:
    """MERGE_HEAD in the main checkout: the hook is NOT called, the task is
    stamped install_pending (persisted) and a loud service event lands in
    the log."""
    app = app_env
    marker = app.project_root / "marker.jsonl"
    make_hook_script(app.project_root, marker)
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    commit_on_branch(app.task_env, task.id)
    make_mid_merge(app.project_root, task.id)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._run_tester_passed_hook(task)
        await pilot.pause()
    assert not marker.exists()  # the hook never ran
    assert "on_tester_passed SKIPPED" in "\n".join(event_texts(task))
    assert f"drcode/{task.id}" in "\n".join(event_texts(task))
    reloaded = TaskManager(app.project_root).get(task.id)
    assert reloaded is not None and reloaded.install_pending


async def test_guard_without_isolation(git_root: Path, configured_env: None) -> None:
    """No isolation: no branch to name, but the guard still holds — the
    shared checkout mid-merge skips the hook the same way (the incident's
    very scenario without isolation)."""
    app = DrCodeApp(git_root)
    marker = git_root / "marker.jsonl"
    make_hook_script(git_root, marker)
    app.workflow = load_hooks(git_root)
    task = finished_task(app)
    # A half-done merge in the shared checkout, unrelated to any branch:
    # divergent README changes on two branches conflict.
    _git(git_root, "checkout", "-q", "-b", "other")
    (git_root / "README.md").write_text("other side")
    _git(git_root, "add", "-A")
    _git(git_root, "commit", "-q", "-m", "other side")
    _git(git_root, "checkout", "-q", "-")
    (git_root / "README.md").write_text("main side")
    _git(git_root, "add", "-A")
    _git(git_root, "commit", "-q", "-m", "main side")
    _git(git_root, "merge", "--no-edit", "other", check=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._run_tester_passed_hook(task)
        await pilot.pause()
    assert not marker.exists()
    assert "on_tester_passed SKIPPED" in "\n".join(event_texts(task))
    assert "unresolved merge in the main checkout" in "\n".join(event_texts(task))
    reloaded = TaskManager(git_root).get(task.id)
    assert reloaded is not None and reloaded.install_pending


async def test_guard_recheck_blocks_rerun_until_clean(app_env: DrCodeApp) -> None:
    """The re-run path re-checks the checkout: a task already stamped
    install_pending whose checkout is STILL mid-merge gets skipped again
    (the flag survives — a skip does not resolve the deferral), and the
    resolved checkout finally gets exactly one install."""
    app = app_env
    marker = app.project_root / "marker.jsonl"
    make_hook_script(app.project_root, marker)
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    task.install_pending = True
    app.manager.storage.save(task)
    commit_on_branch(app.task_env, task.id)
    make_mid_merge(app.project_root, task.id)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._run_tester_passed_hook(task)
        await pilot.pause()
        assert not marker.exists()  # still guarded
        assert task.install_pending
        skipped = "\n".join(event_texts(task)).count("on_tester_passed SKIPPED")
        assert skipped == 1  # no duplicate event for an already-stamped flag
        # Resolve the way a human would: merge again and commit the
        # resolved README.
        _git(app.project_root, "merge", "--abort")
        _git(
            app.project_root,
            "merge",
            "--no-edit",
            branch_for(task.id),
            check=False,
        )
        (app.project_root / "README.md").write_text("resolved")
        _git(app.project_root, "add", "-A")
        _git(app.project_root, "commit", "-q", "-m", f"merge {branch_for(task.id)}")
        await asyncio.to_thread(app._unmerged_completed_ids)
        app._maybe_rerun_pending_installs()
        await wait_until(lambda: marker.exists(), pilot, what="deferred install ran")
        await pilot.pause()
        assert len(marker.read_text().splitlines()) == 1  # exactly once
        assert not task.install_pending
        reloaded = TaskManager(app.project_root).get(task.id)
        assert reloaded is not None and not reloaded.install_pending


# -- 2. the pinned payload ------------------------------------------------------------


async def test_payload_pins_task_branch_commit(app_env: DrCodeApp) -> None:
    """With isolation the payload's git.commit is the task branch tip —
    the verified commit the PASSED verdict is stamped on — even though the
    hook's cwd is the main checkout (the worktree directory is gone under
    worktree_removal: on_completion)."""
    app = app_env
    marker = app.project_root / "marker.jsonl"
    make_hook_script(app.project_root, marker)
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    commit_on_branch(app.task_env, task.id)
    # Simulate the completion removal: the worktree directory is gone,
    # the branch survives.
    app.task_env.remove(task.id)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._run_tester_passed_hook(task)
        await pilot.pause()
    payloads = [json.loads(line) for line in marker.read_text().splitlines()]
    assert len(payloads) == 1
    payload = payloads[0]
    tip = _git(app.project_root, "rev-parse", branch_for(task.id)).stdout.strip()
    assert payload["git"]["commit"] == tip
    assert payload["worktree"] == {"branch": branch_for(task.id)}


async def test_no_isolation_pins_head_commit(
    git_root: Path, configured_env: None
) -> None:
    """Without isolation the shared checkout IS the verified tree: the
    hook runs in the project root (as before) and git.commit is its HEAD
    — behaviour otherwise unchanged."""
    app = DrCodeApp(git_root)
    marker = git_root / "marker.jsonl"
    make_hook_script(git_root, marker)
    app.workflow = load_hooks(git_root)
    task = app.manager.create("Fix the bug", "fix the bug")
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    app.manager.storage.save(task)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._run_tester_passed_hook(task)
        await pilot.pause()
    payloads = [json.loads(line) for line in marker.read_text().splitlines()]
    assert len(payloads) == 1
    payload = payloads[0]
    head = _git(git_root, "rev-parse", "HEAD").stdout.strip()
    assert payload["git"]["commit"] == head
    assert "worktree" not in payload
    assert not task.install_pending  # the hook ran — nothing deferred


# -- 3. the deferred re-run -----------------------------------------------------------


async def test_deferred_rerun_after_manual_merge_once(app_env: DrCodeApp) -> None:
    """A manual `git merge drcode/<id>` resolves the stuck install: the
    unmerged-work tracking notices, the hook runs EXACTLY ONCE and
    install_pending (plus the stale hook_merge_failed) is cleared and
    persisted."""
    app = app_env
    marker = app.project_root / "marker.jsonl"
    make_hook_script(app.project_root, marker)
    app.workflow = load_hooks(app.project_root)
    task = finished_task(app)
    task.install_pending = True
    task.hook_merge_failed = True
    app.manager.storage.save(task)
    commit_on_branch(app.task_env, task.id)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not marker.exists()  # still unmerged — nothing deferred fires
        _git(
            app.project_root,
            "merge",
            "-q",
            "--no-ff",
            branch_for(task.id),
            "-m",
            "manual merge",
        )
        await asyncio.to_thread(app._unmerged_completed_ids)
        app._maybe_rerun_pending_installs()
        await wait_until(lambda: marker.exists(), pilot, what="deferred install ran")
        await pilot.pause()
        assert len(marker.read_text().splitlines()) == 1
        assert not task.install_pending
        assert not task.hook_merge_failed  # cleared by the same pass
        # A second pass finds nothing to do — no second install.
        await asyncio.to_thread(app._unmerged_completed_ids)
        app._maybe_rerun_pending_installs()
        await pilot.pause()
        assert len(marker.read_text().splitlines()) == 1
        reloaded = TaskManager(app.project_root).get(task.id)
        assert reloaded is not None
        assert not reloaded.install_pending
        assert not reloaded.hook_merge_failed


async def test_deferred_rerun_on_startup_scan(
    isolated: Path, configured_env: None
) -> None:
    """The conflict was resolved while the app was down (the branch is
    merged, the checkout clean): the STARTUP scan re-runs the deferred
    install — no live poll needed."""
    marker = isolated / "marker.jsonl"
    make_hook_script(isolated, marker)
    manager = TaskManager(isolated)
    task = manager.create("Fix the bug", "fix the bug")
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    task.install_pending = True
    manager.storage.save(task)
    env = TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    commit_on_branch(env, task.id)
    _git(
        isolated,
        "merge",
        "-q",
        "--no-ff",
        branch_for(task.id),
        "-m",
        "manual merge",
    )
    app = DrCodeApp(isolated)
    async with app.run_test() as pilot:
        await wait_until(
            lambda: marker.exists(), pilot, what="startup-scan deferred install"
        )
    assert len(marker.read_text().splitlines()) == 1
    reloaded = TaskManager(isolated).get(task.id)
    assert reloaded is not None and not reloaded.install_pending


# -- 4. the documented recipe (template smoke) ----------------------------------------


def _extract_template_hook() -> str:
    """The commented-out `on_tester_passed` template hook, uncommented."""
    lines = WORKFLOW_EXAMPLE_CONTENT.splitlines()
    start = next(
        i for i, ln in enumerate(lines) if ln.startswith("# def on_tester_passed")
    )
    picked: list[str] = []
    for line in lines[start:]:
        if not line.startswith("#"):
            break
        picked.append("" if line == "#" else line[2:])
    return textwrap.dedent("\n".join(picked))


def test_template_recipe_builds_from_git_archive() -> None:
    """Smoke test of the documented recipe: the template's commented
    on_tester_passed is valid Python and — with worktree isolation —
    installs from a `git archive` EXPORT of the task branch (a temporary
    directory, removed afterwards), never from the cwd. subprocess is
    faked, so no uv/network is needed."""
    calls: list[tuple[Any, dict[str, Any]]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    class _FakeSubprocess:
        def run(self, args: Any, **kwargs: Any) -> _Result:
            calls.append((args, kwargs))
            return _Result()

    source = _extract_template_hook()
    compile(source, "<template-on_tester_passed>", "exec")  # valid Python
    namespace: dict[str, Any] = {"subprocess": _FakeSubprocess()}
    exec(compile(source, "<template-on_tester_passed>", "exec"), namespace)  # noqa: S102

    # Isolation: export + install from the temp dir, then rmtree.
    result = namespace["on_tester_passed"]({"worktree": {"branch": "drcode/abc123"}})
    assert result == {"text": ""}
    shell_cmds = [c for c, _ in calls if isinstance(c, str)]
    assert any(
        c.startswith("git archive drcode/abc123 | tar -x -C ") for c in shell_cmds
    )
    install_args = [c for c, _ in calls if isinstance(c, list)]
    assert any(
        c[:3] == ["uv", "tool", "install"] and c[3].startswith(tempfile.gettempdir())
        for c in install_args
    ), install_args
    tmp_dirs = [c[3] for c in install_args if c[:3] == ["uv", "tool", "install"]]
    assert all(not Path(d).exists() for d in tmp_dirs)  # cleaned up

    # No isolation: the cwd install — the old recipe, still correct there.
    calls.clear()
    namespace["on_tester_passed"]({})
    assert ["uv", "tool", "install", ".", "--force"] in [
        c for c, _ in calls if isinstance(c, list)
    ]


def test_git_archive_recipe_exports_branch_tree(git_root: Path) -> None:
    """The mechanism the recipe relies on: `git archive <branch> | tar -x
    -C <tmp>` really exports the branch's committed tree into a fresh
    directory — no worktree registration involved."""
    _git(git_root, "checkout", "-q", "-b", "drcode/smoke")
    (git_root / "shipped.txt").write_text("shipped")
    _git(git_root, "add", "-A")
    _git(git_root, "commit", "-q", "-m", "ship it")
    _git(git_root, "checkout", "-q", "-")
    src = Path(tempfile.mkdtemp(prefix="drcode-install-test-"))
    try:
        subprocess.run(
            f"git archive drcode/smoke | tar -x -C {src}",
            shell=True,
            cwd=git_root,
            check=True,
            capture_output=True,
        )
        assert (src / "shipped.txt").read_text() == "shipped"
        assert (src / "README.md").exists()
    finally:
        shutil.rmtree(src, ignore_errors=True)
