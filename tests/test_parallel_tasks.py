"""Parallel development slots (Variant A): several independent tasks run
at once, each in its own git worktree, while the tester stays single.

Covered: the promotion loop (free slots filled from the queue, the rest
waits), interaction with the depends_on gate (blocked tasks never
promoted even with free slots; siblings unblocked together after the
dependency is completed AND merged), the ESC semantics matrix
(parallelism 1 = the historical global stop; parallelism 2 = only the
viewed task's run) under the double-press arming (first ESC arms, second
confirms; debounced fast repeats and expired windows re-arm), the worktree activation race (serialized
ensure_root), the settings gating (parallel > 1 without isolation
behaves as 1, with a warning) and the startup recovery (stale ACTIVE
tasks demoted, slots refilled). The narrowed promotion gate (variant A)
is covered too: with parallel slots the queue keeps flowing while the
tester is busy or a TESTING task waits out its retry backoff, while a
single-slot pipeline and a ready-to-test task keep the absolute testing
priority.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from drcode import agent_model
from drcode.models import Task, TaskStatus
from drcode.settings import (
    EscAbort,
    ProjectSettings,
    ProjectSettingsStore,
    TaskIsolation,
    parallel_tasks_label,
)
from drcode.task_env import TaskEnvironment
from drcode.task_manager import TaskManager
from drcode.tui.actions import ActionsLayer
from drcode.tui.app import DrCodeApp
from tests.network_mocks import ScriptedState, text_response, wait_until

# -- fixtures --------------------------------------------------------------------


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


def _save_settings(
    root: Path, *, isolation: TaskIsolation, parallel: int
) -> ProjectSettings:
    store = ProjectSettingsStore(root)
    settings = store.load()
    settings.task_isolation = isolation
    settings.parallel_tasks = parallel
    store.save(settings)
    return settings


@pytest.fixture
def parallel_app(git_root: Path) -> DrCodeApp:
    """App over a git repo with worktree isolation and 2 slots."""
    _save_settings(git_root, isolation=TaskIsolation.WORKTREE, parallel=2)
    return DrCodeApp(git_root)


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedState:
    state = ScriptedState()
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())
    return state


@pytest.fixture
def esc_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten the armed-ESC timings: pilot presses land within a few
    milliseconds of each other, so the real 250ms debounce would swallow
    every deliberate second press; a short arm window keeps the expiry
    test fast."""
    monkeypatch.setattr(ActionsLayer, "_ESC_DEBOUNCE_SECONDS", 0.0)
    monkeypatch.setattr(ActionsLayer, "_ESC_ARM_SECONDS", 2.0)


def _disarm_window(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """Shorten only the armed window (the debounce stays 0): for the
    expiry test the window must pass while the test waits."""
    monkeypatch.setattr(ActionsLayer, "_ESC_DEBOUNCE_SECONDS", 0.0)
    monkeypatch.setattr(ActionsLayer, "_ESC_ARM_SECONDS", seconds)


def _gate_on(
    scripted: ScriptedState,
    monkeypatch: pytest.MonkeyPatch,
    release: asyncio.Event,
    *markers: str,
) -> None:
    """Make every model stream hold on any of `markers` until `release`
    is set (both parallel runs hang at their first model request)."""

    async def gated_stream(
        messages: list[Any], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        for m in messages:
            if hasattr(m, "parts"):
                for p in m.parts:
                    content = str(getattr(p, "content", ""))
                    if any(marker in content for marker in markers):
                        await release.wait()
        # A stream function must yield text chunks (or delta dicts), not
        # whole ModelResponse objects — an invalid chunk would make
        # pydantic-ai retry until "Exceeded maximum output retries".
        yield "ok"

    monkeypatch.setattr(
        scripted,
        "model",
        lambda: FunctionModel(scripted.fn, stream_function=gated_stream),
    )


def branch_exists(root: Path, branch: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", branch],
            cwd=root,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


async def _dismiss_startup_dialog(pilot: Any) -> None:
    """Decline the modal AgentsMdDialog ("create AGENTS.md?") the app
    pushes on start when the model is configured but the sandbox project
    has no AGENTS.md — its priority ESC binding would otherwise eat the
    key presses the ESC-matrix tests send."""
    await pilot.press("escape")
    await pilot.pause()
    await pilot.pause()


# -- promotion loop -----------------------------------------------------------------


async def test_limit_two_promotes_two_and_third_waits(parallel_app: DrCodeApp) -> None:
    app = parallel_app
    async with app.run_test() as pilot:
        await pilot.pause()
        a = app.manager.create("one", "do a")
        b = app.manager.create("two", "do b")
        c = app.manager.create("three", "do c")
        app._advance()
        await pilot.pause()
        assert app.manager.active_tasks() == [a, b]
        assert a.status is TaskStatus.ACTIVE
        assert b.status is TaskStatus.ACTIVE
        assert c.status is TaskStatus.QUEUED


async def test_limit_change_mid_session(parallel_app: DrCodeApp) -> None:
    app = parallel_app
    async with app.run_test() as pilot:
        await pilot.pause()
        app.manager.create("one", "do a")
        app.manager.create("two", "do b")
        c = app.manager.create("three", "do c")
        app._advance()
        await pilot.pause()
        assert c.status is TaskStatus.QUEUED
        # Raising the limit lets the waiting task start right away.
        app.settings.parallel_tasks = 3
        app._advance()
        await pilot.pause()
        assert c.status is TaskStatus.ACTIVE
        assert len(app.manager.active_tasks()) == 3
        # Lowering it does not kill running tasks — it only stops new
        # promotions until the slots drain below the limit.
        app.settings.parallel_tasks = 1
        app._advance()
        await pilot.pause()
        assert len(app.manager.active_tasks()) == 3


# -- interaction with the dependency gate ---------------------------------------------


def _complete(app: DrCodeApp, task: Task) -> None:
    app.manager.mark_done(task.id)
    app.manager.set_verification(task.id, True)


async def test_chain_and_independent_task_at_limit_two(parallel_app: DrCodeApp) -> None:
    app = parallel_app
    async with app.run_test() as pilot:
        await pilot.pause()
        a = app.manager.create("first", "do a")
        b = app.manager.create("second", "do b")
        app.manager.set_dependencies(b.id, [a.id])
        c = app.manager.create("third", "do c")
        app._advance()
        # The dependency gate routes promotion through the
        # _promote_with_dependencies worker (git check in a thread): poll
        # for the pass to land instead of a single pause.
        await wait_until(
            lambda: len(app.manager.active_tasks()) == 2,
            pilot,
            what="two slots filled",
        )
        # The independent task and the chain head fill the two slots; the
        # dependent waits even though a slot was free when the pass began.
        assert {t.id for t in app.manager.active_tasks()} == {a.id, c.id}
        assert b.status is TaskStatus.QUEUED
        # Completing the head unblocks the dependent into the freed slot.
        _complete(app, a)
        app._advance()
        await wait_until(
            lambda: b.status is TaskStatus.ACTIVE, pilot, what="dependent promoted"
        )
        assert {t.id for t in app.manager.active_tasks()} == {c.id, b.id}


async def test_blocked_tasks_never_promoted_with_free_slots(
    parallel_app: DrCodeApp,
) -> None:
    app = parallel_app
    async with app.run_test() as pilot:
        await pilot.pause()
        dep = app.manager.create("dependency", "do s")
        x = app.manager.create("x", "do x")
        app.manager.set_dependencies(x.id, [dep.id])
        y = app.manager.create("y", "do y")
        app.manager.set_dependencies(y.id, [dep.id])
        # Free slots (limit 2, nothing active) — still nothing to promote.
        app._advance()
        await pilot.pause()
        assert x.status is TaskStatus.QUEUED
        assert y.status is TaskStatus.QUEUED
        # Completing the dependency (its branch holds no commits, so the
        # mergedness half is satisfied) unblocks BOTH siblings into the
        # two slots in one pass.
        _complete(app, dep)
        app._advance()
        await wait_until(
            lambda: {x.status, y.status} == {TaskStatus.ACTIVE},
            pilot,
            what="both siblings promoted",
        )


async def test_unmerged_dependency_branch_keeps_dependents_blocked(
    parallel_app: DrCodeApp, git_root: Path
) -> None:
    """The mergedness half: a COMPLETED dependency whose drcode/<id>
    branch still holds commits blocks its dependents until the branch is
    merged (by hand or by a hook) — and a manual merge unblocks them even
    while the pipeline is otherwise idle (the 30s unblock tick)."""
    app = parallel_app
    async with app.run_test() as pilot:
        await pilot.pause()
        dep = app.manager.create("dependency", "do s")
        x = app.manager.create("x", "do x")
        app.manager.set_dependencies(x.id, [dep.id])
        # Give the dependency real, unmerged work on its branch.
        env = app.task_env
        await asyncio.to_thread(env.ensure_root, dep.id)
        wt = env.worktree_path(dep.id)
        (wt / "feature.txt").write_text("work")
        _git(wt, "add", "-A")
        _git(wt, "commit", "-q", "-m", "dep work")
        _complete(app, dep)
        app._advance()
        await wait_until(
            lambda: app._dep_gate.get(x.id, {}).get(dep.id) == "unmerged",
            pilot,
            what="dependency gate records the unmerged branch",
        )
        assert x.status is TaskStatus.QUEUED
        # The manual merge unblocks the dependent — with the pipeline
        # idle, exactly what the periodic unblock tick re-evaluates.
        _git(git_root, "merge", "--no-ff", "-q", f"drcode/{dep.id}")
        app._poll_dependency_unblock()
        await wait_until(
            lambda: x.status is TaskStatus.ACTIVE, pilot, what="dependent unblocked"
        )


# -- ESC matrix -----------------------------------------------------------------------


async def test_esc_parallel_one_stops_everything(
    git_root: Path,
    configured_env: None,
    scripted: ScriptedState,
    monkeypatch: pytest.MonkeyPatch,
    esc_timing: None,
) -> None:
    _save_settings(git_root, isolation=TaskIsolation.WORKTREE, parallel=1)
    app = DrCodeApp(git_root)
    task = app.manager.create("solo", "work here")
    release = asyncio.Event()
    _gate_on(scripted, monkeypatch, release, "work here")
    scripted.steps = [text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await _dismiss_startup_dialog(pilot)
        await wait_until(lambda: app.agent_running, pilot, what="agent running")
        # esc_abort: double_press — the first ESC only arms the stop.
        await pilot.press("escape")
        await pilot.pause()
        assert app.agent_running, "the first ESC must only arm the stop"
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not app.agent_running
        await wait_until(
            lambda: task.unfinished,
            pilot,
            what="interrupted attempt marked resumable",
        )


async def test_esc_parallel_two_stops_only_viewed_task(
    git_root: Path,
    configured_env: None,
    scripted: ScriptedState,
    monkeypatch: pytest.MonkeyPatch,
    esc_timing: None,
) -> None:
    _save_settings(git_root, isolation=TaskIsolation.WORKTREE, parallel=2)
    app = DrCodeApp(git_root)
    a = app.manager.create("one", "work a")
    b = app.manager.create("two", "work b")
    release = asyncio.Event()
    _gate_on(scripted, monkeypatch, release, "work a", "work b")
    scripted.steps = [text_response("done"), text_response("done 2")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await _dismiss_startup_dialog(pilot)
        app.view_task(a.id)
        await wait_until(
            lambda: len(app._agent_runs) == 2, pilot, what="both runs alive"
        )
        # Each parallel task develops in its own worktree/branch (the
        # runs create them via asyncio.to_thread before streaming).
        await wait_until(
            lambda: (
                (app.project_root / ".drcode" / "worktrees" / a.id).is_dir()
                and (app.project_root / ".drcode" / "worktrees" / b.id).is_dir()
            ),
            pilot,
            what="both worktrees created",
        )
        assert branch_exists(app.project_root, f"drcode/{a.id}")
        assert branch_exists(app.project_root, f"drcode/{b.id}")
        # esc_abort: double_press — arm, then confirm.
        await pilot.press("escape")
        await pilot.pause()
        assert a.id in app._agent_runs, "the first ESC must only arm the stop"
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        # Only the viewed task's run stopped; the sibling keeps working.
        assert a.id not in app._agent_runs
        assert b.id in app._agent_runs
        await wait_until(
            lambda: a.unfinished, pilot, what="viewed attempt marked resumable"
        )
        assert not b.unfinished
        release.set()
        await wait_until(
            lambda: not app._agent_runs, pilot, what="sibling run finished"
        )


async def test_esc_on_non_running_viewed_task_is_noop_with_toast(
    parallel_app: DrCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = parallel_app
    toasts: list[str] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **kw: toasts.append(str(message))
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        running = app.manager.create("busy", "do it")
        parked = app.manager.create("parked", "")  # no input: stays queued
        app._advance()
        await pilot.pause()
        assert running.status is TaskStatus.ACTIVE
        assert parked.status is TaskStatus.QUEUED
        # Simulate a live run for the active task without a model: mark
        # the registry directly (the guard would leave it empty here).
        app._agent_runs[running.id] = 1
        app.view_task(parked.id)
        app.action_abort_run()
        await pilot.pause()
        assert app._agent_runs == {running.id: 1}
        assert any("Task is not running" in t for t in toasts)


# -- worktree activation race -----------------------------------------------------------


async def test_concurrent_activations_create_distinct_worktrees(git_root: Path) -> None:
    """Two parallel activations must not race git worktree prune/add: the
    activation lock serializes them, and both get their own tree/branch."""
    settings = ProjectSettings(task_isolation=TaskIsolation.WORKTREE, parallel_tasks=2)
    env = TaskEnvironment(git_root, settings)
    results = await asyncio.gather(
        asyncio.to_thread(env.ensure_root, "aaaa1111"),
        asyncio.to_thread(env.ensure_root, "bbbb2222"),
    )
    roots = [root for root, _note in results]
    assert roots[0] != roots[1]
    for task_id, root in zip(("aaaa1111", "bbbb2222"), roots):
        assert root == env.worktree_path(task_id)
        assert root.is_dir()
        assert branch_exists(git_root, f"drcode/{task_id}")


# -- settings gating ----------------------------------------------------------------------


def test_settings_roundtrip_and_options(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    assert store.load().parallel_tasks == 1
    settings = store.load()
    settings.parallel_tasks = 3
    store.save(settings)
    assert ProjectSettingsStore(project_root).load().parallel_tasks == 3
    assert parallel_tasks_label(1) == "1 task"
    assert parallel_tasks_label(3) == "3 tasks"


async def test_parallel_without_isolation_behaves_as_one_with_warning(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_settings(project_root, isolation=TaskIsolation.OFF, parallel=2)
    app = DrCodeApp(project_root)
    toasts: list[str] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **kw: toasts.append(str(message))
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.parallel_limit == 1
        assert any("one at a time" in t for t in toasts), (
            "startup must warn that the persisted parallelism is dormant"
        )
        # The picker refuses parallel > 1 without isolation: the value is
        # not saved, a warning explains why.
        app.action_set_parallel_tasks(3)
        assert app.settings.parallel_tasks == 2  # untouched
        assert any("Task isolation" in t for t in toasts)


async def test_picker_saves_parallel_with_isolation(git_root: Path) -> None:
    _save_settings(git_root, isolation=TaskIsolation.WORKTREE, parallel=1)
    app = DrCodeApp(git_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_set_parallel_tasks(4)
        assert app.settings.parallel_tasks == 4
        assert app.parallel_limit == 4
        saved = ProjectSettingsStore(git_root).load()
        assert saved.parallel_tasks == 4


# -- startup recovery -----------------------------------------------------------------------


async def test_stale_active_tasks_demoted_and_slots_refilled(git_root: Path) -> None:
    """Tasks left ACTIVE by a crashed session are demoted at startup and
    the free slots are refilled by the first _advance()."""
    m = TaskManager(git_root)
    t1 = m.create("stale one", "resume one")
    t2 = m.create("stale two", "resume two")
    t3 = m.create("stale three", "resume three")
    for t in (t1, t2, t3):
        t.status = TaskStatus.ACTIVE
        m.storage.save(t)
    _save_settings(git_root, isolation=TaskIsolation.WORKTREE, parallel=2)
    app = DrCodeApp(git_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        # The startup demotion ran in TaskManager.__init__.
        assert all(
            app.manager.get(t.id).status is TaskStatus.QUEUED for t in (t1, t2, t3)
        )
        app._advance()
        await pilot.pause()
        assert len(app.manager.active_tasks()) == 2


# -- watchdog per run -----------------------------------------------------------------------


async def test_watchdog_nudges_only_the_stalled_run(
    git_root: Path,
    configured_env: None,
    scripted: ScriptedState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One run goes silent; the stall deadline of the OTHER run is not
    affected — the nudge aborts and restarts only the stalled task."""
    _save_settings(git_root, isolation=TaskIsolation.WORKTREE, parallel=2)
    app = DrCodeApp(git_root)
    a = app.manager.create("one", "work a")
    b = app.manager.create("two", "work b")
    release = asyncio.Event()
    _gate_on(scripted, monkeypatch, release, "work a", "work b")
    scripted.steps = [text_response("done"), text_response("done 2")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: len(app._agent_runs) == 2, pilot, what="both runs alive"
        )
        # Only run a goes silent (far past the stall limit); b just emitted.
        now = time.monotonic()
        app._last_activity[a.id] = now - 3600
        app._last_activity[b.id] = now
        app._poll_stall()
        # The nudge aborts ONLY the stalled run and restarts it (a fresh
        # run generation resuming the attempt after the restart delay);
        # the healthy run keeps its original run untouched.
        gen_a = app._agent_generations.get(a.id, 0)
        gen_b = app._agent_generations.get(b.id, 0)
        await wait_until(
            lambda: app._agent_generations.get(a.id, 0) > gen_a,
            pilot,
            what="stalled run aborted and restarted",
        )
        assert b.id in app._agent_runs, "the healthy run must keep running"
        assert app._agent_generations.get(b.id, 0) == gen_b
        assert not b.unfinished
        release.set()
        await wait_until(lambda: not app._agent_runs, pilot, what="runs finished")


# -- promotion while the tester is busy ------------------------------------------------
#
# Variant A of the "testing has absolute priority" rule: with parallel
# slots, development is held back only while the tester is FREE and a
# TESTING task is ready to be tested right now. A tester run in flight
# (or a TESTING task waiting for the busy tester's slot / its retry
# backoff) does not stop free development slots from being filled.


def _to_testing(app: DrCodeApp, task: Task) -> None:
    """Move `task` to TESTING directly (as mark_done would after a
    development run) without touching the pipeline."""
    task.status = TaskStatus.TESTING
    app.manager.storage.save(task)


async def test_queued_task_promoted_while_tester_busy(parallel_app: DrCodeApp) -> None:
    """A in TESTING under a running tester, B waits for the tester's slot:
    the free development slot is filled from the queue instead of idling."""
    app = parallel_app
    async with app.run_test() as pilot:
        await pilot.pause()
        a = app.manager.create("one", "do a")
        b = app.manager.create("two", "do b")
        c = app.manager.create("three", "do c")
        _to_testing(app, a)
        _to_testing(app, b)
        app.tester_running = True  # the tester round for a is in flight
        app._advance()
        await pilot.pause()
        assert c.status is TaskStatus.ACTIVE
        assert a.status is TaskStatus.TESTING
        assert b.status is TaskStatus.TESTING
        assert app.tester_running


async def test_promoted_while_testing_task_in_retry_backoff(
    parallel_app: DrCodeApp,
) -> None:
    """A TESTING task whose tester round failed sits in its retry backoff
    window with the tester idle: the queue still flows into free slots."""
    app = parallel_app
    async with app.run_test() as pilot:
        await pilot.pause()
        a = app.manager.create("one", "do a")
        c = app.manager.create("three", "do c")
        _to_testing(app, a)
        # The tester round for a failed recently: retry scheduled far out.
        app._tester_retry_at[a.id] = time.monotonic() + 3600
        app._advance()
        await pilot.pause()
        assert c.status is TaskStatus.ACTIVE
        assert a.status is TaskStatus.TESTING
        assert not app.tester_running, "the tester must not start for a backoff task"


async def test_ready_testing_still_blocks_promotion(parallel_app: DrCodeApp) -> None:
    """The tester is free and a TESTING task is ready to be tested right
    now: the test is dispatched first and the queue is NOT promoted (the
    old semantics for the case the tester can actually start)."""
    app = parallel_app
    async with app.run_test() as pilot:
        await pilot.pause()
        a = app.manager.create("one", "do a")
        c = app.manager.create("three", "do c")
        _to_testing(app, a)
        app._advance()
        # Synchronous assertions: the tester flag is set before the worker
        # dispatch, and the promotion tail was never reached.
        assert app.tester_running
        assert c.status is TaskStatus.QUEUED
        await pilot.pause()


async def test_parallel_one_keeps_absolute_testing_priority(
    project_root: Path,
) -> None:
    """parallel_limit == 1: the historical rule is untouched — no
    development starts while any task is TESTING, whether the tester is
    busy or the only TESTING task waits out its retry backoff."""
    app = DrCodeApp(project_root)
    assert app.parallel_limit == 1
    async with app.run_test() as pilot:
        await pilot.pause()
        a = app.manager.create("one", "do a")
        c = app.manager.create("three", "do c")
        _to_testing(app, a)
        # Tester busy (a round in flight).
        app.tester_running = True
        app._advance()
        await pilot.pause()
        assert c.status is TaskStatus.QUEUED
        # Tester idle, but the TESTING task is within its retry backoff.
        app.tester_running = False
        app._tester_retry_at[a.id] = time.monotonic() + 3600
        app._advance()
        await pilot.pause()
        assert c.status is TaskStatus.QUEUED


async def test_dependency_gate_promotes_while_tester_busy(
    parallel_app: DrCodeApp,
) -> None:
    """The dependency gate with a busy tester: an UNBLOCKED dependent is
    promoted into the free slot (via the unblock tick, which re-fires
    _advance() — a busy tester no longer suppresses it)."""
    app = parallel_app
    async with app.run_test() as pilot:
        await pilot.pause()
        a = app.manager.create("one", "do a")
        dep = app.manager.create("dependency", "do s")
        x = app.manager.create("x", "do x")
        app.manager.set_dependencies(x.id, [dep.id])
        _to_testing(app, a)
        app.tester_running = True
        _complete(app, dep)  # branch holds no commits: mergedness satisfied
        app._poll_dependency_unblock()
        await wait_until(
            lambda: x.status is TaskStatus.ACTIVE,
            pilot,
            what="dependent promoted while the tester is busy",
        )


async def test_dependency_gate_still_blocks_while_tester_busy(
    parallel_app: DrCodeApp,
) -> None:
    """The dependency gate with a busy tester: an unsatisfied dependency
    keeps its dependent out of the free slot."""
    app = parallel_app
    async with app.run_test() as pilot:
        await pilot.pause()
        a = app.manager.create("one", "do a")
        dep = app.manager.create("dependency", "do s")
        x = app.manager.create("x", "do x")
        app.manager.set_dependencies(x.id, [dep.id])
        _to_testing(app, a)
        app.tester_running = True
        app._advance()
        await wait_until(
            lambda: app._dep_gate.get(x.id, {}).get(dep.id) == "not completed",
            pilot,
            what="dependency gate records the unsatisfied dependency",
        )
        assert x.status is TaskStatus.QUEUED


# -- armed ESC (esc_abort: double_press) -------------------------------------------------


def _save_esc_mode(root: Path, mode: EscAbort) -> None:
    store = ProjectSettingsStore(root)
    settings = store.load()
    settings.esc_abort = mode
    store.save(settings)


async def test_esc_single_press_arms_and_toasts(
    project_root: Path, monkeypatch: pytest.MonkeyPatch, esc_timing: None
) -> None:
    """First ESC with a live run: nothing stops, the arming is announced
    and the statusbar shows it."""
    app = DrCodeApp(project_root)
    toasts: list[str] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **kw: toasts.append(str(message))
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await _dismiss_startup_dialog(pilot)
        task = app.manager.create("busy", "do it")
        app._advance()
        await pilot.pause()
        # Simulate a live run without a model.
        app._agent_runs[task.id] = 1
        # The new-task input binds ESC with priority (cancels the form):
        # move focus to the chat input, or the press never reaches the
        # app-level stop action.
        app.chat_input.focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app._agent_runs == {task.id: 1}, "single ESC must not stop"
        assert app._esc_armed_signature is not None, "the run must be armed"
        assert any("Press ESC again" in t for t in toasts)


async def test_esc_double_press_stops_with_mitigation(
    project_root: Path, monkeypatch: pytest.MonkeyPatch, esc_timing: None
) -> None:
    """Second ESC within the window: the run stops, and both the toast
    and the task's service event point at Ctrl+R as the way back."""
    app = DrCodeApp(project_root)
    toasts: list[str] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **kw: toasts.append(str(message))
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await _dismiss_startup_dialog(pilot)
        task = app.manager.create("busy", "do it")
        app._advance()
        await pilot.pause()
        app._agent_runs[task.id] = 1
        # The new-task input binds ESC with priority (cancels the form):
        # move focus to the chat input, or the press never reaches the
        # app-level stop action.
        app.chat_input.focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.press("escape")
        await pilot.pause()
        assert app._agent_runs == {}, "the confirmed ESC must stop the run"
        assert app._esc_armed_signature is None, "firing must disarm"
        assert any("Run stopped" in t for t in toasts)
        events = [e.text for e in app.manager.get(task.id).events]
        assert any("Run aborted by user (ESC)" in e for e in events)


async def test_esc_fast_double_press_is_debounced(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ESC repeat faster than the debounce interval (the accidental
    double-keystroke — exactly the inertia scenario the arming guards
    against) is ignored; a deliberate press after the debounce gap still
    confirms the arming."""
    # pilot.press runs a full screen-idle sync per keystroke — under a
    # loaded suite two back-to-back presses can take longer than the real
    # 250ms window, flipping the test into the "deliberate repeat" branch.
    # Widen the window: the mechanism under test is the debounce, not the
    # concrete default.
    monkeypatch.setattr(ActionsLayer, "_ESC_DEBOUNCE_SECONDS", 30.0)
    app = DrCodeApp(project_root)
    toasts: list[str] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **kw: toasts.append(str(message))
    )
    # Under a loaded (xdist) suite the two pilot presses can drift past the
    # real 250ms debounce and the second press would confirm instead of being
    # ignored. Widen the debounce for this test and simulate the deliberate
    # gap by backdating the last accepted press instead of sleeping it out.
    monkeypatch.setattr(type(app), "_ESC_DEBOUNCE_SECONDS", 10.0)
    async with app.run_test() as pilot:
        await pilot.pause()
        await _dismiss_startup_dialog(pilot)
        task = app.manager.create("busy", "do it")
        app._advance()
        await pilot.pause()
        app._agent_runs[task.id] = 1
        # The new-task input binds ESC with priority (cancels the form):
        # move focus to the chat input, or the press never reaches the
        # app-level stop action.
        app.chat_input.focus()
        await pilot.pause()
        # Two pilot presses land within the (widened) debounce window.
        await pilot.press("escape")
        await pilot.press("escape")
        await pilot.pause()
        assert app._agent_runs == {task.id: 1}, "the debounced press must be ignored"
        assert app._esc_armed_signature is not None, "arming must survive the debounce"
        # A deliberate press after the debounce gap confirms the stop.
        # (Emulate the gap instead of sleeping it out: re-stamp the arming
        # time and age out the last accepted press, so the press reads as
        # "after the window" without a real 30s wait — and without the
        # arming window expiring under a loaded suite first.)
        app._esc_last_press = 0.0
        app._esc_armed_at = time.monotonic()
        app._esc_last_press = time.monotonic() - 11.0
        await pilot.press("escape")
        await pilot.pause()
        assert app._agent_runs == {}


async def test_esc_after_window_expiry_rearms(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ESC after the armed window expired re-arms instead of firing;
    the next press within the fresh window stops the run."""
    _disarm_window(monkeypatch, 0.1)
    app = DrCodeApp(project_root)
    toasts: list[str] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **kw: toasts.append(str(message))
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await _dismiss_startup_dialog(pilot)
        task = app.manager.create("busy", "do it")
        app._advance()
        await pilot.pause()
        app._agent_runs[task.id] = 1
        # The new-task input binds ESC with priority (cancels the form):
        # move focus to the chat input, or the press never reaches the
        # app-level stop action.
        app.chat_input.focus()
        await pilot.pause()
        # Arm with the short window (direct call: the 0.1s expiry timer
        # would race a pilot press + pause).
        app.action_abort_run()
        assert app._esc_armed_signature is not None
        # Outlive the (shortened) window: the expiry timer disarms.
        await asyncio.sleep(0.3)
        await pilot.pause()
        assert app._esc_armed_signature is None, "the window must expire"
        # Restore a comfortable window for the re-arm phase: the fresh
        # confirming press must land INSIDE the new window.
        monkeypatch.setattr(ActionsLayer, "_ESC_ARM_SECONDS", 2.0)
        await pilot.press("escape")
        await pilot.pause()
        assert app._agent_runs == {task.id: 1}, "a fresh ESC must re-arm"
        assert app._esc_armed_signature is not None
        await pilot.press("escape")
        await pilot.pause()
        assert app._agent_runs == {}


async def test_esc_instant_mode_stops_on_single_press(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """esc_abort: instant — the historical single-press stop, with the
    same mitigation toast and service event."""
    _save_esc_mode(project_root, EscAbort.INSTANT)
    app = DrCodeApp(project_root)
    toasts: list[str] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **kw: toasts.append(str(message))
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await _dismiss_startup_dialog(pilot)
        task = app.manager.create("busy", "do it")
        app._advance()
        await pilot.pause()
        app._agent_runs[task.id] = 1
        # The new-task input binds ESC with priority (cancels the form):
        # move focus to the chat input, or the press never reaches the
        # app-level stop action.
        app.chat_input.focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app._agent_runs == {}, "instant mode: one ESC stops the run"
        assert app._esc_armed_signature is None
        assert any("Run stopped" in t for t in toasts)
        events = [e.text for e in app.manager.get(task.id).events]
        assert any("Run aborted by user (ESC)" in e for e in events)


async def test_esc_parallel_two_arms_only_viewed_task(
    parallel_app: DrCodeApp, monkeypatch: pytest.MonkeyPatch, esc_timing: None
) -> None:
    """Parallel slots: arming and the stop both concern only the viewed
    task's run; the sibling keeps working."""
    app = parallel_app
    toasts: list[str] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **kw: toasts.append(str(message))
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await _dismiss_startup_dialog(pilot)
        a = app.manager.create("viewed", "do a")
        b = app.manager.create("sibling", "do b")
        app._advance()
        await pilot.pause()
        app.view_task(a.id)
        # Simulate two live runs without models. The new-task input binds
        # ESC with priority (cancels the form): move focus to the chat
        # input, or the press never reaches the app-level stop action.
        app._agent_runs[a.id] = 1
        app._agent_runs[b.id] = 2
        app.chat_input.focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app._esc_armed_signature is not None
        assert {a.id, b.id} <= set(app._agent_runs)
        await pilot.press("escape")
        await pilot.pause()
        assert a.id not in app._agent_runs, "only the viewed task stops"
        assert b.id in app._agent_runs, "the sibling keeps working"
        assert any("Run stopped" in t for t in toasts)
        events = [e.text for e in app.manager.get(a.id).events]
        assert any("Run aborted by user (ESC)" in e for e in events)
        sibling_events = [e.text for e in app.manager.get(b.id).events]
        assert not any("Run aborted" in e for e in sibling_events)
