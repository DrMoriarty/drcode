"""Regression tests for the conflict-resolver retry state (the a6a6df25
silence): a network failure schedules a retry ("retrying in N s"), but the
round's `finally` block wiped that state whenever `task.needs_human` was
set — including a STALE flag left over from a previous round (it survives
restarts and is only cleared by a successful resolution or the merge
screen). The retry timer then fired into an empty schedule and the resolver
went silent forever.

The scenario: a task with a parked conflict and needs_human=True from a
previous round; a fresh resolver round whose first model request raises a
retryable network error. The retry MUST survive the finally block and the
round must actually resume after the backoff.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage

import drcode.core.pipeline_resolver as pr
from drcode import agent_model
from drcode import merge_engine as me
from drcode import merge_resolver as mr
from drcode.agent import (
    ResolutionVerdict,
    ResolverDeps,
    build_resolver_agent,
    build_resolver_reviewer_agent,
)
from drcode.config import Config
from drcode.merge_engine import ConflictFile, MergeReport
from drcode.merge_resolver import CheckResult
from drcode.models import TaskKind
from drcode.settings import ProjectSettings, ProjectSettingsStore, TaskIsolation
from drcode.task_env import TaskEnvironment
from drcode.task_manager import TaskManager
from drcode.tui.app import DrCodeApp
from tests.network_mocks import (
    ScriptedState,
    api_error,
    event_texts,
    http_error,
    text_response,
    tool_response,
    wait_until,
)


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


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
def conflicted_task(isolated: Path) -> Any:
    """A task whose branch diverged from the base on shared.txt — the
    engine's base-into-worktree merge (run at round start by prepare())
    parks a real conflict in the task's worktree."""
    manager = TaskManager(isolated)
    task = manager.create("Fix the conflict", "fix the conflict")
    task.kind = TaskKind.BUG
    manager.storage.save(task)
    env = TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    wt, _ = env.ensure_root(task.id)
    (wt / "shared.txt").write_text("task side")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "task work")
    (isolated / "shared.txt").write_text("base side")
    _git(isolated, "add", "-A")
    _git(isolated, "commit", "-q", "-m", "base work")
    return task


async def test_retry_survives_stale_needs_human(
    isolated: Path,
    conflicted_task: Any,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A network failure in a fresh round must schedule a retry that
    actually fires — even though needs_human is a stale True from a
    previous round."""
    task = conflicted_task
    task.needs_human = True  # stale flag from the pre-restart round
    TaskManager(isolated).storage.save(task)

    app = DrCodeApp(isolated)

    # Quiesce the background actors so only the test drives the resolver.
    async def _noop() -> None:
        return None

    app._unmerged_work_startup_scan = _noop  # type: ignore[method-assign]
    app._poll_git_indicator = _noop  # type: ignore[method-assign]
    app._poll_pending_installs = _noop  # type: ignore[method-assign]
    # The queued task must NOT be promoted into development by the
    # pipeline's `_advance()` (the resolver round's finally tail): the
    # coding agent is a different machinery — here it would only crash on
    # the missing '#chat-log' DOM node and panic the app mid-test.
    app.run_agent = lambda task_id: None  # type: ignore[method-assign]

    # The resolver model fails with a retryable transport error on every
    # request — the test asserts the retries happen, not the resolution.
    state = ScriptedState()
    state.steps = [api_error(), api_error()]

    def fake_build_agent(
        config: Any, root: Path | None = None, task_id: str | None = None
    ) -> Agent[Any, str]:
        return Agent(state.model())

    import drcode.core.pipeline_resolver as pr

    pr.build_resolver_agent = fake_build_agent
    # Shrink the 5 s backoff so the timer fires within the test.
    monkeypatch.setattr(pr, "_RETRY_BASE_DELAY_S", 0.2)

    async with app.run_test() as pilot:
        await pilot.pause()
        # The app loaded its own Task instance from storage — service
        # events land there, not on the pre-start object.
        live = app.manager.get(task.id)
        assert live.needs_human  # the stale flag survived the load
        app._maybe_start_resolver(task.id)
        # First attempt fails -> the "retrying" event AND a live retry
        # schedule + resume flag (the finally block must not wipe them).
        await wait_until(
            lambda: any("retrying in" in t for t in event_texts(live)),
            pilot,
            what="first retry announcement",
        )
        assert task.id in app._resolver_retry_at
        assert task.id in app._resolver_resume_round
        # The retry fires and the round resumes in place (a second model
        # request) — before the fix the timer fired into an empty schedule
        # and the resolver stayed silent forever.
        await wait_until(
            lambda: len(state.calls) >= 2,
            pilot,
            what="resumed second model request",
        )
        # The second attempt fails too -> another retry is scheduled and
        # the network-attempt counter keeps counting across the resume.
        await wait_until(
            lambda: len([t for t in event_texts(live) if "retrying in" in t]) >= 2,
            pilot,
            what="second retry announcement",
        )
        assert app._resolver_net_count.get(task.id) == 2


# -- the targeted-repair fork (a 'lost' intent-review verdict) -----------------------
# The old behavior on ANY failed funnel level: restore the pristine conflict —
# a 35-file resolution thrown away over a one-line fix the reviewer itself
# described. The repair fork feeds the review report back into the SAME
# resolver transcript and re-runs the FULL funnel; the budget is 2 per
# conflict, and everything that is not a concrete 'lost' finding still rolls
# back.

_LOST_REPORT = (
    "shared.txt: the base side's greeting line was lost — the merged "
    "content must keep the greeting the task branch removed."
)


def _resolve_steps(report: str = "wove both sides") -> list[Any]:
    return [
        tool_response("write_file", {"path": "shared.txt", "content": "merged\n"}),
        tool_response("set_resolution", {"resolved": True, "report": report}),
        text_response("done"),
    ]


def _repair_steps() -> list[Any]:
    # The repair continuation: a point fix in the SAME transcript.
    return [
        tool_response(
            "set_resolution", {"resolved": True, "report": "restored the greeting"}
        ),
        text_response("repaired"),
    ]


def _review_steps(verdict: str, report: str) -> list[Any]:
    return [
        tool_response("set_review", {"verdict": verdict, "report": report}),
        text_response("reviewed"),
    ]


def _repair_app(
    isolated: Path,
    task: Any,
    monkeypatch: pytest.MonkeyPatch,
    state: ScriptedState,
) -> DrCodeApp:
    """The scripted-model app for the repair tests: the real resolver and
    reviewer builders (both draw their model through agent_model.build_model),
    the background actors quiesced, the repair delay shrunk."""
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())
    import drcode.core.pipeline_resolver as pr

    monkeypatch.setattr(pr, "build_resolver_agent", build_resolver_agent)
    monkeypatch.setattr(
        pr, "build_resolver_reviewer_agent", build_resolver_reviewer_agent
    )
    monkeypatch.setattr(pr, "_REPAIR_RETRY_DELAY_S", 0.2)

    app = DrCodeApp(isolated)

    async def _noop() -> None:
        return None

    app._unmerged_work_startup_scan = _noop  # type: ignore[method-assign]
    app._poll_git_indicator = _noop  # type: ignore[method-assign]
    app._poll_pending_installs = _noop  # type: ignore[method-assign]
    app.run_agent = lambda task_id: None  # type: ignore[method-assign]
    return app


async def test_repair_round_fixes_lost_work_and_commits(
    isolated: Path,
    conflicted_task: Any,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 'lost' review verdict with a concrete finding schedules a repair
    round INSTEAD of the rollback: the resolution stays in the worktree,
    the review report continues the SAME transcript, the full funnel
    re-runs, and an 'intact' re-review commits the merged result."""
    task = conflicted_task
    state = ScriptedState()
    state.steps = (
        _resolve_steps()
        + _review_steps("lost", _LOST_REPORT)
        + _repair_steps()
        + _review_steps("intact", "both sides present")
    )
    app = _repair_app(isolated, task, monkeypatch, state)

    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        app._maybe_start_resolver(task.id)
        await wait_until(
            lambda: any("FAILED the intent review" in t for t in event_texts(live)),
            pilot,
            what="the repair-fork announcement",
        )
        assert "nothing is rolled back" in "\n".join(event_texts(live))
        assert not live.needs_human
        # The repair invocation continues the round — the conflict is NOT
        # redeployed over the resolution.
        await wait_until(
            lambda: any("targeted repair" in t for t in event_texts(live)),
            pilot,
            what="the repair continuation event",
        )
        await wait_until(
            lambda: any("verified and committed" in t for t in event_texts(live)),
            pilot,
            what="the committed resolution",
        )
        assert not live.needs_human
        # ONE round entry: the repair turn rides in task.resolver_messages[-1].
        assert len(live.resolver_messages) == 1
        blob = json.dumps(live.resolver_messages[-1])
        assert "intent review report" in blob
        assert "greeting" in blob  # the reviewer's finding reached the agent
        assert app._resolver_repair_count.get(task.id) == 1
        assert not app._resolver_repair_round
        # The merge is concluded in the worktree — a fast-forward for the hook.
        env = TaskEnvironment(
            isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
        )
        wt = env.worktree_path(task.id)
        assert not me.in_progress(wt)
        assert (wt / "shared.txt").read_text() == "merged\n"


async def test_repair_budget_exhaustion_rolls_back(
    isolated: Path,
    conflicted_task: Any,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the repair budget (2) is spent on a persistently 'lost'
    verdict, the behavior is the old one: the conflict is restored and a
    human takes over."""
    task = conflicted_task
    state = ScriptedState()
    state.steps = (
        _resolve_steps()
        + _review_steps("lost", _LOST_REPORT)
        + _repair_steps()  # repair 1
        + _review_steps("lost", _LOST_REPORT)
        + _repair_steps()  # repair 2
        + _review_steps("lost", _LOST_REPORT)
    )
    app = _repair_app(isolated, task, monkeypatch, state)

    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        app._maybe_start_resolver(task.id)
        # The whole 6-step script (3 reviews + 2 repairs + the funnel work
        # between them) is observed through THIS single wait — give it an
        # explicit budget; the 30 s default expires under parallel load.
        await wait_until(
            lambda: any("human must take over" in t for t in event_texts(live)),
            pilot,
            timeout=120,
            what="the final human hand-off",
        )
        assert live.needs_human
        assert app._resolver_repair_count.get(task.id) == 2
        assert not app._resolver_repair_round
        texts = "\n".join(event_texts(live))
        assert "Repair round 1 of 2" in texts
        assert "Repair round 2 of 2" in texts
        # The pristine conflict is back.
        env = TaskEnvironment(
            isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
        )
        wt = env.worktree_path(task.id)
        assert "<<<<<<<" in (wt / "shared.txt").read_text()


async def test_unsure_verdict_never_repairs(
    isolated: Path,
    conflicted_task: Any,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'unsure' is not a concrete finding — no repair round, the old
    rollback runs immediately."""
    task = conflicted_task
    state = ScriptedState()
    state.steps = _resolve_steps() + _review_steps(
        "unsure", "something might be missing but I cannot name it"
    )
    app = _repair_app(isolated, task, monkeypatch, state)

    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        app._maybe_start_resolver(task.id)
        await wait_until(
            lambda: any("human must take over" in t for t in event_texts(live)),
            pilot,
            what="the human hand-off",
        )
        assert live.needs_human
        texts = "\n".join(event_texts(live))
        assert "targeted repair" not in texts
        assert app._resolver_repair_count.get(task.id) is None
        assert not app._resolver_repair_round
        env = TaskEnvironment(
            isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
        )
        wt = env.worktree_path(task.id)
        assert "<<<<<<<" in (wt / "shared.txt").read_text()


async def test_vague_lost_report_never_repairs(
    isolated: Path,
    conflicted_task: Any,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 'lost' verdict whose report names nothing concrete (no file, no
    path, too short to act on) keeps the rollback."""
    task = conflicted_task
    state = ScriptedState()
    state.steps = _resolve_steps() + _review_steps(
        "lost", "Work from one side of the merge appears to be missing."
    )
    app = _repair_app(isolated, task, monkeypatch, state)

    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        app._maybe_start_resolver(task.id)
        await wait_until(
            lambda: any("human must take over" in t for t in event_texts(live)),
            pilot,
            what="the human hand-off",
        )
        assert live.needs_human
        assert "targeted repair" not in "\n".join(event_texts(live))
        assert app._resolver_repair_count.get(task.id) is None


# -- window-aware chunked intent review (the review funnel's level 3) -----------------


class _ScriptedVerdictReviewer:
    """Stands in for `build_resolver_reviewer_agent`: records its prompts
    and reports a scripted verdict per chunk (no model calls)."""

    def __init__(self, labels: list[str], reports: list[str]) -> None:
        self._labels = labels
        self._reports = reports
        self.prompts: list[str] = []

    async def run(self, prompt: str, deps: Any = None, usage_limits: Any = None) -> Any:
        index = len(self.prompts)
        self.prompts.append(prompt)
        deps.verdicts.append(
            ResolutionVerdict(
                resolved=self._labels[index] == "intact",
                report=self._reports[index],
                label=self._labels[index],
            )
        )
        return SimpleNamespace(usage=None)


@pytest.fixture
def conflicted_two_files(isolated: Path) -> Any:
    """A task whose branch diverged from the base in TWO files with large
    differing content — enough for the window-aware review to split into
    two chunks under a small budget."""
    manager = TaskManager(isolated)
    task = manager.create("Fix the conflicts", "fix the conflicts")
    task.kind = TaskKind.BUG
    manager.storage.save(task)
    env = TaskEnvironment(
        isolated, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
    )
    wt, _ = env.ensure_root(task.id)
    for name in ("left.txt", "right.txt"):
        (wt / name).write_text(
            "".join(f"task line {i:03d}\n" for i in range(400)) + f"shared {name}\n"
        )
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "task work")
    for name in ("left.txt", "right.txt"):
        (isolated / name).write_text(
            "".join(f"base line {i:03d}\n" for i in range(400)) + f"shared {name}\n"
        )
    _git(isolated, "add", "-A")
    _git(isolated, "commit", "-q", "-m", "base work")
    return task, wt


async def test_review_chunking_aggregates_verdicts(
    isolated: Path,
    conflicted_two_files: Any,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The intent review is window-aware: a small reviewer window splits
    the two-file review into two chunk prompts, and a 'lost' chunk folds
    the aggregate verdict to 'lost' (the commit is blocked)."""
    task, wt = conflicted_two_files
    # Park the real conflict (the engine's reversed merge)...
    report = mr.prepare(isolated, task.id, wt, "main")
    assert report.ok and report.conflicted
    # ...and "resolve" it by hand: the union of both sides, no markers —
    # levels 1-2 of the funnel pass (no .py files, no tests/ dir).
    for name in ("left.txt", "right.txt"):
        (wt / name).write_text("task line 000\nbase line 000\nshared " + name + "\n")

    # A small reviewer window (with a shrunk overhead reserve) forces the
    # two ~11k-char file diffs into two chunks under the ~14.7k-char budget.
    monkeypatch.setattr(mr, "_REVIEW_OVERHEAD_TOKENS", 400)
    fake_cfg = Config(
        project_root=isolated,
        model="mock-model",
        api_key="mock-key",
        max_context_tokens=5000,
    )

    fake = _ScriptedVerdictReviewer(
        ["lost", "intact"],
        ["left.txt: the base-side notification wiring was dropped", "clean"],
    )
    monkeypatch.setattr(
        pr, "build_resolver_reviewer_agent", lambda config, root=None: fake
    )

    app = DrCodeApp(isolated)
    app.resolved_config = lambda role: fake_cfg  # type: ignore[method-assign]

    deps = ResolverDeps(
        root=wt, manager=app.manager, task_id=task.id, scratch=wt / ".s"
    )
    deps.verdicts.append(
        ResolutionVerdict(resolved=True, report="wove both sides", label="resolved")
    )
    outcome = await app._verify_resolution(task, wt, report, deps)

    # One reviewer prompt per chunk, each scoped to its own file.
    assert len(fake.prompts) == 2
    assert "Review chunk 1/2" in fake.prompts[0] and "left.txt" in fake.prompts[0]
    assert "Review chunk 2/2" in fake.prompts[1] and "right.txt" in fake.prompts[1]
    assert "judge ONLY" in fake.prompts[0]
    # Any 'lost' chunk → the aggregate verdict is 'lost', with the chunk
    # map and the losing chunk's report in the funnel output.
    assert outcome.review_verdict == "lost"
    assert "review split into 2 chunks" in outcome.review_report
    assert "wiring was dropped" in outcome.review_report
    # A lost review blocks the auto-commit.
    assert outcome.verdict == "needs-human"
    assert not outcome.committed


# -- the intent review's own retry budget (#9c3236e8) ----------------------------
#
# The reviewer pass is the CHEAP tail of the verification funnel, but an
# infrastructure failure there used to be silently treated as an 'unsure'
# verdict — rolling back the whole EXPENSIVE resolver round. These tests
# exercise `_verify_resolution` directly with the funnel's mechanical
# parts stubbed green and a scripted reviewer.


class _FakeReviewer:
    """Stands in for the reviewer agent: raises the scripted exceptions
    in order, then succeeds and reports the given verdict label."""

    def __init__(self, failures: list[Exception], label: str = "intact") -> None:
        self.failures = failures
        self.label = label
        self.calls = 0

    async def run(self, prompt: str, deps: ResolverDeps, usage_limits: Any) -> Any:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        deps.verdicts.append(
            ResolutionVerdict(resolved=False, report="nothing lost", label=self.label)
        )
        return SimpleNamespace(usage=RunUsage())


def _report(task_id: str) -> MergeReport:
    return MergeReport(
        task_id=task_id,
        branch=f"drcode/{task_id}",
        base_branch="main",
        worktree="/tmp/nowhere",
        ok=True,
        conflicted=True,
        files=[ConflictFile(path="shared.txt", kind="content")],
    )


def _verify_deps(app: DrCodeApp, task: Any, root: Path) -> ResolverDeps:
    return ResolverDeps(
        root=root,
        manager=app.manager,
        task_id=task.id,
        scratch=Path("/tmp/drcode-does-not-exist"),
        verdicts=[ResolutionVerdict(resolved=True, report="markers cleaned")],
    )


async def test_reviewer_429_retried_then_intact(
    isolated: Path,
    conflicted_task: Any,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two 429s from the reviewer, then a successful 'intact' pass: the
    round must COMMIT (verdict 'resolved') — the reviewer's own retry
    budget absorbs the infra hiccups without burning the round's."""
    task = conflicted_task
    app = DrCodeApp(isolated)
    reviewer = _FakeReviewer([http_error(429), http_error(429)])
    monkeypatch.setattr(pr, "_RETRY_BASE_DELAY_S", 0.01)
    monkeypatch.setattr(pr, "build_resolver_reviewer_agent", lambda *a, **k: reviewer)
    monkeypatch.setattr(pr, "mechanical_checks", lambda *a, **k: [])
    monkeypatch.setattr(
        pr, "run_test_suite", lambda *a, **k: CheckResult(name="pytest", ok=True)
    )
    monkeypatch.setattr(
        pr,
        "review_diff_chunks",
        lambda *a, **k: [mr.ReviewChunk(paths=["shared.txt"], ours="", theirs="")],
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        outcome = await app._verify_resolution(
            live, isolated, _report(task.id), _verify_deps(app, live, isolated)
        )
        assert reviewer.calls == 3
        assert outcome.review_verdict == "intact"
        assert outcome.verdict == "resolved"
        texts = event_texts(live)
        retries = [t for t in texts if "Intent review failed" in t and "retrying" in t]
        assert len(retries) == 2


async def test_reviewer_context_overflow_no_retry(
    isolated: Path,
    conflicted_task: Any,
    configured_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic context-overflow 400 from the reviewer: NO retries
    (every replay is guaranteed the same 400) — straight to 'unsure', the
    review stays mandatory (nothing committed)."""
    task = conflicted_task
    app = DrCodeApp(isolated)
    reviewer = _FakeReviewer(
        [http_error(400, "This model's maximum context length is exceeded")]
    )
    monkeypatch.setattr(pr, "_RETRY_BASE_DELAY_S", 0.01)
    monkeypatch.setattr(pr, "build_resolver_reviewer_agent", lambda *a, **k: reviewer)
    monkeypatch.setattr(pr, "mechanical_checks", lambda *a, **k: [])
    monkeypatch.setattr(
        pr, "run_test_suite", lambda *a, **k: CheckResult(name="pytest", ok=True)
    )
    monkeypatch.setattr(
        pr,
        "review_diff_chunks",
        lambda *a, **k: [mr.ReviewChunk(paths=["shared.txt"], ours="", theirs="")],
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        live = app.manager.get(task.id)
        outcome = await app._verify_resolution(
            live, isolated, _report(task.id), _verify_deps(app, live, isolated)
        )
        assert reviewer.calls == 1  # a blind retry would make this 2+
        assert outcome.review_verdict == "unsure"
        assert outcome.verdict == "needs-human"
        assert "context overflow" in outcome.review_report
        assert any("retrying cannot help" in t for t in event_texts(live))
