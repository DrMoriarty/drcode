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

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage

import drcode.core.pipeline_resolver as pr
from drcode import merge_resolver as mr
from drcode.agent import ResolutionVerdict, ResolverDeps
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
