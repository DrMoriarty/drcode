"""Tests for the automatic conflict-resolution layer (`drcode.merge_resolver`)
and its agent wiring (`build_resolver_agent` / `build_resolver_reviewer_agent`).

Like `test_merge_engine.py`, everything runs against real throwaway git
repositories — the module under test is deliberately LLM-free. The verdict
TOOLS are exercised directly through RunContext (no model calls).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from drcode import merge_engine as me
from drcode import merge_resolver as mr
from drcode.agent import (
    ResolverDeps,
    build_resolver_agent,
    build_resolver_reviewer_agent,
)
from drcode.config import Config
from drcode.models import Task
from drcode.task_manager import TaskManager

# -- helpers --------------------------------------------------------------------


def _run(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if check:
        assert proc.returncode == 0, f"git {args}: {proc.stderr}"
    return proc


@pytest.fixture
def repo(project_root: Path) -> Path:
    """A git repository on `main` with one commit (the merge base)."""
    _run(project_root, "init", "-q", "-b", "main")
    _run(project_root, "config", "user.email", "test@example.com")
    _run(project_root, "config", "user.name", "Test")
    (project_root / "app.py").write_text("a\nb\n")
    _run(project_root, "add", ".")
    _run(project_root, "commit", "-qm", "base")
    return project_root


def _task_worktree(repo: Path, task_id: str) -> Path:
    wt = repo / ".drcode" / "worktrees" / task_id
    _run(repo, "worktree", "add", str(wt), "-b", f"drcode/{task_id}", "HEAD")
    return wt


def _conflicted_repo(repo: Path, task_id: str) -> tuple[Path, me.MergeReport]:
    """A task branch and main both editing app.py → a content conflict
    parked in the task's worktree by the reversed merge."""
    wt = _task_worktree(repo, task_id)
    (wt / "app.py").write_text("task\nb\n")
    _run(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("main\nb\n")
    _run(repo, "commit", "-qam", "main edit")
    report = mr.prepare(repo, task_id, wt, "main")
    assert report.ok and report.conflicted
    return wt, report


@pytest.fixture
def fake_config() -> Config:
    return Config(project_root=Path.cwd(), model="test-model", api_key="sk-test")


@pytest.fixture
def manager(project_root: Path) -> TaskManager:
    return TaskManager(project_root)


# -- prepare + agent context --------------------------------------------------------


def test_prepare_parks_conflict_and_context_describes_both_sides(
    repo: Path, manager: TaskManager
) -> None:
    wt, report = _conflicted_repo(repo, "t1")
    task = manager.create("Wire the feature", "map-reduce wiring detail")
    context = mr.build_agent_context(repo, task.title, task.title, report, wt)
    assert "app.py" in context
    assert "drcode/t1" in context and "main" in context
    assert "Wire the feature" in context
    # Both sides' commits are named.
    assert "task edit" in context and "main edit" in context
    # The triage report path is pointed at.
    assert str(me.report_path(repo, "t1")) in context


def test_prepare_reruns_on_already_parked_merge(repo: Path) -> None:
    # Calling prepare twice must not fail: the second call re-triages the
    # half-done merge in place.
    wt, report = _conflicted_repo(repo, "t1")
    report2 = mr.prepare(repo, "t1", wt, "main")
    assert report2.ok and report2.conflicted
    assert [f.path for f in report.remaining] == [f.path for f in report2.remaining]


# -- verification funnel: mechanics -------------------------------------------------


def test_mechanical_checks_compile_ok_and_failure(repo: Path) -> None:
    results = mr.mechanical_checks(repo)
    compile_check = next(c for c in results if c.name.startswith("compile"))
    assert compile_check.ok and not compile_check.skipped

    (repo / "broken.py").write_text("def f(:\n")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "broken")
    results = mr.mechanical_checks(repo)
    compile_check = next(c for c in results if c.name.startswith("compile"))
    assert not compile_check.ok
    assert "broken.py" in compile_check.output


def test_run_test_suite_skipped_without_tests(repo: Path) -> None:
    result = mr.run_test_suite(repo)
    assert result.skipped and result.ok


def test_run_test_suite_pass_and_fail(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the sys.executable pytest path (deterministic: no uv project
    # environment provisioning inside the test).
    monkeypatch.setattr(mr.shutil, "which", lambda name: None)
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("")
    (repo / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    result = mr.run_test_suite(repo)
    assert result.ok and not result.skipped

    (repo / "tests" / "test_ok.py").write_text("def test_ok():\n    assert False\n")
    result = mr.run_test_suite(repo)
    assert not result.ok and "assert" in result.output


def test_run_test_suite_skipped_when_pytest_unavailable(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mr.shutil, "which", lambda name: None)
    monkeypatch.setattr(mr, "_pytest_importable", lambda: False)
    (repo / "tests").mkdir()
    result = mr.run_test_suite(repo)
    assert result.skipped and result.ok


# -- review diffs, restore, conclude --------------------------------------------------


def test_review_diffs_show_both_sides(repo: Path) -> None:
    wt, report = _conflicted_repo(repo, "t1")
    # "Resolve" by hand: keep both sides' lines.
    (wt / "app.py").write_text("task\nmain\nb\n")
    ours, theirs = mr.review_diffs(wt, [f.path for f in report.files])
    # Relative to the task branch (ours) the resolution ADDS the base
    # side's line; relative to the base branch (theirs) — the task's.
    assert "+main" in ours
    assert "+task" in theirs
    # The cap truncates with an explicit marker.
    ours_capped, _ = mr.review_diffs(wt, [f.path for f in report.files], cap=10)
    assert "truncated" in ours_capped


def test_restore_conflict_returns_pristine_state(repo: Path) -> None:
    wt, _report = _conflicted_repo(repo, "t1")
    # Simulate an agent's failed attempt: the file was edited (markers gone).
    (wt / "app.py").write_text("half resolved\n")
    assert mr.restore_conflict(repo, "t1", wt, "main")
    # The conflict is deployed again: markers back, path still unmerged.
    text = (wt / "app.py").read_text(encoding="utf-8")
    assert "<<<<<<<" in text
    assert me.unmerged_paths(wt) == ["app.py"]
    report2 = me.load_report(repo, "t1")
    assert report2 is not None and report2.conflicted


def test_conclude_commits_and_makes_base_fast_forward(repo: Path) -> None:
    wt, _report = _conflicted_repo(repo, "t1")
    (wt / "app.py").write_text("task\nmain\nb\n")
    _run(wt, "add", ".")
    ok, msg = mr.conclude(wt, "resolved")
    assert ok, msg
    assert me.unmerged_paths(wt) == []
    assert me.is_fast_forward(repo, "drcode/t1", "main")


def test_conclude_refuses_with_unmerged_paths(repo: Path) -> None:
    wt, _report = _conflicted_repo(repo, "t1")
    ok, msg = mr.conclude(wt, "premature")
    assert not ok and "remain" in msg


def test_resolver_outcome_summary(repo: Path) -> None:
    outcome = mr.ResolverOutcome(verdict="resolved")
    outcome.checks = [
        mr.CheckResult(name="compile", ok=True),
        mr.CheckResult(name="ruff check", ok=True, skipped=True),
    ]
    outcome.tests = mr.CheckResult(name="test suite", ok=False, output="1 failed")
    text = outcome.summary()
    assert "needs-human" in text or "resolved" in text
    assert "compile: ok" in text
    assert "ruff check: skipped" in text
    assert "test suite: FAILED" in text
    assert "NOTHING committed" in text


# -- the verdict tools ----------------------------------------------------------------


def _resolver_deps(manager: TaskManager, task: Task, root: Path) -> ResolverDeps:
    return ResolverDeps(
        root=root, manager=manager, task_id=task.id, scratch=root / ".scratch"
    )


def _ctx(deps: ResolverDeps) -> RunContext[ResolverDeps]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


def _tool(agent, name: str):
    return agent._function_toolset.tools[name].function


def test_set_resolution_records_verdict(
    repo: Path, manager: TaskManager, fake_config: Config
) -> None:
    task = manager.create("t", "d")
    deps = _resolver_deps(manager, task, repo)
    agent = build_resolver_agent(fake_config, root=repo)
    set_resolution = _tool(agent, "set_resolution")

    out = set_resolution(_ctx(deps), resolved=True, report="kept both sides")
    assert "verification" in out.lower()
    assert len(deps.verdicts) == 1
    assert deps.verdicts[-1].resolved and deps.verdicts[-1].label == "resolved"

    out = set_resolution(_ctx(deps), resolved=False, report="unsure about hunk 2")
    assert "human" in out.lower()
    assert deps.verdicts[-1].resolved is False and deps.verdicts[-1].label == "refused"


def test_set_review_verdicts(
    repo: Path, manager: TaskManager, fake_config: Config
) -> None:
    task = manager.create("t", "d")
    deps = _resolver_deps(manager, task, repo)
    agent = build_resolver_reviewer_agent(fake_config, root=repo)
    set_review = _tool(agent, "set_review")

    assert set_review(_ctx(deps), verdict="intact", report="both sides present")
    assert deps.verdicts[-1].resolved and deps.verdicts[-1].label == "intact"

    set_review(_ctx(deps), verdict="lost", report="wiring dropped")
    assert not deps.verdicts[-1].resolved and deps.verdicts[-1].label == "lost"

    set_review(_ctx(deps), verdict="unsure", report="")
    assert not deps.verdicts[-1].resolved and deps.verdicts[-1].label == "unsure"

    # An unknown verdict is rejected with a retry prompt, not recorded.
    out = set_review(_ctx(deps), verdict="maybe", report="")
    assert "Invalid verdict" in out
    assert len(deps.verdicts) == 3


def test_resolver_agents_toolsets(
    repo: Path, manager: TaskManager, fake_config: Config
) -> None:
    resolver = build_resolver_agent(fake_config, root=repo)
    names = {t.name for t in resolver._function_toolset.tools.values()}
    # The resolver edits files and reports, but never commits.
    assert {
        "read_file",
        "edit_file",
        "write_file",
        "run_bash",
        "set_resolution",
    } <= names
    assert "git_commit" not in names
    assert "set_verification" not in names

    reviewer = build_resolver_reviewer_agent(fake_config, root=repo)
    names = {t.name for t in reviewer._function_toolset.tools.values()}
    # The reviewer is read-only + its verdict tool.
    assert {"read_file", "grep_search", "glob_files", "set_review"} <= names
    assert not names & {"edit_file", "write_file", "run_bash"}


# -- prompts ------------------------------------------------------------------------


def test_resolver_prompt_placeholders_and_override(
    project_root: Path, repo: Path
) -> None:
    from drcode.prompts import load_resolver_prompt, load_resolver_reviewer_prompt

    prompt = load_resolver_prompt()
    assert "(unknown)" in prompt  # no root: the neutral marker is substituted
    substituted = load_resolver_prompt(repo)
    assert "{project_path}" not in substituted
    assert str(repo) in substituted
    # The reviewer prompt exists and is overridable per project.
    assert load_resolver_reviewer_prompt()
    override_dir = project_root / ".drcode" / "prompts"
    override_dir.mkdir(parents=True, exist_ok=True)
    (override_dir / "resolver_reviewer.md").write_text("custom reviewer rules")
    assert load_resolver_reviewer_prompt(project_root) == "custom reviewer rules"


def test_task_new_fields_default() -> None:
    task = Task(title="t")
    assert task.resolver_messages == []
    assert task.needs_human is False
