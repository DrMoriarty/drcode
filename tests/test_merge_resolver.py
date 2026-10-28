"""Tests for the automatic conflict-resolution layer (`drcode.merge_resolver`)
and its agent wiring (`build_resolver_agent` / `build_resolver_reviewer_agent`).

Like `test_merge_engine.py`, everything runs against real throwaway git
repositories — the module under test is deliberately LLM-free. The verdict
TOOLS are exercised directly through RunContext (no model calls).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
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
    _no_xdist(monkeypatch)
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


# -- the funnel runs the PROJECT'S environment, never a PATH tool -------


def _fake_worktree_venv(repo: Path) -> Path:
    """A .venv whose python is a shim around the running interpreter that
    logs every invocation — lets the tests observe WHICH python the
    funnel picked (the project's own env, never a PATH lookup)."""
    log = repo / ".venv" / "invocations.log"
    shim = repo / ".venv" / "bin" / "python"
    shim.parent.mkdir(parents=True)
    shim.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" >> {shlex.quote(str(log))}\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    shim.chmod(0o755)
    return log


def _no_xdist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite runs single-process: the xdist probe reports
    absent, so the funnel never appends `-n 4`."""
    real = mr._module_importable
    monkeypatch.setattr(
        mr,
        "_module_importable",
        lambda cmd, module, cwd: real(cmd, module, cwd) if module != "xdist" else False,
    )


def test_run_test_suite_uses_worktree_venv(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _fake_worktree_venv(repo)
    monkeypatch.setattr(mr.shutil, "which", lambda name: None)  # no uv on PATH
    _no_xdist(monkeypatch)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    result = mr.run_test_suite(repo)
    assert result.ok and not result.skipped
    lines = log.read_text().splitlines()
    # The pytest probe ran through the venv python, and the suite itself
    # too (`-m pytest`, no bare `pytest` from PATH).
    assert ["-c", "import pytest"] == lines[:2]
    assert lines[-3:] == ["-m", "pytest", "-q"]


def test_run_test_suite_caps_xdist_fanout(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_worktree_venv(repo)
    monkeypatch.setattr(mr.shutil, "which", lambda name: None)
    captured: dict[str, list[str]] = {}

    def spy(
        name: str, cmd: list[str], *args: object, **kwargs: object
    ) -> mr.CheckResult:
        captured["cmd"] = cmd
        return mr.CheckResult(name=name, ok=True)

    monkeypatch.setattr(mr, "_run_checked", spy)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    result = mr.run_test_suite(repo)
    assert result.ok
    # The project's xdist is available → its addopts `-n auto` would pin
    # every core; the funnel caps the fan-out explicitly.
    assert "-m" in captured["cmd"] and "pytest" in captured["cmd"]
    assert ["-n", "4"] == captured["cmd"][-2:]


def test_run_test_suite_falls_through_skeleton_venv(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The 9d29e63d failure class: a .venv exists but holds nothing (a
    # `--no-sync` skeleton) — the funnel must not end up running a random
    # PATH pytest that cannot import the project.
    shim = repo / ".venv" / "bin" / "python"
    shim.parent.mkdir(parents=True)
    shim.write_text("#!/bin/sh\nexit 1\n")  # every import probe fails
    shim.chmod(0o755)
    monkeypatch.setattr(
        mr.shutil, "which", lambda name: "/usr/local/bin/uv" if name == "uv" else None
    )
    _no_xdist(monkeypatch)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    result = mr.run_test_suite(repo)
    # Falls back to the interpreter drcode runs under (pytest available,
    # worktree on the path) — and since the worktree is no uv project, uv
    # is never allowed to discover an ANCESTOR project's environment.
    assert result.ok and not result.skipped


def test_run_test_suite_skips_with_skeleton_venv_and_no_pytest(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shim = repo / ".venv" / "bin" / "python"
    shim.parent.mkdir(parents=True)
    shim.write_text("#!/bin/sh\nexit 1\n")
    shim.chmod(0o755)
    monkeypatch.setattr(mr.shutil, "which", lambda name: None)
    monkeypatch.setattr(mr, "_pytest_importable", lambda: False)
    (repo / "tests").mkdir()
    result = mr.run_test_suite(repo)
    assert result.skipped and result.ok


def test_mechanical_checks_ruff_from_worktree_venv(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _fake_worktree_venv(repo)
    monkeypatch.setattr(mr.shutil, "which", lambda name: None)  # no PATH ruff
    _no_xdist(monkeypatch)
    (repo / "app.py").write_text("value = 1\n")  # lint-clean content
    results = mr.mechanical_checks(repo)
    ruff = next(c for c in results if c.name == "ruff check")
    assert ruff.ok and not ruff.skipped
    lines = log.read_text().splitlines()
    # ruff was invoked through the venv python (`-m ruff`), never a
    # random PATH binary with a different default ruleset. (pyright may
    # follow it in the log — it resolves through the same env.)
    assert ["-c", "import ruff"] == lines[:2]
    windows = [lines[i : i + 4] for i in range(len(lines) - 3)]
    assert ["-m", "ruff", "check", "."] in windows


def test_mechanical_checks_skips_ruff_when_project_env_lacks_it(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_worktree_venv(repo)
    # ruff is importable in THIS process but NOT via the worktree venv
    # shim? The shim execs this interpreter… force the probe negative.
    monkeypatch.setattr(
        mr,
        "_module_importable",
        lambda cmd, module, cwd: module != "ruff" and module != "pyright",
    )
    results = mr.mechanical_checks(repo)
    for name in ("ruff check", "pyright"):
        check = next(c for c in results if c.name == name)
        assert check.ok and check.skipped


def test_pyright_check_pins_the_venv_environment(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The 9d29e63d failure class: pip pyright is a node wrapper — node
    # pyright does its own environment discovery (VIRTUAL_ENV / PATH) and
    # must not land on a foreign interpreter. The funnel pins the plan's
    # venv into the subprocess env explicitly.
    _fake_worktree_venv(repo)
    (repo / ".venv" / "pyvenv.cfg").write_text("home = /fake\n")
    captured: dict[str, dict[str, str] | None] = {}

    def spy(
        name: str, cmd: list[str], *args: object, **kwargs: object
    ) -> mr.CheckResult:
        if name == "pyright":
            captured["env"] = cast("dict[str, str] | None", kwargs.get("env"))
        return mr.CheckResult(name=name, ok=True)

    monkeypatch.setattr(mr, "_run_checked", spy)
    (repo / "app.py").write_text("value = 1\n")
    mr.mechanical_checks(repo)
    env = captured["env"]
    assert env is not None
    assert env["VIRTUAL_ENV"] == str(repo / ".venv")
    assert env["PATH"].startswith(str(repo / ".venv" / "bin") + os.pathsep)


def test_standalone_pyright_check_pins_the_venv_environment(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The 9d29e63d class hit the STANDALONE pyright_check too (the manual
    # merge screen calls it directly): node pyright does its own
    # VIRTUAL_ENV/PATH discovery, so the pin must not live only inside
    # mechanical_checks.
    _fake_worktree_venv(repo)
    (repo / ".venv" / "pyvenv.cfg").write_text("home = /fake\n")
    monkeypatch.setattr(mr.shutil, "which", lambda name: None)  # no uv on PATH
    monkeypatch.setattr(
        mr, "_module_importable", lambda cmd, module, cwd: module == "pyright"
    )
    captured: dict[str, dict[str, str] | None] = {}

    def spy(
        name: str, cmd: list[str], *args: object, **kwargs: object
    ) -> mr.CheckResult:
        if name == "pyright":
            captured["env"] = cast("dict[str, str] | None", kwargs.get("env"))
        return mr.CheckResult(name=name, ok=True)

    monkeypatch.setattr(mr, "_run_checked", spy)
    result = mr.pyright_check(repo)
    assert not result.skipped
    env = captured["env"]
    assert env is not None
    assert env["VIRTUAL_ENV"] == str(repo / ".venv")
    assert env["PATH"].startswith(str(repo / ".venv" / "bin") + os.pathsep)


def test_standalone_ruff_check_pins_the_venv_environment(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Symmetry with pyright: the standalone ruff_check (manual merge
    # screen) pins the plan's venv defensively too.
    _fake_worktree_venv(repo)
    (repo / ".venv" / "pyvenv.cfg").write_text("home = /fake\n")
    monkeypatch.setattr(mr.shutil, "which", lambda name: None)  # no uv on PATH
    monkeypatch.setattr(
        mr, "_module_importable", lambda cmd, module, cwd: module == "ruff"
    )
    captured: dict[str, dict[str, str] | None] = {}

    def spy(
        name: str, cmd: list[str], *args: object, **kwargs: object
    ) -> mr.CheckResult:
        if name == "ruff check":
            captured["env"] = cast("dict[str, str] | None", kwargs.get("env"))
        return mr.CheckResult(name=name, ok=True)

    monkeypatch.setattr(mr, "_run_checked", spy)
    result = mr.ruff_check(repo)
    assert not result.skipped
    env = captured["env"]
    assert env is not None
    assert env["VIRTUAL_ENV"] == str(repo / ".venv")
    assert env["PATH"].startswith(str(repo / ".venv" / "bin") + os.pathsep)


def test_pyright_check_leaves_uv_run_env_alone(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `uv run` already exports VIRTUAL_ENV and prepends the venv bin to
    # the child's PATH — the funnel must not clobber that.
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    captured: dict[str, dict[str, str] | None] = {}

    def spy(
        name: str, cmd: list[str], *args: object, **kwargs: object
    ) -> mr.CheckResult:
        if name == "pyright":
            captured["env"] = cast("dict[str, str] | None", kwargs.get("env"))
        return mr.CheckResult(name=name, ok=True)

    monkeypatch.setattr(mr, "_run_checked", spy)
    monkeypatch.setattr(
        mr.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None
    )
    monkeypatch.setattr(mr, "_module_importable", lambda cmd, module, cwd: True)
    mr.mechanical_checks(repo)
    env = captured["env"]
    assert env is not None
    assert "VIRTUAL_ENV" not in env or env["VIRTUAL_ENV"] != str(repo / ".venv")


# -- conflict-marker scan (the a6a6df25 prose-only resolution class) ---------------


def test_conflict_marker_hits(repo: Path) -> None:
    (repo / "conflicted.py").write_text(
        "x = 1\n<<<<<<< HEAD\n"
        "task\n||||||| 57cb0d9\nbase\n=======\nmain\n>>>>>>> main\n"
    )
    hits = mr.conflict_marker_hits(repo, ["conflicted.py"])
    lines = [h.split(": ", 1)[1] for h in hits]
    # All four zdiff3 marker forms are found, with line numbers.
    assert lines == ["<<<<<<< HEAD", "||||||| 57cb0d9", "=======", ">>>>>>> main"]
    assert hits[0].startswith("conflicted.py:2:")


def test_conflict_marker_hits_ignores_lookalikes(repo: Path) -> None:
    # Eight-plus marker chars (headings, banners) are NOT conflict markers,
    # and clean files yield nothing.
    (repo / "clean.py").write_text(
        "# ========\nx = 1\nheading =======\nseparator =======\n"
    )
    assert mr.conflict_marker_hits(repo, ["clean.py"]) == []
    assert mr.conflict_marker_hits(repo, ["missing.py"]) == []


def test_mechanical_checks_catches_prose_only_resolution(repo: Path) -> None:
    wt, report = _conflicted_repo(repo, "t1")
    # The agent never touched the file — the funnel's FIRST check must
    # fail on the markers, naming file and line.
    results = mr.mechanical_checks(wt, [f.path for f in report.files])
    markers = results[0]
    assert markers.name == "conflict markers"
    assert not markers.ok and not markers.skipped
    assert "app.py" in markers.output and "<<<<<<<" in markers.output

    # An actually written resolution passes the scan.
    (wt / "app.py").write_text("task\nmain\nb\n")
    results = mr.mechanical_checks(wt, [f.path for f in report.files])
    assert results[0].ok


def test_mechanical_checks_marker_scan_defaults_to_unmerged(repo: Path) -> None:
    # Without explicit paths the scan runs over the unmerged index entries;
    # with no merge in progress it is skipped (nothing to scan).
    results = mr.mechanical_checks(repo)
    assert results[0].name == "conflict markers"
    assert results[0].ok and results[0].skipped

    wt, _report = _conflicted_repo(repo, "t1")
    results = mr.mechanical_checks(wt)
    assert results[0].name == "conflict markers"
    assert not results[0].ok


def test_unresolved_conflict_paths_triage_and_fallback(repo: Path) -> None:
    wt, _report = _conflicted_repo(repo, "t1")
    # The triage report's `remaining` set drives the scan…
    assert mr.unresolved_conflict_paths(repo, "t1", wt) == ["app.py"]
    # …with the unmerged index entries as the fallback when the report is
    # missing (task id without a triage report).
    assert mr.unresolved_conflict_paths(repo, "no-report", wt) == ["app.py"]
    # A clean tree has nothing to scan.
    assert mr.unresolved_conflict_paths(repo, "no-report", repo) == []


def test_set_resolution_rejects_markers_with_model_retry(
    repo: Path, manager: TaskManager, fake_config: Config
) -> None:
    wt, _report = _conflicted_repo(repo, "t1")
    # The verdict is keyed to the task id the triage report was saved
    # under — exactly how the pipeline wires ResolverDeps.
    deps = ResolverDeps(root=wt, manager=manager, task_id="t1", scratch=wt / ".s")
    agent = build_resolver_agent(fake_config, root=wt)
    set_resolution = _tool(agent, "set_resolution")

    # The prose-only resolution: resolved=True without the edits — the
    # tool refuses IN ROUND (ModelRetry) and records nothing, so the
    # agent can apply the edits and call again.
    with pytest.raises(ModelRetry) as exc:
        set_resolution(_ctx(deps), resolved=True, report="both sides woven")
    text = str(exc.value)
    assert "app.py" in text and "<<<<<<<" in text
    assert deps.verdicts == []

    # After the file is actually written, the verdict goes through.
    (wt / "app.py").write_text("task\nmain\nb\n")
    out = set_resolution(_ctx(deps), resolved=True, report="both sides woven")
    assert "verification" in out.lower()
    assert deps.verdicts[-1].resolved and deps.verdicts[-1].label == "resolved"


def test_resolver_outcome_summary_couples_rejected_claim(repo: Path) -> None:
    # The funnel rejected the attempt: the agent's "resolved" claim must
    # not read as success in the logged report (a6a6df25: an optimistic
    # weave description right above the FAILED line).
    outcome = mr.ResolverOutcome(verdict="needs-human")
    outcome.agent_resolved = True
    outcome.agent_report = "progress-callback threaded through the cache"
    outcome.checks = [mr.CheckResult(name="conflict markers", ok=False)]
    text = outcome.summary()
    assert "claim REJECTED" in text
    assert "resolver agent: resolved (claim REJECTED" in text

    # A passing round (or a refusal) carries no annotation.
    ok_outcome = mr.ResolverOutcome(verdict="resolved")
    ok_outcome.agent_resolved = True
    assert "claim REJECTED" not in ok_outcome.summary()
    refused = mr.ResolverOutcome(verdict="needs-human")
    refused.agent_resolved = False
    assert "claim REJECTED" not in refused.summary()


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
    # A failed suite must carry its pytest output into the report —
    # a bare FAILED line is undiagnosable in the service events.
    assert "test output:" in text and "1 failed" in text
    assert "NOTHING committed" in text

    # A skipped suite reports no output (nothing ran); a passing one
    # stays compact (the short `-q` tail is enough).
    skipped = mr.ResolverOutcome(verdict="resolved")
    skipped.tests = mr.CheckResult(name="test suite", ok=True, skipped=True)
    assert "test output:" not in skipped.summary()


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
