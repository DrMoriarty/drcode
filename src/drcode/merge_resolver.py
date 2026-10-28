"""Automatic merge-conflict resolution support (`drcode.merge_resolver`).

The second layer on top of the algorithmic merge engine
(`drcode.merge_engine`): the conflicts the provably-safe auto-resolvers
(identical stages, rerere) could not clear go to an AGENT — no human in
the loop, but with a mandatory verification funnel that gates the
auto-commit.

The module is deliberately LLM-free: it owns everything around the agent
round that must be deterministic —

- `prepare` — deploy (or re-detect) the conflicted merge in the task's
  worktree through the engine's reversed merge and return the triage
  report;
- `build_agent_context` — the resolver's assignment: the triage report,
  the task's intent (title/description), what each side changed
  (`drcode/<id>` vs the base branch) and the conflicted paths themselves;
- `mechanical_checks` / `run_test_suite` — level 1 and 2 of the
  verification funnel (the conflict-marker scan + compile/ruff/pyright;
  the project's test suite);
- `review_diffs` — the level-3 payloads: diff(resolved, ours) and
  diff(resolved, theirs), the inputs of the intent-preservation review
  (the direct protection against the #6c065401 incident class — a lost
  piece of someone else's feature inside a "clean" resolution);
- `restore_conflict` — roll a FAILED attempt back to the pristine
  conflicted state (abort + re-run the reversed merge), so a human always
  takes over exactly the conflict the engine deployed — never a
  half-resolved tree;
- `conclude` — commit the verified resolution (`merge_engine.finish`);
  the merge into the base branch becomes a fast-forward.

Contract unchanged: **drcode never merges on its own** — after `conclude`
the resolved branch is ready, but the fast-forward into the user's
checkout stays the workflow hook's / the user's job.

Sync module (plain subprocesses, mirroring `drcode.task_env` /
`drcode.merge_engine`): callers on the event loop wrap the calls in
`asyncio.to_thread`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import merge_engine
from .merge_engine import MergeReport
from .tools import child_priority_kwargs

__all__ = [
    "REVIEW_DIFF_CHAR_CAP",
    "TEST_TIMEOUT",
    "CheckResult",
    "build_agent_context",
    "compile_check",
    "conclude",
    "conflict_marker_check",
    "conflict_marker_hits",
    "failed_names",
    "format_verify_report",
    "mechanical_checks",
    "prepare",
    "pyright_check",
    "restore_conflict",
    "review_diffs",
    "ruff_check",
    "run_test_suite",
    "save_verify_report",
    "unresolved_conflict_paths",
    "verify_report_path",
]
# (check-timings helpers are intentionally not re-exported: they serve the
# interactive funnel only.)

# Longest a single git/subprocess call may run (mirrors the engine).
_TIMEOUT = 120
# The test-suite run of the verification funnel: a full pass may take a
# while, but it must never hang the pipeline forever (the stall watchdog
# does not cover this worker).
TEST_TIMEOUT = 1800
# Cap for each side's review diff (chars): a giant generated diff would
# blow the reviewer's context; the cap is generous enough for any real
# conflict and the truncation is announced in the payload.
REVIEW_DIFF_CHAR_CAP = 120_000


# -- data model ----------------------------------------------------------------


@dataclass
class CheckResult:
    """One verification-funnel check's outcome. `ok` is False for both a
    real failure (nonzero exit) and a missing tool ONLY when the check is
    mandatory; `skipped` marks the best-effort checks a project does not
    have tooling for (they never fail the funnel)."""

    name: str
    ok: bool
    output: str = ""
    skipped: bool = False

    @property
    def line(self) -> str:
        status = "skipped" if self.skipped else ("ok" if self.ok else "FAILED")
        return f"{self.name}: {status}"


# -- preparation -----------------------------------------------------------------


def prepare(
    project_root: Path,
    task_id: str,
    worktree: Path,
    base_branch: str,
    progress: Callable[[str], None] | None = None,
) -> MergeReport:
    """Deploy (or re-detect) the conflicted merge in the task's worktree.

    Runs the engine's reversed merge (`merge_base_into_worktree`): a
    merge already parked there is re-triaged in place (the engine handles
    `resuming`), a fresh one is attempted with the safe auto-resolvers
    switched on. Returns the report; the caller checks `report.ok` /
    `report.conflicted` — a non-conflicted outcome means there is nothing
    for the agent to do (clean merge, already up to date, or the engine
    refused).

    `progress` is an optional callback invoked from the calling thread at
    the step boundaries — ``"worktree"`` (this function's own deploy
    checks), then the engine's ``"merge"`` / ``"triage"`` /
    ``"auto-resolve"`` — so a UI can show per-step progress (the
    ConflictScreen's prepare panel). Plain calls from a worker thread;
    never made when None.
    """
    if progress is not None:
        progress("worktree")
    return merge_engine.merge_base_into_worktree(
        project_root, task_id, worktree, base_branch, save=True, progress=progress
    )


def _git(*args: str, cwd: Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
            **child_priority_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, f"{type(e).__name__}: {e}"
    return proc.returncode, (proc.stdout + proc.stderr).strip()


# -- conflict-marker scan -------------------------------------------------------
# The a6a6df25 incident class: the resolver agent described the resolution in
# prose and reported set_resolution(resolved=True) WITHOUT ever writing the
# files — the funnel then burned compile/ruff/pyright/pytest + a reviewer
# round on a syntactically still-conflicted tree. These helpers are the cheap
# (no-test-suite, near-no-subprocess) pre-checks that catch it immediately.


# A git conflict marker line: exactly seven marker chars at line start,
# optionally followed by a space + label (`<<<<<<< HEAD`, `||||||| 57cb0d9`,
# `=======`, `>>>>>>> main`). Eight or more chars (an RST heading, a `# =====`
# banner) do NOT match — the `(?: .*)?$` tail refuses anything but a space or
# end-of-line after the seventh char.
_CONFLICT_MARKER_RE = re.compile(r"^(<{7}|\|{7}|={7}|>{7})(?: .*)?$")


def conflict_marker_hits(worktree: Path, paths: list[str]) -> list[str]:
    """Scan the given worktree-relative paths for leftover conflict
    markers. Pure file reading, no subprocess; returns `path:line: <line>`
    entries (capped — a pathological file must not flood the caller)."""
    hits: list[str] = []
    for rel in paths:
        try:
            text = (worktree / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if _CONFLICT_MARKER_RE.match(line):
                hits.append(f"{rel}:{lineno}: {line[:80]}")
                if len(hits) >= 40:
                    return hits
    return hits


def _triage_remaining_paths(project_root: Path, task_id: str) -> list[str]:
    """The paths the task's persisted triage report lists as still
    conflicted (its `remaining` array); empty when the report is missing,
    corrupted or already fully resolved."""
    try:
        data = json.loads(
            merge_engine.report_path(project_root, task_id).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return []
    remaining = data.get("remaining") if isinstance(data, dict) else None
    if not isinstance(remaining, list):
        return []
    return [
        entry["path"]
        for entry in remaining
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    ]


def unresolved_conflict_paths(
    project_root: Path, task_id: str, worktree: Path
) -> list[str]:
    """The paths that may still carry conflict markers in `worktree`:
    the triage report's `remaining` set when it is readable, else the
    in-progress merge's unmerged index entries. Empty = nothing to scan."""
    paths = _triage_remaining_paths(project_root, task_id)
    return paths or merge_engine.unmerged_paths(worktree)


def conflict_marker_check(
    worktree: Path, paths: list[str] | None = None
) -> CheckResult:
    """Level 0 of the verification funnel: the conflicted files must not
    carry leftover conflict markers. Scans `paths` (the triage report's
    conflicted set in the automatic funnel); defaults to the in-progress
    merge's unmerged index entries. Skipped when there is nothing to
    scan (no merge in progress)."""
    if paths is None:
        paths = merge_engine.unmerged_paths(worktree)
    if not paths:
        return CheckResult(
            name="conflict markers", ok=True, skipped=True, output="no conflicted paths"
        )
    hits = conflict_marker_hits(worktree, paths)
    return CheckResult(
        name="conflict markers",
        ok=not hits,
        output="\n".join(hits)[:8000],
    )


# -- agent context ---------------------------------------------------------------


def build_agent_context(
    project_root: Path,
    task_title: str,
    task_description: str,
    report: MergeReport,
    worktree: Path,
) -> str:
    """The resolver's assignment text: the triage report, the task's
    intent, and what each side changed around the conflict.

    "ours" is the task branch (`report.branch`), "theirs" the base branch
    (`report.base_branch`) — the merge runs INSIDE the task's worktree,
    so HEAD is the task and MERGE_HEAD the base. The two `git log` ranges
    (and the name-status of the overlapping changes) give the agent the
    semantic picture the zdiff3 hunks alone cannot carry.
    """
    remaining = report.remaining
    files_block = "\n".join(
        f"- {f.path} [{f.kind}]"
        + (
            f" — {f.hunks} hunk(s), conflict lines {f.conflict_lines or {}}"
            if f.hunks is not None
            else ""
        )
        + (f" — resolution: {f.resolution}" if f.resolution else "")
        for f in report.files
    )
    ours_log = _git("log", "--oneline", f"{report.base_branch}..HEAD", cwd=worktree)[1]
    theirs_log = _git("log", "--oneline", f"HEAD..{report.base_branch}", cwd=worktree)[
        1
    ]
    paths = " ".join(f"'{f.path}'" for f in report.files[:50])
    # Three-dot diffs: HEAD...MERGE_HEAD = what the BASE side changed in
    # those paths since the merge base; MERGE_HEAD...HEAD = what the TASK
    # side changed.
    base_touch = (
        _git("diff", "--name-status", "HEAD...MERGE_HEAD", "--", paths, cwd=worktree)[1]
        if paths
        else ""
    )
    task_touch = (
        _git("diff", "--name-status", "MERGE_HEAD...HEAD", "--", paths, cwd=worktree)[1]
        if paths
        else ""
    )
    return (
        f"Task #{report.task_id}: {task_title}\n"
        f"{(task_description or '').strip()[:4000]}\n\n"
        f"The merge of the base branch {report.base_branch} into this "
        f"task's branch {report.branch} conflicted; {len(remaining)} "
        f"path(s) still need a decision (the merge is in progress in THIS "
        f"worktree):\n{files_block}\n\n"
        f"What the TASK side changed relative to the base branch:\n"
        f"{ours_log or '(no commits)'}\n\n"
        f"What the BASE side changed that the task does not have:\n"
        f"{theirs_log or '(no commits)'}\n\n"
        f"How the conflicted paths were changed by each side since the "
        f"merge base (name-status):\n"
        f"base ({report.base_branch}):\n{base_touch or '(no changes)'}\n"
        f"task ({report.branch}):\n{task_touch or '(no changes)'}\n\n"
        f"Triage report: {merge_engine.report_path(project_root, report.task_id)}\n"
        f"Read the conflicted files (zdiff3 markers: ours = {report.branch}, "
        f"theirs = {report.base_branch}, plus the common-ancestor section), "
        f"resolve them, then report with set_resolution."
    )


# -- verification funnel -----------------------------------------------------------


def _run_checked(
    name: str,
    cmd: list[str],
    cwd: Path,
    timeout: int = _TIMEOUT,
    env: dict[str, str] | None = None,
) -> CheckResult:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
            **child_priority_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return CheckResult(name=name, ok=False, output=f"{type(e).__name__}: {e}")
    out = (proc.stdout + "\n" + proc.stderr).strip()
    return CheckResult(name=name, ok=proc.returncode == 0, output=out[:8000])


def _venv_python(worktree: Path) -> Path | None:
    """The worktree's own virtualenv interpreter, when one exists."""
    for rel in (".venv/bin/python", ".venv/Scripts/python.exe"):
        candidate = worktree / rel
        if candidate.exists():
            return candidate
    return None


def _module_importable(python_cmd: list[str], module: str, cwd: Path) -> bool:
    """True when `module` imports in the environment the given python
    command resolves to — probed with the SAME launcher the check would
    use, never a PATH lookup. Generous timeout: for `uv run` the probe
    doubles as the one-time sync of a fresh worktree's environment."""
    try:
        proc = subprocess.run(
            [*python_cmd, "-c", f"import {module}"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            **child_priority_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _project_python(worktree: Path, module: str) -> tuple[list[str], bool] | None:
    """A command that runs the project's OWN python inside `worktree` —
    the interpreter whose environment actually holds the project's
    tooling — plus whether the worktree's code must be put on the path
    explicitly. None when no usable environment exists.

    Preference order: the worktree's synced .venv, then `uv run` (sync
    ALLOWED — a fresh worktree has no .venv, and `--no-sync` would create
    an empty skeleton from which uv silently falls back to whatever
    pytest/ruff sits on PATH, reporting environment artifacts as merge
    regressions), then the interpreter drcode itself runs under. The
    probed `module` (pytest, ruff, …) must really import there — a
    skeleton or partially synced venv falls through to the next option."""
    venv_python = _venv_python(worktree)
    if venv_python is not None and _module_importable(
        [str(venv_python)], module, worktree
    ):
        return [str(venv_python)], False
    if (
        shutil.which("uv") is not None
        and (worktree / "pyproject.toml").is_file()
        and _module_importable(["uv", "run", "python"], module, worktree)
    ):
        return ["uv", "run", "python"], False
    if _pytest_importable() and _module_importable([sys.executable], module, worktree):
        return [sys.executable], True
    return None


def _worktree_pythonpath_env(worktree: Path) -> dict[str, str]:
    """Environment for the drcode-interpreter fallback: the WORKTREE's
    code must be what the checks import, never drcode's own checkout
    (an editable install would otherwise shadow the resolved tree)."""
    env = dict(os.environ)
    src = worktree / "src"
    parts = [str(src if src.is_dir() else worktree)]
    if existing := env.get("PYTHONPATH"):
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _venv_prefix(python_cmd: list[str]) -> Path | None:
    """The virtualenv prefix of the plan's interpreter, when it IS a venv
    python. `uv run` plans are excluded: uv itself exports VIRTUAL_ENV and
    prepends the venv's bin to the child's PATH, so node-based discovery
    already lands correctly there."""
    if python_cmd and python_cmd[0] == "uv":
        return None
    bin_dir = Path(python_cmd[0]).parent
    if bin_dir.name not in ("bin", "Scripts"):
        return None
    prefix = bin_dir.parent
    if not (prefix / "pyvenv.cfg").is_file():
        return None
    return prefix


def _venv_env(python_cmd: list[str], base_env: dict[str, str] | None) -> dict[str, str]:
    """Subprocess environment for a tool that does its OWN python
    discovery — the pip `pyright` package is a node wrapper that ignores
    the interpreter `-m pyright` ran under and probes VIRTUAL_ENV / PATH
    instead. Pin the plan's environment explicitly (VIRTUAL_ENV set, the
    venv's bin prepended to PATH); otherwise that discovery can land on
    an interpreter without the project's deps and report phantom
    reportMissingImports for every third-party import (the 9d29e63d
    failure class)."""
    env = dict(base_env) if base_env is not None else dict(os.environ)
    if prefix := _venv_prefix(python_cmd):
        env["VIRTUAL_ENV"] = str(prefix)
        bin_dir = prefix / ("Scripts" if os.name == "nt" else "bin")
        env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])
    return env


def _python_files(worktree: Path) -> list[str]:
    """Tracked .py files of the worktree (git ls-files — the worktree may
    sit on a branch with different content than the main checkout)."""
    code, out = _git("ls-files", "*.py", cwd=worktree)
    if code != 0:
        return []
    return [line for line in out.splitlines() if line.strip()]


def compile_check(worktree: Path) -> CheckResult:
    """The MANDATORY compile check: every tracked .py file must
    byte-compile. Skipped (best-effort) only when the tree tracks no
    Python files at all."""
    py_files = _python_files(worktree)
    if not py_files:
        return CheckResult(
            name="compile (py_compile of tracked .py files)",
            ok=True,
            skipped=True,
            output="no tracked .py files",
        )
    # Chunk the compile check: a single command line with thousands of
    # paths can overflow the OS arg limit.
    chunk = 400
    ok = True
    output: list[str] = []
    for i in range(0, len(py_files), chunk):
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", *py_files[i : i + chunk]],
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
            **child_priority_kwargs(),
        )
        if proc.returncode != 0:
            ok = False
            output.append((proc.stdout + proc.stderr).strip()[:4000])
    return CheckResult(
        name="compile (py_compile of tracked .py files)",
        ok=ok,
        output="\n".join(output)[:8000],
    )


def ruff_check(worktree: Path) -> CheckResult:
    """The best-effort ruff check — MUST come from the project's own
    environment: a random PATH binary carries a different default
    ruleset/version and reports phantom failures (e.g. DTZ005 from an old
    ruff) that roll back a correct resolution. Skipped when the project
    has no ruff."""
    plan = _project_python(worktree, "ruff")
    if plan is None:
        return CheckResult(name="ruff check", ok=True, skipped=True)
    cmd, fix_path = plan
    return _run_checked(
        "ruff check",
        [*cmd, "-m", "ruff", "check", "."],
        worktree,
        # Same pinning as mechanical_checks: tools with their own
        # environment discovery must not land on a foreign interpreter.
        env=_venv_env(cmd, _worktree_pythonpath_env(worktree) if fix_path else None),
    )


def pyright_check(worktree: Path) -> CheckResult:
    """The best-effort pyright check, from the project's own environment
    (same reasoning as ruff). Skipped when the project has none."""
    plan = _project_python(worktree, "pyright")
    if plan is None:
        return CheckResult(name="pyright", ok=True, skipped=True)
    cmd, fix_path = plan
    return _run_checked(
        "pyright",
        [*cmd, "-m", "pyright"],
        worktree,
        timeout=900,
        # Node pyright discovers the environment itself — pin it to the
        # plan's venv (see _venv_env), same as inside mechanical_checks.
        env=_venv_env(cmd, _worktree_pythonpath_env(worktree) if fix_path else None),
    )


def mechanical_checks(
    worktree: Path, paths: list[str] | None = None
) -> list[CheckResult]:
    """Level 1 of the funnel: the tree must at least be sound. The
    conflict-marker scan runs FIRST (a prose-only resolution — the
    a6a6df25 class — must fail here, not after the test suite); the
    compile check is MANDATORY (every file git tracks must byte-compile);
    ruff/pyright run when the tool is available (best-effort — a project
    without them installed is not failed for it). `paths` limits the
    marker scan to the conflicted set; None scans the unmerged index
    entries (or skips when no merge is in progress)."""
    results: list[CheckResult] = [conflict_marker_check(worktree, paths)]
    py_files = _python_files(worktree)
    if py_files:
        # Chunk the compile check: a single command line with thousands of
        # paths can overflow the OS arg limit.
        chunk = 400
        ok = True
        output: list[str] = []
        for i in range(0, len(py_files), chunk):
            proc = subprocess.run(
                [sys.executable, "-m", "py_compile", *py_files[i : i + chunk]],
                cwd=worktree,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
                check=False,
                **child_priority_kwargs(),
            )
            if proc.returncode != 0:
                ok = False
                output.append((proc.stdout + proc.stderr).strip()[:4000])
        results.append(
            CheckResult(
                name="compile (py_compile of tracked .py files)",
                ok=ok,
                output="\n".join(output)[:8000],
            )
        )
    # ruff/pyright MUST come from the project's own environment: a random
    # PATH binary carries a different default ruleset/version and reports
    # phantom failures (e.g. DTZ005 from an old ruff) that roll back a
    # correct resolution. Skipped (best-effort) when the project has none.
    ruff_plan = _project_python(worktree, "ruff")
    if ruff_plan is not None:
        ruff_cmd, fix_path = ruff_plan
        results.append(
            _run_checked(
                "ruff check",
                [*ruff_cmd, "-m", "ruff", "check", "."],
                worktree,
                env=_worktree_pythonpath_env(worktree) if fix_path else None,
            )
        )
    else:
        results.append(CheckResult(name="ruff check", ok=True, skipped=True))
    pyright_plan = _project_python(worktree, "pyright")
    if pyright_plan is not None:
        pyright_cmd, fix_path = pyright_plan
        base_env = _worktree_pythonpath_env(worktree) if fix_path else None
        results.append(
            _run_checked(
                "pyright",
                [*pyright_cmd, "-m", "pyright"],
                worktree,
                timeout=900,
                # Node pyright discovers the environment itself — pin it
                # to the plan's venv (see _venv_env).
                env=_venv_env(pyright_cmd, base_env),
            )
        )
    else:
        results.append(CheckResult(name="pyright", ok=True, skipped=True))
    return results


def run_test_suite(worktree: Path, timeout: int = TEST_TIMEOUT) -> CheckResult:
    """Level 2 of the funnel: the project's test suite in the resolved
    tree, run with the project's OWN environment (`_project_python`) —
    never a PATH pytest that cannot even `import drcode`. The xdist
    fan-out is capped at 4 workers (a project whose addopts carry
    `-n auto` would otherwise pin every core). Skipped (never a failure)
    when the project has no tests directory or no usable pytest
    environment at all."""
    if not (worktree / "tests").is_dir():
        return CheckResult(
            name="test suite", ok=True, skipped=True, output="no tests/ directory"
        )
    plan = _project_python(worktree, "pytest")
    if plan is None:
        return CheckResult(
            name="test suite", ok=True, skipped=True, output="pytest unavailable"
        )
    python_cmd, fix_path = plan
    # No `-x`: one broken or flaky test at the head of the alphabet must
    # not mask the results of everything after it (the 10fccdfa incident
    # — a single early failure hid a real regression further down). The
    # full pytest output rides in the CheckResult and reaches the service
    # events via ResolverOutcome.summary().
    args = ["-m", "pytest", "-q"]
    if _module_importable(python_cmd, "xdist", worktree):
        args += ["-n", "4"]
    return _run_checked(
        "test suite (pytest -q)",
        [*python_cmd, *args],
        worktree,
        timeout=timeout,
        env=_worktree_pythonpath_env(worktree) if fix_path else None,
    )


def _pytest_importable() -> bool:
    try:
        import pytest  # noqa: F401
    except ImportError:
        return False
    return True


def failed_names(results: list[CheckResult]) -> list[str]:
    """The names of the checks that actually failed (a skipped check
    never counts — it reports the absence of tooling, not a failure)."""
    return [c.name for c in results if not c.ok and not c.skipped]


# -- manual verification report ----------------------------------------------------


def verify_report_path(project_root: Path, task_id: str) -> Path:
    """Where the manual verification report of `task_id` is persisted —
    next to the engine's triage report (`merge_engine.report_path`).
    Rewritten on every manual `v` run of the ConflictScreen; survives
    the screen being closed."""
    return merge_engine.report_path(project_root, task_id).with_name(
        f"{task_id}-verify.txt"
    )


def format_verify_report(checks: list[CheckResult], tests: CheckResult) -> str:
    """The human-readable full report of one manual verification run
    (the ConflictScreen's `v` funnel) — modeled on
    `ResolverOutcome.summary()`: the verdict, one status line per check,
    then every check's full captured output."""
    all_checks = [*checks, tests]
    ok = all(c.ok for c in all_checks)
    lines = [f"Conflict resolution verification: {'PASSED' if ok else 'FAILED'}"]
    failed = failed_names(all_checks)
    if failed:
        lines.append(f"failed checks: {', '.join(failed)}")
    for c in all_checks:
        lines.append("")
        lines.append(c.line)
        if c.output:
            lines.append(textwrap.indent(c.output, "    "))
    return "\n".join(lines) + "\n"


def save_verify_report(
    project_root: Path,
    task_id: str,
    checks: list[CheckResult],
    tests: CheckResult,
) -> Path | None:
    """Persist the verification report (atomically, like the engine's
    triage report); best-effort — None on failure."""
    path = verify_report_path(project_root, task_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".txt.tmp")
        tmp.write_text(format_verify_report(checks, tests), encoding="utf-8")
        tmp.replace(path)
        return path
    except OSError:
        return None


# -- check timings (ETA history for the interactive funnel) -------------------------


def check_timings_path(project_root: Path, task_id: str) -> Path:
    """Where the rolling per-check durations of `task_id` are persisted —
    next to the verify report (`<task-id>-verify-timings.json`). Keyed by
    the short stage key the interactive funnel uses (compile / ruff /
    pyright / pytest), valued in seconds."""
    return merge_engine.report_path(project_root, task_id).with_name(
        f"{task_id}-verify-timings.json"
    )


def _load_timings(path: Path) -> dict[str, float]:
    """The persisted rolling averages from one timings file; an empty
    dict (missing / corrupted file — skipped, never fatal) means no
    history yet."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in data.items()
        if isinstance(value, int | float) and value > 0
    }


def _save_timings(path: Path, timings: dict[str, float]) -> dict[str, float]:
    """Merge one run's measured durations into the persisted rolling
    averages of `path` (weight 0.7 old / 0.3 new) and write the file back
    atomically. Best-effort: any failure is swallowed — the ETA history
    is a UX nicety, never worth crashing over. Returns the merged
    averages."""
    try:
        old = _load_timings(path)
        merged = {
            **old,
            **{
                key: old.get(key, measured) * 0.7 + measured * 0.3
                for key, measured in timings.items()
                if measured > 0
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged), encoding="utf-8")
        tmp.replace(path)
        return merged
    except (OSError, ValueError):
        return {}


def load_check_timings(project_root: Path, task_id: str) -> dict[str, float]:
    """The persisted rolling averages of the task's past check durations;
    an empty dict (missing / corrupted file — skipped, never fatal) means
    no history yet and the funnel falls back to its defaults."""
    return _load_timings(check_timings_path(project_root, task_id))


def save_check_timings(
    project_root: Path, task_id: str, timings: dict[str, float]
) -> dict[str, float]:
    """Merge one run's measured per-check durations into the persisted
    rolling averages (weight 0.7 old / 0.3 new) and write the file back
    atomically. Best-effort: any failure is swallowed — the ETA history
    is a UX nicety, never worth crashing over. Returns the merged
    averages."""
    return _save_timings(check_timings_path(project_root, task_id), timings)


def prepare_timings_path(project_root: Path, task_id: str) -> Path:
    """Where the rolling per-STEP durations of the task's prepare phase
    are persisted — next to the triage report
    (`<task-id>-prepare-timings.json`, same format as the check
    timings). Keyed by the short step key the ConflictScreen's prepare
    panel uses (worktree / merge / triage / auto-resolve), valued in
    seconds."""
    return merge_engine.report_path(project_root, task_id).with_name(
        f"{task_id}-prepare-timings.json"
    )


def load_prepare_timings(project_root: Path, task_id: str) -> dict[str, float]:
    """The persisted rolling averages of the task's past prepare-step
    durations; empty when there is no history yet (the prepare panel
    falls back to its conservative defaults)."""
    return _load_timings(prepare_timings_path(project_root, task_id))


def save_prepare_timings(
    project_root: Path, task_id: str, timings: dict[str, float]
) -> dict[str, float]:
    """Merge one prepare run's measured per-step durations into the
    persisted rolling averages (same format and weighting as the check
    timings). Best-effort — a failure just drops the ETA history."""
    return _save_timings(prepare_timings_path(project_root, task_id), timings)


def review_diffs(
    worktree: Path, paths: list[str], cap: int = REVIEW_DIFF_CHAR_CAP
) -> tuple[str, str]:
    """The level-3 payloads: (diff(resolved, ours), diff(resolved,
    theirs)) limited to the conflicted paths. "ours" is HEAD (the task
    branch), "theirs" MERGE_HEAD (the base branch) of the in-progress
    merge. Each side is capped at `cap` chars with an explicit truncation
    marker — the reviewer must know when it sees a partial picture."""
    ours = _git("diff", "HEAD", "--", *paths, cwd=worktree)[1]
    theirs = _git("diff", "MERGE_HEAD", "--", *paths, cwd=worktree)[1]
    return (_cap(ours, cap), _cap(theirs, cap))


def _cap(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n… [truncated at {cap} chars — the review diff is partial]"


# -- outcome -----------------------------------------------------------------------


def conclude(worktree: Path, message: str) -> tuple[bool, str]:
    """Commit the verified resolution (`git commit` of the merge in
    progress). Refuses while unmerged paths remain — the caller must have
    verified first."""
    return merge_engine.finish(worktree, message)


def restore_conflict(
    project_root: Path, task_id: str, worktree: Path, base_branch: str
) -> bool:
    """Roll a FAILED attempt back to the PRISTINE conflicted state: abort
    the in-progress merge (the resolver's edits are discarded — a failed
    verification must not leave a half-resolved, half-verified tree
    behind) and re-run the reversed merge, which redeploys the conflict
    with fresh zdiff3 markers and a fresh triage. rerere has not recorded
    anything at this point (nothing was committed), so the redeploy is
    faithful. Returns True when the worktree again holds the parked
    conflict."""
    if merge_engine.in_progress(worktree):
        merge_engine.abort(worktree)
    report = prepare(project_root, task_id, worktree, base_branch)
    return bool(report.ok and report.conflicted)


@dataclass
class ResolverOutcome:
    """The full, human-readable funnel report the pipeline logs into the
    task's service events (both on success and on failure)."""

    verdict: str  # "resolved" | "needs-human"
    agent_resolved: bool = False
    agent_report: str = ""
    checks: list[CheckResult] = field(default_factory=list)
    tests: CheckResult | None = None
    review_verdict: str = ""  # "intact" | "lost" | "unsure" | "" (not run)
    review_report: str = ""
    committed: bool = False

    def summary(self) -> str:
        lines = [f"Conflict resolution: {self.verdict}"]
        # When the funnel rejected the attempt, the agent's "resolved"
        # claim must not read as success on its own (the a6a6df25
        # incident: an optimistic weave description right above the
        # FAILED line) — couple the claim to its rejection explicitly.
        rejected = self.agent_resolved and self.verdict != "resolved"
        lines.append(
            "resolver agent: "
            + ("resolved" if self.agent_resolved else "did NOT resolve")
            + (" (claim REJECTED by the verification below)" if rejected else "")
        )
        if self.agent_report:
            lines.append(f"resolver report:\n{self.agent_report[:4000]}")
        for c in self.checks:
            lines.append(c.line)
        if self.tests is not None:
            lines.append(self.tests.line)
            # A bare "test suite: FAILED" is undiagnosable in the service
            # events (the 10fccdfa incident: the tree had to be recovered
            # from dangling git objects to see WHICH test failed) — always
            # carry the pytest output of a failed run.
            if not self.tests.ok and not self.tests.skipped and self.tests.output:
                lines.append(f"test output:\n{self.tests.output}")
        if self.review_verdict:
            lines.append(f"intent review: {self.review_verdict}")
        if self.review_report:
            lines.append(f"review report:\n{self.review_report[:4000]}")
        lines.append("committed" if self.committed else "NOTHING committed")
        return "\n".join(lines)
