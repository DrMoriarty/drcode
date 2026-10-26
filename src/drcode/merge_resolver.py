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
  verification funnel (compile/ruff/pyright; the project's test suite);
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

import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from . import merge_engine
from .merge_engine import MergeReport

__all__ = [
    "REVIEW_DIFF_CHAR_CAP",
    "TEST_TIMEOUT",
    "CheckResult",
    "build_agent_context",
    "conclude",
    "failed_names",
    "format_verify_report",
    "mechanical_checks",
    "prepare",
    "restore_conflict",
    "review_diffs",
    "run_test_suite",
    "save_verify_report",
    "verify_report_path",
]

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
) -> MergeReport:
    """Deploy (or re-detect) the conflicted merge in the task's worktree.

    Runs the engine's reversed merge (`merge_base_into_worktree`): a
    merge already parked there is re-triaged in place (the engine handles
    `resuming`), a fresh one is attempted with the safe auto-resolvers
    switched on. Returns the report; the caller checks `report.ok` /
    `report.conflicted` — a non-conflicted outcome means there is nothing
    for the agent to do (clean merge, already up to date, or the engine
    refused).
    """
    return merge_engine.merge_base_into_worktree(
        project_root, task_id, worktree, base_branch, save=True
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
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, f"{type(e).__name__}: {e}"
    return proc.returncode, (proc.stdout + proc.stderr).strip()


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
    name: str, cmd: list[str], cwd: Path, timeout: int = _TIMEOUT
) -> CheckResult:
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return CheckResult(name=name, ok=False, output=f"{type(e).__name__}: {e}")
    out = (proc.stdout + "\n" + proc.stderr).strip()
    return CheckResult(name=name, ok=proc.returncode == 0, output=out[:8000])


def _python_files(worktree: Path) -> list[str]:
    """Tracked .py files of the worktree (git ls-files — the worktree may
    sit on a branch with different content than the main checkout)."""
    code, out = _git("ls-files", "*.py", cwd=worktree)
    if code != 0:
        return []
    return [line for line in out.splitlines() if line.strip()]


def mechanical_checks(worktree: Path) -> list[CheckResult]:
    """Level 1 of the funnel: the tree must at least be sound. The
    compile check is MANDATORY (every file git tracks must byte-compile);
    ruff/pyright run when the tool is available (best-effort — a project
    without them installed is not failed for it)."""
    results: list[CheckResult] = []
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
    if shutil.which("ruff") is not None:
        results.append(_run_checked("ruff check", ["ruff", "check", "."], worktree))
    else:
        results.append(CheckResult(name="ruff check", ok=True, skipped=True))
    if shutil.which("pyright") is not None:
        results.append(_run_checked("pyright", ["pyright"], worktree, timeout=900))
    else:
        results.append(CheckResult(name="pyright", ok=True, skipped=True))
    return results


def run_test_suite(worktree: Path, timeout: int = TEST_TIMEOUT) -> CheckResult:
    """Level 2 of the funnel: the project's test suite in the resolved
    tree. `uv run pytest` when uv is available (it provisions the
    project's own environment), the current interpreter's pytest as the
    fallback. Skipped (never a failure) when the project has no tests
    directory or no way to run pytest at all."""
    if not (worktree / "tests").is_dir():
        return CheckResult(
            name="test suite", ok=True, skipped=True, output="no tests/ directory"
        )
    if shutil.which("uv") is not None:
        cmd = ["uv", "run", "--no-sync", "pytest", "-q", "-x"]
    elif _pytest_importable():
        cmd = [sys.executable, "-m", "pytest", "-q", "-x"]
    else:
        return CheckResult(
            name="test suite", ok=True, skipped=True, output="pytest unavailable"
        )
    return _run_checked("test suite (pytest -q -x)", cmd, worktree, timeout=timeout)


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
        lines.append(
            "resolver agent: "
            + ("resolved" if self.agent_resolved else "did NOT resolve")
        )
        if self.agent_report:
            lines.append(f"resolver report:\n{self.agent_report[:4000]}")
        for c in self.checks:
            lines.append(c.line)
        if self.tests is not None:
            lines.append(self.tests.line)
        if self.review_verdict:
            lines.append(f"intent review: {self.review_verdict}")
        if self.review_report:
            lines.append(f"review report:\n{self.review_report[:4000]}")
        lines.append("committed" if self.committed else "NOTHING committed")
        return "\n".join(lines)
