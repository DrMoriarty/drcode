"""Shared constants and helpers of the `drcode.task_env` package.

Everything here is import-time-only: the two mixin modules
(`drcode.task_env.lifecycle`, `drcode.task_env.branch_state`) and the
facade (`drcode.task_env`) all depend on this leaf, never on each other's
constants — the package stays free of import cycles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Optional service-event sink wired by the app: `(task_id, text)`. The
# worktree environment calls it when something happens the user must know
# about (a snapshot commit, a branch kept against deletion) but cannot log
# into the task's own event history itself — on the delete/retention paths
# the task is already gone from the manager when the notice fires.
EventLog = Callable[[str, str], None]

# Where the per-task worktrees live (inside the already-gitignored private
# state directory of the project).
WORKTREES_SUBDIR = Path(".drcode") / "worktrees"

# Branch namespace of the per-task worktrees.
BRANCH_PREFIX = "drcode/"

# Git worktrees can be registered but missing on disk (a hand-deleted
# directory) and directories can exist without a registration — both are
# handled by ensure_root's fallbacks (drcode.task_env.lifecycle).

# Dependency/cache directories never copied into a fresh worktree (see the
# package docstring): re-create them there with the package manager instead
# of dragging gigabytes through a file-by-file copy.
_COPY_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".drcode",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".eggs",
        ".hatch",
        ".cache",
        "dist",
        "build",
    }
)

# Hard cap on the file copy into a fresh worktree (beyond it the copy is
# truncated and the pipeline reports it).
_COPY_FILE_LIMIT = 5000


def _copy_skipped(rel: str) -> bool:
    """Should this relative (POSIX) path stay out of the untracked-file
    copy? The skip directories match at ANY path depth — the old
    first-component-only check let nested caches through (`src/**/__pycache__/
    *.pyc`, a `node_modules` buried in another tool's state directory), which
    is where the multi-second activations came from. `*.pyc` files are
    skipped anywhere too (bytecode regenerates, and cold-cache copies of it
    are disproportionately slow)."""
    if rel.endswith(".pyc"):
        return True
    return any(part in _COPY_SKIP_DIRS for part in PurePosixPath(rel).parts)


def branch_for(task_id: str) -> str:
    """The dedicated branch of one task's worktree."""
    return f"{BRANCH_PREFIX}{task_id}"


@dataclass(frozen=True)
class _StartPoint:
    """The resolved start point of a NEW task branch (see
    `TaskEnvironment._start_point`): the ref (branch name — never a raw
    SHA, so the created branch always forks from a ref the user can name)
    and its full SHA for the service-event record. `warning` is a
    human-readable anomaly notice (e.g. the configured start branch does
    not exist and the auto-detected HEAD branch was used instead)."""

    ref: str
    sha: str
    warning: str | None = None
