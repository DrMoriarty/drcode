"""Optional per-task isolation via git worktrees (`drcode.task_env`).

The default ("everything in one tree") mode has every task work directly
in the project directory: an interrupted (unfinished) task leaves its
uncommitted changes behind, and the next task in the queue starts on top
of them. With the `task_isolation: worktree` project setting
(`drcode.settings`) each task instead gets its OWN working tree: on
activation the facade creates

    git worktree add .drcode/worktrees/<task-id> -b drcode/<task-id>

(the `.drcode/worktrees/` directory is gitignored — see `drcode.gitignore`)
and that path becomes the agent's project root: the filesystem tools are
jailed to it, `{project_path}` in the system prompts points at it and the
tester commits (TESTING_COMMIT finalization) land on the task's own
branch. An interrupted task simply keeps its worktree — the next task is
activated in its own, clean one. That is the point of the feature.

Lifecycle contract (enforced here and by the pipeline):

- activation (`ensure_root`): the worktree is created when missing
  (also after a manual deletion or a restart); an existing worktree —
  including an unfinished task's dirty one — is reused as-is.
- unfinished / re-queue: the worktree is PRESERVED (nothing calls
  remove); the task resumes in its own snapshot.
- completion (tester PASSED, research done): the work and the worktree
  STAY on the branch `drcode/<task-id>`. drcode deliberately does NOT
  merge back into the user's base branch on its own: merging is a team
  decision, so it is left to the project's `.drcode/workflow.py`
  `on_accepted` hook (which receives the worktree path and branch in the
  payload — see `drcode.workflow`). The worktree directory lives as long
  as the task does; it is removed when the task is deleted (including
  the retention auto-cleanup).
- cancel (Ctrl+X) / delete (Ctrl+Delete): worktree removed, branch
  `drcode/<task-id>` deleted (wired through TaskManager's removal
  listeners).
- startup: worktree directories and `drcode/*` branches without a live
  task (crashed session, deleted task) are cleaned up
  (`cleanup_orphans`).

Files git does not track are not part of a worktree checkout. To keep
`.env`-style local configuration and ignored artifacts available to the
agent, creating a worktree copies the untracked/ignored files of the
main tree (``git ls-files --others …``) into it, with two documented
limitations: heavy dependency/cache directories (see `_COPY_SKIP_DIRS`:
`.venv`, `node_modules`, …) are skipped — recreate them inside the
worktree with your package manager — and the copy stops at
`_COPY_FILE_LIMIT` files (a pathological tree must not stall the
activation for minutes); both are reported into the task's service-event
log by the pipeline when they bite.

Workflow hooks (`.drcode/workflow.py`): with isolation on, every hook
runs INSIDE the task's worktree (cwd), and the payload carries the
actual `worktree` path/branch — `on_task_start`'s branch checkout and
friends operate on the task's branch, not the user's checkout (see
`drcode.workflow`).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .gitignore import _is_git_repo
from .settings import ProjectSettings, TaskIsolation

# Where the per-task worktrees live (inside the already-gitignored private
# state directory of the project).
WORKTREES_SUBDIR = Path(".drcode") / "worktrees"

# Branch namespace of the per-task worktrees.
BRANCH_PREFIX = "drcode/"

# Git worktrees can be registered but missing on disk (a hand-deleted
# directory) and directories can exist without a registration — both are
# handled by ensure_root's fallbacks below.

# Dependency/cache directories never copied into a fresh worktree (see the
# module docstring): re-create them there with the package manager instead
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


def branch_for(task_id: str) -> str:
    """The dedicated branch of one task's worktree."""
    return f"{BRANCH_PREFIX}{task_id}"


class TaskEnvironment:
    """Worktree lifecycle for one project (the root the agent works in).

    Holds the LIVE project settings instance: the user can toggle
    `task_isolation` in the Settings dialog at any time, and the very next
    activation honours it without rebuilding anything.
    """

    def __init__(self, project_root: Path, settings: ProjectSettings) -> None:
        self.project_root = project_root
        self.settings = settings

    # -- state ---------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """The user asked for worktree isolation (regardless of git)."""
        return self.settings.task_isolation is TaskIsolation.WORKTREE

    @property
    def available(self) -> bool:
        """Isolation can actually work here: the project is a git repo."""
        return _is_git_repo(self.project_root)

    @property
    def active(self) -> bool:
        """Isolation is on AND usable: tasks get their own worktrees."""
        return self.enabled and self.available

    def worktree_path(self, task_id: str) -> Path:
        """Where this task's worktree lives (exists or not)."""
        return self.project_root / WORKTREES_SUBDIR / task_id

    def branch(self, task_id: str) -> str:
        return branch_for(task_id)

    def worktree_info(self, task_id: str) -> dict[str, str] | None:
        """The `worktree` entry for a workflow-hook payload, or None when
        the task is not isolated right now."""
        if not self.active:
            return None
        wt = self.worktree_path(task_id)
        if not wt.is_dir():
            return None
        return {"path": str(wt), "branch": self.branch(task_id)}

    # -- git plumbing (sync subprocesses; callers wrap in to_thread) ----------

    def _git(self, *args: str) -> tuple[int, str]:
        """Run git in the project root; returns (returncode, combined output)."""
        proc = subprocess.run(
            ["git", *args],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        out = (proc.stdout + proc.stderr).strip()
        return proc.returncode, out

    def _branch_exists(self, branch: str) -> bool:
        code, _ = self._git("rev-parse", "--verify", "--quiet", branch)
        return code == 0

    # -- lifecycle -------------------------------------------------------------

    def root_now(self, task_id: str) -> Path:
        """The task's working root WITHOUT creating anything: its worktree
        when isolation is on and the worktree exists, the project root
        otherwise. Cheap — safe to call from sync context."""
        if self.active:
            wt = self.worktree_path(task_id)
            if wt.is_dir():
                return wt
        return self.project_root

    def ensure_root(self, task_id: str) -> tuple[Path, str]:
        """The task's working root, with the worktree CREATED when missing.

        Returns ``(root, note)``: ``note`` is empty on success, otherwise a
        human-readable reason the task fell back to the project root (the
        pipeline logs it into the service-event history). Idempotent: an
        existing worktree — dirty, mid-work, resumed after a restart — is
        reused untouched; a branch left over from a removed worktree is
        re-checked-out so earlier commits survive a manual worktree
        deletion.

        Blocking (git + the untracked-file copy): call via
        `asyncio.to_thread` from the pipeline.
        """
        if not self.enabled:
            # Isolation off: the task works in the project directory. An
            # existing worktree of this task (left over from a session
            # where the setting was on) is NOT reused silently — removing
            # it is `remove`/`cleanup_orphans`' job.
            return self.project_root, ""
        if not self.available:
            if self.enabled:
                # The persisted-but-dormant mode (worktree without a git
                # repo): the app warns at startup; each activation says so
                # too, once, into the task's service-event log.
                return self.project_root, (
                    "the project has no git repository — tasks run in the "
                    "project directory until one exists (or the setting is "
                    "turned off)"
                )
            return self.project_root, ""
        if not self.enabled:
            # Isolation is off (the default): no worktree is ever created —
            # the task runs in the project directory, silently.
            return self.project_root, ""
        wt = self.worktree_path(task_id)
        branch = self.branch(task_id)
        # Stale registrations (directory removed by hand) make `add` fail:
        # prune them first — it never touches healthy worktrees.
        self._git("worktree", "prune")
        wt.parent.mkdir(parents=True, exist_ok=True)
        if wt.is_dir() and wt.iterdir():
            # A non-empty directory: ours, created by an earlier activation.
            return wt, ""
        if self._branch_exists(branch):
            code, out = self._git("worktree", "add", str(wt), branch)
        else:
            code, out = self._git("worktree", "add", str(wt), "-b", branch)
        if code != 0:
            # Last resort: a broken leftover directory blocks `add` —
            # remove it (it is drcode's own private state) and retry.
            shutil.rmtree(wt, ignore_errors=True)
            if self._branch_exists(branch):
                code, out = self._git("worktree", "add", str(wt), branch)
            else:
                code, out = self._git("worktree", "add", str(wt), "-b", branch)
            if code != 0:
                return self.project_root, (
                    f"git worktree add failed, the task falls back to the "
                    f"project directory: {out[:500]}"
                )
        note = self._copy_untracked_files(wt)
        return wt, note

    def _collect_untracked(self) -> list[str]:
        """Untracked (plain + gitignored) files of the main tree, relative
        POSIX paths. The private `.drcode/` state and the heavy dependency
        directories are excluded (see module docstring)."""
        files: dict[str, None] = {}
        for extra in (["--ignored"], []):
            code, out = self._git(
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                *extra,
            )
            if code != 0:
                continue
            for raw in out.split("\0"):
                rel = raw.strip()
                if not rel or rel in files:
                    continue
                first = Path(rel).parts[0] if Path(rel).parts else ""
                if first in _COPY_SKIP_DIRS:
                    continue
                files[rel] = None
        return list(files)

    def _copy_untracked_files(self, wt: Path) -> str:
        """Carry the main tree's untracked/ignored files into a fresh
        worktree (best effort, capped — see module docstring). Returns a
        note for the service-event log (empty when everything fit)."""
        try:
            files = self._collect_untracked()
        except (OSError, subprocess.SubprocessError) as e:
            return f"untracked-file copy failed: {type(e).__name__}: {e}"
        truncated = len(files) > _COPY_FILE_LIMIT
        copied = 0
        try:
            for rel in files[:_COPY_FILE_LIMIT]:
                src = self.project_root / rel
                dst = wt / rel
                if not src.is_file():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
        except OSError as e:
            return f"untracked-file copy stopped early ({copied} files copied): {e}"
        if truncated:
            return (
                f"untracked-file copy truncated at {_COPY_FILE_LIMIT} files "
                f"(of {len(files)}) — the rest stays in the project directory"
            )
        return ""

    def remove(self, task_id: str) -> None:
        """Remove the task's worktree and its `drcode/<task-id>` branch.

        Called on cancel (Ctrl+X) and delete (Ctrl+Delete) through
        TaskManager's removal listeners — and therefore from sync context:
        best-effort, never raises, fast (two git calls). A no-op when
        nothing exists; works regardless of the current setting so
        leftovers of a session where isolation was toggled off are still
        cleaned.
        """
        wt = self.worktree_path(task_id)
        branch = self.branch(task_id)
        try:
            self._git("worktree", "prune")
            if wt.is_dir():
                code, _ = self._git("worktree", "remove", "--force", str(wt))
                if code != 0:
                    shutil.rmtree(wt, ignore_errors=True)
            elif wt.exists():
                wt.rmdir()  # empty leftover dir
            if self._branch_exists(branch):
                self._git("branch", "-D", branch)
        except (OSError, subprocess.SubprocessError):
            pass

    def cleanup_orphans(self, live_task_ids: set[str]) -> list[str]:
        """Startup cleanup: worktree directories and `drcode/*` branches
        whose task no longer exists (crashed session, deleted task, the
        retention auto-cleanup). Returns the removed task ids.

        Worktrees/branches of LIVE tasks — including completed ones whose
        work must stay inspectable — are never touched. Blocking (git
        calls): run via `asyncio.to_thread` at app startup. Skipped
        entirely when there is nothing to sweep (not a git repo, or the
        setting is off and no worktree directory exists).
        """
        if not self.available:
            return []
        if not self.enabled and not (self.project_root / WORKTREES_SUBDIR).exists():
            return []
        removed: list[str] = []
        try:
            self._git("worktree", "prune")
            # Directory-based sweep: our worktrees live in one private dir.
            wt_root = self.project_root / WORKTREES_SUBDIR
            if wt_root.is_dir():
                for entry in wt_root.iterdir():
                    if entry.name in live_task_ids:
                        continue
                    code, _ = self._git("worktree", "remove", "--force", str(entry))
                    if code != 0:
                        shutil.rmtree(entry, ignore_errors=True)
                    removed.append(entry.name)
            elif wt_root.exists():
                wt_root.rmdir()
            # Branch sweep: `drcode/<task-id>` without a live task — e.g.
            # the branch of a cancelled task whose worktree removal never
            # ran — is deleted; a branch still checked out in a registered
            # worktree is left alone (git refuses to delete it anyway).
            code, out = self._git(
                "for-each-ref", "--format=%(refname:short)", "refs/heads"
            )
            if code == 0:
                for branch in out.splitlines():
                    branch = branch.strip()
                    if not branch.startswith(BRANCH_PREFIX):
                        continue
                    task_id = branch[len(BRANCH_PREFIX) :]
                    if task_id in live_task_ids:
                        continue
                    # A branch still checked out in some worktree makes -D
                    # fail — ignored, the worktree sweep above has already
                    # deregistered ours.
                    self._git("branch", "-D", branch)
                    if task_id not in removed:
                        removed.append(task_id)
        except (OSError, subprocess.SubprocessError):
            pass
        return removed
