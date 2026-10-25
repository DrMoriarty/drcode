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
  decision, so it is left to the project's `.drcode/workflow.py` accept
  hook — `on_accepted` for development tasks, `on_research_accepted`
  for research (each receives the worktree path and branch in the
  payload — see `drcode.workflow`). When the finalized task's branch
  still holds commits the main checkout does not have, the pipeline
  reminds the user explicitly (`unmerged_commits`) — a PASSED verdict
  never silently assumes the fix reached their checkout. The worktree
  directory lives as long as the task does; it is removed when the task
  is deleted (including the retention auto-cleanup).
- cancel (Ctrl+X) / delete (Ctrl+Delete): the worktree is removed; the
  branch `drcode/<task-id>` is deleted ONLY when it holds no unmerged
  commits (see the user-facing contract below). Before every removal a
  final snapshot commits a dirty tree onto the branch, so no verdict /
  cancel / purge can strand uncommitted work in a directory that is
  about to disappear (wired through TaskManager's removal listeners).
- startup: worktree directories and `drcode/*` branches without a live
  task (crashed session, deleted task) are cleaned up
  (`cleanup_orphans`) — under the same two rules: a dirty tree is
  snapshotted first, a branch with unmerged commits is kept.

User-facing contract:

    Every task leaves behind a committed branch `drcode/<task-id>`; the
    worktree directory is disposable and can be removed at any time
    without loss. drcode never deletes a branch containing unmerged
    commits and never merges it into the user's base branch itself —
    merging stays with `.drcode/workflow.py` `on_accepted` (research:
    `on_research_accepted`) / the user.

Two rules enforce it (both active only under worktree isolation —
without isolation everything lives in the shared main tree and nothing
here may change its behavior):

1. A verdict never leaves the task with a dirty tree: `snapshot()`
   commits the worktree's changes onto `drcode/<task-id>` whenever the
   task transitions out of active work (tester verdict, cancel) and
   before any removal.
2. A branch with unique commits is never deleted automatically: before
   every `branch -D`, the commits not reachable from the main checkout's
   current branch are counted; if there are any, the branch survives
   (the worktree directory is still removed — commits live in the repo
   object store, the directory is disposable).

Files git does not track are not part of a worktree checkout. To keep
`.env`-style local configuration and ignored artifacts available to the
agent, creating a worktree copies the untracked/ignored files of the
main tree (``git ls-files --others …``) into it, with two documented
limitations: heavy dependency/cache directories (see `_COPY_SKIP_DIRS`:
`.venv`, `node_modules`, …) are skipped at ANY path depth (as are `*.pyc`
files) — recreate them inside the worktree with your package manager — and the copy stops at
`_COPY_FILE_LIMIT` files (a pathological tree must not stall the
activation for minutes); both are reported into the task's service-event
log by the pipeline when they bite.

When the worktree DIRECTORY goes (project setting `worktree_removal`,
`drcode.settings`): `on_delete` (the default) — with the task, i.e. on
cancel (Ctrl+X), delete (Ctrl+Delete) and the retention purge, via
TaskManager's removal listeners; `on_completion` — as soon as the task
reaches COMPLETED (research right in `TaskManager.mark_done`, dev tasks
at the tester's `set_verification` verdict, the no-tester and
review-accept completions alike — every COMPLETED transition funnels
through the completion listeners the app wires to
`TaskEnvironment.on_completed`); `never` — never automatically, and
`remove()` then only anchors a dirty tree on the branch. In every mode
the branch itself is governed by RULE 2 alone: a completion never
touches it, and a removal deletes it only when it holds no unmerged
commits. Under `on_completion` a re-activated completed task (the user
sent new input) simply gets its worktree recreated from the surviving
branch by `ensure_root` — nothing is lost. `cleanup_orphans` IGNORES
the setting: directories of tasks that no longer exist are garbage from
deleted tasks (a crashed session, a manual delete under `never`), and
`never` would otherwise grow `.drcode/worktrees/` without bound.

Workflow hooks (`.drcode/workflow.py`): with isolation on, every hook
runs INSIDE the task's worktree (cwd), and the payload carries the
actual `worktree` path/branch — `on_task_start`'s branch checkout and
friends operate on the task's branch, not the user's checkout (see
`drcode.workflow`). `on_research_accepted` additionally receives the
`project` entry (main checkout path + base branch) so it can merge the
task's docs branch into the user's tree from inside the worktree.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from .gitignore import _is_git_repo
from .models import Task, TaskStatus
from .settings import ProjectSettings, TaskIsolation, WorktreeRemoval

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


class TaskEnvironment:
    """Worktree lifecycle for one project (the root the agent works in).

    Holds the LIVE project settings instance: the user can toggle
    `task_isolation` in the Settings dialog at any time, and the very next
    activation honours it without rebuilding anything.
    """

    def __init__(self, project_root: Path, settings: ProjectSettings) -> None:
        self.project_root = project_root
        self.settings = settings
        # Service-event sink (EventLog): set by the app; best-effort.
        self.event_log: EventLog | None = None
        # Serializes activations (ensure_root): with parallel development
        # slots several runs may activate concurrently, and their git
        # worktree prune/add calls must not interleave (`worktree prune`
        # racing `worktree add` can deregister a worktree another thread
        # is just creating).
        self._activation_lock = threading.Lock()
        # Mergedness cache for the batched `unmerged_ids` (see there):
        # `_unmerged_cache_fingerprint` is the hash of one `show-ref --head`
        # snapshot — HEAD plus every branch tip, i.e. everything the
        # per-branch mergedness answers depend on — and `_unmerged_cache`
        # maps each `drcode/<task-id>` branch name to its mergedness under
        # THAT snapshot. Any commit on any drcode branch or any movement of
        # the base (HEAD) changes the fingerprint and wipes the dict, so a
        # cache hit is always a faithful answer; entries are filled by one
        # batched `for-each-ref --no-merged` instead of a rev-list per
        # branch. Polled every 10–30 s by the header indicator and the
        # dependency gate — in the steady state that is a single cheap
        # `show-ref` per poll instead of ~3 git processes per task id.
        self._unmerged_cache_fingerprint: str | None = None
        self._unmerged_cache: dict[str, bool] = {}
        self._unmerged_cache_lock = threading.Lock()
        # Per-task removal coordination (see schedule_removal / wait_removal):
        # `_removal_locks` serializes the git-touching body of `remove`
        # against `ensure_root` of the same task (a background removal must
        # never race a re-activation's worktree creation), and
        # `_removal_pending` lets the pipeline wait until a scheduled
        # background removal has finished (or skipped) before the accept
        # hooks read the worktree state.
        self._removal_locks: dict[str, threading.RLock] = {}
        self._removal_locks_guard = threading.Lock()
        self._removal_cond = threading.Condition()
        self._removal_pending: set[str] = set()

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

    def _task_lock(self, task_id: str) -> threading.RLock:
        """The per-task serialization lock shared by `ensure_root`,
        `remove` and `snapshot` (lazily created; one small RLock per task
        id — reentrant because remove/cleanup_orphans snapshot from
        inside the lock)."""
        with self._removal_locks_guard:
            lock = self._removal_locks.get(task_id)
            if lock is None:
                lock = threading.RLock()
                self._removal_locks[task_id] = lock
            return lock

    def branch(self, task_id: str) -> str:
        return branch_for(task_id)

    def worktree_info(self, task_id: str) -> dict[str, str] | None:
        """The `worktree` entry for a workflow-hook payload, or None when
        the task is not isolated right now.

        Normally the entry carries both `path` and `branch`. When the
        worktree DIRECTORY is already gone but the branch survives —
        `worktree_removal: on_completion` removes it at the COMPLETED
        transition, while the accept hooks (`on_accepted` /
        `on_research_accepted`) run right AFTER it and the branch is
        kept whenever it holds unmerged commits (RULE 2) — the entry
        degrades to `branch` only: the accept-merge hook needs nothing
        but the branch name (it merges into the main checkout anyway).
        A deleted branch (nothing unmerged to merge) is None."""
        if not self.active:
            return None
        wt = self.worktree_path(task_id)
        if wt.is_dir():
            return {"path": str(wt), "branch": self.branch(task_id)}
        branch = self.branch(task_id)
        if not self._branch_exists(branch):
            return None
        return {"branch": branch}

    # -- git plumbing (sync subprocesses; callers wrap in to_thread) ----------

    def _git(self, *args: str, cwd: Path | None = None) -> tuple[int, str]:
        """Run git in the project root (or `cwd`, e.g. a task's worktree);
        returns (returncode, combined output)."""
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd if cwd is not None else self.project_root,
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

    # -- verified state of the shared checkout ---------------------------------

    def branch_tip(self, task_id: str) -> str | None:
        """Full SHA of the task's `drcode/<id>` branch tip (the commit the
        PASSED verdict is stamped on — the only task state that survives
        the worktree removal), or None when isolation is off or the branch
        is gone. Blocking git — callers wrap in to_thread."""
        if not self.active:
            return None
        code, out = self._git("rev-parse", "--verify", "--quiet", self.branch(task_id))
        return out if code == 0 and out else None

    def head_commit(self) -> str | None:
        """Full SHA of the MAIN checkout's HEAD, or None outside a
        repository (without isolation the shared checkout IS the verified
        tree, so HEAD is what a hook should pin). Blocking git — callers
        wrap in to_thread."""
        code, out = self._git("rev-parse", "HEAD")
        return out if code == 0 and out else None

    def main_checkout_dirty(self) -> bool:
        """Uncommitted TRACKED changes in the main checkout
        (`git status --porcelain -uno`) — False outside a repository.
        Blocking git — callers wrap in to_thread."""
        code, out = self._git("status", "--porcelain", "-uno")
        return code == 0 and bool(out)

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
            # The persisted-but-dormant mode (worktree without a git
            # repo): the app warns at startup; each activation says so
            # too, once, into the task's service-event log.
            return self.project_root, (
                "the project has no git repository — tasks run in the "
                "project directory until one exists (or the setting is "
                "turned off)"
            )
        # One activation at a time: the prune/add sequence below is not
        # race-free against a concurrent activation of another task (the
        # parallel development slots dispatch several ensure_root calls
        # in flight at once). The per-task lock additionally keeps this
        # off the toes of a background removal of the SAME task
        # (schedule_removal): the removal finishes before a worktree is
        # (re-)created here — or, when this activation got here first,
        # the removal's re-check sees the task alive again and skips.
        with self._task_lock(task_id), self._activation_lock:
            return self._ensure_root_locked(task_id)

    def _ensure_root_locked(self, task_id: str) -> tuple[Path, str]:
        """The git-touching body of ensure_root, run under the activation
        lock (see __init__)."""
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
        POSIX paths, sorted. The private `.drcode/` state and the heavy
        dependency/cache directories — at ANY path depth — are excluded
        (see module docstring).

        Enumeration is a plain `os.walk` that prunes the skip directories
        in place (never descends into them, and into nested git
        repositories / submodules) minus git's tracked-file listing —
        which reads the index and never touches the working tree. That
        avoids materializing git's full untracked list only to filter it
        afterwards in Python: a big ignored `.venv` alone is tens of
        thousands of entries, and the enumeration cost was paid on EVERY
        activation. When git cannot even produce the tracked listing
        (broken repository), the git-only fallback below keeps the copy
        working — just slower."""
        code, out = self._git("ls-files", "-z")
        if code != 0:
            return self._collect_untracked_git()
        tracked = {rel for rel in (raw.strip() for raw in out.split("\0")) if rel}
        root = self.project_root
        files: dict[str, None] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            here = Path(dirpath)
            # In-place prune: os.walk never enters a name dropped here.
            # A directory with its own `.git` entry (a submodule, a nested
            # repository) belongs to that repository — git never listed
            # its contents as "others" either.
            dirnames[:] = [
                d
                for d in dirnames
                if d not in _COPY_SKIP_DIRS and not (here / d / ".git").exists()
            ]
            for name in filenames:
                rel = (here / name).relative_to(root).as_posix()
                if rel in tracked or _copy_skipped(rel):
                    continue
                files[rel] = None
        return sorted(files)

    def _collect_untracked_git(self) -> list[str]:
        """The git-only enumeration (fallback when the tracked listing is
        unavailable): two `ls-files --others` passes (gitignored + plain)
        with the same any-depth skip predicate applied afterwards."""
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
                if not rel or rel in files or _copy_skipped(rel):
                    continue
                files[rel] = None
        return sorted(files)

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

    def snapshot(self, task_id: str, note: str = "") -> str | None:
        """Commit every change in the task's worktree onto its branch
        (RULE 1: a verdict never leaves the task with a dirty tree).

        Runs `git add -A -- . ':(exclude).drcode'` + `git commit` inside
        the task's worktree, so the work of a task that just left active
        work (tester verdict, cancel, removal) is anchored on the branch
        `drcode/<task-id>` — the worktree directory itself is disposable
        afterwards. When the tester already committed through its
        git_commit tool (or the tree is simply clean) the snapshot is a
        no-op.

        Returns a human-readable result for the service-event log —
        either "snapshot committed on <branch> @ <sha>" or "worktree
        clean, nothing to snapshot". None is a true no-op: isolation is
        off / not a git repo / the worktree does not exist / git failed.

        This is pipeline-level bookkeeping, NOT model work: blocking git
        calls — run via asyncio.to_thread from the pipeline/listeners.
        """
        if not self.active:
            return None
        wt = self.worktree_path(task_id)
        if not wt.is_dir():
            return None
        # Serialized per task against remove/ensure_root (a background
        # completion removal must not delete the tree mid-commit); the
        # RLock makes the nested snapshot from remove/cleanup_orphans a
        # no-op re-entry.
        with self._task_lock(task_id):
            return self._snapshot_locked(task_id, wt, note)

    def _snapshot_locked(self, task_id: str, wt: Path, note: str) -> str | None:
        branch = self.branch(task_id)
        try:
            code, out = self._git(
                "status", "--porcelain", "--", ".", ":(exclude).drcode", cwd=wt
            )
            if code != 0:
                return None
            if not out.strip():
                return "worktree clean, nothing to snapshot"
            code, _ = self._git("add", "-A", "--", ".", ":(exclude).drcode", cwd=wt)
            if code != 0:
                return None
            message = (
                f"drcode: snapshot of {task_id} ({note})"
                if note
                else f"drcode: snapshot of {task_id}"
            )
            code, _ = self._git("commit", "-m", message, cwd=wt)
            if code != 0:
                return None
            code, sha = self._git("rev-parse", "--short", "HEAD", cwd=wt)
            return f"snapshot committed on {branch} @ {sha if code == 0 else '?'}"
        except (OSError, subprocess.SubprocessError):
            return None

    def base_branch(self) -> str:
        """The main checkout's current branch — the base the unmerged
        commits of a task branch are counted against. Falls back to HEAD
        (detached HEAD, unusual setups): counting against whatever HEAD
        points at still protects the task's own commits."""
        code, out = self._git("symbolic-ref", "--short", "HEAD")
        if code == 0 and out:
            return out
        return "HEAD"

    def _unmerged_count(self, branch: str, base: str | None = None) -> int:
        """How many commits of `branch` are NOT reachable from the base
        branch (RULE 2's delete-protection counter). `base` — an already
        resolved `base_branch()` of the caller (remove / cleanup_orphans
        resolve it once per sweep, not per branch). 0 also on any git
        failure — when git cannot tell, the old delete behavior applies
        (the snapshot above is the real safety net)."""
        if base is None:
            base = self.base_branch()
        code, out = self._git("rev-list", "--count", f"{base}..{branch}")
        if code != 0:
            return 0
        try:
            return int(out or "0")
        except ValueError:
            return 0

    def unmerged_commits(self, task_id: str) -> int:
        """How many commits of the task's `drcode/<task-id>` branch are
        NOT reachable from the main checkout's current branch
        (`base_branch`) — the "is the task's work already merged?" check
        the pipeline runs when a passed task is finalized, so an unmerged
        fix is never left as a silent responsibility of the (absent)
        `on_accepted` hook. 0 when the branch does not exist (no
        isolation, a cancelled task, work already merged and the branch
        cleaned up) or git cannot tell. Blocking (a few git calls): run
        via `asyncio.to_thread` from the pipeline."""
        if not self.available:
            return 0
        branch = self.branch(task_id)
        if not self._branch_exists(branch):
            return 0
        return self._unmerged_count(branch)

    def unmerged_ids(self, task_ids: list[str]) -> set[str]:
        """Which of `task_ids` still have unmerged work: their
        `drcode/<id>` branch exists and holds commits not reachable from
        the base branch — the mergedness half of the dependency promotion
        gate (see drcode.task_manager / the pipeline's promote worker).

        Semantics per id: isolation not active (setting off, no git repo)
        → NOT in the set (nothing to merge — the gate degenerates to the
        dependency's status); branch absent (cancelled/cleaned/never
        isolated) → NOT in the set; branch present with unmerged commits →
        in the set — including ids whose task file is already gone (RULE 2:
        the branch outlives the task), so a deleted dependency with
        unmerged work keeps its dependents blocked. Checks are dynamic:
        the caller re-runs this at every promotion attempt, so a manual
        `git merge drcode/<dep>` unblocks the dependents by itself.

        Cost: batched, not per id. One `show-ref --head` snapshots HEAD
        plus every branch tip — both the existence of the `drcode/<id>`
        branches and a fingerprint of everything their mergedness depends
        on. Unchanged fingerprint → the cached per-branch answers are
        reused as-is (a single cheap git process per call, which is what
        the 10 s header poll and the 30 s dependency tick hit in the
        steady state). Otherwise ONE `for-each-ref --no-merged <base>`
        answers all the requested branches at once — no rev-parse, no
        symbolic-ref and no rev-list per id (the pre-optimization shape,
        kept as `_unmerged_ids_slow` for the fallback when git cannot
        even take the snapshot). Blocking (1–3 git calls total): run via
        `asyncio.to_thread` from the pipeline."""
        if not self.active:
            return set()
        ids = [task_id for task_id in dict.fromkeys(task_ids) if task_id]
        if not ids:
            return set()
        fingerprint, existing = self._branch_snapshot()
        if fingerprint is None:
            # Git could not answer wholesale (broken repository, empty
            # one): the slow per-id path degrades exactly like before.
            return self._unmerged_ids_slow(ids)
        with self._unmerged_cache_lock:
            if fingerprint != self._unmerged_cache_fingerprint:
                # Anything moved (a commit on a drcode branch, a merge
                # into the base, a branch added/removed): every cached
                # answer is stale — recompute from scratch below.
                self._unmerged_cache_fingerprint = fingerprint
                self._unmerged_cache = {}
            cache = self._unmerged_cache
            # The snapshot IS the existence check (no rev-parse per id):
            # absent branches are filtered out here and simply never get
            # a cache entry — they stay out of the answer via .get().
            asked = [b for b in (self.branch(t) for t in ids) if b in existing]
            unknown = [b for b in asked if b not in cache]
            if unknown:
                # One batch over exactly the branches with no cached
                # answer yet (a fresh fingerprint: all of `asked`).
                batch = self._unmerged_branches(unknown)
                cache.update((b, b in batch) for b in unknown)
            return {
                task_id for task_id in ids if cache.get(self.branch(task_id), False)
            }

    def drcode_branch_state(self) -> tuple[set[str], set[str]]:
        """`(existing, unmerged)` for EVERY `drcode/<task-id>` branch in
        one snapshot — the wholesale answer behind the header's
        unmerged-work indicator, the accept-hook merge audit and the
        sidebar's ✓ marker color. `existing` is the set of drcode branch
        names the snapshot saw (the white-vs-colored discriminator: a
        missing branch means isolation is off, there is no git repo, or
        the merged branch was already cleaned up); `unmerged` is the
        subset whose tip holds commits the base branch does not have.
        Unlike `unmerged_ids` this does not need the caller to know
        WHICH ids to ask: one `show-ref --head` plus one `for-each-ref
        --no-merged <base>` glob over `refs/heads/drcode/` answer every
        task branch at once — O(1) git processes, not per branch and not
        per completed task.

        Shares the `_unmerged_cache_fingerprint` mechanism with
        `unmerged_ids` (identical per-branch mergedness semantics, one
        `show-ref --head` snapshot per poll): whichever of the two timers
        (10 s indicator, 30 s dependency tick) runs first pays for the
        snapshot+glob, the other hits the warm cache. Answers for
        branches the snapshot did not see are never cached — a glob
        result is only trusted for branches existing under THAT
        fingerprint.

        Degradation: isolation off / no git repo → both sets empty; the
        snapshot unavailable (broken repository) → empty `existing` and
        the plain glob without caching for `unmerged`; the glob itself
        failing → empty `unmerged` ("merged" — the same degradation as
        the rev-list path). Blocking (2–3 cheap git calls cold, 1 warm):
        run via `asyncio.to_thread`."""
        if not self.active:
            return set(), set()
        fingerprint, existing = self._branch_snapshot()
        if fingerprint is None:
            # The wholesale snapshot could not be taken (broken/empty
            # repository): answer without caching — the glob degrades to
            # empty exactly like the rev-list path on git failure.
            return set(), self._unmerged_branches_glob()
        with self._unmerged_cache_lock:
            if fingerprint != self._unmerged_cache_fingerprint:
                # Anything moved (a commit on a drcode branch, a merge
                # into the base, a branch added/removed): every cached
                # answer is stale — recompute from scratch below.
                self._unmerged_cache_fingerprint = fingerprint
                self._unmerged_cache = {}
            cache = self._unmerged_cache
            unknown = [b for b in existing if b not in cache]
            if unknown:
                # The glob answers ALL drcode branches at once (unmerged
                # ones come back, merged ones do not) — fill the cache
                # for every branch the snapshot vouches for.
                batch = self._unmerged_branches_glob()
                cache.update((b, b in batch) for b in unknown)
            unmerged = {b for b in existing if cache.get(b, False)}
        return existing, unmerged

    def unmerged_drcode_branches(self) -> set[str]:
        """ALL `drcode/<task-id>` branch names whose tip holds commits
        the base branch does not have — the unmerged half of
        `drcode_branch_state()` (same batching, same cache, same
        degradation), kept as the shorthand for callers that only ask
        about mergedness. Blocking (2–3 cheap git calls cold, 1 warm):
        run via `asyncio.to_thread`."""
        return self.drcode_branch_state()[1]

    def _unmerged_branches_glob(self) -> set[str]:
        """Which `drcode/*` branches hold commits the base branch does
        not have — ONE `for-each-ref --no-merged <base>` over the
        prefix glob (no refspec per branch, so the cost does not grow
        with the branch count). Full short branch names, prefix
        included. Empty set on any git failure — when git cannot tell,
        the answer is "merged" (same as the rev-list path)."""
        try:
            code, out = self._git(
                "for-each-ref",
                "--format=%(refname)",
                "--no-merged",
                self.base_branch(),
                f"refs/heads/{BRANCH_PREFIX}",
            )
        except (OSError, subprocess.SubprocessError):
            return set()
        if code != 0:
            return set()
        prefix = "refs/heads/"
        return {
            line.strip()[len(prefix) :]
            for line in out.splitlines()
            if line.strip().startswith(prefix)
        }

    def _unmerged_ids_slow(self, ids: list[str]) -> set[str]:
        """The pre-optimization per-id loop: rev-parse + rev-list per
        branch, base resolved per call. Fallback only — for environments
        where the batch snapshot cannot be taken."""
        unmerged: set[str] = set()
        for task_id in ids:
            branch = self.branch(task_id)
            if not self._branch_exists(branch):
                continue
            if self._unmerged_count(branch) > 0:
                unmerged.add(task_id)
        return unmerged

    def _branch_snapshot(self) -> tuple[str | None, set[str]]:
        """One `show-ref --head` in the project root: a hash of the full
        ref dump (HEAD + every branch tip — the complete input of the
        mergedness answers) and the set of `drcode/<id>` branch names
        that exist. `(None, set())` on any git failure — the caller
        falls back to the per-id path."""
        try:
            code, out = self._git("show-ref", "--head")
        except (OSError, subprocess.SubprocessError):
            return None, set()
        if code != 0:
            return None, set()
        fingerprint = hashlib.sha256(out.encode()).hexdigest()
        existing: set[str] = set()
        prefix = f"refs/heads/{BRANCH_PREFIX}"
        for line in out.splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].startswith(prefix):
                continue
            branch = parts[1][len("refs/heads/") :]
            if branch:  # `drcode/<id>` itself, never a bare "drcode"
                existing.add(branch)
        return fingerprint, existing

    def _unmerged_branches(self, branches: list[str]) -> set[str]:
        """Which of the given `drcode/<id>` branch names hold commits the
        base branch does not have — ONE `for-each-ref --no-merged` for
        the whole batch (branch-tip reachability is exactly the
        `rev-list --count base..branch > 0` question). Only ids of the
        asked branches can come back: a glob-ish task id must not widen
        anyone else's answer. Empty set on any git failure — when git
        cannot tell, the answer is "merged" (same as the rev-list path)."""
        try:
            code, out = self._git(
                "for-each-ref",
                "--format=%(refname)",
                "--no-merged",
                self.base_branch(),
                *(f"refs/heads/{branch}" for branch in branches),
            )
        except (OSError, subprocess.SubprocessError):
            return set()
        if code != 0:
            return set()
        asked = set(branches)
        unmerged: set[str] = set()
        prefix = "refs/heads/"
        for line in out.splitlines():
            branch = line.strip()[len(prefix) :]
            # Branch names come back exactly as asked (glob-ish task ids
            # cannot widen anyone else's answer).
            if branch in asked:
                unmerged.add(branch)
        return unmerged

    def remove(self, task_id: str) -> None:
        """Remove the task's worktree; its `drcode/<task-id>` branch only
        when it holds no unmerged commits (RULE 2 — see the module
        docstring).

        Belt-and-suspenders for RULE 1: a dirty tree is snapshotted first,
        so a delete / retention purge that never passed through a verdict
        transition (whose status listener already snapshots) cannot lose
        uncommitted work either. The snapshot result and any kept-branch
        notice go through `event_log` — the task is often already gone
        from the manager here, so the app surfaces those as toasts.

        Called on cancel (Ctrl+X) and delete (Ctrl+Delete) through
        TaskManager's removal listeners — and therefore from sync context:
        best-effort, never raises (a handful of git calls). A no-op when
        nothing exists; works regardless of the current setting so
        leftovers of a session where isolation was toggled off are still
        cleaned.

        Serialized per task against `ensure_root` and any concurrent
        background removal (`schedule_removal`) through the per-task lock —
        the hot completion path runs this in a daemon thread (see
        `on_completed`), and a re-activation's worktree creation must
        never interleave with the deletion.

        Project setting `worktree_removal` (`drcode.settings`): under
        `never` the directory is NOT removed — the snapshot above still
        runs (RULE 1), and the branch stays too (it is checked out in the
        surviving worktree; git would refuse to delete it anyway). The
        other modes behave identically here: with the task means with the
        task, whether it was cancelled or had just completed.
        """
        with self._task_lock(task_id):
            self._remove_locked(task_id)

    def _remove_locked(self, task_id: str) -> None:
        """The git-touching body of remove, run under the per-task lock
        (see remove)."""
        wt = self.worktree_path(task_id)
        branch = self.branch(task_id)
        try:
            base = self.base_branch()
            snap = self.snapshot(task_id, "before removal")
            if (
                snap is not None
                and snap.startswith("snapshot committed")
                and self.event_log is not None
            ):
                self.event_log(task_id, snap)
            if self.settings.worktree_removal is WorktreeRemoval.NEVER:
                # The user keeps the directory — snapshot (above) is the
                # only thing this call does.
                return
            self._git("worktree", "prune")
            if wt.is_dir():
                code, _ = self._git("worktree", "remove", "--force", str(wt))
                if code != 0:
                    shutil.rmtree(wt, ignore_errors=True)
            elif wt.exists():
                wt.rmdir()  # empty leftover dir
            if self._branch_exists(branch):
                unmerged = self._unmerged_count(branch, base)
                if unmerged > 0:
                    # RULE 2: commits live in the object store — the
                    # directory was disposable, the branch is not.
                    if self.event_log is not None:
                        self.event_log(
                            task_id,
                            f"branch {branch} kept ({unmerged} unmerged commits)",
                        )
                else:
                    self._git("branch", "-D", branch)
        except (OSError, subprocess.SubprocessError):
            pass

    def schedule_removal(self, task_id: str, task: Task | None = None) -> None:
        """Run `remove` in a daemon thread and return immediately.

        The hot path for worktree teardown: agents leave huge trees behind
        (.venv, node_modules, build artifacts — the snapshot's `git add -A`
        and the per-file deletion each take tens of seconds on such a
        worktree), and a synchronous remove on the caller's thread froze
        the Textual event loop for the whole duration (the Header clock
        stopped ticking — every completion looked like a hang).

        Race safety (the reason removal used to be synchronous): a fast
        re-queue of the completed task (new input → re-activation →
        `ensure_root` recreates the worktree) must not interleave with the
        deletion. Two mechanisms, covering both orderings:

        - `ensure_root` serializes against an in-flight removal through
          the same per-task lock — a re-activation WAITS for the removal
          to finish, then recreates the worktree from the surviving
          branch (nothing is lost);
        - the captured `task` (the completion path passes it) is
          re-checked under that lock just before removing: a task that
          got re-queued while the job was pending is left alone (its next
          completion schedules a fresh removal).

        When `task` is None (delete/purge paths — the task is already
        gone from the manager, no re-activation is possible) the removal
        always runs. Best-effort: never raises."""
        with self._removal_cond:
            self._removal_pending.add(task_id)
        thread = threading.Thread(
            target=self._remove_in_background,
            args=(task_id, task),
            name=f"drcode-wt-remove-{task_id}",
            daemon=True,
        )
        thread.start()

    def _remove_in_background(self, task_id: str, task: Task | None) -> None:
        """The daemon-thread body of schedule_removal: re-check the task
        under the per-task lock (so the answer cannot be overtaken by a
        re-activation's ensure_root, which holds the same lock while it
        recreates the worktree), remove, then release the waiters
        (pipeline accept hooks)."""
        try:
            with self._task_lock(task_id):
                if task is not None and task.status is not TaskStatus.COMPLETED:
                    return  # re-queued / re-activated while the job was pending
                self.remove(task_id)  # RLock re-entry: the lock is held
        except Exception:  # noqa: BLE001, S110 - a removal job must never crash anything
            pass
        finally:
            with self._removal_cond:
                self._removal_pending.discard(task_id)
                self._removal_cond.notify_all()

    def wait_removal(self, task_id: str) -> None:
        """Block until a scheduled background removal of the task has
        finished (or skipped). Cheap no-op when none is pending; the
        pipeline awaits this via `asyncio.to_thread` before the accept
        hooks — they must observe the post-removal state (branch-only
        `worktree` payload, the directory really gone) exactly like the
        synchronous contract used to guarantee."""
        with self._removal_cond:
            self._removal_cond.wait_for(lambda: task_id not in self._removal_pending)

    def on_completed(self, task: Task, old: TaskStatus, origin: str) -> None:
        """Completion-transition hook for `worktree_removal:
        on_completion`: the task's worktree directory is removed as soon
        as the task reaches COMPLETED (research done, the tester's
        verdict, a no-tester or review-accept finalization — the signature
        matches `drcode.task_manager.CompletionListener`).

        A no-op under the other modes (`remove` itself honours `never`).
        Synchronous, and safe to run as a background job too (see
        `schedule_removal`): the re-queue race that used to demand the
        synchronous contract — a fast re-queue of the completed task
        (new input) having `ensure_root` recreate the worktree under the
        removal's feet — is now handled by the per-task removal lock
        (ensure_root waits out an in-flight removal) plus the job's
        re-check of the captured task. The app wires the background
        variant; this direct form stays for tests and simple embeddings.
        `remove()` snapshots the dirty tree first (RULE 1) and is
        idempotent, so the removal listeners (cancel/delete) firing
        before or after this do no harm."""
        if self.settings.worktree_removal is not WorktreeRemoval.ON_COMPLETION:
            return
        self.remove(task.id)

    def cleanup_orphans(self, live_task_ids: set[str]) -> list[str]:
        """Startup cleanup: worktree directories and `drcode/*` branches
        whose task no longer exists (crashed session, deleted task, the
        retention auto-cleanup). Returns the removed task ids.

        Worktrees/branches of LIVE tasks — including completed ones whose
        work must stay inspectable — are never touched. Both RULEs apply
        here too: a dead task's dirty tree is snapshotted onto its branch
        before the directory goes, and a branch with unmerged commits is
        never deleted (a kept branch is reported through `event_log`).
        The `worktree_removal` setting is IGNORED here: dead tasks'
        directories are leftovers of deleted tasks, not live state —
        under `never` they would otherwise accumulate without bound (see
        the module docstring).
        Blocking (git calls): run via `asyncio.to_thread` at app startup.
        Skipped entirely when there is nothing to sweep (not a git repo,
        or the setting is off and no worktree directory exists).
        """
        if not self.available:
            return []
        if not self.enabled and not (self.project_root / WORKTREES_SUBDIR).exists():
            return []
        removed: list[str] = []
        try:
            base = self.base_branch()
            self._git("worktree", "prune")
            # Directory-based sweep: our worktrees live in one private dir.
            wt_root = self.project_root / WORKTREES_SUBDIR
            if wt_root.is_dir():
                for entry in wt_root.iterdir():
                    if entry.name in live_task_ids:
                        continue
                    # Same per-task lock as remove/ensure_root: a startup
                    # sweep can overlap a background removal scheduled
                    # just before shutdown of the previous session's
                    # pattern — serialize instead of racing two
                    # snapshot+delete sequences over one directory.
                    with self._task_lock(entry.name):
                        snap = self.snapshot(entry.name, "before orphan cleanup")
                        if (
                            snap is not None
                            and snap.startswith("snapshot committed")
                            and self.event_log is not None
                        ):
                            self.event_log(entry.name, snap)
                        code, _ = self._git("worktree", "remove", "--force", str(entry))
                        if code != 0:
                            shutil.rmtree(entry, ignore_errors=True)
                    removed.append(entry.name)
            elif wt_root.exists():
                wt_root.rmdir()
            # Branch sweep: `drcode/<task-id>` without a live task — e.g.
            # the branch of a cancelled task whose worktree removal never
            # ran — is deleted, UNLESS it holds unmerged commits (RULE 2):
            # those branches are the last home of the task's work and stay.
            # A branch still checked out in some registered worktree makes
            # -D fail — ignored, the worktree sweep above has already
            # deregistered ours.
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
                    unmerged = self._unmerged_count(branch, base)
                    if unmerged > 0:
                        if self.event_log is not None:
                            self.event_log(
                                task_id,
                                f"branch {branch} kept ({unmerged} unmerged commits)",
                            )
                        continue
                    self._git("branch", "-D", branch)
                    if task_id not in removed:
                        removed.append(task_id)
        except (OSError, subprocess.SubprocessError):
            pass
        return removed
