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
tester commits (tester git policy: local or push) land on the task's own
branch. An interrupted task simply keeps its worktree — the next task is
activated in its own, clean one. That is the point of the feature.

Lifecycle contract (enforced here and by the pipeline):

- activation (`ensure_root`): the worktree is created when missing
  (also after a manual deletion or a restart); an existing worktree —
  including an unfinished task's dirty one — is reused as-is.
- isolation pin: the mode a task FIRST activated under is frozen onto it
  (`Task.isolation_pinned`) — toggling `task_isolation` mid-task moves
  neither the running task nor its later rounds (tester, resolver,
  re-queues); the setting only affects tasks that never started.
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
   before any removal. Governed by the `worktree_snapshot` project
   setting (`drcode.settings`): `always` = unconditional (the historic
   behavior); `only-if-no-commits` (the default) = skipped when the run
   made its own commits since the worktree was created — a remaining
   dirty tail is presumed deliberate, the skip is logged as a service
   event and every removal path then KEEPS the worktree directory (the
   unanchored tail must never be destroyed automatically); `never` = off
   except before an actual destruction of a dead task's leftover
   directory (the orphan sweep forces it — the last chance to keep the
   work). The setting never touches the agents' own commit tools
   (`git_policy` governs those).
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

Package layout (one public facade, two implementation mixins):

- `drcode.task_env._common` — shared constants and helpers (`branch_for`,
  the copy-skip rules, `_StartPoint`).
- `drcode.task_env.lifecycle` — `_WorktreeLifecycleMixin`: creation
  (`ensure_root`), snapshot (RULE 1), removal/cleanup.
- `drcode.task_env.branch_state` — `_BranchStateMixin`: mergedness
  queries of the `drcode/*` branches (`unmerged_ids`,
  `duplicate_commits`, … — RULE 2's data source).

All callers keep importing from `drcode.task_env` — this module
re-exports the shared names and defines the composed `TaskEnvironment`.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from ..gitignore import _is_git_repo
from ..settings import ProjectSettings, TaskIsolation
from ..tools import child_priority_kwargs
from ._common import (
    _COPY_FILE_LIMIT,
    _COPY_SKIP_DIRS,
    BRANCH_PREFIX,
    WORKTREES_SUBDIR,
    EventLog,
    _copy_skipped,
    _StartPoint,
    branch_for,
)
from .branch_state import _BranchStateMixin
from .lifecycle import _WorktreeLifecycleMixin

__all__ = [
    "BRANCH_PREFIX",
    "WORKTREES_SUBDIR",
    "_COPY_FILE_LIMIT",
    "_COPY_SKIP_DIRS",
    "EventLog",
    "TaskEnvironment",
    "_StartPoint",
    "_copy_skipped",
    "branch_for",
]


class TaskEnvironment(_WorktreeLifecycleMixin, _BranchStateMixin):
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
        # The branch tip each task's worktree was (re)created from this
        # session (`_remember_activation_tip`, stamped on a fresh
        # `worktree add` only): the baseline the smart snapshot gate
        # (`worktree_snapshot: only-if-no-commits`) measures the run's own
        # commits against. Session-scoped by design — a restart loses the
        # stamps and the gate falls back to the unconditional snapshot
        # (the insurance never gets pickier than the evidence allows).
        self._activation_tips: dict[str, str] = {}

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
            **child_priority_kwargs(),
        )
        out = (proc.stdout + proc.stderr).strip()
        return proc.returncode, out

    def _branch_exists(self, branch: str) -> bool:
        code, _ = self._git("rev-parse", "--verify", "--quiet", branch)
        return code == 0

    def current_branch(self) -> str | None:
        """Short name of the branch the MAIN checkout's HEAD is on, or
        None when HEAD is detached (mid-rebase, mid-checkout, an
        inspection state). Blocking git — callers wrap in to_thread."""
        code, out = self._git("symbolic-ref", "--short", "HEAD")
        return out if code == 0 and out else None

    def local_branches(self) -> list[str]:
        """Short names of the repository's local branches — the options of
        the settings picker for `isolation_start_branch`. Empty on any git
        failure (the picker then offers only the auto mode). Blocking git —
        callers wrap in to_thread."""
        code, out = self._git("for-each-ref", "--format=%(refname:short)", "refs/heads")
        if code != 0:
            return []
        return [line.strip() for line in out.splitlines() if line.strip()]
