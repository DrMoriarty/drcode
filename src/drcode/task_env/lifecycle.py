"""Worktree lifecycle of `drcode.task_env`: creation, snapshot, removal.

`_WorktreeLifecycleMixin` holds the creation half (ensure_root and its
start-point/worktree-add/untracked-copy helpers), the snapshot (RULE 1)
and the removal/cleanup machinery — everything that makes a worktree
appear, survive and disappear. The mergedness queries that gate the
branch deletion (RULE 2) live in `drcode.task_env.branch_state`; the
composed `TaskEnvironment` (`drcode.task_env`) gets both. All methods
keep their blocking-git contract: callers wrap in `asyncio.to_thread`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import TaskStatus
from ..settings import WorktreeRemoval
from ._common import (
    _COPY_FILE_LIMIT,
    _COPY_SKIP_DIRS,
    BRANCH_PREFIX,
    WORKTREES_SUBDIR,
    _copy_skipped,
    _StartPoint,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..models import Task
    from ..settings import ProjectSettings
    from ._common import EventLog


class _WorktreeLifecycleMixin:
    """The creation/snapshot/removal half of `TaskEnvironment` (see
    `drcode.task_env` for the composed class and the lifecycle contract)."""

    if TYPE_CHECKING:
        # Members provided by the composed TaskEnvironment (facade) or the
        # sibling branch_state mixin — declarations only, no runtime impact.
        project_root: Path
        settings: ProjectSettings
        event_log: EventLog | None
        _activation_lock: threading.Lock
        _removal_cond: threading.Condition
        _removal_pending: set[str]

        # Facade properties (read-only state) — declared as properties so
        # the facade's real property overrides stay compatible.
        @property
        def enabled(self) -> bool: ...
        @property
        def available(self) -> bool: ...
        @property
        def active(self) -> bool: ...

        def worktree_path(self, task_id: str) -> Path: ...
        def current_branch(self) -> str | None: ...
        def _task_lock(self, task_id: str) -> threading.RLock: ...
        def _git(self, *args: str, cwd: Path | None = None) -> tuple[int, str]: ...
        def _branch_exists(self, branch: str) -> bool: ...
        def branch(self, task_id: str) -> str: ...
        def base_branch(self) -> str: ...
        def _unmerged_count(self, branch: str, base: str | None = None) -> int: ...

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

    def ff_merge_branch(self, task_id: str) -> tuple[bool, str]:
        """Fast-forward the MAIN checkout's base branch to the task's
        `drcode/<id>` tip (`git merge --ff-only --no-edit` run in
        `project_root`) — the git mechanics behind the conflict screen's
        user-triggered "Merge into <base>" action. The contract stands
        unchanged: drcode never merges into the user's checkout on its
        own; this runs only behind an explicit user action, and the
        `--ff-only` mode makes refusal the only failure shape (a moved
        base or a dirty tree is an error return, never a conflicted
        merge to resolve in the user's tree).

        Returns (ok, combined git output); False when isolation is off
        or the branch does not exist. A refusal leaves the main checkout
        untouched — the caller surfaces the output and lets the user
        re-run the reversed merge instead. Blocking git — callers wrap
        in to_thread."""
        if not self.active:
            return False, "worktree isolation is off"
        branch = self.branch(task_id)
        if not self._branch_exists(branch):
            return False, f"branch {branch} does not exist"
        code, out = self._git("merge", "--ff-only", "--no-edit", branch)
        return code == 0, out

    def clear_identical_untracked_for_ff(
        self, task_id: str
    ) -> tuple[list[str], list[str]]:
        """Pre-process the ff-merge's untracked-file collisions in the MAIN
        checkout: paths the merge would overwrite that exist there as
        untracked files (the "file conflicts with itself" case — a copy
        was snapshotted into the worktree, committed on the branch and
        now blocks its own fast-forward). Files whose local blob is
        IDENTICAL to the branch version are removed (zero risk: after
        the ff git restores exactly those bytes from the repository);
        files that DIFFER are returned untouched — the caller must
        refuse the merge and ask for a manual move/remove.

        Returns (removed, differing) — project-relative path lists, both
        empty when there is nothing to clean (or isolation is off / the
        branch is gone — nothing to pre-process either). Tracked files
        and files absent from the branch are never touched. Blocking
        git — callers wrap in to_thread."""
        if not self.active:
            return [], []
        branch = self.branch(task_id)
        if not self._branch_exists(branch):
            return [], []
        code, out = self._git("merge-base", self.base_branch(), branch)
        if code != 0:
            return [], []
        merge_base = out.strip()
        # Paths the ff-merge would CREATE in the main checkout: added on
        # the task branch since the merge-base (modified/deleted paths
        # are tracked in the main checkout or absent — no collision).
        code, out = self._git(
            "diff", "--name-only", "--diff-filter=A", f"{merge_base}..{branch}"
        )
        if code != 0:
            return [], []
        removed: list[str] = []
        differing: list[str] = []
        for line in out.splitlines():
            path = line.strip()
            if not path:
                continue
            local = self.project_root / path
            if not local.is_file():
                continue
            # Tracked in the main checkout? The merge handles it (and
            # its local modifications) — not our case, never touch it.
            code, tracked = self._git("ls-files", "--", path)
            if code != 0 or tracked.strip():
                continue
            code, branch_hash = self._git("rev-parse", f"{branch}:{path}")
            if code != 0:
                differing.append(path)
                continue
            code, local_hash = self._git("hash-object", "--", str(local))
            if code != 0:
                differing.append(path)
                continue
            if local_hash.strip() == branch_hash.strip():
                try:
                    local.unlink()
                except OSError:
                    differing.append(path)
                    continue
                removed.append(path)
            else:
                differing.append(path)
        return removed, differing

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

    def _isolation_pin(self, task: Task | None) -> bool | None:
        """The task's isolation pin (`task.isolation_pinned`), stamping it
        on the task's FIRST activation.

        The isolation mode is a property of a task's active work, not of
        the moment it is read: a run boundary (unfinished resume, tester
        round, resolver) that re-read the live setting after the user
        toggled it mid-task would silently move the work between the
        worktree and the main checkout — neither tree holds the other's
        changes. So the mode observed at the first activation is frozen
        onto the task and every later activation honors it; the setting
        then only affects tasks that never started. Only None→value
        transitions happen here, so parallel development slots racing
        their first stamps are harmless (all of them read the same live
        value). Callers without a Task (internal/tests) get None and
        keep the historic live-setting behavior."""
        if task is None:
            return None
        if task.isolation_pinned is None:
            task.isolation_pinned = self.active
        return task.isolation_pinned

    def prestamp_isolation_pins(self, tasks: Iterable[Task]) -> list[Task]:
        """Startup hardening of the narrow window the pin cannot cover by
        itself: the app died while tasks were mid-work, and the
        `task_isolation` setting changed before the next start. Each
        candidate (mid-work, never pinned under this feature) gets its
        mode reconstructed from the surviving evidence:

        - a worktree directory or a `drcode/<id>` branch exists → pinned
          True: the task's work lives there, and a fresh worktree forked
          from the branch tip would NOT contain its uncommitted changes;
        - otherwise the task demonstrably ran (unfinished attempt — the
          status filter guarantees it) with no worktree and no branch →
          pinned False: it worked in the shared checkout, which a fresh
          worktree would not contain either.

        Tasks that never started keep None and stamp at their first
        activation as usual (no backfill). Blocking (a `rev-parse` per
        candidate without a worktree directory) — run at app startup
        before the pipeline's first `_advance()` (the resumed tester
        round must not resolve the mode from the live setting first).
        Returns the stamped tasks; the caller persists them."""
        stamped: list[Task] = []
        for task in tasks:
            if task.isolation_pinned is not None:
                continue
            mid_work = task.status is TaskStatus.TESTING or (
                task.status is TaskStatus.QUEUED and task.unfinished
            )
            if not mid_work:
                continue
            if self.worktree_path(task.id).is_dir() or self._branch_exists(
                self.branch(task.id)
            ):
                task.isolation_pinned = True
            else:
                task.isolation_pinned = False
            stamped.append(task)
        return stamped

    def ensure_root(self, task_id: str, task: Task | None = None) -> tuple[Path, str]:
        """The task's working root, with the worktree CREATED when missing.

        Returns ``(root, note)``: ``note`` is empty on success, otherwise a
        human-readable reason the task fell back to the project root (the
        pipeline logs it into the service-event history). Idempotent: an
        existing worktree — dirty, mid-work, resumed after a restart — is
        reused untouched; a branch left over from a removed worktree is
        re-checked-out so earlier commits survive a manual worktree
        deletion.

        When `task` is given, the run is resolved against the task's
        isolation PIN (`_isolation_pin`, stamped on first activation)
        instead of the live `task_isolation` setting: `pinned True` keeps
        the worktree path even after the setting was turned off, `pinned
        False` keeps the project root even after it was turned on.

        Blocking (git + the untracked-file copy): call via
        `asyncio.to_thread` from the pipeline.
        """
        pin = self._isolation_pin(task)
        if pin is False:
            # Pinned to the shared project directory (the task activated
            # before worktree isolation was switched on): the live
            # setting never moves it mid-task.
            return self.project_root, ""
        if not self.enabled and pin is not True:
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

    def _start_point(self) -> tuple[_StartPoint | None, str]:
        """Resolve the start point of a NEW task branch.

        Returns ``(start, refusal_note)``: exactly one of the two is
        meaningful. A non-None `start` is a branch name (never a raw SHA)
        the new `drcode/<task-id>` branch is forked from, with its full
        SHA for the service-event record. A non-empty `refusal_note` is a
        human-readable reason NO start point could be resolved — the main
        checkout sits in a detached HEAD (mid-rebase / mid-checkout), and
        forking from an unknown commit is exactly the silent-wrong-fork
        this method exists to prevent. When both are None/empty the
        caller treats it as a refusal too (git could not even answer
        rev-parse).

        Precedence: the `isolation_start_branch` project setting (when it
        names an existing branch — even with the main checkout detached,
        the worktree can be created correctly from that branch's tip);
        otherwise the branch the main checkout's HEAD is on. A configured
        branch that does not exist degrades to the auto mode with a
        warning (never a crash, never a silent wrong fork)."""
        configured = (self.settings.isolation_start_branch or "").strip()
        warning: str | None = None
        ref: str | None = None
        if configured:
            if self._branch_exists(configured):
                ref = configured
            else:
                warning = (
                    f"configured isolation start branch '{configured}' "
                    "does not exist — falling back to the main checkout's "
                    "current branch"
                )
        if ref is None:
            ref = self.current_branch()
            if ref is None:
                return None, (
                    "checkout is in detached HEAD / mid-rebase — task runs "
                    "in the project directory (no task branch is forked "
                    "from an unknown commit)"
                )
        code, sha = self._git("rev-parse", "--verify", ref)
        if code != 0 or not sha:
            return None, (
                f"cannot resolve the start branch '{ref}' — task runs in "
                "the project directory"
            )
        return _StartPoint(ref=ref, sha=sha, warning=warning), ""

    def _worktree_add(self, wt: Path, branch: str) -> tuple[int, str, str | None]:
        """`git worktree add` for the task's branch, with an EXPLICIT
        start point when the branch must be created (`-b`).

        The historical `git worktree add <wt> -b <branch>` without a
        start point forks from whatever commit the invoking checkout's
        HEAD happens to point at — a detached HEAD stranded mid-rebase
        silently forked task branches from foreign commits (the
        51f2efc8 incident: 3 of 4 branch commits were duplicates of
        already-merged work). A new branch is now created ONLY from a
        named branch: the `isolation_start_branch` setting, else the
        main checkout's current branch; a detached HEAD refuses the
        isolation for this activation instead of forking blindly.

        Returns `(code, combined output, created_note)`; `created_note`
        is the branch-creation service-event text (None when no branch
        was created — an existing branch was merely checked out — or
        when `add` failed), prefixed "Warning: " when the start point
        carried an anomaly."""
        if self._branch_exists(branch):
            code, out = self._git("worktree", "add", str(wt), branch)
            return code, out, None
        start, refusal = self._start_point()
        if start is None:
            # Refusal (detached HEAD / unresolvable ref): report it as a
            # failed add whose note is the human-readable reason — the
            # caller surfaces it into the task's service-event history.
            code, sha = self._git("rev-parse", "HEAD")
            detail = refusal or (
                "no usable start point — task runs in the project directory"
            )
            if code == 0 and sha:
                detail += f" (HEAD at {sha[:12]})"
            return 1, detail, detail
        code, out = self._git("worktree", "add", str(wt), "-b", branch, start.ref)
        created: str | None = None
        if code == 0:
            created = f"branch {branch} created from {start.sha[:12]} ({start.ref})"
            if start.warning:
                created = f"Warning: {start.warning}. {created}"
        return code, out, created

    def _ensure_root_locked(self, task_id: str) -> tuple[Path, str]:
        """The git-touching body of ensure_root, run under the activation
        lock (see the facade's __init__)."""
        wt = self.worktree_path(task_id)
        branch = self.branch(task_id)
        # Stale registrations (directory removed by hand) make `add` fail:
        # prune them first — it never touches healthy worktrees.
        self._git("worktree", "prune")
        wt.parent.mkdir(parents=True, exist_ok=True)
        if wt.is_dir() and wt.iterdir():
            # A non-empty directory: ours, created by an earlier activation.
            return wt, ""
        code, out, created = self._worktree_add(wt, branch)
        if code != 0:
            # Last resort: a broken leftover directory blocks `add` —
            # remove it (it is drcode's own private state) and retry.
            shutil.rmtree(wt, ignore_errors=True)
            code, out, created = self._worktree_add(wt, branch)
            if code != 0:
                return self.project_root, (
                    f"git worktree add failed, the task falls back to the "
                    f"project directory: {(created or out)[:500]}"
                )
        note = self._copy_untracked_files(wt)
        if created is not None and self.event_log is not None:
            # Observability at creation time (the 51f2efc8 lesson): the
            # task's chat records WHICH commit the branch actually forked
            # from — an anomaly is visible as a Warning right away, not
            # as a mystery conflict an hour later.
            self.event_log(task_id, created)
        return wt, note

    def _collect_untracked(self) -> list[str]:
        """Untracked (plain + gitignored) files of the main tree, relative
        POSIX paths, sorted. The private `.drcode/` state and the heavy
        dependency/cache directories — at ANY path depth — are excluded
        (see package docstring).

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
        worktree (best effort, capped — see package docstring). Returns a
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
        off (and this task has no worktree directory — a task PINNED to a
        worktree keeps snapshotting even after the setting moved on,
        RULE 1 protects the task's own tree) / not a git repo / the
        worktree does not exist / git failed.

        This is pipeline-level bookkeeping, NOT model work: blocking git
        calls — run via asyncio.to_thread from the pipeline/listeners.
        """
        if not self.active and not self.worktree_path(task_id).is_dir():
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

    # -- removal -----------------------------------------------------------------

    def remove(self, task_id: str) -> None:
        """Remove the task's worktree; its `drcode/<task-id>` branch only
        when it holds no unmerged commits (RULE 2 — see the package
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
        the package docstring).
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
