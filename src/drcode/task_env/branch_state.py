"""Branch-state queries of `drcode.task_env`: mergedness of the task
branches (RULE 2's data source).

`_BranchStateMixin` answers "which `drcode/<task-id>` branches still hold
commits the base branch does not have" — per task (`unmerged_ids`, the
dependency gate's mergedness half), wholesale (`drcode_branch_state`, the
header indicator / merge audit) and the patch-id twins
(`duplicate_commits`, the accept-hook guard against wrong-forked
branches). Creation/snapshot/removal live in
`drcode.task_env.lifecycle`; the composed `TaskEnvironment`
(`drcode.task_env`) gets both. All methods keep their blocking-git
contract: callers wrap in `asyncio.to_thread`.
"""

from __future__ import annotations

import hashlib
import subprocess
import threading
from typing import TYPE_CHECKING

from ._common import BRANCH_PREFIX

if TYPE_CHECKING:
    from pathlib import Path


class _BranchStateMixin:
    """The mergedness-query half of `TaskEnvironment` (see
    `drcode.task_env` for the composed class)."""

    if TYPE_CHECKING:
        # Members provided by the composed TaskEnvironment (facade) —
        # declarations only, no runtime impact.
        _unmerged_cache_fingerprint: str | None
        _unmerged_cache: dict[str, bool]
        _unmerged_cache_lock: threading.Lock

        # Facade properties (read-only state) — declared as properties so
        # the facade's real property overrides stay compatible.
        @property
        def available(self) -> bool: ...
        @property
        def active(self) -> bool: ...

        def _git(self, *args: str, cwd: Path | None = None) -> tuple[int, str]: ...
        def _branch_exists(self, branch: str) -> bool: ...
        def branch(self, task_id: str) -> str: ...

    # -- mergedness queries ------------------------------------------------------

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

    def duplicate_commits(
        self, task_id: str, base: str | None = None
    ) -> tuple[int, int] | None:
        """The wrong-fork detector behind the accept-hook guard: how many
        commits of the task's `drcode/<id>` branch are PATCH-ID duplicates
        of commits already on the base branch (`git cherry <base>
        <branch>` — the `-` lines), and how many are not (`+` lines).

        A task branch forked from the main checkout's current branch can
        never contain a patch-id its base lacks a copy of — the task's
        work is always new to the base. Duplicates therefore mean the
        branch was forked from a WRONG start point (a detached HEAD
        stranded mid-rebase, a foreign commit): merging would drag the
        already-merged work through a conflict. `(duplicates, ahead)`;
        None when the answer is unavailable (isolation off, no repo,
        branch missing, git failure) — the guard degrades to the
        historical unconditional merge.

        Note the semantics: a branch whose commits are reachable from the
        base (already merged the normal way) yields `cherry` NOTHING —
        `git cherry` only lists commits of `base..branch`, and rebase/
        cherry-pick twins (same patch-id, different SHA) are the case the
        `-` lines catch. Blocking git — callers wrap in to_thread."""
        if not self.available:
            return None
        branch = self.branch(task_id)
        if not self._branch_exists(branch):
            return None
        if base is None:
            base = self.base_branch()
        code, out = self._git("cherry", base, branch)
        if code != 0:
            return None
        lines = [line for line in out.splitlines() if line.strip()]
        duplicates = sum(1 for line in lines if line.startswith("-"))
        ahead = sum(1 for line in lines if line.startswith("+"))
        return duplicates, ahead

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
