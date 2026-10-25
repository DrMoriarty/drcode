"""Custom workflow hooks (`.drcode/workflow.py`).

Split out of the former monolithic `tui/pipeline.py`: the lifecycle
hooks around a task's run (on_task_start, on_development_done,
on_tester_passed, on_accepted / on_research_accepted) plus the
unmerged-work reminder and the research completion finalization.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..models import Task, TaskKind, TaskStatus
from ..task_env import branch_for
from .state import DrCodeState


class PipelineHooksLayer(DrCodeState):
    # -- external review (.drcode/workflow.py hooks) -----------------------------------

    def _hook_env(self, task: Task) -> tuple[Path, dict[str, str] | None]:
        """Where a workflow hook for `task` runs: with per-task worktree
        isolation on — inside the task's worktree (the payload describes
        that tree in `git.*` and carries the `worktree` path/branch entry);
        without it — the project directory as before. A branch-only
        `worktree` entry (the directory is already removed under
        `worktree_removal: on_completion`, the branch with the task's
        work survives) also runs the hook from the project root — the
        accept-merge hooks only need `worktree.branch`."""
        info = self.task_env.worktree_info(task.id)
        if info is not None and "path" in info:
            return Path(info["path"]), info
        return self.project_root, info

    async def _run_task_start_hook(self, task: Task) -> str | None:
        """`on_task_start` before the first activation: branch checkout and
        friends. Returns the hook's "instructions" text (injected into the
        coding agent's first prompt), None when there is nothing to inject.
        Never fatal: a failing hook is logged and the run proceeds. Fires
        only when the transcript is still empty — the hook must not run
        again on every resume; it is the script's job to be idempotent
        beyond that."""
        workflow = self.workflow
        if workflow is None or not workflow.has("on_task_start"):
            return None
        if task.messages:
            return None  # not the first activation
        cwd, wt = self._hook_env(task)
        result = await workflow.call("on_task_start", task, cwd=cwd, worktree=wt)
        if not result.ok:
            self._show_event(
                task,
                f"Workflow on_task_start failed (continuing without it): "
                f"{result.error}",
                kind="workflow",
            )
            return None
        instructions = str(result.output.get("instructions") or "").strip()
        if instructions:
            self._show_event(
                task,
                "Workflow on_task_start: project instructions injected "
                "into the task prompt.",
                kind="workflow",
            )
            return instructions
        return None

    async def _run_development_done_hook(self, task: Task) -> None:
        """`on_development_done` when a development round finishes (push,
        open a PR, …) and before the tester round starts. The returned text
        is kept for the tester's prompt; without a synchronous on_review
        hook this switches the task into the ASYNC review mode — it will
        complete normally and await the verdict via the review poll."""
        workflow = self.workflow
        if workflow is None or not workflow.has("on_development_done"):
            return
        if task.kind is TaskKind.RESEARCH:
            return  # research tasks never go through testing/finalization
        cwd, wt = self._hook_env(task)
        result = await workflow.call("on_development_done", task, cwd=cwd, worktree=wt)
        if not result.ok:
            self._show_event(
                task,
                f"Workflow on_development_done failed (continuing without "
                f"it): {result.error}",
                kind="workflow",
            )
            return
        text = str(result.output.get("text") or "").strip()
        if text:
            self._dev_done_notes[task.id] = text

    async def _run_tester_passed_hook(self, task: Task) -> None:
        """`on_tester_passed` right after the tester stamped the task PASSED
        and the task is finalized (install, deploy, …). Fires on every path
        to COMPLETED-after-PASSED: without an external review directly from
        the tester round's completion, and on a review-mode task when the
        review finally accepts it — after `on_accepted` there (the merge
        must land first, so the install/deploy builds the merged checkout).
        An install step must run no matter which path accepted the task.
        Fail-open like `on_accepted` (an
        install step must not park the task the way the fail-closed
        `on_review` does); its text goes to the service-event log."""
        workflow = self.workflow
        if workflow is None or not workflow.has("on_tester_passed"):
            return
        # See _run_accepted_hook: the background completion removal must
        # have finished before a hook reads the worktree state.
        await asyncio.to_thread(self.task_env.wait_removal, task.id)
        if task.kind is TaskKind.RESEARCH:
            return  # research tasks never go through testing/finalization
        cwd, wt = self._hook_env(task)
        result = await workflow.call("on_tester_passed", task, cwd=cwd, worktree=wt)
        if not result.ok:
            self._show_event(
                task,
                f"Workflow on_tester_passed failed (non-fatal): {result.error}",
                kind="workflow",
            )
            return
        text = str(result.output.get("text") or "").strip()
        if text:
            self._show_event(task, f"on_tester_passed:\n{text}", kind="workflow")

    # Hook-merge failure markers: the workflow template's hooks report a
    # failed `git merge` as free text ("merge of drcode/<id> into main
    # FAILED — CONFLICT …", "NOT merged — the checkout has uncommitted
    # changes"). A crashed/nonzero-exit hook arrives as `not result.ok`
    # instead — covered separately in _record_hook_merge_outcome.
    _HOOK_MERGE_FAILURE_MARKERS = ("FAILED", "NOT merged")

    def _record_hook_merge_outcome(self, task: Task, result: Any) -> None:
        """Stamp `task.hook_merge_failed` with the accept hook's merge
        outcome and persist it. The accept hooks are fail-open (the task
        stays COMPLETED), but a FAILED/NOT merged merge must not vanish:
        the flag re-admits the task into the unmerged-work tracking (the
        header's `⎇ N` counter and the startup reminder) so a human
        resolves it — see DrCodeApp._unmerged_completed_ids. A successful
        merge (or a later successful re-run) clears the flag again."""
        failed = not result.ok or any(
            marker in str(result.output.get("text") or "")
            for marker in self._HOOK_MERGE_FAILURE_MARKERS
        )
        if task.hook_merge_failed is failed:
            return
        task.hook_merge_failed = failed
        task.touch()
        self.manager.storage.save(task)
        if failed:
            # A conflicted merge (likely parked in the task's worktree by
            # the merge engine's reversed merge): hand it to the automatic
            # conflict resolver (no-op when isolation is off, no model is
            # configured for the resolver role, or a round is running).
            self._maybe_start_resolver(task.id)

    async def _run_accepted_hook(self, task: Task) -> None:
        """`on_accepted` after an accept verdict (merge the PR, delete the
        branch, …). Its text goes to the service-event log. Never fatal.

        Serialized through `_accept_hook_lock`: with parallel slots two
        tasks can be accepted around the same moment, and both hooks
        (typically `git merge` into the shared main checkout) must not
        interleave — git's index.lock and conflicting merges would break
        one of them. Fail-open semantics unchanged: a failed hook-merge
        only stamps `task.hook_merge_failed` (unmerged-work tracking picks
        the task up), the task itself stays COMPLETED."""
        workflow = self.workflow
        if workflow is None or not workflow.has("on_accepted"):
            return
        # The completion removal (worktree_removal: on_completion) runs in
        # a background thread — the hook must observe the post-removal
        # state (branch-only worktree payload, directory really gone).
        await asyncio.to_thread(self.task_env.wait_removal, task.id)
        async with self._accept_hook_lock:
            cwd, wt = self._hook_env(task)
            result = await workflow.call("on_accepted", task, cwd=cwd, worktree=wt)
            self._record_hook_merge_outcome(task, result)
            if not result.ok:
                self._show_event(
                    task,
                    f"Workflow on_accepted failed (non-fatal): {result.error}",
                    kind="workflow",
                )
                return
            text = str(result.output.get("text") or "").strip()
            if text:
                self._show_event(task, f"on_accepted:\n{text}", kind="workflow")

    async def _run_research_accepted_hook(self, task: Task) -> None:
        """`on_research_accepted` right after a RESEARCH task completes —
        the research counterpart of `on_accepted` (research skips testing,
        so the review-accept path never runs for it): the team's script
        merges the docs branch `drcode/<task-id>` back, pushes, opens a
        PR, … Its text goes to the service-event log. Never fatal. The
        payload carries `project` (the user's checkout path and base
        branch) so a hook running inside the task's worktree can still
        merge into the main tree (`git -C <project.path> merge <branch>`).

        Serialized through `_accept_hook_lock`, like `on_accepted` (see
        there): parallel finalizations merge one at a time. A failed
        hook-merge stamps `task.hook_merge_failed` (unmerged-work tracking
        picks the task up) — fail-open, the task stays COMPLETED."""
        workflow = self.workflow
        if workflow is None or not workflow.has("on_research_accepted"):
            return
        # The completion removal (worktree_removal: on_completion) runs in
        # a background thread — the hook must observe the post-removal
        # state (branch-only worktree payload, directory really gone).
        await asyncio.to_thread(self.task_env.wait_removal, task.id)
        async with self._accept_hook_lock:
            project: dict[str, Any] = {"path": str(self.project_root)}
            if self.task_env.available:
                project["base_branch"] = await asyncio.to_thread(
                    self.task_env.base_branch
                )
            cwd, wt = self._hook_env(task)
            result = await workflow.call(
                "on_research_accepted",
                task,
                cwd=cwd,
                worktree=wt,
                extra={"project": project},
            )
            self._record_hook_merge_outcome(task, result)
            if not result.ok:
                self._show_event(
                    task,
                    f"Workflow on_research_accepted failed (non-fatal): {result.error}",
                    kind="workflow",
                )
                return
            text = str(result.output.get("text") or "").strip()
            if text:
                self._show_event(
                    task, f"on_research_accepted:\n{text}", kind="workflow"
                )

    async def _anchor_research_snapshot(self, task: Task) -> None:
        """Make sure the completion snapshot has landed on the task's
        branch before the research hook (or the reminder) looks at it:
        `_snapshot_on_settled` commits the worktree in a background job —
        a hook firing concurrently could merge a branch still being
        written to. Waits for the pending snapshot jobs, then snapshots
        once more synchronously (a no-op when the listener already
        committed everything)."""
        if not self.task_env.active:
            return
        if self._snapshot_jobs:
            await asyncio.gather(*list(self._snapshot_jobs), return_exceptions=True)
        msg = await asyncio.to_thread(
            self.task_env.snapshot, task.id, "research completed"
        )
        if msg is not None and msg.startswith("snapshot committed"):
            self._show_event(task, msg)

    async def _finalize_research_completion(self, task: Task) -> None:
        """Post-completion steps of a RESEARCH task (research skips
        testing, so none of the PASSED-finalization steps run for it):
        anchor the completion snapshot on the branch, then the
        on_research_accepted hook (merge the docs branch) — or, without
        the hook, the unmerged-work reminder. Every step is non-fatal."""
        await self._anchor_research_snapshot(task)
        await self._run_research_accepted_hook(task)
        await self._remind_unmerged_work(task)

    async def _remind_unmerged_work(self, task: Task) -> None:
        """After a finalization (PASSED for development tasks, plain
        completion for research): with per-task worktree isolation
        the task's work lives on its own branch `drcode/<task-id>` —
        drcode deliberately never merges it back (merging is a team
        decision, see `drcode.task_env`); it is left to the project's
        `on_accepted` hook or to the user. When the branch still holds
        commits the main checkout's current branch does not have, say so
        explicitly — into the task's service-event log and as a toast —
        instead of leaving a PASSED label imply that the user's checkout
        already has the fix (for a research task, that its docs commit
        reached the checkout; without isolation the work is in the shared
        tree already).

        With the task kind's accept hook defined (`on_accepted` for
        development tasks, `on_research_accepted` for research) the merge
        is (supposed to be) automated by that hook, so the toast is
        skipped there — unless the hook-merge itself failed
        (`task.hook_merge_failed`): then the branch is stuck on a conflict
        and the user must hear about it. The service event above is kept
        either way, as the durable record."""
        env = self.task_env
        if not env.active:
            return  # no isolation — nothing ever lives on a task branch
        # The background completion removal must have finished before the
        # mergedness of the branch is judged (and before a hook payload
        # would be built from it — see _run_accepted_hook).
        await asyncio.to_thread(env.wait_removal, task.id)
        unmerged = await asyncio.to_thread(env.unmerged_commits, task.id)
        if unmerged <= 0:
            return  # merged (or never committed) — nothing to remind about
        branch = branch_for(task.id)
        base = await asyncio.to_thread(env.base_branch)
        plural = "" if unmerged == 1 else "s"
        research = task.kind is TaskKind.RESEARCH
        if research:
            text = (
                f"Documentation committed on branch {branch} and not "
                f"merged into {base} yet ({unmerged} commit{plural}). "
                f"Merging is up to you (or the on_research_accepted "
                f"hook): git merge {branch}"
            )
        else:
            text = (
                f"Work finished on branch {branch} and not merged into "
                f"{base} yet ({unmerged} commit{plural}). Merging is up to "
                f"you (or the on_accepted hook): git merge {branch}"
            )
        self._show_event(task, text, kind="workflow")
        hook = "on_research_accepted" if research else "on_accepted"
        if not (
            self.workflow is not None
            and self.workflow.has(hook)
            and not task.hook_merge_failed
        ):
            # The chat may be open on another task — also raise a toast.
            self.notify(text, title="Unmerged work", severity="warning", timeout=12)

    def _apply_review_accept(self, task: Task) -> None:
        """Finish the task after an accept verdict: a sync-mode task is
        still TESTING (finalize_verified completes it with the PASSED
        label); an async-mode task is already COMPLETED (just clear the
        review-awaiting marker and the rejection count). A task the user
        re-queued while the review ran (QUEUED/HOLD) is left alone."""
        if task.status is TaskStatus.TESTING:
            self.manager.finalize_verified(task.id)
        elif task.status is TaskStatus.COMPLETED:
            task.review_pending = False
            task.review_rejections = 0
            task.touch()
            self.manager.storage.save(task)
