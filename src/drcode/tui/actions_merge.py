"""Merge-conflict actions layer — opening the ConflictScreen.

The user-facing entry points of the manual conflict resolution (the
ConflictScreen, ``tui.merge_screen``): the Ctrl+M binding, the header's
⎇ N click and the Settings-palette command all funnel through
``action_open_merge``. The choice is deliberately conservative — the
screen opens only where a conflict actually exists (a parked merge in
the task's worktree, a `needs_human` fallback flag, or an unmerged
task branch); there is nothing to "open" on a healthy task, and drcode
still never merges anything on its own — the screen itself commits only
behind an explicit user confirmation.
"""

from __future__ import annotations

import asyncio

from textual import work

from .. import merge_engine
from ..models import Task
from .state import DrCodeState


class MergeActionsLayer(DrCodeState):
    def _suggest_conflict_screen(self, task: Task) -> None:
        """The fallback surfacing: the agent resolver gave up on the
        task's parked merge (`needs_human` is set — the sidebar carries
        the persistent ✋ marker) and the screen must OFFER itself. A
        toast instead of an auto-push: a modal popping mid-run would
        steal focus from whatever the user is doing."""
        self.notify(
            f"Conflict in #{task.id} needs a human — Ctrl+M (or the ⎇ "
            f"counter in the header) opens the resolution screen",
            title="Merge conflict",
            severity="warning",
            timeout=8,
        )

    def action_open_merge(self) -> None:
        """Ctrl+M / header-⎇ click: open the conflict screen for the
        viewed task when it carries a conflict, otherwise for the first
        task that does (needs_human first, then unmerged completed)."""
        viewed = self.viewed
        if viewed is not None and viewed.needs_human:
            self.open_merge_for(viewed.id)
            return
        self._open_merge_auto()

    @work(exclusive=True, group="merge-open")
    async def _open_merge_auto(self) -> None:
        viewed = self.viewed
        candidates: list[str] = []
        if viewed is not None:
            candidates.append(viewed.id)
        candidates.extend(t.id for t in self.manager.all_tasks() if t.needs_human)
        candidates.extend(await asyncio.to_thread(self._unmerged_completed_ids))
        seen: set[str] = set()
        for task_id in candidates:
            if task_id in seen or task_id not in self.manager.tasks:
                continue
            seen.add(task_id)
            ok, _reason = await asyncio.to_thread(self._merge_screen_ready, task_id)
            if ok:
                self.open_merge_for(task_id)
                return
        self.notify(
            "No parked merge conflict — nothing to resolve "
            "(a conflicted task opens this screen from Ctrl+M)",
            title="Merge",
            timeout=6,
        )

    def _merge_screen_ready(self, task_id: str) -> tuple[bool, str]:
        """Whether the conflict screen has anything to show for `task_id`
        (sync, blocking git — the callers run it in a thread)."""
        env = self.task_env
        if not env.active:
            return False, "worktree isolation is off"
        if task_id not in self.manager.tasks:
            return False, "no such task"
        if merge_engine.in_progress(env.worktree_path(task_id)):
            return True, ""
        task = self.manager.get(task_id)
        if task.needs_human:
            return True, ""
        # Branch mergedness from the 10 s poll's cached snapshot — no
        # fresh git call here (the poll re-checks git anyway).
        if self._merge_state.get(task_id) == "unmerged":
            return True, ""
        return False, "nothing to merge for this task"

    def open_merge_for(self, task_id: str) -> None:
        """Open the screen for a known-conflicted task. The screen module
        is imported lazily (the codebase's idiom for widget→app edges —
        importing it at the top would make every actions import pay for
        the screen and risk a cycle through the app)."""
        from .merge_screen import ConflictScreen

        if not self.task_env.active:
            self.notify(
                "The conflict screen needs worktree isolation "
                "(project setting `task_isolation: worktree`)",
                title="Merge",
                severity="warning",
                timeout=7,
            )
            return
        self.push_screen(ConflictScreen(task_id))
