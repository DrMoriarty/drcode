"""Activity/watchdog layer of the chat view mixin stack.

The sidebar-bottom run-activity indicator (fresh/stale, current tool,
stall countdown), the git unmerged-work header indicator and the stall
watchdog with its nudge (abort + restart of a single hung run). Split
out of :mod:`drcode.tui.chat_view` — see the composition in
``ChatViewLayer``.
"""

from __future__ import annotations

import asyncio
import time
from functools import partial

from rich.cells import cell_len
from textual.css.query import NoMatches

from ..models import Task, TaskStatus
from ..version import APP_VERSION
from .header import HeaderGitIndicator
from .state import DrCodeState
from .usage import NETWORK_RETRIES


class ActivityLayer(DrCodeState):
    # Delay between aborting a nudged run and starting its replacement: the
    # cancelled worker's handlers (persist + sanitize the transcript) must
    # finish before the resume run reads the task state.
    _NUDGE_RESTART_DELAY = 1.0

    def _touch_activity(
        self, task_id: str, tool: str | None = None, *, done: bool = False
    ) -> None:
        """Record a sign of life from the run of `task_id` (parallel
        slots: every running run keeps its own activity stamp). `tool`
        marks a tool call start; `done` clears the current-tool marker
        (result arrived)."""
        self._last_activity[task_id] = time.monotonic()
        if done:
            self._current_tool[task_id] = None
        elif tool is not None:
            self._current_tool[task_id] = tool

    @staticmethod
    def _fmt_age(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m{seconds % 60:02d}s"
        # Hours and up: drop the seconds — nobody reads them there.
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"

    def _poll_activity(self) -> None:
        """Every second, show in the sidebar whether the run(s) make
        progress or have gone silent (hung model request, long tool,
        deadlock). One live run keeps the detailed rendering; several
        (parallel development slots) collapse into a compact summary."""
        try:
            self._render_model_status()
            # Expanded header: fold task-tree changes into the session
            # counters and refresh the two tall-only lines (no-op while
            # the header is collapsed).
            self._collect_session_stats()
            self._render_header_tall()
            # Live duration counter in the task-statusbar: only ticks when
            # the viewed task is ACTIVE/TESTING (otherwise the header has
            # no timer and the refresh is a no-op re-render anyway).
            self.refresh_task_status()
            # Touch the widget so the teardown race (the 1s timer ticking
            # while the app is being dismantled) raises NoMatches here.
            _ = self.activity_status
        except NoMatches:
            # Teardown race: the 1s timer can tick while the app is being
            # dismantled and its status widgets are already gone.
            return
        agent_ids = list(self._agent_runs)
        tester_id = self._tester_task_id if self.tester_running else None
        if not (agent_ids or tester_id):
            for active in self.manager.active_tasks():
                # Network-retried run waiting for its delayed retry: the
                # task is fine, the re-run timer just has not fired yet.
                retry_at = self._agent_net_retry_at.get(active.id)
                if retry_at is not None and retry_at > time.monotonic():
                    left = max(1, int(retry_at - time.monotonic()))
                    attempt = self._agent_net_count.get(active.id, 0)
                    if attempt > NETWORK_RETRIES:
                        # Outage mode: the budget is spent, the task waits
                        # for the network (probe-gated) — say so instead of
                        # a nonsensical "attempt 7/5".
                        self._render_activity(
                            "⏳ agent: network outage",
                            f"#{active.id} offline — waiting {left}s",
                        )
                    else:
                        self._render_activity(
                            "⏳ agent: network retry",
                            f"#{active.id} attempt {attempt}/{NETWORK_RETRIES} in {left}s",
                        )
                    return
            active_tasks = self.manager.active_tasks()
            if active_tasks:
                # A task claims to be active but no worker owns it (e.g.
                # the user aborted the run, or an older build stranded it):
                # surface the inconsistency instead of a fake idle.
                first = active_tasks[0]
                more = (
                    f" (+{len(active_tasks) - 1} more)" if len(active_tasks) > 1 else ""
                )
                self._render_activity(
                    "⚠ agent: idle",
                    f"#{first.id}{more} active, nothing running — "
                    "send a message or ctrl+r to (re)start",
                    "stale",
                )
                return
            self._render_activity("○ idle", "")
            return
        if len(agent_ids) + (1 if tester_id else 0) == 1:
            # Single live run: the detailed rendering as before.
            if agent_ids:
                task_id, who = agent_ids[0], "agent"
            else:
                task_id, who = tester_id or "", "tester"
            self._render_single_run_activity(task_id, who)
            return
        # Several runs at once (parallel slots): a compact summary line —
        # the count plus the age of the most silent run, so a hung one
        # still surfaces without a per-run detail wall.
        n_agents = len(agent_ids)
        parts = [f"{n_agents} agent" + ("s" if n_agents != 1 else "")]
        if tester_id is not None:
            parts.append("1 tester")
        all_running = agent_ids + ([tester_id] if tester_id else [])
        oldest = min(
            (self._last_activity.get(t, time.monotonic()) for t in all_running),
            default=time.monotonic(),
        )
        age = int(time.monotonic() - oldest)
        detail = " + ".join(parts) + " working"
        if age < 30:
            self._render_activity(
                f"▶ {len(all_running)} runs: working",
                f"{detail} · oldest event {self._fmt_age(age)} ago",
                "fresh",
            )
        else:
            limit = self.settings.stall_nudge.minutes
            if limit is not None:
                left = max(1, limit * 60 - age)
                detail += f" · auto-nudge in {self._fmt_age(int(left))}"
            else:
                detail += " (ctrl+r nudges the viewed task)"
            self._render_activity(
                f"⚠ {len(all_running)} runs: stalled",
                f"{detail} · oldest event {self._fmt_age(age)} ago",
                "stale",
            )

    def _render_single_run_activity(self, task_id: str, who: str) -> None:
        """Detailed single-run rendering of the activity line (the
        historical one-run view): current tool, age, stall countdown."""
        stamped = self._last_activity.get(task_id)
        if stamped is None:
            stamped = time.monotonic()
        age = int(time.monotonic() - stamped)
        tool = self._current_tool.get(task_id)
        if tool:
            detail = f"tool {tool} running {self._fmt_age(age)}"
        else:
            detail = f"last event {self._fmt_age(age)} ago"
        if age < 30:
            self._render_activity(f"▶ {who}: working", detail, "fresh")
        else:
            limit = self.settings.stall_nudge.minutes
            if limit is not None:
                left = max(1, limit * 60 - age)
                detail += f" · auto-nudge in {self._fmt_age(int(left))}"
            else:
                detail += " (esc to stop, ctrl+r to nudge)"
            self._render_activity(f"⚠ {who}: stalled", detail, "stale")

    # -- unmerged-work header indicator -----------------------------------

    async def _poll_git_indicator(self) -> None:
        """Refresh the header's unmerged-work counter: `⎇ N` next to the
        clock while N completed tasks still sit unmerged on their
        `drcode/<id>` branches (same criteria as the startup reminder —
        see _unmerged_completed_ids), hidden entirely once every task's
        work is merged. A manual `git merge drcode/<id>` clears it on the
        next tick, no restart needed. The git calls are blocking — run
        them in a worker thread so a slow repo never stalls the UI."""
        ids = await asyncio.to_thread(self._unmerged_completed_ids)
        count = len(ids)
        try:
            widget = self.query_one("#header-git", HeaderGitIndicator)
        except NoMatches:
            return  # teardown race: the app is being dismantled
        if count > 0:
            plural = "" if count == 1 else "s"
            failed = sum(
                1 for task_id in ids if self.manager.get(task_id).hook_merge_failed
            )
            widget.update(f"⎇ {count}")
            tooltip = (
                f"{count} completed task{plural} not merged into the "
                "base branch yet (git merge drcode/<id>)"
            )
            if failed:
                failed_plural = "" if failed == 1 else "es"
                tooltip += (
                    f" — {failed} hook merge{failed_plural} failed "
                    "(conflict: resolve it in the checkout)"
                )
            widget.tooltip = tooltip
            widget.add_class("-unmerged")
        else:
            widget.remove_class("-unmerged")

    def _render_activity(
        self, status: str, detail: str, state: str | None = None
    ) -> None:
        """Sidebar bottom: agent/tester status on the left of one line, the
        app name and version on the right; the live detail (current tool,
        retry countdown, …) sits on the line below with no blank line in
        between. `state` ("fresh"/"stale"/None) picks the status color.
        """
        label = self.activity_status
        label.remove_class("fresh", "stale")
        if state is not None:
            label.add_class(state)
        version = f"DrCode {APP_VERSION}"
        # Content width (the widget's padding is already excluded); falls
        # back to the composed sidebar width minus padding.
        width = label.container_size.width or 34
        pad = max(1, width - cell_len(status) - cell_len(version))
        label.update(
            f"{status}{' ' * pad}[bold $accent]DrCode[/] [white]{APP_VERSION}[/]\n"
            f"{detail}"
        )

    # -- stall watchdog ---------------------------------------------------------

    def _poll_stall(self) -> None:
        """Watchdog: when a running agent/tester has emitted nothing (no
        stream deltas, no tool events) for the configured stall limit,
        abort the hung run and restart it — ONLY that run: with parallel
        development slots every run gets its own stall deadline and its
        own nudge, and the healthy siblings keep running. A stalled run
        blocks its own task — and through the pipeline gate, the whole
        backlog — so this keeps an overnight run going without human
        intervention."""
        if self._exit or not self.is_running:
            # Teardown race: the 1s interval can tick while the app is
            # being dismantled. With live runs still registered the tick
            # reaches _show_event → _write_history_line → chat_log on an
            # already-parsed-away DOM — NoMatches. Skip the tick (same
            # entry guard as _advance, see 32abee3).
            return
        limit = self.settings.stall_nudge.minutes
        if limit is None:
            return
        now = time.monotonic()
        running: list[tuple[str, str]] = [
            (task_id, "agent") for task_id in list(self._agent_runs)
        ]
        if self.tester_running and self._tester_task_id:
            running.append((self._tester_task_id, "tester"))
        if self._resolver_running and self._resolver_task_id:
            running.append((self._resolver_task_id, "resolver"))
        for task_id, who in running:
            stamped = self._last_activity.get(task_id)
            if stamped is None:
                continue  # the run has not started emitting yet
            stalled_for = now - stamped
            if stalled_for < limit * 60:
                continue
            task = self.manager.tasks.get(task_id)
            if task is None:
                continue
            self._show_event(
                task,
                f"{who} stalled for {self._fmt_age(int(stalled_for))} without any "
                f"activity — aborting the run (stall limit: "
                f"{self.settings.stall_nudge.label})",
            )
            self._nudge_run(task)

    def _nudge_run(self, task: Task) -> None:
        """Abort the hung/stalled run for `task` (if one is in flight) and
        schedule its restart: the agent resumes via a continuation turn (no
        text is re-sent — `task.unfinished` marks the interrupted attempt),
        the tester gets a fresh verification round via the pipeline gate
        (`_advance` re-tests the still-TESTING task). Only this task's run
        is touched — parallel slots keep the other tasks' runs alive."""
        if task.status is TaskStatus.ACTIVE:
            if task.id in self._agent_runs:
                self.workers.cancel_group(self, f"agent-{task.id}")
                self._agent_runs.pop(task.id, None)
                self._current_tool[task.id] = None
                self._clear_stream_label()
                self.refresh_status()
            self._schedule_pipeline_timer(
                self._NUDGE_RESTART_DELAY, partial(self.run_agent, task.id)
            )
        elif task.status is TaskStatus.TESTING:
            if self.tester_running:
                self.workers.cancel_group(self, "tester")
                self.tester_running = False
                self._current_tool[task.id] = None
                self.refresh_status()
            self._schedule_pipeline_timer(self._NUDGE_RESTART_DELAY, self._advance)
        elif self.tester_running and self._tester_task_id == task.id:
            # The run hung AFTER its useful work: set_verification already
            # flipped the task to COMPLETED mid-run. Nothing to restart —
            # but the hung worker keeps tester_running True, which blocks
            # the whole pipeline (_advance is a no-op while testing runs),
            # so it must be aborted and the pipeline re-advanced.
            self.workers.cancel_group(self, "tester")
            self.tester_running = False
            self._current_tool[task.id] = None
            self.refresh_status()
            self._schedule_pipeline_timer(self._NUDGE_RESTART_DELAY, self._advance)
        elif self._resolver_running and self._resolver_task_id == task.id:
            # A hung conflict-resolver round: abort it and restart the
            # round (the transcript is kept; the round resumes as a
            # continuation turn).
            self.workers.cancel_group(self, "resolver")
            self._resolver_running = False
            self._resolver_task_id = None
            self._current_tool[task.id] = None
            self.refresh_status()
            self._schedule_pipeline_timer(
                self._NUDGE_RESTART_DELAY, partial(self.run_resolver, task.id)
            )
