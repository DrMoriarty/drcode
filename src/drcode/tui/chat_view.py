from __future__ import annotations

import statistics
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from pydantic_ai.usage import RunUsage
from rich.cells import cell_len
from rich.console import RenderableType
from rich.markup import escape
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import (
    Static,
)

from ..agent import (
    extract_display,
    is_pending_retry,
    last_context_tokens,
)
from ..providers import AgentRole
from .blocks import (
    AssistantBlock,
    ReasoningBlock,
    ToolCallBlock,
    UserBlock,
)
from .sidebar import TaskItem
from .usage import NETWORK_RETRIES

_MIN_TIME = datetime.min.replace(tzinfo=UTC)
_CHAT_MOUNT_CHUNK = 16
_SCROLL_FOLLOW_THRESHOLD = 4
from ..models import (
    Clarification,
    Task,
    TaskEvent,
    TaskKind,
    TaskStatus,
    VerificationStatus,
)
from ..version import APP_VERSION
from .header import HeaderConnInfo, HeaderSessionInfo, ModelStatusHeader
from .state import DrCodeState


class ChatViewLayer(DrCodeState):
    def refresh_project_info(self) -> None:
        """Top of the sidebar, above the model status: working directory
        with the git branch (`~/pr/drcode:main`), then a blank line. The
        app name/version moved onto the activity-status line below."""
        branch = self._git_branch()
        dir_text = escape(self._display_dir())
        where = f"{dir_text}:{branch}" if branch else dir_text
        self.project_info.update(f"[dim]{where}[/]\n")

    def _display_dir(self) -> str:
        """Working directory with the home prefix collapsed to `~`."""
        path = self.project_root
        try:
            rel = path.relative_to(Path.home())
        except ValueError:
            return str(path)
        if str(rel) == ".":
            return "~"
        return f"~/{rel}"

    def _git_branch(self) -> str | None:
        """Current branch of the git repo in the working directory.

        Parsed straight from `.git` (also handles worktrees, where `.git`
        is a file with a `gitdir:` pointer) so no `git` binary is needed.
        """
        dot_git = self.project_root / ".git"
        try:
            if dot_git.is_dir():
                git_dir: Path | None = dot_git
            elif dot_git.is_file():
                line = dot_git.read_text(encoding="utf-8").strip()
                if not line.startswith("gitdir:"):
                    return None
                git_dir = Path(line.split(":", 1)[1].strip())
                if not git_dir.is_absolute():
                    git_dir = self.project_root / git_dir
            else:
                git_dir = None
        except OSError:
            return None
        if git_dir is None:
            return None
        try:
            content = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if content.startswith("ref: refs/heads/"):
            return content[len("ref: refs/heads/") :]
        if content.startswith("ref: "):
            return content[len("ref: ") :]
        return content[:7]  # detached HEAD

    def _sidebar_fingerprint(self) -> tuple:
        """Lightweight snapshot of everything the sidebar renders: status,
        verification label, title, the viewed-marker and the queue stamp
        (the QUEUE section is order-sensitive) per task."""
        return tuple(
            (
                t.id,
                t.status.value,
                t.kind.value,
                t.verification,
                t.title_display,
                t.id == self.viewed_id,
                t.queue_seq,
                t.queue_head,
            )
            for t in self.manager.all_tasks()
        )

    def _poll_sidebar(self) -> None:
        """Redraw the sidebar only when the underlying task state changed
        since the last render (status transitions, new bug reports from the
        tester, promote_next() activations, …)."""
        if self._sidebar_fingerprint() != self._sidebar_snapshot:
            self.refresh_sidebar()
            # The task status header (and the bottom statusbar) have no
            # poller of their own: they are refreshed explicitly, and mid-run
            # state changes (set_verification flips the viewed task to
            # COMPLETED while its tester run is still finishing) would leave
            # them stale until the run ends — or a view switch.
            self.refresh_status()

    def _sidebar_sections(self) -> list[tuple[str, list[Task]]]:
        """Sidebar content: pipeline sections top to bottom, empty ones
        dropped (the sidebar shows only the stages that have tasks).

        Shared by the sidebar rendering and the startup view pick (the
        opened task must be the one at the top of the sidebar).
        """
        # Group tasks by status; each group keeps the created_at order of
        # all_tasks() unless sorted explicitly below.
        by_status: dict[TaskStatus, list[Task]] = {}
        for t in self.manager.all_tasks():
            by_status.setdefault(t.status, []).append(t)

        queue = list(reversed(self.manager.queued()))
        clarification = sorted(
            by_status.get(TaskStatus.AWAITING_CLARIFICATION, []),
            key=lambda t: t.updated_at,
            reverse=True,
        )
        active = by_status.get(TaskStatus.ACTIVE, [])
        testing = by_status.get(TaskStatus.TESTING, [])
        # Normally one task is being tested; if several are stuck in
        # TESTING, the one the tester is working on goes to the bottom of
        # the section (closest to the pipeline's exit into COMPLETED).
        testing_active_id = self._tester_task_id
        testing = [t for t in testing if t.id != testing_active_id] + [
            t for t in testing if t.id == testing_active_id
        ]
        completed = sorted(
            by_status.get(TaskStatus.COMPLETED, []),
            key=lambda t: t.updated_at,
            reverse=True,
        )
        hold = by_status.get(TaskStatus.HOLD, [])

        sections: list[tuple[str, list[Task]]] = [
            ("QUEUE", queue),
            ("NEEDS CLARIFICATION", clarification),
            ("ACTIVE", active),
            ("TESTING", testing),
            ("COMPLETED", completed),
            ("ON HOLD", hold),
        ]
        return [(name, tasks) for name, tasks in sections if tasks]

    def _sidebar_top_task(self) -> Task | None:
        """First task of the first non-empty sidebar section — what the
        chat pane opens on startup."""
        for _name, tasks in self._sidebar_sections():
            return tasks[0]
        return None

    def refresh_sidebar(self) -> None:
        sidebar = self.sidebar
        sidebar.clear()
        for name, tasks in self._sidebar_sections():
            # A collapsed section keeps only its header line: the "+" shows
            # there is content hidden underneath, "-" means fully expanded.
            collapsed = name in self._collapsed_sections
            marker = "+" if collapsed else "-"
            sidebar.append(
                TaskItem(f"{marker} {name} ({len(tasks)})", header=True, section=name)
            )
            if collapsed:
                continue
            for t in tasks:
                marker = {
                    "queued": "⏳",
                    "hold": "⏸",
                    "active": "▶",
                    "testing": "🧪",
                    "completed": "✓",
                    "clarification": "❓",
                }.get(t.status.value, "⏳")
                kind_letter = {
                    TaskKind.BUG: "[red]B[/]",
                    TaskKind.RESEARCH: "[yellow]R[/]",
                }.get(t.kind, "[green]F[/]")
                if t.verification is VerificationStatus.PASSED:
                    suffix = " ✅"
                elif t.verification is VerificationStatus.FAILED:
                    suffix = " ❌"
                else:
                    suffix = ""
                viewed = " ●" if t.id == self.viewed_id else ""
                item = TaskItem(
                    f"{marker} {kind_letter} #{t.id} "
                    f"{escape(t.title_display)}{suffix}{viewed}",
                    t.id,
                )
                if t.id == self.viewed_id:
                    item.add_class("viewed")
                sidebar.append(item)
        sidebar.refresh()
        self._sidebar_snapshot = self._sidebar_fingerprint()

    def toggle_sidebar_section(self, section: str) -> None:
        """Collapse/expand one sidebar category (its header row).

        The rebuild resets the ListView cursor, so it is put back onto the
        toggled header — the next space/enter keeps working on it.
        """
        if section in self._collapsed_sections:
            self._collapsed_sections.discard(section)
        else:
            self._collapsed_sections.add(section)
        self.refresh_sidebar()
        sidebar = self.sidebar
        for index, item in enumerate(sidebar.children):
            if isinstance(item, TaskItem) and item.section == section:
                sidebar.index = index
                break

    def _refocus_sidebar_task(self, task_id: str) -> None:
        """Put the ListView cursor back onto a task's row after a sidebar
        rebuild (the rebuild resets it), so a series of reorder/hold keys
        keeps working on the same task without re-selecting it."""
        sidebar = self.sidebar
        for index, item in enumerate(sidebar.children):
            if isinstance(item, TaskItem) and item.task_id == task_id:
                sidebar.index = index
                break

    # -- model connection status ---------------------------------------------

    def _recover_from_outage(self) -> None:
        """Error → connected transition: the outage is over, so un-block the
        tasks whose runs died with it. They were excluded from promotion via
        _agent_failed and would stay stranded until restart otherwise. If
        nothing is running, the pipeline picks the next task immediately."""
        self._agent_failed.clear()
        # The connection is back: a later failed-queue rescue (see
        # _promote_failed_queue) must fire immediately, not after a stale
        # backoff window accumulated during the outage.
        self._failed_queue_rounds = 0
        self._failed_queue_retry_at = 0.0
        self._advance()

    def refresh_connection(self) -> None:
        self.run_worker(self._check_model(), exclusive=True, group="model-check")

    def _render_model_status(self) -> None:
        """Header connection label + live activity suffix.

        The suffix is the global "is the model producing" signal: a tok/s
        figure while tokens stream in (regardless of which task is open in
        the chat), "waiting" while a run is alive but the model is silent
        (a tool executing, a non-streamed request in flight), "testing"
        while the tester works, and just the plain connection label when
        nothing runs.
        """
        label = self.model_status
        tps = self._tokens_per_second()
        if tps is not None:
            label.update(f"{self._conn_label} · {tps:.0f} tok/s")
        elif self.agent_running:
            label.update(f"{self._conn_label} · waiting")
        elif self.tester_running:
            label.update(f"{self._conn_label} · testing")
        else:
            label.update(self._conn_label)

    # -- expanded (tall) header: connection details + session statistics ----

    @property
    def header_conn_info(self) -> Static:
        return self.query_one("#header-conn", HeaderConnInfo)

    @property
    def header_session_info(self) -> Static:
        return self.query_one("#header-sessions", HeaderSessionInfo)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """Compact human duration for the header stats line (e.g. 45s, 6m12s)."""
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds}s"
        return f"{seconds // 60}m{seconds % 60:02d}s"

    def _collect_session_stats(self) -> None:
        """Fold task-tree changes into the session counters.

        Runs on the 1 s poll: tasks that appeared since the previous tick
        count as created this session (by kind), tasks that reached
        COMPLETED count as processed, with the activation → completion
        span (when both timestamps exist — a task cancelled straight from
        the queue has no activation stamp and is counted without a
        duration).
        """
        tasks = self.manager.tasks
        for task_id in set(tasks) - self._session_known_ids:
            self._session_known_ids.add(task_id)
            self._session_created[tasks[task_id].kind] += 1
        for task in tasks.values():
            if (
                task.status is TaskStatus.COMPLETED
                and task.id not in self._session_completed_ids
            ):
                self._session_completed_ids.add(task.id)
                self._session_completed += 1
                if task.activated_at and task.completed_at:
                    self._session_durations.append(
                        (task.completed_at - task.activated_at).total_seconds()
                    )

    def _header_conn_text(self) -> str:
        """The expanded header's connection line: the connection label
        (dot + model, error detail on failures), URL, last probe
        round-trip, consecutive probe failures, network retry counters
        and the exact cumulative token usage of every model call of the
        session."""
        coding = self.resolved_config(AgentRole.CODING)
        if not coding.is_ready:
            return "model not configured"
        url = coding.base_url
        probe = (
            f"probe {self._last_probe_ms} ms"
            if self._last_probe_ms is not None
            else "probe —"
        )
        fails = f"probe fails: {self._probe_fail_streak}"
        open_net = sum(self._agent_net_count.values())
        retries = f"net retries: {self._net_retries_total} total"
        if open_net:
            retries += f" / {open_net} active"
        line = f"{self._conn_label} | {url} | {probe} | {fails} | {retries}"
        usage = self._header_usage_text()
        if usage:
            line += f" | {usage}"
        return line

    def _header_stats_text(self) -> str:
        """The expanded header's session-statistics line: tasks completed
        this session, average/median activation→completion time and the
        bugs:features ratio of the tasks created this session (research
        tasks are excluded from the ratio: they are neither bug fixes nor
        implemented features)."""
        durations = self._session_durations
        if durations:
            avg = statistics.fmean(durations)
            median = statistics.median(durations)
            times = (
                f"avg {self._fmt_duration(avg)} | median {self._fmt_duration(median)}"
            )
        else:
            times = "avg — | median —"
        bugs = self._session_created[TaskKind.BUG]
        features = self._session_created[TaskKind.FEATURE]
        line = (
            f"tasks done: {self._session_completed} | {times} | "
            f"bugs:features {bugs}:{features}"
        )
        return f"{line} | {self._header_github_text()}"

    def _header_github_text(self) -> str:
        """GitHub-integration status for the expanded header's bottom line:
        configured/not configured (token + repository set), active (the
        sync switch is on) and the date/time of the last successful sync
        (or the failure reason of the last attempt)."""
        settings = self.settings
        if not (settings.github_token and settings.github_repo):
            return "GitHub: not configured"
        if not settings.github_enabled:
            return "GitHub: configured, off"
        if self._github_syncing:
            status = "GitHub: on | syncing…"
        else:
            last = self._github_state.last_sync
            stamp = last.astimezone().strftime("%d.%m.%Y %H:%M") if last else "never"
            status = f"GitHub: on | last sync {stamp}"
        if self._github_last_error:
            status += f" | error: {self._github_last_error}"
        return status

    @staticmethod
    def _fmt_tokens(n: int) -> str:
        """Compact human token count for the header (e.g. 45k, 1.2M, 980)."""

        def fmt(v: float, unit: str) -> str:
            return f"{v:.1f}".rstrip("0").rstrip(".") + unit

        if n >= 1_000_000:
            return fmt(n / 1_000_000, "M")
        if n >= 1_000:
            k = f"{n / 1_000:.1f}".rstrip("0").rstrip(".")
            if float(k) < 1000:  # 999_950+ rounds up into M territory
                return f"{k}k"
            return fmt(n / 1_000_000, "M")
        return str(n)

    def _record_session_usage(self, usage: RunUsage | None) -> None:
        """Fold one model call's exact token usage into the session
        totals (main agent, tester and compaction summarizer alike) and
        refresh the expanded header line for an instant update — the 1 s
        poller keeps it live while the header is collapsed (no-op there)."""
        if usage is None:
            return
        self._session_usage.add(usage)
        self._render_header_tall()

    def _header_usage_text(self) -> str:
        """Exact cumulative token usage of the session, compactly: the
        four per-response counters pydantic-ai reports. Zero fields are
        omitted (providers without prompt caching simply show no cache
        part); an all-zero counter renders as an empty string."""
        u = self._session_usage
        parts: list[str] = []
        if u.input_tokens:
            parts.append(f"tok in {self._fmt_tokens(u.input_tokens)}")
        if u.cache_read_tokens or u.cache_write_tokens:
            parts.append(
                f"cache {self._fmt_tokens(u.cache_read_tokens)}"
                f"/{self._fmt_tokens(u.cache_write_tokens)}"
            )
        if u.output_tokens:
            parts.append(f"out {self._fmt_tokens(u.output_tokens)}")
        return " | ".join(parts)

    def _render_header_tall(self) -> None:
        """Refresh the two expanded-header lines (no-op while collapsed:
        the widgets are display:none and their content is refreshed on
        toggle / by the next poll tick anyway)."""
        header = self.query_one(ModelStatusHeader)
        if not header.has_class("-tall"):
            return
        self.header_conn_info.update(self._header_conn_text())
        self.header_session_info.update(self._header_stats_text())

    def on_model_status_header_tall_toggled(
        self, event: ModelStatusHeader.TallToggled
    ) -> None:
        """The header was clicked: fill the expanded lines immediately (the
        1 s poll keeps them live afterwards)."""
        if event.tall:
            self._render_header_tall()

    # -- run activity indicator ----------------------------------------------

    def _touch_activity(self, tool: str | None = None, *, done: bool = False) -> None:
        """Record a sign of life from the agent/tester run. `tool` marks a
        tool call start; `done` clears the current-tool marker (result
        arrived)."""
        self._last_activity = time.monotonic()
        if done:
            self._current_tool = None
        elif tool is not None:
            self._current_tool = tool

    @staticmethod
    def _fmt_age(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds}s"
        return f"{seconds // 60}m{seconds % 60:02d}s"

    def _poll_activity(self) -> None:
        """Every second, show in the sidebar whether the run makes progress
        or has gone silent (hung model request, long tool, deadlock)."""
        try:
            self._render_model_status()
            # Expanded header: fold task-tree changes into the session
            # counters and refresh the two tall-only lines (no-op while
            # the header is collapsed).
            self._collect_session_stats()
            self._render_header_tall()
            # Touch the widget so the teardown race (the 1s timer ticking
            # while the app is being dismantled) raises NoMatches here.
            _ = self.activity_status
        except NoMatches:
            # Teardown race: the 1s timer can tick while the app is being
            # dismantled and its status widgets are already gone.
            return
        if not (self.agent_running or self.tester_running):
            active = self.manager.active
            if active is not None:
                # Network-retried run waiting for its delayed retry: the task
                # is fine, the re-run timer just has not fired yet.
                retry_at = self._agent_net_retry_at.get(active.id)
                if retry_at is not None and retry_at > time.monotonic():
                    left = max(1, int(retry_at - time.monotonic()))
                    attempt = self._agent_net_count.get(active.id, 0)
                    self._render_activity(
                        "⏳ agent: network retry",
                        f"#{active.id} attempt {attempt}/{NETWORK_RETRIES} in {left}s",
                    )
                    return
                # Task claims to be active but no worker owns it (e.g. the
                # user aborted the run, or an older build stranded it):
                # surface the inconsistency instead of showing a fake idle.
                self._render_activity(
                    "⚠ agent: idle",
                    f"#{active.id} active, nothing running — "
                    "send a message or ctrl+r to (re)start",
                    "stale",
                )
                return
            self._render_activity("○ idle", "")
            return
        who = "agent" if self.agent_running else "tester"
        age = int(time.monotonic() - self._last_activity)
        if self._current_tool:
            detail = f"tool {self._current_tool} running {self._fmt_age(age)}"
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

    # Delay between aborting a nudged run and starting its replacement: the
    # cancelled worker's handlers (persist + sanitize the transcript) must
    # finish before the resume run reads the task state.
    _NUDGE_RESTART_DELAY = 1.0

    def _poll_stall(self) -> None:
        """Watchdog: when the running agent/tester has emitted nothing (no
        stream deltas, no tool events) for the configured stall limit, abort
        the hung run and restart it. A stalled run blocks its own task — and
        through the pipeline gate, the whole backlog — so this keeps an
        overnight run going without human intervention."""
        limit = self.settings.stall_nudge.minutes
        if limit is None:
            return
        if not (self.agent_running or self.tester_running):
            return
        stalled_for = time.monotonic() - self._last_activity
        if stalled_for < limit * 60:
            return
        if self.agent_running:
            task = self.manager.active
        else:
            task = (
                self.manager.get(self._tester_task_id) if self._tester_task_id else None
            )
        if task is None:
            return
        who = "agent" if self.agent_running else "tester"
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
        (`_advance` re-tests the still-TESTING task)."""
        if task.status is TaskStatus.ACTIVE:
            if self.agent_running:
                self.workers.cancel_group(self, "agent")
                self.agent_running = False
                self._current_tool = None
                self._clear_stream_label()
                self.refresh_status()
            self.set_timer(self._NUDGE_RESTART_DELAY, partial(self.run_agent, task.id))
        elif task.status is TaskStatus.TESTING:
            if self.tester_running:
                self.workers.cancel_group(self, "tester")
                self.tester_running = False
                self._current_tool = None
                self.refresh_status()
            self.set_timer(self._NUDGE_RESTART_DELAY, self._advance)
        elif self.tester_running and self._tester_task_id == task.id:
            # The run hung AFTER its useful work: set_verification already
            # flipped the task to COMPLETED mid-run. Nothing to restart —
            # but the hung worker keeps tester_running True, which blocks
            # the whole pipeline (_advance is a no-op while testing runs),
            # so it must be aborted and the pipeline re-advanced.
            self.workers.cancel_group(self, "tester")
            self.tester_running = False
            self._current_tool = None
            self.refresh_status()
            self.set_timer(self._NUDGE_RESTART_DELAY, self._advance)

    # -- chat pane ---------------------------------------------------------------

    def _log_event(self, task: Task, text: str, *, kind: str = "system") -> None:
        """Persist a service event into the task's event log.

        These lines are shown in the chat but must never reach the model:
        they live in `task.events`, outside the transcripts that
        resumable_history() reads, so a resumed agent never sees them.
        """
        task.events.append(TaskEvent(kind=kind, text=text))
        task.touch()
        self.manager.storage.save(task)

    @staticmethod
    def _format_event(event: TaskEvent) -> str:
        ts = event.at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        return f"[dim]{ts}[/] {escape(event.text)}"

    def _show_event(self, task: Task, text: str, *, kind: str = "system") -> None:
        """Persist an event and, if the task is open, render it live."""
        self._log_event(task, text, kind=kind)
        if self.viewed_id == task.id:
            self._write_history_line("event", self._format_event(task.events[-1]))

    def view_task(self, task_id: str) -> None:
        # Opening a task cancels a pending new-task creation: the click on a
        # sidebar item is an explicit "leave the form" from the user.
        if self._new_task_mode:
            self._leave_new_task_mode()
        self.viewed_id = task_id
        task = self.manager.get(task_id)
        # Wipe marker for the streaming loop: any block created before this
        # point belongs to a dead view and must be re-mounted on the next
        # delta (its accumulated content is carried over, so the live text
        # never loses the tail of an in-flight response).
        self._chat_epoch += 1
        self.chat_log.remove_children()
        # Drop any un-drained chunks of a previous view: their drain is
        # stale now (the epoch guard below would drop them anyway, but the
        # new _render_task_history appends into the same list).
        self._chat_batch.clear()
        self._chat_batching = True
        try:
            self._render_task_history(task, task_id)
        finally:
            # Opening a task always lands on the latest entry (previously
            # this happened implicitly because every mount scrolled to the
            # end). The finally keeps a mid-render exception from leaving
            # batching on (later live writes would be swallowed).
            self._flush_chat_batch()
        self.refresh_sidebar()
        self.refresh_status()

    def _render_task_history(self, task: Task, task_id: str) -> None:
        """Write the task's full history into the (batched) chat log."""
        self._chat_has_content = False
        self._pending_rendered = []
        self._clear_stream_label()
        # Rebuild the full history as ONE chronological timeline. Both the
        # serialized transcripts (per-message timestamps) and the service
        # events (TaskEvent.at) carry wall-clock stamps, so a stable sort
        # by timestamp reproduces exactly the order in which the entries
        # were rendered live — the same order on every (re)open, instead of
        # clumping all events at the tail.
        timeline: list[tuple[datetime, int, str, str]] = []
        seq = 0

        def _add(at: datetime | None, role: str, text: str) -> None:
            nonlocal seq
            timeline.append((at or _MIN_TIME, seq, role, text))
            seq += 1

        for at, role, text in extract_display(task.messages):
            _add(at, role, text)
        # Tester rounds: the synthetic prompt ("Verify completed task #…")
        # is not shown; each round is announced by its own "testing"
        # separator anchored at the round's first message.
        for round_messages in task.tester_messages:
            if not round_messages:
                continue
            round_items = extract_display(round_messages)
            round_at = next((at for at, _, _ in round_items if at is not None), None)
            _add(round_at, "testing-sep", "")
            for at, role, text in round_items:
                if role == "user":
                    continue
                if role == "tool-call":
                    text = f"tester: {text}"
                _add(at, role, text)
        # Service events (failures, skips) — human-only, with timestamps;
        # interleaved at the position they occurred at.
        for event in task.events:
            _add(event.at, "event", self._format_event(event))
        # Queued user input — interleaved by the moment it was TYPED, not
        # pinned to the tail: comments sent while the task was queued or
        # running belong between the messages that were on screen then.
        for entry in task.pending_input:
            if is_pending_retry(task.messages, entry.text):
                # The failed/interrupted run left this prompt in the
                # transcript: it is already rendered from task.messages
                # above — writing the pending copy again duplicates it.
                continue
            _add(entry.at, "user-pending", entry.text)
        timeline.sort(key=lambda item: item[:2])
        for at, _, role, text in timeline:
            block = self._write_history_line(
                role, text, None if at == _MIN_TIME else at
            )
            if role == "tool-call" and isinstance(block, ToolCallBlock):
                # A tool still executing (run in flight) gets its spinner
                # back on the freshly rendered block.
                self._reattach_tool_spinner(block, text, task_id)
        if task.status is TaskStatus.AWAITING_CLARIFICATION and task.clarification:
            self._write_clarification(task.clarification)
            self._rendered_clarification = task.clarification

    def _chat_at_bottom(self) -> bool:
        """True when the reader is tracking the chat tail (auto-scroll follows).

        Allows a small threshold of trailing rows below the viewport edge:
        markdown reflow can suddenly add a couple of lines and a fresh model
        chunk arrives at the same time, so the tail edge jumps below the
        visible area. A 1-line tolerance would then falsely report "not at
        bottom" and stop following the tail; within this threshold we still
        treat the reader as tracking the tail.

        Tolerance absorbs rounding of fractional scroll offsets.
        """
        log = self.chat_log
        return log.scroll_offset.y >= log.max_scroll_y - _SCROLL_FOLLOW_THRESHOLD

    def _stream_follow(self) -> None:
        """Keep the chat tail visible while streaming, coalesced to ~10 Hz.

        Called per streamed delta; each scroll_end schedules a refresh+scroll
        pass over the (growing) chat log — doing that per delta saturates the
        event loop and makes keyboard/mouse input unresponsive.
        """
        now = time.monotonic()
        if now - self._last_follow >= 0.1:
            self._last_follow = now
            self.chat_log.call_after_refresh(self.chat_log.scroll_end, animate=False)

    # -- tool-call spinners ------------------------------------------------------

    def _tool_call_started(
        self,
        run_key: tuple[str, int],
        task_id: str,
        call_id: str,
        call_text: str,
        block: ToolCallBlock,
    ) -> None:
        """Register a tool call whose execution just began (spinner on)."""
        self._active_tools.setdefault(run_key, {})[call_id] = (
            task_id,
            call_text,
            block,
        )

    def _tool_call_finished(self, call_id: str) -> None:
        """Tool result arrived: stop the spinner of its block (in whichever
        run registered it — agent and tester never overlap, but the lookup
        is over all runs anyway)."""
        for per_run in self._active_tools.values():
            entry = per_run.pop(call_id, None)
            if entry is not None:
                entry[2].stop_spinner()

    def _stop_run_tools(self, run_key: tuple[str, int]) -> None:
        """End-of-run cleanup: anything still marked running for this run
        (a result event lost to an abort/error) loses its spinner."""
        for _task_id, _text, block in self._active_tools.pop(run_key, {}).values():
            block.stop_spinner()

    def _reattach_tool_spinner(
        self, block: ToolCallBlock, call_text: str, task_id: str
    ) -> None:
        """After a mid-run view re-render, hand a still-running call's
        spinner over to the freshly rendered block (matched by the exact
        rendered call text — same tool with the same arguments)."""
        for per_run in self._active_tools.values():
            for call_id, (tid, text, _old) in per_run.items():
                if tid == task_id and text == call_text:
                    per_run[call_id] = (tid, text, block)
                    block.start_spinner()
                    return

    def _write_chat(
        self, content: RenderableType | Widget, *, expand: bool = False
    ) -> None:
        """Mount one block to the chat log.

        Scrolls to the end only when the user is already looking at the
        bottom: a scrolled-up reader must not be yanked down by new tool
        calls / reasoning blocks arriving while the agent runs.

        While batching (view_task's history rebuild and its chunked drain),
        the block is only collected — _flush_chat_batch() mounts the
        timeline in chunks afterwards.
        """
        if isinstance(content, Widget):
            content.add_class("chat-block")
        else:
            content = Static(content, expand=expand, classes="chat-block")
        if self._chat_batching:
            self._chat_batch.append(content)
            self._chat_has_content = True
            return
        follow = self._chat_at_bottom()
        self.chat_log.mount(content)
        if follow:
            self.chat_log.scroll_end(animate=False)
        self._chat_has_content = True

    def _flush_chat_batch(self) -> None:
        """Start mounting the blocks collected by a batched history rebuild.

        Mounts in chunks, yielding to the event loop between them: one
        mounted block costs a few ms (textual re-applies styles over all
        mounted widgets per mount), so a fat 600-block history would freeze
        keyboard/mouse input for ~2s if mounted in a single synchronous
        pass. Late live writes (a stream starting while the drain is still
        going) keep appending to _chat_batch — batching stays on until the
        last chunk — and ride along with the remaining chunks.
        """
        self._chat_batching = True
        self._chat_batch_epoch = self._chat_epoch
        self._drain_chat_batch()

    def _drain_chat_batch(self) -> None:
        """Mount one chunk of the pending batch, reschedule for the rest."""
        if self._chat_batch_epoch != self._chat_epoch:
            # The view switched mid-drain: the new view_task wiped the log
            # and owns it now — drop the stale tail.
            self._chat_batching = False
            self._chat_batch.clear()
            return
        chunk = self._chat_batch[:_CHAT_MOUNT_CHUNK]
        del self._chat_batch[:_CHAT_MOUNT_CHUNK]
        try:
            if chunk:
                self.chat_log.mount(*chunk)
            if self._chat_batch:
                self.call_after_refresh(self._drain_chat_batch)
            else:
                self._chat_batching = False
                # Opening a task always lands on the latest entry (previously
                # this happened implicitly because every mount scrolled to
                # the end).
                self.chat_log.scroll_end(animate=False)
        except Exception:
            # Never leave batching stuck on: later live writes would be
            # silently swallowed.
            self._chat_batching = False
            self._chat_batch.clear()
            raise

    def _write_history_line(
        self, role: str, text: str, at: datetime | None = None
    ) -> Widget | None:
        """Render one history entry into the chat log.

        Returns the mounted block widget (None for markup-string lines) so
        callers can tweak it afterwards — view_task re-attaches the running
        spinner to tool-call blocks this way.
        """
        block: Widget | None = None
        if role == "user":
            block = UserBlock(text)
        elif role == "user-pending":
            self._pending_rendered.append(text)
            block = UserBlock(text)
        elif role == "reasoning":
            block = ReasoningBlock(at)
            block.set_content(text)
        elif role == "tool-call":
            block = ToolCallBlock(text, at)
        elif role == "assistant":
            block = AssistantBlock(text)
        elif role == "tool":
            self._write_chat(f"[dim]{escape(text)}[/]")
        elif role == "tester":
            self._write_chat(f"[bold magenta]tester ›[/] {text}")
        elif role == "testing-sep":
            label = f"testing: {text}" if text else "testing"
            self._write_chat(f"[bold magenta]──── {label} ────[/]")
        elif role == "summary":
            # Compaction summary: a marker separator, then the summary
            # itself rendered as markdown (it is the model's output the
            # agent continues from, not a user message).
            self._write_chat("[bold magenta]──── context compacted ────[/]")
            block = AssistantBlock(text)
        elif role == "event":
            self._write_chat(f"[bold red]event ›[/] {text}")
        elif role == "system":
            self._write_chat(f"[bold red]error ›[/] {text}")
        if block is not None:
            self._write_chat(block)
        return block

    def _write_clarification(self, clarification: Clarification) -> None:
        lines = [f"[bold blue]drcode ❓[/] {escape(clarification.question)}"]
        for i, option in enumerate(clarification.options):
            lines.append(
                f"  ▸ [@click=app.pick_option({i})]"
                f"[underline cyan]{escape(option)}[/][/]"
            )
        lines.append("[dim]Click an option or answer in the input below.[/]")
        self._write_chat("\n".join(lines))

    def refresh_status(self) -> None:
        active = self.manager.active
        queue_len = len(self.manager.queued())
        testing_len = len(
            [t for t in self.manager.all_tasks() if t.status is TaskStatus.TESTING]
        )
        state = "running" if self.agent_running else "idle"
        tester = "running" if self.tester_running else "idle"
        who = f"#{active.id} {active.title_display}" if active else "none"
        model = self.resolved_config(AgentRole.CODING).model or (
            "(model not configured)"
        )
        self.query_one("#statusbar", Static).update(
            f" active: {who}   queue: {queue_len}   testing: {testing_len}   "
            f"agent: {state}   tester: {tester}   model: {model} "
        )
        self.refresh_task_status()

    def refresh_task_status(self) -> None:
        # The footer mirrors the state check_action() gates on (viewed task,
        # its status, running flags): re-evaluate the visible bindings.
        self.refresh_bindings()
        left = self.query_one("#task-status-left", Static)
        right = self.query_one("#task-status-right", Static)
        # The new-task form is not tied to any task: keep the bar empty so
        # the previously viewed task doesn't keep updating behind the form.
        if self._new_task_mode:
            left.update("")
            right.update("")
            return
        task = self.viewed
        if task is None:
            left.update("")
            right.update("")
            return
        created = task.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
        kind = {
            TaskKind.BUG: "[bold red]bug[/]",
            TaskKind.RESEARCH: "[bold yellow]research[/]",
        }.get(task.kind, "[bold green]feature[/]")
        status_colors = {
            TaskStatus.QUEUED: "yellow",
            TaskStatus.HOLD: "blue",
            TaskStatus.ACTIVE: "cyan",
            TaskStatus.TESTING: "magenta",
            TaskStatus.COMPLETED: "green",
            TaskStatus.AWAITING_CLARIFICATION: "yellow",
        }
        status = f"[{status_colors[task.status]}]{task.status.value}[/]"
        parts = [
            kind,
            f"#{task.id} {escape(task.title_display)}",
            f"[dim]{created}[/]",
            status,
        ]
        if task.verification is VerificationStatus.PASSED:
            parts.append("[green]✓ testing[/]")
        elif task.verification is VerificationStatus.FAILED:
            parts.append("[red]✗ testing[/]")
        left.update("   ".join(parts))
        self._update_ctx_counter(right, task)

    @staticmethod
    def _fmt_ctx_tokens(n: int) -> str:
        if n >= 1000:
            return f"{n / 1000:.1f}".rstrip("0").rstrip(".") + "k"
        return str(n)

    def _update_ctx_counter(self, right: Static, task: Task) -> None:
        """Right corner of the task status bar: current context fill in
        tokens and percent of the allowed maximum.

        Source of truth: the latest model request usage. For a run in
        progress — the live value; otherwise (finished task, history view)
        — the value computed from the persisted transcript, i.e. the final
        value is saved in the task history together with the per-response
        usage and survives restarts.
        """
        right.remove_class("ctx-warn", "ctx-crit")
        tokens = self._live_ctx_tokens.get(task.id)
        if tokens is None:
            tokens = last_context_tokens(task.messages)
        if not tokens:
            right.update("")
            return
        max_tokens = self.resolved_config(self._task_role(task)).max_context_tokens
        if max_tokens > 0:
            pct = round(tokens * 100 / max_tokens)
            text = (
                f"🧠 {self._fmt_ctx_tokens(tokens)}/"
                f"{self._fmt_ctx_tokens(max_tokens)}·{pct}%"
            )
            # Red from the compaction threshold, yellow when the remaining
            # headroom is twice the red-zone headroom: threshold 80 →
            # yellow from 60, threshold 90 → yellow from 80. The threshold
            # follows the task's serving role's resolved config (research
            # tasks use the research model), so a per-model compaction
            # limit from the catalog is reflected here too.
            role_config = self.resolved_config(self._task_role(task))
            threshold = min(max(role_config.compaction_threshold, 1), 100)
            warn_pct = 2 * threshold - 100
            if pct >= threshold:
                right.add_class("ctx-crit")
            elif warn_pct > 0 and pct >= warn_pct:
                right.add_class("ctx-warn")
        else:
            text = f"🧠 {self._fmt_ctx_tokens(tokens)}"
        right.update(text)

    # -- agent run -----------------------------------------------------------------
