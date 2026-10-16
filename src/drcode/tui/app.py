from __future__ import annotations

import asyncio
import json
import shutil
import ssl
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar

import httpx
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRun
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    TextPartDelta,
    ThinkingPartDelta,
)
from rich.console import RenderableType
from rich.markdown import Markdown
from rich.markup import escape
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
    TextArea,
)

from ..agent import (
    USER_AGENT,
    AgentDeps,
    TesterDeps,
    build_agent,
    build_tester_agent,
    extract_display,
    last_context_tokens,
    resumable_history,
    run_context_tokens,
    serialize_messages,
    strip_retried_prompt,
)
from ..config import Config
from ..models import (
    Clarification,
    Task,
    TaskEvent,
    TaskKind,
    TaskStatus,
    VerificationStatus,
)
from ..task_manager import TaskManager


class TaskItem(ListItem):
    def __init__(
        self, text: str, task_id: str | None = None, *, header: bool = False
    ) -> None:
        super().__init__(
            Label(text, classes="section-header" if header else "task-item")
        )
        self.task_id = task_id


class NewTaskInput(TextArea):
    """Multi-line editor for a new task; Tab toggles the task type."""

    BINDINGS: ClassVar = [
        Binding("tab", "toggle_kind", "Toggle type", priority=True),
        Binding("ctrl+enter", "submit_new_task", "Create", priority=True),
        Binding("escape", "cancel_new_task", "Cancel", priority=True),
    ]

    def _app(self) -> DrCodeApp:
        app = self.app
        assert isinstance(app, DrCodeApp)
        return app

    def action_toggle_kind(self) -> None:
        self._app().toggle_new_task_kind()

    def action_submit_new_task(self) -> None:
        self._app().submit_new_task()

    def action_cancel_new_task(self) -> None:
        self._app().cancel_new_task()


class UserBlock(Static):
    """User message: full-width background with an accent bar on the left.

    Plain Text content (not a rich Table) so selection extraction stays clean.
    """

    DEFAULT_CSS = """
    UserBlock {
        background: $panel;
        border-left: thick $accent;
        padding: 1 2;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(Text(text), expand=True)


class ReasoningBlock(Static):
    """Collapsible model-reasoning block; collapsed by default, click to toggle.

    While the model is thinking, `append()` streams text into the block.
    """

    def __init__(self) -> None:
        super().__init__(classes="reasoning collapsed")
        self._content = ""
        self._streaming = False

    def begin(self) -> None:
        self._streaming = True
        self._redraw()

    def append(self, delta: str) -> None:
        self._content += delta
        self._redraw()

    def set_content(self, text: str) -> None:
        """Fill from history (not streaming)."""
        self._content = text
        self._redraw()

    def finish(self) -> None:
        self._streaming = False
        self._redraw()

    def on_click(self) -> None:
        self.toggle_class("collapsed")
        self._redraw()

    def _redraw(self) -> None:
        collapsed = self.has_class("collapsed")
        arrow = "▸" if collapsed else "▾"
        status = "…" if self._streaming else ""
        header = f"[bold dim]{arrow} Reasoning{status}[/]"
        if collapsed:
            self.update(header)
        else:
            body = escape(self._content) or "[dim italic](empty)[/]"
            self.update(f"{header}\n{body}")


class AssistantBlock(Static):
    """Live assistant answer, streamed directly into the chat log.

    Grows in place like ReasoningBlock, so the text appears between the
    reasoning and tool-call blocks while the model is still producing it
    (instead of a preview panel under the chat log).
    """

    def __init__(self) -> None:
        super().__init__(classes="assistant")
        self._content = ""

    def append(self, delta: str) -> None:
        self._content += delta
        self.update(Markdown(self._content, code_theme="ansi_dark"))


class ToolCallBlock(Static):
    """Collapsible tool-call block; collapsed by default, click to toggle.

    Header shows only the tool name ("⚙ grep_search") in gray; expanding
    reveals the arguments, also in gray.
    """

    def __init__(self, call: str) -> None:
        super().__init__(classes="tool-call collapsed")
        self._header, self._args = self._split(call)
        self._redraw()

    @staticmethod
    def _split(call: str) -> tuple[str, str]:
        """Split "name(args)" (possibly with a prefix like "tester: ") into
        (header, args). Falls back to (call, "") if the shape is unexpected."""
        open_idx = call.find("(")
        if open_idx != -1 and call.endswith(")"):
            return call[:open_idx].rstrip(), call[open_idx + 1 : -1]
        return call, ""

    def on_click(self) -> None:
        self.toggle_class("collapsed")
        self._redraw()

    def _redraw(self) -> None:
        arrow = "▸" if self.has_class("collapsed") else "▾"
        header = f"{arrow} ⚙ {escape(self._header)}"
        if self.has_class("collapsed") or not self._args:
            self.update(header)
        else:
            self.update(f"{header}\n{escape(self._args)}")


class DrCodeApp(App[None]):
    CSS = """
    Screen { layers: base toast; }
    #main { height: 1fr; }
    #sidebar-col { width: 36; border-right: solid $primary; }
    #sidebar { height: 1fr; border: none; }
    # Sidebar bottom: model-status then activity-status, plain flow
    # (no dock) so they always stack in compose order under the list.
    #model-status { height: auto; padding: 0 1; }
    #model-status.ok { color: $success; }
    #model-status.warn { color: $warning; }
    #model-status.error { color: $error; }
    #activity-status { height: auto; padding: 0 1; color: $text-disabled; }
    #activity-status.fresh { color: $success; }
    #activity-status.stale { color: $warning; }
    #chat-col { height: 1fr; }
    #task-statusbar { dock: top; height: 1; background: $panel; padding: 0 1; }
    #task-status-left { width: 1fr; color: $text; }
    #task-status-right { width: auto; color: $text; }
    # Context counter turns amber near the limit and red at it.
    #task-status-right.ctx-warn { color: $warning; }
    #task-status-right.ctx-crit { color: $error; }
    # 1-column empty gutter on both sides: no text sticks to the window edge.
    #chat-log { height: 1fr; border: none; background: $surface; padding: 0 1; }
    #chat-log > * { margin-bottom: 1; }
    .reasoning { padding: 0 4; color: $text-muted; }
    .reasoning:hover { background: $panel; }
    .tool-call { padding: 0 4; color: $text-muted; }
    .tool-call:hover { background: $panel; }
    #statusbar { dock: bottom; height: 1; background: $panel; color: $text; padding: 0 1; }
    .section-header { color: $accent; text-style: bold; padding: 0 1; }
    .task-item { padding: 0 1; }
    # Highlight for the task currently open in the chat pane.
    TaskItem.viewed { background: $panel; }
    TaskItem.viewed > .task-item { text-style: bold; }
    #copy-toast {
        layer: toast;
        dock: bottom;
        width: auto;
        height: auto;
        margin: 0 0 3 1;
        padding: 0 2;
        background: $panel;
        border: solid $success;
        color: $text;
    }
    .hidden { display: none; }
    #new-task {
        height: 1fr; width: 1fr;
        align: center middle;
    }
    #new-task.hidden { display: none; }
    #nt-input {
        width: 80%; height: 10;
        border: solid $primary;
    }
    # Labels must stay width: auto: a full-width (1fr) sibling disables
    # horizontal centering for the whole #new-task container, leaving the
    # input glued to the left edge.
    #nt-kind-label {
        width: auto;
        margin-top: 1;
        text-style: bold;
    }
    #nt-kind-label.kind-feature { color: $success; }
    #nt-kind-label.kind-bug { color: $error; }
    #nt-kind-label.kind-documentation { color: yellow; }
    #nt-hint { width: auto; color: $text-disabled; }
    """
    BINDINGS: ClassVar = [
        ("ctrl+n", "new_task", "New task"),
        ("ctrl+d", "done_task", "Done"),
        ("ctrl+o", "focus_sidebar", "Tasks"),
        ("ctrl+i", "focus_input", "Input"),
        ("escape", "abort_run", "Stop agent"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.config = Config()
        self.manager = TaskManager(project_root)
        self.viewed_id: str | None = None
        self.agent_running = False
        self.tester_running = False
        self.new_task_kind: TaskKind = TaskKind.FEATURE
        # Incremented on every run_agent() start; the finally block of a
        # superseded (cancelled) run must not touch shared state.
        self._agent_generation = 0
        # Pending messages already drawn in the current chat view, so the
        # agent run doesn't re-render them when it consumes them.
        self._pending_rendered: list[str] = []
        # Whether the chat log currently holds at least one block.
        self._chat_has_content = False
        # Auto-hide timer for the "copied" toast.
        self._copy_toast_timer: Timer | None = None
        # Per-task tester failure tracking: consecutive failure count and
        # the earliest monotonic time the task may be retried. A failed
        # tester run is retried with backoff, not dropped: tasks in TESTING
        # block the whole pipeline, so they must eventually get their round.
        self._tester_fail_count: dict[str, int] = {}
        self._tester_retry_at: dict[str, float] = {}
        # Same idea for agent runs: a task whose run crashed (e.g. the model
        # gateway is unreachable) is demoted to QUEUED and skipped by
        # promote_next for the rest of the session, so one broken task (or a
        # dead gateway) doesn't loop the whole queue through the error.
        self._agent_failed: set[str] = set()
        # True while the new-task form covers the chat view. Must exist from
        # the start: the streaming loop reads it on every text delta, and an
        # unset attribute would crash the run (it did — the AttributeError
        # surfaced as a red "error ›" line and killed the first streamed run
        # after a startup resume).
        self._new_task_mode = False
        # Snapshot of the task state the sidebar was last rendered from; the
        # poller compares against it to redraw only on actual change.
        self._sidebar_snapshot: tuple = ()
        # Last moment an agent/tester event (stream delta, tool call/result)
        # was seen; the activity label compares it against now. Zero means
        # the current run has not emitted anything yet.
        self._last_activity = 0.0
        # Tool the run is currently waiting on (name at call start, cleared
        # on the result event) — lets the label distinguish "tool busy" from
        # "total silence".
        self._current_tool: str | None = None
        # Context fill (tokens) seen by the latest completed model request,
        # per task, updated live while the run streams. The stored transcript
        # is only refreshed at persist points, so the status bar reads the
        # live value first and falls back to the persisted one (the final
        # value of a finished task — it stays in the history via the saved
        # per-response usage).
        self._live_ctx_tokens: dict[str, int] = {}
        # Context fill (tokens) seen by the latest model request per task,
        # updated live while the agent runs. The task transcript only picks
        # this up at persist points, so the status bar reads from here
        # first and falls back to the stored transcript (final value).
        self._live_ctx_tokens: dict[str, int] = {}

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="sidebar-col"):
                yield ListView(id="sidebar")
                yield Static("", id="model-status")
                yield Static("", id="activity-status")
            with Vertical(id="chat-col"):
                with Horizontal(id="task-statusbar"):
                    yield Static("", id="task-status-left")
                    yield Static("", id="task-status-right")
                yield VerticalScroll(id="chat-log")
                yield Input(placeholder="Type a message…", id="chat-input")
                with Vertical(id="new-task", classes="hidden"):
                    yield NewTaskInput(placeholder="Describe the task…", id="nt-input")
                    yield Static("type: feature", id="nt-kind-label")
                    yield Static(
                        "tab — switch type   ctrl+enter — create   esc — cancel",
                        id="nt-hint",
                    )
        yield Static("", id="statusbar")
        yield Static("", id="copy-toast", classes="hidden")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_sidebar()
        tasks = self.manager.all_tasks()
        active = self.manager.active
        initial = active or (tasks[0] if tasks else None)
        if initial:
            self.view_task(initial.id)
        else:
            self.action_new_task()
        self.refresh_status()
        self.refresh_connection()
        self.set_interval(30, self.refresh_connection)
        # The sidebar must follow task-status changes on its own: statuses
        # also mutate mid-run (agent tools, promote_next in the worker
        # finally-blocks) where no one calls refresh_sidebar().
        self.set_interval(1.0, self._poll_sidebar)
        self.set_interval(1.0, self._poll_activity)
        # Resume after restart. _advance() re-tests TESTING tasks (the app
        # died during testing) FIRST — the startup demotion (active ->
        # queued) leaves pending_input stranded unless the queue is
        # promoted here.
        self._advance()

    # -- widgets helpers ------------------------------------------------------

    @property
    def chat_log(self) -> VerticalScroll:
        return self.query_one("#chat-log", VerticalScroll)

    @property
    def chat_input(self) -> Input:
        return self.query_one("#chat-input", Input)

    @property
    def sidebar(self) -> ListView:
        return self.query_one("#sidebar", ListView)

    @property
    def model_status(self) -> Static:
        return self.query_one("#model-status", Static)

    @property
    def activity_status(self) -> Static:
        return self.query_one("#activity-status", Static)

    @property
    def viewed(self) -> Task | None:
        return self.manager.tasks.get(self.viewed_id) if self.viewed_id else None

    # -- copy on select -------------------------------------------------------

    def _selected_text(self) -> str | None:
        """Extract the currently selected text from rendered lines.

        Done at the app level instead of Widget.get_selection, which works on
        logical (unwrapped) text and returns garbled results for wrapped lines.
        """
        parts: list[str] = []
        for widget, selection in self.screen.selections.items():
            if not widget.is_attached:
                continue
            lines: list[str] = []
            for y in range(widget.size.height):
                span = selection.get_span(y)
                if span is None:
                    continue
                start, end = span
                line = widget.render_line(y).text
                lines.append((line[start:] if end == -1 else line[start:end]).rstrip())
            if lines:
                parts.append("\n".join(lines).rstrip())
        if not parts:
            return None
        return "\n".join(parts)

    def _copy_with_fallback(self, text: str) -> None:
        # copy_to_clipboard uses OSC52, which macOS Terminal.app ignores;
        # pbcopy always works there.
        self.copy_to_clipboard(text)
        if sys.platform == "darwin" and shutil.which("pbcopy"):
            subprocess.run(["pbcopy"], input=text.encode(), check=False)

    def _show_copy_toast(self, text: str) -> None:
        preview = escape(text if len(text) <= 40 else text[:39] + "…")
        toast = self.query_one("#copy-toast", Static)
        toast.update(f"✓ copied: {preview}")
        toast.remove_class("hidden")
        if self._copy_toast_timer is not None:
            self._copy_toast_timer.stop()
        self._copy_toast_timer = self.set_timer(2.0, lambda: toast.add_class("hidden"))

    def on_text_selected(self, event: events.TextSelected) -> None:
        text = self._selected_text()
        if not text:
            return
        self._copy_with_fallback(text)
        self._show_copy_toast(text)

    # -- sidebar ---------------------------------------------------------------

    def _sidebar_fingerprint(self) -> tuple:
        """Lightweight snapshot of everything the sidebar renders: status,
        verification label, title and the viewed-marker per task."""
        return tuple(
            (
                t.id,
                t.status.value,
                t.verification,
                t.title_display,
                t.id == self.viewed_id,
            )
            for t in self.manager.all_tasks()
        )

    def _poll_sidebar(self) -> None:
        """Redraw the sidebar only when the underlying task state changed
        since the last render (status transitions, new bug reports from the
        tester, promote_next() activations, …)."""
        if self._sidebar_fingerprint() != self._sidebar_snapshot:
            self.refresh_sidebar()

    def refresh_sidebar(self) -> None:
        sidebar = self.sidebar
        sidebar.clear()
        sections: list[tuple[str, list[Task]]] = [
            (
                "ACTIVE",
                [t for t in self.manager.all_tasks() if t.status is TaskStatus.ACTIVE],
            ),
            (
                "NEEDS CLARIFICATION",
                [
                    t
                    for t in self.manager.all_tasks()
                    if t.status is TaskStatus.AWAITING_CLARIFICATION
                ],
            ),
            ("QUEUE", self.manager.queued()),
            (
                "TESTING",
                [t for t in self.manager.all_tasks() if t.status is TaskStatus.TESTING],
            ),
            (
                "COMPLETED",
                [
                    t
                    for t in self.manager.all_tasks()
                    if t.status is TaskStatus.COMPLETED
                ],
            ),
        ]
        for name, tasks in sections:
            sidebar.append(TaskItem(f"{name} ({len(tasks)})", header=True))
            for t in tasks:
                marker = {
                    "queued": "⏳",
                    "active": "▶",
                    "testing": "🧪",
                    "completed": "✓",
                    "clarification": "❓",
                }.get(t.status.value, "⏳")
                kind_letter = {
                    TaskKind.BUG: "[red]B[/]",
                    TaskKind.DOCUMENTATION: "[yellow]D[/]",
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

    # -- model connection status ---------------------------------------------

    def refresh_connection(self) -> None:
        self.run_worker(self._check_model(), exclusive=True, group="model-check")

    async def _check_model(self) -> None:
        label = self.model_status
        if not self.config.is_ready:
            label.remove_class("ok", "error")
            label.add_class("warn")
            label.update("○ model not configured — set DRCODE_MODEL / DRCODE_API_KEY")
            return
        model = self.config.model
        # Системное хранилище сертификатов: корпоративные CA (как у
        # api-copilot.x5.ru) отсутствуют в certifi, из-за чего httpx даёт
        # CERTIFICATE_VERIFY_FAILED, который оборачивается в ConnectError.
        try:
            import truststore

            verify: ssl.SSLContext | bool = truststore.SSLContext(
                ssl.PROTOCOL_TLS_CLIENT
            )
        except ImportError:
            verify = True
        try:
            async with httpx.AsyncClient(timeout=5, verify=verify) as client:
                # Зонд через chat/completions, а не /models: этот гейтвей не
                # отдаёт /models (503). Пустой messages даёт 400 от бэкенда —
                # значит TLS, ключ, UA и имя модели приняты.
                resp = await client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "User-Agent": USER_AGENT,
                    },
                    json={"model": model, "messages": []},
                )
        except httpx.HTTPError as e:
            label.remove_class("ok", "warn")
            label.add_class("error")
            label.update(f"✗ {model} — unavailable ({type(e).__name__})")
            return
        label.remove_class("warn", "error")
        if resp.status_code in (401, 403):
            label.add_class("error")
            label.update(f"✗ {model} — auth failed (HTTP {resp.status_code})")
        elif resp.status_code < 500:
            label.add_class("ok")
            label.update(f"● {model} — connected")
        else:
            label.add_class("error")
            label.update(f"✗ {model} — unavailable (HTTP {resp.status_code})")

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
        label = self.activity_status
        if not (self.agent_running or self.tester_running):
            active = self.manager.active
            if active is not None:
                # Task claims to be active but no worker owns it (e.g. the
                # user aborted the run, or an older build stranded it):
                # surface the inconsistency instead of showing a fake idle.
                label.remove_class("fresh")
                label.add_class("stale")
                label.update(
                    f"⚠ #{active.id} active, nothing running — "
                    "send a message to (re)start"
                )
                return
            label.remove_class("fresh", "stale")
            label.update("○ idle")
            return
        who = "agent" if self.agent_running else "tester"
        age = int(time.monotonic() - self._last_activity)
        if self._current_tool:
            detail = f"tool {self._current_tool} running {self._fmt_age(age)}"
        else:
            detail = f"last event {self._fmt_age(age)} ago"
        if age < 30:
            label.remove_class("stale")
            label.add_class("fresh")
            label.update(f"▶ {who}: working · {detail}")
        else:
            label.remove_class("fresh")
            label.add_class("stale")
            label.update(f"⚠ {who}: stalled? · {detail} (esc to stop)")

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

    def _next_testing(self) -> Task | None:
        """Oldest task in TESTING (crash/abort mid-test) whose retry backoff
        has expired. Tasks whose tester run failed recently are skipped
        until their scheduled retry time."""
        now = time.monotonic()
        return next(
            (
                t
                for t in self.manager.all_tasks()
                if t.status is TaskStatus.TESTING
                and self._tester_retry_at.get(t.id, 0.0) <= now
            ),
            None,
        )

    def _advance(self) -> None:
        """The single pipeline gate: testing always has absolute priority.

        Every path that used to promote the queue goes through here, so a
        new development run can never start while any task is awaiting
        verification. While a tester run is in flight this is a no-op —
        the tester's own completion path advances the pipeline.
        """
        if self.tester_running:
            return
        testing = self._next_testing()
        if testing is not None:
            # Set synchronously: two _advance() calls within one event-loop
            # tick must not start two tester workers (the exclusive group
            # would cancel the first before it does anything).
            self.tester_running = True
            self.run_tester(testing.id)
            return
        if any(t.status is TaskStatus.TESTING for t in self.manager.all_tasks()):
            # Every TESTING task is within its retry backoff window: the
            # rule "no new development while any testing is incomplete"
            # holds even then — the retry timer will call _advance() again.
            return
        promoted = self.manager.promote_next(exclude=self._agent_failed)
        if promoted:
            self.run_agent(promoted.id)
        self.refresh_status()

    def view_task(self, task_id: str) -> None:
        self.viewed_id = task_id
        task = self.manager.get(task_id)
        self.chat_log.remove_children()
        self._chat_has_content = False
        self._pending_rendered = []
        self._clear_stream_label()
        for role, text in extract_display(task.messages):
            self._write_history_line(role, text)
        # Tester rounds come after the development transcript, each behind
        # its own "testing" separator. The tester's synthetic prompt
        # ("Verify completed task #…") is not shown.
        for round_messages in task.tester_messages:
            if not round_messages:
                continue
            self._write_history_line("testing-sep", "")
            for role, text in extract_display(round_messages):
                if role == "user":
                    continue
                if role == "tool-call":
                    text = f"tester: {text}"
                self._write_history_line(role, text)
        # Service events (failures, skips) — human-only, with timestamps.
        if task.events:
            self._write_history_line("events-sep", "")
            for event in task.events:
                self._write_history_line("event", self._format_event(event))
        for pending in task.pending_input:
            self._write_history_line("user-pending", pending)
        if task.status is TaskStatus.AWAITING_CLARIFICATION and task.clarification:
            self._write_clarification(task.clarification)
        # Opening a task always lands on the latest entry (previously this
        # happened implicitly because every mount scrolled to the end).
        self.chat_log.scroll_end(animate=False)
        self.refresh_sidebar()
        self.refresh_status()

    def _chat_at_bottom(self) -> bool:
        """True when the last line of the chat log is in the viewport.

        Tolerance of 1 line absorbs rounding of fractional scroll offsets.
        """
        log = self.chat_log
        return log.scroll_offset.y >= log.max_scroll_y - 1

    def _write_chat(
        self, content: RenderableType | Widget, *, expand: bool = False
    ) -> None:
        """Mount one block to the chat log.

        Scrolls to the end only when the user is already looking at the
        bottom: a scrolled-up reader must not be yanked down by new tool
        calls / reasoning blocks arriving while the agent runs.
        """
        follow = self._chat_at_bottom()
        if isinstance(content, Widget):
            content.add_class("chat-block")
        else:
            content = Static(content, expand=expand, classes="chat-block")
        self.chat_log.mount(content)
        if follow:
            self.chat_log.scroll_end(animate=False)
        self._chat_has_content = True

    def _write_history_line(self, role: str, text: str) -> None:
        if role == "user":
            self._write_chat(UserBlock(text))
        elif role == "user-pending":
            self._pending_rendered.append(text)
            self._write_chat(UserBlock(text))
        elif role == "reasoning":
            block = ReasoningBlock()
            block.set_content(text)
            self._write_chat(block)
        elif role == "tool-call":
            self._write_chat(ToolCallBlock(text))
        elif role == "assistant":
            self._write_chat(Markdown(text, code_theme="ansi_dark"))
        elif role == "tool":
            self._write_chat(f"[dim]{escape(text)}[/]")
        elif role == "tester":
            self._write_chat(f"[bold magenta]tester ›[/] {text}")
        elif role == "testing-sep":
            label = f"testing: {text}" if text else "testing"
            self._write_chat(f"[bold magenta]──── {label} ────[/]")
        elif role == "events-sep":
            self._write_chat("[bold red]──── service events ────[/]")
        elif role == "event":
            self._write_chat(f"[bold red]event ›[/] {text}")
        elif role == "system":
            self._write_chat(f"[bold red]error ›[/] {text}")

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
        model = self.config.model or "(model not configured)"
        self.query_one("#statusbar", Static).update(
            f" active: {who}   queue: {queue_len}   testing: {testing_len}   "
            f"agent: {state}   tester: {tester}   model: {model} "
        )
        self.refresh_task_status()

    def refresh_task_status(self) -> None:
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
            TaskKind.DOCUMENTATION: "[bold yellow]documentation[/]",
        }.get(task.kind, "[bold green]feature[/]")
        status_colors = {
            TaskStatus.QUEUED: "yellow",
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
        max_tokens = self.config.max_context_tokens
        if max_tokens > 0:
            pct = round(tokens * 100 / max_tokens)
            text = (
                f"🧠 {self._fmt_ctx_tokens(tokens)} / "
                f"{self._fmt_ctx_tokens(max_tokens)} · {pct}%"
            )
            if pct >= 95:
                right.add_class("ctx-crit")
            elif pct >= 80:
                right.add_class("ctx-warn")
        else:
            text = f"🧠 {self._fmt_ctx_tokens(tokens)}"
        right.update(text)

    # -- agent run -----------------------------------------------------------------

    def _persist_task_messages(
        self, task: Task, run: AgentRun[AgentDeps, str] | None
    ) -> None:
        """Snapshot the transcript produced so far into the task.

        One agent turn can run for minutes (many model requests and tool
        calls). Saving only on turn completion leaves the stored history
        stale while the agent works: switching to the task then shows
        nothing new (for a fresh task — nothing at all), and an interrupted
        turn (abort, a follow-up comment cancelling the exclusive worker)
        lost everything it had done. Persisting after every completed node
        keeps both the on-disk history and `view_task` current.

        The snapshot may end with an incomplete exchange (tool calls
        without results), which the model API would reject on resume —
        `resumable_history()` trims that tail when the transcript is next
        loaded as history.

        `run` is None when the failure happened before the run context was
        entered (e.g. `agent.iter()` or `resumable_history()` raised); there
        is nothing new to persist then — the pending input is re-queued by
        the caller and retried on the next activation.
        """
        if run is None:
            return
        try:
            messages = run.all_messages()
        except Exception:  # noqa: BLE001 - state may be mid-teardown after cancel
            return
        task.messages = serialize_messages(messages)
        self.manager.storage.save(task)

    def _persist_tester_messages(
        self, task: Task, run: AgentRun[TesterDeps, str] | None
    ) -> None:
        """Snapshot the tester transcript into the task's last testing round.

        Mirrors _persist_task_messages for the tester run: without it the
        tester's reasoning, tool calls and answer would exist only in the
        chat widgets and vanish from the history on the next view_task().
        The round entry is created up front by run_tester; `run` is None
        when the failure happened before the run context was entered.
        """
        if run is None:
            return
        try:
            messages = run.all_messages()
        except Exception:  # noqa: BLE001 - state may be mid-teardown after cancel
            return
        if not task.tester_messages:
            task.tester_messages.append([])
        task.tester_messages[-1] = serialize_messages(messages)
        self.manager.storage.save(task)

    @work(exclusive=True, group="agent")
    async def run_agent(self, task_id: str) -> None:
        self._agent_generation += 1
        generation = self._agent_generation
        task = self.manager.get(task_id)
        # Bound only once `async with agent.iter(...)` is entered; the
        # except handlers below must not touch a run that never started.
        # Initialized before the try: build_agent() itself may raise.
        run: AgentRun[AgentDeps, str] | None = None
        cancelled = False
        try:
            if not self.config.is_ready:
                self._show_event(
                    task,
                    "Model is not configured. Set DRCODE_MODEL, DRCODE_BASE_URL "
                    "and DRCODE_API_KEY (env vars or .env); the message is kept "
                    "and will be sent once the model is available.",
                )
                return
            self.agent_running = True
            self._agent_failed.discard(task_id)
            self._current_tool = None
            self._touch_activity()
            self.refresh_status()
            agent = build_agent(self.config, kind=task.kind)
            while task.pending_input and task.status is TaskStatus.ACTIVE:
                run = None
                text = task.pending_input.pop(0)
                history = resumable_history(task.messages)
                # A retried message may already be present in the persisted
                # transcript (the interrupted run saved its prompt and partial
                # work before re-queueing the text): cut that stale attempt,
                # otherwise the prompt is duplicated in the transcript and
                # drawn a second time at the end of the chat.
                history, is_retry = strip_retried_prompt(history, text)
                buffer: list[str] = []
                if self.viewed_id == task_id:
                    if text in self._pending_rendered:
                        # Already shown as a queued block; don't repeat it.
                        self._pending_rendered.remove(text)
                    elif not is_retry:
                        self._write_history_line("user", text)
                try:
                    async with agent.iter(
                        text,
                        message_history=history,
                        deps=AgentDeps(
                            root=self.project_root,
                            manager=self.manager,
                            task_id=task_id,
                        ),
                    ) as run:
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                reasoning_block: ReasoningBlock | None = None
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        self._touch_activity()
                                        if not isinstance(event, PartDeltaEvent):
                                            continue
                                        delta = event.delta
                                        if (
                                            isinstance(delta, ThinkingPartDelta)
                                            and delta.content_delta
                                        ):
                                            if self.viewed_id == task_id:
                                                if reasoning_block is None:
                                                    reasoning_block = ReasoningBlock()
                                                    reasoning_block.begin()
                                                    self._write_chat(reasoning_block)
                                                    reasoning_block.append(
                                                        delta.content_delta
                                                    )
                                                else:
                                                    # Streaming grows the block
                                                    # without a new mount: keep
                                                    # following only if the tail
                                                    # was visible before the
                                                    # delta arrived.
                                                    follow = self._chat_at_bottom()
                                                    reasoning_block.append(
                                                        delta.content_delta
                                                    )
                                                    if follow:
                                                        self.chat_log.call_after_refresh(
                                                            self.chat_log.scroll_end,
                                                            animate=False,
                                                        )
                                        elif isinstance(delta, TextPartDelta):
                                            # The live preview widget was
                                            # removed: the text is shown after
                                            # the turn as an assistant block.
                                            buffer.append(delta.content_delta)
                                if reasoning_block is not None:
                                    reasoning_block.finish()
                                # The stream is closed: this request's usage
                                # is final. Refresh the context counter in
                                # the task status bar for the running task.
                                ctx_tokens = run_context_tokens(run)
                                if ctx_tokens is not None:
                                    self._live_ctx_tokens[task_id] = ctx_tokens
                                    if self.viewed_id == task_id:
                                        self.refresh_task_status()
                            elif Agent.is_call_tools_node(node):
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        if isinstance(event, FunctionToolCallEvent):
                                            self._touch_activity(event.part.tool_name)
                                        elif isinstance(event, FunctionToolResultEvent):
                                            self._touch_activity(done=True)
                                        if self.viewed_id != task_id:
                                            continue
                                        if isinstance(event, FunctionToolCallEvent):
                                            args = event.part.args
                                            args_str = (
                                                args
                                                if isinstance(args, str)
                                                else json.dumps(
                                                    args or {}, ensure_ascii=False
                                                )
                                            )
                                            self._write_chat(
                                                ToolCallBlock(
                                                    f"{event.part.tool_name}"
                                                    f"({args_str})"
                                                )
                                            )
                    # Keep the stored history current while the turn runs:
                    # view_task() renders from task.messages, so without this
                    # a switch away+back shows the pre-run transcript (for a
                    # fresh task — an empty chat).
                    self._persist_task_messages(task, run)
                    if run.result is not None:
                        task.messages = serialize_messages(run.result.all_messages())
                        self.manager.storage.save(task)
                    if self.viewed_id == task_id:
                        self._clear_stream_label()
                        self._write_history_line("assistant", "".join(buffer))
                except asyncio.CancelledError:
                    # Interrupted (abort or a new comment): keep the message so
                    # the agent retries it on the next activation.
                    cancelled = True
                    task.pending_input.insert(0, text)
                    # Persist whatever the interrupted turn managed to do,
                    # otherwise its whole work survives only in the widgets
                    # and vanishes from the history on the next view_task().
                    self._persist_task_messages(task, run)
                    self.manager.storage.save(task)
                    # Drop our streaming preview; a newer run (if any) owns
                    # the label now and must not be touched.
                    self._drop_stream_preview(generation)
                    raise
                except Exception as e:  # noqa: BLE001 - show agent errors in the chat
                    # Persist the failure even for a superseded run: it
                    # happened, and the human must be able to see it.
                    self._show_event(task, f"{type(e).__name__}: {e}")
                    if generation == self._agent_generation:
                        # A mid-stream failure leaves a half-streamed orange
                        # preview above the input: clear it, otherwise the
                        # stale fragments hang in the dialog (and would be
                        # shown for whatever the user does next).
                        self._drop_stream_preview(generation)
                    task.pending_input.insert(
                        0, text
                    )  # not lost, retried on the next user nudge
                    # Demote honestly: an ACTIVE task with no runner is a
                    # state nothing ever picks up again (promote_next skips
                    # it) — the task goes back to the queue instead, and the
                    # queue continues with the next task.
                    self._persist_task_messages(task, run)
                    task.status = TaskStatus.QUEUED
                    self._agent_failed.add(task_id)
                    self.manager.storage.save(task)
                    break
        finally:
            if generation == self._agent_generation:
                self.agent_running = False
                self.refresh_status()
            # A cancelled run (user abort, or a newer run taking over) must
            # not restart the pipeline: the abort stops everything until the
            # user acts, and the newer run advances on its own completion.
            if not cancelled and generation == self._agent_generation:
                if task.status is TaskStatus.AWAITING_CLARIFICATION:
                    # The user may have answered while the run was still
                    # finishing (the answer sits in pending_input and send()'s
                    # clarification transition never fired): don't wait forever
                    # for an answer that has already arrived.
                    self.manager.requeue_answered_clarification(task.id)
                    # The agent paused the task on a clarifying question: show it
                    # and let the pipeline continue while the user answers.
                    if self.viewed_id == task_id and task.clarification:
                        self._write_clarification(task.clarification)
                    self.refresh_sidebar()
                    self._advance()
                elif task.status is TaskStatus.ACTIVE and not task.pending_input:
                    # The agent answered and nothing is pending: the task is
                    # done automatically (as if the user pressed done).
                    self.manager.mark_done(task.id)
                    self.refresh_sidebar()
                    if self.viewed_id == task_id:
                        self.view_task(task.id)
                    # Testing (this task or an older one in the backlog)
                    # always goes before any new development.
                    self._advance()
                elif task.status is TaskStatus.QUEUED:
                    # The run failed and demoted the task back to the queue:
                    # keep the pipeline going with the next task (failed ones
                    # are excluded, so a dead gateway can't loop the queue).
                    self._advance()

    # -- tester run ------------------------------------------------------------------

    def _schedule_tester_retry(self, task_id: str) -> None:
        """Backoff before the next tester attempt for a failed task.

        TESTING tasks block the whole pipeline, so a failed tester run is
        retried (with capped exponential backoff) instead of being dropped
        from the backlog until restart."""
        count = self._tester_fail_count.get(task_id, 1)
        delay = min(5 * 2 ** (count - 1), 300)
        self._tester_retry_at[task_id] = time.monotonic() + delay
        self.set_timer(delay, self._advance)

    @work(exclusive=True, group="tester")
    async def run_tester(self, task_id: str) -> None:
        """After any task completes, verify the project and label the task."""
        task = self.manager.get(task_id)
        # Initialized before the try: build_tester_agent() itself may raise,
        # and the handlers below reference `run`.
        run: AgentRun[TesterDeps, str] | None = None
        cancelled = False
        try:
            if not self.config.is_ready:
                self._tester_fail_count[task_id] = (
                    self._tester_fail_count.get(task_id, 0) + 1
                )
                self._schedule_tester_retry(task_id)
                self._show_event(
                    task,
                    "Tester skipped: model is not configured. Set DRCODE_MODEL, "
                    "DRCODE_BASE_URL and DRCODE_API_KEY (env vars or .env).",
                )
                return
            # Idempotent: _advance() sets this before starting the worker so
            # two _advance() calls in one tick cannot double-start the tester.
            self.tester_running = True
            self._current_tool = None
            self._touch_activity()
            self.refresh_status()
            # New testing round: created up front, filled by
            # _persist_tester_messages as the run progresses. An empty round
            # (crash before any snapshot) is skipped when rendering.
            task.tester_messages.append([])
            if self.viewed_id == task_id:
                self._write_history_line("testing-sep", "")
            tester = build_tester_agent(self.config)
            deps = TesterDeps(
                root=self.project_root, manager=self.manager, task_id=task_id
            )
            try:
                async with tester.iter(
                    f"Verify completed task #{task.id}: {task.title}", deps=deps
                ) as run:
                    async for node in run:
                        if Agent.is_call_tools_node(node):
                            async with node.stream(run.ctx) as stream:
                                async for event in stream:
                                    if isinstance(event, FunctionToolCallEvent):
                                        self._touch_activity(event.part.tool_name)
                                    elif isinstance(event, FunctionToolResultEvent):
                                        self._touch_activity(done=True)
                                    if self.viewed_id == task_id and isinstance(
                                        event, FunctionToolCallEvent
                                    ):
                                        args = event.part.args
                                        args_str = (
                                            args
                                            if isinstance(args, str)
                                            else json.dumps(
                                                args or {}, ensure_ascii=False
                                            )
                                        )
                                        self._write_chat(
                                            ToolCallBlock(
                                                f"tester: {event.part.tool_name}"
                                                f"({args_str})"
                                            )
                                        )
                        # Keep the stored history current while the test
                        # runs: switching to the task mid-test shows the
                        # tester's work done so far.
                        self._persist_tester_messages(task, run)
                if run.result is not None and self.viewed_id == task_id:
                    # Re-render from the persisted transcript: the streamed
                    # tool-call blocks above are replaced by the full history
                    # view (separator + reasoning + tool calls + answer).
                    self.view_task(task_id)
            except asyncio.CancelledError:
                # Aborted (or superseded): keep whatever the tester did so
                # far, then re-raise so the worker teardown stays clean.
                cancelled = True
                self._persist_tester_messages(task, run)
                raise
            except Exception as e:  # noqa: BLE001 - show tester errors in the chat
                self._persist_tester_messages(task, run)
                self._tester_fail_count[task_id] = (
                    self._tester_fail_count.get(task_id, 0) + 1
                )
                self._schedule_tester_retry(task_id)
                self._show_event(task, f"tester failed: {type(e).__name__}: {e}")
        finally:
            self.tester_running = False
            self.refresh_status()
            self.refresh_sidebar()
            if task.status is TaskStatus.COMPLETED:
                # Success: give the next testing round of this task a fresh
                # backoff budget.
                self._tester_fail_count.pop(task_id, None)
                self._tester_retry_at.pop(task_id, None)
            # A cancelled run (user abort, or a newer tester taking over)
            # must not restart the pipeline.
            if not cancelled:
                self._advance()

    # -- actions ---------------------------------------------------------------------

    # -- new task mode ---------------------------------------------------------------

    @property
    def new_task_input(self) -> NewTaskInput:
        return self.query_one("#nt-input", NewTaskInput)

    def _clear_stream_label(self) -> None:
        """Legacy hook of the removed streaming-preview widget.

        The preview was replaced by in-place assistant blocks; nothing to
        clear — kept as a no-op so abort/new-task paths stay simple.
        """

    def _drop_stream_preview(self, generation: int) -> None:
        """Clear the streaming preview after an agent run ends, but only if
        no newer run has (already) taken over the label."""
        if generation == self._agent_generation:
            self._clear_stream_label()

    def action_new_task(self) -> None:
        # Every entry into the new-task screen starts from a clean editor.
        self.new_task_input.clear()
        self._new_task_mode = True
        # The form is not tied to any task: drop the viewed highlight so the
        # sidebar doesn't single out a task while the user composes a new one.
        self.viewed_id = None
        self.refresh_sidebar()
        # The streaming preview belongs to the chat view, not the form: a
        # run in progress must not leak its orange text over the form.
        self._clear_stream_label()
        self.chat_log.add_class("hidden")
        self.chat_input.add_class("hidden")
        self.query_one("#new-task", Vertical).remove_class("hidden")
        self._update_kind_label()
        self.refresh_task_status()
        self.new_task_input.focus()

    def _leave_new_task_mode(self) -> None:
        self._new_task_mode = False
        self.query_one("#new-task", Vertical).add_class("hidden")
        self.chat_log.remove_class("hidden")
        # Content may be stale (a run streamed while the form was open, or
        # finished in the background): don't resurrect it.
        self._clear_stream_label()
        self.chat_input.remove_class("hidden")
        self.refresh_task_status()

    def cancel_new_task(self) -> None:
        self._leave_new_task_mode()
        self.chat_input.focus()

    # Tab cycles the new-task type through all three kinds.
    _KIND_CYCLE: ClassVar = (TaskKind.FEATURE, TaskKind.BUG, TaskKind.DOCUMENTATION)

    def toggle_new_task_kind(self) -> None:
        idx = self._KIND_CYCLE.index(self.new_task_kind)
        self.new_task_kind = self._KIND_CYCLE[(idx + 1) % len(self._KIND_CYCLE)]
        self._update_kind_label()

    def _update_kind_label(self) -> None:
        label = self.query_one("#nt-kind-label", Static)
        label.remove_class("kind-feature", "kind-bug", "kind-documentation")
        if self.new_task_kind is TaskKind.BUG:
            label.update("type: bug")
            label.add_class("kind-bug")
        elif self.new_task_kind is TaskKind.DOCUMENTATION:
            label.update("type: documentation")
            label.add_class("kind-documentation")
        else:
            label.update("type: feature")
            label.add_class("kind-feature")

    def submit_new_task(self) -> None:
        text = self.new_task_input.text.strip()
        if not text:
            return
        task = self.manager.create("", text, kind=self.new_task_kind)
        self._leave_new_task_mode()
        self.refresh_sidebar()
        self.view_task(task.id)
        self.chat_input.focus()
        # Starts immediately unless testing is pending or the queue is busy.
        self._advance()

    def action_done_task(self) -> None:
        task = self.viewed
        if task is None or task.status is TaskStatus.COMPLETED:
            return
        self.workers.cancel_group(self, "agent")
        self.agent_running = False
        self.manager.mark_done(task.id)
        self.refresh_sidebar()
        self.view_task(task.id)
        # Testing backlog first (possibly an older TESTING task), then queue.
        self._advance()

    def action_abort_run(self) -> None:
        if self.agent_running or self.tester_running:
            self.workers.cancel_group(self, "agent")
            self.workers.cancel_group(self, "tester")
            self.agent_running = False
            self.tester_running = False
            self._current_tool = None
            # Hide AND empty: the aborted run's partial text must not be
            # resurrected by a later add_class("visible").
            self._clear_stream_label()
            self.refresh_status()

    def action_focus_sidebar(self) -> None:
        self.sidebar.focus()

    def action_focus_input(self) -> None:
        self.chat_input.focus()

    # -- events -------------------------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, TaskItem) and item.task_id:
            self.view_task(item.task_id)
            self.chat_input.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "chat-input":
            return
        text = event.value.strip()
        if not text:
            return
        task = self.viewed
        if task is None:
            return
        event.input.clear()
        self._submit_user_text(task, text)

    def _submit_user_text(self, task: Task, text: str) -> None:
        self.manager.send(task.id, text)
        if task.status is TaskStatus.ACTIVE:
            # A comment to the running task interrupts it: the exclusive
            # "agent" group cancels the in-flight run (its message goes back
            # to pending_input) and the new run resumes with the comment.
            self._clear_stream_label()
            self.run_agent(task.id)
        else:
            # Queued/clarification/... input: advance the pipeline — testing
            # backlog first, then the queue.
            self._advance()
        self.refresh_sidebar()
        self.refresh_status()
        # For an active task run_agent() renders the message itself.
        if task.status is not TaskStatus.ACTIVE and self.viewed_id == task.id:
            self._write_history_line("user-pending", text)

    def action_pick_option(self, index: int) -> None:
        """User clicked a clarifying-question option: submit it as their answer."""
        task = self.viewed
        if task is None or task.clarification is None:
            return
        if not 0 <= index < len(task.clarification.options):
            return
        self._submit_user_text(task, task.clarification.options[index])


def run_app(project_root: Path) -> None:
    DrCodeApp(project_root).run()
