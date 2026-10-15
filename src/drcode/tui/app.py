from __future__ import annotations

import asyncio
import ssl
from pathlib import Path
from typing import ClassVar

import httpx
from pydantic_ai import Agent
from pydantic_ai.messages import FunctionToolCallEvent, PartDeltaEvent, TextPartDelta
from rich.console import RenderableType
from rich.markdown import Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
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
    load_messages,
    serialize_messages,
)
from ..config import Config
from ..models import Clarification, Task, TaskKind, TaskStatus, VerificationStatus
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


class DrCodeApp(App[None]):
    CSS = """
    #main { height: 1fr; }
    #sidebar-col { width: 36; border-right: solid $primary; }
    #sidebar { height: 1fr; border: none; }
    #model-status { dock: bottom; height: auto; padding: 0 1; }
    #model-status.ok { color: $success; }
    #model-status.warn { color: $warning; }
    #model-status.error { color: $error; }
    #chat-col { height: 1fr; }
    #task-statusbar { dock: top; height: 1; background: $panel; color: $text; padding: 0 1; }
    #chat-log { height: 1fr; border: none; background: $surface; }
    #stream-label { display: none; padding: 0 1; color: $accent; }
    #stream-label.visible { display: block; }
    #statusbar { dock: bottom; height: 1; background: $panel; color: $text; padding: 0 1; }
    .section-header { color: $accent; text-style: bold; padding: 0 1; }
    .task-item { padding: 0 1; }
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
    #nt-kind-label {
        margin-top: 1;
        text-style: bold;
    }
    #nt-kind-label.kind-feature { color: $success; }
    #nt-kind-label.kind-bug { color: $error; }
    #nt-hint { color: $text-disabled; }
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

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="sidebar-col"):
                yield ListView(id="sidebar")
                yield Static("", id="model-status")
            with Vertical(id="chat-col"):
                yield Static("", id="task-statusbar")
                yield RichLog(id="chat-log", markup=True, wrap=True, highlight=False)
                yield Static("", id="stream-label")
                yield Input(placeholder="Type a message…", id="chat-input")
                with Vertical(id="new-task", classes="hidden"):
                    yield NewTaskInput(placeholder="Describe the task…", id="nt-input")
                    yield Static("type: feature", id="nt-kind-label")
                    yield Static(
                        "tab — switch type   ctrl+enter — create   esc — cancel",
                        id="nt-hint",
                    )
        yield Static("", id="statusbar")
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
        # Resume after restart: the startup demotion (active -> queued) leaves
        # pending_input stranded unless the queue is promoted here.
        promoted = self.manager.promote_next()
        if promoted:
            self.run_agent(promoted.id)

    # -- widgets helpers ------------------------------------------------------

    @property
    def chat_log(self) -> RichLog:
        return self.query_one("#chat-log", RichLog)

    @property
    def stream_label(self) -> Static:
        return self.query_one("#stream-label", Static)

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
    def viewed(self) -> Task | None:
        return self.manager.tasks.get(self.viewed_id) if self.viewed_id else None

    # -- sidebar ---------------------------------------------------------------

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
                if t.kind is TaskKind.BUG:
                    marker = "🐞"
                else:
                    marker = {
                        "queued": "⏳",
                        "active": "▶",
                        "completed": "✓",
                        "clarification": "❓",
                    }.get(t.status.value, "⏳")
                if t.verification is VerificationStatus.PASSED:
                    suffix = " ✅"
                elif t.verification is VerificationStatus.FAILED:
                    suffix = " ❌"
                else:
                    suffix = ""
                viewed = " ●" if t.id == self.viewed_id else ""
                sidebar.append(
                    TaskItem(
                        f"{marker} #{t.id} {t.title_display}{suffix}{viewed}", t.id
                    )
                )
        sidebar.refresh()

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

    # -- chat pane ---------------------------------------------------------------

    def view_task(self, task_id: str) -> None:
        self.viewed_id = task_id
        task = self.manager.get(task_id)
        self.chat_log.clear()
        self._chat_has_content = False
        self._pending_rendered = []
        self.stream_label.update("")
        for role, text in extract_display(task.messages):
            self._write_history_line(role, text)
        for pending in task.pending_input:
            self._write_history_line("user-pending", pending)
        if task.status is TaskStatus.AWAITING_CLARIFICATION and task.clarification:
            self._write_clarification(task.clarification)
        self.refresh_sidebar()
        self.refresh_status()

    def _user_block(self, text: str) -> RenderableType:
        variables = self.get_css_variables()
        bg = variables.get("panel", "blue")
        accent = variables.get("accent", "cyan")
        grid = Table.grid(expand=True)
        grid.add_column(width=1)
        grid.add_column(ratio=1)
        body = Text(text, style=f"on {bg}")
        body_lines = max(1, len(text.splitlines()))
        border = Text("\n".join("▌" for _ in range(body_lines)), style=accent)
        grid.add_row(border, Padding(body, (0, 2), style=f"on {bg}"))
        # Vertical padding is filled with the background too, so the block
        # visually extends one line above and one line below the text.
        return Padding(grid, (1, 2), style=f"on {bg}")

    def _write_chat(self, content: RenderableType, *, expand: bool = False) -> None:
        """Write one block to the chat log, separated by a blank line."""
        if self._chat_has_content:
            self.chat_log.write("")
        self.chat_log.write(content, expand=expand)
        self._chat_has_content = True

    def _write_history_line(self, role: str, text: str) -> None:
        if role == "user":
            self._write_chat(self._user_block(text), expand=True)
        elif role == "user-pending":
            self._pending_rendered.append(text)
            self._write_chat(self._user_block(text), expand=True)
        elif role == "assistant":
            self._write_chat(Markdown(text, code_theme="ansi_dark"))
        elif role == "tester":
            self._write_chat(f"[bold magenta]tester ›[/] {text}")
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
        state = "running" if self.agent_running else "idle"
        tester = "running" if self.tester_running else "idle"
        who = f"#{active.id} {active.title_display}" if active else "none"
        model = self.config.model or "(model not configured)"
        self.query_one("#statusbar", Static).update(
            f" active: {who}   queue: {queue_len}   agent: {state}   "
            f"tester: {tester}   model: {model} "
        )
        self.refresh_task_status()

    def refresh_task_status(self) -> None:
        bar = self.query_one("#task-statusbar", Static)
        task = self.viewed
        if task is None:
            bar.update("")
            return
        created = task.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
        parts = [
            f"#{task.id} {task.title_display}",
            f"created: {created}",
            f"status: {task.status.value}",
        ]
        if task.verification is VerificationStatus.PASSED:
            parts.append("✅ тестирование пройдено")
        elif task.verification is VerificationStatus.FAILED:
            parts.append("❌ тестирование провалено")
        bar.update("   ".join(parts))

    # -- agent run -----------------------------------------------------------------

    @work(exclusive=True, group="agent")
    async def run_agent(self, task_id: str) -> None:
        self._agent_generation += 1
        generation = self._agent_generation
        task = self.manager.get(task_id)
        try:
            if not self.config.is_ready:
                if self.viewed_id == task_id:
                    self._write_history_line(
                        "system",
                        "Model is not configured. Set DRCODE_MODEL, DRCODE_BASE_URL "
                        "and DRCODE_API_KEY (env vars or .env); the message is kept "
                        "and will be sent once the model is available.",
                    )
                return
            self.agent_running = True
            self.refresh_status()
            agent = build_agent(self.config)
            while task.pending_input and task.status is TaskStatus.ACTIVE:
                text = task.pending_input.pop(0)
                history = load_messages(task.messages)
                buffer: list[str] = []
                if self.viewed_id == task_id:
                    if text in self._pending_rendered:
                        # Already shown as a queued block; don't repeat it.
                        self._pending_rendered.remove(text)
                    else:
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
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        if isinstance(
                                            event, PartDeltaEvent
                                        ) and isinstance(event.delta, TextPartDelta):
                                            buffer.append(event.delta.content_delta)
                                            if self.viewed_id == task_id:
                                                self.stream_label.add_class("visible")
                                                self.stream_label.update(
                                                    "".join(buffer)
                                                )
                            elif Agent.is_call_tools_node(node):
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        if self.viewed_id != task_id:
                                            continue
                                        if isinstance(event, FunctionToolCallEvent):
                                            self._write_chat(
                                                f"[dim]⚙ {event.part.tool_name}[/]"
                                            )
                    if run.result is not None:
                        task.messages = serialize_messages(run.result.all_messages())
                        self.manager.storage.save(task)
                    if self.viewed_id == task_id:
                        self.stream_label.remove_class("visible")
                        self._write_history_line("assistant", "".join(buffer))
                        self.stream_label.update("")
                except asyncio.CancelledError:
                    # Interrupted (abort or a new comment): keep the message so
                    # the agent retries it on the next activation.
                    task.pending_input.insert(0, text)
                    self.manager.storage.save(task)
                    raise
                except Exception as e:  # noqa: BLE001 - show agent errors in the chat
                    if self.viewed_id == task_id:
                        self._write_history_line("system", f"{type(e).__name__}: {e}")
                    task.pending_input.insert(
                        0, text
                    )  # not lost, retry on next activation
                    break
        finally:
            if generation == self._agent_generation:
                self.agent_running = False
                self.refresh_status()
                if task.status is TaskStatus.AWAITING_CLARIFICATION:
                    # The agent paused the task on a clarifying question: show it
                    # and let the next queued task run while the user answers.
                    if self.viewed_id == task_id and task.clarification:
                        self._write_clarification(task.clarification)
                    self.refresh_sidebar()
                    promoted = self.manager.promote_next()
                    if promoted:
                        self.run_agent(promoted.id)
                        self.refresh_status()
                elif task.status is TaskStatus.ACTIVE and not task.pending_input:
                    # The agent answered and nothing is pending: the task is
                    # done automatically (as if the user pressed done). The
                    # tester promotes the next queued task when it finishes.
                    self.manager.mark_done(task.id)
                    self.refresh_sidebar()
                    if self.viewed_id == task_id:
                        self.view_task(task.id)
                    self.run_tester(task.id)

    # -- tester run ------------------------------------------------------------------

    @work(exclusive=True, group="tester")
    async def run_tester(self, task_id: str) -> None:
        """After any task completes, verify the project and label the task."""
        task = self.manager.get(task_id)
        try:
            if not self.config.is_ready:
                if self.viewed_id == task_id:
                    self._write_history_line(
                        "system",
                        "Tester skipped: model is not configured. Set DRCODE_MODEL, "
                        "DRCODE_BASE_URL and DRCODE_API_KEY (env vars or .env).",
                    )
                return
            self.tester_running = True
            self.refresh_status()
            if self.viewed_id == task_id:
                self._write_history_line(
                    "tester", f"Verifying task #{task.id}: {task.title}"
                )
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
                                    if self.viewed_id == task_id and isinstance(
                                        event, FunctionToolCallEvent
                                    ):
                                        self._write_chat(
                                            f"[dim]⚙ tester: {event.part.tool_name}[/]"
                                        )
                if run.result is not None and self.viewed_id == task_id:
                    self._write_history_line("tester", run.result.output)
            except Exception as e:  # noqa: BLE001 - show tester errors in the chat
                if self.viewed_id == task_id:
                    self._write_history_line(
                        "system", f"tester failed: {type(e).__name__}: {e}"
                    )
        finally:
            self.tester_running = False
            self.refresh_status()
            self.refresh_sidebar()
            # Tester finished: pick the next available task from the queue.
            promoted = self.manager.promote_next()
            if promoted:
                self.run_agent(promoted.id)
                self.refresh_status()

    # -- actions ---------------------------------------------------------------------

    # -- new task mode ---------------------------------------------------------------

    @property
    def new_task_input(self) -> NewTaskInput:
        return self.query_one("#nt-input", NewTaskInput)

    def action_new_task(self) -> None:
        self.chat_log.add_class("hidden")
        self.stream_label.add_class("hidden")
        self.chat_input.add_class("hidden")
        self.query_one("#new-task", Vertical).remove_class("hidden")
        self._update_kind_label()
        self.new_task_input.focus()

    def _leave_new_task_mode(self) -> None:
        self.query_one("#new-task", Vertical).add_class("hidden")
        self.chat_log.remove_class("hidden")
        self.stream_label.remove_class("hidden")
        self.chat_input.remove_class("hidden")

    def cancel_new_task(self) -> None:
        self._leave_new_task_mode()
        self.chat_input.focus()

    def toggle_new_task_kind(self) -> None:
        self.new_task_kind = (
            TaskKind.BUG if self.new_task_kind is TaskKind.FEATURE else TaskKind.FEATURE
        )
        self._update_kind_label()

    def _update_kind_label(self) -> None:
        label = self.query_one("#nt-kind-label", Static)
        if self.new_task_kind is TaskKind.BUG:
            label.update("type: bug")
            label.remove_class("kind-feature")
            label.add_class("kind-bug")
        else:
            label.update("type: feature")
            label.remove_class("kind-bug")
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
        # A task with a description and no active task starts immediately.
        promoted = (
            task if task.status is TaskStatus.ACTIVE else self.manager.promote_next()
        )
        if promoted:
            self.run_agent(promoted.id)
            self.refresh_status()

    def action_done_task(self) -> None:
        task = self.viewed
        if task is None or task.status is TaskStatus.COMPLETED:
            return
        self.workers.cancel_group(self, "agent")
        self.agent_running = False
        self.manager.mark_done(task.id)
        self.refresh_sidebar()
        self.view_task(task.id)
        # Completed: run the tester, it promotes the next queued task when done.
        self.run_tester(task.id)

    def action_abort_run(self) -> None:
        if self.agent_running or self.tester_running:
            self.workers.cancel_group(self, "agent")
            self.workers.cancel_group(self, "tester")
            self.agent_running = False
            self.tester_running = False
            self.stream_label.remove_class("visible")
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
        was_active = task.status is TaskStatus.ACTIVE
        if was_active:
            # A comment to the running task interrupts it: the exclusive
            # "agent" group cancels the in-flight run (its message goes back
            # to pending_input) and the new run resumes with the comment.
            self.stream_label.remove_class("visible")
            self.stream_label.update("")
            self.run_agent(task.id)
        self.refresh_sidebar()
        self.refresh_status()
        # For an active task run_agent() renders the message itself.
        if not was_active and self.viewed_id == task.id:
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
