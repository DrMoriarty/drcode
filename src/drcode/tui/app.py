from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic_ai import Agent
from pydantic_ai.messages import FunctionToolCallEvent, PartDeltaEvent, TextPartDelta
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Select,
    Static,
    TextArea,
)

from ..agent import (
    TesterDeps,
    build_agent,
    build_tester_agent,
    extract_display,
    load_messages,
    serialize_messages,
)
from ..config import Config
from ..models import Task, TaskKind, TaskStatus, VerificationStatus
from ..task_manager import TaskManager


class TaskItem(ListItem):
    def __init__(
        self, text: str, task_id: str | None = None, *, header: bool = False
    ) -> None:
        super().__init__(
            Label(text, classes="section-header" if header else "task-item")
        )
        self.task_id = task_id


class NewTaskScreen(ModalScreen[Task | None]):
    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self.kind_select = Select[str](
            [
                ("Feature request", TaskKind.FEATURE.value),
                ("Bug report", TaskKind.BUG.value),
            ],
            value=TaskKind.FEATURE.value,
            id="nt-kind",
            allow_blank=False,
            prompt="Task type",
        )
        self.title_input = Input(placeholder="Task title", id="nt-title")
        self.desc_input = TextArea(
            placeholder="Describe the feature request or bug report…", id="nt-desc"
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="nt-dialog"):
            yield Label("New task", id="nt-heading")
            yield self.kind_select
            yield self.title_input
            yield self.desc_input
            with Horizontal(id="nt-buttons"):
                yield Button("Create", variant="primary", id="nt-create")
                yield Button("Cancel", id="nt-cancel")

    def on_mount(self) -> None:
        self.title_input.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.desc_input.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "nt-create":
            self._submit()
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _submit(self) -> None:
        title = self.title_input.value.strip()
        desc = self.desc_input.text.strip()
        if not title and not desc:
            self.dismiss(None)
            return
        app = self.app
        assert isinstance(app, DrCodeApp)
        kind = TaskKind(self.kind_select.value)
        self.dismiss(app.manager.create(title, desc, kind=kind))


class DrCodeApp(App[None]):
    CSS = """
    #main { height: 1fr; }
    #sidebar { width: 36; border-right: solid $primary; }
    #chat-col { height: 1fr; }
    #chat-log { height: 1fr; border: none; background: $surface; }
    #stream-label { display: none; padding: 0 1; color: $accent; }
    #stream-label.visible { display: block; }
    #statusbar { dock: bottom; height: 1; background: $panel; color: $text; padding: 0 1; }
    .section-header { color: $accent; text-style: bold; padding: 0 1; }
    .task-item { padding: 0 1; }
    #nt-dialog {
        width: 70%; height: auto; max-height: 80%;
        border: solid $primary; background: $surface;
        padding: 1 2; align: center top;
    }
    #nt-heading { text-style: bold; margin-bottom: 1; }
    #nt-kind { margin-bottom: 1; }
    #nt-desc { height: 10; margin-top: 1; }
    #nt-buttons { height: auto; margin-top: 1; align-horizontal: right; }
    #nt-buttons Button { margin-left: 1; }
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

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            yield ListView(id="sidebar")
            with Vertical(id="chat-col"):
                yield RichLog(id="chat-log", markup=True, wrap=True, highlight=False)
                yield Static("", id="stream-label")
                yield Input(placeholder="Type a message…", id="chat-input")
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
                    marker = {"queued": "⏳", "active": "▶", "completed": "✓"}[
                        t.status.value
                    ]
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

    # -- chat pane ---------------------------------------------------------------

    def view_task(self, task_id: str) -> None:
        self.viewed_id = task_id
        task = self.manager.get(task_id)
        self.chat_log.clear()
        self.stream_label.update("")
        for role, text in extract_display(task.messages):
            self._write_history_line(role, text)
        for pending in task.pending_input:
            self._write_history_line("user-pending", pending)
        self.refresh_sidebar()
        self.refresh_status()

    def _write_history_line(self, role: str, text: str) -> None:
        if role == "user":
            self.chat_log.write(f"[bold cyan]you ›[/] {text}")
        elif role == "user-pending":
            self.chat_log.write(f"[bold yellow]you (queued) ›[/] {text}")
        elif role == "assistant":
            self.chat_log.write(f"[bold green]drcode ›[/] {text}")
        elif role == "tester":
            self.chat_log.write(f"[bold magenta]tester ›[/] {text}")
        elif role == "system":
            self.chat_log.write(f"[bold red]error ›[/] {text}")

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

    # -- agent run -----------------------------------------------------------------

    @work(exclusive=True, group="agent")
    async def run_agent(self, task_id: str) -> None:
        task = self.manager.get(task_id)
        if not self.config.is_ready:
            if self.viewed_id == task_id:
                self._write_history_line(
                    "system",
                    "Model is not configured. Set DRCODE_MODEL, DRCODE_BASE_URL and "
                    "DRCODE_API_KEY (env vars or .env), then send the message again.",
                )
            return
        self.agent_running = True
        self.refresh_status()
        try:
            agent = build_agent(self.config)
            while task.pending_input and task.status is TaskStatus.ACTIVE:
                text = task.pending_input.pop(0)
                history = load_messages(task.messages)
                buffer: list[str] = []
                if self.viewed_id == task_id:
                    self._write_history_line("user", text)
                try:
                    async with agent.iter(
                        text, message_history=history, deps=self.project_root
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
                                            self.chat_log.write(
                                                f"[dim]⚙ {event.part.tool_name}[/]"
                                            )
                    if run.result is not None:
                        task.messages = serialize_messages(run.result.all_messages())
                        self.manager.storage.save(task)
                    if self.viewed_id == task_id:
                        self.stream_label.remove_class("visible")
                        self._write_history_line("assistant", "".join(buffer))
                        self.stream_label.update("")
                except Exception as e:  # noqa: BLE001 - show agent errors in the chat
                    if self.viewed_id == task_id:
                        self._write_history_line("system", f"{type(e).__name__}: {e}")
                    task.pending_input.insert(
                        0, text
                    )  # not lost, retry on next activation
                    break
        finally:
            self.agent_running = False
            self.refresh_status()

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
                                        self.chat_log.write(
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

    def action_new_task(self) -> None:
        def _created(task: Task | None) -> None:
            if task is None:
                return
            self.refresh_sidebar()
            self.view_task(task.id)
            self.chat_input.focus()
            # A task with a description and no active task starts immediately.
            promoted = (
                task
                if task.status is TaskStatus.ACTIVE
                else self.manager.promote_next()
            )
            if promoted:
                self.run_agent(promoted.id)
                self.refresh_status()

        self.push_screen(NewTaskScreen(), _created)

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
        if not self.config.is_ready:
            self._write_history_line(
                "system",
                "Model is not configured. Set DRCODE_MODEL, DRCODE_BASE_URL and DRCODE_API_KEY "
                "(env vars or .env), then retry.",
            )
            return
        self.manager.send(task.id, text)
        event.input.clear()
        was_active = task.status is TaskStatus.ACTIVE
        if was_active and not self.agent_running:
            self.run_agent(task.id)
        self.refresh_sidebar()
        self.refresh_status()
        # For an active task run_agent() renders the message itself.
        if not was_active and self.viewed_id == task.id:
            self._write_history_line("user-pending", text)


def run_app(project_root: Path) -> None:
    DrCodeApp(project_root).run()
