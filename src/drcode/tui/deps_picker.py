"""Dependency picker — attach a `depends_on` edge from the task header.

Opened by the cyan ``+`` fragment of the task statusbar's second row (see
:mod:`drcode.tui.task_status_view`), which is offered only for the
statuses where editing dependencies makes sense (QUEUED / HOLD — the same
gate as ``_can_edit_dependencies``).

The modal copies the ``GoalAssignPicker`` pattern (filter input on top, a
``ListView`` of :class:`~drcode.tui.sidebar.TaskItem` rows below, Enter to
pick, ESC to close). Rows read ``#<id> <kind> <title>`` and the filter
matches on both the title and the id, so a short hash pasted from the
chat/search dialog finds its task.

The caller pre-filters the candidates (the app's action layer): every
non-COMPLETED task except the viewed one and the ones already in its
`depends_on` — a completed task can never block anything, and an existing
dependency is visible (and removable) in the header itself. The screen
only renders and picks, it never touches the manager.
"""

from __future__ import annotations

from typing import ClassVar

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, ListView, Static

from ..models import Task, TaskKind
from .sidebar import TaskItem

_KIND_COLORS: dict[TaskKind, str] = {
    TaskKind.BUG: "red",
    TaskKind.RESEARCH: "yellow",
}


def _kind_label(task: Task) -> str:
    """The kind word of a row, colored like the sidebar's kind letters
    and the statusbar's kind fragment (anything not bug/research is a
    feature)."""
    color = _KIND_COLORS.get(task.kind, "green")
    return f"[{color}]{task.kind.value}[/]"


class DependencyPicker(ModalScreen[str | None]):
    """Pick a task to add to the viewed task's `depends_on` list.

    Dismisses with the picked task's id, None on ESC / no pick. The
    caller owns the mutation, its validation and the refresh — this screen
    is deliberately manager-free (the ``GoalAssignPicker`` contract)."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Close", priority=True),
    ]

    DEFAULT_CSS = """
    DependencyPicker {
        align: center middle;
    }
    #deps-picker-dialog {
        width: 80;
        max-width: 95%;
        height: 24;
        max-height: 80%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #deps-picker-title { text-style: bold; margin-bottom: 1; }
    #deps-picker-list {
        height: 1fr;
        margin-top: 1;
        background: $surface-lighten-1;
    }
    #deps-picker-list > ListItem { padding: 0 1; }
    #deps-picker-list ListItem Label { width: 1fr; }
    #deps-picker-hint {
        margin-top: 1;
        color: $text-disabled;
        text-align: right;
    }
    """

    def __init__(self, task: Task, candidates: list[Task]) -> None:
        super().__init__()
        # NB: not `self._task` — Textual's MessagePump owns a `_task`
        # attribute (the asyncio task of the widget), and shadowing it
        # crashes compose() with a TypeError.
        self._subject = task
        self._candidates = candidates

    def compose(self) -> ComposeResult:
        with Vertical(id="deps-picker-dialog"):
            yield Static(
                f"#{self._subject.id} {escape(self._subject.title_display)} waits for…",
                id="deps-picker-title",
            )
            yield Input(
                placeholder="filter tasks by title or id…", id="deps-picker-input"
            )
            yield ListView(id="deps-picker-list")
            yield Static("enter — add dependency · esc — close", id="deps-picker-hint")

    def on_mount(self) -> None:
        self._rebuild("")
        self.query_one("#deps-picker-input", Input).focus()

    @on(Input.Changed, "#deps-picker-input")
    def _input_changed(self, event: Input.Changed) -> None:
        self._rebuild(event.value.strip().lower())

    @on(Input.Submitted, "#deps-picker-input")
    def _input_submitted(self) -> None:
        """Enter in the filter takes the first (best) row: the focus stays
        in the input while typing, so reaching for the list would
        otherwise need a Tab first — "type a few chars, Enter" is the
        path the picker is built for."""
        listing = self.query_one("#deps-picker-list", ListView)
        for child in listing.children:
            if isinstance(child, TaskItem) and child.task_id:
                self.dismiss(child.task_id)
                return

    def _rebuild(self, query: str) -> None:
        listing = self.query_one("#deps-picker-list", ListView)
        listing.clear()
        shown = 0
        for candidate in self._candidates:
            text = (
                f"#[bold cyan]{candidate.id}[/] {_kind_label(candidate)} "
                f"{escape(candidate.title_display)}"
            )
            hay = f"{candidate.title} {candidate.id}".lower()
            if query and query not in hay:
                continue
            listing.append(TaskItem(text, candidate.id))
            shown += 1
        if not shown:
            # An empty list renders as a blank pane with a cursor nothing
            # can point at — say why instead (the filter can wipe it out).
            listing.append(
                TaskItem(
                    "[dim]No matching task — every other non-completed task "
                    "may already be a dependency.[/]"
                )
            )

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(ListView.Selected, "#deps-picker-list")
    def _picked(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, TaskItem) and item.task_id:
            self.dismiss(item.task_id)
