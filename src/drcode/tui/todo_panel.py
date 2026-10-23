"""Read-only todo checklist panel of the chat pane.

Sits between the chat history scroll (`#chat-log`) and the user input
(`#chat-input`) and renders the viewed task's `task.todos` checklist.
Strictly read-only: the agent owns the list (todo_create / todo_update
tools, feature tasks only) — the user just watches it.
"""

from __future__ import annotations

from textual.widgets import Static


class TodoPanel(Static):
    """Collapsible checklist strip; `.hidden` while there is nothing to
    show (no viewed task, new-task form open, empty `task.todos`)."""
