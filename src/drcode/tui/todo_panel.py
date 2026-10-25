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
    show (no viewed task, new-task form open, empty `task.todos`).

    Collapse state: the header line is a click link — clicking it toggles
    the checklist between the collapsed counter line and the full item
    list. `_user_collapsed` is that manual override (`None` = no override:
    the panel auto-collapses only when every item is done). Session-only —
    never persisted, and reset when another task is viewed.
    """

    # Instance state (class defaults keep the compose call unchanged):
    # the viewed task the current render belongs to (an override lives
    # only within one task's view) and the manual collapse override.
    task_id: str | None = None
    user_collapsed: bool | None = None
