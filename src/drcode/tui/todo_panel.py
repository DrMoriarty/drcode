"""Read-only todo checklist panel of the chat pane.

Sits between the chat history scroll (`#chat-log`) and the user input
(`#chat-input`) and renders the viewed task's `task.todos` checklist.
Strictly read-only: the agent owns the list (todo_create / todo_update
tools, feature tasks only) — the user just watches it.
"""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static


class TodoPanel(VerticalScroll):
    """Collapsible checklist strip; `.hidden` while there is nothing to
    show (no viewed task, new-task form open, empty `task.todos`).

    The checklist body is a child `Static`; the panel itself is the
    scroll viewport — CSS caps it (`#todo-panel` max-height) and a longer
    list scrolls instead of being clipped (a bare `Static` cannot scroll
    its own render: the compositor gives leaf widgets
    `virtual_size == region.size`).

    Collapse state: the header line is a click link — clicking it toggles
    the checklist between the collapsed counter line and the full item
    list. `_user_collapsed` is that manual override (`None` = no override:
    the panel auto-collapses only when every item is done). Session-only —
    never persisted, and reset when another task is viewed.

    Not focusable (`can_focus=False`): a click on the header must not
    steal focus from the chat input, and the read-only list needs no
    keyboard bindings — the wheel scrolls it either way.
    """

    # Instance state (class defaults keep the compose call unchanged):
    # the viewed task the current render belongs to (an override lives
    # only within one task's view) and the manual collapse override.
    task_id: str | None = None
    user_collapsed: bool | None = None

    def __init__(
        self,
        content: str = "",
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        # The positional `content` keeps the historic `TodoPanel("", id=…)`
        # compose call unchanged; it seeds the body Static.
        super().__init__(
            name=name, id=id, classes=classes, disabled=disabled, can_focus=False
        )
        self._body = Static(content)

    def compose(self):
        yield self._body

    def update(self, content: str) -> None:
        """Same contract as `Static.update` — replaces the checklist text."""
        self._body.update(content)
