from __future__ import annotations

from typing import ClassVar, cast

from textual import events
from textual.binding import Binding
from textual.geometry import Offset
from textual.widgets import Label, ListItem, ListView


class TaskItem(ListItem):
    def __init__(
        self,
        text: str,
        task_id: str | None = None,
        *,
        header: bool = False,
        section: str | None = None,
    ) -> None:
        super().__init__(
            Label(text, classes="section-header" if header else "task-item")
        )
        self.task_id = task_id
        self.text = text
        # For section headers: the collapsible category they belong to
        # (None on task rows). Enter/space/click on a header toggles it.
        self.section = section


class Sidebar(ListView):
    """Task list with collapsible pipeline sections.

    Space on a highlighted section header toggles it (Enter and mouse
    clicks are handled via the ListView.Selected event in the app).
    """

    BINDINGS: ClassVar = [
        Binding("space", "toggle_section", "Expand/collapse", show=False),
    ]

    # Where the current mouse press started (screen coordinates); None
    # while no button is held. Used to tell a drag (text selection) apart
    # from a click.
    _mouse_down_offset: Offset | None = None

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        self._mouse_down_offset = event.screen_offset

    def action_toggle_section(self) -> None:
        item = self.highlighted_child
        if isinstance(item, TaskItem) and item.section is not None:
            from .app import DrCodeApp

            app = cast("DrCodeApp", self.app)
            app.toggle_sidebar_section(item.section)

    def _on_list_item__child_clicked(self, event: ListItem._ChildClicked) -> None:
        """Ignore clicks on items removed by a sidebar rebuild.

        refresh_sidebar() clears and re-appends the items, and
        ListView.clear() removes the old widgets asynchronously (it returns
        an AwaitRemove) — so a click can land on an item that is still in
        the DOM but already pruned from the ListView's node list. The stock
        handler does an unchecked index lookup and crashes on it.

        Textual dispatches this handler for every class in the MRO, so the
        ListView base implementation runs after this one — prevented here
        via prevent_default(); no super() call (it would double-run both).
        """
        if event.item not in self.children:
            event.stop()
            event.prevent_default()
            return
        # A press-and-drag (text selection inside the row) ends on the same
        # widget as it started, so Textual still synthesizes a Click from it.
        # The mouse has moved between press and release though — suppress
        # the selection so it doesn't open the task; only a click where the
        # pointer didn't move navigates.
        if (
            self._mouse_down_offset is not None
            and self.app.mouse_position != self._mouse_down_offset
        ):
            event.stop()
            event.prevent_default()
