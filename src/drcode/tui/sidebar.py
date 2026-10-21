from __future__ import annotations

from typing import ClassVar, cast

from textual.binding import Binding
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

