"""User-facing goal UI: capture, edit, reparent and the full-screen tree.

- ``GoalEditScreen`` — the single goal modal with two modes: CREATE (the
  capture form: title required, description/notes TextAreas, type cycled
  with ctrl+t, an optional parent preset — the GoalScreen's ``N`` creates
  a sub-goal — and an optional pre-enabled "link viewed task" checkbox
  when the palette's "Create goal…" runs over an open task) and EDIT
  (the GoalScreen's Enter, formerly E: description/notes only, the
  title is fixed).
  `notes` is the long free-form markdown the AGENT reads only via its
  read_goal tool — it is never injected into a system prompt.
- ``GoalParentPicker`` — the "M" reparent picker (SearchDialog pattern):
  a filter input over the candidate goals plus a "Make top-level" entry.
- ``GoalAssignPicker`` — the same pattern minus "Make top-level": the
  task statusbar's "+ assign" link assigns a goal to the viewed task.
- ``GoalScreen`` — the full-screen roadmap view: milestone → sub-goal
  tree with n/m progress counters, a status filter and the keyboard
  action set (N/R/T/M/A/D/Ctrl+Up/Down/Esc). Enter on a goal row opens
  the edit dialog (the old E); a CLICK on a goal row (or → / Right)
  toggles the details — description/notes preview plus EVERY linked
  task (all statuses) under the goal's whole subtree; Enter or a click
  on a task row opens that task and closes the screen. The tree itself
  is collapsible: → (a button and the Right key) toggles the cursor
  goal's sub-goals and detail rows together, ← (a button and the Left
  key) lifts the cursor to the parent goal. The screen is a viewer +
  decision pad:
  every mutation goes through the app's goal actions layer
  (``tui.actions_goals``), which owns persistence and sidebar refresh.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, ListView, Static, TextArea

from ..models import (
    Goal,
    GoalStatus,
    GoalType,
    TaskKind,
    VerificationStatus,
)
from .sidebar import TaskItem

if TYPE_CHECKING:
    from .app import DrCodeApp

# Status filters of the GoalScreen's top line, cycled with F.
FILTERS: list[str] = ["active", "ideas", "done", "archived", "all"]

_TYPE_CYCLE: list[GoalType] = [
    GoalType.IDEA,
    GoalType.GOAL,
    GoalType.MILESTONE,
]

_TYPE_HINTS: dict[GoalType, str] = {
    GoalType.IDEA: "a captured thought, refined later",
    GoalType.GOAL: "an approved direction of development",
    GoalType.MILESTONE: "a decomposable checkpoint",
}


def _goal_type_marker(goal: Goal) -> str:
    """The tree marker of a goal: ◆ for a goal (bold for a milestone),
    the hollow ◇ for a raw idea."""
    if goal.type is GoalType.IDEA:
        return "[dim]◇[/]"
    if goal.type is GoalType.MILESTONE:
        return "[bold magenta]◆[/]"
    return "[magenta]◆[/]"


def _goal_progress_label(goal: Goal, progress: tuple[int, int]) -> str:
    """The "n/m" counter (done/total of the goal's linked tasks, a
    milestone sums its subtree); a goal without any linked task shows
    "—" instead of a hollow 0/0."""
    done, total = progress
    return "—" if total == 0 else f"{done}/{total}"


@dataclass
class GoalCaptureResult:
    """What the create mode of :class:`GoalEditScreen` dismisses with:
    the captured title, the description/notes texts, the dialog's type
    classification and whether the user asked to link the viewed task."""

    title: str
    description: str
    notes: str
    goal_type: GoalType
    link_task: bool


class GoalEditScreen(ModalScreen[GoalCaptureResult | tuple[str, str] | None]):
    """The single goal modal, both modes in one class.

    CREATE (``goal is None``): title (Input, required), description and
    notes (TextAreas), the type cycled with ctrl+t, an optional preset
    parent (the GoalScreen's N — a read-only line, re-parenting is the
    GoalScreen's M) and — only when the caller passes `link_task_id` (the
    palette's "Create goal…" over a viewed task) — a pre-enabled checkbox
    to link that task to the created goal right away. Dismisses with a
    :class:`GoalCaptureResult`.

    EDIT (``goal`` given, the GoalScreen's Enter, formerly ``e``): the
    title is a fixed line,
    only description and notes are editable. Dismisses with
    (description, notes).

    Both dismiss None on ESC/Cancel. `notes` is the long free-form
    markdown the agent reads ONLY via its read_goal tool — it never
    enters a system prompt."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+enter", "save", "Save", priority=True),
        Binding("ctrl+t", "cycle_type", "Change type", priority=True),
    ]

    DEFAULT_CSS = """
    GoalEditScreen {
        align: center middle;
    }
    #goal-edit-dialog {
        width: 90;
        max-width: 95%;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    /* The middle fields scroll so the buttons row and the hint always
    stay visible (the ts-body pattern from task_settings_dialog). CSS
    alone can't adapt the cap to the terminal: an fr under the
    auto-height dialog does not clamp, so the floor here keeps a small
    terminal safe and _fit_fields raises the cap on mount/resize when
    the terminal has room. */
    #goal-edit-fields {
        height: auto;
        max-height: 12;
        overflow-y: auto;
    }
    #goal-edit-title { text-style: bold; margin-bottom: 1; }
    #goal-note { color: $text; margin-bottom: 1; }
    #goal-parent { color: $text-muted; margin-bottom: 1; }
    #goal-title { margin-bottom: 1; }
    #goal-edit-desc-label, #goal-edit-notes-label {
        color: $text-muted;
        margin-top: 0; /* compact widgets need no border-spacing */
    }
    #goal-edit-desc { height: 4; border: solid $primary; }
    #goal-edit-notes { height: 12; border: solid $primary; }
    #goal-link-task { margin-top: 1; height: auto; }
    #goal-error { color: $error; height: auto; }
    #goal-type-label {
        width: auto;
        margin-right: 1;
        text-style: bold;
        color: yellow;
    }
    #goal-type-hint { color: $text-disabled; }
    #goal-type-line { margin-top: 1; }
    #goal-edit-buttons { height: auto; margin-top: 1; }
    #goal-edit-buttons Button { width: auto; margin-right: 1; }
    #goal-edit-hint { color: $text-disabled; }
    """

    def __init__(
        self,
        goal: Goal | None = None,
        *,
        parent_id: str | None = None,
        parent_label: str | None = None,
        link_task_id: str | None = None,
    ) -> None:
        super().__init__()
        self._goal = goal
        self._goal_type = goal.type if goal is not None else GoalType.IDEA
        # Preset parent (sub-goal creation from the GoalScreen): shown as
        # a read-only line — the picker for re-parenting is the GoalScreen's M.
        self._parent_id = parent_id
        self._parent_label = parent_label
        # The viewed task offered for linking (create mode only, the
        # palette's "Create goal…"); None — no checkbox at all.
        self._link_task_id = link_task_id

    @property
    def _creating(self) -> bool:
        return self._goal is None

    # -- composition ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="goal-edit-dialog"):
            if self._creating:
                yield Static(
                    "New product goal — the roadmap item the research agent "
                    "plans against. The title is enough to start; classification "
                    "and description can be refined later.",
                    id="goal-note",
                )
                if self._parent_id:
                    yield Static(
                        f"sub-goal of #{self._parent_id} "
                        f"{escape(self._parent_label or '')}",
                        id="goal-parent",
                    )
                yield Input(placeholder="Goal title (required)", id="goal-title")
            else:
                assert self._goal is not None
                yield Static(
                    f"Edit goal #{self._goal.id} — {escape(self._goal.title_display)}",
                    id="goal-edit-title",
                )
            with VerticalScroll(id="goal-edit-fields"):
                yield Static(
                    "description — one short paragraph; it rides along in the "
                    "research agent's prompt injection",
                    id="goal-edit-desc-label",
                )
                yield TextArea(
                    self._goal.description if self._goal else "",
                    id="goal-edit-desc",
                )
                yield Static(
                    "notes — free-form markdown; the agent reads it only via "
                    "read_goal on demand (never injected into the prompt)",
                    id="goal-edit-notes-label",
                )
                yield TextArea(
                    self._goal.notes if self._goal else "",
                    id="goal-edit-notes",
                )
                if self._creating:
                    yield Horizontal(
                        Static("", id="goal-type-label"),
                        Static("", id="goal-type-hint"),
                        id="goal-type-line",
                    )
                    if self._link_task_id:
                        yield Checkbox(
                            f"link viewed task #{self._link_task_id}",
                            True,
                            id="goal-link-task",
                        )
                    yield Static("", id="goal-error")
            with Horizontal(id="goal-edit-buttons"):
                yield Button(
                    "Create goal" if self._creating else "Save",
                    variant="primary",
                    id="goal-save",
                )
                yield Button("Cancel", id="goal-cancel")
            yield Static("ctrl+enter — save   esc — cancel", id="goal-edit-hint")
        if self._creating:
            self._update_type_label()

    def on_mount(self) -> None:
        self.call_after_refresh(self._fit_fields)
        if self._creating:
            self.query_one("#goal-title", Input).focus()
        else:
            self.query_one("#goal-edit-desc", TextArea).focus()

    def on_resize(self) -> None:
        self.call_after_refresh(self._fit_fields)

    def _fit_fields(self) -> None:
        """Cap #goal-edit-fields so the buttons row and the hint always
        fit inside the dialog's max-height: 90% — CSS fr can't do it
        under the auto-height dialog (it doesn't clamp, the dialog just
        clips its bottom). Sums the natural heights of everything around
        the field area (measured post-layout, so wrapped Statics count
        exactly) and gives the rest to the scrollable fields; the floor
        of 6 rows keeps the field area usable on tiny terminals (the
        CSS floor of 12 stands until this runs)."""
        fields = self.query_one("#goal-edit-fields", VerticalScroll)
        dialog = self.query_one("#goal-edit-dialog", Vertical)
        fixed = dialog.styles.gutter.height
        for child in dialog.children:
            if child is fields:
                continue
            fixed += (
                child.region.height
                + child.styles.margin.top
                + child.styles.margin.bottom
            )
        budget = int(self.size.height * 0.9) - fixed
        fields.styles.max_height = max(6, budget)

    @property
    def type_label(self) -> Static:
        return self.query_one("#goal-type-label", Static)

    def _update_type_label(self) -> None:
        if not self.is_mounted:
            return
        self.type_label.update(f"type: {self._goal_type.value}")
        self.query_one("#goal-type-hint", Static).update(_TYPE_HINTS[self._goal_type])

    # -- actions ------------------------------------------------------------------

    def action_cycle_type(self) -> None:
        if not self._creating:
            return  # ctrl+t is a create-mode control; the edit mode ignores it
        index = _TYPE_CYCLE.index(self._goal_type)
        self._goal_type = _TYPE_CYCLE[(index + 1) % len(_TYPE_CYCLE)]
        self._update_type_label()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        self._save()

    @on(Button.Pressed, "#goal-save")
    def _save_pressed(self) -> None:
        self._save()

    @on(Button.Pressed, "#goal-cancel")
    def _cancel_pressed(self) -> None:
        self.dismiss(None)

    @on(Input.Submitted, "#goal-title")
    def _title_submitted(self) -> None:
        self._save()

    def _save(self) -> None:
        if self._creating:
            title = self.query_one("#goal-title", Input).value.strip()
            if not title:
                self.query_one("#goal-error", Static).update(
                    "The title is required — one line naming the goal."
                )
                self.query_one("#goal-title", Input).focus()
                return
            description = self.query_one("#goal-edit-desc", TextArea).text.strip()
            notes = self.query_one("#goal-edit-notes", TextArea).text.strip()
            link = (
                self.query_one("#goal-link-task", Checkbox).value
                if self._link_task_id
                else False
            )
            self.dismiss(
                GoalCaptureResult(
                    title=title,
                    description=description,
                    notes=notes,
                    goal_type=self._goal_type,
                    link_task=link,
                )
            )
        else:
            description = self.query_one("#goal-edit-desc", TextArea).text.strip()
            notes = self.query_one("#goal-edit-notes", TextArea).text
            self.dismiss((description, notes.strip()))


class GoalParentPicker(ModalScreen[str | None]):
    """The reparent picker (the GoalScreen's M, SearchDialog pattern):
    a filter input over the candidate goals plus a "Make top-level"
    entry. Dismisses with the new parent's id, "__top__" for top-level,
    None on ESC."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Close", priority=True),
        # The filter input keeps the focus (typing must land there); the
        # arrows move the list cursor from the screen level — Input binds
        # no up/down, so the keys bubble here (also shows the blurred
        # highlight cursor that plain mouse/Tab navigation never revealed).
        Binding("down", "list_down", show=False),
        Binding("up", "list_up", show=False),
    ]

    DEFAULT_CSS = """
    GoalParentPicker {
        align: center middle;
    }
    #goal-parent-dialog {
        width: 72;
        max-width: 95%;
        height: 24;
        max-height: 80%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #goal-parent-title { text-style: bold; margin-bottom: 1; }
    #goal-parent-list {
        height: 1fr;
        margin-top: 1;
        background: $surface-lighten-1;
    }
    #goal-parent-list > ListItem { padding: 0 1; }
    #goal-parent-list ListItem Label { width: 1fr; }
    #goal-parent-hint {
        margin-top: 1;
        color: $text-disabled;
        text-align: right;
    }
    """

    _TOP = "__top__"

    def __init__(self, goal: Goal, candidates: list[tuple[Goal, int]]) -> None:
        super().__init__()
        self._goal = goal
        # (goal, tree depth) pairs — the caller flattens the store's
        # display tree (GoalStore.flattened) so sub-goals render indented
        # under their parents.
        self._candidates = candidates

    def compose(self) -> ComposeResult:
        with Vertical(id="goal-parent-dialog"):
            yield Static(
                f"Move goal #{self._goal.id} {escape(self._goal.title_display)} under…",
                id="goal-parent-title",
            )
            yield Input(placeholder="filter goals…", id="goal-parent-input")
            yield ListView(id="goal-parent-list")
            yield Static("enter — move under it · esc — close", id="goal-parent-hint")

    def on_mount(self) -> None:
        self._rebuild("")
        self.query_one("#goal-parent-input", Input).focus()

    @on(Input.Changed, "#goal-parent-input")
    def _input_changed(self, event: Input.Changed) -> None:
        self._rebuild(event.value.strip().lower())

    def _rebuild(self, query: str) -> None:
        listing = self.query_one("#goal-parent-list", ListView)
        listing.clear()
        if not query or "top".startswith(query) or "make".startswith(query):
            listing.append(TaskItem("[bold]Make top-level[/]", None, goal_id=self._TOP))
        for goal, depth in self._candidates:
            text = (
                f"{'  ' * depth}{_goal_type_marker(goal)} {escape(goal.title_display)}"
            )
            if goal.status is GoalStatus.DONE:
                text += " [green]done[/]"
            elif goal.status is GoalStatus.ARCHIVED:
                text += " [dim]archived[/]"
            hay = f"{goal.title} {goal.id}".lower()
            if query and query not in hay:
                continue
            listing.append(TaskItem(text, None, goal_id=goal.id))

    def action_list_down(self) -> None:
        self.query_one("#goal-parent-list", ListView).action_cursor_down()

    def action_list_up(self) -> None:
        self.query_one("#goal-parent-list", ListView).action_cursor_up()

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Input.Submitted, "#goal-parent-input")
    def _input_submitted(self) -> None:
        """Enter in the filter picks the highlighted row (the first one
        before the arrows were touched) — "type a few chars, Enter"."""
        item = self.query_one("#goal-parent-list", ListView).highlighted_child
        if isinstance(item, TaskItem) and item.goal_id:
            self.dismiss(item.goal_id)

    @on(ListView.Selected, "#goal-parent-list")
    def _picked(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, TaskItem) and item.goal_id:
            # stop() — the event must not bubble to the app's
            # on_list_view_selected, which routes goal_id rows to the
            # full-screen GoalScreen.
            event.stop()
            self.dismiss(item.goal_id)


class GoalAssignPicker(ModalScreen[str | None]):
    """The "assign a goal to the viewed task" picker (the task
    statusbar's "+ assign" link): the GoalParentPicker pattern minus the
    "Make top-level" entry — a filter input over the candidate goals
    (the caller pre-filters: non-archived, not already linked). Dismisses
    with the picked goal's id, None on ESC."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Close", priority=True),
        # The filter input keeps the focus (typing must land there); the
        # arrows move the list cursor from the screen level — Input binds
        # no up/down, so the keys bubble here (also shows the blurred
        # highlight cursor that plain mouse/Tab navigation never revealed).
        Binding("down", "list_down", show=False),
        Binding("up", "list_up", show=False),
    ]

    DEFAULT_CSS = """
    GoalAssignPicker {
        align: center middle;
    }
    #goal-assign-dialog {
        width: 72;
        max-width: 95%;
        height: 24;
        max-height: 80%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #goal-assign-title { text-style: bold; margin-bottom: 1; }
    #goal-assign-list {
        height: 1fr;
        margin-top: 1;
        background: $surface-lighten-1;
    }
    #goal-assign-list > ListItem { padding: 0 1; }
    #goal-assign-list ListItem Label { width: 1fr; }
    #goal-assign-hint {
        margin-top: 1;
        color: $text-disabled;
        text-align: right;
    }
    """

    def __init__(self, candidates: list[tuple[Goal, int]]) -> None:
        super().__init__()
        # (goal, tree depth) pairs — the caller flattens the store's
        # display tree (GoalStore.flattened) so sub-goals render indented
        # under their parents.
        self._candidates = candidates

    def compose(self) -> ComposeResult:
        with Vertical(id="goal-assign-dialog"):
            yield Static("Assign goal…", id="goal-assign-title")
            yield Input(placeholder="filter goals…", id="goal-assign-input")
            yield ListView(id="goal-assign-list")
            yield Static("enter — assign it · esc — close", id="goal-assign-hint")

    def on_mount(self) -> None:
        self._rebuild("")
        self.query_one("#goal-assign-input", Input).focus()

    @on(Input.Changed, "#goal-assign-input")
    def _input_changed(self, event: Input.Changed) -> None:
        self._rebuild(event.value.strip().lower())

    def _rebuild(self, query: str) -> None:
        listing = self.query_one("#goal-assign-list", ListView)
        listing.clear()
        for goal, depth in self._candidates:
            text = (
                f"{'  ' * depth}{_goal_type_marker(goal)} {escape(goal.title_display)}"
            )
            if goal.status is GoalStatus.DONE:
                text += " [green]done[/]"
            hay = f"{goal.title} {goal.id}".lower()
            if query and query not in hay:
                continue
            listing.append(TaskItem(text, None, goal_id=goal.id))

    def action_list_down(self) -> None:
        self.query_one("#goal-assign-list", ListView).action_cursor_down()

    def action_list_up(self) -> None:
        self.query_one("#goal-assign-list", ListView).action_cursor_up()

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Input.Submitted, "#goal-assign-input")
    def _input_submitted(self) -> None:
        """Enter in the filter picks the highlighted row (the first one
        before the arrows were touched) — "type a few chars, Enter"."""
        item = self.query_one("#goal-assign-list", ListView).highlighted_child
        if isinstance(item, TaskItem) and item.goal_id:
            self.dismiss(item.goal_id)

    @on(ListView.Selected, "#goal-assign-list")
    def _picked(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, TaskItem) and item.goal_id:
            # stop() — the event must not bubble to the app's
            # on_list_view_selected, which routes goal_id rows to the
            # full-screen GoalScreen (the reported bug: a click assigned
            # the goal AND popped the roadmap view open).
            event.stop()
            self.dismiss(item.goal_id)


class GoalScreen(ModalScreen[None]):
    """The full-screen roadmap view (opened from the sidebar's GOALS
    section, a goal row included): a status filter line on top, the
    milestone → sub-goal tree in the middle and a key-hint footer.
    Enter on a goal row (or on its detail rows) opens the edit dialog;
    a CLICK on a goal row toggles its detail section — description/
    notes preview right under the row plus, below the goal's WHOLE
    sub-tree, one row per linked task (every status: draft, hold,
    completed, …); Enter or a click on a task row opens that task and
    closes the screen. Local bindings shadow the app's globals (the
    merge-screen pattern): N new (sub-)goal · R research task for the
    goal · T task from the goal (the standard Ctrl+N form with the link
    preset) · Enter edit description/notes · M reparent (picker or Make
    top-level) · A archive/unarchive · D done/undone (toggle
    GoalStatus.DONE) · Ctrl+Up/Down reorder among siblings · F cycle
    the status filter · → expand/collapse the cursor goal's sub-goals
    AND its detail rows together ·
    ← lift the cursor to the parent goal · Esc close."""

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Close", priority=True),
        # Priority: must win over ListView's own Enter (= Selected), so
        # the key can open the edit dialog / the task while a click
        # stays on the Selected handler (details toggle / task open).
        Binding("enter", "tree_enter", "Edit", priority=True, show=False),
        Binding("right", "toggle_subtree", "Subtree", show=False),
        Binding("left", "cursor_parent", "Up hierarchy", show=False),
        Binding("f", "cycle_filter", "Filter"),
        Binding("n", "new_goal", "New goal"),
        Binding("r", "research_task", "Research task"),
        Binding("t", "task_from_goal", "Task from goal"),
        Binding("m", "reparent", "Move"),
        Binding("a", "archive", "Archive"),
        Binding("d", "toggle_done", "Done"),
        Binding("ctrl+up", "move_up", "Move up", show=False),
        Binding("ctrl+down", "move_down", "Move down", show=False),
        # Alt duplicates: macOS intercepts ctrl+arrows system-wide.
        Binding("alt+up", "move_up", "Move up", show=False),
        Binding("alt+down", "move_down", "Move down", show=False),
    ]

    DEFAULT_CSS = """
    GoalScreen {
        align: center middle;
    }
    #goal-screen-dialog {
        width: 98%;
        max-width: 200;
        height: 98%;
        padding: 0 1 1 1;
        background: $panel;
        border: thick $accent;
    }
    #goal-screen-top {
        height: auto;
        padding: 0 1;
    }
    #goal-screen-top-label {
        width: 1fr;
        height: 1;
        padding: 0 1 0 0;
        color: $text;
        text-style: bold;
    }
    /* Compact one-line tree controls (→ subtree toggle, ← cursor to the
    parent): the default Button is a 3-row block — too heavy for the
    tree's header line. */
    #goal-screen-top Button {
        height: 1;
        min-width: 3;
        border: none;
        padding: 0;
        margin-left: 1;
    }
    #goal-tree {
        height: 1fr;
        background: $surface;
    }
    #goal-tree > ListItem { padding: 0 1; }
    #goal-tree ListItem Label { width: 1fr; }
    #goal-screen-footer {
        dock: bottom;
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, focus_goal_id: str | None = None) -> None:
        super().__init__()
        self._filter = "active"
        # Goal ids whose description/notes detail rows are expanded
        # (Enter toggles; session state, not persisted).
        self._expanded: set[str] = set()
        # Goal ids whose sub-goal subtrees are hidden (the → button /
        # Right key folds it together with the detail rows; session
        # state, not persisted). A collapsed goal still shows its own
        # row, its descendants don't render.
        self._collapsed: set[str] = set()
        # The goal a refresh must put the cursor back onto (the ListView
        # index is position-based, a rebuild would drop it).
        self._focus_goal_id = focus_goal_id
        # Same for a task row (the linked-task rows of an expanded goal):
        # restored by task_id after a rebuild, the goal hint is the
        # fallback when the task row is no longer rendered.
        self._focus_task_id: str | None = None
        # Parent preset of an in-flight N capture (read back in the
        # dismissal callback — the cursor may have moved by then).
        self._pending_parent: str | None = None
        self._pending_parent_label: str | None = None

    # -- helpers ------------------------------------------------------------

    def _app(self) -> DrCodeApp:
        from .app import DrCodeApp  # lazy: widget→app edge, avoids a cycle

        app = self.app
        assert isinstance(app, DrCodeApp)
        return app

    def _matches(self, goal: Goal) -> bool:
        """The status filter: `active`/`done`/`archived` by status,
        `ideas` shows non-archived ideas, `all` everything."""
        if self._filter == "active":
            return goal.status is GoalStatus.ACTIVE
        if self._filter == "ideas":
            return goal.type is GoalType.IDEA and goal.status is not GoalStatus.ARCHIVED
        if self._filter == "done":
            return goal.status is GoalStatus.DONE
        if self._filter == "archived":
            return goal.status is GoalStatus.ARCHIVED
        return True

    def _visible_rows(self) -> list[tuple[Goal, int]]:
        """(goal, depth) pairs of the filtered tree. A goal whose own
        status does not match still shows when a descendant matches (a
        matching sub-goal must not orphan under a hidden parent); the
        non-matching intermediate parents render as plain rows. A goal
        in `self._collapsed` shows its own row but its subtree is not
        walked (the → button / Right key toggles it)."""
        goals = self._app().manager.goals
        rows: list[tuple[Goal, int]] = []

        def subtree_matches(goal_id: str) -> bool:
            return any(
                self._matches(child) or subtree_matches(child.id)
                for child in goals.display_children(goal_id)
            )

        def walk(parent_id: str | None, depth: int) -> None:
            children = (
                goals.top_level()
                if parent_id is None
                else goals.display_children(parent_id)
            )
            for goal in children:
                matched = self._matches(goal)
                if matched or subtree_matches(goal.id):
                    if matched:
                        rows.append((goal, depth))
                    if goal.id not in self._collapsed:
                        walk(goal.id, depth + 1)

        walk(None, 0)
        return rows

    def _cursor_goal_id(self) -> str | None:
        item = self.query_one("#goal-tree", ListView).highlighted_child
        if isinstance(item, TaskItem):
            return item.goal_id
        return None

    def _cursor_goal(self) -> Goal | None:
        goal_id = self._cursor_goal_id()
        if goal_id is None or goal_id not in self._app().manager.goals:
            return None
        return self._app().manager.goals.get(goal_id)

    # -- composition / rendering ---------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="goal-screen-dialog"):
            with Horizontal(id="goal-screen-top"):
                yield Static("", id="goal-screen-top-label", markup=False)
                yield Button(
                    "→",
                    id="goal-tree-expand",
                    tooltip=(
                        "expand/collapse the cursor goal's sub-goals and "
                        "details (Right)"
                    ),
                )
                yield Button(
                    "←",
                    id="goal-tree-parent",
                    tooltip="move the cursor to the parent goal (Left)",
                )
            yield ListView(id="goal-tree")
            yield Static(
                "n new · r research task · t task · enter edit goal · "
                "click goal = details · m move · "
                "a archive/unarchive · d done/undone · f filter · "
                "ctrl+↑/↓ reorder · enter on task = open · ←/→ tree · esc close",
                id="goal-screen-footer",
            )

    def on_mount(self) -> None:
        if self._focus_goal_id:
            self._expanded.add(self._focus_goal_id)
        # The first render waits one layout round: ListView.appends that
        # run inside on_mount are dropped (the list itself is still
        # mounting — its prune/mount round-trips have not started yet).
        self.call_after_refresh(self._first_render)

    def _first_render(self) -> None:
        if not self.is_mounted:
            return
        self._rebuild()
        self.query_one("#goal-tree", ListView).focus()

    def _update_top(self) -> None:
        rows = self._visible_rows()
        total = len(self._app().manager.goals.goals)
        self.query_one("#goal-screen-top-label", Static).update(
            f"filter: {self._filter} — {len(rows)} shown of {total} goal(s) "
            "(f cycles the filter)"
        )

    def _row_text(self, goal: Goal, depth: int, app: DrCodeApp) -> str:
        indent = "  " * depth
        done, total = app.manager.goal_progress(goal)
        line = (
            f"{indent}{_goal_type_marker(goal)} {escape(goal.title_display)}  "
            f"[dim]{_goal_progress_label(goal, (done, total))}[/]"
        )
        if goal.status is GoalStatus.DONE:
            line += " [green]✓ done[/]"
        elif goal.status is GoalStatus.ARCHIVED:
            line += " [dim]archived[/]"
        viewed = app.viewed
        if viewed is not None and viewed.goal_id == goal.id:
            # The link marker of the task this screen was opened from
            # (the same ◆ the sidebar paints next to the task).
            line += f" [magenta]◆[/]#{viewed.id}"
        return line

    def _detail_rows(self, goal: Goal, depth: int) -> list[TaskItem]:
        """The Enter-expanded rows right under a goal line: description
        (short), a notes preview and the meta line. The notes preview is
        UI-only — the full text stays behind the agent's read_goal."""
        indent = "  " * (depth + 1)
        rows: list[TaskItem] = []
        description = " ".join(goal.description.split())
        if len(description) > 120:
            description = description[:119] + "…"
        rows.append(
            TaskItem(
                f"{indent}[italic]{escape(description) if description else '[dim]no description — E to edit[/]'}[/]",
                None,
                goal_id=goal.id,
            )
        )
        notes_first = next(
            (ln.strip() for ln in goal.notes.splitlines() if ln.strip()), ""
        )
        if len(notes_first) > 120:
            notes_first = notes_first[:119] + "…"
        if notes_first:
            rows.append(TaskItem(f"{indent}[dim]notes: {escape(notes_first)}[/]", None))
        rows.append(
            TaskItem(
                f"{indent}[dim]#{goal.id} · {goal.type.value} · "
                f"created {goal.created_at:%Y-%m-%d}[/]",
                None,
            )
        )
        return rows

    _TASK_STATUS_MARKERS: ClassVar[dict[str, str]] = {
        "queued": "◷",
        "draft": "✎",
        "hold": "⏸",
        "active": "▶",
        "testing": "⚑",
        "completed": "✓",
        "clarification": "❓",
    }

    # Sort rank of the linked-task rows: work in motion first, parked
    # and finished tasks last (created_at breaks the ties).
    _TASK_STATUS_RANK: ClassVar[dict[str, int]] = {
        "active": 0,
        "testing": 1,
        "clarification": 2,
        "queued": 3,
        "draft": 4,
        "hold": 5,
        "completed": 6,
    }

    def _task_rows(self, goal: Goal, depth: int, app: DrCodeApp) -> list[TaskItem]:
        """The linked-task rows of an expanded goal: EVERY task with
        ``task.goal_id == goal.id`` — all statuses, draft/hold/completed
        included — one line each (kind, #id, title, status), indented one
        level deeper than the sub-goal rows. Enter or a click on a row
        opens the task (the sidebar's row format, ``Sidebar._task_row_text``)."""
        linked = [t for t in app.manager.tasks.values() if t.goal_id == goal.id]
        if not linked:
            return []
        linked.sort(
            key=lambda t: (self._TASK_STATUS_RANK.get(t.status.value, 9), t.created_at)
        )
        indent = "  " * (depth + 2)
        # Rough visible width of the list for the title clip (chars, not
        # cells — good enough, the markers are narrow glyphs).
        width = self.query_one("#goal-tree", ListView).size.width or 80
        rows: list[TaskItem] = []
        for t in linked:
            marker = self._TASK_STATUS_MARKERS.get(t.status.value, "⏳")
            kind_letter = {
                TaskKind.BUG: "[red]B[/]",
                TaskKind.RESEARCH: "[yellow]R[/]",
            }.get(t.kind, "[green]F[/]")
            if t.verification is VerificationStatus.PASSED:
                verify = " ✅"
            elif t.verification is VerificationStatus.FAILED:
                verify = " ❌"
            else:
                verify = ""
            prefix = f"{indent}{marker} {kind_letter} #{t.id} "
            budget = max(8, width - len(indent) - 8 - len(t.id) - len(t.status.value))
            title = t.title_display
            if len(title) > budget:
                title = title[: budget - 1] + "…"
            rows.append(
                TaskItem(
                    f"{prefix}{escape(title)} [dim]{t.status.value}[/]{verify}",
                    t.id,
                )
            )
        return rows

    def _rebuild(self) -> None:
        """Re-render the tree, keeping the cursor on its goal.

        The actual re-render runs in an exclusive worker: ListView's
        clear()/extend() settle asynchronously (the old rows are pruned
        on a later message), so restoring the cursor synchronously would
        scan the doomed pre-clear children and highlight a stale row —
        the lost-cursor bug of Ctrl+Up/Down. A newer rebuild (key repeat)
        cancels the in-flight one and starts from the current store."""
        if not self.is_mounted:
            return
        self.run_worker(self._rebuild_worker(), exclusive=True, group="goal-rebuild")

    async def _rebuild_worker(self) -> None:
        app = self._app()
        listing = self.query_one("#goal-tree", ListView)
        highlighted = listing.highlighted_child
        cursor_task_id = (
            highlighted.task_id if isinstance(highlighted, TaskItem) else None
        )
        cursor_id = self._cursor_goal_id() or self._focus_goal_id
        # Compute every row BEFORE clearing: clear() yields to the event
        # loop, so anything computed after it delays the re-mount and the
        # user sees an empty list for that whole gap.
        rows = self._visible_rows()
        items: list[TaskItem] = []
        # Expanded goals whose sub-tree is currently open (a row at depth
        # d closes every open sub-tree of depth >= d). The linked-task
        # rows sit BELOW the whole sub-tree, so they are emitted when the
        # sub-tree closes — innermost first; a collapsed goal's sub-tree
        # is not walked at all, so its tasks land right under its detail
        # rows.
        open_expanded: list[tuple[Goal, int]] = []
        for goal, depth in rows:
            while open_expanded and open_expanded[-1][1] >= depth:
                closed, closed_depth = open_expanded.pop()
                items.extend(self._task_rows(closed, closed_depth, app))
            items.append(
                TaskItem(self._row_text(goal, depth, app), None, goal_id=goal.id)
            )
            if goal.id in self._expanded:
                items.extend(self._detail_rows(goal, depth))
                open_expanded.append((goal, depth))
        while open_expanded:
            closed, closed_depth = open_expanded.pop()
            items.extend(self._task_rows(closed, closed_depth, app))
        # Swap the whole tree in one batch: clear() and mount() are
        # synchronous calls returning awaitables, so keeping them (and the
        # batch) free of awaits means no frame is painted between the
        # removal and the mount — no empty-list flash.
        async with self.batch():
            removed = listing.clear()
            mounted = listing.extend(items)
        await removed
        await mounted

        def restore(match: Callable[[TaskItem], bool]) -> bool:
            for index, child in enumerate(listing.children):
                if isinstance(child, TaskItem) and match(child):
                    listing.index = index
                    return True
            return False

        # The task row first (its position shifts with every detail row);
        # when it is gone (the details were collapsed), fall back to the
        # goal hint, then to the plain goal cursor.
        if not (
            (cursor_task_id and restore(lambda c: c.task_id == cursor_task_id))
            or (cursor_id and restore(lambda c: c.goal_id == cursor_id))
            or (
                self._focus_goal_id
                and restore(lambda c: c.goal_id == self._focus_goal_id)
            )
        ):
            restore(lambda c: c.task_id == self._focus_task_id)
        if listing.index is None and listing.children:
            # No cursor to restore (first open, or the filter hid the old
            # row): land on the first row so the action keys work right
            # away, without an extra arrow press.
            listing.index = 0
        # The cursor hint for the next rebuild: while a previous rebuild
        # is still settling, highlighted_child is None and the move
        # actions fall back to the goal it was about to restore.
        self._focus_goal_id = cursor_id
        self._focus_task_id = cursor_task_id
        self._update_top()

    def _refresh_all(self) -> None:
        self._app().refresh_sidebar()
        self._rebuild()

    # -- events ----------------------------------------------------------------

    @on(ListView.Selected, "#goal-tree")
    def _goal_selected(self, event: ListView.Selected) -> None:
        """A CLICK on a row (Enter is intercepted by the screen's
        priority ``enter`` binding and never reaches the ListView, so
        Selected here means a mouse click): on a goal row it toggles the
        detail section, on a task row it opens the task."""
        item = event.item
        if not isinstance(item, TaskItem):
            return
        if item.task_id:
            event.stop()
            self._open_task(item.task_id)
            return
        if item.goal_id:
            if item.goal_id in self._expanded:
                self._expanded.discard(item.goal_id)
            else:
                self._expanded.add(item.goal_id)
            event.stop()
            self._rebuild()

    def action_tree_enter(self) -> None:
        """Enter (the priority binding): on a task row — open the task
        and close the screen; on a goal or detail row — the edit dialog
        (the old ``e``). A no-op when the tree does not hold the focus,
        so a focused →/← button keeps its own Enter."""
        listing = self.query_one("#goal-tree", ListView)
        if self.focused is not listing:
            return
        item = listing.highlighted_child
        if isinstance(item, TaskItem):
            if item.task_id:
                self._open_task(item.task_id)
            elif item.goal_id:
                self.action_edit_goal()

    def _open_task(self, task_id: str) -> None:
        """Dismiss the screen and open the task — the SearchDialog flow
        (view_task + chat_input.focus(), view_task clears the new-task
        mode; a DRAFT redirect keeps the focus in the editor)."""
        from .app import DrCodeApp  # lazy: widget→app edge, avoids a cycle

        app = self.app
        self.dismiss(None)
        assert isinstance(app, DrCodeApp)
        app.view_task(task_id)
        if not app._new_task_mode:
            app.chat_input.focus()

    @on(Button.Pressed, "#goal-tree-expand")
    def _expand_pressed(self) -> None:
        self.action_toggle_subtree()
        # A click focuses the button, stealing the arrow keys from the
        # tree — put the cursor (and focus) back into the list.
        self.query_one("#goal-tree", ListView).focus()

    @on(Button.Pressed, "#goal-tree-parent")
    def _parent_pressed(self) -> None:
        self.action_cursor_parent()
        self.query_one("#goal-tree", ListView).focus()

    # -- actions ------------------------------------------------------------------

    def action_close(self) -> None:
        self.dismiss(None)

    def action_cycle_filter(self) -> None:
        self._filter = FILTERS[(FILTERS.index(self._filter) + 1) % len(FILTERS)]
        self._rebuild()

    def action_toggle_subtree(self) -> None:
        """→ (the button / Right key): toggle the cursor goal's whole
        expansion in one move — sub-goals AND the Enter-style detail
        rows together. When anything is showing (the subtree is open or
        the details are expanded) both are hidden; otherwise both are
        expanded back."""
        goal = self._cursor_goal()
        if goal is None:
            return
        has_children = bool(self._app().manager.goals.children(goal.id))
        showing_children = has_children and goal.id not in self._collapsed
        if showing_children or goal.id in self._expanded:
            # Hide both: fold the subtree and the detail rows.
            self._expanded.discard(goal.id)
            if has_children:
                self._collapsed.add(goal.id)
        else:
            # Show both: open the subtree and the detail rows.
            if has_children:
                self._collapsed.discard(goal.id)
            self._expanded.add(goal.id)
        self._rebuild()

    def action_cursor_parent(self) -> None:
        """← (the button / Left key): lift the cursor one level up the
        hierarchy — onto the parent goal's row. A no-op on a top-level
        goal; the parent may be hidden by the status filter (its row
        still renders as an intermediate when a descendant matches, but
        a fully hidden branch leaves nothing to land on)."""
        goal = self._cursor_goal()
        if goal is None or not goal.parent_id:
            return
        listing = self.query_one("#goal-tree", ListView)
        for index, child in enumerate(listing.children):
            if isinstance(child, TaskItem) and child.goal_id == goal.parent_id:
                listing.index = index
                return
        self.notify(
            f"parent #{goal.parent_id} is hidden by the filter",
            title="Goals",
            timeout=4,
        )

    def _require_cursor(self) -> Goal | None:
        goal = self._cursor_goal()
        if goal is None:
            self.notify(
                "Put the cursor on a goal row first",
                title="Goals",
                timeout=4,
            )
        return goal

    def action_new_goal(self) -> None:
        """N: a new goal — a sub-goal under the cursor goal, top-level
        when no row is selected (or the cursor sits on a detail row)."""
        cursor = self._cursor_goal()
        self._pending_parent = cursor.id if cursor else None
        self._pending_parent_label = cursor.title_display if cursor else None
        self.app.push_screen(
            GoalEditScreen(
                parent_id=self._pending_parent,
                parent_label=self._pending_parent_label,
            ),
            self._captured,
        )

    def _captured(self, result: GoalCaptureResult | tuple[str, str] | None) -> None:
        # The create mode dismisses a GoalCaptureResult (the edit-tuple
        # and None — ESC/Cancel — are not capture results).
        if not isinstance(result, GoalCaptureResult):
            return
        self._app().create_goal_from_capture(result, parent_id=self._pending_parent)
        self._refresh_all()

    def action_research_task(self) -> None:
        """R: create a research task linked to the cursor goal — a plain
        research task with the goal reference (no separate "goal chat"):
        the chat input gets the «Цель: … Задание:» stub and the user
        completes and sends it."""
        goal = self._require_cursor()
        if goal is None:
            return
        self.dismiss(None)
        self._app().start_goal_research(goal.id)

    def action_task_from_goal(self) -> None:
        """T: a standard development task from the goal — the regular
        Ctrl+N form with the goal link preset (the form's goal line)."""
        goal = self._require_cursor()
        if goal is None:
            return
        self.dismiss(None)
        self._app().start_goal_task(goal.id)

    def action_edit_goal(self) -> None:
        """Enter (formerly ``e``): edit the cursor goal's description/
        notes."""
        goal = self._require_cursor()
        if goal is None:
            return
        self.app.push_screen(GoalEditScreen(goal), self._edited)

    def _edited(self, result: GoalCaptureResult | tuple[str, str] | None) -> None:
        # The edit mode dismisses (description, notes); a capture result
        # or None (ESC/Cancel) is not an edit.
        if result is None or isinstance(result, GoalCaptureResult):
            return
        goal_id = self._cursor_goal_id()
        if goal_id is None:
            return
        description, notes = result
        app = self._app()
        try:
            app.manager.goals.update(goal_id, description=description, notes=notes)
        except KeyError:
            self.notify("The goal was deleted", title="Goals", severity="warning")
            return
        self._refresh_all()

    def action_reparent(self) -> None:
        """M: move the cursor goal under another goal (the picker lists
        everything except the goal itself and its own subtree) or make
        it top-level."""
        goal = self._require_cursor()
        if goal is None:
            return
        app = self._app()
        excluded = app.manager.goals.descendant_ids(goal.id) | {goal.id}
        candidates = app.manager.goals.flattened(lambda g: g.id not in excluded)
        app.push_screen(GoalParentPicker(goal, candidates), self._reparented)

    def _reparented(self, new_parent: str | None) -> None:
        if new_parent is None:
            return
        goal_id = self._cursor_goal_id()
        if goal_id is None:
            return
        app = self._app()
        target = None if new_parent == GoalParentPicker._TOP else new_parent
        try:
            app.manager.goals.reparent(goal_id, target)
        except ValueError as e:
            self.notify(str(e), title="Goals", severity="error", timeout=6)
            return
        label = (
            f"#{goal_id} made top-level"
            if target is None
            else f"#{goal_id} moved under #{target}"
        )
        self.notify(label, title="Goals", timeout=4)
        self._refresh_all()

    def action_archive(self) -> None:
        """A: archive the cursor goal (goals are never deleted) — or
        return an archived one to active."""
        goal = self._require_cursor()
        if goal is None:
            return
        app = self._app()
        to_archived = goal.status is not GoalStatus.ARCHIVED
        app.manager.goals.update(
            goal.id,
            status=GoalStatus.ARCHIVED if to_archived else GoalStatus.ACTIVE,
        )
        self.notify(
            f"#{goal.id} {goal.title_display} "
            + ("archived" if to_archived else "returned to active"),
            title="Goals",
            timeout=4,
        )
        self._refresh_all()

    def action_toggle_done(self) -> None:
        """D: mark the cursor goal done — the status the "done" filter
        (F) selects on — or return an already done one to active (an
        undo for a mis-strike). Archive (A) stays a separate toggle:
        done means "achieved", archived means "shelved"."""
        goal = self._require_cursor()
        if goal is None:
            return
        app = self._app()
        to_done = goal.status is not GoalStatus.DONE
        app.manager.goals.update(
            goal.id,
            status=GoalStatus.DONE if to_done else GoalStatus.ACTIVE,
        )
        self.notify(
            f"#{goal.id} {goal.title_display} "
            + ("marked done" if to_done else "returned to active"),
            title="Goals",
            timeout=4,
        )
        self._refresh_all()

    def _move(self, *, up: bool) -> None:
        goal_id = self._cursor_goal_id()
        if goal_id is None:
            # A rebuild worker may still be settling (cleared ListView):
            # fall back to the cursor it was about to restore, so a held
            # Ctrl+Up/Down keeps moving the same goal instead of no-oping.
            hint = self._focus_goal_id
            goal_id = hint if hint and hint in self._app().manager.goals else None
        if goal_id is None:
            return
        self._app().manager.goals.move(goal_id, up=up)
        self._refresh_all()

    def action_move_up(self) -> None:
        self._move(up=True)

    def action_move_down(self) -> None:
        self._move(up=False)
