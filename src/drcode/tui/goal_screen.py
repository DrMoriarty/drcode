"""User-facing goal UI: capture, edit, reparent and the full-screen tree.

- ``GoalCaptureScreen`` — the minimal "New goal" modal (the title is the
  only required field; type and description optional, refined later by
  the user or the research agent). Optionally presets a parent goal (the
  GoalScreen's ``N`` creates a sub-goal).
- ``GoalEditScreen`` — description/notes editing (two TextArea fields;
  `notes` is the long free-form markdown the AGENT reads only via its
  read_goal tool — it is never injected into a system prompt).
- ``GoalParentPicker`` — the "M" reparent picker (SearchDialog pattern):
  a filter input over the candidate goals plus a "Make top-level" entry.
- ``GoalScreen`` — the full-screen roadmap view: milestone → sub-goal
  tree with n/m progress counters, a status filter, Enter-expandable
  description/notes details and the keyboard action set
  (N/R/T/E/M/A/Ctrl+Up/Down/Esc). The screen is a viewer + decision pad:
  every mutation goes through the app's goal actions layer
  (``tui.actions_goals``), which owns persistence and sidebar refresh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, ListView, Static, TextArea

from ..models import Goal, GoalStatus, GoalType
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


class GoalCaptureScreen(ModalScreen[tuple[str, str, GoalType] | None]):
    """Modal "New goal" form: title (required), description (optional),
    type cycled with ctrl+t. Dismisses with
    (title, description, type) on create, None on ESC/Cancel. When a
    parent goal is preset (the GoalScreen's N), the dialog says so and
    the created goal becomes its sub-goal."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+t", "cycle_type", "Change type", priority=True),
    ]

    DEFAULT_CSS = """
    GoalCaptureScreen {
        align: center middle;
    }
    #goal-dialog {
        width: 72;
        max-width: 95%;
        height: auto;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #goal-title { margin-bottom: 1; }
    #goal-desc { margin-bottom: 1; }
    #goal-note { color: $text; margin-bottom: 1; }
    #goal-parent { color: $text-muted; margin-bottom: 1; }
    #goal-error { color: $error; height: auto; }
    #goal-type-label {
        width: auto;
        margin-right: 1;
        text-style: bold;
        color: yellow;
    }
    #goal-type-hint { color: $text-disabled; }
    #goal-buttons {
        height: auto;
        margin-top: 1;
    }
    #goal-buttons Button { width: auto; margin-right: 1; }
    """

    def __init__(
        self,
        parent_id: str | None = None,
        parent_label: str | None = None,
    ) -> None:
        super().__init__()
        self._goal_type = GoalType.IDEA
        # Preset parent (sub-goal creation from the GoalScreen): shown as
        # a read-only line — the picker for re-parenting is the GoalScreen's M.
        self._parent_id = parent_id
        self._parent_label = parent_label

    # -- composition ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="goal-dialog"):
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
            yield Input(placeholder="Description (optional)", id="goal-desc")
            yield Horizontal(
                Static("", id="goal-type-label"),
                Static("", id="goal-type-hint"),
                id="goal-type-line",
            )
            yield Static("", id="goal-error")
            with Horizontal(id="goal-buttons"):
                yield Button("Create goal", variant="primary", id="goal-create")
                yield Button("Cancel", id="goal-cancel")
        self._update_type_label()

    def on_mount(self) -> None:
        self.query_one("#goal-title", Input).focus()

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
        index = _TYPE_CYCLE.index(self._goal_type)
        self._goal_type = _TYPE_CYCLE[(index + 1) % len(_TYPE_CYCLE)]
        self._update_type_label()

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#goal-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#goal-create")
    def _create(self) -> None:
        self._submit()

    @on(Input.Submitted, "#goal-title")
    def _title_submitted(self) -> None:
        self._submit()

    @on(Input.Submitted, "#goal-desc")
    def _desc_submitted(self) -> None:
        self._submit()

    def _submit(self) -> None:
        title = self.query_one("#goal-title", Input).value.strip()
        if not title:
            self.query_one("#goal-error", Static).update(
                "The title is required — one line naming the goal."
            )
            self.query_one("#goal-title", Input).focus()
            return
        description = self.query_one("#goal-desc", Input).value.strip()
        self.dismiss((title, description, self._goal_type))


class GoalEditScreen(ModalScreen[tuple[str, str] | None]):
    """Edit a goal's `description` and `notes` (the GoalScreen's E).

    Two TextArea fields: the description is the short line the research
    agent's prompt injection carries (and the tree shows), the notes are
    the long free-form markdown the agent reads ONLY via its read_goal
    tool — they never enter a system prompt. Dismisses with
    (description, notes) on save, None on ESC/Cancel."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+enter", "save", "Save", priority=True),
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
    #goal-edit-title { text-style: bold; margin-bottom: 1; }
    #goal-edit-desc-label, #goal-edit-notes-label {
        color: $text-muted;
        margin-top: 1;
    }
    #goal-edit-desc { height: 4; border: solid $primary; }
    #goal-edit-notes { height: 12; border: solid $primary; }
    #goal-edit-buttons { height: auto; margin-top: 1; }
    #goal-edit-buttons Button { width: auto; margin-right: 1; }
    """

    def __init__(self, goal: Goal) -> None:
        super().__init__()
        self._goal = goal

    def compose(self) -> ComposeResult:
        with Vertical(id="goal-edit-dialog"):
            yield Static(
                f"Edit goal #{self._goal.id} — {escape(self._goal.title_display)}",
                id="goal-edit-title",
            )
            yield Static(
                "description — one short paragraph; it rides along in the "
                "research agent's prompt injection",
                id="goal-edit-desc-label",
            )
            yield TextArea(self._goal.description, id="goal-edit-desc")
            yield Static(
                "notes — free-form markdown; the agent reads it only via "
                "read_goal on demand (never injected into the prompt)",
                id="goal-edit-notes-label",
            )
            yield TextArea(self._goal.notes, id="goal-edit-notes")
            with Horizontal(id="goal-edit-buttons"):
                yield Button("Save", variant="primary", id="goal-edit-save")
                yield Button("Cancel", id="goal-edit-cancel")
            yield Static("ctrl+enter — save   esc — cancel", id="goal-edit-hint")

    def on_mount(self) -> None:
        self.query_one("#goal-edit-desc", TextArea).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        self._save()

    @on(Button.Pressed, "#goal-edit-save")
    def _save_pressed(self) -> None:
        self._save()

    @on(Button.Pressed, "#goal-edit-cancel")
    def _cancel_pressed(self) -> None:
        self.dismiss(None)

    def _save(self) -> None:
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

    def __init__(self, goal: Goal, candidates: list[Goal]) -> None:
        super().__init__()
        self._goal = goal
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
        for goal in self._candidates:
            text = f"{_goal_type_marker(goal)} {escape(goal.title_display)}"
            if goal.status is GoalStatus.DONE:
                text += " [green]done[/]"
            elif goal.status is GoalStatus.ARCHIVED:
                text += " [dim]archived[/]"
            hay = f"{goal.title} {goal.id}".lower()
            if query and query not in hay:
                continue
            listing.append(TaskItem(text, None, goal_id=goal.id))

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(ListView.Selected, "#goal-parent-list")
    def _picked(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, TaskItem) and item.goal_id:
            self.dismiss(item.goal_id)


class GoalScreen(ModalScreen[None]):
    """The full-screen roadmap view (opened from the sidebar's GOALS
    section, a goal row included): a status filter line on top, the
    milestone → sub-goal tree in the middle (Enter expands the goal's
    description/notes as collapsible rows right under it) and a key-hint
    footer. Local bindings shadow the app's globals (the merge-screen
    pattern): N new (sub-)goal · R research task for the goal · T task
    from the goal (the standard Ctrl+N form with the link preset) · E
    edit description/notes · M reparent (picker or Make top-level) · A
    archive/unarchive · Ctrl+Up/Down reorder among siblings · F cycle
    the status filter · Esc close."""

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Close", priority=True),
        Binding("f", "cycle_filter", "Filter"),
        Binding("n", "new_goal", "New goal"),
        Binding("r", "research_task", "Research task"),
        Binding("t", "task_from_goal", "Task from goal"),
        Binding("e", "edit_goal", "Edit"),
        Binding("m", "reparent", "Move"),
        Binding("a", "archive", "Archive"),
        Binding("ctrl+up", "move_up", "Move up", show=False),
        Binding("ctrl+down", "move_down", "Move down", show=False),
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
        height: 1;
        padding: 0 1;
        color: $text;
        text-style: bold;
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
        # The goal a refresh must put the cursor back onto (the ListView
        # index is position-based, a rebuild would drop it).
        self._focus_goal_id = focus_goal_id
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
        non-matching intermediate parents render as plain rows."""
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
            yield Static("", id="goal-screen-top", markup=False)
            yield ListView(id="goal-tree")
            yield Static(
                "n new · r research task · t task · e edit · m move · "
                "a archive/unarchive · f filter · ctrl+↑/↓ reorder · "
                "enter details · esc close",
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
        self.query_one("#goal-screen-top", Static).update(
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

    def _rebuild(self) -> None:
        """Re-render the tree, keeping the cursor on its goal."""
        if not self.is_mounted:
            return
        app = self._app()
        listing = self.query_one("#goal-tree", ListView)
        cursor_id = self._cursor_goal_id() or self._focus_goal_id
        listing.clear()
        for goal, depth in self._visible_rows():
            listing.append(
                TaskItem(self._row_text(goal, depth, app), None, goal_id=goal.id)
            )
            if goal.id in self._expanded:
                for row in self._detail_rows(goal, depth):
                    listing.append(row)
        if cursor_id:
            for index, child in enumerate(listing.children):
                if isinstance(child, TaskItem) and child.goal_id == cursor_id:
                    listing.index = index
                    break
        if listing.index is None and listing.children:
            # No cursor to restore (first open, or the filter hid the old
            # row): land on the first row so the action keys work right
            # away, without an extra arrow press.
            listing.index = 0
        self._update_top()

    def _refresh_all(self) -> None:
        self._app().refresh_sidebar()
        self._rebuild()

    # -- events ----------------------------------------------------------------

    @on(ListView.Selected, "#goal-tree")
    def _goal_selected(self, event: ListView.Selected) -> None:
        """Enter (or a click) on a goal row toggles its description/notes
        detail section."""
        item = event.item
        if isinstance(item, TaskItem) and item.goal_id:
            if item.goal_id in self._expanded:
                self._expanded.discard(item.goal_id)
            else:
                self._expanded.add(item.goal_id)
            event.stop()
            self._rebuild()

    # -- actions ------------------------------------------------------------------

    def action_close(self) -> None:
        self.dismiss(None)

    def action_cycle_filter(self) -> None:
        self._filter = FILTERS[(FILTERS.index(self._filter) + 1) % len(FILTERS)]
        self._rebuild()

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
            GoalCaptureScreen(self._pending_parent, self._pending_parent_label),
            self._captured,
        )

    def _captured(self, result: tuple[str, str, GoalType] | None) -> None:
        if result is None:
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
        """E: edit the cursor goal's description/notes."""
        goal = self._require_cursor()
        if goal is None:
            return
        self.app.push_screen(GoalEditScreen(goal), self._edited)

    def _edited(self, result: tuple[str, str] | None) -> None:
        if result is None:
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
        candidates = [
            g for g in app.manager.goals.goals.values() if g.id not in excluded
        ]
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

    def _move(self, *, up: bool) -> None:
        goal_id = self._cursor_goal_id()
        if goal_id is None:
            return
        self._app().manager.goals.move(goal_id, up=up)
        self._refresh_all()

    def action_move_up(self) -> None:
        self._move(up=True)

    def action_move_down(self) -> None:
        self._move(up=False)
