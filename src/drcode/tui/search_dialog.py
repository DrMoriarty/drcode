"""Task-archive search dialog (Ctrl+F) — find a task by keywords.

A modal over the main screen: a keyword input on top, the ranked result
list below. The heavy work never touches the event loop:

- the task archive (`<project>/.drcode/tasks/*.json`) is read into memory
  ONCE per dialog session, in a worker thread (blocking file I/O — the
  same treatment as the blocking git calls in `task_env`);
- every keystroke afterwards only re-filters the in-memory records
  (`task_search.search_records` — pure computation), debounced ~300 ms;
- mergedness markers (blocking git, `task_search.annotate_unmerged`) run
  in the same worker thread as the filtering, and only for short result
  lists (`MERGEDNESS_MAX_IDS`).

Row format: one logical line per hit — the head `#<id> [<status>/<kind>]
<markers> <date>` followed on the SAME line by the first prompt (capped,
but much longer than one row line could hold), which visually wraps onto
the second line — more of the original task description stays in view.
Styling follows the task statusbar (task_status_view): the id in bold
cyan, the status in its statusbar color, the kind letter colors and the
markers after the sidebar's TaskItem (verification cross, orange
unmerged sign); the date is deliberately dimmed so it recedes next to
the highlighted id and status.
The dialog has no return value
— it acts itself:

- enter on a highlighted row (or enter in the input with results) opens
  the task in the chat view (`view_task` + `chat_input.focus()`, exactly
  like a sidebar click);
- "c" on a highlighted row copies the task id to the clipboard (a toast
  confirms it) — for pasting into a new task's description or the
  dependency field (`set_task_dependencies` accepts short ids);
- escape closes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, ListView, Static

from .. import task_search
from ..task_search import TaskHit
from .sidebar import TaskItem

# Keystroke debounce: the in-memory filtering is cheap, but re-rendering
# the list per character still costs layout passes — wait for a pause.
_SEARCH_DEBOUNCE = 0.3
# Display cap of the first-prompt snippet on a result row's context
# line — the prompt gets a whole second line, so it can be much longer
# than a single-line row could show.
_PROMPT_CAP = 200
# Display cap of the last-error text appended to a row.
_ERROR_CAP = 50

# Result rows shown per query — a dialog list longer than that is useless
# (refine the query instead); also keeps the mergedness git calls cheap.
_DIALOG_LIMIT = 30


def _literal(text: str) -> str:
    """Escape *text* for textual's console markup so it renders literally.

    rich.markup.escape is NOT enough: it leaves a bracket that does not
    form a valid rich tag (e.g. "[queued/") untouched, but textual's
    stricter parser still trips over it.
    """
    return text.replace("\\", "\\\\").replace("[", "\\[")


class SearchDialog(ModalScreen[None]):
    """Modal task search: keyword input, ranked results, enter to open."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Close", priority=True),
        # No priority: while the input is focused it consumes "c" as text;
        # the binding fires only when the results list has focus.
        Binding("c", "copy_id", "Copy id"),
        Binding("down", "focus_results", "Results", show=False),
    ]

    DEFAULT_CSS = """
    SearchDialog {
        align: center middle;
    }
    #srch-dialog {
        width: 90;
        max-width: 95%;
        /* Two lines per result row — taller than the old single-line
         * dialog so roughly the same number of tasks stays visible. */
        height: 32;
        max-height: 90%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #srch-title { color: $text; margin-bottom: 1; text-style: bold; }
    #srch-results {
        height: 1fr;
        margin-top: 1;
        border: none;
        background: $surface-lighten-1;
    }
    # Row padding on the ListItem itself (the .task-item class lives on
    # the Label inside it) and a full-width label so the prompt wrapped
    # after the head actually flows onto the second line instead of
    # being clipped by the Label's auto width.
    #srch-results > ListItem { padding: 0 1; }
    #srch-results ListItem Label { width: 1fr; }
    #srch-hint {
        margin-top: 1;
        color: $text-disabled;
        text-align: right;
    }
    #srch-status {
        margin-top: 1;
        min-height: 1;
        color: $text-muted;
    }
    """

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        # The MAIN checkout's .drcode/tasks/ — never a task worktree (see
        # the worktree-isolation note in task_search's docstring).
        self._project_root = Path(project_root)
        # The archive, loaded once per session in a worker thread; None
        # until the load worker finishes. Each (task_id, data, fields) —
        # the shape task_search.load_archive returns.
        self._records: (
            list[tuple[str, dict[str, Any], list[tuple[str, str]]]] | None
        ) = None
        self._search_timer = None

    def compose(self) -> ComposeResult:
        with Vertical(id="srch-dialog"):
            yield Static("Find task", id="srch-title")
            yield Input(
                placeholder="keywords: title, prompt, edited files, agent summary…",
                id="srch-input",
            )
            yield ListView(id="srch-results")
            yield Static("", id="srch-status")
            yield Static(
                "enter — open task · c — copy id · esc — close", id="srch-hint"
            )

    def on_mount(self) -> None:
        self.query_one("#srch-input", Input).focus()
        self._set_status("Loading task archive…")
        self.run_worker(self._load_archive(), exclusive=True, group="srch-load")

    def action_cancel(self) -> None:
        self.dismiss(None)

    # -- archive loading (worker thread) --------------------------------------

    async def _load_archive(self) -> None:
        records = await asyncio.to_thread(task_search.load_archive, self._project_root)
        self._records = records
        self._set_status("")
        # The user may have typed while the archive was loading — run the
        # pending query now.
        self._run_search_now()

    # -- debounced in-memory search -------------------------------------------

    @on(Input.Changed, "#srch-input")
    def _input_changed(self, event: Input.Changed) -> None:
        self._schedule_search()

    def _schedule_search(self) -> None:
        if self._search_timer is not None:
            self._search_timer.stop()
        self._search_timer = self.set_timer(_SEARCH_DEBOUNCE, self._run_search_now)

    def _run_search_now(self) -> None:
        self._search_timer = None
        if self._records is None:
            return  # still loading; _load_archive re-runs when ready
        self.run_worker(self._search(), exclusive=True, group="srch-search")

    async def _search(self) -> None:
        query = self.query_one("#srch-input", Input).value
        assert self._records is not None
        try:
            hits = await asyncio.to_thread(self._search_sync, query)
        except ValueError:
            # Empty keyword list (whitespace query): show the idle hint.
            self._show_hits([])
            return
        # A newer keystroke superseded this run (the exclusive worker
        # usually prevents that; a race at the timer boundary remains).
        if self.query_one("#srch-input", Input).value != query:
            return
        self._show_hits(hits)

    def _search_sync(self, query: str) -> list[TaskHit]:
        """Filtering + mergedness in one off-loop step (blocking git)."""
        assert self._records is not None
        hits = task_search.search_records(
            self._records, query.split(), limit=_DIALOG_LIMIT
        )
        task_search.annotate_unmerged(self._project_root, hits)
        return hits

    # -- rendering ------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self.query_one("#srch-status", Static).update(text)

    @staticmethod
    def _hit_line(hit: TaskHit) -> str:
        """One result row: the head (#id, [status/kind], markers, date)
        followed on the same line by the first prompt, which wraps onto
        the second line at render time — more of the original task
        description stays in view. The id is bold cyan, the status wears
        the color of the task statusbar (task_status_view.refresh_
        task_status), the date is dimmed so it recedes."""
        kind_color = {"bug": "red", "research": "yellow"}.get(hit.kind, "green")
        status_color = {
            "queued": "yellow",
            "hold": "blue",
            "active": "cyan",
            "testing": "magenta",
            "completed": "green",
            "clarification": "yellow",
        }.get(hit.status, "white")
        # Only the id and kind letter carry their own colors; every
        # literal segment ("[status/", "]", prompt, error text) goes
        # through _literal so no accidental markup survives.
        head = f"[bold cyan]{_literal('#' + hit.task_id)}[/]"
        head += f" [{status_color}]{_literal('[' + hit.status + '/')}[/]"
        head += f"[{kind_color}]{_literal(hit.kind)}[/]"
        head += f"[{status_color}]{_literal(']')}[/]"
        markers = ""
        if hit.verification == "failed":
            markers += " [red]❌[/]"
        if hit.unfinished:
            markers += " [red]unfinished[/]"
        if hit.hook_merge_failed:
            markers += " [red]merge-failed[/]"
        if hit.unmerged:
            markers += " [orange]⎇ unmerged[/]"
        error = ""
        if hit.last_error_event:
            text = hit.last_error_event
            if len(text) > _ERROR_CAP:
                text = text[:_ERROR_CAP] + "…"
            error = f" [dim red]· {_literal(text)}[/]"
        head_line = f"{head}{markers} [dim]{hit.date}[/]{error}"
        prompt = task_search.first_prompt_snippet(hit.first_prompt, _PROMPT_CAP)
        if not prompt:
            return head_line
        return f"{head_line} {_literal(prompt)}"

    def _show_hits(self, hits: list[TaskHit]) -> None:
        results = self.query_one("#srch-results", ListView)
        highlighted = results.highlighted_child
        old_id = (
            highlighted.task_id
            if isinstance(highlighted, TaskItem) and highlighted.task_id
            else None
        )
        results.clear()
        query = self.query_one("#srch-input", Input).value.strip()
        if hits:
            for hit in hits:
                results.append(TaskItem(self._hit_line(hit), hit.task_id))
            # Keep the highlighted row stable across re-filters when it
            # still matches; otherwise land on the best hit.
            index = next(
                (
                    i
                    for i, child in enumerate(results.children)
                    if isinstance(child, TaskItem) and child.task_id == old_id
                ),
                0,
            )
            results.index = index
            self._set_status(
                f"{len(hits)} task(s)"
                + ("" if len(hits) < _DIALOG_LIMIT else "+")
                + " — enter opens, c copies the id"
            )
        elif query:
            results.append(
                TaskItem("[dim]No tasks matched — try fewer or shorter keywords.[/]")
            )
            self._set_status("")
        else:
            results.append(
                TaskItem("[dim]Type keywords to search the task archive.[/]")
            )
            self._set_status("")

    # -- actions ----------------------------------------------------------------

    @on(ListView.Selected, "#srch-results")
    def _result_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, TaskItem) and item.task_id:
            self._open_task(item.task_id)

    @on(Input.Submitted, "#srch-input")
    def _input_submitted(self, event: Input.Submitted) -> None:
        results = self.query_one("#srch-results", ListView)
        item = results.highlighted_child
        if isinstance(item, TaskItem) and item.task_id:
            self._open_task(item.task_id)
        else:
            results.focus()

    def _open_task(self, task_id: str) -> None:
        """Dismiss and open the task — the sidebar-click flow
        (on_list_view_selected: view_task + chat_input.focus(); view_task
        itself clears the new-task mode)."""
        from .app import DrCodeApp  # lazy: .app imports this module's user

        app = self.app
        self.dismiss(None)
        assert isinstance(app, DrCodeApp)
        app.view_task(task_id)
        app.chat_input.focus()

    def action_focus_results(self) -> None:
        self.query_one("#srch-results", ListView).focus()

    def action_copy_id(self) -> None:
        from .app import DrCodeApp  # lazy: .app imports this module's user

        item = self.query_one("#srch-results", ListView).highlighted_child
        if not isinstance(item, TaskItem) or not item.task_id:
            return
        app = self.app
        assert isinstance(app, DrCodeApp)
        # copy_to_clipboard alone is ignored by macOS Terminal.app (OSC52);
        # the app's helper adds the pbcopy fallback there.
        app._copy_with_fallback(f"#{item.task_id}")
        app.notify(f"#{item.task_id} скопирован", title="Find task", timeout=4)
