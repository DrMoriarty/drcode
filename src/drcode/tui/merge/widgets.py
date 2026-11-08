"""List-item widgets of the conflict screen's file pane.

The conflicted-file list (the conflict mode) and the numstat list (the
ff-ready / staged modes) — one module, since the two item kinds are
interchangeable in the selection handlers (`FileItem.path` /
`DiffStatItem.path`).
"""

from __future__ import annotations

from rich.markup import escape
from textual.widgets import Label, ListItem

from ...conflict import DiffStat, FileState
from .stages import _elide_middle


class FileItem(ListItem):
    """One row of the conflicted-file list; carries the path for the
    selection handler."""

    def __init__(self, state: FileState) -> None:
        super().__init__(Label(f"{state.path}\n   {state.badge()}"))
        self.path = state.path


def _diffstat_row_text(stat: DiffStat, max_row: int | None) -> str:
    """The rendered row text of a numstat list item: the (middle-elided)
    path, two spaces and the `+42 −7` counts (a binary file shows
    `bin`). `max_row` is the whole row's cell budget (path + separator +
    counts TOGETHER): the counts are always rendered in full — eliding
    THEM would defeat the row's purpose — so the path gets whatever
    cells remain after its own counts block; a long path can therefore
    never push the counts off the pane. Pure string work —
    unit-testable without an app."""
    if stat.binary:
        counts, plain = "[dim]bin[/]", "bin"
    else:
        counts = f"[$success]+{stat.added}[/] [$error]−{stat.deleted}[/]"
        plain = f"+{stat.added} −{stat.deleted}"
    path = stat.path
    if max_row is not None:
        budget = max(max_row - len(plain) - 2, 4)
        path = _elide_middle(path, budget)
    # escape: the path is arbitrary repo content — literal [brackets]
    # in it must never trip the markup.
    return f"{escape(path)}  {counts}"


class DiffStatItem(ListItem):
    """One row of the numstat file list (ff-ready / staged modes):
    `path  +42 −7`, additions green, deletions red; a binary file shows
    `bin`. `max_path` bounds the path's OWN cell width — the counts are
    always rendered in full (eliding them would defeat the row's
    purpose), the path is middle-elided (the file name's tail kept) to
    fit. Carries the path for the selection handler — clicking a row
    loads the file's worktree content into the viewer (the worktree
    holds the branch/resolved state in both diff modes)."""

    def __init__(self, stat: DiffStat, max_path: int | None = None) -> None:
        if stat.binary:
            counts = "[dim]bin[/]"
        else:
            counts = f"[$success]+{stat.added}[/] [$error]−{stat.deleted}[/]"
        # Middle-elide to the caller's budget so the count block stays
        # inside the label even for deep paths; None = no budget known.
        shown = stat.path if max_path is None else _elide_middle(stat.path, max_path)
        # escape: the path is arbitrary repo content — literal [brackets]
        # in it must never trip the markup.
        super().__init__(Label(f"{escape(shown)}  {counts}"))
        self.path = stat.path
