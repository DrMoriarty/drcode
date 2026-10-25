"""The interactive conflict-resolution screen (`drcode.tui.merge_screen`).

The visual fallback of the three-layer conflict stack: the algorithmic
merge engine (#d5841a9f) parks the conflicted merge in the task's
worktree, the agent resolver (#59d85021) tries to clear it unattended —
and when it gives up (`task.needs_human`) or the user simply wants to
decide by hand, this screen opens on the parked conflict.

The screen is a viewer + decision pad, never an autonomous actor:

- left: the conflicted files of the triage report with their LIVE status
  (resolved / N hunks / untouched — refreshed from the worktree, the
  report alone would go stale as the user works);
- right: the selected file in its zdiff3 form — ours (the task branch),
  the common ancestor (the `|||||||` section the engine's zdiff3 markers
  embed) and theirs (the base branch), with a cursor over the hunks;
- keys: ``o`` / ``t`` / ``b`` apply ours / theirs / both to the cursor
  hunk, ``e`` opens the file in the user's editor (or ``git mergetool``
  when no editor is configured), ``v`` runs the verification funnel of
  layer 2 (compile/ruff/pyright + the test suite) IN the worktree and
  shows the result, ``s`` commits the resolution on the task branch and
  ``a`` aborts the merge — both only behind an explicit second-press
  confirmation, mirroring the armed-ESC convention of the app.

The contract holds unchanged: the commit lands on the task's own branch
`drcode/<id>`; the merge into the user's checkout afterwards is a
fast-forward that the workflow hook or the user performs — the screen
never merges into the base branch itself. Every blocking call (git,
file IO, the checks) runs in a worker thread; every worker writes back
to the UI from the event loop only.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from typing import TYPE_CHECKING, Any, ClassVar, cast

from rich.segment import Segment
from rich.style import Style
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.geometry import Size
from textual.screen import ModalScreen
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.widgets import Label, ListItem, ListView, Static

from .. import merge_resolver
from ..conflict import (
    ConflictWorktree,
    FileState,
    Hunk,
    parse_hunks,
)
from ..merge_engine import MergeReport

if TYPE_CHECKING:
    from .app import DrCodeApp

# The two-second confirmation window of the armed commit/abort keys
# (mirrors the app's armed-ESC: a destructive key must not fire on the
# first, possibly accidental press).
_ARM_SECONDS = 2.0

# The gutter in front of every content line (cursor marker + space).
_GUTTER = "  "

# Section styles of the zdiff3 view: ours green, the common ancestor dim
# italic, theirs blue, the marker lines dim.
_SECTION_STYLES = {
    "head": "dim",
    "tail": "dim",
    "ours": "$success",
    "base": "italic dim",
    "theirs": "$accent",
}


class ConflictView(ScrollView, can_focus=True):
    """The conflict-file viewer: no-wrap line rendering with a hunk
    cursor.

    A line-API widget (own ``render_line``) instead of a Static child:
    conflict files are code — lines must never soft-wrap (a wrapped line
    would shift every following hunk row and break the cursor/scroll
    mapping), horizontal overflow scrolls instead."""

    BINDINGS: ClassVar = [
        Binding("up,k", "cursor_up", "Previous hunk", show=False),
        Binding("down,j", "cursor_down", "Next hunk", show=False),
    ]

    DEFAULT_CSS = """
    ConflictView {
        background: $surface;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lines: list[tuple[str, str | None]] = []
        self._hunk_lines: dict[int, int] = {}  # content line -> hunk index
        self._hunks: list[Hunk] = []
        self.cursor: int = -1

    # -- content -----------------------------------------------------------------

    def set_file(self, text: str, hunks: list[Hunk]) -> None:
        """Render a file's content with the conflict sections colored and
        reset the cursor to the first hunk."""
        self._hunks = hunks
        self._hunk_lines = {}
        for idx, hunk in enumerate(hunks):
            for line_no in range(hunk.start, hunk.end):
                self._hunk_lines[line_no] = idx
        self._lines = self._styled_lines(text.split("\n"), hunks)
        self.cursor = 0 if hunks else -1
        self.virtual_size = Size(
            max((len(line) for line in self._lines), default=1) + 2,
            len(self._lines),
        )
        self.scroll_to(x=0, y=0, animate=False)
        self.refresh()

    @staticmethod
    def _styled_lines(
        lines: list[str], hunks: list[Hunk]
    ) -> list[tuple[str, str | None]]:
        """Content lines as rich Texts, colored per zdiff3 section."""
        section_of: dict[int, str] = {}
        for hunk in hunks:
            for line_no in range(hunk.start, hunk.end):
                if line_no == hunk.start:
                    section_of[line_no] = "head"
                elif line_no == hunk.end - 1:
                    section_of[line_no] = "tail"
                elif line_no <= hunk.start + len(hunk.ours):
                    section_of[line_no] = "ours"
                elif hunk.base and line_no <= (
                    hunk.start + len(hunk.ours) + 1 + len(hunk.base)
                ):
                    section_of[line_no] = "base"
                else:
                    section_of[line_no] = "theirs"
        result: list[tuple[str, str | None]] = []
        for line_no, line in enumerate(lines):
            section = section_of.get(line_no)
            result.append((line, _SECTION_STYLES.get(section) if section else None))
        return result

    # -- cursor ------------------------------------------------------------------

    def _set_cursor(self, index: int) -> None:
        if not self._hunks:
            return
        old = self.cursor
        self.cursor = max(0, min(index, len(self._hunks) - 1))
        if old != self.cursor:
            for line_no in range(self._hunks[old].start, self._hunks[old].end):
                self.refresh_line(line_no)
        hunk = self._hunks[self.cursor]
        for line_no in range(hunk.start, hunk.end):
            self.refresh_line(line_no)
        self.scroll_to(y=max(0, hunk.start - 2), animate=False)

    def action_cursor_up(self) -> None:
        self._set_cursor(self.cursor - 1)

    def action_cursor_down(self) -> None:
        self._set_cursor(self.cursor + 1)

    def current_hunk(self) -> Hunk | None:
        if 0 <= self.cursor < len(self._hunks):
            return self._hunks[self.cursor]
        return None

    # -- rendering -----------------------------------------------------------------

    def _rich_style(self, spec: str | None) -> Style | None:
        """A rich Style from a spec that may carry Textual CSS variables
        (`$success`) — rich cannot parse those, resolve them against the
        app's active theme first."""
        if not spec:
            return None
        variables = self.app.get_css_variables()
        resolved = " ".join(
            variables.get(word[1:], word) if word.startswith("$") else word
            for word in spec.split()
        )
        return Style.parse(resolved)

    def render_line(self, y: int) -> Strip:
        line_no = self.scroll_offset.y + y
        if 0 <= line_no < len(self._lines):
            text, style_spec = self._lines[line_no]
            hunk_idx = self._hunk_lines.get(line_no)
            gutter = _GUTTER
            style = self._rich_style(style_spec)
            if hunk_idx is not None and hunk_idx == self.cursor:
                gutter = "❯ "
                style = (style or Style()) + Style(bold=True)
            padded = (gutter + text).ljust(
                max(self.virtual_size.width, self.size.width) - 2
            )
            return Strip([Segment(padded, style)])
        return Strip.blank(self.size.width)


class FileItem(ListItem):
    """One row of the conflicted-file list; carries the path for the
    selection handler."""

    def __init__(self, state: FileState) -> None:
        super().__init__(Label(f"{state.path}\n   {state.badge()}"))
        self.path = state.path


class ConflictScreen(ModalScreen[None]):
    """Manual merge-conflict resolution for one task's parked merge."""

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Close", priority=True),
        Binding("o", "resolve('ours')", "Ours", priority=True),
        Binding("t", "resolve('theirs')", "Theirs", priority=True),
        Binding("b", "resolve('both')", "Both", priority=True),
        Binding("e", "external_edit", "Edit", priority=True),
        Binding("v", "verify", "Check", priority=True),
        Binding("s", "commit", "Commit", priority=True),
        Binding("a", "abort_merge", "Abort", priority=True),
        Binding("r", "refresh_state", "Refresh", priority=True),
    ]

    DEFAULT_CSS = """
    ConflictScreen {
        align: center middle;
    }
    #merge-dialog {
        width: 95%;
        max-width: 160;
        height: 95%;
        padding: 0 1 1 1;
        background: $panel;
        border: thick $accent;
    }
    #merge-title {
        height: 1;
        padding: 0 1;
        color: $text;
        text-style: bold;
    }
    #merge-body { height: 1fr; }
    #merge-files {
        width: 34;
        min-width: 20;
        border-right: solid $panel;
        background: $surface;
    }
    #merge-files ListItem { padding: 0 1; }
    #merge-view { width: 1fr; }
    #merge-footer {
        dock: bottom;
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }
    #merge-checks {
        height: auto;
        max-height: 12;
        padding: 0 1;
        color: $text-muted;
        display: none;
        border-top: solid $panel;
    }
    #merge-checks.visible { display: block; }
    """

    def __init__(self, task_id: str) -> None:
        super().__init__()
        self.task_id = task_id
        self.wt: ConflictWorktree | None = None
        self._armed: str | None = None
        self._checks_ok: bool | None = None

    # -- composition ------------------------------------------------------------

    @property
    def files_list(self) -> ListView:
        return self.query_one("#merge-files", ListView)

    @property
    def view(self) -> ConflictView:
        return self.query_one("#merge-view", ConflictView)

    def compose(self) -> ComposeResult:
        with Vertical(id="merge-dialog"):
            yield Static("Merge conflicts", id="merge-title")
            with Horizontal(id="merge-body"):
                yield ListView(id="merge-files")
                yield ConflictView(id="merge-view")
            yield Static("", id="merge-checks")
            yield Static(
                "o/t/b hunk ours/theirs/both · e edit · v check · "
                "s commit · a abort · r refresh · esc close",
                id="merge-footer",
            )

    # -- preparation (the engine's reversed merge) -------------------------------

    def on_mount(self) -> None:
        self._prepare()

    @work(exclusive=True, group="merge-prepare")
    async def _prepare(self) -> None:
        """Deploy (or re-detect) the parked conflict through the engine
        (`merge_resolver.prepare`): an already-parked merge is re-triaged
        in place, a fresh one is attempted with the safe auto-resolvers
        on. Blocking git — off the UI thread."""
        app = cast("DrCodeApp", self.app)
        env = app.task_env
        if not env.active:
            self._set_title("Merge conflicts — worktree isolation is off")
            return
        root, note = await asyncio.to_thread(env.ensure_root, self.task_id)
        if note:
            self._set_title(f"Merge conflicts — unavailable: {note}")
            return
        base = await asyncio.to_thread(env.base_branch)
        report: MergeReport = await asyncio.to_thread(
            merge_resolver.prepare, app.project_root, self.task_id, root, base
        )
        self.wt = ConflictWorktree(app.project_root, self.task_id, root, report=report)
        if not report.ok:
            self._set_title(f"Merge conflicts — {report.message}")
            return
        if not report.conflicted:
            self._set_title(
                f"{report.message} — nothing to resolve; "
                "`git merge --ff-only` from the base checkout finishes it"
            )
            return
        self._set_title(
            f"#{self.task_id} — merge {report.base_branch} into "
            f"{report.branch}: {len(report.remaining)} file(s) to resolve"
        )
        await self._reload_files()

    def _set_title(self, text: str) -> None:
        # A worker may reach this after the user already closed the
        # screen (ESC mid-prepare) — an unmounted screen is a no-op.
        if not self.is_mounted:
            return
        self.query_one("#merge-title", Static).update(text)

    # -- file list / viewer ---------------------------------------------------------

    async def _reload_files(self) -> None:
        """Rebuild the file list from the LIVE worktree state (resolved
        files stay visible — the user watches their own progress). The
        git/file reads run in a thread, the widget updates here."""
        if self.wt is None or not self.is_mounted:
            return
        try:
            selected = self._selected_path()
            states = await asyncio.to_thread(self.wt.file_states)
            items = [FileItem(s) for s in states]
            self.files_list.clear()
            if items:
                self.files_list.extend(items)
            # Keep the selection (or default to the first file), then load
            # it into the viewer — the highlighted event fires on the
            # index set and refreshes again; the explicit call covers the
            # no-event edge.
            target = 0
            if selected is not None:
                for index, item in enumerate(items):
                    if item.path == selected:
                        target = index
                        break
            if items:
                self.files_list.index = target
            await self._refresh_view()
        except NoMatches:
            # The user closed the screen mid-refresh (ESC while a worker
            # holds the loop) — an unmounted screen is a no-op.
            return
        if states and all(s.resolved for s in states):
            self._set_title(
                f"#{self.task_id} — all conflicts resolved; "
                "v to check, s to commit the merge"
            )

    def _selected_path(self) -> str | None:
        try:
            item = self.files_list.highlighted_child
        except NoMatches:  # the screen is already unmounted (a late worker)
            return None
        return item.path if isinstance(item, FileItem) else None

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if isinstance(event.item, FileItem):
            self.run_worker(self._refresh_view(), exclusive=True, group="merge-view")

    async def _refresh_view(self) -> None:
        """Load the selected file into the viewer (with its hunks)."""
        if self.wt is None or not self.is_mounted:
            return
        path = self._selected_path()
        if path is None:
            self.view.set_file("", [])
            return
        content = await asyncio.to_thread(self.wt.read_file, path)
        if content is None:
            self.view.set_file("(file absent in the worktree)", [])
            return
        self.view.set_file(content, parse_hunks(content))

    # -- hunk decisions -----------------------------------------------------------

    def action_resolve(self, choice: str) -> None:
        """o / t / b: apply the decision to the cursor hunk (staged as
        soon as the file is marker-free)."""
        if self.wt is None:
            return
        path = self._selected_path()
        hunk = self.view.current_hunk()
        if path is None or hunk is None:
            self.notify("No hunk under the cursor", title="Merge", timeout=3)
            return
        self._apply(path, hunk, choice)

    @work(exclusive=False, group="merge-op")
    async def _apply(self, path: str, hunk: Hunk, choice: str) -> None:
        if self.wt is None:
            return
        ok, message = await asyncio.to_thread(self.wt.apply_hunk, path, hunk, choice)
        if not ok:
            self.notify(message, title="Merge", severity="error", timeout=5)
        await self._reload_files()
        # No disarm here: the armed state belongs to the commit/abort keys
        # (action_commit / action_abort_merge validate the merge state
        # themselves before firing), and a blanket wipe would swallow an
        # arm the user made while this worker was still finishing — the
        # confirmation press would then silently re-arm instead of
        # committing (seen as a flaky "resolution committed" timeout).

    # -- external editor / mergetool ----------------------------------------------

    def action_external_edit(self) -> None:
        """e: open the file in the user's editor ($VISUAL/$EDITOR), or
        `git mergetool` for the whole merge when none is configured. The
        app is suspended for the duration — an interactive subprocess
        cannot run inside Textual's alternate screen."""
        if self.wt is None:
            return
        path = self._selected_path()
        if path is None:
            return
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if editor:
            cmd: list[str] = [*editor.split(), str(self.wt.worktree / path)]
        else:
            cmd = ["git", "mergetool", "--", path]
        try:
            with self.app.suspend():
                subprocess.run(cmd, cwd=self.wt.worktree, check=False)
        except Exception as e:  # noqa: BLE001 - any launch failure is a toast
            self.notify(
                f"Editor failed: {type(e).__name__}: {e}",
                title="Merge",
                severity="error",
                timeout=6,
            )
            return
        self.action_refresh_state()

    # -- verification funnel (layer 2, in the worktree) --------------------------

    def action_verify(self) -> None:
        if self.wt is None or not self.wt.in_progress():
            self.notify("No merge is in progress", title="Merge", timeout=4)
            return
        panel = self.query_one("#merge-checks", Static)
        panel.add_class("visible")
        panel.update("Running checks… (compile / ruff / pyright, then tests)")
        self._run_checks()

    @work(exclusive=True, group="merge-checks")
    async def _run_checks(self) -> None:
        if self.wt is None:
            return
        worktree = self.wt.worktree
        checks = await asyncio.to_thread(merge_resolver.mechanical_checks, worktree)
        tests = await asyncio.to_thread(merge_resolver.run_test_suite, worktree)
        self._checks_ok = all(c.ok for c in checks) and tests.ok
        panel = self.query_one("#merge-checks", Static)
        lines = [c.line for c in checks] + [tests.line]
        panel.update("Checks:\n" + "\n".join(f"  {line}" for line in lines))
        self.notify(
            "All checks passed" if self._checks_ok else "Checks FAILED",
            title="Merge",
            severity="information" if self._checks_ok else "error",
            timeout=5,
        )

    # -- commit / abort (armed, second-press confirmation) -----------------------

    def _disarm(self) -> None:
        self._armed = None

    def _arm(self, action: str) -> None:
        self._armed = action
        verb = "commit the resolution" if action == "commit" else "ABORT the merge"
        extra = ""
        if action == "commit" and self._checks_ok is not True:
            extra = " (checks not passed — v runs them)"
        self.notify(
            f"Press s again within {int(_ARM_SECONDS)}s to {verb}{extra}",
            title="Merge",
            timeout=_ARM_SECONDS + 0.5,
        )

    def action_commit(self) -> None:
        if self.wt is None:
            return
        if self._armed != "commit":
            self._arm("commit")
            return
        self._disarm()
        if not self.wt.in_progress():
            self.notify("No merge is in progress", title="Merge", timeout=4)
            return
        self._do_commit()

    @work(exclusive=True, group="merge-commit")
    async def _do_commit(self) -> None:
        if self.wt is None or self.wt.report is None:
            return
        branch = self.wt.report.branch or f"drcode/{self.task_id}"
        base = self.wt.report.base_branch or "base"
        message = (
            f"Merge {base} into {branch}: resolve conflicts "
            f"(manual, via the DrCode conflict screen)"
        )
        ok, out = await asyncio.to_thread(self.wt.commit, message)
        app = cast("DrCodeApp", self.app)
        task = app.manager.get(self.task_id)
        task.needs_human = False
        task.touch()
        app.manager.storage.save(task)
        if ok:
            app._show_event(
                task,
                f"Conflict resolution committed on {branch} via the "
                f"conflict screen — the merge into {base} is now a "
                f"fast-forward (`git merge --ff-only {branch}`).",
                kind="workflow",
            )
            self._set_title(
                f"#{self.task_id} — resolution committed on {branch}; "
                f"`git merge --ff-only {branch}` finishes it"
            )
            self.notify(f"Committed on {branch}", title="Merge", timeout=5)
            await self._reload_files()
        else:
            self.notify(
                f"Commit refused: {out}", title="Merge", severity="error", timeout=8
            )

    def action_abort_merge(self) -> None:
        if self.wt is None:
            return
        if self._armed != "abort":
            self._arm("abort")
            return
        self._disarm()
        self._do_abort()

    @work(exclusive=True, group="merge-abort")
    async def _do_abort(self) -> None:
        if self.wt is None:
            return
        ok, out = await asyncio.to_thread(self.wt.abort)
        if ok:
            self.notify(
                "Merge aborted — worktree back to the pre-merge state",
                title="Merge",
                timeout=5,
            )
            self.dismiss()
        else:
            self.notify(
                f"Abort failed: {out}", title="Merge", severity="error", timeout=6
            )

    # -- refresh / close -----------------------------------------------------------

    def action_refresh_state(self) -> None:
        """r: re-triage the live merge and reload everything (an external
        editor may have changed files while the screen was suspended)."""
        if self.wt is None:
            return
        self._refresh()

    @work(exclusive=True, group="merge-refresh")
    async def _refresh(self) -> None:
        if self.wt is None or not self.is_mounted:
            return
        await asyncio.to_thread(self.wt.refresh)
        await self._reload_files()

    def action_close(self) -> None:
        self.dismiss()
