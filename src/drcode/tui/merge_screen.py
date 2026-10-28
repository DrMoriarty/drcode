"""The interactive conflict-resolution screen (`drcode.tui.merge_screen`).

The visual fallback of the three-layer conflict stack: the algorithmic
merge engine (#d5841a9f) parks the conflicted merge in the task's
worktree, the agent resolver (#59d85021) tries to clear it unattended —
and when it gives up (`task.needs_human`) or the user simply wants to
decide by hand, this screen opens on the parked conflict.

The ``v`` funnel runs with visible per-stage progress: a panel line per
check (braille spinner + elapsed seconds while running, ok/FAILED/skip
marks when done, plus a percent/ETA footer estimated from the rolling
per-check durations persisted across runs next to the verify report) —
the checks themselves stay serial and off the UI thread, the panel is
animated by an independent timer while the worker awaits them.

The same panel also carries the OPENING prepare phase (the engine's
reversed merge + triage + safe auto-resolvers — seconds on small
repositories, tens of seconds on large ones): one line per step driven
by `merge_resolver.prepare`'s progress callback, its own percent/ETA
footer over the rolling per-step durations persisted next to the triage
report — and it disappears once the outcome is in and the file list has
loaded. The short blocking ops (``r`` refresh, ``s`` commit, ``a``
abort, ``f`` ff-merge) show a single busy spinner line in the same
panel and lock every key but ``esc`` (``check_action``) until their
worker finishes — a git operation must never race a key press.

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
  shows the result (a failure is named check-by-check in the toast),
  ``d`` opens the full output of the selected check in a log viewer
  (``[`` / ``]`` step through the checks, ESC returns here — the
  resolution state is untouched), ``E`` opens the persisted full report
  in the user's editor, ``s`` commits the resolution on the task branch
  and ``a`` aborts the merge — both only behind an explicit second-press
  confirmation, mirroring the armed-ESC convention of the app. A
  successful commit or ff-merge is the logical end of the work here:
  the screen closes itself (the outcome toast outlives it).

When the engine's auto-resolution left NOTHING to resolve AND the
report vouches the fast-forward is real (`fast_forward is True` — a
clean merge, an up-to-date branch or the rerere replay that concluded),
the screen switches to its "ff-ready" mode instead: the
resolution keys are disabled (the footer names them as unavailable) and
``f`` becomes the one relevant action — it fast-forwards the MAIN
checkout's base branch to `drcode/<id>` (`task_env.ff_merge_branch`).
That is the contract's user-triggered half: the screen still never
merges into the base on its own, only behind the explicit ``f`` press,
and only `--ff-only` — a refusal (the base moved ahead, a dirty tree)
is an error toast, never a conflicted merge to resolve in the user's
tree; the user re-runs the reversed merge by reopening the screen. A
successful ``f`` closes the screen, same as a successful ``s``.

When the engine's auto-resolution left NOTHING to resolve (the report
is ok and not conflicted — a clean merge, an up-to-date branch or the
rerere replay), the screen switches to its "ff-ready" mode instead: the
resolution keys are disabled (the footer names them as unavailable) and
``f`` becomes the one relevant action — it fast-forwards the MAIN
checkout's base branch to `drcode/<id>` (`task_env.ff_merge_branch`).
That is the contract's user-triggered half: the screen still never
merges into the base on its own, only behind the explicit ``f`` press,
and only `--ff-only` — a refusal (the base moved ahead, a dirty tree)
is an error toast, never a conflicted merge to resolve in the user's
tree; the user re-runs the reversed merge by reopening the screen.

Every ``v`` run also rewrites the full funnel report (verdict, status
lines, complete check outputs — modeled on `ResolverOutcome.summary()`)
to ``.drcode/merge_reports/<task-id>-verify.txt`` next to the engine's
triage report, so the failure details survive the screen being closed.

The contract holds unchanged: the commit lands on the task's own branch
`drcode/<id>`; the merge into the user's checkout afterwards is a
fast-forward that the workflow hook or the user performs — and the only
merge the screen itself performs into the base is the explicit, `--ff-only`
``f`` action above (refusal, never a conflict, when the base is not an
ancestor). Every blocking call (git, file IO, the checks) runs in a
worker thread; every worker writes back to the UI from the event loop
only.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass
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
from textual.timer import Timer
from textual.widgets import Label, ListItem, ListView, Static

from .. import merge_resolver
from ..conflict import (
    ConflictWorktree,
    FileState,
    Hunk,
    parse_hunks,
)
from ..merge_engine import MergeReport, is_fast_forward
from ..merge_resolver import CheckResult
from ..tools import child_priority_kwargs

if TYPE_CHECKING:
    from .app import DrCodeApp

# The two-second confirmation window of the armed commit/abort keys
# (mirrors the app's armed-ESC: a destructive key must not fire on the
# first, possibly accidental press).
_ARM_SECONDS = 2.0

# The gutter in front of every content line (cursor marker + space).
_GUTTER = "  "

# The verification-funnel progress indicator: braille spinner frames and
# how often the panel's animation timer redraws the running line.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_INTERVAL = 0.15


def _fmt_seconds(seconds: float | None) -> str:
    """Compact human duration for the progress panel ("42s", "1m20s")."""
    if seconds is None:
        return "?"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


@dataclass
class _CheckStage:
    """One stage of the interactive funnel's live progress: `key` names
    the persisted-timing slot, `fn_name` the `merge_resolver` function to
    resolve at CALL time (never captured — tests monkeypatch the module
    attribute), `status` is pending / running / done."""

    key: str
    fn_name: str
    label: str
    status: str = "pending"
    result: CheckResult | None = None
    started: float | None = None
    duration: float | None = None


@dataclass
class _PrepareStage:
    """One step of the screen's prepare phase (the engine's reversed
    merge + triage + safe auto-resolvers) rendered in the same progress
    panel as the funnel: `key` names the persisted-timing slot
    (`<task-id>-prepare-timings.json`), `status` is pending / running /
    done. The steps advance from `merge_resolver.prepare`'s progress
    callback, applied on the event loop."""

    key: str
    label: str
    status: str = "pending"
    started: float | None = None
    duration: float | None = None


def _stages_estimate(
    stage: _CheckStage | _PrepareStage,
    averages: dict[str, float],
    defaults: dict[str, float],
) -> float:
    """The historical duration of `stage`: the persisted rolling average
    when there is one, else the built-in default."""
    return averages.get(stage.key) or defaults.get(stage.key, 30.0)


def _stages_eta(
    stages: list[_CheckStage] | list[_PrepareStage],
    averages: dict[str, float],
    defaults: dict[str, float],
) -> float:
    """ETA over the not-yet-done stages: each remaining stage's
    historical duration, the running one credited with
    `historical − elapsed` (floored at 10% of it)."""
    now = time.monotonic()
    total = 0.0
    for stage in stages:
        if stage.status == "done":
            continue
        estimate = _stages_estimate(stage, averages, defaults)
        if stage.status == "running" and stage.started is not None:
            total += max(estimate - (now - stage.started), estimate * 0.1)
        else:
            total += estimate
    return total


def _stages_percent(
    stages: list[_CheckStage] | list[_PrepareStage],
    averages: dict[str, float],
    defaults: dict[str, float],
) -> int:
    """Completed / total stages, the running one credited proportionally
    to its historical duration (capped at 95% — it never claims to be
    done while it is still running)."""
    if not stages:
        return 0
    now = time.monotonic()
    done = 0.0
    for stage in stages:
        if stage.status == "done":
            done += 1
        elif stage.status == "running" and stage.started is not None:
            done += min(
                (now - stage.started) / _stages_estimate(stage, averages, defaults),
                0.95,
            )
    return int(done / len(stages) * 100)


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


class CheckOutputView(ScrollView, can_focus=True):
    """The raw output of one verification check: no-wrap line rendering
    with virtual sizing computed from the longest line — pyright's lines
    are wide, horizontal overflow scrolls instead of soft-wrapping (the
    same line-API approach as `ConflictView`)."""

    DEFAULT_CSS = """
    CheckOutputView {
        background: $surface;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lines: list[str] = []

    def set_output(self, text: str) -> None:
        """Replace the shown output and scroll back to the top-left."""
        self._lines = text.split("\n") if text else ["(no output)"]
        self.virtual_size = Size(
            max((len(line) for line in self._lines), default=1) + 2,
            len(self._lines),
        )
        self.scroll_to(x=0, y=0, animate=False)
        self.refresh()

    def render_line(self, y: int) -> Strip:
        line_no = self.scroll_offset.y + y
        if 0 <= line_no < len(self._lines):
            padded = ("  " + self._lines[line_no]).ljust(
                max(self.virtual_size.width, self.size.width) - 2
            )
            return Strip([Segment(padded, None)])
        return Strip.blank(self.size.width)


class CheckDetailsScreen(ModalScreen[None]):
    """Full-screen viewer of one check's raw output, pushed OVER the
    ConflictScreen: esc returns without touching the resolution state
    (the conflict screen stays mounted underneath), ``[`` / ``]`` step
    through the check list — skipped checks show their explanatory
    output ("no tests/ directory", "pytest unavailable")."""

    BINDINGS: ClassVar = [
        Binding("escape", "back", "Back", priority=True),
        Binding("left_square_bracket", "prev_check", "Previous check", show=False),
        Binding("right_square_bracket", "next_check", "Next check", show=False),
    ]

    DEFAULT_CSS = """
    CheckDetailsScreen {
        align: center middle;
    }
    #check-details-dialog {
        width: 95%;
        max-width: 160;
        height: 95%;
        padding: 0 1 1 1;
        background: $panel;
        border: thick $accent;
    }
    #check-details-header {
        height: 1;
        padding: 0 1;
        color: $text;
        text-style: bold;
    }
    #check-details-view { height: 1fr; }
    #check-details-footer {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, checks: list[CheckResult], index: int = 0) -> None:
        super().__init__()
        self._checks = checks
        self._index = max(0, min(index, len(checks) - 1))

    def compose(self) -> ComposeResult:
        with Vertical(id="check-details-dialog"):
            # markup=False: check names / the "[n/m]" counter would trip
            # Rich markup.
            yield Static("", id="check-details-header", markup=False)
            yield CheckOutputView(id="check-details-view")
            yield Static(
                "[ ] prev/next check · arrows scroll · esc back",
                id="check-details-footer",
                markup=False,
            )

    def on_mount(self) -> None:
        self._show()

    def _show(self) -> None:
        check = self._checks[self._index]
        self.query_one("#check-details-header", Static).update(
            f"[{self._index + 1}/{len(self._checks)}] {check.line}"
        )
        self.query_one("#check-details-view", CheckOutputView).set_output(check.output)

    def action_prev_check(self) -> None:
        if self._checks:
            self._index = (self._index - 1) % len(self._checks)
            self._show()

    def action_next_check(self) -> None:
        if self._checks:
            self._index = (self._index + 1) % len(self._checks)
            self._show()

    def action_back(self) -> None:
        self.dismiss()


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
        Binding("d", "check_details", "Details", priority=True),
        Binding("E", "open_report", "Report", priority=True),
        Binding("s", "commit", "Commit", priority=True),
        Binding("a", "abort_merge", "Abort", priority=True),
        Binding("f", "ff_merge", "Merge into base", priority=True),
        Binding("r", "refresh_state", "Refresh", priority=True),
    ]

    # The resolution keys that lose their meaning in the ff-ready mode
    # ("nothing to resolve"): check_action disables them so a stray
    # press does nothing, and the footer names them as unavailable —
    # the keys stay documented instead of silently vanishing.
    _FF_DISABLED_ACTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "resolve",
            "external_edit",
            "verify",
            "check_details",
            "commit",
            "abort_merge",
        }
    )

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
        # ff-ready mode: the engine's auto-resolution left nothing to
        # resolve (report ok, not conflicted) — the branch is ready to
        # be fast-forwarded into the base. `_ff_done` marks the merge
        # already performed here (further f presses would be no-ops).
        self._ff_ready: bool = False
        self._ff_done: bool = False
        # The last verification run (checks + the test suite), newest
        # first-shown check ordered as in the panel — the details viewer
        # reads from here.
        self.check_results: list[CheckResult] = []
        # Live progress of the running funnel: per-stage state, the
        # panel's animation timer (independent of the worker — it must
        # tick while the blocking checks run in their threads), the
        # persisted rolling averages feeding the ETA, and a generation
        # counter so a cancelled run's cleanup cannot stop the NEXT
        # run's timer.
        self._stages: list[_CheckStage] = []
        self._checks_timer: Timer | None = None
        self._spinner_frame = 0
        self._avg_timings: dict[str, float] = {}
        self._checks_generation = 0
        # Live progress of the prepare phase (on_mount's reversed merge):
        # same panel, same animation mechanics — the steps advance from
        # `prepare`'s progress callback (applied on the event loop), the
        # panel disappears once the file list has loaded.
        self._prepare_stages: list[_PrepareStage] = []
        self._prepare_timer: Timer | None = None
        self._prepare_avg_timings: dict[str, float] = {}
        self._prepare_measured: dict[str, float] = {}
        self._prepare_failed = False
        # Busy state of the short blocking ops (r/s/a/f): one spinner
        # line in the same panel, plus a check_action gate so git cannot
        # race a key press. `_panel_saved` holds the panel content the
        # busy line temporarily replaced (e.g. a finished funnel's
        # summary), restored when the op ends.
        self._busy: str | None = None
        self._busy_started: float | None = None
        self._busy_timer: Timer | None = None
        self._busy_visible = False
        self._busy_generation = 0
        self._panel_saved: str | None = None
        # What the merge actually carries (the task branch's last commit
        # subject, falling back to the task title): fetched once at the
        # very start of prepare and woven into EVERY title via
        # `_set_title`, so it survives all the title rewrites of the
        # prepare's final states. None = unavailable — the titles stay
        # as they were.
        self._merge_desc: str | None = None

    # -- composition ------------------------------------------------------------

    @property
    def files_list(self) -> ListView:
        return self.query_one("#merge-files", ListView)

    @property
    def view(self) -> ConflictView:
        return self.query_one("#merge-view", ConflictView)

    def compose(self) -> ComposeResult:
        with Vertical(id="merge-dialog"):
            # markup=False: the title embeds a commit subject (and git
            # notes/messages) that may carry literal [brackets] — they
            # must never trip Rich markup.
            yield Static("Merge conflicts", id="merge-title", markup=False)
            with Horizontal(id="merge-body"):
                yield ListView(id="merge-files")
                yield ConflictView(id="merge-view")
            # markup=False: the progress lines carry literal [ok] / [ ]
            # brackets that must never trip Rich markup.
            yield Static("", id="merge-checks", markup=False)
            yield Static(
                "o/t/b hunk ours/theirs/both · e edit · v check · "
                "d details · E report · s commit · a abort · r refresh · esc close",
                id="merge-footer",
            )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Gate the key set by the mode: while a blocking op is busy the
        whole key set except ``esc`` is inert (a git operation must not
        race a second one), in the ff-ready state the resolution keys
        are inert (nothing to resolve — the footer names them as
        unavailable), and `f` exists only there (and only until the
        merge actually lands)."""
        if action == "close":
            return True  # esc stays live: a worker's write-backs are guarded
        if self._busy is not None:
            return False
        if action == "ff_merge":
            return self._ff_ready and not self._ff_done
        return not (self._ff_ready and action in self._FF_DISABLED_ACTIONS)

    def _update_footer(self) -> None:
        """Rewrite the key hint line for the current mode — the footer
        is the screen's only key documentation, so the ff-ready state
        must swap it explicitly (a late worker may land after unmount:
        a no-op there)."""
        if not self.is_mounted:
            return
        footer = self.query_one("#merge-footer", Static)
        if self._ff_ready:
            base = "base"
            if self.wt is not None and self.wt.report is not None:
                base = self.wt.report.base_branch or "base"
            if self._ff_done:
                text = f"merged into {base} · r refresh · esc close"
            else:
                text = (
                    f"f merge into {base} (ff-only) · r refresh · esc close · "
                    "o/t/b/e/v/d/s/a disabled — nothing to resolve"
                )
        else:
            text = (
                "o/t/b hunk ours/theirs/both · e edit · v check · "
                "d details · E report · s commit · a abort · r refresh · esc close"
            )
        footer.update(text)

    # -- preparation (the engine's reversed merge) -------------------------------

    def on_mount(self) -> None:
        self._prepare()

    @work(exclusive=True, group="merge-prepare")
    async def _prepare(self) -> None:
        """Deploy (or re-detect) the parked conflict through the engine
        (`merge_resolver.prepare`): an already-parked merge is re-triaged
        in place, a fresh one is attempted with the safe auto-resolvers
        on. Blocking git — off the UI thread. The steps surface live in
        the progress panel via `prepare`'s progress callback (applied on
        the event loop, so the transitions land strictly before this
        coroutine resumes); the panel disappears once the outcome is in
        and the file list has loaded."""
        app = cast("DrCodeApp", self.app)
        env = app.task_env
        # The merge description first — one cheap git call BEFORE the
        # heavy prepare — so the title (and the progress panel's first
        # line) already say WHAT is being resolved from the first frame.
        self._merge_desc = await asyncio.to_thread(self._fetch_merge_desc)
        self._begin_prepare()
        if self._merge_desc:
            self._set_title(f"#{self.task_id} — merge conflicts")
        # The persisted rolling averages (previous runs' per-step
        # durations) feed the ETA line from the first tick on — blocking
        # file IO, off the UI thread.
        self._prepare_avg_timings = await asyncio.to_thread(
            merge_resolver.load_prepare_timings, app.project_root, self.task_id
        )
        self._render_prepare_panel()
        # The task's isolation PIN overrides the live setting here too:
        # a task pinned to its worktree keeps its parked merge reachable
        # even after `task_isolation` was switched off mid-task.
        task = app.manager.tasks.get(self.task_id)
        if not env.active and not (task is not None and task.isolation_pinned is True):
            self._set_title("Merge conflicts — worktree isolation is off")
            self._hide_prepare()
            return
        pinned_before = task is not None and task.isolation_pinned is not None
        root, note = await asyncio.to_thread(env.ensure_root, self.task_id, task)
        if task is not None:
            app._isolation_pin_note(task, save=not pinned_before)
        if note:
            self._set_title(f"Merge conflicts — unavailable: {note}")
            self._hide_prepare()
            return
        base = await asyncio.to_thread(env.base_branch)

        def progress(stage: str) -> None:
            # Worker thread → event loop. call_from_thread is
            # synchronous: by the time the callback returns, the panel
            # transition has been applied — the finalize below can never
            # overtake a step boundary.
            try:
                self.app.call_from_thread(self._on_prepare_stage, stage)
            except Exception:  # noqa: BLE001, S110 - best-effort UI update
                pass

        report: MergeReport = await asyncio.to_thread(
            merge_resolver.prepare,
            app.project_root,
            self.task_id,
            root,
            base,
            progress=progress,
        )
        self._finish_prepare_stages(ok=report.ok)
        if self._prepare_measured:
            # The measured step durations merge into the persisted
            # rolling averages feeding the next run's ETA — blocking
            # file IO, off the UI thread; best-effort by contract.
            await asyncio.to_thread(
                merge_resolver.save_prepare_timings,
                app.project_root,
                self.task_id,
                self._prepare_measured,
            )
        self.wt = ConflictWorktree(app.project_root, self.task_id, root, report=report)
        if not report.ok:
            self._ff_ready = False
            self._set_title(f"Merge conflicts — {report.message}")
            self._update_footer()
            self._hide_prepare()
            return
        if not report.conflicted:
            # ff-ready ONLY when the engine vouches the fast-forward is
            # actually possible (the base tip is an ancestor of the
            # branch tip): ok+not-conflicted alone also covers a branch
            # already contained in the base (a --ff-only merge would
            # refuse) and a resumed merge staged but never concluded (no
            # merge commit exists yet) — neither may offer f.
            if report.fast_forward is True:
                self._ff_ready = True
                self._set_title(
                    f"{report.message} — nothing to resolve; "
                    f"press f to merge {report.branch} into "
                    f"{report.base_branch} (ff-only)"
                )
                self._update_footer()
                self._hide_prepare()
                return
            self._ff_ready = False
            if report.fast_forward is False:
                self._set_title(
                    f"{report.message} — {report.branch} holds no commits "
                    f"beyond {report.base_branch}; there is nothing to "
                    "merge into the base"
                )
            else:
                # fast_forward is None: the merge is resolved and staged
                # in the worktree but not committed — s (not f) is the
                # next step, and it stays live below.
                self._set_title(
                    f"#{self.task_id} — the merge is resolved but NOT "
                    f"committed on {report.branch}; press s to commit it"
                )
            self._update_footer()
            self._hide_prepare()
            return
        self._ff_ready = False
        self._set_title(
            f"#{self.task_id} — merge {report.base_branch} into "
            f"{report.branch}: {len(report.remaining)} file(s) to resolve"
        )
        await self._reload_files()
        self._hide_prepare()

    def _titled(self, text: str) -> str:
        """`text` with the merge description (`_merge_desc`) appended —
        the single place every title passes through, so the description
        survives all the rewrites of `_prepare`'s final states (isolation
        off, note, !report.ok, ff-ready, not-committed, conflicted) and
        of the later commit/ff outcomes."""
        if self._merge_desc:
            return f"{text} — «{self._merge_desc}»"
        return text

    def _set_title(self, text: str) -> None:
        # A worker may reach this after the user already closed the
        # screen (ESC mid-prepare) — an unmounted screen is a no-op.
        if not self.is_mounted:
            return
        self.query_one("#merge-title", Static).update(self._titled(text))

    def _fetch_merge_desc(self) -> str | None:
        """What the merge carries, by descending preference: the subject
        of the task branch's last commit (`git log -1 --format=%s
        drcode/<id>` — best describes what actually lands in the base),
        then the task's title, then None (the titles stay as they were).
        One cheap git call — designed to run BEFORE the heavy prepare,
        in a worker thread."""
        app = cast("DrCodeApp", self.app)
        branch = f"drcode/{self.task_id}"
        try:
            proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(app.project_root),
                    "log",
                    "-1",
                    "--format=%s",
                    branch,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                **child_priority_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            proc = None
        if proc is not None and proc.returncode == 0:
            subject = proc.stdout.strip()
            if subject:
                return subject.splitlines()[0]
        task = app.manager.tasks.get(self.task_id)
        if task is not None and task.title:
            return task.title
        return None

    # -- prepare progress panel (the same #merge-checks zone) --------------------

    # The prepare phase's observable steps, in engine order. `merge` is
    # by far the longest (the actual reversed git merge), the rest are
    # fast — the defaults below are deliberately conservative so the
    # first-run ETA overestimates rather than lies.
    _PREPARE_STAGES: ClassVar[tuple[tuple[str, str], ...]] = (
        ("worktree", "worktree"),
        ("merge", "merge"),
        ("triage", "triage"),
        ("auto-resolve", "auto-resolve"),
    )
    _DEFAULT_PREPARE_ETAS: ClassVar[dict[str, float]] = {
        "worktree": 3.0,
        "merge": 30.0,
        "triage": 5.0,
        "auto-resolve": 10.0,
    }

    def _begin_prepare(self) -> None:
        """Reset the prepare steps, mark the first one running and start
        the panel's animation timer (independent of the worker — it must
        tick while the blocking git runs in its thread)."""
        self._stop_prepare_animation()
        self._prepare_stages = [
            _PrepareStage(key, label) for key, label in self._PREPARE_STAGES
        ]
        first = self._prepare_stages[0]
        first.status = "running"
        first.started = time.monotonic()
        self._prepare_avg_timings = {}
        self._prepare_measured = {}
        self._prepare_failed = False
        self._spinner_frame = 0
        if self.is_mounted:
            panel = self.query_one("#merge-checks", Static)
            panel.add_class("visible")
            self._render_prepare_panel()
        self._prepare_timer = self.set_interval(
            _SPINNER_INTERVAL, self._tick_prepare_animation
        )

    def _on_prepare_stage(self, key: str) -> None:
        """One progress-boundary transition from the worker thread,
        applied HERE on the event loop: the running step (if any) is
        closed with its measured duration, the named one starts. The
        first boundary (``worktree``) may name an already-running step —
        idempotent."""
        if not self.is_mounted:
            return
        now = time.monotonic()
        for stage in self._prepare_stages:
            if stage.status == "running":
                stage.status = "done"
                stage.duration = now - (stage.started or now)
        target = next((s for s in self._prepare_stages if s.key == key), None)
        if target is not None and target.status != "running":
            target.status = "running"
            target.started = now
        self._render_prepare_panel()

    def _finish_prepare_stages(self, ok: bool) -> None:
        """Close the running step (the callbacks have all been applied —
        `call_from_thread` is synchronous), drop the steps the run never
        reached (a clean merge returns before triage), and freeze the
        final [ok]/[FAILED] render before the panel is hidden."""
        self._prepare_failed = not ok
        now = time.monotonic()
        for stage in self._prepare_stages:
            if stage.status == "running":
                stage.status = "done"
                stage.duration = now - (stage.started or now)
        self._prepare_stages = [s for s in self._prepare_stages if s.status == "done"]
        self._prepare_measured = {
            s.key: s.duration for s in self._prepare_stages if s.duration
        }
        self._stop_prepare_animation()
        self._render_prepare_panel()

    def _hide_prepare(self) -> None:
        """Prepare is over and the outcome is on screen — the panel goes
        away (unless a busy op or a funnel run owns it right now: the
        hide runs at the tail of the prepare worker, and a fast `v`
        press can already have claimed the zone)."""
        self._stop_prepare_animation()
        if not self.is_mounted or self._busy is not None or self._stages:
            return
        try:
            self.query_one("#merge-checks", Static).remove_class("visible")
        except NoMatches:
            # The screen is being torn down (is_mounted is still truthy
            # mid-dismiss) — a late hide is a no-op.
            pass

    def _stop_prepare_animation(self) -> None:
        if self._prepare_timer is not None:
            self._prepare_timer.stop()
            self._prepare_timer = None

    def _tick_prepare_animation(self) -> None:
        self._spinner_frame = (self._spinner_frame + 1) % len(_SPINNER_FRAMES)
        if any(stage.status == "running" for stage in self._prepare_stages):
            self._render_prepare_panel()

    def _render_prepare_panel(self) -> None:
        """The live prepare view — identical mechanics to the funnel
        panel: `[⠹] merge…  (12s)` for the running step, `[ok]` /
        `[FAILED]` for the finished ones, plus the percent/ETA footer."""
        if not self.is_mounted:
            return
        panel = self.query_one("#merge-checks", Static)
        lines = ["Preparing merge:"]
        if self._merge_desc:
            lines.append(f"#{self.task_id}: «{self._merge_desc}»")
        now = time.monotonic()
        for stage in self._prepare_stages:
            if stage.status == "done":
                mark = "FAILED" if self._prepare_failed else "ok"
                lines.append(
                    f"[{mark}] {stage.label}  ({_fmt_seconds(stage.duration)})"
                )
            elif stage.status == "running":
                frame = _SPINNER_FRAMES[self._spinner_frame]
                elapsed = (now - stage.started) if stage.started is not None else 0.0
                lines.append(f"[{frame}] {stage.label}…  ({_fmt_seconds(elapsed)})")
            else:
                lines.append(f"[ ] {stage.label}")
        done_count = sum(1 for stage in self._prepare_stages if stage.status == "done")
        lines.append(
            f"Preparing: {done_count}/{len(self._prepare_stages)} · "
            f"{_stages_percent(self._prepare_stages, self._prepare_avg_timings, self._DEFAULT_PREPARE_ETAS)}% · "
            f"ETA ~{_fmt_seconds(_stages_eta(self._prepare_stages, self._prepare_avg_timings, self._DEFAULT_PREPARE_ETAS))}"
        )
        panel.update("\n".join(lines))

    # -- busy state of the short blocking ops (r / s / a / f) --------------------

    def _show_busy(self, label: str) -> None:
        """A short blocking op (refresh/commit/abort/ff-merge) starts:
        one spinner line with elapsed seconds in the same panel, and
        `check_action` locks every key but esc until `_end_busy`. When a
        funnel run is mid-flight its panel is left alone (the busy gate
        still applies) — a running panel must not be overwritten."""
        self._busy_generation += 1
        self._busy = label
        self._busy_started = time.monotonic()
        if not self.is_mounted or any(
            stage.status == "running" for stage in self._stages
        ):
            self._busy_visible = False
            return
        self._busy_visible = True
        self._spinner_frame = 0
        panel = self.query_one("#merge-checks", Static)
        if panel.has_class("visible"):
            # A finished funnel's summary (or anything else the panel
            # held) comes back when the op ends.
            self._panel_saved = str(panel.content)
        else:
            self._panel_saved = None
            panel.add_class("visible")
        self._render_busy_panel()
        self._busy_timer = self.set_interval(
            _SPINNER_INTERVAL, self._tick_busy_animation
        )

    def _end_busy(self) -> None:
        """The op is over: restore the panel content the busy line had
        replaced, or hide the panel again. Safe for a late worker after
        the screen was closed (unmount stops the timer separately)."""
        self._busy = None
        self._stop_busy_animation()
        if not self.is_mounted or not self._busy_visible:
            self._busy_visible = False
            return
        self._busy_visible = False
        try:
            panel = self.query_one("#merge-checks", Static)
        except NoMatches:
            return
        if self._panel_saved is not None:
            panel.update(self._panel_saved)
            self._panel_saved = None
        else:
            panel.remove_class("visible")

    def _stop_busy_animation(self) -> None:
        if self._busy_timer is not None:
            self._busy_timer.stop()
            self._busy_timer = None

    def _tick_busy_animation(self) -> None:
        self._spinner_frame = (self._spinner_frame + 1) % len(_SPINNER_FRAMES)
        if self._busy is not None:
            self._render_busy_panel()

    def _render_busy_panel(self) -> None:
        if not self.is_mounted or self._busy is None:
            return
        elapsed = (
            (time.monotonic() - self._busy_started)
            if self._busy_started is not None
            else 0.0
        )
        frame = _SPINNER_FRAMES[self._spinner_frame]
        self.query_one("#merge-checks", Static).update(
            f"[{frame}] {self._busy}…  ({_fmt_seconds(elapsed)})"
        )

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
        try:
            view = self.view
        except NoMatches:
            # The screen is being torn down (is_mounted is still truthy
            # mid-dismiss) — a late refresh is a no-op.
            return
        if path is None:
            view.set_file("", [])
            return
        content = await asyncio.to_thread(self.wt.read_file, path)
        if content is None:
            view.set_file("(file absent in the worktree)", [])
            return
        try:
            view = self.view
        except NoMatches:
            return
        view.set_file(content, parse_hunks(content))

    # -- hunk decisions -----------------------------------------------------------

    def action_resolve(self, choice: str) -> None:
        """o / t / b: apply the decision to the cursor hunk (staged as
        soon as the file is marker-free)."""
        if self._busy is not None:
            return
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
        if self._busy is not None:
            return
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

    # The funnel's four serial stages — the order matters (tree soundness
    # before the expensive suite). `fn_name` is resolved on the module at
    # call time, so monkeypatching per-stage functions in tests works.
    _CHECK_STAGES: ClassVar[tuple[tuple[str, str, str], ...]] = (
        ("compile", "compile_check", "compile"),
        ("ruff", "ruff_check", "ruff check"),
        ("pyright", "pyright_check", "pyright"),
        ("pytest", "run_test_suite", "test suite"),
    )
    # ETA defaults (seconds) for a stage with no persisted history yet —
    # replaced by the rolling averages after the first real runs.
    _DEFAULT_STAGE_ETAS: ClassVar[dict[str, float]] = {
        "compile": 5.0,
        "ruff": 10.0,
        "pyright": 90.0,
        "pytest": 180.0,
    }

    def action_verify(self) -> None:
        if self._busy is not None:
            return
        if self.wt is None or not self.wt.in_progress():
            self.notify("No merge is in progress", title="Merge", timeout=4)
            return
        self._begin_checks()
        self._run_checks()

    def _begin_checks(self) -> None:
        """Reset the per-stage progress state and start the panel's
        animation timer. The timer is deliberately independent of the
        worker: the checks run in worker threads, and a braille spinner
        plus elapsed/ETA line must keep moving while the event loop idles
        between the `asyncio.to_thread` awaits."""
        self._checks_generation += 1
        self._stop_checks_animation()
        self._stages = [
            _CheckStage(key, fn_name, label)
            for key, fn_name, label in self._CHECK_STAGES
        ]
        self._spinner_frame = 0
        self._avg_timings = {}
        panel = self.query_one("#merge-checks", Static)
        panel.add_class("visible")
        self._render_checks_panel()
        self._checks_timer = self.set_interval(
            _SPINNER_INTERVAL, self._tick_checks_animation
        )

    def _stop_checks_animation(self) -> None:
        if self._checks_timer is not None:
            self._checks_timer.stop()
            self._checks_timer = None

    def on_unmount(self) -> None:
        self._stop_checks_animation()
        self._stop_prepare_animation()
        self._stop_busy_animation()

    def _tick_checks_animation(self) -> None:
        """One spinner frame; redraw only while a stage is actually
        running (a finished panel is static and needs no redraws)."""
        self._spinner_frame = (self._spinner_frame + 1) % len(_SPINNER_FRAMES)
        if any(stage.status == "running" for stage in self._stages):
            self._render_checks_panel()

    def _checks_eta(self) -> float:
        """ETA over the not-yet-done funnel stages (see `_stages_eta`)."""
        return _stages_eta(self._stages, self._avg_timings, self._DEFAULT_STAGE_ETAS)

    def _checks_percent(self) -> int:
        """Completed / total checks, the running stage credited
        proportionally to its historical duration (see
        `_stages_percent`)."""
        return _stages_percent(
            self._stages, self._avg_timings, self._DEFAULT_STAGE_ETAS
        )

    def _render_checks_panel(self) -> None:
        """The live progress view: one status line per stage
        ("[ok] compile (0.4s)", "[⠹] pyright… (12s)", "[ ] test suite")
        plus the percent/ETA footer. Plain text (the panel is
        markup=False) — redrawn between stages and from the animation
        timer, never from the worker threads."""
        if not self.is_mounted:
            return
        panel = self.query_one("#merge-checks", Static)
        lines = ["Verification funnel:"]
        now = time.monotonic()
        for stage in self._stages:
            if stage.status == "done":
                mark = (
                    "skip"
                    if (stage.result and stage.result.skipped)
                    else ("ok" if (stage.result and stage.result.ok) else "FAILED")
                )
                lines.append(
                    f"[{mark}] {stage.label}  ({_fmt_seconds(stage.duration)})"
                )
            elif stage.status == "running":
                frame = _SPINNER_FRAMES[self._spinner_frame]
                elapsed = (now - stage.started) if stage.started is not None else 0.0
                lines.append(f"[{frame}] {stage.label}…  ({_fmt_seconds(elapsed)})")
            else:
                lines.append(f"[ ] {stage.label}")
        done_count = sum(1 for stage in self._stages if stage.status == "done")
        lines.append(
            f"Checks: {done_count}/{len(self._stages)} · "
            f"{self._checks_percent()}% · ETA ~{_fmt_seconds(self._checks_eta())}"
        )
        panel.update("\n".join(lines))

    @work(exclusive=True, group="merge-checks")
    async def _run_checks(self) -> None:
        if self.wt is None:
            return
        generation = self._checks_generation
        worktree = self.wt.worktree
        app = cast("DrCodeApp", self.app)
        try:
            # The persisted rolling averages (previous runs' per-check
            # durations) feed the ETA line from the first tick on —
            # blocking file IO, off the UI thread.
            self._avg_timings = await asyncio.to_thread(
                merge_resolver.load_check_timings, app.project_root, self.task_id
            )
            self._render_checks_panel()
            for stage in self._stages:
                stage.status = "running"
                stage.started = time.monotonic()
                self._render_checks_panel()
                fn = getattr(merge_resolver, stage.fn_name)
                result = await asyncio.to_thread(fn, worktree)
                stage.duration = time.monotonic() - stage.started
                stage.result = result
                stage.status = "done"
                self._render_checks_panel()
            self.check_results = [
                s.result for s in self._stages if s.result is not None
            ]
            self._checks_ok = all(c.ok for c in self.check_results)
            # The full funnel report is persisted next to the engine's
            # triage report, and this run's measured durations merge into
            # the persisted rolling averages (skipped stages excluded —
            # a 2ms skip would poison the average). Blocking file IO, off
            # the UI thread. Both best-effort: a write failure just drops
            # the path line / the ETA history, never the run itself.
            report_path = await asyncio.to_thread(
                merge_resolver.save_verify_report,
                app.project_root,
                self.task_id,
                [s.result for s in self._stages[:-1] if s.result is not None],
                self._stages[-1].result or CheckResult(name="test suite", ok=True),
            )
            measured = {
                stage.key: stage.duration
                for stage in self._stages
                if stage.duration is not None
                and stage.result is not None
                and not stage.result.skipped
            }
            if measured:
                self._avg_timings = await asyncio.to_thread(
                    merge_resolver.save_check_timings,
                    app.project_root,
                    self.task_id,
                    measured,
                )
            if self.is_mounted:
                panel = self.query_one("#merge-checks", Static)
                lines = [c.line for c in self.check_results]
                if report_path is not None:
                    lines.append(f"Full report: {report_path}")
                panel.update("Checks:\n" + "\n".join(f"  {line}" for line in lines))
            if self._checks_ok:
                message = "All checks passed"
            else:
                failed = merge_resolver.failed_names(self.check_results)
                message = "Checks FAILED: " + ", ".join(failed)
                if report_path is not None:
                    message += f" — report: {report_path}"
            self.notify(
                message,
                title="Merge",
                severity="information" if self._checks_ok else "error",
                timeout=5,
            )
        finally:
            # Stop OUR generation's timer only — a cancelled run (an
            # exclusive restart by a fresh v press) must not kill the
            # successor's animation.
            if generation == self._checks_generation:
                self._stop_checks_animation()

    def action_check_details(self) -> None:
        """d: open the full output of a check in the details viewer —
        defaulting to the first FAILED check when any failed."""
        if not self.check_results:
            self.notify("No checks yet — press v to run them", title="Merge", timeout=4)
            return
        start = 0
        for index, check in enumerate(self.check_results):
            if not check.ok and not check.skipped:
                start = index
                break
        self.app.push_screen(CheckDetailsScreen(self.check_results, start))

    def action_open_report(self) -> None:
        """E: open the persisted verification report in the user's
        editor (same suspend-and-subprocess pattern as `e`)."""
        app = cast("DrCodeApp", self.app)
        path = merge_resolver.verify_report_path(app.project_root, self.task_id)
        if not path.exists():
            self.notify(
                "No report yet — press v to run the checks", title="Merge", timeout=4
            )
            return
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor:
            self.notify(
                "No editor configured ($VISUAL/$EDITOR)",
                title="Merge",
                severity="error",
                timeout=5,
            )
            return
        try:
            with self.app.suspend():
                subprocess.run([*editor.split(), str(path)], check=False)
        except Exception as e:  # noqa: BLE001 - any launch failure is a toast
            self.notify(
                f"Editor failed: {type(e).__name__}: {e}",
                title="Merge",
                severity="error",
                timeout=6,
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
        if self._busy is not None:
            return
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
        self._show_busy("committing")
        try:
            await self._commit_resolution()
        finally:
            self._end_busy()

    async def _commit_resolution(self) -> None:
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
            # A successful commit is the screen's logical end of work —
            # close it automatically (the toast outlives the screen; the
            # abort path below already works the same way).
            self.dismiss()
        else:
            self.notify(
                f"Commit refused: {out}", title="Merge", severity="error", timeout=8
            )

    def action_abort_merge(self) -> None:
        if self._busy is not None:
            return
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
        self._show_busy("aborting")
        try:
            ok, out = await asyncio.to_thread(self.wt.abort)
        finally:
            self._end_busy()
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

    # -- ff-merge (ff-ready mode: nothing left to resolve) -----------------------

    def action_ff_merge(self) -> None:
        """f: fast-forward the MAIN checkout's base branch to the task's
        branch — available only in the ff-ready state (report ok, not
        conflicted) and only once (check_action gates both)."""
        if self._busy is not None:
            return
        if self.wt is None or not self._ff_ready or self._ff_done:
            return
        self._do_ff_merge()

    @work(exclusive=True, group="merge-ff")
    async def _do_ff_merge(self) -> None:
        if self.wt is None:
            return
        self._show_busy("fast-forwarding")
        try:
            await self._run_ff_merge()
        finally:
            self._end_busy()

    async def _run_ff_merge(self) -> None:
        app = cast("DrCodeApp", self.app)
        env = app.task_env
        report = self.wt.report if self.wt is not None else None
        if report is None:
            return
        branch = report.branch or f"drcode/{self.task_id}"
        base = report.base_branch or "base"
        # The base may have moved between the screen's prepare and this
        # press — re-verify the ff-possibility at press time so the
        # refusal names the reason instead of raw git output.
        if not await asyncio.to_thread(is_fast_forward, app.project_root, branch, base):
            self.notify(
                f"git merge --ff-only {branch} into {base} refused: "
                f"{base} is no longer an ancestor of {branch} (the base "
                "moved ahead) — reopen the screen to re-run the merge there",
                title="Merge",
                severity="error",
                timeout=8,
            )
            return
        ok, out = await asyncio.to_thread(env.ff_merge_branch, self.task_id)
        if not ok:
            # --ff-only refuses instead of conflicting: the base moved
            # ahead (or the tree is dirty). Nothing else to try here —
            # the user re-runs the reversed merge by reopening the
            # screen (prepare re-detects and re-triages).
            self.notify(
                f"git merge --ff-only {branch} into {base} refused: {out} "
                "— reopen the screen to re-run the merge there",
                title="Merge",
                severity="error",
                timeout=8,
            )
            return
        self._ff_done = True
        self._set_title(f"#{self.task_id} — {branch} merged into {base} (ff-only)")
        self._update_footer()
        task = app.manager.get(self.task_id)
        if task.needs_human:
            task.needs_human = False
            task.touch()
            app.manager.storage.save(task)
        app._show_event(
            task,
            f"merged {branch} into {base} (ff-only) via the conflict screen",
            kind="workflow",
        )
        self.notify(
            f"{branch} merged into {base}",
            title="Merge",
            timeout=5,
        )
        # Refresh the mergedness snapshot (_merge_state) and the header's
        # ⎇ counter in one pass — the poll's fingerprint cache sees the
        # moved base and recomputes.
        await app._poll_git_indicator()
        # A successful merge is the screen's logical end of work — close
        # it automatically (the toast above outlives the screen).
        self.dismiss()

    # -- refresh / close -----------------------------------------------------------

    def action_refresh_state(self) -> None:
        """r: re-triage the live merge and reload everything (an external
        editor may have changed files while the screen was suspended)."""
        if self._busy is not None:
            return
        if self.wt is None:
            return
        self._refresh()

    @work(exclusive=True, group="merge-refresh")
    async def _refresh(self) -> None:
        if self.wt is None or not self.is_mounted:
            return
        self._show_busy("refreshing")
        try:
            await asyncio.to_thread(self.wt.refresh)
            await self._reload_files()
        finally:
            self._end_busy()

    def action_close(self) -> None:
        self.dismiss()
