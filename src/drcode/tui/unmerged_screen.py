"""The unmerged-branches list screen (`drcode.tui.unmerged_screen`).

Ctrl+M with MORE than one unmerged task: a small modal list — one row
per task whose `drcode/<id>` branch still holds commits the base branch
does not have (or whose merge is parked as a conflict) — where the user
picks a CONCRETE task and acts on it without opening the per-task
conflict screen first. Exactly one candidate never reaches this screen
(`action_open_merge` opens the ConflictScreen directly, as before).

Each row shows the task (kind letter F/B/R, id, title), its
`drcode/<id>` branch (deepskyblue, the header's branch color) and the
merge state (computed fresh on every refresh — never from a cached
snapshot, the user acts the moment the list paints):

- ``conflict`` — the engine parked a conflicted merge in the task's
  worktree (`needs_human` or a live in-progress merge): resolve it by
  hand (``enter`` opens the ConflictScreen) or hand it to the automatic
  resolver (``a``);
- ``ff-ready`` — nothing is left to resolve (a clean reversed merge,
  the safe auto-resolvers concluded, or a plain unmerged branch): ``m``
  fast-forwards the MAIN checkout's base branch to `drcode/<id>` via
  `task_env.ff_merge_branch` — `--ff-only`, a refusal is an error toast,
  never a merge in the user's tree;
- ``staged`` — the merge is resolved and staged in the worktree but not
  concluded: ``m`` commits it on the task branch (the fast-forward it
  enables becomes the next ``m``);
- ``unmerged`` — the branch holds commits the base lacks and nothing is
  parked: ``m`` prepares it (the engine's reversed merge) and then takes
  whatever the report vouches for (a clean merge is concluded and merged,
  a conflict is announced — ``enter``/``a`` take over).

All three action groups stay available from BOTH this list and the
ConflictScreen (its ``f``/``s`` keys are the same mechanics inlined per
task; the list's ``a`` and the screen's ``A`` both go through the app's
`launch_resolver` guard — a refusal is a toast, and a started round
closes the screen: the progress lives in the task's chat, the funnel and
the service events already report there).

The contract holds unchanged: the only merge into the user's checkout is
the explicit, `--ff-only` ``m`` behind the base-is-an-ancestor
re-verification (same double check as the ConflictScreen's ``f``) —
everything else lands on the task's own branch.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

from rich.markup import escape
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Label, ListItem, ListView, Static

from .. import merge_engine, merge_resolver
from ..merge_engine import is_fast_forward
from ..models import TaskKind
from ..task_env import WorktreeUnavailable
from .merge.stages import _fmt_seconds

if TYPE_CHECKING:
    from .app import DrCodeApp

# The busy spinner of the short blocking ops (r / m): braille frames and
# the footer redraw interval — the same look as the ConflictScreen's
# progress panel (which owns the canonical copies).
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_INTERVAL = 0.15

# The footer's idle content (the hotkey line) — restored when a busy op
# ends.
_FOOTER_HINT = (
    "enter manual resolve · a auto resolve · m merge/commit · r refresh · esc close"
)


@dataclass
class _Entry:
    """One list row: the task, its branch and the merge state the actions
    dispatch on (`kind`: conflict / ff / staged / unmerged)."""

    task_id: str
    title: str
    branch: str
    kind: str
    base: str
    task_kind: TaskKind | None = None

    @property
    def state_label(self) -> str:
        if self.kind == "conflict":
            return "conflict — needs resolution"
        if self.kind == "ff":
            return f"ready — merge into {self.base} (ff-only)"
        if self.kind == "staged":
            return "resolved, not committed — m commits it"
        return f"unmerged — merge into {self.base} not done"


class UnmergedListScreen(ModalScreen[None]):
    """The >1-candidates face of Ctrl+M: pick a task, act on it."""

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Close", priority=True),
        Binding("enter", "open_screen", "Manual resolve", priority=True),
        Binding("a", "auto_resolve", "Auto resolve", priority=True),
        Binding("m", "land", "Merge / commit", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
    ]

    DEFAULT_CSS = """
    UnmergedListScreen {
        align: center middle;
    }
    #unmerged-dialog {
        width: 80%;
        max-width: 110;
        height: auto;
        max-height: 80%;
        padding: 0 1 1 1;
        background: $panel;
        border: thick $accent;
    }
    #unmerged-title {
        height: 1;
        padding: 0 1;
        color: $text;
        text-style: bold;
    }
    #unmerged-list {
        height: auto;
        max-height: 100%;
        background: $surface;
    }
    #unmerged-list ListItem { padding: 0 1; }
    #unmerged-footer {
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, task_ids: list[str]) -> None:
        super().__init__()
        self.task_ids = list(dict.fromkeys(task_ids))
        self._entries: list[_Entry] = []
        # A short blocking op (m's prepare/commit/ff chain) is running:
        # every key but esc is inert (check_action) — git must not race a
        # second press.
        self._busy = False
        # The busy spinner's state (the animated footer line): the label
        # names the current phase of the op, the clock runs from the op's
        # START (phase changes keep it — the user watches the total).
        self._busy_label: str | None = None
        self._busy_started: float | None = None
        self._busy_timer: Timer | None = None
        self._spinner_frame = 0

    # -- composition ----------------------------------------------------------

    @property
    def list_view(self) -> ListView:
        return self.query_one("#unmerged-list", ListView)

    def compose(self) -> ComposeResult:
        with Vertical(id="unmerged-dialog"):
            # markup=False: titles are commit subjects that may carry
            # literal [brackets] — they must never trip Rich markup.
            yield Static("Unmerged task branches", id="unmerged-title", markup=False)
            yield ListView(id="unmerged-list")
            yield Static(_FOOTER_HINT, id="unmerged-footer", markup=False)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "close":
            return True  # esc stays live: a worker's write-backs are guarded
        if self._busy:
            return False
        if action in ("open_screen", "auto_resolve", "land"):
            return self._selected() is not None
        return True

    # -- busy state of the short blocking ops (r / m) -----------------------------

    def _show_busy(self, label: str) -> None:
        """A short blocking op (refresh / the m prepare→commit→ff chain)
        starts or enters its next phase: one spinner line with the
        elapsed seconds replaces the footer's hotkey line, and
        `check_action` locks every key but esc until `_end_busy`. The
        clock keeps running across phases — it measures the whole
        operation, the label alone names the current phase."""
        self._busy = True
        self._busy_label = label
        if self._busy_started is None:
            self._busy_started = time.monotonic()
        if not self.is_mounted:
            return
        self._render_busy_footer()
        if self._busy_timer is None:
            self._busy_timer = self.set_interval(
                _SPINNER_INTERVAL, self._tick_busy_animation
            )

    def _end_busy(self) -> None:
        """The op is over: restore the footer's hotkey line. Safe for a
        late worker after the screen was dismissed/unmounted (unmount
        stops the timer separately, the query is guarded)."""
        self._busy = False
        self._busy_label = None
        self._busy_started = None
        if self._busy_timer is not None:
            self._busy_timer.stop()
            self._busy_timer = None
        if not self.is_mounted:
            return
        try:
            footer = self.query_one("#unmerged-footer", Static)
        except NoMatches:
            return
        footer.update(_FOOTER_HINT)

    def _tick_busy_animation(self) -> None:
        self._spinner_frame = (self._spinner_frame + 1) % len(_SPINNER_FRAMES)
        self._render_busy_footer()

    def _render_busy_footer(self) -> None:
        if not self.is_mounted or self._busy_label is None:
            return
        try:
            footer = self.query_one("#unmerged-footer", Static)
        except NoMatches:
            return
        elapsed = (
            (time.monotonic() - self._busy_started)
            if self._busy_started is not None
            else 0.0
        )
        frame = _SPINNER_FRAMES[self._spinner_frame]
        footer.update(f"[{frame}] {self._busy_label}…  ({_fmt_seconds(elapsed)})")

    # -- state ------------------------------------------------------------------

    def on_screen_resume(self) -> None:
        """Recompute the rows every time the screen becomes active — on
        push, and after a ConflictScreen opened from here was closed (the
        user may have resolved or committed something meanwhile)."""
        self._refresh_entries()

    def _selected(self) -> _Entry | None:
        if not self._entries or not self.is_mounted:
            return None
        try:
            index = self.list_view.index
        except NoMatches:
            # Shutdown race: is_mounted can still be true while the
            # screen's widgets are already torn down (app exit with a
            # refresh/land worker in flight) — nothing is selected then.
            return None
        if index is None or not 0 <= index < len(self._entries):
            return None
        return self._entries[index]

    @staticmethod
    def _collect_entry(app: DrCodeApp, task_id: str) -> _Entry | None:
        """One task's merge state, fresh from git (sync, blocking — the
        callers run it in a worker thread). None = nothing to merge
        anymore (the task vanished, or the branch holds nothing beyond
        the base) — the row is dropped."""
        env = app.task_env
        task = app.manager.tasks.get(task_id)
        if task is None or not env.active:
            return None
        branch = env.branch(task_id)
        base = env.base_branch()
        existing, unmerged = env.drcode_branch_state()
        if branch not in existing:
            return None  # no branch — nothing to merge
        if branch not in unmerged:
            return None  # already merged into the base — the row is done
        wt = env.worktree_path(task_id)
        report = merge_engine.load_report(app.project_root, task_id)
        if task.needs_human or merge_engine.in_progress(wt):
            return _Entry(
                task_id=task_id,
                title=task.title_display,
                branch=branch,
                kind="conflict",
                base=base,
                task_kind=task.kind,
            )
        if report is not None and report.ok and not report.conflicted:
            if report.fast_forward is True:
                kind = "ff"
            elif report.fast_forward is None:
                kind = "staged"
            else:
                return None  # up to date — nothing beyond the base
        elif is_fast_forward(app.project_root, branch, base):
            kind = "ff"
        else:
            kind = "unmerged"
        return _Entry(
            task_id=task_id,
            title=task.title_display,
            branch=branch,
            kind=kind,
            base=base,
            task_kind=task.kind,
        )

    def _collect_entries(self, task_ids: list[str]) -> list[_Entry]:
        app = cast("DrCodeApp", self.app)
        entries: list[_Entry] = []
        for task_id in task_ids:
            entry = self._collect_entry(app, task_id)
            if entry is not None:
                entries.append(entry)
        return entries

    # -- refresh / rendering ------------------------------------------------------

    def action_refresh(self) -> None:
        if self._busy:
            return
        self._refresh_entries(show_busy=True)

    @work(exclusive=True, group="unmerged-refresh")
    async def _refresh_entries(self, show_busy: bool = False) -> None:
        # The explicit r press gets the busy spinner (the git sweep can
        # take seconds); the automatic refreshes (on resume, after a
        # land) stay silent — the user did not press anything.
        if show_busy:
            self._show_busy("refreshing")
        try:
            await self._run_refresh()
        finally:
            if show_busy:
                self._end_busy()

    async def _run_refresh(self) -> None:
        sel = self._selected()
        selected = sel.task_id if sel is not None else None
        entries = await asyncio.to_thread(self._collect_entries, self.task_ids)
        self._entries = entries
        if not self.is_mounted:
            return
        if not entries:
            self.notify(
                "Nothing unmerged — every branch is merged into the base",
                title="Merge",
                timeout=5,
            )
            self.dismiss()
            return
        try:
            self._render_rows(selected)
        except NoMatches:
            # Unmounted between the is_mounted check and the render —
            # nothing left to paint.
            pass

    # The kind letter of a row (F/B/R), the same colors the sidebar uses
    # for the task kinds. Always shown: this screen itself only opens
    # when MORE than one branch is ready to merge.
    _KIND_LETTER: ClassVar[dict[TaskKind, str]] = {
        TaskKind.BUG: "[red]B[/]",
        TaskKind.RESEARCH: "[yellow]R[/]",
    }

    @classmethod
    def _kind_letter(cls, kind: TaskKind | None) -> str:
        if kind is None:
            return ""
        return f"{cls._KIND_LETTER.get(kind, '[green]F[/]')} "

    def _render_rows(self, selected: str | None = None) -> None:
        list_view = self.list_view
        list_view.clear()
        # markup=True on both rows: the kind letter and the deepskyblue
        # branch are painted via Rich markup — every dynamic fragment
        # (title, branch, state label) goes through escape() so literal
        # [brackets] in commit subjects never trip the parser.
        for entry in self._entries:
            marker = "❯" if entry.task_id == selected else " "
            kind_letter = self._kind_letter(entry.task_kind)
            list_view.append(
                ListItem(
                    Label(
                        f"{marker} {kind_letter}#{entry.task_id}  "
                        f"{escape(entry.title)}",
                        classes="unmerged-row-title",
                    ),
                    Label(
                        f"    [deepskyblue]{escape(entry.branch)}[/] · "
                        f"{escape(entry.state_label)}",
                        classes="unmerged-row-state",
                    ),
                )
            )
        if selected is not None:
            for index, entry in enumerate(self._entries):
                if entry.task_id == selected:
                    list_view.index = index
                    break
        elif self._entries:
            list_view.index = 0

    # -- actions ---------------------------------------------------------------------

    def action_open_screen(self) -> None:
        """enter: open the ConflictScreen for the selected task — the
        manual zdiff3 path. The list stays underneath and refreshes when
        the screen closes."""
        entry = self._selected()
        if entry is None or self._busy:
            return
        app = cast("DrCodeApp", self.app)
        app.open_merge_for(entry.task_id)

    def action_auto_resolve(self) -> None:
        """a: hand the selected task's parked conflict to the automatic
        resolver (the app's guard refuses with a toast when a round is
        already running / no model / no isolation). On a successful start
        the list closes — the round's progress is visible in the task's
        chat; after a successful round Ctrl+M shows the branch ff-ready."""
        entry = self._selected()
        if entry is None or self._busy:
            return
        app = cast("DrCodeApp", self.app)
        if app.launch_resolver(entry.task_id):
            self.dismiss()

    def action_land(self) -> None:
        """m: make the selected task's branch land. The task is prepared
        first (`merge_resolver.prepare` — idempotent, a parked merge is
        re-triaged in place), then the report decides: ff-ready → the
        `--ff-only` merge into the base; staged → the concluding commit;
        conflicted → a toast pointing at enter/a (nothing is committed
        behind the user's back)."""
        entry = self._selected()
        if entry is None or self._busy:
            return
        self._do_land(entry.task_id)

    @work(exclusive=True, group="unmerged-land")
    async def _do_land(self, task_id: str) -> None:
        self._show_busy(f"merging #{task_id} — preparing")
        app = cast("DrCodeApp", self.app)
        env = app.task_env
        try:
            task = app.manager.tasks.get(task_id)
            if task is None:
                return
            # Fast path BEFORE ensure_root: a strictly ff-able branch
            # (the base is an ancestor of the branch tip AND the two
            # tips differ) whose worktree is GONE needs no worktree —
            # recreating one (worktree_removal: on_completion) would
            # re-copy the main checkout's untracked files into the fresh
            # worktree and let the removal snapshot commit them onto the
            # finished branch (the 3f13160b incident). The ff-merge
            # itself runs in the MAIN checkout (`_ff_merge`:
            # is_fast_forward re-check, clear_identical_untracked_for_ff,
            # ff_merge_branch) and needs nothing else. An existing
            # worktree keeps the full prepare; the strictness keeps the
            # row's "ff" kind honest: a branch already contained in the
            # base ("already up to date") falls through to the full
            # prepare, which reports it as holding nothing beyond the
            # base.
            branch = env.branch(task_id)
            base = await asyncio.to_thread(env.base_branch)
            if not env.worktree_path(task_id).is_dir() and await asyncio.to_thread(
                merge_engine.strict_fast_forward, app.project_root, branch, base
            ):
                await self._ff_merge(task_id, branch, base)
                return
            try:
                root, note = await asyncio.to_thread(env.ensure_root, task_id, task)
            except WorktreeUnavailable as e:
                # `worktree add` failed: no fallback into the user's
                # checkout (the cb4937bb/540a500d incident) — show the
                # full git reason and stop.
                self.notify(
                    f"#{task_id}: worktree unavailable — {e}",
                    title="Merge",
                    severity="error",
                    timeout=8,
                )
                return
            if note:
                self.notify(
                    f"#{task_id}: worktree unavailable — {note}",
                    title="Merge",
                    severity="error",
                    timeout=8,
                )
                return
            base = await asyncio.to_thread(env.base_branch)
            report = await asyncio.to_thread(
                merge_resolver.prepare, app.project_root, task_id, root, base
            )
            if not report.ok:
                self.notify(
                    f"#{task_id}: {report.message}",
                    title="Merge",
                    severity="error",
                    timeout=8,
                )
                return
            if report.conflicted:
                self.notify(
                    f"#{task_id}: {len(report.remaining)} conflicted file(s) — "
                    "enter opens the manual screen, a runs the automatic resolver",
                    title="Merge",
                    severity="warning",
                    timeout=8,
                )
                return
            if report.fast_forward is True:
                await self._ff_merge(
                    task_id,
                    report.branch or f"drcode/{task_id}",
                    report.base_branch or base,
                )
                return
            if report.fast_forward is None:
                # The merge is resolved and staged but not concluded —
                # the commit on the task branch is the next step (after
                # it the branch becomes ff-ready, the next m lands it).
                from ..conflict import ConflictWorktree

                self._show_busy("committing")
                branch = report.branch or f"drcode/{task_id}"
                message = (
                    f"Merge {report.base_branch or base} into {branch}: "
                    f"conclude the resolved merge (via the unmerged list)"
                )
                w = ConflictWorktree(app.project_root, task_id, root, report=report)
                ok, out = await asyncio.to_thread(w.commit, message)
                if ok:
                    if task.needs_human:
                        task.needs_human = False
                        task.touch()
                        app.manager.storage.save(task)
                    # The conflict is concluded — its triage report is
                    # obsolete (a leftover would keep the footer's
                    # Merge… entry alive).
                    await asyncio.to_thread(
                        merge_engine.remove_report, app.project_root, task_id
                    )
                    app._show_event(
                        task,
                        f"Resolved merge committed on {branch} via the "
                        f"unmerged list — the merge into "
                        f"{report.base_branch or base} is now a "
                        f"fast-forward (`git merge --ff-only {branch}`).",
                        kind="workflow",
                    )
                    self.notify(
                        f"Committed on {branch} — press m again to merge it "
                        f"into {report.base_branch or base}",
                        title="Merge",
                        timeout=6,
                    )
                else:
                    self.notify(
                        f"#{task_id}: commit refused: {out}",
                        title="Merge",
                        severity="error",
                        timeout=8,
                    )
                return
            self.notify(
                f"#{task_id}: {report.message}",
                title="Merge",
                timeout=6,
            )
        finally:
            self._end_busy()
            await app._poll_git_indicator()
            if self.is_mounted:
                self._refresh_entries()

    async def _ff_merge(self, task_id: str, branch: str, base: str) -> None:
        """The list's half of the ConflictScreen's ``f``: fast-forward the
        MAIN checkout's base branch to `drcode/<id>` — only `--ff-only`,
        with the same press-time re-verification (the base may have moved
        since the row was painted); a refusal is an error toast, never a
        conflicted merge in the user's tree."""
        self._show_busy("fast-forwarding")
        app = cast("DrCodeApp", self.app)
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
        # Untracked files in the main checkout that the merge would
        # overwrite: identical copies are removed (git restores the very
        # same bytes from the repository right after the ff); differing
        # ones refuse the merge — the user moves/removes them manually.
        task = app.manager.get(task_id)
        removed, differing = await asyncio.to_thread(
            app.task_env.clear_identical_untracked_for_ff, task_id
        )
        if removed:
            app._show_event(
                task,
                f"removed {len(removed)} untracked file(s) identical to the "
                f"branch version before ff-merge: {', '.join(removed)}",
                kind="workflow",
            )
        if differing:
            self.notify(
                f"git merge --ff-only {branch} into {base} refused: untracked "
                "file(s) differ from the branch version — move or remove them "
                f"manually: {', '.join(differing)}",
                title="Merge",
                severity="error",
                timeout=10,
            )
            return
        ok, out = await asyncio.to_thread(app.task_env.ff_merge_branch, task_id)
        if not ok:
            self.notify(
                f"git merge --ff-only {branch} into {base} refused: {out}",
                title="Merge",
                severity="error",
                timeout=8,
            )
            return
        if task.needs_human:
            task.needs_human = False
            task.touch()
            app.manager.storage.save(task)
        # The merge concluded — the triage report is obsolete (a leftover
        # would keep the footer's Merge… entry alive).
        await asyncio.to_thread(merge_engine.remove_report, app.project_root, task_id)
        app._show_event(
            task,
            f"merged {branch} into {base} (ff-only) via the unmerged list",
            kind="workflow",
        )
        self.notify(f"{branch} merged into {base}", title="Merge", timeout=5)

    # -- close ------------------------------------------------------------------

    def action_close(self) -> None:
        self.dismiss()
