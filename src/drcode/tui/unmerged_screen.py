"""The unmerged-branches list screen (`drcode.tui.unmerged_screen`).

Ctrl+M with MORE than one unmerged task: a small modal list — one row
per task whose `drcode/<id>` branch still holds commits the base branch
does not have (or whose merge is parked as a conflict) — where the user
picks a CONCRETE task and acts on it without opening the per-task
conflict screen first. Exactly one candidate never reaches this screen
(`action_open_merge` opens the ConflictScreen directly, as before).

Each row shows the task (id, title), its `drcode/<id>` branch and the
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static

from .. import merge_engine, merge_resolver
from ..merge_engine import is_fast_forward

if TYPE_CHECKING:
    from .app import DrCodeApp


@dataclass
class _Entry:
    """One list row: the task, its branch and the merge state the actions
    dispatch on (`kind`: conflict / ff / staged / unmerged)."""

    task_id: str
    title: str
    branch: str
    kind: str
    base: str

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
            yield Static(
                "enter manual resolve · a auto resolve · m merge/commit · "
                "r refresh · esc close",
                id="unmerged-footer",
                markup=False,
            )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "close":
            return True  # esc stays live: a worker's write-backs are guarded
        if self._busy:
            return False
        if action in ("open_screen", "auto_resolve", "land"):
            return self._selected() is not None
        return True

    # -- state ------------------------------------------------------------------

    def on_screen_resume(self) -> None:
        """Recompute the rows every time the screen becomes active — on
        push, and after a ConflictScreen opened from here was closed (the
        user may have resolved or committed something meanwhile)."""
        self._refresh_entries()

    def _selected(self) -> _Entry | None:
        if not self._entries or not self.is_mounted:
            return None
        index = self.list_view.index
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
        self._refresh_entries()

    @work(exclusive=True, group="unmerged-refresh")
    async def _refresh_entries(self) -> None:
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
        self._render_rows(selected)

    def _render_rows(self, selected: str | None = None) -> None:
        list_view = self.list_view
        list_view.clear()
        for entry in self._entries:
            marker = "❯" if entry.task_id == selected else " "
            list_view.append(
                ListItem(
                    Label(
                        f"{marker} #{entry.task_id}  {entry.title}",
                        markup=False,
                        classes="unmerged-row-title",
                    ),
                    Label(
                        f"    {entry.branch} · {entry.state_label}",
                        markup=False,
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
        self._busy = True
        app = cast("DrCodeApp", self.app)
        env = app.task_env
        try:
            task = app.manager.tasks.get(task_id)
            if task is None:
                return
            root, note = await asyncio.to_thread(env.ensure_root, task_id, task)
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
            self._busy = False
            await app._poll_git_indicator()
            self._refresh_entries()

    async def _ff_merge(self, task_id: str, branch: str, base: str) -> None:
        """The list's half of the ConflictScreen's ``f``: fast-forward the
        MAIN checkout's base branch to `drcode/<id>` — only `--ff-only`,
        with the same press-time re-verification (the base may have moved
        since the row was painted); a refusal is an error toast, never a
        conflicted merge in the user's tree."""
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
        ok, out = await asyncio.to_thread(app.task_env.ff_merge_branch, task_id)
        if not ok:
            self.notify(
                f"git merge --ff-only {branch} into {base} refused: {out}",
                title="Merge",
                severity="error",
                timeout=8,
            )
            return
        task = app.manager.get(task_id)
        if task.needs_human:
            task.needs_human = False
            task.touch()
            app.manager.storage.save(task)
        app._show_event(
            task,
            f"merged {branch} into {base} (ff-only) via the unmerged list",
            kind="workflow",
        )
        self.notify(f"{branch} merged into {base}", title="Merge", timeout=5)

    # -- close ------------------------------------------------------------------

    def action_close(self) -> None:
        self.dismiss()
