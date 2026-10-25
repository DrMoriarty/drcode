"""Sidebar layer of the chat view mixin stack.

Everything that renders (and keeps fresh) the left column of the main
screen: the project-info line, the task sections, the todo checklist
panel and the pollers that watch the task directory for external
changes. Split out of :mod:`drcode.tui.chat_view` — see the composition
in ``ChatViewLayer``.
"""

from __future__ import annotations

from pathlib import Path

from rich.markup import escape
from textual.css.query import NoMatches
from textual.widgets import ListView

from ..models import Task, TaskKind, TaskStatus, VerificationStatus
from ..providers import AgentRole
from .sidebar import TaskItem
from .state import DrCodeState


class SidebarLayer(DrCodeState):
    def refresh_project_info(self) -> None:
        """Top of the sidebar, above the model status: working directory
        with the git branch (`~/pr/drcode:main`), then a blank line. The
        app name/version moved onto the activity-status line below."""
        branch = self._git_branch()
        dir_text = escape(self._display_dir())
        where = f"{dir_text}:{branch}" if branch else dir_text
        self.project_info.update(f"[dim]{where}[/]\n")

    def _display_dir(self) -> str:
        """Working directory with the home prefix collapsed to `~`."""
        path = self.project_root
        try:
            rel = path.relative_to(Path.home())
        except ValueError:
            return str(path)
        if str(rel) == ".":
            return "~"
        return f"~/{rel}"

    def _git_branch(self) -> str | None:
        """Current branch of the git repo in the working directory.

        Parsed straight from `.git` (also handles worktrees, where `.git`
        is a file with a `gitdir:` pointer) so no `git` binary is needed.
        """
        dot_git = self.project_root / ".git"
        try:
            if dot_git.is_dir():
                git_dir: Path | None = dot_git
            elif dot_git.is_file():
                line = dot_git.read_text(encoding="utf-8").strip()
                if not line.startswith("gitdir:"):
                    return None
                git_dir = Path(line.split(":", 1)[1].strip())
                if not git_dir.is_absolute():
                    git_dir = self.project_root / git_dir
            else:
                git_dir = None
        except OSError:
            return None
        if git_dir is None:
            return None
        try:
            content = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if content.startswith("ref: refs/heads/"):
            return content[len("ref: refs/heads/") :]
        if content.startswith("ref: "):
            return content[len("ref: ") :]
        return content[:7]  # detached HEAD

    def _sidebar_fingerprint(self) -> tuple:
        """Lightweight snapshot of everything the sidebar renders: status,
        verification label, title, the viewed-marker, the queue stamp
        (the QUEUE section is order-sensitive) and the mergedness color
        of the ✓ marker per task. The merge state lives outside the task
        objects (updated off the event loop by the 10 s git poll), so
        without it here a color-only change would not redraw — the same
        staleness trap _poll_sidebar's fingerprint guards against."""
        merge_state = self._merge_state
        return tuple(
            (
                t.id,
                t.status.value,
                t.kind.value,
                t.verification,
                t.title_display,
                t.id == self.viewed_id,
                t.queue_seq,
                t.queue_head,
                t.needs_human,
                merge_state.get(t.id, ""),
            )
            for t in self.manager.all_tasks()
        )

    def _poll_sidebar(self) -> None:
        """Redraw the sidebar only when the underlying task state changed
        since the last render (status transitions, new bug reports from the
        tester, promote_next() activations, …). Also keeps the todo panel
        in sync (fallback path — todo tool calls refresh it instantly)."""
        todo_fp = self._todo_fingerprint()
        if todo_fp != self._todo_snapshot:
            self._todo_snapshot = todo_fp
            self.refresh_todo_panel()
        self._poll_chat_drain()
        if self._sidebar_fingerprint() != self._sidebar_snapshot:
            self.refresh_sidebar()
            # The task status header (and the bottom statusbar) have no
            # poller of their own: they are refreshed explicitly, and mid-run
            # state changes (set_verification flips the viewed task to
            # COMPLETED while its tester run is still finishing) would leave
            # them stale until the run ends — or a view switch.
            self.refresh_status()

    def _todo_fingerprint(self) -> tuple:
        """Everything the todo panel renders: new-task mode, the viewed
        task's id and its checklist items (text + done flag)."""
        if self._new_task_mode:
            return ("new-task",)
        task = self.viewed
        if task is None:
            return ()
        return (task.id, tuple((t.text, t.done) for t in task.todos))

    def refresh_todo_panel(self) -> None:
        """Re-render the todo panel from the viewed task's checklist.

        Visible (auto-sized) only when the task has a non-empty `todos`;
        hidden otherwise. The default collapse state matches the historic
        behavior: a fully completed checklist collapses to the single
        `todos N/N` counter line, anything else shows all items. Clicking
        the header toggles the state manually (session-only override on
        the panel, reset when another task is viewed). Done items render
        struck-through-dim (rich does
        not strike through reliably across terminals — dim is the portable
        equivalent); unchecked ones in the normal text color. Read-only:
        the agent owns the list."""
        try:
            panel = self.todo_panel
        except NoMatches:
            return
        task = self.viewed
        if task is None or not task.todos:
            panel.add_class("hidden")
            return
        # A manual collapse override lives only within one task's view:
        # switching tasks falls back to the default auto behavior.
        if panel.task_id != task.id:
            panel.task_id = task.id
            panel.user_collapsed = None
        done = len(task.todos) - len(task.unchecked_todos)
        collapsed = (
            panel.user_collapsed
            if panel.user_collapsed is not None
            else done == len(task.todos)
        )
        # The whole header line is the click target (`@click` action, same
        # mechanism as the task-id mention links); Textual underlines link
        # spans on hover, which doubles as the affordance hint.
        lines = [
            (
                f"[@click=app.toggle_todo_panel][bold $accent]todos[/] "
                f"[dim]{done}/{len(task.todos)}[/dim][/]"
            ),
        ]
        if collapsed:
            # Collapsed: the counter line alone — either the default (a
            # fully completed checklist carries no useful item signal
            # anymore) or a user override.
            panel.update("\n".join(lines))
            panel.remove_class("hidden")
            return
        for i, item in enumerate(task.todos, 1):
            text = escape(item.text)
            # The [x]/[ ] markers are escaped (\[) so rich keeps them
            # literal instead of eating [x] as a markup tag.
            if item.done:
                lines.append(f"[dim]  \\[x] {i}. {text}[/]")
            else:
                lines.append(f"  \\[ ] {i}. {text}")
        panel.update("\n".join(lines))
        panel.remove_class("hidden")

    def action_toggle_todo_panel(self) -> None:
        """`@click` target of the todo panel's header line: flip the
        checklist between collapsed and expanded.

        The first click on an untouched panel inverts its DEFAULT state
        (a fully-completed checklist auto-collapses, anything else is
        expanded); further clicks simply toggle. The override is
        session-only — `refresh_todo_panel` resets it on a task switch."""
        try:
            panel = self.todo_panel
        except NoMatches:
            return
        task = self.viewed
        if task is None or not task.todos:
            return
        done = len(task.todos) - len(task.unchecked_todos)
        if panel.user_collapsed is None:
            # Invert the auto state: default-collapsed → expand, and
            # vice versa.
            panel.user_collapsed = done != len(task.todos)
        else:
            panel.user_collapsed = not panel.user_collapsed
        self.refresh_todo_panel()

    def _poll_new_tasks(self) -> None:
        """Watch `<project>/.drcode/tasks/` for tasks created by other
        processes while the app is running (`drcode --feature-request DESC`
        in a second terminal, a one-shot GitHub sync).

        promote_next() reloads new files itself, but only when the pipeline
        actively searches the queue — a settled idle pipeline has nothing
        left to trigger it. This poller closes that gap: external tasks
        appear in the sidebar and, when the pipeline is idle, are promoted
        right away instead of idling until a restart. The reload is a cheap
        directory scan (known files are not re-read), so a 1s interval is
        safe; a failed scan just waits for the next tick.
        """
        try:
            new = self.manager.reload_new_tasks()
        except Exception:  # noqa: BLE001 - a broken scan must not kill the poller
            return
        if not new:
            return
        for task in new:
            self._show_event(
                task,
                "Task created outside this session "
                "(e.g. `drcode --feature-request`) — loaded from the "
                "project directory.",
            )
        self.refresh_sidebar()
        self.refresh_status()
        # Same gate as the startup advance (on_mount): without a configured
        # model the queue stays parked — the tasks remain visible in the
        # sidebar and are promoted once a model is picked. Advancing into
        # run_agent anyway would promote a task to ACTIVE and strand it
        # there (run_agent's not-configured branch leaves it be).
        if self.resolved_config(AgentRole.CODING).is_ready:
            self._advance()

    def _sidebar_sections(self) -> list[tuple[str, list[Task]]]:
        """Sidebar content: pipeline sections top to bottom, empty ones
        dropped (the sidebar shows only the stages that have tasks).

        Shared by the sidebar rendering and the startup view pick (the
        opened task must be the one at the top of the sidebar).
        """
        # Group tasks by status; each group keeps the created_at order of
        # all_tasks() unless sorted explicitly below.
        by_status: dict[TaskStatus, list[Task]] = {}
        for t in self.manager.all_tasks():
            by_status.setdefault(t.status, []).append(t)

        queue = list(reversed(self.manager.queued()))
        clarification = sorted(
            by_status.get(TaskStatus.AWAITING_CLARIFICATION, []),
            key=lambda t: t.updated_at,
            reverse=True,
        )
        active = by_status.get(TaskStatus.ACTIVE, [])
        testing = by_status.get(TaskStatus.TESTING, [])
        # Normally one task is being tested; if several are stuck in
        # TESTING, the one the tester is working on goes to the bottom of
        # the section (closest to the pipeline's exit into COMPLETED).
        testing_active_id = self._tester_task_id
        testing = [t for t in testing if t.id != testing_active_id] + [
            t for t in testing if t.id == testing_active_id
        ]
        completed = sorted(
            by_status.get(TaskStatus.COMPLETED, []),
            key=lambda t: t.updated_at,
            reverse=True,
        )
        hold = by_status.get(TaskStatus.HOLD, [])

        sections: list[tuple[str, list[Task]]] = [
            ("QUEUE", queue),
            ("NEEDS CLARIFICATION", clarification),
            ("ACTIVE", active),
            ("TESTING", testing),
            ("COMPLETED", completed),
            ("ON HOLD", hold),
        ]
        return [(name, tasks) for name, tasks in sections if tasks]

    def _sidebar_top_task(self) -> Task | None:
        """First task of the first non-empty sidebar section — what the
        chat pane opens on startup."""
        for _name, tasks in self._sidebar_sections():
            return tasks[0]
        return None

    def _task_row_text(self, t: Task) -> str:
        """Rendered label of a task row (the incremental refresh compares
        it against the mounted row to skip unchanged labels)."""
        if t.status.value == "completed":
            # The ✓ color tracks the mergedness of the task's
            # `drcode/<id>` branch (refreshed by the 10 s git poll):
            # orange while the work sits unmerged, green once merged,
            # plain when no branch exists (isolation off, no repo, or
            # the merged branch was cleaned up) — the accepted
            # "temporary green" limitation: the cleanup deletes a merged
            # branch, turning the task white again.
            color = {
                "unmerged": "orange",
                "merged": "green",
            }.get(self._merge_state.get(t.id, "none"))
            marker = f"[{color}]✓[/]" if color else "✓"
        else:
            marker = {
                "queued": "⏳",
                "hold": "⏸",
                "active": "▶",
                "testing": "🧪",
                "completed": "✓",
                "clarification": "❓",
            }.get(t.status.value, "⏳")
        kind_letter = {
            TaskKind.BUG: "[red]B[/]",
            TaskKind.RESEARCH: "[yellow]R[/]",
        }.get(t.kind, "[green]F[/]")
        if t.verification is VerificationStatus.PASSED:
            suffix = " ✅"
        elif t.verification is VerificationStatus.FAILED:
            suffix = " ❌"
        else:
            suffix = ""
        # Dependency gate: a task whose depends_on is not satisfied yet
        # gets the lock marker and the ids it waits for.
        deps_suffix = self._dependency_marker(t)
        # The failed-conflict-fallback marker: the agent resolver gave up
        # on this task's parked merge — a human must take over (Ctrl+M).
        human_suffix = " [bold red]✋[/]" if t.needs_human else ""
        viewed = " ●" if t.id == self.viewed_id else ""
        return (
            f"{marker} {kind_letter} #{t.id} "
            f"{escape(t.title_display)}{suffix}{deps_suffix}{human_suffix}{viewed}"
        )

    def _sidebar_row(
        self,
        key: tuple[str, str],
        text: str,
        *,
        task_id: str | None = None,
        header: bool = False,
        section: str | None = None,
        viewed: bool = False,
    ) -> TaskItem:
        """Reuse (or create) the mounted row for *key* and sync it to the
        desired *text*/`viewed` class.

        Rows are tracked in `_sidebar_items` so a refresh only creates the
        rows that appeared and rewrites the labels that changed — the
        full rebuild of ~2 widgets per row (ListItem + Label) with mount
        messages and layout passes is what made the sidebar redraw take
        over a second at a few hundred completed tasks."""
        item = self._sidebar_items.get(key)
        if item is None:
            item = TaskItem(text, task_id, header=header, section=section)
            if viewed:
                item.add_class("viewed")
            self._sidebar_items[key] = item
            return item
        if item.text != text:
            item.set_text(text)
        if viewed != item.has_class("viewed"):
            if viewed:
                item.add_class("viewed")
            else:
                item.remove_class("viewed")
        return item

    def _reconcile_sidebar(
        self, sidebar: ListView, desired: list[TaskItem], tracked: set[TaskItem]
    ) -> None:
        """Bring the sidebar's row order to *desired*, reusing widgets.

        Rows already mounted keep their place while they match the desired
        order; new rows are mounted (before the first following row that
        is already in place, or appended at the end); tracked rows sitting
        at a wrong position are moved with `move_child` — an in-place
        reorder of the node list, since textual has no mount-time
        reparenting and re-creating a row would defeat the reuse. Rows
        that were removed from the tracking dict (pending their async
        prune) are invisible to this: they never match a desired row and
        are skipped as placement anchors.
        """
        # Mirror of the tracked rows' DOM order, updated after every
        # mount/move below. Each row is placed at its absolute desired
        # position: if live[:i] is already correct, the row now occupying
        # position i is exactly what must follow the row being placed —
        # anchoring against anything else let a moved row jump over a
        # freshly mounted header (two tasks promoted at once could land
        # ABOVE their "ACTIVE (2)" header).
        live = [child for child in sidebar._nodes if child in tracked]
        for i, item in enumerate(desired):
            if i < len(live) and live[i] is item:
                continue
            following = live[i] if i < len(live) else None
            if item in live:
                live.remove(item)
            if following is not None:
                if item in tracked:
                    if item in sidebar._nodes:
                        sidebar.move_child(item, before=following)
                    # else: the row's mount (or its position among the
                    # desired rows) is still an in-flight async round-trip
                    # — moving it now would raise WidgetError (not a child
                    # YET); the next refresh reconciles its position.
                else:
                    # A row created by THIS refresh (never mounted yet).
                    sidebar.mount(item, before=following)
                live.insert(i, item)
            else:
                if item in tracked:
                    if live and item in sidebar._nodes:
                        sidebar.move_child(item, after=live[-1])
                    # else: no tracked row left to anchor against, or the
                    # row's mount is still in flight — its position among
                    # the pending-prune rows is irrelevant either way.
                else:
                    sidebar.mount(item)
                live.append(item)

    def refresh_sidebar(self) -> None:
        # Called from worker finally-blocks too — it can fire while the
        # app is shutting down and the screen (with the sidebar) is
        # already gone; a NoMatches there would panic the whole app.
        try:
            sidebar = self.sidebar
        except NoMatches:
            return
        mounted_before = set(self._sidebar_items.values())
        # Incremental reconcile instead of clear()+append-everything: the
        # typical update (a status flip, a new bug report at the queue
        # head) touches a handful of rows instead of recreating them all.
        # The highlighted row is detected via the ListItem.highlighted
        # reactive, not highlighted_child: the latter is index-based and
        # desyncs when an async prune has already shrunk the node list
        # under an unchanged index (see the tail of this method).
        highlighted = next(
            (w for w in sidebar._nodes if isinstance(w, TaskItem) and w.highlighted),
            None,
        )
        # The cursor counts as legitimate only when the user is actively
        # navigating the sidebar (it has focus) or it sits on the viewed
        # task's row. A cursor parked anywhere else — left behind by a
        # click or a task switch — is stale: kept, it would paint that row
        # in the cursor colour while the chat pane shows a different task,
        # or nothing at all while the new-task form is open.
        keep_cursor = highlighted is not None and (
            sidebar.has_focus
            or (
                isinstance(highlighted, TaskItem)
                and highlighted.task_id == self.viewed_id
            )
        )
        items: list[TaskItem] = []
        keys: set[tuple[str, str]] = set()
        for name, tasks in self._sidebar_sections():
            keys.add(("header", name))
            # A collapsed section keeps only its header line: the "+" shows
            # there is content hidden underneath, "-" means fully expanded.
            collapsed = name in self._collapsed_sections
            marker = "+" if collapsed else "-"
            items.append(
                self._sidebar_row(
                    ("header", name),
                    f"{marker} {name} ({len(tasks)})",
                    header=True,
                    section=name,
                )
            )
            if collapsed:
                continue
            for t in tasks:
                keys.add(("task", t.id))
                items.append(
                    self._sidebar_row(
                        ("task", t.id),
                        self._task_row_text(t),
                        task_id=t.id,
                        viewed=t.id == self.viewed_id,
                    )
                )
        # Rows whose key disappeared: drop from the tracking dict and
        # remove. The removal is async (a Prune round-trip), so a row can
        # linger in the DOM for a moment — the sidebar's click handler
        # already ignores items that are no longer ListView children, and
        # the reconcile below treats only tracked rows as anchors.
        for key in [*self._sidebar_items]:
            if key not in keys:
                item = self._sidebar_items.pop(key)
                mounted_before.discard(item)
                if item is highlighted:
                    keep_cursor = False
                item.remove()
        self._reconcile_sidebar(sidebar, items, mounted_before)
        # The reconcile is position-based and mount/move do not shift the
        # reactive index: put the cursor back onto the row it was on (a
        # full rebuild used to drop it entirely; callers restore it via
        # _refocus_sidebar_task / toggle_sidebar_section when they need a
        # specific row highlighted).
        if not keep_cursor:
            if highlighted is not None:
                sidebar.index = None
        else:
            assert highlighted is not None
            sidebar.index = sidebar._nodes.index(highlighted)
        # ListView.watch_index clears the old highlight by *position*
        # (_nodes[old_index]), not by widget. When rows moved under the
        # reconcile or an async prune shrank the node list between two
        # refreshes, the widget at the old index is not the one that was
        # painted — the painted row keeps its -highlight class forever,
        # and every such mismatch adds another stuck highlighted row
        # (several rows lit in the cursor colour, none of them the viewed
        # task). Clear the stale highlight reactives directly; only the
        # widget the index actually points at may stay lit.
        cursor = sidebar.highlighted_child
        for item in sidebar._nodes:
            if isinstance(item, TaskItem) and item.highlighted and item is not cursor:
                item.highlighted = False
        self._sidebar_snapshot = self._sidebar_fingerprint()

    def _dependency_marker(self, task: Task) -> str:
        """Sidebar suffix of the dependency gate: " 🔒 ⇢ <id1>,<id2>" for a
        task whose dependencies are not satisfied yet, "" otherwise.

        Status half from the manager (a dep exists and is not COMPLETED);
        the mergedness half from the promote worker's last computed gate
        (covers deps whose task file is gone but whose branch still holds
        unmerged work)."""
        if not task.depends_on:
            return ""
        gate = self._dep_gate.get(task.id)
        if gate:
            waiting = list(gate)
        elif self.manager.blocking_dependencies(task.id):
            waiting = list(task.depends_on)
        else:
            return ""
        return f" [red]🔒[/] ⇢ {','.join(waiting)}"

    def toggle_sidebar_section(self, section: str) -> None:
        """Collapse/expand one sidebar category (its header row).

        The cursor is put back onto the toggled header — the next
        space/enter keeps working on it.
        """
        if section in self._collapsed_sections:
            self._collapsed_sections.discard(section)
        else:
            self._collapsed_sections.add(section)
        self.refresh_sidebar()
        sidebar = self.sidebar
        for index, item in enumerate(sidebar.children):
            if isinstance(item, TaskItem) and item.section == section:
                sidebar.index = index
                break

    def _refocus_sidebar_task(self, task_id: str) -> None:
        """Put the ListView cursor onto a task's row, so a series of
        reorder/hold keys keeps working on the same task without
        re-selecting it."""
        sidebar = self.sidebar
        for index, item in enumerate(sidebar.children):
            if isinstance(item, TaskItem) and item.task_id == task_id:
                sidebar.index = index
                break

    def _sync_sidebar_cursor(self, task_id: str | None) -> None:
        """Align the ListView cursor with the viewed task: onto its row, or
        dropped entirely when nothing is viewed (new-task form / empty
        backlog).

        With a task viewed the sync is skipped while the sidebar has
        focus — the user's arrow-key position wins there (browsing must
        not be yanked back by a background view_task of the running
        task). Clearing is unconditional: with nothing viewed there is no
        position to protect, and the next arrow key re-establishes the
        cursor."""
        try:
            sidebar = self.sidebar
        except NoMatches:
            return
        if task_id is None:
            sidebar.index = None
            for item in sidebar._nodes:
                if isinstance(item, TaskItem) and item.highlighted:
                    item.highlighted = False
            return
        if sidebar.has_focus:
            return
        self._refocus_sidebar_task(task_id)
