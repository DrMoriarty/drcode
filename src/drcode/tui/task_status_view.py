"""Task-status layer of the chat view mixin stack.

The bottom statusbar (active task, queue length, agent/tester state,
model) and the task statusbar under the chat log (kind, id, title,
creation date, live duration, dependency gate, context-fill counter
with its modal dialog). Split out of :mod:`drcode.tui.chat_view` — see
the composition in ``ChatViewLayer``.
"""

from __future__ import annotations

from rich.cells import cell_len
from rich.markup import escape
from rich.text import Text
from textual.css.query import NoMatches
from textual.widgets import Static

from ..agent import all_user_prompts, last_context_tokens
from ..context_layout import (
    AGENT_DEV,
    AGENT_RESOLVER,
    AGENT_TESTER,
    agent_transcript,
    last_agent,
    task_reports,
)
from ..models import Task, TaskKind, TaskStatus, VerificationStatus, _now
from ..providers import AgentRole
from ..settings import WebFetchMode
from ..web_fetch import web_fetch_policy_for
from .ctx_dialog import ContextDialog
from .state import DrCodeState, update_static_if_changed


class TaskStatusLayer(DrCodeState):
    def refresh_status(self) -> None:
        # Called from worker finally-blocks too — it can fire while the
        # app is shutting down and the screen (with the statusbar) is
        # already gone; a NoMatches there would panic the whole app.
        try:
            statusbar = self.query_one("#statusbar", Static)
        except NoMatches:
            return
        active = self.manager.active_tasks()
        queue_len = len(self.manager.queued())
        testing_len = len(
            [t for t in self.manager.all_tasks() if t.status is TaskStatus.TESTING]
        )
        state = "running" if self.agent_running else "idle"
        tester = "running" if self.tester_running else "idle"
        # Armed-ESC confirmation (esc_abort: double_press): show that the
        # next ESC within the window will stop the run(s).
        esc = " (esc armed)" if self._esc_armed_signature else ""
        if active:
            who = f"#{active[0].id} {active[0].title_display}"
            if len(active) > 1:
                # Parallel slots: several tasks in development at once.
                who += f" (+{len(active) - 1})"
        else:
            who = "none"
        model = self.resolved_config(AgentRole.CODING).model or (
            "(model not configured)"
        )
        statusbar.update(
            f" active: {who}   queue: {queue_len}   testing: {testing_len}   "
            f"agent: {state}   tester: {tester}{esc}   model: {model} "
            f"{self._mcp_status_text()}"
        )
        self.refresh_task_status()

    def _mcp_status_text(self) -> str:
        """The statusbar's MCP field (design doc §6: "MCP :51234").

        Shows the REAL server state, never the setting: `off` with the
        setting disabled, `:PORT` while our server runs, `busy` when
        another instance serves this project (§3 — our own server is
        deliberately not started then), and `error` when the setting is on
        but nothing came up. The TOKEN is never rendered — the endpoint
        file's path is the only detail the UI offers (§6).
        """
        service = getattr(self, "mcp_service", None)
        if service is None:  # pragma: no cover - pre-init render
            return "mcp: off "
        if service.running:
            return f"mcp: :{service.port} "
        if not self.settings.mcp_enabled:
            return "mcp: off "
        if service.foreign_pid is not None:
            return "mcp: busy "
        return "mcp: error "

    def refresh_task_status(self) -> None:
        # The footer mirrors the state check_action() gates on (viewed task,
        # its status, running flags): re-evaluate the visible bindings —
        # but only when they actually changed. This also runs on the 1 s
        # activity poll, and a bindings publish that changes nothing is NOT
        # free: textual's Footer recomposes from scratch on every publish
        # (every FooterKey remounted, styles re-applied, screen relaid out)
        # — on a fat archive that used to burn continuous idle CPU.
        # active_bindings is the exact input Footer.compose() renders.
        try:
            fp = tuple(
                (
                    key,
                    binding.key,
                    binding.description,
                    binding.show,
                    enabled,
                    tooltip,
                )
                for key, (_, binding, enabled, tooltip) in sorted(
                    self.screen.active_bindings.items()
                )
            )
        except Exception:  # noqa: BLE001 - teardown race: keep the old behavior
            self.refresh_bindings()
        else:
            if fp != self._footer_bindings_fp:
                self._footer_bindings_fp = fp
                self.refresh_bindings()
        try:
            left = self.query_one("#task-status-left", Static)
            mid = self.query_one("#task-status-mid", Static)
            right = self.query_one("#task-status-right", Static)
            extra = self.query_one("#task-status-extra", Static)
        except NoMatches:
            return
        # The second row's fragments are deliberately colored (cyan deps /
        # magenta goal / deepskyblue branch), and almost every one of them is also
        # a link (@click). Textual applies the widget's link style ON TOP of
        # any segment carrying @click meta, and the auto link color (near-
        # white over the panel background) plus the underline paints over
        # the markup color. auto_links = False skips that overlay — click
        # dispatch works off the segment meta, not the link style, so the
        # links keep working; only the underline/hover highlight is given
        # up (the colors carry the clickability). Not settable via CSS —
        # auto_links is a reactive, not a style rule.
        extra.auto_links = False
        # The new-task form is not tied to any task: keep the bar empty so
        # the previously viewed task doesn't keep updating behind the form.
        if self._new_task_mode:
            self._set_extra_row(extra, "")
            update_static_if_changed(left, "")
            update_static_if_changed(mid, "")
            update_static_if_changed(right, "")
            return
        task = self.viewed
        if task is None:
            self._set_extra_row(extra, "")
            update_static_if_changed(left, "")
            update_static_if_changed(mid, "")
            update_static_if_changed(right, "")
            return
        kind = {
            TaskKind.BUG: "[bold red]bug[/]",
            TaskKind.RESEARCH: "[bold yellow]research[/]",
        }.get(task.kind, "[bold green]feature[/]")
        # The task's own id is a link too: clicking it copies the
        # short id (with the `#` prefix) to the clipboard
        # (action_copy_task_id).
        id_markup = f"[@click=app.copy_task_id('{task.id}')]#{task.id}[/]"
        # Full whitespace-normalized title (NOT title_display — that one
        # hard-caps at 40 chars for the narrow sidebar): the adaptive
        # layout below gives the title all room between #id and the right
        # cluster and truncates only when the row is physically too short.
        title_plain = " ".join(task.title.split())
        status_colors = {
            TaskStatus.DRAFT: "bright_black",
            TaskStatus.QUEUED: "yellow",
            TaskStatus.HOLD: "blue",
            TaskStatus.ACTIVE: "cyan",
            TaskStatus.TESTING: "magenta",
            TaskStatus.COMPLETED: "green",
            TaskStatus.AWAITING_CLARIFICATION: "yellow",
        }
        status_color = status_colors[task.status]
        # Duration counter: only for the statuses where the time spent is
        # meaningful and live — the developer working (ACTIVE) and the
        # tester verifying (TESTING). Stamped at every status transition
        # (fire_status_change), so this measures the current continuous
        # stay, never a cumulative one; survives restarts via the stamp.
        # Rendered in the status's own color, one space after the status.
        timer_plain: str | None = None
        if (
            task.status in (TaskStatus.ACTIVE, TaskStatus.TESTING)
            and task.status_changed_at is not None
        ):
            seconds = max(0, int((_now() - task.status_changed_at).total_seconds()))
            timer_plain = f"⏱ {self._fmt_age(seconds)}"
        # The verification marker stays glued to the status fragment
        # (single space — it reads as one unit with the status word).
        # Both the markup and its plain-text width are kept in lockstep:
        # the measurement below must mirror the rendered fragments exactly.
        marker_plain = ""
        marker_markup = ""
        if task.verification is VerificationStatus.PASSED:
            marker_plain, marker_markup = " ✓ testing", " [green]✓ testing[/]"
        elif task.verification is VerificationStatus.FAILED:
            marker_plain, marker_markup = " ✗ testing", " [red]✗ testing[/]"
        # Context counter (rightmost of the right cluster, in its own
        # widget so its [@click] area and the ctx-warn/ctx-crit coloring
        # cover exactly the counter). The helper drives both.
        counter = self._ctx_counter(task)
        right.remove_class("ctx-warn", "ctx-crit")
        if counter is None:
            right.tooltip = None
            counter_markup = counter_plain = ""
        else:
            counter_markup, counter_plain, warn_class = counter
            if warn_class:
                right.add_class(warn_class)
            right.tooltip = "Click: context layout (≈ breakdown + exact usage)"

        # --- adaptive layout ------------------------------------------------
        # The title fills ALL the room between the #id and the right cluster
        # (date · status · counter). When less than 40 cells remain for it,
        # the right cluster degrades stepwise (compact date → timer-only or
        # hidden status → truncated title); the room is recomputed after
        # every step. Widths are measured over PLAIN text (markup tags take
        # no columns) with rich's cell_len — the same measure the rendering
        # itself uses, so emoji (🧠/⏱) are counted at their true cell width.
        row = self.query_one("#task-status-row")
        total = row.container_size.width or None
        kind_plain = Text.from_markup(kind).plain

        def title_room(date_plain: str, status_plain: str) -> int | None:
            """Cells left for the title under the given right-cluster
            fragments (None when the row has no real width yet — pre-layout,
            then nothing degrades or truncates)."""
            if total is None:
                return None
            left_fixed = cell_len(kind_plain) + 3 + cell_len(f"#{task.id}") + 1
            right_widths = [
                w
                for w in (
                    cell_len(date_plain),
                    cell_len(status_plain),
                    cell_len(counter_plain),
                )
                if w
            ]
            return (
                total - left_fixed - sum(right_widths) - max(0, len(right_widths) - 1)
            )

        date_plain = task.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
        # The MEASURED status fragment must mirror the RENDERED markup
        # exactly — status word + (" " + timer) + marker, glued pieces
        # included. Measuring the bare status word alone overestimates the
        # title room by the timer/marker widths: the degradation steps fire
        # late and step (c) truncates the title past the physical row, so
        # the tail is hard-clipped without an ellipsis.
        status_full_plain = (
            task.status.value
            + (f" {timer_plain}" if timer_plain is not None else "")
            + marker_plain
        )
        # Step-b variant: a timer status keeps only the live ⏱ (in the
        # status's color, the verification marker still glued to it), a
        # timerless status (queued/hold/completed/draft/
        # awaiting_clarification) hides entirely — marker included.
        status_step_b_plain = (
            f"{timer_plain}{marker_plain}" if timer_plain is not None else ""
        )
        status_plain = status_full_plain
        status_word_hidden = False
        room = title_room(date_plain, status_plain)
        if room is not None and room < 40:
            # Step a: compact the creation date.
            date_plain = task.created_at.astimezone().strftime("%d.%m %H:%M")
            room = title_room(date_plain, status_plain)
        if room is not None and room < 40:
            # Step b: drop the status word.
            status_plain = status_step_b_plain
            status_word_hidden = True
            room = title_room(date_plain, status_plain)
        if room is not None and room < 0:
            # Step d (physical-fit floor): a negative room means the right
            # cluster no longer FITS beside the fixed left prefix (type +
            # #id) — the auto-width widgets win the layout and squeeze
            # the 1fr left below its own content, hard-clipping the id.
            # The timer is the most ephemeral piece → drop it first, then
            # the date; the counter stays (the spec keeps it untouched).
            status_plain = ""
            status_word_hidden = True
            room = title_room(date_plain, status_plain)
        if room is not None and room < 0:
            date_plain = ""
            room = title_room(date_plain, status_plain)
        if room is not None:
            # Step c (and the general overflow case): the title is cut with
            # an ellipsis to exactly the remaining room.
            title_plain = self._truncate_cells(title_plain, room)

        date_markup = f"[dim]{date_plain}[/]" if date_plain else ""
        if status_plain:
            if status_word_hidden:
                status_markup = f"[{status_color}]{timer_plain}[/]{marker_markup}"
            else:
                status_markup = (
                    f"[{status_color}]{task.status.value}[/]"
                    + (f" [{status_color}]{timer_plain}[/]" if timer_plain else "")
                    + marker_markup
                )
        else:
            status_markup = ""
        left_markup = f"{kind}   {id_markup} {escape(title_plain)}"
        # The right cluster in text order: date → status → counter, single
        # spaces between the fragments. The date+status live in the mid
        # widget, the counter in the right one — the single space between
        # them is the trailing space of the mid content, so the adjacency
        # of the two auto-width widgets reads as one cluster.
        mid_markup = " ".join(
            fragment for fragment in (date_markup, status_markup) if fragment
        )
        if mid_markup:
            mid_markup += " "
        update_static_if_changed(left, left_markup)
        update_static_if_changed(mid, mid_markup)
        update_static_if_changed(right, counter_markup)
        self._set_extra_row(extra, self._task_status_extra(task))

    def _dependency_status_part(self, task: Task, label: str = "deps: ") -> str:
        """Task-statusbar fragment listing the task's dependencies with
        their per-dependency gate state:

        - `#id ✓` — the dependency is COMPLETED;
        - `#id …` — the dependency task exists but is not completed yet;
        - `#id ⇴` — the promote worker's last gate run found unmerged
          commits on the dependency's `drcode/<id>` branch (also for a
          dep whose task file is already gone — the branch outlives it);
        - `#id ?` — an id that is not in the manager and has no branch
          (a stale entry; the gate treats it as satisfied).

        Each id is a link that opens the dependency task. For the
        statuses where the gate is still editable (QUEUED / HOLD — the
        ``_can_edit_dependencies`` gate) every single dependency is
        followed by its own small red ✕: the click removes THAT
        dependency (the ``action_remove_viewed_dependency`` link asks for
        a confirmation first — the removal is a force-start). The ✕ hugs
        its own `#id`, so the pair reads as one unit and the right one is
        dropped; for every other status the list stays read-only.

        Empty string when the task has no dependencies; `label` prefixes
        the list ("deps: " on the first header row, "depends on: " on the
        second).

        The whole fragment is tinted cyan (the dependency-list color of
        the second header row, distinct from the magenta goal and the
        deepskyblue branch); the per-dependency state colors below override it
        inside their own spans."""
        if not task.depends_on:
            return ""
        gate = self._dep_gate.get(task.id, {})
        # Same gate as the "+" link of the group: only a task that has not
        # started yet can have its dependencies edited.
        editable = self._can_edit_dependencies(task)
        bits: list[str] = []
        for dep in task.depends_on:
            # Each #id marker is itself a link to the dependency task (ids
            # are lowercase hex — no escaping needed inside the markup tag).
            link = f"[@click=app.open_task('{dep.lower()}')]#{dep.lower()}[/]"
            dep_task = self.manager.tasks.get(dep)
            if gate.get(dep) == "unmerged":
                bit = f"[red]{link}⇴[/]"
            elif dep_task is not None and dep_task.status is TaskStatus.COMPLETED:
                bit = f"[green]{link}✓[/]"
            elif dep_task is not None:
                bit = f"[yellow]{link}…[/]"
            else:
                bit = f"[dim]{link}?[/]"
            if editable:
                # The per-dependency ✕, modeled on the ✏/✕ icon pair of the
                # goal group: it carries the id in the click action, so a
                # click always targets THIS dependency, and it is drawn in
                # the deps group's cyan context but red on its own.
                bit += (
                    f" [@click=app.remove_viewed_dependency('{dep.lower()}')]"
                    "[red]✕[/][/]"
                )
            bits.append(bit)
        return f"[cyan]{label}{' '.join(bits)}[/]"

    def _task_status_extra(self, task: Task) -> str:
        """The second header row (under the kind/id/date/status line):
        the full dependency list with its gate colors — each of them
        followed by its own ✕ removal link for a queued/held task — plus
        the cyan "+" picker link (same statuses), the goal link
        with its ✏/✕ icon fragments (or the "+ assign" picker
        link when unattached), the GitHub issue number, the
        worktree branch (clickable — copies the branch name), the
        round counters (dev/test/resolve, dim, hidden while all zero)
        and the warning markers that the sidebar also carries
        (needs_human, unfinished). Fragments the task has none of are omitted; an
        entirely empty result hides the row (`_set_extra_row`)."""
        bits: list[str] = []
        deps = self._dependency_status_part(task, label="depends on: ")
        if deps:
            bits.append(deps)
        # The "+" of the dependency group: only for the statuses where the
        # gate can still change (queued/held — the _can_edit_dependencies
        # gate), and rendered both with and without existing dependencies
        # (an unconditional "add another one" right after the list). Cyan,
        # the color of the group it belongs to.
        if self._can_edit_dependencies(task):
            if not deps:
                # A bare "+" would read as noise at the head of the row:
                # label the empty state the way the list is labelled.
                bits.append("[@click=app.pick_dependency][cyan]depends on: +[/][/]")
            else:
                bits.append("[@click=app.pick_dependency][cyan]+[/][/]")
        # Goal link — a task attached to a product goal opens the
        # GoalScreen focused on it, with the ✏ (re-assign to another
        # goal via the picker) and ✕ (detach) icon buttons before the
        # goal title — the leftmost fragments of the goal group; a
        # dangling id (the goal was deleted) is shown dim and keeps the
        # assign link so the user can re-link.
        goal = (
            self.manager.goals.get(task.goal_id)
            if task.goal_id and task.goal_id in self.manager.goals
            else None
        )
        if goal is not None:
            # The two compact icon buttons come first (left of the goal
            # fragment): ✏ opens the assign picker, ✕ detaches the task
            # from the goal. The whole group is one fragment joined with
            # single spaces — the icons hug the goal title (the row's
            # three-space fragment separator would leave a double gap).
            bits.append(
                "[@click=app.assign_goal][deepskyblue]✏[/][/] "
                "[@click=app.unlink_viewed_goal][red]✕[/][/] "
                # The goal fragment is magenta (the goal color of this
                # row, distinct from the cyan deps and the deepskyblue branch).
                # No goal id in the text: the sidebar shows goals by title
                # only, the id means nothing to the user (the click target
                # still carries it).
                f"[magenta][@click=app.open_goals('{goal.id}')]goal: "
                f"◆ {escape(goal.title_display)}[/][/]"
            )
        elif task.goal_id:
            # Dangling link (the goal was deleted): no title to show and
            # the bare id tells the user nothing — just a dim marker, the
            # "+ assign" link below handles re-linking.
            bits.append("[dim]goal: ?[/]")
        if goal is None:
            # Same magenta family as the goal fragment it stands in for.
            bits.append("[@click=app.assign_goal][magenta]+ assign[/][/]")
        if task.github_issue is not None:
            bits.append(f"issue: [dim]#{task.github_issue}[/]")
        if task.isolation_pinned:
            # The branch fragment is deepskyblue (the branch color of this row,
            # distinct from the cyan deps and the magenta goal).
            branch = f"drcode/{task.id}"
            bits.append(
                f"[@click=app.copy_task_branch('{branch}')][deepskyblue]branch: {branch}[/][/]"
            )
        # Round counters (dev / test / resolve) — one per started round:
        # dev = user prompts of the dev transcript (the initial assignment,
        # clarifications and tester bug-reports; a continuation of an
        # interrupted attempt adds no prompt, compaction machinery is
        # filtered by all_user_prompts), test/resolve = transcript entries
        # (one appended per fresh round; a network retry or a compaction
        # pause resumes the same round and does not bump the count). Zero
        # values are dropped and the whole fragment hides while every
        # counter is zero — a fresh task shows nothing here. The fragment
        # goes AFTER the branch — the rightmost info fragment of the row
        # (only the ✋/⏸ warning markers may follow it).
        dev_rounds = len(all_user_prompts(task.messages))
        test_rounds = len(task.tester_messages)
        resolve_rounds = len(task.resolver_messages)
        if dev_rounds or test_rounds or resolve_rounds:
            shown = [
                f"{name} {n}"
                for name, n in (
                    ("dev", dev_rounds),
                    ("test", test_rounds),
                    ("resolve", resolve_rounds),
                )
                if n
            ]
            bits.append(f"[dim]rounds: {' · '.join(shown)}[/]")
        if task.needs_human:
            bits.append("[bold red]✋ needs human[/]")
        if task.unfinished:
            bits.append("[yellow]⏸ unfinished[/]")
        return "   ".join(bits)

    @staticmethod
    def _set_extra_row(widget: Static, markup: str) -> None:
        """Redraw the second header row atomically with the first: the
        same update-only-on-change pattern, plus the visibility toggle —
        the row collapses entirely when the task has nothing to show."""
        if widget.has_class("hidden") == bool(markup):
            widget.set_class(not markup, "hidden")
        update_static_if_changed(widget, markup)

    def action_copy_task_id(self, task_id: str) -> None:
        """Click on the viewed task's own #id in the header: copy the short
        id WITH the `#` prefix to the clipboard (a pasted `#<hex>` mention
        renders as a clickable task link — the bare id does not). A
        drag-selection ending on the link is copying text, not navigating
        (same guard as action_open_task)."""
        if self.screen.selections:
            return
        task_id = task_id.lower()
        self._copy_with_fallback(f"#{task_id}")
        self._show_copy_toast(f"#{task_id}")

    def action_copy_task_branch(self, branch: str) -> None:
        """Click on the `branch: drcode/<id>` fragment: copy the branch
        name (for a manual merge / checkout)."""
        if self.screen.selections:
            return
        self._copy_with_fallback(branch)
        self._show_copy_toast(branch)

    @staticmethod
    def _truncate_cells(text: str, max_cells: int) -> str:
        """Plain-text truncation by DISPLAY cells (not characters): cut
        `text` so it fits `max_cells` terminal columns, marking the cut
        with an ellipsis. Markup is never passed here — the caller
        truncates first and escapes afterwards (tags take no columns)."""
        if max_cells <= 0:
            return ""
        if cell_len(text) <= max_cells:
            return text
        # Text.truncate crops IN PLACE and returns None (not self) —
        # mutate a throwaway Text and read the result back.
        cropped = Text(text)
        cropped.truncate(max_cells, overflow="ellipsis")
        return cropped.plain

    def _ctx_counter(self, task: Task) -> tuple[str, str, str] | None:
        """The context-counter fragment (markup, plain text) plus the warn
        class to put on the right widget — or None when the task has no
        context reading yet.

        The counter is clickable: it opens the modal context-layout dialog
        (action_show_ctx_dialog) — the 🧠 figure itself is wrapped in the
        same [@click] markup the task-id mention links use.
        """
        agent, tokens = self._agent_ctx(task)
        if not tokens:
            return None
        role_config = self.resolved_config(self._agent_ctx_role(task, agent))
        max_tokens = role_config.max_context_tokens
        warn_class = ""
        if max_tokens > 0:
            pct = round(tokens * 100 / max_tokens)
            text = (
                f"🧠{self._fmt_ctx_tokens(tokens)}/"
                f"{self._fmt_ctx_tokens(max_tokens)}·{pct}%"
            )
            # Red from the compaction threshold, yellow when the remaining
            # headroom is twice the red-zone headroom: threshold 80 →
            # yellow from 60, threshold 90 → yellow from 80. The threshold
            # follows the shown agent's role's resolved config (a research
            # task's dev transcript uses the research model, a testing
            # round the tester's, a resolver round the resolver's), so a
            # per-model compaction limit from the catalog is reflected
            # here too.
            threshold = min(max(role_config.compaction_threshold, 1), 100)
            warn_pct = 2 * threshold - 100
            if pct >= threshold:
                warn_class = "ctx-crit"
            elif warn_pct > 0 and pct >= warn_pct:
                warn_class = "ctx-warn"
        else:
            text = f"🧠{self._fmt_ctx_tokens(tokens)}"
        return f"[@click=app.show_ctx_dialog]{text}[/]", text, warn_class

    @staticmethod
    def _fmt_ctx_tokens(n: int) -> str:
        if n >= 1000:
            return f"{n / 1000:.1f}".rstrip("0").rstrip(".") + "k"
        return str(n)

    def _agent_ctx(self, task: Task) -> tuple[str, int]:
        """Which agent's context the task's counter must show and its token
        reading: an agent with a run in flight (per-agent live maps) wins,
        otherwise the agent that ran LAST (the shared `last_agent`
        heuristic) measured over its persisted transcript. Shared by the
        status-bar counter and the context dialog — the single definition
        of "active/last agent"."""
        tester_live = self._live_ctx_tester.get(task.id)
        if tester_live is not None:
            return AGENT_TESTER, tester_live
        resolver_live = self._live_ctx_resolver.get(task.id)
        if resolver_live is not None:
            return AGENT_RESOLVER, resolver_live
        dev_live = self._live_ctx_tokens.get(task.id)
        if dev_live is not None:
            return AGENT_DEV, dev_live
        agent = last_agent(task)
        return agent, last_context_tokens(agent_transcript(task, agent)) or 0

    def _agent_ctx_role(self, task: Task, agent: str) -> AgentRole:
        """The role whose resolved config (window / compaction threshold)
        the given agent's figures are measured against."""
        if agent == AGENT_TESTER:
            return AgentRole.TESTER
        if agent == AGENT_RESOLVER:
            return AgentRole.RESOLVER
        return self._task_role(task)

    def _subagent_limit_for(self, role: AgentRole) -> int:
        # Deferred: pydantic_ai (subagents) off the TUI cold start.
        from ..agent import subagent_limit_for

        return subagent_limit_for(self.settings, role)

    def _code_index_tools_for(self, role: AgentRole) -> bool:
        # Deferred: tree-sitter (code_index) off the TUI cold start.
        from ..agent import code_index_tools_for

        return code_index_tools_for(self.settings, role)

    def action_show_ctx_dialog(self) -> None:
        """Click on the context counter: open the modal dialog with the
        ≈ context layout of the viewed task's agents (the development
        transcript on the serving role's model, the latest testing round
        on the tester's, the latest resolver round on the resolver's —
        tabbed), plus the exact API figures and the whole-task cumulative
        usage. The tab of the active/last agent is selected (the same
        selector the counter uses)."""
        task = self.viewed
        if task is None or self._new_task_mode:
            return
        if (
            not task.messages
            and not task.tester_messages
            and not task.resolver_messages
        ):
            self.notify("Task has no context yet — no model request was made.")
            return
        role = self._task_role(task)
        dev, tester, resolver, usage, requests = task_reports(
            task,
            self.resolved_config(role),
            self.resolved_config(AgentRole.TESTER),
            self.resolved_config(AgentRole.RESOLVER),
            role.value.upper(),
            self.project_root,
            live_tokens=self._live_ctx_tokens.get(task.id),
            tester_live_tokens=self._live_ctx_tester.get(task.id),
            resolver_live_tokens=self._live_ctx_resolver.get(task.id),
            tester_allow_commit=self.settings.git_policy.tester.commits,
            tester_allow_push=self.settings.git_policy.tester.pushes,
            research_git=self.settings.git_policy.research,
            subagent_limit=self._subagent_limit_for(role),
            research_web_fetch=web_fetch_policy_for(self.settings, role),
            # Mirror rule (§8.2): the counted tool schemas must match the
            # real wiring — the code-index tools ride the dev, tester and
            # resolver reports when the mode is ON (the helper returns
            # the same project decision for all three non-research
            # roles).
            code_index=self._code_index_tools_for(role),
            # Mirror rule (§8.2): the run_bash network description follows
            # the project's `unrestricted` web-fetch mode.
            network_unrestricted=(
                self.settings.web_fetch_mode is WebFetchMode.UNRESTRICTED
            ),
            # The goals data for the dev report's dynamic-instructions line:
            # the roadmap block needs the active list, the per-task block
            # the goal behind task.goal_id (None when absent/deleted).
            active_goals=self.manager.goals.active(),
            linked_goal=(
                self.manager.goals.goals.get(task.goal_id)
                if task.goal_id is not None
                else None
            ),
        )
        agent, _ = self._agent_ctx(task)
        self.push_screen(
            ContextDialog(
                [
                    (AGENT_DEV, dev),
                    (AGENT_TESTER, tester),
                    (AGENT_RESOLVER, resolver),
                ],
                usage,
                requests,
                default_agent=agent,
            )
        )
