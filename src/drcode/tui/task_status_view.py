"""Task-status layer of the chat view mixin stack.

The bottom statusbar (active task, queue length, agent/tester state,
model) and the task statusbar under the chat log (kind, id, title,
creation date, live duration, dependency gate, context-fill counter
with its modal dialog). Split out of :mod:`drcode.tui.chat_view` — see
the composition in ``ChatViewLayer``.
"""

from __future__ import annotations

from rich.markup import escape
from textual.css.query import NoMatches
from textual.widgets import Static

from ..agent import last_context_tokens
from ..context_layout import task_reports
from ..models import Task, TaskKind, TaskStatus, VerificationStatus, _now
from ..providers import AgentRole
from .ctx_dialog import ContextDialog
from .state import DrCodeState


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
        )
        self.refresh_task_status()

    def refresh_task_status(self) -> None:
        # The footer mirrors the state check_action() gates on (viewed task,
        # its status, running flags): re-evaluate the visible bindings.
        # Guarded like refresh_status: worker finally-blocks can call this
        # while the screen is already dismantled.
        self.refresh_bindings()
        try:
            left = self.query_one("#task-status-left", Static)
            right = self.query_one("#task-status-right", Static)
        except NoMatches:
            return
        # The new-task form is not tied to any task: keep the bar empty so
        # the previously viewed task doesn't keep updating behind the form.
        if self._new_task_mode:
            left.update("")
            right.update("")
            return
        task = self.viewed
        if task is None:
            left.update("")
            right.update("")
            return
        created = task.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
        kind = {
            TaskKind.BUG: "[bold red]bug[/]",
            TaskKind.RESEARCH: "[bold yellow]research[/]",
        }.get(task.kind, "[bold green]feature[/]")
        status_colors = {
            TaskStatus.QUEUED: "yellow",
            TaskStatus.HOLD: "blue",
            TaskStatus.ACTIVE: "cyan",
            TaskStatus.TESTING: "magenta",
            TaskStatus.COMPLETED: "green",
            TaskStatus.AWAITING_CLARIFICATION: "yellow",
        }
        status_color = status_colors[task.status]
        status = f"[{status_color}]{task.status.value}[/]"
        # Duration counter: only for the statuses where the time spent is
        # meaningful and live — the developer working (ACTIVE) and the
        # tester verifying (TESTING). Stamped at every status transition
        # (fire_status_change), so this measures the current continuous
        # stay, never a cumulative one; survives restarts via the stamp.
        # Rendered in the status's own color, one space after the status
        # (the other fragments stay separated by the wide join).
        if (
            task.status in (TaskStatus.ACTIVE, TaskStatus.TESTING)
            and task.status_changed_at is not None
        ):
            seconds = max(0, int((_now() - task.status_changed_at).total_seconds()))
            status += f" [{status_color}]⏱ {self._fmt_age(seconds)}[/]"
        parts = [
            kind,
            f"#{task.id} {escape(task.title_display)}",
            f"[dim]{created}[/]",
            status,
        ]
        deps_part = self._dependency_status_part(task)
        if deps_part:
            parts.append(deps_part)
        if task.verification is VerificationStatus.PASSED:
            parts.append("[green]✓ testing[/]")
        elif task.verification is VerificationStatus.FAILED:
            parts.append("[red]✗ testing[/]")
        left.update("   ".join(parts))
        self._update_ctx_counter(right, task)

    def _dependency_status_part(self, task: Task) -> str:
        """Task-statusbar fragment listing the task's dependencies with
        their per-dependency gate state:

        - `#id ✓` — the dependency is COMPLETED;
        - `#id …` — the dependency task exists but is not completed yet;
        - `#id ⇴` — the promote worker's last gate run found unmerged
          commits on the dependency's `drcode/<id>` branch (also for a
          dep whose task file is already gone — the branch outlives it);
        - `#id ?` — an id that is not in the manager and has no branch
          (a stale entry; the gate treats it as satisfied).

        Empty string when the task has no dependencies."""
        if not task.depends_on:
            return ""
        gate = self._dep_gate.get(task.id, {})
        bits: list[str] = []
        for dep in task.depends_on:
            # Each #id marker is itself a link to the dependency task (ids
            # are lowercase hex — no escaping needed inside the markup tag).
            link = f"[@click=app.open_task('{dep.lower()}')]#{dep.lower()}[/]"
            dep_task = self.manager.tasks.get(dep)
            if gate.get(dep) == "unmerged":
                bits.append(f"[red]{link}⇴[/]")
            elif dep_task is not None and dep_task.status is TaskStatus.COMPLETED:
                bits.append(f"[green]{link}✓[/]")
            elif dep_task is not None:
                bits.append(f"[yellow]{link}…[/]")
            else:
                bits.append(f"[dim]{link}?[/]")
        return "deps: " + " ".join(bits)

    @staticmethod
    def _fmt_ctx_tokens(n: int) -> str:
        if n >= 1000:
            return f"{n / 1000:.1f}".rstrip("0").rstrip(".") + "k"
        return str(n)

    def _update_ctx_counter(self, right: Static, task: Task) -> None:
        """Right corner of the task status bar: current context fill in
        tokens and percent of the allowed maximum.

        Source of truth: the latest model request usage. For a run in
        progress — the live value; otherwise (finished task, history view)
        — the value computed from the persisted transcript, i.e. the final
        value is saved in the task history together with the per-response
        usage and survives restarts.

        The counter is clickable: it opens the modal context-layout dialog
        (action_show_ctx_dialog) — the 🧠 figure itself is wrapped in the
        same [@click] markup the task-id mention links use.
        """
        right.remove_class("ctx-warn", "ctx-crit")
        tokens = self._live_ctx_tokens.get(task.id)
        if tokens is None:
            tokens = last_context_tokens(task.messages)
        if not tokens:
            right.update("")
            right.tooltip = None
            return
        right.tooltip = "Click: context layout (≈ breakdown + exact usage)"
        max_tokens = self.resolved_config(self._task_role(task)).max_context_tokens
        if max_tokens > 0:
            pct = round(tokens * 100 / max_tokens)
            text = (
                f"🧠 {self._fmt_ctx_tokens(tokens)}/"
                f"{self._fmt_ctx_tokens(max_tokens)}·{pct}%"
            )
            # Red from the compaction threshold, yellow when the remaining
            # headroom is twice the red-zone headroom: threshold 80 →
            # yellow from 60, threshold 90 → yellow from 80. The threshold
            # follows the task's serving role's resolved config (research
            # tasks use the research model), so a per-model compaction
            # limit from the catalog is reflected here too.
            role_config = self.resolved_config(self._task_role(task))
            threshold = min(max(role_config.compaction_threshold, 1), 100)
            warn_pct = 2 * threshold - 100
            if pct >= threshold:
                right.add_class("ctx-crit")
            elif warn_pct > 0 and pct >= warn_pct:
                right.add_class("ctx-warn")
        else:
            text = f"🧠 {self._fmt_ctx_tokens(tokens)}"
        right.update(f"[@click=app.show_ctx_dialog]{text}[/]")

    def action_show_ctx_dialog(self) -> None:
        """Click on the context counter: open the modal dialog with the
        ≈ context layout of the viewed task (development transcript on the
        serving role's model, the latest testing round on the tester's,
        plus the exact API figures and the whole-task cumulative usage)."""
        task = self.viewed
        if task is None or self._new_task_mode:
            return
        if not task.messages and not task.tester_messages:
            self.notify("Task has no context yet — no model request was made.")
            return
        role = self._task_role(task)
        dev, tester, usage, requests = task_reports(
            task,
            self.resolved_config(role),
            self.resolved_config(AgentRole.TESTER),
            role.value.upper(),
            self.project_root,
            live_tokens=self._live_ctx_tokens.get(task.id),
            tester_allow_commit=self.settings.git_policy.tester.commits,
            tester_allow_push=self.settings.git_policy.tester.pushes,
            research_git=self.settings.git_policy.research,
        )
        self.push_screen(
            ContextDialog([r for r in (dev, tester) if r is not None], usage, requests)
        )
