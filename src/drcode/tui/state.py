"""Shared instance state for the :class:`~drcode.tui.app.DrCodeApp` layer mixins.

The coding UI is split into several mixin layers (:mod:`chat_view`,
:mod:`actions`, :mod:`pipeline`) that are combined with the base ``App`` in a
single :class:`~drcode.tui.app.DrCodeApp`.  Every layer references a large set
of shared instance attributes (task manager, settings, live-run bookkeeping,
the chat-log widgets, …) that are only *assigned* inside ``DrCodeApp.__init__``.
Pyright cannot see those assignments from a sibling mixin, so it reports
"attribute unknown" everywhere the layers touch them.

This module declares those attributes once, as class-level annotations, so all
layers can typecheck their ``self`` references.  There is intentionally no
runtime value here — ``DrCodeApp.__init__`` still owns the actual assignment —
so behaviour is unchanged (reading an attribute before ``__init__`` still
raises ``AttributeError`` exactly as before).
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_ai.agent import AgentRun
from pydantic_ai.usage import RunUsage
from rich.console import RenderableType
from textual import work
from textual.app import App
from textual.containers import VerticalScroll
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import ListView, Static

from ..agent import AgentDeps, TesterDeps
from ..config import Config
from ..github import GitHubSyncState, GitHubSyncStateStore
from ..models import Clarification, PendingInput, Task, TaskEvent, TaskKind
from ..notifications import Notifier
from ..providers import AgentRole, ModelRef
from ..settings import (
    CompletedRetention,
    ProjectSettings,
    ProjectSettingsStore,
    StallNudge,
    TaskFinalization,
    TaskIsolation,
)
from ..task_env import TaskEnvironment
from ..task_manager import TaskManager
from ..workflow import WorkflowHooks
from .blocks import ToolCallBlock
from .github_dialog import GitHubSetup
from .header import ModelStatusHeader
from .inputs import ChatInput, NewTaskInput
from .setup_dialog import ModelSetup
from .sidebar import Sidebar
from .todo_panel import TodoPanel
from .usage import SessionTokenUsage


class DrCodeState(App[None]):
    def run_agent(self, task_id: str) -> None: ...

    async def _run_agent_worker(self, task_id: str) -> None: ...

    @property
    def agent_running(self) -> bool: ...

    @property
    def parallel_limit(self) -> int: ...

    async def _check_model(self) -> None: ...
    def _git_branch(self) -> str | None: ...
    def _display_dir(self) -> str: ...

    def refresh_project_info(self) -> None: ...

    def _sidebar_fingerprint(self) -> tuple: ...

    def _poll_sidebar(self) -> None: ...

    def _sidebar_sections(self) -> list[tuple[str, list[Task]]]: ...

    def _sidebar_top_task(self) -> Task | None: ...

    def refresh_sidebar(self) -> None: ...

    def _todo_fingerprint(self) -> tuple: ...

    def refresh_todo_panel(self) -> None: ...

    @staticmethod
    def _todo_instructions(task: Task) -> str | None: ...

    def _todo_completion_warning(self, task: Task) -> None: ...

    def toggle_sidebar_section(self, section: str) -> None: ...

    def _refocus_sidebar_task(self, task_id: str) -> None: ...

    def _recover_from_outage(self) -> None: ...

    def refresh_connection(self) -> None: ...

    def _render_model_status(self) -> None: ...

    @property
    def header_conn_info(self) -> Static: ...

    @property
    def header_session_info(self) -> Static: ...

    @staticmethod
    def _fmt_duration(seconds: float) -> str: ...

    def _collect_session_stats(self) -> None: ...

    def _header_conn_text(self) -> str: ...

    def _header_stats_text(self) -> str: ...

    def _header_github_text(self) -> str: ...

    @staticmethod
    def _fmt_tokens(n: int) -> str: ...

    def _record_session_usage(self, usage: RunUsage | None) -> None: ...

    def _header_usage_text(self) -> str: ...

    def _render_header_tall(self) -> None: ...

    def on_model_status_header_tall_toggled(
        self, event: ModelStatusHeader.TallToggled
    ) -> None: ...

    def _touch_activity(
        self, task_id: str, tool: str | None = None, *, done: bool = False
    ) -> None: ...

    @staticmethod
    def _fmt_age(seconds: int) -> str: ...

    def _poll_activity(self) -> None: ...

    def _render_activity(
        self, status: str, detail: str, state: str | None = None
    ) -> None: ...

    def _poll_stall(self) -> None: ...

    def _nudge_run(self, task: Task) -> None: ...

    def _log_event(self, task: Task, text: str, *, kind: str = "system") -> None: ...

    @staticmethod
    def _format_event(event: TaskEvent) -> str: ...

    def _show_event(self, task: Task, text: str, *, kind: str = "system") -> None: ...

    def view_task(self, task_id: str) -> None: ...

    def _try_delta_view(self, task_id: str) -> bool: ...

    def _invalidate_chat_delta(self) -> None: ...

    def _render_task_history(
        self, task: Task, task_id: str
    ) -> list[tuple[datetime, int, str, str]]: ...

    def _chat_at_bottom(self) -> bool: ...

    def _stream_follow(self) -> None: ...

    def _tool_call_started(
        self,
        run_key: tuple[str, str, int],
        task_id: str,
        call_id: str,
        call_text: str,
        block: ToolCallBlock,
    ) -> None: ...

    def _tool_call_finished(self, call_id: str) -> None: ...

    def _stop_run_tools(self, run_key: tuple[str, str, int]) -> None: ...

    def _reattach_tool_spinner(
        self, block: ToolCallBlock, call_text: str, task_id: str
    ) -> None: ...

    def _write_chat(
        self, content: RenderableType | Widget, *, expand: bool = False
    ) -> None: ...

    def _flush_chat_batch(self) -> None: ...

    def _drain_chat_batch(self) -> None: ...

    def _write_history_line(
        self, role: str, text: str, at: datetime | None = None
    ) -> Widget | None: ...

    def _write_clarification(self, clarification: Clarification) -> None: ...

    def refresh_status(self) -> None: ...

    def refresh_task_status(self) -> None: ...

    @staticmethod
    def _fmt_ctx_tokens(n: int) -> str: ...

    def _update_ctx_counter(self, right: Static, task: Task) -> None: ...

    def action_show_ctx_dialog(self) -> None: ...

    def _model_setup_done(self, setup: ModelSetup | None) -> None: ...

    # -- per-role model resolution (implemented in DrCodeApp / ActionsLayer) --

    def resolved_config(self, role: AgentRole) -> Config: ...

    def invalidate_resolved_configs(self) -> None: ...

    def _coding_config(self) -> Config: ...

    @staticmethod
    def _task_role(task: Task) -> AgentRole: ...

    def role_model_label(self, role: AgentRole) -> str: ...

    def action_open_model_catalog(self) -> None: ...

    def action_change_coding_model(self) -> None: ...

    def action_change_research_model(self) -> None: ...

    def action_change_tester_model(self) -> None: ...

    def action_change_compaction_model(self) -> None: ...

    def action_set_role_model(self, role: AgentRole, ref: ModelRef | None) -> None: ...

    def action_activate_role_custom(self, role: AgentRole) -> None: ...

    def action_custom_role_model(self, role: AgentRole) -> None: ...

    def _maybe_offer_agents_md(self) -> None: ...

    def _agents_md_answered(self, create: bool | None) -> None: ...

    @property
    def new_task_input(self) -> NewTaskInput: ...

    def _clear_stream_label(self) -> None: ...

    def _drop_stream_preview(self, task_id: str, generation: int) -> None: ...

    def action_new_task(self) -> None: ...

    def _new_task_chrome(self) -> None: ...

    def action_edit_task(self) -> None: ...

    def _leave_new_task_mode(self) -> None: ...

    def cancel_new_task(self) -> None: ...

    def toggle_new_task_kind(self) -> None: ...

    def _update_kind_label(self) -> None: ...

    def _update_kind_descriptions(self) -> None: ...

    def submit_new_task(self) -> None: ...

    def action_submit_new_task_hold(self) -> None: ...

    def action_save_task_edit_hold(self) -> None: ...

    def _submit_new_form(self, *, hold: bool) -> None: ...

    def _finish_task_edit(self, *, queue: bool) -> None: ...

    def action_done_task(self) -> None: ...

    def action_cancel_task(self) -> None: ...

    def action_abort_run(self) -> None: ...

    def action_nudge_run(self) -> None: ...

    def action_cycle_task_kind(self) -> None: ...

    def action_move_task_up(self) -> None: ...

    def action_move_task_down(self) -> None: ...

    def _move_viewed_task(self, *, up: bool) -> None: ...

    def action_hold_task(self) -> None: ...

    def action_resume_task(self) -> None: ...

    def action_delete_task(self) -> None: ...

    def action_focus_sidebar(self) -> None: ...

    def action_focus_input(self) -> None: ...

    def action_settings(self) -> None: ...

    def action_set_completed_retention(self, retention: CompletedRetention) -> None: ...

    def action_change_completed_retention(self) -> None: ...

    def action_set_task_finalization(self, finalization: TaskFinalization) -> None: ...

    def action_change_task_finalization(self) -> None: ...

    def action_set_max_dev_rounds(self, rounds: int) -> None: ...

    def action_change_max_dev_rounds(self) -> None: ...

    def action_set_stall_nudge(self, nudge: StallNudge) -> None: ...

    def action_change_stall_nudge(self) -> None: ...

    def action_set_task_isolation(self, isolation: TaskIsolation) -> None: ...

    def action_change_task_isolation(self) -> None: ...

    def action_set_parallel_tasks(self, count: int) -> None: ...

    def action_change_parallel_tasks(self) -> None: ...

    def _task_isolation_label(self) -> str: ...

    def workflow_status_label(self) -> str: ...

    def action_workflow_info(self) -> None: ...

    def action_change_github(self) -> None: ...

    def _github_settings_done(self, setup: GitHubSetup | None) -> None: ...

    def _poll_github_sync(self) -> None: ...

    def action_update_agents_md(self) -> None: ...

    def _create_agents_task(self) -> None: ...

    def action_change_theme(self) -> None: ...

    def on_list_view_selected(self, event: ListView.Selected) -> None: ...

    def submit_chat_input(self) -> None: ...

    def _submit_user_text(self, task: Task, text: str) -> None: ...

    def action_pick_option(self, index: int) -> None: ...

    def _record_stream_tokens(self, task_id: str, text: str) -> None: ...

    def _prune_tps_samples(self, now: float) -> None: ...

    def _tokens_per_second(self) -> float | None: ...

    def _next_testing(self) -> Task | None: ...

    def _advance(self) -> None: ...

    def _promote_failed_queue(self, blocked: set[str] | None = None) -> Task | None: ...

    def _poll_dependency_unblock(self) -> None: ...

    def _dispatch_promotions(self, blocked: set[str] | None) -> bool: ...

    @work(exclusive=True, group="promote")
    async def _promote_with_dependencies(self) -> None: ...

    def _dep_gate_sync(self, dep_ids: list[str]) -> tuple[set[str], str]: ...

    def _announce_dependency_gate(
        self, queued: list[Task], gate: dict[str, dict[str, str]], base: str
    ) -> None: ...

    def _persist_task_messages(
        self, task: Task, run: AgentRun[AgentDeps, str] | None
    ) -> None: ...

    def _persist_tester_messages(
        self, task: Task, run: AgentRun[TesterDeps, str] | None
    ) -> None: ...

    def _fill_over_threshold(
        self,
        live_tokens: int | None,
        messages: list[dict[str, Any]],
    ) -> int | None: ...

    def _compaction_due(self, task: Task) -> int | None: ...

    def _schedule_tester_retry(self, task_id: str) -> None: ...

    # -- external review / workflow hooks (implemented in PipelineLayer) ----

    def _next_review(self) -> Task | None: ...

    async def _run_task_start_hook(self, task: Task) -> str | None: ...

    async def _run_development_done_hook(self, task: Task) -> None: ...

    async def _run_accepted_hook(self, task: Task) -> None: ...

    async def _run_research_accepted_hook(self, task: Task) -> None: ...

    async def _anchor_research_snapshot(self, task: Task) -> None: ...

    async def _finalize_research_completion(self, task: Task) -> None: ...

    def _apply_review_accept(self, task: Task) -> None: ...

    @work(exclusive=True, group="review")
    async def run_external_review(self, task_id: str) -> None: ...

    def _poll_workflow_review(self) -> None: ...

    # DrCodeApp.__init__ (see the region that builds self.manager, self.config,
    # the live-run bookkeeping dicts, the session stats, …). Declaring the
    # attributes here is what lets the chat/actions/pipeline mixins reference
    # them without pyright complaining.

    # -- core services ------------------------------------------------------
    project_root: Path
    config: Config
    manager: TaskManager
    settings_store: ProjectSettingsStore
    settings: ProjectSettings
    # Per-task worktree isolation (project setting `task_isolation`); the
    # optional root the agent/tester runs in comes from here.
    task_env: TaskEnvironment
    # In-flight worktree snapshot jobs (see _snapshot_in_background) — the
    # research finalization awaits them before running its hook.
    _snapshot_jobs: set[asyncio.Task[None]]
    # Custom team workflow (.drcode/workflow.py hooks); None = absent.
    workflow: WorkflowHooks | None

    # -- GitHub Issues sync bookkeeping -------------------------------------
    _github_state_store: GitHubSyncStateStore
    _github_state: GitHubSyncState
    _github_next_sync: float
    _github_syncing: bool
    _github_last_error: str | None
    _purged_tasks: list[Task]

    # -- ntfy.sh notifications ------------------------------------------------
    notifier: Notifier
    _notifications_primed: bool

    # -- run state ----------------------------------------------------------
    viewed_id: str | None
    # Counted run state (parallel development slots): task id -> the
    # generation of its in-flight agent run. `agent_running` above is a
    # property over `_agent_runs`; the tester stays single (a plain flag).
    _agent_runs: dict[str, int]
    tester_running: bool
    _tester_task_id: str | None
    new_task_kind: TaskKind
    # Per-task run generation (superseded-run bookkeeping): a run's
    # teardown may only touch shared state while it is still the task's
    # newest run.
    _agent_generations: dict[str, int]
    _pending_rendered: list[str]
    # Prompts a run has already consumed (popped from pending_input) but
    # that are not persisted into the transcript yet: the first node
    # boundary is what saves them. Until then a view_task() re-render finds
    # neither copy and would wipe the prompt from the chat — the record
    # lets _render_task_history draw it anyway. See run_agent's consumption.
    _consumed_prompts: dict[str, tuple[str, list[PendingInput]]]
    _chat_has_content: bool

    # -- copy toast ---------------------------------------------------------
    _copy_toast_timer: Timer | None

    # -- tester failure / retry bookkeeping ---------------------------------
    _tester_fail_count: dict[str, int]
    _tester_retry_at: dict[str, float]
    _tester_net_count: dict[str, int]
    _tester_resume_round: set[str]

    # -- agent failure / network-retry bookkeeping --------------------------
    _agent_failed: set[str]
    _agent_net_count: dict[str, int]
    _agent_net_retry_at: dict[str, float]
    _net_retries_total: int
    # Failed-queue deadlock escape (see PipelineLayer._promote_failed_queue):
    # how many rescue rounds have fired this session and the earliest
    # monotonic time the next one may fire at.
    _failed_queue_rounds: int
    _failed_queue_retry_at: float

    # -- dependency gate (depends_on promotion condition) ---------------------
    # Last computed gate of the promote worker: viewed task id -> unsatisfied
    # dependency id -> "not completed" | "unmerged". Drives the sidebar lock
    # markers, the task-statusbar dependency list and the settled-pipeline
    # notification. Session-only; empty until the first gated promotion.
    _dep_gate: dict[str, dict[str, str]]
    # Tasks that already received their "Task waits: …" service event and
    # have not been announced as eligible again — dedups the gate's events.
    _dep_block_notified: set[str]

    # -- external review (workflow hooks) bookkeeping ------------------------
    _review_running: bool
    _review_task_id: str | None
    _review_retry_at: dict[str, float]
    _review_fail_count: dict[str, int]
    _review_poll_at: float
    _dev_done_notes: dict[str, str]
    # Serializes the accept hooks (on_accepted / on_research_accepted):
    # two tasks finalizing in parallel must not run concurrent git merges
    # into the shared main checkout.
    _accept_hook_lock: asyncio.Lock

    # -- session statistics (expanded header, session-scoped) ---------------
    _session_known_ids: set[str]
    _session_completed_ids: set[str]
    _session_created: dict[TaskKind, int]
    _session_completed: int
    _session_usage: SessionTokenUsage
    _session_durations: list[float]

    # -- new-task form ------------------------------------------------------
    _new_task_mode: bool
    _new_task_prev_id: str | None
    _editing_task_id: str | None

    # -- sidebar / activity -------------------------------------------------
    _sidebar_snapshot: tuple
    # Todo-panel snapshot of the last render (see _todo_fingerprint).
    _todo_snapshot: tuple
    _collapsed_sections: set[str]
    # Per-run activity bookkeeping (parallel slots): last sign of life,
    # the tool each run is waiting on and the tok/s sample window.
    _last_activity: dict[str, float]
    _current_tool: dict[str, str | None]

    # -- live stream / connection -------------------------------------------
    _tps_samples: dict[str, deque[tuple[float, int]]]
    _conn_label: str
    _last_probe_ms: int | None
    _probe_fail_streak: int
    _active_tools: dict[tuple[str, str, int], dict[str, tuple[str, str, ToolCallBlock]]]
    _tester_generation: int

    # -- context fill / compaction ------------------------------------------
    _live_ctx_tokens: dict[str, int]
    _compaction_failed: set[str]
    _last_follow: float
    _chat_epoch: int

    # -- chat batch drain ---------------------------------------------------
    _chat_batching: bool
    _chat_batch: list[Widget]
    _rendered_clarification: Clarification | None

    # -- delta re-render bookkeeping ----------------------------------------
    # When the open chat is a canonical history render (no live/streamed
    # extras mounted since), _rendered_count is the number of timeline
    # entries already mounted and _rendered_prefix their (timestamp, role)
    # fingerprint — a view_task() of the SAME task then appends only the
    # missing tail instead of wiping and remounting the whole history. Any
    # live _write_chat() mount or log wipe invalidates both (None/()).
    _rendered_count: int | None
    _rendered_prefix: tuple[tuple[str, str], ...]

    # -- widget getters (defined in the DrCodeApp skeleton) -----------------
    @property
    def chat_log(self) -> VerticalScroll:
        return self.query_one("#chat-log", VerticalScroll)

    @property
    def chat_input(self) -> ChatInput: ...

    @property
    def sidebar(self) -> Sidebar: ...

    @property
    def todo_panel(self) -> TodoPanel: ...

    @property
    def project_info(self) -> Static: ...

    @property
    def model_status(self) -> Static: ...

    @property
    def activity_status(self) -> Static: ...

    @property
    def viewed(self) -> Task | None: ...
