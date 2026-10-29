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
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from pydantic_ai.agent import AgentRun
from pydantic_ai.usage import RunUsage
from rich.console import RenderableType
from textual import work
from textual.app import App
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import ListView, Static
from textual.worker import Worker

from ..agent import AgentDeps, ResolverDeps, TesterDeps
from ..config import Config
from ..github import GitHubSyncState, GitHubSyncStateStore
from ..keepawake import KeepAwake
from ..models import Clarification, PendingInput, Task, TaskEvent, TaskKind
from ..notifications import Notifier
from ..providers import AgentRole, ModelRef
from ..settings import (
    CompletedRetention,
    GitPolicy,
    ProjectSettings,
    ProjectSettingsStore,
    StallNudge,
    TaskIsolation,
    Verification,
    WorktreeRemoval,
)
from ..task_env import TaskEnvironment
from ..task_manager import TaskManager
from ..workflow import WorkflowHooks
from .blocks import AssistantBlock, ChatLog, ReasoningBlock, ToolCallBlock
from .command_providers import WorkflowPreset
from .github_dialog import GitHubSetup
from .header import ModelStatusHeader
from .inputs import ChatInput, NewTaskInput
from .setup_dialog import ModelSetup
from .sidebar import Sidebar, TaskItem
from .todo_panel import TodoPanel
from .usage import SessionTokenUsage
from .voice_input import VoiceInputController

# A live-tail ledger entry (see chat_view._write_chat's `note`): (role,
# text, block). `block` is set only for streamed ReasoningBlock/
# AssistantBlock mounts — their final text lives in the widget's `_content`
# (still partial at mount time); every other entry is compared by the exact
# `text` written.
LiveTailEntry = tuple[str, str, ReasoningBlock | AssistantBlock | None]


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

    def action_toggle_todo_panel(self) -> None: ...

    @staticmethod
    def _todo_instructions(task: Task) -> str | None: ...

    def _todo_completion_warning(self, task: Task) -> None: ...

    def toggle_sidebar_section(self, section: str) -> None: ...

    def _refocus_sidebar_task(self, task_id: str) -> None: ...

    def _sync_sidebar_cursor(self, task_id: str | None) -> None: ...

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

    def _unmerged_completed_ids(self) -> list[str]: ...

    def _suggest_conflict_screen(self, task: Task) -> None: ...

    async def _poll_git_indicator(self) -> None: ...

    def _ensure_unmerged_tracking(self) -> None: ...

    def _poll_stall(self) -> None: ...

    def _nudge_run(self, task: Task) -> None: ...

    def _schedule_pipeline_timer(
        self, delay: float, callback: Callable[[], Any]
    ) -> None: ...

    def _log_event(self, task: Task, text: str, *, kind: str = "system") -> None: ...

    @staticmethod
    def _format_event(event: TaskEvent) -> str: ...

    def _show_event(self, task: Task, text: str, *, kind: str = "system") -> None: ...

    def _isolation_pin_note(self, task: Task, *, save: bool) -> None: ...

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
        self,
        content: RenderableType | Widget,
        *,
        expand: bool = False,
        note: LiveTailEntry | None = None,
    ) -> None: ...

    def _flush_chat_batch(self) -> None: ...

    def _drain_chat_batch(self) -> None: ...

    def _poll_chat_drain(self) -> None: ...

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

    def action_change_resolver_model(self) -> None: ...

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

    def action_search_tasks(self) -> None: ...

    def action_settings(self) -> None: ...

    def action_set_completed_retention(self, retention: CompletedRetention) -> None: ...

    def action_change_completed_retention(self) -> None: ...

    def action_set_verification(self, verification: Verification) -> None: ...

    def action_change_verification(self) -> None: ...

    def action_set_git_tester(self, policy: GitPolicy) -> None: ...

    def action_change_git_tester(self) -> None: ...

    def action_set_git_research(self, policy: GitPolicy) -> None: ...

    def action_change_git_research(self) -> None: ...

    def action_set_workflow_preset(self, preset: WorkflowPreset) -> None: ...

    def action_workflow_preset_custom(self) -> None: ...

    def action_change_workflow_preset(self) -> None: ...

    def workflow_preset_label(self) -> str: ...

    def action_set_worktree_removal(self, removal: WorktreeRemoval) -> None: ...

    def action_change_worktree_removal(self) -> None: ...

    def action_set_max_dev_rounds(self, rounds: int) -> None: ...

    def action_change_max_dev_rounds(self) -> None: ...

    def action_set_stall_nudge(self, nudge: StallNudge) -> None: ...

    def action_change_stall_nudge(self) -> None: ...

    def action_set_task_isolation(self, isolation: TaskIsolation) -> None: ...

    def action_change_task_isolation(self) -> None: ...

    def action_set_isolation_start_branch(self, branch: str) -> None: ...

    def action_change_isolation_start_branch(self) -> None: ...

    def _isolation_start_branch_label(self) -> str: ...

    def action_set_parallel_tasks(self, count: int) -> None: ...

    def action_change_parallel_tasks(self) -> None: ...

    def action_set_subagents_coder(self, count: int) -> None: ...

    def action_change_subagents_coder(self) -> None: ...

    def action_set_subagents_researcher(self, count: int) -> None: ...

    def action_change_subagents_researcher(self) -> None: ...

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

    def _note_network_recovered(self, task: Task, counts: dict[str, int]) -> None: ...

    def _outage_active(self) -> bool: ...

    def _outage_retry_tick(self, task_id: str, role: AgentRole) -> None: ...

    def _next_testing(self) -> Task | None: ...

    def _advance(self) -> None: ...

    def _update_keepawake(self) -> None: ...

    def _promote_failed_queue(self, blocked: set[str] | None = None) -> Task | None: ...

    def _poll_dependency_unblock(self) -> None: ...

    def _dispatch_promotions(self, blocked: set[str] | None) -> bool: ...

    def _notify_pipeline_settled(self) -> None: ...

    @work(exclusive=True, group="promote")
    async def _promote_with_dependencies(self) -> None: ...

    def _dep_gate_sync(self, dep_ids: list[str]) -> tuple[set[str], str]: ...

    def _announce_dependency_gate(
        self, queued: list[Task], gate: dict[str, dict[str, str]], base: str
    ) -> None: ...

    def _persist_task_messages(
        self, task: Task, run: AgentRun[AgentDeps, str] | None
    ) -> None: ...

    def _flush_in_flight_prompt(self, task: Task) -> None: ...

    def _persist_tester_messages(
        self, task: Task, run: AgentRun[TesterDeps, str] | None
    ) -> None: ...

    def _fill_over_threshold(
        self,
        live_tokens: int | None,
        messages: list[dict[str, Any]],
        role: AgentRole = AgentRole.CODING,
    ) -> int | None: ...

    def _fill_over_hard_limit(
        self,
        live_tokens: int | None,
        messages: list[dict[str, Any]],
        role: AgentRole = AgentRole.CODING,
    ) -> int | None: ...

    def _compaction_due(self, task: Task) -> int | None: ...

    def _hard_limit_due(self, task: Task) -> bool: ...

    def _threshold_observation(
        self,
        task: Task,
        tokens: int | None,
        counter: dict[str, int],
    ) -> None: ...

    async def _maybe_compact(self, task: Task) -> None: ...

    async def _maybe_compact_tester_round(
        self, task: Task, live_tokens: int | None
    ) -> bool: ...

    def _schedule_tester_retry(self, task_id: str) -> None: ...

    # The real run_tester is a @work-decorated async worker; the stub only
    # fixes the decorated signature (Worker[None]) for the mixin layers.
    def run_tester(self, task_id: str) -> Worker[None]: ...

    def _maybe_start_resolver(self, task_id: str) -> None: ...

    def run_resolver(self, task_id: str) -> Worker[None]: ...

    def _retry_resolver(self) -> None: ...

    async def _verify_resolution(
        self,
        task: Task,
        root: Path,
        report: Any,
        deps: ResolverDeps,
    ) -> Any: ...

    async def _apply_resolver_outcome(
        self, task: Task, root: Path, base: str, outcome: Any
    ) -> None: ...

    async def _give_up_to_human(self, task: Task, root: Path, base: str) -> None: ...

    def _persist_resolver_messages(
        self, task: Task, run: AgentRun[Any, str] | None
    ) -> None: ...

    async def _maybe_compact_resolver_round(
        self, task: Task, live_tokens: int | None
    ) -> bool: ...

    # -- external review / workflow hooks (implemented in PipelineLayer) ----

    def _next_review(self) -> Task | None: ...

    def _hook_env(self, task: Task) -> tuple[Path, dict[str, str] | None]: ...

    async def _run_task_start_hook(self, task: Task) -> str | None: ...

    async def _run_development_done_hook(self, task: Task) -> None: ...

    async def _run_accepted_hook(self, task: Task) -> None: ...

    async def _run_research_accepted_hook(self, task: Task) -> None: ...

    async def _anchor_research_snapshot(self, task: Task) -> None: ...

    async def _finalize_research_completion(self, task: Task) -> None: ...

    async def _run_tester_passed_hook(self, task: Task) -> None: ...

    async def _remind_unmerged_work(self, task: Task) -> None: ...

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
    # Idempotence guard for _ensure_unmerged_tracking (assigned in __init__).
    _unmerged_tracking_registered: bool
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

    # -- keep-awake (system-sleep assertion while work is in flight) ----------
    # Created in DrCodeApp.__init__; the busy detection lives in the
    # pipeline layer (_keepawake_busy/_update_keepawake).
    keepawake: KeepAwake

    # -- run state ----------------------------------------------------------
    viewed_id: str | None
    # Counted run state (parallel development slots): task id -> the
    # generation of its in-flight agent run. `agent_running` above is a
    # property over `_agent_runs`; the tester stays single (a plain flag).
    _agent_runs: dict[str, int]
    tester_running: bool
    _tester_task_id: str | None
    # Armed-ESC confirmation (project setting `esc_abort`, default
    # double_press): the run-state signature the arming saw (None = not
    # armed), when it was armed, the last accepted ESC press time (the
    # debounce base) and the window-expiry timer.
    _esc_armed_signature: tuple | None
    _esc_armed_at: float
    _esc_last_press: float
    _esc_disarm_timer: Timer | None
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

    # -- merge-conflict resolver bookkeeping (single-slot worker) -----------
    _resolver_running: bool
    _resolver_task_id: str | None
    _resolver_generation: int
    _resolver_fail_count: dict[str, int]
    _resolver_retry_at: dict[str, float]
    _resolver_net_count: dict[str, int]
    _resolver_resume_round: set[str]
    # Compaction-failure bookkeeping of the resolver rounds — separate
    # from the agent/tester sets so a broken summarizer seen by one role
    # never disables another role's gate.
    _resolver_compaction_failed: set[str]
    _resolver_hard_limit_exhausted: set[str]
    # Consecutive hard-limit compactions of the resolver round (livelock
    # guard, see _livelock_stop).
    _resolver_hard_limit_compactions: dict[str, int]
    # Same counter at the threshold level (see _livelock_stop).
    _resolver_threshold_compactions: dict[str, int]

    # -- agent failure / network-retry bookkeeping --------------------------
    _agent_failed: set[str]
    # Deterministically-failed subset of _agent_failed (transcript over the
    # model's context window): the rescue escape never lifts it.
    _agent_dead: set[str]
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

    # -- deferred installs (the on_tester_passed MERGE_HEAD guard) -----------
    # Ids collected by _install_rerun_candidates (a worker thread inside
    # _unmerged_completed_ids): deferred on_tester_passed installs
    # (task.install_pending) whose blocker is gone — handed to
    # _maybe_rerun_pending_installs by the 10 s poll / the startup scan.
    _install_rerun_ids: list[str]
    _install_rerun_running: bool

    def _install_rerun_candidates(self, unmerged_ids: list[str]) -> list[str]: ...

    def _maybe_rerun_pending_installs(self) -> Any: ...

    async def _poll_pending_installs(self) -> None: ...

    # -- session statistics (expanded header, session-scoped) ---------------
    _session_known_ids: set[str]
    _session_completed_ids: set[str]
    _session_created: dict[TaskKind, int]
    _session_completed: int
    _session_usage: SessionTokenUsage
    _session_durations: list[float]

    # -- new-task form ------------------------------------------------------
    # Tab-cycle order of the task kinds (TaskFormLayer; the lifecycle
    # layer's kind cycle for existing tasks reads the same order).
    _KIND_CYCLE: ClassVar[tuple[TaskKind, ...]]
    _new_task_mode: bool
    _new_task_prev_id: str | None
    _editing_task_id: str | None

    # -- push-to-talk dictation (Ctrl+B; created in DrCodeApp.__init__) ------
    voice_input: VoiceInputController

    # -- sidebar / activity -------------------------------------------------
    _sidebar_snapshot: tuple
    # Mounted sidebar rows by key ("header", section) / ("task", id) —
    # reused by the incremental refresh instead of a full rebuild.
    _sidebar_items: dict[tuple[str, str], TaskItem]
    # Todo-panel snapshot of the last render (see _todo_fingerprint).
    _todo_snapshot: tuple
    _collapsed_sections: set[str]
    # Per-run activity bookkeeping (parallel slots): last sign of life,
    # the tool each run is waiting on and the tok/s sample window.
    _last_activity: dict[str, float]
    _current_tool: dict[str, str | None]
    # Per-task mergedness of COMPLETED tasks' drcode/<id> branches
    # ("none"/"unmerged"/"merged") — refreshes off the event loop in
    # _unmerged_completed_ids; the sidebar colors the ✓ marker by it.
    _merge_state: dict[str, str]

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
    _hard_limit_exhausted: set[str]
    # Livelock guard: consecutive hard-limit compactions per task that never
    # landed a turn under the hard limit (see _livelock_stop). Shared by
    # the main-agent and tester gates (like the sets above); the resolver
    # keeps its own counter.
    _hard_limit_compactions: dict[str, int]
    # The same guard at the threshold level — one counter per layer (the
    # main agent's, the tester rounds' own): a stale count of one layer
    # must never disable another layer's compaction gate. Reset is
    # progress-gated (see _threshold_observation).
    _threshold_compactions: dict[str, int]
    _tester_threshold_compactions: dict[str, int]
    # Progress streak toward the progress-gated reset of the threshold
    # counters (consecutive under-threshold gate observations; see
    # _threshold_observation).
    _threshold_under_obs: dict[str, int]
    _last_follow: float
    _chat_epoch: int

    # -- chat batch drain ---------------------------------------------------
    _chat_batching: bool
    _chat_batch: list[Widget]
    _chat_drain_at: float
    # Scroll policy of the running drain: a history rebuild always lands on
    # the latest entry (True); a drain seeded by a split fat live write
    # follows the tail only while the reader is already at the bottom
    # (False — checked per chunk, see _drain_chat_batch).
    _chat_batch_follow: bool
    _rendered_clarification: Clarification | None

    # -- delta re-render bookkeeping ----------------------------------------
    # When the open chat is a canonical history render (no live/streamed
    # extras mounted since), _rendered_count is the number of timeline
    # entries already mounted and _rendered_prefix their (timestamp, role)
    # fingerprint — a view_task() of the SAME task then appends only the
    # missing tail instead of wiping and remounting the whole history. Live
    # writes that mirror the next timeline entries are tracked in the
    # live-tail ledger and reconciled by _try_delta_view; a non-timeline
    # live _write_chat() mount or log wipe invalidates both (None/()).
    _rendered_count: int | None
    _rendered_prefix: tuple[tuple[str, ...], ...]
    _live_tail: list[LiveTailEntry]

    # -- widget getters (defined in the DrCodeApp skeleton) -----------------
    @property
    def chat_log(self) -> ChatLog:
        return self.query_one("#chat-log", ChatLog)

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
