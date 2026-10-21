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
from ..models import Clarification, Task, TaskEvent, TaskKind
from ..notifications import Notifier
from ..settings import (
    CompletedRetention,
    ProjectSettings,
    ProjectSettingsStore,
    StallNudge,
    TaskFinalization,
)
from ..task_manager import TaskManager
from .blocks import ToolCallBlock
from .github_dialog import GitHubSetup
from .header import ModelStatusHeader
from .inputs import ChatInput, NewTaskInput
from .setup_dialog import ModelSetup
from .usage import SessionTokenUsage


class DrCodeState(App[None]):

    @work(exclusive=True, group="agent")
    async def run_agent(self, task_id: str) -> None: ...

    async def _check_model(self) -> None: ...
    def _git_branch(self) -> str | None: ...
    def _display_dir(self) -> str: ...

    def refresh_project_info(self) -> None: ...


    def _sidebar_fingerprint(self) -> tuple: ...


    def _poll_sidebar(self) -> None: ...


    def _sidebar_sections(self) -> list[tuple[str, list[Task]]]: ...


    def _sidebar_top_task(self) -> Task | None: ...


    def refresh_sidebar(self) -> None: ...


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


    def _touch_activity(self, tool: str | None = None, *, done: bool = False) -> None: ...


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


    def _render_task_history(self, task: Task, task_id: str) -> None: ...


    def _chat_at_bottom(self) -> bool: ...


    def _stream_follow(self) -> None: ...


    def _tool_call_started(
        self,
        run_key: tuple[str, int],
        task_id: str,
        call_id: str,
        call_text: str,
        block: ToolCallBlock,
    ) -> None: ...


    def _tool_call_finished(self, call_id: str) -> None: ...


    def _stop_run_tools(self, run_key: tuple[str, int]) -> None: ...


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


    def _model_setup_done(self, setup: ModelSetup | None) -> None: ...


    def _maybe_offer_agents_md(self) -> None: ...


    def _agents_md_answered(self, create: bool | None) -> None: ...


    @property
    def new_task_input(self) -> NewTaskInput: ...


    def _clear_stream_label(self) -> None: ...


    def _drop_stream_preview(self, generation: int) -> None: ...


    def action_new_task(self) -> None: ...


    def _leave_new_task_mode(self) -> None: ...


    def cancel_new_task(self) -> None: ...


    def toggle_new_task_kind(self) -> None: ...


    def _update_kind_label(self) -> None: ...


    def _update_kind_descriptions(self) -> None: ...


    def submit_new_task(self) -> None: ...


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


    def action_open_model_setup(self) -> None: ...


    def action_set_completed_retention(self, retention: CompletedRetention) -> None: ...


    def action_change_completed_retention(self) -> None: ...


    def action_set_task_finalization(self, finalization: TaskFinalization) -> None: ...


    def action_change_task_finalization(self) -> None: ...


    def action_set_stall_nudge(self, nudge: StallNudge) -> None: ...


    def action_change_stall_nudge(self) -> None: ...


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


    def _record_stream_tokens(self, text: str) -> None: ...


    def _prune_tps_samples(self, now: float) -> None: ...


    def _tokens_per_second(self) -> float | None: ...


    def _next_testing(self) -> Task | None: ...


    def _advance(self) -> None: ...


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
    agent_running: bool
    tester_running: bool
    _tester_task_id: str | None
    new_task_kind: TaskKind
    _agent_generation: int
    _pending_rendered: list[str]
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

    # -- sidebar / activity -------------------------------------------------
    _sidebar_snapshot: tuple
    _collapsed_sections: set[str]
    _last_activity: float
    _current_tool: str | None

    # -- live stream / connection -------------------------------------------
    _tps_samples: deque[tuple[float, int]]
    _conn_label: str
    _last_probe_ms: int | None
    _probe_fail_streak: int
    _active_tools: dict[tuple[str, int], dict[str, tuple[str, str, ToolCallBlock]]]
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

    # -- widget getters (defined in the DrCodeApp skeleton) -----------------
    @property
    def chat_log(self) -> VerticalScroll:
        return self.query_one("#chat-log", VerticalScroll)

    @property
    def chat_input(self) -> ChatInput:
        ...

    @property
    def sidebar(self) -> ListView:
        ...

    @property
    def project_info(self) -> Static:
        ...

    @property
    def model_status(self) -> Static:
        ...

    @property
    def activity_status(self) -> Static:
        ...

    @property
    def viewed(self) -> Task | None:
        ...
