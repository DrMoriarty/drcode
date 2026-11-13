"""UI-neutral shared state of the application layers.

The coding app is split into mixin layers that are combined into the
concrete app class (``DrCodeApp`` in the TUI today). Every layer
references a large set of shared instance attributes (task manager,
settings, live-run bookkeeping, …) that are only *assigned* inside the
app's ``__init__``. This module declares those attributes once, as
class-level annotations, so all layers can typecheck their ``self``
references — without importing any UI package.

Two kinds of members live here:

- annotations (no runtime value — the app's ``__init__`` owns the
  actual assignment; reading an attribute before that still raises
  ``AttributeError`` exactly as before);
- the layer protocol: stub declarations (``...``) for methods the
  pipeline layers call that the UI layers implement (refreshes, chat
  writes, activity bookkeeping), plus the per-role model resolution
  implemented here directly (pure logic over the core config layers).

UI-bound members — widget getters, timers, widget-typed bookkeeping —
stay in ``tui.state.DrCodeState``, which combines this class with the
Textual ``App``. The ``host`` attribute is the UI seam: the pipeline
layers reach the frontend only through its callback protocol
(:mod:`drcode.core.host`).
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from pydantic_ai.agent import AgentRun
from pydantic_ai.usage import RunUsage

from ..agent import AgentDeps, ResolverDeps, TesterDeps
from ..config import Config
from ..github import GitHubSyncState, GitHubSyncStateStore
from ..keepawake import KeepAwake
from ..models import (
    Clarification,
    PendingInput,
    Task,
    TaskKind,
)
from ..notifications import Notifier
from ..providers import (
    AgentRole,
    resolve_model_key,
)
from ..providers import resolve_effective as resolve_model_config
from ..settings import (
    ProjectSettings,
    ProjectSettingsStore,
)
from ..task_env import TaskEnvironment
from ..task_manager import TaskManager
from ..usage_store import UsageStore
from ..workflow import WorkflowHooks
from .host import PipelineHost
from .usage import SessionTokenUsage
from .voice_input import VoiceInputController


class CoreState:
    # -- UI seam ------------------------------------------------------------
    # The callback protocol the pipeline layers use instead of a UI base
    # class (toasts, timers, workers, chat blocks). Assigned by the app
    # shell's __init__ before any pipeline activity.
    host: PipelineHost

    # -- per-role model resolution (implemented here; caches assigned in
    #    the app's __init__) --------------------------------------------------

    def resolved_config(self, role: AgentRole) -> Config:
        """Effective model connection for an agent role (coding / research /
        tester / compaction): project `models[role]` > user-level default >
        the flat config. The research role falls back to the coding model
        when nothing is selected for it (`providers.resolve_effective`).
        Cached until invalidated. A PINNED resolution (the availability
        aware choice of the latest run start, see `_pin_resolution`)
        outranks the cache — every consumer of this method (usage
        accounting, compaction gates, header) must see the model the run
        actually uses."""
        pinned = self._pinned_resolutions.get(role)
        if pinned is not None:
            return pinned[0]
        cached = self._resolved_configs.get(role)
        if cached is None:
            cached = resolve_model_config(self.config, role)
            self._resolved_configs[role] = cached
        return cached

    def resolved_model_key(self, role: AgentRole) -> str:
        """Usage-accounting model id of the role's effective model, in the
        catalog's "<provider>/<model-id>" format (bare id for custom
        models and the env override — they carry no provider name; empty
        when nothing is selected). Cached alongside `resolved_config`,
        invalidated with it; a pinned run-start availability choice
        outranks the cache (see `_pin_resolution`)."""
        pinned = self._pinned_resolutions.get(role)
        if pinned is not None:
            return pinned[1]
        cached = self._resolved_model_keys.get(role)
        if cached is None:
            cached = resolve_model_key(self.config, role)
            self._resolved_model_keys[role] = cached
        return cached

    def _pin_resolution(self, role: AgentRole, resolution: Any) -> None:
        """Pin the FACTUAL run-start resolution (an
        `providers.AvailableResolution`) of a role: from now on
        `resolved_config` / `resolved_model_key` return the model the run
        actually uses (after the availability fallback), not the
        theoretical one. Re-pinned at every run start — a provider that
        came back (or went down) is picked up on the next start, so the
        pin never goes stale between runs. Cleared together with the
        caches on any config/catalog change — EXCEPT for a role with a
        live agent run, whose pin survives until the run ends (see
        `invalidate_resolved_configs`)."""
        self._pinned_resolutions[role] = (resolution.config, resolution.model_key)

    def _live_pin_roles(self) -> set[AgentRole]:
        """Roles currently served by a live dev-agent run: the roles of
        the tasks registered in `_agent_runs`. A run keeps the model it
        was built with (the agent is constructed at run start), so its
        pin must outlive a config invalidation — the run's gauges
        (compaction/eviction gates, fill status) measure through
        `resolved_config` and would otherwise silently switch to the
        newly selected model mid-run (incident #cca920d5)."""
        runs = getattr(self, "_agent_runs", None)
        if not runs:
            return set()
        roles: set[AgentRole] = set()
        for task_id in runs:
            try:
                task = self.manager.get(task_id)
            except (AttributeError, KeyError):
                continue
            roles.add(self._task_role(task))
        return roles

    def _drop_run_pin_if_idle(self, role: AgentRole) -> None:
        """Drop the role's run-start pin once no live dev run of the
        role remains (called from the run worker's teardown): between
        runs the resolution must again reflect the CURRENT configuration
        — the pin is re-created at the next run start anyway. Another
        live run of the same role (parallel slots) keeps the pin."""
        if not self._pinned_resolutions.get(role):
            return
        if role in self._live_pin_roles():
            return
        self._pinned_resolutions.pop(role, None)

    def invalidate_resolved_configs(self) -> None:
        """Drop the cached per-role resolutions (after any change to the
        flat config, the project model slots or the user catalog).

        The run-start pin of a role with a LIVE agent run survives: the
        run keeps serving on the model it was built with, so dropping
        its pin mid-run would let the run's gauges resolve the NEW
        model and silently skip the old model's compaction hard limit
        until the next request is deterministically rejected (incident
        #cca920d5). The surviving pin is dropped when the run ends
        (`_drop_run_pin_if_idle`)."""
        self._resolved_configs.clear()
        self._resolved_model_keys.clear()
        live = self._live_pin_roles()
        if not live:
            self._pinned_resolutions.clear()
            return
        for role in list(self._pinned_resolutions):
            if role not in live:
                del self._pinned_resolutions[role]

    # -- per-task agent-model event bookkeeping ------------------------------
    # The effective model key each task's agent last STARTED a run on
    # (task id → "<provider>/<model-id>"; a session-scoped dict, never
    # persisted): lets the run-start event report a model SWITCH between
    # attempts (e.g. after a request-limit demotion the user picked a
    # different model) instead of a bare model line. Assigned in the
    # app's __init__ alongside the resolution caches above.
    _last_agent_model_keys: dict[str, str]

    def _snapshot_model_keys(self) -> dict[AgentRole, str]:
        """The cached per-role model keys as they resolve RIGHT NOW —
        the "before" half of a manual model change (the caller invalidates
        the caches right after and passes the snapshot to
        `_emit_manual_model_change_events`)."""
        return {role: self.resolved_model_key(role) for role in AgentRole}

    def _emit_run_model_event(self, task: Task, role: AgentRole) -> None:
        """Run-start service event naming the model the agent's run is
        served by. Fires on every run start that follows a STATUS
        transition (first activation, re-entry after a queue demotion)
        — but NOT on the in-place resume of a network retry (the task
        never changed status, the model line would only repeat — see
        `_agent_retry_resumes`). When the task's previous run started on
        a different key (the user picked another model in the meantime),
        the wording says so — a switch is reported even on a retry
        resume.

        A task whose key has no entry yet (first run of the session —
        the dict above is session-scoped and not persisted) always gets
        the plain "Model: <key>" line. Like every service event this goes
        to `task.events` — shown in the chat, never fed to the model."""
        key = self.resolved_model_key(role)
        if not key:
            return
        previous = self._last_agent_model_keys.get(task.id)
        self._last_agent_model_keys[task.id] = key
        if previous and previous != key:
            self._show_event(task, f"Model switched: {previous} → {key}")
            self._agent_retry_resumes.discard(task.id)
            return
        if task.id in self._agent_retry_resumes:
            # An in-place network-retry resume: no status change, the
            # model is unchanged — the plain line would only repeat.
            self._agent_retry_resumes.discard(task.id)
            return
        self._show_event(task, f"Model: {key}")

    def _emit_manual_model_change_events(self, before: dict[AgentRole, str]) -> None:
        """Manual model-change events: the user picked a different model
        for a role while a task's agent run is in flight. The current run
        keeps the model it was built with — the event says the change
        applies from the next run. `before` is the snapshot taken before
        `invalidate_resolved_configs()`; the "after" half is resolved
        fresh here — deliberately BYPASSING the run-start pin, which
        survives the invalidation for roles with a live run (the event
        must name the model the NEXT run will pick up). Only roles
        whose effective key actually changed and only tasks with a LIVE
        agent run (the dev agent, `_agent_runs`) of that role get an
        event."""
        for role in AgentRole:
            old = before.get(role) or ""
            new = resolve_model_key(self.config, role)
            if not old or not new or old == new:
                continue
            for task in self.manager.active_tasks():
                if task.id not in self._agent_runs:
                    continue
                if self._task_role(task) is not role:
                    continue
                self._show_event(
                    task,
                    f"Model changed: {old} → {new} "
                    "(the current run keeps the old model; "
                    "the change applies from the next run)",
                )

    def _zombie_active_ids(self) -> set[str]:
        """ACTIVE tasks no live run owns and nothing legitimately pauses:
        not in the run registry, not an unfinished attempt (abort /
        mid-run compaction pause — its handlers left the traces and the
        user or the restart-time auto-resume picks it up), not waiting on
        a scheduled network retry, and a model exists to run it at all.

        This is the silent-worker-exit state (incident #a2dc8424):
        promote_next never re-activates a non-queued task, so such a task
        would sit ACTIVE forever — and through the raw active count block
        a development slot forever. The stall watchdog requeues or
        restarts them after the stall limit (see _poll_stall); the slot
        accounting refuses to count them as occupied slots in the
        meantime."""
        return {
            t.id
            for t in self.manager.active_tasks()
            if t.id not in self._agent_runs
            and not t.unfinished
            and t.id not in self._agent_net_retry_at
            and self.resolved_config(self._task_role(t)).is_ready
        }

    _resolved_configs: dict[AgentRole, Config]
    _resolved_model_keys: dict[AgentRole, str]
    # The factual run-start availability resolution per role
    # (role → (Config, model key); see `_pin_resolution`). Assigned in
    # the app's __init__ alongside the caches above.
    _pinned_resolutions: dict[AgentRole, tuple[Config, str]]

    # -- layer protocol (implemented in the UI layers) -----------------------
    # Signatures are Any-typed where the implementations handle widgets: the
    # core must not import a UI package, and `Any` keeps the overrides
    # compatible in both directions.

    def run_agent(self, task_id: str) -> None: ...

    async def _run_agent_worker(self, task_id: str) -> None: ...

    @property
    def agent_running(self) -> bool: ...

    @property
    def parallel_limit(self) -> int: ...

    async def _check_model(self) -> None: ...

    def refresh_sidebar(self) -> None: ...

    def refresh_status(self) -> None: ...

    def refresh_task_status(self) -> None: ...

    def refresh_todo_panel(self) -> None: ...

    def _todo_fingerprint(self) -> tuple: ...

    @staticmethod
    def _todo_instructions(task: Task) -> str | None: ...

    def _todo_tail_block(self, task_id: str) -> str | None: ...

    def _todo_completion_warning(self, task: Task) -> None: ...

    def _record_session_usage(
        self, usage: RunUsage | None, role: AgentRole, model_id: str
    ) -> None: ...

    def _touch_activity(
        self, task_id: str, tool: str | None = None, *, done: bool = False
    ) -> None: ...

    def _poll_activity(self) -> None: ...

    def _render_activity(
        self, status: str, detail: str, state: str | None = None
    ) -> None: ...

    def _render_model_status(self) -> None: ...

    def _active_header_role(self) -> AgentRole: ...

    def _poll_stall(self) -> None: ...

    def _nudge_run(self, task: Task, *, source: str = "watchdog") -> bool: ...

    def _abort_task_runs(self, task_id: str) -> None: ...

    def _schedule_pipeline_timer(
        self, delay: float, callback: Callable[[], Any]
    ) -> None: ...

    def _log_event(self, task: Task, text: str, *, kind: str = "system") -> None: ...

    def _show_event(self, task: Task, text: str, *, kind: str = "system") -> None: ...

    def _show_event_once(
        self, task: Task, text: str, *, kind: str = "system"
    ) -> None: ...

    def _isolation_pin_note(self, task: Task, *, save: bool) -> None: ...

    def view_task(self, task_id: str) -> None: ...

    def _try_delta_view(self, task_id: str) -> bool: ...

    def _invalidate_chat_delta(self) -> None: ...

    def _tool_call_started(
        self,
        run_key: tuple[str, str, int],
        task_id: str,
        call_id: str,
        call_text: str,
        block: Any,
    ) -> None: ...

    def _tool_call_finished(self, call_id: str) -> None: ...

    def _stop_run_tools(self, run_key: tuple[str, str, int]) -> None: ...

    def _write_chat(
        self,
        content: Any,
        *,
        expand: bool = False,
        note: Any = None,
    ) -> None: ...

    def _write_history_line(self, role: str, text: str, at: Any = None) -> Any: ...

    def _write_clarification(self, clarification: Clarification) -> None: ...

    def _drop_stream_preview(self, task_id: str, generation: int) -> None: ...

    def _clear_stream_label(self) -> None: ...

    def _record_stream_tokens(self, task_id: str, text: str) -> None: ...

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

    def _promote_with_dependencies(self) -> Any: ...

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

    def _eviction_due(self, task: Task) -> bool: ...

    def _maybe_evict_marked(self, task: Task, force: bool = False) -> int: ...

    def _threshold_observation(
        self,
        task: Task,
        tokens: int | None,
        counter: dict[str, int],
    ) -> None: ...

    async def _maybe_compact(self, task: Task) -> None: ...

    async def _maybe_compact_after_idle(self, task: Task) -> None: ...

    async def _maybe_compact_tester_round(
        self, task: Task, live_tokens: int | None, *, force: bool = False
    ) -> bool: ...

    def _mechanical_framing(
        self, task: Task, role: AgentRole
    ) -> tuple[int, int, int] | None: ...

    def _schedule_tester_retry(self, task_id: str) -> None: ...

    def run_tester(self, task_id: str) -> Any: ...

    def _maybe_start_resolver(self, task_id: str) -> None: ...

    def run_resolver(self, task_id: str) -> Any: ...

    def _retry_resolver(self) -> None: ...

    async def _verify_resolution(
        self,
        task: Task,
        root: Path,
        report: Any,
        deps: ResolverDeps,
    ) -> Any: ...

    # Returns True when a targeted repair round was scheduled instead of
    # the rollback (the caller's invocation stays UNconcluded).
    async def _apply_resolver_outcome(
        self, task: Task, root: Path, base: str, outcome: Any, paths: list[str]
    ) -> bool: ...

    async def _give_up_to_human(self, task: Task, root: Path, base: str) -> None: ...

    def _persist_resolver_messages(
        self, task: Task, run: AgentRun[Any, str] | None
    ) -> None: ...

    async def _maybe_compact_resolver_round(
        self, task: Task, live_tokens: int | None
    ) -> bool: ...

    # -- external review / workflow hooks (implemented in the pipeline) ------

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

    def run_external_review(self, task_id: str) -> Any: ...

    def _poll_workflow_review(self) -> None: ...

    def _install_rerun_candidates(self, unmerged_ids: list[str]) -> list[str]: ...

    async def _poll_pending_installs(self) -> None: ...

    @staticmethod
    def _task_role(task: Task) -> AgentRole: ...

    def _coding_config(self) -> Config: ...

    def _recover_from_outage(self) -> None: ...

    def _tokens_per_second(self) -> float | None: ...

    def _unmerged_completed_ids(self) -> list[str]: ...

    def _remove_merged_completed_worktrees(self) -> None: ...

    def _suggest_conflict_screen(self, task: Task) -> None: ...

    def _maybe_rerun_pending_installs(self) -> Any: ...

    async def _check_model_inner(self) -> None: ...

    @property
    def voice_dictation_configured(self) -> bool: ...

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
    keepawake: KeepAwake

    # -- run state ----------------------------------------------------------
    viewed_id: str | None
    # Counted run state (parallel development slots): task id -> the
    # generation of its in-flight agent run. `agent_running` above is a
    # property over `_agent_runs`; the tester keeps its own singular
    # `tester_running`.
    _agent_runs: dict[str, int]
    tester_running: bool
    _tester_task_id: str | None
    # Armed-ESC confirmation (project setting `esc_abort`, default
    # double_press): the run-state signature the arming saw (None = not
    # armed), when it was armed, the last accepted ESC press time (the
    # debounce base) and the window-expiry timer (a UI object — declared
    # in tui.state).
    _esc_armed_signature: tuple | None
    _esc_armed_at: float
    _esc_last_press: float
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

    # -- tester failure / retry bookkeeping ---------------------------------
    _tester_fail_count: dict[str, int]
    _tester_retry_at: dict[str, float]
    _tester_net_count: dict[str, int]
    _tester_resume_round: set[str]
    # Tasks whose testing round already burned its ONE forced-compaction
    # recovery of a deterministic context-overflow rejection (see
    # run_tester's overflow branch): a second overflow gives the round up.
    _tester_overflow_retried: set[str]
    # Tasks whose pending round-resume follows a SUCCESSFUL compaction (see
    # run_tester): the continuation turn carries the resume note.
    _tester_compaction_resumed: set[str]
    # Tasks whose testing round had ALREADY issued its set_verification
    # verdict by the time a compaction landed (#490d5652 class): the resumed
    # round must be warned not to re-issue it. Sticky across the round's
    # compactions (the verdict call itself is summarized away by the first
    # one); cleared when a FRESH round starts.
    _tester_compaction_verdict: set[str]

    # -- merge-conflict resolver bookkeeping (single-slot worker) -----------
    _resolver_running: bool
    _resolver_task_id: str | None
    _resolver_generation: int
    _resolver_fail_count: dict[str, int]
    _resolver_retry_at: dict[str, float]
    _resolver_net_count: dict[str, int]
    _resolver_resume_round: set[str]
    # The dirty-paths baseline of the running resolver round (the scope
    # guard, c3b17718): captured when a FRESH round starts — the engine
    # refuses to deploy a merge onto a dirty tracked tree, so everything
    # dirty then is the merge's own staging plus pre-existing untracked
    # files, never the resolver's work — kept across the round's
    # scheduled retries, popped when the round concludes.
    _resolver_round_baseline: dict[str, set[str]]
    # The pending targeted-repair continuation of a resolver round: task
    # id -> the repair prompt (the failing checks' output or the review
    # report + point-fix instructions). Set by _apply_resolver_outcome
    # when a CONCRETE failure follows a successful agent claim — a failed
    # mechanical check / test suite, or a 'lost' review verdict with an
    # actionable finding — the resolved worktree is NOT rolled back; the
    # next invocation continues the SAME round transcript
    # (task.resolver_messages[-1]) with this prompt. Popped when the
    # repair invocation starts.
    _resolver_repair_round: dict[str, str]
    # Repair rounds already spent per conflict (budget: _MAX_REPAIR_ROUNDS):
    # incremented per repair, reset when a FRESH round deploys/re-detects
    # the conflict (a new prepare/triage) — a rollback starts a clean slate.
    _resolver_repair_count: dict[str, int]
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
    # Task ids whose NEXT run start is the in-place resume of a scheduled
    # network retry (set by _agent_turn_network_error, consumed by
    # _emit_run_model_event): such a resume never changed the task's
    # status, so the plain "Model:" line is suppressed there. Session-only.
    _agent_retry_resumes: set[str]
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

    # -- session usage (expanded header, session-scoped) ---------------------
    _session_usage: SessionTokenUsage
    _usage_store: UsageStore

    # -- new-task form ------------------------------------------------------
    # Tab-cycle order of the task kinds (TaskFormLayer; the lifecycle
    # layer's kind cycle for existing tasks reads the same order).
    _KIND_CYCLE: ClassVar[tuple[TaskKind, ...]]
    _new_task_mode: bool
    _new_task_prev_id: str | None
    _editing_task_id: str | None
    # The goal link preset into the new-task form (the GoalScreen's T);
    # consumed by the fresh-creation submit, cleared with the form.
    _new_task_goal_id: str | None
    # What the in-flight goal-capture dialog should wire its result to:
    # (parent goal id, task id to link).
    _goal_capture_context: tuple[str | None, str | None]

    # The clarification currently rendered in the open chat view (the
    # end-of-run epilogue compares against it to avoid a double render).
    _rendered_clarification: Clarification | None

    @property
    def viewed(self) -> Task | None:
        """The task whose chat view is open (None on the empty state)."""
        return self.manager.tasks.get(self.viewed_id) if self.viewed_id else None

    # -- push-to-talk dictation (Ctrl+B; created in the app's __init__) ------
    voice_input: VoiceInputController

    # -- sidebar / activity -------------------------------------------------
    _sidebar_snapshot: tuple
    # Todo-panel snapshot of the last render (see _todo_fingerprint).
    _todo_snapshot: tuple
    _collapsed_sections: set[str]
    # Per-run activity bookkeeping (parallel slots): last sign of life,
    # the tool each run is waiting on and the tok/s sample window.
    _last_activity: dict[str, float]
    _current_tool: dict[str, str | None]
    # Per-task mergedness of COMPLETED tasks' drcode/<id> branches
    # ("none"/"unmerged"/"merged") — refreshed off the event loop in
    # _unmerged_completed_ids; the sidebar colors the ✓ marker by it.
    _merge_state: dict[str, str]
    # Task ids already handed to the post-merge worktree removal (see
    # DrCodeApp._remove_merged_completed_worktrees) — one scheduled
    # removal per task per session.
    _post_merge_removal_done: set[str]
    # Footer visibility of the Merge… binding as of the last git-indicator
    # poll (None = not evaluated yet); see check_action "open_merge".
    _merge_binding_visible: bool | None
    # True once the unmerged-work tracking has completed at least one fill
    # (the poll's first pass, the startup scan or action_open_merge); until
    # then the Merge… candidate state is unknown and must not gate the key.
    _merge_poll_done: bool

    # -- live stream / connection -------------------------------------------
    _tps_samples: dict[str, deque[tuple[float, int]]]
    _conn_label: str
    _conn_role: AgentRole | None
    _last_probe_ms: int | None
    _probe_fail_streak: int
    _tester_generation: int

    # -- context fill / compaction ------------------------------------------
    _live_ctx_tokens: dict[str, int]
    # Eviction credit per task: the ≈tokens the batch eviction of the
    # agent-marked tool results collapsed (see
    # PipelineCompactionLayer._maybe_evict_marked). Every fill reading
    # taken before the next real usage report over-reports by exactly
    # that; the dev-transcript gates subtract the credit until a fresh
    # reading consumes it (pipeline_agent_run._agent_model_request_done)
    # or a full compaction rewrites the transcript. The tester/resolver
    # rounds never evict — their gates must not see a dev credit.
    _evicted_ctx_tokens: dict[str, int]
    # The same live readings for the two round-running agents: tester
    # rounds (the shared map above belongs to the dev transcript and must
    # not be polluted with the tester's) and resolver rounds. Presence of
    # an entry = that agent is running for the task; popped in the run's
    # finally-block.
    _live_ctx_tester: dict[str, int]
    _live_ctx_resolver: dict[str, int]
    # Compaction-failure bookkeeping of the DEV/RESEARCH transcript gate —
    # separate from the tester/resolver round sets so a broken summarizer
    # (or a livelock) seen by one role's transcript never disables another
    # role's gate.
    _compaction_failed: set[str]
    _hard_limit_exhausted: set[str]
    # Same bookkeeping for the tester ROUND gate (see
    # _maybe_compact_tester_round): an agent-transcript compaction failure
    # or livelock must never disable the testing round's mid-round gate.
    _tester_compaction_failed: set[str]
    _tester_hard_limit_exhausted: set[str]
    # Livelock guard: consecutive hard-limit compactions per task that never
    # landed a turn under the hard limit (see _livelock_stop). One counter
    # per layer (dev transcript, tester round; the resolver keeps its own)
    # — a stale count of one layer must never stop another layer's gate.
    _hard_limit_compactions: dict[str, int]
    _tester_hard_limit_compactions: dict[str, int]
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
    _chat_epoch: int

    # -- chat batch drain ---------------------------------------------------
    _chat_batching: bool
    _chat_drain_at: float
    # Watchdog diagnostics for the drain (see _poll_chat_drain /
    # _report_drain_stall): re-kicks within the current drain episode and
    # the monotonic stamp of the last emitted stall event (throttle).
    _chat_drain_rekicks: int
    _chat_drain_diag_at: float
    # Answers picked by CLICKING a package question's options (question
    # index → option text): rendered into the chat input as one line per
    # question and submitted together on Enter (see action_pick_option).
    _clarification_picks: dict[int, str]
    # Accumulated clicks of MULTIPLE-choice questions (question index →
    # selected indices in the combined option list): toggled by clicks,
    # rendered "; "-joined into the question's draft line (see
    # action_pick_option).
    _clarification_multi: dict[int, set[int]]

    # -- delta re-render bookkeeping ----------------------------------------
    # When the open chat is a canonical history render (no live/streamed
    # extras mounted since), _rendered_count is the number of timeline
    # entries already mounted and _rendered_prefix their (timestamp, role)
    # fingerprint — a view_task() of the SAME task then appends only the
    # missing tail instead of wiping and remounting the whole history. Live
    # writes that mirror the next timeline entries are tracked in the
    # live-tail ledger (a UI object — declared in tui.state) and
    # reconciled by _try_delta_view; a non-timeline live _write_chat()
    # mount or log wipe invalidates both (None/()).
    _rendered_count: int | None
    _rendered_prefix: tuple[tuple[str, ...], ...]

    # Fingerprint of the screen's active bindings at the last footer refresh
    # (see refresh_task_status): the 1 s poll must not publish
    # bindings_updated when nothing changed — every publish makes textual's
    # Footer recompose from scratch (a remount of every FooterKey plus a
    # stylesheet pass and a full screen relayout), which on a fat archive
    # burns continuous CPU while idle.
    _footer_bindings_fp: tuple | None
