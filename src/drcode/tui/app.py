from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from rich.markup import escape
from textual import events
from textual.app import (
    App,
    ComposeResult,
    SystemCommand,
    get_system_commands_provider,
)
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import (
    Footer,
    Static,
)

from ..config import Config
from ..github import GitHubSyncState, GitHubSyncStateStore
from ..models import (
    Clarification,
    PendingInput,
    Task,
    TaskKind,
    TaskStatus,
    VerificationStatus,
)
from ..notifications import Notifier
from ..providers import (
    AgentRole,
    UserModelsStore,
    migrate_project_config,
    migrate_user_config,
    project_has_model_selection,
)
from ..providers import resolve_effective as resolve_model_config
from ..settings import ProjectSettings, ProjectSettingsStore, TaskIsolation
from ..task_env import TaskEnvironment
from ..task_manager import TaskManager
from ..tools import cleanup_orphan_scratches
from ..voice import TtsSpeaker
from ..workflow import load_hooks
from .actions import ActionsLayer
from .blocks import AssistantBlock, ReasoningBlock, ToolCallBlock  # noqa: F401
from .chat_view import ChatViewLayer
from .command_providers import (
    DependencyCommandsProvider,
    ModelSettingsCommandProvider,
    ProjectSettingsCommandsProvider,
    ServiceCommandsProvider,
)
from .header import ModelStatusHeader
from .inputs import ChatInput, DepsInput, NewTaskInput
from .pipeline import PipelineLayer
from .setup_dialog import ModelSetupDialog
from .sidebar import Sidebar, TaskItem
from .state import LiveTailEntry
from .todo_panel import TodoPanel
from .usage import SessionTokenUsage
from .voice_input import VoiceInputController


class DrCodeApp(ChatViewLayer, ActionsLayer, PipelineLayer, App[None]):
    # The stock command-palette binding (ctrl+p, shown as "palette" in the
    # footer) is replaced by our own "Settings" binding (ctrl+s) below;
    # disabling the built-in also stops it from shadowing the app's own
    # settings binding.
    ENABLE_COMMAND_PALETTE: ClassVar[bool] = False

    # Header title: the stock Textual default is the class name (DrCodeApp).
    # Unannotated on purpose — Textual declares TITLE as an instance
    # variable, so a ClassVar override trips pyright.
    TITLE = "DrCode"

    # Terminal tab title (iTerm2 etc.): Textual never emits an OSC title
    # sequence itself, so the tab falls back to the process name ("Python").
    # We write OSC 0 on mount (see _set_terminal_title) and clear it on
    # unmount so the title doesn't linger after the app exits.
    TERMINAL_TITLE: ClassVar[str] = "DrCode"

    # Providers for the Settings palette (it passes no explicit providers,
    # so it reads this set): the system commands (theme, quit, …) plus our
    # own service commands (AGENTS.md refresh and friends). The model
    # entries (provider catalog, per-role picks) live in the "Model
    # settings…" sub-menu, whose providers are passed explicitly by
    # action_open_model_settings.
    COMMANDS: ClassVar = {
        get_system_commands_provider,
        ModelSettingsCommandProvider,
        ServiceCommandsProvider,
        ProjectSettingsCommandsProvider,
        DependencyCommandsProvider,
    }

    CSS = """
    Screen { layers: base toast; }


    #main { height: 1fr; }
    #sidebar-col { width: 36; border-right: solid $primary; }
    #sidebar { height: 1fr; border: none; }
    # Sidebar bottom: project-info then activity-status, plain flow (no
    # dock) so they always stack in compose order under the list.
    # project-info is the directory line plus one blank line; the activity
    # line (agent status left, app name+version right) and the detail line
    # below it are rendered by _render_activity().
    #project-info { height: auto; padding: 0 1; text-align: right; }
    #project-info .project-dir { color: $text-muted; }
    # Connection status lives in the header (see tui/header.py), docked
    # left between the icon and the title; the ok/warn/error classes are
    # set by _check_model().
    #model-status { width: auto; padding: 0 1; }
    #model-status.ok { color: $success; }
    #model-status.warn { color: $warning; }
    #model-status.error { color: $error; }
    #activity-status {
        height: auto; padding: 0 1;
        color: $text-disabled;
    }
    #activity-status.fresh { color: $success; }
    #activity-status.stale { color: $warning; }
    #chat-col { height: 1fr; }
    #task-statusbar { dock: top; height: 1; background: $panel; padding: 0 1; }
    #task-status-left { width: 1fr; color: $text; }
    #task-status-right { width: auto; color: $text; }
    # Context counter: red from the compaction threshold, yellow from
    # twice the remaining headroom (threshold 80 → yellow from 60%).
    #task-status-right.ctx-warn { color: $warning; }
    #task-status-right.ctx-crit { color: $error; }
    # The counter is a link (opens the context-layout dialog).
    #task-status-right:hover { text-style: underline; }
    # 1-column empty gutter on both sides: no text sticks to the window edge.
    #chat-log { height: 1fr; border: none; background: $surface; padding: 0 1; }
    #chat-log > * { margin-bottom: 1; }
    # Read-only todo checklist between the chat log and the input: hidden
    # while the viewed task has no checklist, auto-sized to its items
    # otherwise (capped, so a long list cannot crush the chat log).
    #todo-panel {
        height: auto;
        max-height: 10;
        border-top: solid $panel;
        background: $surface;
        padding: 0 1;
    }
    # Multi-line chat editor: height is driven from Python (wrapped line
    # count, capped); auto here is just the pre-measure fallback.
    #chat-input {
        height: auto;
        min-height: 2;
        padding: 0 1;
        border: none;
        # Always visibly distinct from the $surface chat log above, even
        # without focus; focus brightens it one more step.
        background: $surface-lighten-1;
    }
    #chat-input:focus {
        background: $surface-lighten-2;
    }
    .reasoning { padding: 0 4; color: $text-muted; }
    .reasoning:hover { background: $panel; }
    .tool-call { padding: 0 4; color: $text-muted; }
    .tool-call:hover { background: $panel; }
    # Fixed width: the markdown is rendered to a Text at the content width,
    # so an auto-width widget would resize (and rewrap) as lines arrive.
    .assistant { width: 100%; }
    #statusbar { dock: bottom; height: 1; background: $panel; color: $text; padding: 0 1; }
    .section-header { color: $accent; text-style: bold; padding: 0 1; }
    .task-item { padding: 0 1; }
    # Highlight for the task currently open in the chat pane.
    TaskItem.viewed { background: $panel; }
    TaskItem.viewed > .task-item { text-style: bold; }
    #copy-toast {
        layer: toast;
        dock: bottom;
        width: auto;
        height: auto;
        margin: 0 0 3 1;
        padding: 0 2;
        background: $panel;
        border: solid $success;
        color: $text;
    }
    # Persistent dictation indicator, bottom-right corner (on the toast
    # layer so it paints over the statusbar/footer row). Shown for the
    # whole RECORDING/TRANSCRIBING session — the auto-dismissing toasts
    # left a long utterance with no visible sign that dictation is still
    # running. Background stays transparent: the strip only draws its
    # right-aligned text, never a full-width panel.
    #voice-indicator {
        layer: toast;
        dock: bottom;
        width: 100%;
        height: 1;
        text-align: right;
        padding: 0 1 0 0;
        background: transparent;
        color: $warning;
        text-style: bold;
    }
    .hidden { display: none; }
    #new-task {
        height: 1fr; width: 1fr;
        align: center middle;
    }
    #new-task.hidden { display: none; }
    #nt-note {
        width: 80%;
        color: $text-disabled;
        text-align: center;
        margin-bottom: 1;
    }
    #nt-input {
        width: 80%; height: 10;
        border: solid $primary;
    }
    #nt-deps {
        width: 80%;
        border: none;
        background: $surface-lighten-1;
    }
    #nt-deps-note {
        width: 80%;
        color: $error;
    }
    # Labels must stay width: auto: a full-width (1fr) sibling disables
    # horizontal centering for the whole #new-task container, leaving the
    # input glued to the left edge.
    #nt-kind-label {
        width: auto;
        margin-top: 1;
        text-style: bold;
    }
    # Literal colors: they must match the markup colors used for the kind
    # descriptions (#nt-kinds, _KIND_COLOR) and the task status bar — the
    # theme's $success/$error are different shades and would not line up.
    #nt-kind-label.kind-feature { color: green; }
    #nt-kind-label.kind-bug { color: red; }
    #nt-kind-label.kind-research { color: yellow; }
    #nt-hint { width: auto; color: $text-disabled; }
    #nt-kinds {
        width: auto;
        margin-top: 1;
        color: $text-disabled;
    }
    """
    BINDINGS: ClassVar = [
        ("ctrl+n", "new_task", "New task"),
        ("ctrl+d", "done_task", "Done"),
        ("ctrl+x", "cancel_task", "Cancel task"),
        ("ctrl+o", "focus_sidebar", "Tasks"),
        ("ctrl+i", "focus_input", "Input"),
        ("ctrl+t", "cycle_task_kind", "Change type"),
        ("ctrl+up", "move_task_up", "Move up"),
        ("ctrl+down", "move_task_down", "Move down"),
        ("ctrl+e", "edit_task", "Edit held"),
        ("ctrl+s", "settings", "Settings"),
        # Three bindings on ctrl+h plus two on the held-task keys: the
        # footer (and the key dispatch) picks the first one whose
        # check_action allows it — so exactly one of Create on hold /
        # Save on hold / Hold / Resume is shown, depending on whether the
        # new-task form is open (fresh vs editing) and the viewed task's
        # status. The TextArea does not bind ctrl+h, so the key reaches the
        # app while the form editor is focused.
        ("ctrl+h", "submit_new_task_hold", "Create on hold"),
        ("ctrl+h", "save_task_edit_hold", "Save on hold"),
        ("ctrl+h", "hold_task", "Hold"),
        ("ctrl+h", "resume_task", "Resume"),
        ("ctrl+delete", "delete_task", "Delete held"),
        ("escape", "abort_run", "Stop agent"),
        ("ctrl+r", "nudge_run", "Nudge agent"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        # One-time migrations BEFORE the config is built: legacy flat model
        # keys in the user-level / project drcode.json move into the user
        # catalog and the project's per-role custom models, and are removed
        # from the json files (both idempotent, never fatal).
        migrate_user_config()
        migrate_project_config(project_root)
        # Layered config: env > project (.env / drcode.json) >
        # ~/.config/drcode/drcode.json > defaults. The json layers carry
        # only non-provider settings; the model comes from the catalog /
        # project selections / env (see drcode.providers.resolve).
        self.config = Config(project_root=project_root)
        self.manager = TaskManager(project_root)
        # Project-scoped preferences (<project>/.drcode/settings.json), then
        # the startup cleanup: completed tasks older than the retention
        # setting are removed before anything is shown or resumed.
        self.settings_store = ProjectSettingsStore(project_root)
        self.settings: ProjectSettings = self.settings_store.load()
        # Per-task worktree isolation (project setting `task_isolation`,
        # default off): holds the LIVE settings instance, so toggling the
        # setting in the dialog applies from the next activation. Removal
        # of a task (cancel/delete/purge) tears its worktree down here.
        self.task_env = TaskEnvironment(project_root, self.settings)
        # Worktree service events (snapshot commits, kept-branch notices)
        # surface through the task's event history — or a toast when the
        # task is already gone (delete / retention purge).
        self.task_env.event_log = self._task_env_event
        self.manager.add_removal_listener(self.task_env.remove)
        # RULE 1 (see drcode.task_env): a verdict never leaves the task
        # with a dirty tree — every transition out of active work
        # (tester verdict, cancel) snapshots the worktree onto the task's
        # branch before anything can tear it down.
        self.manager.add_status_listener(self._snapshot_on_settled)
        # Strong references to the in-flight background snapshot tasks
        # (a bare create_task result can be garbage-collected mid-flight).
        self._snapshot_jobs: set[asyncio.Task[None]] = set()
        # GitHub Issues sync: persisted bookkeeping (last sync, seen issue
        # numbers) and the monotonic time of the next scheduled cycle. The
        # first cycle runs shortly after startup.
        self._github_state_store = GitHubSyncStateStore(project_root)
        self._github_state: GitHubSyncState = self._github_state_store.load()
        self._github_next_sync = time.monotonic() + 15.0
        # Expanded-header bookkeeping: whether a sync cycle is in flight
        # right now and the last cycle's failure reason (None = the last
        # cycle either succeeded or has not run yet).
        self._github_syncing = False
        self._github_last_error: str | None = None
        # ntfy.sh push notifications: the notifier reads the project
        # settings on every event, so a change made in the settings dialog
        # applies immediately. Delivery failures are surfaced as in-app
        # toasts (the delivery itself runs in a daemon thread). The same
        # notifier voices the configured events (TTS announcements via the
        # audio gateway): the speaker serializes playback one phrase at a
        # time in its own daemon thread and shares the settings getter.
        self.voice_tts = TtsSpeaker(lambda: self.settings, on_error=self._notify_error)
        self.notifier = Notifier(
            lambda: self.settings,
            project=project_root.name,
            project_root=project_root,
            on_error=self._notify_error,
            speaker=self.voice_tts.speak,
        )
        # Push-to-talk voice dictation (Ctrl+B in the input fields).
        self.voice_input = VoiceInputController(self)
        # The "queue empty" / "pipeline blocked" notifications are sent only
        # on state transitions; the very first pipeline evaluation after
        # startup only primes the state (a cold start on an empty queue
        # must not announce "queue is empty").
        self._notifications_primed = False
        self._purged_tasks = self.manager.purge_expired_completed(
            self.settings.completed_retention.days
        )
        # Scratch hygiene: remove /tmp/drcode/<id> directories whose task no
        # longer exists — leftovers from a crashed previous session (scratch
        # directories of live tasks, including interrupted ones, are kept).
        cleanup_orphan_scratches(set(self.manager.tasks))
        # Custom team workflow (opt-in by file presence): .drcode/workflow.py
        # hooks (branch checkout, dev-done, external review, accept) and
        # .drcode/git-workflow.md rules (read at prompt-build time). None =
        # the default single-developer workflow, the pipeline unchanged.
        self.workflow = load_hooks(project_root)
        # External-review bookkeeping (session-scoped): whether a review
        # worker is in flight (set synchronously in _advance, like
        # tester_running) and which task it verifies; per-task retry backoff
        # and failure counts of the on_review hook; the monotonic time the
        # async review poll may fire at next; the on_development_done notes
        # waiting for the tester prompt; and the tasks whose dev-done hook
        # ran in async review mode.
        self._review_running = False
        self._review_task_id: str | None = None
        self._review_retry_at: dict[str, float] = {}
        self._review_fail_count: dict[str, int] = {}
        self._review_poll_at = 0.0
        self._dev_done_notes: dict[str, str] = {}
        # Serializes the accept hooks (on_accepted / on_research_accepted):
        # with parallel slots two tasks can finalize at once, and their
        # merges into the shared main checkout must not interleave (git
        # index.lock, conflicting merges).
        self._accept_hook_lock = asyncio.Lock()
        self.viewed_id: str | None = None
        # Counted run state (parallel development slots): the in-flight
        # agent runs (task id -> generation) and the per-task run
        # generation counter. The tester stays single (plain flag).
        self._agent_runs: dict[str, int] = {}
        self.tester_running = False
        # Task id the tester worker is currently verifying (for Ctrl+X).
        self._tester_task_id: str | None = None
        self.new_task_kind: TaskKind = TaskKind.FEATURE
        # Per task: incremented on every run_agent() start; the finally
        # block of a superseded (cancelled) run must not touch shared
        # state.
        self._agent_generations: dict[str, int] = {}
        # Pending messages already drawn in the current chat view, so the
        # agent run doesn't re-render them when it consumes them.
        self._pending_rendered: list[str] = []
        # Per task: the prompt a run consumed (merged pending input) that is
        # not persisted into the transcript yet — see _consumed_prompts in
        # state.py and the consumption in run_agent.
        self._consumed_prompts: dict[str, tuple[str, list[PendingInput]]] = {}
        # Whether the chat log currently holds at least one block.
        self._chat_has_content = False
        # Auto-hide timer for the "copied" toast.
        self._copy_toast_timer: Timer | None = None
        # Per-task tester failure tracking: consecutive failure count and
        # the earliest monotonic time the task may be retried. A failed
        # tester run is retried with backoff, not dropped: tasks in TESTING
        # block the whole pipeline, so they must eventually get their round.
        self._tester_fail_count: dict[str, int] = {}
        self._tester_retry_at: dict[str, float] = {}
        # Network-retry bookkeeping for tester runs (mirrors the agent's
        # dicts below): a testing round killed by a transient network error
        # is retried in place — the round transcript is kept and the retry
        # resumes it as a continuation turn — instead of restarting the
        # round from scratch. `_tester_resume_round` marks tasks whose next
        # tester run must continue the current round rather than start a
        # new one.
        self._tester_net_count: dict[str, int] = {}
        self._tester_resume_round: set[str] = set()
        # Same idea for agent runs: a task whose run crashed (e.g. the model
        # gateway is unreachable) is demoted to QUEUED and skipped by
        # promote_next for the rest of the session, so one broken task (or a
        # dead gateway) doesn't loop the whole queue through the error.
        self._agent_failed: set[str] = set()
        # Deadlock escape for a queue where every task failed this session
        # (see PipelineLayer._promote_failed_queue): the rescue-round counter
        # (backoff base 60s doubling, capped) and the earliest monotonic time
        # the next rescue may fire at. Reset when a model call succeeds.
        self._failed_queue_rounds = 0
        self._failed_queue_retry_at = 0.0
        # Dependency gate (depends_on promotion condition): the last
        # computed gate (task id -> unsatisfied dep id -> reason) for the
        # sidebar/task-status rendering and the settled notification, plus
        # the per-task dedup flags of its service events. Filled by
        # PipelineLayer._promote_with_dependencies; session-only.
        self._dep_gate: dict[str, dict[str, str]] = {}
        self._dep_block_notified: set[str] = set()
        # Network-retry bookkeeping for agent runs, per task: consecutive
        # transient network failures this session (retry budget) and the
        # earliest monotonic time the delayed re-run fires at. Unlike a
        # generic failure (immediate requeue), a network-retried run keeps
        # the task ACTIVE and is retried after a backoff delay.
        self._agent_net_count: dict[str, int] = {}
        self._agent_net_retry_at: dict[str, float] = {}
        # Total network-retry backoffs this session (the per-task dict above
        # holds only the current retry budgets); shown in the expanded
        # header.
        self._net_retries_total = 0
        # Session statistics for the expanded header (never persisted):
        # tasks present at startup don't count as "created this session",
        # tasks already COMPLETED at startup don't count as "processed".
        self._session_known_ids: set[str] = set(self.manager.tasks)
        self._session_completed_ids: set[str] = {
            t.id
            for t in self.manager.tasks.values()
            if t.status is TaskStatus.COMPLETED
        }
        self._session_created: dict[TaskKind, int] = {kind: 0 for kind in TaskKind}
        self._session_completed = 0
        # Exact cumulative token usage of every model call this session
        # (main agent + tester + compaction summarizer) — shown in the
        # expanded header's bottom line. Session-scoped: resets on restart.
        self._session_usage = SessionTokenUsage()
        # Activation → completion spans (seconds) of the tasks completed
        # this session; feeds the average/median figures.
        self._session_durations: list[float] = []
        # True while the new-task form covers the chat view. Must exist from
        # the start: the streaming loop reads it on every text delta, and an
        # unset attribute would crash the run (it did — the AttributeError
        # surfaced as a red "error ›" line and killed the first streamed run
        # after a startup resume).
        self._new_task_mode = False
        # Task that was open in the chat when the new-task form was entered;
        # cancel (ESC) restores its view (action_new_task() clears
        # viewed_id, but the chat log keeps showing the previous task).
        self._new_task_prev_id: str | None = None
        # While the form edits an existing held task (instead of composing a
        # new one): the id of that task, None otherwise. submit/hold keys
        # branch on it (create vs merge-and-save).
        self._editing_task_id: str | None = None
        # Snapshot of the task state the sidebar was last rendered from; the
        # poller compares against it to redraw only on actual change.
        self._sidebar_snapshot: tuple = ()
        # Mounted sidebar rows by key (("header", section) / ("task", id)):
        # refresh_sidebar() reuses these widgets instead of rebuilding the
        # whole list every time (a full rebuild of a few hundred tasks
        # costs over a second of mounting + layout).
        self._sidebar_items: dict[tuple[str, str], TaskItem] = {}
        # Same idea for the todo panel: (new-task mode, viewed id, checklist
        # items) of the last render — the 1s poll re-renders the panel only
        # when this changes (todo tools refresh it instantly themselves).
        self._todo_snapshot: tuple = ()
        # Sidebar sections currently collapsed (Enter/space/click on a
        # section header toggles it); session-only, not persisted.
        self._collapsed_sections: set[str] = set()
        # Per-run activity bookkeeping (parallel slots): the last moment a
        # run emitted an event (stream delta, tool call/result) — the
        # activity label and the stall watchdog compare it against now.
        self._last_activity: dict[str, float] = {}
        # Tool each run is currently waiting on (name at call start,
        # cleared on the result event) — lets the label distinguish "tool
        # busy" from "total silence".
        self._current_tool: dict[str, str | None] = {}
        # Sliding window of streamed output per run ((monotonic stamp,
        # approximated token count) samples) — feeds the live tok/s
        # figure in the header connection status. See
        # _record_stream_tokens/_tokens_per_second.
        self._tps_samples: dict[str, deque[tuple[float, int]]] = {}
        # Base text of the header connection label ("● model" / "✗ …"),
        # written by _check_model; _render_model_status appends the live
        # activity suffix (tok/s / waiting / testing) on top of it.
        self._conn_label = "○ checking…"
        # Connection probe bookkeeping for the expanded header: last
        # round-trip of probe_model in ms (None = no probe finished yet)
        # and the streak of consecutive failed probes (reset on success).
        self._last_probe_ms: int | None = None
        self._probe_fail_streak = 0
        # Tool calls currently executing, per run (the run key is
        # ("agent"|"tester", task_id, generation)): tool_call_id ->
        # (task_id, rendered call text, live block with the spinner). The
        # text lets a mid-run view re-render (view_task wipes the chat log)
        # re-attach the running spinner to the freshly rendered block.
        self._active_tools: dict[
            tuple[str, str, int], dict[str, tuple[str, str, ToolCallBlock]]
        ] = {}
        # Monotonic stamp distinguishing successive tester runs.
        self._tester_generation = 0
        # Context fill (tokens) seen by the latest completed model request,
        # per task, updated live while the run streams. The stored transcript
        # is only refreshed at persist points, so the status bar reads the
        # live value first and falls back to the persisted one (the final
        # value of a finished task — it stays in the history via the saved
        # per-response usage).
        self._live_ctx_tokens: dict[str, int] = {}
        # Tasks whose latest compaction attempt failed: the mid-run
        # compaction gate must not stop the run for them (a persistently
        # failing summarizer would otherwise loop break→continuation→break
        # forever); turn-boundary compaction still retries each turn.
        self._compaction_failed: set[str] = set()
        # Monotonic stamp of the last coalesced follow-the-tail scroll while
        # streaming (see _stream_follow).
        self._last_follow = 0.0
        # Bumped by every view_task(): the streaming loop compares it with
        # each block's stream_epoch to detect that a mid-node re-render
        # orphaned the live blocks and they must be re-mounted.
        self._chat_epoch = 0
        # While True, _write_chat() collects blocks into _chat_batch instead
        # of mounting them one by one (view_task's history rebuild mounts the
        # whole timeline chunked + one final scroll — per-block mounting of
        # a fat transcript forces a layout/scroll pass per block and freezes
        # the event loop for seconds).
        self._chat_batching = False
        self._chat_batch: list[Widget] = []
        # _chat_epoch value the running drain belongs to: a view switch
        # mid-drain bumps the epoch and the remaining stale chunks are
        # dropped instead of mounting into the new view's log.
        self._chat_batch_epoch = 0
        # monotonic() stamp of the last drain-chunk progress: the 1 s poll
        # watchdog (_poll_chat_drain) re-kicks a drain whose reschedule
        # chain died or starved (see chat_view._schedule_drain).
        self._chat_drain_at = 0.0
        # Delta re-render bookkeeping (see _try_delta_view): None/() until
        # the first full history render; live writes that mirror the next
        # timeline entries are recorded in _live_tail instead of
        # invalidating, so a subsequent view_task can append just the tail.
        self._rendered_count: int | None = None
        self._rendered_prefix: tuple[tuple[str, ...], ...] = ()
        # Live-tail ledger: (role, text, block) of every chat write mounted
        # since the last delta-consistent point that _try_delta_view has not
        # reconciled against the timeline yet. `block` (a ReasoningBlock /
        # AssistantBlock) is set when the entry's final text lives in the
        # widget's own `_content` (streamed blocks whose text was still
        # partial at mount time); otherwise `text` is the exact timeline-
        # comparable string. Cleared by full renders and invalidation.
        self._live_tail: list[LiveTailEntry] = []
        # Clarification object currently rendered in the open chat view.
        # Compared by identity: a newly asked question is a fresh instance,
        # so the agent run's end-of-run handler never re-draws a question
        # that a view_task() re-render has already put on screen (which
        # used to show the question twice while the dialog stayed open).
        self._rendered_clarification: Clarification | None = None
        # Per-role resolved model connections (see drcode.providers.resolve),
        # cached: resolution reads the project config + the user-level
        # catalog from disk. Invalidated whenever the flat config, the
        # project `models` slots or the catalog change.
        self._resolved_configs: dict[AgentRole, Config] = {}

    # -- per-role model resolution -------------------------------------------

    def resolved_config(self, role: AgentRole) -> Config:
        """Effective model connection for an agent role (coding / research /
        tester / compaction): project `models[role]` > user-level default >
        the flat config. The research role falls back to the coding model
        when nothing is selected for it (`providers.resolve_effective`).
        Cached until invalidated."""
        cached = self._resolved_configs.get(role)
        if cached is None:
            cached = resolve_model_config(self.config, role)
            self._resolved_configs[role] = cached
        return cached

    def invalidate_resolved_configs(self) -> None:
        """Drop the cached per-role resolutions (after any change to the
        flat config, the project model slots or the user catalog)."""
        self._resolved_configs.clear()

    def push_screen(self, *args: Any, **kwargs: Any) -> Any:
        """A modal screen is a context change: any in-flight dictation
        belongs to the screen it was started on and must not outlive it
        (its insert-time validation would drop the text anyway — this
        just releases the microphone immediately)."""
        self.voice_input.cancel(silent=True)
        return super().push_screen(*args, **kwargs)

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield ModelStatusHeader(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="sidebar-col"):
                yield Sidebar(id="sidebar")
                yield Static("", id="project-info")
                yield Static("", id="activity-status")
            with Vertical(id="chat-col"):
                with Horizontal(id="task-statusbar"):
                    yield Static("", id="task-status-left")
                    yield Static("", id="task-status-right")
                yield VerticalScroll(id="chat-log")
                yield TodoPanel("", id="todo-panel", classes="hidden")
                yield ChatInput(placeholder="Type a message…", id="chat-input")
                with Vertical(id="new-task", classes="hidden"):
                    yield Static(
                        "This is not a live chat with the agent — "
                        "you are composing a task to be executed later.",
                        id="nt-note",
                    )
                    yield NewTaskInput(placeholder="Describe the task…", id="nt-input")
                    yield DepsInput(
                        placeholder="Depends on — comma-separated task ids (Ctrl+G)",
                        id="nt-deps",
                    )
                    yield Static("", id="nt-deps-note")
                    yield Static("type: feature", id="nt-kind-label")
                    yield Static(
                        "tab — switch type   ctrl+enter — create   "
                        "ctrl+h — create on hold   esc — cancel",
                        id="nt-hint",
                    )
                    yield Static("", id="nt-kinds")
        yield Static("", id="statusbar")
        yield Static("", id="copy-toast", classes="hidden")
        yield Static("", id="voice-indicator", classes="hidden")
        yield Footer()

    def _set_terminal_title(self, title: str) -> None:
        """Write an OSC 0 sequence so the terminal shows `title` in its tab.

        iTerm2 (and other xterm-like terminals) title a tab from the last
        title escape received, falling back to the job name — which is why
        an untitled TUI shows up as "Python".
        """
        driver = self._driver
        if driver is not None:
            driver.write(f"\x1b]0;{title}\x07")
            driver.flush()

    def _agent_badge(self) -> str:
        """Letter of the agent currently working, empty when idle.

        Mirrors the pipeline: an ACTIVE task means the development agent runs
        (its letter is the task kind's, same F/B/R the sidebar uses), any
        TESTING task means the tester runs. Everything else — queued, held,
        awaiting clarification, completed — is idle: no agent letter.
        """
        active = self.manager.active
        if active is not None:
            return {
                TaskKind.FEATURE: "F",
                TaskKind.BUG: "B",
                TaskKind.RESEARCH: "R",
            }[active.kind]
        if any(t.status is TaskStatus.TESTING for t in self.manager.all_tasks()):
            return "T"
        return ""

    def _update_terminal_title(self) -> None:
        """Recompute the tab title from the current pipeline state.

        Runs on the app thread (status listeners hop here via call_later,
        since they may fire from worker threads).
        """
        badge = self._agent_badge()
        base = f"{self.TERMINAL_TITLE} — {self.project_root.name}"
        self._set_terminal_title(
            f"{self.TERMINAL_TITLE} [{badge}] — {self.project_root.name}"
            if badge
            else base
        )

    def _on_title_status_change(
        self, task: Task, old: TaskStatus, new: TaskStatus, origin: str
    ) -> None:
        """Status listener keeping the tab title in sync with the pipeline.

        The title must reflect the agent switch at the exact moment a task
        changes status. Listeners may be called from worker threads, so the
        actual redraw is scheduled onto the app thread (post_message is
        thread-safe); it is a no-op once the app is shutting down.
        """
        self.call_later(self._update_terminal_title)

    def _notify_error(self, text: str) -> None:
        """Delivery failures of the notifications module, surfaced in-app.

        Called from the delivery's daemon thread: hop into the app thread
        (a broken ntfy.sh must never crash anything, hence the blanket
        except — the failure is already on stderr at that point)."""
        try:
            self.call_from_thread(
                self.notify,
                f"Failed to send notification: {text}",
                title="Notifications",
                severity="warning",
                timeout=6,
            )
        except Exception as e:  # noqa: BLE001 - best-effort surface only
            print(f"drcode: notification error surfacing failed: {e}", file=sys.stderr)

    # -- worktree snapshots (task-isolation bookkeeping) -------------------------

    def _on_app_thread(self) -> bool:
        return getattr(self, "_thread_id", None) == threading.get_ident()

    def _snapshot_on_settled(
        self, task: Task, old: TaskStatus, new: TaskStatus, origin: str
    ) -> None:
        """Status listener: RULE 1 of the worktree contract — a verdict
        never leaves the task with a dirty tree. On every transition out
        of active work into COMPLETED (the tester's PASSED/FAILED
        verdict, Ctrl+X cancel, finalize without verification) the task's
        worktree is committed onto its branch `drcode/<task-id>` in the
        background; the result lands in the task's service-event history.
        When the tester already committed via its git_commit tool, the
        snapshot is a no-op. Without isolation this never fires: the work
        lives in the shared main tree and nothing here may touch it."""
        if new is not TaskStatus.COMPLETED or old not in (
            TaskStatus.ACTIVE,
            TaskStatus.TESTING,
        ):
            return
        if not self.task_env.active:
            return
        if origin == "user":
            note = "cancelled"
        elif task.verification is not None:
            note = f"tester verdict: {task.verification.value}"
        else:
            note = "completed without verification"
        self._snapshot_in_background(task.id, note)

    def _snapshot_in_background(self, task_id: str, note: str) -> None:
        """Run task_env.snapshot off the UI thread (blocking git calls —
        this is bookkeeping, not model work) and log the result into the
        task's service-event history."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # A worker thread (e.g. the tester's set_verification tool):
            # no loop here — a plain daemon thread does the job.

            def _run() -> None:
                msg = self.task_env.snapshot(task_id, note)
                if msg is not None:
                    self._task_env_event(task_id, msg)

            threading.Thread(target=_run, name="drcode-snapshot", daemon=True).start()
        else:

            async def _job() -> None:
                msg = await asyncio.to_thread(self.task_env.snapshot, task_id, note)
                if msg is not None:
                    self._task_env_event(task_id, msg)

            # Hold a strong reference: a bare create_task result can be
            # garbage-collected mid-flight.
            job = loop.create_task(_job())
            self._snapshot_jobs.add(job)
            job.add_done_callback(self._snapshot_jobs.discard)

    def _task_env_event(self, task_id: str, text: str) -> None:
        """Service-event sink for the worktree environment
        (TaskEnvironment.event_log): snapshot commits and kept-branch
        notices. Logged into the task's service-event history when the
        task still exists; otherwise (delete / retention purge / orphan
        sweep — the task's chat is already gone) surfaced as a toast, so
        the user still learns that a branch with its work was kept."""
        task = self.manager.tasks.get(task_id)
        if task is not None:
            if self._on_app_thread():
                self._show_event(task, text)
            else:
                # A worker thread (e.g. the tester's set_verification tool
                # snapshot): persisting alone would leave the line invisible
                # until the next full re-render — hop into the app thread and
                # draw it live, one appended line (never fatal).
                try:
                    self.call_from_thread(self._show_event, task, text)
                except Exception:  # noqa: BLE001 - app not running/shutting down
                    self._log_event(task, text)
            return
        self._announce_cleanup(text)

    def _announce_cleanup(self, text: str) -> None:
        """Best-effort toast for worktree notices of tasks that no longer
        exist (never fatal, also before the app is mounted / from worker
        threads)."""
        if not self.is_running:
            return
        try:
            if self._on_app_thread():
                self.notify(text, title="Worktree cleanup", timeout=8)
            else:
                self.call_from_thread(
                    self.notify, text, title="Worktree cleanup", timeout=8
                )
        except Exception as e:  # noqa: BLE001 - the toast is best-effort
            print(
                f"drcode: worktree notice failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )

    def on_unmount(self) -> None:
        # Clear the tab title so "DrCode — <project>" doesn't stick around
        # in the terminal after the app has exited.
        self._set_terminal_title("")

    def on_mount(self) -> None:
        # The app thread is the thread running the asyncio loop: recorded
        # here so _on_app_thread() can tell live UI writes (safe) from
        # worker-thread calls that must hop over via call_from_thread.
        self._thread_id = threading.get_ident()
        self._update_terminal_title()
        # Keep the tab title's agent badge in sync with the pipeline: every
        # status transition (active/testing/queued/…) re-renders the title.
        self.manager.add_status_listener(self._on_title_status_change)
        # Status observers (notifications): wired on mount, so the startup
        # demotions in TaskManager.__init__ never produce events.
        self.manager.add_status_listener(self.notifier.task_status)
        # TTS "bug report created" hook: the tester's create_bug_report
        # tool calls manager.create directly (no status transition), so
        # the voice path needs the creation listener. The startup reload
        # (purge/demotion) does not re-create tasks — wiring on mount is
        # quiet by construction.
        self.manager.add_creation_listener(self.notifier.bug_report_created)
        if self._purged_tasks:
            # The startup cleanup removed something — say so, so deleted
            # history never "silently vanishes".
            self.notify(
                f"Deleted completed tasks: {len(self._purged_tasks)} "
                f"({self.settings.completed_retention.label})",
                title="Auto cleanup",
                timeout=6,
            )
        if (
            self.settings.task_isolation is TaskIsolation.WORKTREE
            and not self.task_env.available
        ):
            # Worktree isolation is persisted even without a git repo (the
            # user's choice is never rewritten silently) — but then it is
            # dormant: warn at EVERY startup until the situation resolves
            # (git init, or the setting switched back to off).
            self.notify(
                "Task isolation is set to worktree, but this project "
                "has no git repository — the mode will not work and "
                "tasks run in the project directory. Run `git init` "
                "(and commit) or switch the setting to off in "
                "Settings (Ctrl+S).",
                title="Task isolation",
                severity="warning",
                timeout=12,
            )
        if self.settings.parallel_tasks > 1 and not self.task_env.active:
            # Parallel development needs per-task worktrees: without a
            # usable isolation the persisted value is dormant and the
            # pipeline behaves as 1 (warned once per startup).
            self.notify(
                f"Parallel tasks: {self.settings.parallel_tasks} is saved, "
                "but needs Task isolation: worktree (a git repository) — "
                "tasks run one at a time until it is on.",
                title="Parallel tasks",
                severity="warning",
                timeout=12,
            )
        if self.task_env.available:
            # Sweep worktrees/branches of tasks that no longer exist
            # (crashed session, retention cleanup) off the startup
            # path: a few git calls in a worker thread. Runs regardless
            # of the setting: `cleanup_orphans` itself no-ops when the
            # setting is off and there is nothing left to sweep, so
            # leftovers of a session with isolation toggled on are still
            # cleaned after the user switches it off.
            self.run_worker(
                self._worktree_startup_cleanup,
                exclusive=True,
                group="worktree-cleanup",
            )
        if self.task_env.active:
            # Remind about completed tasks whose work never reached the
            # current branch (old "stuck" fixes on drcode/* branches —
            # see _unmerged_work_startup_scan).
            self.run_worker(
                self._unmerged_work_startup_scan,
                exclusive=True,
                group="unmerged-scan",
            )
            # Header's unmerged-work counter (`⎇ N` left of the clock):
            # an immediate first fill plus a 10 s poll — a manual
            # `git merge drcode/<id>` clears it without a restart (see
            # _poll_git_indicator / _unmerged_completed_ids).
            self.run_worker(
                self._poll_git_indicator,
                exclusive=True,
                group="git-indicator",
            )
            self.set_interval(10, self._poll_git_indicator)
        self.refresh_project_info()
        self.refresh_sidebar()
        # Open the chat on the task at the top of the first non-empty
        # sidebar section (same pick the sidebar renders first), not on the
        # active task / newest entry.
        initial = self._sidebar_top_task()
        if initial:
            self.view_task(initial.id)
        else:
            self.action_new_task()
        self.refresh_status()
        self.refresh_connection()
        self.set_interval(30, self.refresh_connection)
        # The sidebar must follow task-status changes on its own: statuses
        # also mutate mid-run (agent tools, promote_next in the worker
        # finally-blocks) where no one calls refresh_sidebar().
        self.set_interval(1.0, self._poll_sidebar)
        # External task pickup: task files written by other processes
        # (drcode --feature-request, one-shot GitHub sync) join the running
        # app's queue without a restart (see _poll_new_tasks).
        self.set_interval(1.0, self._poll_new_tasks)
        self.set_interval(1.0, self._poll_activity)
        # Stall watchdog: auto-nudge a run that stopped showing signs of
        # life for the configured period (see _poll_stall).
        self.set_interval(1.0, self._poll_stall)
        # GitHub Issues sync scheduler: every 30s, checks whether the
        # configured interval has elapsed (see _poll_github_sync).
        self.set_interval(30, self._poll_github_sync)
        # Async external review poll (custom workflow without a synchronous
        # on_review hook): asks the hook for the verdict of every completed
        # task with an open PR (see _poll_workflow_review).
        self.set_interval(30, self._poll_workflow_review)
        # Dependency-gate unblock tick: a manual `git merge drcode/<dep>`
        # (or a foreign merge) must unblock waiting dependents even when
        # the pipeline is otherwise idle and nothing re-fires _advance()
        # (see _poll_dependency_unblock).
        self.set_interval(30, self._poll_dependency_unblock)
        # Resume after restart. _advance() re-tests TESTING tasks (the app
        # died during testing) FIRST — the startup demotion (active ->
        # queued) leaves pending_input stranded unless the queue is
        # promoted here. With no model configured anywhere, the setup
        # dialog takes precedence: the pipeline stays idle until the
        # dialog is saved (or skipped with ESC — tasks can still be
        # created and browsed, they just never start).
        #
        # Dialog condition: the env vars suppress it as before; after the
        # startup migrations it only appears when the user catalog has no
        # providers AND the project has no active model entry (catalog
        # reference or valid custom model for at least one role).
        if not (
            os.environ.get("DRCODE_MODEL")
            or os.environ.get("DRCODE_BASE_URL")
            or UserModelsStore().load().providers
            or project_has_model_selection(self.project_root)
        ):
            self.push_screen(ModelSetupDialog(), self._model_setup_done)
        elif self.resolved_config(AgentRole.CODING).is_ready:
            self._advance()
            # Model settings exist — offer to create AGENTS.md if the
            # project has none (the setup-dialog path offers it too,
            # right after the settings are saved).
            self._maybe_offer_agents_md()

    @property
    def chat_input(self) -> ChatInput:
        return self.query_one("#chat-input", ChatInput)

    async def _worktree_startup_cleanup(self) -> None:
        """Startup sweep of the worktree environment: worktree directories
        and `drcode/*` branches whose task no longer exists are removed
        (see drcode.task_env.cleanup_orphans). Blocking git calls — run in
        this worker, off the startup path."""
        removed = await asyncio.to_thread(
            self.task_env.cleanup_orphans, set(self.manager.tasks)
        )
        if removed:
            self.notify(
                f"Removed orphaned task worktrees: {len(removed)}",
                title="Task isolation",
                timeout=6,
            )

    def _unmerged_completed_ids(self) -> list[str]:
        """Synchronous half of the unmerged-work tracking: ids of
        completed tasks whose `drcode/<task-id>` branch still holds
        commits the main checkout's current branch does not have — the
        "stuck" fixes the startup scan reminds about and the header's
        unmerged-work indicator counts. Skipped per task kind when the
        merge is automated by the kind's accept hook (`on_accepted` /
        `on_research_accepted`). Blocking git calls — run in a worker
        thread."""
        env = self.task_env
        if not env.active:
            return []
        workflow = self.workflow
        candidates: list[str] = []
        for task in list(self.manager.tasks.values()):
            if task.status is not TaskStatus.COMPLETED:
                continue
            if task.kind is TaskKind.RESEARCH:
                if workflow is not None and workflow.has("on_research_accepted"):
                    continue  # the merge is the research hook's job
            else:
                if workflow is not None and workflow.has("on_accepted"):
                    continue  # the merge is the hook's job — no reminders
                if task.verification is not VerificationStatus.PASSED:
                    continue
                if task.review_pending:
                    continue  # the external review has not accepted it yet
            candidates.append(task.id)
        if not candidates:
            return []
        unmerged = env.unmerged_ids(candidates)
        return [task_id for task_id in candidates if task_id in unmerged]

    async def _unmerged_work_startup_scan(self) -> None:
        """Startup reminder for old "stuck" fixes: completed tasks whose
        `drcode/<task-id>` branch still holds commits the main checkout's
        current branch does not have (drcode never merges — the branch
        survives the retention cleanup and nothing nags about it, so a
        PASSED task from a previous session can sit unmerged forever —
        and a finished research task's docs commit can too).
        One toast for the whole session, listing the branch names; each
        task's chat already carries the per-task reminder written at its
        finalization. Same criteria as the header's unmerged-work
        indicator (see _unmerged_completed_ids)."""
        env = self.task_env
        stuck_ids = await asyncio.to_thread(self._unmerged_completed_ids)
        if not stuck_ids:
            return
        stuck = [env.branch(task_id) for task_id in stuck_ids]
        base = await asyncio.to_thread(env.base_branch)
        shown = ", ".join(stuck[:5]) + ("…" if len(stuck) > 5 else "")
        self.notify(
            f"Completed task work not merged into {base} yet: {shown}. "
            "Merge it (git merge <branch>) or wire an on_accepted / "
            "on_research_accepted hook in .drcode/workflow.py.",
            title="Unmerged work",
            severity="warning",
            timeout=12,
        )

    @property
    def sidebar(self) -> Sidebar:
        return self.query_one("#sidebar", Sidebar)

    @property
    def todo_panel(self) -> TodoPanel:
        return self.query_one("#todo-panel", TodoPanel)

    @property
    def project_info(self) -> Static:
        return self.query_one("#project-info", Static)

    @property
    def model_status(self) -> Static:
        return self.query_one("#model-status", Static)

    @property
    def activity_status(self) -> Static:
        return self.query_one("#activity-status", Static)

    @property
    def viewed(self) -> Task | None:
        return self.manager.tasks.get(self.viewed_id) if self.viewed_id else None

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Gate the app-level bindings on the current state: a binding that
        can't do anything right now (no viewed task, wrong status, nothing
        running) is hidden from the footer instead of failing silently.

        True = enabled+visible, False = disabled+hidden, None = disabled
        but grayed out in the footer.
        """
        task = self.viewed
        if action == "done_task":
            return task is not None and task.status is not TaskStatus.COMPLETED
        if action == "cancel_task":
            return task is not None and task.status is not TaskStatus.COMPLETED
        if action == "cycle_task_kind":
            # Only a queued (not yet started) task can be re-typed.
            return task is not None and task.status is TaskStatus.QUEUED
        if action in ("move_task_up", "move_task_down"):
            # Manual reordering works on queued tasks only; the binding is
            # hidden while the task sits at the corresponding edge of the
            # queue or at the bug-block boundary (bugs always run first).
            if task is None or task.status is not TaskStatus.QUEUED:
                return False
            order = self.manager.queued()
            if len(order) < 2 or task not in order:
                return False
            i = order.index(task)
            j = i - 1 if action == "move_task_down" else i + 1
            if not 0 <= j < len(order):
                return False
            return (order[j].kind is TaskKind.BUG) == (task.kind is TaskKind.BUG)
        if action == "edit_task":
            # Only a held task can be edited: the editor merges its text and
            # comments; nothing is running for a parked task.
            return task is not None and task.status is TaskStatus.HOLD
        if action == "submit_new_task_hold":
            return self._new_task_mode and self._editing_task_id is None
        if action == "save_task_edit_hold":
            return self._new_task_mode and self._editing_task_id is not None
        if action == "hold_task":
            return task is not None and task.status in (
                TaskStatus.QUEUED,
                TaskStatus.AWAITING_CLARIFICATION,
            )
        if action == "resume_task":
            return task is not None and task.status is TaskStatus.HOLD
        if action == "delete_task":
            return task is not None and task.status is TaskStatus.HOLD
        if action == "abort_run":
            return self.agent_running or self.tester_running
        if action == "nudge_run":
            # A nudge makes sense on a task the pipeline is (or was) working
            # on: abort the hung run and restart it. Also the one-key resume
            # for "active, nothing running" after a previous abort.
            return task is not None and task.status in (
                TaskStatus.ACTIVE,
                TaskStatus.TESTING,
            )
        return True

    # -- copy on select -------------------------------------------------------

    def _selected_text(self) -> str | None:
        """Extract the currently selected text.

        Screen.get_selected_text maps the selection through the per-cell
        meta offsets of the rendered strips, so wrapped display rows and
        content lines stay aligned (a row-based extraction here drifts by
        every padding/wrapped row above the selection).
        """
        return self.screen.get_selected_text()

    def _copy_with_fallback(self, text: str) -> None:
        # copy_to_clipboard uses OSC52, which macOS Terminal.app ignores;
        # pbcopy always works there.
        self.copy_to_clipboard(text)
        if sys.platform == "darwin" and shutil.which("pbcopy"):
            subprocess.run(["pbcopy"], input=text.encode(), check=False)

    def _show_copy_toast(self, text: str) -> None:
        preview = escape(text if len(text) <= 40 else text[:39] + "…")
        toast = self.query_one("#copy-toast", Static)
        toast.update(f"✓ copied: {preview}")
        toast.remove_class("hidden")
        if self._copy_toast_timer is not None:
            self._copy_toast_timer.stop()
        self._copy_toast_timer = self.set_timer(2.0, lambda: toast.add_class("hidden"))

    def on_text_selected(self, event: events.TextSelected) -> None:
        text = self._selected_text()
        if not text:
            return
        self._copy_with_fallback(text)
        self._show_copy_toast(text)

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        for command in super().get_system_commands(screen):
            if command.title == "Maximize":
                continue
            yield command


def run_app(project_root: Path) -> None:
    DrCodeApp(project_root).run()
