"""The automatic merge-conflict resolver round.

Split out of the pipeline like the tester (`pipeline_tester.py`), whose
shape it mirrors: a SINGLE worker (`group="resolver"`, never occupies a
development slot), an isolated per-round transcript
(`task.resolver_messages` — never mixed into the coding agent's or the
tester's history), mid-round compaction gates measured against the
RESOLVER role's resolved config, and network-retry/outage handling.

The funnel (all of it must pass before anything is committed):

1. the resolver agent resolves the conflicts parked in the task's
   worktree (zdiff3 markers, both sides' intent must survive) and
   reports via set_resolution;
2. mechanical checks in the resolved tree (compile / ruff / pyright —
   `drcode.merge_resolver.mechanical_checks`);
3. the project's test suite in the resolved tree;
4. the intent-preservation review: a read-only reviewer pass receives
   diff(resolved, ours) and diff(resolved, theirs) and answers whether a
   piece of either side was lost (the #6c065401 incident class). The
   review is window-aware: the diff budget derives from the reviewer
   model's context window and a many-file conflict is reviewed in
   several self-contained chunks whose verdicts fold (any 'lost' →
   'lost', else any 'unsure' → 'unsure').

Before the funnel a SCOPE GUARD rejects the round outright when the
worktree carries changes outside the triage report's paths (the
#c3b17718 incident class — a foreign edit would be silently unverified,
silently uncommitted, and would block the accept hook with a dirty
worktree), and EVERY round end — success, failure, error/retry,
cancellation — reports any uncommitted leftovers outside the set into
the task's service events (end-of-round hygiene).

Only a full pass commits (`merge_engine.finish`) — the merge into the
base branch then becomes a fast-forward. Any failure rolls the worktree
back to the PRISTINE conflicted state, marks the task `needs_human` and
logs the full report into the service events; nothing is ever committed.

The one bounded exception: an intent review that finds a CONCRETE loss
(verdict 'lost' with an actionable report) schedules up to
`_MAX_REPAIR_ROUNDS` targeted repair rounds — the resolution stays in
the worktree, the review report continues the SAME round transcript
(the coding agent's reject-rework pattern), and the FULL funnel re-runs
after the fix. A spent budget, 'unsure', a scope expansion or a vague
report all keep the rollback.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRun
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)
from pydantic_ai.usage import UsageLimits

from ..agent import (
    ResolverDeps,
    all_user_prompts,
    build_resolver_agent,
    build_resolver_reviewer_agent,
    last_response_usage,
    resumable_history,
    run_context_tokens,
    serialize_messages,
)
from ..core.usage import (
    NETWORK_RETRIES,
    _is_context_overflow_error,
    _is_retryable_network_error,
)
from ..merge_engine import MergeReport, in_progress, remove_report, report_path
from ..merge_resolver import (
    ResolverOutcome,
    ReviewChunk,
    aggregate_review_verdicts,
    build_agent_context,
    conclude,
    dirty_paths,
    mechanical_checks,
    out_of_scope_changes,
    prepare,
    restore_conflict,
    review_char_budget,
    review_diff_chunks,
    run_test_suite,
    stage_resolved,
)
from ..models import Task, TaskStatus
from ..providers import AgentRole
from ..tools import scratch_dir
from .pipeline_stream import NodeStreamView
from .state import CoreState
from .worker_runtime import work

# How many failed resolver attempts each error class gets before the
# round gives up and a human takes over (the resolver is unattended by
# design — an endless retry loop against a broken model would park the
# worktree half-done for hours). Model errors: _MAX_RESOLVER_ATTEMPTS;
# transient network errors: a more generous budget (an outage must not
# burn the human hand-off), with the same capped exponential backoff —
# the resolver deliberately does NOT enter the tester's TCP-gated outage
# mode: that machinery is coupled to the agent/tester states, and a
# resolver round waiting out an outage in 300 s backoffs is fine.
_MAX_RESOLVER_ATTEMPTS = 5
_MAX_RESOLVER_NETWORK_ATTEMPTS = NETWORK_RETRIES * 3
# Base of the capped exponential backoff between attempts (5, 10, 20 ... s).
_RETRY_BASE_DELAY_S = 5.0
# Targeted-repair rounds after a 'lost' intent-review verdict (per
# CONFLICT, not per round): instead of rolling the whole resolution back,
# the review report is fed back into the SAME resolver transcript and the
# round continues with a point fix — then the FULL funnel re-runs. After
# the budget is spent the behavior is the old one: restore + needs_human.
_MAX_REPAIR_ROUNDS = 2
# The tiny gap between scheduling a repair round and its invocation (the
# scheduler re-enters run_resolver through the retry timer; the current
# invocation's finally-block must run first).
_REPAIR_RETRY_DELAY_S = 1.0
# A review report shorter than this is never "actionable" (a concrete
# finding needs somewhere to name the lost work).
_MIN_ACTIONABLE_REPORT_CHARS = 40
# A report this long is treated as actionable even without a recognizable
# file mention — it almost certainly names the finding.
_DETAILED_REPORT_CHARS = 160


def _review_report_actionable(report: str, paths: list[str]) -> bool:
    """Whether a 'lost' review report names its finding concretely enough
    for a targeted repair: it must point at the lost work — a conflicted
    file (full path or bare name — the reviewer receives that list in its
    prompt), any path-like token (`dir/file`, `file.ext`), or be detailed
    enough to act on. A vague "something may be missing" is not a
    finding — that is 'unsure' territory and keeps the old rollback."""
    text = (report or "").strip()
    if len(text) < _MIN_ACTIONABLE_REPORT_CHARS:
        return False
    lowered = text.lower()
    for p in paths:
        name = p.rsplit("/", 1)[-1].lower()
        if name and (p.lower() in lowered or name in lowered):
            return True
    if re.search(r"\S+/\S+", text):
        return True
    return len(text) >= _DETAILED_REPORT_CHARS


def _build_repair_prompt(review_report: str, number: int) -> str:
    """The repair continuation turn fed into the SAME resolver transcript
    (mirrors the coding agent's reject-rework pattern): the resolution
    stays in the worktree, the review's finding gets fixed in place."""
    return (
        f"The verification funnel's intent-preservation review REJECTED "
        f"your resolution: it found work from one side of the merge that "
        f"the resolution lost (verdict: lost; repair round {number} of "
        f"{_MAX_REPAIR_ROUNDS}). The resolved tree is still in place in "
        f"the worktree — NOTHING was rolled back. Apply the review's "
        f"findings as a targeted repair: change only what the review "
        f"names (plus what that change necessarily touches), leave the "
        f"rest of the resolution exactly as it is, then report the "
        f"repaired state again with set_resolution.\n\n"
        f"=== intent review report (verdict: lost) ===\n"
        f"{(review_report or '').strip()[:4000]}"
    )


# The intent-preservation review's OWN retry budget (the #9c3236e8 class):
# an infrastructure failure of the CHEAP reviewer pass used to be silently
# treated as an 'unsure' verdict — rolling back tens of minutes of resolver
# work over one dropped HTTP call. Retryable failures (429s, transport
# errors — see `_is_retryable_network_error`) now get a small budget of
# their own with the same capped exponential backoff, NOT spending the
# round's `_MAX_RESOLVER_ATTEMPTS`; a deterministic context-overflow 400
# (or any other non-retryable error) goes straight to 'unsure' as before —
# replaying the same oversized review prompt is guaranteed the same 400.
_MAX_REVIEW_ATTEMPTS = 3


# -- intent-preservation review (level 3 of the funnel) ------------------------------


def _review_chunk_prompt(
    task: Task,
    report: MergeReport,
    agent_report: str,
    chunk: ReviewChunk,
    index: int,
    total: int,
) -> str:
    """One reviewer prompt for one chunk. A single-chunk review keeps the
    historical shape; a multi-chunk one names the chunk's file scope
    explicitly — the reviewer must judge ONLY its chunk (the remaining
    conflicted files are reviewed in their own chunks and the verdicts
    fold in the pipeline)."""
    lines = [f"Task #{task.id}: {task.title}"]
    if total > 1:
        lines.append(
            f"Review chunk {index + 1}/{total} — judge ONLY these files: "
            + ", ".join(chunk.paths)
        )
    lines.append(f"Conflicted files: {', '.join(f.path for f in report.files)}")
    lines.append("")
    lines.append("The resolver's own report of what it did:")
    lines.append(agent_report[:4000])
    lines.append("")
    lines.append(
        f"=== diff(resolved, ours) — changes relative to the task "
        f"branch {report.branch} ==="
    )
    lines.append(chunk.ours or "(no differences)")
    lines.append("")
    lines.append(
        f"=== diff(resolved, theirs) — changes relative to the base "
        f"branch {report.base_branch} ==="
    )
    lines.append(chunk.theirs or "(no differences)")
    lines.append("")
    if total > 1:
        lines.append(
            "Deliver your verdict with set_review — scope: only the "
            "chunk's files named above (the remaining conflicted files "
            "are reviewed in separate chunks)."
        )
    else:
        lines.append("Deliver your verdict with set_review.")
    return "\n".join(lines)


def _review_chunk_map(chunks: list[ReviewChunk]) -> str:
    """Header of a multi-chunk review report: which chunk covered which
    files — the service-event reader must be able to reconstruct the
    scope behind every per-chunk verdict."""
    lines = [f"review split into {len(chunks)} chunks:"]
    for i, chunk in enumerate(chunks):
        lines.append(f"  chunk {i + 1}: {', '.join(chunk.paths)}")
    return "\n".join(lines) + "\n\n"


class PipelineResolverLayer(CoreState):
    # -- triggering ----------------------------------------------------------------

    def _parked_merge_exists(self, task_id: str) -> bool:
        """Whether a conflicted merge is actually parked for `task_id`:
        the engine's persisted triage report (`.drcode/merge_reports/`),
        or a half-done merge (MERGE_HEAD set) in a still-existing
        worktree. The no-op guard of the resolver round (see run_resolver)
        reads it — blocking git, call via `asyncio.to_thread`."""
        if report_path(self.project_root, task_id).exists():
            return True
        wt = self.task_env.worktree_path(task_id)
        return wt.is_dir() and in_progress(wt)

    def _resolver_start_refusal(self, task_id: str) -> str | None:
        """Why a resolver round must NOT start for `task_id` right now
        (None = it may): worktree isolation is active (or the task is
        PINNED to a worktree from before the setting was turned off — its
        parked merge lives in its own worktree regardless of the live
        setting), the RESOLVER role resolves to a model (it falls back to
        the coding model) and no round is already running. The single
        guard behind both trigger paths: `_maybe_start_resolver` (the
        automatic markers, silent) and `launch_resolver` (the
        user-triggerable one, which turns the reason into a toast)."""
        if self._resolver_running:
            return "an automatic resolution round is already running"
        task = self.manager.tasks.get(task_id)
        if task is None:
            return "no such task"
        if not self.task_env.active and task.isolation_pinned is not True:
            return "worktree isolation is off"
        if not self.resolved_config(AgentRole.RESOLVER).is_ready:
            return "no model is configured for the resolver role"
        return None

    def _maybe_start_resolver(self, task_id: str) -> None:
        """Start a resolver round for `task_id` when the situation calls
        for one (see `_resolver_start_refusal`). Triggered from the
        hook-merge failure marker and the startup scan; silently does
        nothing otherwise (a conflict without a model is business as
        usual — the human path)."""
        if self._resolver_start_refusal(task_id) is None:
            self.run_resolver(task_id)

    def launch_resolver(self, task_id: str) -> bool:
        """The user-triggerable half of the resolver trigger (the
        ConflictScreen's ``A`` key and the unmerged list's ``a``): the
        same guard as the automatic marker, but a refusal is announced —
        a user pressed a key and must learn why nothing started. Returns
        True when the round was started (the caller closes its screen —
        the round's progress lives in the task's chat)."""
        reason = self._resolver_start_refusal(task_id)
        if reason is not None:
            self.host.host_notify(
                f"Cannot start the automatic resolver for #{task_id}: {reason}",
                title="Merge",
                severity="warning",
                timeout=7,
            )
            return False
        self.run_resolver(task_id)
        return True

    @work(exclusive=True, group="resolver")
    async def run_resolver(self, task_id: str) -> None:
        """One resolver round: prepare the conflicted merge, run the
        resolver agent, then the mandatory verification funnel."""
        self._resolver_running = True
        self._resolver_task_id = task_id
        self.refresh_sidebar()
        self._resolver_generation += 1
        run_key: tuple[str, str, int] = (
            "resolver",
            task_id,
            self._resolver_generation,
        )
        task = self.manager.get(task_id)
        # A network-retry continues the current round in place instead of
        # starting a new one (mirrors the tester's resume flag). A
        # repair round is the same continuation idea after a 'lost'
        # intent-review verdict: the resolved tree stays, the review
        # report is fed into the SAME transcript as a new user turn.
        resume_round = task_id in self._resolver_resume_round
        self._resolver_resume_round.discard(task_id)
        repair_prompt = self._resolver_repair_round.pop(task_id, None)
        continuation_round = resume_round or repair_prompt is not None
        run: AgentRun[ResolverDeps, str] | None = None
        cancelled = False
        # Statistics: when the report turns out genuinely conflicted, the
        # round is timed from here to its conclusion (success, human
        # hand-off) — resolving is not a status, the dwell lives only
        # here. `concluded` marks an invocation that ENDED the logical
        # round (a scheduled retry continues it in the next invocation:
        # its elapsed time still accumulates, the round is counted once).
        round_t0: float | None = None
        concluded = False
        # True when the except-branch scheduled a retry: the finally-block
        # must then leave the retry state alone (needs_human may be a
        # stale flag from a PREVIOUS round — popping here would cancel the
        # retry the same moment it was scheduled, the a6a6df25 silence).
        scheduled_retry = False
        # Initialized before the try: the handlers below reference them
        # even when the failure happened before ensure_root/base_branch.
        root: Path | None = None
        base = ""
        # The scope guard's inputs, filled once the round reaches the
        # agent stage (the finally-block's hygiene report is gated on
        # reached_agent — a round refused by prepare never touched the
        # tree, its pre-existing dirt is not the resolver's doing).
        triage_paths: list[str] = []
        baseline: set[str] = set()
        reached_agent = False
        try:
            self._current_tool[task_id] = None
            self._touch_activity(task_id)
            # A trigger that raced a merge: the audit re-stamp / startup
            # scan acted on a batched mergedness snapshot, and between
            # that snapshot and this round the hook (or a hand) merged
            # the branch into the base. Re-read mergedness FRESH (no
            # batch cache) BEFORE touching anything: a branch that no
            # longer exists or is fully reachable from the base has
            # nothing to resolve — skip the round without ensure_root
            # (which would re-create a worktree removed at completion)
            # and without prepare (which would park a pointless merge
            # commit, re-flagging the branch as unmerged). A scheduled
            # retry (resume_round) continues its in-flight round instead
            # — its branch cannot have become reachable on its own.
            if not resume_round and (
                await asyncio.to_thread(self.task_env.unmerged_commits, task_id) == 0
            ):
                if task.hook_merge_failed:
                    task.hook_merge_failed = False
                    task.touch()
                    self.manager.storage.save(task)
                self._show_event(
                    task,
                    f"Conflict resolver: {self.task_env.branch(task_id)} is "
                    f"already merged — resolver round skipped.",
                    kind="workflow",
                )
                return
            # A COMPLETED task whose worktree is GONE and whose merge left
            # nothing parked — no triage report and no half-done merge —
            # has nothing for the resolver to re-detect. Running on would
            # only `ensure_root` a worktree back into existence
            # (worktree_removal: on_completion had removed it) and let
            # prepare plant a meaningless "Merge <base> into drcode/<id>"
            # commit on the branch — which then counts as unmerged again
            # and lures a manual ff for work that was already merged.
            # Skip instead. A SURVIVING worktree is deliberately not
            # skipped here: prepare's base-into-worktree merge may park a
            # real conflict that this round exists to resolve.
            if (
                task.status is TaskStatus.COMPLETED
                and not self.task_env.worktree_path(task_id).is_dir()
                and not (await asyncio.to_thread(self._parked_merge_exists, task_id))
            ):
                self._show_event(
                    task,
                    "Conflict resolver: nothing to resolve — no parked "
                    "merge and no triage report; the worktree is not "
                    "recreated.",
                    kind="workflow",
                )
                return
            # The resolver works in the task's worktree, resolved against
            # its isolation PIN (see run_agent): a mid-task toggle of the
            # setting must not move the parked merge out from under it.
            pinned_before = task.isolation_pinned is not None
            root, iso_note = await asyncio.to_thread(
                self.task_env.ensure_root, task_id, task
            )
            self._isolation_pin_note(task, save=not pinned_before)
            if iso_note:
                self._show_event(task, f"Task isolation unavailable: {iso_note}")
                return
            base = await asyncio.to_thread(self.task_env.base_branch)
            # Prepare (or re-detect) the parked conflict. A merge the
            # workflow hook already parked is re-triaged in place.
            report = await asyncio.to_thread(
                prepare, self.project_root, task_id, root, base
            )
            if not report.ok:
                self._show_event(
                    task, f"Conflict resolver: {report.message}", kind="workflow"
                )
                return
            if not report.conflicted:
                # Nothing left for the agent: the engine concluded the
                # merge (clean, or via the safe auto-resolvers) or it was
                # already up to date. The branch is ready — fast-forward
                # it from the checkout (the hook or the user).
                task.needs_human = False
                task.touch()
                self.manager.storage.save(task)
                self._show_event(
                    task, f"Conflict resolver: {report.summary()}", kind="workflow"
                )
                return
            if repair_prompt is not None:
                # The targeted-repair continuation: the resolution is
                # still in the worktree, the review report rides in the
                # prompt — say so, instead of the fresh-start text.
                self._show_event(
                    task,
                    f"Conflict resolver: continuing the round with a "
                    f"targeted repair (repair "
                    f"{self._resolver_repair_count.get(task_id, 0)} of "
                    f"{_MAX_REPAIR_ROUNDS}) — the intent review found "
                    f"lost work, the resolution stays in the worktree.",
                    kind="workflow",
                )
            elif resume_round:
                # A network-retry continues the interrupted round in place
                # (resume via `resumable_history`) — say so, instead of the
                # fresh-start text that made consecutive 429 retries look
                # like full resolver restarts.
                self._show_event(
                    task,
                    f"Conflict resolver: resuming after a network error — "
                    f"{len(report.remaining)} conflicted file(s) left.",
                    kind="workflow",
                )
            else:
                self._show_event(
                    task,
                    f"Conflict resolver: {len(report.remaining)} conflicted "
                    f"file(s) — starting an automatic resolution round.",
                    kind="workflow",
                )
            round_t0 = time.monotonic()
            reached_agent = True

            # -- the agent round -----------------------------------------------------
            if not continuation_round:
                # A FRESH round on a (re)deployed conflict: a new
                # prepare/triage means a new conflict — the repair budget
                # resets (the counter counts repairs per conflict, not
                # per round).
                self._resolver_repair_count.pop(task_id, None)
                task.resolver_messages.append([])
                # The round's dirty-paths baseline (the scope guard,
                # c3b17718): everything dirty NOW predates the agent —
                # the merge's own staging plus pre-existing untracked
                # files. Kept across scheduled retries and repair
                # continuations, so a continuation round still counts the
                # interrupted attempt's own edits as its own; popped when
                # the round concludes.
                baseline = await asyncio.to_thread(dirty_paths, root)
                self._resolver_round_baseline[task_id] = baseline
                # A fresh round starts with a clean threshold-livelock
                # slate (see run_tester / _threshold_observation).
                self._resolver_threshold_compactions.pop(task_id, None)
                self._threshold_under_obs.pop(task_id, None)
                if self.viewed_id == task_id:
                    self._write_history_line("resolving-sep", "")
            agent = build_resolver_agent(
                self.resolved_config(AgentRole.RESOLVER),
                root=root,
                # The real scratch path is baked into the write tools'
                # descriptions (no literal <task-id> placeholder for the
                # model to copy verbatim into a path argument).
                task_id=task_id,
            )
            deps = ResolverDeps(
                root=root,
                manager=self.manager,
                task_id=task_id,
                scratch=scratch_dir(task_id),
            )
            prompt = build_agent_context(
                self.project_root,
                task.title,
                "\n\n".join(
                    p for p in all_user_prompts(task.messages) if p != task.title
                ),
                report,
                root,
            )
            # Which turn the agent gets: a fresh round re-sends the full
            # assignment; a network-retry resume replays the stored
            # transcript with no new text; a repair round appends the
            # review report to the SAME transcript (task.resolver_messages
            # [-1]) as a new user turn — the point-fix continues in place.
            has_transcript = bool(task.resolver_messages and task.resolver_messages[-1])
            repair_continuation = repair_prompt is not None and has_transcript
            if repair_continuation:
                text: str | None = repair_prompt
            elif resume_round and has_transcript:
                text = None
            else:
                text = prompt
            live_tokens: int | None = None
            view = NodeStreamView(self, task_id, record_tps=False, track_dirty=False)
            while True:
                history: list[Any] | None = None
                if text is None or repair_continuation:
                    history = resumable_history(task.resolver_messages[-1])
                paused = False
                async with agent.iter(
                    text,
                    message_history=history,
                    deps=deps,
                    usage_limits=UsageLimits(request_limit=self.config.request_limit),
                ) as run:
                    async for node in run:
                        if Agent.is_model_request_node(node):
                            view.begin_node()
                            async with node.stream(run.ctx) as node_stream:
                                try:
                                    async for event in node_stream:
                                        view.feed(event)
                                finally:
                                    # A mid-stream death (network error,
                                    # abort) skips the normal node end:
                                    # stop the spinners here, or they keep
                                    # animating through the retry backoff
                                    # (and the outage wait).
                                    view.finish_node()
                            live_tokens = run_context_tokens(run)
                            if live_tokens is not None:
                                self._live_ctx_resolver[task_id] = live_tokens
                                if self.viewed_id == task_id:
                                    self.refresh_task_status()
                            self._record_session_usage(
                                last_response_usage(run),
                                AgentRole.RESOLVER,
                                self.resolved_model_key(AgentRole.RESOLVER),
                            )
                            self._note_network_recovered(task, self._resolver_net_count)
                            self._resolver_net_count.pop(task_id, None)
                        elif Agent.is_call_tools_node(node):
                            async with node.stream(run.ctx) as stream:
                                async for event in stream:
                                    if isinstance(event, FunctionToolCallEvent):
                                        self._touch_activity(
                                            task_id, event.part.tool_name
                                        )
                                    elif isinstance(event, FunctionToolResultEvent):
                                        self._tool_call_finished(
                                            event.part.tool_call_id
                                        )
                                        self._touch_activity(task_id, done=True)
                                    if self.viewed_id == task_id and isinstance(
                                        event, FunctionToolCallEvent
                                    ):
                                        args = event.part.args
                                        args_str = (
                                            args
                                            if isinstance(args, str)
                                            else json.dumps(
                                                args or {}, ensure_ascii=False
                                            )
                                        )
                                        call_text = (
                                            f"resolver: {event.part.tool_name}"
                                            f"({args_str})"
                                        )
                                        block = self.host.host_new_tool_call_block(
                                            call_text
                                        )
                                        self._tool_call_started(
                                            run_key,
                                            task_id,
                                            event.part.tool_call_id,
                                            call_text,
                                            block,
                                        )
                                        block.start_spinner()
                                        self._write_chat(
                                            block,
                                            note=("tool-call", call_text, None),
                                        )
                            # Mid-round compaction gate at a tool boundary
                            # (the only safe stopping point), measured
                            # against the RESOLVER role's resolved config.
                            tokens = self._fill_over_threshold(
                                live_tokens,
                                task.resolver_messages[-1],
                                AgentRole.RESOLVER,
                            )
                            # Progress-gated threshold-livelock reset (a
                            # single under-threshold observation proves
                            # nothing — see _threshold_observation).
                            self._threshold_observation(
                                task,
                                tokens,
                                self._resolver_threshold_compactions,
                            )
                            hard_due = (
                                tokens is not None
                                and self._fill_over_hard_limit(
                                    live_tokens,
                                    task.resolver_messages[-1],
                                    AgentRole.RESOLVER,
                                )
                                is not None
                            )
                            if tokens is not None and (
                                task.id not in self._resolver_compaction_failed
                                or (
                                    hard_due
                                    and task.id
                                    not in self._resolver_hard_limit_exhausted
                                )
                            ):
                                self._persist_resolver_messages(task, run)
                                paused = True
                                break
                        # Keep the stored history current (same as the tester).
                        self._persist_resolver_messages(task, run)
                if paused:
                    if await self._maybe_compact_resolver_round(task, live_tokens):
                        live_tokens = None
                    text = None
                    continue
                if run.result is not None and self.viewed_id == task_id:
                    self.view_task(task_id)
                break

            # -- the scope guard -------------------------------------------------------
            # Before ANY verdict is applied: every worktree change outside
            # the triage report's set is a scope expansion (the c3b17718
            # incident — a foreign feature ported into the resolution) —
            # it would bypass the funnel's checks AND the concluding
            # commit's staging. The round is rejected outright (rollback +
            # needs_human) with the files named in the service events;
            # the funnel does not run on a tree the report never covered.
            triage_paths = [f.path for f in report.files]
            extra = await asyncio.to_thread(
                out_of_scope_changes,
                root,
                triage_paths,
                self._resolver_round_baseline.get(task_id, set()),
            )
            if extra:
                verdict_record = deps.verdicts[-1] if deps.verdicts else None
                outcome = ResolverOutcome(verdict="needs-human")
                outcome.agent_resolved = bool(
                    verdict_record and verdict_record.resolved
                )
                outcome.agent_report = verdict_record.report if verdict_record else ""
                outcome.scope_violations = extra
            else:
                # -- the verification funnel -------------------------------------------
                outcome = await self._verify_resolution(task, root, report, deps)
            repaired = await self._apply_resolver_outcome(
                task, root, base, outcome, triage_paths
            )
            if not repaired:
                concluded = True
        except asyncio.CancelledError:
            cancelled = True
            self._persist_resolver_messages(task, run)
            if self.viewed_id == task_id:
                self.view_task(task_id)
            raise
        except Exception as e:  # noqa: BLE001 - resolver errors must be visible
            self._persist_resolver_messages(task, run)
            if _is_context_overflow_error(e):
                # Deterministic rejection (HTTP 400 "context length
                # exceeded"): every replay of the same round is guaranteed
                # the same 400, so the retry budget below would only delay
                # the inevitable — hand the parked conflict to a human now.
                self._show_event(
                    task,
                    f"The resolver model rejects the round with a context "
                    f"overflow ({type(e).__name__}) — retrying cannot help, "
                    "a human takes over.",
                )
                await self._give_up_to_human(task, root, base)
                concluded = True
                return
            network = _is_retryable_network_error(e)
            if network:
                attempt = self._resolver_net_count.get(task_id, 0) + 1
                self._resolver_net_count[task_id] = attempt
                self._net_retries_total += 1
                budget = _MAX_RESOLVER_NETWORK_ATTEMPTS
            else:
                self._resolver_net_count.pop(task_id, None)
                attempt = self._resolver_fail_count.get(task_id, 0) + 1
                self._resolver_fail_count[task_id] = attempt
                budget = _MAX_RESOLVER_ATTEMPTS
            if attempt >= budget:
                self._show_event(
                    task,
                    f"Conflict resolver failed repeatedly "
                    f"({type(e).__name__}: {e}) — a human takes over.",
                )
                await self._give_up_to_human(task, root, base)
                concluded = True
                return
            delay = min(_RETRY_BASE_DELAY_S * 2 ** (attempt - 1), 300)
            self._resolver_retry_at[task_id] = time.monotonic() + delay
            # The round continues where it stopped (the transcript is
            # kept) — no re-sent prompt.
            self._resolver_resume_round.add(task_id)
            scheduled_retry = True
            self._show_event(
                task,
                f"Conflict resolver failed ({type(e).__name__}: {e}) — "
                f"retrying in {int(delay)} s",
            )
            self._schedule_pipeline_timer(delay, self._retry_resolver)
            return
        finally:
            self._stop_run_tools(run_key)
            self._resolver_running = False
            self._resolver_task_id = None
            self._live_ctx_resolver.pop(task_id, None)
            self.refresh_status()
            self.refresh_sidebar()
            if round_t0 is not None and not cancelled:
                self._flush_resolver_time(task, time.monotonic() - round_t0, concluded)
            if reached_agent and root is not None:
                # End-of-round hygiene (the c3b17718 incident): whatever
                # the round leaves uncommitted OUTSIDE the conflicted set
                # gets a named trace in the task's chat — success,
                # verification failure, error/retry and cancellation
                # alike. A silent dirty tail used to block the accept
                # hook hours later with the cause recoverable only from
                # the transcript.
                self._report_resolver_leftovers(
                    task,
                    root,
                    triage_paths,
                    # A resumed round reports against the fresh round's
                    # baseline (still in the dict — a retry never
                    # concludes); the local covers every other path.
                    self._resolver_round_baseline.get(task_id, baseline),
                )
            if concluded:
                self._resolver_round_baseline.pop(task_id, None)
            if not cancelled:
                # Clean the retry state only when THIS attempt handed the
                # task to a human (needs_human freshly set, no retry
                # pending). A stale needs_human from a previous round must
                # not erase a just-scheduled retry.
                if task.needs_human and not scheduled_retry:
                    self._resolver_fail_count.pop(task_id, None)
                    self._resolver_retry_at.pop(task_id, None)
                    self._resolver_resume_round.discard(task_id)
                self._advance()

    def _flush_resolver_time(self, task: Task, elapsed: float, concluded: bool) -> None:
        """Fold one resolver invocation's wall time into the statistics
        (see drcode.stats): the task's cumulative `resolver_seconds` plus
        the day bucket — a full second-granularity chunk goes to the day
        the invocation ended. A round is counted once: a scheduled retry
        continues the same logical round, so only a concluded invocation
        (outcome applied / handed to a human) bumps `resolver_rounds`.
        The task save is explicit: the conclusion paths that persist the
        task may already have run (the finally here is later)."""
        if elapsed <= 0:
            return
        task.resolver_seconds += elapsed
        self.manager.stats.add_time("resolver", elapsed, rounds=1 if concluded else 0)
        if task.id in self.manager.tasks:
            self.manager.storage.save(task)

    def _retry_resolver(self) -> None:
        """A scheduled resolver retry: pick the task waiting out its
        backoff (if it still exists) and restart its round."""
        now = time.monotonic()
        # A hair of tolerance: a timer firing marginally before its
        # deadline must not leave the entry stranded with no timer left
        # to ever pick it up.
        due = [tid for tid, at in self._resolver_retry_at.items() if at <= now + 0.05]
        if not due:
            return
        task_id = next(
            (tid for tid in due if tid in self.manager.tasks),
            None,
        )
        if task_id is None:
            # Due entries whose tasks no longer exist would otherwise
            # linger forever and swallow every later timer fire silently.
            self.host.host_log(
                f"resolver retry timer fired, but scheduled task(s) "
                f"{due} no longer exist — dropping them"
            )
            for tid in due:
                self._resolver_retry_at.pop(tid, None)
            return
        self._resolver_retry_at.pop(task_id, None)
        self._maybe_start_resolver(task_id)

    # -- verification funnel ---------------------------------------------------------

    async def _verify_resolution(
        self,
        task: Task,
        root: Path,
        report: MergeReport,
        deps: ResolverDeps,
    ) -> ResolverOutcome:
        """Run the three verification levels over the agent's resolution.
        Short-circuits on the first failure (the outcome carries what was
        checked so far)."""
        outcome = ResolverOutcome(verdict="needs-human")
        verdict = deps.verdicts[-1] if deps.verdicts else None
        outcome.agent_resolved = bool(verdict and verdict.resolved)
        outcome.agent_report = verdict.report if verdict else ""
        if not outcome.agent_resolved:
            return outcome
        # Level 1: mechanics. The marker scan inside mechanical_checks
        # runs over the triage report's conflicted paths (staged or not)
        # — a prose-only resolution fails here in seconds, before the
        # compile/ruff/pyright/pytest battery burns on a still-conflicted
        # tree (the a6a6df25 incident class).
        outcome.checks = await asyncio.to_thread(
            mechanical_checks, root, [f.path for f in report.files]
        )
        if any(not c.ok and not c.skipped for c in outcome.checks):
            return outcome
        # Level 2: tests.
        outcome.tests = await asyncio.to_thread(run_test_suite, root)
        if not outcome.tests.ok and not outcome.tests.skipped:
            return outcome
        # Level 3: intent-preservation review — window-aware and chunked.
        # The diff budget derives from the reviewer model's context window
        # (the legacy REVIEW_DIFF_CHAR_CAP is only the unknown-window
        # fallback); when the diffs still overflow it, the review runs once
        # per chunk of conflicted paths and the verdicts fold: any 'lost'
        # chunk → 'lost', else any 'unsure' → 'unsure', else 'intact'.
        cfg = self.resolved_config(AgentRole.RESOLVER)
        chunks = await asyncio.to_thread(
            review_diff_chunks,
            root,
            [f.path for f in report.files],
            review_char_budget(cfg.max_context_tokens),
        )
        reviewer = build_resolver_reviewer_agent(cfg, root=root)
        rdeps = ResolverDeps(
            root=root,
            manager=self.manager,
            task_id=task.id,
            scratch=scratch_dir(task.id),
        )
        results: list[tuple[str, str]] = []
        for index, chunk in enumerate(chunks):
            prompt = _review_chunk_prompt(
                task, report, outcome.agent_report, chunk, index, len(chunks)
            )
            label, chunk_report, stop = await self._run_reviewer_pass(
                task, prompt, reviewer, rdeps, index + 1, len(chunks)
            )
            results.append((label, chunk_report))
            if stop:
                break
        outcome.review_verdict, outcome.review_report = aggregate_review_verdicts(
            results
        )
        if len(chunks) > 1:
            outcome.review_report = _review_chunk_map(chunks) + outcome.review_report
        if outcome.review_verdict == "intact":
            outcome.verdict = "resolved"
        return outcome

    async def _run_reviewer_pass(
        self,
        task: Task,
        prompt: str,
        reviewer: Agent[Any, str],
        rdeps: ResolverDeps,
        chunk_index: int,
        chunk_total: int,
    ) -> tuple[str, str, bool]:
        """One intent-review chunk's reviewer pass with its own retry
        budget: retryable failures (429s, transport errors) are retried
        IN PLACE with the same capped exponential backoff as the round —
        the expensive resolver round must not die to one dropped HTTP
        call in the cheap tail — while a deterministic context-overflow
        400 or any other non-retryable error goes straight to an 'unsure'
        verdict (replaying the same oversized prompt is guaranteed the
        same 400). Returns (label, report, stop): `stop` is True when the
        pass failed terminally, so the remaining chunks must not be
        reviewed either (a failure of this class will not heal itself on
        them, and the folded verdict already blocks the commit)."""
        failures: list[str] = []
        result = None
        prefix = f"chunk {chunk_index}/{chunk_total}: "
        for attempt in range(1, _MAX_REVIEW_ATTEMPTS + 1):
            try:
                result = await reviewer.run(
                    prompt,
                    deps=rdeps,
                    usage_limits=UsageLimits(request_limit=self.config.request_limit),
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - a failed review blocks the commit
                failure = f"{type(e).__name__}: {e}"
                failures.append(failure)
                if _is_context_overflow_error(e):
                    # Deterministic: every replay of the same prompt is
                    # guaranteed the same 400 — do not burn the budget.
                    self._show_event(
                        task,
                        f"Intent review failed with a context overflow "
                        f"({failure}) — retrying cannot help, the round is "
                        "treated as unsure.",
                        kind="workflow",
                    )
                    return (
                        "unsure",
                        prefix
                        + f"the reviewer pass failed with a context "
                        f"overflow ({failure}) — retrying cannot help, "
                        "treated as unsure",
                        True,
                    )
                if not _is_retryable_network_error(e):
                    self._show_event(
                        task,
                        f"Intent review failed ({failure}) — the error is "
                        "not transient, the round is treated as unsure.",
                        kind="workflow",
                    )
                    return (
                        "unsure",
                        prefix + f"the reviewer pass failed ({failure}) — treated as unsure",
                        True,
                    )
                if attempt >= _MAX_REVIEW_ATTEMPTS:
                    self._show_event(
                        task,
                        f"Intent review failed {attempt} times "
                        f"(last: {failure}) — retries exhausted, the round "
                        "is treated as unsure.",
                        kind="workflow",
                    )
                    return (
                        "unsure",
                        prefix
                        + f"the reviewer pass failed {attempt} times "
                        f"({'; '.join(failures)}) — retries exhausted, "
                        "treated as unsure",
                        True,
                    )
                delay = min(_RETRY_BASE_DELAY_S * 2 ** (attempt - 1), 300)
                self._show_event(
                    task,
                    f"Intent review failed ({failure}) — retrying in "
                    f"{int(delay)} s (attempt "
                    f"{attempt + 1}/{_MAX_REVIEW_ATTEMPTS})",
                    kind="workflow",
                )
                await asyncio.sleep(delay)
                continue
            break
        assert result is not None  # the loop only exits via break or return
        self._record_session_usage(
            result.usage,
            AgentRole.RESOLVER,
            self.resolved_model_key(AgentRole.RESOLVER),
        )
        rverdict = rdeps.verdicts[-1] if rdeps.verdicts else None
        return (
            rverdict.label if rverdict else "unsure",
            rverdict.report if rverdict else "",
            False,
        )

    async def _apply_resolver_outcome(
        self,
        task: Task,
        root: Path,
        base: str,
        outcome: ResolverOutcome,
        paths: list[str],
    ) -> bool:
        """Commit a fully verified resolution — schedule a targeted repair
        round on an actionable 'lost' review verdict — or restore the
        pristine conflict and mark the task needs_human. `paths` are the
        resolved conflict paths (the triage report's files — the same set
        the funnel verified): they are staged here (`git add`) because the
        resolver agent never touches the index, and `finish` refuses
        while unmerged entries remain (the bcd4a7ad incident: a fully
        verified resolution lost to an unstaged index).

        Returns True when a repair round was scheduled INSTEAD of the
        rollback: the resolution stays in the worktree, the review report
        continues the SAME resolver transcript, and the caller must treat
        its invocation as UNconcluded (the logical round continues)."""
        branch = self.task_env.branch(task.id)
        if outcome.verdict == "resolved":
            staged, stage_msg = await asyncio.to_thread(stage_resolved, root, paths)
            if staged:
                message = (
                    f"Merge {base} into {branch}: resolve conflicts "
                    f"(drcode automatic resolver)"
                )
                ok, msg = await asyncio.to_thread(conclude, root, message)
            else:
                ok, msg = False, f"staging the resolved files failed: {stage_msg}"
            if ok:
                outcome.committed = True
                task.needs_human = False
                task.touch()
                self.manager.storage.save(task)
                # The conflict is concluded — its triage report is
                # obsolete (a leftover would keep the footer's Merge…
                # entry alive).
                await asyncio.to_thread(remove_report, self.project_root, task.id)
                self._show_event(
                    task,
                    f"Conflict resolution verified and committed on "
                    f"{branch} — the merge into {base} is now a "
                    f"fast-forward (`git merge --ff-only {branch}`).\n"
                    f"{outcome.summary()}",
                    kind="workflow",
                )
                return False
            outcome.review_report = (
                f"the concluding commit failed: {msg}\n{outcome.review_report}"
            )
        # The repair fork (before any rollback): a 'lost' review verdict
        # that NAMES its finding gets a bounded number of targeted repair
        # rounds — the ~20-minute resolution is not thrown away over a
        # one-line fix the reviewer itself described (the review report
        # continues the SAME resolver transcript; the FULL funnel re-runs
        # afterwards). 'unsure', a scope expansion, a vague/empty report
        # and a spent budget all keep the old rollback.
        repair = self._plan_repair(task, outcome, paths)
        if repair is not None:
            # A stale needs_human from an earlier round must not make the
            # finally-block wipe the repair schedule the moment it was
            # set (the a6a6df25 class).
            task.needs_human = False
            task.touch()
            self.manager.storage.save(task)
            self._resolver_repair_round[task.id] = repair
            self._resolver_retry_at[task.id] = time.monotonic() + _REPAIR_RETRY_DELAY_S
            self._schedule_pipeline_timer(_REPAIR_RETRY_DELAY_S, self._retry_resolver)
            self._show_event(
                task,
                f"Automatic conflict resolution FAILED the intent review — "
                f"a lost change was named. Repair round "
                f"{self._resolver_repair_count.get(task.id, 0)} of "
                f"{_MAX_REPAIR_ROUNDS} continues the resolution in the "
                f"worktree (nothing is rolled back); the full funnel "
                f"re-runs afterwards.\n{outcome.summary()}",
                kind="workflow",
            )
            return True
        # Anything short of a full pass: nothing committed, conflict
        # restored, human takes over.
        restored = await asyncio.to_thread(
            restore_conflict,
            self.project_root,
            task.id,
            root,
            base,
            outcome.scope_violations,
        )
        task.needs_human = True
        task.touch()
        self.manager.storage.save(task)
        # The rollback claim must match reality: only a verified-clean
        # restore may say "untouched" — a failed one leaves the worktree
        # possibly carrying uncommitted resolver edits (the 9c3236e8
        # incident: "the conflict is back untouched" next to a
        # could-not-rollback note, while 6 dirty files blocked the merge).
        tail = (
            " — NOTHING was committed, the pristine conflict is back in "
            "the worktree, a human must take over."
            if restored
            else (
                " — NOTHING was committed, but the worktree could NOT be "
                "rolled back cleanly: uncommitted resolver edits may remain "
                "in it alongside the conflict; inspect it by hand and clean "
                "it up before resolving."
            )
        )
        # A scope-expansion rejection never reached the funnel — the
        # event must say WHY the round failed (the c3b17718 incident's
        # "successfully concluded" round hid a foreign edit).
        headline = (
            "Automatic conflict resolution REJECTED — the resolver changed "
            "files outside the conflicted set (scope expansion)"
            if outcome.scope_violations
            else "Automatic conflict resolution FAILED verification"
        )
        self._show_event(
            task,
            f"{headline}{tail}\n{outcome.summary()}",
            kind="workflow",
        )
        self._suggest_conflict_screen(task)
        return False

    def _plan_repair(
        self, task: Task, outcome: ResolverOutcome, paths: list[str]
    ) -> str | None:
        """The repair-eligibility fork: the repair prompt when the round
        qualifies for a targeted repair round, None for the rollback.
        Qualification (ALL of it): the round did not conclude successfully
        (a commit-machinery failure after an 'intact' verdict rolls back —
        the tree is fine, the git state is not), no scope expansion, the
        review verdict is exactly 'lost' ('unsure' is not a finding), the
        report names something concrete, and the per-conflict repair
        budget is not spent. Incrementing the counter is deliberate:
        scheduling the round spends it."""
        if outcome.verdict == "resolved" or outcome.committed:
            return None
        if outcome.scope_violations:
            return None
        if outcome.review_verdict != "lost":
            return None
        report_text = (outcome.review_report or "").strip()
        if not _review_report_actionable(report_text, paths):
            return None
        used = self._resolver_repair_count.get(task.id, 0)
        if used >= _MAX_REPAIR_ROUNDS:
            return None
        self._resolver_repair_count[task.id] = used + 1
        return _build_repair_prompt(report_text, used + 1)

    async def _give_up_to_human(self, task: Task, root: Path | None, base: str) -> None:
        """The last-resort path (repeated failures): restore the pristine
        conflict and hand over — no verification funnel to report. With
        `root` None the failure happened before the worktree was even
        reached: nothing to restore, just flag the task."""
        restored = (
            await asyncio.to_thread(
                restore_conflict, self.project_root, task.id, root, base
            )
            if root is not None and base
            else False
        )
        task.needs_human = True
        task.touch()
        self.manager.storage.save(task)
        tail = (
            " — the pristine conflict is back in the worktree, a human "
            "must take over."
            if restored
            else (
                " — the worktree could NOT be rolled back cleanly: "
                "uncommitted resolver edits may remain in it alongside the "
                "conflict; inspect it by hand and clean it up before "
                "resolving."
            )
        )
        self._show_event(
            task,
            f"Automatic conflict resolution abandoned after repeated "
            f"failures{tail}",
            kind="workflow",
        )
        self._suggest_conflict_screen(task)

    def _report_resolver_leftovers(
        self,
        task: Task,
        root: Path,
        allowed: list[str],
        baseline: set[str],
    ) -> None:
        """End-of-round hygiene (the c3b17718 incident): name every
        uncommitted worktree change OUTSIDE the conflicted set in the
        task's service events. Deliberately SYNC (one fast `git status
        --porcelain`): it also runs on the cancellation path, where an
        await after the cancel is not safe to rely on. Never raises."""
        try:
            extra = out_of_scope_changes(root, allowed, baseline)
        except Exception:  # noqa: BLE001 - teardown hygiene must not crash
            return
        if not extra:
            return
        self._show_event(
            task,
            "Resolver round left uncommitted changes outside the conflicted "
            "set (a sign of scope expansion — inspect before resolving):\n"
            + "\n".join(f"- {p}" for p in extra),
            kind="workflow",
        )

    # -- persistence -----------------------------------------------------------------

    def _persist_resolver_messages(
        self, task: Task, run: AgentRun[ResolverDeps, str] | None
    ) -> None:
        """Snapshot the resolver transcript into the task's last resolver
        round (mirrors _persist_tester_messages)."""
        if run is None:
            return
        try:
            messages = run.all_messages()
        except Exception:  # noqa: BLE001 - state may be mid-teardown after cancel
            return
        if not task.resolver_messages:
            task.resolver_messages.append([])
        task.resolver_messages[-1] = serialize_messages(messages)
        self.manager.storage.save(task)
