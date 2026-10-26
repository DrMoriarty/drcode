"""The tester's verification run.

Split out of the former monolithic `tui/pipeline.py`: the retry
backoff schedule and the single-slot tester worker — round
continuation, streaming render, mid-round compaction pauses and the
network-retry/outage handling mirroring the agent's.
"""

from __future__ import annotations

import asyncio
import json
import time
from functools import partial
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRun
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)
from pydantic_ai.usage import UsageLimits
from textual import work

from ..agent import (
    TesterDeps,
    all_user_prompts,
    build_tester_agent,
    last_response_usage,
    resumable_history,
    run_context_tokens,
)
from ..github import issue_commit_hint
from ..models import TaskStatus, VerificationStatus
from ..providers import AgentRole
from ..tools import scratch_dir
from .blocks import ToolCallBlock
from .pipeline_stream import NodeStreamView
from .state import DrCodeState
from .usage import (
    NETWORK_RETRIES,
    _in_outage,
    _is_retryable_network_error,
    _network_retry_delay,
)


class PipelineTesterLayer(DrCodeState):
    # -- tester run ------------------------------------------------------------------

    def _schedule_tester_retry(self, task_id: str) -> None:
        """Backoff before the next tester attempt for a failed task.

        A TESTING task must not be dropped from the backlog until restart,
        so a failed tester run is retried with capped exponential backoff
        (the timer re-fires _advance(); under parallel slots the queue
        keeps flowing while the task waits out its window)."""
        count = self._tester_fail_count.get(task_id, 1)
        delay = min(5 * 2 ** (count - 1), 300)
        self._tester_retry_at[task_id] = time.monotonic() + delay
        self._schedule_pipeline_timer(delay, self._advance)

    @work(exclusive=True, group="tester")
    async def run_tester(self, task_id: str) -> None:
        """After any task completes, verify the project and label the task."""
        self._tester_task_id = task_id
        self._tester_generation += 1
        run_key: tuple[str, str, int] = ("tester", task_id, self._tester_generation)
        task = self.manager.get(task_id)
        # A network-retry continues the current testing round in place
        # instead of starting a new one (the flag is set by the network
        # handler below).
        # Ignored when the task left TESTING in the meantime: the round
        # would test stale state.
        resume_round = task_id in self._tester_resume_round and (
            task.status is TaskStatus.TESTING
        )
        self._tester_resume_round.discard(task_id)
        # Initialized before the try: build_tester_agent() itself may raise,
        # and the handlers below reference `run`.
        run: AgentRun[TesterDeps, str] | None = None
        cancelled = False
        try:
            if not self.resolved_config(AgentRole.TESTER).is_ready:
                self._tester_fail_count[task_id] = (
                    self._tester_fail_count.get(task_id, 0) + 1
                )
                self._schedule_tester_retry(task_id)
                self._show_event(
                    task,
                    "Tester skipped: model is not configured. Pick a model "
                    "for the tester role in Settings (or set DRCODE_MODEL / "
                    "DRCODE_BASE_URL env vars).",
                )
                return
            # Idempotent: _advance() sets this before starting the worker so
            # two _advance() calls in one tick cannot double-start the tester.
            self.tester_running = True
            self._current_tool[task_id] = None
            self._tps_samples.pop(task_id, None)
            self._touch_activity(task_id)
            self.refresh_status()
            # New testing round: created up front, filled by
            # _persist_tester_messages as the run progresses. An empty round
            # (crash before any snapshot) is skipped when rendering.
            if not resume_round:
                task.tester_messages.append([])
                # A fresh round starts with a clean threshold-livelock
                # slate: the previous round's counter (and its
                # under-threshold progress streak) must not fire the
                # guard on this round's first compactions (see
                # _threshold_observation).
                self._tester_threshold_compactions.pop(task_id, None)
                self._threshold_under_obs.pop(task_id, None)
                if self.viewed_id == task_id:
                    self._write_history_line("testing-sep", "")
            # The tester verifies the task in ITS worktree (per-task
            # worktree isolation, see run_agent): the round runs against
            # exactly the tree the development agent worked in, and the
            # tester's git policy commits onto the task's branch.
            # A worktree lost to a manual deletion is recreated here (the
            # branch with the task's commits is re-checked-out).
            root, iso_note = await asyncio.to_thread(self.task_env.ensure_root, task_id)
            if iso_note:
                self._show_event(task, f"Task isolation unavailable: {iso_note}")
            tester = build_tester_agent(
                self.resolved_config(AgentRole.TESTER),
                allow_commit=self.settings.git_policy.tester.commits,
                allow_push=self.settings.git_policy.tester.pushes,
                root=root,
                # The external review (`.drcode/workflow.py` on_review)
                # splits "the tester passed" from "the task is accepted":
                # a PASSED verdict then leaves the task TESTING.
                review_gate=(
                    self.workflow is not None and self.workflow.has("on_review")
                ),
            )
            deps = TesterDeps(
                root=root,
                manager=self.manager,
                task_id=task_id,
                scratch=scratch_dir(task_id),
                review_gate=(
                    self.workflow is not None and self.workflow.has("on_review")
                ),
                # The fix-loop cap: how many consecutive failed development
                # rounds (retry_here verdicts) the task gets before its
                # failure completes it as FAILED (and the tester creates
                # the bug report).
                max_dev_rounds=self.settings.max_dev_rounds,
            )
            try:
                prompt = f"Verify completed task #{task.id}: {task.title}"
                # The tester needs the original assignment (to paraphrase it
                # into a commit message — see the git_commit step in its
                # prompt) plus every clarification the user sent while the
                # task was being worked on, so the verification targets the
                # CURRENT intent, not a stale initial one.
                user_prompts = [
                    p for p in all_user_prompts(task.messages) if p != task.title
                ]
                if user_prompts:
                    prompt += f"\n\nOriginal task:\n{user_prompts[0]}"
                if len(user_prompts) > 1:
                    prompt += (
                        "\n\nUser clarifications sent during development "
                        "(in order, latest last):\n"
                        + "\n\n".join(f"- {c}" for c in user_prompts[1:])
                    )
                # A GitHub-imported task must link its commit back to the
                # originating issue ("Fixes #N" keyword in the message).
                if task.github_issue is not None:
                    prompt += f"\n\n{issue_commit_hint(task.github_issue)}"
                # Custom workflow: context returned by on_development_done
                # (the opened PR, the pushed branch, CI links, …).
                dev_note = self._dev_done_notes.pop(task_id, None)
                if dev_note:
                    prompt += (
                        "\n\nProject workflow notes (from the on_development_done "
                        f"step):\n{dev_note}"
                    )
                # Round continuation loop: a testing round is a single
                # `tester.iter()` run, but it can make many model requests
                # (each `read_file`/`run_bash` result grows the context).
                # When the fill crosses the compaction threshold at a tool
                # boundary, the run is stopped, the ROUND transcript is
                # compacted in place (task.messages is never touched) and
                # the round resumes as a continuation turn from the
                # compacted transcript — otherwise a pathological tool
                # result could blow the model window before the tester
                # ever calls set_verification, looping the round in
                # fruitless retries.
                # A resumed round continues from the persisted round
                # transcript (no new prompt — the while-loop below builds
                # the continuation history from it); an empty round (the
                # failed request was never recorded) re-sends the
                # prompt as usual.
                text: str | None = (
                    None
                    if resume_round
                    and task.tester_messages
                    and task.tester_messages[-1]
                    else prompt
                )
                # Live context reading of the current round; the shared
                # _live_ctx_tokens map belongs to the agent transcript and
                # must not be polluted with the tester's.
                live_tokens: int | None = None
                # The shared streaming renderer (see pipeline_stream): the
                # tester variant records no tok/s, does not pin the chat
                # tail and does not track off-screen dirtiness (the round
                # re-renders from the transcript at its end).
                view = NodeStreamView(
                    self,
                    task_id,
                    record_tps=False,
                    follow=False,
                    track_dirty=False,
                )
                while True:
                    # Reset per attempt: if tester.iter() raises before the
                    # `as run` binding, the except handlers must not
                    # re-persist the PREVIOUS attempt's transcript over the
                    # (possibly compacted) round.
                    run = None
                    history: list[Any] | None = None
                    if text is None:
                        # Continuation of a paused round: resume from the
                        # stored round transcript (compacted, or — when the
                        # summarizer failed — the original one with its
                        # invalid tail trimmed).
                        history = resumable_history(task.tester_messages[-1])
                    paused = False
                    async with tester.iter(
                        text,
                        message_history=history,
                        deps=deps,
                        usage_limits=UsageLimits(
                            request_limit=self.config.request_limit
                        ),
                    ) as run:  # noqa: F811 - rebinding per attempt is intended
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                # Stream the round's reasoning/answer into
                                # the open chat view (the shared renderer,
                                # mirrors run_agent): everything the tester
                                # produces must already be on screen when
                                # the end-of-round re-render runs, so that
                                # re-render can go through the delta path
                                # (append the missing tail) instead of
                                # wiping and remounting the whole history.
                                # Every event is a sign of life (the view's
                                # _touch_activity): without it a long
                                # (minutes) report generation reads as
                                # "stalled" — and the stall watchdog aborts
                                # a perfectly healthy run.
                                view.begin_node()
                                async with node.stream(run.ctx) as node_stream:
                                    async for event in node_stream:
                                        view.feed(event)
                                view.finish_node()
                                live_tokens = run_context_tokens(run)
                                # Exact per-request usage of the completed
                                # model call, into the session totals.
                                self._record_session_usage(last_response_usage(run))
                                # A model call went through: the network-retry
                                # budget for this testing round starts over,
                                # so the NEXT network error counts from one
                                # again (mirrors the agent's per-request
                                # reset). A count past the budget = the round
                                # was in outage mode: announce the recovery
                                # before the reset.
                                self._note_network_recovered(
                                    task, self._tester_net_count
                                )
                                self._tester_net_count.pop(task_id, None)
                            elif Agent.is_call_tools_node(node):
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        if isinstance(event, FunctionToolCallEvent):
                                            self._touch_activity(
                                                task_id, event.part.tool_name
                                            )
                                        elif isinstance(event, FunctionToolResultEvent):
                                            # Result stops the spinner even when
                                            # the task is off-screen.
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
                                                f"tester: {event.part.tool_name}"
                                                f"({args_str})"
                                            )
                                            block = ToolCallBlock(call_text)
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
                                # Mid-round compaction gate: a tool boundary
                                # is the only safe stopping point (same as
                                # the agent's mid-run gate). Skipped when a
                                # previous compaction attempt failed, or a
                                # persistently broken summarizer would loop
                                # pause→resume forever — except at the
                                # HARD-LIMIT level, which fires regardless
                                # of `_compaction_failed` (a broken
                                # summarizer there escalates to the
                                # mechanical trim; unless the escalation
                                # proved unable to shrink the round —
                                # `_hard_limit_exhausted`, or the livelock
                                # guard stopped the gate after consecutive
                                # compactions that never landed under the
                                # hard limit — both flags set together).
                                tokens = self._fill_over_threshold(
                                    live_tokens,
                                    task.tester_messages[-1],
                                    AgentRole.TESTER,
                                )
                                # Under the threshold: credit progress
                                # toward the progress-gated reset of the
                                # threshold-livelock counter — a single
                                # observation proves nothing (the first
                                # boundary of a resumed round reads the
                                # compacted transcript's lagging live
                                # usage, which predates the tool result
                                # that re-crosses the threshold — see
                                # _threshold_observation). Over it: the
                                # streak breaks.
                                self._threshold_observation(
                                    task,
                                    tokens,
                                    self._tester_threshold_compactions,
                                )
                                hard_due = (
                                    tokens is not None
                                    and self._fill_over_hard_limit(
                                        live_tokens,
                                        task.tester_messages[-1],
                                        AgentRole.TESTER,
                                    )
                                    is not None
                                )
                                if tokens is not None and (
                                    task.id not in self._compaction_failed
                                    or (
                                        hard_due
                                        and task.id not in self._hard_limit_exhausted
                                    )
                                ):
                                    pct = round(
                                        tokens
                                        * 100
                                        / self.resolved_config(
                                            AgentRole.TESTER
                                        ).max_context_tokens
                                    )
                                    self._show_event(
                                        task,
                                        f"Context fill {pct}% — pausing the "
                                        "tester round to compact history…",
                                    )
                                    self._persist_tester_messages(task, run)
                                    paused = True
                                    break
                            # Keep the stored history current while the test
                            # runs: switching to the task mid-test shows the
                            # tester's work done so far.
                            self._persist_tester_messages(task, run)
                    if paused:
                        if await self._maybe_compact_tester_round(task, live_tokens):
                            # The round transcript was replaced by the
                            # original prompt plus the summary: the live
                            # token reading belongs to the old transcript —
                            # drop it so the counter falls back to the
                            # compacted one (mirrors _maybe_compact).
                            live_tokens = None
                        # Either way (compacted, or the summarizer failed
                        # and the gate is now disabled for this task) the
                        # round resumes as a continuation turn: no new
                        # text, history from the round transcript.
                        text = None
                        continue
                    if run.result is not None and self.viewed_id == task_id:
                        # Re-render from the persisted transcript: everything
                        # the round streamed live (separator, tool calls,
                        # reasoning, answer) is on screen and mirrored by the
                        # delta ledger, so this appends only the entries the
                        # live view is missing (usually nothing) instead of
                        # wiping and remounting the whole history.
                        self.view_task(task_id)
                    break
            except asyncio.CancelledError:
                # Aborted (or superseded): keep whatever the tester did so
                # far, then re-raise so the worker teardown stays clean.
                cancelled = True
                self._persist_tester_messages(task, run)
                if self.viewed_id == task_id:
                    # Watchdog abort (or esc) of a hung run: re-render from
                    # the persisted transcript, so the open view shows what
                    # was actually done instead of stale blocks (e.g. a tool
                    # call that never returned) until the user switches views.
                    # Sync only — no awaits inside a cancelled coroutine.
                    self.view_task(task_id)
                raise
            except Exception as e:  # noqa: BLE001 - show tester errors in the chat
                # A transient network error (transport failure like a
                # connection refused/timeout, or HTTP 429 rate limiting)
                # is transient: instead of restarting the whole testing
                # round, retry it in place (the task stays TESTING, the
                # round transcript is kept and the retry resumes it as a
                # continuation turn) after a backoff delay, at most
                # NETWORK_RETRIES times — past the budget the run enters
                # outage mode instead (mirrors run_agent's network
                # handling): the task keeps TESTING and keeps retrying on
                # the long outage backoff until the connection returns;
                # only NON-network errors reach the ordinary failure path
                # below.
                if _is_retryable_network_error(e):
                    attempt = self._tester_net_count.get(task_id, 0) + 1
                    self._tester_net_count[task_id] = attempt
                    self._net_retries_total += 1
                    outage = _in_outage(attempt)
                    delay = _network_retry_delay(attempt, outage)
                    if outage:
                        if attempt == NETWORK_RETRIES + 1:
                            self._show_event(
                                task,
                                f"Network outage: the model gateway has been "
                                f"unreachable for {NETWORK_RETRIES} tester "
                                f"retries ({type(e).__name__}) — the testing "
                                f"round is kept and waits out the outage, "
                                f"retrying in place until the connection "
                                f"returns.",
                            )
                        else:
                            self._show_event(
                                task,
                                f"Network outage: still unreachable "
                                f"({type(e).__name__}) — tester retry "
                                f"{attempt} in {int(delay)} s",
                            )
                    else:
                        self._show_event(
                            task,
                            f"Network error ({type(e).__name__}): tester retry "
                            f"{attempt}/{NETWORK_RETRIES} in {int(delay)} s",
                        )
                    self._persist_tester_messages(task, run)
                    self._tester_resume_round.add(task_id)
                    self._tester_retry_at[task_id] = time.monotonic() + delay
                    if outage:
                        # Outage mode: gate the retry behind the cheap TCP
                        # probe (the tick calls _advance when the host is
                        # back — _next_testing then picks the task up).
                        self._schedule_pipeline_timer(
                            delay,
                            partial(self._outage_retry_tick, task_id, AgentRole.TESTER),
                        )
                    else:
                        self._schedule_pipeline_timer(delay, self._advance)
                    return
                self._persist_tester_messages(task, run)
                self._tester_resume_round.discard(task_id)
                self._tester_fail_count[task_id] = (
                    self._tester_fail_count.get(task_id, 0) + 1
                )
                self._schedule_tester_retry(task_id)
                self._show_event(task, f"tester failed: {type(e).__name__}: {e}")
        finally:
            # Stop any spinner whose result event was lost to an abort or a
            # tester error (normally all of them are already stopped).
            self._stop_run_tools(run_key)
            self.tester_running = False
            self._tester_task_id = None
            self.refresh_status()
            self.refresh_sidebar()
            if task.status is TaskStatus.COMPLETED:
                # Success: give the next testing round of this task a fresh
                # backoff and network-retry budget. A count past the budget
                # = the round was in outage mode: announce the recovery.
                self._tester_fail_count.pop(task_id, None)
                self._note_network_recovered(task, self._tester_net_count)
                self._tester_net_count.pop(task_id, None)
                self._tester_resume_round.discard(task_id)
                self._tester_retry_at.pop(task_id, None)
            # Custom workflow: on_tester_passed after a PASSED finalization
            # (no external review — with the review gate the task is still
            # TESTING here and the hook fires from the review's accept path
            # instead). State-checked, so it also fires when the run was
            # aborted after set_verification had already finalized the task
            # (the stall nudge), and never for FAILED labels, research
            # tasks, cancels, or the "finalization: none" path (no label).
            if (
                task.status is TaskStatus.COMPLETED
                and task.verification is VerificationStatus.PASSED
                and not task.review_pending
            ):
                await self._run_tester_passed_hook(task)
                await self._remind_unmerged_work(task)
            # A cancelled run (user abort, or a newer tester taking over)
            # must not restart the pipeline.
            if not cancelled:
                self._advance()
