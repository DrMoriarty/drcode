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
from collections.abc import Callable
from functools import partial
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRun
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)
from pydantic_ai.usage import UsageLimits

from ..agent import (
    COMPACTION_RESUME_PREFIX,
    TesterDeps,
    all_user_prompts,
    build_tester_agent,
    last_response_usage,
    resumable_history,
    run_context_tokens,
)
from ..core.usage import (
    NETWORK_RETRIES,
    _in_outage,
    _is_context_overflow_error,
    _is_retryable_network_error,
    _network_retry_delay,
)
from ..github import issue_commit_hint
from ..models import Task, TaskStatus, VerificationStatus
from ..providers import AgentRole
from ..tools import scratch_dir
from .pipeline_stream import NodeStreamView
from .state import CoreState
from .worker_runtime import work

# The user message a paused testing round is resumed with after a SUCCESSFUL
# mid-round compaction. Without it the model sees only the original prompt
# plus the summary and — nudged by the system prompt's "run each detected
# check" — starts the whole verification cycle from scratch (re-running
# every check, re-collecting every diff: the 8eebbdb8 carousel). The note
# tells the model the summary is a trustworthy replacement for the dropped
# tool results. It rides behind COMPACTION_RESUME_PREFIX so the compaction
# machinery recognizes it: split_original_prompts keeps it OUT of the
# preserved verbatim prefix (the next compaction summarizes it into `rest`),
# and the user-prompt/skeleton extractors skip it.
TESTER_COMPACTION_RESUME_PROMPT = (
    f"{COMPACTION_RESUME_PREFIX} The round history was compacted into the "
    "summary above. The summary records the checks already run and their "
    "exact outcomes, and the files/diffs already reviewed with their "
    "verdicts. Do NOT re-run checks or re-collect evidence that is already "
    "recorded there — continue the verification from the remaining steps "
    "only."
)


def tester_compaction_resume_prompt(verdict_issued: bool) -> str:
    """The resume note for a compacted tester round. When the round had
    ALREADY issued its set_verification verdict before the compaction, an
    explicit warning is appended (the #490d5652 class): a model waking up
    into a summarized round has re-issued the verdict — and on a task the
    first verdict already requeued, the second one bypasses the retry_here
    machinery and completes the task as failed. The verdict is final; a
    second set_verification in the same round is forbidden."""
    if not verdict_issued:
        return TESTER_COMPACTION_RESUME_PROMPT
    return (
        f"{TESTER_COMPACTION_RESUME_PROMPT}\n\n"
        "WARNING: a set_verification verdict was ALREADY issued earlier in "
        "this round, before the compaction — it is recorded in the summary "
        "and is final. Do NOT call set_verification again in this round; "
        "if the summary's verdict line conflicts with what you see in the "
        "worktree, trust the verdict line (the developer's fix, if any, is "
        "queued as their input and lands later, in their own run)."
    )


# -- start-prompt pre-flight gate (the #f425b421 class) ------------------------
#
# The testing round's start prompt embeds the original assignment verbatim;
# a huge assignment can alone exceed the serving model's window, and the
# round then dies on its FIRST request — no mid-round compaction gate ever
# fires (they all wait for a tool boundary the round never reaches) and the
# ordinary failure path loops the same oversized prompt on backoff forever.
# The pre-flight measures the prompt against the tester role's resolved
# config before the first request: trim the embedded texts to head+tail, or
# never send the request at all.

# Conservative chars/token conversion for the pre-flight: the conventional
# len/4 estimate under-counts Cyrillic- and pseudographics-heavy text (≈2.5–3
# real characters per token), and it is the GATEWAY's count the 400 checks —
# the #f425b421 prompt estimated ≈137k at len/4 while the real count crossed
# 65537. The gate assumes 3 chars per token instead.
_PREFLIGHT_CHARS_PER_TOKEN = 3
# The start prompt may take this percent of the threshold-scaled window (the
# rest goes to the measured framing and the round's own growth before the
# mid-round compaction gates take over).
_PREFLIGHT_SLACK_PCT = 90
# Below this a trimmed prompt is useless anyway: give up without a request.
_PREFLIGHT_MIN_PROMPT_TOKENS = 1_000
# Embedded texts shorter than this stay verbatim (trimming them saves
# nothing and mangles the intent).
_PREFLIGHT_BLOCK_FLOOR_CHARS = 2_000


def _trim_oversized_text(text: str, keep_chars: int, task_id: str) -> str:
    """Head+tail of an oversized embedded text around an explicit note
    pointing at the full text in the task's storage file."""
    head = max(keep_chars * 2 // 3, 0)
    tail = max(keep_chars - head, 0)
    note = (
        "\n\n[… The text was shortened to fit the verifier model's context "
        f"window: {len(text)} characters total, the full text is in "
        f".drcode/tasks/{task_id}.transcript.json …]\n\n"
    )
    return text[:head] + note + (text[-tail:] if tail > 0 else "")


class PipelineTesterLayer(CoreState):
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

    def _preflight_tester_prompt(
        self,
        task: Task,
        prompt: str,
        render: Callable[[str | None, list[str]], str],
        original: str | None,
        clarifications: list[str],
    ) -> str | None:
        """Pre-flight gate of the testing round's FIRST request: the start
        prompt must fit the tester model's window before it is sent — the
        mid-round compaction gates all fire at tool boundaries, and a round
        whose first request is rejected never reaches one (the #f425b421
        carousel: six identical 400s on exponential backoff, never a single
        tool call).

        The fill is measured conservatively (`_PREFLIGHT_CHARS_PER_TOKEN`
        instead of the len/4 convention — the heuristic under-counts
        Cyrillic/pseudographics, and it is the gateway's count the 400
        checks) from the MEASURED request framing of the tester role
        (`_mechanical_framing`: system prompt, AGENTS.md, tool schemas) plus
        the output reserve, against the window.

        Returns the prompt to send — the original one when it fits, otherwise
        a rebuilt one with the oversized embedded texts (the assignment, the
        clarifications) trimmed to head+tail around an explicit note — or
        None when even a trimmed prompt cannot fit: the caller must not send
        the request (it is guaranteed the same 400) and gives the round up.
        An unmeasurable wiring or an unknown window (≤ 0) skips the gate:
        without figures there is nothing to decide on."""
        measured = self._mechanical_framing(task, AgentRole.TESTER)
        if measured is None:
            return prompt
        window, framing, output = measured
        reserve = framing + output
        prompt_tokens = len(prompt) // _PREFLIGHT_CHARS_PER_TOKEN + 1
        if reserve >= window or prompt_tokens + reserve <= window:
            # Fits the window (conservatively estimated) — or the framing
            # alone already overflows it, which no trim of the prompt can
            # fix (a config problem the overflow handling owns, not this
            # gate). Send as-is: the mid-round gates compact from there.
            return prompt
        config = self.resolved_config(AgentRole.TESTER)
        threshold = min(max(config.compaction_threshold, 1), 100)
        limit = window * threshold // 100 * _PREFLIGHT_SLACK_PCT // 100
        budget = max(limit - reserve, 0)
        if budget < _PREFLIGHT_MIN_PROMPT_TOKENS:
            # The threshold-scaled target is unreachable: fall back to the
            # window itself (with a 10% margin) — fitting at all beats
            # giving up when the threshold headroom cannot be met.
            budget = max(int(window * 0.9) - reserve, 0)
        blocks = ([original] if original is not None else []) + list(clarifications)
        big_total = sum(len(b) for b in blocks if len(b) > _PREFLIGHT_BLOCK_FLOOR_CHARS)
        excess = len(prompt) - budget * _PREFLIGHT_CHARS_PER_TOKEN
        if budget < _PREFLIGHT_MIN_PROMPT_TOKENS or not blocks or big_total <= excess:
            # Nothing trimmable (or the trimmable part alone overflows):
            # the request would be rejected deterministically.
            return None
        scale = (big_total - excess) / big_total
        trimmed = [
            b
            if len(b) <= _PREFLIGHT_BLOCK_FLOOR_CHARS
            else _trim_oversized_text(b, int(len(b) * scale), task.id)
            for b in blocks
        ]
        new_prompt = render(
            trimmed[0] if original is not None else None,
            trimmed[1:] if original is not None else trimmed,
        )
        new_tokens = len(new_prompt) // _PREFLIGHT_CHARS_PER_TOKEN + 1
        if new_tokens + reserve > window:
            return None
        self._show_event(
            task,
            f"The original assignment is too large for the tester model's "
            f"{window}-token window — the start prompt was trimmed "
            f"({len(prompt) - len(new_prompt)} characters cut, head and "
            "tail kept, the full text is in this task's storage file).",
        )
        return new_prompt

    def _tester_overflow_give_up(self, task: Task, reason: str) -> None:
        """Deterministic death of a testing round: the model rejects the
        requests with a context overflow and nothing can shrink the round
        below the window. Retrying the same prompt is guaranteed the same
        400 (the #f425b421 carousel), so the task completes with the FAILED
        label instead — the work stays on the branch, and sending input to
        the task re-queues it (a fresh round then passes the pre-flight
        gate against the model the user picked meanwhile)."""
        window = self.resolved_config(AgentRole.TESTER).max_context_tokens
        self._show_event(
            task,
            f"The testing round is given up: the tester model's requests "
            f"are deterministically rejected with a context overflow — "
            f"{reason} (window {window} tokens). Retry loops cannot help. "
            "Pick a larger-window model for the tester role (Settings → "
            "Provider catalog) or shorten the task, then send input to the "
            "task to re-queue it.",
        )
        self.notifier.task_context_overflow(task.title)
        self._tester_fail_count.pop(task.id, None)
        self._tester_retry_at.pop(task.id, None)
        self._tester_resume_round.discard(task.id)
        self._tester_overflow_retried.discard(task.id)
        self._tester_compaction_resumed.discard(task.id)
        self._tester_compaction_verdict.discard(task.id)
        self.manager.set_verification(task.id, success=False)

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
        if not resume_round:
            # A fresh testing round: the previous round's verdict-at-
            # compaction fact no longer applies (see
            # _maybe_compact_tester_round).
            self._tester_compaction_verdict.discard(task_id)
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
            # branch with the task's commits is re-checked-out). Passing
            # the task resolves the round against its isolation PIN —
            # a mid-task toggle of the setting must not send the tester
            # to the other tree (it would verify an empty worktree or a
            # main checkout without the task's work).
            pinned_before = task.isolation_pinned is not None
            root, iso_note = await asyncio.to_thread(
                self.task_env.ensure_root, task_id, task
            )
            self._isolation_pin_note(task, save=not pinned_before)
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
                # The real scratch path is baked into run_bash's description
                # (no literal <task-id> placeholder for the model to copy
                # verbatim into a path argument).
                task_id=task_id,
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
                # create_bug_report creates a DRAFT task when the setting
                # is on (the user finishes it in the editor).
                agent_creates_drafts=self.settings.agent_creates_drafts,
            )
            try:
                # The tester needs the original assignment (to paraphrase it
                # into a commit message — see the git_commit step in its
                # prompt) plus every clarification the user sent while the
                # task was being worked on, so the verification targets the
                # CURRENT intent, not a stale initial one.
                user_prompts = [
                    p for p in all_user_prompts(task.messages) if p != task.title
                ]
                original = user_prompts[0] if user_prompts else None
                clarifications = user_prompts[1:]
                # Custom workflow: context returned by on_development_done
                # (the opened PR, the pushed branch, CI links, …).
                dev_note = self._dev_done_notes.pop(task_id, None)

                def render_prompt(
                    original_text: str | None, clar_texts: list[str]
                ) -> str:
                    prompt = f"Verify completed task #{task.id}: {task.title}"
                    if original_text is not None:
                        prompt += f"\n\nOriginal task:\n{original_text}"
                    if clar_texts:
                        prompt += (
                            "\n\nUser clarifications sent during development "
                            "(in order, latest last):\n"
                            + "\n\n".join(f"- {c}" for c in clar_texts)
                        )
                    # A GitHub-imported task must link its commit back to the
                    # originating issue ("Fixes #N" keyword in the message).
                    if task.github_issue is not None:
                        prompt += f"\n\n{issue_commit_hint(task.github_issue)}"
                    if dev_note:
                        prompt += (
                            "\n\nProject workflow notes (from the "
                            "on_development_done step):\n"
                            f"{dev_note}"
                        )
                    return prompt

                # Pre-flight gate: the start prompt must fit the tester
                # model's window BEFORE the first request (see the
                # _PREFLIGHT_* notes above) — an oversized one is trimmed
                # here, or the round is given up without a doomed request.
                prompt = self._preflight_tester_prompt(
                    task,
                    render_prompt(original, clarifications),
                    render_prompt,
                    original,
                    clarifications,
                )
                if prompt is None:
                    self._tester_overflow_give_up(
                        task,
                        "the start prompt (the original assignment alone) "
                        "does not fit the tester model's context window even "
                        "after trimming the embedded task text",
                    )
                    return
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
                # transcript; when the resume follows a SUCCESSFUL
                # compaction (mid-round pause, or the forced recovery of a
                # context-overflow death), the continuation turn also
                # carries TESTER_COMPACTION_RESUME_PROMPT — the summary
                # must be treated as evidence already collected. An empty
                # round (the failed request was never recorded) re-sends
                # the prompt as usual.
                resume_from_round = bool(
                    resume_round and task.tester_messages and task.tester_messages[-1]
                )
                resumed_compacted = resume_from_round and (
                    task_id in self._tester_compaction_resumed
                )
                self._tester_compaction_resumed.discard(task_id)
                text: str | None = (
                    tester_compaction_resume_prompt(
                        task_id in self._tester_compaction_verdict
                    )
                    if resumed_compacted
                    else None
                    if resume_from_round
                    else prompt
                )
                # Live context reading of the current round; it goes into
                # the tester's own map (`_live_ctx_tester`) — the shared
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
                    if resume_from_round:
                        # Continuation of a paused round: resume from the
                        # stored round transcript (compacted, or — when the
                        # summarizer failed — the original one with its
                        # invalid tail trimmed). The continuation text (a
                        # resume note, or None for a plain continuation)
                        # rides ON TOP of this history.
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
                                    try:
                                        async for event in node_stream:
                                            view.feed(event)
                                    finally:
                                        # A mid-stream death (network error,
                                        # abort) skips the normal node end:
                                        # stop the spinners here, or they
                                        # keep animating through the retry
                                        # backoff (and the outage wait).
                                        view.finish_node()
                                live_tokens = run_context_tokens(run)
                                if live_tokens is not None:
                                    self._live_ctx_tester[task_id] = live_tokens
                                    if self.viewed_id == task_id:
                                        self.refresh_task_status()
                                # Exact per-request usage of the completed
                                # model call, into the session totals and
                                # the persistent per-day usage store.
                                self._record_session_usage(
                                    last_response_usage(run),
                                    AgentRole.TESTER,
                                    self.resolved_model_key(AgentRole.TESTER),
                                )
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
                        resume_from_round = True
                        if await self._maybe_compact_tester_round(task, live_tokens):
                            # The round transcript was replaced by the
                            # original prompt plus the summary: the live
                            # token reading belongs to the old transcript —
                            # drop it so the counter falls back to the
                            # compacted one (mirrors _maybe_compact).
                            live_tokens = None
                            # Nudge the model: the summary records the
                            # checks already run and their outcomes — the
                            # round must continue from the remaining steps,
                            # not restart the whole verification cycle (the
                            # 8eebbdb8 carousel). A round that already
                            # issued its verdict before the compaction gets
                            # an explicit "verdict is final" warning — a
                            # second set_verification here would complete
                            # the requeued task as failed (the #490d5652
                            # class).
                            text = tester_compaction_resume_prompt(
                                task_id in self._tester_compaction_verdict
                            )
                        else:
                            # The summarizer failed and the gate is now
                            # disabled for this task: resume as a plain
                            # continuation turn — no new text, history from
                            # the round transcript.
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
                if not _is_context_overflow_error(e) and _is_retryable_network_error(e):
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
                if _is_context_overflow_error(e):
                    # Deterministic rejection (HTTP 400 "context length
                    # exceeded"): the same transcript is refused on every
                    # retry — the ordinary backoff path below would loop it
                    # forever (the #f425b421 carousel). Shrink the round
                    # once (forced compaction — the round never reached the
                    # tool boundary the ordinary gate waits at) and retry
                    # it as a continuation; a second overflow (or nothing
                    # left to shrink) gives the round up.
                    self._persist_tester_messages(task, run)
                    if (
                        task_id not in self._tester_overflow_retried
                        and await self._maybe_compact_tester_round(
                            task, None, force=True
                        )
                    ):
                        self._tester_overflow_retried.add(task_id)
                        self._tester_resume_round.add(task_id)
                        # The forced compaction succeeded: the round resumes
                        # (after the retry timer) as a continuation turn that
                        # must carry the resume note (same rationale as the
                        # mid-round pause path above).
                        self._tester_compaction_resumed.add(task_id)
                        self._tester_fail_count.pop(task_id, None)
                        self._tester_retry_at[task_id] = time.monotonic() + 5
                        self._schedule_pipeline_timer(5, self._advance)
                        return
                    self._tester_overflow_give_up(
                        task,
                        "compaction cannot shrink the round below the "
                        f"model's window ({type(e).__name__})",
                    )
                    return
                self._persist_tester_messages(task, run)
                self._tester_resume_round.discard(task_id)
                self._tester_compaction_resumed.discard(task_id)
                self._tester_compaction_verdict.discard(task_id)
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
            self._live_ctx_tester.pop(task_id, None)
            self.refresh_status()
            self.refresh_sidebar()
            if task.status is TaskStatus.COMPLETED:
                # Success: give the next testing round of this task a fresh
                # backoff and network-retry budget. A count past the budget
                # = the round was in outage mode: announce the recovery.
                self._tester_fail_count.pop(task_id, None)
                self._tester_overflow_retried.discard(task_id)
                self._tester_compaction_resumed.discard(task_id)
                self._tester_compaction_verdict.discard(task_id)
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
