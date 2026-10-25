"""Context compaction: threshold gates, LLM summarizer, mechanical floor.

Split out of the former monolithic `tui/pipeline.py`: the fill
measurement against the serving role's resolved config, the
single-shot / map-reduce summarizer and the LLM-free mechanical
trim — for both the main agent transcript and a testing round.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..agent import (
    MECHANICAL_HEADROOM_TOKENS,
    build_compaction_agent,
    build_compaction_prompt,
    build_reduce_prompt,
    chunk_history_for_summary,
    compacted_messages,
    extract_previous_summary,
    is_summary_message,
    last_context_tokens,
    mechanical_compaction_messages,
    prompt_fits_context,
    prompt_fits_window,
    render_history_for_summary,
    slice_budget_tokens,
    split_original_prompts,
)
from ..compaction import _MAX_REDUCE_DEPTH, CHARS_PER_TOKEN
from ..config import Config
from ..models import Task, TaskKind, render_todos
from ..prompts import load_compaction_prompts
from ..providers import AgentRole
from ..transcript import estimate_tokens
from .state import DrCodeState
from .usage import _is_context_overflow_error

# Hard-limit compaction level (percent of max_context_tokens): the last
# safe line before the serving model deterministically rejects the next
# request with a context-overflow error (HTTP 400). The THRESHOLD level
# pauses and compacts via the LLM summarizer (and backs off when the
# summarizer is broken); the HARD-LIMIT level fires even when a previous
# compaction failed — growth past it must never be allowed, so a failing
# summarizer there escalates to the mechanical trim instead.
_HARD_LIMIT_PCT = 97

# Livelock guard: the maximum number of CONSECUTIVE hard-limit compactions
# (no gate observation under the hard limit in between — every resumed turn
# lands over the hard limit again) before the gate gives up. Past it the
# role's cap is below the transcript's irreducible floor (system prompt +
# tool schemas + original prompts verbatim + a summary that cannot shrink
# further) and compacting again can only burn the API budget: the gate is
# disabled for the task and the diagnostic event tells the user what to fix.
_HARD_LIMIT_COMPACTION_LIMIT = 3


class _ChunkingImpossible(Exception):
    """The history cannot be chunked into map-reduce slices that fit the
    COMPACTION role's context window (a tiny window the slice budget's
    floor exceeds, or no summarizable slices at all): every map call would
    deterministically overflow — the deterministic signal for the caller
    to apply the mechanical floor instead of calling the model."""


class PipelineCompactionLayer(DrCodeState):
    def _fill_over_threshold(
        self,
        live_tokens: int | None,
        messages: list[dict[str, Any]],
        role: AgentRole = AgentRole.CODING,
    ) -> int | None:
        """Context fill (tokens) when the compaction threshold is reached
        (percent of max_context_tokens), None otherwise. Prefers the live
        reading of the running turn over the persisted transcript, whose
        usage lags behind by the not-yet-persisted requests.

        Both the context size and the threshold come from the resolved
        config of `role` — the model actually serving the transcript — so
        a per-model compaction limit from the catalog (models.json) is
        honored instead of the flat config's value."""
        config = self.resolved_config(role)
        max_tokens = config.max_context_tokens
        if max_tokens <= 0:
            return None
        threshold = min(max(config.compaction_threshold, 1), 100)
        tokens = live_tokens
        if tokens is None:
            tokens = last_context_tokens(messages)
        if not tokens or tokens * 100 < threshold * max_tokens:
            return None
        return tokens

    def _compaction_due(self, task: Task) -> int | None:
        """Compaction threshold check for the main agent transcript,
        measured against the model actually serving it (research tasks use
        the research role's resolved config, everything else the coding
        one)."""
        return self._fill_over_threshold(
            self._live_ctx_tokens.get(task.id), task.messages, self._task_role(task)
        )

    def _fill_over_hard_limit(
        self,
        live_tokens: int | None,
        messages: list[dict[str, Any]],
        role: AgentRole = AgentRole.CODING,
    ) -> int | None:
        """Context fill (tokens) when the HARD-LIMIT level is reached
        (_HARD_LIMIT_PCT percent of max_context_tokens), None otherwise.

        Unlike the threshold level, the hard limit is measured
        independently of `_compaction_failed`: past it the next model
        request is rejected deterministically, so a broken summarizer must
        never disable this check — it can only downgrade the compaction
        itself to the mechanical floor (see `_compact_transcript`)."""
        config = self.resolved_config(role)
        max_tokens = config.max_context_tokens
        if max_tokens <= 0:
            return None
        tokens = live_tokens
        if tokens is None:
            tokens = last_context_tokens(messages)
        if not tokens or tokens * 100 < _HARD_LIMIT_PCT * max_tokens:
            return None
        return tokens

    def _hard_limit_due(self, task: Task) -> bool:
        """Hard-limit check for the main agent transcript (same role
        resolution as `_compaction_due`)."""
        return (
            self._fill_over_hard_limit(
                self._live_ctx_tokens.get(task.id),
                task.messages,
                self._task_role(task),
            )
            is not None
        )

    def _hard_livelock_stop(
        self,
        task: Task,
        counter: dict[str, int],
        exhausted: set[str],
        gate_failed: set[str],
        role: AgentRole,
        pct: int,
        what: str,
    ) -> bool:
        """Livelock guard of the hard-limit compaction gates — call at gate
        entry when the fill is over the hard limit, right before compacting.

        Counts the CONSECUTIVE hard-level compactions of the task (the
        caller resets the counter whenever a gate observation lands under
        the hard limit — a previous compaction worked). Past
        `_HARD_LIMIT_COMPACTION_LIMIT` of them every resumed turn lands
        over the hard limit again: the cap is below the transcript's
        irreducible floor and the pause→compact→resume cycle would repeat
        forever (each cycle a successful compaction — the estimate-based
        no-op guards never fire, the wire-level fill stays over 100%).
        The gate is then disabled for the task (both the hard-level bypass
        flag and the threshold-level failed set, so the mid-round pauses
        stop too) and a diagnostic service event names the misconfigured
        cap. Returns True when the caller must NOT compact: the round is
        resumed as-is and the next model request dies with the
        deterministic context overflow, whose ordinary error handling
        (agent: `_agent_dead`; tester/resolver: backoff) takes over."""
        count = counter.get(task.id, 0) + 1
        counter[task.id] = count
        if count <= _HARD_LIMIT_COMPACTION_LIMIT:
            return False
        if task.id in exhausted:
            # Already stopped for this task: a later fire (e.g. a direct
            # gate call in a test) must not repeat the diagnostic.
            return True
        exhausted.add(task.id)
        gate_failed.add(task.id)
        cap = self.resolved_config(role).max_context_tokens
        self._show_event(
            task,
            f"Context fill {pct}% is still over the hard limit after "
            f"{_HARD_LIMIT_COMPACTION_LIMIT} consecutive compactions — "
            f"compaction cannot bring the {what} below the hard limit: "
            f"the {role.value} model's context cap ({cap} tokens) is at "
            "or below its irreducible floor (system prompt, tool schemas "
            "and the original prompt(s) kept verbatim). Compaction "
            "stopped; the next model request will fail with a context "
            "overflow. Raise the model's max_context_tokens in the "
            "provider catalog (drcode-models).",
        )
        return True

    def _messages_estimate(self, messages: list[dict[str, Any]]) -> int:
        """≈ wire size of a serialized transcript in tokens (chars/4 on the
        json dump — same deliberately conservative estimate the mechanical
        floor uses). Used to detect that a hard-limit compaction failed to
        shrink the transcript at all (no-op → the gate must stop firing)."""
        return estimate_tokens(json.dumps(messages, ensure_ascii=False))

    def _mechanical_token_budget(self, role: AgentRole) -> int:
        """Token budget of the mechanical compaction floor for `role`: the
        serving model's window under its compaction threshold, minus a
        headroom reserve (system prompt, tool schemas, the next turn's
        output — MECHANICAL_HEADROOM_TOKENS, capped at a quarter of the
        window). An unknown serving window falls back to the compaction
        model's (smaller) one; both unknown → 0 (only the original prompts
        and the note survive)."""
        cfg = self.resolved_config(role)
        window = cfg.max_context_tokens
        if window <= 0:
            window = self.resolved_config(AgentRole.COMPACTION).max_context_tokens
        if window <= 0:
            return 0
        threshold = min(max(cfg.compaction_threshold, 1), 100)
        headroom = min(MECHANICAL_HEADROOM_TOKENS, window // 4)
        return max(window * threshold // 100 - headroom, 0)

    def _mechanical_compact(
        self, messages: list[dict[str, Any]], summary_role: AgentRole, what: str
    ) -> tuple[list[dict[str, Any]], str]:
        """The LLM-free floor: trim `messages` at message boundaries into a
        transcript that fits `summary_role`'s window, plus the success event
        text announcing it."""
        trimmed = mechanical_compaction_messages(
            messages, self._mechanical_token_budget(summary_role)
        )
        event = (
            "Context compacted mechanically: the transcript did not fit the "
            f"summarizer model's context window, {what} was trimmed at "
            "message boundaries without a summarizer call (original task "
            "prompt(s), the previous summary and the most recent messages "
            "kept)."
        )
        return trimmed, event

    async def _compact_transcript(
        self,
        messages: list[dict[str, Any]],
        summary_role: AgentRole,
        task: Task,
        what: str,
        checklist: str | None = None,
        hard: bool = False,
    ) -> tuple[list[dict[str, Any]], str, bool]:
        """Summarize `messages` into a compacted transcript; returns it,
        the success event text and whether the LLM summarizer path failed
        (`llm_failed` — callers keep `_compaction_failed` set for such
        tasks, meaning "the summarizer is unavailable", and clear it when
        the summarizer ran fine).

        The LLM path goes through `_summarize_history`, which owns the
        single-shot pre-flight (the rendered prompt must fit the COMPACTION
        role's context window) and falls back to a chunked map-reduce
        summarization when the history does not fit — so an oversized
        history is still summarized, never skipped. Two deterministic
        outcomes bypass the LLM and land on the mechanical floor
        (`mechanical_compaction_messages`): the window is too small for
        even one minimal map-reduce slice (`_ChunkingImpossible`), or the
        gateway rejects the summary request anyway with a deterministic
        context-overflow error (see `_is_context_overflow_error`) — the
        chars/4 estimate can under-shoot.

        Any OTHER summarizer failure is re-raised — a transient outage
        should retry later, not throw away history — EXCEPT at the
        hard-limit level (`hard=True`): there the fill is at/near the hard
        limit, growth past it means a deterministic death of the run, so
        the failure escalates to the mechanical trim instead. In other
        words `_compaction_failed` may mean "the summarizer is
        unavailable", never "growth past the limit is allowed".
        """
        try:
            summary = await self._summarize_history(task, messages, checklist=checklist)
        except asyncio.CancelledError:
            raise
        except _ChunkingImpossible:
            # Chunking cannot produce a slice that fits the window: the
            # mechanical floor is the only path that never overflows.
            trimmed, event = self._mechanical_compact(messages, summary_role, what)
            return trimmed, event, False
        except Exception as e:
            if _is_context_overflow_error(e):
                # The estimate under-shot and the gateway rejected the
                # summary request: fall back to the mechanical floor
                # instead of failing the whole compaction.
                self._show_event(
                    task,
                    "The summarizer model rejected the transcript "
                    f"({type(e).__name__}: context overflow) — trimming "
                    "mechanically instead.",
                )
                trimmed, event = self._mechanical_compact(messages, summary_role, what)
                return trimmed, event, True
            if not hard:
                raise
            # The fill is at the hard-limit level: retrying later is not an
            # option (the run would die deterministically before it), so a
            # broken summarizer here escalates to the mechanical trim.
            self._show_event(
                task,
                f"The summarizer failed ({type(e).__name__}: {e}) with the "
                "fill at the hard limit — trimming mechanically instead.",
            )
            trimmed, event = self._mechanical_compact(messages, summary_role, what)
            return trimmed, event, True
        event = (
            f"Context compacted: {what} summarized, original prompt(s) kept verbatim."
        )
        return compacted_messages(messages, summary), event, False

    async def _run_summarizer(self, prompt: str) -> str:
        """One compaction model call: run the tool-less summarizer and
        record its usage into the session totals like every other model
        call. Raises on failure — the callers turn that into a non-fatal
        compaction-failed event."""
        agent = build_compaction_agent(self.resolved_config(AgentRole.COMPACTION))
        result = await agent.run(prompt)
        # The summarizer's model calls count toward the session usage too.
        self._record_session_usage(result.usage)
        return result.output

    async def _reduce_partials(
        self,
        partials: list[str],
        template: str,
        checklist: str | None,
        depth: int = 0,
    ) -> str:
        """Merge the map step's partial summaries into the final structured
        summary. When the accumulated partials overflow the merge call, the
        partials are grouped into fitting batches and each batch is reduced
        first (hierarchical reduce, depth-capped); past the cap the payload
        is mechanically trimmed per partial so the merge call always
        happens (the guaranteed floor — a bounded summary beats none)."""
        prompt = build_reduce_prompt(partials, template, checklist)
        compaction_cfg = self.resolved_config(AgentRole.COMPACTION)
        if prompt_fits_window(prompt, compaction_cfg) or not partials:
            return await self._run_summarizer(prompt)
        if depth >= _MAX_REDUCE_DEPTH:
            trimmed = self._mechanically_trim_partials(
                partials, compaction_cfg, checklist
            )
            return await self._run_summarizer(
                build_reduce_prompt(trimmed, template, checklist)
            )
        # Hierarchical reduce: batch the partials so each batch's prompt
        # fits, reduce every batch, then merge the intermediates.
        batch_budget = slice_budget_tokens(compaction_cfg, checklist=checklist)
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for p in partials:
            size = len(p) // CHARS_PER_TOKEN + 1
            if current and current_tokens + size > batch_budget:
                batches.append(current)
                current, current_tokens = [], 0
            current.append(p)
            current_tokens += size
        if current:
            batches.append(current)
        if len(batches) == 1 and len(batches[0]) == len(partials):
            # Single batch covering everything and still overflowing (one
            # gigantic partial): trim mechanically instead of recursing on
            # an unchanged payload.
            trimmed = self._mechanically_trim_partials(
                partials, compaction_cfg, checklist
            )
            return await self._run_summarizer(
                build_reduce_prompt(trimmed, template, checklist)
            )
        intermediates = [
            await self._reduce_partials(batch, template, checklist, depth + 1)
            for batch in batches
        ]
        return await self._reduce_partials(
            intermediates, template, checklist, depth + 1
        )

    @staticmethod
    def _mechanically_trim_partials(
        partials: list[str],
        compaction_cfg: Config,
        checklist: str | None,
    ) -> list[str]:
        """Guaranteed-floor fallback: trim each partial to an equal char
        share of the reduce budget so the merge call always happens (a
        bounded, degraded summary beats no compaction at all)."""
        budget_chars = (
            max(slice_budget_tokens(compaction_cfg, checklist=checklist), 1)
            * CHARS_PER_TOKEN
        )
        per_partial = max(budget_chars // max(len(partials), 1), 200)
        return [
            p[:per_partial] + ("\n…[truncated]" if len(p) > per_partial else "")
            for p in partials
        ]

    async def _summarize_history(
        self,
        task: Task,
        messages: list[dict[str, Any]],
        *,
        checklist: str | None = None,
    ) -> str:
        """LLM summarization of `messages` for compaction, shared by the
        main-agent and tester-round gates (both via _compact_transcript).

        Single-shot when the rendered history fits the COMPACTION role's
        window (the chars/4 pre-flight, `prompt_fits_context`); otherwise a
        map-reduce: the history is chunked at conversation boundaries into
        slices that fit (with headroom for the system prompt, output
        template, anchors and the reduce step), the slices are summarized
        sequentially — each map call sees the running partial summaries as
        its <previous-summary> anchor, mirroring the previous-summary
        anchoring of the single-shot path — and a final reduce call merges
        the partials into the structured summary (hierarchical reduce, then
        a mechanical per-partial trim, when the partials themselves
        overflow the merge call). When even one minimal slice cannot fit
        the window (a tiny window the slice budget's floor exceeds), raises
        `_ChunkingImpossible` — the deterministic signal for the caller to
        apply the mechanical floor instead of calling the model.

        Every model call's usage is recorded via _record_session_usage, so
        total extra calls stay visible exactly like the single-shot path.
        Returns the summary text; a summarizer failure propagates to the
        caller (_compact_transcript), which keeps the mechanical-floor /
        retry-later / hard-limit-escalation semantics (compaction must not
        kill the run)."""
        system_prompt, template = load_compaction_prompts()
        anchor = extract_previous_summary(messages)
        prompt = build_compaction_prompt(
            anchor,
            render_history_for_summary(messages),
            template,
            checklist=checklist,
        )
        compaction_cfg = self.resolved_config(AgentRole.COMPACTION)
        if prompt_fits_context(prompt, system_prompt, compaction_cfg):
            return await self._run_summarizer(prompt)
        budget = slice_budget_tokens(compaction_cfg, anchor, checklist)
        if budget > compaction_cfg.max_context_tokens:
            raise _ChunkingImpossible(
                f"the {compaction_cfg.max_context_tokens}-token window is "
                f"smaller than one minimal {budget}-token slice"
            )
        slices = chunk_history_for_summary(messages, budget)
        if not slices:
            raise _ChunkingImpossible("no summarizable slices")
        self._show_event(
            task,
            f"History too large for one summarizer call — map-reduce "
            f"over {len(slices)} slice(s)…",
        )
        partials: list[str] = []
        for chunk in slices:
            running = "\n\n".join(partials) if partials else anchor
            partials.append(
                await self._run_summarizer(
                    build_compaction_prompt(
                        running, chunk, template, checklist=checklist
                    )
                )
            )
        return await self._reduce_partials(partials, template, checklist)

    async def _maybe_compact(self, task: Task) -> None:
        """Context compaction gate, called between agent turns (a turn
        boundary is the only safe point: the transcript never ends with
        unanswered tool calls there).

        When the context fill reaches the configured threshold (percent of
        max_context_tokens), the main agent's work is paused and the
        transcript is compacted: a tool-less summarizer agent rewrites it
        into a structured summary (anchored to the previous one, if any) —
        single-shot when it fits the compaction model's window, else a
        chunked map-reduce — or, when the summarizer itself cannot be run
        (the gateway rejects the summary request with a deterministic
        context-overflow error), the transcript is trimmed mechanically
        without any LLM call. Either
        way the transcript is replaced by the original user prompt(s) plus
        the summary/note, so compaction always terminates with a history
        that fits the serving model's context. A compaction failure is
        never fatal: it is logged into the service events and the run
        continues with the uncompressed history — except at the hard-limit
        level, where a failed summarizer escalates to the mechanical trim
        (growth past the hard limit must never be allowed). When even
        successful compactions cannot bring the fill under the hard limit
        (the cap is below the transcript's irreducible floor),
        `_hard_livelock_stop` bounds the pause→compact→resume cycle: after
        `_HARD_LIMIT_COMPACTION_LIMIT` consecutive hard-level compactions
        the gate disables itself for the task with a diagnostic event and
        the next model request dies with the deterministic context
        overflow.
        """
        tokens = self._compaction_due(task)
        hard = self._hard_limit_due(task)
        if not hard:
            # A reading under the hard limit: the previous compaction (if
            # any) landed — the consecutive hard-compaction counter of the
            # livelock guard starts over.
            self._hard_limit_compactions.pop(task.id, None)
        if tokens is None:
            return
        role_config = self.resolved_config(self._task_role(task))
        pct = round(tokens * 100 / role_config.max_context_tokens)
        _, rest = split_original_prompts(task.messages)
        if not any(not is_summary_message(m) for m in rest):
            # Nothing to summarize (only the original prompt(s), possibly
            # plus a previous summary): compacting again would be a no-op.
            if hard:
                # At the hard-limit level this no-op must also stop the
                # gate from firing again: the mechanical floor keeps the
                # original prompts verbatim, so nothing can shrink this
                # transcript and pause→resume would loop forever.
                self._hard_limit_exhausted.add(task.id)
            return
        if hard and self._hard_livelock_stop(
            task,
            self._hard_limit_compactions,
            self._hard_limit_exhausted,
            self._compaction_failed,
            self._task_role(task),
            pct,
            "history",
        ):
            return
        self._show_event(
            task,
            f"Context fill {pct}% — pausing the agent and compacting history…",
        )
        try:
            new_messages, event, llm_failed = await self._compact_transcript(
                task.messages,
                summary_role=self._task_role(task),
                task=task,
                what="history",
                checklist=(
                    render_todos(task.todos)
                    if task.kind is TaskKind.FEATURE and task.todos
                    else None
                ),
                hard=hard,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - compaction must not kill the run
            self._compaction_failed.add(task.id)
            if not hard:
                self._show_event(task, f"compaction failed: {type(e).__name__}: {e}")
                return
            # Hard-limit escalation: the run must not continue
            # uncompressed past the hard limit — trim mechanically.
            self._show_event(
                task,
                f"compaction failed ({type(e).__name__}: {e}) with the fill "
                "at the hard limit — trimming mechanically instead.",
            )
            new_messages, event = self._mechanical_compact(
                task.messages, self._task_role(task), "history"
            )
            llm_failed = True
        if llm_failed:
            self._compaction_failed.add(task.id)
            self._hard_limit_exhausted.discard(task.id)
        else:
            self._compaction_failed.discard(task.id)
            self._hard_limit_exhausted.discard(task.id)
        if hard and self._messages_estimate(new_messages) >= self._messages_estimate(
            task.messages
        ):
            # The hard-level compaction could not shrink the transcript
            # (only the verbatim original prompts are left): disable the
            # hard-level gate bypass for this task — firing it again would
            # loop pause→resume with no effect.
            self._hard_limit_exhausted.add(task.id)
        task.messages = new_messages
        task.touch()
        self.manager.storage.save(task)
        # The live token reading is stale now (it belongs to the old
        # transcript); drop it so the counter falls back to the compacted
        # transcript until the next model request reports real usage.
        self._live_ctx_tokens.pop(task.id, None)
        self._show_event(task, event)
        if self.viewed_id == task.id:
            self.view_task(task.id)

    async def _maybe_compact_tester_round(
        self, task: Task, live_tokens: int | None
    ) -> bool:
        """Compaction gate for the current testing round, called after
        run_tester paused the round at a tool boundary.

        Mirrors _maybe_compact for the tester, with one crucial difference:
        the summarizer rewrites the ROUND transcript
        (`task.tester_messages[-1]`) in place — `task.messages` (the
        developer transcript) is never touched, the two histories must not
        mix (tester rounds are persisted separately precisely so they never
        leak into the coding agent's model history). The same summarizer /
        mechanical-floor guarantees apply: when the summarizer cannot be
        run, the round is trimmed mechanically instead of looping through a
        guaranteed failure.

        Returns True when the round was compacted: run_tester must resume
        it as a continuation turn from the compacted transcript (and drop
        the stale live token reading). Returns False when compaction was
        not possible — the threshold is not reached, there is nothing to
        summarize (a no-op), or the summarizer failed with a non-overflow
        error below the hard limit — in which case the round is resumed
        uncompressed and the caller adds the task to `_compaction_failed`,
        so the mid-round gate stays off at the threshold level (a
        persistently broken summarizer must not loop pause→resume forever);
        turn-boundary compaction of the main agent still retries on later
        turns. At the hard-limit level a failed summarizer escalates to the
        mechanical trim instead of giving up (growth past the hard limit is
        never allowed) — unless even the mechanical floor cannot shrink the
        round (nothing to summarize), in which case the task lands in
        `_hard_limit_exhausted` and the hard-level gate bypass stops
        firing. The `_hard_livelock_stop` guard bounds the cycle itself:
        past `_HARD_LIMIT_COMPACTION_LIMIT` consecutive hard-level
        compactions without a turn landing under the hard limit (the cap is
        below the round's irreducible context floor) the gate stops
        compacting, disables itself for the task (both sets) and emits a
        diagnostic event instead of burning the API budget forever.
        """
        round_messages = task.tester_messages[-1] if task.tester_messages else []
        tokens = self._fill_over_threshold(
            live_tokens, round_messages, AgentRole.TESTER
        )
        if tokens is None:
            return False
        hard = (
            self._fill_over_hard_limit(live_tokens, round_messages, AgentRole.TESTER)
            is not None
        )
        if not hard:
            # A reading under the hard limit: the previous compaction (if
            # any) landed — the consecutive hard-compaction counter of the
            # livelock guard starts over.
            self._hard_limit_compactions.pop(task.id, None)
        pct = round(
            tokens * 100 / self.resolved_config(AgentRole.TESTER).max_context_tokens
        )
        _, rest = split_original_prompts(round_messages)
        if not any(not is_summary_message(m) for m in rest):
            # Nothing to summarize (only the original prompt(s), possibly
            # plus a previous summary): compacting again would be a no-op —
            # resuming the round as-is would hit the gate again.
            self._compaction_failed.add(task.id)
            if hard:
                # The mechanical floor keeps the original prompts verbatim
                # too: nothing can shrink this round — stop the hard-level
                # gate bypass or pause→resume would loop forever.
                self._hard_limit_exhausted.add(task.id)
            return False
        if hard and self._hard_livelock_stop(
            task,
            self._hard_limit_compactions,
            self._hard_limit_exhausted,
            self._compaction_failed,
            AgentRole.TESTER,
            pct,
            "testing round",
        ):
            return False
        self._show_event(
            task,
            f"Context fill {pct}% — compacting the testing round…",
        )
        try:
            new_messages, event, llm_failed = await self._compact_transcript(
                round_messages,
                summary_role=AgentRole.TESTER,
                task=task,
                what="testing round",
                hard=hard,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - compaction must not kill the round
            self._compaction_failed.add(task.id)
            if not hard:
                self._show_event(task, f"compaction failed: {type(e).__name__}: {e}")
                return False
            # Hard-limit escalation: the round must not continue
            # uncompressed past the hard limit — trim mechanically.
            self._show_event(
                task,
                f"compaction failed ({type(e).__name__}: {e}) with the fill "
                "at the hard limit — trimming mechanically instead.",
            )
            new_messages, event = self._mechanical_compact(
                round_messages, AgentRole.TESTER, "testing round"
            )
            llm_failed = True
        if llm_failed:
            self._compaction_failed.add(task.id)
            self._hard_limit_exhausted.discard(task.id)
        else:
            self._compaction_failed.discard(task.id)
            self._hard_limit_exhausted.discard(task.id)
        if hard and self._messages_estimate(new_messages) >= self._messages_estimate(
            round_messages
        ):
            # The hard-level compaction could not shrink the round:
            # disable the hard-level gate bypass for this task.
            self._hard_limit_exhausted.add(task.id)
        task.tester_messages[-1] = new_messages
        task.touch()
        self.manager.storage.save(task)
        self._show_event(task, event)
        if self.viewed_id == task.id:
            self.view_task(task.id)
        return True

    async def _maybe_compact_resolver_round(
        self, task: Task, live_tokens: int | None
    ) -> bool:
        """Compaction gate for the current conflict-resolver round, called
        after run_resolver paused the round at a tool boundary.

        Mirrors _maybe_compact_tester_round exactly, but rewrites
        `task.resolver_messages[-1]` in place and measures the fill
        against the RESOLVER role's resolved config. The round failure
        bookkeeping uses the resolver's own sets
        (`_resolver_compaction_failed` / `_resolver_hard_limit_exhausted`)
        so a broken summarizer seen by the agent or the tester never
        disables the resolver's gate (and vice versa).
        """
        round_messages = task.resolver_messages[-1] if task.resolver_messages else []
        tokens = self._fill_over_threshold(
            live_tokens, round_messages, AgentRole.RESOLVER
        )
        if tokens is None:
            return False
        hard = (
            self._fill_over_hard_limit(live_tokens, round_messages, AgentRole.RESOLVER)
            is not None
        )
        if not hard:
            # A reading under the hard limit: the previous compaction (if
            # any) landed — the consecutive hard-compaction counter of the
            # livelock guard starts over.
            self._resolver_hard_limit_compactions.pop(task.id, None)
        pct = round(
            tokens * 100 / self.resolved_config(AgentRole.RESOLVER).max_context_tokens
        )
        _, rest = split_original_prompts(round_messages)
        if not any(not is_summary_message(m) for m in rest):
            self._resolver_compaction_failed.add(task.id)
            if hard:
                self._resolver_hard_limit_exhausted.add(task.id)
            return False
        if hard and self._hard_livelock_stop(
            task,
            self._resolver_hard_limit_compactions,
            self._resolver_hard_limit_exhausted,
            self._resolver_compaction_failed,
            AgentRole.RESOLVER,
            pct,
            "conflict-resolution round",
        ):
            return False
        self._show_event(
            task,
            f"Context fill {pct}% — compacting the conflict-resolution round…",
        )
        try:
            new_messages, event, llm_failed = await self._compact_transcript(
                round_messages,
                summary_role=AgentRole.RESOLVER,
                task=task,
                what="conflict-resolution round",
                hard=hard,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - compaction must not kill the round
            self._resolver_compaction_failed.add(task.id)
            if not hard:
                self._show_event(task, f"compaction failed: {type(e).__name__}: {e}")
                return False
            self._show_event(
                task,
                f"compaction failed ({type(e).__name__}: {e}) with the fill "
                "at the hard limit — trimming mechanically instead.",
            )
            new_messages, event = self._mechanical_compact(
                round_messages, AgentRole.RESOLVER, "conflict-resolution round"
            )
            llm_failed = True
        if llm_failed:
            self._resolver_compaction_failed.add(task.id)
            self._resolver_hard_limit_exhausted.discard(task.id)
        else:
            self._resolver_compaction_failed.discard(task.id)
            self._resolver_hard_limit_exhausted.discard(task.id)
        if hard and self._messages_estimate(new_messages) >= self._messages_estimate(
            round_messages
        ):
            self._resolver_hard_limit_exhausted.add(task.id)
        task.resolver_messages[-1] = new_messages
        task.touch()
        self.manager.storage.save(task)
        self._show_event(task, event)
        if self.viewed_id == task.id:
            self.view_task(task.id)
        return True
