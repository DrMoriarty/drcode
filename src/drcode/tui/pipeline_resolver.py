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
   piece of either side was lost (the #6c065401 incident class).

Only a full pass commits (`merge_engine.finish`) — the merge into the
base branch then becomes a fast-forward. Any failure rolls the worktree
back to the PRISTINE conflicted state, marks the task `needs_human` and
logs the full report into the service events; nothing is ever committed.
"""

from __future__ import annotations

import asyncio
import json
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
from textual import work

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
from ..merge_engine import MergeReport
from ..merge_resolver import (
    ResolverOutcome,
    build_agent_context,
    conclude,
    mechanical_checks,
    prepare,
    restore_conflict,
    review_diffs,
    run_test_suite,
)
from ..models import Task
from ..providers import AgentRole
from ..tools import scratch_dir
from .blocks import ToolCallBlock
from .pipeline_stream import NodeStreamView
from .state import DrCodeState
from .usage import NETWORK_RETRIES, _is_retryable_network_error

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


class PipelineResolverLayer(DrCodeState):
    # -- triggering ----------------------------------------------------------------

    def _maybe_start_resolver(self, task_id: str) -> None:
        """Start a resolver round for `task_id` when the situation calls
        for one: worktree isolation is active, the RESOLVER role resolves
        to a model (it falls back to the coding model) and no round is
        already running. Triggered from the hook-merge failure marker and
        the startup scan; silently does nothing otherwise (a conflict
        without a model is business as usual — the human path)."""
        if self._resolver_running or not self.task_env.active:
            return
        if task_id not in self.manager.tasks:
            return
        if not self.resolved_config(AgentRole.RESOLVER).is_ready:
            return
        self.run_resolver(task_id)

    @work(exclusive=True, group="resolver")
    async def run_resolver(self, task_id: str) -> None:
        """One resolver round: prepare the conflicted merge, run the
        resolver agent, then the mandatory verification funnel."""
        self._resolver_running = True
        self._resolver_task_id = task_id
        self._resolver_generation += 1
        run_key: tuple[str, str, int] = (
            "resolver",
            task_id,
            self._resolver_generation,
        )
        task = self.manager.get(task_id)
        # A network-retry continues the current round in place instead of
        # starting a new one (mirrors the tester's resume flag).
        resume_round = task_id in self._resolver_resume_round
        self._resolver_resume_round.discard(task_id)
        run: AgentRun[ResolverDeps, str] | None = None
        cancelled = False
        # Initialized before the try: the handlers below reference them
        # even when the failure happened before ensure_root/base_branch.
        root: Path | None = None
        base = ""
        try:
            self._current_tool[task_id] = None
            self._touch_activity(task_id)
            root, iso_note = await asyncio.to_thread(self.task_env.ensure_root, task_id)
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
            self._show_event(
                task,
                f"Conflict resolver: {len(report.remaining)} conflicted "
                f"file(s) — starting an automatic resolution round.",
                kind="workflow",
            )

            # -- the agent round -----------------------------------------------------
            if not resume_round:
                task.resolver_messages.append([])
                if self.viewed_id == task_id:
                    self._write_history_line("resolving-sep", "")
            agent = build_resolver_agent(
                self.resolved_config(AgentRole.RESOLVER), root=root
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
            text: str | None = (
                None
                if resume_round
                and task.resolver_messages
                and task.resolver_messages[-1]
                else prompt
            )
            live_tokens: int | None = None
            view = NodeStreamView(
                self, task_id, record_tps=False, follow=False, track_dirty=False
            )
            while True:
                history: list[Any] | None = None
                if text is None:
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
                                async for event in node_stream:
                                    view.feed(event)
                            view.finish_node()
                            live_tokens = run_context_tokens(run)
                            self._record_session_usage(last_response_usage(run))
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
                            # Mid-round compaction gate at a tool boundary
                            # (the only safe stopping point), measured
                            # against the RESOLVER role's resolved config.
                            tokens = self._fill_over_threshold(
                                live_tokens,
                                task.resolver_messages[-1],
                                AgentRole.RESOLVER,
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

            # -- the verification funnel ----------------------------------------------
            outcome = await self._verify_resolution(task, root, report, deps)
            await self._apply_resolver_outcome(task, root, base, outcome)
        except asyncio.CancelledError:
            cancelled = True
            self._persist_resolver_messages(task, run)
            if self.viewed_id == task_id:
                self.view_task(task_id)
            raise
        except Exception as e:  # noqa: BLE001 - resolver errors must be visible
            self._persist_resolver_messages(task, run)
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
                return
            delay = min(5 * 2 ** (attempt - 1), 300)
            self._resolver_retry_at[task_id] = time.monotonic() + delay
            # The round continues where it stopped (the transcript is
            # kept) — no re-sent prompt.
            self._resolver_resume_round.add(task_id)
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
            self.refresh_status()
            self.refresh_sidebar()
            if not cancelled:
                if task.needs_human:
                    self._resolver_fail_count.pop(task_id, None)
                    self._resolver_retry_at.pop(task_id, None)
                    self._resolver_resume_round.discard(task_id)
                self._advance()

    def _retry_resolver(self) -> None:
        """A scheduled resolver retry: pick the task waiting out its
        backoff (if it still exists) and restart its round."""
        now = time.monotonic()
        task_id = next(
            (
                tid
                for tid, at in self._resolver_retry_at.items()
                if at <= now and tid in self.manager.tasks
            ),
            None,
        )
        if task_id is not None:
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
        # Level 1: mechanics.
        outcome.checks = await asyncio.to_thread(mechanical_checks, root)
        if any(not c.ok and not c.skipped for c in outcome.checks):
            return outcome
        # Level 2: tests.
        outcome.tests = await asyncio.to_thread(run_test_suite, root)
        if not outcome.tests.ok and not outcome.tests.skipped:
            return outcome
        # Level 3: intent-preservation review.
        ours_diff, theirs_diff = await asyncio.to_thread(
            review_diffs,
            root,
            [f.path for f in report.files],
        )
        reviewer = build_resolver_reviewer_agent(
            self.resolved_config(AgentRole.RESOLVER), root=root
        )
        rdeps = ResolverDeps(
            root=root,
            manager=self.manager,
            task_id=task.id,
            scratch=scratch_dir(task.id),
        )
        prompt = (
            f"Task #{task.id}: {task.title}\n"
            f"Conflicted files: {', '.join(f.path for f in report.files)}\n\n"
            f"The resolver's own report of what it did:\n"
            f"{outcome.agent_report[:4000]}\n\n"
            f"=== diff(resolved, ours) — changes relative to the task "
            f"branch {report.branch} ===\n"
            f"{ours_diff or '(no differences)'}\n\n"
            f"=== diff(resolved, theirs) — changes relative to the base "
            f"branch {report.base_branch} ===\n"
            f"{theirs_diff or '(no differences)'}\n\n"
            f"Deliver your verdict with set_review."
        )
        try:
            result = await reviewer.run(
                prompt,
                deps=rdeps,
                usage_limits=UsageLimits(request_limit=self.config.request_limit),
            )
            self._record_session_usage(result.usage)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - a failed review blocks the commit
            outcome.review_verdict = "unsure"
            outcome.review_report = (
                f"the reviewer pass failed ({type(e).__name__}: {e}) — "
                "treated as unsure"
            )
            return outcome
        rverdict = rdeps.verdicts[-1] if rdeps.verdicts else None
        outcome.review_verdict = rverdict.label if rverdict else "unsure"
        outcome.review_report = rverdict.report if rverdict else ""
        if outcome.review_verdict == "intact":
            outcome.verdict = "resolved"
        return outcome

    async def _apply_resolver_outcome(
        self, task: Task, root: Path, base: str, outcome: ResolverOutcome
    ) -> None:
        """Commit a fully verified resolution — or restore the pristine
        conflict and mark the task needs_human."""
        branch = self.task_env.branch(task.id)
        if outcome.verdict == "resolved":
            message = (
                f"Merge {base} into {branch}: resolve conflicts "
                f"(drcode automatic resolver)"
            )
            ok, msg = await asyncio.to_thread(conclude, root, message)
            if ok:
                outcome.committed = True
                task.needs_human = False
                task.touch()
                self.manager.storage.save(task)
                self._show_event(
                    task,
                    f"Conflict resolution verified and committed on "
                    f"{branch} — the merge into {base} is now a "
                    f"fast-forward (`git merge --ff-only {branch}`).\n"
                    f"{outcome.summary()}",
                    kind="workflow",
                )
                return
            outcome.review_report = (
                f"the concluding commit failed: {msg}\n{outcome.review_report}"
            )
        # Anything short of a full pass: nothing committed, conflict
        # restored, human takes over.
        restored = await asyncio.to_thread(
            restore_conflict, self.project_root, task.id, root, base
        )
        task.needs_human = True
        task.touch()
        self.manager.storage.save(task)
        note = (
            ""
            if restored
            else (
                " (the worktree could not be rolled back cleanly — inspect "
                "it by hand before resolving)"
            )
        )
        self._show_event(
            task,
            f"Automatic conflict resolution FAILED verification{note} — "
            f"NOTHING was committed, the conflict is back untouched in the "
            f"worktree, a human must take over.\n{outcome.summary()}",
            kind="workflow",
        )
        self._suggest_conflict_screen(task)

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
        note = (
            ""
            if restored
            else (
                " (the worktree could not be rolled back cleanly — inspect it by hand)"
            )
        )
        self._show_event(
            task,
            f"Automatic conflict resolution abandoned after repeated "
            f"failures{note} — the conflict is untouched in the worktree, "
            f"a human must take over.",
            kind="workflow",
        )
        self._suggest_conflict_screen(task)

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
