"""Network-outage handling and the live tok/s estimate.

Split out of the former monolithic `tui/pipeline.py`: the model-status
probe feeding the header connection line, the outage-mode retry loop
(TCP-gated probes instead of burning model requests at a dead
gateway) and the streamed-output rate estimator (header tok/s).
"""

from __future__ import annotations

import time
from collections import deque
from functools import partial

from textual.css.query import NoMatches

from ..agent import probe_model
from ..models import Task, TaskStatus
from ..providers import AgentRole
from .state import DrCodeState
from .usage import NETWORK_OUTAGE_PROBE_INTERVAL, NETWORK_RETRIES, tcp_reachable


class PipelineNetLayer(DrCodeState):
    async def _check_model(self) -> None:
        try:
            await self._check_model_inner()
        except NoMatches:
            # Teardown race: the probe round-trip can outlive the app (ESC
            # while the gateway answers slowly) — the header label is gone
            # by the time the worker resumes; the indicator is cosmetic,
            # so a vanished widget just ends the check quietly.
            pass

    async def _check_model_inner(self) -> None:
        label = self.model_status
        was_error = label.has_class("error")
        # The header shows the CODING model's connection: the role the
        # pipeline spends most of its life in.
        config = self._coding_config()
        if not config.is_ready:
            self._conn_label = "○ model not configured"
            self._last_probe_ms = None
            label.remove_class("ok", "error")
            label.add_class("warn")
            self._render_model_status()
            return
        model = config.model
        started = time.monotonic()
        # The cached health_check_url is probed first (cheaper, no generation);
        # the checked_url itself is discarded here — the header only needs the
        # ok/detail status.
        ok, detail, _checked_url = await probe_model(
            model,
            config.base_url,
            config.api_key,
            health_check_url=config.health_check_url,
        )
        # Round-trip of the probe itself, shown on the expanded header line.
        self._last_probe_ms = int((time.monotonic() - started) * 1000)
        if not ok:
            self._probe_fail_streak += 1
            self._conn_label = f"✗ {model} — {detail}"
            label.remove_class("ok", "warn")
            label.add_class("error")
            self._render_model_status()
            return
        self._probe_fail_streak = 0
        self._conn_label = f"● {model}"
        label.remove_class("warn", "error")
        label.add_class("ok")
        self._render_model_status()
        if was_error:
            self._recover_from_outage()

    def _note_network_recovered(self, task: Task, counts: dict[str, int]) -> None:
        """Outage exit on success: when the task's consecutive-failure count
        says it was in outage mode, one service event announces the recovery
        before the caller pops the counter. Called from the per-request
        success resets of run_agent/run_tester — the first model call that
        gets through ends the outage."""
        if counts.get(task.id, 0) > NETWORK_RETRIES:
            self._show_event(
                task,
                "Network restored: the model gateway responds again — resuming.",
            )

    def _outage_active(self) -> bool:
        """True while any task is in the outage-mode wait (its consecutive
        network-failure count is past the ordinary retry budget). Gates the
        failed-queue rescue (see _promote_failed_queue): rescue rounds must
        not fire model requests while the outage probe says the gateway is
        down. Session-only by construction — the counts live in dicts that
        every success/reset/pop clears."""
        return any(
            c > NETWORK_RETRIES
            for c in [*self._agent_net_count.values(), *self._tester_net_count.values()]
        )

    def _outage_retry_tick(self, task_id: str, role: AgentRole) -> None:
        """Outage-mode retry tick: gate the next model attempt behind a
        cheap TCP connect to the gateway host. Reachable — the ordinary
        re-run (run_agent / a pipeline advance for the tester) fires;
        still down — only re-probe at NETWORK_OUTAGE_PROBE_INTERVAL,
        without burning full model requests (and the transcript entries
        they would leave behind) on a host that cannot even accept a
        TCP connection."""
        if self._exit or not self.is_running:
            return
        self.run_worker(
            partial(self._outage_probe, task_id, role),
            name=f"outage probe {task_id}",
            group=f"outage-{task_id}",
            exclusive=True,
        )

    async def _outage_probe(self, task_id: str, role: AgentRole) -> None:
        tester = role is AgentRole.TESTER
        try:
            task = self.manager.get(task_id)
        except KeyError:
            return
        expected = TaskStatus.TESTING if tester else TaskStatus.ACTIVE
        if task.status is not expected:
            # The task left the waiting state while the probe was pending
            # (cancelled, re-queued, completed): drop the outage state, or
            # a stale count would keep _outage_active (and the failed-queue
            # rescue coordination) latched forever.
            if tester:
                self._tester_net_count.pop(task_id, None)
            else:
                self._agent_net_count.pop(task_id, None)
            return
        base_url = self.resolved_config(role).base_url
        if await tcp_reachable(base_url):
            # Only when nothing took over in the meantime: a recovery
            # (probe/_recover_from_outage) may already have re-run the task.
            if tester:
                if not self.tester_running:
                    self._tester_retry_at.pop(task_id, None)
                    self._advance()
            elif task_id not in self._agent_runs:
                self.run_agent(task_id)
        else:
            retry_at = time.monotonic() + NETWORK_OUTAGE_PROBE_INTERVAL
            if tester:
                self._tester_retry_at[task_id] = retry_at
            else:
                self._agent_net_retry_at[task_id] = retry_at
            self._schedule_pipeline_timer(
                NETWORK_OUTAGE_PROBE_INTERVAL,
                partial(self._outage_retry_tick, task_id, role),
            )

    # -- live output rate (header tok/s) --------------------------------------

    # Window over which the streamed output rate is averaged, and the minimum
    # span of samples before a rate is shown at all (the very first second of
    # a stream would otherwise produce a wildly spiky figure).
    _TPS_WINDOW = 6.0
    _TPS_MIN_SPAN = 1.5

    def _record_stream_tokens(self, task_id: str, text: str) -> None:
        """Record a streamed delta of `task_id`'s run for the live tok/s
        estimate.

        Streaming APIs emit deltas roughly token-sized, so the character
        count divided by the conventional ~4 chars/token is a good enough
        approximation — the point is the visual "is it producing" signal,
        not billing-grade accounting. With parallel slots the samples of
        all running tasks are summed (one aggregate "is the model
        producing" figure).
        """
        now = time.monotonic()
        samples = self._tps_samples.setdefault(task_id, deque())
        samples.append((now, max(1, len(text) // 4)))
        self._prune_tps_samples(now)

    def _prune_tps_samples(self, now: float) -> None:
        cutoff = now - self._TPS_WINDOW
        for task_id in list(self._tps_samples):
            samples = self._tps_samples[task_id]
            while samples and samples[0][0] < cutoff:
                samples.popleft()
            if not samples:
                del self._tps_samples[task_id]

    def _tokens_per_second(self) -> float | None:
        """Aggregate output rate over the sliding window across every
        run's samples (None = nothing measurable: no recent stream
        output, or too little of it to average yet)."""
        now = time.monotonic()
        self._prune_tps_samples(now)
        all_samples: list[tuple[float, int]] = []
        for samples in self._tps_samples.values():
            all_samples.extend(samples)
        if not all_samples:
            return None
        span = now - min(t for t, _ in all_samples)
        if span < self._TPS_MIN_SPAN:
            return None
        return sum(n for _, n in all_samples) / span
