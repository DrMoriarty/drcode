"""Header layer of the chat view mixin stack.

The connection status in the header (icon + label + live activity
suffix), the expanded (tall) header's connection details and session
statistics, and the persistent bottom-right voice-dictation indicator.
Split out of :mod:`drcode.tui.chat_view` — see the composition in
``ChatViewLayer``.
"""

from __future__ import annotations

import statistics

from pydantic_ai.usage import RunUsage
from rich.cells import cell_len
from textual.css.query import NoMatches
from textual.widgets import Static

from ..models import TaskKind, TaskStatus
from ..providers import AgentRole
from .header import (
    HeaderConnInfo,
    HeaderSessionInfo,
    ModelStatusHeader,
)
from .state import DrCodeState
from .usage import NETWORK_RETRIES


class HeaderLayer(DrCodeState):
    def _recover_from_outage(self) -> None:
        """Error → connected transition: the outage is over, so un-block the
        tasks whose runs died with it. They were excluded from promotion via
        _agent_failed and would stay stranded until restart otherwise. If
        nothing is running, the pipeline picks the next task immediately."""
        self._agent_failed.clear()
        # _agent_dead is deliberately NOT cleared: a context-overflow
        # failure is deterministic — the outage recovery cannot fix an
        # oversized transcript.
        # The connection is back: a later failed-queue rescue (see
        # _promote_failed_queue) must fire immediately, not after a stale
        # backoff window accumulated during the outage.
        self._failed_queue_rounds = 0
        self._failed_queue_retry_at = 0.0
        # Outage-mode tasks wait on their own TCP-probe ticks: this header
        # probe just proved the gateway answers — wake them right away
        # instead of letting them wait out the remaining probe interval.
        # (The pending probe ticks stay harmless: _outage_probe skips a
        # task whose run is already back.)
        for task_id in [
            tid
            for tid, c in self._agent_net_count.items()
            if c > NETWORK_RETRIES and tid not in self._agent_runs
        ]:
            try:
                task = self.manager.get(task_id)
            except KeyError:
                self._agent_net_count.pop(task_id, None)
                continue
            if task.status is TaskStatus.ACTIVE:
                self._agent_net_retry_at.pop(task_id, None)
                self.run_agent(task_id)
            else:
                # Cancelled/re-queued while waiting: drop the stale outage
                # state (it would keep _outage_active latched).
                self._agent_net_count.pop(task_id, None)
        for task_id in [
            tid
            for tid, c in self._tester_net_count.items()
            if c > NETWORK_RETRIES and not self.tester_running
        ]:
            self._tester_retry_at.pop(task_id, None)
        self._advance()

    def refresh_connection(self) -> None:
        self.run_worker(self._check_model(), exclusive=True, group="model-check")

    def _render_model_status(self) -> None:
        """Header connection label + live activity suffix.

        The suffix is the global "is the model producing" signal: a tok/s
        figure while tokens stream in (regardless of which task is open in
        the chat), "waiting" while a run is alive but the model is silent
        (a tool executing, a non-streamed request in flight), "testing"
        while the tester works, and just the plain connection label when
        nothing runs. A voice dictation in flight outranks all of that:
        its recording indicator must stay visible for the whole utterance.

        Also re-renders the persistent bottom-right dictation indicator —
        both render paths (the immediate _sync_indicator and the 1 s
        poll) go through here, so the corner never goes stale."""
        label = self.model_status
        voice = self.voice_input.indicator
        # Keep-awake marker: the assertion is held right now (the machine
        # will not sleep under this run). Appended to whatever the label
        # shows, below the voice dictation indicator.
        awake = " · no-sleep" if self.keepawake.active else ""
        self._render_voice_indicator()
        if voice is not None:
            # The dictation indicator outranks the model-activity suffix:
            # a recording that loses its only persistent cue looks broken
            # to the user (the toast auto-dismisses after a few seconds).
            label.update(f"{self._conn_label} · {voice}")
            return
        tps = self._tokens_per_second()
        if tps is not None:
            label.update(f"{self._conn_label} · {tps:.0f} tok/s{awake}")
        elif self.agent_running:
            label.update(f"{self._conn_label} · waiting{awake}")
        elif self.tester_running:
            label.update(f"{self._conn_label} · testing{awake}")
        else:
            label.update(f"{self._conn_label}{awake}")

    # -- expanded (tall) header: connection details + session statistics ----

    def _render_voice_indicator(self) -> None:
        """Persistent bottom-right dictation indicator (#voice-indicator).

        Visible for the whole RECORDING/TRANSCRIBING session (the corner
        never times out, unlike the toasts it replaces); it disappears
        only when the controller returns to IDLE — insert, cancel or
        error. Text comes from the controller's ``corner_indicator`` so
        the header and the corner can never disagree."""
        try:
            widget = self.query_one("#voice-indicator", Static)
        except NoMatches:
            return  # teardown race: the widget is already gone
        text = self.voice_input.corner_indicator
        if text is None:
            widget.add_class("hidden")
            widget.update("")
        else:
            widget.remove_class("hidden")
            widget.update(text)
            # Pin the shrink-to-fit strip to the right edge: a docked
            # widget always lands at x=0, and the offset must track the
            # current text length (it differs between recording and
            # transcribing) and the terminal width. Re-evaluated on every
            # render — state changes plus the 1 s activity poll cover
            # terminal resizes within a second.
            widget.styles.offset = (
                max(0, self.size.width - cell_len(text) - 1),  # 1: right padding
                0,
            )

    @property
    def header_conn_info(self) -> Static:
        return self.query_one("#header-conn", HeaderConnInfo)

    @property
    def header_session_info(self) -> Static:
        return self.query_one("#header-sessions", HeaderSessionInfo)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """Compact human duration for the header stats line (e.g. 45s, 6m12s)."""
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds}s"
        return f"{seconds // 60}m{seconds % 60:02d}s"

    def _collect_session_stats(self) -> None:
        """Fold task-tree changes into the session counters.

        Runs on the 1 s poll: tasks that appeared since the previous tick
        count as created this session (by kind), tasks that reached
        COMPLETED count as processed, with the activation → completion
        span (when both timestamps exist — a task cancelled straight from
        the queue has no activation stamp and is counted without a
        duration).
        """
        tasks = self.manager.tasks
        for task_id in set(tasks) - self._session_known_ids:
            self._session_known_ids.add(task_id)
            self._session_created[tasks[task_id].kind] += 1
        for task in tasks.values():
            if (
                task.status is TaskStatus.COMPLETED
                and task.id not in self._session_completed_ids
            ):
                self._session_completed_ids.add(task.id)
                self._session_completed += 1
                if task.activated_at and task.completed_at:
                    self._session_durations.append(
                        (task.completed_at - task.activated_at).total_seconds()
                    )

    def _header_conn_text(self) -> str:
        """The expanded header's connection line: the connection label
        (dot + model, error detail on failures), URL, last probe
        round-trip, consecutive probe failures, network retry counters
        and the exact cumulative token usage of every model call of the
        session."""
        coding = self.resolved_config(AgentRole.CODING)
        if not coding.is_ready:
            return "model not configured"
        url = coding.base_url
        probe = (
            f"probe {self._last_probe_ms} ms"
            if self._last_probe_ms is not None
            else "probe —"
        )
        fails = f"probe fails: {self._probe_fail_streak}"
        open_net = sum(self._agent_net_count.values())
        retries = f"net retries: {self._net_retries_total} total"
        if open_net:
            retries += f" / {open_net} active"
        line = f"{self._conn_label} | {url} | {probe} | {fails} | {retries}"
        usage = self._header_usage_text()
        if usage:
            line += f" | {usage}"
        return line

    def _header_stats_text(self) -> str:
        """The expanded header's session-statistics line: tasks completed
        this session, average/median activation→completion time and the
        bugs:features ratio of the tasks created this session (research
        tasks are excluded from the ratio: they are neither bug fixes nor
        implemented features)."""
        durations = self._session_durations
        if durations:
            avg = statistics.fmean(durations)
            median = statistics.median(durations)
            times = (
                f"avg {self._fmt_duration(avg)} | median {self._fmt_duration(median)}"
            )
        else:
            times = "avg — | median —"
        bugs = self._session_created[TaskKind.BUG]
        features = self._session_created[TaskKind.FEATURE]
        line = (
            f"tasks done: {self._session_completed} | {times} | "
            f"bugs:features {bugs}:{features}"
        )
        return f"{line} | {self._header_github_text()}"

    def _header_github_text(self) -> str:
        """GitHub-integration status for the expanded header's bottom line:
        configured/not configured (token + repository set), active (the
        sync switch is on) and the date/time of the last successful sync
        (or the failure reason of the last attempt)."""
        settings = self.settings
        if not (settings.github_token and settings.github_repo):
            return "GitHub: not configured"
        if not settings.github_enabled:
            return "GitHub: configured, off"
        if self._github_syncing:
            status = "GitHub: on | syncing…"
        else:
            last = self._github_state.last_sync
            stamp = last.astimezone().strftime("%d.%m.%Y %H:%M") if last else "never"
            status = f"GitHub: on | last sync {stamp}"
        if self._github_last_error:
            status += f" | error: {self._github_last_error}"
        return status

    @staticmethod
    def _fmt_tokens(n: int) -> str:
        """Compact human token count for the header (e.g. 45k, 1.2M, 980)."""

        def fmt(v: float, unit: str) -> str:
            return f"{v:.1f}".rstrip("0").rstrip(".") + unit

        if n >= 1_000_000:
            return fmt(n / 1_000_000, "M")
        if n >= 1_000:
            k = f"{n / 1_000:.1f}".rstrip("0").rstrip(".")
            if float(k) < 1000:  # 999_950+ rounds up into M territory
                return f"{k}k"
            return fmt(n / 1_000_000, "M")
        return str(n)

    def _record_session_usage(self, usage: RunUsage | None) -> None:
        """Fold one model call's exact token usage into the session
        totals (main agent, tester and compaction summarizer alike) and
        refresh the expanded header line for an instant update — the 1 s
        poller keeps it live while the header is collapsed (no-op there)."""
        if usage is None:
            return
        self._session_usage.add(usage)
        self._render_header_tall()

    def _header_usage_text(self) -> str:
        """Exact cumulative token usage of the session, compactly: the
        four per-response counters pydantic-ai reports. Zero fields are
        omitted (providers without prompt caching simply show no cache
        part); an all-zero counter renders as an empty string."""
        u = self._session_usage
        parts: list[str] = []
        if u.input_tokens:
            parts.append(f"tok in {self._fmt_tokens(u.input_tokens)}")
        if u.cache_read_tokens or u.cache_write_tokens:
            parts.append(
                f"cache {self._fmt_tokens(u.cache_read_tokens)}"
                f"/{self._fmt_tokens(u.cache_write_tokens)}"
            )
        if u.output_tokens:
            parts.append(f"out {self._fmt_tokens(u.output_tokens)}")
        return " | ".join(parts)

    def _render_header_tall(self) -> None:
        """Refresh the two expanded-header lines (no-op while collapsed:
        the widgets are display:none and their content is refreshed on
        toggle / by the next poll tick anyway)."""
        header = self.query_one(ModelStatusHeader)
        if not header.has_class("-tall"):
            return
        self.header_conn_info.update(self._header_conn_text())
        self.header_session_info.update(self._header_stats_text())

    def on_model_status_header_tall_toggled(
        self, event: ModelStatusHeader.TallToggled
    ) -> None:
        """The header was clicked: fill the expanded lines immediately (the
        1 s poll keeps them live afterwards)."""
        if event.tall:
            self._render_header_tall()
