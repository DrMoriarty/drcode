"""Chat-timeline layer of the chat view mixin stack.

Rendering the viewed task's history into the chat log: the
chronological timeline build (transcript + tester rounds + service
events + pending input), the chunked history-drain batching, the delta
re-render (append-only view switches), the live-tail ledger, the
tool-call spinners and the service-event writer. Split out of
:mod:`drcode.tui.chat_view` — see the composition in ``ChatViewLayer``.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from functools import partial

from rich.console import RenderableType
from rich.markup import escape
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Static

from ..agent import (
    all_user_prompts,
    extract_display,
    is_pending_retry,
)
from ..models import (
    Clarification,
    Task,
    TaskEvent,
    TaskStatus,
)
from .blocks import (
    TASK_MENTION_RE,
    AssistantBlock,
    ReasoningBlock,
    ToolCallBlock,
    UserBlock,
)
from .state import DrCodeState, LiveTailEntry

_MIN_TIME = datetime.min.replace(tzinfo=UTC)
_CHAT_MOUNT_CHUNK = 16
_SCROLL_FOLLOW_THRESHOLD = 4
# A history drain chunk that has not progressed for this long while blocks
# are still pending is considered dead (see _poll_chat_drain).
_CHAT_DRAIN_STUCK = 2.0


def _mention_markup(text: str) -> str:
    """Wrap task-id mentions (#69c8bd30) in click markup for markup-string
    lines (tool output, tester/system/event lines, statusbar fragments).

    *text* must already be ESCAPED (rich `escape`): the wrapper only touches
    `#`+hex runs, which escaping never alters, and Textual's markup parser
    fires the `@click` action on the span (same mechanism as the
    clarification options' `[@click=…]` tags). Unknown ids are caught at
    click time by the action's "not found" toast.
    """

    def _wrap(match: re.Match[str]) -> str:
        task_id = match.group(1).lower()
        return f"[@click=app.open_task('{task_id}')]#{task_id}[/]"

    return TASK_MENTION_RE.sub(_wrap, text)


class ChatTimelineLayer(DrCodeState):
    def _log_event(self, task: Task, text: str, *, kind: str = "system") -> None:
        """Persist a service event into the task's event log.

        These lines are shown in the chat but must never reach the model:
        they live in `task.events`, outside the transcripts that
        resumable_history() reads, so a resumed agent never sees them.
        """
        task.events.append(TaskEvent(kind=kind, text=text))
        task.touch()
        self.manager.storage.save(task)

    @staticmethod
    def _format_event(event: TaskEvent) -> str:
        ts = event.at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        return f"[dim]{ts}[/] {_mention_markup(escape(event.text))}"

    def _show_event(self, task: Task, text: str, *, kind: str = "system") -> None:
        """Persist an event and, if the task is open, render it live."""
        self._log_event(task, text, kind=kind)
        if self.viewed_id == task.id:
            self._write_history_line("event", self._format_event(task.events[-1]))

    def view_task(self, task_id: str) -> None:
        # Opening a task cancels a pending new-task creation: the click on a
        # sidebar item is an explicit "leave the form" from the user.
        if self._new_task_mode:
            self._leave_new_task_mode()
        elif task_id != self.viewed_id:
            # A task switch is a context change: the shared ChatInput is
            # about to show another task's view, so an in-flight dictation
            # must not land there. Re-viewing the SAME task keeps it.
            self.voice_input.cancel(silent=True)
        # Same task already on screen and the mounted blocks still mirror a
        # rendered timeline prefix: append only the missing tail instead of
        # the wipe-and-remount (mounting cost grows with the number of
        # mounted blocks — a full rebuild of a fat history freezes the UI).
        if task_id == self.viewed_id and self._try_delta_view(task_id):
            self.refresh_todo_panel()
            self.refresh_sidebar()
            self.refresh_status()
            return
        self.viewed_id = task_id
        task = self.manager.get(task_id)
        # Wipe marker for the streaming loop: any block created before this
        # point belongs to a dead view and must be re-mounted on the next
        # delta (its accumulated content is carried over, so the live text
        # never loses the tail of an in-flight response).
        self._chat_epoch += 1
        self._invalidate_chat_delta()
        self.chat_log.remove_children()
        # Drop any un-drained chunks of a previous view: their drain is
        # stale now (the epoch guard below would drop them anyway, but the
        # new _render_task_history appends into the same list).
        self._chat_batch.clear()
        self._chat_batching = True
        try:
            timeline = self._render_task_history(task, task_id)
        finally:
            # Opening a task always lands on the latest entry (previously
            # this happened implicitly because every mount scrolled to the
            # end). The finally keeps a mid-render exception from leaving
            # batching on (later live writes would be swallowed).
            self._flush_chat_batch()
        # The log now mirrors the rendered timeline: delta re-renders can
        # append from here. A live write landing while the chunked drain is
        # still going keeps the ledger (its note rides along); noted writes
        # made by THIS render (the notes above) are dropped.
        self._live_tail.clear()
        self._rendered_count = len(timeline)
        self._rendered_prefix = self._timeline_fingerprint(timeline)
        self.refresh_todo_panel()
        self.refresh_sidebar()
        self.refresh_status()

    def _try_delta_view(self, task_id: str) -> bool:
        """Delta re-render of the currently open task: mount only the
        timeline entries that appeared since the last full render.

        Falls back to False (caller does the full rebuild) whenever the
        guarantee "mounted blocks == rendered timeline prefix (+ reconciled
        live tail)" cannot be trusted: a live write that does not mirror
        the next timeline entries (a clarification, a still-growing partial
        stream, merged prompts, an event sorting INTO the middle), the
        drain of the last rebuild still running, or the history prefix
        changed underneath (compaction rewriting the transcript). Queued
        comments and consumed-but-unpersisted prompts ARE timeline entries
        ("user-pending", canonicalized to "user" with the prompt text in
        the fingerprint — see _timeline_fingerprint), so they don't block
        the delta. Writes that DO mirror the timeline (streamed blocks,
        tool-call lines, event lines) are tracked in the live-tail ledger
        (_write_chat's note) and reconciled here — the rebuild then appends
        only the missing tail. Returns True also when nothing new at all —
        the rebuild is skipped entirely.
        """
        count = self._rendered_count
        if count is None or self._chat_batching:
            return False
        task = self.manager.get(task_id)
        if task is None:
            return False
        if task.status is TaskStatus.AWAITING_CLARIFICATION and (
            task.clarification is not self._rendered_clarification
        ):
            # A fresh clarifying question is on screen only via the full
            # render (the dialog is rendered outside the timeline).
            return False
        timeline = self._build_task_timeline(task, task_id)
        # Reconcile the live-tail ledger: every live write since the last
        # delta-consistent point must mirror the timeline entries that
        # follow the rendered prefix, in order. Streamed blocks are
        # compared by their final `_content` (the stream has ended by the
        # time a view_task re-render happens); everything else by the exact
        # text that was written. A mismatch (a partial stream still
        # growing, merged prompts, reordered entries) falls back to the
        # full rebuild.
        tail = self._live_tail
        end = count + len(tail)
        if len(timeline) < end:
            return False
        for offset, (tail_role, tail_text, tail_block) in enumerate(tail):
            entry = timeline[count + offset]
            if self._fingerprint_role(entry[2]) != self._fingerprint_role(tail_role):
                return False
            if tail_block is not None:
                if tail_block._content != entry[3]:
                    return False
            elif tail_text != entry[3]:
                return False
        if self._timeline_fingerprint(timeline[:count]) != self._rendered_prefix:
            return False
        if len(timeline) == end:
            # Nothing to mount: the live tail IS the timeline tail — fold
            # it into the rendered prefix so later deltas build on it.
            self._live_tail.clear()
            self._rendered_count = end
            self._rendered_prefix = self._timeline_fingerprint(timeline)
            return True
        for at, _, role, text in timeline[end:]:
            block = self._write_history_line(
                role, text, None if at == _MIN_TIME else at
            )
            if role == "tool-call" and isinstance(block, ToolCallBlock):
                self._reattach_tool_spinner(block, text, task_id)
        # The mounted tail is now part of the rendered prefix; the writes
        # above noted themselves into the ledger — drop the whole record.
        self._live_tail.clear()
        self._rendered_count = len(timeline)
        self._rendered_prefix = self._timeline_fingerprint(timeline)
        return True

    def _invalidate_chat_delta(self) -> None:
        """Drop the delta re-render bookkeeping: the mounted blocks no
        longer mirror a rendered timeline prefix (a live block arrived or
        the log was wiped), so the next view_task of the same task must do
        a full rebuild."""
        self._rendered_count = None
        self._rendered_prefix = ()
        self._live_tail.clear()

    @staticmethod
    def _timeline_fingerprint(
        timeline: list[tuple[datetime, int, str, str]],
    ) -> tuple[tuple[str, ...], ...]:
        """Compact identity of a rendered timeline prefix: (timestamp,
        role) of every entry — plus the text for user prompts. Text is
        intentionally left out for everything else: a prefix whose entries
        kept their positions but changed text means the transcript was
        rewritten (compaction), which the length check and the fingerprint
        both tend to catch anyway; timestamps+roles catch the reordering
        case (an event sorted into the middle).

        "user-pending" canonicalizes to "user" (see _fingerprint_role): a
        queued prompt rendered as a pending bubble becomes a regular user
        entry (same timestamp — the request is backdated to the typing
        time) once the run consumes it into the transcript; the mounted
        UserBlock is identical, so the transition must not force a rebuild.
        The user-prompt TEXT is part of the fingerprint precisely for this
        transition: a consumed batch (comments merged into one prompt,
        workflow instructions prepended) changes it, and only a rebuild
        replaces the stale bubble(s)."""
        out: list[tuple[str, ...]] = []
        for at, _, role, text in timeline:
            if role in ("user", "user-pending"):
                out.append((at.isoformat(), "user", text))
            else:
                out.append((at.isoformat(), role))
        return tuple(out)

    @staticmethod
    def _fingerprint_role(role: str) -> str:
        """Canonical timeline role: "user-pending" and "user" render the
        same UserBlock and may swap underneath a mounted one (the run
        consuming the queued prompt into the transcript)."""
        return "user" if role == "user-pending" else role

    def _build_task_timeline(
        self, task: Task, task_id: str
    ) -> list[tuple[datetime, int, str, str]]:
        """Build the task's full history as ONE chronological timeline of
        (timestamp, sequence, role, text) entries. Both the serialized
        transcripts (per-message timestamps) and the service events
        (TaskEvent.at) carry wall-clock stamps, so a stable sort by
        timestamp reproduces exactly the order in which the entries were
        rendered live — the same order on every (re)open, instead of
        clumping all events at the tail."""
        timeline: list[tuple[datetime, int, str, str]] = []
        seq = 0

        def _add(at: datetime | None, role: str, text: str) -> None:
            nonlocal seq
            timeline.append((at or _MIN_TIME, seq, role, text))
            seq += 1

        for at, role, text in extract_display(task.messages):
            _add(at, role, text)
        # Tester rounds: the synthetic prompt ("Verify completed task #…")
        # is not shown; each round is announced by its own "testing"
        # separator anchored at the round's first message.
        for round_messages in task.tester_messages:
            if not round_messages:
                continue
            round_items = extract_display(round_messages)
            round_at = next((at for at, _, _ in round_items if at is not None), None)
            _add(round_at, "testing-sep", "")
            for at, role, text in round_items:
                if role == "user":
                    continue
                if role == "tool-call":
                    text = f"tester: {text}"
                _add(at, role, text)
        # Service events (failures, skips) — human-only, with timestamps;
        # interleaved at the position they occurred at.
        for event in task.events:
            _add(event.at, "event", self._format_event(event))
        # Queued user input — interleaved by the moment it was TYPED, not
        # pinned to the tail: comments sent while the task was queued or
        # running belong between the messages that were on screen then.
        for entry in task.pending_input:
            if is_pending_retry(task.messages, entry.text):
                # The failed/interrupted run left this prompt in the
                # transcript: it is already rendered from task.messages
                # above — writing the pending copy again duplicates it.
                continue
            _add(entry.at, "user-pending", entry.text)
        # A prompt a running agent has already consumed (popped from
        # pending_input) but that is not persisted into the transcript yet
        # (the first node boundary saves it): without this record a
        # re-render in between shows an empty chat — the brief would
        # disappear from the screen until the next re-render.
        in_flight = self._consumed_prompts.get(task_id)
        if in_flight is not None and in_flight[0] not in all_user_prompts(
            task.messages
        ):
            for entry in in_flight[1]:
                _add(entry.at, "user-pending", entry.text)
        timeline.sort(key=lambda item: item[:2])
        return timeline

    def _render_task_history(
        self, task: Task, task_id: str
    ) -> list[tuple[datetime, int, str, str]]:
        """Write the task's full history into the (batched) chat log.
        Returns the rendered timeline (for the delta re-render bookkeeping
        in view_task)."""
        self._chat_has_content = False
        self._pending_rendered = []
        self._clear_stream_label()
        timeline = self._build_task_timeline(task, task_id)
        for at, _, role, text in timeline:
            block = self._write_history_line(
                role, text, None if at == _MIN_TIME else at
            )
            if role == "tool-call" and isinstance(block, ToolCallBlock):
                # A tool still executing (run in flight) gets its spinner
                # back on the freshly rendered block.
                self._reattach_tool_spinner(block, text, task_id)
        if task.status is TaskStatus.AWAITING_CLARIFICATION and task.clarification:
            self._write_clarification(task.clarification)
            self._rendered_clarification = task.clarification
        return timeline

    def _chat_at_bottom(self) -> bool:
        """True when the reader is tracking the chat tail (auto-scroll follows).

        Allows a small threshold of trailing rows below the viewport edge:
        markdown reflow can suddenly add a couple of lines and a fresh model
        chunk arrives at the same time, so the tail edge jumps below the
        visible area. A 1-line tolerance would then falsely report "not at
        bottom" and stop following the tail; within this threshold we still
        treat the reader as tracking the tail.

        Tolerance absorbs rounding of fractional scroll offsets.
        """
        log = self.chat_log
        return log.scroll_offset.y >= log.max_scroll_y - _SCROLL_FOLLOW_THRESHOLD

    def _stream_follow(self) -> None:
        """Keep the chat tail visible while streaming, coalesced to ~10 Hz.

        Called per streamed delta; each scroll_end schedules a refresh+scroll
        pass over the (growing) chat log — doing that per delta saturates the
        event loop and makes keyboard/mouse input unresponsive.
        """
        now = time.monotonic()
        if now - self._last_follow >= 0.1:
            self._last_follow = now
            self.chat_log.call_after_refresh(self.chat_log.scroll_end, animate=False)

    # -- tool-call spinners ------------------------------------------------------

    def _tool_call_started(
        self,
        run_key: tuple[str, str, int],
        task_id: str,
        call_id: str,
        call_text: str,
        block: ToolCallBlock,
    ) -> None:
        """Register a tool call whose execution just began (spinner on)."""
        self._active_tools.setdefault(run_key, {})[call_id] = (
            task_id,
            call_text,
            block,
        )

    def _tool_call_finished(self, call_id: str) -> None:
        """Tool result arrived: stop the spinner of its block (in whichever
        run registered it — agent and tester never overlap, but the lookup
        is over all runs anyway)."""
        for per_run in self._active_tools.values():
            entry = per_run.pop(call_id, None)
            if entry is not None:
                entry[2].stop_spinner()

    def _stop_run_tools(self, run_key: tuple[str, str, int]) -> None:
        """End-of-run cleanup: anything still marked running for this run
        (a result event lost to an abort/error) loses its spinner."""
        for _task_id, _text, block in self._active_tools.pop(run_key, {}).values():
            block.stop_spinner()

    def _reattach_tool_spinner(
        self, block: ToolCallBlock, call_text: str, task_id: str
    ) -> None:
        """After a mid-run view re-render, hand a still-running call's
        spinner over to the freshly rendered block (matched by the exact
        rendered call text — same tool with the same arguments)."""
        for per_run in self._active_tools.values():
            for call_id, (tid, text, _old) in per_run.items():
                if tid == task_id and text == call_text:
                    per_run[call_id] = (tid, text, block)
                    block.start_spinner()
                    return

    def _write_chat(
        self,
        content: RenderableType | Widget,
        *,
        expand: bool = False,
        note: LiveTailEntry | None = None,
    ) -> None:
        """Mount one block to the chat log.

        Scrolls to the end only when the user is already looking at the
        bottom: a scrolled-up reader must not be yanked down by new tool
        calls / reasoning blocks arriving while the agent runs.

        While batching (view_task's history rebuild and its chunked drain),
        the block is only collected — _flush_chat_batch() mounts the
        timeline in chunks afterwards.

        *note* marks the write as one that mirrors the NEXT timeline entry
        of the open task: (role, text, block) — `text` is the exact
        timeline-comparable string (for markup lines and tool calls), or
        `block` is a ReasoningBlock/AssistantBlock whose `_content` holds
        the final text once the stream ends. Noted writes extend the delta
        bookkeeping (the live-tail ledger) instead of invalidating it:
        _try_delta_view reconciles the ledger against the timeline and
        appends only the missing tail. A write without a note (a
        clarification dialog, an error line, anything outside the
        timeline) invalidates as before.
        """
        if isinstance(content, Widget):
            content.add_class("chat-block")
        else:
            content = Static(content, expand=expand, classes="chat-block")
        if note is not None and self._rendered_count is not None:
            # Live write whose timeline counterpart is known: keep the
            # "mounted log == rendered timeline (+ reconciled live tail)"
            # invariant instead of dropping it. (During a batched rebuild
            # the counters are stale/None — view_task resets both the
            # counters and the ledger after the render.)
            self._live_tail.append(note)
        elif note is None:
            # Any live (non-timeline) mount — a streamed block, an event
            # line, a clarification — breaks the "log == rendered timeline
            # prefix" invariant the delta re-render relies on: the next
            # view_task must rebuild fully.
            self._invalidate_chat_delta()
        if self._chat_batching:
            self._chat_batch.append(content)
            self._chat_has_content = True
        else:
            try:
                follow = self._chat_at_bottom()
                self.chat_log.mount(content)
                if follow:
                    self.chat_log.scroll_end(animate=False)
            except NoMatches:
                # Teardown race: a worker streaming into the open chat (a
                # tester verdict, an agent delta, an event line) can hit an
                # already-parsed-away DOM — the app is being dismantled
                # under it. Drop the write (same pattern as _poll_activity
                # and _drain_chat_batch): the transcript is persisted, the
                # next session renders it from there.
                return
            self._chat_has_content = True

    def _flush_chat_batch(self) -> None:
        """Start mounting the blocks collected by a batched history rebuild.

        Mounts in chunks, yielding to the event loop between them: one
        mounted block costs a few ms (textual re-applies styles over all
        mounted widgets per mount), so a fat 600-block history would freeze
        keyboard/mouse input for ~2s if mounted in a single synchronous
        pass. Late live writes (a stream starting while the drain is still
        going) keep appending to _chat_batch — batching stays on until the
        last chunk — and ride along with the remaining chunks.
        """
        self._chat_batching = True
        self._chat_batch_epoch = self._chat_epoch
        self._chat_drain_at = time.monotonic()
        self._drain_chat_batch()

    def _drain_chat_batch(self, epoch: int | None = None) -> None:
        """Mount one chunk of the pending batch, reschedule for the rest.

        *epoch* is the `_chat_epoch` value captured when THIS invocation was
        scheduled: a view switch since then makes the chain stale, and the
        guard below actually detects that (comparing the shared attributes
        at run time never could — both always hold the current values).
        A stale chain returns without touching `_chat_batching`/`_chat_batch`:
        the newer view_task that bumped the epoch already re-flushed and owns
        the shared state.
        """
        self._chat_drain_at = time.monotonic()
        if epoch is None:
            epoch = self._chat_batch_epoch
        if epoch != self._chat_epoch:
            # The view switched mid-drain: the new view_task wiped the log
            # and owns it now — drop the stale tail. (A watchdog re-kick
            # under an orphaned drain — epoch bumped without a re-flush,
            # e.g. action_delete — cleans the shared state up instead.)
            if self._chat_batching and self._chat_batch_epoch == epoch:
                self._chat_batching = False
                self._chat_batch.clear()
            return
        chunk = self._chat_batch[:_CHAT_MOUNT_CHUNK]
        del self._chat_batch[:_CHAT_MOUNT_CHUNK]
        try:
            if chunk:
                self.chat_log.mount(*chunk)
            if self._chat_batch:
                self._schedule_drain(self._chat_epoch)
            else:
                self._chat_batching = False
                # Opening a task always lands on the latest entry (previously
                # this happened implicitly because every mount scrolled to
                # the end).
                self.chat_log.scroll_end(animate=False)
        except NoMatches:
            # Teardown race: the drain chain (call_after_refresh / the 0.1s
            # timer fallback) can fire into an already-parsed-away DOM —
            # nothing left to render into. Drop the batch (same pattern as
            # _poll_activity's NoMatches catch).
            self._chat_batching = False
            self._chat_batch.clear()
        except Exception:
            # Never leave batching stuck on: later live writes would be
            # silently swallowed.
            self._chat_batching = False
            self._chat_batch.clear()
            raise

    def _schedule_drain(self, epoch: int) -> None:
        """Schedule the next drain chunk. `call_after_refresh` may silently
        NOT schedule its callback (it returns False when the message pump is
        closed/closing) — its InvokeLater message also only ever runs on a
        clean screen idle, so under a busy terminal (streaming deltas,
        spinners, tickers keeping widgets dirty) the chain can starve.
        The fallbacks here plus the _poll_chat_drain watchdog keep the drain
        self-healing: a dropped or starved schedule must never leave a
        permanently truncated history with batching stuck on."""
        if self.call_after_refresh(self._drain_chat_batch, epoch):
            return
        # InvokeLater could not be queued — try a plain Callback message
        # (processed in this pump's own queue, not gated on screen idle).
        if not self.call_later(self._drain_chat_batch, epoch):
            # Last resort while the app is still alive: a timer. If the app
            # is really shutting down this never fires — nothing left to
            # render into anyway.
            self.set_timer(0.1, partial(self._drain_chat_batch, epoch))

    def _poll_chat_drain(self) -> None:
        """Watchdog for the chunked history drain (1 s poll): if a drain is
        pending but no chunk has progressed for a while, the reschedule
        chain died (a dropped `call_after_refresh` schedule, or InvokeLater
        callbacks starved by a continuously dirty screen — both invisible
        to headless run_test() repros). Re-kick it; _drain_chat_batch's
        epoch guard drops the tail if the view moved on in the meantime."""
        if (
            self._chat_batching
            and self._chat_batch
            and time.monotonic() - self._chat_drain_at > _CHAT_DRAIN_STUCK
        ):
            self._drain_chat_batch()

    def _write_history_line(
        self, role: str, text: str, at: datetime | None = None
    ) -> Widget | None:
        """Render one history entry into the chat log.

        Returns the mounted block widget (None for markup-string lines) so
        callers can tweak it afterwards — view_task re-attaches the running
        spinner to tool-call blocks this way.
        """
        block: Widget | None = None
        # Live-tail note: (role, timeline-text, streamed-block-or-None).
        # Timeline roles note themselves (the ledger reconciles them
        # against the next timeline entry); "system" and the summary's
        # separator line are outside the timeline and keep invalidating.
        # For streamed blocks (reasoning/assistant) the note carries the
        # widget — its `_content` holds the final text once the stream
        # ends; for everything else the exact text compares.
        note: LiveTailEntry | None = (role, text, None)
        if role == "user":
            block = UserBlock(text)
        elif role == "user-pending":
            self._pending_rendered.append(text)
            block = UserBlock(text)
        elif role == "reasoning":
            block = ReasoningBlock(at)
            block.set_content(text)
        elif role == "tool-call":
            block = ToolCallBlock(text, at)
        elif role == "assistant":
            block = AssistantBlock(text)
        elif role == "tool":
            self._write_chat(f"[dim]{_mention_markup(escape(text))}[/]", note=note)
        elif role == "tester":
            # tester output is raw model text: escape it before the markup
            # parser sees it (a stray bracket would crash rendering), then
            # wrap task-id mentions into click links.
            self._write_chat(
                f"[bold magenta]tester ›[/] {_mention_markup(escape(text))}",
                note=note,
            )
        elif role == "testing-sep":
            label = f"testing: {text}" if text else "testing"
            self._write_chat(f"[bold magenta]──── {label} ────[/]", note=note)
        elif role == "summary":
            # Compaction summary: a marker separator, then the summary
            # itself rendered as markdown (it is the model's output the
            # agent continues from, not a user message).
            self._write_chat("[bold magenta]──── context compacted ────[/]")
            block = AssistantBlock(text)
        elif role == "event":
            self._write_chat(f"[bold red]event ›[/] {text}", note=note)
        elif role == "system":
            note = None
            self._write_chat(f"[bold red]error ›[/] {_mention_markup(escape(text))}")
        if block is not None:
            if isinstance(block, ReasoningBlock | AssistantBlock):
                self._write_chat(block, note=(role, text, block))
            elif note is not None:
                self._write_chat(block, note=(role, text, None))
            else:
                self._write_chat(block)
        return block

    def _write_clarification(self, clarification: Clarification) -> None:
        lines = [
            f"[bold blue]drcode ❓[/] {_mention_markup(escape(clarification.question))}"
        ]
        for i, option in enumerate(clarification.options):
            lines.append(
                f"  ▸ [@click=app.pick_option({i})]"
                f"[underline cyan]{escape(option)}[/][/]"
            )
        lines.append("[dim]Click an option or answer in the input below.[/]")
        self._write_chat("\n".join(lines))
