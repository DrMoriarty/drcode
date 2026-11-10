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
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Static

from ..agent import (
    all_user_prompts,
    extract_display,
    is_pending_retry,
)
from ..core.blocks import TASK_MENTION_RE
from ..models import (
    Clarification,
    Task,
    TaskEvent,
    TaskStatus,
)
from .blocks import (
    AssistantBlock,
    ReasoningBlock,
    ToolCallBlock,
    UserBlock,
)
from .state import DrCodeState, LiveTailEntry

_MIN_TIME = datetime.min.replace(tzinfo=UTC)
_CHAT_MOUNT_CHUNK = 16
# A history drain chunk that has not progressed for this long while blocks
# are still pending is considered dead (see _poll_chat_drain).
_CHAT_DRAIN_STUCK = 2.0
# Minimum gap between stuck-drain diagnostic events (see _report_drain_stall).
_CHAT_DRAIN_DIAG_GAP = 10.0
# Live writes (tool results, tester output) longer than this are NOT mounted
# as one Static: one fat block is a single long render pass on the event loop
# (textual also re-applies styles over all mounted widgets per mount) and the
# keyboard echo lags around it. They are split into parts and mounted through
# the chunked drain instead (same machinery as the history rebuild).
_CHAT_LIVE_MOUNT_LIMIT = 16_000
# Split-part target size: small enough that one drain tick stays cheap.
_CHAT_LIVE_PART = 4_000
# Per-tick mount budget of the drain, in "mount cost" units (the text length
# of the split parts; plain rebuild blocks carry no cost and chunk by count
# only, as before).
_CHAT_MOUNT_BYTES = 16_384

# A rich-markup tag (skipping backslash-escaped brackets, which rich's
# escape() emits for literal text).
_RICH_TAG_RE = re.compile(r"(?<!\\)\[[^\[\]]*\]")


def _full_escape(text: str) -> str:
    r"""Fully escape raw text (model/tool output) for a markup string.

    rich's escape() only escapes tag-LIKE sequences (`[a-z#/@…]`), so a
    stray bracket sequence such as `[?]` passes through raw — and then
    the splitter's tag stack (which counts any `[...]`) disagrees with
    the Textual parser (which treats it as literal text), producing
    extra closing `[/]` on split and a MarkupError. Escaping EVERY
    bracket restores the agreement: the splitter's lookbehind skips
    `\[?` and Textual un-escapes it back to a literal `[`.

    Backslashes are deliberately left alone: Textual renders a `\\`
    pair literally (it does not un-escape it), so doubling them would
    show two backslashes for every one in the raw output; a raw `\`
    before a bracket already shields it from both parsers. The only
    residual quirk — a raw `\` directly abutting one of OUR structural
    tags (e.g. the final `[/]` of the wrapper) makes that tag render
    literally — is shared with rich's own escape() and is cosmetic (no
    MarkupError: the swallowed tag never opens, so closers never
    outnumber opens).
    """

    return text.replace("[", "\\[")


def _safe_markup_cut(line: str, limit: int) -> int:
    """Largest cut index <= limit that does not split a markup tag or a
    backslash escape in half — the fallback for a single over-long line
    (minified output, a base64 blob) that has no newline to split at."""
    depth = 0
    cut = 0
    prev = ""
    for i, ch in enumerate(line[:limit]):
        if ch == "[" and prev != "\\":
            if depth == 0:
                cut = i  # cutting right before a tag is safe
            depth += 1
        elif ch == "]" and prev != "\\":
            depth = max(depth - 1, 0)
            cut = i + 1  # cutting right after a closed tag is safe
        elif depth == 0:
            cut = i + 1
        prev = ch
    # Never split a `\[` escape: the next part must not start with a bare `[`.
    if cut > 1 and line[cut - 1] == "\\":
        cut -= 1
    # Pathological (a tag longer than limit): force a cut rather than loop.
    return cut if cut > 0 else max(limit, 1)


def _split_chat_text(text: str, limit: int) -> list[str]:
    """Split an over-long markup string into parts of at most ~limit
    characters each, so each part renders standalone: split at line
    boundaries (hard-splitting individual over-long lines at markup-safe
    positions), close the still-open markup tags at each part end and
    re-open them at the next part start."""
    if len(text) <= limit:
        return [text]
    lines: list[str] = []
    # Whether a newline precedes this line-piece in the original text
    # (False between hard-split pieces of one over-long line — joining
    # those with "\n" would invent line breaks that were never there).
    nl_before: list[bool] = []
    for li, raw in enumerate(text.split("\n")):
        first_piece = True
        while len(raw) > limit:
            cut = _safe_markup_cut(raw, limit)
            lines.append(raw[:cut])
            nl_before.append(first_piece and li > 0)
            raw = raw[cut:]
            first_piece = False
        lines.append(raw)
        nl_before.append(first_piece and li > 0)
    # Open-tag stack after each line: parts are closed/reopened so each
    # renders standalone (an unclosed [dim] would raise MarkupError).
    stack: list[str] = []
    stacks: list[list[str]] = []
    for line in lines:
        for m in _RICH_TAG_RE.finditer(line):
            tag = m.group(0)
            if tag.startswith("[/"):
                if stack:
                    stack.pop()
            else:
                stack.append(tag)
        stacks.append(stack.copy())
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    reopen = ""
    for i, line in enumerate(lines):
        sep = "\n" if nl_before[i] else ""
        if buf and size + len(sep) + len(line) > limit:
            # The separator that preceded this line belongs to the END of
            # the closing part (before the appended closers) — dropping it
            # would silently swallow one newline per part boundary.
            parts.append(reopen + "".join(buf) + sep + "[/]" * len(stacks[i - 1]))
            buf, size = [], 0
            reopen = "".join(stacks[i - 1])
            buf.append(reopen + line)
            size = len(reopen) + len(line)
        else:
            buf.append(sep + line if buf else line)
            size += len(sep) + len(line)
    if buf:
        parts.append(reopen + "".join(buf))
    return parts


class _ChatPart(Static):
    """One part of a split over-long live write. Carries the part's text
    length as its mount cost — _drain_chat_batch bounds each tick by the
    summed cost, so a fat write spreads over several event-loop ticks
    instead of one long render pass."""

    mount_cost: int

    def __init__(self, text: str, expand: bool, *, cont: bool) -> None:
        # `cont` (a middle part) drops the inter-block margin so the parts
        # of one logical entry render seamlessly (see the .chat-cont CSS).
        classes = "chat-block chat-cont" if cont else "chat-block"
        super().__init__(text, expand=expand, classes=classes)
        self.mount_cost = len(text)


def _mention_markup(text: str) -> str:
    """Wrap task-id mentions (#69c8bd30) in click markup for markup-string
    lines (tool output, tester/system/event lines, statusbar fragments).

    *text* must already be ESCAPED (`_full_escape`): the wrapper only
    touches `#`+hex runs, which escaping never alters, and Textual's
    markup parser
    fires the `@click` action on the span (same mechanism as the
    clarification options' `[@click=…]` tags). Unknown ids are caught at
    click time by the action's "not found" toast.
    """

    def _wrap(match: re.Match[str]) -> str:
        task_id = match.group(1).lower()
        return f"[@click=app.open_task('{task_id}')]#{task_id}[/]"

    return TASK_MENTION_RE.sub(_wrap, text)


# Marker the system prompts USED to tell the model to prefix its OWN
# initiative options with (canonical English form, plus the localized
# fallbacks a model may drift into). Matched case-insensitively anywhere
# in the option text — models put it at the head AND at the tail. New
# clarifications carry initiative options structurally (Clarification
# .agent_proposals); this fallback only cleans up OLD stored options.
AGENT_SUGGESTION_RE = re.compile(
    r"agent's suggestion|not from your request|предложение агента|моя инициатива",
    re.IGNORECASE,
)

# A parenthesized span that contains the marker: "(agent's suggestion,
# not from your request)" and variants. Used to cut the marker out of
# the middle/tail of an option.
AGENT_SUGGESTION_PAREN_RE = re.compile(
    r"\([^()]*(?:agent's suggestion|not from your request"
    r"|предложение агента|моя инициатива)[^()]*\)",
    re.IGNORECASE,
)


def _clean_marker_removal(text: str) -> str:
    """Tidy up a string after a parenthesized marker was cut out:
    collapse doubled spaces, fix a space before punctuation, drop a
    leading/trailing hanging comma or dash."""
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip().strip(",;:-–—").strip()


def _split_agent_marker(option: str) -> tuple[bool, str]:
    """Split an agent-initiative marker off a clarification option.

    Returns ``(is_agent_suggestion, display_text)``. The marker stays in
    the stored option (it is part of the answer the model receives when
    the user clicks) — only the RENDERED text loses it, replaced by a ★.

    The marker is searched over the WHOLE option text (models put it at
    the head and at the tail alike). At the head the canonical shape is
    "(marker …): real text" — cut through the colon; in the middle or at
    the tail the parenthesized "(…marker…)" span is removed and the
    leftovers (doubled spaces, hanging punctuation) cleaned up.
    """
    match = AGENT_SUGGESTION_RE.search(option)
    if match is None:
        return False, option
    # Marker near the head: canonical "(marker …): real text" prefix —
    # cut through the colon that closes it. Without a colon, strip the
    # parenthesized marker if it is one, else keep the text whole.
    if match.start() < 60:
        _, sep, rest = option.partition(":")
        if sep and AGENT_SUGGESTION_RE.search(option[: option.index(":")]):
            return True, rest.strip()
        stripped = AGENT_SUGGESTION_PAREN_RE.sub("", option)
        if stripped != option:
            return True, _clean_marker_removal(stripped)
        return True, option.strip()
    # Marker in the middle or at the tail: remove the parenthesized span
    # holding it (if the text is parenthesized there) and tidy up.
    stripped = AGENT_SUGGESTION_PAREN_RE.sub("", option)
    if stripped != option:
        return True, _clean_marker_removal(stripped)
    return True, option.strip()


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
        return f"[dim]{ts}[/] {_mention_markup(_full_escape(event.text))}"

    def _show_event(self, task: Task, text: str, *, kind: str = "system") -> None:
        """Persist an event and, if the task is open, render it live."""
        self._log_event(task, text, kind=kind)
        if self.viewed_id == task.id:
            self._write_history_line("event", self._format_event(task.events[-1]))

    def _show_event_once(self, task: Task, text: str, *, kind: str = "system") -> None:
        """`_show_event` deduplicated per task and wording: an identical
        line already sitting in the task's event log is not appended
        again.

        For announcements that fire on EVERY activation of a run — the
        worktree-isolation notes: a network retry re-enters `run_agent`
        whole (`_agent_turn_network_error` reschedules it), the worktree
        is merely reused, and without the dedupe each retry would repeat
        the same "Task isolated in worktree …" line into the chat."""
        if not any(e.text == text for e in task.events):
            self._show_event(task, text, kind=kind)

    def _isolation_pin_note(self, task: Task, *, save: bool) -> None:
        """Run-boundary bookkeeping for the isolation pin
        (`task.isolation_pinned`, stamped by `ensure_root` on the task's
        first activation): persist a fresh stamp and — when the run
        activated under a pin that differs from the LIVE
        `task_isolation` setting — say so in the task's chat (the user
        toggled the setting mid-task; the task keeps its mode for the
        rest of its life). Deduplicated per task and wording: the
        identical line is not appended twice."""
        pin = task.isolation_pinned
        if pin is None:
            return
        if pin != self.task_env.active:
            text = (
                "Task pinned to worktree isolation (the setting is now off)."
                if pin
                else "Task pinned to the project directory (worktree "
                "isolation is now on)."
            )
            if not any(e.text == text for e in task.events):
                # _log_event persists the task — a freshly stamped pin
                # rides along with the event.
                self._show_event(task, text)
            return
        if save:
            # Pin matches the live setting: no event, but a stamp that
            # only lives in memory must be persisted anyway (a crash
            # before the next regular save would otherwise lose it).
            task.touch()
            self.manager.storage.save(task)

    def view_task(self, task_id: str) -> None:
        # A DRAFT task has no chat view: every selection path (sidebar
        # click, mention, startup pick) opens the new-task editor on it
        # instead — the user must finish the draft before it enters the
        # queue. The exception is the form already editing THIS draft
        # (re-clicks in the sidebar while composing must not reset it).
        target = self.manager.tasks.get(task_id)
        if (
            target is not None
            and target.status is TaskStatus.DRAFT
            and not (self._new_task_mode and self._editing_task_id == task_id)
        ):
            self.action_open_draft(task_id)
            return
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
        # Opening a task always lands on the latest entry: re-arm the log's
        # anchor right after the wipe (while the log is still empty), so the
        # compositor pins the mounting chunks to the bottom — including the
        # last chunk's layout pass, which the drain chain may outpace.
        self.chat_log.anchor(True)
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
        # Resolver rounds (merge-conflict resolution): same shape as the
        # tester rounds — an own separator, tool calls prefixed, the
        # synthetic assignment prompt not shown.
        for round_messages in task.resolver_messages:
            if not round_messages:
                continue
            round_items = extract_display(round_messages)
            round_at = next((at for at, _, _ in round_items if at is not None), None)
            _add(round_at, "resolving-sep", "")
            for at, role, text in round_items:
                if role == "user":
                    continue
                if role == "tool-call":
                    text = f"resolver: {text}"
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

        No scrolling here: the log is anchored (ChatLog.__init__ /
        view_task's wipe), so the compositor keeps the tail on screen for
        a reader at the bottom; a reader scrolled up has released the
        anchor and is never yanked down.

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
        parts: list[_ChatPart] | None = None
        widget: Widget
        if isinstance(content, str) and len(content) > _CHAT_LIVE_MOUNT_LIMIT:
            # A fat live write (a big tool result): one Static of the whole
            # text is a single long render pass on the event loop — keyboard
            # echo lags around it. Split into parts and mount through the
            # chunked drain (live-follow mode: scroll only while the reader
            # is at the bottom), yielding to the loop between ticks — same
            # machinery as the history rebuild's _drain_chat_batch. The
            # ledger note appended below still counts the write as ONE live
            # entry: it carries the full text, so _try_delta_view's
            # reconciliation is unaffected by how many widgets the write
            # became.
            split = _split_chat_text(content, _CHAT_LIVE_PART)
            parts = [
                _ChatPart(part, expand, cont=0 < j < len(split) - 1)
                for j, part in enumerate(split)
            ]
            widget = parts[0]
        elif isinstance(content, Widget):
            widget = content
            widget.add_class("chat-block")
        else:
            widget = Static(content, expand=expand, classes="chat-block")
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
            self._chat_batch.append(widget)
            if parts is not None:
                self._chat_batch.extend(parts[1:])
            self._chat_has_content = True
        elif parts is not None:
            try:
                # Seed the chunked drain with the parts: subsequent live
                # writes keep appending to _chat_batch and ride along, so
                # block order is preserved. The drain's epoch guard drops
                # the tail on a mid-append view switch. The anchored log
                # keeps the tail on screen for a reader at the bottom and
                # leaves a scrolled-up reader alone — no follow switch.
                self._chat_batch.extend(parts)
                self._chat_batch_epoch = self._chat_epoch
                self._chat_batching = True
                self._chat_drain_at = time.monotonic()
                self._drain_chat_batch()
            except NoMatches:
                # Teardown race (see the plain-mount branch below).
                return
            self._chat_has_content = True
        else:
            try:
                self.chat_log.mount(widget)
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
        chunk: list[Widget] = []
        cost = 0
        # Size-aware chunking: split parts of a fat live write carry their
        # text length as mount_cost, so a tick's total work stays bounded
        # even when the parts are few; plain rebuild blocks cost 0 and chunk
        # by count only (as before).
        while self._chat_batch and len(chunk) < _CHAT_MOUNT_CHUNK:
            widget = self._chat_batch[0]
            widget_cost = getattr(widget, "mount_cost", 0)
            if chunk and cost + widget_cost > _CHAT_MOUNT_BYTES:
                break
            chunk.append(widget)
            cost += widget_cost
            del self._chat_batch[0]
        try:
            if chunk:
                # No follow logic here either: the anchored log keeps the
                # tail on screen as the chunks grow the virtual size (the
                # compositor pins the scroll position at each arrange —
                # a same-tick scroll_end would compute against the stale
                # max_scroll_y, the very race the anchor removes).
                self.chat_log.mount(*chunk)
            if self._chat_batch:
                self._schedule_drain(self._chat_epoch)
            else:
                self._chat_batching = False
        except NoMatches:
            # Teardown race: the drain chain (call_later / the timer
            # fallback) can fire into an already-parsed-away DOM —
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
        """Schedule the next drain chunk.

        The primary path is `call_later`: a plain Callback message in this
        pump's own queue, processed between the other pending messages and
        NOT gated on screen idle. `call_after_refresh` is deliberately not
        used: its InvokeLater callback only ever runs on a fully clean
        screen (Screen._on_idle defers it while any widget/layout flag is
        dirty), so a continuously dirty screen starves the chain — measured
        on a ~12 h session: the drain advanced solely via the 1 s watchdog
        (~2 s per chunk, a 123-block history took ~17 s to appear).

        `call_later` returns False only when the message pump is
        closed/closing; the timer fallback covers that (if the app is
        really shutting down it never fires — nothing left to render into
        anyway). The _poll_chat_drain watchdog remains the self-healing
        backstop: a dropped schedule must never leave a permanently
        truncated history with batching stuck on."""
        if self.call_later(self._drain_chat_batch, epoch):
            return
        self.set_timer(0.05, partial(self._drain_chat_batch, epoch))

    def _poll_chat_drain(self) -> None:
        """Watchdog for the chunked history drain (1 s poll): if a drain is
        pending but no chunk has progressed for a while, the reschedule
        chain died (a dropped schedule, or the pump wedged on something
        else — invisible to headless run_test() repros). Re-kick it;
        _drain_chat_batch's epoch guard drops the tail if the view moved on
        in the meantime. Every re-kick reports the stall (throttled) so a
        recurring stall pattern is diagnosable in the task's chat."""
        if not (self._chat_batching and self._chat_batch):
            # No drain in flight: reset the re-kick streak so the next
            # stall episode starts its count and its diagnostics fresh.
            self._chat_drain_rekicks = 0
            return
        stuck = time.monotonic() - self._chat_drain_at
        if stuck <= _CHAT_DRAIN_STUCK:
            return
        self._chat_drain_rekicks += 1
        try:
            self._report_drain_stall(stuck)
        except Exception:  # noqa: BLE001, S110 - diagnostics must never break the re-kick
            pass
        self._drain_chat_batch()

    def _report_drain_stall(self, stuck: float) -> None:
        """Emit a diagnostic for a stalled drain (throttled to one event per
        _CHAT_DRAIN_DIAG_GAP seconds — a permanently dead chain is advanced
        by this watchdog alone, so a per-re-kick event would spam one line
        per chunk). The point is to capture WHY the chain stalled while it
        is happening: the pending block count and the screen's dirt state
        (dirty-widget count, layout/repaint/recompose flags, is_current,
        textual's frame-batch counter) distinguish a wedged pump from the
        historic "screen forever dirty, InvokeLater starved" class and a
        widget that keeps itself dirty forever (e.g. an orphaned spinner)."""
        now = time.monotonic()
        if now - self._chat_drain_diag_at < _CHAT_DRAIN_DIAG_GAP:
            return
        self._chat_drain_diag_at = now
        try:
            screen = self.screen
            flags = (
                "+".join(
                    name
                    for name, on in (
                        ("layout", screen._layout_required),
                        ("repaint", screen._repaint_required),
                        ("recompose", screen._recompose_required),
                    )
                    if on
                )
                or "clean"
            )
            detail = (
                f"{len(screen._dirty_widgets)} dirty widgets, flags {flags}, "
                f"is_current={screen.is_current}"
            )
            batch_count = getattr(self, "_batch_count", None)
            if batch_count is not None:
                detail += f", app._batch_count={batch_count}"
        except Exception:  # noqa: BLE001 - a probe miss must not kill the poller
            detail = "screen state unavailable"
        text = (
            f"Chat drain stalled {stuck:.1f} s (watchdog re-kick "
            f"#{self._chat_drain_rekicks}, {len(self._chat_batch)} blocks "
            f"pending; screen: {detail})."
        )
        self.log.warning(text)
        task = self.viewed
        if task is not None:
            self._show_event(task, text)

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
            self._write_chat(
                f"[dim]{_mention_markup(_full_escape(text))}[/]", note=note
            )
        elif role == "tester":
            # tester output is raw model text: escape it before the markup
            # parser sees it (a stray bracket would crash rendering), then
            # wrap task-id mentions into click links.
            self._write_chat(
                f"[bold magenta]tester ›[/] {_mention_markup(_full_escape(text))}",
                note=note,
            )
        elif role == "testing-sep":
            label = f"testing: {text}" if text else "testing"
            self._write_chat(f"[bold magenta]──── {label} ────[/]", note=note)
        elif role == "resolving-sep":
            label = f"conflict resolution: {text}" if text else "conflict resolution"
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
            self._write_chat(
                f"[bold red]error ›[/] {_mention_markup(_full_escape(text))}"
            )
        if block is not None:
            if isinstance(block, ReasoningBlock | AssistantBlock):
                self._write_chat(block, note=(role, text, block))
            elif note is not None:
                self._write_chat(block, note=(role, text, None))
            else:
                self._write_chat(block)
        return block

    def _write_clarification(self, clarification: Clarification) -> None:
        if clarification is not self._rendered_clarification:
            # A fresh question/package replaces the previous one: stale
            # click picks from the old package must not leak into the new
            # draft (see action_pick_option).
            self._clarification_picks = {}
            self._clarification_multi = {}
        # A package holds one or more questions (the agent is prompted to
        # batch ALL its open questions into one call). Each question renders
        # separately with its own clickable options; a package question is
        # numbered and its options carry the question index in the click id
        # (app.pick_option(<q>, <i>)) so a click answers exactly that
        # question — a single-question package keeps the legacy flat id
        # (app.pick_option(<i>)) and the immediate-submit behaviour, UNLESS
        # the question is multiple-choice: those always get the question-
        # indexed id and the accumulating (toggle) click behaviour, never
        # the immediate submit.
        questions = clarification.questions
        lines: list[str] = []
        packaged = len(questions) > 1
        for qi, item in enumerate(questions):
            header = f"{qi + 1}. {item.question}" if packaged else item.question
            lines.append(
                f"[bold blue]drcode ❓[/] {_mention_markup(_full_escape(header))}"
            )
            # Plain options first (a legacy agent-marker inside an old
            # option's text still gets its ★ via _split_agent_marker), then
            # the agent's own proposals with a ★ — indices CONTINUE after
            # `options`, and action_pick_option validates against the
            # combined list.
            has_agent_option = False
            question_indexed = packaged or item.multiple

            def _option_line(
                index: int,
                display: str,
                *,
                star: bool,
                qi: int = qi,
                question_indexed: bool = question_indexed,
            ) -> str:
                click = (
                    f"app.pick_option({qi},{index})"
                    if question_indexed
                    else f"app.pick_option({index})"
                )
                return (
                    f"  ▸ {'[bold yellow]★[/] ' if star else ''}"
                    f"[@click={click}]"
                    f"[underline cyan]{_full_escape(display)}[/][/]"
                )

            for i, option in enumerate(item.options):
                is_agent, display = _split_agent_marker(option)
                has_agent_option = has_agent_option or is_agent
                lines.append(_option_line(i, display, star=is_agent))
            for j, proposal in enumerate(item.agent_proposals, start=len(item.options)):
                has_agent_option = True
                lines.append(_option_line(j, proposal, star=True))
            if has_agent_option:
                lines.append(
                    "[dim]★ — option proposed by the agent on its own "
                    "initiative, not part of your request.[/]"
                )
            if item.multiple:
                # The click log stays append-only (no ticks are drawn into
                # the rendered history) — the accumulated selection is
                # visible only in the input draft line, so point at it.
                lines.append(
                    "[dim]Multiple choices allowed — click every option you "
                    "need (click again to unselect), then press Enter; "
                    "nothing selected means \u201cnone of them\u201d.[/]"
                )
        if len(questions) > 1:
            lines.append(
                "[dim]Click an option per question or type answers — one "
                "line per question — then press Enter to send them all at "
                "once.[/]"
            )
        elif not any(q.multiple for q in questions):
            lines.append("[dim]Click an option or answer in the input below.[/]")
        self._write_chat("\n".join(lines))
