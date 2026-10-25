"""Shared live-stream renderer of a model-request node.

Used by both run loops (`pipeline_agent_run.run_agent` and
`pipeline_tester.run_tester`): it classifies pydantic-ai stream events
into thinking/text deltas and grows the reasoning/assistant chat blocks
in place, surviving mid-node `view_task()` re-renders (a wiped view is
detected via the block's `stream_epoch` and the already-streamed text is
carried over).

The two loops differ only in three switches:

- `record_tps` — the agent feeds the header's live tok/s estimate, the
  tester does not;
- `follow` — the agent keeps the chat tail pinned while the user is at
  the bottom (`_chat_at_bottom`/`_stream_follow`);
- `track_dirty` — the agent notes "content was produced off-screen" in
  `view_dirty` (the end-of-turn re-render then goes through the delta
  path); the tester re-renders unconditionally at the round's end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)

from .blocks import AssistantBlock, ReasoningBlock

if TYPE_CHECKING:
    from .state import DrCodeState


def stream_delta(event: object) -> ThinkingPartDelta | TextPartDelta | None:
    """The thinking/text delta carried by a stream event, None for any
    other event kind (and for content-less start parts)."""
    if isinstance(event, PartStartEvent):
        # The first chunk of a new text/thinking part arrives as
        # PartStartEvent carrying the content itself (only subsequent
        # chunks come as deltas); dropping it loses the beginning of the
        # message from the live view.
        part = event.part
        if isinstance(part, ThinkingPart):
            return ThinkingPartDelta(content_delta=part.content)
        if isinstance(part, TextPart):
            return TextPartDelta(content_delta=part.content)
        return None
    if isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, ThinkingPartDelta | TextPartDelta):
            return event.delta
        return None
    return None


class NodeStreamView:
    """Per-node accumulator of the streamed reasoning/assistant blocks."""

    def __init__(
        self,
        layer: DrCodeState,
        task_id: str,
        *,
        record_tps: bool,
        follow: bool,
        track_dirty: bool,
    ) -> None:
        self._layer = layer
        self.task_id = task_id
        self._record_tps = record_tps
        self._follow = follow
        self._track_dirty = track_dirty
        self.view_dirty = False
        self._reasoning: ReasoningBlock | None = None
        self._assistant: AssistantBlock | None = None

    def begin_node(self) -> None:
        """Fresh blocks for a new model-request node."""
        self._reasoning = None
        self._assistant = None

    def feed(self, event: object) -> None:
        """Consume one stream event: classify it and grow the blocks."""
        layer = self._layer
        layer._touch_activity(self.task_id)
        delta = stream_delta(event)
        if delta is None:
            return
        content = delta.content_delta
        if not content:
            return
        if self._record_tps:
            layer._record_stream_tokens(self.task_id, content)
        if isinstance(delta, ThinkingPartDelta):
            self._feed_thinking(delta, content)
        else:
            self._feed_text(delta, content)

    def finish_node(self) -> None:
        """The node's stream is closed: stop both blocks' spinners."""
        if self._reasoning is not None:
            self._reasoning.finish()
        if self._assistant is not None:
            self._assistant.finish()

    # -- internals ---------------------------------------------------------

    def _feed_thinking(self, delta: ThinkingPartDelta, content: str) -> None:
        layer = self._layer
        block = self._reasoning
        if block is None or block.stream_epoch != layer._chat_epoch:
            old, block = block, ReasoningBlock(block.at if block else None)
            block.stream_epoch = layer._chat_epoch
            if old is not None:
                # A mid-node view_task() re-render orphaned the old block:
                # carry the already-streamed text over, or the rest of this
                # response would never become visible.
                block.set_content(old._content)
            self._reasoning = block
            block.begin()
            if layer.viewed_id == self.task_id:
                # Noted: the block mirrors the timeline "reasoning" entry
                # the persisted transcript gains at the node boundary —
                # the end-of-turn view_task then appends instead of
                # wiping.
                layer._write_chat(block, note=("reasoning", "", block))
            elif self._track_dirty:
                self.view_dirty = True
            block.append(content)
        else:
            # Streaming grows the block without a new mount: keep
            # following only if the tail was visible before the delta
            # arrived. Deltas are appended even when the task is
            # off-screen: they must survive a later re-render.
            block.thinking(True)
            self._append(block, content)

    def _feed_text(self, delta: TextPartDelta, content: str) -> None:
        layer = self._layer
        # Stream the answer into the chat log as it is produced: the block
        # grows in place, so the text shows up between the reasoning and
        # tool-call blocks (no separate preview area under the log).
        # Thinking is over for now — stop the reasoning block's spinner.
        if self._reasoning is not None:
            self._reasoning.thinking(False)
        block = self._assistant
        if block is None or block.stream_epoch != layer._chat_epoch:
            old, block = block, AssistantBlock()
            block.stream_epoch = layer._chat_epoch
            if old is not None:
                # See the reasoning branch: re-mount after a mid-node
                # view_task() wipe, keeping the streamed so far.
                block.set_content(old._content)
            self._assistant = block
            if layer.viewed_id == self.task_id:
                # Noted: mirrors the timeline "assistant" entry (see the
                # reasoning branch above).
                layer._write_chat(block, note=("assistant", "", block))
            elif self._track_dirty:
                self.view_dirty = True
            block.append(content)
        else:
            self._append(block, content)

    def _append(self, block: ReasoningBlock | AssistantBlock, content: str) -> None:
        """Append to a mounted block, following the chat tail when the
        open view is pinned to the bottom (agent runs only)."""
        layer = self._layer
        viewed = layer.viewed_id == self.task_id
        if self._track_dirty and not viewed:
            self.view_dirty = True
        follow = layer._chat_at_bottom() if viewed and self._follow else False
        block.append(content)
        if follow:
            layer._stream_follow()
