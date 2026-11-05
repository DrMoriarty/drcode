"""Selective context eviction driven by the agent's own marks.

Research (and feature) rounds read dozens of files / run dozens of
searches whose results are dead weight once the agent has extracted what
it needed. The agent marks such results with the `mark_unneeded` tool
(recording tool-call ids on `Task.marked_unneeded`); this module holds
the LLM-free mechanics of the DEFERRED batch eviction:

- the stub a marked tool-return's content collapses into,
- the ≈token accounting of the pending marks (chars/4, same convention
  as compaction/context_layout — every figure built from it is "≈"),
- the collapse itself: replace the CONTENT of each marked `tool-return`
  part with the stub, never touching the neighboring `tool-call` part —
  the call/return pair (matched by tool_call_id) must survive intact or
  the transcript stops being API-valid.

Eviction never happens at mark time: the pipeline fires it in one batch
(one prompt-cache invalidation per batch, not per mark) from the gates in
`core/pipeline_compaction.py` — between turns, never mid-stream.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .transcript import estimate_tokens

# The fraction of the serving role's context window at which pending marks
# are worth one batch eviction (one prompt-cache invalidation): below it
# the marks ride along cheaply.
EVICTION_WINDOW_FRACTION = 3

# The args summary in the stub is capped: the stub must stay a short,
# self-describing line whatever the original call looked like.
ARGS_SUMMARY_CAP = 120


def evicted_stub(tool_name: str, args: Any) -> str:
    """The stub text replacing an evicted tool result's content."""
    if isinstance(args, str):
        args_text = args
    else:
        try:
            args_text = json.dumps(args or {}, ensure_ascii=False)
        except (TypeError, ValueError):
            args_text = str(args)
    if len(args_text) > ARGS_SUMMARY_CAP:
        args_text = args_text[:ARGS_SUMMARY_CAP] + "…"
    return (
        f"[tool result evicted: {tool_name}({args_text}) — re-run the tool if needed]"
    )


def is_evicted_stub(content: Any) -> bool:
    """True when a tool-return content is already an eviction stub."""
    return isinstance(content, str) and content.startswith("[tool result evicted:")


def _return_text(content: Any) -> str:
    """The chars/4 input for a tool-return content (string or JSON-ish)."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


@dataclass
class EvictionResult:
    """Outcome of one `evict_marked` pass over a serialized transcript."""

    new_messages: list[dict[str, Any]]
    # Tool-call ids whose return content was actually replaced by a stub.
    evicted: list[str] = field(default_factory=list)
    # Marked ids with NO tool-return part in the transcript at all — stale
    # marks (the history was rewritten by a full compaction around them);
    # the caller prunes them alongside the evicted ones.
    missing: list[str] = field(default_factory=list)


def marked_tokens(messages: list[dict[str, Any]], marked: Iterable[str]) -> int:
    """≈ tokens (chars/4) held by the NOT-yet-evicted marked tool returns.

    The batch-eviction gate's threshold input: the sum the pipeline
    compares against a third of the role's context window. Stubs count as
    zero (their content is already gone)."""
    marked_set = set(marked)
    if not marked_set:
        return 0
    total = 0
    for msg in messages:
        for part in msg.get("parts", []):
            if (
                isinstance(part, dict)
                and part.get("part_kind") == "tool-return"
                and part.get("tool_call_id") in marked_set
                and not is_evicted_stub(part.get("content"))
            ):
                total += estimate_tokens(_return_text(part.get("content")))
    return total


def evict_marked(
    messages: list[dict[str, Any]], marked: Iterable[str]
) -> EvictionResult:
    """Collapse every marked tool-return's content into the stub.

    Walks the serialized transcript once; each `tool-return` part whose
    `tool_call_id` is in `marked` gets its content replaced by
    `evicted_stub` (built from the tool name and the args of the sibling
    `tool-call` part — the pair is re-associated by id, not assumed
    adjacent). Messages without marked returns are reused as-is; only the
    mutated ones are copied, so the input list is never modified.

    The tool-call parts are never touched: the call/return pair required
    by the model APIs survives, and the stub itself tells the model how to
    get the content back (re-run the tool).
    """
    marked_set = set(marked)
    result = EvictionResult(new_messages=list(messages))
    if not marked_set:
        return result
    # tool_call_id -> (tool_name, args) for the stub text.
    calls: dict[str, tuple[str, Any]] = {}
    for msg in messages:
        for part in msg.get("parts", []):
            if (
                isinstance(part, dict)
                and part.get("part_kind") == "tool-call"
                and part.get("tool_call_id") in marked_set
            ):
                calls[part["tool_call_id"]] = (
                    part.get("tool_name", "?"),
                    part.get("args"),
                )
    seen: set[str] = set()
    new_messages: list[dict[str, Any]] = []
    for msg in messages:
        parts = msg.get("parts", [])
        touched = False
        new_parts: list[Any] = []
        for part in parts:
            if (
                isinstance(part, dict)
                and part.get("part_kind") == "tool-return"
                and part.get("tool_call_id") in marked_set
                and not is_evicted_stub(part.get("content"))
            ):
                call_id = part["tool_call_id"]
                seen.add(call_id)
                tool_name, args = calls.get(call_id, (part.get("tool_name", "?"), {}))
                new_parts.append({**part, "content": evicted_stub(tool_name, args)})
                touched = True
                result.evicted.append(call_id)
            else:
                new_parts.append(part)
        new_messages.append({**msg, "parts": new_parts} if touched else msg)
    result.new_messages = new_messages
    result.missing = sorted(marked_set - seen)
    return result
