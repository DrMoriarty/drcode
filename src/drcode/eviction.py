"""Selective context eviction driven by the agent's own marks.

Research (and feature) rounds read dozens of files / run dozens of
searches whose results are dead weight once the agent has extracted what
it needed. The agent marks such results with the `mark_unneeded` tool
(recording tool-call ids on `Task.marked_unneeded`); this module holds
the LLM-free mechanics of the DEFERRED batch eviction:

- the stub a marked tool-return's content collapses into — a one-liner
  plus an EXTRACTIVE digest of the original (verbatim copies of the most
  informative lines, picked by cheap per-tool-kind heuristics — the
  Paritok insight that a function signature is worth more than a debug
  line; no LLM, no disk reads, built in the same batch pass),
- the ≈token accounting of the pending marks (chars/4, same convention
  as compaction/context_layout — every figure built from it is "≈"),
- the collapse itself: replace the CONTENT of each marked `tool-return`
  part with the stub, never touching the neighboring `tool-call` part —
  the call/return pair (matched by tool_call_id) must survive intact or
  the transcript stops being API-valid.

Eviction never happens at mark time: the pipeline fires it in one batch
(one prompt-cache invalidation per batch, not per mark) from the gates in
`core/pipeline_compaction.py` — between turns, never mid-stream.

On top of the agent's own marks runs the AUTOMATIC layer (tool result
clearing, the Claude Code recipe — "the safest, lightest touch form of
compaction"): tool results older than `AUTO_EVICTION_MIN_AGE_TURNS` model
exchanges — except the freshest `AUTO_EVICTION_KEEP_RECENT` ones — are
collapsed into the same stubs by the same batch pass, no mark required.
`auto_evictable_ids` is the pure criterion; the pipeline union-folds it
into the same eviction batch the marks go through.
"""

from __future__ import annotations

import json
import re
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

# Automatic tool result clearing (the age criterion, no marks involved):
# a tool result is auto-evictable once this many model exchanges
# (assistant responses — the "turns" of an agentic run) have happened
# AFTER the exchange that produced it. Anthropic's measurements of the
# recipe (+29% agentic-search quality, −84% tokens over a 100-turn eval)
# come from exactly this regime: fresh results the agent still works from
# stay verbatim, stale ones degrade to re-runnable stubs. The auto layer
# itself is a project setting (auto_context_unloading, default OFF — the
# gates in core/pipeline_compaction check it before anything auto-evicts);
# this module knob, like EVICTION_WINDOW_FRACTION, only shapes HOW MUCH
# age/freshness the opted-in layer grants.
AUTO_EVICTION_MIN_AGE_TURNS = 10

# The freshness anchor of the automatic clearing: the last K tool results
# of the transcript are NEVER auto-evicted, however old — the newest
# conclusions (test output, the last file read) are what the agent is
# still actively reasoning over, and the age estimate cannot see that.
AUTO_EVICTION_KEEP_RECENT = 5

# Mid-run pause gate of the automatic clearing: between turns the pass
# fires on any auto-evictable result at all (see _maybe_evict_marked),
# but pausing a LIVE run costs a stop/resume cycle — mid-run it fires
# only when the auto-evictable results' ≈size (chars/4) reached this
# fraction of the serving role's context window. Deliberately tighter
# than the marks gate's EVICTION_WINDOW_FRACTION (3): auto-evictable
# volume renews itself every turn once past the age horizon, and a
# pause per turn would be churn.
AUTO_EVICTION_PAUSE_FRACTION = 6


# -- extractive digest ----------------------------------------------------------------
#
# The Paritok insight, applied LLM-free: a stub that keeps verbatim copies of
# the most informative lines of the ORIGINAL result (its "function signature
# is worth more than a debug line") lets the agent remember what the result
# held without re-running the tool. The extraction is cheap per-tool-kind
# heuristics over the content being evicted — no disk reads, no model calls —
# and runs inside the same batch pass as the collapse itself, so nothing is
# added between turns.

# Total chars/4-style budget for the digest block (header + lines): the stub
# must stay short whatever the original weighed.
DIGEST_BUDGET_CHARS = 600

# One digest line is capped, so a single huge line (minified JS, a wall of
# one-character output) cannot eat the whole budget.
DIGEST_LINE_CAP = 160

# How many leading lines count as the result's "head" (the header + the first
# significant lines).
DIGEST_HEAD_LINES = 4

# How many trailing lines count as the "tail" (the outcome/summary of a
# command's output lives at the end).
DIGEST_TAIL_LINES = 6

# Hard cap on digest lines (budget permitting).
DIGEST_MAX_LINES = 12

# The file-inspecting tool kinds: their results are read like source/paths —
# head + signature lines are what matters. Every other tool (run_bash and
# the rest) gets the output heuristic: error/warning lines first, then the
# tail, then the head.
DIGEST_FILE_TOOLS = frozenset({"read_file", "grep_search", "glob_files"})

_ERROR_LINE_RE = re.compile(
    r"\b(errors?|warnings?|failed|failure|fail(?:ed|ure)?s?|exceptions?|"
    r"traceback|fatal|panic|refused|denied|abort(?:ed|ion)?)\b",
    re.IGNORECASE,
)

_SIGNATURE_RE = re.compile(
    r"^\s*(?:async\s+def\s|def\s|class\s|import\s|from\s+\S+\s+import|"
    r"export\s+(?:default\s+)?(?:function|class|const|let)|"
    r"function\s|fn\s|pub\s+fn\s|interface\s|type\s)"
)


def _digest_picked_indices(lines: list[str], tool_name: str) -> list[int]:
    """Which line indices of the (non-empty) result lines the digest keeps.

    Priority for the file-inspecting kinds: the head, then signature lines
    (def/class/import — identifiers, the expensive stuff), then error lines.
    For run_bash and every other tool: error/warning lines, then the tail
    (the outcome), then the head. Duplicate indices collapse; the output is
    returned in the original order so the digest stays readable as an
    excerpt of the result.
    """
    errors = [i for i, ln in enumerate(lines) if _ERROR_LINE_RE.search(ln)]
    heads = list(range(min(DIGEST_HEAD_LINES, len(lines))))
    if tool_name in DIGEST_FILE_TOOLS:
        sigs = [i for i, ln in enumerate(lines) if _SIGNATURE_RE.match(ln)]
        priority = heads + sigs + errors
    else:
        tails = (
            list(range(max(heads[-1] + 1, len(lines) - DIGEST_TAIL_LINES), len(lines)))
            if len(lines) > DIGEST_HEAD_LINES
            else []
        )
        priority = errors + tails + heads
    seen: set[int] = set()
    picked: list[int] = []
    for i in priority:
        if i in seen:
            continue
        seen.add(i)
        picked.append(i)
        if len(picked) >= DIGEST_MAX_LINES:
            break
    return sorted(picked)


def extractive_digest(tool_name: str, content: Any) -> str:
    """The extractive digest block for an evicted tool-return content.

    Copies of the most informative lines of the original (per-tool-kind
    heuristics, see _digest_picked_indices), each capped at
    DIGEST_LINE_CAP, all together bounded by DIGEST_BUDGET_CHARS, every
    line prefixed `| ` so the excerpt cannot be mistaken for the full
    result. Built ONLY from the content being evicted — nothing is read
    from disk. Returns "" when the heuristics yield nothing (blank
    content, nothing fits the budget): the caller falls back to the
    historical one-line stub.
    """
    text = _return_text(content)
    if not text.strip():
        return ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    picked = _digest_picked_indices(lines, tool_name)
    if not picked:
        return ""
    header = f"[extractive digest: {len(picked)} of {len(lines)} lines]"
    out = [header]
    used = len(header)
    for i in picked:
        line = lines[i]
        if len(line) > DIGEST_LINE_CAP:
            line = line[:DIGEST_LINE_CAP] + "…"
        piece = f"| {line}"
        if used + 1 + len(piece) > DIGEST_BUDGET_CHARS:
            break
        out.append(piece)
        used += 1 + len(piece)
    if len(out) == 1:
        return ""
    return "\n".join(out)


def evicted_stub(tool_name: str, args: Any, content: Any = None) -> str:
    """The stub text replacing an evicted tool result's content.

    The first line is the historical one-liner — its
    `[tool result evicted:` prefix is what `is_evicted_stub` (and every
    consumer of it: `marked_tokens`, the auto-layer's re-selection, the
    pipeline gates) recognizes, so it must never change. When the
    ORIGINAL content is passed (as `evict_marked` does at collapse time)
    an extractive digest of the most informative lines is appended after
    it; without content, or when the digest heuristics yield nothing
    (blank result, budget exhausted), the stub is exactly the historical
    one-liner.
    """
    if isinstance(args, str):
        args_text = args
    else:
        try:
            args_text = json.dumps(args or {}, ensure_ascii=False)
        except (TypeError, ValueError):
            args_text = str(args)
    if len(args_text) > ARGS_SUMMARY_CAP:
        args_text = args_text[:ARGS_SUMMARY_CAP] + "…"
    base = (
        f"[tool result evicted: {tool_name}({args_text}) — re-run the tool if needed]"
    )
    if content is None:
        return base
    digest = extractive_digest(tool_name, content)
    if not digest:
        return base
    return f"{base}\n{digest}"


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


def auto_evictable_ids(
    messages: list[dict[str, Any]],
    *,
    min_age_turns: int = AUTO_EVICTION_MIN_AGE_TURNS,
    keep_recent: int = AUTO_EVICTION_KEEP_RECENT,
) -> set[str]:
    """The tool-call ids the AUTOMATIC age criterion would clear now.

    A turn is one model exchange (a serialized `kind: "response"`
    message — the unit Anthropic's 100-turn agentic evals count). A
    tool return's age is the number of exchanges that happened AFTER
    the exchange carrying its `tool-call` sibling (the return landing
    directly behind its call is age 0 — brand new). Eligible: live
    (not yet a stub) returns with age ≥ `min_age_turns`, minus the
    freshness anchor — the last `keep_recent` live returns of the
    transcript in chronological order, however old.

    Never eligible, on top of that: `mark_unneeded`'s own returns
    (one-liners, and the marks machinery assumes them present — the
    same exclusion `_eviction_candidates` makes), and returns with no
    `tool-call` sibling in the transcript at all (a broken pair is
    sanitized away elsewhere; conservatively kept here). The
    tool-call parts are never touched by the collapse — the criterion
    selects whole call/return pairs, matching evict_marked's
    pair-intact invariant by construction.

    One scan, no mutation — the pipeline calls it at every gate
    observation to union the ids into the batch the marks go through.
    """
    exchanges = 0
    call_exchange: dict[str, int] = {}
    live: list[str] = []  # evictable-candidate call ids, chronological
    seen: set[str] = set()
    for msg in messages:
        kind = msg.get("kind") or msg.get("role")
        if kind == "response":
            exchanges += 1
            for part in msg.get("parts", []):
                if (
                    isinstance(part, dict)
                    and part.get("part_kind") == "tool-call"
                    and part.get("tool_call_id")
                ):
                    call_exchange[part["tool_call_id"]] = exchanges
            continue
        for part in msg.get("parts", []):
            if not (isinstance(part, dict) and part.get("part_kind") == "tool-return"):
                continue
            call_id = part.get("tool_call_id")
            if (
                not call_id
                or call_id in seen
                or part.get("tool_name") == "mark_unneeded"
                or is_evicted_stub(part.get("content"))
            ):
                continue
            if call_id not in call_exchange:
                continue  # no sibling call: conservatively kept
            seen.add(call_id)
            live.append(call_id)
    if not live:
        return set()
    # The age is measured against the FINAL exchange count: a return seen
    # right behind its call sits at age 0 NOW, but the exchanges still to
    # come age it — the criterion judges the transcript as it stands.
    ages = {call_id: exchanges - call_exchange[call_id] for call_id in live}
    anchor = set(live[-keep_recent:]) if keep_recent > 0 else set()
    return {call_id for call_id, age in ages.items() if age >= min_age_turns} - anchor


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
    by the model APIs survives. The stub tells the model how to get the
    content back (re-run the tool) AND carries an extractive digest —
    verbatim copies of the most informative lines of the original,
    selected by the per-tool-kind heuristics of `extractive_digest` at
    collapse time (no disk reads, no model calls, bounded by
    DIGEST_BUDGET_CHARS).
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
                new_parts.append(
                    {
                        **part,
                        "content": evicted_stub(tool_name, args, part.get("content")),
                    }
                )
                touched = True
                result.evicted.append(call_id)
            else:
                new_parts.append(part)
        new_messages.append({**msg, "parts": new_parts} if touched else msg)
    result.new_messages = new_messages
    result.missing = sorted(marked_set - seen)
    return result
