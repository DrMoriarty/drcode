"""Context compaction: summarise a task's history into an anchored summary.

Turns the serialized transcript into readable text, feeds it to a tool-less
summarizer (system prompt + output template from the bundled `compaction.txt` /
`compaction.md` templates) and rebuilds the transcript as the original user
prompts verbatim plus one synthetic request carrying the new summary. Kept
apart from the coding/tester agents: the summarizer only reads the rendered
history and returns a summary string.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, UserPromptPart

from . import agent_model
from .agent_model import EXTRA_HEADERS
from .config import Config
from .prompts import load_compaction_prompts
from .transcript import (
    COMPACTION_SUMMARY_PREFIX,
    estimate_tokens,
    sanitized_messages,
    serialize_messages,
)

# Tool results carry most of the working context (file reads, command
# output), but a single pathological result must not blow up the
# summarizer's own request — everything above the cap is cut.
_COMPACTION_TOOL_RESULT_CAP = 20_000

# Headroom (tokens) the mechanical compaction floor reserves in the serving
# model's window for everything the transcript does not carry: the system
# prompt, the tool schemas and the next turn's output. Capped at a quarter
# of the window so small local models keep a usable budget.
MECHANICAL_HEADROOM_TOKENS = 32_000

# Tokens reserved on top of the measured prompt pieces (system prompt,
# template, anchors) for the <conversation>/<previous-summary> scaffolding
# and rounding slack of the chars/4 heuristic.
_PROMPT_SCAFFOLD_ALLOWANCE = 512

# Tokens the summarizer may spend on its own output: it must fit into the
# window alongside the prompt (input + output share the context), and the
# reduce call's output is the final summary.
_OUTPUT_ALLOWANCE = 4_096

# Safety margin applied to the computed slice budget: the chars/4 estimate
# is a heuristic, and gateways count system messages and chat framing too.
_SLICE_SAFETY_MARGIN = 0.9

# Floor for the slice budget when the window is tiny (tests, exotic
# gateways): below this the chunker would degenerate into per-block slices.
_MIN_SLICE_BUDGET = 2_000

# Depth cap of the hierarchical reduce: each level merges already-summarized
# text, so it converges long before this; past the cap the remaining
# partials are mechanically trimmed to fit (the guaranteed floor).
_MAX_REDUCE_DEPTH = 3

# The chars/4 token-size heuristic — same convention as context_layout
# (duplicated rather than imported: context_layout → agent → compaction is
# a circular import).
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """≈ tokens of `text` by the chars/4 heuristic (0 for empty text)."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def build_compaction_agent(config: Config) -> Agent[None, str]:
    """Tool-less summarizer agent used to compact a task's history.

    System prompt and output structure come from the bundled compaction.txt
    / compaction.md templates; the conversation itself is passed as a
    plain-text user prompt (see build_compaction_prompt). No tools, no deps:
    the agent only reads the rendered transcript and returns the summary.
    """
    system_prompt, _ = load_compaction_prompts()
    return Agent(
        agent_model.build_model(config),
        instructions=system_prompt,
        model_settings={
            "extra_headers": EXTRA_HEADERS,
            "max_tokens": config.max_tokens,
        },
    )


def _is_request(msg: dict[str, Any]) -> bool:
    return msg.get("kind") == "request" or msg.get("role") == "request"


def _user_prompt_parts(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """String user-prompt parts of a serialized request, in order."""
    return [
        part
        for part in msg.get("parts", [])
        if isinstance(part, dict)
        and part.get("part_kind") == "user-prompt"
        and isinstance(part.get("content"), str)
        and part["content"]
    ]


def _is_summary_part(part: dict[str, Any]) -> bool:
    content = part.get("content")
    return isinstance(content, str) and content.startswith(COMPACTION_SUMMARY_PREFIX)


def _user_prompt_content(msg: dict[str, Any]) -> str | None:
    """First string user-prompt content of a serialized request, if any."""
    parts = _user_prompt_parts(msg)
    return parts[0]["content"] if parts else None


def is_summary_message(msg: dict[str, Any]) -> bool:
    """True for a request carrying the compaction summary.

    pydantic-ai merges the synthetic summary request with the adjacent
    original-prompt request on resume (_merge_consecutive_messages), so a
    merged request holds the original prompt AND the summary as adjacent
    user-prompt parts — any summary part makes it a summary carrier.
    """
    if not _is_request(msg):
        return False
    return any(_is_summary_part(p) for p in _user_prompt_parts(msg))


def _without_summary_parts(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Copy of a serialized request with its summary user-prompt parts
    dropped; None when nothing but summary parts remains."""
    parts = [
        p
        for p in msg.get("parts", [])
        if not (
            isinstance(p, dict)
            and p.get("part_kind") == "user-prompt"
            and _is_summary_part(p)
        )
    ]
    if len(parts) == len(msg.get("parts", [])):
        return msg
    return {**msg, "parts": parts} if parts else None


def split_original_prompts(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the transcript into (original user prompts, rest).

    The leading consecutive user requests are the task's original
    assignment(s) — they are kept verbatim across compactions (the summary
    message, if a previous compaction left one, ends this prefix: it lives
    in `rest` so the summarizer sees it as <previous-summary>).

    Part-aware: a leading request that pydantic-ai merged with a summary
    (adjacent user-prompt parts after a resume) contributes only its
    non-summary parts to `preserved` — stale summaries must not survive
    into the next compaction, or the transcript grows on every cycle.
    """
    i = 0
    preserved: list[dict[str, Any]] = []
    while i < len(messages):
        msg = messages[i]
        parts = _user_prompt_parts(msg) if _is_request(msg) else []
        if not parts:
            break
        if any(_is_summary_part(p) for p in parts):
            trimmed = _without_summary_parts(msg)
            if trimmed is not None:
                preserved.append(trimmed)
            break
        preserved.append(msg)
        i += 1
    return preserved, messages[i:]


def extract_previous_summary(messages: list[dict[str, Any]]) -> str | None:
    """Body of the latest compaction summary in the transcript (None if the
    history was never compacted)."""
    found: str | None = None
    for msg in messages:
        if not _is_request(msg):
            continue
        for part in _user_prompt_parts(msg):
            if _is_summary_part(part):
                found = part["content"][len(COMPACTION_SUMMARY_PREFIX) :].strip()
    return found


def render_history_blocks(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Render the serialized transcript as a list of (kind, text) blocks for
    the summarizer — the chunkable unit structure behind
    render_history_for_summary.

    Kinds: "user", "assistant", "tool-call", "tool-result". User prompts,
    assistant text, tool calls and tool results are kept (tool results are
    the bulk of the working context — file contents and command output the
    summary must capture); thinking parts are dropped; compaction-summary
    user-prompt parts are skipped (they are passed separately as
    <previous-summary>) — including the merged ones pydantic-ai folds into
    the original-prompt request after a resume.
    """
    blocks: list[tuple[str, str]] = []
    for msg in messages:
        is_request = _is_request(msg)
        for part in msg.get("parts", []):
            if not isinstance(part, dict):
                continue
            kind = part.get("part_kind")
            if kind == "user-prompt":
                content = part.get("content")
                if (
                    isinstance(content, str)
                    and content
                    and not content.startswith(COMPACTION_SUMMARY_PREFIX)
                ):
                    blocks.append(("user", f"[user]\n{content}"))
            elif kind == "tool-return":
                # Tool returns live in request messages (ModelRequest with a
                # ToolReturnPart) — they must be rendered BEFORE the generic
                # "skip everything else in a request" branch below, or the
                # summarizer loses the bulk of the working context.
                content = part.get("content")
                if isinstance(content, str):
                    text = content
                else:
                    try:
                        text = json.dumps(content, ensure_ascii=False, indent=2)
                    except (TypeError, ValueError):
                        text = str(content)
                if len(text) > _COMPACTION_TOOL_RESULT_CAP:
                    text = text[:_COMPACTION_TOOL_RESULT_CAP] + "\n…[truncated]"
                blocks.append(
                    (
                        "tool-result",
                        f"[tool result for {part.get('tool_name', '?')}]\n{text}",
                    )
                )
            elif is_request:
                continue
            elif kind == "text":
                content = part.get("content", "")
                if content:
                    blocks.append(("assistant", f"[assistant]\n{content}"))
            elif kind == "tool-call":
                args = part.get("args")
                args_str = (
                    args
                    if isinstance(args, str)
                    else json.dumps(args or {}, ensure_ascii=False)
                )
                blocks.append(
                    (
                        "tool-call",
                        f"[tool call] {part.get('tool_name', '?')}({args_str})",
                    )
                )
    return blocks


def render_history_for_summary(messages: list[dict[str, Any]]) -> str:
    """Render the serialized transcript as readable text for the summarizer
    (the blocks of render_history_blocks joined by blank lines)."""
    return "\n\n".join(text for _, text in render_history_blocks(messages))


def build_compaction_prompt(
    previous_summary: str | None,
    transcript: str,
    template: str,
    checklist: str | None = None,
) -> str:
    """User prompt for the summarizer: the conversation, the anchored
    previous summary (if any), the task's current todo checklist (when it
    has one) and the required output structure."""
    parts = [
        "Below is the conversation history of a coding session.",
        "",
        "<conversation>",
        transcript,
        "</conversation>",
        "",
        "<previous-summary>",
        previous_summary or "(none)",
        "</previous-summary>",
    ]
    if checklist:
        # The checklist lives in `task.todos`, outside the transcript — the
        # summarizer cannot see it in the conversation text. Passed (and
        # rendered) explicitly so the "## Task checklist" section of the
        # output template carries the true current state into the summary
        # (belt-and-suspenders alongside the structured task field).
        parts.extend(
            [
                "",
                "<task-checklist>",
                checklist,
                "</task-checklist>",
            ]
        )
    parts.extend(
        [
            "",
            "Produce the updated summary following exactly this structure:",
            "",
            template,
        ]
    )
    return "\n".join(parts)


def build_reduce_prompt(
    partials: list[str],
    template: str,
    checklist: str | None = None,
) -> str:
    """User prompt for the reduce step: merge the sequential partial
    summaries (one per map slice) into the single structured summary."""
    rendered = "\n\n".join(
        f"<partial index={i}>\n{p.strip()}\n</partial>"
        for i, p in enumerate(partials, 1)
        if p.strip()
    )
    parts = [
        (
            "Below are sequential partial summaries of one coding session, "
            "produced while reading its conversation in order."
        ),
        "",
        "<partial-summaries>",
        rendered,
        "</partial-summaries>",
    ]
    if checklist:
        parts.extend(
            [
                "",
                "<task-checklist>",
                checklist,
                "</task-checklist>",
            ]
        )
    parts.extend(
        [
            "",
            (
                "Merge them into a single summary of the whole session: keep "
                "still-true details, drop stale ones, resolve contradictions in "
                "favor of the latest partial. Do not lose file paths, identifiers "
                "or decisions mentioned in any partial."
            ),
            "",
            "Produce the merged summary following exactly this structure:",
            "",
            template,
        ]
    )
    return "\n".join(parts)


def _output_allowance(config: Config) -> int:
    """Room reserved for the summarizer's own output: capped by the model's
    max_tokens and the output allowance, and clamped to a quarter of the
    window so a small window (tests, exotic gateways) still leaves room for
    a prompt at all."""
    base = min(config.max_tokens, _OUTPUT_ALLOWANCE)
    if config.max_context_tokens <= 0:
        return base
    return min(base, max(config.max_context_tokens // 4, 128))


def slice_budget_tokens(
    config: Config,
    anchor: str | None = None,
    checklist: str | None = None,
) -> int:
    """Token budget for one history slice (or one reduce partials payload):
    the COMPACTION role's context window minus headroom for the system
    prompt, the output template, the anchors riding along on every call,
    the model's own output and the chars/4 heuristic's slack."""
    system_prompt, template = load_compaction_prompts()
    reserve = (
        _estimate_tokens(system_prompt)
        + _estimate_tokens(template)
        + _estimate_tokens(anchor or "")
        + _estimate_tokens(checklist or "")
        + _PROMPT_SCAFFOLD_ALLOWANCE
        + _output_allowance(config)
    )
    budget = int(max(config.max_context_tokens - reserve, 0) * _SLICE_SAFETY_MARGIN)
    return max(budget, _MIN_SLICE_BUDGET)


def prompt_fits_window(prompt: str, config: Config) -> bool:
    """True when `prompt` (plus room for the summarizer's output) fits the
    COMPACTION role's window — the single-shot pre-flight check. An unknown
    window (max_context_tokens <= 0) is assumed to fit, preserving the
    pre-map-reduce behavior."""
    if config.max_context_tokens <= 0:
        return True
    return _estimate_tokens(prompt) + _output_allowance(config) <= (
        config.max_context_tokens
    )


# -- map-reduce chunking ------------------------------------------------------


def _atomic_groups(
    blocks: list[tuple[str, str]],
) -> list[list[tuple[str, str]]]:
    """Smallest block groups that must never be separated: a tool call stays
    glued to its tool return, and a user prompt stays glued to (at least the
    first group of) its answer."""
    groups: list[list[tuple[str, str]]] = []
    i = 0
    while i < len(blocks):
        kind, text = blocks[i]
        if (
            kind == "tool-call"
            and i + 1 < len(blocks)
            and blocks[i + 1][0] == "tool-result"
        ):
            groups.append([blocks[i], blocks[i + 1]])
            i += 2
        else:
            groups.append([(kind, text)])
            i += 1
    merged: list[list[tuple[str, str]]] = []
    j = 0
    while j < len(groups):
        group = groups[j]
        if (
            len(group) == 1
            and group[0][0] == "user"
            and j + 1 < len(groups)
            and groups[j + 1][0][0] != "user"
        ):
            merged.append(group + groups[j + 1])
            j += 2
        else:
            merged.append(group)
            j += 1
    return merged


def _hard_split(text: str, budget_chars: int) -> list[str]:
    """Last-resort char-level split of an oversized single block (a huge
    user prompt or a capped-but-still-huge tool result)."""
    return [text[i : i + budget_chars] for i in range(0, len(text), budget_chars)]


def _pack_groups(
    groups: list[list[tuple[str, str]]],
    budget_chars: int,
) -> list[str]:
    """Greedy packing of atomic groups into slice texts within the char
    budget; a single oversized group is hard-split."""
    slices: list[str] = []
    current: list[str] = []
    current_size = 0
    for group in groups:
        texts = [text for _, text in group]
        group_text = "\n\n".join(texts)
        group_size = _estimate_tokens(group_text) * CHARS_PER_TOKEN
        if group_size > budget_chars:
            # Flush what we have, then emit the oversized group as its own
            # hard-split slices.
            if current:
                slices.append("\n\n".join(current))
                current, current_size = [], 0
            joined = "\n\n".join(texts)
            parts = (
                _hard_split(joined, budget_chars)
                if len(texts) == 1
                else _pack_groups([[block] for block in group], budget_chars)
            )
            slices.extend(parts)
            continue
        if current and current_size + group_size > budget_chars:
            slices.append("\n\n".join(current))
            current, current_size = [], 0
        current.append(group_text)
        current_size += group_size
    if current:
        slices.append("\n\n".join(current))
    return slices


def chunk_history_for_summary(
    messages: list[dict[str, Any]],
    budget_tokens: int,
) -> list[str]:
    """Split the rendered history into slices each within ~budget_tokens
    (chars/4 heuristic), cutting only at conversation boundaries.

    Boundaries: a turn (one [user] block up to the next) is never broken
    while it fits the budget; inside an oversized turn the cut falls only
    between atomic groups — a tool call is never separated from its tool
    return, a user prompt never from the first group of its answer. A single
    group larger than the whole budget (a giant prompt under the tool-result
    cap is impossible, but a huge user prompt is not) is hard-split at char
    boundaries so the chunker always terminates with slices <= budget.
    """
    blocks = render_history_blocks(messages)
    if not blocks:
        return []
    groups = _atomic_groups(blocks)
    # Turns: cut before every group that STARTS with a user block.
    turns: list[list[list[tuple[str, str]]]] = [[]]
    for group in groups:
        if group[0][0] == "user" and turns[-1]:
            turns.append([])
        turns[-1].append(group)
    budget_chars = max(budget_tokens, 1) * CHARS_PER_TOKEN

    def join_groups(groups: list[list[tuple[str, str]]]) -> str:
        return "\n\n".join(text for group in groups for _, text in group)

    slices: list[str] = []
    current: list[list[tuple[str, str]]] = []
    current_size = 0
    for turn in turns:
        turn_size = sum(
            _estimate_tokens(text) * CHARS_PER_TOKEN
            for group in turn
            for _, text in group
        )
        if turn_size > budget_chars:
            if current:
                slices.append(join_groups(current))
                current, current_size = [], 0
            slices.extend(_pack_groups(turn, budget_chars))
            continue
        if current and current_size + turn_size > budget_chars:
            slices.append(join_groups(current))
            current, current_size = [], 0
        current.extend(turn)
        current_size += turn_size
    if current:
        slices.append(join_groups(current))
    return slices


def compacted_messages(
    messages: list[dict[str, Any]], summary: str
) -> list[dict[str, Any]]:
    """New transcript after compaction: the original user prompts verbatim
    plus one synthetic request carrying the summary."""
    preserved, _ = split_original_prompts(messages)
    request = ModelRequest(
        parts=[
            UserPromptPart(content=f"{COMPACTION_SUMMARY_PREFIX}\n\n{summary.strip()}")
        ]
    )
    return preserved + serialize_messages([request])


# -- mechanical compaction floor (no LLM call) --------------------------------


def prompt_fits_context(prompt: str, system_prompt: str, config: Config) -> bool:
    """Whether the summarizer request plausibly fits the compaction model's
    context window: the chars/4 estimate of the rendered prompt plus the
    system prompt plus the configured output reserve (`max_tokens`).

    A window ≤ 0 (unknown) counts as fitting — the LLM path is simply tried,
    and a deterministic rejection is handled by the caller."""
    if config.max_context_tokens <= 0:
        return True
    needed = (
        estimate_tokens(prompt)
        + estimate_tokens(system_prompt)
        + max(config.max_tokens, 0)
    )
    return needed <= config.max_context_tokens


_MECHANICAL_NOTE = (
    "The middle of the conversation history was dropped mechanically to fit "
    "the context window: the summarizer model could not process the full "
    "transcript, so everything between the original task prompt(s) at the "
    "top and the recent messages right above this note was removed without "
    "summarization."
)


def _message_tokens(msg: dict[str, Any]) -> int:
    """≈ wire size of one serialized message in tokens (json + chars/4 —
    json escaping makes this deliberately conservative: overestimating only
    trims more)."""
    return estimate_tokens(json.dumps(msg, ensure_ascii=False))


def _has_tool_returns(msg: dict[str, Any]) -> bool:
    return any(
        isinstance(part, dict) and part.get("part_kind") == "tool-return"
        for part in msg.get("parts", [])
    )


def mechanical_compaction_messages(
    messages: list[dict[str, Any]], token_budget: int
) -> list[dict[str, Any]]:
    """LLM-free compaction floor: the original user prompts verbatim, the
    previous summary verbatim (embedded in the note), and as many of the
    LAST messages as fit `token_budget`.

    Used when the summarizer itself cannot be run — the rendered prompt does
    not fit the compaction model's window, or the gateway rejected the
    summary request with a deterministic context-overflow error — because a
    summary call would be guaranteed to fail. The transcript is trimmed at
    message boundaries instead.

    The note rides in a COMPACTION_SUMMARY_PREFIX request, so the next
    compaction (LLM or mechanical) extracts it as the previous summary; the
    last real summary is embedded in it, so the anchored chain never loses
    the condensed history.

    An orphan tool-return request whose tool-call response was trimmed is
    dropped (a tool call is never separated from its result), and the result
    goes through `sanitized_messages` — the transcript stays API-valid no
    matter where the cut lands. Note that `preserved` cannot be trimmed (it
    IS the task assignment): when the original prompts alone exceed the
    budget, the result keeps them anyway."""
    preserved, rest = split_original_prompts(messages)
    previous = extract_previous_summary(messages)
    note = _MECHANICAL_NOTE
    if previous:
        note = f"{note}\n\n<previous-summary>\n{previous}\n</previous-summary>"
    stub = serialize_messages(
        [
            ModelRequest(
                parts=[UserPromptPart(content=f"{COMPACTION_SUMMARY_PREFIX}\n\n{note}")]
            )
        ]
    )
    used = sum(_message_tokens(m) for m in preserved)
    used += sum(_message_tokens(m) for m in stub)
    cut = len(rest)
    while cut > 0:
        tokens = _message_tokens(rest[cut - 1])
        if used + tokens > token_budget:
            break
        cut -= 1
        used += tokens
    # Never separate a tool call from its result: when the cut lands right
    # before a request carrying tool returns (their calls were trimmed),
    # drop the orphan results too.
    while cut < len(rest) and _has_tool_returns(rest[cut]):
        cut += 1
    return sanitized_messages(preserved + rest[cut:] + stub)
