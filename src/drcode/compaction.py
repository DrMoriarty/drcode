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


def render_history_for_summary(messages: list[dict[str, Any]]) -> str:
    """Render the serialized transcript as readable text for the summarizer.

    User prompts, assistant text, tool calls and tool results are kept
    (tool results are the bulk of the working context — file contents and
    command output the summary must capture); thinking parts are dropped;
    compaction-summary user-prompt parts are skipped (they are passed
    separately as <previous-summary>) — including the merged ones pydantic-ai
    folds into the original-prompt request after a resume.
    """
    lines: list[str] = []
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
                    lines.append(f"[user]\n{content}")
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
                lines.append(f"[tool result for {part.get('tool_name', '?')}]\n{text}")
            elif is_request:
                continue
            elif kind == "text":
                content = part.get("content", "")
                if content:
                    lines.append(f"[assistant]\n{content}")
            elif kind == "tool-call":
                args = part.get("args")
                args_str = (
                    args
                    if isinstance(args, str)
                    else json.dumps(args or {}, ensure_ascii=False)
                )
                lines.append(f"[tool call] {part.get('tool_name', '?')}({args_str})")
    return "\n\n".join(lines)


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
