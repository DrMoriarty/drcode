"""Transcript helpers: serialize/load messages, track usage, render for chat.

Serialization (`serialize_messages` / `load_messages`) round-trips the
pydantic-ai transcript; the API-valid truncation helpers (`sanitized_messages`,
`resumable_history`, `ends_with_open_request`) guard an interrupted tail; the
usage helpers (`_usage_context_tokens`, `last_context_tokens`,
`run_context_tokens`, `last_response_usage`) feed the TUI accumulator; and
`extract_display` renders the transcript into timestamped (role, text) lines
for the chat thread.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic_ai.agent import AgentRun
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.usage import RunUsage

# Marker prefix of the synthetic user message that carries the compaction
# summary inside the transcript. Lets us (a) feed the previous summary back
# to the summarizer as <previous-summary> on the next compaction and
# (b) render it in the chat as a "context compacted" block instead of a
# regular user message.
COMPACTION_SUMMARY_PREFIX = "[context summary]"


def serialize_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    return ModelMessagesTypeAdapter.dump_python(messages, mode="json")  # type: ignore[return-value]


def sanitized_messages(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transcript truncated to its API-valid prefix (resumable_history rule).

    Applied when a run is interrupted: the invalid tail (a response with
    tool calls whose results never arrived, the empty interrupted request
    pydantic-ai appends) is dropped so the stored transcript stays valid no
    matter in which order later user messages get appended to it.
    """
    return serialize_messages(resumable_history(data))


def ends_with_open_request(data: list[dict[str, Any]]) -> bool:
    """True when the transcript ends with an unanswered user request —
    i.e. there is an interrupted attempt that a continuation turn can pick
    up (the transcript is expected to be sanitized at that point)."""
    if not data:
        return False
    last = data[-1]
    return last.get("kind") == "request" or last.get("role") == "request"


def _usage_context_tokens(input_tokens: int, output_tokens: int) -> int:
    """Tokens held by the model's context slot: prompt + generated output,
    matching llama.cpp's slot `n_tokens`. pydantic-ai normalizes usage from
    all providers, so `input_tokens` already INCLUDES cache reads/writes
    (cache fields must not be added on top)."""
    return input_tokens + output_tokens


def _context_status(
    tokens: int | None, max_tokens: int, compaction_threshold: int
) -> str:
    """Calm context-fill report for the model: how many tokens it already
    holds, what percent of the window that is, and how many tokens remain
    before the compaction threshold. Only states the numbers — never alarms —
    so the model can manage its own reading instead of guessing, or fretting
    over a leaky "nearly full" signal."""
    if not tokens:
        return "Context fill: empty (0 tokens loaded)."
    if max_tokens <= 0:
        return f"Context fill: {tokens} tokens loaded (no maximum configured)."
    pct = round(tokens * 100 / max_tokens)
    threshold_tokens = round(compaction_threshold * max_tokens / 100)
    headroom = threshold_tokens - tokens
    return (
        f"Context fill: {tokens} tokens loaded of {max_tokens} "
        f"({pct}% used). {headroom} tokens of headroom before the "
        f"{compaction_threshold}% compaction threshold ({threshold_tokens} tokens)."
    )


def last_context_tokens(data: list[dict[str, Any]]) -> int | None:
    """Context fill (tokens) at the end of a serialized transcript.

    Each ModelResponse carries the usage of the request that produced it,
    and its input + output tokens are the full context the model holds
    (history + tools + prompt + generated output). The last response with
    non-zero usage therefore gives the final context fill of the task — for
    finished tasks as well, since the transcript with usage is persisted in
    the task history.
    """
    for msg in reversed(data):
        if msg.get("kind") != "response" and msg.get("role") != "response":
            continue
        usage = msg.get("usage") or {}
        tokens = _usage_context_tokens(
            usage.get("input_tokens") or 0,
            usage.get("output_tokens") or 0,
        )
        if tokens:
            return tokens
    return None


def run_context_tokens(run: AgentRun[Any, Any]) -> int | None:
    """Context fill (tokens) after the latest completed model request of a
    live run (usage of the last ModelResponse in the run's transcript)."""
    try:
        messages = run.all_messages()
    except Exception:  # noqa: BLE001 - state may be mid-teardown after cancel
        return None
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            usage = msg.usage
            tokens = _usage_context_tokens(usage.input_tokens, usage.output_tokens)
            if tokens:
                return tokens
    return None


def ctx_context_tokens(ctx: Any) -> int | None:
    """Context fill (tokens) of the model's context slot for a live
    `RunContext` — same semantics as `run_context_tokens`, but read from
    the run transcript the context carries instead of the run object.

    `RunContext.usage` is a cumulative counter over the whole run (every
    model response adds to it), while the actual window fill is the
    input + output tokens of the *latest* request — so tools must not
    consult `ctx.usage` directly.
    """
    for msg in reversed(ctx.messages):
        if isinstance(msg, ModelResponse):
            usage = msg.usage
            tokens = _usage_context_tokens(usage.input_tokens, usage.output_tokens)
            if tokens:
                return tokens
    return None


def last_response_usage(run: AgentRun[Any, Any]) -> RunUsage | None:
    """Exact token usage of the latest completed model request of a live run.

    Each ModelResponse carries the usage of the request that produced it;
    the last response with non-zero usage therefore gives the per-request
    input/cache/output figures for the accumulator in the TUI (distinct
    from run_context_tokens, which only sums the input side). None when
    the run has no completed model request yet.
    """
    try:
        messages = run.all_messages()
    except Exception:  # noqa: BLE001 - state may be mid-teardown after cancel
        return None
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            u = msg.usage
            if (
                u.input_tokens
                or u.output_tokens
                or u.cache_read_tokens
                or u.cache_write_tokens
            ):
                return RunUsage(
                    input_tokens=u.input_tokens,
                    cache_read_tokens=u.cache_read_tokens,
                    cache_write_tokens=u.cache_write_tokens,
                    output_tokens=u.output_tokens,
                )
    return None


def load_messages(data: list[dict[str, Any]]) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_python(data)


def resumable_history(data: list[dict[str, Any]]) -> list[ModelMessage]:
    """Load persisted messages for resuming an interrupted/previous run.

    The only tail a model API cannot accept is a half-done exchange: a
    model response with tool calls whose results never arrived, or a
    response the stream cut off before any part arrived (an empty
    assistant message). The transcript is truncated right before such a
    response. Everything else — user prompts, tool returns, retry prompts,
    completed text answers — is a valid point to continue from and is
    kept, so a retried run resumes the attempt where it stopped instead of
    redoing it from scratch.
    """
    messages = list(load_messages(data))
    cut = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest) or (
            isinstance(msg, ModelResponse)
            and any(isinstance(p, TextPart) and p.content for p in msg.parts)
            and not any(isinstance(p, ToolCallPart) for p in msg.parts)
        ):
            cut = i + 1
    return messages[:cut]


def is_pending_retry(messages: list[dict[str, Any]], text: str) -> bool:
    """True when `text` is already the prompt of an unfinished attempt in the
    serialized transcript (possible only for legacy tasks whose interrupted
    prompt was re-queued into `pending_input` by older builds). The chat
    renders the prompt and the partial work from the transcript itself;
    showing the pending copy too would draw the same message twice.
    """
    idx: int | None = None
    for i, msg in enumerate(messages):
        if msg.get("kind") != "request" and msg.get("role") != "request":
            continue
        for part in msg.get("parts", []):
            if part.get("part_kind") == "user-prompt" and part.get("content") == text:
                idx = i
    if idx is None:
        return False
    for msg in reversed(messages[idx + 1 :]):
        if msg.get("kind") == "response" or msg.get("role") == "response":
            parts = msg.get("parts", [])
            has_text = any(
                p.get("part_kind") == "text" and p.get("content") for p in parts
            )
            has_tool_call = any(p.get("part_kind") == "tool-call" for p in parts)
            return not (has_text and not has_tool_call)
    return True


def backdate_user_prompt(
    messages: list[ModelMessage], text: str, at: datetime
) -> None:
    """Stamp the user request carrying `text` with `at` — the moment the
    user typed the prompt — in place.

    pydantic-ai timestamps the ModelRequest it builds from a consumed
    pending prompt with the consumption time, which can be minutes after
    the message was actually sent (the prompt may sit in pending_input
    while other work runs). The chat thread is ordered by these stamps, so
    the request is backdated to keep the transcript chronological.
    """
    for msg in reversed(messages):
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if isinstance(part, UserPromptPart) and part.content == text:
                part.timestamp = at
                msg.timestamp = at
                return


def first_user_prompt(messages: list[dict[str, Any]]) -> str:
    """First user prompt of a serialized transcript — the original task text.

    Used to hand the tester the task's initial assignment (the title alone is
    often too terse to paraphrase into a commit message).
    """
    for msg in messages:
        if msg.get("kind") != "request" and msg.get("role") != "request":
            continue
        for part in msg.get("parts", []):
            if part.get("part_kind") == "user-prompt":
                content = part.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return ""


def _message_time(msg: dict[str, Any]) -> datetime | None:
    """Timestamp of a serialized ModelMessage (None when not recorded).

    pydantic-ai 2.x puts a top-level `timestamp` on both ModelRequest and
    ModelResponse; older transcripts only carried per-part timestamps on
    requests (UserPromptPart / ToolReturnPart), so those are used as a
    fallback (the latest part stamp)."""
    raw = msg.get("timestamp")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            raw = None
    if isinstance(raw, datetime):
        return raw
    stamps: list[datetime] = []
    for part in msg.get("parts", []):
        if not isinstance(part, dict):
            continue
        part_raw = part.get("timestamp")
        if isinstance(part_raw, str):
            try:
                stamps.append(datetime.fromisoformat(part_raw))
            except ValueError:
                pass
        elif isinstance(part_raw, datetime):
            stamps.append(part_raw)
    return max(stamps) if stamps else None


def extract_display(
    messages: list[dict[str, Any]],
) -> list[tuple[datetime | None, str, str]]:
    """Turn serialized ModelMessages into (timestamp, role, text) triples
    for the chat UI.

    Roles: "user", "reasoning" (model thinking), "assistant" (text),
    "tool-call" (name(args) line; the TUI shows the name as the collapsed
    header and the args in the expandable body), rendered by the TUI in
    message order. The timestamp (message creation time, None when the
    transcript format didn't record one) lets the TUI interleave these
    entries with other timestamped streams (service events, tester rounds)
    into a single chronological timeline.
    """
    out: list[tuple[datetime | None, str, str]] = []
    reasoning: list[str] = []
    chunks: list[str] = []
    for msg in messages:
        # pydantic-ai 2.x serializes ModelRequest with "kind": "request"
        # (older versions used "role"); support both.
        at = _message_time(msg)
        if msg.get("kind") == "request" or msg.get("role") == "request":
            for part in msg.get("parts", []):
                if part.get("part_kind") == "user-prompt":
                    content = part.get("content")
                    if isinstance(content, str) and content:
                        if content.startswith(COMPACTION_SUMMARY_PREFIX):
                            # Synthetic compaction summary, not a user message:
                            # rendered as its own "context compacted" block.
                            out.append(
                                (
                                    at,
                                    "summary",
                                    content[len(COMPACTION_SUMMARY_PREFIX) :].strip(),
                                )
                            )
                        else:
                            out.append((at, "user", content))
            continue
        for part in msg.get("parts", []):
            kind = part.get("part_kind")
            if kind == "thinking":
                content = part.get("content", "")
                if content:
                    if chunks:
                        out.append((at, "assistant", "\n".join(chunks)))
                        chunks = []
                    reasoning.append(content)
            elif kind == "text":
                content = part.get("content", "")
                if content:
                    if reasoning:
                        out.append((at, "reasoning", "\n\n".join(reasoning)))
                        reasoning = []
                    chunks.append(content)
            elif kind == "tool-call":
                args = part.get("args")
                args_str = (
                    args
                    if isinstance(args, str)
                    else json.dumps(args or {}, ensure_ascii=False)
                )
                if reasoning:
                    out.append((at, "reasoning", "\n\n".join(reasoning)))
                    reasoning = []
                if chunks:
                    out.append((at, "assistant", "\n".join(chunks)))
                    chunks = []
                out.append(
                    (at, "tool-call", f"{part.get('tool_name', '?')}({args_str})")
                )
        if reasoning:
            out.append((at, "reasoning", "\n\n".join(reasoning)))
            reasoning = []
        if chunks:
            out.append((at, "assistant", "\n".join(chunks)))
            chunks = []
    return out
