from __future__ import annotations

import asyncio
import urllib.parse
from dataclasses import dataclass

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.usage import RunUsage

# Transient network-failure handling for every agent's model calls (coding
# agent and tester alike): a run that dies with a transport error
# (ModelAPIError — connection refused, timeout, DNS failure) or HTTP 429
# rate limiting is retried in place (the task keeps its status) after a
# backoff delay, at most NETWORK_RETRIES times — only when the budget is
# exhausted does the run fall back to the ordinary failure path (requeue +
# session exclusion). The retry budget resets after any successful model
# call (see the per-request counter resets in run_agent/run_tester).
NETWORK_RETRIES = 5
# Backoff: 5s doubling per failure, capped here (5, 10, 20, 40, 60).
NETWORK_DELAY_MAX = 60.0
# Outage mode: when the in-place retry budget is exhausted (consecutive
# retryable failures exceed NETWORK_RETRIES with no successful model call
# in between — a sustained outage, e.g. the machine slept through the
# gateway's downtime), the run stops heading for the ordinary failure
# path (requeue + session exclusion) and switches to outage mode: the
# task KEEPS its status and keeps retrying in place on a longer backoff —
# the same 5s doubling schedule, capped here (15 min) instead of 60s —
# until a model request succeeds again. Any success clears the outage
# (and the per-task counters with it).
NETWORK_OUTAGE_DELAY_MAX = 900.0
# Before an outage-mode retry re-dials the model, a cheap TCP connect to
# the gateway host gates the attempt: while the host is unreachable the
# tick only re-probes at this interval instead of firing full model
# requests (which would pile junk transcript entries onto a dead run).
NETWORK_OUTAGE_PROBE_INTERVAL = 60.0


def _is_retryable_network_error(e: BaseException) -> bool:
    """Transient network failures the in-place retry branch handles:
    transport-level errors (any plain ModelAPIError: connection refused,
    timeouts, DNS failures) and HTTP 429 rate limiting. ModelHTTPError
    subclasses ModelAPIError, so the HTTP branch is checked first — every
    other HTTP status (400/401/500/…) is not a network error and falls
    through to the generic failure path."""
    if isinstance(e, ModelHTTPError):
        return e.status_code == 429
    return isinstance(e, ModelAPIError)


# Body/message fragments identifying a deterministic context-window
# overflow: ModelHTTPError 400s whose text matches are NOT transient —
# the same transcript will be rejected on every retry, so requeueing and
# rescuing the task would only loop it through the error forever.
# Matched case-insensitively against str(error) (which embeds the
# response body) — the list covers the phrasings of the common gateways
# (OpenAI "maximum context length" / "context_length_exceeded",
# Anthropic "prompt is too long", generic "context too long" / "too many
# tokens" / "reduce the length of the input").
_CONTEXT_OVERFLOW_MARKERS = (
    "context length",
    "context_length",
    "context too long",
    "context window",
    "prompt is too long",
    "input is too long",
    "input too long",
    "too many tokens",
    "reduce the length of the input",
    "longer than the model's context",
)


def _is_context_overflow_error(e: BaseException) -> bool:
    """Deterministic model failure: HTTP 400 whose body says the request
    exceeds the model's context window. Unlike transport errors and 429s
    this cannot succeed on retry — the transcript must shrink (mechanical
    compaction) or the task must be handled manually."""
    if not isinstance(e, ModelHTTPError) or e.status_code != 400:
        return False
    text = str(e).lower()
    return any(marker in text for marker in _CONTEXT_OVERFLOW_MARKERS)


def _in_outage(attempt: int) -> bool:
    """True when retry `attempt` (1-based, consecutive retryable failures
    since the last successful model call) is past the ordinary retry
    budget — i.e. the run is in outage mode."""
    return attempt > NETWORK_RETRIES


def _network_retry_delay(attempt: int, outage: bool = False) -> float:
    """Backoff before retry `attempt` (1-based): 5s doubling, capped —
    at NETWORK_DELAY_MAX inside the ordinary budget, at the (much higher)
    NETWORK_OUTAGE_DELAY_MAX in outage mode (the schedule just keeps
    doubling past 60s: 160, 320, 640, 900, 900, …)."""
    cap = NETWORK_OUTAGE_DELAY_MAX if outage else NETWORK_DELAY_MAX
    return min(5 * 2 ** (attempt - 1), cap)


async def tcp_reachable(base_url: str, timeout: float = 5.0) -> bool:
    """Cheap outage gate: can we open a TCP connection to `base_url`'s
    host at all? No HTTP, no model request — a single connect with a
    short timeout, so an outage-mode tick can tell "the laptop just
    woke up, the network is still down" from "worth a real attempt"
    without burning full model requests (and the transcript entries
    they would leave behind).

    Conservative on purpose: an URL that cannot be parsed (no host) or
    a scheme without a default port counts as reachable, degrading to
    the ordinary retry path instead of blocking a working setup on a
    probe bug."""
    try:
        parsed = urllib.parse.urlparse(base_url)
        host = parsed.hostname
        if not host:
            return True
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return True
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    except (OSError, TimeoutError):
        # Connection refused, DNS failure, timeouts — host unreachable.
        # (asyncio.TimeoutError is an alias of TimeoutError on 3.11+.)
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return True


@dataclass
class SessionTokenUsage:
    """Cumulative exact token usage of every model call this session.

    Fed from the four per-response usage fields pydantic-ai reports on
    each ModelResponse (main agent, tester and the compaction summarizer
    alike) — billing-grade numbers, unlike the char/4 approximation the
    live tok/s figure uses. Session-scoped only: resets on restart, never
    backfilled from persisted transcripts.
    """

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage: RunUsage | None) -> None:
        """Fold one request's (or run's) usage in; None is a no-op."""
        if usage is None:
            return
        self.input_tokens += usage.input_tokens or 0
        self.cache_read_tokens += usage.cache_read_tokens or 0
        self.cache_write_tokens += usage.cache_write_tokens or 0
        self.output_tokens += usage.output_tokens or 0
