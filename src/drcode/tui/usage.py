from __future__ import annotations

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


def _network_retry_delay(attempt: int) -> float:
    """Backoff before retry `attempt` (1-based): 5s doubling, capped."""
    return min(5 * 2 ** (attempt - 1), NETWORK_DELAY_MAX)


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

