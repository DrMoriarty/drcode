"""The pydantic-ai/openai half of the model layer.

Split out of :mod:`drcode.agent_model` for cold-start latency: that
module stays importable WITHOUT pydantic_ai/openai (the probe and the
/models auto-discovery only need httpx), so the TUI startup never pays
the ~2 s pydantic_ai+openai import. Everything that constructs or
wraps an actual model lives here and loads on the first real use (the
first `build_model` call — the first agent run, probe-with-generation,
compaction summarizer…).

`drcode.agent_model` re-exports these names lazily (module
`__getattr__`), so historical imports
(`from .agent_model import build_model`, `agent_model.build_model(...)`,
tests' `monkeypatch.setattr(agent_model, "build_model", ...)`) keep
working — with the one documented caveat that monkeypatching the
re-export binds the name on `agent_model`, while the pipeline imports
it through the same attribute access, so both resolve identically.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from openai import DEFAULT_TIMEOUT, AsyncOpenAI
from openai import Timeout as timeout_cls
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from . import agent_model
from .agent_model import (
    EXTRA_HEADERS,
    WaitAnnouncer,
    _cancel_task,
    _throttle_pool,
    _ThrottlePool,
)
from .config import Config
from .providers import REASONING_EFFORT_LEVELS

__all__ = [
    "DEFAULT_TIMEOUT",
    "REASONING_EFFORT_LEVELS",
    "AsyncOpenAI",
    "TailInjectionModel",
    "ThrottledModel",
    "base_model_settings",
    "build_model",
    "build_throttled_model",
    "openai_client_of",
    "unwrap_model",
]

# Флаг prompt-кэширования OpenAI-совместимых серверов (vLLM, llama.cpp
# server): true — сервер кэширует префикс промпта (системный промпт +
# неизменяемое начало истории), что заметно ускоряет и удешевляет
# повторные запросы с длинным общим префиксом. Уходит в тело POST
# /chat/completions через extra_body (pydantic-ai ModelSettings);
# строгие OpenAI-совместимые серверы могут отвергнуть неизвестное поле.
# The dict form kept separately: merging provider extra-body fields over it
# needs a plainly-typed dict[str, Any] (ModelSettings["extra_body"] is an
# optional key of a wide union type — not directly copyable per pyright).
CACHE_PROMPT_EXTRA_BODY: dict[str, Any] = {"cache_prompt": True}
CACHE_PROMPT_SETTINGS: ModelSettings = {"extra_body": CACHE_PROMPT_EXTRA_BODY}


def base_model_settings(config: Config) -> ModelSettings:
    """The shared ModelSettings of every drcode agent: the neutral
    User-Agent headers, the per-response token cap, the prompt-cache flag
    and the resolved provider's `extra_body` fields.

    The provider extra body (catalog `Provider.extra_body`, carried into
    the resolved Config as `provider_extra_body` by the resolver) is MERGED
    into the cache-prompt extra body — never a replacement: an empty dict
    leaves the settings exactly as before (nothing is added).

    The reasoning knobs of the resolved connection ride the same extra
    body ON TOP of the provider fields (a per-model explicit knob outranks
    the provider-wide routing fields):

    - `reasoning_effort` (valid level only — "off" and None send nothing)
      as the `reasoning_effort` body field;
    - `thinking_budget` (> 0) as the Anthropic-style
      `thinking: {type: enabled, budget_tokens: N}` block.

    Setting BOTH knobs is allowed but unusual: the gateway decides which
    one it honors — set one per model (see `ModelEntry`)."""
    body: dict[str, Any] = {**CACHE_PROMPT_EXTRA_BODY}
    provider_extra = config.provider_extra_body
    if provider_extra:
        # Merge OVER the cache-prompt flag: the provider's own values win…
        body.update(provider_extra)
    # …but the explicit per-model reasoning knobs win over everything:
    # they were set for THIS model specifically.
    if config.reasoning_effort in REASONING_EFFORT_LEVELS:
        body["reasoning_effort"] = config.reasoning_effort
    if config.thinking_budget is not None and config.thinking_budget > 0:
        body["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.thinking_budget,
        }
    settings: ModelSettings = {
        "extra_headers": EXTRA_HEADERS,
        "max_tokens": config.max_tokens,
        "extra_body": body,
    }
    return settings


class TailInjectionModel(WrapperModel):
    """Trailing-injection wrapper: appends the per-request blocks supplied by
    its suppliers as a synthetic last message of EVERY outgoing request.

    Why a tail and not dynamic instructions: pydantic-ai renders dynamic
    instructions into the LEADING system messages, so any mutation of a
    block (todo_create / todo_update mid-activation) moves the request
    divergence point to the top of the dialogue and invalidates the
    server-side prefix cache almost entirely (production incident #a0a55d3d:
    four todo mutations re-read the whole ~34k-token context at full price,
    ~65% of the run's bill). A trailing block sits AFTER the whole history:
    mutating it only re-reads a few tokens of the tail.

    Every block rides the SAME synthetic message: several suppliers (the
    task's live checklist and the run's delegation note, say) are read at
    request time and their non-None texts joined into ONE `ModelRequest` —
    one extra user message per request at most, never one per block.

    The blocks are injected at REQUEST time — `request` and
    `request_stream` both copy the outgoing message list and append that
    synthetic `ModelRequest` (a user message) carrying the fresh text.
    Reading the suppliers per request is what keeps them live mid-run: a
    todo_update executed in the current run is visible to the very next
    model request of that run.

    The synthetic message is never persisted: pydantic-ai passes the model
    a shallow copy of the run's message history, and this wrapper copies
    again before appending — the run state and therefore the stored
    transcript (the "instructions never touch the transcript" rule, see
    `todo_instructions`) stay clean.

    Note: appending one more user message after the history (even after a
    tool-return turn) is universally valid for OpenAI-compatible wire
    formats; no leading system zone is touched.
    """

    def __init__(
        self,
        wrapped: Model,
        blocks: Sequence[Callable[[], str | None]],
    ) -> None:
        super().__init__(wrapped)
        self._blocks = list(blocks)

    def _messages_with_tail(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """A copy of the outgoing messages with the fresh blocks appended as
        the LAST message (no non-None block → the list unchanged)."""
        parts = [
            block
            for block in (supplier() for supplier in self._blocks)
            if block is not None and block.strip()
        ]
        if not parts:
            return messages
        return [
            *messages,
            ModelRequest(parts=[UserPromptPart(content="\n\n".join(parts))]),
        ]

    async def request(self, messages, model_settings, model_request_parameters):  # type: ignore[no-untyped-def]
        return await self.wrapped.request(
            self._messages_with_tail(messages),
            model_settings,
            model_request_parameters,
        )

    @asynccontextmanager
    async def request_stream(  # type: ignore[override]
        self,
        messages: list[ModelMessage],
        model_settings: Any,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncGenerator[StreamedResponse]:
        async with self.wrapped.request_stream(
            self._messages_with_tail(messages),
            model_settings,
            model_request_parameters,
            run_context,
        ) as response_stream:
            yield response_stream


class ThrottledModel(WrapperModel):
    """Client-side concurrency limiter for one provider connection.

    Wraps the model produced by `build_model` (the INNERMOST wrapper, so the
    slot is taken as late as possible — after the checklist tail and any
    other request shaping) and serializes outgoing HTTP requests through the
    process-wide pool of the provider's `(base_url, api_key)`.

    `limit <= 0` means unlimited: `build_throttled_model` then returns the
    model unwrapped, so the zero-cost path has no wrapper at all.

    `announce` (optional) is called when a request has been waiting for a
    slot for `THROTTLE_WAIT_ANNOUNCE_SECONDS` — and every
    `THROTTLE_WAIT_ANNOUNCE_EVERY` seconds after that — with the pool's
    (limit, busy) pair. It runs on the event loop; a raising announcer must
    never break the request (the call is guarded).

    The pool is looked up on FIRST USE, inside the running event loop:
    `build_model` is routinely called off-loop (the catalog screen's probe,
    the context-layout report), where creating a semaphore would bind it to
    a loop that never runs.
    """

    def __init__(
        self,
        wrapped: Model,
        provider_key: tuple[str, str],
        limit: int,
        announce: WaitAnnouncer | None = None,
    ) -> None:
        super().__init__(wrapped)
        self._provider_key = provider_key
        self._limit = limit
        self._announce = announce
        self._pool: _ThrottlePool | None = None

    @property
    def pool(self) -> _ThrottlePool:
        """The shared pool of this connection (created on first use)."""
        if self._pool is None:
            base_url, api_key = self._provider_key
            self._pool = _throttle_pool(base_url, api_key, self._limit)
        return self._pool

    # -- slot acquisition ---------------------------------------------------

    @asynccontextmanager
    async def _slot(self) -> AsyncGenerator[None]:
        """Hold one provider slot for the duration of the block, announcing
        a long wait (see the class docstring)."""
        pool = self.pool
        watcher: asyncio.Task[None] | None = None
        if pool.sem.locked():
            # The request is about to queue behind a full pool: announce
            # the wait if it does not get a slot soon enough.
            watcher = asyncio.create_task(self._announce_while_waiting())
        try:
            await pool.sem.acquire()
        finally:
            await _cancel_task(watcher)
        pool.busy += 1
        try:
            yield
        finally:
            pool.busy -= 1
            pool.sem.release()

    async def _announce_while_waiting(self) -> None:
        """Announce the queued request once it has waited long enough, then
        repeat at a slower cadence while it is still queued."""
        # The two tunables are read THROUGH `agent_model` (not bound at
        # import time) so a test monkeypatching
        # `agent_model.THROTTLE_WAIT_ANNOUNCE_*` reaches the wrapper.
        await asyncio.sleep(agent_model.THROTTLE_WAIT_ANNOUNCE_SECONDS)
        while True:
            self._emit_announce()
            await asyncio.sleep(agent_model.THROTTLE_WAIT_ANNOUNCE_EVERY)

    def _emit_announce(self) -> None:
        if self._announce is None:
            return
        pool = self.pool
        try:
            self._announce(pool.limit, max(pool.busy, 1))
        except Exception:  # noqa: BLE001, S110 - never break a run for a note
            pass

    # -- model interface ----------------------------------------------------

    async def request(self, messages, model_settings, model_request_parameters):  # type: ignore[no-untyped-def]
        async with self._slot():
            return await self.wrapped.request(
                messages, model_settings, model_request_parameters
            )

    @asynccontextmanager
    async def request_stream(  # type: ignore[override]
        self,
        messages: list[ModelMessage],
        model_settings: Any,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncGenerator[StreamedResponse]:
        # The slot lives until the caller leaves the context manager, i.e.
        # until the stream is fully consumed — the provider keeps counting
        # the request as open for exactly as long.
        async with (
            self._slot(),
            self.wrapped.request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as response_stream,
        ):
            yield response_stream


def _client_timeout(config: Config) -> Any:
    """The `timeout` argument for `AsyncOpenAI`.

    A configured `request_timeout` (> 0) becomes a per-phase timeout with
    the SDK's own connect/pool values kept — raising only read/write so a
    slow-but-alive generation is not cut off, and no request stalls forever
    either. 0/absent keeps the SDK default (`DEFAULT_TIMEOUT`, 600 s
    read/write), i.e. `None` is passed through unchanged.
    """
    if config.request_timeout <= 0:
        return DEFAULT_TIMEOUT
    return timeout_cls(
        connect=DEFAULT_TIMEOUT.connect,
        read=config.request_timeout,
        write=config.request_timeout,
        pool=DEFAULT_TIMEOUT.pool,
    )


def build_model(config: Config, announce_wait: WaitAnnouncer | None = None):
    """Build the OpenAI-compatible model of a resolved role config.

    When the provider carries a concurrency cap
    (`Config.provider_max_parallel_requests` > 0, from the catalog entry or
    the project's custom model), the model is wrapped in a
    `ThrottledModel`: every outgoing HTTP request of this connection — from
    every role and every agent in this process — draws from one shared pool
    of slots, so the provider never sees more parallel requests than it
    allows. 0 = unlimited: the model is returned unwrapped (no overhead).

    `announce_wait` is forwarded to the wrapper: it is called while a
    request is queued for longer than `THROTTLE_WAIT_ANNOUNCE_SECONDS`, so
    the pipeline can surface the wait in the task's chat.
    """
    # `AsyncOpenAI` is read THROUGH `agent_model` (not bound at import
    # time) so a test monkeypatching `agent_model.AsyncOpenAI` reaches the
    # construction.
    client = agent_model.AsyncOpenAI(
        base_url=config.base_url,
        # Keyless gateways (local servers) still need a syntactically
        # valid Authorization header: "Bearer " is rejected by httpx.
        api_key=config.api_key or "no-key",
        default_headers=EXTRA_HEADERS,
        # Per-role request budget (Config.request_timeout, written by
        # providers.resolve from ProjectSettings.role_request_timeout):
        # 0 = the SDK default. Only the compaction role sets it today —
        # a huge summarizer prefill needs a longer read window than a
        # coding turn.
        timeout=_client_timeout(config),
        # No SILENT SDK-internal retries (the openai default is
        # max_retries=2): a stalled request (e.g. a gateway holding the
        # call in its processing queue while the model slot is busy) is
        # then retried invisibly twice more after the 600 s read timeout —
        # up to ~30 min of total mute with zero events, the "active,
        # nothing happening" hang of task #a2dc8424. drcode has its own
        # retry layer for transport errors (core/usage.py: NETWORK_RETRIES
        # + backoff via _agent_turn_network_error) which announces every
        # attempt in the task's service events — the first timeout must
        # surface THERE, not be absorbed here (the test mocks in
        # tests/network_mocks.py already run with max_retries=0 for the
        # same reason).
        max_retries=0,
    )
    # Строгие OpenAI-совместимые шлюзы (vLLM) отвергают запрос с больше чем
    # одним ведущим `system`-сообщением: "System message must be at the
    # beginning." (400). pydantic-ai по умолчанию превращает каждый
    # instruction-part (статические instructions + динамические: чек-лист
    # задач, roadmap goals, подсказка mark_unneeded) в ОТДЕЛЬНОЕ ведущее
    # system-сообщение и склеивает их только при профиле
    # openai_chat_supports_multiple_system_messages=False — задаём его явно
    # (дефолтный профиль рассчитан на OpenAI, который терпит несколько).
    return build_throttled_model(
        OpenAIChatModel(
            config.model,
            provider=OpenAIProvider(openai_client=client),
            profile=OpenAIModelProfile(
                openai_chat_supports_multiple_system_messages=False
            ),
        ),
        config,
        announce_wait,
    )


def build_throttled_model(
    model: Model, config: Config, announce_wait: WaitAnnouncer | None = None
) -> Model:
    """Wrap `model` in a `ThrottledModel` when the resolved config asks for
    a concurrency cap; return it unchanged at 0 (unlimited).

    Applied INSIDE `build_model`, so the limiter is the INNERMOST wrapper
    (the checklist tail and any other request-shaping wrapper sit outside
    it): the slot is taken as late as possible and released as soon as the
    transport is done. The pool is keyed by `(base_url, api_key)`: two
    roles on one provider share it, two providers never do. The pool
    itself is resolved lazily (on first use, inside the running event
    loop) — `build_model` is routinely called off-loop (catalog screen
    probes, context estimation)."""
    limit = config.provider_max_parallel_requests
    if limit <= 0:
        return model
    return ThrottledModel(
        model,
        provider_key=(config.base_url, config.api_key),
        limit=limit,
        announce=announce_wait,
    )


def unwrap_model(model: Model) -> Model:
    """Strip drcode's request-shaping wrappers (`ThrottledModel`,
    `TailInjectionModel`) off `model` and return the underlying one.

    Only the truly INNER model — the `OpenAIChatModel` with its
    `AsyncOpenAI` client — carries the connection settings, so introspection
    (tests reading `model.client`, a future diagnostics screen) must peel
    the wrappers first. Unknown wrapper types are left alone: the outer
    loop stops at the first model it does not recognise."""
    while isinstance(model, (ThrottledModel, TailInjectionModel)):
        model = model.wrapped
    return model


def openai_client_of(model: Model) -> AsyncOpenAI:
    """The `AsyncOpenAI` client behind `model` (wrappers peeled off).

    Raises `TypeError` when the chain does not end at an
    `OpenAIChatModel` — a caller asking for the client of a stubbed model
    has a wiring bug, not a runtime condition to tolerate."""
    inner = unwrap_model(model)
    if not isinstance(inner, OpenAIChatModel):
        raise TypeError(f"no OpenAI client behind {type(model).__name__}")
    return inner.client
