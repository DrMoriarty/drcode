"""Model/client layer: build an OpenAI-compatible model and probe endpoints.

Owns everything that touches the network / the model client — the neutral
User-Agent, the OpenAI client construction (`build_model`) and the endpoint
availability probe (`probe_model`). Kept separate from the coding/tester
agent wiring in `agent.py` so the agent module stays an orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import weakref
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from openai import DEFAULT_TIMEOUT, AsyncOpenAI
from openai import Timeout as timeout_cls
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from .config import Config
from .providers import ModelEntry
from .version import APP_VERSION

# Гейтвей блокирует запросы с UA известных CLI/SDK
# (curl, OpenAI/Python, pydantic-ai, …) — против автоматизации.
# Нейтральный UA проходит; ставим его на уровне клиента и в model_settings
# (pydantic-ai перебивает client-заголовки per-request заголовком pydantic-ai/x.y).
USER_AGENT = f"drcode/{APP_VERSION}"
EXTRA_HEADERS = {"User-Agent": USER_AGENT}

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
    leaves the settings exactly as before (nothing is added)."""
    settings: ModelSettings = {
        "extra_headers": EXTRA_HEADERS,
        "max_tokens": config.max_tokens,
        **CACHE_PROMPT_SETTINGS,
    }
    provider_extra = config.provider_extra_body
    if provider_extra:
        # Merge OVER the cache-prompt flag: the provider's own values win.
        settings["extra_body"] = {**CACHE_PROMPT_EXTRA_BODY, **provider_extra}
    return settings


class TodoTailModel(WrapperModel):
    """Trailing-injection wrapper: appends the task's CURRENT checklist block
    as a synthetic last message of EVERY outgoing request.

    Why a tail and not dynamic instructions: pydantic-ai renders dynamic
    instructions into the LEADING system messages, so any mutation of the
    block (todo_create / todo_update mid-activation) moves the request
    divergence point to the top of the dialogue and invalidates the
    server-side prefix cache almost entirely (production incident #a0a55d3d:
    four todo mutations re-read the whole ~34k-token context at full price,
    ~65% of the run's bill). A trailing block sits AFTER the whole history:
    mutating it only re-reads a few tokens of the tail.

    The block is injected at REQUEST time — `request` and `request_stream`
    both copy the outgoing message list and append one synthetic
    `ModelRequest` (a user message) carrying the fresh block. Reading the
    supplier per request is what keeps the list live mid-activation: a
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

    def __init__(self, wrapped: Model, todo_block: Callable[[], str | None]) -> None:
        super().__init__(wrapped)
        self._todo_block = todo_block

    def _messages_with_tail(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """A copy of the outgoing messages with the fresh checklist block
        appended as the LAST message (None supplier result → unchanged)."""
        block = self._todo_block()
        if block is None:
            return messages
        return [
            *messages,
            ModelRequest(parts=[UserPromptPart(content=block)]),
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


# -- client-side provider throttling -------------------------------------------
#
# Providers commonly cap the number of CONCURRENT requests one client may
# have open — regardless of the model — and answer the excess with HTTP 429
# or a stalled read. drcode reaches one provider from several roles at once
# (the coding run, the tester round, a compaction summarizer, the resolver,
# a title generator, read-only subagents), so the cap has to be enforced
# per CONNECTION, not per model, role or agent.
#
# The pool is process-local: `_THROTTLE_POOLS` maps the provider identity
# `(base_url, api_key)` to one `_ThrottlePool` per event loop. Nothing is
# shared between drcode processes (the desktop shell forks one process per
# tab) on purpose — a cross-process limiter would need a lock file or a
# broker and the failure modes of both are worse than the provider's 429s.
#
# A slot is held for ONE outgoing HTTP request: `request` releases it when
# the response has arrived, `request_stream` when the stream is fully read
# (that is exactly what "concurrent open requests" means to a provider).
# Model calls are flat — a run never nests a model call inside another
# model call (compaction, the reviewer pass and a subagent are all separate
# top-level calls) — so the pool can never deadlock on itself.

# How long a request may wait for a slot before the wait is ANNOUNCED (a
# service event in the task's chat when the run can be identified). The
# consumer of that visibility is the "active, nothing happening" confusion
# of incident #a2dc8424: a queued request is otherwise indistinguishable
# from a hung one.
THROTTLE_WAIT_ANNOUNCE_SECONDS = 5.0
# Repeats of the announcement while the wait continues: a long queue must
# not look silent again.
THROTTLE_WAIT_ANNOUNCE_EVERY = 30.0


@dataclass
class _ThrottlePool:
    """The semaphore of one provider connection plus its live counters.

    `sem` is the limiting semaphore; `busy` counts the slots currently
    taken and `limit` is the configured cap, so the wait announcement can
    say "waiting for provider slot: 3/3 busy".
    """

    sem: asyncio.Semaphore
    limit: int
    busy: int = 0


# asyncio loop → {(base_url, api_key, limit) → pool}. Keyed BY the loop:
# `asyncio.Semaphore` binds to the running loop, and tests (or a future
# second loop) must not inherit a dead loop's semaphore. The loop object is
# held weakly so a finished loop's pools do not outlive it.
_THROTTLE_POOLS: weakref.WeakKeyDictionary[
    Any, dict[tuple[str, str, int], _ThrottlePool]
] = weakref.WeakKeyDictionary()


def _throttle_pool(base_url: str, api_key: str, limit: int) -> _ThrottlePool:
    """The process-wide pool of one provider connection at `limit`.

    The key is the provider IDENTITY (base URL + key), never the model: two
    roles on the same provider share the pool. A LIMIT CHANGE (the user
    edits the catalog) gets its own pool — the configured cap is part of
    the key, so a stale semaphore never lingers. Must be called from the
    event loop the pool will be used on."""
    loop = asyncio.get_running_loop()
    pools = _THROTTLE_POOLS.setdefault(loop, {})
    key = (base_url, api_key, limit)
    pool = pools.get(key)
    if pool is None:
        pool = _ThrottlePool(sem=asyncio.Semaphore(limit), limit=limit)
        pools[key] = pool
    return pool


def reset_throttle_pools() -> None:
    """Forget every pooled semaphore (tests; also a clean slate after a
    catalog change that alters a provider's cap mid-session)."""
    _THROTTLE_POOLS.clear()


async def _cancel_task(task: asyncio.Task[None] | None) -> None:
    """Cancel `task` (when there is one) and wait for it to unwind.

    Awaiting the cancelled watcher matters: `ThrottledModel` is a hot path
    and a fire-and-forget cancellation would leave "Task was destroyed but
    it is pending" noise (and a stray announcer running) behind. Shielded
    against a second cancellation of the CALLER, which must not prevent
    the cleanup. Never raises."""
    if task is None or task.done():
        return
    task.cancel()
    try:
        await asyncio.shield(task)
    except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110 - cleanup
        pass


# Consumer of the slot-wait announcement: called with (limit, busy) while a
# request is still queued. Wired by the model's builder (the pipeline passes
# a per-run callable that logs a service event into the task's chat) and
# never by the model itself — `agent_model` knows nothing about tasks.
WaitAnnouncer = Callable[[int, int], None]


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
        await asyncio.sleep(THROTTLE_WAIT_ANNOUNCE_SECONDS)
        while True:
            self._emit_announce()
            await asyncio.sleep(THROTTLE_WAIT_ANNOUNCE_EVERY)

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


def _candidate_urls(base_url: str, health_check_url: str) -> list[tuple[str, bool]]:
    """Ordered (url, needs_model_present) endpoints to probe for availability.

    ``needs_model_present`` marks a metadata endpoint whose success requires
    the requested model to appear in the response body; availability and
    metrics endpoints only need a 200. The cached ``health_check_url`` (a
    pure availability endpoint) is tried first when present.
    """
    out: list[tuple[str, bool]] = []
    if health_check_url.strip():
        out.append((health_check_url.strip(), False))
    # Canonical OpenAI-compatible /models (base_url carries the /v1 prefix).
    out.append((f"{base_url}/models", True))
    # Pure availability ping.
    out.append((f"{base_url.rstrip('/')}/health", False))
    # llama-swap: currently-loaded models.
    out.append((f"{base_url.rstrip('/')}/running", True))
    # Server metrics.
    out.append((f"{base_url.rstrip('/')}/metrics", False))
    # llama-swap: full model listing.
    out.append((f"{base_url.rstrip('/')}/models", True))
    return out


def _parse_json(resp: httpx.Response) -> Any:
    """Parse the response body as JSON; return None on an empty/invalid body."""
    try:
        body = resp.text.strip()
        if not body:
            return None
        return json.loads(body)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _model_in_data(payload: Any, model: str) -> bool:
    """Confirm `model` appears among the entries of a models-style response.

    The canonical `/v1/models` returns `{"data": [{"id": "..."}, ...]}` (a
    list), while llama-swap `/running` and `/models` return `{"data": [...]}`
    of ids; a single-object `/models` may itself be the entry. We look for
    `model` in the ids (falling back to `name`).
    """
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    if data is None:
        return False
    if isinstance(data, list):
        return any(
            item.get("id") == model or item.get("name") == model for item in data
        )
    if isinstance(data, dict):
        return data.get("id") == model or data.get("name") == model
    return False


def _ssl_verify() -> ssl.SSLContext | bool:
    """Trust-store-backed TLS verification, falling back to httpx defaults.

    Системное хранилище сертификатов: корпоративные CA отсутствуют в
    certifi, из-за чего httpx даёт CERTIFICATE_VERIFY_FAILED, который
    оборачивается в ConnectError.
    """
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return True


def _auth_headers(api_key: str) -> dict[str, str]:
    """Neutral-UA headers with the Authorization header only when a key
    is set (an empty key must be omitted entirely: "Bearer " with a
    trailing space is an illegal header value for h11/httpx)."""
    headers: dict[str, str] = {"User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def probe_model(
    model: str,
    base_url: str,
    api_key: str,
    *,
    timeout: float = 5.0,
    health_check_url: str = "",
) -> tuple[bool, str, str]:
    """Probe an OpenAI-compatible endpoint for availability without generating.

    Instead of forcing the model to produce a real reply (which stalls in the
    llama-swap processing queue at parallelism=1 and can time out), this tries
    a series of availability/metadata endpoints. Each candidate is a GET that
    needs no generation; the first that answers 200 (without a model error for
    the metadata endpoints) wins and is returned as ``checked_url``.

    Candidate order:

    1. a previously cached ``health_check_url`` (an availability endpoint);
    2. the canonical OpenAI-compatible ``/models`` ({base_url}/models), which
       reports the requested model with ``status: loaded``;
    3. ``{base_url}/health`` — a pure availability ping (200 OK);
    4. ``{base_url}/running`` — llama-swap's list of loaded models;
    5. ``{base_url}/metrics`` — server metrics;
    6. ``{base_url}/models`` — llama-swap's model listing.

    The ``/models`` and ``/running`` responses must contain the requested
    model among their entries to count as a match; the pure endpoints only
    need 200. ``detail`` is empty on success; ``checked_url`` is the winning
    URL (or the fallback ``/chat/completions`` when no metadata endpoint
    answered, or "" on failure).
    """
    verify = _ssl_verify()
    headers = _auth_headers(api_key)

    for url, needs_model in _candidate_urls(base_url, health_check_url):
        try:
            async with httpx.AsyncClient(timeout=timeout, verify=verify) as client:
                resp = await client.get(url, headers=headers)
        except httpx.HTTPError:
            # Transport failure: skip this candidate and try the next.
            continue
        if resp.status_code in (401, 403):
            continue
        if resp.status_code >= 500:
            continue
        # 200-ish: the metadata endpoints must name the requested model.
        if needs_model and not _model_in_data(_parse_json(resp), model):
            continue
        return True, "", url

    # No metadata/availability endpoint answered — fall back to a real
    # generation ping so the connection is still validated.
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
    except httpx.HTTPError as e:
        return False, f"unavailable ({type(e).__name__})", ""
    if resp.status_code in (401, 403):
        return False, f"auth failed (HTTP {resp.status_code})", ""
    if resp.status_code >= 500:
        return False, f"unavailable (HTTP {resp.status_code})", ""
    return True, "", f"{base_url.rstrip('/')}/chat/completions"


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
    client = AsyncOpenAI(
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
    `TodoTailModel`) off `model` and return the underlying one.

    Only the truly INNER model — the `OpenAIChatModel` with its
    `AsyncOpenAI` client — carries the connection settings, so introspection
    (tests reading `model.client`, a future diagnostics screen) must peel
    the wrappers first. Unknown wrapper types are left alone: the outer
    loop stops at the first model it does not recognise."""
    while isinstance(model, (ThrottledModel, TodoTailModel)):
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


# -- model auto-discovery ------------------------------------------------------

# Extra context-size keys many OpenAI-compatible gateways add to their
# /models entries (the spec-compliant response does not carry one).
# Checked in order; llama.cpp's nested meta.n_ctx handled separately.
_CONTEXT_KEYS = (
    "max_model_len",
    "context_length",
    "max_context_tokens",
    "context_window",
)


class ModelFetchError(Exception):
    """Auto-discovery of a provider's model list failed (detail inside)."""


def _models_urls(base_url: str) -> list[str]:
    """Candidate /models endpoints: the canonical one first (base_url
    normally carries the /v1 prefix), then the /v1-prefixed fallback when
    the base_url does not already end with /v1."""
    base = base_url.rstrip("/")
    urls = [f"{base}/models"]
    if not base.endswith("/v1"):
        urls.append(f"{base}/v1/models")
    return urls


def _entry_context(item: dict[str, Any]) -> int:
    """Context size (tokens) of one /models entry."""
    for key in _CONTEXT_KEYS:
        value = item.get(key)
        if isinstance(value, int) and value > 0:
            return value
    meta = item.get("meta")
    if isinstance(meta, dict):
        value = meta.get("n_ctx")  # llama.cpp
        if isinstance(value, int) and value > 0:
            return value
    return 200_000


def _parse_models_payload(payload: Any) -> list[ModelEntry]:
    """Extract model entries from a /models response body.

    Canonical shape: {"data": [{"id": ...}, ...]}; some gateways return a
    bare list of entries — accept both. Entries without an id are skipped.
    """
    items: Any = None
    if isinstance(payload, dict):
        items = payload.get("data")
    elif isinstance(payload, list):
        items = payload
    if not isinstance(items, list):
        return []
    entries: list[ModelEntry] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        context = _entry_context(item)
        entries.append(
            ModelEntry(
                id=model_id,
                max_context_tokens=context,
            )
        )
    return entries


async def fetch_models(
    base_url: str,
    api_key: str,
    *,
    timeout: float = 10.0,
    _transport: httpx.AsyncBaseTransport | None = None,
) -> list[ModelEntry]:
    """Discover the models an OpenAI-compatible endpoint serves.

    GETs `{base_url}/models` (with a `{base}/v1/models` fallback when the
    base_url has no /v1 suffix) and parses `data[].id`. Context sizes are
    taken from optional gateway-specific keys when present
    (`_CONTEXT_KEYS` / llama.cpp `meta.n_ctx`); otherwise the entry gets
    the 200_000 default (adjust by hand in the catalog UI if needed).

    Raises `ModelFetchError` when no endpoint answers with a model list.
    The `_transport` parameter exists for tests (httpx.MockTransport).
    """
    verify = _ssl_verify()
    headers = _auth_headers(api_key)
    detail = ""
    for url in _models_urls(base_url):
        try:
            async with httpx.AsyncClient(
                timeout=timeout, verify=verify, transport=_transport
            ) as client:
                resp = await client.get(url, headers=headers)
        except httpx.HTTPError as e:
            detail = f"unavailable ({type(e).__name__})"
            continue
        if resp.status_code in (401, 403):
            raise ModelFetchError(f"auth failed (HTTP {resp.status_code})")
        if resp.status_code >= 400:
            detail = f"HTTP {resp.status_code}"
            continue
        entries = _parse_models_payload(_parse_json(resp))
        if entries:
            return entries
        detail = "no models in response"
    raise ModelFetchError(detail or "unavailable")
