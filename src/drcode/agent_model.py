"""Model/client layer: build an OpenAI-compatible model and probe endpoints.

Owns everything that touches the network / the model client — the neutral
User-Agent, the OpenAI client construction (`build_model`) and the endpoint
availability probe (`probe_model`). Kept separate from the coding/tester
agent wiring in `agent.py` so the agent module stays an orchestrator.
"""

from __future__ import annotations

import json
import ssl
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Config
from .providers import ModelEntry
from .version import APP_VERSION

# Гейейвей api-copilot.x5.ru блокирует запросы с UA известных CLI/SDK
# (curl, OpenAI/Python, pydantic-ai, …) — против автоматизации.
# Нейтральный UA проходит; ставим его на уровне клиента и в model_settings
# (pydantic-ai перебивает client-заголовки per-request заголовком pydantic-ai/x.y).
USER_AGENT = f"drcode/{APP_VERSION}"
EXTRA_HEADERS = {"User-Agent": USER_AGENT}


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


def build_model(config: Config):
    client = AsyncOpenAI(
        base_url=config.base_url,
        # Keyless gateways (local servers) still need a syntactically
        # valid Authorization header: "Bearer " is rejected by httpx.
        api_key=config.api_key or "no-key",
        default_headers=EXTRA_HEADERS,
    )
    return OpenAIChatModel(config.model, provider=OpenAIProvider(openai_client=client))


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


def _entry_context(item: dict[str, Any]) -> tuple[int, bool]:
    """(context_tokens, reported_by_gateway) of one /models entry."""
    for key in _CONTEXT_KEYS:
        value = item.get(key)
        if isinstance(value, int) and value > 0:
            return value, True
    meta = item.get("meta")
    if isinstance(meta, dict):
        value = meta.get("n_ctx")  # llama.cpp
        if isinstance(value, int) and value > 0:
            return value, True
    return 200_000, False


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
        context, reported = _entry_context(item)
        entries.append(
            ModelEntry(
                id=model_id,
                max_context_tokens=context,
                context_reported=reported,
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
    the 200_000 default and `context_reported=False` ("needs manual
    confirmation" in the catalog UI).

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
