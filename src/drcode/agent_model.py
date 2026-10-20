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
            item.get("id") == model or item.get("name") == model
            for item in data
        )
    if isinstance(data, dict):
        return data.get("id") == model or data.get("name") == model
    return False


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
    # Системное хранилище сертификатов: корпоративные CA отсутствуют в
    # certifi, из-за что httpx даёт CERTIFICATE_VERIFY_FAILED, который
    # оборачивается в ConnectError.
    try:
        import truststore

        verify: ssl.SSLContext | bool = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        verify = True

    headers: dict[str, str] = {"User-Agent": USER_AGENT}
    if api_key:
        # An empty key must be omitted entirely: "Bearer " (trailing
        # space) is an illegal header value for h11/httpx.
        headers["Authorization"] = f"Bearer {api_key}"

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
