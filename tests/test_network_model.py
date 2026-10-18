"""Model-layer network tests: the real OpenAIChatModel against a fake HTTP
endpoint (httpx.MockTransport — no real network).

Covers the failure modes a live endpoint produces and verifies they surface
as the pydantic-ai exception types the rest of the code base branches on
(`ModelHTTPError` for HTTP statuses, `ModelAPIError` for transport
failures), plus the request configuration drcode sends (base_url, auth,
User-Agent override).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from drcode import agent as agent_mod
from drcode.config import Config
from drcode.version import APP_VERSION
from tests.network_mocks import (
    completion_payload,
    mock_openai_model,
    raising_handler,
    sse_text_stream,
    status_handler,
)


def make_config(**overrides: object) -> Config:
    kwargs: dict[str, object] = {
        "project_root": Path.cwd(),
        "model": "mock-model",
        "api_key": "sk-test",
        "base_url": "http://mock.local/v1",
    }
    kwargs.update(overrides)
    return Config(**kwargs)  # type: ignore[arg-type]


# -- HTTP status errors ------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 502, 503])
async def test_http_status_errors_surface_as_model_http_error(status: int) -> None:
    ag = Agent(mock_openai_model(status_handler(status)))
    with pytest.raises(ModelHTTPError) as exc_info:
        await ag.run("hi")
    assert exc_info.value.status_code == status
    assert exc_info.value.model_name == "mock-model"


async def test_rate_limit_error_keeps_status_for_retry_logic() -> None:
    """The app's 429 branch keys off ModelHTTPError.status_code == 429."""
    ag = Agent(mock_openai_model(status_handler(429)))
    with pytest.raises(ModelHTTPError) as exc_info:
        await ag.run("hi")
    assert exc_info.value.status_code == 429


# -- transport-level failures --------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("connection refused"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
    ],
)
async def test_transport_errors_surface_as_model_api_error(
    exc: httpx.HTTPError,
) -> None:
    ag = Agent(mock_openai_model(raising_handler(exc)))
    with pytest.raises(ModelAPIError) as exc_info:
        await ag.run("hi")
    # Transport failures carry no HTTP status — the app's generic failure
    # path (not the 429 retry branch) must handle them.
    assert not isinstance(exc_info.value, ModelHTTPError)
    assert exc_info.value.model_name == "mock-model"


# -- request configuration -----------------------------------------------------------


def test_build_model_client_uses_configured_base_url_and_key(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, json=completion_payload("ok"))

    real_cls = agent_mod.AsyncOpenAI

    def factory(**kwargs: object) -> AsyncOpenAI:
        kwargs["http_client"] = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        kwargs["max_retries"] = 0
        return real_cls(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(agent_mod, "AsyncOpenAI", factory)
    model = agent_mod.build_model(make_config(api_key="sk-secret"))

    resp = asyncio.run(
        model.client.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert resp.choices[0].message.content == "ok"
    assert seen["url"].startswith("http://mock.local/v1/chat/completions")
    assert seen["auth"] == "Bearer sk-secret"
    # The neutral UA (gateway policy — see agent.py) must be set on the
    # client itself.
    assert seen["ua"] == f"drcode/{APP_VERSION}"


def test_build_model_default_base_url_is_openai() -> None:
    model = agent_mod.build_model(make_config(base_url="https://api.openai.com/v1"))
    assert str(model.client.base_url).rstrip("/").endswith("/v1")


async def test_streamed_roundtrip_and_per_request_user_agent_override() -> None:
    """Happy streaming path (agent.iter + node.stream, exactly the loop the
    TUI runs) and the model_settings UA override beating the pydantic-ai
    default 'pydantic-ai/x.y' header."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers.get("user-agent")
        return httpx.Response(
            200,
            content=sse_text_stream("mock says hi"),
            headers={"content-type": "text/event-stream"},
        )

    ag = Agent(
        mock_openai_model(handler),
        model_settings={"extra_headers": {"User-Agent": f"drcode/{APP_VERSION}"}},
    )
    async with ag.iter("hello") as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for _ in stream:
                        pass

    assert run.result is not None
    assert run.result.output == "mock says hi"
    assert seen["ua"] == f"drcode/{APP_VERSION}"
    assert run.result.usage.input_tokens > 0
