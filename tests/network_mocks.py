"""Reusable mocks for network-interaction tests.

Two mocking layers are provided:

1. `ScriptedState` + `FunctionModel` — pydantic-ai's in-process model stub.
   No HTTP at all: each model request consumes the next scripted step
   (a `ModelResponse` or an exception). Used for app-level and
   transcript-recovery tests, where the *reaction* to a network failure
   matters, not its transport.

2. `mock_openai_model` + `httpx.MockTransport` — the real `OpenAIChatModel`
   talking to a fake HTTP endpoint. Used for model-layer tests: it exercises
   the actual OpenAI SDK / pydantic-ai error mapping (HTTP statuses,
   connection failures, timeouts) and the request headers drcode sends.

Neither layer touches the real network.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import httpx
from openai import AsyncOpenAI
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

MODEL_NAME = "mock-model"
MOCK_BASE_URL = "http://mock.local/v1"


# -- scripted responses / errors -------------------------------------------------


def http_error(status: int, message: str = "request failed") -> ModelHTTPError:
    """A pydantic-ai model-layer HTTP error (what the OpenAI SDK's 4xx/5xx
    responses are mapped to by pydantic-ai)."""
    return ModelHTTPError(
        status_code=status,
        model_name=MODEL_NAME,
        body={"error": {"message": message}},
    )


def api_error(message: str = "Connection error.") -> ModelAPIError:
    """A pydantic-ai model-layer transport error (what connection failures
    and timeouts are mapped to)."""
    return ModelAPIError(model_name=MODEL_NAME, message=message)


def text_response(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def tool_response(
    tool: str, args: dict[str, Any], call_id: str = "c1"
) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name=tool, args=args, tool_call_id=call_id)]
    )


def last_user_text(messages: list[ModelMessage]) -> str:
    """Concatenated user-prompt text of a request — lets tests route by
    prompt content (e.g. detect a compaction request) and assert on what
    the model actually received."""
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    parts.append(part.content)
    return "\n".join(parts)


class ScriptedState:
    """A queue of model responses / exceptions consumed in request order.

    `steps` may be refilled between runs (e.g. after a simulated outage).
    Every request records its user-prompt text in `calls` so tests can
    assert on the exact number of model round-trips.
    """

    def __init__(self) -> None:
        self.steps: list[ModelResponse | Exception] = []
        self.calls: list[str] = []

    def fn(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls.append(last_user_text(messages))
        if not self.steps:
            raise AssertionError("scripted model queue is empty")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    async def stream_fn(
        self, messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        """Streamed variant: replays the scripted response as deltas (the
        TUI consumes model responses through the streaming API)."""
        response = self.fn(messages, info)
        for part in response.parts:
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                args = (
                    part.args
                    if isinstance(part.args, str)
                    else json.dumps(part.args or {})
                )
                yield {
                    0: DeltaToolCall(
                        name=part.tool_name,
                        json_args=args,
                        tool_call_id=part.tool_call_id,
                    )
                }

    def model(self) -> FunctionModel:
        """A FunctionModel consuming this script, usable for both plain and
        streamed requests."""
        return FunctionModel(self.fn, stream_function=self.stream_fn)


# -- raw HTTP mocks (real OpenAIChatModel through a fake transport) ----------------


def mock_async_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def mock_openai_model(
    handler: Callable[[httpx.Request], httpx.Response],
    model_name: str = MODEL_NAME,
    base_url: str = MOCK_BASE_URL,
) -> OpenAIChatModel:
    """A real OpenAIChatModel whose HTTP traffic goes through `handler`.

    max_retries=0: the SDK's built-in retry/backoff would silently absorb
    (and sleep on) the very failures these tests simulate.
    """
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="sk-test",
        # openai vendors its own httpx (httpx2) — the type checkers see two
        # distinct AsyncClient types for the same runtime object.
        http_client=cast(Any, mock_async_client(handler)),
        max_retries=0,
    )
    return OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=client))


def status_handler(status: int, body: dict[str, Any] | None = None):
    """Handler answering every request with a fixed HTTP status."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json=body if body is not None else {"error": {"message": f"HTTP {status}"}},
            request=request,
        )

    return handler


def raising_handler(exc: Exception):
    """Handler simulating a transport-level failure (connection refused,
    read timeout, ...)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


def completion_payload(text: str, usage: tuple[int, int] = (12, 4)) -> dict[str, Any]:
    """Non-streaming chat-completions JSON body."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[0] + usage[1],
        },
    }


def sse_text_stream(text: str, usage: tuple[int, int] = (12, 4)) -> str:
    """Streaming chat-completions body (SSE) delivering a single text part."""
    first = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": None,
            }
        ],
    }
    last = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[0] + usage[1],
        },
    }
    return sse_body(first, last)


def sse_body(*chunks: dict[str, Any]) -> str:
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


# -- async test helpers ----------------------------------------------------------


async def wait_until(
    pred: Callable[[], bool],
    pilot: Any = None,
    timeout: float = 10.0,
    what: str = "condition",
) -> None:
    """Poll `pred` (yielding to the event loop, and to the textual app when
    a pilot is given) until it holds or the deadline passes."""
    import asyncio
    import time

    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for: {what}")
        if pilot is not None:
            await pilot.pause(0.01)
        else:
            await asyncio.sleep(0.01)


def event_texts(task: Any) -> list[str]:
    return [e.text for e in task.events]
