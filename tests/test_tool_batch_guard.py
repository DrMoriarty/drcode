"""Tests for the degenerate tool-batch guard (tool_runtime.BatchGuard).

Some models emit degenerate parallel tool batches (up to 32 calls in one
response, often with exact duplicates — task 1ad9c359 on
xiaomi/mimo-v2.6-flash). The guard is wired into every drcode agent as a
WrapperToolset around the agent's own function toolset; these tests exercise
it through a real pydantic-ai Agent with a scripted FunctionModel, so the
tool_call_id-per-call protocol, the run_step epoch and the parallel
dispatch all behave exactly as in production.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from drcode.config import Config
from drcode.tool_runtime import (
    DEFAULT_TOOL_BATCH_LIMIT,
    BatchGuardToolset,
    attach_batch_guard,
)


def _scripted_model(responses: list[ModelResponse]) -> FunctionModel:
    """A FunctionModel consuming one scripted response per model request."""

    steps = list(responses)

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert steps, "scripted model queue is empty"
        return steps.pop(0)

    return FunctionModel(fn)


def _call(call_id: str, label: str) -> ToolCallPart:
    return ToolCallPart(tool_name="probe", args={"label": label}, tool_call_id=call_id)


def _build_agent(
    responses: list[ModelResponse],
    limit: int,
    executed: list[str],
    failing: set[str] | None = None,
) -> Agent[None, str]:
    """An agent with one counting `probe` tool and the batch guard attached.
    A label in `failing` raises ModelRetry on its FIRST execution only, so a
    scripted re-issue of the same call succeeds (the retry-prompt round)."""
    agent = Agent(_scripted_model(responses), retries=3)
    attempts: dict[str, int] = {}

    @agent.tool
    async def probe(ctx: RunContext[Any], label: str) -> str:
        executed.append(label)
        attempts[label] = attempts.get(label, 0) + 1
        if failing and label in failing and attempts[label] == 1:
            raise ModelRetry(f"bad label: {label}")
        return f"result:{label}"

    attach_batch_guard(agent, limit)
    return agent


def _tool_returns(messages: list[ModelMessage]) -> dict[str, str]:
    """tool_call_id → tool-return content across the whole run."""
    returns: dict[str, str] = {}
    for message in messages:
        for part in getattr(message, "parts", []):
            if isinstance(part, ToolReturnPart) and isinstance(part.content, str):
                returns[part.tool_call_id] = part.content
    return returns


async def test_duplicate_calls_in_one_batch_execute_once() -> None:
    """Two identical (tool, args) calls in ONE response: the first executes,
    the twin gets the SAME result content under its own tool_call_id."""
    executed: list[str] = []
    agent = _build_agent(
        [
            ModelResponse(parts=[_call("c1", "a"), _call("c2", "a"), _call("c3", "b")]),
            ModelResponse(parts=[TextPart(content="done")]),
        ],
        limit=DEFAULT_TOOL_BATCH_LIMIT,
        executed=executed,
    )

    result = await agent.run("go")

    # "a" executed exactly once despite two calls for it.
    assert executed == ["a", "b"]
    returns = _tool_returns(result.all_messages())
    # The protocol is intact: every tool_call_id got its answer.
    assert set(returns) == {"c1", "c2", "c3"}
    assert returns["c1"] == returns["c2"] == "result:a"
    assert returns["c3"] == "result:b"


async def test_batch_over_limit_refuses_extra_calls() -> None:
    """Calls beyond the per-response limit are NOT executed: each gets a
    plain (non-retry) tool result telling the model to re-issue later."""
    executed: list[str] = []
    agent = _build_agent(
        [
            ModelResponse(
                parts=[
                    _call("c1", "a"),
                    _call("c2", "b"),
                    _call("c3", "c"),
                    _call("c4", "d"),
                ]
            ),
            ModelResponse(parts=[TextPart(content="done")]),
        ],
        limit=2,
        executed=executed,
    )

    result = await agent.run("go")

    # Emission order is execution order (pydantic-ai creates one task per
    # call in response order; the guard counts at task start, before any
    # await) — the first two run, the rest are refused.
    assert executed == ["a", "b"]
    returns = _tool_returns(result.all_messages())
    assert set(returns) == {"c1", "c2", "c3", "c4"}
    assert returns["c1"] == "result:a"
    assert returns["c2"] == "result:b"
    assert returns["c3"].startswith("Batch limit exceeded")
    assert returns["c4"].startswith("Batch limit exceeded")


async def test_same_args_in_next_response_execute_again() -> None:
    """A repeated call with the same args in a LATER response is legitimate
    (re-read after an edit) and must execute — the dedup epoch resets with
    every model response."""
    executed: list[str] = []
    agent = _build_agent(
        [
            ModelResponse(parts=[_call("c1", "a")]),
            ModelResponse(parts=[_call("c2", "a")]),
            ModelResponse(parts=[TextPart(content="done")]),
        ],
        limit=DEFAULT_TOOL_BATCH_LIMIT,
        executed=executed,
    )

    await agent.run("go")

    assert executed == ["a", "a"]


async def test_failed_call_is_retryable_in_next_response() -> None:
    """A ModelRetry failure is not cached as a result: the model's re-issue
    in the next response (the retry prompt round) executes and succeeds."""
    executed: list[str] = []
    agent = _build_agent(
        [
            ModelResponse(parts=[_call("c1", "a")]),
            ModelResponse(parts=[_call("c2", "a")]),
            ModelResponse(parts=[TextPart(content="done")]),
        ],
        limit=DEFAULT_TOOL_BATCH_LIMIT,
        executed=executed,
        failing={"a"},
    )

    result = await agent.run("go")

    # The first attempt failed (ModelRetry), the retry round executed again.
    assert executed == ["a", "a"]
    returns = _tool_returns(result.all_messages())
    assert returns["c2"] == "result:a"


async def test_disabled_guard_leaves_toolset_untouched() -> None:
    """limit=0 disables the guard entirely (toolset not wrapped)."""
    agent = Agent(_scripted_model([ModelResponse(parts=[TextPart(content="x")])]))
    original = agent._function_toolset
    attach_batch_guard(agent, 0)
    assert agent._function_toolset is original

    attach_batch_guard(agent, 3)
    assert isinstance(agent._function_toolset, BatchGuardToolset)
    assert agent._function_toolset.guard.limit == 3


async def test_tool_registered_after_attach_still_guarded() -> None:
    """`@agent.tool` after attach_batch_guard must still work (the wrapper
    forwards registration into the inner toolset) and stay guarded."""
    executed: list[str] = []
    agent = Agent(
        _scripted_model(
            [
                ModelResponse(parts=[_call("c1", "a"), _call("c2", "a")]),
                ModelResponse(parts=[TextPart(content="done")]),
            ]
        ),
        retries=3,
    )
    attach_batch_guard(agent, DEFAULT_TOOL_BATCH_LIMIT)

    @agent.tool
    async def probe(ctx: RunContext[Any], label: str) -> str:
        executed.append(label)
        return f"result:{label}"

    result = await agent.run("go")

    assert executed == ["a"]
    returns = _tool_returns(result.all_messages())
    assert returns["c1"] == returns["c2"] == "result:a"


def test_config_default_is_sane() -> None:
    """The config field exists and defaults to the documented 8."""
    config = Config()
    assert config.tool_batch_limit == 8
