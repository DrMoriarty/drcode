"""Client-side provider throttling: the `agent_model.ThrottledModel` wrapper.

A provider may cap the number of CONCURRENT requests one client keeps open
(regardless of the model) and answer the excess with HTTP 429 or a stalled
read. drcode therefore serializes the outgoing HTTP requests of one provider
CONNECTION through a process-local semaphore pool, keyed by
`(base_url, api_key)` — never by model, role or agent. These tests pin:

* the pool key is the provider identity (two models / two roles on one
  provider share the pool, two providers never do);
* `max_parallel_requests=0` leaves the model completely unwrapped (the
  historical behavior, zero overhead);
* the cap is per HTTP REQUEST, and a streamed request holds its slot until
  the stream is fully consumed;
* a request queued for a long time is announced through the `announce`
  callback (the pipeline turns that into a service event, so a throttled
  run never looks hung — incident #a2dc8424);
* `build_model` reads the cap from the resolved config, so a catalog
  provider and a project custom model each reach the wrapper.

Everything runs through pydantic-ai's in-process `FunctionModel` stubs (see
`tests/network_mocks.py`): no HTTP, no real network.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.settings import ModelSettings

from drcode import agent_model
from drcode.agent_model import (
    THROTTLE_WAIT_ANNOUNCE_SECONDS,
    ThrottledModel,
    build_throttled_model,
    reset_throttle_pools,
)
from drcode.config import Config
from drcode.providers import AgentRole
from drcode.tui.app import DrCodeApp
from tests.network_mocks import text_response

BASE = "http://provider.local/v1"
KEY = "sk-a"


@pytest.fixture
def app_factory(project_root, monkeypatch: pytest.MonkeyPatch):
    """Build a TUI app over the temp project (the pipeline methods — the
    announcer in particular — are exercised without a running app)."""
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")

    def make() -> DrCodeApp:
        return DrCodeApp(project_root)

    return make


@pytest.fixture(autouse=True)
def _clean_pools() -> Iterator[None]:
    """Every test starts with an empty pool registry: the semaphores are
    process-global, so a leftover pool from another test (or another event
    loop) would silently share slots."""
    reset_throttle_pools()
    yield
    reset_throttle_pools()


def make_config(**overrides: Any) -> Config:
    kwargs: dict[str, Any] = {
        "model": "mock-model",
        "api_key": KEY,
        "base_url": BASE,
    }
    kwargs.update(overrides)
    return Config(**kwargs)


class _ConcurrencyProbe:
    """A `FunctionModel` body that records the peak number of overlapping
    calls — the figure a provider's own limiter measures."""

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.calls = 0

    async def _enter(self) -> None:
        self.active += 1
        self.calls += 1
        self.peak = max(self.peak, self.active)

    def _leave(self) -> None:
        self.active -= 1

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return text_response("ok")

    async def stream(
        self, messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        await self._enter()
        try:
            await asyncio.sleep(self.delay)
            yield "ok"
        finally:
            self._leave()

    async def request(
        self, messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        await self._enter()
        try:
            await asyncio.sleep(self.delay)
            return text_response("ok")
        finally:
            self._leave()


def _throttled(
    probe: _ConcurrencyProbe,
    *,
    base_url: str = BASE,
    api_key: str = KEY,
    limit: int = 1,
    announce: Any = None,
    stream: bool = False,
) -> ThrottledModel:
    inner = (
        FunctionModel(stream_function=probe.stream)
        if stream
        else FunctionModel(probe.request)
    )
    return ThrottledModel(
        inner, provider_key=(base_url, api_key), limit=limit, announce=announce
    )


async def _call(model: Any) -> None:
    """One bare (non-streaming) model request through `model`."""
    await model.request([], None, ModelRequestParameters())


def _parameters() -> ModelRequestParameters:
    return ModelRequestParameters()


# -- the zero path ---------------------------------------------------------------


def test_zero_limit_returns_the_model_unwrapped() -> None:
    """`max_parallel_requests=0` (the default) means unlimited: the model is
    returned as-is, so the historical behavior carries no wrapper at all."""
    inner = FunctionModel(lambda m, i: text_response("ok"))
    assert build_throttled_model(inner, make_config()) is inner
    assert (
        build_throttled_model(inner, make_config(provider_max_parallel_requests=0))
        is inner
    )


def test_positive_limit_wraps_the_model() -> None:
    inner = FunctionModel(lambda m, i: text_response("ok"))
    wrapped = build_throttled_model(
        inner, make_config(provider_max_parallel_requests=2)
    )
    assert isinstance(wrapped, ThrottledModel)
    assert wrapped.wrapped is inner


def test_build_model_reads_the_cap_from_the_resolved_config() -> None:
    """The resolver's `provider_max_parallel_requests` is what makes
    `build_model` wrap the real OpenAI model — the single seam every agent
    builder and the compaction/title/subagent paths go through."""
    plain = agent_model.build_model(make_config())
    assert isinstance(plain, agent_model.OpenAIChatModel)

    wrapped = agent_model.build_model(make_config(provider_max_parallel_requests=4))
    assert isinstance(wrapped, ThrottledModel)
    assert wrapped.wrapped is not None
    # The connection settings stay reachable through the wrapper chain.
    assert (
        agent_model.openai_client_of(wrapped).base_url
        == agent_model.openai_client_of(plain).base_url
    )


# -- the pool key ----------------------------------------------------------------


async def test_slot_is_held_for_the_duration_of_a_request() -> None:
    """limit=1: two concurrent requests reach the provider one at a time."""
    probe = _ConcurrencyProbe()
    model = _throttled(probe, limit=1)

    await asyncio.gather(*(_call(model) for _ in range(3)))

    assert probe.calls == 3
    assert probe.peak == 1


async def test_slot_is_shared_by_two_models_of_one_provider() -> None:
    """The cap belongs to the CONNECTION, not the model: two `ThrottledModel`
    instances of the same (base_url, api_key) draw from ONE pool — this is
    the case a per-model semaphore would get wrong."""
    probe = _ConcurrencyProbe()
    first = _throttled(probe, limit=1)
    second = _throttled(probe, limit=1)

    await asyncio.gather(*(_call(m) for m in (first, second, first, second)))

    assert probe.calls == 4
    assert probe.peak == 1


async def test_two_providers_do_not_share_the_pool() -> None:
    """A different base_url (or a different key) is a different provider and
    gets its own slots."""
    probe = _ConcurrencyProbe()
    one = _throttled(probe, limit=1, base_url="http://one.local/v1")
    two = _throttled(probe, limit=1, base_url="http://two.local/v1")

    await asyncio.gather(*(_call(m) for m in (one, two)))

    assert probe.calls == 2
    assert probe.peak == 2


async def test_a_different_key_is_a_different_provider() -> None:
    """Same host, another API key = another client as far as the provider's
    per-client limiter is concerned."""
    probe = _ConcurrencyProbe()
    one = _throttled(probe, limit=1, api_key="sk-1")
    two = _throttled(probe, limit=1, api_key="sk-2")

    await asyncio.gather(*(_call(m) for m in (one, two)))

    assert probe.peak == 2


async def test_limit_above_one_allows_exactly_that_many() -> None:
    probe = _ConcurrencyProbe(delay=0.05)
    model = _throttled(probe, limit=2)

    await asyncio.gather(*(_call(model) for _ in range(6)))

    assert probe.calls == 6
    assert probe.peak == 2


# -- streaming -------------------------------------------------------------------


async def test_stream_holds_the_slot_until_the_stream_is_consumed() -> None:
    """The slot covers the WHOLE streamed response: releasing it at the
    first chunk would let a second request in while the provider still
    counts the first one as open."""
    probe = _ConcurrencyProbe()
    model = _throttled(probe, limit=1, stream=True)

    async def consume_one() -> None:
        async with model.request_stream([], None, _parameters(), None) as streamed:
            async for _ in streamed:
                pass

    await asyncio.gather(*(consume_one() for _ in range(3)))

    assert probe.calls == 3
    assert probe.peak == 1


async def test_stream_releases_the_slot_when_abandoned_early() -> None:
    """Leaving the stream context early (aborted run) still frees the slot —
    a leaked slot would starve the provider pool for the rest of the
    session."""
    probe = _ConcurrencyProbe()
    model = _throttled(probe, limit=1, stream=True)

    async with model.request_stream([], None, _parameters(), None):
        pass  # enter and leave without reading anything

    pool = model.pool
    assert pool.busy == 0
    assert not pool.sem.locked()


# -- the wait announcement --------------------------------------------------------


async def test_long_wait_is_announced(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request queued behind a saturated pool reports the wait (the
    pipeline turns it into a service event) instead of looking hung."""
    monkeypatch.setattr(agent_model, "THROTTLE_WAIT_ANNOUNCE_SECONDS", 0.01)
    seen: list[tuple[int, int]] = []
    probe = _ConcurrencyProbe(delay=0.1)
    model = _throttled(probe, limit=1, announce=lambda l, b: seen.append((l, b)))

    await asyncio.gather(*(_call(model) for _ in range(2)))

    assert seen, "a request that had to queue must be announced"
    assert seen[0][0] == 1  # the configured limit
    assert seen[0][1] >= 1  # slots busy at that moment


async def test_short_wait_is_silent() -> None:
    """The announcement is a latch for a genuinely long queue, not a note
    for every request that happens to wait a few milliseconds."""
    seen: list[tuple[int, int]] = []
    probe = _ConcurrencyProbe(delay=0.01)
    model = _throttled(probe, limit=1, announce=lambda l, b: seen.append((l, b)))

    await asyncio.gather(*(_call(model) for _ in range(2)))

    assert seen == []
    assert THROTTLE_WAIT_ANNOUNCE_SECONDS >= 1.0


async def test_a_raising_announcer_never_breaks_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The announcer is a side channel: whatever it does, the model request
    must complete."""
    monkeypatch.setattr(agent_model, "THROTTLE_WAIT_ANNOUNCE_SECONDS", 0.01)
    probe = _ConcurrencyProbe(delay=0.1)

    def boom(limit: int, busy: int) -> None:
        raise RuntimeError("announcer exploded")

    model = _throttled(probe, limit=1, announce=boom)
    await asyncio.gather(*(_call(model) for _ in range(2)))

    assert probe.calls == 2


# -- cancellation ---------------------------------------------------------------


async def test_cancelled_queued_request_frees_nothing_and_leaks_nothing() -> None:
    """A queued request cancelled mid-wait (aborted run) never decrements
    the busy counter of a slot it does not hold."""
    probe = _ConcurrencyProbe(delay=0.1)
    model = _throttled(probe, limit=1)

    holder = asyncio.create_task(_call(model))
    await asyncio.sleep(0.02)  # let the holder take the slot
    queued = asyncio.create_task(_call(model))
    await asyncio.sleep(0.02)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    await holder

    pool = model.pool
    assert pool.busy == 0
    assert not pool.sem.locked()


# -- end-to-end through an Agent ---------------------------------------------------


async def test_agent_run_goes_through_the_slot() -> None:
    """A real `Agent` run (the wrapper is the model) still throttles: the
    wrapper must be transparent to pydantic-ai's own run machinery."""
    probe = _ConcurrencyProbe()
    agent = Agent(_throttled(probe, limit=1), instructions="be brief")

    results = await asyncio.gather(
        agent.run("one"), agent.run("two"), agent.run("three")
    )

    assert [r.output for r in results] == ["ok", "ok", "ok"]
    assert probe.peak == 1


async def test_streamed_agent_run_goes_through_the_slot() -> None:
    probe = _ConcurrencyProbe()
    agent = Agent(_throttled(probe, limit=1, stream=True), instructions="be brief")

    async def run_one(prompt: str) -> str:
        async with agent.run_stream(prompt) as streamed:
            return await streamed.get_output()

    results = await asyncio.gather(*(run_one(p) for p in ("a", "b", "c")))

    assert results == ["ok", "ok", "ok"]
    assert probe.peak == 1


# -- registry hygiene -------------------------------------------------------------


async def test_a_model_call_inside_a_tool_never_deadlocks() -> None:
    """The one real nesting shape — a tool body that itself runs a model
    (`delegate_research` subagents) — must not deadlock on the pool: the
    outer slot is released when the response ARRIVES, not when the whole
    run ends, so the inner call always finds a free slot even at limit=1.

    Both streamed paths are exercised (a plain `run`, `run_stream` and the
    `agent.iter` shape the coding runs use): a streamed outer request that
    held its slot past the response would wedge the inner one forever."""
    from pydantic_ai.messages import TextPart, ToolCallPart
    from pydantic_ai.models.function import DeltaToolCall

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # The nested sub-agent has no tools of its own.
        if not info.function_tools:
            return text_response("inner")
        if any(type(p).__name__ == "ToolReturnPart" for m in messages for p in m.parts):
            return text_response("outer")
        return ModelResponse(parts=[ToolCallPart(tool_name="nested", args={})])

    async def stream_answer(messages: list[ModelMessage], info: AgentInfo):
        part = answer(messages, info).parts[0]
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {
                0: DeltaToolCall(
                    name=part.tool_name, json_args="{}", tool_call_id=part.tool_call_id
                )
            }

    model = ThrottledModel(
        FunctionModel(answer, stream_function=stream_answer),
        provider_key=(BASE, KEY),
        limit=1,
    )
    agent = Agent(model)

    @agent.tool_plain
    async def nested() -> str:
        sub = Agent(model)
        return (await sub.run("inner question")).output

    assert (await agent.run("outer")).output == "outer"
    async with agent.run_stream("outer") as streamed:
        assert await streamed.get_output() == "outer"
    async with agent.iter("outer") as run:
        async for _node in run:
            pass
    assert run.result is not None
    assert run.result.output == "outer"


async def test_pool_identity_is_stable_within_a_provider() -> None:
    first = _throttled(_ConcurrencyProbe(), limit=2)
    second = _throttled(_ConcurrencyProbe(), limit=2)
    await _call(first)
    await _call(second)
    assert first.pool is second.pool


async def test_a_changed_limit_gets_a_fresh_pool() -> None:
    """The configured cap is part of the pool key: editing the catalog must
    not resurrect a semaphore built for the old limit."""
    before = _throttled(_ConcurrencyProbe(), limit=1)
    after = _throttled(_ConcurrencyProbe(), limit=3)
    await _call(before)
    await _call(after)
    assert before.pool is not after.pool
    assert after.pool.limit == 3


async def test_reset_throttle_pools_drops_the_registry() -> None:
    model = _throttled(_ConcurrencyProbe(), limit=1)
    await _call(model)
    pool_before = model.pool
    reset_throttle_pools()
    other = ThrottledModel(
        FunctionModel(lambda m, i: text_response("ok")),
        provider_key=(BASE, KEY),
        limit=1,
    )
    await _call(other)
    assert other.pool is not pool_before


# -- settings plumbing sanity -----------------------------------------------------


async def test_model_settings_are_forwarded_unchanged() -> None:
    """The wrapper must not reshape the request: settings and messages reach
    the inner model untouched."""
    seen: dict[str, Any] = {}

    async def request(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["messages"] = messages
        return text_response("ok")

    inner = FunctionModel(request)
    model = ThrottledModel(inner, provider_key=(BASE, KEY), limit=1)
    settings: ModelSettings = {"max_tokens": 7}
    await model.request([], settings, _parameters())
    assert seen["messages"] == []


# -- the pipeline's slot-wait announcer -------------------------------------------


def _seed_task(manager: Any) -> Any:
    from drcode.models import TaskKind

    return manager.create("docs", "wait for a slot", kind=TaskKind.RESEARCH)


def test_wait_announcer_logs_a_service_event(app_factory: Any) -> None:
    """The announcer the pipeline hands to `build_model` turns a long slot
    wait into a service event in the task's chat: a throttled run must
    never look like the "active, nothing happening" hang (#a2dc8424)."""
    from tests.network_mocks import event_texts

    app = app_factory()
    task = _seed_task(app.manager)

    announce = app._wait_announcer(task.id, AgentRole.CODING)
    assert announce is not None
    announce(2, 2)

    texts = event_texts(task)
    assert any("provider slot" in t for t in texts)
    assert any("2/2" in t for t in texts)
    # The event is persisted and never enters the model transcript.
    assert app.manager.get(task.id).messages == []


def test_wait_announcer_is_deduplicated(app_factory: Any) -> None:
    """The announcer repeats while the wait continues — the chat must not
    fill with identical lines."""
    from tests.network_mocks import event_texts

    app = app_factory()
    task = _seed_task(app.manager)

    announce = app._wait_announcer(task.id, AgentRole.CODING)
    assert announce is not None
    announce(2, 2)
    announce(2, 2)
    announce(2, 2)

    assert len([t for t in event_texts(task) if "provider slot" in t]) == 1


def test_wait_announcer_is_none_without_a_task(app_factory: Any) -> None:
    """No task (the title generator, the catalog probe) means no chat to
    explain the wait in — the model is built without an announcer."""
    app = app_factory()
    assert app._wait_announcer(None, AgentRole.CODING) is None


def test_wait_announcer_survives_a_deleted_task(app_factory: Any) -> None:
    """A task deleted while its request was queued must not raise through
    the model request."""
    app = app_factory()
    task = _seed_task(app.manager)
    announce = app._wait_announcer(task.id, AgentRole.CODING)
    assert announce is not None
    app.manager.delete(task.id)
    announce(1, 1)  # no raise


def test_build_agent_forwards_the_announcer(
    app_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`build_agent` must pass the announcer into `build_model` — the wiring
    that makes the throttle's wait visible at all."""
    app = app_factory()
    seen: dict[str, Any] = {}
    real = agent_model.build_model

    def spy(config: Config, announce_wait: Any = None):  # type: ignore[no-untyped-def]
        seen["announce"] = announce_wait
        return real(config, announce_wait)

    monkeypatch.setattr(agent_model, "build_model", spy)
    from drcode.agent_build import build_agent

    task = _seed_task(app.manager)
    announce = app._wait_announcer(task.id, AgentRole.CODING)
    build_agent(app.resolved_config(AgentRole.CODING), announce_wait=announce)
    assert seen["announce"] is announce


@pytest.mark.parametrize(
    "builder_name",
    [
        "build_tester_agent",
        "build_resolver_agent",
        "build_resolver_reviewer_agent",
        "build_title_agent",
        "build_compaction_agent",
    ],
)
def test_every_model_builder_forwards_the_announcer(
    app_factory: Any, monkeypatch: pytest.MonkeyPatch, builder_name: str
) -> None:
    """Every builder that serves a TASK carries the announcer into
    `build_model`: the tester round, the resolver round and its reviewer,
    the title generator and the compaction summarizer all share the same
    provider pool as the coding run, so a queue behind the cap must be
    equally audible there."""
    import drcode.agent as agent_mod

    app = app_factory()
    seen: dict[str, Any] = {}
    real = agent_model.build_model

    def spy(config: Config, announce_wait: Any = None):  # type: ignore[no-untyped-def]
        seen["announce"] = announce_wait
        return real(config, announce_wait)

    monkeypatch.setattr(agent_model, "build_model", spy)
    task = _seed_task(app.manager)
    announce = app._wait_announcer(task.id, AgentRole.CODING)
    builder = getattr(agent_mod, builder_name)
    builder(app.resolved_config(AgentRole.CODING), announce_wait=announce)
    assert seen["announce"] is announce


def test_delegate_research_forwards_the_announcer(
    app_factory: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The nested `delegate_research` subagent draws from the SAME provider
    pool as its parent: a queued subagent request must be announced in the
    parent task's chat (the parent run is what looks stalled), so
    `AgentDeps.wait_announcer` is forwarded into the subagent's model."""
    from drcode import subagents
    from drcode.agent_build import AgentDeps

    app = app_factory()
    seen: dict[str, Any] = {}

    def fake_model(config: Config, announce_wait: Any = None):  # type: ignore[no-untyped-def]
        seen["announce"] = announce_wait
        return FunctionModel(lambda m, i: text_response("report"))

    monkeypatch.setattr(subagents, "_build_subagent_model", fake_model)
    task = _seed_task(app.manager)
    announce = app._wait_announcer(task.id, AgentRole.CODING)
    deps = AgentDeps(
        root=tmp_path,
        manager=app.manager,
        task_id=task.id,
        scratch=tmp_path / "scratch",
        subagent_gate=subagents.SubagentGate(1),
        wait_announcer=announce,
    )

    async def run() -> str:
        return await subagents.run_delegate_tool(
            deps, app.resolved_config(AgentRole.CODING), "look around"
        )

    assert "report" in asyncio.run(run())
    assert seen["announce"] is announce
