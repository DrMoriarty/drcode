"""App-level network-interaction tests: the real TUI pipeline (run_agent /
run_tester workers, the _advance gate, retry bookkeeping) with the model
layer stubbed out.

Two failure surfaces are simulated:
- scripted model errors (FunctionModel raising ModelHTTPError/ModelAPIError
  on cue) — the model request path;
- a mocked httpx.AsyncClient for the connection-probe worker (_check_model).

Rate-limit and tester retry timers are fired immediately (the real backoff
delays are recorded instead of slept), so the tests stay fast while the
full retry bookkeeping runs for real.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from drcode import agent as agent_mod
from drcode.agent import (
    ends_with_open_request,
    extract_display,
    is_summary_message,
    last_context_tokens,
    serialize_messages,
)
from drcode.models import TaskKind, TaskStatus, VerificationStatus
from drcode.tui.app import DrCodeApp
from tests.network_mocks import (
    ScriptedState,
    api_error,
    event_texts,
    http_error,
    status_handler,
    text_response,
    tool_response,
    wait_until,
)

# -- fixtures ---------------------------------------------------------------------


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured (but never contacted) model endpoint. The base URL
    points at a black-hole port: nothing must dial it — the model layer is
    mocked — but a real dial fails instantly instead of hanging."""
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


@pytest.fixture
def net_app(project_root: Path, configured_env: None) -> DrCodeApp:
    return DrCodeApp(project_root)


@pytest.fixture
def net_app_small_ctx(
    project_root: Path, configured_env: None, monkeypatch: pytest.MonkeyPatch
) -> DrCodeApp:
    monkeypatch.setenv("DRCODE_MAX_CONTEXT_TOKENS", "1000")
    return DrCodeApp(project_root)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedState:
    """The model layer of every agent drcode builds (coding, tester and
    compaction alike) is a FunctionModel consuming `state.steps`."""
    state = ScriptedState()
    monkeypatch.setattr(agent_mod, "build_model", lambda cfg: state.model())
    return state


@pytest.fixture
def instant_timers(net_app: DrCodeApp, monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Fire retry timers immediately, recording the requested backoff
    delays so tests can still assert on the backoff schedule."""
    delays: list[float] = []

    def fake_set_timer(
        delay: float, callback: Callable[[], Any], *args: Any, **kwargs: Any
    ) -> None:
        delays.append(delay)
        net_app.call_later(callback, *args, **kwargs)

    monkeypatch.setattr(net_app, "set_timer", cast(Any, fake_set_timer))
    return delays


# -- agent run: happy path ----------------------------------------------------------


async def test_agent_tool_roundtrip_completes_task(
    net_app: DrCodeApp, scripted: ScriptedState, project_root: Path
) -> None:
    (project_root / "README.md").write_text("hello readme")
    task = net_app.manager.create(
        "docs", "summarize readme", kind=TaskKind.DOCUMENTATION
    )
    scripted.steps = [
        tool_response("read_file", {"path": "README.md"}),
        text_response("readme says hello"),
    ]
    async with net_app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED,
            pilot,
            what="documentation task completed",
        )

    assert task.pending_input == []
    assert scripted.calls == ["summarize readme", "summarize readme"]
    kinds = [p["part_kind"] for m in task.messages for p in m["parts"]]
    assert "tool-call" in kinds
    assert "tool-return" in kinds
    # No error events: the run was clean end to end.
    assert not [t for t in event_texts(task) if "failed" in t or "Error" in t]


# -- agent run: rate limiting ---------------------------------------------------------


async def test_rate_limited_run_retries_with_backoff_then_exhausts_budget(
    net_app: DrCodeApp, scripted: ScriptedState, instant_timers: list[float]
) -> None:
    task = net_app.manager.create("limited", "do work", kind=TaskKind.DOCUMENTATION)
    # 5 retries within the budget, then one more 429 that exhausts it.
    scripted.steps = [http_error(429) for _ in range(6)]

    async with net_app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.QUEUED,
            pilot,
            what="task demoted after rate-limit budget exhausted",
        )

    texts = event_texts(task)
    retries = [t for t in texts if t.startswith("HTTP 429")]
    assert [f"retry {i}/5" in t for i, t in enumerate(retries, 1)] == [True] * 5
    # Backoff schedule 5s doubling, capped at 60 (RATE_LIMIT_RETRIES = 5).
    assert instant_timers == [5.0, 10.0, 20.0, 40.0, 60.0]
    # The last failure fell through to the generic failure path.
    assert any(t.startswith("ModelHTTPError") for t in texts)
    assert task.unfinished
    assert task.id in net_app._agent_failed
    # The prompt was consumed into the transcript (not re-queued): the
    # retry resumes the attempt instead of re-sending the text.
    assert task.pending_input == []
    assert ends_with_open_request(task.messages)


# -- agent run: connection outage -------------------------------------------------------


async def test_connection_outage_demotes_task_and_recovery_resumes_it(
    net_app: DrCodeApp, scripted: ScriptedState
) -> None:
    task = net_app.manager.create("docs", "write the docs", kind=TaskKind.DOCUMENTATION)
    scripted.steps = [api_error("Connection error.")]

    async with net_app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.QUEUED and task.unfinished,
            pilot,
            what="task demoted after connection failure",
        )
        assert task.id in net_app._agent_failed
        assert any(t.startswith("ModelAPIError") for t in event_texts(task))
        # The consumed prompt lives in the sanitized transcript.
        assert len(task.messages) == 1
        assert ends_with_open_request(task.messages)

        # Outage over: model answers now; recovery un-blocks the task and
        # the pipeline resumes the interrupted attempt.
        scripted.steps = [text_response("recovered answer")]
        net_app._recover_from_outage()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED,
            pilot,
            what="task completed after outage recovery",
        )

    assert [m["kind"] for m in task.messages] == ["request", "response"]
    assert [
        text for _, role, text in extract_display(task.messages) if role == "user"
    ] == ["write the docs"]
    assert scripted.calls == ["write the docs", "write the docs"]


# -- tester run: network failure -------------------------------------------------------


async def test_tester_network_failure_retried_then_verification_passes(
    net_app: DrCodeApp, scripted: ScriptedState
) -> None:
    task = net_app.manager.create("feature", "implement x", kind=TaskKind.FEATURE)
    scripted.steps = [
        text_response("work done"),  # development run
        api_error("Connection error."),  # tester round 1 dies
        tool_response("set_verification", {"success": True}),  # round 2
        text_response("all checks passed"),
    ]

    async with net_app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: any("tester failed" in t for t in event_texts(task)),
            pilot,
            what="tester failure logged",
        )
        assert task.status is TaskStatus.TESTING
        assert net_app._tester_fail_count[task.id] == 1

        # Fast-forward the backoff window once the failed attempt has fully
        # wound down, and let the pipeline retry.
        net_app._tester_retry_at[task.id] = time.monotonic() - 1
        net_app._advance()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED,
            pilot,
            what="tester retry completed the task",
        )
        # The verification tool flips the status mid-run; the tester worker
        # keeps going (final summary request, re-render) — wait for it to
        # fully wind down before tearing the app down.
        await wait_until(
            lambda: not net_app.tester_running,
            pilot,
            what="tester worker fully finished",
        )

    assert task.verification is VerificationStatus.PASSED
    # Two testing rounds persisted (the failed one included).
    assert len(task.tester_messages) == 2
    assert task.id not in net_app._tester_fail_count
    assert task.id not in net_app._tester_retry_at


# -- missing configuration ---------------------------------------------------------------


async def test_model_not_configured_keeps_prompt_pending(project_root: Path) -> None:
    """No model/key configured (conftest clears the env): the run stops at
    the guard, the prompt is kept for a later send."""
    app = DrCodeApp(project_root)
    task = app.manager.create("docs", "some prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: bool(event_texts(task)), pilot, what="guard event logged"
        )
    assert any("Model is not configured" in t for t in event_texts(task))
    assert task.status is TaskStatus.ACTIVE
    assert task.pending_input == ["some prompt"]
    assert not app.agent_running


# -- context compaction over the network -------------------------------------------------


def _seed_transcript(app: DrCodeApp, prompt: str) -> Any:
    """A persisted transcript whose last response reports a big context
    fill (2k tokens) — past the compaction threshold of the small-ctx app."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )
    from pydantic_ai.usage import RequestUsage

    task = app.manager.create("docs", "", kind=TaskKind.DOCUMENTATION)
    response = ModelResponse(
        parts=[TextPart("partial work")],
        usage=RequestUsage(input_tokens=2000, output_tokens=8),
    )
    task.messages = serialize_messages(
        [ModelRequest(parts=[UserPromptPart(prompt)]), response]
    )
    app.manager.storage.save(task)
    return task


async def test_compaction_failure_is_not_fatal(
    net_app_small_ctx: DrCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai.models.function import FunctionModel

    def failing(messages: list, info: Any) -> Any:
        raise api_error("Connection error.")

    monkeypatch.setattr(agent_mod, "build_model", lambda cfg: FunctionModel(failing))
    task = _seed_transcript(net_app_small_ctx, "long task " * 200)

    async with net_app_small_ctx.run_test() as pilot:
        await pilot.pause()
        await net_app_small_ctx._maybe_compact(task)
        await pilot.pause()

    assert any(
        e.text.startswith("compaction failed: ModelAPIError") for e in task.events
    )
    # Uncompressed history: the run continues with what it has.
    assert not any(is_summary_message(m) for m in task.messages)
    assert last_context_tokens(task.messages) == 2000


async def test_compaction_success_replaces_history(
    net_app_small_ctx: DrCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai.models.function import FunctionModel

    state = ScriptedState()
    state.steps = [text_response("THE SUMMARY BODY")]
    monkeypatch.setattr(agent_mod, "build_model", lambda cfg: FunctionModel(state.fn))
    prompt = "long task " * 200
    task = _seed_transcript(net_app_small_ctx, prompt)

    async with net_app_small_ctx.run_test() as pilot:
        await pilot.pause()
        await net_app_small_ctx._maybe_compact(task)
        await pilot.pause()

    assert any("Context compacted" in t for t in event_texts(task))
    # Original prompt preserved verbatim + one synthetic summary request.
    assert len(task.messages) == 2
    assert not is_summary_message(task.messages[0])
    assert is_summary_message(task.messages[1])
    assert "THE SUMMARY BODY" in task.messages[1]["parts"][0]["content"]
    assert not state.steps  # exactly one summarizer request happened


# -- connection probe (_check_model) ------------------------------------------------------


def _patch_probe_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            kwargs.pop("verify", None)
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(**kwargs)

    monkeypatch.setattr("drcode.tui.app.httpx.AsyncClient", PatchedAsyncClient)


@pytest.mark.parametrize(
    "status,expected",
    [
        (200, "● mock-model"),
        (401, "auth failed (HTTP 401)"),
        (503, "unavailable (HTTP 503)"),
    ],
)
async def test_check_model_status_labels(
    net_app: DrCodeApp, monkeypatch: pytest.MonkeyPatch, status: int, expected: str
) -> None:
    _patch_probe_transport(monkeypatch, status_handler(status))
    async with net_app.run_test() as pilot:
        await pilot.pause()
        await net_app._check_model()
        await pilot.pause()
        label = str(net_app.model_status.render())
    assert expected in label


@pytest.mark.parametrize(
    "exc", [httpx.ConnectError("refused"), httpx.ReadTimeout("timed out")]
)
async def test_check_model_transport_error_shows_unavailable(
    net_app: DrCodeApp, monkeypatch: pytest.MonkeyPatch, exc: httpx.HTTPError
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    _patch_probe_transport(monkeypatch, handler)
    async with net_app.run_test() as pilot:
        await pilot.pause()
        await net_app._check_model()
        await pilot.pause()
        label = str(net_app.model_status.render())
    assert f"unavailable ({type(exc).__name__})" in label


async def test_check_model_ok_after_outage_triggers_recovery(
    net_app: DrCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    state: dict[str, int] = {"status": 503}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(state["status"], json={}, request=request)

    _patch_probe_transport(monkeypatch, handler)
    recoveries: list[int] = []
    monkeypatch.setattr(net_app, "_recover_from_outage", lambda: recoveries.append(1))
    async with net_app.run_test() as pilot:
        await pilot.pause()
        await net_app._check_model()
        assert "unavailable" in str(net_app.model_status.render())
        # The outage is over — the recovery must un-block failed tasks.
        state["status"] = 200
        await net_app._check_model()
        await pilot.pause()
        assert "● mock-model" in str(net_app.model_status.render())
    assert recoveries == [1]
