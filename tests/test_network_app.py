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

from drcode import agent_model
from drcode.agent import (
    ends_with_open_request,
    extract_display,
    is_summary_message,
    last_context_tokens,
    serialize_messages,
)
from drcode.models import TaskKind, TaskStatus, VerificationStatus
from drcode.tui import pipeline as pipeline_module
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
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())
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


async def test_queued_comments_merge_into_one_first_prompt(
    net_app: DrCodeApp, scripted: ScriptedState
) -> None:
    """The original brief plus comments typed before the run starts (while
    the task sat in the queue) reach the agent as ONE prompt, so it solves
    the task as a whole instead of one message per turn."""
    task = net_app.manager.create("t", "original brief", kind=TaskKind.RESEARCH)
    net_app.manager.send(task.id, "first comment")
    net_app.manager.send(task.id, "second comment")
    scripted.steps = [text_response("done")]
    async with net_app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED,
            pilot,
            what="research task completed",
        )

    assert task.pending_input == []
    # Exactly one model round-trip, carrying the whole brief at once.
    assert len(scripted.calls) == 1
    assert scripted.calls[0] == "original brief\n\nfirst comment\n\nsecond comment"


async def test_agent_tool_roundtrip_completes_task(
    net_app: DrCodeApp, scripted: ScriptedState, project_root: Path
) -> None:
    (project_root / "README.md").write_text("hello readme")
    task = net_app.manager.create("docs", "summarize readme", kind=TaskKind.RESEARCH)
    scripted.steps = [
        tool_response("read_file", {"path": "README.md"}),
        text_response("readme says hello"),
    ]
    async with net_app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED,
            pilot,
            what="research task completed",
        )

    assert task.pending_input == []
    assert scripted.calls == ["summarize readme", "summarize readme"]
    kinds = [p["part_kind"] for m in task.messages for p in m["parts"]]
    assert "tool-call" in kinds
    assert "tool-return" in kinds
    # No error events: the run was clean end to end.
    assert not [t for t in event_texts(task) if "failed" in t or "Error" in t]


async def test_research_task_builds_agent_with_research_role_config(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_agent resolves the role per task kind: a research task gets the
    RESEARCH role's resolved config (here a dedicated research custom
    model), so a project can serve research from a different endpoint than
    coding. No env override — the env layer would win for all roles."""
    from drcode.providers import AgentRole, CustomModelEntry, set_project_custom_model

    state = ScriptedState()
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())
    seen: list[tuple[str, str, str]] = []
    real_build_agent = pipeline_module.build_agent

    def spy_build_agent(config: Any, *, kind: Any = TaskKind.FEATURE, root: Any = None):
        seen.append((kind.value, config.model, config.base_url))
        return real_build_agent(config, kind=kind, root=root)

    monkeypatch.setattr(pipeline_module, "build_agent", spy_build_agent)
    set_project_custom_model(
        project_root,
        AgentRole.RESEARCH,
        CustomModelEntry(
            model="research-m",
            base_url="http://research/v1",
            max_context_tokens=12345,
        ),
    )
    set_project_custom_model(
        project_root,
        AgentRole.CODING,
        CustomModelEntry(model="coding-m", base_url="http://coding/v1"),
    )
    app = DrCodeApp(project_root)
    # Sanity: the two roles resolve independently.
    assert app.resolved_config(AgentRole.RESEARCH).model == "research-m"
    assert app.resolved_config(AgentRole.CODING).model == "coding-m"

    task = app.manager.create("r", "look around", kind=TaskKind.RESEARCH)
    state.steps = [text_response("analysis done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED,
            pilot,
            what="research task completed",
        )

    assert seen == [("research", "research-m", "http://research/v1")]


# -- agent run: rate limiting ---------------------------------------------------------


async def test_rate_limited_run_retries_with_backoff_then_exhausts_budget(
    net_app: DrCodeApp, scripted: ScriptedState, instant_timers: list[float]
) -> None:
    task = net_app.manager.create("limited", "do work", kind=TaskKind.RESEARCH)
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
    retries = [t for t in texts if t.startswith("Network error (ModelHTTPError)")]
    assert [f"retry {i}/5" in t for i, t in enumerate(retries, 1)] == [True] * 5
    # Backoff schedule 5s doubling, capped at 60 (NETWORK_RETRIES = 5),
    # plus the failed-queue rescue window (the task is the only queued one
    # and is failed, so the deadlock escape arms its 60s backoff timer).
    assert instant_timers == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0]
    # The last failure fell through to the generic failure path.
    assert any(t.startswith("ModelHTTPError") for t in texts)
    assert task.unfinished
    assert task.id in net_app._agent_failed
    # The prompt was consumed into the transcript (not re-queued): the
    # retry resumes the attempt instead of re-sending the text.
    assert task.pending_input == []
    assert ends_with_open_request(task.messages)


# -- agent run: connection outage -------------------------------------------------------


async def test_connection_outage_retried_then_demoted_and_recovery_resumes_it(
    net_app: DrCodeApp, scripted: ScriptedState, instant_timers: list[float]
) -> None:
    task = net_app.manager.create("docs", "write the docs", kind=TaskKind.RESEARCH)
    # 5 in-place retries within the budget, then one more connection error
    # that exhausts it.
    scripted.steps = [api_error("Connection error.") for _ in range(6)]

    async with net_app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: (
                task.status is TaskStatus.QUEUED
                and task.unfinished
                # The failed-queue rescue fired too (the task is the only
                # queued one): it re-ran the attempt, which failed on the
                # now-empty script and was re-excluded with a backoff.
                and net_app._failed_queue_rounds == 1
            ),
            pilot,
            what="task demoted after connection-retry budget exhausted",
        )
        await pilot.pause()
        assert task.id in net_app._agent_failed
        texts = event_texts(task)
        retries = [t for t in texts if t.startswith("Network error (ModelAPIError)")]
        assert [f"retry {i}/5" in t for i, t in enumerate(retries, 1)] == [True] * 5
        assert instant_timers == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0]
        # The last failure fell through to the generic failure path.
        assert any(t.startswith("ModelAPIError") for t in texts)
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
    # Six failed attempts (each resumed without re-sending text), the
    # failed-queue rescue attempt (failed on the empty script) and the
    # successful recovery call.
    assert scripted.calls == ["write the docs"] * 8


# -- tester run: network failure -------------------------------------------------------


async def test_tester_network_failure_retried_then_verification_passes(
    net_app: DrCodeApp, scripted: ScriptedState, instant_timers: list[float]
) -> None:
    task = net_app.manager.create("feature", "implement x", kind=TaskKind.FEATURE)
    scripted.steps = [
        text_response("work done"),  # development run
        api_error("Connection error."),  # tester round 1 dies
        tool_response("set_verification", {"success": True}),  # resumed round
        text_response("all checks passed"),
    ]

    async with net_app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: any("tester retry" in t for t in event_texts(task)),
            pilot,
            what="tester network retry logged",
        )
        assert task.status is TaskStatus.TESTING
        # The round is retried in place (like 429), not via the generic
        # failure backoff.
        assert task.id not in net_app._tester_fail_count

        # Fast-forward the retry window (the timer itself was recorded, not
        # slept) and let the pipeline resume the round.
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
    # The failed round was resumed in place — one testing round total.
    assert len(task.tester_messages) == 1
    retries = [
        t
        for t in event_texts(task)
        if t.startswith("Network error (ModelAPIError): tester retry")
    ]
    assert [f"retry {i}/5" in t for i, t in enumerate(retries, 1)] == [True]
    assert instant_timers == [5.0]
    assert task.id not in net_app._tester_fail_count
    assert task.id not in net_app._tester_retry_at


# -- missing configuration ---------------------------------------------------------------


async def test_model_not_configured_keeps_prompt_pending(project_root: Path) -> None:
    """No model/key configured (conftest clears the env): the app opens the
    first-start setup dialog instead of starting the pipeline; skipping it
    (ESC) leaves the app usable, and the next advance stops at the guard —
    the prompt is kept for a later send."""
    from drcode.tui.setup_dialog import ModelSetupDialog

    app = DrCodeApp(project_root)
    task = app.manager.create("docs", "some prompt")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ModelSetupDialog)
        await pilot.press("escape")
        await pilot.pause()
        # The pipeline only advances once the dialog is saved/skipped and
        # the user creates/queues work: trigger it like submit_new_task does.
        app._advance()
        await wait_until(
            lambda: bool(event_texts(task)), pilot, what="guard event logged"
        )
    assert any("Model is not configured" in t for t in event_texts(task))
    assert task.status is TaskStatus.ACTIVE
    assert [e.text for e in task.pending_input] == ["some prompt"]
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

    task = app.manager.create("docs", "", kind=TaskKind.RESEARCH)
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

    monkeypatch.setattr(agent_model, "build_model", lambda cfg: FunctionModel(failing))
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
    assert last_context_tokens(task.messages) == 2008  # 2000 input + 8 output


async def test_compaction_success_replaces_history(
    net_app_small_ctx: DrCodeApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic_ai.models.function import FunctionModel

    state = ScriptedState()
    state.steps = [text_response("THE SUMMARY BODY")]
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: FunctionModel(state.fn))
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


# -- map-reduce compaction (history larger than the summarizer's window) ----


def _routing_summarizer(
    map_prompts: list[str], calls: dict[str, Any]
) -> Callable[[list, Any], Any]:
    """A FunctionModel routing map vs reduce compaction calls: map prompts
    (carrying a <conversation>) answer with a numbered partial, reduce
    prompts (carrying <partial-summaries>) with the merged final summary."""
    from tests.network_mocks import last_user_text

    def fn(messages: list, info: Any) -> Any:
        prompt = last_user_text(messages)
        if "<partial-summaries>" in prompt:
            calls["reduce"] += 1
            calls["last_reduce_prompt"] = prompt
            return text_response("FINAL MERGED SUMMARY")
        calls["map"] += 1
        map_prompts.append(prompt)
        return text_response(f"PARTIAL {calls['map']}")

    return fn


async def test_map_reduce_compaction_over_window(
    project_root: Path, configured_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transcript whose rendered history is several times the compaction
    role's window is still compacted into a proper structured summary: the
    history is chunked at conversation boundaries, each slice is summarized
    sequentially (each map call seeing the running partials as anchor), and
    a final reduce call merges the partials — with every extra model call
    recorded via _record_session_usage like the single-shot path."""
    monkeypatch.setenv("DRCODE_MAX_CONTEXT_TOKENS", "2000")
    from pydantic_ai.models.function import FunctionModel

    map_prompts: list[str] = []
    calls: dict[str, Any] = {"map": 0, "reduce": 0}
    monkeypatch.setattr(
        agent_model,
        "build_model",
        lambda cfg: FunctionModel(_routing_summarizer(map_prompts, calls)),
    )

    recorded: list[Any] = []
    orig_record = DrCodeApp._record_session_usage
    monkeypatch.setattr(
        DrCodeApp,
        "_record_session_usage",
        lambda self, usage: (recorded.append(usage), orig_record(self, usage)),
    )

    app = DrCodeApp(project_root)
    # Rendered history ~25k chars (~6k estimated tokens) vs a 2k-token
    # window: several times over, and past the compaction threshold (the
    # seeded response reports a 2k-token fill).
    task = _seed_transcript(app, "word " * 5_000)

    async with app.run_test() as pilot:
        await pilot.pause()
        await app._maybe_compact(task)
        await pilot.pause()

    assert any("map-reduce" in t for t in event_texts(task))
    assert any("Context compacted" in t for t in event_texts(task))
    assert len(task.messages) == 2
    assert is_summary_message(task.messages[1])
    assert "FINAL MERGED SUMMARY" in task.messages[1]["parts"][0]["content"]
    # The map step really chunked: several slices, each well below the
    # rendered history, and the reduce call merged them.
    assert calls["map"] >= 2
    assert calls["reduce"] == 1
    rendered_len = len("word " * 5_000)
    assert all(len(p) < rendered_len for p in map_prompts)
    # Each map call sees the running partial summaries as its anchor
    # (mirroring the previous-summary anchoring).
    assert "PARTIAL 1" in map_prompts[1]
    # Total extra calls bounded and all recorded into the session usage.
    assert len(recorded) == calls["map"] + calls["reduce"]
    assert "<partial-summaries>" in calls["last_reduce_prompt"]


async def test_reduce_partials_hierarchical_on_overflow(
    project_root: Path, configured_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partials whose accumulated prompt overflows the reduce call are
    batched and reduced hierarchically, then merged: bounded extra calls,
    still a structured summary."""
    monkeypatch.setenv("DRCODE_MAX_CONTEXT_TOKENS", "2000")
    from pydantic_ai.models.function import FunctionModel

    calls: dict[str, Any] = {"map": 0, "reduce": 0}
    monkeypatch.setattr(
        agent_model,
        "build_model",
        lambda cfg: FunctionModel(_routing_summarizer([], calls)),
    )
    app = DrCodeApp(project_root)
    partials = ["p" * 60_000 for _ in range(3)]
    async with app.run_test() as pilot:
        await pilot.pause()
        result = await app._reduce_partials(partials, "TEMPLATE", None)
    assert result == "FINAL MERGED SUMMARY"
    # 3 batch reduces (one gigantic partial per batch) + 1 final merge.
    assert calls["reduce"] == 4


async def test_reduce_partials_mechanical_trim_at_depth_cap(
    project_root: Path, configured_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the hierarchical-reduce depth cap the payload is mechanically
    trimmed per partial so the merge call always happens (the guaranteed
    floor)."""
    monkeypatch.setenv("DRCODE_MAX_CONTEXT_TOKENS", "2000")
    from pydantic_ai.models.function import FunctionModel

    calls: dict[str, Any] = {"map": 0, "reduce": 0}
    monkeypatch.setattr(
        agent_model,
        "build_model",
        lambda cfg: FunctionModel(_routing_summarizer([], calls)),
    )
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        result = await app._reduce_partials(
            ["p" * 60_000 for _ in range(2)], "TEMPLATE", None, depth=3
        )
    assert result == "FINAL MERGED SUMMARY"
    assert calls["reduce"] == 1
    assert "…[truncated]" in calls["last_reduce_prompt"]


# -- mid-run context compaction ------------------------------------------------------------


@pytest.fixture
def scripted_fills(monkeypatch: pytest.MonkeyPatch):
    """Script the context fill reported after each completed model request.

    The streamed FunctionModel mock recomputes usage from the message
    content (real usage never flows through it), so the fill the app reads
    live after every model request node is scripted instead: one value per
    model request, consumed in order."""

    def install(fills: list[int]) -> None:
        queue = iter(fills)
        monkeypatch.setattr(
            pipeline_module, "run_context_tokens", lambda run: next(queue, None)
        )

    return install


async def test_mid_run_compaction_pauses_and_resumes(
    net_app_small_ctx: DrCodeApp,
    scripted: ScriptedState,
    scripted_fills: Any,
    project_root: Path,
) -> None:
    """The fill crossing the threshold inside a single turn (one run makes
    many model requests) stops the run at the next tool boundary, compacts
    and resumes via a continuation turn — the work is not lost."""
    (project_root / "README.md").write_text("hello readme")
    task = net_app_small_ctx.manager.create("docs", "long task", kind=TaskKind.RESEARCH)
    scripted_fills([100, 900])
    scripted.steps = [
        # 100 of 1000 — below the 80% threshold, the run goes on.
        tool_response("read_file", {"path": "README.md"}, "c1"),
        # 900 of 1000 — past the threshold: the run must pause here.
        tool_response("read_file", {"path": "README.md"}, "c2"),
        text_response("THE SUMMARY BODY"),  # compaction summarizer
        text_response("final answer"),  # continuation turn
    ]

    async with net_app_small_ctx.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and not net_app_small_ctx.agent_running
            ),
            pilot,
            what="task completed after mid-run compaction",
        )

    texts = event_texts(task)
    assert any(t.startswith("Context fill 90% — pausing the run") for t in texts)
    assert any("Context compacted" in t for t in texts)
    # Transcript replaced: original prompt verbatim + one summary request
    # (+ the continuation's answer). pydantic-ai merges the two consecutive
    # requests of the compacted history into one, keeping both prompts.
    assert [m["kind"] for m in task.messages] == ["request", "response"]
    prompt_contents = [
        p["content"]
        for p in task.messages[0]["parts"]
        if p["part_kind"] == "user-prompt"
    ]
    assert prompt_contents[0] == "long task"
    assert any("[context summary]" in c for c in prompt_contents)
    assert any("THE SUMMARY BODY" in c for c in prompt_contents)
    # Four model round-trips: two development requests, the summarizer, the
    # continuation; the summarizer saw the rendered transcript, the
    # continuation started from the summary.
    assert len(scripted.calls) == 4
    assert "read_file" in scripted.calls[2]
    assert "[context summary]" in scripted.calls[3]
    assert not task.unfinished


async def test_mid_run_compaction_skipped_when_summarizer_keeps_failing(
    net_app_small_ctx: DrCodeApp,
    scripted: ScriptedState,
    scripted_fills: Any,
    project_root: Path,
) -> None:
    """A failing summarizer must not turn the mid-run gate into a
    break→continuation loop: after one failure the run continues
    uncompressed (turn-boundary compaction still retries next turn)."""
    (project_root / "README.md").write_text("hello readme")
    task = net_app_small_ctx.manager.create("docs", "long task", kind=TaskKind.RESEARCH)
    scripted_fills([100, 900, 950])
    scripted.steps = [
        tool_response("read_file", {"path": "README.md"}, "c1"),
        tool_response("read_file", {"path": "README.md"}, "c2"),
        api_error("Connection error."),  # compaction summarizer dies
        # The continuation runs past the threshold again — the gate must
        # stay silent now, or the task would never finish.
        tool_response("read_file", {"path": "README.md"}, "c3"),
        text_response("carried on"),
    ]

    async with net_app_small_ctx.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and not net_app_small_ctx.agent_running
            ),
            pilot,
            what="task completed despite failing compaction",
        )

    texts = event_texts(task)
    assert len([t for t in texts if t.startswith("compaction failed")]) == 1
    assert len([t for t in texts if "pausing the run" in t]) == 1
    assert not any("Context compacted" in t for t in texts)
    # Uncompressed history carried the run to the end.
    assert not any(is_summary_message(m) for m in task.messages)
    assert len(scripted.calls) == 5
    assert not task.unfinished


async def test_no_mid_run_pause_below_threshold(
    net_app_small_ctx: DrCodeApp,
    scripted: ScriptedState,
    scripted_fills: Any,
    project_root: Path,
) -> None:
    """A turn whose fill stays under the threshold runs to the end without
    a pause and without touching the summarizer."""
    (project_root / "README.md").write_text("hello readme")
    task = net_app_small_ctx.manager.create("docs", "long task", kind=TaskKind.RESEARCH)
    scripted_fills([500])
    scripted.steps = [
        tool_response("read_file", {"path": "README.md"}, "c1"),
        text_response("done"),
    ]

    async with net_app_small_ctx.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and not net_app_small_ctx.agent_running
            ),
            pilot,
            what="task completed without compaction",
        )

    texts = event_texts(task)
    assert not [t for t in texts if "Context fill" in t or "Context compacted" in t]
    assert not any(is_summary_message(m) for m in task.messages)
    assert len(scripted.calls) == 2
    assert not task.unfinished


# -- tester round context compaction --------------------------------------------------------


async def test_tester_round_compaction_replaces_round_transcript(
    net_app_small_ctx: DrCodeApp,
    scripted: ScriptedState,
    scripted_fills: Any,
    project_root: Path,
) -> None:
    """A testing round whose fill crosses the threshold mid-round is paused
    at the tool boundary, compacted and resumed as a continuation turn —
    the round still reaches set_verification instead of looping in
    context-overflow retries."""
    (project_root / "README.md").write_text("hello readme")
    task = net_app_small_ctx.manager.create("feature", "implement x")
    scripted_fills([100, 900])
    scripted.steps = [
        text_response("work done"),  # development run
        # Tester request 1 reports 900 of 1000 — past the threshold:
        # the round must pause after this tool round.
        tool_response("read_file", {"path": "README.md"}),
        text_response("THE SUMMARY BODY"),  # compaction summarizer
        tool_response("set_verification", {"success": True}),  # continuation
        text_response("all checks passed"),
    ]

    async with net_app_small_ctx.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and not net_app_small_ctx.tester_running
            ),
            pilot,
            what="tester round completed after mid-round compaction",
        )

    texts = event_texts(task)
    assert any(
        t.startswith("Context fill 90% — pausing the tester round") for t in texts
    )
    assert any("Context compacted: testing round summarized" in t for t in texts)
    assert task.verification is VerificationStatus.PASSED
    # Five model round-trips: development, tester request, summarizer,
    # continuation (set_verification), final summary. The summarizer saw
    # the rendered ROUND transcript, the continuation started from the
    # summary (plus the original verify prompt).
    assert len(scripted.calls) == 5
    # The summarizer's transcript is the paused round: prompt, tool call,
    # (the just-executed round's result is uncommitted at the pause point —
    # same price the agent's mid-run gate pays) and nothing else.
    assert "[tool call] read_file" in scripted.calls[2]
    assert scripted.calls[2].count("[user]") == 1
    assert "[context summary]" in scripted.calls[3]
    assert "THE SUMMARY BODY" in scripted.calls[3]
    # The round transcript was replaced in place: original verify prompt
    # verbatim + the summary (merged by pydantic-ai into one request).
    final_round = task.tester_messages[-1]
    prompts = [
        p["content"]
        for m in final_round
        for p in m["parts"]
        if p["part_kind"] == "user-prompt"
    ]
    assert prompts[0].startswith("Verify completed task #")
    assert any("[context summary]" in c and "THE SUMMARY BODY" in c for c in prompts)


async def test_tester_round_compaction_failure_is_not_fatal(
    net_app_small_ctx: DrCodeApp,
    scripted: ScriptedState,
    scripted_fills: Any,
    project_root: Path,
) -> None:
    """A failing summarizer must not kill the round: it resumes
    uncompressed as a continuation turn, the gate stays off for the task
    (no pause→resume loop) and the verification still lands."""
    (project_root / "README.md").write_text("hello readme")
    task = net_app_small_ctx.manager.create("feature", "implement x")
    scripted_fills([100, 900])
    scripted.steps = [
        text_response("work done"),
        tool_response("read_file", {"path": "README.md"}),  # fill 900 → pause
        api_error("Connection error."),  # compaction summarizer dies
        tool_response("set_verification", {"success": True}),
        text_response("all checks passed"),
    ]

    async with net_app_small_ctx.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and not net_app_small_ctx.tester_running
            ),
            pilot,
            what="tester round completed despite failing compaction",
        )

    texts = event_texts(task)
    assert len([t for t in texts if t.startswith("compaction failed")]) == 1
    assert len([t for t in texts if "pausing the tester round" in t]) == 1
    assert not any("Context compacted" in t for t in texts)
    assert task.verification is VerificationStatus.PASSED
    # The round went through uncompressed: no summary message anywhere.
    assert not any(is_summary_message(m) for m in task.messages)
    assert not any(is_summary_message(m) for rnd in task.tester_messages for m in rnd)
    assert len(scripted.calls) == 5


async def test_tester_round_compaction_never_touches_agent_transcript(
    net_app_small_ctx: DrCodeApp,
    scripted: ScriptedState,
    scripted_fills: Any,
    project_root: Path,
) -> None:
    """Compacting the testing round must leave task.messages — the coding
    agent's transcript — byte-for-byte intact (the two histories never
    mix)."""
    (project_root / "README.md").write_text("hello readme")
    task = net_app_small_ctx.manager.create("feature", "implement x")
    scripted_fills([100, 900])
    scripted.steps = [
        text_response("work done"),
        tool_response("read_file", {"path": "README.md"}),
        text_response("THE SUMMARY BODY"),
        tool_response("set_verification", {"success": True}),
        text_response("all checks passed"),
    ]

    async with net_app_small_ctx.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and not net_app_small_ctx.tester_running
            ),
            pilot,
            what="tester round completed after mid-round compaction",
        )

    assert any("Context compacted" in t for t in event_texts(task))
    # The agent transcript is exactly the development exchange (original
    # prompt + answer): no summary request, no tester tool calls leaked in.
    assert [m["kind"] for m in task.messages] == ["request", "response"]
    texts = [p["content"] for m in task.messages for p in m["parts"] if "content" in p]
    assert any(c == "implement x" for c in texts)
    assert any(c == "work done" for c in texts)
    assert not any("[context summary]" in c or "read_file" in c for c in texts)
    assert not any(is_summary_message(m) for m in task.messages)


async def test_second_mid_run_compaction_does_not_grow_summary(
    net_app_small_ctx: DrCodeApp,
    scripted: ScriptedState,
    scripted_fills: Any,
    project_root: Path,
) -> None:
    """Regression: pydantic-ai merges the synthetic summary request into the
    adjacent original-prompt request on resume, so the next compaction saw
    the whole merged request as an "original prompt" and re-appended another
    summary — the transcript grew by one summary per pause→resume cycle and
    a long run could never compact its way below the threshold. Two
    consecutive compactions must leave exactly ONE summary part, holding
    the LATEST summary."""
    (project_root / "README.md").write_text("hello readme")
    task = net_app_small_ctx.manager.create("docs", "long task", kind=TaskKind.RESEARCH)
    # The pause sanitizes away the trailing unanswered tool call, so the
    # second compaction must find fresh committed activity in `rest`: the
    # continuation runs one below-threshold round (fill 500) before the
    # fill crosses again.
    scripted_fills([100, 900, 500, 900])
    scripted.steps = [
        tool_response("read_file", {"path": "README.md"}, "c1"),
        tool_response("read_file", {"path": "README.md"}, "c2"),  # 900 → compact
        text_response("SUMMARY ONE"),
        tool_response("read_file", {"path": "README.md"}, "c3"),  # 500 → carry on
        tool_response("read_file", {"path": "README.md"}, "c4"),  # 900 → compact again
        text_response("SUMMARY TWO"),
        text_response("final answer"),
    ]

    async with net_app_small_ctx.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and not net_app_small_ctx.agent_running
            ),
            pilot,
            what="task completed after two mid-run compactions",
        )

    texts = event_texts(task)
    assert len([t for t in texts if "Context compacted" in t]) == 2
    assert len(scripted.calls) == 7
    prompt_contents = [
        p["content"]
        for m in task.messages
        for p in m["parts"]
        if p["part_kind"] == "user-prompt" and isinstance(p["content"], str)
    ]
    summaries = [c for c in prompt_contents if c.startswith("[context summary]")]
    assert len(summaries) == 1
    assert "SUMMARY TWO" in summaries[0]
    assert "SUMMARY ONE" not in summaries[0]
    # The stale summary reaches the summarizer only as <previous-summary>
    # (the anchored chain), never as [user] conversation content.
    second_summary_prompt = scripted.calls[5]
    assert (
        "<previous-summary>\nSUMMARY ONE\n</previous-summary>" in second_summary_prompt
    )
    assert "[user]\nSUMMARY ONE" not in second_summary_prompt


async def test_second_tester_round_compaction_does_not_grow_summary(
    net_app_small_ctx: DrCodeApp,
    scripted: ScriptedState,
    scripted_fills: Any,
    project_root: Path,
) -> None:
    """Regression mirror for the tester: two consecutive mid-round
    compactions of the SAME round must leave the round transcript with a
    single (latest) summary part, not an ever-growing chain — otherwise the
    round oscillates pause → compact → resume forever without a verdict."""
    (project_root / "README.md").write_text("hello readme")
    task = net_app_small_ctx.manager.create("feature", "implement x")
    scripted_fills([100, 900, 900])
    scripted.steps = [
        text_response("work done"),
        tool_response("read_file", {"path": "README.md"}),  # 900 → compact
        text_response("SUMMARY ONE"),
        tool_response("read_file", {"path": "README.md"}),  # 900 → compact again
        text_response("SUMMARY TWO"),
        tool_response("set_verification", {"success": True}),
        text_response("all checks passed"),
    ]

    async with net_app_small_ctx.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and not net_app_small_ctx.tester_running
            ),
            pilot,
            what="tester round completed after two mid-round compactions",
        )

    texts = event_texts(task)
    assert len([t for t in texts if "Context compacted: testing round" in t]) == 2
    assert task.verification is VerificationStatus.PASSED
    assert len(scripted.calls) == 7
    final_round = task.tester_messages[-1]
    prompt_contents = [
        p["content"]
        for m in final_round
        for p in m["parts"]
        if p["part_kind"] == "user-prompt" and isinstance(p["content"], str)
    ]
    summaries = [c for c in prompt_contents if c.startswith("[context summary]")]
    assert len(summaries) == 1
    assert "SUMMARY TWO" in summaries[0]
    assert "SUMMARY ONE" not in summaries[0]
    # Stale summary as <previous-summary> only, never as [user] content.
    assert "<previous-summary>\nSUMMARY ONE\n</previous-summary>" in scripted.calls[4]
    assert "[user]\nSUMMARY ONE" not in scripted.calls[4]


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

    # The probe lives in drcode.agent_model.probe_model since the model
    # layer was split out of `agent` — patch its client, not the app's.
    monkeypatch.setattr("drcode.agent_model.httpx.AsyncClient", PatchedAsyncClient)


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


# -- failed-queue deadlock escape (_promote_failed_queue) --------------------------


async def test_all_failed_queue_is_rescued_and_redeveloped(
    net_app: DrCodeApp, scripted: ScriptedState, instant_timers: list[float]
) -> None:
    """When every queued task has failed this session and nothing is
    active, the error exclusion is lifted from all of them (with a
    notification) and development resumes by the ordinary priority rules
    — no restart needed."""
    a = net_app.manager.create("a", "brief a", kind=TaskKind.RESEARCH)
    b = net_app.manager.create("b", "brief b", kind=TaskKind.RESEARCH)
    # Both tasks fail instantly and non-retryably (HTTP 500), exhausting
    # the queue; then the rescue round re-runs them successfully.
    scripted.steps = [
        http_error(500),
        http_error(500),
        text_response("done a"),
        text_response("done b"),
    ]
    rescued: list[int] = []
    monkeypatch_ref = net_app.notifier
    monkeypatch_ref.failed_tasks_requeued = lambda count: rescued.append(count)  # type: ignore[method-assign]

    async with net_app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: (
                a.status is TaskStatus.COMPLETED and b.status is TaskStatus.COMPLETED
            ),
            pilot,
            what="both rescued tasks completed",
        )
        # The backoff state was reset by the successful model calls: a
        # later single failure must be rescued immediately again.
        assert net_app._failed_queue_rounds == 0

    assert rescued == [2]
    assert not {a.id, b.id} & net_app._agent_failed
    rescue_events = [
        t for t in event_texts(a) + event_texts(b) if "exclusion lifted" in t
    ]
    assert len(rescue_events) == 2
    # Each brief was sent exactly once (the failure never re-queues text).
    assert scripted.calls == ["brief a", "brief b", "brief a", "brief b"]


async def test_repeated_failed_queue_rescue_backs_off(
    net_app: DrCodeApp,
    scripted: ScriptedState,
    instant_timers: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a rescue round whose tasks fail again, the next rescue waits
    out a doubled backoff (60s → 120s): a persistently broken model must
    not hot-loop the backlog."""
    a = net_app.manager.create("a", "brief a", kind=TaskKind.RESEARCH)
    b = net_app.manager.create("b", "brief b", kind=TaskKind.RESEARCH)
    # task1 fails, task2 fails, rescue round 1: both fail again — the
    # second rescue round must now wait out its backoff.
    scripted.steps = [http_error(500)] * 4
    rescued: list[int] = []
    net_app.notifier.failed_tasks_requeued = lambda count: rescued.append(count)  # type: ignore[method-assign]

    # Controllable clock for the pipeline's backoff windows (the stdlib
    # `time` module is replaced only in the pipeline's namespace, so real
    # timers and the test's own wait_until keep working).
    class _FakeTime:
        def __init__(self) -> None:
            self.now = 1000.0

        def monotonic(self) -> float:
            return self.now

    fake_time = _FakeTime()
    monkeypatch.setattr(pipeline_module, "time", fake_time)

    async with net_app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: (
                a.status is TaskStatus.QUEUED
                and b.status is TaskStatus.QUEUED
                and {a.id, b.id} <= net_app._agent_failed
                # Rescue round 1 fired (immediately) and failed again.
                and net_app._failed_queue_rounds == 1
            ),
            pilot,
            what="queue settled after the first rescue failed again",
        )
        await pilot.pause()
        assert rescued == [2]
        assert instant_timers == [60.0]
        assert a.unfinished and b.unfinished

        # The 60s window expires: the re-advance fires rescue round 2,
        # which succeeds — the queue drains without a restart.
        fake_time.now += 130.0
        scripted.steps = [text_response("done a"), text_response("done b")]
        net_app._advance()
        await wait_until(
            lambda: (
                a.status is TaskStatus.COMPLETED and b.status is TaskStatus.COMPLETED
            ),
            pilot,
            what="both tasks completed by the second rescue round",
        )

    assert rescued == [2, 2]
    # The second round was scheduled behind the doubled delay.
    assert instant_timers == [60.0, 120.0]
    # Successful model calls reset the rescue backoff.
    assert net_app._failed_queue_rounds == 0
