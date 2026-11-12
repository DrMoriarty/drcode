"""Service events naming the model that serves an agent's run.

Chat-only bookkeeping (never the model context), three cases:

1. every run start that follows a STATUS transition logs
   ``Model: <key>`` (first activation, re-entry after a queue demotion)
   — the in-place resume of a network retry does NOT repeat the line
   (the task never changed status, the model is unchanged);
2. when the effective model of the serving role changed since the
   task's previous attempt, the run-start wording reports the switch
   (even on a retry resume — e.g. the user picked a different model
   while the retry was pending);
3. a manual model change while a run is in flight logs a
   ``Model changed: …`` event into the running task's chat, honestly
   saying the current run keeps the old model.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from drcode import agent_model
from drcode.models import TaskKind, TaskStatus
from drcode.providers import (
    AgentRole,
    CustomModelEntry,
    ModelEntry,
    ModelRef,
    Provider,
    UserModelsConfig,
    UserModelsStore,
    resolve_available,
    set_project_custom_model,
)
from drcode.tui.app import DrCodeApp
from tests.network_mocks import (
    ScriptedState,
    api_error,
    event_texts,
    text_response,
    wait_until,
)


@pytest.fixture
def catalog() -> None:
    """A user-level catalog with one provider and a coding default.

    No DRCODE_* env (the conftest sandbox strips it): the model
    resolution is driven by the catalog/project slots, so a test can
    change the effective model the way the Settings UI does."""
    UserModelsStore().save(
        UserModelsConfig(
            providers=[
                Provider(
                    name="local-swap",
                    base_url="http://localhost:8080/v1",
                    models=[ModelEntry(id="m-one"), ModelEntry(id="m-two")],
                )
            ],
            defaults={
                AgentRole.CODING: [ModelRef(provider="local-swap", model="m-one")]
            },
        )
    )


@pytest.fixture
def app(project_root: Path, catalog: None) -> DrCodeApp:
    return DrCodeApp(project_root)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedState:
    """Every agent drcode builds consumes `state.steps` (no network)."""
    state = ScriptedState()
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())
    return state


def _bare_active_task(app: DrCodeApp, title: str) -> Any:
    """A task marked ACTIVE with nothing queued: it never auto-runs, so
    a test can fake an in-flight run at the registry level (`_agent_runs`)
    and exercise the manual-change events deterministically."""
    task = app.manager.create(title, "prompt")
    task.pending_input.clear()
    task.status = TaskStatus.ACTIVE
    app.manager.storage.save(task)
    return task


async def test_run_start_event_and_switch_after_model_change(
    app: DrCodeApp, scripted: ScriptedState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run-start model event fires on the first run; the in-place
    network retry's re-entry (a run start like any other) reports the
    switch when the user changed the model while the retry was pending."""
    held: list[Callable[[], Any]] = []

    def hold_timers(delay: float, callback: Callable[[], Any]) -> None:
        held.append(callback)

    monkeypatch.setattr(app, "_schedule_pipeline_timer", hold_timers)
    task = app.manager.create("docs", "do things", kind=TaskKind.RESEARCH)
    scripted.steps = [api_error("Connection error."), text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: "Model: local-swap/m-one" in event_texts(task) and bool(held),
            pilot,
            what="run-start model event + the held retry timer",
        )
        assert task.status is TaskStatus.ACTIVE
        assert not any(
            "Model switched" in t or "Model changed" in t for t in event_texts(task)
        )
        # The user picks a different model while the retry is pending
        # (the same write the Settings pickers do)...
        set_project_custom_model(
            app.project_root,
            AgentRole.CODING,
            CustomModelEntry(model="m-two", base_url="http://localhost:8080/v1"),
        )
        app.invalidate_resolved_configs()
        # ...and the retry fires: the resumed run reports the switch
        # instead of a bare model line.
        for callback in held:
            callback()
            await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED,
            pilot,
            what="resumed run completed the research task",
        )
    texts = event_texts(task)
    assert "Model: local-swap/m-one" in texts
    assert "Model switched: local-swap/m-one → m-two" in texts


async def test_run_start_event_not_repeated_by_retry_resume(
    app: DrCodeApp, scripted: ScriptedState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run start on the same model logs the plain event once; the
    in-place network retry's re-entry into run_agent (no status change,
    same model) stays silent."""
    fired: list[Callable[[], Any]] = []

    def fire_later(delay: float, callback: Callable[[], Any]) -> None:
        fired.append(callback)
        app.call_later(callback)

    monkeypatch.setattr(app, "_schedule_pipeline_timer", fire_later)
    task = app.manager.create("docs", "first things", kind=TaskKind.RESEARCH)
    scripted.steps = [api_error("Connection error."), text_response("done")]
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED,
            pilot,
            what="research run completed after the in-place retry",
        )
    texts = event_texts(task)
    assert texts.count("Model: local-swap/m-one") == 1
    assert not any("Model switched" in t for t in texts)


async def test_manual_change_event_while_run_in_flight(
    app: DrCodeApp,
) -> None:
    """Changing the model through the Settings action while a task's run
    is in flight logs the change into THAT task's chat (and only there),
    saying the current run keeps the old model."""
    running = _bare_active_task(app, "docs")
    idle = _bare_active_task(app, "other")
    async with app.run_test() as pilot:
        await pilot.pause()
        # A live agent run, faked at the registry level: the event layer
        # only needs "this task has an in-flight run of the role".
        app._agent_runs[running.id] = 1
        app.action_set_role_model(
            AgentRole.CODING, ModelRef(provider="local-swap", model="m-two")
        )
        await pilot.pause()
    changed = [t for t in event_texts(running) if t.startswith("Model changed:")]
    assert len(changed) == 1
    assert changed[0].startswith("Model changed: local-swap/m-one → local-swap/m-two")
    assert "next run" in changed[0]
    # No live run for the idle task: no event there.
    assert not any(t.startswith("Model changed:") for t in event_texts(idle))


async def test_manual_change_no_event_when_key_unchanged(
    app: DrCodeApp,
) -> None:
    """Re-selecting the effectively-same model (here: the user default
    pinned as a project slot) is not a change — no event, no spam."""
    running = _bare_active_task(app, "docs")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._agent_runs[running.id] = 1
        app.action_set_role_model(
            AgentRole.CODING, ModelRef(provider="local-swap", model="m-one")
        )
        await pilot.pause()
    assert event_texts(running) == []


async def test_emit_run_model_event_wording(
    app: DrCodeApp,
) -> None:
    """Direct wording check of the run-start helper: plain line while the
    key stays, switch line after the effective key changes."""
    task = _bare_active_task(app, "docs")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._emit_run_model_event(task, AgentRole.CODING)
        app._emit_run_model_event(task, AgentRole.CODING)
        set_project_custom_model(
            app.project_root,
            AgentRole.CODING,
            CustomModelEntry(model="m-two", base_url="http://localhost:8080/v1"),
        )
        app.invalidate_resolved_configs()
        # The RESEARCH role inherits coding: switching the coding model
        # shows up in the research-served task's event too.
        app._emit_run_model_event(task, AgentRole.RESEARCH)
    texts = event_texts(task)
    assert texts.count("Model: local-swap/m-one") == 2
    assert "Model switched: local-swap/m-one → m-two" in texts


async def test_manual_change_through_custom_model_dialog(
    app: DrCodeApp,
) -> None:
    """The other manual-change call sites emit the same event: the
    free-form custom-model dialog callback (and the project model
    settings dialog callback) log it for in-flight runs."""
    from drcode.tui.setup_dialog import ModelSetup

    running = _bare_active_task(app, "docs")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._agent_runs[running.id] = 1
        app._custom_role_model_done(
            AgentRole.CODING,
            ModelSetup(
                model="m-two",
                base_url="http://localhost:8080/v1",
                api_key="k",
                max_context_tokens=200_000,
                max_tokens=8192,
            ),
        )
        await pilot.pause()
    changed = [t for t in event_texts(running) if t.startswith("Model changed:")]
    assert len(changed) == 1
    assert changed[0].startswith("Model changed: local-swap/m-one → m-two")
    assert "next run" in changed[0]


async def test_mid_run_model_change_keeps_serving_window_gates(
    project_root: Path,
) -> None:
    """Incident #cca920d5: a run serving on model A (window 1000) keeps A
    after the user switches the coding model to B (window 100000) mid-run.

    The run-start pin must survive `invalidate_resolved_configs()` for a
    role with a live run: the compaction gates measure through
    `resolved_config`, and a mid-run pin drop let them resolve B's huge
    window — both gates went silent at 124–129k of a 131k window and the
    next request died deterministically on A's context-overflow 400.
    """
    UserModelsStore().save(
        UserModelsConfig(
            providers=[
                Provider(
                    name="local-swap",
                    base_url="http://localhost:8080/v1",
                    models=[
                        ModelEntry(id="m-one", max_context_tokens=1000),
                        ModelEntry(id="m-big", max_context_tokens=100_000),
                    ],
                )
            ],
            defaults={
                AgentRole.CODING: [ModelRef(provider="local-swap", model="m-one")]
            },
        )
    )
    app = DrCodeApp(project_root)
    task = _bare_active_task(app, "docs")
    async with app.run_test() as pilot:
        await pilot.pause()
        # The run started on m-one: the factual run-start resolution is
        # pinned (the custom project slot path — no probing, like
        # _run_agent_worker does).
        set_project_custom_model(
            app.project_root,
            AgentRole.CODING,
            CustomModelEntry(
                model="m-one",
                base_url="http://localhost:8080/v1",
                max_context_tokens=1000,
            ),
        )
        app.invalidate_resolved_configs()
        resolution = await resolve_available(app.config, AgentRole.CODING)
        app._pin_resolution(AgentRole.CODING, resolution)
        app._agent_runs[task.id] = 1
        assert app.resolved_config(AgentRole.CODING).max_context_tokens == 1000

        # Fill 90% of A's window: the threshold gate fires...
        assert app._fill_over_threshold(900, [], AgentRole.CODING) == 900
        # ...and the hard-limit gate fires past 97%.
        assert app._fill_over_hard_limit(980, [], AgentRole.CODING) == 980

        # Mid-run the user switches the coding model to the big one (the
        # same write the Settings pickers do).
        set_project_custom_model(
            app.project_root,
            AgentRole.CODING,
            CustomModelEntry(
                model="m-big",
                base_url="http://localhost:8080/v1",
                max_context_tokens=100_000,
            ),
        )
        app.invalidate_resolved_configs()
        # The pin SURVIVES for the live role: the gates keep measuring
        # against the serving model's window and still fire.
        assert app.resolved_config(AgentRole.CODING).max_context_tokens == 1000
        assert app._fill_over_threshold(900, [], AgentRole.CODING) == 900
        assert app._fill_over_hard_limit(980, [], AgentRole.CODING) == 980

        # The change is still announced — naming the model the NEXT run
        # picks up (resolved fresh, bypassing the surviving pin).
        app._emit_manual_model_change_events(app._snapshot_model_keys())
        changed = [t for t in event_texts(task) if t.startswith("Model changed:")]
        assert len(changed) == 1
        assert changed[0].startswith("Model changed: m-one → m-big")

        # Run over: with no live run of the role left, the stale pin is
        # dropped — between runs the resolution reflects the current
        # configuration again.
        app._agent_runs.pop(task.id)
        app._drop_run_pin_if_idle(AgentRole.CODING)
        assert app.resolved_config(AgentRole.CODING).max_context_tokens == 100_000
