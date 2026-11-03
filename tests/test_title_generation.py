"""The agent-generated task titles (project setting ``generate_task_titles``).

Covered behaviour:

- flag off (the default) — nothing changes: the title stays the
  first-line derivation, no title request ever reaches the model;
- flag on — the title request runs BEFORE the main run (its model call is
  first), the generated name replaces the derived one, the stamp persists,
  and a re-activation does NOT re-generate;
- tasks the agent never sees (HOLD, DRAFT) keep the derived title;
- a failed generation is non-fatal: the old title stays, the run
  completes, a service event tells the user (and the stamp stays off, so
  the next activation may retry).

All model traffic goes through the in-process ScriptedState stub — no
network anywhere.
"""

from pathlib import Path
from typing import Any

import pytest

from drcode import agent_model
from drcode.models import TaskKind, TaskStatus
from drcode.settings import ProjectSettings, ProjectSettingsStore
from drcode.task_manager import TaskManager
from drcode.tui.app import DrCodeApp
from tests.network_mocks import (
    ScriptedState,
    api_error,
    text_response,
    tool_response,
    wait_until,
)

DESCRIPTION = "починить падение формы логина при пустом пароле"
GENERATED_TITLE = "Починить падение формы логина"

# The title request is distinguishable from every main-run request by its
# prompt marker (see drcode.title_gen.build_title_prompt).
TITLE_MARKER = "<task-description>"

# A full scripted cycle: [optional title call] + the dev run's answer +
# the tester's verdict round (set_verification tool call + its closing
# text response — the tool result needs one more model turn).
DEV_AND_TESTER_STEPS = [
    text_response("готово"),
    tool_response("set_verification", {"success": True}),
    text_response("Задача проверена."),
]


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedState:
    state = ScriptedState()
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())
    return state


def enable_titles(project_root: Path) -> None:
    ProjectSettingsStore(project_root).save(ProjectSettings(generate_task_titles=True))


def title_calls(state: ScriptedState) -> list[str]:
    return [c for c in state.calls if TITLE_MARKER in c]


async def settle(pilot: Any) -> None:
    """A few pilot pauses so pending UI updates (sidebar refresh, event
    lines) land before the assertions run."""
    for _ in range(5):
        await pilot.pause()


async def test_flag_off_keeps_derived_title(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
) -> None:
    task = TaskManager(project_root).create("first line", DESCRIPTION)
    scripted.steps = [*DEV_AND_TESTER_STEPS]
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        live = app.manager.get(task.id)
        await wait_until(
            lambda: live.status is TaskStatus.COMPLETED,
            pilot,
            what="task completed",
        )
        await settle(pilot)
        assert live.title == "first line"
        assert not live.title_generated
        assert title_calls(scripted) == []


async def test_flag_on_generates_title_before_run(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
) -> None:
    enable_titles(project_root)
    task = TaskManager(project_root).create("первая строка", DESCRIPTION)
    scripted.steps = [
        text_response(GENERATED_TITLE),
        *DEV_AND_TESTER_STEPS,
    ]
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        live = app.manager.get(task.id)
        await wait_until(
            lambda: live.status is TaskStatus.COMPLETED and live.title_generated,
            pilot,
            what="title generated and task completed",
        )
        await settle(pilot)
        assert live.title == GENERATED_TITLE
        # The title request came BEFORE the main run's own request: the
        # first model call is the namer (marker prompt), the second is the
        # dev run consuming the task description verbatim.
        assert len(title_calls(scripted)) == 1
        assert TITLE_MARKER in scripted.calls[0]
        assert DESCRIPTION in scripted.calls[0]
        assert scripted.calls[1] == DESCRIPTION
        # The generated name is persisted, not just in-memory.
        reloaded = TaskManager(project_root).get(task.id)
        assert reloaded.title == GENERATED_TITLE
        assert reloaded.title_generated


async def test_regeneration_never_repeats(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
) -> None:
    """A re-activation (new input on a completed task) does not re-generate:
    the stamp is set on the first success and generation is exactly-once."""
    enable_titles(project_root)
    task = TaskManager(project_root).create("первая строка", DESCRIPTION)
    # Both activations' steps up front: the trailing async tail of the
    # first tester round must never steal the second activation's steps.
    scripted.steps = [
        text_response(GENERATED_TITLE),
        *DEV_AND_TESTER_STEPS,
        *DEV_AND_TESTER_STEPS,
    ]
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        live = app.manager.get(task.id)
        await wait_until(
            lambda: live.status is TaskStatus.COMPLETED and live.title_generated,
            pilot,
            what="first activation done",
        )
        # Re-queue with a new comment: the second activation runs under the
        # same title setting, but the stamp must keep the generator away.
        app.manager.send(task.id, "дополнительный комментарий")
        await wait_until(
            lambda: (
                live.status is TaskStatus.COMPLETED
                and any("комментарий" in c for c in scripted.calls)
            ),
            pilot,
            what="second activation done",
        )
        await settle(pilot)
        assert len(title_calls(scripted)) == 1
        assert live.title == GENERATED_TITLE


async def test_hold_and_draft_keep_derived_title(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
) -> None:
    enable_titles(project_root)
    manager = TaskManager(project_root)
    held = manager.create("первая строка", DESCRIPTION, hold=True)
    draft = manager.create("черновик", DESCRIPTION, kind=TaskKind.BUG, draft=True)
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
        held_live = app.manager.get(held.id)
        draft_live = app.manager.get(draft.id)
        assert held_live.status is TaskStatus.HOLD
        assert draft_live.status is TaskStatus.DRAFT
        assert held_live.title == "первая строка"
        assert draft_live.title == "черновик"
        assert not held_live.title_generated
        assert not draft_live.title_generated
        assert scripted.calls == []


async def test_failed_generation_does_not_break_the_run(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
) -> None:
    enable_titles(project_root)
    task = TaskManager(project_root).create("первая строка", DESCRIPTION)
    # The title request fails (network/model error); the main run follows
    # and must complete regardless.
    scripted.steps = [
        api_error("connection refused"),
        *DEV_AND_TESTER_STEPS,
    ]
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        live = app.manager.get(task.id)
        await wait_until(
            lambda: live.status is TaskStatus.COMPLETED,
            pilot,
            what="task completed despite title failure",
        )
        await settle(pilot)
        assert live.title == "первая строка"
        assert not live.title_generated
        assert any("Title generation failed" in e.text for e in live.events)
