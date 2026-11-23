"""The agent-generated task titles (project setting ``generate_task_titles``).

Covered behaviour:

- flag off (the default) — nothing changes: the title stays the
  first-line derivation, no title request ever reaches the model;
- flag on — the title request runs BEFORE the main run (its model call is
  first), the generated name replaces the derived one, the stamp persists,
  and a re-activation does NOT re-generate;
- an explicit author title (a non-empty title at creation — the agents'
  report tools, the GitHub import, a goal's title) is NEVER regenerated;
  edit_task re-derives the title and makes it generable again;
- tasks the agent never sees (HOLD, DRAFT) keep the derived title;
- a failed generation is non-fatal: the old title stays, the run
  completes, a service event tells the user (and the stamp stays off, so
  the next activation may retry);
- the title call is served by the COMPACTION role's model (the same
  one-shot profile as the compaction summarizer — a cheap model, and the
  usage lands under compaction/<model>), falling back to the task's
  serving model only when no compaction model is selected anywhere
  (COMPACTION has no fallback of its own in providers.resolve_effective).

All model traffic goes through the in-process ScriptedState stub — no
network anywhere.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from drcode import agent_model, title_gen
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
    monkeypatch.setattr(
        agent_model, "build_model", lambda cfg, announce_wait=None: state.model()
    )
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
    task = TaskManager(project_root).create("", DESCRIPTION)
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
        assert live.title == DESCRIPTION
        assert not live.title_generated
        assert title_calls(scripted) == []


async def test_flag_on_generates_title_before_run(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
) -> None:
    enable_titles(project_root)
    # An empty title argument: the title is the derived first line —
    # exactly what the TUI "New task" form and the CLI produce.
    task = TaskManager(project_root).create("", DESCRIPTION)
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
    task = TaskManager(project_root).create("", DESCRIPTION)
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


async def test_explicit_title_is_never_regenerated(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
) -> None:
    """An explicit author title (non-empty title argument at creation —
    the path of the agents' report tools / GitHub import / goal titles)
    is never overwritten by the generator: no title call, the first model
    request is the main run's own."""
    enable_titles(project_root)
    task = TaskManager(project_root).create("Авторский заголовок", DESCRIPTION)
    assert task.title_explicit
    # No title step scripted: the dev run is the FIRST model call.
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
        assert live.title == "Авторский заголовок"
        assert not live.title_generated
        assert title_calls(scripted) == []
        assert scripted.calls[0] == DESCRIPTION


async def test_edit_task_rederived_title_is_generated(
    project_root: Path,
    configured_env: None,
    scripted: ScriptedState,
) -> None:
    """edit_task re-derives the title from the first line: an explicitly
    titled task becomes generable again after a rewrite."""
    enable_titles(project_root)
    manager = TaskManager(project_root)
    task = manager.create("Авторский заголовок", DESCRIPTION, hold=True)
    assert task.title_explicit
    manager.edit_task(task.id, "переписанное условие задачи")
    edited = manager.get(task.id)
    assert edited.title == "переписанное условие задачи"
    assert not edited.title_explicit
    scripted.steps = [
        text_response(GENERATED_TITLE),
        *DEV_AND_TESTER_STEPS,
    ]
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        live = app.manager.get(task.id)
        app.manager.set_hold(task.id, on=False)
        app._advance()
        await wait_until(
            lambda: live.status is TaskStatus.COMPLETED and live.title_generated,
            pilot,
            what="title generated and task completed",
        )
        await settle(pilot)
        assert live.title == GENERATED_TITLE
        assert len(title_calls(scripted)) == 1


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
    task = TaskManager(project_root).create("", DESCRIPTION)
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
        assert live.title == DESCRIPTION
        assert not live.title_generated
        assert any("Title generation failed" in e.text for e in live.events)


def write_custom_models(project_root: Path, customs: dict[str, dict[str, str]]) -> None:
    """Activate per-role custom models in the project's drcode.json (no
    env, no user catalog — the custom entries are the only connections)."""
    cfg_dir = project_root / ".drcode"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "drcode.json").write_text(
        json.dumps(
            {
                "models": {role: "custom" for role in customs},
                "custom_models": customs,
            }
        )
    )


@pytest.fixture
def title_spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture the config the pipeline hands to the title generator (the
    `drcode.title_gen.build_title_agent` seam — imported at call time by
    `_maybe_generate_title`, so the monkeypatch is honored) while keeping
    the real builder behavior."""
    seen: dict[str, Any] = {}
    real = title_gen.build_title_agent

    def spy(config: Any, announce_wait: Any = None) -> Any:
        seen["model"] = config.model
        seen["announce"] = announce_wait
        return real(config, announce_wait)

    monkeypatch.setattr(title_gen, "build_title_agent", spy)
    return seen


async def test_title_runs_on_compaction_model(
    project_root: Path,
    title_spy: dict[str, Any],
    scripted: ScriptedState,
) -> None:
    """With a compaction model selected, the namer runs on it — not on the
    task's coding model — and its usage is accounted under
    compaction/<model>."""
    enable_titles(project_root)
    write_custom_models(
        project_root,
        {
            "coding": {
                "model": "coding-m",
                "base_url": "http://coding.local/v1",
                "api_key": "k",
            },
            "compaction": {
                "model": "cheap-m",
                "base_url": "http://cheap.local/v1",
                "api_key": "k",
            },
        },
    )
    task = TaskManager(project_root).create("", DESCRIPTION)
    # Only the title step and one dev answer are needed — the assertions
    # fire as soon as the title is generated, before the tester round.
    scripted.steps = [text_response(GENERATED_TITLE), text_response("готово")]
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        live = app.manager.get(task.id)
        await wait_until(
            lambda: live.title_generated,
            pilot,
            what="title generated",
        )
        await settle(pilot)
        assert live.title == GENERATED_TITLE
        assert title_spy["model"] == "cheap-m"
        # The slot-wait announcer still rides along (provider cap
        # visibility), worded on the task's slot.
        assert callable(title_spy["announce"])
        # The title call's usage landed under the COMPACTION role.
        today = next(iter(app._usage_store.load().values()), {})
        assert "compaction/cheap-m" in today
        assert "coding/cheap-m" not in today


async def test_title_falls_back_to_task_model_without_compaction(
    project_root: Path,
    title_spy: dict[str, Any],
    scripted: ScriptedState,
) -> None:
    """No compaction model anywhere (env unset, no catalog, no slot): the
    COMPACTION role resolves not-ready and the namer keeps the historical
    behavior — the model serving the task."""
    enable_titles(project_root)
    write_custom_models(
        project_root,
        {
            "coding": {
                "model": "coding-m",
                "base_url": "http://coding.local/v1",
                "api_key": "k",
            },
        },
    )
    task = TaskManager(project_root).create("", DESCRIPTION)
    scripted.steps = [text_response(GENERATED_TITLE), text_response("готово")]
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        live = app.manager.get(task.id)
        await wait_until(
            lambda: live.title_generated,
            pilot,
            what="title generated",
        )
        await settle(pilot)
        assert live.title == GENERATED_TITLE
        assert title_spy["model"] == "coding-m"
