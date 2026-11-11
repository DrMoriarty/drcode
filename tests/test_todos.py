"""Tests for the feature-task todo checklist.

Covers the data model + persistence (todos ride with the task JSON), the
todo tools (registration per kind, degenerate-list guard, index/text
matching), the context injection (`_todo_instructions`), the completion
soft gate and the TUI panel rendering.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel

from drcode import agent as agent_mod
from drcode.agent import AgentDeps
from drcode.agent_model import TodoTailModel
from drcode.config import Config
from drcode.models import Task, TaskKind, TaskStatus, TodoItem
from drcode.storage import Storage
from drcode.tui.app import DrCodeApp
from drcode.tui.todo_panel import TodoPanel


@pytest.fixture
def fake_config() -> Config:
    return Config(project_root=Path.cwd(), model="test-model", api_key="sk-test")


def _tool_names(ag) -> set[str]:
    return {t.name for t in ag._function_toolset.tools.values()}


# -- tool registration per kind -------------------------------------------------


def test_todo_tools_only_for_feature(fake_config: Config) -> None:
    from drcode.agent import build_agent, build_tester_agent

    feature = build_agent(fake_config, kind=TaskKind.FEATURE)
    names = _tool_names(feature)
    assert {"todo_create", "todo_update"} <= names

    for ag in (
        build_agent(fake_config, kind=TaskKind.BUG),
        build_agent(fake_config, kind=TaskKind.RESEARCH),
        build_tester_agent(fake_config),
    ):
        tool_names = _tool_names(ag)
        assert "todo_create" not in tool_names
        assert "todo_update" not in tool_names


# -- todo_create ----------------------------------------------------------------


def _deps(manager, task: Task, root: Path) -> AgentDeps:
    return AgentDeps(
        root=root,
        manager=manager,
        task_id=task.id,
        scratch=agent_mod.tools.scratch_dir(task.id),
    )


def test_todo_create_persists_checklist(manager, project_root) -> None:
    from drcode.agent import _todo_create

    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    result = _todo_create(_deps(manager, task, project_root), ["a", "b", "c"])
    stored = manager.get(task.id)
    assert [t.text for t in stored.todos] == ["a", "b", "c"]
    assert all(not t.done for t in stored.todos)
    assert "3 items" in result
    # Persisted with the task, not only in memory.
    assert (
        json.loads(
            (project_root / ".drcode" / "tasks" / f"{task.id}.json").read_text()
        )["todos"][0]["text"]
        == "a"
    )


async def test_todo_create_rejects_degenerate_list(manager, tmp_path) -> None:
    """Fewer than 3 items is the one-item-list anti-pattern: the ValueError
    from the tool becomes a ModelRetry so the model corrects in-session."""
    from drcode.agent import _run_tool, _todo_create

    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    deps = _deps(manager, task, tmp_path)
    with pytest.raises(ModelRetry, match="do not use the checklist"):
        await _run_tool(_todo_create, deps, ["only one"])
    # Nothing was stored.
    assert manager.get(task.id).todos == []


def test_todo_create_replaces_existing_list(manager, tmp_path) -> None:
    from drcode.agent import _todo_create

    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    deps = _deps(manager, task, tmp_path)
    _todo_create(deps, ["a", "b", "c"])
    result = _todo_create(deps, ["x", "y", "z"])
    assert "replaced" in result.lower()
    assert [t.text for t in manager.get(task.id).todos] == ["x", "y", "z"]


# -- todo_update ----------------------------------------------------------------


def test_todo_update_by_index_and_text(manager, tmp_path) -> None:
    from drcode.agent import _todo_create, _todo_update

    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    deps = _deps(manager, task, tmp_path)
    _todo_create(deps, ["write tests", "add module", "run linters"])

    result = _todo_update(deps, [1, "lint"], done=True)
    todos = manager.get(task.id).todos
    assert todos[0].done and todos[2].done and not todos[1].done
    assert "2 item(s) marked done" in result


def test_todo_update_unknown_target_raises(manager, tmp_path) -> None:
    from drcode.agent import _todo_create, _todo_update

    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    deps = _deps(manager, task, tmp_path)
    _todo_create(deps, ["write tests", "add module", "run linters"])

    with pytest.raises(ValueError, match="out of range"):
        _todo_update(deps, [4], done=True)
    with pytest.raises(ValueError, match="No checklist item matches"):
        _todo_update(deps, ["nonexistent"], done=True)


def test_todo_update_without_checklist_raises(manager, tmp_path) -> None:
    from drcode.agent import _todo_update

    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    with pytest.raises(ValueError, match="no checklist"):
        _todo_update(_deps(manager, task, tmp_path), [1], done=True)


# -- data model / persistence ---------------------------------------------------


def test_todos_survive_storage_roundtrip(project_root: Path) -> None:
    storage = Storage(project_root)
    task = Task(title="t", todos=[TodoItem(text="a"), TodoItem(text="b", done=True)])
    storage.save(task)
    loaded = storage.load_one(project_root / ".drcode" / "tasks" / f"{task.id}.json")
    assert loaded is not None
    assert [(t.text, t.done) for t in loaded.todos] == [("a", False), ("b", True)]


def test_old_task_json_without_todos_loads(project_root: Path) -> None:
    """The field is a plain default: task JSONs written before the feature
    existed load without migration."""
    storage = Storage(project_root)
    task = Task(title="t")
    storage.save(task)
    path = project_root / ".drcode" / "tasks" / f"{task.id}.json"
    data = json.loads(path.read_text())
    data.pop("todos", None)
    path.write_text(json.dumps(data))
    loaded = storage.load_one(path)
    assert loaded is not None
    assert loaded.todos == []


def test_unchecked_todos_property() -> None:
    task = Task(
        title="t",
        todos=[
            TodoItem(text="a", done=True),
            TodoItem(text="b"),
        ],
    )
    assert [t.text for t in task.unchecked_todos] == ["b"]


def test_render_todos_numbered() -> None:
    from drcode.models import render_todos

    text = render_todos([TodoItem(text="a", done=True), TodoItem(text="b")])
    assert text == "1. [x] a\n2. [ ] b"


# -- tail injection (request tail, NOT leading instructions) ---------------------


class _CreateThenFinishModel(TestModel):
    """TestModel variant simulating the #fbb0d9fe scenario: the model
    calls todo_create on the FIRST request of a run, then just answers —
    all within ONE uninterrupted activation."""

    def _get_tool_calls(self, model_request_parameters):  # type: ignore[no-untyped-def]
        if not getattr(self, "_created", False):
            self._created = True
            defs = {t.name: t for t in model_request_parameters.function_tools}
            return [("todo_create", defs["todo_create"])]
        return []

    def gen_tool_args(self, tool_def):  # type: ignore[no-untyped-def]
        if tool_def.name == "todo_create":
            return {"items": ["explore code", "implement", "run tests"]}
        return super().gen_tool_args(tool_def)


class _CapturingModel(WrapperModel):
    """Records the message list every wrapped request actually receives —
    the wire boundary the TodoTailModel injection must show up at."""

    def __init__(self, wrapped):  # type: ignore[no-untyped-def]
        super().__init__(wrapped)
        self.seen: list[list] = []

    async def request(self, messages, model_settings, model_request_parameters):  # type: ignore[no-untyped-def]
        self.seen.append(list(messages))
        return await self.wrapped.request(
            messages, model_settings, model_request_parameters
        )

    @asynccontextmanager
    async def request_stream(  # type: ignore[no-untyped-def]
        self, messages, model_settings, model_request_parameters, run_context=None
    ):
        self.seen.append(list(messages))
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as response_stream:
            yield response_stream


def _tail_text(seen_request: list) -> str:
    """The injected tail: the LAST message of one recorded request must be
    a ModelRequest whose text parts carry the checklist block."""
    from pydantic_ai.messages import UserPromptPart

    last = seen_request[-1]
    assert isinstance(last, ModelRequest)
    return "\n".join(
        part.content
        for part in last.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


def _wire_supplier(manager, task_id: str):  # type: ignore[no-untyped-def]
    return lambda: agent_mod.todo_instructions(manager.get(task_id))


async def _run_once(
    ag,
    deps,
    inner,
    supplier,
    prompt: str = "do the work",
    *,
    stream_nodes: bool = False,
) -> None:
    """One agent turn through the wrapped model. With stream_nodes=True the
    model-request nodes are consumed via node.stream() — the same streaming
    path the pipeline uses, i.e. model.request_stream."""
    from pydantic_ai import Agent

    model = TodoTailModel(inner, supplier)
    async with ag.iter(prompt, model=model, deps=deps) as run:
        async for node in run:
            if stream_nodes and Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as node_stream:
                    async for _event in node_stream:
                        pass


async def test_todo_tail_reflects_state_within_activation(
    manager, fake_config: Config, project_root: Path
) -> None:
    """Regression (#fbb0d9fe / #a0a55d3d): a checklist created at the start
    of an activation must be visible on the LATER model requests of the
    SAME activation, and it must ride the request TAIL — as the LAST
    message — never the leading system zone, where a mutation would blow
    away the server-side prefix cache."""
    from drcode.agent import build_agent

    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    # The deps root is the agent's WRITABLE root: always a tmp dir — TestModel
    # auto-issues tool calls (write_file etc.) against it, a real cwd would
    # let them land in the checkout.
    deps = _deps(manager, task, project_root)
    inner = _CapturingModel(_CreateThenFinishModel())
    ag = build_agent(fake_config, kind=TaskKind.FEATURE)
    supplier = _wire_supplier(manager, task.id)

    await _run_once(ag, deps, inner, supplier)

    assert len(inner.seen) >= 2
    # Before todo_create: no checklist block in the request tail.
    assert "Task checklist (current state)" not in _tail_text(inner.seen[0])
    # After todo_create, same activation: the live state rides the LAST
    # message of the request.
    final_tail = _tail_text(inner.seen[-1])
    assert "# Task checklist (current state)" in final_tail
    assert "1. [ ] explore code" in final_tail
    assert "3 of 3 items are unchecked" in final_tail


async def test_todo_tail_absent_from_leading_zone(
    manager, fake_config: Config, project_root: Path
) -> None:
    """Structural cache guarantee: on the wire-level message list the block
    appears ONLY in the trailing message — every leading message (system /
    instructions / history) is clean."""
    from drcode.agent import build_agent

    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    # Tmp writable root: TestModel auto-issues tool calls against deps.root.
    deps = _deps(manager, task, project_root)
    agent_mod._todo_create(deps, ["write tests", "add module", "run linters"])
    inner = _CapturingModel(TestModel(custom_output_text="ok"))
    ag = build_agent(fake_config, kind=TaskKind.FEATURE)

    await _run_once(ag, deps, inner, _wire_supplier(manager, task.id))

    wire_request = inner.seen[0]
    for msg in wire_request[:-1]:
        assert "Task checklist (current state)" not in str(msg)
    assert "Task checklist (current state)" in str(wire_request[-1])


async def test_todo_tail_live_after_update_and_never_persisted(
    manager, fake_config: Config, project_root: Path
) -> None:
    """The supplier re-reads the task per REQUEST: a todo_update executed
    mid-activation is visible to the next request's tail; and the injected
    synthetic message never lands in the persisted transcript."""
    from drcode.agent import build_agent

    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    # Tmp writable root: TestModel auto-issues tool calls against deps.root.
    deps = _deps(manager, task, project_root)
    agent_mod._todo_create(deps, ["write tests", "add module", "run linters"])
    inner = _CapturingModel(TestModel(custom_output_text="ok"))
    ag = build_agent(fake_config, kind=TaskKind.FEATURE)
    supplier = _wire_supplier(manager, task.id)

    # One request with the initial state...
    await _run_once(ag, deps, inner, supplier)
    assert "1. [ ] write tests" in _tail_text(inner.seen[0])
    # ...then update the checklist and make ONE more request: the tail
    # carries the fresh state on the very next request.
    agent_mod._todo_update(deps, [1], done=True)
    inner.seen.clear()
    await _run_once(ag, deps, inner, supplier, prompt="continue")
    tail = _tail_text(inner.seen[-1])
    assert "1. [x] write tests" in tail
    assert "2 of 3 items are unchecked" in tail
    # The transcript stays clean: the tail never touches it.
    stored = manager.get(task.id).messages
    assert all("Task checklist (current state)" not in str(m) for m in stored)


async def test_todo_tail_model_semantics() -> None:
    """Wrapper-level contract, no agent involved: None supplier → the
    wrapped model gets EXACTLY the original list; a block → appended as
    one synthetic trailing ModelRequest, and the caller's list object is
    never mutated (the injection must not leak into the run state)."""
    from pydantic_ai.messages import (
        ModelMessage,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )
    from pydantic_ai.models import Model

    class _EchoModel(Model):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[ModelMessage] | None = None

        @property
        def model_name(self) -> str:
            return "echo"

        @property
        def system(self) -> str:
            return "test"

        async def request(  # type: ignore[no-untyped-def]
            self, messages, model_settings, model_request_parameters
        ):
            self.seen = list(messages)
            return ModelResponse(parts=[TextPart(content="ok")])

    echo = _EchoModel()
    model = TodoTailModel(echo, lambda: None)
    msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="hi")])]
    await model.request(msgs, None, None)  # type: ignore[arg-type]
    assert echo.seen == msgs

    echo2 = _EchoModel()
    model2 = TodoTailModel(echo2, lambda: "BLOCK")
    await model2.request(msgs, None, None)  # type: ignore[arg-type]
    assert echo2.seen is not None
    assert len(echo2.seen) == 2
    assert echo2.seen[0] is msgs[0]
    assert "BLOCK" in _tail_text(echo2.seen)
    # The caller's list was not mutated.
    assert len(msgs) == 1


def test_build_agent_wraps_model_only_with_supplier(fake_config: Config) -> None:
    """The tail wrapper is installed exactly when a supplier is wired, and
    stays a pass-through design elsewhere (tester/resolver/estimation)."""
    from drcode.agent import build_agent

    assert isinstance(
        build_agent(fake_config, todo_supplier=lambda: None).model, TodoTailModel
    )
    assert not isinstance(build_agent(fake_config).model, TodoTailModel)


async def test_todo_tail_streaming_request(
    manager, fake_config: Config, project_root: Path
) -> None:
    """The streaming path (the pipeline streams every node) injects the
    tail too: the wrapped model's request_stream sees the trailing block."""
    from drcode.agent import build_agent

    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    # Tmp writable root: TestModel auto-issues tool calls against deps.root.
    agent_mod._todo_create(
        _deps(manager, task, project_root), ["a first", "a second", "a third"]
    )
    inner = _CapturingModel(TestModel(custom_output_text="ok"))
    ag = build_agent(fake_config, kind=TaskKind.FEATURE)
    await _run_once(
        ag,
        _deps(manager, task, project_root),
        inner,
        _wire_supplier(manager, task.id),
        stream_nodes=True,
    )
    assert inner.seen
    assert "1. [ ] a first" in _tail_text(inner.seen[0])


def test_todo_create_result_reminds_to_update(manager, tmp_path) -> None:
    """The tool result itself carries the discipline reminder — the model
    sees it right when the list appears (its very first request already
    happened, so instructions alone can't reach that turn)."""
    from drcode.agent import _todo_create

    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    result = _todo_create(_deps(manager, task, tmp_path), ["a", "b", "c"])
    assert "todo_update immediately after" in result


def test_todo_instructions_feature_with_items() -> None:
    from drcode.tui.pipeline import PipelineLayer

    task = Task(
        title="t",
        kind=TaskKind.FEATURE,
        todos=[TodoItem(text="a", done=True), TodoItem(text="b")],
    )
    text = PipelineLayer._todo_instructions(task)
    assert text is not None
    assert "# Task checklist (current state)" in text
    assert "1. [x] a" in text
    assert "2. [ ] b" in text
    assert "1 of 2 items are unchecked" in text


def test_todo_instructions_absent_otherwise() -> None:
    from drcode.tui.pipeline import PipelineLayer

    assert PipelineLayer._todo_instructions(Task(title="t")) is None
    assert (
        PipelineLayer._todo_instructions(
            Task(title="t", kind=TaskKind.BUG, todos=[TodoItem(text="a")])
        )
        is None
    )
    assert (
        PipelineLayer._todo_instructions(
            Task(title="t", kind=TaskKind.FEATURE, todos=[])
        )
        is None
    )


# -- completion soft gate -------------------------------------------------------


def test_completion_warning_logs_unchecked(manager) -> None:
    mgr = manager

    class Host:
        viewed_id = None  # the task is not open in the chat
        manager = mgr
        _show_event = DrCodeApp._show_event
        _log_event = DrCodeApp._log_event
        _format_event = DrCodeApp._format_event

    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    task.status = TaskStatus.ACTIVE
    task.todos = [TodoItem(text="leftover"), TodoItem(text="done", done=True)]
    DrCodeApp._todo_completion_warning(Host(), task)  # type: ignore[arg-type]
    assert any(
        "1 unchecked todo item" in e.text and "leftover" in e.text
        for e in manager.get(task.id).events
    )


def test_completion_warning_silent_when_all_done(manager) -> None:
    mgr = manager

    class Host:
        viewed_id = None
        manager = mgr
        _show_event = DrCodeApp._show_event
        _log_event = DrCodeApp._log_event
        _format_event = DrCodeApp._format_event

    task = manager.create("feature", "brief", kind=TaskKind.FEATURE)
    task.todos = [TodoItem(text="a", done=True)]
    DrCodeApp._todo_completion_warning(Host(), task)  # type: ignore[arg-type]
    assert manager.get(task.id).events == []


# -- compaction template --------------------------------------------------------


def test_compaction_prompt_carries_checklist() -> None:
    from drcode.agent import build_compaction_prompt

    plain = build_compaction_prompt(None, "transcript", "template")
    assert "<task-checklist>" not in plain
    with_list = build_compaction_prompt(
        None, "transcript", "template", checklist="1. [ ] a"
    )
    assert "<task-checklist>\n1. [ ] a\n</task-checklist>" in with_list


def test_compaction_template_has_checklist_section() -> None:
    from drcode.prompts import load_compaction_prompts

    _, template = load_compaction_prompts()
    assert "## Task checklist" in template


# -- system prompt --------------------------------------------------------------


def test_feature_prompt_documents_todo_flow() -> None:
    from drcode.prompts import BUG_SYSTEM_PROMPT, FEATURE_SYSTEM_PROMPT

    assert "todo_create" in FEATURE_SYSTEM_PROMPT
    assert "todo_update" in FEATURE_SYSTEM_PROMPT
    # Bug tasks get no todo tools — the prompt must not encourage a list.
    assert "todo_" not in BUG_SYSTEM_PROMPT


# -- TUI panel ------------------------------------------------------------------


async def test_todo_panel_hidden_without_items(project_root: Path) -> None:
    app = DrCodeApp(project_root)
    task = app.manager.create("feature", "brief", kind=TaskKind.FEATURE)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        panel = app.query_one("#todo-panel", TodoPanel)
        assert panel.has_class("hidden")


async def test_todo_panel_renders_items(project_root: Path) -> None:
    app = DrCodeApp(project_root)
    task = app.manager.create("feature", "brief", kind=TaskKind.FEATURE)
    task.todos = [TodoItem(text="first"), TodoItem(text="second", done=True)]
    app.manager.storage.save(task)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        panel = app.query_one("#todo-panel", TodoPanel)
        assert not panel.has_class("hidden")
        text = str(panel.render())
        assert "[ ] 1. first" in text
        assert "[x] 2. second" in text


async def test_todo_panel_collapses_when_all_done(project_root: Path) -> None:
    app = DrCodeApp(project_root)
    task = app.manager.create("feature", "brief", kind=TaskKind.FEATURE)
    task.todos = [
        TodoItem(text="first", done=True),
        TodoItem(text="second", done=True),
    ]
    app.manager.storage.save(task)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        panel = app.query_one("#todo-panel", TodoPanel)
        assert not panel.has_class("hidden")
        text = str(panel.render())
        assert "todos" in text and "2/2" in text
        assert "first" not in text
        assert "second" not in text


def _panel_text(app: DrCodeApp) -> str:
    panel = app.query_one("#todo-panel", TodoPanel)
    return str(panel.render())


async def test_todo_panel_click_toggles_completed_list(project_root: Path) -> None:
    """A fully-completed checklist defaults to collapsed; a header click
    expands it, a second click collapses it back."""
    app = DrCodeApp(project_root)
    task = app.manager.create("feature", "brief", kind=TaskKind.FEATURE)
    task.todos = [
        TodoItem(text="first", done=True),
        TodoItem(text="second", done=True),
    ]
    app.manager.storage.save(task)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        # Default: collapsed to the counter line (unchanged behavior).
        assert "first" not in _panel_text(app)
        # Click → expand.
        app.action_toggle_todo_panel()
        await pilot.pause()
        text = _panel_text(app)
        assert "first" in text and "second" in text
        # Second click → collapse again.
        app.action_toggle_todo_panel()
        await pilot.pause()
        assert "first" not in _panel_text(app)


async def test_todo_panel_click_toggles_open_list(project_root: Path) -> None:
    """A checklist with unchecked items defaults to expanded; a header
    click collapses it to the counter line."""
    app = DrCodeApp(project_root)
    task = app.manager.create("feature", "brief", kind=TaskKind.FEATURE)
    task.todos = [TodoItem(text="first"), TodoItem(text="second", done=True)]
    app.manager.storage.save(task)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task.id)
        await pilot.pause()
        assert "first" in _panel_text(app)
        app.action_toggle_todo_panel()
        await pilot.pause()
        text = _panel_text(app)
        assert "todos" in text and "1/2" in text
        assert "first" not in text
        assert "second" not in text
        app.action_toggle_todo_panel()
        await pilot.pause()
        assert "first" in _panel_text(app)


async def test_todo_panel_override_resets_on_task_switch(project_root: Path) -> None:
    """The manual collapse override is per-view: switching to another task
    and back falls back to the default auto state."""
    app = DrCodeApp(project_root)
    task_a = app.manager.create("feature", "brief a", kind=TaskKind.FEATURE)
    task_a.todos = [TodoItem(text="alpha"), TodoItem(text="beta")]
    app.manager.storage.save(task_a)
    task_b = app.manager.create("feature", "brief b", kind=TaskKind.FEATURE)
    task_b.todos = [TodoItem(text="gamma")]
    app.manager.storage.save(task_b)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_task(task_a.id)
        await pilot.pause()
        app.action_toggle_todo_panel()
        await pilot.pause()
        assert "alpha" not in _panel_text(app)
        app.view_task(task_b.id)
        await pilot.pause()
        # Other task: default expanded, untouched by task A's override.
        assert "gamma" in _panel_text(app)
        app.view_task(task_a.id)
        await pilot.pause()
        # Override was dropped on the switch — back to the default.
        assert "alpha" in _panel_text(app)
