"""Tests for the product-goal registry (stage A) and the research agent's
planning layer on top of it (stage B).

Covers: the Goal model + GoalStore persistence (atomic per-goal JSON,
corrupted-file backup), the TaskManager integration (goal_id linking at
task creation, the completion service event "цель Y: n/m задач
завершено"), the research agent's goal tools (list/read/create/update +
service events, registration per kind), the compact dynamic-instructions
injection and the goal_id parameter of create_bug_report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from drcode import agent as agent_mod
from drcode.agent import AgentDeps
from drcode.config import Config
from drcode.goals import GoalStore
from drcode.models import Goal, GoalStatus, GoalType, Task, TaskKind, TaskStatus


@pytest.fixture
def fake_config() -> Config:
    return Config(project_root=Path.cwd(), model="test-model", api_key="sk-test")


def _tool_names(ag) -> set[str]:
    return {t.name for t in ag._function_toolset.tools.values()}


def _deps(manager, task: Task, root: Path) -> AgentDeps:
    return AgentDeps(
        root=root,
        manager=manager,
        task_id=task.id,
        scratch=agent_mod.tools.scratch_dir(task.id),
    )


def _ctx(deps: AgentDeps):
    from pydantic_ai.tools import RunContext

    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


def _research_task(manager, title: str = "research") -> Task:
    return manager.create(title, "brief", kind=TaskKind.RESEARCH)


# -- Goal model ---------------------------------------------------------------


def test_goal_model_defaults() -> None:
    g = Goal(title="Async core")
    assert g.type is GoalType.IDEA
    assert g.status is GoalStatus.ACTIVE
    assert g.parent_id is None
    assert g.linked_task_ids == []
    assert g.description == ""


def test_goal_summary_line_caps_description() -> None:
    g = Goal(title="Async core", description="x" * 500)
    line = g.summary_line(160)
    assert "Async core" in line
    assert len(line.split("— ", 1)[1]) == 160
    assert "—" not in g.summary_line()


# -- GoalStore persistence ------------------------------------------------------


def test_goal_store_persists_and_reloads(project_root: Path) -> None:
    store = GoalStore(project_root)
    parent = store.create("Milestone: async core", description="ship it")
    child = store.create("Sub: migrate storage", parent_id=parent.id)

    assert (project_root / ".drcode" / "goals" / f"{parent.id}.json").exists()
    reloaded = GoalStore(project_root)
    assert set(reloaded.goals) == {parent.id, child.id}
    assert reloaded.get(child.id).parent_id == parent.id
    assert reloaded.get(parent.id).description == "ship it"
    # Creation order is stable.
    assert [g.id for g in reloaded.load_all()] == [parent.id, child.id]


def test_goal_store_corrupted_file_backed_up_never_fatal(project_root: Path) -> None:
    store = GoalStore(project_root)
    good = store.create("fine")
    bad = project_root / ".drcode" / "goals" / "broken123.json"
    bad.write_text("{not json")

    reloaded = GoalStore(project_root)
    assert good.id in reloaded.goals
    assert not bad.exists()
    assert bad.with_suffix(".json.bak").exists()


def test_goal_store_create_validates_parent(project_root: Path) -> None:
    store = GoalStore(project_root)
    with pytest.raises(ValueError, match="unknown parent goal id"):
        store.create("orphan", parent_id="nope1234")


def test_goal_store_update_partial_fields(project_root: Path) -> None:
    store = GoalStore(project_root)
    goal = store.create("idea")
    updated = store.update(goal.id, title="real goal", type=GoalType.GOAL)
    assert updated.title == "real goal"
    assert updated.type is GoalType.GOAL
    assert updated.status is GoalStatus.ACTIVE
    assert store.get(goal.id).title == "real goal"


def test_goal_store_update_rejects_bad_fields_and_values(project_root: Path) -> None:
    store = GoalStore(project_root)
    goal = store.create("idea")
    with pytest.raises(ValueError, match="unknown goal fields"):
        store.update(goal.id, nonexistent=1)
    with pytest.raises(ValueError, match="invalid goal status"):
        store.update(goal.id, status="vanished")
    with pytest.raises(ValueError, match="unknown parent goal id"):
        store.update(goal.id, parent_id="nope1234")
    with pytest.raises(ValueError, match="descendant"):
        store.update(goal.id, parent_id=goal.id)


def test_goal_store_decomposition_is_acyclic(project_root: Path) -> None:
    store = GoalStore(project_root)
    parent = store.create("milestone")
    child = store.create("sub", parent_id=parent.id)
    # Making the parent a child of its own descendant would create a cycle.
    with pytest.raises(ValueError, match="descendant"):
        store.update(parent.id, parent_id=child.id)


def test_goal_store_active_filters_closed_goals(project_root: Path) -> None:
    store = GoalStore(project_root)
    a = store.create("active one")
    b = store.create("done one")
    c = store.create("archived one")
    store.update(b.id, status=GoalStatus.DONE)
    store.update(c.id, status=GoalStatus.ARCHIVED)
    assert [g.id for g in store.active()] == [a.id]


def test_goal_store_delete(project_root: Path) -> None:
    store = GoalStore(project_root)
    goal = store.create("temporary")
    store.delete(goal.id)
    assert goal.id not in store.goals
    assert not (project_root / ".drcode" / "goals" / f"{goal.id}.json").exists()


# -- TaskManager integration -----------------------------------------------------


def test_manager_create_links_task_to_goal(manager, project_root: Path) -> None:
    goal = manager.goals.create("Async core")
    task = manager.create("migrate storage", "do it", goal_id=goal.id)

    assert task.goal_id == goal.id
    assert goal.linked_task_ids == [task.id]
    # Persisted on both sides.
    reloaded = GoalStore(project_root)
    assert task.id in reloaded.get(goal.id).linked_task_ids
    loaded_task = json.loads(
        (project_root / ".drcode" / "tasks" / f"{task.id}.json").read_text()
    )
    assert loaded_task["goal_id"] == goal.id


def test_manager_create_rejects_unknown_goal(manager) -> None:
    with pytest.raises(ValueError, match="unknown goal id"):
        manager.create("t", "brief", goal_id="nope1234")
    assert not manager.tasks


def test_completion_appends_goal_progress_event(manager) -> None:
    goal = manager.goals.create("Async core")
    first = manager.create(
        "migrate storage", "do it", kind=TaskKind.RESEARCH, goal_id=goal.id
    )
    second = manager.create(
        "migrate agent", "do it", kind=TaskKind.RESEARCH, goal_id=goal.id
    )

    manager.mark_done(first.id)
    events = [e.text for e in manager.get(first.id).events if e.kind == "goals"]
    assert len(events) == 1
    assert "1/2 задач завершено" in events[0]
    assert goal.title_display in events[0]
    # The second task's own completion sees both done.
    manager.mark_done(second.id)
    events2 = [e.text for e in manager.get(second.id).events if e.kind == "goals"]
    assert len(events2) == 1
    assert "2/2 задач завершено" in events2[0]


def test_completion_without_goal_has_no_goal_event(manager) -> None:
    task = manager.create("plain", "brief", kind=TaskKind.RESEARCH)
    manager.mark_done(task.id)
    assert not [e for e in manager.get(task.id).events if e.kind == "goals"]


def test_completion_with_deleted_goal_is_skipped(manager) -> None:
    goal = manager.goals.create("to be deleted")
    task = manager.create("t", "brief", kind=TaskKind.RESEARCH, goal_id=goal.id)
    manager.goals.delete(goal.id)
    manager.mark_done(task.id)
    assert not [e for e in manager.get(task.id).events if e.kind == "goals"]


def test_requeued_completed_task_keeps_goal_link_and_counts_once(manager) -> None:
    goal = manager.goals.create("Async core")
    task = manager.create("migrate", "do it", kind=TaskKind.RESEARCH, goal_id=goal.id)
    manager.mark_done(task.id)
    # Re-queue via send (a completed task with input re-enters), complete again.
    manager.send(task.id, "one more thing")
    assert task.status is TaskStatus.QUEUED
    assert task.goal_id == goal.id
    assert goal.linked_task_ids == [task.id]
    manager.mark_done(task.id)
    events = [e.text for e in task.events if e.kind == "goals"]
    assert len(events) == 2
    assert "1/1 задач завершено" in events[-1]


# -- goal tools (agent layer) -----------------------------------------------------


def test_goal_tools_registered_only_for_research(fake_config: Config) -> None:
    from drcode.agent import build_agent, build_tester_agent

    names = _tool_names(build_agent(fake_config, kind=TaskKind.RESEARCH))
    assert {"list_goals", "read_goal", "create_goal", "update_goal"} <= names

    for ag in (
        build_agent(fake_config, kind=TaskKind.FEATURE),
        build_agent(fake_config, kind=TaskKind.BUG),
        build_tester_agent(fake_config),
    ):
        tool_names = _tool_names(ag)
        assert not tool_names & {
            "list_goals",
            "read_goal",
            "create_goal",
            "update_goal",
        }


def test_list_goals_and_read_goal(manager, project_root: Path) -> None:
    store = manager.goals
    parent = store.create("Milestone: async", description="the big rewrite")
    child = store.create("Sub: storage", parent_id=parent.id)
    done = store.create("shipped thing")
    store.update(done.id, status=GoalStatus.DONE)
    task = _research_task(manager)
    goal_task = manager.create("migrate", "do it", goal_id=parent.id)

    deps = _deps(manager, task, project_root)
    listing = agent_mod._list_goals(deps)
    assert parent.id in listing and "Milestone: async" in listing
    assert child.id in listing
    assert done.id not in listing  # done goals are hidden by default
    assert "include_done=True" in listing

    full = agent_mod._list_goals(deps, include_done=True)
    assert done.id in full

    details = agent_mod._read_goal(deps, parent.id)
    assert "the big rewrite" in details
    assert child.id in details  # sub-goals listed
    assert goal_task.id in details  # linked tasks with live status
    assert "queued" in details


def test_read_goal_unknown_id_raises(manager, project_root: Path) -> None:
    task = _research_task(manager)
    deps = _deps(manager, task, project_root)
    with pytest.raises(ValueError, match="unknown goal id"):
        agent_mod._read_goal(deps, "nope1234")


def test_create_goal_sets_type_and_logs_service_event(
    manager, project_root: Path
) -> None:
    task = _research_task(manager)
    deps = _deps(manager, task, project_root)
    result = agent_mod._create_goal(
        deps, "Milestone: async", description="rewrite", goal_type="milestone"
    )
    assert len(manager.goals.goals) == 1
    goal = next(iter(manager.goals.goals.values()))
    assert goal.type is GoalType.MILESTONE
    assert goal.description == "rewrite"
    assert f"#{goal.id}" in result
    # Never silent: the research task's chat carries the service event.
    events = [e.text for e in manager.get(task.id).events if e.kind == "goals"]
    assert events and "создана агентом" in events[0] and goal.id in events[0]


def test_create_goal_invalid_type_raises(manager, project_root: Path) -> None:
    task = _research_task(manager)
    deps = _deps(manager, task, project_root)
    with pytest.raises(ValueError, match="invalid goal type"):
        agent_mod._create_goal(deps, "t", goal_type="someday")
    assert not manager.goals.goals


def test_update_goal_partial_and_service_event(manager, project_root: Path) -> None:
    store = manager.goals
    parent = store.create("parent")
    goal = store.create("idea")
    task = _research_task(manager)
    deps = _deps(manager, task, project_root)

    agent_mod._update_goal(
        deps, goal.id, goal_type="goal", parent_id=parent.id, description="refined"
    )
    updated = store.get(goal.id)
    assert updated.type is GoalType.GOAL
    assert updated.parent_id == parent.id
    assert updated.description == "refined"
    assert updated.status is GoalStatus.ACTIVE  # untouched field
    events = [e.text for e in manager.get(task.id).events if e.kind == "goals"]
    assert events and "уточнена агентом" in events[0]
    assert "тип → goal" in events[0]
    assert f"#{parent.id}" in events[0]


def test_update_goal_status_done_and_detach(manager, project_root: Path) -> None:
    store = manager.goals
    parent = store.create("parent")
    goal = store.create("sub", parent_id=parent.id)
    task = _research_task(manager)
    deps = _deps(manager, task, project_root)

    agent_mod._update_goal(deps, goal.id, parent_id="")
    assert store.get(goal.id).parent_id is None
    agent_mod._update_goal(deps, goal.id, status="done")
    assert store.get(goal.id).status is GoalStatus.DONE
    assert goal.id not in [g.id for g in store.active()]


def test_update_goal_validates_everything(manager, project_root: Path) -> None:
    store = manager.goals
    parent = store.create("parent")
    child = store.create("sub", parent_id=parent.id)
    task = _research_task(manager)
    deps = _deps(manager, task, project_root)

    with pytest.raises(ValueError, match="unknown goal id"):
        agent_mod._update_goal(deps, "nope1234", title="x")
    with pytest.raises(ValueError, match="invalid goal status"):
        agent_mod._update_goal(deps, child.id, status="finished")
    with pytest.raises(ValueError, match="descendant"):
        agent_mod._update_goal(deps, parent.id, parent_id=child.id)
    with pytest.raises(ValueError, match="nothing to change"):
        agent_mod._update_goal(deps, child.id)


# -- dynamic instructions injection -----------------------------------------------


def test_goals_injection_research_only(manager, project_root: Path) -> None:
    goal = manager.goals.create("Async core", description="the direction")
    research = _research_task(manager)
    feature = manager.create("feature", "brief", kind=TaskKind.FEATURE)

    block = agent_mod.goals_dynamic_instructions(
        _ctx(_deps(manager, research, project_root))
    )
    assert block is not None
    assert goal.id in block and "Async core" in block
    assert "the direction" in block

    assert (
        agent_mod.goals_dynamic_instructions(
            _ctx(_deps(manager, feature, project_root))
        )
        is None
    )


def test_goals_injection_excludes_closed_and_empty(manager, project_root: Path) -> None:
    research = _research_task(manager)
    deps = _deps(manager, research, project_root)
    assert agent_mod.goals_dynamic_instructions(_ctx(deps)) is None

    goal = manager.goals.create("done already")
    manager.goals.update(goal.id, status=GoalStatus.DONE)
    assert agent_mod.goals_dynamic_instructions(_ctx(deps)) is None


def test_goals_injection_truncates_long_description(
    manager, project_root: Path
) -> None:
    research = _research_task(manager)
    deps = _deps(manager, research, project_root)
    manager.goals.create("big", description="word " * 200)
    block = agent_mod.goals_dynamic_instructions(_ctx(deps))
    assert block is not None
    # Context discipline: the block stays compact, the tail is not included.
    assert "word word" in block
    assert len(block) < 1200


def test_goal_instructions_none_without_goals() -> None:
    assert agent_mod.goals_instructions([]) is None


def test_goal_instructions_lists_every_active_goal() -> None:
    block = agent_mod.goals_instructions(
        [
            Goal(title="First", description="d1"),
            Goal(title="Second", type=GoalType.MILESTONE, description="d2"),
        ]
    )
    assert block is not None
    assert "First" in block and "Second" in block
    assert "(milestone)" in block
    assert "read_goal" in block


# -- create_bug_report with goal_id -------------------------------------------------


def test_validated_goal_id_returns_none_or_id(manager, project_root: Path) -> None:
    from drcode.agent_build import _validated_goal_id

    goal = manager.goals.create("g")
    research = _research_task(manager)
    deps = _deps(manager, research, project_root)
    ctx = _ctx(deps)

    assert _validated_goal_id(ctx, None) is None
    assert _validated_goal_id(ctx, "  ") is None
    assert _validated_goal_id(ctx, f" {goal.id} ") == goal.id
    with pytest.raises(ModelRetry, match="unknown goal id"):
        _validated_goal_id(ctx, "nope1234")


class _CreateBugModel(TestModel):
    """TestModel variant simulating the research agent: ONE
    create_bug_report call with a goal_id, then a final answer."""

    def __init__(self, goal_id: str) -> None:
        super().__init__()
        self.goal_id = goal_id

    def _get_tool_calls(self, model_request_parameters):  # type: ignore[no-untyped-def]
        if not getattr(self, "_called", False):
            self._called = True
            defs = {t.name: t for t in model_request_parameters.function_tools}
            return [("create_bug_report", defs["create_bug_report"])]
        return []

    def gen_tool_args(self, tool_def):  # type: ignore[no-untyped-def]
        if tool_def.name == "create_bug_report":
            return {
                "title": "goal-driven bugfix",
                "description": "do it",
                "goal_id": self.goal_id,
            }
        return super().gen_tool_args(tool_def)


async def test_create_bug_report_links_goal(
    fake_config: Config, manager, project_root: Path
) -> None:
    from pydantic_ai import capture_run_messages

    from drcode.agent_build import build_agent

    goal = manager.goals.create("Async core")
    research = _research_task(manager)
    agent = build_agent(fake_config, kind=TaskKind.RESEARCH, root=project_root)
    deps = _deps(manager, research, project_root)
    before = set(manager.tasks)

    with capture_run_messages():
        async with agent.iter(
            "file the bug", model=_CreateBugModel(goal.id), deps=deps
        ) as run:
            async for _node in run:
                pass

    created = [t for t in manager.all_tasks() if t.id not in before]
    assert len(created) == 1
    assert created[0].goal_id == goal.id
    assert created[0].id in manager.goals.get(goal.id).linked_task_ids
    assert created[0].kind is TaskKind.BUG


def test_task_model_round_trip_keeps_goal_id() -> None:
    task = Task(title="t")
    assert task.model_dump()["goal_id"] is None
    task.goal_id = "ab12cd34"
    assert task.model_dump()["goal_id"] == "ab12cd34"
