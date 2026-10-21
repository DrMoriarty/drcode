"""Lost-update race between tool calls of a single model response.

pydantic-ai executes the tool calls of one model response in parallel, and
drcode runs each tool in a worker thread — two concurrent `edit_file` calls
to the same file both read the original snapshot and the later write
silently overwrote the earlier edit (task #5e16576b: pairs of edits reported
success while one of them vanished). The mutating file tools are therefore
registered as `sequential=True` barriers.

The test drives the real agent with a scripted model that answers with a
pair of `edit_file` calls in a single response; the patched tool widens the
race window so the parallel execution would deterministically lose an edit
without the barrier.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from drcode import agent as agent_mod
from drcode import agent_model, tools
from drcode.agent import AgentDeps
from drcode.config import Config
from drcode.models import TaskKind
from drcode.task_manager import TaskManager


@pytest.fixture
def scripted(monkeypatch):
    from tests.network_mocks import ScriptedState

    state = ScriptedState()
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())
    return state


@pytest.fixture
def racy_edit(monkeypatch):
    """Patch `tools.edit_file` with a read-modify-write that parks after the
    read: under parallel execution both calls of the pair read the same
    snapshot; with the sequential barrier the calls never overlap."""
    from drcode.tools import resolve_edit_path

    def slow_edit(root: Path, path: str, old_string: str, new_string: str) -> str:
        p = resolve_edit_path(root, path, "edit")
        text = p.read_text(errors="replace")
        time.sleep(0.3)
        p.write_text(text.replace(old_string, new_string, 1))
        return f"edited {path}"

    monkeypatch.setattr(tools, "edit_file", slow_edit)


def two_edit_response() -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="edit_file",
                args={"path": "f.txt", "old_string": "aaa", "new_string": "bbb"},
                tool_call_id="c1",
            ),
            ToolCallPart(
                tool_name="edit_file",
                args={"path": "f.txt", "old_string": "xxx", "new_string": "yyy"},
                tool_call_id="c2",
            ),
        ]
    )


async def test_pair_of_edits_in_one_response_both_apply(
    project_root: Path, scripted, racy_edit
) -> None:
    (project_root / "f.txt").write_text("aaa\nxxx\n")
    config = Config(project_root=project_root, model="mock-model", api_key="sk-test")
    ag = agent_mod.build_agent(config, kind=TaskKind.RESEARCH, root=project_root)
    mgr = TaskManager(project_root)
    mgr.create("docs", "the task")
    task = mgr.all_tasks()[0]
    deps = AgentDeps(root=project_root, manager=mgr, task_id=task.id)

    scripted.steps = [two_edit_response(), ModelResponse(parts=[TextPart("done")])]
    async with ag.iter("the task", deps=deps) as run:
        async for _node in run:
            pass

    # Both tool calls succeeded...
    returns = [
        p.content
        for m in run.all_messages()
        for p in getattr(m, "parts", [])
        if p.part_kind == "tool-return"
    ]
    assert returns == ["edited f.txt", "edited f.txt"]
    # ...and both edits survived in the file.
    assert (project_root / "f.txt").read_text() == "bbb\nyyy\n"
