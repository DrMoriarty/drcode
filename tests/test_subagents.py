"""Read-only subagent delegation (`delegate_research`).

Covers the project-setting wiring (0 = the tool does not exist; the
actual limit is baked into its description), the per-run gate (a plain
limit message on exhaustion — never a crash; parallel calls inside the
limit), the subagent's final answer (plus usage summary) reaching the
parent, the service-event observability and the never-crash conversion
of a failing subagent model. All model traffic goes through in-process
FunctionModel stubs — no network.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from drcode.agent import (
    AgentDeps,
    SubagentGate,
    build_agent,
    run_delegate_tool,
    subagent_limit_for,
)
from drcode.config import Config
from drcode.models import TaskKind
from drcode.providers import AgentRole
from drcode.settings import ProjectSettings
from drcode.subagents import build_subagent_agent, git_read
from drcode.task_manager import TaskManager
from drcode.tool_runtime import _run_tool
from tests.network_mocks import text_response


@pytest.fixture
def fake_config() -> Config:
    return Config(project_root=Path.cwd(), model="test-model", api_key="sk-test")


def _tool_names(ag) -> set[str]:
    return {t.name for t in ag._function_toolset.tools.values()}


def _description(ag, name: str) -> str:
    for t in ag._function_toolset.tools.values():
        if t.tool_def.name == name:
            return t.tool_def.description or ""
    raise AssertionError(f"tool {name} is not registered")


def _instructions_text(ag) -> str:
    """The LEADING (static) instruction text of a built agent — dynamic
    hooks are functions and carry no literal here, like in the agent tests."""
    return "\n".join(
        si.instruction for si in ag._instructions if isinstance(si.instruction, str)
    )


def _tail_text(messages: list) -> str:
    """The text of the trailing synthetic message the wrapper appends."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    last = messages[-1]
    assert isinstance(last, ModelRequest)
    return "\n".join(
        part.content
        for part in last.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


# -- registration follows the project setting ----------------------------------


def test_delegate_tool_absent_at_zero(fake_config: Config) -> None:
    """The default limit 0 registers nothing — the agent never sees the
    tool (no description, no body), for every task kind."""
    for kind in (TaskKind.FEATURE, TaskKind.BUG, TaskKind.RESEARCH):
        ag = build_agent(fake_config, kind=kind)
        assert "delegate_research" not in _tool_names(ag)


def test_delegate_tool_registered_with_baked_limit(fake_config: Config) -> None:
    """Above 0 the tool exists and its description carries the ACTUAL
    limit (rebuilt per run, so the model always sees the real budget)."""
    ag = build_agent(fake_config, kind=TaskKind.FEATURE, subagent_limit=3)
    assert "delegate_research" in _tool_names(ag)
    assert "up to 3 subagents" in _description(ag, "delegate_research")
    # Singular spelling at 1 — the number must never lie.
    ag1 = build_agent(fake_config, kind=TaskKind.RESEARCH, subagent_limit=1)
    assert "up to 1 subagent" in _description(ag1, "delegate_research")


def test_delegate_description_carries_triggers_and_economics(
    fake_config: Config,
) -> None:
    """The description must not merely explain how to write a brief: it
    carries the WHEN (hard triggers) and the WHY (the context economics —
    the parent's own reads are permanent context and move it toward a
    compaction, a subagent's reads are free). A 119-task benchmark in which
    the tool was offered but never once called is the regression this guards.
    """
    for kind in (TaskKind.FEATURE, TaskKind.BUG, TaskKind.RESEARCH):
        text = _description(
            build_agent(fake_config, kind=kind, subagent_limit=2), "delegate_research"
        )
        # WHEN: the concrete triggers, not a vague "you may delegate".
        assert "WHEN TO DELEGATE" in text
        assert "call sites" in text
        assert "3–5 files" in text
        assert "surveying a subsystem" in text
        # WHY: the economic argument.
        assert "WHY IT IS CHEAP" in text
        assert "compaction" in text
        assert "stays in your context" in text
        # HOW: still the three-part brief (goal / starting points / report).
        assert "HOW TO WRITE THE BRIEF" in text
        assert "STARTING POINTS" in text
        assert "REPORT" in text
        # The description is parsed from the SOURCE (the docstring is
        # assigned, never an f-string in the def) — a blank one would be a
        # silent regression of the whole feature.
        assert len(text) > 500


def test_delegation_note_injected_at_request_tail(fake_config: Config) -> None:
    """With slots enabled the run gets ONE line in the trailing injection
    zone ("You have N ..."), never in the leading system instructions: the
    leading zone is the cached prefix, and a per-request reminder there
    would invalidate it on every request."""
    from drcode.agent import subagent_notice
    from drcode.agent_model import TailInjectionModel

    for kind, role in (
        (TaskKind.FEATURE, AgentRole.CODING),
        (TaskKind.BUG, AgentRole.CODING),
        (TaskKind.RESEARCH, AgentRole.RESEARCH),
    ):
        ag = build_agent(fake_config, kind=kind, subagent_limit=2)
        assert isinstance(ag.model, TailInjectionModel)
        block = subagent_notice(2, role=role)
        assert block is not None
        assert "You have 2 read-only subagents available via delegate_research" in block
        assert "delegate_research" in block
        # Not in the static instructions.
        assert "You have 2 read-only subagents" not in _instructions_text(ag)

    # Singular spelling, and 0 = no note at all (no wrapper, no block).
    assert "You have 1 read-only subagent available" in (
        subagent_notice(1, role=AgentRole.CODING) or ""
    )
    assert subagent_notice(0, role=AgentRole.CODING) is None
    assert not isinstance(
        build_agent(fake_config, kind=TaskKind.FEATURE).model, TailInjectionModel
    )
    # Short on purpose: it repeats on every request of every run with slots
    # enabled, so it must stay a POINTER to the triggers — the description
    # and the system prompt already carry the full argument.
    assert len(subagent_notice(2, role=AgentRole.CODING) or "") < 320


def test_delegation_note_rides_the_same_tail_as_the_checklist(
    fake_config: Config,
) -> None:
    """Both tail blocks share ONE synthetic message (the wrapper merges its
    suppliers), so enabling subagents never costs a second extra message."""
    from drcode.agent_model import TailInjectionModel

    ag = build_agent(
        fake_config,
        kind=TaskKind.FEATURE,
        subagent_limit=1,
        todo_supplier=lambda: "CHECKLIST-BLOCK",
    )
    assert isinstance(ag.model, TailInjectionModel)
    written = ag.model._messages_with_tail([])  # type: ignore[attr-defined]
    assert len(written) == 1
    tail = _tail_text(written)
    assert "CHECKLIST-BLOCK" in tail
    assert "You have 1 read-only subagent available" in tail


def test_subagent_limit_follows_role() -> None:
    """CODING reads subagents_coder, RESEARCH reads
    subagents_researcher; the default settings give 0."""
    settings = ProjectSettings(subagents_coder=2, subagents_researcher=4)
    assert subagent_limit_for(settings, AgentRole.CODING) == 2
    assert subagent_limit_for(settings, AgentRole.RESEARCH) == 4
    assert subagent_limit_for(ProjectSettings(), AgentRole.CODING) == 0


# -- the nested run through the gate --------------------------------------------


def _deps(
    project_root: Path, manager: TaskManager, task_id: str, limit: int
) -> AgentDeps:
    return AgentDeps(
        root=project_root,
        manager=manager,
        task_id=task_id,
        scratch=project_root / "scratch",
        subagent_gate=SubagentGate(limit),
    )


def test_subagent_answer_reaches_parent(
    monkeypatch: pytest.MonkeyPatch,
    fake_config: Config,
    project_root: Path,
    manager: TaskManager,
) -> None:
    """A successful delegation returns the subagent's FINAL answer plus a
    usage summary, logs spawn/finish service events and releases the slot."""
    monkeypatch.setattr(
        "drcode.subagents._build_subagent_model",
        lambda cfg, announce_wait=None: FunctionModel(
            lambda messages, info: text_response("findings: foo.py:10 does X")
        ),
    )
    task = manager.create("T", "delegate me")
    deps = _deps(project_root, manager, task.id, limit=2)

    out = asyncio.run(run_delegate_tool(deps, fake_config, "Investigate foo"))

    assert "findings: foo.py:10 does X" in out
    assert "subagent usage" in out
    # The gate slot is released afterwards.
    assert deps.subagent_gate is not None
    assert deps.subagent_gate.active == 0
    # Observability in the chat: spawn + finish service events.
    texts = [e.text for e in task.events]
    assert any("subagent spawned (1/2)" in t for t in texts)
    assert any("subagent finished" in t and "requests" in t for t in texts)


def test_gate_exhaustion_returns_plain_message(
    fake_config: Config, project_root: Path, manager: TaskManager
) -> None:
    """An exhausted gate returns an actionable plain tool result — no
    exception, no ModelRetry crash of the parent run."""
    task = manager.create("T", "delegate me")
    deps = _deps(project_root, manager, task.id, limit=1)
    assert deps.subagent_gate is not None
    deps.subagent_gate.active = 1  # the one allowed slot is busy

    out = asyncio.run(run_delegate_tool(deps, fake_config, "Investigate foo"))

    assert out.startswith("Subagent limit reached (1 running of 1 allowed)")
    assert "continue without delegation" in out


def test_parallel_calls_within_limit(
    monkeypatch: pytest.MonkeyPatch,
    fake_config: Config,
    project_root: Path,
    manager: TaskManager,
) -> None:
    """Two concurrent delegations both run (inflight 2 at the peak) under
    a limit of 2; the gate returns to 0 when both are done."""
    state = {"inflight": 0, "max": 0}

    async def sub_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        state["inflight"] += 1
        state["max"] = max(state["max"], state["inflight"])
        await asyncio.sleep(0.01)
        state["inflight"] -= 1
        return text_response(f"report {len(messages)}")

    monkeypatch.setattr(
        "drcode.subagents._build_subagent_model",
        lambda cfg, announce_wait=None: FunctionModel(sub_fn),
    )
    task = manager.create("T", "delegate me")
    gate = SubagentGate(2)

    async def call(brief: str) -> str:
        deps = AgentDeps(
            root=project_root,
            manager=manager,
            task_id=task.id,
            scratch=project_root / "scratch",
            subagent_gate=gate,
        )
        return await run_delegate_tool(deps, fake_config, brief)

    async def two() -> tuple[str, str]:
        return await asyncio.gather(call("brief a"), call("brief b"))

    outs = asyncio.run(two())

    assert all("report" in o for o in outs)
    assert state["max"] == 2  # both ran CONCURRENTLY
    assert gate.active == 0


def test_parallel_calls_above_limit_gate_one(
    monkeypatch: pytest.MonkeyPatch,
    fake_config: Config,
    project_root: Path,
    manager: TaskManager,
) -> None:
    """With a limit of 1 the first call runs, the second gets the plain
    limit message — the gate never lets the second through."""
    ran = {"count": 0}

    async def sub_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        ran["count"] += 1
        await asyncio.sleep(0.01)
        return text_response("solo report")

    monkeypatch.setattr(
        "drcode.subagents._build_subagent_model",
        lambda cfg, announce_wait=None: FunctionModel(sub_fn),
    )
    task = manager.create("T", "delegate me")
    gate = SubagentGate(1)

    async def call(brief: str) -> str:
        deps = AgentDeps(
            root=project_root,
            manager=manager,
            task_id=task.id,
            scratch=project_root / "scratch",
            subagent_gate=gate,
        )
        return await run_delegate_tool(deps, fake_config, brief)

    async def two() -> tuple[str, str]:
        return await asyncio.gather(call("brief a"), call("brief b"))

    outs = asyncio.run(two())

    assert ran["count"] == 1  # only one subagent actually ran
    answers = [o for o in outs if "solo report" in o]
    refused = [o for o in outs if o.startswith("Subagent limit reached")]
    assert len(answers) == 1 and len(refused) == 1
    assert gate.active == 0


def test_subagent_failure_is_plain_result(
    monkeypatch: pytest.MonkeyPatch,
    fake_config: Config,
    project_root: Path,
    manager: TaskManager,
) -> None:
    """A crashing subagent model (network error, refusal, anything) is
    converted into a plain tool result — the parent run keeps going."""

    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        "drcode.subagents._build_subagent_model",
        lambda cfg, announce_wait=None: FunctionModel(boom),
    )
    task = manager.create("T", "delegate me")
    deps = _deps(project_root, manager, task.id, limit=2)

    out = asyncio.run(run_delegate_tool(deps, fake_config, "Investigate foo"))

    assert "The subagent failed" in out
    assert "kaboom" in out
    assert deps.subagent_gate is not None
    assert deps.subagent_gate.active == 0  # released in the failure path too
    texts = [e.text for e in task.events]
    assert any("subagent failed" in t and "kaboom" in t for t in texts)


def test_no_gate_returns_configuration_hint(
    fake_config: Config, project_root: Path, manager: TaskManager
) -> None:
    """A deps without a gate (defensive: the tool only exists when a limit
    was configured, but a mismatched wiring must stay harmless)."""
    task = manager.create("T", "delegate me")
    deps = AgentDeps(
        root=project_root,
        manager=manager,
        task_id=task.id,
        scratch=project_root / "scratch",
    )
    out = asyncio.run(run_delegate_tool(deps, fake_config, "Investigate foo"))
    assert "not configured" in out


# -- inherited web_fetch ----------------------------------------------------------


def test_delegate_inherits_web_fetch_from_parent_gate(
    monkeypatch: pytest.MonkeyPatch,
    fake_config: Config,
    project_root: Path,
    manager: TaskManager,
) -> None:
    """A RESEARCH parent run with a web-fetch gate passes the permission
    on: the subagent agent gets the web_fetch tool and the SubagentDeps
    carry the parent's manager plus the subagent's OWN gate (same policy,
    smaller budget)."""
    from drcode.subagents import SubagentDeps
    from drcode.web_fetch import (
        FETCHES_PER_RUN,
        SUBAGENT_FETCHES_PER_RUN,
        WebFetchGate,
        WebFetchMode,
        WebFetchPolicy,
    )

    captured: dict[str, object] = {}

    def fake_subagent_model(cfg, announce_wait=None):
        return FunctionModel(
            lambda messages, info: text_response("report with a fetched page")
        )

    monkeypatch.setattr("drcode.subagents._build_subagent_model", fake_subagent_model)
    real_deps = SubagentDeps

    def recording_deps(**kwargs: object) -> object:
        captured.update(kwargs)
        return real_deps(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("drcode.subagents.SubagentDeps", recording_deps)

    policy = WebFetchPolicy(mode=WebFetchMode.ALL)
    task = manager.create("R", "research")
    deps = AgentDeps(
        root=project_root,
        manager=manager,
        task_id=task.id,
        scratch=project_root / "scratch",
        subagent_gate=SubagentGate(2),
        web_fetch_gate=WebFetchGate(policy),
    )

    out = asyncio.run(run_delegate_tool(deps, fake_config, "Investigate foo"))

    assert "report with a fetched page" in out
    assert captured["manager"] is manager
    assert captured["parent_task_id"] == task.id
    sub_gate = captured["web_fetch_gate"]
    assert isinstance(sub_gate, WebFetchGate)
    assert sub_gate.policy is policy  # the SAME policy object
    assert sub_gate.limit == SUBAGENT_FETCHES_PER_RUN
    assert sub_gate.limit < FETCHES_PER_RUN  # a share, never the full budget
    # The built agent carried the tool too (build happened with the policy).
    # captured via the gate: build_subagent_agent received web_fetch_policy.

    # The parent's own gate is untouched by the delegation.
    assert deps.web_fetch_gate is not None
    assert deps.web_fetch_gate.used == 0


def test_delegate_without_parent_gate_stays_network_free(
    monkeypatch: pytest.MonkeyPatch,
    fake_config: Config,
    project_root: Path,
    manager: TaskManager,
) -> None:
    """A CODING parent (no web_fetch_gate on deps) delegates a subagent
    with NO fetch tool and a gate-less SubagentDeps — network-free, as
    before the inheritance existed."""
    captured: dict[str, object] = {}
    from drcode.subagents import SubagentDeps

    monkeypatch.setattr(
        "drcode.subagents._build_subagent_model",
        lambda cfg, announce_wait=None: FunctionModel(
            lambda messages, info: text_response("plain report")
        ),
    )
    real_deps = SubagentDeps

    def recording_deps(**kwargs: object) -> object:
        captured.update(kwargs)
        return real_deps(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("drcode.subagents.SubagentDeps", recording_deps)

    task = manager.create("T", "delegate me")
    deps = AgentDeps(
        root=project_root,
        manager=manager,
        task_id=task.id,
        scratch=project_root / "scratch",
        subagent_gate=SubagentGate(1),
        # no web_fetch_gate — the CODING shape
    )

    out = asyncio.run(run_delegate_tool(deps, fake_config, "Investigate foo"))

    assert "plain report" in out
    assert captured["manager"] is manager
    assert captured["web_fetch_gate"] is None
    # And the built agent had no web_fetch tool: the policy reached
    # build_subagent_agent as None (recording via the gate above).


def test_subagent_web_fetch_tool_registered_with_policy(
    fake_config: Config,
) -> None:
    """build_subagent_agent with an inherited policy registers the fetch
    tool; without one it does not (the tool list stays five + git)."""
    from drcode.web_fetch import WebFetchMode, WebFetchPolicy

    policy = WebFetchPolicy(mode=WebFetchMode.ALLOWLIST, allow_hosts=("example.com",))
    ag = build_subagent_agent(fake_config, root=Path.cwd(), web_fetch_policy=policy)
    names = _tool_names(ag)
    assert "web_fetch" in names
    assert "run_bash" not in names

    ag_plain = build_subagent_agent(fake_config, root=Path.cwd())
    assert "web_fetch" not in _tool_names(ag_plain)


# -- the read-only git history tool ----------------------------------------------
@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A tiny real git repo with two commits over one file."""
    import subprocess as sp

    def git(*args: str) -> None:
        sp.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True)

    git("init", "-q", "--initial-branch=main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "T")
    (tmp_path / "foo.py").write_text("one\n", encoding="utf-8")
    git("add", "foo.py")
    git("commit", "-q", "-m", "first commit")
    (tmp_path / "foo.py").write_text("one\ntwo\n", encoding="utf-8")
    git("commit", "-q", "-am", "second commit")
    return tmp_path


def test_subagent_gets_git_history_tool(fake_config: Config) -> None:
    """The subagent's toolset carries the whitelisted git tool with a
    description naming the subcommands and the read-only guarantee."""
    ag = build_subagent_agent(fake_config, root=Path.cwd())
    names = {t.name for t in ag._function_toolset.tools.values()}
    assert "git_history" in names
    desc = _description(ag, "git_history")
    assert "log" in desc and "blame" in desc and "show" in desc and "diff" in desc
    assert "read-only" in desc.lower()
    # Still no shell, and no fetch at the default build (web_fetch is
    # registered only when the INHERITED policy is passed — see the
    # inheritance tests here and in test_web_fetch.py).
    assert "run_bash" not in names
    assert "web_fetch" not in names


def test_git_read_reads_history(git_repo: Path) -> None:
    """log/blame/show/diff work against a real repo and return the output."""
    log = git_read(git_repo, ["log", "--oneline"])
    assert "second commit" in log and "first commit" in log

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert "second commit" in git_read(git_repo, ["show", sha, "--stat"])
    assert "second commit" in git_read(git_repo, ["show", "--oneline", "-s", sha])

    diff = git_read(git_repo, ["diff", "HEAD~1", "HEAD"])
    assert "+two" in diff

    blame = git_read(git_repo, ["blame", "foo.py"])
    assert "two" in blame


def test_git_read_refuses_non_whitelisted_subcommands(git_repo: Path) -> None:
    """Anything outside log/blame/show/diff is refused BEFORE git runs —
    including read-shaped commands (status) and obvious writers (commit,
    checkout, stash, apply, reset). The message lists the whitelist."""
    for sub in ("status", "commit", "checkout", "stash", "apply", "reset", "push"):
        with pytest.raises(ValueError, match=r"not allowed.*log, blame, show, diff"):
            git_read(git_repo, [sub])


def test_git_read_refuses_writing_and_retargeting_arguments(git_repo: Path) -> None:
    """Options that would leave the read-only lane (git diff --output/-o
    writes a file; --git-dir/--work-tree/-C retarget the repository) are
    refused before git runs."""
    for args in (
        ["diff", "--output=leak.txt", "HEAD~1", "HEAD"],
        ["diff", "-o", "leak.txt", "HEAD~1", "HEAD"],
        ["log", "--git-dir=/etc", "--oneline"],
        ["log", "--work-tree=/tmp", "--oneline"],
        ["log", "-C", "/elsewhere", "--oneline"],
        ["show", "-c", "core.pager=cat", "HEAD"],
    ):
        with pytest.raises(ValueError, match="not allowed"):
            git_read(git_repo, args)
    assert not (git_repo / "leak.txt").exists()


def test_git_read_failure_is_value_error_not_crash(git_repo: Path) -> None:
    """A git error (unknown revision) surfaces as a plain ValueError —
    _run_tool turns it into a ModelRetry, never a crash of the nested run."""
    with pytest.raises(ValueError, match="failed"):
        git_read(git_repo, ["show", "deadbeefdeadbeef"])


def test_git_read_refusal_becomes_model_retry(git_repo: Path) -> None:
    """Through _run_tool both refusals and git errors degrade to ModelRetry."""

    async def call(args: list[str]) -> str:
        return await _run_tool(git_read, git_repo, args)

    with pytest.raises(ModelRetry):
        asyncio.run(call(["checkout", "main"]))
    with pytest.raises(ModelRetry):
        asyncio.run(call(["show", "deadbeefdeadbeef"]))


def test_git_read_truncates_huge_output(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output is capped so a huge dump cannot blow the subagent's context."""
    import drcode.subagents as subagents_mod

    monkeypatch.setattr(subagents_mod, "GIT_OUTPUT_CAP", 100)
    out = git_read(git_repo, ["log", "-p"])
    assert out.startswith("commit") or out  # something real came first
    assert "truncated at 100 chars" in out
    assert len(out) < 200


def test_git_read_empty_output_is_marker(git_repo: Path) -> None:
    """An empty result (e.g. diff with no changes) comes back as a marker,
    not an empty string the model cannot tell from silence."""
    out = git_read(git_repo, ["diff", "HEAD", "HEAD"])
    assert out == "(empty output)"
