"""MCP client, stage 2: safety and budget wrappers (§3-§5 of
docs/mcp-client-design.md).

SafeToolset degrade + event dedup, the tool-list budget gate, the
tools_allow/tools_deny filtering (deny wins), the UNTRUSTED framing of
call results, the description warning and the tool_budget_tokens
setting. The real stdio-server lifecycle tests live in
test_mcp_stdio.py; everything here runs on synthetic toolsets only.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import (
    AbstractToolset,
    FilteredToolset,
    FunctionToolset,
    PrefixedToolset,
    PreparedToolset,
)
from pydantic_ai.usage import RunUsage

from drcode.mcp.client_config import (
    DEFAULT_TOOL_BUDGET_TOKENS,
    McpServerEntry,
    McpServersStore,
    user_servers_file,
)
from drcode.mcp.client_toolsets import (
    UNTRUSTED_DESCRIPTION_SUFFIX,
    UNTRUSTED_MCP_HEADER,
    SafeToolset,
    _allow_deny_filter,
    _build_toolset,
    _untrusted_prepare,
    estimate_toolset_cost,
    frame_mcp_output,
    mcp_toolsets_for,
)
from drcode.providers import AgentRole

_UNTRUSTED_START = "----- BEGIN UNTRUSTED MCP TOOL OUTPUT -----"
_UNTRUSTED_END = "----- END UNTRUSTED MCP TOOL OUTPUT -----"


# -- helpers ---------------------------------------------------------------


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def user_config(data: object) -> None:
    write_json(user_servers_file(), data)


def _ctx() -> RunContext[Any]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), retries={})


def _get_tools(toolset: Any) -> dict[str, Any]:
    async def run() -> dict[str, Any]:
        return await toolset.get_tools(_ctx())

    return asyncio.run(run())


def _call(safe: Any, name: str, args: dict[str, Any], tool: Any = None) -> str:
    """call_tool; when no explicit `tool` handle is given, fetch one via
    get_tools (FunctionToolset asserts on the real ToolsetTool)."""

    async def run() -> Any:
        handle = tool
        if handle is None:
            tools = await safe.get_tools(_ctx())
            handle = tools.get(name)  # None for stub wrappers
        return await safe.call_tool(name, args, _ctx(), handle)

    return asyncio.run(run())


def _entry(**extra: Any) -> McpServerEntry:
    return McpServerEntry.model_validate(
        {"id": "srv", "transport": "stdio", "command": "x", **extra}
    )


class _ExplodingToolset(AbstractToolset[Any]):
    """Wrapped toolset failing at the configured points."""

    @property
    def id(self) -> str:
        return "exploding"

    def __init__(
        self,
        get_error: Exception | None = None,
        call_error: Exception | None = None,
        result: Any = "ok",
    ) -> None:
        self.get_error = get_error
        self.call_error = call_error
        self.result = result

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, Any]:
        if self.get_error is not None:
            raise self.get_error
        return {}

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: Any
    ) -> Any:
        if self.call_error is not None:
            raise self.call_error
        return self.result


def _three_tool_toolset() -> FunctionToolset:
    def alpha() -> None:
        """Alpha."""

    def beta() -> None:
        """Beta."""

    def gamma() -> None:
        """Gamma."""

    return FunctionToolset([alpha, beta, gamma])


def _echo_toolset() -> FunctionToolset:
    def echo(text: str) -> str:
        """Echo the text back."""
        return f"echo:{text}"

    return FunctionToolset([echo])


# -- SafeToolset: degrade on get_tools failure (§5) -------------------------


def test_get_tools_failure_degrades_with_one_deduped_event() -> None:
    notes: list[str] = []
    safe = SafeToolset(
        _ExplodingToolset(get_error=RuntimeError("boom")),
        server_id="dead",
        notify=notes.append,
    )
    assert _get_tools(safe) == {}
    assert _get_tools(safe) == {}
    assert len(notes) == 1  # ONE event, deduplicated across calls
    assert "MCP server 'dead' unavailable" in notes[0]
    assert "boom" in notes[0]
    assert "tools not attached" in notes[0]


def test_get_tools_success_announces_nothing() -> None:
    notes: list[str] = []
    safe = SafeToolset(_echo_toolset(), server_id="srv", notify=notes.append)
    assert sorted(_get_tools(safe)) == ["echo"]
    assert notes == []


def test_call_failure_becomes_model_retry() -> None:
    notes: list[str] = []
    safe = SafeToolset(
        _ExplodingToolset(call_error=RuntimeError("kaboom")),
        server_id="srv",
        notify=notes.append,
    )
    with pytest.raises(ModelRetry, match="kaboom"):
        _call(safe, "echo", {})
    assert any("srv/echo failed" in n for n in notes)


def test_call_model_retry_passes_through_unwrapped() -> None:
    # The inner MCPToolset (tool_error_behavior="retry") already raises
    # ModelRetry for tool errors — SafeToolset must not double-wrap it.
    notes: list[str] = []
    safe = SafeToolset(
        _ExplodingToolset(call_error=ModelRetry("retry me")),
        server_id="srv",
        notify=notes.append,
    )
    with pytest.raises(ModelRetry, match="^retry me$"):
        _call(safe, "echo", {})
    assert any("failed" in n for n in notes)  # still announced


# -- SafeToolset: UNTRUSTED framing (§4.4) ----------------------------------


def test_call_result_is_framed_and_announced() -> None:
    notes: list[str] = []
    safe = SafeToolset(_echo_toolset(), server_id="srv", notify=notes.append)
    out = _call(safe, "echo", {"text": "hi"})
    assert UNTRUSTED_MCP_HEADER in out
    assert _UNTRUSTED_START in out and _UNTRUSTED_END in out
    assert "echo:hi" in out
    # §4.5: per-call event — server, tool, status, size.
    assert any("MCP call srv/echo ok" in n for n in notes)
    assert any("7 chars" in n for n in notes)


def test_structured_result_is_framed_too() -> None:
    def payload() -> dict[str, Any]:
        """Return structured data."""
        return {"a": 1, "nested": {"b": [1, 2]}}

    safe = SafeToolset(
        FunctionToolset([payload]), server_id="srv", notify=lambda text: None
    )
    out = _call(safe, "payload", {})
    assert _UNTRUSTED_START in out and '"a": 1' in out
    assert '"nested"' in out


def test_empty_result_framed_with_placeholder() -> None:
    def nothing() -> None:
        """Return nothing."""

    safe = SafeToolset(
        FunctionToolset([nothing]), server_id="srv", notify=lambda text: None
    )
    out = _call(safe, "nothing", {})
    assert "(empty result)" in out
    assert _UNTRUSTED_START in out


def test_frame_mcp_output_direct() -> None:
    out = frame_mcp_output("data")
    assert out.startswith(UNTRUSTED_MCP_HEADER)
    assert "----- BEGIN UNTRUSTED MCP TOOL OUTPUT -----\ndata" in out


# -- Budget gate (§3.3) ------------------------------------------------------


def test_budget_gate_blocks_oversized_tool_list_once() -> None:
    notes: list[str] = []

    def big() -> str:
        return "data"

    big.__doc__ = "x" * 8000  # ≈2000 "tokens" of description alone

    safe = SafeToolset(
        FunctionToolset([big]),
        server_id="fat",
        notify=notes.append,
        budget_tokens=100,
    )
    assert _get_tools(safe) == {}
    budget_notes = [n for n in notes if "budget" in n]
    assert len(budget_notes) == 1
    assert "MCP server 'fat' tool list ≈" in budget_notes[0]
    assert "tools not attached" in budget_notes[0]
    assert "tools_allow" in budget_notes[0]
    # Later calls stay degraded without re-announcing.
    assert _get_tools(safe) == {}
    assert len([n for n in notes if "budget" in n]) == 1


def test_under_budget_tool_list_passes() -> None:
    safe = SafeToolset(_echo_toolset(), server_id="srv", notify=lambda text: None)
    assert sorted(_get_tools(safe)) == ["echo"]


def test_estimate_toolset_cost_is_positive_and_small() -> None:
    tools = _get_tools(SafeToolset(_echo_toolset(), server_id="s"))
    cost = estimate_toolset_cost(tools.values())
    assert 0 < cost < 500  # a one-tool echo list must never approach 10k


def test_budget_setting_flows_from_config(tmp_path: Path) -> None:
    user_config(
        {
            "servers": [{"id": "ssh", "transport": "stdio", "command": "x"}],
            "roles": {"coding": ["ssh"]},
            "tool_budget_tokens": 777,
        }
    )
    toolsets = mcp_toolsets_for(AgentRole.CODING, project_root=tmp_path)
    assert len(toolsets) == 1
    assert toolsets[0].budget_tokens == 777


# -- tools_allow / tools_deny (§3.2): deny always wins -----------------------


def _filtered_names(server: McpServerEntry) -> list[str]:
    filtered = FilteredToolset(
        _three_tool_toolset(), filter_func=_allow_deny_filter(server)
    )
    return sorted(_get_tools(filtered))


def test_no_lists_admit_everything() -> None:
    assert _filtered_names(_entry()) == ["alpha", "beta", "gamma"]


def test_allow_list_restricts() -> None:
    assert _filtered_names(_entry(tools_allow=["alpha"])) == ["alpha"]


def test_deny_list_removes() -> None:
    assert _filtered_names(_entry(tools_deny=["beta"])) == ["alpha", "gamma"]


def test_deny_wins_over_allow() -> None:
    # web_fetch BLOCK-list semantics, mirrored per §3.2.
    assert _filtered_names(
        _entry(tools_allow=["alpha", "beta"], tools_deny=["beta"])
    ) == ["alpha"]
    assert _filtered_names(_entry(tools_allow=["alpha"], tools_deny=["alpha"])) == []


def test_filtered_names_are_bare_inside_the_prefix() -> None:
    # The filter sits INSIDE PrefixedToolset: config names are the server's
    # bare tool names, never the prefixed ones.
    server = _entry(tools_allow=["alpha"])
    filtered = FilteredToolset(
        _three_tool_toolset(), filter_func=_allow_deny_filter(server)
    )
    prefixed = PrefixedToolset(filtered, "srv")
    assert sorted(_get_tools(prefixed)) == ["srv_alpha"]


# -- Description warning (PreparedToolset, §4.4) -----------------------------


def test_description_warning_appended_once() -> None:
    prepared = PreparedToolset(_echo_toolset(), prepare_func=_untrusted_prepare)
    tools = _get_tools(prepared)
    assert UNTRUSTED_DESCRIPTION_SUFFIX in tools["echo"].tool_def.description
    # Idempotent: preparing an already-warned definition changes nothing.
    again = _untrusted_prepare(None, [tools["echo"].tool_def])[0]
    assert again.description == tools["echo"].tool_def.description


def test_empty_description_gets_the_warning() -> None:
    def nodoc() -> str:
        return ""

    prepared = PreparedToolset(
        FunctionToolset([nodoc]), prepare_func=_untrusted_prepare
    )
    tools = _get_tools(prepared)
    assert tools["nodoc"].tool_def.description == UNTRUSTED_DESCRIPTION_SUFFIX


# -- Full composition (§2.2, via a stub inner toolset) -----------------------


def test_build_toolset_composition_order_and_filters() -> None:
    class _FakeMCPToolset(FunctionToolset):
        # Accepts the MCPToolset constructor kwargs, never connects.
        def __init__(self, client: Any = None, **kwargs: Any) -> None:
            def alpha() -> None:
                """Alpha."""

            def beta() -> None:
                """Beta."""

            super().__init__([alpha, beta])

    notes: list[str] = []
    server = _entry(
        transport="http",
        url="http://127.0.0.1:9/mcp",
        tools_allow=["alpha"],
    )
    safe = _build_toolset(
        server, _FakeMCPToolset, None, None, {}, set(), notes.append, 1234
    )
    assert isinstance(safe, SafeToolset)
    assert safe.budget_tokens == 1234
    assert isinstance(safe.wrapped, PrefixedToolset)
    assert isinstance(safe.wrapped.wrapped, PreparedToolset)
    assert isinstance(safe.wrapped.wrapped.wrapped, FilteredToolset)
    assert isinstance(safe.wrapped.wrapped.wrapped.wrapped, _FakeMCPToolset)
    # Prefixed AND filtered: only the allowed tool reaches the model.
    assert sorted(_get_tools(safe)) == ["srv_alpha"]
    assert UNTRUSTED_DESCRIPTION_SUFFIX in (
        _get_tools(safe)["srv_alpha"].tool_def.description
    )


# -- tool_budget_tokens config tolerance & merge -----------------------------


def test_tool_budget_tokens_parsed_from_file(tmp_path: Path) -> None:
    path = tmp_path / "mcp-servers.json"
    write_json(path, {"servers": [], "roles": {}, "tool_budget_tokens": 25_000})
    parsed, _ = McpServersStore(user_path=path).load()
    assert parsed.tool_budget_tokens == 25_000


def test_tool_budget_tokens_junk_degrades_to_default(tmp_path: Path) -> None:
    path = tmp_path / "mcp-servers.json"
    for junk in ("junk", 0, -5, True, 1.5):
        write_json(path, {"tool_budget_tokens": junk})
        parsed, _ = McpServersStore(user_path=path).load()
        assert parsed.tool_budget_tokens == DEFAULT_TOOL_BUDGET_TOKENS


def test_tool_budget_tokens_project_layer_wins(tmp_path: Path) -> None:
    user = tmp_path / "user.json"
    project = tmp_path / "project.json"
    write_json(user, {"tool_budget_tokens": 25_000})
    write_json(project, {"tool_budget_tokens": 500})
    parsed, _ = McpServersStore(user_path=user, project_path=project).load()
    assert parsed.tool_budget_tokens == 500


def test_tool_budget_tokens_absent_project_keeps_user(tmp_path: Path) -> None:
    user = tmp_path / "user.json"
    project = tmp_path / "project.json"
    write_json(user, {"tool_budget_tokens": 25_000})
    write_json(project, {"servers": []})
    parsed, _ = McpServersStore(user_path=user, project_path=project).load()
    assert parsed.tool_budget_tokens == 25_000
