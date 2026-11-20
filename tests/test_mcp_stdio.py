"""MCP client, stage 2: lifecycle tests against a REAL stdio echo server
(§5 and §7 of docs/mcp-client-design.md).

The fixture (tests/fixtures/mcp_echo_server.py) runs on the mcp SDK —
fastmcp-slim (the client-side dependency) has no server support. The
tests cover: list_tools + framed call through the full SafeToolset
chain, the env whitelist (a host variable does NOT leak to the server),
a server dying at startup (skipped, event, run continues), a call on a
dead connection (ModelRetry), and the no-zombie invariant after a clean
run and after an abort mid-call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from drcode.mcp.client_toolsets import mcp_toolsets_for
from drcode.providers import AgentRole

pytest.importorskip("fastmcp")

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
_UNTRUSTED_START = "----- BEGIN UNTRUSTED MCP TOOL OUTPUT -----"
_UNTRUSTED_END = "----- END UNTRUSTED MCP TOOL OUTPUT -----"


# -- helpers ---------------------------------------------------------------


def write_user_config(data: object) -> None:
    from drcode.mcp.client_config import user_servers_file

    path = user_servers_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _server_record(tmp_path: Path, **extra: Any) -> tuple[dict[str, Any], Path]:
    """A stdio record for the echo fixture + the pidfile path it will write."""
    pidfile = tmp_path / "server.pid"
    record: dict[str, Any] = {
        "id": "echo",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(FIXTURE)],
        "env": {"ECHO_PIDFILE": str(pidfile)},
        "init_timeout": 30.0,
        "call_timeout": 15.0,
    }
    record.update(extra)
    return record, pidfile


def _safe_toolset(record: dict[str, Any], notes: list[str]) -> Any:
    write_user_config({"servers": [record], "roles": {"coding": [record["id"]]}})
    toolsets = mcp_toolsets_for(AgentRole.CODING, notify=notes.append)
    assert len(toolsets) == 1, f"no toolset built; notes: {notes}"
    return toolsets[0]


def _ctx() -> RunContext[Any]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), retries={})


def _get_tools(toolset: Any, notes: list[str] | None = None) -> dict[str, Any]:
    async def run() -> dict[str, Any]:
        return await toolset.get_tools(_ctx())

    tools = asyncio.run(run())
    if notes is not None:
        assert tools, f"no tools attached; notes: {notes}"
    return tools


def _in_session(safe: Any, body: Any) -> Any:
    """Run `body()` inside ONE held session on ONE event loop — the shape
    a real agent run has (pydantic-ai holds the toolsets in the run's
    AsyncExitStack). A pre-built fastmcp Client is bound to the loop that
    first opened it: a second asyncio.run() would fail with 'Event loop
    is closed', so list+call sequences must share the session."""

    async def run() -> Any:
        async with _innermost(safe):
            return await body()

    return asyncio.run(run())


def _await_pidfile(pidfile: Path, timeout: float = 20.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pidfile.exists():
            return int(pidfile.read_text().strip())
        time.sleep(0.1)
    pytest.fail(f"echo server never wrote its pidfile to {pidfile}")


def _wait_pid_gone(pid: int, timeout: float = 20.0) -> None:
    """The stdio child must be terminated AND reaped (no zombies)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    pytest.fail(f"stdio server process {pid} still alive after {timeout}s")


def _innermost(safe: Any) -> Any:
    """Safe → Prefixed → Prepared → Filtered → MCPToolset."""
    return safe.wrapped.wrapped.wrapped.wrapped


# -- clean lifecycle: list, call, terminate ---------------------------------


def test_stdio_list_call_and_clean_exit(tmp_path: Path) -> None:
    notes: list[str] = []
    record, pidfile = _server_record(tmp_path)
    safe = _safe_toolset(record, notes)

    async def body() -> str:
        tools = await safe.get_tools(_ctx())
        assert "echo_echo" in tools  # prefix = the server id
        # §4.4: the description warning rides on the prefixed definition.
        assert "untrusted" in tools["echo_echo"].tool_def.description
        return await safe.call_tool(
            "echo_echo", {"text": "hello"}, _ctx(), tools["echo_echo"]
        )

    out = _in_session(safe, body)
    assert "echo:hello" in out
    assert _UNTRUSTED_START in out and _UNTRUSTED_END in out
    # §4.5: per-call service event (server, tool, status, size).
    assert any("MCP call echo/echo_echo ok" in n for n in notes)

    # The session is closed → the child is gone, not a zombie.
    _wait_pid_gone(_await_pidfile(pidfile))


def test_env_whitelist_keeps_host_variables_out(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DRCODE_MCP_CANARY", "top-secret")
    record, _ = _server_record(tmp_path)
    record["env"]["ECHO_TOKEN"] = "s3cret"
    safe = _safe_toolset(record, [])

    # The record's env passes through; the host process's own variable does
    # NOT (§1.2: the SDK's whitelist-based default environment).
    async def body() -> tuple[str, str]:
        tools = await safe.get_tools(_ctx())
        canary = await safe.call_tool(
            "echo_env_probe",
            {"name": "DRCODE_MCP_CANARY"},
            _ctx(),
            tools["echo_env_probe"],
        )
        token = await safe.call_tool(
            "echo_env_probe", {"name": "ECHO_TOKEN"}, _ctx(), tools["echo_env_probe"]
        )
        return canary, token

    canary, token = _in_session(safe, body)
    assert "<absent>" in canary
    assert "s3cret" in token


def test_agent_run_framed_output_and_termination(tmp_path: Path) -> None:
    """The true §5 'ран' path: an Agent run enters/exits the toolset and
    the stdio child must not survive the run."""
    from pydantic_ai.messages import ToolReturnPart

    notes: list[str] = []
    record, pidfile = _server_record(tmp_path)
    safe = _safe_toolset(record, notes)

    agent = Agent(
        TestModel(call_tools=["echo_echo"], custom_output_text="done"),
        toolsets=[safe],
    )
    result = asyncio.run(agent.run("use the echo tool"))
    returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert returns, "the run never called the echo tool"
    assert any(
        _UNTRUSTED_START in str(content) and _UNTRUSTED_END in str(content)
        for content in returns
    )
    _wait_pid_gone(_await_pidfile(pidfile))


# -- server failure modes (§5/§7) --------------------------------------------


def test_stdio_server_dying_at_startup_is_skipped(tmp_path: Path) -> None:
    notes: list[str] = []
    safe = _safe_toolset(
        {
            "id": "dead",
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-c", "import sys; sys.exit(3)"],
            "init_timeout": 20.0,
        },
        notes,
    )
    assert _get_tools(safe) == {}  # empty set, never an exception
    assert len(notes) == 1
    assert "MCP server 'dead' unavailable" in notes[0]
    assert "tools not attached" in notes[0]


def test_http_server_on_dead_port_is_skipped() -> None:
    notes: list[str] = []
    safe = _safe_toolset(
        {
            "id": "void",
            "transport": "http",
            "url": "http://127.0.0.1:1/mcp",  # nothing listens on port 1
            "init_timeout": 10.0,
        },
        notes,
    )
    assert _get_tools(safe) == {}
    assert any("MCP server 'void' unavailable" in n for n in notes)


def test_call_on_dead_connection_is_model_retry(tmp_path: Path) -> None:
    notes: list[str] = []
    record, pidfile = _server_record(tmp_path)
    safe = _safe_toolset(record, notes)
    inner = _innermost(safe)

    async def scenario() -> int:
        async with inner:  # persistent session, the child is alive
            tools = await safe.get_tools(_ctx())
            tool = tools["echo_echo"]
            pid = _await_pidfile(pidfile)
            os.kill(pid, signal.SIGKILL)
            await asyncio.sleep(0.5)  # let the transport notice the EOF
            with pytest.raises(ModelRetry):
                await safe.call_tool("echo_echo", {"text": "x"}, _ctx(), tool)
            return pid

    pid = asyncio.run(scenario())
    # The killed child is reaped once the session unwinds.
    _wait_pid_gone(pid)
    assert any("failed" in n for n in notes)


def test_abort_mid_call_leaves_no_zombie(tmp_path: Path) -> None:
    """Cancelling the run (stall-nudge / ESC) cascades through the session:
    the stdio child is terminated and reaped (§5, "никаких зомби")."""
    record, pidfile = _server_record(tmp_path)
    safe = _safe_toolset(record, [])
    inner = _innermost(safe)

    async def hold_session() -> None:
        async with inner:
            await inner.direct_call_tool("slow", {"seconds": 60})

    async def scenario() -> int:
        task = asyncio.create_task(hold_session())
        deadline = time.monotonic() + 20
        while not pidfile.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        assert pidfile.exists(), "server never started"
        pid = int(pidfile.read_text().strip())
        await asyncio.sleep(1.0)  # the slow call is now in flight
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return pid

    pid = asyncio.run(scenario())
    _wait_pid_gone(pid)
