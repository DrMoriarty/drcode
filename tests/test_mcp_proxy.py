"""The stdio shim (`drcode-mcp` / `python -m drcode.mcp_proxy`,
docs/mcp-server-design.md §1).

Three contracts, each at the cheapest level that proves it:

- **Discovery failure** (no `.drcode/mcp.json`) → the human-readable
  stderr hint and exit code 1, never a traceback and never an auto-start
  of a TUI.
- **Stale descriptor** (dead owning pid) → the same stderr contract:
  a crashed TUI must look exactly like a switched-off one to the client.
- **Round trip over a real stdio bridge**: the in-process server from
  the transport phase (an `McpService` on an ephemeral loopback socket,
  serving the read surface's tools) plus the shim as a REAL subprocess
  speaking the stdio framing — `initialize`, `notifications/initialized`,
  then a read-only `list_tasks` tool call whose answer crosses
  HTTP → shim → stdout and names a task created on the live manager.

The tests never leave loopback and never guess a port (the descriptor
is read back from `.drcode/mcp.json`, like a real client would).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from drcode.mcp import (
    McpEndpoint,
    McpService,
    endpoint_path,
    load_endpoint,
    save_endpoint,
)

#: The stderr hint of §1 — asserted verbatim: it is the documented
#: recovery path a confused external agent must be able to follow.
NO_SERVER_MESSAGE = (
    "drcode TUI не запущен или MCP выключен — включите в Settings → MCP server"
)

TOKEN = "fedcba9876543210" * 4


# -- fixtures ------------------------------------------------------------------


@pytest.fixture
def sdk_server():
    """An SDK server with one trivial tool (same shape as the transport tests)."""
    from drcode.mcp.app import build_server

    server = build_server()

    @server.tool()
    def ping() -> str:
        """Answer "pong"."""
        return "pong"

    return server


@pytest.fixture
def read_surface(manager, tmp_path: Path):
    """The full read surface over the tmp manager (the TUI's wiring)."""
    from drcode.mcp.wiring import McpSurface
    from drcode.usage_store import UsageStore

    return McpSurface(manager, usage_store=UsageStore(path=tmp_path / "usage.json"))


@pytest.fixture
def events() -> list[str]:
    return []


def _service(project_root: Path, events: list[str], factory) -> McpService:
    return McpService(project_root, event_sink=events.append, server_factory=factory)


class StdioSession:
    """Line-framed JSON-RPC conversation with the shim subprocess.

    stdout is drained by a daemon thread into a queue, so a read can
    time out instead of hanging the suite when the shim is broken.
    """

    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self.proc = proc
        self._lines: queue.Queue[str] = queue.Queue()
        self._drain = threading.Thread(target=self._drain_stdout, daemon=True)
        self._drain.start()

    def _drain_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._lines.put(line)

    def send(self, payload: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def recv(self, timeout: float = 15.0) -> dict:
        line = self._lines.get(timeout=timeout)
        return json.loads(line)

    def close(self, timeout: float = 10.0) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:  # pragma: no cover - hostile env
            self.proc.kill()
            self.proc.wait(timeout=5)

    def request(self, id_: int, method: str, params: dict | None = None) -> dict:
        self.send(
            {"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}}
        )
        return self.recv()

    def notify(self, method: str, params: dict | None = None) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params or {}})


def _spawn_shim(project_root: Path, *extra: str) -> subprocess.Popen[str]:
    """The shim exactly as an MCP stdio client would spawn it: cwd = project."""
    return subprocess.Popen(
        [sys.executable, "-m", "drcode.mcp_proxy", *extra],
        cwd=project_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


# -- discovery failures (§1: clean stderr + exit 1, never an auto-start) -------


def test_missing_descriptor_fails_with_the_hint(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No `.drcode/mcp.json` → the §1 hint on stderr, exit code 1."""
    from drcode.mcp.proxy import main

    monkeypatch.chdir(project_root)
    assert main([]) == 1
    err = capsys.readouterr().err
    assert NO_SERVER_MESSAGE in err
    assert "mcp.json" in err  # the file is named: the user can check it
    assert not endpoint_path(project_root).exists()  # nothing was fabricated


def test_dead_pid_fails_with_the_hint(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A descriptor left by a crashed TUI (dead pid) → the hint, exit 1."""
    from drcode.mcp.proxy import main

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.terminate()
    proc.wait(timeout=5)
    assert proc.pid > 0
    save_endpoint(project_root, McpEndpoint(port=1, token=TOKEN, pid=proc.pid))
    # The port is dead too (nothing listens on port 1), but the pid check
    # must fire first and name the real cause.
    monkeypatch.chdir(project_root)
    assert main([]) == 1
    err = capsys.readouterr().err
    assert NO_SERVER_MESSAGE in err
    assert str(proc.pid) in err


def test_explicit_port_with_a_dead_port_fails_with_the_hint(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--port` overrides the file, but a silent port is still exit 1."""
    from drcode.mcp.proxy import main

    save_endpoint(project_root, McpEndpoint(port=1, token=TOKEN, pid=os.getpid()))
    monkeypatch.chdir(project_root)
    assert main(["--port", "1", "--token", TOKEN]) == 1
    assert NO_SERVER_MESSAGE in capsys.readouterr().err


# -- the stdio bridge end to end ----------------------------------------------


@pytest.fixture
def live_read_service(project_root: Path, events: list[str], read_surface):
    """The in-process server (§0: the REAL one, uvicorn thread + guard)."""
    service = _service(project_root, events, lambda: read_surface.server)
    assert service.start() is True
    published = load_endpoint(project_root)
    assert published is not None
    try:
        yield service
    finally:
        service.stop()
    assert not endpoint_path(project_root).exists()


def _initialize(session: StdioSession) -> dict:
    """The stateful Streamable HTTP session dance, over stdio lines."""
    response = session.request(
        1,
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "drcode-tests", "version": "1"},
        },
    )
    session.notify("notifications/initialized")
    return response


def test_stdio_round_trip_through_the_bridge(
    project_root: Path, manager, live_read_service: McpService
) -> None:
    """A read-only tool call walks client-stdio → shim → HTTP → server back.

    The shim is a real subprocess spawned in the project root (§1: that
    is how MCP clients launch it); the server side is the in-process
    `McpService` from the server task. The tool answer must name a task
    created AFTER the shim connected — live state, not a recording.
    """
    assert live_read_service.port is not None
    proc = _spawn_shim(project_root)
    session = StdioSession(proc)
    try:
        response = _initialize(session)
        info = response["result"]["serverInfo"]
        assert info["name"] == "drcode"

        manager.create("Bridged task", "over the stdio bridge")

        listing = session.request(2, "tools/list")
        names = [tool["name"] for tool in listing["result"]["tools"]]
        assert "list_tasks" in names

        call = session.request(3, "tools/call", {"name": "list_tasks", "arguments": {}})
        assert call["result"]["isError"] is False
        text = call["result"]["content"][0]["text"]
        assert "Bridged task" in text
        assert f"#{manager.all_tasks()[0].id}" in text
    finally:
        session.close()
    assert proc.returncode == 0


def test_stdio_shim_takes_cli_overrides(
    project_root: Path, manager, live_read_service: McpService
) -> None:
    """`--port/--token` work without the file (the non-standard path, §1)."""
    published = load_endpoint(project_root)
    assert published is not None
    endpoint_path(project_root).unlink()  # the file must NOT be the source here

    proc = _spawn_shim(
        project_root,
        "--port",
        str(published.port),
        "--token",
        published.token,
    )
    session = StdioSession(proc)
    try:
        _initialize(session)
        manager.create("Override task", "via --port/--token")
        call = session.request(4, "tools/call", {"name": "list_tasks", "arguments": {}})
        assert call["result"]["isError"] is False
        assert "Override task" in call["result"]["content"][0]["text"]
    finally:
        session.close()
    assert proc.returncode == 0


def test_shim_reports_a_missing_endpoint_as_a_subprocess(
    project_root: Path,
) -> None:
    """The spawned-process view of the failure contract: exit 1, hint, no stdout."""
    proc = _spawn_shim(project_root)
    try:
        out, _err = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - hostile env
        proc.kill()
        raise
    assert proc.returncode == 1
    assert NO_SERVER_MESSAGE in _err
    # stdout carries JSON-RPC frames and nothing else — on a startup
    # failure not a single byte of it may leak to the client's parser.
    assert out == ""
