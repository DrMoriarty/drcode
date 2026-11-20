"""A minimal stdio MCP echo server for the client lifecycle tests (§7 of
docs/mcp-client-design.md).

Runs on the mcp SDK itself — a full drcode dependency (the fastmcp-slim
variant installed for the client path has no server support).

Startup side channel for the tests: when `ECHO_PIDFILE` is set, the
server writes its own pid there, so a test can verify the stdio child
is terminated (no zombies) after a run or an abort.

Environment contract: the transport env is the SDK's whitelist-based
default plus the record's `env` — the test relies on `ECHO_TOKEN`
passing through and `DRCODE_MCP_CANARY` NOT passing through.
"""

from __future__ import annotations

import os
import time

from mcp.server.mcpserver import MCPServer

server = MCPServer("echo-server")


@server.tool()
def echo(text: str) -> str:
    """Echo the text back prefixed with 'echo:'."""
    return f"echo:{text}"


@server.tool()
def env_probe(name: str) -> str:
    """Return the value of an environment variable ('<absent>' if unset)."""
    return os.environ.get(name, "<absent>")


@server.tool()
def slow(seconds: float) -> str:
    """Sleep for the given number of seconds, then return."""
    time.sleep(seconds)
    return "woke up"


def main() -> None:
    pidfile = os.environ.get("ECHO_PIDFILE")
    if pidfile:
        with open(pidfile, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
            f.flush()
    server.run()  # stdio by default


if __name__ == "__main__":
    main()
