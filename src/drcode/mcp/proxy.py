"""The stdio shim behind the `drcode-mcp` console script (design doc §1).

External MCP clients come in two kinds: the modern ones speak Streamable
HTTP and can read `.drcode/mcp.json` themselves; older clients only know
how to **spawn a subprocess and talk JSON-RPC over stdin/stdout**. For
those, drcode ships this shim: a small proxy process that owns a
`mcp.server.stdio` transport on one side and a `mcp.client.streamable_http`
transport on the other and forwards `SessionMessage`s between the two
pairs of memory streams — both directions, including the server→client
SSE stream (notifications ride it after the first POST opens it).

Design rules, all binding:

- **The endpoint is discovered, never guessed** (§1): `<cwd>/.drcode/mcp.json`
  is the source of truth — MCP clients spawn servers with the project root
  as cwd, so the file sits right there. `--port`/`--token` exist for
  non-standard scenarios and override the file field-by-field; the file
  remains the primary source.
- **A missing or dead endpoint is a clean failure, never a crash and never
  an auto-start** (§1): the shim prints the human-readable hint to stderr
  and exits 1. It must NEVER start a TUI itself — lifecycle mutations are
  only valid through the live pipeline (§0), and a headless TaskManager
  would silently break that contract.
- **stdio discipline**: everything the shim wants to say goes to stderr;
  stdout carries JSON-RPC frames and nothing else. The SDK's
  `stdio_server` additionally re-points the raw fd 1 while serving, so
  even a stray child-process print cannot corrupt the wire.

The bridge itself is transport-agnostic and testable in-process: it
connects two already-built stream pairs. Only the thin `serve()` wrapper
knows where the stdio pair comes from.

Importing this module is cheap: the `mcp` SDK is imported lazily inside
the async functions, so `drcode-mcp --help` pays nothing for it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .endpoint import (
    LOOPBACK_HOST,
    MCP_PATH,
    endpoint_path,
    load_endpoint,
    pid_alive,
    probe_healthz,
)

#: The stderr hint when the project has no live MCP endpoint (§1, verbatim).
NO_SERVER_MESSAGE: Final = (
    "drcode TUI не запущен или MCP выключен — включите в Settings → MCP server"
)

#: Timeout of the startup `/healthz` probe. Slightly more generous than
#: `endpoint.PROBE_TIMEOUT` (which gates a TUI's second-instance check):
#: the shim is spawned cold by a client and a loaded machine may need a
#: beat to answer.
PROBE_TIMEOUT: Final = 2.0

#: Prefix every stderr diagnostic carries, so a client's log shows whose
#: noise this is.
PROG: Final = "drcode-mcp"


class ProxyStartupError(Exception):
    """The endpoint cannot be reached before any JSON-RPC flows.

    The message is what goes to stderr verbatim; the process exits 1.
    """


class ProxyConnectionError(Exception):
    """The HTTP endpoint died mid-session (after a successful start)."""


@dataclass(frozen=True)
class ResolvedEndpoint:
    """The endpoint the shim will talk to: URL + bearer token."""

    port: int
    token: str

    @property
    def url(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self.port}{MCP_PATH}"


def resolve_endpoint(
    project_root: Path,
    *,
    port: int | None = None,
    token: str | None = None,
) -> ResolvedEndpoint:
    """Find the endpoint to bridge to; raise `ProxyStartupError` if none.

    Precedence per field (§1): an explicit CLI option wins, the
    descriptor fills the rest. The liveness gate is two-fold: a
    descriptor whose owning pid is dead means a crashed TUI left the file
    behind — report it rather than dial a port some other process may
    have reused; then `/healthz` must answer (a file can be minutes old,
    only an answering endpoint counts).
    """
    descriptor = load_endpoint(project_root)
    if port is None and descriptor is None:
        # No overrides and nothing published: the ordinary "TUI is off"
        # case — the exact hint the design doc mandates, plus the file
        # that was checked so the user can verify the state themselves.
        raise ProxyStartupError(
            f"{NO_SERVER_MESSAGE} (не найден {endpoint_path(project_root)})"
        )
    if port is None and descriptor is not None and not pid_alive(descriptor.pid):
        raise ProxyStartupError(
            f"{NO_SERVER_MESSAGE} (устаревший {endpoint_path(project_root)}: "
            f"pid {descriptor.pid} больше не жив)"
        )
    effective_port = (
        port if port is not None else (descriptor.port if descriptor else 0)
    )
    effective_token = (
        token
        if token is not None
        else (descriptor.token if descriptor is not None else None)
    )
    if effective_port <= 0 or not effective_token:
        raise ProxyStartupError(NO_SERVER_MESSAGE)
    if not probe_healthz(effective_port, timeout=PROBE_TIMEOUT):
        raise ProxyStartupError(
            f"{NO_SERVER_MESSAGE} (порт {effective_port} не отвечает на /healthz)"
        )
    assert effective_token is not None  # narrowing for the type checkers
    return ResolvedEndpoint(port=effective_port, token=effective_token)


async def bridge(
    stdio_read,
    stdio_write,
    url: str,
    *,
    token: str,
) -> None:
    """Forward JSON-RPC between a stdio stream pair and the HTTP endpoint.

    `stdio_read`/`stdio_write` are the pair `mcp.server.stdio.stdio_server`
    yields (client → shim and shim → client); the HTTP side is created
    here with the bearer token baked into the client's default headers —
    the token never passes through any JSON-RPC frame.

    Either side finishing tears the whole bridge down (the task group's
    cancel scope), so a client closing its stdin or a server dropping
    the session both end the shim cleanly instead of leaving a half-open
    proxy hanging. A malformed client line (the stdio transport reports
    those as `Exception` items) is dropped: there is no request id to
    answer with, and the client's own timeout is the right recovery.
    """
    import anyio
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    headers = {"Authorization": f"Bearer {token}"}
    async with (
        create_mcp_http_client(headers=headers) as http_client,
        streamable_http_client(url, http_client=http_client) as (
            http_read,
            http_write,
        ),
        anyio.create_task_group() as tg,
    ):

        async def client_to_server() -> None:
            try:
                async with http_write:
                    async for message in stdio_read:
                        if isinstance(message, Exception):
                            continue  # unparseable client line: no id to answer
                        await http_write.send(message)
            finally:
                tg.cancel_scope.cancel()

        async def server_to_client() -> None:
            try:
                async with stdio_write:
                    async for message in http_read:
                        if isinstance(message, Exception):
                            raise ProxyConnectionError(
                                f"HTTP endpoint error: {message}"
                            ) from message
                        await stdio_write.send(message)
            finally:
                tg.cancel_scope.cancel()

        tg.start_soon(client_to_server)
        tg.start_soon(server_to_client)


async def serve(endpoint: ResolvedEndpoint) -> None:
    """Run the shim until the stdio client disconnects or the endpoint dies."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (stdio_read, stdio_write):
        await bridge(stdio_read, stdio_write, endpoint.url, token=endpoint.token)


def _fail(message: str) -> int:
    """One stderr line + exit code 1 — the §1 failure contract."""
    print(f"{PROG}: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Console entry: resolve the endpoint, then bridge until done."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "stdio→Streamable-HTTP MCP shim for the drcode TUI's embedded "
            "server. Reads the endpoint from <cwd>/.drcode/mcp.json."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="override the endpoint port from mcp.json (non-standard setups)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="override the bearer token from mcp.json (non-standard setups)",
    )
    args = parser.parse_args(argv)
    try:
        endpoint = resolve_endpoint(Path.cwd(), port=args.port, token=args.token)
    except ProxyStartupError as e:
        return _fail(str(e))
    try:
        import anyio

        anyio.run(serve, endpoint)
    except ProxyConnectionError as e:
        return _fail(f"HTTP endpoint lost mid-session: {e}")
    except OSError as e:
        # A refused/timeouted connect that surfaced as a transport error
        # instead of a stream Exception — same contract: say it, exit 1.
        return _fail(f"{NO_SERVER_MESSAGE} ({e})")
    except Exception as e:  # noqa: BLE001 - a proxy must not traceback at a client
        return _fail(f"stdio bridge failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":  # pragma: no cover - `python -m drcode.mcp_proxy`
    sys.exit(main())
