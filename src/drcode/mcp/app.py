"""ASGI composition of the embedded MCP server (design doc §1/§2).

Two HTTP surfaces share one loopback port:

- `POST/GET/DELETE /mcp` — the SDK's **Streamable HTTP** transport in its
  stateful mode (`Mcp-Session-Id` is managed by the SDK, the GET stream
  carries server-side notifications). Everything below it is the SDK's
  own Starlette app, which is why nothing here reimplements JSON-RPC:
  protocol churn (spec versions, sessions, framing) stays the SDK's
  problem.
- `GET /healthz` — an unauthenticated `{"ok": true}`, the liveness probe
  of `endpoint.probe_healthz` (the second-instance rule of §3 and the
  stdio shim both lean on it). It carries no data, so it needs no token
  (§2), and it must stay OUTSIDE the guard — a probe cannot know a token.

The composition is deliberately tiny — a parent Starlette app with the
health route plus the SDK app mounted under the guard middleware — and
built by one factory so the lifecycle controller, the tests and the
context/health accounting can never assemble it differently.

One subtlety keeps this non-obvious: Starlette does not propagate the
lifespan of a mounted sub-app. The SDK app's lifespan is what starts the
Streamable HTTP session manager (`session_manager.run()`), and without it
every `/mcp` request fails with "task group not initialized" — hence the
explicit `lifespan=inner.router.lifespan_context` below.

No MCP tools are registered in this phase: the server exposes an empty
tool list until the tool layer lands. The factory already accepts a
pre-built `MCPServer`, so adding tools never touches the transport.
"""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.types import ASGIApp

from ..version import APP_VERSION
from .endpoint import HEALTH_PATH, LOOPBACK_HOST, MCP_PATH
from .guard import GuardMiddleware

#: Name the server reports to MCP clients.
SERVER_NAME = "drcode"

#: The single field of the health answer — nothing else is ever exposed (§2).
HEALTH_PAYLOAD: dict[str, bool] = {"ok": True}


def build_server(*, instructions: str | None = None, bus=None):
    """Build the SDK server instance (no tools yet — infrastructure only).

    Kept as a function rather than a module-level singleton: the SDK
    server owns per-instance managers (tool/resource/prompt) and a
    lifespan, so one process could legitimately host several (the tests
    do). `version` comes from the single source of truth so a client can
    tell builds apart.

    `bus` is the `SubscriptionBus` the server's `subscriptions/listen`
    fan-out runs on (§5). The default `None` lets the SDK build its own
    in-memory bus — fine for a server nobody publishes to; the read
    surface (`wiring.ReadSurface`) passes its own so the event bridge
    holds the very instance the notifications go out on.
    """
    from mcp.server.mcpserver import MCPServer

    return MCPServer(
        name=SERVER_NAME,
        version=APP_VERSION,
        instructions=instructions,
        # WARNING: the SDK logs a line per request/session at INFO, which
        # would spam the TUI's stderr for something the service events
        # already report in human terms.
        log_level="WARNING",
        subscriptions=bus,
    )


async def healthz(request: Request) -> JSONResponse:
    """The unauthenticated `/healthz` endpoint (§2)."""
    return JSONResponse(HEALTH_PAYLOAD)


def build_app(*, token: str, server=None) -> Starlette:
    """Assemble the full ASGI app for the given token and SDK server.

    `server` defaults to a fresh `build_server()` — the tests pass their
    own instance (with stub tools) to talk to. Returns the concrete
    `Starlette` app rather than a bare `ASGIApp`: callers need
    `router.lifespan_context` to drive the session manager (an ASGI
    transport in the tests does exactly that).
    """
    if server is None:
        server = build_server()
    inner = server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        # loopback in every spelling: the SDK then auto-enables its own
        # DNS-rebinding settings for exactly this host (defence in depth —
        # `GuardMiddleware` is the primary, token-carrying check).
        host=LOOPBACK_HOST,
    )
    guarded: ASGIApp = GuardMiddleware(inner, token=token)
    # A bare Router for the guard: `Mount` would strip the path prefix and
    # the inner app would then match "/" instead of "/mcp".
    return Starlette(
        routes=[
            Route(HEALTH_PATH, healthz, methods=["GET"]),
            Mount("/", app=guarded),
        ],
        # Mount does NOT run the sub-app's lifespan — pass the SDK app's
        # own context explicitly, or the session manager never starts.
        lifespan=inner.router.lifespan_context,
        # No CORS middleware on purpose (§2): a browser client is not a
        # supported deployment, and a permissive CORS policy would defeat
        # the Origin check of `guard.py`.
    )


__all__ = [
    "HEALTH_PAYLOAD",
    "SERVER_NAME",
    "build_app",
    "build_server",
    "healthz",
]
