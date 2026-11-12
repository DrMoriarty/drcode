"""The embedded MCP server (design doc `docs/mcp-server-design.md`).

drcode exposes itself to external AI agents over **Streamable HTTP on
loopback**, serving the live TUI process — not a headless copy. That is
the whole point of §0 of the design: lifecycle mutations must run through
the running pipeline, whose single `_advance()` gate only exists inside
the app instance.

The package lives here, next to `core/` rather than inside it, because
`core/` is the UI-neutral pipeline and `tests/test_core_purity.py`
forbids any server/transport code there (core → outside imports are
fine, the reverse is not).

Module map:

- `endpoint` — the `.drcode/mcp.json` descriptor (port, bearer token,
  owning pid), the `/healthz` probe and pid liveness. It is also the
  mutex of the one-server-per-project rule (§3).
- `guard` — the ASGI middleware over the SDK app: bearer-token auth on
  `/mcp` (401 without details) plus Origin/Host validation against DNS
  rebinding.
- `app` — the ASGI composition: the unauthenticated `/healthz` route
  next to the SDK's guarded Streamable HTTP app.
- `service` — the lifecycle controller: the ephemeral socket, the
  `uvicorn.Server` in a daemon thread, start/stop and the conflicts of
  two TUIs over one project.

No MCP tools are registered yet — this is the transport, auth and
lifecycle foundation; the tool surface (§4/§7 of the design) is the next
layer and plugs into `app.build_server()`.

The token is never displayed in the UI: the user is told the path of
`.drcode/mcp.json` (mode 0600), which is also how a client obtains it.

Importing this package is cheap — the `mcp` SDK and uvicorn are imported
lazily inside `service._build_app`/`_serve`, so a session that never
enables the server pays nothing for it.
"""

from __future__ import annotations

from .endpoint import (
    HEALTH_PATH,
    LOOPBACK_HOST,
    MCP_PATH,
    McpEndpoint,
    endpoint_is_live,
    endpoint_path,
    load_endpoint,
    new_token,
    probe_healthz,
    remove_endpoint_if_owned_by,
    save_endpoint,
)
from .service import McpService, disabled_note

__all__ = [
    "HEALTH_PATH",
    "LOOPBACK_HOST",
    "MCP_PATH",
    "McpEndpoint",
    "McpService",
    "disabled_note",
    "endpoint_is_live",
    "endpoint_path",
    "load_endpoint",
    "new_token",
    "probe_healthz",
    "remove_endpoint_if_owned_by",
    "save_endpoint",
]
