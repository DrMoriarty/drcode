"""The read surface assembled: tools + resources + the event bridge (§5/§7).

The transport (§1/§2) and the read half (§7) meet here. `ReadSurface` is
the ONE object the TUI keeps a reference to: it builds the SDK server,
registers the nine read tools and the four resources over a single
`ProjectView`, and owns the subscription bus the event bridge publishes
through. Owning the bus explicitly (instead of letting the SDK build its
own) is what makes `start_notifications` possible: the bridge's
`NotificationRouter` publishes onto exactly the bus whose listen streams
the clients hang off.

Lifecycle contract with `McpService`:

- the factory (the TUI's closure) builds a fresh `ReadSurface` per START —
  an SDK server is per-instance state (tool managers, the streamable
  HTTP session manager) and must not be reused across a stop/start cycle;
- `start_notifications(loop)` runs AFTER the service is healthy (the loop
  argument is the uvicorn thread's loop, which `McpService.loop` exposes)
  and attaches the bridge to the manager — from then on every
  `fire_status_change` ends in a coalesced `notifications/resources/updated`;
- `stop_notifications()` detaches before the server thread dies — a
  stopped surface must keep feeding no loop.
"""

from __future__ import annotations

import asyncio

from .app import build_app, build_server
from .bridge import EventBridge, NotificationRouter, bridge_for
from .reads import ProjectView
from .resources import register_resources
from .tools import register_tools


class ReadSurface:
    """The read half of the embedded server, assembled and lifecycle-aware."""

    def __init__(self, manager, *, usage_store=None) -> None:
        from mcp.server.subscriptions import InMemorySubscriptionBus

        self.view = ProjectView(manager, usage_store=usage_store)
        self.bus = InMemorySubscriptionBus()
        self.server = build_server(bus=self.bus)
        register_tools(self.server, self.view)
        register_resources(self.server, self.view)
        self.router: NotificationRouter | None = None
        self.bridge: EventBridge | None = None

    def asgi(self, token: str):
        """The guarded ASGI app this surface serves (the `McpService` payload)."""
        return build_app(token=token, server=self.server)

    # -- the event bridge (§5) ---------------------------------------------------

    def start_notifications(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        window: float | None = None,
    ) -> None:
        """Attach the bridge to the manager, publishing on `loop`.

        Idempotent: a second call with the router already bound to the
        same loop is a no-op (the TUI's `_sync_mcp_server` re-runs on every
        settings apply). The `attach` registers the ONE thin listener
        quartet of §5 — the bridge maps each manager event to the URIs of
        the affected resources (`resources.affected_uris_*`), the router
        coalesces and publishes. `window` overrides the coalescing window
        (tests shrink it; production uses the §5 default).
        """
        router = self.router
        if router is not None and router.loop is loop:
            return
        if router is not None:
            # A live router on another loop (hot restart): detach first.
            self.stop_notifications()
        kwargs: dict[str, float] = {} if window is None else {"window": window}
        router, bridge = bridge_for(self.bus, loop, **kwargs)
        bridge.attach(self.view.manager)
        self.router = router
        self.bridge = bridge

    def stop_notifications(self) -> None:
        """Detach the listeners and close the router (server teardown)."""
        bridge, self.bridge = self.bridge, None
        router, self.router = self.router, None
        if bridge is not None:
            bridge.detach()
        if router is not None:
            router.close()


__all__ = ["ReadSurface"]
