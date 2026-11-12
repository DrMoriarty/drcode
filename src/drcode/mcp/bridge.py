"""Event bridge: TaskManager transitions → MCP resource notifications (§5).

The design's threading rule is the whole difficulty here. A status change
happens on whatever thread touched the task — the app thread, a worker
thread running the tester's tools, a pipeline callback — while the MCP
server owns its own event loop on the uvicorn daemon thread. The bridge is
therefore two pieces:

1. `EventBridge` — a SYNC listener registered into the `TaskManager`. Its
   only job is to tell an MCP-loop callable "these URIs changed". That is
   `sim` in the design's terms: `loop.call_soon_threadsafe`, no locks on
   the hot path, no exceptions (the manager's own listeners all must be
   fast and non-raising, and ours is fire-and-forget).
2. `NotificationRouter` — the MCP-loop half. Ugly details live there:
   the pending-URI set, the ~250 ms coalescing window, and the actual
   `notifications/resources/updated` publish.

**Why coalescing is not optional.** A single development round fires
dozens of transitions (ACTIVE → TESTING → COMPLETED, tester verdicts,
queue reshuffles), and a burst of 50 notifications would just be a DoS of
the client's own refetch logic. The rule is per-URI: the FIRST event for a
URI arms a timer; everything that arrives inside `COALESCE_WINDOW` joins
the same batch; when the timer fires, each distinct URI gets exactly ONE
notification. Level-triggered semantics make this safe — the notification
only means "refetch if you care", so a coalesced burst loses nothing (the
last state is what a refetch returns).

**Only affected URIs** (`resources.affected_uris_*`). A status change of an
unlinked task must not wake every `drcode://goals/*` subscriber, and a
stats change must not churn `drcode://tasks/*`. That routing lives in
`resources.py` next to the URI builders so a new resource cannot be added
without deciding its trigger.

**No subscriptions, no loss.** The agent that never subscribes simply
polls `list_tasks`/`get_task` (§9); this module is an optimization, which
is why every failure inside it is swallowed — a broken notification must
never break the pipeline.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Iterable

from ..models import Task
from .resources import (
    affected_uris_for_completion,
    affected_uris_for_creation,
    affected_uris_for_removal,
    affected_uris_for_status,
    dedupe,
)

#: The coalescing window of §5: a burst of transitions inside this span
#: collapses into one notification per URI.
COALESCE_WINDOW = 0.25

#: Publisher signature: an async callable announcing one URI. The SDK's
#: `ctx.notify_resource_updated` for the request-bound path, or a bus
#: publish for the loop-owning path — the router does not care which.
Publisher = Callable[[str], "asyncio.Future[None] | None"]


def _report(text: str) -> None:
    """Bridge failures are logged, never raised (they are not the TUI's problem)."""
    print(f"drcode mcp: {text}", file=sys.stderr)


class NotificationRouter:
    """Coalescing scheduler on the MCP loop: pending URIs → one publish each.

    Loop-affine: `schedule` may be called from ANY thread (that is its
    purpose — it is what `loop.call_soon_threadsafe` ends up running), but
    the timer callback and `flush` must run on the router's loop.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        publish: Publisher,
        *,
        window: float = COALESCE_WINDOW,
    ) -> None:
        self._loop = loop
        self._publish = publish
        self._window = window
        self._pending: dict[str, None] = {}
        self._sources: dict[str, None] = {}
        self._timer: asyncio.TimerHandle | None = None
        self._closed = False
        # Diagnostics for the tests (bounded: only the last batch is kept).
        self.published: list[list[str]] = []

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """The loop publishes run on (the MCP server thread's loop)."""
        return self._loop

    # -- producer side ---------------------------------------------------------

    def schedule(self, uris: Iterable[str], *, source: str = "") -> None:
        """Queue URIs for the next batch — from any thread.

        `source` is only diagnostics: the routing bug it exposes ("a task
        change woke up the goals") is otherwise invisible.
        """
        batch = dedupe([uri for uri in uris if uri])
        if not batch:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            # Already on the MCP loop (a handler publishing, tests): no hop.
            self._enqueue(batch, source)
        else:
            try:
                self._loop.call_soon_threadsafe(self._enqueue, batch, source)
            except RuntimeError as e:  # loop closed / never started
                _report(f"cannot schedule {batch!r}: {type(e).__name__}: {e}")

    def _enqueue(self, batch: list[str], source: str) -> None:
        """Merge a batch into the pending set and (re)arm the window."""
        if self._closed:
            return
        for uri in batch:
            self._pending.setdefault(uri, None)
        if self._timer is None:
            self._timer = self._loop.call_later(self._window, self._fire)
        # Diagnostics only: the first source of a batch explains the batch.
        if source:
            self._sources.setdefault(source, None)
        else:  # pragma: no cover - defensive
            pass

    # -- consumer side ---------------------------------------------------------

    def _fire(self) -> None:
        """The window elapsed: publish one notification per pending URI."""
        self._timer = None
        uris = list(self._pending)
        self._pending.clear()
        self._sources.clear()
        if not uris:
            return
        self.published.append(uris)
        if len(self.published) > 16:  # bounded diagnostic buffer
            del self.published[0]
        for uri in uris:
            self._notify(uri)

    def _notify(self, uri: str) -> None:
        """Publish ONE resource-updated event; any failure is logged."""
        try:
            result = self._publish(uri)
        except Exception as e:  # noqa: BLE001 - a notification never breaks the pipeline
            _report(f"resource notify {uri} failed: {type(e).__name__}: {e}")
            return
        if result is None:
            return
        try:
            self._loop.create_task(_swallow(result, uri))
        except Exception as e:  # noqa: BLE001, pragma: no cover - defensive
            _report(f"resource notify {uri} not scheduled: {type(e).__name__}: {e}")

    def close(self) -> None:
        """Stop the timer and drop the backlog (server shutdown)."""
        self._closed = True
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
        self._pending.clear()
        self._sources.clear()


async def _swallow(awaitable, uri: str) -> None:
    """Await a publish future, swallowing its failure (logged, never raised)."""
    try:
        await awaitable
    except Exception as e:  # noqa: BLE001 - fire-and-forget by contract
        _report(f"resource notify {uri} failed: {type(e).__name__}: {e}")


class EventBridge:
    """Sync TaskManager listeners feeding a `NotificationRouter`.

    One instance per server. `attach(manager)` registers the four thin
    listeners; `detach(manager)` removes them again on shutdown — a stopped
    server must not keep a dying TUI's state machine feeding a closed loop.

    The listeners hold no state and do no work: they map a manager event to
    a URI list (`resources.affected_uris_*`) and hand it to the router.
    """

    def __init__(self, router: NotificationRouter) -> None:
        self._router = router
        self._manager = None
        self._listeners: list[tuple[str, object]] = []

    # -- wiring ----------------------------------------------------------------

    def attach(self, manager) -> None:
        """Register the listeners on `manager` (idempotent)."""
        if self._manager is manager:
            return
        if self._manager is not None:
            self.detach(self._manager)
        status = self._on_status
        removal = self._on_removal
        completion = self._on_completion
        creation = self._on_creation
        manager.add_status_listener(status)  # type: ignore[arg-type]
        manager.add_removal_listener(removal)  # type: ignore[arg-type]
        manager.add_completion_listener(completion)  # type: ignore[arg-type]
        manager.add_creation_listener(creation)  # type: ignore[arg-type]
        self._listeners = [
            ("status", status),
            ("removal", removal),
            ("completion", completion),
            ("creation", creation),
        ]
        self._manager = manager

    def detach(self, manager=None) -> None:
        """Remove our listeners from `manager` (or the attached one).

        The TaskManager keeps its listener lists private and offers no
        removal API (every existing listener is registered for the process
        lifetime), so the lists are filtered directly — with a comment here
        because it is the one place reaching into them.
        """
        target = manager if manager is not None else self._manager
        if target is None:
            return
        for kind, listener in self._listeners:
            attr = {
                "status": "_status_listeners",
                "removal": "_removal_listeners",
                "completion": "_completion_listeners",
                "creation": "_creation_listeners",
            }[kind]
            try:
                listeners = getattr(target, attr)
                setattr(
                    target,
                    attr,
                    [item for item in listeners if item is not listener],
                )
            except Exception as e:  # noqa: BLE001 - teardown is best-effort
                _report(f"detach {kind} listener failed: {type(e).__name__}: {e}")
        self._listeners = []
        self._manager = None

    # -- listeners (sync, fast, non-raising) -----------------------------------

    def _on_status(self, task: Task, _old, _new, _origin: str) -> None:
        self._router.schedule(affected_uris_for_status(task), source="status")

    def _on_removal(self, task_id: str) -> None:
        self._router.schedule(affected_uris_for_removal(task_id), source="removal")

    def _on_completion(self, task: Task, _old, _origin: str) -> None:
        self._router.schedule(affected_uris_for_completion(task), source="completion")

    def _on_creation(self, task: Task, _origin: str) -> None:
        self._router.schedule(affected_uris_for_creation(task), source="creation")


class LoopPublisher:
    """`Publisher` implementation that owns a "publish to subscribers" seam.

    The MCPServer publishes resource updates through its `SubscriptionBus`
    (`Context.notify_resource_updated` is just `bus.publish`). A listener
    cannot hold a request-bound `Context` — it fires from arbitrary threads
    and has no request at all — so the bridge publishes through the bus the
    server was CONSTRUCTED with. `LoopPublisher` wraps that: it is the
    publisher callable the router gets, and it is the only place that knows
    the bus.

    Failures (a closed bus, a client that vanished) are the router's
    business: this returns the publish coroutine and lets it be awaited.
    """

    def __init__(self, bus) -> None:
        self._bus = bus

    def __call__(self, uri: str):
        from mcp.shared.subscriptions import ResourceUpdated

        loop = asyncio.get_running_loop()
        return loop.create_task(self._bus.publish(ResourceUpdated(uri=uri)))


def current_loop() -> asyncio.AbstractEventLoop | None:
    """The running loop, or None (the router can then be built later)."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def bridge_for(
    bus,
    loop: asyncio.AbstractEventLoop,
    *,
    window: float = COALESCE_WINDOW,
) -> tuple[NotificationRouter, EventBridge]:
    """Build the router + bridge pair for a server's bus on `loop`."""
    router = NotificationRouter(loop, LoopPublisher(bus), window=window)
    return router, EventBridge(router)


__all__ = [
    "COALESCE_WINDOW",
    "EventBridge",
    "LoopPublisher",
    "NotificationRouter",
    "bridge_for",
    "current_loop",
]
