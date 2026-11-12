"""Lifecycle controller of the embedded MCP server (design doc §3/§6).

Owns the one thing the ASGI layers cannot: *whether* a server exists for
this project, on which port, and how it goes away.

Threading model (§0): the server is a `uvicorn.Server` in a **daemon**
thread with its own event loop — the pattern `notifications.py` uses for
delivery. Nothing here ever runs a blocking call on the TUI's loop: the
socket is bound synchronously (fast, and the port must be known before
anything is published), while the serve loop lives on the daemon thread.
Tool handlers will hop into the app thread through `call_from_thread`;
this module only ever reports events.

**One server = one project** (§3). Two TUIs over the same project are
already dangerous (two TaskManagers over one set of task files), so the
second one must NOT expose a second endpoint: the descriptor
(`.drcode/mcp.json`) plus a `/healthz` probe is the mutex. The rules are
implemented literally:

1. descriptor present, pid alive AND `/healthz` answering → the project
   is served elsewhere: don't start, report whose pid it is;
2. descriptor present but stale (dead pid or silent port, e.g. a crashed
   TUI) → overwrite it and start;
3. no descriptor → publish one and start.

That check runs twice on purpose: once on entry (the cheap refusal) and
again right before the descriptor is written — the second one is what
closes the race of two instances starting simultaneously (§6: "второй
инстанс перепроверяет healthz до записи файла").

**Token reuse** (§2): the token is generated at the first enable and
regenerated only when the file is lost. A stale descriptor therefore
still lends its token to the restarted server, so a client configured
against `.drcode/mcp.json` keeps working across TUI restarts.

Everything is failure-tolerant by contract: `start` returns a bool and
`stop` never raises — an MCP endpoint is a convenience, never a
precondition for the TUI to run.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from collections.abc import Callable
from typing import Final

from .endpoint import (
    LOOPBACK_HOST,
    McpEndpoint,
    endpoint_is_live,
    endpoint_path,
    load_endpoint,
    new_token,
    probe_healthz,
    remove_endpoint_if_owned_by,
    save_endpoint,
)

#: How long a freshly started server may take to answer `/healthz` before
#: the start is declared failed. Generous for a cold interpreter under a
#: loaded machine, short enough that a broken start cannot wedge the TUI.
START_TIMEOUT: Final = 10.0

#: Grace period of `stop()` — a graceful shutdown closes the listening
#: socket and lets an open GET-SSE stream finish. The thread is a daemon,
#: so a hung shutdown can never keep the process alive past this join.
STOP_TIMEOUT: Final = 5.0

#: Poll interval while waiting for the first `/healthz` answer.
_START_POLL: Final = 0.05

#: Serializes the descriptor write of instances starting at the same
#: moment (module-level: the mutex is per project directory in practice,
#: and a process-wide lock is the stricter, simpler choice).
_START_LOCK = threading.Lock()

EventSink = Callable[[str], None]
"""Callback receiving one human-readable service-event line per lifecycle
step (started / stopped / refused / failed). Called from whatever thread
triggered the step — the TUI's sink hops threads itself."""


class McpService:
    """Starts, tracks and stops the project's embedded MCP server.

    The instance is inert until `start()` is called and stays harmless
    forever: every failure path ends in an event line instead of an
    exception, so the TUI never dies because an endpoint could not be
    published.
    """

    def __init__(
        self,
        project_root,
        *,
        event_sink: EventSink | None = None,
        server_factory: Callable[[], object] | None = None,
    ) -> None:
        self.project_root = project_root
        self._event_sink = event_sink
        # Injectable for the tests: the SDK server instance (and thus the
        # tool set) is not this module's business.
        self._server_factory = server_factory
        self._server = None
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._endpoint: McpEndpoint | None = None
        # The event loop of the server thread (set once the thread runs its
        # own loop instead of letting asyncio.run create a hidden one). The
        # event bridge (§5) publishes on it, so the TUI needs it after
        # start() — by then the loop is alive (the healthz wait guarantees
        # the request handlers are already serving on it).
        self._loop: asyncio.AbstractEventLoop | None = None
        # pid of another live instance serving this project, when we were
        # refused (the TUI shows it and explains why nothing is exposed).
        self._foreign_pid: int | None = None

    # -- state ---------------------------------------------------------------

    @property
    def running(self) -> bool:
        """Whether OUR server thread is alive (a refused instance is not)."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def port(self) -> int | None:
        """The published port, or None while no server of ours runs."""
        if not self.running or self._endpoint is None:
            return None
        return self._endpoint.port

    @property
    def url(self) -> str | None:
        """The Streamable HTTP endpoint URL clients connect to."""
        if not self.running or self._endpoint is None:
            return None
        return self._endpoint.url

    @property
    def foreign_pid(self) -> int | None:
        """The pid of the other instance that owns the endpoint, if any."""
        return self._foreign_pid

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The server thread's event loop while a server of ours runs.

        The event bridge's notifications must be published ON the loop the
        SDK's listen streams live on — this is it. None before the first
        start and after the shutdown, so callers can treat it as "the
        server's loop, if one is serving".
        """
        if not self.running:
            return None
        return self._loop

    @property
    def endpoint_file(self) -> str:
        """Path of the descriptor — the ONLY thing the UI may show (§6)."""
        return str(endpoint_path(self.project_root))

    def set_event_sink(self, sink: EventSink | None) -> None:
        """Attach/replace the service-event reporter (late wiring)."""
        self._event_sink = sink

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> bool:
        """Bring the server up unless the project is served elsewhere.

        Returns True when OUR server answers `/healthz` at the end of the
        call. Idempotent: a second call with a live server is a no-op.
        """
        if self.running:
            return True
        try:
            return self._start()
        except Exception as e:  # noqa: BLE001 - an endpoint must never crash the TUI
            self._cleanup_failed_start()
            self._emit(f"MCP: failed to start the server: {type(e).__name__}: {e}")
            return False

    def _start(self) -> bool:
        existing = load_endpoint(self.project_root)
        if existing is not None and existing.pid != os.getpid():
            if endpoint_is_live(existing):
                self._foreign_pid = existing.pid
                self._emit(
                    "MCP: endpoint already served by another instance "
                    f"(pid {existing.pid}) — not starting a second one"
                )
                return False
        else:
            self._foreign_pid = None
        # Reuse a surviving token (§2: it is regenerated only when the
        # file is lost) so configured clients survive a TUI restart.
        token = existing.token if existing is not None else new_token()
        app = self._build_app(token)
        sock = _bind_ephemeral()
        self._socket = sock
        port = sock.getsockname()[1]
        server = self._serve(app, sock)
        thread = threading.Thread(
            target=self._serve_loop,
            args=(server, sock),
            name="drcode-mcp",
            daemon=True,
        )
        self._server = server
        self._thread = thread
        thread.start()
        if not _wait_healthy(port):
            self.stop()
            self._emit(f"MCP: server on port {port} did not become healthy")
            return False
        with _START_LOCK:
            # Second check, now that a port of ours is live: another
            # instance may have published its descriptor while we were
            # coming up (§6 — the only real race, closed right here).
            competitor = load_endpoint(self.project_root)
            if (
                competitor is not None
                and competitor.pid != os.getpid()
                and endpoint_is_live(competitor)
            ):
                self.stop()
                self._foreign_pid = competitor.pid
                self._emit(
                    "MCP: endpoint already served by another instance "
                    f"(pid {competitor.pid}) — not starting a second one"
                )
                return False
            endpoint = McpEndpoint(
                port=port,
                token=token,
                pid=os.getpid(),
            )
            try:
                save_endpoint(self.project_root, endpoint)
            except OSError as e:
                self.stop()
                self._emit(f"MCP: cannot write {endpoint_path(self.project_root)}: {e}")
                return False
        self._endpoint = endpoint
        self._foreign_pid = None
        self._emit(
            f"MCP: server listening on http://{LOOPBACK_HOST}:{port}/mcp "
            f"(endpoint: {endpoint_path(self.project_root)})"
        )
        return True

    def stop(self) -> None:
        """Shut OUR server down and remove the descriptor we published.

        Never raises and never touches a descriptor owned by somebody else
        (§3: its pid is not ours — deleting it would tear down a
        stranger's endpoint).
        """
        server = self._server
        thread = self._thread
        had_endpoint = self._endpoint is not None or self.running
        self._server = None
        self._thread = None
        self._endpoint = None
        try:
            if server is not None:
                server.should_exit = True
            if thread is not None:
                thread.join(STOP_TIMEOUT)
        except Exception as e:  # noqa: BLE001 - shutdown is best-effort
            self._emit(f"MCP: shutdown problem: {type(e).__name__}: {e}")
        self._close_socket()
        try:
            removed = remove_endpoint_if_owned_by(self.project_root, os.getpid())
        except OSError:  # pragma: no cover - defensive
            removed = False
        if had_endpoint:
            self._emit(
                "MCP: server stopped"
                + (f", removed {endpoint_path(self.project_root)}" if removed else "")
            )

    def apply(self, enabled: bool) -> bool:
        """Reconcile the service with the `mcp_enabled` setting (§2).

        The hot switch of the Settings palette: on → start, off → stop.
        Returns the resulting "a server of ours is running" state.
        """
        if enabled:
            return self.start()
        self.stop()
        return False

    # -- internals -----------------------------------------------------------

    def _build_app(self, token: str):
        """The ASGI app; the SDK server comes from the factory when given.

        Imported lazily: building the app drags in the whole `mcp` SDK and
        uvicorn, and a user who never enables the server should not pay
        that import on every TUI start.
        """
        from .app import build_app

        server = self._server_factory() if self._server_factory else None
        return build_app(token=token, server=server)

    @staticmethod
    def _serve(app, sock: socket.socket):
        """A configured `uvicorn.Server` for `app` on the bound `sock`.

        `log_config=None` is not cosmetic: uvicorn's default logging config
        REPLACES the root handlers process-wide, which would silence or
        duplicate drcode's own logging. `access_log=False` because every
        request of interest is reported as a service event in the TUI,
        and stderr spam from a background thread would corrupt the
        Textual screen.
        """
        import uvicorn

        config = uvicorn.Config(
            app,
            host=LOOPBACK_HOST,
            port=sock.getsockname()[1],
            log_config=None,
            log_level="warning",
            access_log=False,
            # The shutdown path waits for open connections; an idle GET-SSE
            # stream would otherwise hold it for the whole default timeout.
            timeout_graceful_shutdown=2,
        )
        return uvicorn.Server(config)

    def _serve_loop(self, server, sock: socket.socket) -> None:
        """The server thread's body: OWN the loop instead of `Server.run`'s
        hidden `asyncio.run`.

        `Server.run()` would create the loop invisibly and close it without
        telling anyone; the event bridge needs the very loop object to
        publish on (`loop`), so the thread runs the same
        `loop.run_until_complete(server.serve(...))` sequence explicitly
        and records the loop while it lives.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(server.serve(sockets=[sock]))
        finally:
            self._loop = None
            asyncio.set_event_loop(None)
            try:
                loop.close()
            except RuntimeError:  # pragma: no cover - defensive
                pass

    def _cleanup_failed_start(self) -> None:
        """Undo whatever a half-done start did (never raises)."""
        try:
            self._close_socket()
        except Exception:  # noqa: BLE001, S110 - already in a failure path
            pass
        self._server = None
        self._thread = None
        self._endpoint = None
        try:
            remove_endpoint_if_owned_by(self.project_root, os.getpid())
        except Exception:  # noqa: BLE001, S110 - already in a failure path
            pass

    def _close_socket(self) -> None:
        sock = self._socket
        self._socket = None
        if sock is None:
            return
        try:
            sock.close()
        except OSError:  # pragma: no cover - already closed by uvicorn
            pass

    def _emit(self, text: str) -> None:
        """Report one line to the sink; a broken sink is never fatal."""
        sink = self._event_sink
        if sink is None:
            return
        try:
            sink(text)
        except Exception:  # noqa: BLE001, S110 - reporting never breaks reporting
            pass


def _bind_ephemeral() -> socket.socket:
    """Bind a loopback socket on an ephemeral port (§1).

    The socket is bound HERE, not left to uvicorn: the port it picked is
    what gets published in the descriptor, and a port uvicorn chose later
    could not be announced in advance. `SO_REUSEADDR` is deliberately not
    set — a second instance must not be able to sneak onto the port.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((LOOPBACK_HOST, 0))
    except OSError:
        sock.close()
        raise
    return sock


def _wait_healthy(port: int, *, timeout: float = START_TIMEOUT) -> bool:
    """Wait for the freshly started server to answer `/healthz`.

    The probe is the same one the second-instance rule uses, so "started"
    here means exactly what "alive" means there — no second notion of
    readiness to keep in sync.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe_healthz(port):
            return True
        time.sleep(_START_POLL)
    return False


def disabled_note(project_root) -> str:
    """One-line description of the disabled state for the UI (§6).

    The token is never shown anywhere in the UI — only the path of the
    descriptor, which is what a client needs to find it (`mcp.json` is
    mode 0600, so reading it is the client-side authorization step).
    """
    path = endpoint_path(project_root)
    if path.exists():
        return f"MCP: off (stale endpoint file {path})"
    return "MCP: off"


__all__ = [
    "START_TIMEOUT",
    "STOP_TIMEOUT",
    "EventSink",
    "McpService",
    "disabled_note",
]
