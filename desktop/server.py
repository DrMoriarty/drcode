"""aiohttp server: static web/ + the /ws WebSocket <-> PTY bridge.

Listens on 127.0.0.1 only, on an ephemeral port, and requires a random
per-run token for the WebSocket — pywebview's own HTTP helper has no
WebSocket support (and must not be used), so the shell carries its own
tiny server. The bridge speaks a minimal JSON control protocol over text
frames and raw terminal bytes over binary frames. Tabs: every browser
tab (one per project) uses its OWN WebSocket connection, addressed by a
client-chosen ``tab`` id (query param of the connect URL; control
messages may restate it). Events are always marked with the same tab:

  connect:   /ws?token=<token>&tab=<tab-id>
  client -> server (text, JSON):
    {"type": "start",   "project": "/path", "cols": 120, "rows": 30}
    {"type": "resize",  "cols": 120, "rows": 30}
    {"type": "restart", "cols": 120, "rows": 30}   # kill + respawn same project
    {"type": "kill"}
  server -> client (text, JSON):
    {"event": "started", "project": "...", "tab": "..."}
    {"event": "exited",  "code": 0, "tab": "..."}  # 0 = clean, else crashed
    {"event": "error",   "message": "...", "tab": "..."}

Each tab holds at most one PTY session; the context holds any number of
tabs (a tab's session outlives a ws reconnect by design — the client
kills it explicitly with the ``kill`` control message).

A project (normalized absolute path — realpath, so symlinks and
``/private/tmp``-style aliases resolve to the same key) can back at most
ONE live session across all tabs: drcode keeps its state in the shared
``<project>/.drcode/`` (tasks, worktrees, search index, settings) and
two instances over one project would race. ``start``/``restart`` on an
occupied project is answered with an ``{"event": "error", ...}`` frame
instead of a spawn.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web
from pty_session import PtySession, payload_argv

WEB_DIR = Path(__file__).resolve().parent / "web"

# Tab id for connections that do not name one (tests, first-run probes).
DEFAULT_TAB = "default"


class TabState:
    """One tab: its PTY session plus the sinks of the attached websocket."""

    def __init__(self, tab_id: str) -> None:
        self.tab_id = tab_id
        self.session: PtySession | None = None
        self.project: str | None = None
        # Filled in by the ws handler while a browser tab is attached.
        # Typed as raw callables; None when no browser is connected.
        self.frame_sink: Callable[[bytes], None] | None = None
        self.event_sink: Callable[[dict[str, Any]], None] | None = None


class ShellContext:
    """State shared by routes: token + the live PTY sessions, one per tab."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.token = secrets.token_urlsafe(24)
        self.tabs: dict[str, TabState] = {}
        # Filled in by app.py / the server thread:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.base_url: str | None = None
        self.startup_error: str | None = None
        # Callables invoked in loop context (set by app.py):
        self.on_clean_exit: Callable[[str], None] | None = None
        self.on_project_launched: Callable[[str], None] | None = None

    # -- tab registry -------------------------------------------------------

    def tab(self, tab_id: str) -> TabState:
        """Get (creating on demand) the state for a tab id."""
        state = self.tabs.get(tab_id)
        if state is None:
            state = TabState(tab_id)
            self.tabs[tab_id] = state
        return state

    def live_project(self) -> str | None:
        """Project of the first tab with a live session (informational)."""
        for state in self.tabs.values():
            if state.session is not None and state.project:
                return state.project
        return None

    def project_tab(self, project: str, exclude_tab: str | None = None) -> str | None:
        """Tab id already running ``project`` in a live session, or None.

        Comparison is on realpath-normalized absolute paths, so ``/tmp/x``,
        a symlink to it, and macOS's ``/private/tmp/x`` all count as the
        same project. A tab whose session has exited (crash, kill) does
        not hold the project anymore. ``exclude_tab`` lets a tab restart
        its own project.
        """
        if not project:
            return None
        real = os.path.realpath(project)
        for tab_id, state in self.tabs.items():
            if tab_id == exclude_tab or state.session is None or not state.project:
                continue
            if os.path.realpath(state.project) == real:
                return tab_id
        return None

    # -- session management -------------------------------------------------

    async def start_session(
        self, tab_id: str, project: str, rows: int, cols: int
    ) -> None:
        """Spawn the payload in ``project`` for ``tab_id``, replacing any
        previous session of that tab."""
        await self.stop_session(tab_id)
        if not Path(project).is_dir():
            raise FileNotFoundError(f"not a directory: {project}")
        state = self.tab(tab_id)

        def _on_output(data: bytes) -> None:
            sink = state.frame_sink
            if sink is not None:
                sink(data)

        def _on_exit(code: int) -> None:
            if state.session is session:
                state.session = None
            event = state.event_sink
            if event is not None:
                event({"event": "exited", "code": code, "tab": tab_id})
            if code == 0 and self.on_clean_exit is not None:
                self.on_clean_exit(tab_id)

        session = PtySession(
            payload_argv(self.payload),
            project,
            on_output=_on_output,
            on_exit=_on_exit,
        )
        state.session = session
        # Claim the project BEFORE the await so two near-simultaneous
        # starts cannot both pass the project_tab() check (the second one
        # then sees this tab's session even though the PTY is still
        # booting). A failed start clears the session; the stale project
        # string is harmless (never counted while session is None) and is
        # overwritten by the next successful spawn.
        state.project = project
        if os.environ.get("DRCODE_DESKTOP_DEBUG") == "1":
            print(
                f"[desktop] spawning {self.payload} in {project} (tab {tab_id})",
                flush=True,
            )
        try:
            await session.start(initial_size=(rows, cols))
        except Exception:
            if os.environ.get("DRCODE_DESKTOP_DEBUG") == "1":
                import traceback

                traceback.print_exc()
            if state.session is session:
                state.session = None
            raise
        if os.environ.get("DRCODE_DESKTOP_DEBUG") == "1":
            print(
                f"[desktop] session started, tab={tab_id}, "
                f"pid={session._proc.pid if session._proc else None}",
                flush=True,
            )
        event = state.event_sink
        if event is not None:
            event({"event": "started", "project": project, "tab": tab_id})
        if self.on_project_launched is not None:
            self.on_project_launched(project)

    async def stop_session(self, tab_id: str) -> None:
        state = self.tabs.get(tab_id)
        if state is None or state.session is None:
            return
        session, state.session = state.session, None
        session.terminate()

    async def stop_all(self) -> None:
        """Terminate every tab's session (app shutdown)."""
        for tab_id in list(self.tabs):
            await self.stop_session(tab_id)


def make_app(ctx: ShellContext) -> web.Application:
    app = web.Application()
    app.router.add_get("/", _index)
    app.router.add_static("/static/", WEB_DIR)
    app.router.add_get("/ws", _websocket(ctx))
    return app


async def _index(request: web.Request) -> web.FileResponse:
    print(f"[desktop] GET / from {request.remote}", flush=True)
    return web.FileResponse(WEB_DIR / "index.html")


def _websocket(
    ctx: ShellContext,
) -> Callable[[web.Request], Awaitable[web.WebSocketResponse]]:
    async def handler(request: web.Request) -> web.WebSocketResponse:
        if request.query.get("token") != ctx.token:
            print("[desktop] ws refused: bad token", flush=True)
            raise web.HTTPForbidden(text="bad token")
        tab_id = request.query.get("tab") or DEFAULT_TAB
        print(f"[desktop] ws connected (tab {tab_id})", flush=True)
        ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)

        state = ctx.tab(tab_id)
        loop = asyncio.get_running_loop()
        out_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def _frame_sink(data: bytes) -> None:
            loop.call_soon_threadsafe(out_queue.put_nowait, ("bytes", data))

        def _event_sink(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(out_queue.put_nowait, ("json", event))

        state.frame_sink = _frame_sink
        state.event_sink = _event_sink

        async def pump() -> None:
            while True:
                kind, payload = await out_queue.get()
                if kind == "close":
                    return
                if kind == "bytes":
                    await ws.send_bytes(payload)
                else:
                    await ws.send_json(payload)

        pump_task = loop.create_task(pump())
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    # Raw stdin bytes from the terminal widget.
                    if state.session is not None:
                        state.session.write(msg.data)
                elif msg.type == WSMsgType.TEXT:
                    await _handle_control(ctx, msg.data, tab_id)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            # Clear the sinks only if a newer connection has not already
            # replaced them (ws reconnect to the same tab).
            if state.frame_sink is _frame_sink:
                state.frame_sink = None
            if state.event_sink is _event_sink:
                state.event_sink = None
            out_queue.put_nowait(("close", None))
            pump_task.cancel()
        return ws

    return handler


async def _handle_control(ctx: ShellContext, raw: str, conn_tab: str) -> None:
    if os.environ.get("DRCODE_DESKTOP_DEBUG") == "1":
        print(f"[desktop] control: {raw[:200]}", flush=True)
    try:
        msg = json.loads(raw)
    except ValueError:
        return
    if not isinstance(msg, dict):
        return
    kind = msg.get("type")
    # The connection's tab is the default; a message may restate (or, from
    # a shared connection, explicitly pick) the target tab.
    tab_id = str(msg.get("tab") or conn_tab)
    state = ctx.tab(tab_id)
    rows = int(msg.get("rows") or 30)
    cols = int(msg.get("cols") or 100)

    if kind in ("start", "restart"):
        project = str(msg.get("project") or state.project or "")
        # One live drcode instance per project: its state lives in the
        # shared <project>/.drcode/ (tasks, worktrees, search index), so
        # a second tab over the same path would race it. A tab may always
        # restart its own project (exclude_tab below).
        holder = ctx.project_tab(project, exclude_tab=tab_id)
        if holder is not None:
            event = state.event_sink
            if event is not None:
                event(
                    {
                        "event": "error",
                        "message": (
                            "This project is already open in another tab: "
                            f"{project} (tab {holder}). Close that tab first."
                        ),
                        "tab": tab_id,
                    }
                )
            return
        try:
            await ctx.start_session(tab_id, project, rows, cols)
        except Exception as exc:  # noqa: BLE001 - any spawn failure becomes an error event
            event = state.event_sink
            if event is not None:
                event({"event": "error", "message": str(exc), "tab": tab_id})
    elif kind == "resize":
        if state.session is not None:
            state.session.resize(rows, cols)
    elif kind == "kill":
        await ctx.stop_session(tab_id)


async def run_server(ctx: ShellContext) -> tuple[web.AppRunner, str]:
    """Start the aiohttp site on 127.0.0.1:<ephemeral>; return (runner, base_url)."""
    app = make_app(ctx)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, f"http://127.0.0.1:{port}"
