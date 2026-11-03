"""aiohttp server: static web/ + the /ws WebSocket <-> PTY bridge.

Listens on 127.0.0.1 only, on an ephemeral port, and requires a random
per-run token for the WebSocket — pywebview's own HTTP helper has no
WebSocket support (and must not be used), so the shell carries its own
tiny server. The bridge speaks a minimal JSON control protocol over text
frames and raw terminal bytes over binary frames:

  client -> server (text, JSON):
    {"type": "start",   "project": "/path", "cols": 120, "rows": 30}
    {"type": "resize",  "cols": 120, "rows": 30}
    {"type": "restart", "cols": 120, "rows": 30}   # kill + respawn same project
    {"type": "kill"}
  server -> client (text, JSON):
    {"event": "started", "project": "..."}
    {"event": "exited",  "code": 0}              # 0 = clean, else crashed
    {"event": "error",   "message": "..."}

The shell holds at most ONE session at a time (one pty per window —
tabs are explicitly out of scope).
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


class ShellContext:
    """State shared by routes: token + the single live PTY session."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.token = secrets.token_urlsafe(24)
        self.session: PtySession | None = None
        self.current_project: str | None = None
        # Filled in by app.py / the server thread:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.base_url: str | None = None
        self.startup_error: str | None = None
        # Callables invoked in loop context (set by app.py):
        self.on_clean_exit: Callable[[], None] | None = None
        self.on_project_launched: Callable[[str], None] | None = None

    # -- session management ------------------------------------------------

    async def start_session(self, project: str, rows: int, cols: int) -> None:
        await self.stop_session()
        if not Path(project).is_dir():
            raise FileNotFoundError(f"not a directory: {project}")

        def _on_output(data: bytes) -> None:
            sink = self._ws_frame_sink
            if sink is not None:
                sink(data)

        def _on_exit(code: int) -> None:
            if self.session is session:
                self.session = None
            event = self._ws_event_sink
            if event is not None:
                event({"event": "exited", "code": code})
            if code == 0 and self.on_clean_exit is not None:
                self.on_clean_exit()

        session = PtySession(
            payload_argv(self.payload),
            project,
            on_output=_on_output,
            on_exit=_on_exit,
        )
        self.session = session
        if os.environ.get("DRCODE_DESKTOP_DEBUG") == "1":
            print(f"[desktop] spawning {self.payload} in {project}", flush=True)
        try:
            await session.start(initial_size=(rows, cols))
        except Exception:
            if os.environ.get("DRCODE_DESKTOP_DEBUG") == "1":
                import traceback

                traceback.print_exc()
            self.session = None
            raise
        self.current_project = project
        if os.environ.get("DRCODE_DESKTOP_DEBUG") == "1":
            print(
                f"[desktop] session started, pid={session._proc.pid if session._proc else None}",
                flush=True,
            )
        event = self._ws_event_sink
        if event is not None:
            event({"event": "started", "project": project})
        if self.on_project_launched is not None:
            self.on_project_launched(project)

    async def stop_session(self) -> None:
        if self.session is None:
            return
        session, self.session = self.session, None
        session.terminate()

    # Sinks are filled in by the ws handler while a browser is attached.
    # Typed as raw callables; None when no browser is connected.
    _ws_frame_sink: Callable[[bytes], None] | None = None
    _ws_event_sink: Callable[[dict[str, Any]], None] | None = None


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
        print("[desktop] ws connected", flush=True)
        ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)

        loop = asyncio.get_running_loop()
        out_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def _frame_sink(data: bytes) -> None:
            loop.call_soon_threadsafe(out_queue.put_nowait, ("bytes", data))

        def _event_sink(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(out_queue.put_nowait, ("json", event))

        ctx._ws_frame_sink = _frame_sink
        ctx._ws_event_sink = _event_sink

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
                    if ctx.session is not None:
                        ctx.session.write(msg.data)
                elif msg.type == WSMsgType.TEXT:
                    await _handle_control(ctx, msg.data)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            ctx._ws_frame_sink = None
            ctx._ws_event_sink = None
            out_queue.put_nowait(("close", None))
            pump_task.cancel()
        return ws

    return handler


async def _handle_control(ctx: ShellContext, raw: str) -> None:
    if os.environ.get("DRCODE_DESKTOP_DEBUG") == "1":
        print(f"[desktop] control: {raw[:200]}", flush=True)
    try:
        msg = json.loads(raw)
    except ValueError:
        return
    if not isinstance(msg, dict):
        return
    kind = msg.get("type")
    rows = int(msg.get("rows") or 30)
    cols = int(msg.get("cols") or 100)

    if kind in ("start", "restart"):
        project = str(msg.get("project") or ctx.current_project or "")
        try:
            await ctx.start_session(project, rows, cols)
        except Exception as exc:  # noqa: BLE001 - any spawn failure becomes an error event
            event = ctx._ws_event_sink
            if event is not None:
                event({"event": "error", "message": str(exc)})
    elif kind == "resize":
        if ctx.session is not None:
            ctx.session.resize(rows, cols)
    elif kind == "kill":
        await ctx.stop_session()


async def run_server(ctx: ShellContext) -> tuple[web.AppRunner, str]:
    """Start the aiohttp site on 127.0.0.1:<ephemeral>; return (runner, base_url)."""
    app = make_app(ctx)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, f"http://127.0.0.1:{port}"
