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
    {"type": "pause",  "tab": "..."}   # flow control: withhold binary frames
    {"type": "resume", "tab": "..."}   # ...release the withheld bytes again
  server -> client (text, JSON):
    {"event": "started", "project": "...", "tab": "..."}
    {"event": "exited",  "code": 0, "tab": "..."}  # 0 = clean, else crashed
    {"event": "error",   "message": "...", "tab": "..."}

Raw terminal bytes go out as binary frames. The pump coalesces whatever
has already accumulated in the outgoing queue into one frame (capped at
FRAME_MAX_BYTES) — under a burst of PTY output this is the difference
between hundreds of tiny ws frames per second and a few large ones. For
an even (non-bursty) stream the queue is empty at the moment the pump
wakes, so before draining it first waits out a small latency window
(COALESCE_WINDOW_SECONDS) that lets the stream pile up; an interactive
keystroke pays exactly that window and nothing more.
Order is strictly preserved; JSON events are never mixed into the byte
stream (separate text frames, in queue order relative to the bytes).

Flow control: the client counts the bytes it has queued into xterm.js but
not yet processed (the term.write callback) and reports congestion with
{"type": "pause"} — the pump then withholds that connection's binary
frames in memory (JSON events keep flowing; stdin is never blocked). The
withheld bytes are buffered up to PAUSE_BUFFER_MAX_BYTES; beyond the cap
the pause is given up and sending resumes wholesale, because a lost
terminal stream is worse than a lost throttle. The pause state lives on
the connection (not the tab), so a fresh ws attach to the same tab starts
unpaused.

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

# Upper bound of one coalesced binary frame (pump() drains the queue into
# a batch until it reaches this size). Bounds memory under a firehose of
# PTY output; the batch may overshoot by at most one PTY read chunk
# (READ_CHUNK in pty_session.py).
FRAME_MAX_BYTES = 256 * 1024

# Flow control: while the client has signaled {"type": "pause"}, the pump
# withholds binary frames in memory up to this many bytes. Past the cap it
# gives the pause up and resumes sending — terminal output must not be
# lost, an ignored pause only costs the throttling effect.
PAUSE_BUFFER_MAX_BYTES = 16 * 1024 * 1024

# Latency window: how long the pump may hold a byte batch before its first
# send when the queue is otherwise empty. An even (non-bursty) stream —
# exactly what a ``pytest -n auto`` run produces — delivers each PTY chunk
# while the pump is already asleep in ``out_queue.get()``, so the
# drain-what-is-queued pass alone still ships one tiny ws frame per chunk.
# The window gives such a stream a few milliseconds to pile up; a single
# keystroke pays exactly the window (≤ COALESCE_WINDOW_SECONDS added
# latency, imperceptible for a terminal), a firehose pays it at most once
# per frame (a batch that already reached FRAME_MAX_BYTES skips it).
COALESCE_WINDOW_SECONDS = 0.008


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
            # frame_sink is wired to PtySession._on_output, which fires from
            # _on_readable — a loop.add_reader callback, i.e. ALWAYS on THIS
            # loop's thread. A direct put is therefore safe (asyncio.Queue is
            # not thread-safe, but this never crosses a thread) and skips the
            # call_soon_threadsafe handle + wake-up per chunk on a busy output
            # stream. The event_sink below MUST keep call_soon_threadsafe:
            # exit/error events arrive from the reaper running in
            # asyncio.to_thread.
            out_queue.put_nowait(("bytes", data))

        def _event_sink(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(out_queue.put_nowait, ("json", event))

        def _flow_signal(direction: str) -> None:
            # Pause/resume arrive on the loop thread (ws text frames), so a
            # direct put is safe; it doubles as the wake-up the pump needs
            # while it sits paused in out_queue.get().
            out_queue.put_nowait(("flow", direction))

        state.frame_sink = _frame_sink
        state.event_sink = _event_sink

        async def pump() -> None:
            # Coalescing: under a burst of PTY output (a pytest -n auto run,
            # a big diff) the queue accumulates dozens of small chunks; one
            # ws frame per chunk (loopback TCP framing → WKWebView → JS
            # onmessage → term.write each time) makes the UI lag. Drain
            # everything ALREADY queued into one binary frame instead —
            # order is preserved, and an interactive print still pays only
            # the latency of the single frame it was anyway (the queue is
            # empty by then). The batch is capped so a firehose cannot grow
            # the buffer unboundedly; whatever does not fit stays queued
            # and goes out in the next frame. JSON events are NEVER merged
            # into the byte stream: they ride as separate text frames, in
            # queue order relative to the bytes around them.
            #
            # The drain-what-is-queued pass alone only collapses bursts
            # that piled up WHILE the pump was busy. An even stream —
            # pytest -n auto progress lines, a steady download log — hands
            # each chunk over while the pump is asleep in get(), so each
            # would still ship as its own tiny frame. Hence the latency
            # window: when the queue is empty right after the first chunk
            # (and the batch is under the cap), hold the batch for
            # COALESCE_WINDOW_SECONDS before draining, giving the stream
            # time to accumulate. The sleep runs on the loop, so
            # call_soon_threadsafe puts from the PTY reader thread land in
            # the queue during it — no extra yield needed. A single
            # keystroke pays exactly the window; a firehose never waits
            # past it (batch ≥ FRAME_MAX_BYTES skips the window and any
            # further draining).
            pending: tuple[str, Any] | None = None
            # Client-side flow control: while the client is paused, binary
            # output is withheld here instead of being sent (JSON events
            # still flow; stdin is untouched). None = not paused.
            held: bytearray | None = None

            async def _flush_held(buf: bytearray) -> None:
                # Send withheld bytes in FRAME_MAX_BYTES-sized frames,
                # order intact — the withheld stream keeps the framing
                # contract of the unpumped path.
                while len(buf) > FRAME_MAX_BYTES:
                    await ws.send_bytes(bytes(buf[:FRAME_MAX_BYTES]))
                    del buf[:FRAME_MAX_BYTES]
                if buf:
                    await ws.send_bytes(bytes(buf))

            while True:
                if pending is None:
                    kind, payload = await out_queue.get()
                else:
                    kind, payload = pending
                    pending = None
                if kind == "close":
                    return
                if kind == "flow":
                    # {"type": "pause"} / {"type": "resume"} from the client.
                    if payload == "pause":
                        held = bytearray()
                    elif held is not None:  # resume
                        await _flush_held(held)
                        held = None
                    continue
                if kind == "json":
                    # Control events are never flow-controlled: the client
                    # must learn about start/exit/error regardless of the
                    # byte-stream pause.
                    await ws.send_json(payload)
                    continue
                if held is not None:
                    # Paused: withhold bytes in memory. JSON events keep
                    # flowing (handled above); a resume flushes in order.
                    # ``buf`` is the same buffer under a non-Optional name —
                    # the pause can only end by clearing ``held`` below.
                    buf = held
                    buf.extend(payload)
                    overflow = False
                    while True:
                        try:
                            nkind, npayload = out_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if nkind == "close":
                            return
                        if nkind == "flow":
                            if npayload == "resume":
                                await _flush_held(buf)
                                held = None
                                break
                            continue  # pause while already paused: no-op
                        if nkind == "json":
                            await ws.send_json(npayload)
                            continue
                        buf.extend(npayload)
                        if len(buf) > PAUSE_BUFFER_MAX_BYTES:
                            overflow = True
                            break
                    if overflow:
                        # The client cannot keep up even with the buffer:
                        # resume sending wholesale — losing the throttle is
                        # recoverable, losing terminal output is not. The
                        # client's later resume becomes a harmless no-op.
                        await _flush_held(buf)
                        held = None
                    continue
                # bytes: coalesce everything already queued, in order —
                # and, via the latency window, what an even stream
                # delivers within the window before the first send.
                batch = bytearray(payload)
                if out_queue.empty() and len(batch) < FRAME_MAX_BYTES:
                    await asyncio.sleep(COALESCE_WINDOW_SECONDS)
                while len(batch) < FRAME_MAX_BYTES:
                    try:
                        nkind, npayload = out_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if nkind == "bytes":
                        batch.extend(npayload)
                    else:
                        # A json event (or the close/flow sentinel): it must
                        # be its own text frame (or handled on its own),
                        # AFTER the bytes queued before it — hold it back,
                        # flush the batch, handle next.
                        pending = (nkind, npayload)
                        break
                await ws.send_bytes(bytes(batch))

        pump_task = loop.create_task(pump())
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    # Raw stdin bytes from the terminal widget.
                    if state.session is not None:
                        state.session.write(msg.data)
                elif msg.type == WSMsgType.TEXT:
                    await _handle_control(
                        ctx, msg.data, tab_id, flow_signal=_flow_signal
                    )
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


async def _handle_control(
    ctx: ShellContext,
    raw: str,
    conn_tab: str,
    flow_signal: Callable[[str], None] | None = None,
) -> None:
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
    elif kind in ("pause", "resume") and flow_signal is not None:
        # Client-side flow control of the binary stream. The pause state
        # lives in the connection's pump (the "flow" queue item), NOT on
        # the tab: a reconnect to the same tab gets a fresh pump and is
        # therefore never stuck paused. Unknown payload values are ignored.
        flow_signal(kind)


async def run_server(ctx: ShellContext) -> tuple[web.AppRunner, str]:
    """Start the aiohttp site on 127.0.0.1:<ephemeral>; return (runner, base_url)."""
    app = make_app(ctx)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, f"http://127.0.0.1:{port}"
