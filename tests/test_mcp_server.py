"""Embedded MCP server: transport, auth and lifecycle
(docs/mcp-server-design.md §1/§2/§3/§6).

Three layers are exercised here, each at the cheapest level that still
proves the design:

- `drcode.mcp.endpoint` — the `.drcode/mcp.json` descriptor, the pid probe
  and the `/healthz` probe. Pure I/O, no server: a real file, a real
  process for the "is it alive?" half and a stdlib HTTP server for the
  probe half.
- `drcode.mcp.guard` + `drcode.mcp.app` — the ASGI stack, driven through
  `httpx.ASGITransport` (in-process, no socket). Auth (401 without and
  with a wrong bearer token, a valid one passes), the Origin/Host
  DNS-rebinding checks and `/healthz`'s token-free answer.
- `drcode.mcp.service` — the lifecycle controller on a REAL ephemeral
  loopback socket, including the one-server-per-project refusal of §3 and
  one end-to-end smoke test with the SDK's own Streamable HTTP client.

The tests never touch the network beyond loopback and never rely on a
fixed port (every server here binds `127.0.0.1:0`).
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from drcode.gitignore import GITIGNORE_ENTRIES
from drcode.mcp import (
    HEALTH_PATH,
    LOOPBACK_HOST,
    MCP_PATH,
    McpEndpoint,
    McpService,
    disabled_note,
    endpoint_is_live,
    endpoint_path,
    load_endpoint,
    new_token,
    probe_healthz,
    remove_endpoint_if_owned_by,
    save_endpoint,
)
from drcode.mcp.app import HEALTH_PAYLOAD, build_app, build_server
from drcode.mcp.guard import (
    GuardMiddleware,
    is_allowed_origin,
    is_loopback_host,
    token_matches,
)
from drcode.settings import ProjectSettings

#: The token every ASGI-level test publishes. 64 hex chars like the real
#: one (§2: `secrets.token_hex(32)`), so nothing depends on the shape of a
#: shorter test value.
TOKEN = "0123456789abcdef" * 4

#: Base URL of the in-process ASGI client: a loopback host (the guard
#: rejects anything else) with a port that is never actually connected to.
BASE_URL = "http://127.0.0.1:9"

#: Headers every real MCP client sends: the SDK's Streamable HTTP
#: transport negotiates between JSON and SSE, so `Accept` carries both.
MCP_ACCEPT = "application/json, text/event-stream"

#: A minimal but VALID `initialize` call — the first request of every
#: stateful Streamable HTTP session (the SDK answers it with a session id).
INITIALIZE_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "drcode-tests", "version": "1"},
    },
}


def _auth(token: str = TOKEN) -> dict[str, str]:
    """The bearer header of a valid (or deliberately wrong) request."""
    return {"Authorization": f"Bearer {token}", "Accept": MCP_ACCEPT}


# -- fixtures ------------------------------------------------------------------


@pytest.fixture
def sdk_server():
    """An SDK server with one trivial tool, so a client has something to call."""
    server = build_server()

    @server.tool()
    def ping() -> str:
        """Answer "pong" — the smoke test's proof of a live round trip."""
        return "pong"

    return server


@pytest.fixture
def guarded_app(sdk_server):
    """The full ASGI stack (healthz + guarded Streamable HTTP) with TOKEN."""
    return build_app(token=TOKEN, server=sdk_server)


@pytest.fixture
def live_healthz() -> Iterator[int]:
    """A real loopback HTTP server answering `{"ok": true}` on `/healthz`.

    Only `/healthz` is served: the one-server-per-project rule needs *some*
    live endpoint to probe, and using a stdlib server (instead of a second
    real MCP service) keeps the refusal test free of a second SDK stack.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # http.server's dispatch contract
            if self.path != HEALTH_PATH:
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(HEALTH_PAYLOAD).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Silence per-request logging (it would land in the pytest log)."""

    httpd = ThreadingHTTPServer((LOOPBACK_HOST, 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def foreign_process() -> Iterator[subprocess.Popen[bytes]]:
    """A live, unrelated process whose pid can be published in a descriptor.

    Blocked on stdin, so it stays alive for the whole test and dies with
    the fixture — the "another TUI already serves this project" half of §3.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
    )
    try:
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - hostile env
            proc.kill()
            proc.wait(timeout=5)


# -- /healthz ------------------------------------------------------------------


async def test_healthz_answers_without_a_token(guarded_app) -> None:
    """§2: `/healthz` is the token-free liveness probe — nothing else in it."""
    async with guarded_app.router.lifespan_context(guarded_app):
        transport = httpx.ASGITransport(app=guarded_app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.get(HEALTH_PATH)
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    # A health answer must not leak anything else (§2: "и больше ничего").
    assert response.json() == HEALTH_PAYLOAD


async def test_healthz_ignores_a_bearer_token(guarded_app) -> None:
    """The probe sits OUTSIDE the guard: a token cannot make it fail either."""
    async with guarded_app.router.lifespan_context(guarded_app):
        transport = httpx.ASGITransport(app=guarded_app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.get(HEALTH_PATH, headers=_auth("nonsense"))
    assert response.status_code == 200


# -- bearer-token authentication (§2) -----------------------------------------


@pytest.mark.parametrize("method", ["POST", "GET", "DELETE"])
async def test_mcp_without_a_token_is_401(guarded_app, method: str) -> None:
    """Every `/mcp` verb is guarded — the GET-SSE stream and DELETE included."""
    async with guarded_app.router.lifespan_context(guarded_app):
        transport = httpx.ASGITransport(app=guarded_app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.request(method, MCP_PATH, json=INITIALIZE_BODY)
    assert response.status_code == 401
    # §2: 401 "without details" — no hint about the expected token, no
    # mention of the descriptor file, no echo of what was presented.
    assert response.content == b""
    assert TOKEN not in response.headers.get("www-authenticate", "")


@pytest.mark.parametrize("method", ["POST", "GET", "DELETE"])
async def test_mcp_with_a_wrong_token_is_401(guarded_app, method: str) -> None:
    async with guarded_app.router.lifespan_context(guarded_app):
        transport = httpx.ASGITransport(app=guarded_app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.request(
                method, MCP_PATH, json=INITIALIZE_BODY, headers=_auth("f" * 64)
            )
    assert response.status_code == 401
    assert response.content == b""


async def test_mcp_with_a_non_bearer_scheme_is_401(guarded_app) -> None:
    """`Authorization: Basic …` (or a bare token) never authenticates."""
    async with guarded_app.router.lifespan_context(guarded_app):
        transport = httpx.ASGITransport(app=guarded_app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            basic = await client.post(
                MCP_PATH,
                json=INITIALIZE_BODY,
                headers={"Authorization": f"Basic {TOKEN}"},
            )
            bare = await client.post(
                MCP_PATH,
                json=INITIALIZE_BODY,
                headers={"Authorization": TOKEN},
            )
    assert basic.status_code == 401
    assert bare.status_code == 401


async def test_mcp_with_a_valid_token_reaches_the_sdk(guarded_app) -> None:
    """A valid token passes the guard: the SDK answers `initialize`.

    The `Mcp-Session-Id` response header is the SDK's own proof that the
    request was handled by the stateful Streamable HTTP transport — not
    just let through to a 4xx.
    """
    async with guarded_app.router.lifespan_context(guarded_app):
        transport = httpx.ASGITransport(app=guarded_app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.post(
                MCP_PATH, json=INITIALIZE_BODY, headers=_auth()
            )
    assert response.status_code == 200
    assert response.headers.get("mcp-session-id")
    assert "drcode" in response.text


# -- Origin/Host validation (DNS-rebinding defence, §2) -----------------------


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example",
        "http://evil.example:8080",
        "https://127.0.0.1:1234",  # right host, wrong scheme
        "null",  # opaque origin of a sandboxed/file page
        "http://127.0.0.1.evil.example",
        "http://notlocalhost",
    ],
)
async def test_foreign_origin_is_rejected(guarded_app, origin: str) -> None:
    """A browser Origin that is not loopback is refused even WITH a valid token."""
    headers = {**_auth(), "Origin": origin}
    async with guarded_app.router.lifespan_context(guarded_app):
        transport = httpx.ASGITransport(app=guarded_app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.post(
                MCP_PATH, json=INITIALIZE_BODY, headers=headers
            )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "origin",
    ["http://127.0.0.1:51234", "http://localhost:51234", "http://[::1]:51234"],
)
async def test_loopback_origin_is_accepted(guarded_app, origin: str) -> None:
    headers = {**_auth(), "Origin": origin}
    async with guarded_app.router.lifespan_context(guarded_app):
        transport = httpx.ASGITransport(app=guarded_app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.post(
                MCP_PATH, json=INITIALIZE_BODY, headers=headers
            )
    assert response.status_code == 200


async def test_absent_origin_is_accepted(guarded_app) -> None:
    """A CLI client sends no Origin at all — the header is browser-only."""
    async with guarded_app.router.lifespan_context(guarded_app):
        transport = httpx.ASGITransport(app=guarded_app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.post(
                MCP_PATH, json=INITIALIZE_BODY, headers=_auth()
            )
    assert response.status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "evil.example:51234", "10.0.0.5"])
async def test_foreign_host_is_rejected(guarded_app, host: str) -> None:
    """DNS rebinding: the Host header is what tells a rebind attempt apart."""
    headers = {**_auth(), "Host": host}
    async with guarded_app.router.lifespan_context(guarded_app):
        transport = httpx.ASGITransport(app=guarded_app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            response = await client.post(
                MCP_PATH, json=INITIALIZE_BODY, headers=headers
            )
    assert response.status_code == 421


# -- guard helpers (unit) -----------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "127.0.0.1",
        "127.0.0.1:51234",
        "localhost",
        "localhost:1",
        "::1",
        "[::1]",
        "[::1]:80",
    ],
)
def test_loopback_host_accepts_loopback_spellings(value: str) -> None:
    assert is_loopback_host(value) is True


@pytest.mark.parametrize(
    "value",
    ["", None, "evil.example", "127.0.0.1.evil.example", "10.0.0.1:80", "0.0.0.0"],
)
def test_loopback_host_rejects_everything_else(value: str | None) -> None:
    assert is_loopback_host(value) is False


def test_allowed_origin_semantics() -> None:
    assert is_allowed_origin(None) is True  # non-browser client
    assert is_allowed_origin("http://127.0.0.1:1") is True
    assert is_allowed_origin("http://localhost") is True
    assert is_allowed_origin("https://127.0.0.1") is False  # wrong scheme
    assert is_allowed_origin("http://evil.example") is False
    assert is_allowed_origin("null") is False


def test_token_matches_is_exact_and_fails_closed() -> None:
    assert token_matches(TOKEN, TOKEN) is True
    assert token_matches(TOKEN, TOKEN[:-1]) is False
    assert token_matches(TOKEN, TOKEN + "0") is False
    assert token_matches(TOKEN, TOKEN.upper()) is False
    assert token_matches(TOKEN, None) is False
    # A misconfigured server (empty expectation) must never accept anything.
    assert token_matches("", "") is False
    assert token_matches("", TOKEN) is False


async def test_guard_passes_non_http_scopes_through() -> None:
    """Lifespan/websocket scopes are not HTTP requests: never guarded.

    The app's lifespan is what starts the SDK's session manager — a guard
    that swallowed it would leave every request with "task group is not
    initialized".
    """
    seen: list[str] = []

    async def inner(scope, receive, send) -> None:
        seen.append(str(scope["type"]))

    middleware = GuardMiddleware(inner, token=TOKEN)
    await middleware({"type": "lifespan"}, None, None)  # type: ignore[arg-type]
    await middleware({"type": "websocket"}, None, None)  # type: ignore[arg-type]
    assert seen == ["lifespan", "websocket"]


async def test_guard_only_covers_the_mcp_path() -> None:
    """Anything the parent app routes elsewhere passes unauthenticated.

    `/healthz` reaches the guard today only if the routing changes, but the
    path-exact check makes the middleware safe to mount over a wider tree.
    """
    seen: list[str] = []

    async def inner(scope, receive, send) -> None:
        seen.append(str(scope["path"]))

    middleware = GuardMiddleware(inner, token=TOKEN)
    await middleware({"type": "http", "path": "/healthz"}, None, None)  # type: ignore[arg-type]
    assert seen == ["/healthz"]


# -- the descriptor: .drcode/mcp.json (§2/§3) ---------------------------------


def test_endpoint_path_is_inside_the_state_dir(project_root: Path) -> None:
    assert endpoint_path(project_root) == project_root / ".drcode" / "mcp.json"


def test_mcp_json_is_gitignored() -> None:
    """The descriptor carries a bearer token — it must never be committable."""
    assert ".drcode/mcp.json" in GITIGNORE_ENTRIES


def test_new_token_shape() -> None:
    """32 bytes hex-encoded, and two calls never agree (§2)."""
    first = new_token()
    assert len(first) == 64
    assert all(c in "0123456789abcdef" for c in first)
    assert new_token() != first


def test_save_endpoint_roundtrip(project_root: Path) -> None:
    endpoint = McpEndpoint(port=51234, token=TOKEN, pid=os.getpid())
    path = save_endpoint(project_root, endpoint)
    assert path == endpoint_path(project_root)
    loaded = load_endpoint(project_root)
    assert loaded is not None
    assert loaded.port == 51234
    assert loaded.token == TOKEN
    assert loaded.pid == os.getpid()
    assert loaded.started_at == endpoint.started_at


def test_saved_endpoint_is_mode_0600(project_root: Path) -> None:
    """The token is a secret: nobody but the user may read the file."""
    save_endpoint(project_root, McpEndpoint(port=1, token=TOKEN, pid=1))
    mode = endpoint_path(project_root).stat().st_mode & 0o777
    assert mode == 0o600


def test_save_endpoint_leaves_no_temp_file(project_root: Path) -> None:
    save_endpoint(project_root, McpEndpoint(port=1, token=TOKEN, pid=1))
    leftovers = [
        p
        for p in (project_root / ".drcode").iterdir()
        if p != endpoint_path(project_root)
    ]
    assert leftovers == []


def test_missing_endpoint_is_none(project_root: Path) -> None:
    assert load_endpoint(project_root) is None


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json at all",
        "[]",
        '"a string"',
        '{"port": 0, "token": "x", "pid": 1}',  # port out of range
        '{"port": 1, "token": "", "pid": 1}',  # empty token
        '{"port": 1, "token": "x"}',  # pid missing
    ],
)
def test_corrupted_endpoint_is_none(project_root: Path, content: str) -> None:
    """A corrupted descriptor is "no descriptor" — never a crash (§2)."""
    endpoint_path(project_root).parent.mkdir(parents=True, exist_ok=True)
    endpoint_path(project_root).write_text(content, encoding="utf-8")
    assert load_endpoint(project_root) is None


def test_endpoint_urls(project_root: Path) -> None:
    endpoint = McpEndpoint(port=51234, token=TOKEN, pid=1)
    assert endpoint.url == "http://127.0.0.1:51234/mcp"
    assert endpoint.health_url == "http://127.0.0.1:51234/healthz"


# -- pid liveness --------------------------------------------------------------


def test_pid_alive_for_our_own_process() -> None:
    from drcode.mcp.endpoint import pid_alive

    assert pid_alive(os.getpid()) is True


def test_pid_alive_for_a_reaped_process(foreign_process) -> None:
    """A pid of an exited (reaped) process is dead — the stale-file case."""
    from drcode.mcp.endpoint import pid_alive

    pid = foreign_process.pid
    assert pid_alive(pid) is True
    foreign_process.terminate()
    foreign_process.wait(timeout=5)
    assert pid_alive(pid) is False


@pytest.mark.parametrize("pid", [0, -1, -12345])
def test_pid_alive_rejects_nonsense(pid: int) -> None:
    """Signal 0 to 0/-1 would target a whole process group — refused upfront."""
    from drcode.mcp.endpoint import pid_alive

    assert pid_alive(pid) is False


# -- /healthz probe ------------------------------------------------------------


def test_probe_healthz_against_a_live_endpoint(live_healthz: int) -> None:
    assert probe_healthz(live_healthz) is True


def test_probe_healthz_on_a_closed_port() -> None:
    """A bound-then-closed port: nothing listens, the probe answers False."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((LOOPBACK_HOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    assert probe_healthz(port) is False


def test_probe_healthz_rejects_a_wrong_payload() -> None:
    """`{"ok": true}` is the contract — any other 200 answer is not alive."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # http.server's dispatch contract
            body = b'{"ok": false}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Silence logging."""

    httpd = ThreadingHTTPServer((LOOPBACK_HOST, 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        assert probe_healthz(httpd.server_address[1]) is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# -- endpoint_is_live: both halves of §3 --------------------------------------


def test_endpoint_is_live_needs_a_live_pid_and_an_answering_port(
    live_healthz: int, foreign_process
) -> None:
    live = McpEndpoint(port=live_healthz, token=TOKEN, pid=foreign_process.pid)
    assert endpoint_is_live(live) is True
    # A live pid with a silent port (a crashed server behind a reused pid).
    silent = McpEndpoint(port=live_healthz + 1, token=TOKEN, pid=foreign_process.pid)
    assert endpoint_is_live(silent) is False
    # An answering port with a dead owner pid.
    dead = McpEndpoint(port=live_healthz, token=TOKEN, pid=2**22 + 12345)
    assert endpoint_is_live(dead) is False


# -- remove_endpoint_if_owned_by (§3/§6) --------------------------------------


def test_remove_endpoint_only_when_pid_is_ours(project_root: Path) -> None:
    save_endpoint(project_root, McpEndpoint(port=1, token=TOKEN, pid=os.getpid()))
    # Somebody else's descriptor: never touched.
    assert remove_endpoint_if_owned_by(project_root, os.getpid() + 1) is False
    assert endpoint_path(project_root).exists()
    # Ours: removed.
    assert remove_endpoint_if_owned_by(project_root, os.getpid()) is True
    assert not endpoint_path(project_root).exists()


def test_remove_endpoint_is_a_no_op_without_a_descriptor(project_root: Path) -> None:
    assert remove_endpoint_if_owned_by(project_root, os.getpid()) is False


def test_disabled_note_reports_a_stale_file(project_root: Path) -> None:
    """§6: the UI says "off", and points at a leftover descriptor if any."""
    assert disabled_note(project_root) == "MCP: off"
    save_endpoint(project_root, McpEndpoint(port=1, token=TOKEN, pid=1))
    assert "off" in disabled_note(project_root)
    assert "mcp.json" in disabled_note(project_root)


# -- McpService: lifecycle on a real socket (§1/§3/§6) -----------------------


@pytest.fixture
def events() -> list[str]:
    """Every service-event line the lifecycle produced, in order."""
    return []


def _service(project_root: Path, events: list[str], factory) -> McpService:
    return McpService(project_root, event_sink=events.append, server_factory=factory)


def test_start_publishes_the_endpoint_and_stop_removes_it(
    project_root: Path, events: list[str], sdk_server
) -> None:
    """The §6 lifecycle: port → mcp.json → server, and the mirror-image teardown."""
    service = _service(project_root, events, lambda: sdk_server)
    try:
        assert service.start() is True
        assert service.running is True
        port = service.port
        assert port is not None and port > 0
        assert service.url == f"http://{LOOPBACK_HOST}:{port}{MCP_PATH}"
        # Something is actually listening on it.
        assert probe_healthz(port) is True
        published = load_endpoint(project_root)
        assert published is not None
        assert published.port == port
        assert published.pid == os.getpid()
        assert published.token == TOKEN or len(published.token) == 64
        assert service.endpoint_file.endswith("mcp.json")
        assert "listening" in events[0]
    finally:
        service.stop()
    assert service.running is False
    assert service.port is None
    assert not endpoint_path(project_root).exists()
    assert any("stopped" in line for line in events)


def test_start_is_idempotent(project_root: Path, events: list[str], sdk_server) -> None:
    service = _service(project_root, events, lambda: sdk_server)
    try:
        assert service.start() is True
        port = service.port
        assert service.start() is True
        assert service.port == port  # the same server, no second bind
        assert sum("listening" in line for line in events) == 1
    finally:
        service.stop()


def test_stop_is_idempotent(project_root: Path, events: list[str], sdk_server) -> None:
    """A repeated stop must never raise — teardown may run twice."""
    service = _service(project_root, events, lambda: sdk_server)
    assert service.start() is True
    service.stop()
    service.stop()
    assert service.running is False


def test_apply_switches_the_server_on_and_off(
    project_root: Path, events: list[str], sdk_server
) -> None:
    """The hot switch behind the Settings toggle (§2)."""
    service = _service(project_root, events, lambda: sdk_server)
    try:
        assert service.apply(True) is True
        assert service.running is True
        assert service.apply(False) is False
        assert service.running is False
        assert not endpoint_path(project_root).exists()
        # Applying "off" again is harmless.
        assert service.apply(False) is False
    finally:
        service.stop()


def test_start_recovers_after_a_dead_descriptor(
    project_root: Path, events: list[str], sdk_server
) -> None:
    """§3 rule 2: a stale descriptor (dead pid) is overwritten, not honoured."""
    stale = McpEndpoint(port=1, token="a" * 64, pid=2**22 + 4242)
    save_endpoint(project_root, stale)
    service = _service(project_root, events, lambda: sdk_server)
    try:
        assert service.start() is True
        assert service.foreign_pid is None
        published = load_endpoint(project_root)
        assert published is not None
        assert published.pid == os.getpid()
        assert published.port == service.port
        # §2: the surviving token is reused, so configured clients keep working.
        assert published.token == stale.token
    finally:
        service.stop()


def test_start_refuses_when_another_live_instance_serves_the_project(
    project_root: Path,
    events: list[str],
    sdk_server,
    live_healthz: int,
    foreign_process,
) -> None:
    """§3 rule 1: one server = one project — and the descriptor stays untouched."""
    foreign = McpEndpoint(port=live_healthz, token="b" * 64, pid=foreign_process.pid)
    save_endpoint(project_root, foreign)
    before = endpoint_path(project_root).read_text(encoding="utf-8")
    service = _service(project_root, events, lambda: sdk_server)
    try:
        assert service.start() is False
        assert service.running is False
        assert service.port is None
        assert service.foreign_pid == foreign_process.pid
        # No socket was published and the stranger's file was not rewritten.
        assert endpoint_path(project_root).read_text(encoding="utf-8") == before
        assert any(
            "already served by another instance" in line
            and str(foreign_process.pid) in line
            for line in events
        )
        # Teardown of a refused instance must not delete a stranger's file.
        service.stop()
        assert endpoint_path(project_root).exists()
    finally:
        service.stop()


def test_own_descriptor_does_not_block_a_restart(
    project_root: Path, events: list[str], sdk_server
) -> None:
    """A descriptor still carrying OUR pid is our own leftover, not a rival."""
    save_endpoint(
        project_root,
        McpEndpoint(port=1, token="c" * 64, pid=os.getpid()),
    )
    service = _service(project_root, events, lambda: sdk_server)
    try:
        assert service.start() is True
        published = load_endpoint(project_root)
        assert published is not None
        assert published.port == service.port
        assert published.token == "c" * 64
    finally:
        service.stop()


def test_start_failure_is_reported_not_raised(
    project_root: Path, events: list[str]
) -> None:
    """Any start failure degrades to an event line — never to an exception."""

    def boom() -> object:
        raise RuntimeError("no SDK for you")

    service = _service(project_root, events, boom)
    assert service.start() is False
    assert service.running is False
    assert not endpoint_path(project_root).exists()
    assert any("failed to start" in line for line in events)


def test_a_broken_event_sink_never_breaks_the_lifecycle(
    project_root: Path, sdk_server
) -> None:
    """The sink is reporting, not logic (§6): its own crash is swallowed."""

    def broken(_text: str) -> None:
        raise RuntimeError("sink is down")

    service = McpService(
        project_root, event_sink=broken, server_factory=lambda: sdk_server
    )
    try:
        assert service.start() is True
    finally:
        service.stop()


# -- end-to-end smoke test: a real socket + the SDK's own client (§1) --------


async def test_streamable_http_round_trip_over_a_real_socket(
    project_root: Path, events: list[str], sdk_server
) -> None:
    """The whole stack for real: uvicorn thread, guard, SDK transport, client.

    Nothing is stubbed here except the tool body: an ephemeral loopback
    port is bound by `McpService`, the descriptor's bearer token is read
    back from `.drcode/mcp.json` (exactly like a real client would) and the
    SDK's Streamable HTTP client lists the tools and calls one.
    """
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    service = _service(project_root, events, lambda: sdk_server)
    try:
        assert service.start() is True
        published = load_endpoint(project_root)
        assert published is not None
        url = published.url
        headers = {"Authorization": f"Bearer {published.token}"}
        async with httpx2.AsyncClient(headers=headers) as http_client:
            transport = streamable_http_client(url, http_client=http_client)
            async with Client(transport) as client:
                tools = await client.list_tools()
                assert [tool.name for tool in tools.tools] == ["ping"]
                result = await client.call_tool("ping", {})
        assert result.is_error is False
        assert result.content[0].text == "pong"  # type: ignore[union-attr]
    finally:
        service.stop()
    assert not endpoint_path(project_root).exists()


async def test_smoke_client_is_refused_without_the_token(
    project_root: Path, events: list[str], sdk_server
) -> None:
    """The same real socket, but a token-less client never gets a session."""
    import httpx2

    service = _service(project_root, events, lambda: sdk_server)
    try:
        assert service.start() is True
        assert service.port is not None
        async with httpx2.AsyncClient() as http_client:
            response = await http_client.post(
                service.url or "",
                json=INITIALIZE_BODY,
                headers={"Accept": MCP_ACCEPT},
            )
        assert response.status_code == 401
    finally:
        service.stop()


async def test_read_surface_answers_over_the_real_socket(
    project_root: Path, events: list[str], read_surface, manager
) -> None:
    """§8 smoke for the read block: the REAL socket, the read tools.

    The service factory builds the very surface the TUI would build; the
    client reads `.drcode/mcp.json` for the token (like a real agent) and
    calls `list_tasks` against live manager state.
    """
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    service = _service(project_root, events, lambda: read_surface.server)
    try:
        assert service.start() is True
        # The loop property lives exactly as long as the server thread.
        assert service.loop is not None
        manager.create("Socket task", "over the real socket")
        published = load_endpoint(project_root)
        assert published is not None
        headers = {"Authorization": f"Bearer {published.token}"}
        async with httpx2.AsyncClient(headers=headers) as http_client:
            transport = streamable_http_client(published.url, http_client=http_client)
            async with Client(transport) as client:
                tools = await client.list_tools()
                assert "list_tasks" in [tool.name for tool in tools.tools]
                result = await client.call_tool("list_tasks", {})
        assert result.is_error is False
        assert "Socket task" in result.content[0].text  # type: ignore[union-attr]
        assert f"#{manager.all_tasks()[0].id}" in result.content[0].text  # type: ignore[union-attr]
    finally:
        service.stop()
    assert service.loop is None
    assert not endpoint_path(project_root).exists()


# -- UI surface (§6): the state strings, without mounting a whole TUI --------


class _FakeService:
    """Stand-in for `McpService` in the rendering tests."""

    def __init__(
        self, *, running: bool, port: int | None, foreign_pid: int | None
    ) -> None:
        self.running = running
        self.port = port
        self.foreign_pid = foreign_pid
        self.endpoint_file = "/tmp/project/.drcode/mcp.json"


class _UiStub:
    """The slice of `DrCodeApp` the two indicator methods actually read."""

    def __init__(self, service: _FakeService, *, enabled: bool) -> None:
        self.mcp_service = service
        self.settings = ProjectSettings(mcp_enabled=enabled)


def _status_text(stub: _UiStub) -> str:
    from drcode.tui.task_status_view import TaskStatusLayer

    return TaskStatusLayer._mcp_status_text(stub)  # pyright: ignore[reportArgumentType]


def _header_text(stub: _UiStub) -> str:
    from drcode.tui.header_view import HeaderLayer

    return HeaderLayer._header_mcp_text(stub)  # pyright: ignore[reportArgumentType]


def test_statusbar_shows_the_live_port() -> None:
    stub = _UiStub(
        _FakeService(running=True, port=51234, foreign_pid=None), enabled=True
    )
    assert _status_text(stub).strip() == "mcp: :51234"


def test_statusbar_states_are_honest() -> None:
    """off = setting disabled, busy = another instance, error = enabled but dead."""
    off = _UiStub(
        _FakeService(running=False, port=None, foreign_pid=None), enabled=False
    )
    assert _status_text(off).strip() == "mcp: off"
    busy = _UiStub(_FakeService(running=False, port=None, foreign_pid=42), enabled=True)
    assert _status_text(busy).strip() == "mcp: busy"
    error = _UiStub(
        _FakeService(running=False, port=None, foreign_pid=None), enabled=True
    )
    assert _status_text(error).strip() == "mcp: error"


def test_header_shows_the_endpoint_url() -> None:
    stub = _UiStub(
        _FakeService(running=True, port=51234, foreign_pid=None), enabled=True
    )
    assert _header_text(stub) == "MCP: on | 127.0.0.1:51234/mcp"


def test_header_states_are_honest() -> None:
    off = _UiStub(
        _FakeService(running=False, port=None, foreign_pid=None), enabled=False
    )
    assert _header_text(off) == "MCP: off"
    busy = _UiStub(_FakeService(running=False, port=None, foreign_pid=42), enabled=True)
    assert _header_text(busy) == "MCP: other instance (pid 42)"
    error = _UiStub(
        _FakeService(running=False, port=None, foreign_pid=None), enabled=True
    )
    text = _header_text(error)
    assert text.startswith("MCP: on but not serving")
    assert "mcp.json" in text


def test_the_token_is_never_rendered() -> None:
    """§6: the UI shows the endpoint, never the secret."""
    for enabled, service in (
        (True, _FakeService(running=True, port=51234, foreign_pid=None)),
        (True, _FakeService(running=False, port=None, foreign_pid=42)),
        (True, _FakeService(running=False, port=None, foreign_pid=None)),
        (False, _FakeService(running=False, port=None, foreign_pid=None)),
    ):
        stub = _UiStub(service, enabled=enabled)
        for text in (_status_text(stub), _header_text(stub)):
            assert "token" not in text.lower()
            assert TOKEN not in text


# -- the read surface (§5/§7): tools, resources, the event bridge -------------
#
# Everything below runs IN-PROCESS: the full guarded ASGI stack is driven
# through an ASGI transport (no socket), with the SDK's own Streamable HTTP
# client on top — the same client a real external agent would use. The
# TaskManager is the conftest `manager` over a tmp project, so every tool
# contract is checked against live state, not fixtures of strings.


@pytest.fixture
def read_surface(manager, tmp_path: Path):
    """A ReadSurface over the tmp manager (usage store on a tmp file)."""
    from drcode.mcp.wiring import ReadSurface
    from drcode.usage_store import UsageStore

    return ReadSurface(manager, usage_store=UsageStore(path=tmp_path / "usage.json"))


@asynccontextmanager
async def mcp_client(surface, *, headers=None):
    """A connected SDK client over the in-process ASGI transport.

    `surface` is a `wiring.ReadSurface`; `.asgi()` builds a FRESH guarded
    app per session because the SDK's session manager is single-use. The
    app's lifespan (the session manager) runs around the block, an
    `httpx2.ASGITransport` answers the HTTP half and
    `streamable_http_client` negotiates the MCP session — no socket
    anywhere, exactly the §8 testing prescription.
    """
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    app = surface.asgi(TOKEN)
    async with app.router.lifespan_context(app):
        transport = httpx2.ASGITransport(app=app)
        async with (
            httpx2.AsyncClient(
                transport=transport, base_url=BASE_URL, headers=headers or _auth()
            ) as http,
            Client(
                streamable_http_client(BASE_URL + MCP_PATH, http_client=http)
            ) as client,
        ):
            yield client


async def call_text(client, name: str, arguments: dict | None = None) -> str:
    """Call one tool, assert a non-error answer, return the text content."""
    result = await client.call_tool(name, arguments or {})
    assert result.is_error is False, getattr(result.content[0], "text", "")
    return result.content[0].text  # type: ignore[union-attr]


async def call_text_error(client, name: str, arguments: dict) -> str:
    """Call one tool, assert the §4 isError contract, return the message."""
    result = await client.call_tool(name, arguments)
    assert result.is_error is True
    return result.content[0].text  # type: ignore[union-attr]


# -- the tool list ------------------------------------------------------------


async def test_the_read_tools_are_listed(read_surface) -> None:
    """The server advertises exactly the nine read tools of §7, in order."""
    from drcode.mcp.tools import READ_TOOLS

    async with mcp_client(read_surface) as client:
        tools = await client.list_tools()
        assert [tool.name for tool in tools.tools] == list(READ_TOOLS)


# -- list_tasks ----------------------------------------------------------------


async def test_list_tasks_rows_and_filters(read_surface, manager) -> None:
    """Compact rows: `#id [status/kind] title — markers`, filters applied."""
    plain = manager.create("Plain feature", "do the thing")
    held = manager.create("Held bug", "fix the crash", kind="bug", hold=True)
    async with mcp_client(read_surface) as client:
        text = await call_text(client, "list_tasks")
    lines = text.splitlines()
    assert lines[0].startswith("2 task(s)")
    assert f"#{plain.id} [queued/feature] Plain feature" in lines[1]
    assert f"#{held.id} [hold/bug] Held bug" in lines[2]
    # The held task carries the documented markers: hold + one queued input.
    assert "hold" in lines[2]
    assert "input=1" in lines[2]
    # The plain task's only marker is its one queued input message.
    assert lines[1].endswith("— input=1")
    assert "hold" not in lines[1]
    # A status filter narrows to the matching tasks only.
    async with mcp_client(read_surface) as client:
        held_only = await call_text(
            client, "list_tasks", {"status": "hold", "kind": "bug"}
        )
    assert held_only.startswith("1 task(s) (status=hold, kind=bug)")
    assert held.id in held_only and plain.id not in held_only


async def test_list_tasks_limit_and_unknown_filter(read_surface, manager) -> None:
    manager.create("First", "one")
    manager.create("Second", "two")
    async with mcp_client(read_surface) as client:
        capped = await call_text(client, "list_tasks", {"limit": 1})
        assert capped.startswith("2 task(s), showing the first 1")
        assert "...and 1 more" in capped
        message = await call_text_error(client, "list_tasks", {"status": "banana"})
        assert "unknown status" in message and "banana" in message


# -- get_task ------------------------------------------------------------------


async def test_get_task_full_metadata_contract(read_surface, manager) -> None:
    """JSON body: metadata + skeleton + events + clarification, NO transcript."""
    from drcode.models import Clarification

    task = manager.create("Detailed", "the whole prompt")
    task.clarification = Clarification.single("Which variant?", options=["A", "B"])
    manager.storage.save(task)
    async with mcp_client(read_surface) as client:
        raw = await call_text(client, "get_task", {"task_id": task.id})
    payload = json.loads(raw)
    assert payload["id"] == task.id
    assert payload["title"] == "Detailed"
    assert payload["status"] == "queued"
    assert payload["kind"] == "feature"
    assert payload["hold"] is False
    assert payload["verification"] is None
    # The pending input is structural: count + the verbatim text.
    assert payload["pending_input"]["count"] == 1
    assert payload["pending_input"]["items"][0]["text"] == "the whole prompt"
    # The clarification package comes through structurally.
    assert payload["clarification"]["questions"][0]["question"] == "Which variant?"
    assert payload["clarification"]["questions"][0]["options"] == ["A", "B"]
    # Cheap fields every consumer needs.
    assert payload["blocking_dependencies"] == []
    assert payload["events"]["total"] == 0
    assert payload["skeleton"]["prompts"] == []
    # THE hard rule of the layer: no transcript field is ever exposed.
    assert "messages" not in payload and "tester_messages" not in payload


async def test_get_task_unknown_id_is_an_error_with_known_ids(
    read_surface, manager
) -> None:
    task = manager.create("Known", "known")
    async with mcp_client(read_surface) as client:
        message = await call_text_error(client, "get_task", {"task_id": "nope"})
    assert "unknown task id" in message and "nope" in message
    assert f"#{task.id}" in message  # the hint names real ids


# -- search_tasks --------------------------------------------------------------


async def test_search_tasks_finds_the_archive(read_surface, manager) -> None:
    needle = manager.create("Needle task", "the grawlixquuzz story")
    manager.create("Other task", "something else entirely")
    async with mcp_client(read_surface) as client:
        text = await call_text(client, "search_tasks", {"keywords": "grawlixquuzz"})
    assert f"#{needle.id}" in text
    assert "Other task" not in text


async def test_search_tasks_rejects_empty_keywords(read_surface) -> None:
    async with mcp_client(read_surface) as client:
        message = await call_text_error(client, "search_tasks", {"keywords": "  , "})
    assert "at least one non-empty word" in message


# -- get_queue -----------------------------------------------------------------


async def test_get_queue_order_and_blockers(read_surface, manager) -> None:
    first = manager.create("First", "the dependency")
    second = manager.create("Second", "depends on first", depends_on=[first.id])
    async with mcp_client(read_surface) as client:
        text = await call_text(client, "get_queue")
    lines = text.splitlines()
    assert lines[0].startswith("queue: 2 task(s)")
    # The dependency sink rule: the blocker is promoted first.
    assert lines[1].startswith(f"1. #{first.id}")
    assert lines[2].startswith(f"2. #{second.id}")
    assert f"blocked by #{first.id}" in lines[2]
    assert "blocked by" not in lines[1]


# -- goals ---------------------------------------------------------------------


async def test_goals_list_and_get_contract(read_surface, manager) -> None:
    goal = manager.goals.create("The Roadmap", description="where this goes")
    task = manager.create("Linked task", "works on the roadmap", goal_id=goal.id)
    async with mcp_client(read_surface) as client:
        listing = await call_text(client, "list_goals")
        raw = await call_text(client, "get_goal", {"goal_id": goal.id})
    assert f"#{goal.id}" in listing
    assert "0/1 tasks" in listing  # live progress: the linked task is not done
    payload = json.loads(raw)
    assert payload["id"] == goal.id
    assert payload["title"] == "The Roadmap"
    assert payload["description"] == "where this goes"
    assert payload["progress"] == {"done": 0, "total": 1}
    assert payload["complete"] is False
    assert [entry["id"] for entry in payload["tasks"]] == [task.id]
    async with mcp_client(read_surface) as client:
        message = await call_text_error(client, "get_goal", {"goal_id": "missing"})
    assert "unknown goal id" in message


# -- stats / usage -------------------------------------------------------------


async def test_get_stats_contract(read_surface, manager) -> None:
    manager.create("Counted", "the created counter moves")
    async with mcp_client(read_surface) as client:
        text = await call_text(client, "get_stats")
    lines = text.splitlines()
    # The fixed header names every column; the day rows and the total follow.
    assert lines[0].split() == [
        "day",
        "created",
        "completed",
        "cancelled",
        "failed",
        "dev_avg",
        "testing_avg",
        "resolver_avg",
    ]
    assert "total (last 14 days)" in text


async def test_get_usage_contract(read_surface, manager) -> None:
    # One recorded day (today, so any window that includes "now" finds it)
    # straight into the injected store's file.
    store = read_surface.view._usage_store
    today = date.today().isoformat()
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps(
            {
                "version": 1,
                "days": {
                    today: {
                        "coding/test-model": {
                            "requests": 2,
                            "input": 100,
                            "output": 40,
                            "cache_read": 7,
                            "cache_write": 0,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    async with mcp_client(read_surface) as client:
        text = await call_text(client, "get_usage", {"days": 30})
    assert f"{today} coding/test-model requests=2 input=100 output=40" in text
    assert "total (last 30 days): requests=2 input=100 output=40" in text


async def test_get_usage_empty_window(read_surface, manager) -> None:
    async with mcp_client(read_surface) as client:
        empty = await call_text(client, "get_usage", {"days": 3})
    assert "no model calls recorded in the last 3 day(s)" in empty


# -- check_file_overlaps -------------------------------------------------------


async def test_check_file_overlaps_contract(read_surface, manager) -> None:
    manager.create("Alpha work", "implement src/alpha.py properly")
    async with mcp_client(read_surface) as client:
        text = await call_text(
            client, "check_file_overlaps", {"text": "refactor src/alpha.py"}
        )
        clean = await call_text(
            client, "check_file_overlaps", {"text": "touch src/unrelated.py"}
        )
        no_paths = await call_text(
            client, "check_file_overlaps", {"text": "no paths here at all"}
        )
    assert f"#{manager.all_tasks()[0].id}" in text
    assert "src/alpha.py" in text
    assert "no unfinished task mentions" in clean
    assert "No path-like tokens" in no_paths
    # A declared dependency is excluded from the report — it IS the mitigation.
    async with mcp_client(read_surface) as client:
        declared = await call_text(
            client,
            "check_file_overlaps",
            {
                "text": "refactor src/alpha.py",
                "depends_on": [manager.all_tasks()[0].id],
            },
        )
    assert "no unfinished task mentions" in declared


# -- resources: same bodies as the tools (§7) ----------------------------------


async def test_resources_are_listed(read_surface) -> None:
    from drcode.mcp.resources import QUEUE_URI, STATS_URI, TASK_URI_TEMPLATE

    async with mcp_client(read_surface) as client:
        listed = await client.list_resources()
        templates = await client.list_resource_templates()
    uris = {str(resource.uri) for resource in listed.resources}
    assert QUEUE_URI in uris and STATS_URI in uris
    assert any(
        template.uri_template == TASK_URI_TEMPLATE
        for template in templates.resource_templates
    )


async def test_resource_bodies_equal_the_tool_output(read_surface, manager) -> None:
    from mcp.types import TextResourceContents

    first = manager.create("First", "the dependency")
    manager.create("Second", "depends on first", depends_on=[first.id])
    from drcode.mcp.resources import QUEUE_URI, task_uri

    async with mcp_client(read_surface) as client:
        resource = await client.read_resource(QUEUE_URI)
        tool_text = await call_text(client, "get_queue")
        task_resource = await client.read_resource(task_uri(first.id))
        task_tool_text = await call_text(client, "get_task", {"task_id": first.id})
    # The subscription story and the polling story CANNOT drift: the
    # resource body is byte-identical with the matching tool answer. The
    # contents union (text|blob) narrows with isinstance — every resource
    # here is registered with a str return, so TextResourceContents it is.
    queue_body = resource.contents[0]
    task_body = task_resource.contents[0]
    assert isinstance(queue_body, TextResourceContents)
    assert isinstance(task_body, TextResourceContents)
    assert queue_body.text == tool_text
    assert task_body.text == task_tool_text
    assert json.loads(task_body.text)["id"] == first.id


async def test_unknown_task_resource_is_not_found(read_surface) -> None:
    from drcode.mcp.resources import task_uri

    async with mcp_client(read_surface) as client:
        with pytest.raises(Exception, match="unknown task id"):
            await client.read_resource(task_uri("nope"))


# -- the event bridge (§5): routing + coalescing --------------------------------
#
# NOTE: the listen stream is a NEVER-ENDING SSE response, which the in-process
# `httpx2.ASGITransport` cannot stream (it waits for the response body to
# end) — so the subscription tests run the SDK's DIRECT in-memory client
# (`Client(server, mode="2026-07-28")`, no HTTP at all). The server-side path
# under test — bus → listen handler → typed events — is identical; the HTTP
# surface itself is covered by the transport tests above and the #22b76acf
# suite (a real socket smoke test included).


@asynccontextmanager
async def direct_client(surface):
    """A connected SDK client speaking to the server in-process."""
    from mcp import Client

    async with Client(surface.server, mode="2026-07-28") as client:
        yield client


async def drain(sub, count: int, *, timeout: float = 5.0):
    """Collect `count` resource-updated events from a listen stream."""
    events = []
    async with asyncio.timeout(timeout):
        async for event in sub:
            events.append(event)
            if len(events) >= count:
                break
    return events


async def test_bridge_routes_and_coalesces(read_surface, manager) -> None:
    """Burst on one task → ONE notification per URI; only affected URIs."""
    from drcode.mcp.resources import QUEUE_URI, goal_uri, task_uri

    goal = manager.goals.create("Bridge goal")
    watched = manager.create("Watched", "watched", goal_id=goal.id)
    other = manager.create("Other", "other")
    loop = asyncio.get_running_loop()
    read_surface.start_notifications(loop, window=0.05)

    async with (
        direct_client(read_surface) as client,
        client.listen(
            resource_subscriptions=[
                task_uri(watched.id),
                task_uri(other.id),
                QUEUE_URI,
                goal_uri(goal.id),
            ]
        ) as sub,
    ):
        # Burst: three status transitions of the SAME task inside the
        # coalescing window — one task notification, one queue.
        manager.set_hold(watched.id, on=True)
        manager.set_hold(watched.id, on=False)
        manager.set_hold(watched.id, on=True)
        # The other task's change must NOT touch the watched task's URI
        # or the goal's — only its own URI (the queue joins the same batch).
        manager.set_hold(other.id, on=True)
        events = await drain(sub, 4)
    uris = [str(event.uri) for event in events]
    from collections import Counter

    counts = Counter(uris)
    # Coalesced burst: exactly ONE per distinct URI, despite 4 transitions.
    assert counts == Counter(
        {
            task_uri(watched.id): 1,
            QUEUE_URI: 1,
            task_uri(other.id): 1,
            goal_uri(goal.id): 1,
        }
    )


async def test_bridge_does_not_wake_unaffected_subscribers(
    read_surface, manager
) -> None:
    """A subscriber of ONE task's URI hears nothing about other tasks."""
    from drcode.mcp.resources import task_uri

    watched = manager.create("Watched", "watched")
    bystander = manager.create("Bystander", "bystander")
    loop = asyncio.get_running_loop()
    read_surface.start_notifications(loop, window=0.05)

    async with (
        direct_client(read_surface) as client,
        client.listen(resource_subscriptions=[task_uri(watched.id)]) as sub,
    ):
        # The bystander's transitions must not produce ANY event.
        manager.set_hold(bystander.id, on=True)
        manager.set_hold(bystander.id, on=False)
        # Long enough to cover the coalescing window plus slack.
        await asyncio.sleep(0.3)
        got_during_noise = await drain_without_events(sub)
        assert got_during_noise is True  # nothing arrived
        # Now the watched task changes: exactly one event arrives.
        manager.set_hold(watched.id, on=True)
        events = await drain(sub, 1)
    assert [str(event.uri) for event in events] == [task_uri(watched.id)]


async def drain_without_events(sub, *, quiet: float = 0.1) -> bool:
    """True when NO event arrives within `quiet` seconds (negative test)."""
    try:
        async with asyncio.timeout(quiet):
            async for _event in sub:
                return False
    except TimeoutError:
        return True
    return True  # pragma: no cover - the iterator ended without events


async def test_bridge_completing_a_task_notifies_stats(read_surface, manager) -> None:
    """§5 routing: the transition INTO COMPLETED adds `drcode://stats`."""
    from drcode.mcp.resources import QUEUE_URI, STATS_URI, task_uri

    task = manager.create("Finishing", "finishing", kind="research")
    loop = asyncio.get_running_loop()
    read_surface.start_notifications(loop, window=0.05)

    async with (
        direct_client(read_surface) as client,
        client.listen(
            resource_subscriptions=[task_uri(task.id), STATS_URI, QUEUE_URI]
        ) as sub,
    ):
        manager.mark_done(task.id)  # research → COMPLETED directly
        events = await drain(sub, 3)
    uris = {str(event.uri) for event in events}
    assert uris == {task_uri(task.id), STATS_URI, QUEUE_URI}


async def test_bridge_is_detached_on_stop(read_surface, manager) -> None:
    """stop_notifications removes the listeners: the manager stops feeding."""
    task = manager.create("Detached", "detached")
    loop = asyncio.new_event_loop()
    try:
        read_surface.start_notifications(loop, window=0.01)
        assert read_surface.bridge is not None
        read_surface.stop_notifications()
        assert read_surface.bridge is None and read_surface.router is None
        # A transition AFTER the detach reaches nothing and must not raise.
        manager.set_hold(task.id, on=True)
    finally:
        loop.close()


def test_the_router_never_raises_on_a_closed_loop(manager) -> None:
    """Scheduling onto a dead loop is logged, never raised (§5 contract)."""
    from drcode.mcp.bridge import EventBridge, LoopPublisher, NotificationRouter

    loop = asyncio.new_event_loop()
    router = NotificationRouter(loop, LoopPublisher(None), window=0.01)
    bridge = EventBridge(router)
    loop.close()
    # The listeners are sync and must not raise even on a dead loop.
    bridge._on_status(manager.create("Zombie", "zombie"), None, None, "user")
    bridge._on_removal("ghost")
    router.close()


# -- the mutation surface (§4/§6): gating matrix + re-advance -------------------
#
# Same in-process discipline as the read half: the FULL guarded ASGI stack
# is driven by the SDK's own Streamable HTTP client, the TaskManager is the
# live conftest `manager`, and the §6 re-advance is observed through a
# recorder injected as the surface's MutationHooks (in production the TUI
# passes wrappers that hop into the app thread — the tests record instead,
# which is the mock-_advance observation the design prescribes).

from drcode.mcp.mutations import (
    MUTATION_TOOLS,
    MutationHooks,
)
from drcode.models import (
    Clarification,
    ClarificationItem,
    Task,
    TaskStatus,
)


class AdvanceRecorder:
    """The mutation hooks as test observers: every §6 post-step is counted."""

    def __init__(self) -> None:
        self.advances: list[str] = []
        self.aborted: list[str] = []
        self.restarted: list[str] = []
        self.hooks = MutationHooks(
            advance=lambda: self.advances.append("advance"),
            abort_runs=self.abort_runs,
            restart_run=self.restart_run,
        )

    def abort_runs(self, task_id: str) -> None:
        self.aborted.append(task_id)

    def restart_run(self, task_id: str) -> None:
        self.restarted.append(task_id)


@pytest.fixture
def mut(manager, tmp_path: Path):
    """A surface WITH the mutation tools, plus the recorder of its hooks."""
    from drcode.mcp.wiring import McpSurface
    from drcode.usage_store import UsageStore

    recorder = AdvanceRecorder()
    surface = McpSurface(
        manager,
        usage_store=UsageStore(path=tmp_path / "usage.json"),
        hooks=recorder.hooks,
    )
    return surface, recorder


# The §4 matrix, literally: row = tool, column = status, value = outcome.
# ALLOWED — the mutation applies and (for the §6 set) re-advance is planned;
# BUSY — denied with the matrix's verbatim ACTIVE/TESTING advice
# («отмените (cancel) или дождитесь»); DENIED — any other refusal, still a
# normal isError carrying the current status.
ALLOWED, DENIED, BUSY = "allowed", "denied", "busy"

_STATUSES = (
    "queued",
    "draft",
    "hold",
    "clarification",
    "active",
    "testing",
    "completed",
)

_MUTATION_MATRIX: dict[str, dict[str, str]] = {
    "send_input": dict.fromkeys(_STATUSES, ALLOWED),
    "answer_clarification": dict.fromkeys(_STATUSES, DENIED)
    | {"clarification": ALLOWED},
    "edit_task": dict.fromkeys(_STATUSES, DENIED)
    | {"queued": ALLOWED, "draft": ALLOWED, "hold": ALLOWED},
    "set_dependencies": dict.fromkeys(_STATUSES, DENIED)
    | {
        "queued": ALLOWED,
        "draft": ALLOWED,
        "hold": ALLOWED,
        "active": BUSY,
        "testing": BUSY,
    },
    "set_kind": dict.fromkeys(_STATUSES, DENIED)
    | {
        "queued": ALLOWED,
        "draft": ALLOWED,
        "hold": ALLOWED,
        "active": BUSY,
        "testing": BUSY,
    },
    # The matrix row is set_hold(ON); the resume half has its own tests below.
    "set_hold": dict.fromkeys(_STATUSES, DENIED)
    | {
        "queued": ALLOWED,
        "draft": ALLOWED,
        "hold": ALLOWED,
        "active": BUSY,
        "testing": BUSY,
    },
    "move_in_queue": dict.fromkeys(_STATUSES, DENIED) | {"queued": ALLOWED},
    "cancel": dict.fromkeys(_STATUSES, ALLOWED) | {"completed": DENIED},
    "delete": dict.fromkeys(_STATUSES, DENIED)
    | {
        "hold": ALLOWED,
        "draft": ALLOWED,
        "active": BUSY,
        "testing": BUSY,
    },
}

#: The §6 re-advance set. `edit_task`/`set_kind`/`set_hold(on)` are absent
#: on purpose: the TUI does not press Enter after them (a held/draft task
#: is never promotable; a re-type only changes the queue position).
#: `set_dependencies` IS one: dropping a blocker is the TUI's
#: force-start (action_remove_dependency advances).
_ADVANCE_AFTER = {
    "send_input",
    "answer_clarification",
    "set_dependencies",
    "move_in_queue",
    "cancel",
    "delete",
}


def _task_in_status(manager, status: str, *, title: str | None = None) -> Task:
    """A real task parked in the given status (direct transitions — the
    gating only reads the status, so no pipeline round-trip is needed)."""
    name = title or f"Task {status}"
    if status == "draft":
        return manager.create(name, f"condition of {status}", draft=True)
    if status == "hold":
        return manager.create(name, f"condition of {status}", hold=True)
    task = manager.create(name, f"condition of {status}")
    if status == "clarification":
        task.status = TaskStatus.AWAITING_CLARIFICATION
        task.clarification = Clarification.single("Which variant?", options=["A", "B"])
    elif status != "queued":
        task.status = TaskStatus(status)
    if status != "queued":
        manager.storage.save(task)
    return task


def _mutation_args(tool: str, task: Task, manager) -> dict:
    """Minimal valid arguments for one mutation against `task`."""
    if tool == "send_input":
        return {"task_id": task.id, "text": "a fresh external comment"}
    if tool == "answer_clarification":
        return {"task_id": task.id, "answers": ["A"]}
    if tool == "edit_task":
        return {"task_id": task.id, "text": "rewritten condition"}
    if tool == "set_dependencies":
        dep = manager.create("Dep target", "the dependency condition")
        return {"task_id": task.id, "deps": [dep.id]}
    if tool == "set_kind":
        return {"task_id": task.id, "kind": "bug"}
    if tool == "set_hold":
        return {"task_id": task.id, "on": True}
    if tool == "move_in_queue":
        return {"task_id": task.id, "up": True}
    return {"task_id": task.id}


async def _mutation_call(mut, tool: str, arguments: dict) -> tuple[bool, str]:
    """One tool call over the in-process client: (is_error, text)."""
    surface, _recorder = mut
    async with mcp_client(surface) as client:
        result = await client.call_tool(tool, arguments)
    return result.is_error, result.content[0].text  # type: ignore[union-attr]


@pytest.mark.parametrize("status", _STATUSES)
@pytest.mark.parametrize("tool", list(_MUTATION_MATRIX))
async def test_gating_matrix_cell(mut, manager, tool: str, status: str) -> None:
    """Every row × column of the §4 matrix, driven through the ASGI client."""
    _surface, recorder = mut
    expected = _MUTATION_MATRIX[tool][status]
    task = _task_in_status(manager, status)
    if tool == "move_in_queue" and status == "queued":
        # queued() runs next-to-run FIRST, and up=True swaps with the entry
        # BEHIND — the target needs a later queued task to swap with.
        manager.create("Later queued", "queued later")
    arguments = _mutation_args(tool, task, manager)
    recorder.advances.clear()

    is_error, text = await _mutation_call(mut, tool, arguments)

    if expected is ALLOWED:
        assert is_error is False, text
        # The mutation is reflected in the TaskManager — the one source of truth.
        if tool == "send_input":
            fresh = manager.get(task.id)
            assert any(
                e.text == "a fresh external comment" for e in fresh.pending_input
            )
        elif tool == "answer_clarification":
            fresh = manager.get(task.id)
            assert fresh.clarification is None
            assert fresh.status is TaskStatus.QUEUED
        elif tool == "edit_task":
            fresh = manager.get(task.id)
            assert [e.text for e in fresh.pending_input] == ["rewritten condition"]
        elif tool == "set_dependencies":
            assert manager.get(task.id).depends_on == [arguments["deps"][0]]
        elif tool == "set_kind":
            assert manager.get(task.id).kind.value == "bug"
        elif tool == "set_hold":
            fresh = manager.get(task.id)
            assert fresh.status is TaskStatus.HOLD or status == "draft"
        elif tool == "move_in_queue":
            assert manager.queued()[-1].id == task.id  # moved up = away from execution
        elif tool == "cancel":
            fresh = manager.get(task.id)
            assert fresh.status is TaskStatus.COMPLETED and fresh.cancelled
        elif tool == "delete":
            assert task.id not in manager.tasks
        # §6: the re-advance is planned exactly for the TUI's Enter set.
        if tool in _ADVANCE_AFTER:
            assert recorder.advances == ["advance"], tool
        else:
            assert recorder.advances == [], tool
    else:
        # A forbidden mutation is a NORMAL isError with the current status
        # and advice — not a crash, not a silent no-op.
        assert is_error is True, text
        assert status in text
        assert task.status.value == status  # nothing moved
        if expected is BUSY:
            # The matrix's verbatim advice: cancel it, or wait for it.
            assert "cancel" in text
        if tool in (
            "set_kind",
            "set_dependencies",
            "set_hold",
            "delete",
            "move_in_queue",
        ):
            # The denial really denied: the field is untouched.
            fresh = manager.get(task.id)
            assert fresh.status is TaskStatus(task.status.value)


async def test_the_surface_with_hooks_lists_reads_then_mutations(mut) -> None:
    from drcode.mcp.tools import READ_TOOLS

    surface, _recorder = mut
    async with mcp_client(surface) as client:
        tools = await client.list_tools()
    assert [tool.name for tool in tools.tools] == [*READ_TOOLS, *MUTATION_TOOLS]


async def test_the_forbidden_tools_are_never_published(mut) -> None:
    """§4 «Не публикуются» — the tester/hook/config/transcript internals
    stay unreachable: the tester alone owns the COMPLETED verdict."""
    surface, _recorder = mut
    forbidden = (
        "set_verification",
        "finalize_verified",
        "requeue_failed_fix",
        "requeue_review_rejection",
        "complete_without_verification",
        "purge_expired_completed",
        "link_goal",
        "unlink_goal",
        "write_transcript",
        "set_model",
        "configure_models",
    )
    async with mcp_client(surface) as client:
        tools = await client.list_tools()
    names = {tool.name for tool in tools.tools}
    assert names & set(forbidden) == set()


async def test_cancel_description_declares_the_power_action(mut) -> None:
    """§4: the cancel tool's description MUST say in plain words that it
    aborts the run and completes past the tester."""
    surface, _recorder = mut
    async with mcp_client(surface) as client:
        tools = await client.list_tools()
    description = next(t.description for t in tools.tools if t.name == "cancel")
    # The SDK types Tool.description as `str | None` — narrow first.
    assert description is not None
    assert "POWER-ACTION" in description
    assert "ABORTS" in description
    assert "COMPLETED" in description
    assert "NEVER runs" in description


# -- re-advance details (§6) ----------------------------------------------------


async def test_send_input_to_active_restarts_the_run(mut, manager) -> None:
    """§4 ACTIVE cell: a comment interrupts the run — the TUI's run_agent
    supersede is planned (restart hook) alongside the re-advance."""
    surface, recorder = mut
    task = _task_in_status(manager, "active")
    async with mcp_client(surface) as client:
        text = await call_text(
            client, "send_input", {"task_id": task.id, "text": "mid-flight comment"}
        )
    assert "restarts" in text
    assert recorder.restarted == [task.id]
    assert recorder.advances == ["advance"]
    fresh = manager.get(task.id)
    assert fresh.status is TaskStatus.ACTIVE
    assert fresh.pending_input[-1].text == "mid-flight comment"


async def test_send_input_to_completed_requeues_and_clears_stale_marks(
    mut, manager
) -> None:
    """§4 completed cell: «перезакидывает в очередь», dropping the stale
    verification label / cancel marker — manager.send's own transition."""
    surface, recorder = mut
    task = _task_in_status(manager, "completed")
    task.cancelled = True
    manager.storage.save(task)
    async with mcp_client(surface) as client:
        await call_text(
            client, "send_input", {"task_id": task.id, "text": "one more round"}
        )
    fresh = manager.get(task.id)
    assert fresh.status is TaskStatus.QUEUED
    assert fresh.cancelled is False
    assert fresh.verification is None
    assert recorder.advances == ["advance"]


async def test_set_hold_resume_advances_and_refuses_the_rest(mut, manager) -> None:
    """set_hold(off) is a §6 trigger; a non-held task refuses honestly."""
    surface, recorder = mut
    task = _task_in_status(manager, "hold")
    async with mcp_client(surface) as client:
        text = await call_text(client, "set_hold", {"task_id": task.id, "on": False})
        assert "resumed" in text
        message = await call_text_error(
            client, "set_hold", {"task_id": task.id, "on": False}
        )
    assert recorder.advances == ["advance"]
    assert manager.get(task.id).status is TaskStatus.QUEUED
    # A second resume is refused: the task is not on hold anymore.
    assert "not on hold" in message


async def test_move_in_queue_refusals_carry_the_tui_reasons(mut, manager) -> None:
    """An impossible move is an isError with the TUI toast's reason — a
    silent no-op would look like the tool ignored the call. queued() runs
    next-to-run first: the bug sits at the head, the dependency-blocked
    task sinks to the tail."""
    surface, recorder = mut
    bug = manager.create("The bug", "fix it", kind="bug")
    free = manager.create("The feature", "build it")
    blocked = manager.create("The blocked", "waits", depends_on=[bug.id])
    async with mcp_client(surface) as client:
        edge = await call_text_error(
            client, "move_in_queue", {"task_id": blocked.id, "up": True}
        )
        cross = await call_text_error(
            client, "move_in_queue", {"task_id": bug.id, "up": True}
        )
        tier = await call_text_error(
            client, "move_in_queue", {"task_id": free.id, "up": True}
        )
    assert "already farthest" in edge
    assert "bug block" in cross
    assert "waits for its dependencies" in tier
    assert recorder.advances == []  # refusals do not advance
    assert [t.id for t in manager.queued()] == [bug.id, free.id, blocked.id]


async def test_trigger_advance_schedules_the_gate(mut) -> None:
    surface, recorder = mut
    async with mcp_client(surface) as client:
        text = await call_text(client, "trigger_advance")
    assert "scheduled" in text
    assert recorder.advances == ["advance"]


async def test_mutation_with_unknown_task_id_is_an_error(mut, manager) -> None:
    surface, _recorder = mut
    async with mcp_client(surface) as client:
        message = await call_text_error(
            client, "send_input", {"task_id": "nope", "text": "hi"}
        )
    assert "unknown task id" in message and "nope" in message


# -- create_task ----------------------------------------------------------------


async def test_create_task_contract(mut, manager) -> None:
    surface, recorder = mut
    async with mcp_client(surface) as client:
        text = await call_text(
            client,
            "create_task",
            {"text": "implement src/thing.py", "kind": "bug", "title": "Thing"},
        )
    created = manager.all_tasks()[-1]
    assert f"#{created.id}" in text and "[bug]" in text and "status queued" in text
    assert created.title == "Thing"  # an explicit title is kept verbatim
    assert created.kind.value == "bug"
    assert created.pending_input[-1].text == "implement src/thing.py"
    assert recorder.advances == ["advance"]  # §6: creation re-advances


async def test_create_task_held_and_dependencies(mut, manager) -> None:
    surface, _recorder = mut
    dep = manager.create("The dep", "the blocker")
    async with mcp_client(surface) as client:
        text = await call_text(
            client,
            "create_task",
            {
                "text": "the parked work",
                "kind": "feature",
                "hold": True,
                "depends_on": [dep.id],
            },
        )
    created = manager.all_tasks()[-1]
    assert created.status is TaskStatus.HOLD
    assert created.depends_on == [dep.id]
    assert "hold" in text


async def test_create_task_validates_kind_goal_and_text(mut, manager) -> None:
    surface, _recorder = mut
    async with mcp_client(surface) as client:
        bad_kind = await call_text_error(
            client, "create_task", {"text": "x", "kind": "banana"}
        )
        bad_goal = await call_text_error(
            client, "create_task", {"text": "x", "kind": "bug", "goal_id": "ghost"}
        )
        bad_dep = await call_text_error(
            client, "create_task", {"text": "x", "kind": "bug", "depends_on": ["ghost"]}
        )
        empty = await call_text_error(
            client, "create_task", {"text": "  ", "kind": "bug"}
        )
    assert "unknown kind" in bad_kind
    assert "unknown goal id" in bad_goal
    assert "unknown task id" in bad_dep
    assert "must not be empty" in empty


async def test_create_task_reports_file_overlaps(mut, manager) -> None:
    """The pre-flight warning: an unfinished task already mentions the same
    files — a WARNING in the answer (and a service event), never a block."""
    surface, _recorder = mut
    manager.create("Alpha work", "implement src/alpha.py properly")
    async with mcp_client(surface) as client:
        text = await call_text(
            client, "create_task", {"text": "refactor src/alpha.py", "kind": "feature"}
        )
    assert "src/alpha.py" in text
    assert "also appear in unfinished task" in text
    created = manager.all_tasks()[-1]
    assert any("overlap" in e.text.lower() for e in created.events)


# -- answer_clarification --------------------------------------------------------


async def test_answer_clarification_joins_one_line_per_question(mut, manager) -> None:
    """Count validation + the exact TUI Enter semantics: one line per
    question, submitted as ONE pending input; the task re-queues at the head."""
    surface, recorder = mut
    task = _task_in_status(manager, "clarification")
    task.clarification = Clarification(
        questions=[
            ClarificationItem(question="Which variant?", options=["A", "B"]),
            ClarificationItem(question="Which color?", options=["red", "green"]),
        ]
    )
    manager.storage.save(task)
    async with mcp_client(surface) as client:
        # The wrong count is refused BEFORE anything is sent…
        mismatch = await call_text_error(
            client, "answer_clarification", {"task_id": task.id, "answers": ["A"]}
        )
        assert "expected 2 answer(s), got 1" in mismatch
        assert "Which color?" in mismatch  # the questions are listed
        fresh = manager.get(task.id)
        assert fresh.clarification is not None  # …the package is intact
        # …and the right count submits one line per question.
        text = await call_text(
            client,
            "answer_clarification",
            {"task_id": task.id, "answers": ["Variant B", "green"]},
        )
    fresh = manager.get(task.id)
    assert fresh.clarification is None
    assert fresh.status is TaskStatus.QUEUED
    assert fresh.queue_head is True
    assert [e.text for e in fresh.pending_input] == [
        "condition of clarification",
        "Variant B\ngreen",
    ]
    assert recorder.advances == ["advance"]
    assert "queued" in text


async def test_answer_clarification_without_a_package_is_refused(mut, manager) -> None:
    surface, _recorder = mut
    task = manager.create("Plain", "plain")
    async with mcp_client(surface) as client:
        message = await call_text_error(
            client, "answer_clarification", {"task_id": task.id, "answers": ["x"]}
        )
    assert "no pending clarification" in message


# -- edit_task --------------------------------------------------------------------


async def test_edit_task_refuses_when_nothing_is_editable(mut, manager) -> None:
    """§4 «по editable_condition (причина в ошибке)»: consumed text is
    history, not a form field — the error says so."""
    surface, _recorder = mut
    task = manager.create("Started", "original condition", hold=True)
    task.pending_input.clear()  # everything consumed before the hold
    manager.storage.save(task)
    async with mcp_client(surface) as client:
        message = await call_text_error(
            client, "edit_task", {"task_id": task.id, "text": "rewritten"}
        )
    assert "nothing editable" in message
    assert manager.get(task.id).pending_input == []
