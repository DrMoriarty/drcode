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

import json
import os
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
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
