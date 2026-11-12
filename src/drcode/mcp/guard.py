"""The access guard in front of the embedded MCP server (design doc §2).

The server listens on loopback, but "loopback" is not an access control:
any local process — including a web page in the user's browser driving a
DNS-rebinding attack — can reach `127.0.0.1:<port>`. Two independent
checks therefore wrap the SDK's Streamable HTTP app:

1. **Bearer token** — every `/mcp` request (POST, the GET-SSE stream and
   the session-terminating DELETE alike) must carry
   `Authorization: Bearer <token>` with the token published in
   `.drcode/mcp.json`. A missing or wrong one is answered with a bare
   `401` and a `WWW-Authenticate` challenge that names no details — the
   response must not help a guesser (no "expected 64 hex chars", no
   hint about the file).
2. **Origin/Host validation** — the standard DNS-rebinding defence of
   HTTP transports: a request whose `Origin` is not a loopback origin is
   refused, and so is one whose `Host` is not a loopback host. That is
   what stops `evil.example` from resolving to `127.0.0.1` and talking to
   the port with the user's browser as the confused deputy. A missing
   `Origin` is fine (a plain CLI client sends none — the header is a
   browser-only concept, so its absence cannot be a rebinding attempt).

Both checks live in ONE pure ASGI middleware rather than a Starlette
`BaseHTTPMiddleware`: the GET-SSE stream is a long-lived response, and
`BaseHTTPMiddleware`'s task-group indirection historically breaks
background disconnect propagation for exactly such streams. Being plain
ASGI also keeps this module free of any dependency beyond the stdlib, so
it can be unit-tested without starting a server.

CORS is NOT configured on purpose — no browser client is expected, and a
permissive CORS header would undercut exactly the Origin check above.
"""

from __future__ import annotations

import hmac
from typing import Final

from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from .endpoint import MCP_PATH

#: Header carrying the bearer token (§2).
AUTH_HEADER: Final = "authorization"

#: Scheme prefix of the `Authorization` header value.
BEARER_PREFIX: Final = "bearer "

#: Hostnames accepted in `Host` and `Origin` — loopback in its IPv4, IPv6
#: and RFC 6761 (`localhost`) spellings, nothing else. A port, when
#: present, is always accepted: the port is ephemeral by design (§1), so
#: pinning it cannot be a static list.
LOOPBACK_HOSTNAMES: Final[frozenset[str]] = frozenset(
    {"127.0.0.1", "localhost", "::1", "[::1]"}
)


def _hostname_of(host_header: str) -> str:
    """The hostname part of a `Host` header, lower-cased and port-stripped.

    Handles the bracketed IPv6 form `[::1]:51234` as well as the plain
    `127.0.0.1:51234`; an unbracketed IPv6 literal (`::1`) carries no
    port and is returned as-is.
    """
    host = host_header.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            return host[: end + 1]
        return host
    if host.count(":") == 1:
        return host.split(":", 1)[0]
    return host


def is_loopback_host(host_header: str | None) -> bool:
    """Whether a `Host` header names a loopback host (any port)."""
    if not host_header:
        return False
    return _hostname_of(host_header) in LOOPBACK_HOSTNAMES


def is_allowed_origin(origin: str | None) -> bool:
    """Whether a request's `Origin` is acceptable for the embedded server.

    An ABSENT origin is allowed (non-browser clients never send one);
    a present one must be `http://<loopback host>[:port]`. The scheme is
    pinned to `http` because the server itself speaks plain HTTP on
    loopback — an `https://` origin (or the opaque `null`) belongs to a
    different, non-loopback context by definition.
    """
    if origin is None:
        return True
    value = origin.strip().lower()
    if not value.startswith("http://"):
        return False
    return is_loopback_host(value[len("http://") :])


def _bearer_token(scope: Scope) -> str | None:
    """The token of an `Authorization: Bearer …` header, or None."""
    for raw_name, raw_value in scope.get("headers", ()):
        if raw_name.decode("latin-1").lower() != AUTH_HEADER:
            continue
        value = raw_value.decode("latin-1").strip()
        if value.lower().startswith(BEARER_PREFIX):
            return value[len(BEARER_PREFIX) :].strip()
        return None
    return None


def _header(scope: Scope, name: str) -> str | None:
    wanted = name.lower()
    for raw_name, raw_value in scope.get("headers", ()):
        if raw_name.decode("latin-1").lower() == wanted:
            return raw_value.decode("latin-1")
    return None


def token_matches(expected: str, presented: str | None) -> bool:
    """Constant-time comparison of the presented token with the expected one.

    `hmac.compare_digest` on purpose: a plain `==` leaks the length of the
    matching prefix through timing, which is exactly the signal a local
    brute-forcer would use. An empty expectation never matches (a
    misconfigured server must fail closed, not accept everything).
    """
    if not expected or presented is None:
        return False
    return hmac.compare_digest(expected, presented)


class GuardMiddleware:
    """Pure ASGI middleware: Origin/Host validation + bearer-token auth.

    Wraps the SDK's Streamable HTTP app. Only the `/mcp` path is guarded
    (everything else the parent app routes — `/healthz` — passes
    untouched, though it never goes through this middleware thanks to the
    parent's routing); the check is repeated path-exact so the middleware
    stays safe to mount over a wider tree.
    """

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # websocket / lifespan: pass through
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path != MCP_PATH and not path.startswith(f"{MCP_PATH}/"):
            await self.app(scope, receive, send)
            return
        denial = self._deny(scope)
        if denial is not None:
            await denial(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def _deny(self, scope: Scope) -> Response | None:
        """The rejection response for an unacceptable request, or None.

        Order matters: the header checks come first because they are the
        cheap, informative-to-the-operator gate, and the token challenge
        is what a compliant MCP client actually reacts to.
        """
        if not is_loopback_host(_header(scope, "host")):
            # DNS rebinding: a browser talking to evil.example that
            # resolves to loopback carries its own Host header.
            return PlainTextResponse("Invalid Host header", status_code=421)
        if not is_allowed_origin(_header(scope, "origin")):
            return PlainTextResponse("Invalid Origin header", status_code=403)
        if not token_matches(self.token, _bearer_token(scope)):
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="drcode"'},
            )
        return None
