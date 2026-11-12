"""The `.drcode/mcp.json` endpoint descriptor (design doc §2/§3).

The embedded MCP server listens on an **ephemeral** loopback port
(§1: a fixed default port would sooner or later collide with a foreign
service), so nothing could find it by convention — the port, the bearer
token and the owning TUI's pid are published in a small file next to the
rest of the project's private state:

```json
{"port": 51234, "token": "<64 hex chars>", "pid": 4711, "started_at": "…"}
```

Why a file and not the settings: the token is a SECRET, and
`.drcode/settings.json` is a user-editable, potentially committed file —
`mcp.json` is written with mode 0600 and is listed in the gitignore
entries of `drcode.gitignore`.

The file is also the mutex of the "one server = one project" rule
(§3): before writing it, a starting instance checks the descriptor it
finds — a live pid plus an answering `/healthz` means the project is
already served by another TUI, and the second instance must stay silent
instead of firing up a competing server over the same task files. A dead
pid or a silent port means the file is stale and gets overwritten.

Everything here is I/O-only and telemetry-free: an unreadable, corrupted
or partial descriptor is simply "no descriptor" (never fatal — a lost
file just costs a regenerated token, §2).
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx
from pydantic import BaseModel, Field, ValidationError

from ..config import PROJECT_STATE_DIRNAME

#: Name of the descriptor inside `<project>/.drcode/`.
ENDPOINT_FILENAME: Final = "mcp.json"

#: The Streamable HTTP endpoint path (§1) and the unauthenticated health probe.
MCP_PATH: Final = "/mcp"
HEALTH_PATH: Final = "/healthz"

#: Bind address of the server — loopback only, never a routable interface.
LOOPBACK_HOST: Final = "127.0.0.1"

#: Timeout of one `/healthz` probe. Deliberately short: the probe gates a
#: TUI's startup and must not hang on a half-dead foreign process.
PROBE_TIMEOUT: Final = 1.0

#: 32 bytes of entropy, hex-encoded (64 characters) — §2.
TOKEN_BYTES: Final = 32


def endpoint_path(project_root: Path) -> Path:
    """`<project>/.drcode/mcp.json`."""
    return Path(project_root) / PROJECT_STATE_DIRNAME / ENDPOINT_FILENAME


def new_token() -> str:
    """A fresh bearer token: 32 random bytes, hex-encoded (§2)."""
    return secrets.token_hex(TOKEN_BYTES)


class McpEndpoint(BaseModel):
    """The published endpoint of a running embedded MCP server.

    `port` is the ephemeral port the server actually bound (never 0 — it
    is resolved before the file is written, §6), `token` the bearer
    secret every `/mcp` request must carry, `pid` the owning TUI process
    (the "is this file still ours?" marker, §3) and `started_at` a
    human-readable stamp for debugging.
    """

    port: int = Field(ge=1, le=65535)
    token: str = Field(min_length=1)
    pid: int
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def url(self) -> str:
        """The full Streamable HTTP endpoint URL a client should connect to."""
        return f"http://{LOOPBACK_HOST}:{self.port}{MCP_PATH}"

    @property
    def health_url(self) -> str:
        """The unauthenticated liveness probe URL of this endpoint."""
        return f"http://{LOOPBACK_HOST}:{self.port}{HEALTH_PATH}"


def load_endpoint(project_root: Path) -> McpEndpoint | None:
    """Read the descriptor; None when absent, unreadable or corrupted.

    A corrupted file is treated exactly like a missing one: the next
    start overwrites it (and regenerates the token), which is the
    documented cost of losing the file (§2) — never a crash.
    """
    path = endpoint_path(project_root)
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return McpEndpoint.model_validate(data)
    except ValidationError:
        return None


def save_endpoint(project_root: Path, endpoint: McpEndpoint) -> Path:
    """Publish `endpoint` atomically, with mode 0600.

    The temp file is created by `tempfile.mkstemp`, which already opens
    with 0600 — the secret is never readable by other users, not even
    for the instant before a later `chmod`. `os.replace` keeps the
    inode's mode, so the published file inherits it.
    """
    path = endpoint_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(endpoint.model_dump_json(indent=2))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def remove_endpoint_if_owned_by(project_root: Path, pid: int) -> bool:
    """Delete the descriptor when `pid` owns it; returns whether it went.

    A file whose `pid` is somebody else's is left alone (§3: another TUI
    may be publishing it — deleting it would tear a stranger's endpoint
    down for everybody). An unreadable file is not ours either.
    """
    endpoint = load_endpoint(project_root)
    if endpoint is None or endpoint.pid != pid:
        return False
    try:
        endpoint_path(project_root).unlink()
    except OSError:
        return False
    return True


def pid_alive(pid: int) -> bool:
    """Whether the OS still has a process with this pid.

    Signal 0 on POSIX probes without touching the process; a
    `PermissionError` means it exists but belongs to somebody else (still
    alive). Windows gets the equivalent through `OpenProcess` — `os.kill`
    there would TERMINATE the process instead of probing it.
    """
    if pid <= 0:
        return False
    if os.name == "nt":  # pragma: no cover - Windows only
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pid_alive_windows(pid: int) -> bool:  # pragma: no cover - Windows only
    """`OpenProcess` + `GetExitCodeProcess` liveness probe (Windows)."""
    import ctypes

    # "windll" only exists on Windows; pyright's stubs hide it, ruff's
    # B009 forbids a constant getattr — a text lookup satisfies both.
    windll = getattr(ctypes, "windll")  # noqa: B009 - Windows-only attribute
    kernel32 = windll.kernel32
    process_query_limited_information = 0x1000
    still_active = 259
    handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong(0)
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True  # cannot tell — assume alive rather than steal the file
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def probe_healthz(port: int, *, timeout: float = PROBE_TIMEOUT) -> bool:
    """Whether the endpoint on `port` answers `/healthz` with `{"ok": true}`.

    `trust_env=False` on purpose: the probe targets loopback and must
    never be routed through an `HTTP_PROXY` the environment happens to
    carry (a proxy is exactly the way this check would silently lie).
    """
    url = f"http://{LOOPBACK_HOST}:{port}{HEALTH_PATH}"
    try:
        response = httpx.get(url, timeout=timeout, trust_env=False)
    except (httpx.HTTPError, OSError):
        return False
    if response.status_code != 200:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("ok") is True


def endpoint_is_live(endpoint: McpEndpoint) -> bool:
    """Whether `endpoint` describes a running server of a live process.

    Both halves of §3 are required: the pid must still exist (a crashing
    TUI leaves the file behind) AND the published port must still answer
    the health probe (a pid-reusing unrelated process must not keep a
    dead descriptor alive).
    """
    return pid_alive(endpoint.pid) and probe_healthz(endpoint.port)
