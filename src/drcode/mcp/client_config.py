"""External MCP servers as tool sources — configuration.

The config layer of the MCP client (§1 of docs/mcp-client-design.md):
external MCP servers an agent may connect to live in their OWN catalog
file, `mcp-servers.json`, by the model of `models.json` — a list of
records with nested env/headers secrets does not belong into
`drcode.json` (whose `_EXAMPLE_PLACEHOLDERS` machinery) or
`settings.json` (deliberately committable, see `mcp/endpoint.py` for
the "secrets never in a committed file" rule).

Two layers, merged on load:

- user level: ``~/.config/drcode/mcp-servers.json`` — servers useful
  everywhere (e.g. SSH execution);
- project level: ``.drcode/mcp-servers.json`` — servers of one project.

Union by record ``id``; on a duplicate the PROJECT record wins ("the
specific layer sits on top", as in ``providers._resolve_connection``).
The project file contains secrets (env/headers), so it is listed in
``gitignore.GITIGNORE_ENTRIES`` and written with mode 0600.

Tolerance rules (the "isError, never fatal" philosophy of the server
stack): a corrupted/unreadable file is skipped WHOLE with a warning
(never an exception, the file is left for the user to fix), while junk
records inside a valid file are dropped individually — one bad server
never invalidates the others (the `_normalized_hosts` pattern of
`settings.py`).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from drcode import config as _config

MCP_SERVERS_FILENAME = "mcp-servers.json"

# Role keys of the per-role gating matrix ("roles" in the file). COMPACTION
# is listed for completeness of the format; the implementation IGNORES it —
# the compaction summarizer must never carry tools at all.
MCP_ROLE_KEYS: tuple[str, ...] = (
    "coding",
    "tester",
    "research",
    "resolver",
    "compaction",
)

DEFAULT_INIT_TIMEOUT = 5.0
DEFAULT_CALL_TIMEOUT = 300.0

# §3.3 of the design doc: after a successful list_tools the tool list is
# cost-estimated by the chars/4 heuristic (transcript.estimate_tokens); a
# server whose list exceeds the budget does NOT attach — one full catalog
# from a real server (bvisible/mcp-ssh-manager, 37 tools) is ≈43.5k "tokens",
# a third of a typical window. Overridable per file with the top-level
# `tool_budget_tokens` key (project layer wins on merge).
DEFAULT_TOOL_BUDGET_TOKENS = 10_000


class McpServerEntry(BaseModel):
    """One external MCP server record (§1.2 of the design doc).

    Validation failures are not allowed to escape to the file level:
    `McpServersFile`'s tolerant before-validator drops a record that does
    not validate, keeping the rest of the file alive.
    """

    model_config = ConfigDict(extra="ignore")  # tolerant to unknown keys

    id: str
    transport: Literal["stdio", "http"] = "stdio"
    # stdio:
    command: str = ""
    args: list[str] = Field(default_factory=list)
    # ADDS to the mcp-SDK default environment (whitelist-based) — the host
    # process's own environment never leaks to the server wholesale (§1.2).
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    # http:
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    # common (tools_allow/tools_deny budgets are stage 2, the fields are
    # part of the format already — tolerated and preserved, unused):
    tools_allow: list[str] = Field(default_factory=list)
    tools_deny: list[str] = Field(default_factory=list)
    prefix: str | None = None  # default tool-name prefix = id
    init_timeout: float = DEFAULT_INIT_TIMEOUT
    call_timeout: float = DEFAULT_CALL_TIMEOUT  # read_timeout of the client
    enabled: bool = True

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        # Non-string env/header values are stringified, not fatal: the
        # transport needs strings anyway.
        for key in ("env", "headers"):
            value = data.get(key)
            if isinstance(value, dict):
                data[key] = {
                    str(k): v if isinstance(v, str) else str(v)
                    for k, v in value.items()
                }
        for key in ("id", "command", "url"):
            if isinstance(data.get(key), str):
                data[key] = data[key].strip()
        for key in ("prefix", "cwd"):
            if isinstance(data.get(key), str):
                data[key] = data[key].strip() or None
        timeout_fields = (
            ("init_timeout", DEFAULT_INIT_TIMEOUT),
            ("call_timeout", DEFAULT_CALL_TIMEOUT),
        )
        for key, default in timeout_fields:
            value = data.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                data[key] = default  # junk → default, never fatal
            elif value <= 0:
                data[key] = default
        return data

    @model_validator(mode="after")
    def _check_transport(self) -> McpServerEntry:
        if not self.id:
            raise ValueError("server record requires a non-empty 'id'")
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio server requires 'command'")
        if self.transport == "http" and not self.url:
            raise ValueError("http server requires 'url'")
        return self


class McpServersFile(BaseModel):
    """The parsed content of one mcp-servers.json file (tolerant)."""

    model_config = ConfigDict(extra="ignore")

    servers: list[McpServerEntry] = Field(default_factory=list)
    roles: dict[str, list[str]] = Field(default_factory=dict)
    # Estimated-tokens ceiling for one server's tool list (§3.3); junk in
    # the file degrades to the default, never invalidates the layer.
    tool_budget_tokens: int = DEFAULT_TOOL_BUDGET_TOKENS

    @model_validator(mode="before")
    @classmethod
    def _tolerant(cls, data: Any) -> Any:
        """Drop junk records, keep valid ones — one bad server must never
        invalidate the file (settings.py's `_normalized_hosts` pattern)."""
        if not isinstance(data, dict):
            raise TypeError("mcp-servers.json must be a JSON object")
        data = dict(data)
        raw_servers = data.get("servers", [])
        servers: list[McpServerEntry] = []
        if isinstance(raw_servers, list):
            for raw in raw_servers:
                if not isinstance(raw, dict):
                    continue
                try:
                    servers.append(McpServerEntry.model_validate(raw))
                except ValidationError:
                    continue  # junk record dropped, the rest survives
        data["servers"] = servers
        raw_roles = data.get("roles", {})
        roles: dict[str, list[str]] = {}
        if isinstance(raw_roles, dict):
            for key, ids in raw_roles.items():
                if key not in MCP_ROLE_KEYS or not isinstance(ids, list):
                    continue  # unknown role key / junk value: ignored
                roles[key] = [i for i in ids if isinstance(i, str) and i.strip()]
        data["roles"] = roles
        budget = data.get("tool_budget_tokens")
        if (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or budget <= 0
            or (isinstance(budget, float) and not budget.is_integer())
        ):
            # Junk = the key is treated as ABSENT (the field default
            # applies, the layer does not override the other one).
            data.pop("tool_budget_tokens", None)
        else:
            data["tool_budget_tokens"] = int(budget)
        return data

    def server_by_id(self, server_id: str) -> McpServerEntry | None:
        for server in self.servers:
            if server.id == server_id:
                return server
        return None


def user_servers_file() -> Path:
    """Path of the user-level catalog (XDG_CONFIG_HOME-aware).

    Resolved dynamically through the config module (never imported by
    value): the conftest sandbox and the per-task run_bash redirect both
    repoint `config._USER_CONFIG_DIR`, and a binding captured at import
    time would pin the store to whichever env was live first (the
    incident-ebe6ba15 trap `usage_store.usage_file` documents).
    """
    return _config._USER_CONFIG_DIR / MCP_SERVERS_FILENAME


def project_servers_file(project_root: Path) -> Path:
    """Path of the project-level catalog: `<project>/.drcode/mcp-servers.json`."""
    return project_root / ".drcode" / MCP_SERVERS_FILENAME


class McpServersStore:
    """Loads the two mcp-servers.json layers and merges them.

    Constructor accepts explicit paths (tests, tooling); the defaults
    resolve dynamically at load time, so the conftest sandbox applies.
    `load()` never raises: corrupted layers come back as warnings (texts
    for service events — they contain file paths and error classes,
    never file CONTENT: env/headers secrets must not reach the log).
    """

    def __init__(
        self,
        user_path: Path | None = None,
        project_path: Path | None = None,
    ) -> None:
        self.user_path = user_path
        self.project_path = project_path

    def _read_layer(self, path: Path) -> tuple[McpServersFile | None, str | None]:
        """Parse one file: (file, None) on success, (None, warning) when the
        layer must be skipped whole, (None, None) when absent."""
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None, None  # absent layer is fine
        except (OSError, UnicodeDecodeError) as exc:
            return None, f"unreadable ({type(exc).__name__}), layer skipped"
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None, "invalid JSON, layer skipped"
        try:
            return McpServersFile.model_validate(data), None
        except (ValidationError, TypeError, ValueError) as exc:
            # Defensive: _tolerant absorbs junk records; a non-dict
            # document (TypeError from the before-validator) is still
            # only a skipped layer, never a crash.
            if isinstance(exc, ValidationError):
                return (
                    None,
                    f"invalid structure ({exc.error_count()} errors), layer skipped",
                )
            return None, "invalid structure, layer skipped"

    def load(self) -> tuple[McpServersFile, list[str]]:
        """Merged view (project wins per id) + warning texts.

        Role matrix merge: a role key present in the PROJECT file overrides
        that whole role's list (the specific layer sits on top); keys only
        in the user file pass through.
        """
        merged = McpServersFile()
        warnings: list[str] = []
        by_id: dict[str, McpServerEntry] = {}
        user_roles: dict[str, list[str]] = {}
        project_roles: dict[str, list[str]] | None = None
        user_budget: int | None = None
        project_budget: int | None = None

        for path, layer in self._layers():
            if path is None:
                continue
            parsed, warning = self._read_layer(path)
            if warning is not None:
                warnings.append(f"MCP config {path}: {warning}")
                continue
            if parsed is None:
                continue
            for server in parsed.servers:
                by_id[server.id] = server  # later (project) layer wins
            if layer == "project":
                project_roles = dict(parsed.roles)
                if "tool_budget_tokens" in parsed.model_fields_set:
                    project_budget = parsed.tool_budget_tokens
            else:
                user_roles.update(parsed.roles)
                if "tool_budget_tokens" in parsed.model_fields_set:
                    user_budget = parsed.tool_budget_tokens

        merged.servers = list(by_id.values())
        merged.roles = {**user_roles, **(project_roles or {})}
        # The specific layer sits on top, as for the role matrix.
        if project_budget is not None:
            merged.tool_budget_tokens = project_budget
        elif user_budget is not None:
            merged.tool_budget_tokens = user_budget
        return merged, warnings

    def _layers(self) -> list[tuple[Path | None, str]]:
        layers: list[tuple[Path | None, str]] = [
            (self.user_path or user_servers_file(), "user")
        ]
        project = self.project_path
        if project is None:
            return layers
        return [*layers, (project, "project")]

    def save_project(self, data: McpServersFile) -> Path:
        """Atomically write the PROJECT file with mode 0600.

        `tempfile.mkstemp` already opens with 0600 — the secrets are never
        readable by other users, not even for the instant before a later
        chmod; `os.replace` keeps the inode's mode (endpoint.py's
        save_endpoint pattern). Refuses to run inside a drcode agent's
        verification snippet without explicit paths (incident ebe6ba15: a
        default-resolved store must never clobber the real user files).
        """
        if self.project_path is None:
            raise ValueError(
                "save_project requires an explicit project_path"
                " (a default-resolved store must never write the real file)"
            )
        return self._write(self.project_path, data)

    def save_user(self, data: McpServersFile) -> Path:
        """Atomically write the USER file (mode 0600 — it carries secrets too)."""
        return self._write(self.user_path or user_servers_file(), data)

    @staticmethod
    def _write(path: Path, data: McpServersFile) -> Path:
        if os.environ.get("DRCODE_TASK_ID"):  # agent snippet guard
            raise ValueError(
                "refusing to write mcp-servers.json from inside a drcode run"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(json.loads(data.model_dump_json()), indent=2, ensure_ascii=False)
            + "\n"
        )
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return path
