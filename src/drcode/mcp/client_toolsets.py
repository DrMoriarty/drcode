"""External MCP servers as tool sources — toolset assembly.

The single seam between the mcp-servers.json config and the agents
(§2 of docs/mcp-client-design.md): `mcp_toolsets_for(role, ...)` reads
the two config layers through `McpServersStore` and returns the
pydantic-ai toolsets for ONE role. `agent_build` stays thin — the
builders just accept a precomputed list; the pipeline callers resolve
it at activation.

Guarantees (stage 1):

- **Safe default**: an empty/absent `roles` matrix yields `[]` for every
  role — no config, no MCP tools, exactly the pre-feature behavior.
- **Fresh instances per activation**: `MCPToolset`/`Client` objects are
  created anew on every call — SDK objects are per-instance state and
  must never be reused across a stop/start cycle (the wiring.py lesson),
  and the refcounted client cache would not survive a mid-run
  `list_changed` notification anyway. No module-level singletons.
- **Prefixed names only**: every toolset is wrapped in
  `PrefixedToolset(prefix=...)`, so server tools reach the model as
  `<prefix>_<tool_name>` — never in bare form, which rules out
  collisions with the native tools (`read_file`, `run_bash`, ...) and
  between two servers.
- **Lazy and never fatal**: the connection opens on first use (the
  MCPToolset session is lazy/refcounted); the fastmcp imports happen on
  first assembly, and a missing/broken fastmcp degrades to `[]` + a
  warning, never a crash.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from drcode.mcp.client_config import McpServerEntry, McpServersStore
from drcode.providers import AgentRole

# Warn sink for operational notes (service events at the callers): receives
# the text only — config secrets (env/headers) never enter these strings.
Notify = Callable[[str], None]


def resolve_prefix(server: McpServerEntry) -> str:
    """The effective tool-name prefix of a server record.

    `prefix` when set, else the id. An explicit EMPTY prefix (the §2.2
    opt-out) is coerced to the id: bare server names in the run would
    risk collisions with the native tools, and the synchronous assembly
    cannot inspect the server's tool list to check for them — degrading
    to the safe prefixed behavior is the documented outcome.
    """
    return (server.prefix or "").strip() or server.id


def _default_notify(text: str) -> None:  # pragma: no cover - trivial
    return None


def mcp_toolsets_for(
    role: AgentRole,
    project_root: Path | None = None,
    notify: Notify | None = None,
) -> list:
    """Assemble the MCP toolsets of ONE role (fresh instances, every call).

    - `role`: the serving role. COMPACTION and SUBAGENT always get `[]` —
      the summarizer must never carry tools, and the read-only research
      subagents must never gain external side-effecting ones.
    - `project_root`: when given, the project layer
      (`<root>/.drcode/mcp-servers.json`) is read on top of the user
      layer; the project record wins per id.
    - `notify`: optional sink for warning texts (corrupted config layers,
      skipped servers, prefix collisions) — the callers turn them into
      service events; failures inside the sink are swallowed here.
    """
    warn = notify or _default_notify
    if role in (AgentRole.COMPACTION, AgentRole.SUBAGENT):
        return []

    store = McpServersStore(
        project_path=(
            project_root / ".drcode" / "mcp-servers.json"
            if project_root is not None
            else None
        )
    )
    try:
        config, warnings = store.load()
    except Exception as exc:  # noqa: BLE001 - a config problem must not
        warn(  # kill a run
            f"MCP config load failed ({type(exc).__name__}); no MCP tools"
        )
        return []
    for warning in warnings:
        warn(warning)

    role_ids = config.roles.get(role.value, [])
    if not role_ids:
        return []

    try:
        toolset_cls, client_cls, stdio_transport_cls, default_env = (
            _import_client_stack(warn)
        )
    except ImportError:
        return []

    toolsets = []
    used_prefixes: set[str] = set()
    for server_id in role_ids:
        server = config.server_by_id(server_id)
        if server is None:
            warn(
                f"MCP server '{server_id}' listed for role '{role.value}' "
                "but not defined in any mcp-servers.json layer — skipped"
            )
            continue
        if not server.enabled:
            warn(f"MCP server '{server.id}' is disabled — skipped")
            continue
        toolset = _build_toolset(
            server,
            toolset_cls,
            client_cls,
            stdio_transport_cls,
            default_env,
            used_prefixes,
            warn,
        )
        if toolset is not None:
            toolsets.append(toolset)
    return toolsets


def _import_client_stack(warn: Notify) -> tuple:
    """Import the fastmcp-backed client stack lazily.

    drcode's server code never imports fastmcp, so a missing/broken
    install would otherwise surface as an unrelated startup crash;
    here it degrades to "no MCP tools" + a warning instead.
    """
    try:
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport
        from mcp.client.stdio import get_default_environment
        from pydantic_ai.mcp import MCPToolset
    except ImportError as exc:
        warn(
            "fastmcp/MCPToolset unavailable "
            f"({type(exc).__name__}: {exc}); no MCP tools for this run"
        )
        raise
    return MCPToolset, Client, StdioTransport, get_default_environment()


def _build_toolset(
    server: McpServerEntry,
    toolset_cls,
    client_cls,
    stdio_transport_cls,
    default_env: dict[str, str],
    used_prefixes: set[str],
    warn: Notify,
):
    """Build one server's toolset: PrefixedToolset(MCPToolset(...)).

    The MCPToolset is constructed but NOT connected — the session opens
    lazily on first `list_tools`/`call_tool` and terminates on exit; a
    server that is down surfaces at use time, not here.
    """
    prefix = resolve_prefix(server)
    if prefix in used_prefixes:
        # Two enabled servers of the role resolving to the same prefix:
        # force-disambiguate the LATER one instead of silently merging
        # their tool namespaces under one prefix.
        forced = f"{prefix}_{server.id}"
        warn(
            f"MCP servers '{prefix}' prefix collision — server "
            f"'{server.id}' forced to prefix '{forced}'"
        )
        prefix = forced
    used_prefixes.add(prefix)

    common = {
        "id": server.id,
        "tool_error_behavior": "retry",  # call errors → ModelRetry for the model
    }
    if server.transport == "stdio":
        # The record's env is ADDED to the SDK's whitelist-based default
        # environment; the host process's own environment never leaks
        # wholesale (documented §1.2 property, made explicit here).
        env = {**default_env, **server.env}
        transport = stdio_transport_cls(
            server.command, server.args, env=env, cwd=server.cwd
        )
        # A pre-built Client carries the timeouts itself — MCPToolset
        # refuses init_timeout/read_timeout alongside one.
        client = client_cls(
            transport,
            init_timeout=server.init_timeout,
            timeout=server.call_timeout,
        )
        inner = toolset_cls(client, **common)
    else:
        inner = toolset_cls(
            server.url,
            headers=server.headers or None,
            init_timeout=server.init_timeout,
            read_timeout=server.call_timeout,
            **common,
        )
    from pydantic_ai.toolsets import PrefixedToolset

    return PrefixedToolset(inner, prefix)
