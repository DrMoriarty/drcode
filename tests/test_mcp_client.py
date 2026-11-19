"""MCP client, stage 1 (docs/mcp-client-design.md §1-§2, §6-§7).

Covers the config layer (McpServersStore: two-layer merge, tolerance,
gitignore entry, mode-0600 writes), the role gating of
`mcp_toolsets_for` (empty matrix = no tools, ever) and the builder seam
(`toolsets=` is a pure extension point: the default builds stay exactly
as before, prefixed tools appear when passed).
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset, PrefixedToolset
from pydantic_ai.usage import RunUsage

from drcode import gitignore
from drcode.agent_build import (
    build_agent,
    build_resolver_agent,
    build_resolver_reviewer_agent,
    build_tester_agent,
)
from drcode.config import Config
from drcode.mcp.client_config import (
    MCP_SERVERS_FILENAME,
    McpServersFile,
    McpServersStore,
    project_servers_file,
    user_servers_file,
)
from drcode.mcp.client_toolsets import mcp_toolsets_for, resolve_prefix
from drcode.models import TaskKind
from drcode.providers import AgentRole

ALL_AGENT_ROLES = (
    AgentRole.CODING,
    AgentRole.RESEARCH,
    AgentRole.TESTER,
    AgentRole.RESOLVER,
)


@pytest.fixture
def fake_config() -> Config:
    return Config(project_root=Path.cwd(), model="test-model", api_key="sk-test")


# -- helpers ---------------------------------------------------------------


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def user_config(data: object) -> None:
    write_json(user_servers_file(), data)


def project_config(root: Path, data: object) -> None:
    write_json(project_servers_file(root), data)


def _tool_names(toolsets: list) -> list[str]:
    return sorted(t.prefix for t in toolsets if isinstance(t, PrefixedToolset))


def _agent_tool_names(agent) -> list[str]:
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), retries={})

    async def run():
        tools = await agent._get_toolset().get_tools(ctx)
        return sorted(t if isinstance(t, str) else t.name for t in tools)

    import asyncio

    return asyncio.run(run())


# -- config: load / merge / tolerance ---------------------------------------


def test_missing_files_yield_empty_default(project_root: Path) -> None:
    config, warnings = McpServersStore(
        project_path=project_servers_file(project_root)
    ).load()
    assert config.servers == []
    assert config.roles == {}
    assert warnings == []


def test_merge_union_by_id_project_wins(project_root: Path) -> None:
    user_config(
        {
            "servers": [
                {
                    "id": "ssh-remote",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@tufantunc/ssh-mcp"],
                },
                {
                    "id": "legacy",
                    "transport": "http",
                    "url": "http://127.0.0.1:9000/mcp",
                },
            ],
            "roles": {"coding": ["ssh-remote", "legacy"]},
        }
    )
    # The project redefines ssh-remote (transport change) and adds one.
    project_config(
        project_root,
        {
            "servers": [
                {
                    "id": "ssh-remote",
                    "transport": "http",
                    "url": "http://127.0.0.1:9001/mcp",
                    "headers": {"Authorization": "Bearer x"},
                },
                {"id": "extra", "transport": "stdio", "command": "uvx", "args": ["m"]},
            ],
            "roles": {"coding": ["ssh-remote", "extra"]},
        },
    )
    config, warnings = McpServersStore(
        project_path=project_servers_file(project_root)
    ).load()
    assert [s.id for s in config.servers] == ["ssh-remote", "legacy", "extra"]
    ssh = config.server_by_id("ssh-remote")
    assert ssh is not None and ssh.transport == "http"
    assert ssh.headers == {"Authorization": "Bearer x"}
    # A role key present in the project file overrides that role's list.
    assert config.roles == {"coding": ["ssh-remote", "extra"]}
    assert warnings == []


def test_corrupted_file_is_skipped_with_warning(project_root: Path) -> None:
    user_config(
        {
            "servers": [{"id": "ok", "transport": "stdio", "command": "x"}],
            "roles": {"coding": ["ok"]},
        }
    )
    secret = "Bearer super-secret-token"
    broken_path = project_servers_file(project_root)
    broken_path.parent.mkdir(parents=True, exist_ok=True)
    broken_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "id": "p",
                        "transport": "http",
                        "url": "http://x/mcp",
                        "headers": {"Authorization": secret},
                    }
                ]
            }
        ).replace("}", "broken"),
        encoding="utf-8",
    )
    config, warnings = McpServersStore(
        project_path=project_servers_file(project_root)
    ).load()
    # The broken layer is skipped WHOLE, the user layer survives.
    assert [s.id for s in config.servers] == ["ok"]
    assert len(warnings) == 1 and "invalid JSON" in warnings[0]
    # Secrets never reach the warning texts (file paths and error classes
    # only — the env/headers content is never quoted).
    assert all(secret not in w for w in warnings)


def test_junk_records_dropped_valid_ones_survive(project_root: Path) -> None:
    user_config(
        {
            "servers": [
                {"id": "good", "transport": "stdio", "command": "x"},
                {"id": "no-transport"},  # stdio without command → junk
                {"junk": True},  # no id → junk
                "not-even-a-dict",
                42,
            ],
            "roles": {"tester": ["good", "no-transport"], "nonsense": ["x"]},
        }
    )
    config, warnings = McpServersStore(
        project_path=project_servers_file(project_root)
    ).load()
    assert [s.id for s in config.servers] == ["good"]
    # Unknown role keys and references to dropped records are not fatal —
    # the resolver just never finds them.
    assert config.roles == {"tester": ["good", "no-transport"]}
    assert warnings == []


def test_record_coercion_is_tolerant() -> None:
    file = McpServersFile.model_validate(
        {
            "servers": [
                {
                    "id": "ssh",
                    "transport": "stdio",
                    "command": "npx",
                    "env": {"N": 1, "S": "v"},  # non-str values stringified
                    "init_timeout": "junk",  # → default
                    "call_timeout": -5,  # → default
                    "unknown_key": "ignored",  # tolerated, not fatal
                },
                {"id": "", "transport": "stdio", "command": "x"},  # junk
                {"id": "h", "transport": "http"},  # http without url → junk
            ]
        }
    )
    assert [s.id for s in file.servers] == ["ssh"]
    entry = file.servers[0]
    assert entry.env == {"N": "1", "S": "v"}
    assert entry.init_timeout == 5.0 and entry.call_timeout == 300.0


def test_project_file_with_roles_only_is_valid(project_root: Path) -> None:
    user_config(
        {
            "servers": [{"id": "ssh", "transport": "stdio", "command": "x"}],
            "roles": {"coding": ["ssh"], "tester": ["ssh"]},
        }
    )
    project_config(project_root, {"roles": {"tester": []}})
    config, warnings = McpServersStore(
        project_path=project_servers_file(project_root)
    ).load()
    assert config.roles == {"coding": ["ssh"], "tester": []}
    assert warnings == []


def test_gitignore_entry_present() -> None:
    # The project file carries secrets (env/headers) — it must be in the
    # private subset, next to the `.drcode/mcp.json` precedent.
    assert ".drcode/mcp-servers.json" in gitignore.GITIGNORE_ENTRIES


def test_save_project_writes_mode_0600(project_root: Path) -> None:
    path = project_servers_file(project_root)
    store = McpServersStore(project_path=path)
    data = McpServersFile.model_validate(
        {
            "servers": [
                {
                    "id": "ssh",
                    "transport": "http",
                    "url": "http://x/mcp",
                    "headers": {"Authorization": "Bearer s"},
                }
            ],
            "roles": {"coding": ["ssh"]},
        }
    )
    written = store.save_project(data)
    assert written == path
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600
    roundtripped, _ = McpServersStore(project_path=path).load()
    assert roundtripped.servers[0].headers == {"Authorization": "Bearer s"}


def test_save_refuses_default_resolved_store_inside_agent(monkeypatch) -> None:
    # Incident ebe6ba15 pattern: a store constructed inside a drcode agent's
    # verification snippet must never write the real user-level catalog.
    monkeypatch.setenv("DRCODE_TASK_ID", "4bcf3332")
    with pytest.raises(ValueError, match="drcode run"):
        McpServersStore(user_path=None).save_user(McpServersFile())


def test_user_file_path_resolves_dynamically(monkeypatch, tmp_path: Path) -> None:
    from drcode import config as config_module

    monkeypatch.setattr(config_module, "_USER_CONFIG_DIR", tmp_path / "cfg")
    assert user_servers_file() == tmp_path / "cfg" / MCP_SERVERS_FILENAME


# -- gating (§1.3) -----------------------------------------------------------


def test_empty_matrix_yields_no_toolsets_for_any_role(
    project_root: Path,
) -> None:
    user_config({"servers": [{"id": "ssh", "transport": "stdio", "command": "x"}]})
    for role in ALL_AGENT_ROLES:
        (toolsets,) = (mcp_toolsets_for(role, project_root=project_root),)
        assert toolsets == []
    # With no files at all, too.
    empty_root = project_root / "nowhere"
    for role in ALL_AGENT_ROLES:
        assert mcp_toolsets_for(role, project_root=empty_root) == []


def test_compaction_and_subagent_never_get_toolsets(project_root: Path) -> None:
    # COMPACTION is listed in the format for completeness; the summarizer
    # must never carry tools. SUBAGENT (read-only research) is not gateable
    # at all.
    user_config(
        {
            "servers": [{"id": "ssh", "transport": "stdio", "command": "x"}],
            "roles": {
                "compaction": ["ssh"],
                "coding": ["ssh"],
                "tester": ["ssh"],
                "research": ["ssh"],
                "resolver": ["ssh"],
            },
        }
    )
    assert mcp_toolsets_for(AgentRole.COMPACTION, project_root=project_root) == []
    assert mcp_toolsets_for(AgentRole.SUBAGENT, project_root=project_root) == []
    for role in (AgentRole.CODING, AgentRole.TESTER):
        assert mcp_toolsets_for(role, project_root=project_root)


def test_role_gating_is_exact(project_root: Path) -> None:
    user_config(
        {
            "servers": [{"id": "ssh", "transport": "stdio", "command": "x"}],
            "roles": {"coding": ["ssh"]},
        }
    )
    coding = mcp_toolsets_for(AgentRole.CODING, project_root=project_root)
    assert _tool_names(coding) == ["ssh"]  # prefix defaults to the id
    assert all(isinstance(t, PrefixedToolset) for t in coding)
    for role in (AgentRole.RESEARCH, AgentRole.TESTER, AgentRole.RESOLVER):
        assert mcp_toolsets_for(role, project_root=project_root) == []


def test_disabled_server_skipped_with_note(project_root: Path) -> None:
    user_config(
        {
            "servers": [
                {
                    "id": "off",
                    "transport": "stdio",
                    "command": "x",
                    "enabled": False,
                }
            ],
            "roles": {"coding": ["off"]},
        }
    )
    notes: list[str] = []
    assert (
        mcp_toolsets_for(
            AgentRole.CODING, project_root=project_root, notify=notes.append
        )
        == []
    )
    assert len(notes) == 1 and "disabled" in notes[0]


def test_unknown_server_id_warns_but_never_fails(project_root: Path) -> None:
    user_config({"roles": {"coding": ["ghost"]}})
    notes: list[str] = []
    assert (
        mcp_toolsets_for(
            AgentRole.CODING, project_root=project_root, notify=notes.append
        )
        == []
    )
    assert len(notes) == 1 and "ghost" in notes[0]


def test_project_layer_read_from_drcode_dir(project_root: Path) -> None:
    project_config(
        project_root,
        {
            "servers": [
                {
                    "id": "proj",
                    "transport": "http",
                    "url": "http://127.0.0.1:9000/mcp",
                    "prefix": "pfx",
                }
            ],
            "roles": {"research": ["proj"]},
        },
    )
    toolsets = mcp_toolsets_for(AgentRole.RESEARCH, project_root=project_root)
    assert _tool_names(toolsets) == ["pfx"]


def test_prefix_collision_forces_disambiguation(project_root: Path) -> None:
    user_config(
        {
            "servers": [
                {
                    "id": "one",
                    "transport": "stdio",
                    "command": "x",
                    "prefix": "ssh",
                },
                {
                    "id": "two",
                    "transport": "stdio",
                    "command": "y",
                    "prefix": "ssh",
                },
            ],
            "roles": {"coding": ["one", "two"]},
        }
    )
    notes: list[str] = []
    toolsets = mcp_toolsets_for(
        AgentRole.CODING, project_root=project_root, notify=notes.append
    )
    assert _tool_names(toolsets) == ["ssh", "ssh_two"]
    assert any("collision" in n for n in notes)


def test_resolve_prefix_coerces_empty_to_id() -> None:
    from drcode.mcp.client_config import McpServerEntry

    base = {"transport": "stdio", "command": "x"}
    assert resolve_prefix(McpServerEntry.model_validate({"id": "a", **base})) == "a"
    assert (
        resolve_prefix(McpServerEntry.model_validate({"id": "a", "prefix": "", **base}))
        == "a"
    )
    assert (
        resolve_prefix(
            McpServerEntry.model_validate({"id": "a", "prefix": "ssh", **base})
        )
        == "ssh"
    )


# -- the builder seam (§2.1) -------------------------------------------------


def test_default_builders_register_no_user_toolsets(fake_config: Config) -> None:
    # Regression: with no MCP config the agents must be constructed exactly
    # as before the feature — `toolsets=None`, empty user toolsets.
    assert build_agent(fake_config)._user_toolsets == []
    assert build_tester_agent(fake_config)._user_toolsets == []
    assert build_resolver_agent(fake_config)._user_toolsets == []
    assert build_resolver_reviewer_agent(fake_config)._user_toolsets == []
    # An explicitly EMPTY list is the same thing.
    assert build_agent(fake_config, mcp_toolsets=[])._user_toolsets == []


def test_passed_toolsets_land_under_prefix_and_native_set_unchanged(
    fake_config: Config,
) -> None:
    def ping() -> str:
        return "pong"

    native = _agent_tool_names(build_agent(fake_config))
    mcp_agent = build_agent(
        fake_config,
        mcp_toolsets=[PrefixedToolset(FunctionToolset([ping]), "mcpx")],
    )
    names = _agent_tool_names(mcp_agent)
    assert "mcpx_ping" in names
    # No bare names leak: the prefixed toolset adds exactly one tool.
    assert [n for n in names if not n.startswith("mcpx_")] == native


def test_two_servers_same_tool_names_do_not_collide(fake_config: Config) -> None:
    def ping() -> str:
        return "pong"

    mcp_agent = build_agent(
        fake_config,
        kind=TaskKind.FEATURE,
        mcp_toolsets=[
            PrefixedToolset(FunctionToolset([ping]), "one"),
            PrefixedToolset(FunctionToolset([ping]), "two"),
        ],
    )
    names = _agent_tool_names(mcp_agent)
    assert "one_ping" in names and "two_ping" in names


def test_all_builder_kinds_accept_the_seam(fake_config: Config) -> None:
    def ping() -> str:
        return "pong"

    toolsets = [PrefixedToolset(FunctionToolset([ping]), "m")]
    for builder in (
        build_tester_agent,
        build_resolver_agent,
        build_resolver_reviewer_agent,
    ):
        agent = builder(fake_config, mcp_toolsets=toolsets)
        assert len(agent._user_toolsets) == 1
