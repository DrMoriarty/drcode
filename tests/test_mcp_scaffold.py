"""The bundled MCP-servers template: package data + per-project scaffolding.

`ensure_project_example` (drcode.mcp.client_config) drops
`.drcode/mcp-servers.json.example` into every project on startup — strictly
if-absent and never fatal — so installed users (uv tool install / pip /
binary) get the configuration hint that the repository copy used to carry.
"""

from __future__ import annotations

import json
from pathlib import Path

from drcode.mcp.client_config import (
    _EXAMPLE_PATH,
    MCP_EXAMPLE_FILENAME,
    MCP_SERVERS_FILENAME,
    McpServersFile,
    ensure_project_example,
    project_example_file,
)
from drcode.storage import Storage

# -- package data -----------------------------------------------------------


def test_template_is_shipped_as_package_data() -> None:
    """The template lives next to client_config.py and is non-empty."""
    assert _EXAMPLE_PATH == Path(__file__).parent.parent / (
        "src/drcode/mcp/mcp-servers.json.example"
    )
    assert _EXAMPLE_PATH.is_file()


def test_package_template_parses_as_mcp_servers_file() -> None:
    """The shipped template must stay in sync with the schema: a full
    `McpServersFile.model_validate` round-trip (extra `_`-keys ignored,
    the three example servers and the roles matrix survive)."""
    raw = json.loads(_EXAMPLE_PATH.read_text(encoding="utf-8"))
    parsed = McpServersFile.model_validate(raw)
    assert [s.id for s in parsed.servers] == [
        "ssh-remote",
        "filesystem",
        "project-gateway",
    ]
    assert parsed.roles["coding"] == ["ssh-remote", "filesystem"]
    assert parsed.tool_budget_tokens == 10_000
    # Every example server ships disabled — the template must never be
    # able to start a real process by being copied verbatim.
    assert all(not s.enabled for s in parsed.servers)


def test_template_roles_keys_are_known() -> None:
    """The template's roles matrix only references role keys the loader
    knows (MCP_ROLE_KEYS) — a renamed role must not silently strand the
    template's servers."""
    from drcode.mcp.client_config import MCP_ROLE_KEYS

    raw = json.loads(_EXAMPLE_PATH.read_text(encoding="utf-8"))
    assert set(raw["roles"]) <= set(MCP_ROLE_KEYS)


# -- scaffolding -------------------------------------------------------------


def test_scaffold_creates_example_lazily(project_root: Path) -> None:
    """The `.drcode/` directory is created only by the write itself."""
    assert not (project_root / ".drcode").exists()
    ensure_project_example(project_root)
    target = project_example_file(project_root)
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == _EXAMPLE_PATH.read_text(
        encoding="utf-8"
    )
    assert target.name == MCP_EXAMPLE_FILENAME


def test_scaffold_never_overwrites_existing(project_root: Path) -> None:
    """A user-edited (or deleted-and-customized) template is never touched."""
    target = project_example_file(project_root)
    target.parent.mkdir(parents=True)
    target.write_text('{"custom": true}', encoding="utf-8")
    ensure_project_example(project_root)
    assert target.read_text(encoding="utf-8") == '{"custom": true}'


def test_scaffold_idempotent(project_root: Path) -> None:
    ensure_project_example(project_root)
    target = project_example_file(project_root)
    ensure_project_example(project_root)
    assert target.read_text(encoding="utf-8") == _EXAMPLE_PATH.read_text(
        encoding="utf-8"
    )


def test_scaffold_never_fatal_when_drcode_is_a_file(project_root: Path) -> None:
    """`.drcode` occupied by a regular file makes the lazy mkdir fail —
    the scaffold must swallow it (no permissions / odd layouts rhyme)."""
    (project_root / ".drcode").write_text("not a dir", encoding="utf-8")
    ensure_project_example(project_root)  # must not raise
    assert (project_root / ".drcode").is_file()


def test_scaffold_never_fatal_when_root_missing(tmp_path: Path) -> None:
    """A nonexistent project root: the lazy mkdir creates the chain and
    the file lands (the OSError guard covers the unwritable variants)."""
    ensure_project_example(tmp_path / "gone")  # must not raise
    assert project_example_file(tmp_path / "gone").is_file()


def test_storage_init_scaffolds_example(project_root: Path) -> None:
    """The startup hook: TaskManager → Storage.__init__ scaffolds the
    template next to the gitignore pass."""
    Storage(project_root)
    assert project_example_file(project_root).is_file()


def test_scaffold_does_not_touch_real_servers_file(project_root: Path) -> None:
    ensure_project_example(project_root)
    assert not (project_root / ".drcode" / MCP_SERVERS_FILENAME).exists()
