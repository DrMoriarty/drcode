"""Tests for drcode.config: layered configuration and user-config scaffolding."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import drcode.config as config_module
from drcode.config import (
    Config,
    _parsable_or_none,
    _resolve_user_config_dir,
    ensure_user_config_dir,
)


@pytest.fixture
def user_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A redirected (already-sandboxed) user config dir, as a plain path."""
    return config_module._USER_CONFIG_DIR


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _project_config(project_root: Path) -> Path:
    return project_root / ".drcode" / "drcode.json"


# -- is_ready -------------------------------------------------------------------


def test_is_ready(project_root: Path) -> None:
    assert not Config(project_root=project_root).is_ready
    # base_url defaults to the OpenAI API and api_key is optional (keyless
    # local gateways work): a model alone is enough to be ready.
    assert Config(project_root=project_root, model="m").is_ready
    assert not Config(project_root=project_root, api_key="k").is_ready
    assert Config(project_root=project_root, model="m", api_key="k").is_ready


def test_defaults(project_root: Path) -> None:
    cfg = Config(project_root=project_root)
    assert cfg.base_url == "https://api.openai.com/v1"
    assert cfg.max_context_tokens == 200_000
    assert cfg.compaction_threshold == 80
    assert cfg.request_limit == 500
    assert cfg.model == ""
    assert cfg.api_key == ""


# -- layering ---------------------------------------------------------------------


def test_env_beats_everything(project_root: Path, monkeypatch) -> None:
    _write_json(_project_config(project_root), {"compaction_threshold": 10})
    monkeypatch.setenv("DRCODE_COMPACTION_THRESHOLD", "99")
    cfg = Config(project_root=project_root)
    assert cfg.compaction_threshold == 99


def test_project_json_beats_user_json(
    project_root: Path, user_config_dir: Path
) -> None:
    _write_json(user_config_dir / "drcode.json", {"compaction_threshold": 20})
    _write_json(_project_config(project_root), {"compaction_threshold": 30})
    cfg = Config(project_root=project_root)
    assert cfg.compaction_threshold == 30


def test_user_json_beats_defaults(project_root: Path, user_config_dir: Path) -> None:
    _write_json(user_config_dir / "drcode.json", {"compaction_threshold": 40})
    cfg = Config(project_root=project_root)
    assert cfg.compaction_threshold == 40


def test_init_kwargs_win(project_root: Path, monkeypatch) -> None:
    monkeypatch.setenv("DRCODE_MAX_CONTEXT_TOKENS", "123")
    cfg = Config(project_root=project_root, max_context_tokens=777)
    assert cfg.max_context_tokens == 777


def test_corrupted_project_json_skipped(
    project_root: Path, user_config_dir: Path
) -> None:
    _write_json(user_config_dir / "drcode.json", {"compaction_threshold": 20})
    _project_config(project_root).parent.mkdir(parents=True, exist_ok=True)
    _project_config(project_root).write_text("{broken")
    cfg = Config(project_root=project_root)
    # Startup survives and falls through to the user-level layer.
    assert cfg.compaction_threshold == 20


def test_corrupted_user_json_skipped(project_root: Path) -> None:
    config_module._USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_module.USER_CONFIG_FILE.write_text("not json")
    cfg = Config(project_root=project_root)
    assert cfg.compaction_threshold == 80


def test_project_config_file_property(project_root: Path) -> None:
    cfg = Config(project_root=project_root)
    assert cfg.project_config_file == project_root / ".drcode" / "drcode.json"


def test_unknown_json_keys_ignored(project_root: Path) -> None:
    _write_json(_project_config(project_root), {"no_such_field": 1})
    cfg = Config(project_root=project_root)
    assert cfg.compaction_threshold == 80


# -- legacy project-root drcode.json -----------------------------------------


def test_legacy_root_config_still_read(project_root: Path) -> None:
    # Configs written by older versions sat in the project root; they are
    # honored until the file is re-saved to the new location.
    _write_json(project_root / "drcode.json", {"compaction_threshold": 33})
    cfg = Config(project_root=project_root)
    assert cfg.compaction_threshold == 33


def test_current_config_beats_legacy(project_root: Path) -> None:
    _write_json(project_root / "drcode.json", {"compaction_threshold": 11})
    _write_json(_project_config(project_root), {"compaction_threshold": 22})
    cfg = Config(project_root=project_root)
    assert cfg.compaction_threshold == 22


def test_json_layers_never_provide_model_fields(
    project_root: Path, user_config_dir: Path
) -> None:
    """The flat model keys of a drcode.json are ignored as settings (they
    are consumed by the startup migrations instead): only env/dotenv and
    the catalog/project selections provide the model."""
    _write_json(
        _project_config(project_root),
        {
            "model": "flat-m",
            "base_url": "http://flat/v1",
            "api_key": "flat-key",
            "compaction_threshold": 42,
        },
    )
    _write_json(
        user_config_dir / "drcode.json",
        {"model": "user-m", "max_context_tokens": 123},
    )
    cfg = Config(project_root=project_root)
    assert cfg.model == ""
    assert cfg.api_key == ""
    assert cfg.base_url == "https://api.openai.com/v1"  # built-in default
    assert cfg.max_context_tokens == 200_000  # not taken from user json
    # Non-provider settings still flow through the json layers.
    assert cfg.compaction_threshold == 42


# -- _parsable_or_none ---------------------------------------------------------------


def test_parsable_or_none(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert _parsable_or_none(missing) == missing  # absent is fine

    good = tmp_path / "good.json"
    good.write_text("{}")
    assert _parsable_or_none(good) == good

    broken = tmp_path / "broken.json"
    broken.write_text("{")
    assert _parsable_or_none(broken) is None

    binary = tmp_path / "bin.json"
    binary.write_bytes(b"\xff\xfe\x00")
    assert _parsable_or_none(binary) is None


# -- user config dir resolution (DRCODE_USER_CONFIG_DIR priority) ----------------------


def test_resolve_user_config_dir_env_beats_xdg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drcode-specific override wins over XDG_CONFIG_HOME: run_bash uses
    it to point task children at the per-task sandbox copy (incident
    ebe6ba15) without touching XDG itself (git/gh must keep working)."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("DRCODE_USER_CONFIG_DIR", str(tmp_path / "sandbox"))
    assert _resolve_user_config_dir() == tmp_path / "sandbox"


def test_resolve_user_config_dir_xdg_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("DRCODE_USER_CONFIG_DIR", raising=False)
    assert _resolve_user_config_dir() == tmp_path / "xdg" / "drcode"


def test_resolve_user_config_dir_home_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DRCODE_USER_CONFIG_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert _resolve_user_config_dir() == Path.home() / ".config" / "drcode"


def test_child_process_import_honors_env(tmp_path: Path) -> None:
    """End-to-end: a freshly started python process (what an agent-spawned
    child actually is — the module constant is computed at import time)
    resolves its user config into DRCODE_USER_CONFIG_DIR when set."""
    override = tmp_path / "sandbox"
    env = os.environ.copy()
    env["DRCODE_USER_CONFIG_DIR"] = str(override)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import drcode.config as c\n"
                "print(c._USER_CONFIG_DIR)\n"
                "print(c.USER_CONFIG_FILE)"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0] == str(override)
    assert lines[1] == str(override / "drcode.json")


# -- user config scaffolding -----------------------------------------------------------


def test_ensure_user_config_dir_creates_example(user_config_dir: Path) -> None:
    from drcode.config import _PROVIDER_FIELDS

    ensure_user_config_dir()
    example_file = config_module.USER_CONFIG_EXAMPLE_FILE
    assert example_file.is_file()
    data = json.loads(example_file.read_text())
    # Every non-provider configurable field is present.
    assert set(data) == set(Config.model_fields) - _PROVIDER_FIELDS
    # The model/provider fields are deliberately absent: models live in
    # the user catalog / project selections / env.
    for name in _PROVIDER_FIELDS:
        assert name not in data


def test_ensure_user_config_dir_idempotent(user_config_dir: Path) -> None:
    ensure_user_config_dir()
    example_file = config_module.USER_CONFIG_EXAMPLE_FILE
    example_file.write_text("user edited this")
    ensure_user_config_dir()
    # An existing template is never overwritten.
    assert example_file.read_text() == "user edited this"


def test_ensure_user_config_dir_defaults_in_example(user_config_dir: Path) -> None:
    ensure_user_config_dir()
    data = json.loads(config_module.USER_CONFIG_EXAMPLE_FILE.read_text())
    assert data["compaction_threshold"] == 80
    assert data["request_limit"] == 500


def test_example_values_use_placeholders() -> None:
    # Fields listed in _EXAMPLE_PLACEHOLDERS get the fake value in the
    # generated example; everything else gets its built-in default. The
    # placeholders currently hold no entries: the provider/model fields are
    # excluded from the example entirely, the rest have sensible defaults.
    fields = Config.model_fields
    assert config_module._EXAMPLE_PLACEHOLDERS == {}
    assert (
        config_module._example_value(
            "compaction_threshold", fields["compaction_threshold"]
        )
        == 80
    )
    # Placeholders must never reference unknown fields.
    assert set(config_module._EXAMPLE_PLACEHOLDERS) <= set(fields)
