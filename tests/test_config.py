"""Tests for drcode.config: layered configuration and user-config scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import drcode.config as config_module
from drcode.config import Config, _parsable_or_none, ensure_user_config_dir


@pytest.fixture
def user_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A redirected (already-sandboxed) user config dir, as a plain path."""
    return config_module._USER_CONFIG_DIR


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


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
    _write_json(project_root / "drcode.json", {"compaction_threshold": 10})
    monkeypatch.setenv("DRCODE_COMPACTION_THRESHOLD", "99")
    cfg = Config(project_root=project_root)
    assert cfg.compaction_threshold == 99


def test_project_json_beats_user_json(
    project_root: Path, user_config_dir: Path
) -> None:
    _write_json(user_config_dir / "drcode.json", {"compaction_threshold": 20})
    _write_json(project_root / "drcode.json", {"compaction_threshold": 30})
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
    (project_root / "drcode.json").write_text("{broken")
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
    assert cfg.project_config_file == project_root / "drcode.json"


def test_unknown_json_keys_ignored(project_root: Path) -> None:
    _write_json(project_root / "drcode.json", {"no_such_field": 1})
    cfg = Config(project_root=project_root)
    assert cfg.compaction_threshold == 80


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


# -- user config scaffolding -----------------------------------------------------------


def test_ensure_user_config_dir_creates_example(user_config_dir: Path) -> None:
    ensure_user_config_dir()
    example_file = config_module.USER_CONFIG_EXAMPLE_FILE
    assert example_file.is_file()
    data = json.loads(example_file.read_text())
    # Every configurable field is present.
    assert set(data) == set(Config.model_fields)
    # Fields without a sensible default get fake placeholders.
    assert data["api_key"] == "sk-your-api-key-here"
    assert data["model"] == "gpt-4o"


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
    assert data["base_url"] == "https://api.openai.com/v1"
    assert data["compaction_threshold"] == 80


def test_example_values_use_placeholders() -> None:
    # Fields listed in _EXAMPLE_PLACEHOLDERS get the fake value in the
    # generated example; everything else gets its built-in default.
    fields = Config.model_fields
    assert (
        config_module._example_value("api_key", fields["api_key"])
        == "sk-your-api-key-here"
    )
    assert config_module._example_value("model", fields["model"]) == "gpt-4o"
    assert (
        config_module._example_value("base_url", fields["base_url"])
        == "https://api.openai.com/v1"
    )
    assert (
        config_module._example_value(
            "compaction_threshold", fields["compaction_threshold"]
        )
        == 80
    )
    # Placeholders must never reference unknown fields.
    assert set(config_module._EXAMPLE_PLACEHOLDERS) <= set(fields)
