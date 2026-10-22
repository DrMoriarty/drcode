"""Layered configuration for drcode.

Sources, from highest to lowest priority (a higher layer wins):

1. process environment (`DRCODE_*` variables),
2. project-level files: `<project>/.env` (dotenv) and
   `<project>/.drcode/drcode.json`,
3. user-level file: `~/.config/drcode/drcode.json` (XDG_CONFIG_HOME-aware),
4. built-in defaults.

On first start `ensure_user_config_dir()` creates the user config directory
and seeds it with a `drcode.json.example` listing the user-configurable
non-provider parameters (compaction_threshold, request_limit, …) with their
defaults. The per-model connection settings deliberately live elsewhere: in
the user-level model catalog (`~/.config/drcode/models.json`, see
`drcode.providers`), in the project's per-role selections
(`.drcode/drcode.json`) or in `DRCODE_*` environment variables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from pydantic_settings.sources import ConfigFileSourceType

# env_file в pydantic-settings резолвится относительно cwd, а TUI может быть
# запущен из любой директории — якоримся на корень репозитория через __file__.
# Последний файл в кортеже побеждает: локальный .env переопределяет репозиторный.
# Файл учитывается только для запуска из исходников (рядом pyproject.toml):
# у pip-установленной или замороженной копии parents[2] — это site-packages
# или временный каталог распаковки, искать там .env бессмысленно.
_REPO_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_REPO_ENV_FILES: tuple[Path | str, ...] = (
    (_REPO_ENV_PATH, ".env")
    if (_REPO_ENV_PATH.parent / "pyproject.toml").is_file()
    else (".env",)
)

# User-level config directory/file. $XDG_CONFIG_HOME is honored so Linux
# packagers/redirectors can move the config tree; defaults to ~/.config.
_USER_CONFIG_DIR = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "drcode"
)
USER_CONFIG_FILE = _USER_CONFIG_DIR / "drcode.json"
USER_CONFIG_EXAMPLE_FILE = _USER_CONFIG_DIR / "drcode.json.example"

# Project-level json file, resolved per instance (the TUI passes the project
# root, which is normally the cwd the app was started in). It lives inside
# `.drcode/` together with the task storage, so a single gitignore entry
# covers both (see `drcode.gitignore`).
PROJECT_STATE_DIRNAME = ".drcode"
PROJECT_CONFIG_FILENAME = "drcode.json"
# Configs written by older versions sat in the project root; they are still
# read (and migrated on save) so an upgrade doesn't lose the model settings.
_LEGACY_PROJECT_CONFIG_FILENAME = "drcode.json"

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE for maintainers: whenever you add/change a user-configurable field of
# `Config` below, ALSO check the example-file generation in
# `ensure_user_config_dir` (fields without a sensible default need a fake
# placeholder in `_EXAMPLE_PLACEHOLDERS`) — the generated
# `drcode.json.example` is built from the model fields.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Fake/placeholder values for example-file fields that have no meaningful
# default; fields not listed here get their built-in default in the example.
# The model/provider fields are not part of the user-level example at all
# (see `_PROVIDER_FIELDS`): models live in the user catalog
# (~/.config/drcode/models.json) or in the project's per-role selections.
_EXAMPLE_PLACEHOLDERS: dict[str, str] = {}

# The per-model connection fields of `Config`. They stay on the model (the
# env/dotenv layers still populate them — DRCODE_MODEL etc. — and
# `drcode.providers.resolve` writes its result into a copy of Config), but
# they are deliberately NOT read from any json layer: the model selection
# lives in the user-level catalog and the project's `models` /
# `custom_models` keys. Legacy flat keys inside a drcode.json are consumed
# by the startup migrations (`drcode.providers.migrate_user_config` /
# `migrate_project_config`) instead.
_PROVIDER_FIELDS: frozenset[str] = frozenset(
    {
        "model",
        "base_url",
        "api_key",
        "health_check_url",
        "max_context_tokens",
        "max_tokens",
    }
)


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ENV_FILES, env_prefix="DRCODE_", extra="ignore"
    )

    # Root of the project this config belongs to (private, not a setting).
    _project_root: Path = PrivateAttr(default_factory=Path.cwd)

    model: str = ""  # e.g. "gpt-4o", "qwen3-coder-30b", model id of your provider
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    # A verified availability/metadata endpoint (no generation) found by the
    # first-start probe, cached for cheaper availability checks on later runs.
    health_check_url: str = ""
    # Context window of the model, tokens: the task status bar shows context
    # fill as a percentage of this. Providers rarely report it — set via
    # DRCODE_MAX_CONTEXT_TOKENS when the default doesn't match the model.
    max_context_tokens: int = 200_000
    # Per-response output token cap (tokens): the `max_tokens` model setting of
    # pydantic-ai's OpenAIChatModel, distinct from max_context_tokens (the whole
    # context window). It caps a single model response's output length. With
    # no explicit value the provider's default is used, which for many gateways
    # is far too small and fails with
    # "Model token limit (provider default) exceeded before any response was
    # generated." Set via DRCODE_MAX_TOKENS when the default doesn't match the
    # model; keep it at or below max_context_tokens.
    max_tokens: int = 8192
    # Context fill (percent of max_context_tokens) at which the agent's
    # history is compacted before the next turn: a tool-less summarizer
    # agent rewrites the transcript into a structured summary while the
    # original user prompt(s) stay verbatim. Also drives the status-bar
    # colors: red from this fill, yellow from half the remaining headroom
    # (2 * threshold - 100, e.g. 60% at the default 80). A model entry in
    # the user-level catalog (models.json, see drcode.providers) may carry
    # its own `compaction_threshold`, which overrides this value for every
    # agent role the model is assigned to.
    compaction_threshold: int = 80
    # Cap on model turns (model responses acted upon) per agent/tester run —
    # pydantic-ai's UsageLimits.request_limit. Its built-in default of 50 is
    # too low for coding tasks, so drcode raises it; set to null to disable
    # the cap entirely. The counter is per run: a new user message starts
    # from zero again.
    request_limit: int | None = 500

    def __init__(self, project_root: Path | None = None, **data: object) -> None:
        # Scaffold the user config dir (idempotent, no-op when it exists)
        # before the sources read from it.
        ensure_user_config_dir()
        # Settings sources are class-level callables with no access to the
        # instance, so publish the project root for them before loading.
        # Single-threaded TUI startup — no locking needed.
        global _pending_project_root
        _pending_project_root = project_root or Path.cwd()
        super().__init__(**data)  # type: ignore[arg-type]
        # AFTER super().__init__(): pydantic initializes private attributes
        # to their defaults during model construction, which would wipe a
        # value assigned before it (the default is Path.cwd() — wrong for
        # the TUI operating on another directory).
        self._project_root = _pending_project_root

    @property
    def project_config_file(self) -> Path:
        return self._project_root / PROJECT_STATE_DIRNAME / PROJECT_CONFIG_FILENAME

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Earlier sources win. Project files override the user-level file;
        # env/dotenv override both; defaults come last. A corrupted json
        # file must not keep the app from starting — it's skipped instead.
        # The json sources deliberately never provide the model/provider
        # fields (see `_PROVIDER_FIELDS`): the model selection lives in the
        # user catalog / project slots / env, not in drcode.json.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _NonProviderJsonSource(
                settings_cls, json_file=_parsable_or_none(_project_json_file())
            ),
            _NonProviderJsonSource(
                settings_cls, json_file=_parsable_or_none(USER_CONFIG_FILE)
            ),
            file_secret_settings,
        )

    @property
    def is_ready(self) -> bool:
        """Model and endpoint are known. The API key is deliberately NOT
        required here: local/OpenAI-compatible gateways often work without
        one — the setup dialog asks for it as optional."""
        return bool(self.model and self.base_url)


class _NonProviderJsonSource(JsonConfigSettingsSource):
    """Json settings source that never provides the model/provider fields.

    The per-model connection keys of a legacy drcode.json are ignored as
    settings (they are consumed by the startup migrations in
    `drcode.providers` instead); only the non-provider fields
    (compaction_threshold, request_limit, …) flow through.
    """

    def _read_files(
        self, files: ConfigFileSourceType | None, deep_merge: bool = False
    ) -> dict[str, Any]:
        data = super()._read_files(files, deep_merge=deep_merge)
        return {k: v for k, v in data.items() if k not in _PROVIDER_FIELDS}


def _parsable_or_none(path: Path) -> Path | None:
    """Return `path` when it holds valid JSON, else None (source skipped).

    Corrupted config files must never crash the startup; the file is left
    untouched so the user can fix it, and lower layers provide the values.
    """
    try:
        json.loads(path.read_text())
    except FileNotFoundError:
        return path  # absent file is fine — the source yields nothing
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return path


def _project_json_file() -> Path:
    """Project-level drcode.json for the config currently being built.

    The value comes from `_pending_project_root`, which `Config.__init__`
    publishes right before `super().__init__()` runs the settings sources
    (defaults to cwd for bare `Config()` calls). When the current
    `.drcode/drcode.json` does not exist, a legacy root-level `drcode.json`
    left by an older version is read instead, so pre-migration projects
    keep their settings until the file is re-saved.
    """
    root = _pending_project_root
    current = root / PROJECT_STATE_DIRNAME / PROJECT_CONFIG_FILENAME
    if current.is_file():
        return current
    legacy = root / _LEGACY_PROJECT_CONFIG_FILENAME
    return legacy if legacy.is_file() else current


# Set by `Config.__init__` just before `super().__init__()` so the
# class-level settings sources can resolve the project-level json file.
_pending_project_root: Path = Path.cwd()


def _example_value(name: str, field: FieldInfo) -> object:
    placeholder = _EXAMPLE_PLACEHOLDERS.get(name)
    if placeholder is not None:
        return placeholder
    return field.get_default(call_default_factory=True)


def ensure_user_config_dir() -> None:
    """Create `~/.config/drcode` (or `$XDG_CONFIG_HOME/drcode`) and the
    `drcode.json.example` seed file on first start. Never fails the app:
    an unwritable home just means no user-level config file."""

    example = {
        name: _example_value(name, field)
        for name, field in Config.model_fields.items()
        # The provider/model fields are not user-level json settings:
        # models live in the catalog / project selections / env.
        if name not in _PROVIDER_FIELDS
    }
    payload = json.dumps(example, indent=2, ensure_ascii=False) + "\n"
    try:
        _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if not USER_CONFIG_EXAMPLE_FILE.exists():
            # First start (or the user deleted the template): drop it next
            # to where the user config would live, so it's discoverable.
            USER_CONFIG_EXAMPLE_FILE.write_text(payload)
        # One-line hint for the user-level model catalog (written/managed
        # exclusively by the `drcode-models` utility or the main TUI's
        # "Provider catalog…" screen — never by hand).
        if not (_USER_CONFIG_DIR / "models.json.example").exists():
            (_USER_CONFIG_DIR / "models.json.example").write_text(
                json.dumps(
                    {
                        "providers": [
                            {
                                "name": "local-swap",
                                "base_url": "http://localhost:8080/v1",
                                "api_key": "",
                                "health_check_url": "",
                                "models": [
                                    {
                                        "id": "qwen3-coder-30b",
                                        "max_context_tokens": 200000,
                                        "max_tokens": None,
                                        "compaction_threshold": None,
                                    }
                                ],
                            }
                        ],
                        "defaults": {
                            "coding": "local-swap/qwen3-coder-30b",
                            "tester": None,
                            "compaction": None,
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
    except OSError:
        # Read-only home or exotic sandbox — user-level config is optional.
        pass
