"""Layered configuration for drcode.

Sources, from highest to lowest priority (a higher layer wins):

1. process environment (`DRCODE_*` variables),
2. project-level files: `<project>/.env` (dotenv) and `<project>/drcode.json`,
3. user-level file: `~/.config/drcode/drcode.json` (XDG_CONFIG_HOME-aware),
4. built-in defaults.

On first start `ensure_user_config_dir()` creates the user config directory
and seeds it with a `drcode.json.example` listing every user-configurable
parameter with its default (or a fake placeholder when there is no sensible
default, e.g. the API key).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import PrivateAttr
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

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
# root, which is normally the cwd the app was started in).
PROJECT_CONFIG_FILENAME = "drcode.json"

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE for maintainers: whenever you add/change a user-configurable field of
# `Config` below, ALSO add it to `_EXAMPLE_PLACEHOLDERS` (for fields without a
# sensible default) — the generated `drcode.json.example` is built from the
# model fields, so new fields appear there automatically, but fake placeholder
# values for defaultless fields (secrets, model ids) must be listed by hand.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Fake/placeholder values for example-file fields that have no meaningful
# default; fields not listed here get their built-in default in the example.
_EXAMPLE_PLACEHOLDERS: dict[str, str] = {
    "model": "gpt-4o",
    "api_key": "sk-your-api-key-here",
}


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ENV_FILES, env_prefix="DRCODE_", extra="ignore"
    )

    # Root of the project this config belongs to (private, not a setting).
    _project_root: Path = PrivateAttr(default_factory=Path.cwd)

    model: str = ""  # e.g. "gpt-4o", "qwen3-coder-30b", model id of your provider
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    # Context window of the model, tokens: the task status bar shows context
    # fill as a percentage of this. Providers rarely report it — set via
    # DRCODE_MAX_CONTEXT_TOKENS when the default doesn't match the model.
    max_context_tokens: int = 200_000
    # Context fill (percent of max_context_tokens) at which the agent's
    # history is compacted before the next turn: a tool-less summarizer
    # agent rewrites the transcript into a structured summary while the
    # original user prompt(s) stay verbatim. Also drives the status-bar
    # colors: red from this fill, yellow from half the remaining headroom
    # (2 * threshold - 100, e.g. 60% at the default 80).
    compaction_threshold: int = 80

    def __init__(self, project_root: Path | None = None, **data: object) -> None:
        # Scaffold the user config dir (idempotent, no-op when it exists)
        # before the sources read from it.
        ensure_user_config_dir()
        # Settings sources are class-level callables with no access to the
        # instance, so publish the project root for them before loading.
        # Single-threaded TUI startup — no locking needed.
        global _pending_project_root
        _pending_project_root = project_root or Path.cwd()
        self._project_root = _pending_project_root
        super().__init__(**data)  # type: ignore[arg-type]

    @property
    def project_config_file(self) -> Path:
        return self._project_root / PROJECT_CONFIG_FILENAME

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
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            JsonConfigSettingsSource(
                settings_cls, json_file=_parsable_or_none(_project_json_file())
            ),
            JsonConfigSettingsSource(
                settings_cls, json_file=_parsable_or_none(USER_CONFIG_FILE)
            ),
            file_secret_settings,
        )

    @property
    def is_ready(self) -> bool:
        return bool(self.model and self.api_key)


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
    (defaults to cwd for bare `Config()` calls).
    """
    return _pending_project_root / PROJECT_CONFIG_FILENAME


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
        name: _example_value(name, field) for name, field in Config.model_fields.items()
    }
    payload = json.dumps(example, indent=2, ensure_ascii=False) + "\n"
    try:
        _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if not USER_CONFIG_EXAMPLE_FILE.exists():
            # First start (or the user deleted the template): drop it next
            # to where the user config would live, so it's discoverable.
            USER_CONFIG_EXAMPLE_FILE.write_text(payload)
    except OSError:
        # Read-only home or exotic sandbox — user-level config is optional.
        pass
