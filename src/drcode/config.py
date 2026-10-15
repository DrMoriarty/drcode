from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# env_file в pydantic-settings резолвится относительно cwd, а TUI может быть
# запущен из любой директории — якоримся на корень репозитория через __file__.
# Последний файл в кортеже побеждает: локальный .env переопределяет репозиторный.
_REPO_ENV = Path(__file__).resolve().parents[2] / ".env"


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_REPO_ENV, ".env"), env_prefix="DRCODE_", extra="ignore"
    )

    model: str = ""  # e.g. "gpt-4o", "qwen3-coder-30b", model id of your provider
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""

    @property
    def is_ready(self) -> bool:
        return bool(self.model and self.api_key)
