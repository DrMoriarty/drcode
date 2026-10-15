from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="DRCODE_", extra="ignore"
    )

    model: str = ""  # e.g. "gpt-4o", "qwen3-coder-30b", model id of your provider
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""

    @property
    def is_ready(self) -> bool:
        return bool(self.model and self.api_key)
