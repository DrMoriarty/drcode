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

    @property
    def is_ready(self) -> bool:
        return bool(self.model and self.api_key)
