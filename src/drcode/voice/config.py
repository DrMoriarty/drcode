"""Voice configuration resolution: project-settings → :class:`VoiceConfig`.

The announcement-language selection (``voice_language`` with a
terminal-derived fallback and the phrase tables) also lives here — it is
pure configuration, no audio or network.

Configuration comes from the project settings (``voice_*`` fields of
:class:`drcode.settings.ProjectSettings`): everything empty = the feature is
off. The API key may be left empty to reuse the token of the user-catalog
provider whose host matches the audio URL's host (no second token for the
common single-gateway case).

The ASR/TTS model ids (e.g. ``gigaam_rnnt_v3_e2e``) contain ``/`` — the
separator of the agent-catalog ``<provider>/<model-id>`` reference format
(:func:`drcode.providers._reject_slash`) — so they are stored in the project
settings directly, never in the model catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..providers import UserModelsStore

if TYPE_CHECKING:
    from ..settings import ProjectSettings

# Default audio-API base URL (offered as the dialog's prefill; the setting
# stays empty until the user saves the voice dialog — empty = voice off).
DEFAULT_VOICE_API_URL = "https://provider.ru/v1"

# Default speaker voice when voice_tts_voice is empty (OpenAI TTS names).
DEFAULT_TTS_VOICE = "alloy"

# Spoken-announcement languages (voice_language setting): "ru" or "en".
# Empty setting = auto-derive from the terminal locale.
VOICE_LANGUAGE_RU = "ru"
VOICE_LANGUAGE_EN = "en"


def terminal_voice_language() -> str:
    """The announcement language derived from the terminal environment:
    the language subtag of ``LC_ALL``/``LC_MESSAGES``/``LANG`` (in that
    precedence, POSIX form like ``ru_RU.UTF-8``), falling back to Python's
    ``locale`` when none is set. Russian reads as "ru", everything else
    (including no locale at all) as "en"."""
    import os

    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(var, "").strip()
        if not value:
            continue
        # Strip the encoding and modifier parts: "ru_RU.UTF-8" → "ru".
        lang = value.split(".", 1)[0].split("@", 1)[0].split("_", 1)[0].lower()
        if lang:
            return VOICE_LANGUAGE_RU if lang == "ru" else VOICE_LANGUAGE_EN
    try:
        import locale

        lang = (locale.getlocale()[0] or "").split("_", 1)[0].lower()
    except Exception:  # noqa: BLE001 - a broken locale must never crash
        lang = ""
    return VOICE_LANGUAGE_RU if lang == "ru" else VOICE_LANGUAGE_EN


def effective_voice_language(settings: ProjectSettings) -> str:
    """The language the announcements are phrased in: the explicit
    ``voice_language`` setting ("ru"/"en") or, when it is empty, the
    terminal-derived default."""
    stored = settings.voice_language.strip().lower()
    if stored in (VOICE_LANGUAGE_RU, VOICE_LANGUAGE_EN):
        return stored
    return terminal_voice_language()


# The spoken announcement phrases, keyed by language (a format template
# per event kind; ``{title}`` is the task title). The TTS speaks whatever
# text it is given — choosing the language is choosing the phrase table.
VOICE_PHRASES: dict[str, dict[str, str]] = {
    VOICE_LANGUAGE_RU: {
        "bug_report": "Заведён баг-репорт: {title}",
        "completed": "Задача «{title}» завершена",
        "completed_failed": "Задача «{title}» завершена с ошибкой",
        "requeued": (
            "Тестирование провалено — задача «{title}» возвращена в разработку"
        ),
        "queue_empty": "Очередь пуста — все задачи обработаны",
    },
    VOICE_LANGUAGE_EN: {
        "bug_report": "Bug report created: {title}",
        "completed": 'Task "{title}" completed',
        "completed_failed": 'Task "{title}" completed with an error',
        "requeued": 'Testing failed — task "{title}" returned to development',
        "queue_empty": "Queue is empty — all tasks processed",
    },
}


def announcement_phrase(language: str, key: str, **kwargs: object) -> str:
    """A spoken phrase from :data:`VOICE_PHRASES`, formatted; an unknown
    language falls back to English (the table is always complete for both,
    but a hand-edited settings.json may carry anything)."""
    table = VOICE_PHRASES.get(
        language if language in VOICE_PHRASES else VOICE_LANGUAGE_EN,
        VOICE_PHRASES[VOICE_LANGUAGE_EN],
    )
    return str(table.get(key, table.get("completed", ""))).format(**kwargs)


class VoiceError(Exception):
    """A voice feature could not do its job (config missing, network or
    API error). The message is user-readable on purpose."""


@dataclass(frozen=True)
class VoiceConfig:
    """Resolved connection + model selection for the audio API."""

    base_url: str
    api_key: str
    asr_model: str = ""
    tts_model: str = ""
    tts_voice: str = DEFAULT_TTS_VOICE

    def _base(self) -> str:
        return self.base_url.rstrip("/")


def catalog_api_key_for_host(base_url: str) -> str:
    """The api_key of the user-catalog provider whose base_url host matches
    ``base_url``'s host ("" when nothing matches). Audio gateways that also
    serve the chat models reuse the same token — the user then never enters
    it twice."""
    from urllib.parse import urlsplit

    host = urlsplit(base_url).netloc.rsplit("@", 1)[-1].lower()
    host = host.rsplit(":", 1)[0] if host else ""
    if not host:
        return ""
    for provider in UserModelsStore().load().providers:
        provider_host = urlsplit(provider.base_url).netloc.rsplit("@", 1)[-1].lower()
        if provider_host.rsplit(":", 1)[0] == host and provider.api_key:
            return provider.api_key
    return ""


def resolve_api_key(settings: ProjectSettings, base_url: str) -> str:
    """The effective audio-API key: the explicit setting wins, the host
    match against the user catalog is the fallback."""
    key = settings.voice_api_key.strip()
    if key:
        return key
    return catalog_api_key_for_host(base_url)


def config_from_settings(settings: ProjectSettings) -> VoiceConfig | None:
    """The resolved config, or None when voice is off (no base URL).
    api_key resolution against the catalog happens here, once, so callers
    never re-implement the fallback."""
    base_url = settings.voice_api_url.strip()
    if not base_url:
        return None
    return VoiceConfig(
        base_url=base_url,
        api_key=resolve_api_key(settings, base_url),
        asr_model=settings.voice_asr_model.strip(),
        tts_model=settings.voice_tts_model.strip(),
        tts_voice=settings.voice_tts_voice.strip() or DEFAULT_TTS_VOICE,
    )
