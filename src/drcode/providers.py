"""User-level model catalog and per-role model resolution.

While `config.Config` stays a flat, env-driven description of ONE model
connection (shared by every agent), this module adds two more layers:

1. A user-level catalog of OpenAI-compatible providers
   (`~/.config/drcode/models.json`, XDG_CONFIG_HOME-aware), each with its
   base URL, API key, cached health-check endpoint and a list of models
   (auto-discovered via `agent_model.fetch_models` or entered by hand).
   The file is written exclusively by the `drcode-models` utility (or the
   catalog screen mounted inside the main TUI) — never by hand. When the
   file does not exist yet but the user-level drcode.json already
   describes a model, the catalog screen pre-fills itself with one
   provider/model built from that data (`load_seeded_catalog`) — in
   memory only, until the user saves.
2. A project-level selection of the model per agent role (coding / tester /
   compaction) stored under the optional `models` key of
   `<project>/.drcode/drcode.json`, values of the form
   `"<provider-name>/<model-id>"` (provider names must not contain `/`;
   model ids may). A `null` / absent slot means "use the user-level
   default, else the flat config".

Resolution precedence for a role (`resolve`):

1. `DRCODE_MODEL` / `DRCODE_BASE_URL` environment variables — when either
   is set, they win globally for ALL roles (the existing "env > everything"
   contract and the simple single-model scenario stay intact; note that
   project `.env` files flow through the ordinary Config layering and do
   NOT trigger this branch).
2. the project `models[role]` slot;
3. `UserModelsConfig.defaults[role]`;
4. the flat config as-is (backward compatibility).

The resolver returns a `config.model_copy(update={...})` so
`agent_model.build_model(config)` keeps working unchanged — and so the
network test-suite's `build_model` monkeypatching (signature `(config)`)
is unaffected.

The catalog deliberately does NOT ride on `Config`: `Config` is a flat
BaseSettings whose corrupted json layer is silently skipped, which is the
wrong behavior for a hand-curated catalog — here a corrupted file falls
back to the `.bak` copy of the previous version and only ever warns.
"""

from __future__ import annotations

import enum
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ValidationError

from . import config as _config
from .config import (
    _LEGACY_PROJECT_CONFIG_FILENAME,
    PROJECT_STATE_DIRNAME,
    Config,
)
from .gitignore import ensure_drcode_gitignored

# User-level catalog file, next to the user-level drcode.json. The
# directory is read dynamically from the config module (not bound at
# import time) so tests can redirect the whole user config dir by
# monkeypatching `config._USER_CONFIG_DIR`.
USER_MODELS_FILENAME = "models.json"


def user_models_file() -> Path:
    """Path of the user-level model catalog (XDG_CONFIG_HOME-aware)."""
    return _config._USER_CONFIG_DIR / USER_MODELS_FILENAME


class ModelEntry(BaseModel):
    """One model of a provider: its id at the provider plus the token caps."""

    id: str
    # Context window of the model, tokens: drives the context-fill display
    # and the compaction threshold. Gateways rarely report it — when
    # auto-discovery cannot find it, the default below is used and the
    # entry is flagged (`context_reported=False`) as needing manual
    # confirmation in the catalog UI.
    max_context_tokens: int = 200_000
    # Per-response output cap; None = keep whatever the flat config / the
    # provider default says.
    max_tokens: int | None = None
    # True when the gateway itself reported the context size (trusted);
    # False = the default above, confirm/adjust by hand in the UI.
    context_reported: bool = False


class Provider(BaseModel):
    """One OpenAI-compatible endpoint and the models served by it."""

    # Unique key, e.g. "local-swap"; must not contain "/" (it is the head
    # of the "<provider>/<model-id>" reference format).
    name: str
    base_url: str
    api_key: str = ""
    # Cached verified availability endpoint (same idea as
    # Config.health_check_url): probed first on later availability checks.
    health_check_url: str = ""
    models: list[ModelEntry] = []


class AgentRole(str, enum.Enum):
    """Agent roles a model can be selected for."""

    CODING = "coding"
    TESTER = "tester"
    COMPACTION = "compaction"


class ModelRef(BaseModel):
    """Reference to a catalog model: provider name + model id."""

    provider: str
    model: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.provider}/{self.model}"


class UserModelsConfig(BaseModel):
    """The whole user-level catalog as persisted in models.json."""

    providers: list[Provider] = []
    # Per-role fallback when the project does not pin a model itself.
    defaults: dict[AgentRole, ModelRef | None] = {}


# -- project-level selection (.drcode/drcode.json "models" key) --------------


def _project_config_path(project_root: Path) -> Path:
    """The project's drcode.json, with the legacy root-level fallback."""
    current = project_root / PROJECT_STATE_DIRNAME / "drcode.json"
    if current.is_file():
        return current
    legacy = project_root / _LEGACY_PROJECT_CONFIG_FILENAME
    return legacy if legacy.is_file() else current


def _read_json(path: Path) -> dict[str, object]:
    """Read a json object; missing/corrupt file → {} (never fatal)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_ref(value: object) -> ModelRef | None:
    """Parse one `models[role]` value: "provider/model" (string), a
    {"provider": …, "model": …} object, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        if "/" not in value:
            return None
        provider, _, model = value.partition("/")
        if not provider or not model:
            return None
        return ModelRef(provider=provider, model=model)
    if isinstance(value, dict):
        try:
            return ModelRef.model_validate(value)
        except ValidationError:
            return None
    return None


def load_project_model_refs(project_root: Path) -> dict[AgentRole, ModelRef]:
    """The per-role model selection of the project (absent slots omitted)."""
    models = _read_json(_project_config_path(project_root)).get("models")
    if not isinstance(models, dict):
        return {}
    out: dict[AgentRole, ModelRef] = {}
    for role in AgentRole:
        ref = _parse_ref(models.get(role.value))
        if ref is not None:
            out[role] = ref
    return out


def set_project_model(
    project_root: Path, role: AgentRole, ref: ModelRef | None
) -> Path:
    """Persist one role's model selection into `.drcode/drcode.json`.

    Only the role's own slot is touched; every other key (including the
    other two role slots) is preserved. `ref=None` clears the slot (null =
    user-level default / flat config). Returns the written path.
    """
    path = _project_config_path(project_root)
    data = _read_json(path)
    models_raw = data.get("models")
    models = dict(models_raw) if isinstance(models_raw, dict) else {}
    models[role.value] = None if ref is None else f"{ref.provider}/{ref.model}"
    if any(v is not None for v in models.values()):
        data["models"] = models
    else:
        # Every slot is null: drop the key entirely (absence == null).
        data.pop("models", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    # `.drcode/` also holds task storage — make sure git ignores it
    # (no-op outside a git repository).
    ensure_drcode_gitignored(project_root)
    return path


# -- user-level catalog store -------------------------------------------------


def _validate_catalog(data: object) -> tuple[UserModelsConfig | None, str | None]:
    try:
        return UserModelsConfig.model_validate(data), None
    except ValidationError as e:
        msgs = [str(err.get("msg", "")) for err in e.errors()]
        return None, msgs[0] if msgs else type(e).__name__


class UserModelsStore:
    """Loads/saves the user-level model catalog with a `.bak` fallback.

    A corrupted models.json must never crash the startup: the load falls
    back to the `.bak` copy of the previous version, else to defaults, and
    records a warning in `last_warning` for the UI to surface.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or user_models_file()
        self.last_warning: str | None = None

    def load(self) -> UserModelsConfig:
        try:
            data: object = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return UserModelsConfig()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            data = None
            detail = f"{type(e).__name__}: {e}"
        else:
            parsed, detail = _validate_catalog(data)
            if parsed is not None:
                return parsed
        # The primary file is absent or unreadable — try the backup of the
        # previous version before giving up.
        parsed, bak_detail = None, None
        try:
            bak_data: object = json.loads(
                self.path.with_suffix(".json.bak").read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            bak_data = None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            bak_data = None
            bak_detail = f"{type(e).__name__}: {e}"
        if bak_data is not None:
            parsed, bak_detail = _validate_catalog(bak_data)
        if parsed is not None:
            self.last_warning = (
                f"{self.path.name} is unreadable ({detail}); using the .bak backup"
            )
            return parsed
        if detail is not None:
            self.last_warning = f"{self.path.name} ignored ({detail})" + (
                f"; backup also unreadable ({bak_detail})" if bak_detail else ""
            )
        return UserModelsConfig()

    def save(self, catalog: UserModelsConfig) -> Path:
        """Atomically persist the catalog, backing up the previous version."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                shutil.copyfile(self.path, self.path.with_suffix(".json.bak"))
            except OSError:
                pass  # backup is best-effort
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(catalog.model_dump_json(indent=2) + "\n")
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return self.path


# -- one-time seeding from the user-level flat config -------------------------


def _provider_name_for(base_url: str) -> str:
    """Derive a provider name from the endpoint: its host (without port,
    credentials or path), e.g. `https://api.openai.com/v1` → `api.openai.com`.
    Falls back to a generic name for unparseable URLs."""
    netloc = urlsplit(base_url).netloc
    # Drop userinfo ("user:pass@host") and the port ("host:8080").
    host = netloc.rsplit("@", 1)[-1].rsplit(":", 1)[0].strip()
    return host or "provider"


def load_seeded_catalog(
    store: UserModelsStore,
) -> tuple[UserModelsConfig, str | None]:
    """Load the catalog, pre-filling it from the user-level drcode.json.

    Migration convenience for `drcode-models` (and the main TUI's catalog
    screen): when the catalog file does not exist yet but the user-level
    drcode.json already describes the model in use, an in-memory catalog
    with a single provider and its single model is built from that data
    (the provider name is derived from the base URL host; the flat config
    defaults fill in whatever the file omits).

    Nothing is persisted here — models.json appears only when the user
    explicitly saves the catalog screen; quitting without saving discards
    the seed and the flat config keeps working as before.

    Returns the catalog plus an optional human-readable note for the UI
    (set when the seed actually happened).
    """
    catalog = store.load()
    if store.path.exists():
        # A catalog already exists — it is the source of truth, never reseed.
        return catalog, None
    data = _read_json(_config.USER_CONFIG_FILE)
    model = str(data.get("model") or "").strip()
    if not model:
        # Nothing (usable) described in the flat config — empty catalog.
        return catalog, None
    base_url = str(data.get("base_url") or "").strip() or str(
        Config.model_fields["base_url"].default
    )
    max_context = data.get("max_context_tokens")
    if not isinstance(max_context, int) or max_context <= 0:
        max_context = ModelEntry.model_fields["max_context_tokens"].default
    max_tokens = data.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = None
    provider = Provider(
        name=_provider_name_for(base_url),
        base_url=base_url,
        api_key=str(data.get("api_key") or ""),
        health_check_url=str(data.get("health_check_url") or ""),
        models=[
            ModelEntry(
                id=model,
                max_context_tokens=max_context,
                max_tokens=max_tokens,
                # The flat config value is not a gateway report — keep the
                # "confirm by hand" flag of a hand-entered entry.
                context_reported=False,
            )
        ],
    )
    catalog.providers.append(provider)
    note = (
        f"Pre-filled from drcode.json: provider “{provider.name}”, model "
        f"“{model}”. Not saved yet — press Save to write {USER_MODELS_FILENAME}."
    )
    return catalog, note


# -- resolution ---------------------------------------------------------------


def resolve(config: Config, role: AgentRole) -> Config:
    """Resolve the effective model connection for one agent role.

    See the module docstring for the precedence. Always returns a `Config`
    (the input unchanged when nothing overrides it), so callers can feed it
    straight into `agent_model.build_model`.
    """
    # 1. Env override: DRCODE_MODEL / DRCODE_BASE_URL win globally for all
    #    roles — they are already baked into `config` by the ordinary
    #    Config layering, so nothing to resolve here.
    if os.environ.get("DRCODE_MODEL") or os.environ.get("DRCODE_BASE_URL"):
        return config

    catalog = UserModelsStore().load()
    # 2. Project-level selection, 3. user-level default: the first slot
    #    whose provider still exists in the catalog wins (a dangling
    #    reference — provider renamed/deleted — falls through instead of
    #    breaking the run).
    candidates = [
        load_project_model_refs(config._project_root).get(role),
        catalog.defaults.get(role),
    ]
    provider: Provider | None = None
    ref: ModelRef | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        provider = next(
            (p for p in catalog.providers if p.name == candidate.provider), None
        )
        if provider is not None:
            ref = candidate
            break
    if provider is None or ref is None:
        # 4. Flat config as-is.
        return config

    update: dict[str, object] = {
        "model": ref.model,
        "base_url": provider.base_url,
        "api_key": provider.api_key,
        "health_check_url": provider.health_check_url,
    }
    entry = next((m for m in provider.models if m.id == ref.model), None)
    if entry is not None:
        update["max_context_tokens"] = entry.max_context_tokens
        if entry.max_tokens is not None:
            update["max_tokens"] = entry.max_tokens
    return config.model_copy(update=update)
