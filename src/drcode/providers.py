"""User-level model catalog and per-role model resolution.

While `config.Config` remains the flat, env-driven container of ONE model
connection (the json layers no longer provide its model fields — see
`drcode.config._PROVIDER_FIELDS`), this module adds two more layers:

1. A user-level catalog of OpenAI-compatible providers
   (`~/.config/drcode/models.json`, XDG_CONFIG_HOME-aware), each with its
   base URL, API key, cached health-check endpoint and a list of models
   (auto-discovered via `agent_model.fetch_models` or entered by hand).
   The file is written exclusively by the catalog screen mounted inside
   the main TUI ("Provider catalog…" in Settings) — never by hand. When the
   file does not exist yet but the user-level drcode.json (legacy flat
   config) or the project's `custom_models` already describe models, the
   catalog screen pre-fills itself from that data (`load_seeded_catalog`)
   — in memory only, until the user saves.
2. A project-level selection of the model per agent role (coding / research /
   tester / compaction) stored under the optional `models` key of
   `<project>/.drcode/drcode.json`: either a catalog reference of the form
   `"<provider-name>/<model-id>"` (provider names must not contain `/`;
   model ids may — a reference is always split at the FIRST slash) or the
   string `"custom"` — use the per-role free-form model
   stored under the sibling `custom_models[role]` key (all six connection
   fields: model/base_url/api_key/health_check_url/max_context_tokens/
   max_tokens). A `null` / absent slot means "use the user-level default".
   A `"custom"` slot without a valid `custom_models[role]` entry is treated
   as "nothing selected" (the same as `null`).

Resolution precedence for a role (`resolve`):

1. `DRCODE_MODEL` / `DRCODE_BASE_URL` environment variables — when either
   is set, they win globally for ALL roles (the existing "env > everything"
   contract and the simple single-model scenario stay intact; note that
   project `.env` files flow through the ordinary Config layering and do
   NOT trigger this branch).
2. the project `models[role]` slot (catalog reference or `"custom"`);
3. the user-level ordered chain `UserModelsConfig.defaults[role]` — the
   PRIMARY model (the historical single default) first, then the
   availability fallbacks in order; the first reference that resolves to
   a catalog provider wins;
4. nothing — the role resolves to "model not set" (an empty `model` in the
   returned config): the header shows "not configured" and the pipeline
   stays idle. The flat config's model fields are deliberately NOT a
   fallback anymore; on startup `migrate_user_config` /
   `migrate_project_config` move any legacy flat keys into the catalog /
   project custom models and remove them from the json files.

A reference resolves to its provider in two steps (`_match_ref`): the
exact `<provider>/<model-id>` match first; when the named provider does
not exist on this machine, a match BY MODEL ID across the catalog —
project drcode.json files stay portable between machines whose user
catalogs name the same gateways differently. No notification is emitted
for such a cross-provider match: it is a legitimate way of working, not
an anomaly.

`resolve` stays a SYNCHRONOUS, pure config resolution (network-free).
Availability-aware selection — probing each candidate provider's health
endpoint at run start and taking the first AVAILABLE candidate by
priority — is the separate async `resolve_available` (see below); the
pipeline applies it once per run start, never mid-run.

Independently of the model connection, two per-role REQUEST overrides are
overlaid on the result of every path (including the env shortcut), from
`<project>/.drcode/settings.json` (see `drcode.settings.ProjectSettings`) —
the most specific layer, beating the catalog entry and the flat config:

- `role_max_tokens[role]` — the per-response output cap (beats the catalog
  entry's `max_tokens` and the flat config default 8192 / DRCODE_MAX_TOKENS);
- `role_request_timeout[role]` — the role's OpenAI client timeout in
  seconds (`Config.request_timeout`, consumed by
  `agent_model.build_model`); absent = the SDK default 600 s. The UI
  exposes it for the COMPACTION role only.

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

import asyncio
import enum
import json
import os
import shutil
import ssl
import tempfile
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ValidationError, field_serializer, field_validator

from . import config as _config
from .config import (
    _LEGACY_PROJECT_CONFIG_FILENAME,
    PROJECT_CONFIG_FILENAME,
    PROJECT_STATE_DIRNAME,
    Config,
)
from .gitignore import ensure_drcode_gitignored
from .settings import ProjectSettingsStore
from .version import APP_VERSION

# User-level catalog file, next to the user-level drcode.json. The
# directory is read dynamically from the config module (not bound at
# import time) so tests can redirect the whole user config dir by
# monkeypatching `config._USER_CONFIG_DIR`.
USER_MODELS_FILENAME = "models.json"

# Legacy flat model keys of a drcode.json: after the migrations (see
# `migrate_user_config` / `migrate_project_config`) these are no longer
# read as settings from any json layer — the model comes from the user
# catalog, the project slots/custom models or the DRCODE_* env instead.
_FLAT_MODEL_KEYS: tuple[str, ...] = (
    "model",
    "base_url",
    "api_key",
    "health_check_url",
    "max_context_tokens",
    "max_tokens",
)


def user_models_file() -> Path:
    """Path of the user-level model catalog (XDG_CONFIG_HOME-aware)."""
    return _config._USER_CONFIG_DIR / USER_MODELS_FILENAME


def _reject_slash(value: str) -> str:
    """Shared validator body: `/` is the separator of the
    "<provider>/<model-id>" reference format and is forbidden in provider
    NAMES (its head). Model ids may contain `/` — a reference is split at
    the first slash, so the id is everything after it."""
    if "/" in value:
        raise ValueError("must not contain '/'")
    return value


class ModelEntry(BaseModel):
    """One model of a provider: its id at the provider plus the token caps."""

    # Unique id at the provider, e.g. "qwen3-coder-30b"; may contain "/"
    # (it is the tail of the "<provider>/<model-id>" reference format,
    # which is split at the FIRST slash).
    id: str
    # Context window of the model, tokens: drives the context-fill display
    # and the compaction threshold. Gateways rarely report it — when
    # auto-discovery cannot find it, the default below is used (adjust by
    # hand in the catalog UI if needed).
    max_context_tokens: int = 200_000
    # Per-response output cap; None = keep whatever the flat config / the
    # provider default says.
    max_tokens: int | None = None
    # Context fill (percent of max_context_tokens) at which THIS model's
    # transcript is compacted; None = inherit the flat config's
    # `compaction_threshold`. Stored on the model entry (not per role) so
    # the model–context–compaction-limit triple is one indivisible unit:
    # assigning the model to an agent role brings all three with it.
    compaction_threshold: int | None = None
    # Favorite level: 0 = not a favorite, 1..3 = favorite priority tier.
    # The MEANING of the tiers is the user's own (cheap/expensive,
    # fast/quality — any criterion); drcode only uses the ordering: the
    # availability fallback chain of a role is built from favorite models
    # first (see the role-default lists and `resolve_available`), and the
    # catalog UI (a separate task) will sort favorites to the top. Stored
    # on the model entry in the user-level models.json.
    favorite: int = 0

    @field_validator("favorite")
    @classmethod
    def _favorite_range(cls, value: int) -> int:
        if not 0 <= value <= 3:
            raise ValueError("favorite must be 0..3 (0 = not a favorite)")
        return value


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
    # Extra JSON fields merged into the body of EVERY request to this
    # provider's models (pydantic-ai ModelSettings `extra_body` → POST
    # /chat/completions), e.g. {"cache_prompt": true} for vLLM or gateway
    # routing knobs. Edited only by hand in models.json (no UI); merging
    # into the requests happens in `agent_model.base_model_settings`.
    # Must be a JSON object; empty = nothing is added.
    extra_body: dict[str, Any] = {}
    models: list[ModelEntry] = []

    @field_validator("name")
    @classmethod
    def _name_no_slash(cls, value: str) -> str:
        return _reject_slash(value)


class AgentRole(str, enum.Enum):
    """Agent roles a model can be selected for.

    RESEARCH serves research tasks (analysis/docs load profile: read-heavy,
    long tool chains — a cheap-input large-window model beats the best
    codegen model there). When nothing is selected for it anywhere, research
    tasks fall back to the coding model (`resolve_effective`), so existing
    configs keep working unchanged.

    RESOLVER serves the automatic merge-conflict resolver rounds (a
    conflicted merge parked in the task's worktree by the merge engine):
    same fallback to the coding model when nothing is selected for it.
    """

    CODING = "coding"
    RESEARCH = "research"
    TESTER = "tester"
    COMPACTION = "compaction"
    RESOLVER = "resolver"


class ModelRef(BaseModel):
    """Reference to a catalog model: provider name + model id."""

    provider: str
    model: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.provider}/{self.model}"


class CustomModelEntry(BaseModel):
    """A project-level free-form ("custom") model of one agent role.

    The full connection description lives in the project's drcode.json
    under `custom_models[role]`; the role's `models[role]` slot holds the
    string "custom" to activate it. Unlike catalog entries, custom models
    are per-role: the tester's and the coding agent's custom models never
    mix. `compaction_threshold` is deliberately absent — a non-provider
    setting, it stays with the ordinary Config layering.
    """

    model: str
    base_url: str
    api_key: str = ""
    health_check_url: str = ""
    max_context_tokens: int = 200_000
    # None = keep whatever the flat config / env provides.
    max_tokens: int | None = None

    @field_validator("model", "base_url")
    @classmethod
    def _required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


# Maximum length of a role's ordered user-level reference list: the
# primary model plus up to two availability fallbacks. The list lives on
# the user level because the three availability tiers (works behind a
# VPN / works without a VPN, needs the internet / works offline, local)
# are a property of the USER'S MACHINE, not of any project.
MAX_ROLE_DEFAULTS = 3


def _normalize_role_refs(value: object) -> list[ModelRef]:
    """Normalize one persisted defaults slot into an ordered ModelRef list.

    Accepts the CURRENT list form (strings "provider/model" and/or
    `{"provider": …, "model": …}` objects), the LEGACY single-ref forms
    (the same string or object, or an already-parsed ModelRef) and
    None/absent (= no default). Unparsable items are dropped silently;
    the list is capped at `MAX_ROLE_DEFAULTS` entries.
    """
    items: list[object]
    if isinstance(value, (str, dict, ModelRef)):
        # Legacy single-ref slot (or a list forgotten by an old writer).
        items = [value]
    elif isinstance(value, list):
        items = list(value)
    else:
        return []
    refs: list[ModelRef] = []
    for item in items:
        ref = item if isinstance(item, ModelRef) else _parse_ref(item)
        if ref is not None and ref not in refs:
            refs.append(ref)
        if len(refs) >= MAX_ROLE_DEFAULTS:
            break
    return refs


class UserModelsConfig(BaseModel):
    """The whole user-level catalog as persisted in models.json."""

    providers: list[Provider] = []
    # Per-role ordered fallback chain for when the project does not pin a
    # model itself: defaults[role][0] is the PRIMARY model of the role
    # (the historical single default), the remaining entries are
    # availability fallbacks tried in order (see `resolve_available`).
    # Persisted as a list of "provider/model" strings per role; the
    # LEGACY single-string form ("defaults": {"coding": "p/m"}) is
    # normalized on load into a one-element list.
    defaults: dict[AgentRole, list[ModelRef]] = {}

    @field_validator("defaults", mode="before")
    @classmethod
    def _parse_default_slots(cls, value: object) -> object:
        """Accept the persisted per-role list (and the legacy single-ref
        string / `{"provider": …, "model": …}` object / null) instead of
        the list-of-ModelRef in-memory form."""
        if not isinstance(value, dict):
            return value
        normalized = {key: _normalize_role_refs(slot) for key, slot in value.items()}
        # Empty chains are absence: don't keep null/empty slots around.
        return {key: refs for key, refs in normalized.items() if refs}

    @field_serializer("defaults", when_used="json")
    def _serialize_default_slots(
        self, value: dict[AgentRole, list[ModelRef]]
    ) -> dict[str, list[str]]:
        return {
            role.value: [str(ref) for ref in refs]
            for role, refs in value.items()
            if refs
        }

    def role_refs(self, role: AgentRole) -> list[ModelRef]:
        """The role's ordered reference chain (empty when unset)."""
        return list(self.defaults.get(role, ()))

    def default_ref(self, role: AgentRole) -> ModelRef | None:
        """The role's PRIMARY reference (the historical single default)."""
        refs = self.defaults.get(role)
        return refs[0] if refs else None

    def set_default(self, role: AgentRole, ref: ModelRef | None) -> None:
        """Set (ref) or clear (None) the role's PRIMARY reference — the
        single-slot API the catalog screen used before the ordered lists;
        it replaces the whole chain of the role."""
        if ref is None:
            self.defaults.pop(role, None)
        else:
            self.defaults[role] = [ref]

    def all_refs(self) -> list[ModelRef]:
        """Every reference of every role, roles in enum order (used by the
        catalog UI to keep references pointing at renamed providers/models)."""
        return [ref for role in AgentRole for ref in self.role_refs(role)]


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
    """Parse one `models[role]` / `defaults[role]` slot: "provider/model"
    (string — provider = the head up to the FIRST slash, model = the rest,
    slashes allowed), a {"provider": …, "model": …} object (legacy), or
    None."""
    if value is None:
        return None
    if isinstance(value, str):
        provider, sep, model = value.partition("/")
        if not sep or not provider or not model:
            return None
        return ModelRef(provider=provider, model=model)
    if isinstance(value, dict):
        try:
            return ModelRef.model_validate(value)
        except ValidationError:
            return None
    return None


def load_project_model_slots(
    project_root: Path,
) -> dict[AgentRole, ModelRef | str]:
    """The per-role model selection of the project (absent slots omitted).

    A slot is either a `ModelRef` (a catalog reference "provider/model")
    or the string "custom" (use `custom_models[role]`). Anything else
    (null, unparsable) is omitted — the same as "nothing selected".
    """
    models = _read_json(_project_config_path(project_root)).get("models")
    if not isinstance(models, dict):
        return {}
    out: dict[AgentRole, ModelRef | str] = {}
    for role in AgentRole:
        raw = models.get(role.value)
        if raw == "custom":
            out[role] = "custom"
            continue
        ref = _parse_ref(raw)
        if ref is not None:
            out[role] = ref
    return out


def load_project_model_refs(project_root: Path) -> dict[AgentRole, ModelRef]:
    """The per-role catalog references of the project (absent slots and
    "custom" slots omitted)."""
    return {
        role: slot
        for role, slot in load_project_model_slots(project_root).items()
        if isinstance(slot, ModelRef)
    }


def load_project_custom_models(
    project_root: Path,
) -> dict[AgentRole, CustomModelEntry]:
    """The per-role custom models of the project (`custom_models` key).

    Broken entries (missing model/base_url, wrong types) are silently
    skipped here — they read as "nothing selected"; the UI surfaces a
    warning when the user tries to select such a role.
    """
    customs = _read_json(_project_config_path(project_root)).get("custom_models")
    if not isinstance(customs, dict):
        return {}
    out: dict[AgentRole, CustomModelEntry] = {}
    for role in AgentRole:
        try:
            out[role] = CustomModelEntry.model_validate(customs.get(role.value))
        except ValidationError:
            continue
    return out


def project_has_model_selection(project_root: Path) -> bool:
    """Whether the project has at least one ACTIVE model entry: a catalog
    reference in `models[role]` or a VALID `custom_models[role]`. Drives
    the first-start setup dialog condition."""
    slots = load_project_model_slots(project_root)
    customs = load_project_custom_models(project_root)
    return bool(customs) or any(isinstance(s, ModelRef) for s in slots.values())


def _write_project_json(project_root: Path, data: dict[str, object]) -> Path:
    """Atomically persist `data` as the project's drcode.json (the legacy
    root-level file is left alone — migrated data always lands in
    `.drcode/`)."""
    path = project_root / PROJECT_STATE_DIRNAME / PROJECT_CONFIG_FILENAME
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


def _apply_project_slot(data: dict[str, object], role: AgentRole, raw: object) -> None:
    """Write one `models[role]` slot into a raw json dict, keeping the
    "no non-null slots → drop the key" invariant."""
    models_raw = data.get("models")
    models = dict(models_raw) if isinstance(models_raw, dict) else {}
    models[role.value] = raw
    if any(v is not None for v in models.values()):
        data["models"] = models
    else:
        # Every slot is null: drop the key entirely (absence == null).
        data.pop("models", None)


def set_project_model(
    project_root: Path, role: AgentRole, ref: ModelRef | None
) -> Path:
    """Persist one role's model selection into `.drcode/drcode.json`.

    Only the role's own slot is touched; every other key (including the
    other two role slots and the custom_models entries) is preserved.
    `ref=None` clears the slot (null = user-level default); a saved
    custom model of the role survives — it can be re-activated from the
    picker. Returns the written path.
    """
    path = _project_config_path(project_root)
    data = _read_json(path)
    _apply_project_slot(
        data, role, None if ref is None else f"{ref.provider}/{ref.model}"
    )
    return _write_project_json(project_root, data)


def set_project_custom_active(project_root: Path, role: AgentRole) -> Path:
    """Activate the role's saved custom model (`models[role] = "custom"`),
    without touching the stored `custom_models[role]` entry."""
    data = _read_json(_project_config_path(project_root))
    _apply_project_slot(data, role, "custom")
    return _write_project_json(project_root, data)


def set_project_custom_model(
    project_root: Path, role: AgentRole, entry: CustomModelEntry
) -> Path:
    """Store (or overwrite) the role's custom model AND activate it
    (`models[role] = "custom"`) in one atomic write."""
    data = _read_json(_project_config_path(project_root))
    _apply_project_slot(data, role, "custom")
    customs_raw = data.get("custom_models")
    customs = dict(customs_raw) if isinstance(customs_raw, dict) else {}
    customs[role.value] = json.loads(entry.model_dump_json())
    data["custom_models"] = customs
    return _write_project_json(project_root, data)


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
        # Whether the path was passed explicitly matters for the agent-guard
        # in `save` (incident ebe6ba15): a verification snippet default-
        # constructed inside a drcode agent's run_bash child must never
        # resolve to the real user catalog.
        self._explicit_path = path is not None
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
        if os.environ.get("DRCODE_TASK_ID") and not self._explicit_path:
            # Defense in depth (incident ebe6ba15): a default-constructed
            # store inside a run_bash child resolves to the REAL
            # ~/.config/drcode/models.json. The error reaches the agent as
            # the snippet's output (same mechanism as ModelRetry /
            # PathViolation) and the agent fixes itself by passing an
            # explicit tmp path. The TUI runs without
            # DRCODE_TASK_ID and are unaffected.
            raise ValueError(
                "UserModelsStore: running under a drcode agent (DRCODE_TASK_ID "
                "is set); writing to the real user catalog is forbidden — "
                "pass an explicit path (a tmp directory) to UserModelsStore()."
            )
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


# -- one-time seeding from legacy flat configs --------------------------------


def _provider_name_for(base_url: str) -> str:
    """Derive a provider name from the endpoint: its host (without port,
    credentials or path), e.g. `https://api.openai.com/v1` → `api.openai.com`.
    Falls back to a generic name for unparseable URLs."""
    netloc = urlsplit(base_url).netloc
    # Drop userinfo ("user:pass@host") and the port ("host:8080").
    host = netloc.rsplit("@", 1)[-1].rsplit(":", 1)[0].strip()
    return host or "provider"


def _provider_from_flat(
    data: dict[str, object], *, carry_compaction: bool = True
) -> Provider | None:
    """Build one provider/model from a legacy flat-config dict (the model
    keys of an old drcode.json), or None when it describes no model."""
    model = str(data.get("model") or "").strip()
    if not model:
        return None
    base_url = str(data.get("base_url") or "").strip() or str(
        Config.model_fields["base_url"].default
    )
    max_context = data.get("max_context_tokens")
    if not isinstance(max_context, int) or max_context <= 0:
        max_context = ModelEntry.model_fields["max_context_tokens"].default
    max_tokens = data.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = None
    # The compaction limit is only carried over when the user set it
    # explicitly; otherwise the entry stays `None` (= inherit), so the
    # flat config / project setting keeps applying until the user pins a
    # per-model value in the catalog UI. The startup migrations pass
    # carry_compaction=False: the setting stays in the json files there,
    # so baking it into the catalog entry would freeze it.
    compaction_threshold = data.get("compaction_threshold")
    if (
        not carry_compaction
        or not isinstance(compaction_threshold, int)
        or not (1 <= compaction_threshold <= 100)
    ):
        compaction_threshold = None
    return Provider(
        name=_provider_name_for(base_url),
        base_url=base_url,
        api_key=str(data.get("api_key") or ""),
        health_check_url=str(data.get("health_check_url") or ""),
        models=[
            ModelEntry(
                id=model,
                max_context_tokens=max_context,
                max_tokens=max_tokens,
                compaction_threshold=compaction_threshold,
            )
        ],
    )


def _providers_from_custom_models(project_root: Path) -> list[Provider]:
    """Build catalog providers from the project's `custom_models` entries
    (used to seed an empty catalog): one provider per distinct connection
    (base_url/api_key/health_check_url), named after the base URL host;
    same-connection models are merged into one provider. Role defaults
    are NOT derived — the seeding only lists what exists."""
    out: list[Provider] = []
    for entry in load_project_custom_models(project_root).values():
        for provider in out:
            if (
                provider.base_url == entry.base_url
                and provider.api_key == entry.api_key
                and provider.health_check_url == entry.health_check_url
            ):
                if not any(m.id == entry.model for m in provider.models):
                    provider.models.append(ModelEntry(id=entry.model))
                break
        else:
            name = _provider_name_for(entry.base_url)
            base = name
            n = 2
            while any(p.name == name for p in out):
                name = f"{base}-{n}"
                n += 1
            out.append(
                Provider(
                    name=name,
                    base_url=entry.base_url,
                    api_key=entry.api_key,
                    health_check_url=entry.health_check_url,
                    models=[ModelEntry(id=entry.model)],
                )
            )
    return out


def load_seeded_catalog(
    store: UserModelsStore, project_root: Path | None = None
) -> tuple[UserModelsConfig, str | None]:
    """Load the catalog, pre-filling it from legacy flat model data.

    Migration convenience for the main TUI's catalog screen: when the
    catalog file does not exist yet but the user-level
    drcode.json and/or the project's `custom_models` already describe
    models, an in-memory catalog with the corresponding providers is built
    from that data (the provider name is derived from the base URL host).

    Nothing is persisted here — models.json appears only when the user
    explicitly saves the catalog screen; quitting without saving discards
    the seed. Role defaults are never derived from the seed (the startup
    migration `migrate_user_config` sets them when it creates the file).

    Returns the catalog plus an optional human-readable note for the UI
    (set when the seed actually happened).
    """
    catalog = store.load()
    if store.path.exists():
        # A catalog already exists — it is the source of truth, never reseed.
        return catalog, None
    seeded: list[Provider] = []
    notes: list[str] = []
    data = _read_json(_config.USER_CONFIG_FILE)
    flat = _provider_from_flat(data)
    if flat is not None:
        seeded.append(flat)
        notes.append(
            f"provider “{flat.name}”, model “{flat.models[0].id}” (from drcode.json)"
        )
    if project_root is not None:
        for provider in _providers_from_custom_models(project_root):
            seeded.append(provider)
            notes.append(f"provider “{provider.name}” (from project custom models)")
    if not seeded:
        return catalog, None
    catalog.providers.extend(seeded)
    note = (
        f"Pre-filled: {'; '.join(notes)}. Not saved yet — press Save to "
        f"write {USER_MODELS_FILENAME}."
    )
    return catalog, note


# -- resolution ---------------------------------------------------------------


def _apply_role_overrides(config: Config, role: AgentRole) -> Config:
    """Overlay the project's per-role overrides over the resolved role
    config — the LAST resolution layer (the role — the most specific scope
    — beats the catalog entry and the flat config / env):

    - `role_max_tokens[role]` beats the catalog entry's `max_tokens` and
      the flat config's default / DRCODE_MAX_TOKENS;
    - `role_request_timeout[role]` sets the role's `request_timeout`
      (seconds; the OpenAI client timeout of `agent_model.build_model`).
      Only roles present in the map get a value — every other role keeps
      0 = the SDK default (600 s).

    See `ProjectSettings.role_max_tokens` / `role_request_timeout`. One
    settings read for both maps (they are edited in the same dialog)."""
    settings = ProjectSettingsStore(config._project_root).load()
    update: dict[str, object] = {}
    tokens = settings.role_max_tokens.get(role.value)
    if tokens is not None and tokens != config.max_tokens:
        update["max_tokens"] = tokens
    timeout = settings.role_request_timeout.get(role.value)
    if timeout is not None and timeout != config.request_timeout:
        update["request_timeout"] = float(timeout)
    if not update:
        return config
    return config.model_copy(update=update)


def resolve(config: Config, role: AgentRole) -> Config:
    """Resolve the effective model connection for one agent role.

    See the module docstring for the precedence. Always returns a `Config`
    (the input unchanged when only the env layer applies), so callers can
    feed it straight into `agent_model.build_model`. A role with no
    selection anywhere resolves to "model not set": an empty `model` in
    the returned config (`is_ready` is False). The project's per-role
    overrides (`ProjectSettings.role_max_tokens` /
    `role_request_timeout`) are applied to every outcome — they are the
    most specific layer.
    """
    return _apply_role_overrides(_resolve_connection(config, role), role)


def _match_ref(catalog: UserModelsConfig, ref: ModelRef) -> Provider | None:
    """Resolve a reference to a catalog provider.

    1. The exact provider-name match: the provider whose name is the
       reference's head wins even when it does not list the model id
       (a hand-entered reference to an unlisted model stays usable — the
       historical behavior).
    2. When the named provider does not exist on THIS machine, match BY
       MODEL ID across the catalog (catalog order): the portable way for
       a project drcode.json committed to git to keep working on machines
       whose user catalog names the same gateway differently. Legitimate,
       never notified.
    """
    provider = next((p for p in catalog.providers if p.name == ref.provider), None)
    if provider is not None:
        return provider
    return next(
        (p for p in catalog.providers if any(m.id == ref.model for m in p.models)),
        None,
    )


def _apply_provider(config: Config, ref: ModelRef, provider: Provider) -> Config:
    """Write the provider's connection (and the catalog entry's token
    caps, when one is listed) over `config`."""
    update: dict[str, object] = {
        "model": ref.model,
        "base_url": provider.base_url,
        "api_key": provider.api_key,
        "health_check_url": provider.health_check_url,
        # The provider-wide extra request-body fields ride with the
        # connection (consumed by `agent_model.base_model_settings`).
        "provider_extra_body": dict(provider.extra_body),
    }
    entry = next((m for m in provider.models if m.id == ref.model), None)
    if entry is not None:
        update["max_context_tokens"] = entry.max_context_tokens
        if entry.max_tokens is not None:
            update["max_tokens"] = entry.max_tokens
        # The compaction limit rides with the model: an entry-level
        # value overrides the flat config, None keeps it (project/user
        # setting).
        if entry.compaction_threshold is not None:
            update["compaction_threshold"] = entry.compaction_threshold
    return config.model_copy(update=update)


def _resolve_connection(config: Config, role: AgentRole) -> Config:
    """Resolve the role's model CONNECTION (without the per-role
    max_tokens overlay) — the body of the historical `resolve`."""
    # 1. Env override: DRCODE_MODEL / DRCODE_BASE_URL win globally for all
    #    roles — they are already baked into `config` by the ordinary
    #    Config layering, so nothing to resolve here.
    if os.environ.get("DRCODE_MODEL") or os.environ.get("DRCODE_BASE_URL"):
        return config

    project_root = config._project_root
    catalog = UserModelsStore().load()

    # 2. Project-level selection: a catalog reference (resolved via the
    #    exact → cross-provider id match) or the role's custom model.
    slot = load_project_model_slots(project_root).get(role)
    if isinstance(slot, ModelRef):
        provider = _match_ref(catalog, slot)
        if provider is not None:
            return _apply_provider(config, slot, provider)
    elif slot == "custom":
        custom = load_project_custom_models(project_root).get(role)
        if custom is not None:
            update: dict[str, object] = {
                "model": custom.model,
                "base_url": custom.base_url,
                "api_key": custom.api_key,
                "health_check_url": custom.health_check_url,
                "max_context_tokens": custom.max_context_tokens,
            }
            if custom.max_tokens is not None:
                update["max_tokens"] = custom.max_tokens
            return config.model_copy(update=update)

    # 3. The user-level ordered chain: primary first, then the availability
    #    fallbacks; the first reference resolving to a catalog provider
    #    wins (dangling ones fall through to the next candidate).
    for ref in catalog.role_refs(role):
        provider = _match_ref(catalog, ref)
        if provider is not None:
            return _apply_provider(config, ref, provider)

    # 4. Nothing selected for this role: "model not set".
    return config.model_copy(update={"model": ""})


def resolve_model_key(config: Config, role: AgentRole) -> str:
    """The usage-accounting model id of the role's EFFECTIVE model: the
    catalog's "<provider>/<model-id>" reference format when the model
    comes from the catalog; the bare model id when it comes from a
    custom slot or the DRCODE_* env override (neither carries a provider
    name). Empty when nothing is selected for the role.

    Mirrors `resolve`/`resolve_effective` precedence — env override, the
    project slot (exact → cross-provider id match, same as
    `_resolve_connection`), the user-level ordered chain — including the
    research/resolver fallback to the coding model. The provider name in
    the key is the one the reference ACTUALLY resolved to (after a
    cross-provider id match), not the one written in the slot. Feeds the
    TUI's usage-accounting key ("<role>/<model-id>", see
    `drcode.usage_store`); kept in sync with `_resolve_connection` by
    construction (same walk, same fall-throughs). Note: this is the
    THEORETICAL resolution — after an availability fallback at run start
    the pipeline pins the factually chosen key over it (see
    `resolve_available`).
    """
    resolved = resolve(config, role)
    if role in (AgentRole.RESEARCH, AgentRole.RESOLVER) and not resolved.is_ready:
        return resolve_model_key(config, AgentRole.CODING)
    if os.environ.get("DRCODE_MODEL") or os.environ.get("DRCODE_BASE_URL"):
        # The env layer wins for all roles and carries no provider name.
        return resolved.model
    catalog = UserModelsStore().load()
    project_root = config._project_root
    slot = load_project_model_slots(project_root).get(role)
    if isinstance(slot, ModelRef):
        provider = _match_ref(catalog, slot)
        if provider is not None:
            return f"{provider.name}/{slot.model}"
    elif slot == "custom":
        custom = load_project_custom_models(project_root).get(role)
        if custom is not None:
            return custom.model
    for ref in catalog.role_refs(role):
        provider = _match_ref(catalog, ref)
        if provider is not None:
            return f"{provider.name}/{ref.model}"
    return resolved.model


def resolve_effective(config: Config, role: AgentRole) -> Config:
    """Resolve the effective model for a role, with the fallback roles.

    The RESEARCH and RESOLVER roles have no default of their own: when
    nothing is selected for them (project slot, custom model, catalog
    default — all unset), they fall back to the resolved CODING model, so
    research tasks and conflict-resolver rounds transparently run on
    whatever the coding role uses and existing configs keep working.
    The env override (`DRCODE_MODEL` / `DRCODE_BASE_URL`) applies to every
    role exactly the same — and makes the fallbacks unreachable, since the
    env layer leaves every role ready. Every other role resolves as usual
    (`resolve`).
    """
    resolved = resolve(config, role)
    if role in (AgentRole.RESEARCH, AgentRole.RESOLVER) and not resolved.is_ready:
        return resolve(config, AgentRole.CODING)
    return resolved


# -- availability-aware resolution (run start) ---------------------------------

# Per-probe timeout, seconds: every candidate provider of a role's chain is
# probed IN PARALLEL with this timeout, so the total run-start delay is
# bounded by ONE timeout — never the sum over the chain. Short on purpose:
# a hanging health endpoint must not stall the run start.
AVAILABILITY_PROBE_TIMEOUT = 2.5


def _probe_ssl_verify() -> ssl.SSLContext | bool:
    """System certificate store when available (same rationale as
    `agent_model._ssl_verify` — duplicated here because agent_model imports
    this module, not the other way round)."""
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return True


async def provider_available(
    provider: Provider, *, timeout: float = AVAILABILITY_PROBE_TIMEOUT
) -> bool:
    """Probe one provider's availability via a cheap GET.

    Endpoint: the cached `health_check_url` when the provider defines one,
    else the canonical OpenAI-compatible `{base_url}/models`. Any answer
    below 500 (except 401/403 — an auth failure is a configuration error,
    not a reachability signal) counts as available; a transport error or a
    5xx means unavailable.

    Graceful degrade for providers WITHOUT a cached health endpoint: an
    unreachable /models probe is not proof of unavailability (some
    gateways only serve /chat/completions), so such a provider is assumed
    available and the run's own network-retry layer deals with a dead
    endpoint. A provider with a health_check_url that does not answer IS
    unavailable (that is the whole point of caching a verified endpoint).
    Never raises.
    """
    base = provider.base_url.strip()
    if not base:
        return False
    health = provider.health_check_url.strip()
    url = health or f"{base.rstrip('/')}/models"
    headers = {"User-Agent": f"drcode/{APP_VERSION}"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    try:
        async with httpx.AsyncClient(
            timeout=timeout, verify=_probe_ssl_verify()
        ) as client:
            resp = await client.get(url, headers=headers)
    except httpx.HTTPError:
        return not health
    if resp.status_code in (401, 403):
        return False
    return resp.status_code < 500


def _availability_candidates(
    config: Config, role: AgentRole
) -> list[tuple[ModelRef, Provider]]:
    """The role's ordered candidate chain: the project slot reference
    (when one is set) first, then the user-level ordered list (primary +
    availability fallbacks). Duplicate references dropped; each reference
    matched to its provider via `_match_ref`; references resolving to no
    provider are dropped."""
    catalog = UserModelsStore().load()
    refs: list[ModelRef] = []
    slot = load_project_model_slots(config._project_root).get(role)
    if isinstance(slot, ModelRef):
        refs.append(slot)
    for ref in catalog.role_refs(role):
        if ref not in refs:
            refs.append(ref)
    out: list[tuple[ModelRef, Provider]] = []
    for ref in refs:
        provider = _match_ref(catalog, ref)
        if provider is not None:
            out.append((ref, provider))
    return out


class AvailableResolution(NamedTuple):
    """The outcome of an availability-aware run-start resolution."""

    # The connection to serve the run with (already overlaid with the
    # role's max_tokens / request-timeout overrides, same as `resolve`).
    config: Config
    # Usage-accounting key of the FACTUALLY chosen model
    # ("<provider>/<model-id>", bare id for custom/env).
    model_key: str
    # True when the chosen provider answered its availability probe — or
    # when no probing applied (env override, custom slot, empty chain).
    # False means the whole candidate chain was probed and every provider
    # is unreachable.
    available: bool
    # Providers probed and found unreachable, in chain priority order
    # (empty when nothing was probed or everything answered).
    unavailable: list[str]


async def _resolve_available_chain(
    config: Config, role: AgentRole, *, timeout: float
) -> AvailableResolution:
    """Availability-aware resolution over the role's candidate chain
    (env/custom shortcuts already handled by the caller)."""
    theoretical_config = resolve(config, role)
    theoretical_key = resolve_model_key(config, role)
    # A "custom" project slot pins one exact connection: the user-level
    # chain (built for availability portability) is deliberately not
    # consulted — and no probing either.
    if load_project_model_slots(config._project_root).get(role) == "custom":
        return AvailableResolution(theoretical_config, theoretical_key, True, [])
    candidates = _availability_candidates(config, role)
    if not candidates:
        return AvailableResolution(theoretical_config, theoretical_key, True, [])
    # Distinct providers in chain-priority order — the availability TIER
    # is the provider as a whole, so two references on one provider are
    # probed (and judged) together.
    unique: dict[str, Provider] = {}
    for _, provider in candidates:
        unique.setdefault(provider.name, provider)
    names = list(unique)
    results = await asyncio.gather(
        *(provider_available(p, timeout=timeout) for p in unique.values())
    )
    up = {name for name, ok in zip(names, results) if ok}
    down = [name for name in names if name not in up]
    chosen = next(((r, p) for r, p in candidates if p.name in up), None)
    if chosen is None:
        # Every candidate is unreachable: degrade — hand back the
        # theoretical primary with the failure list (never crash; the
        # caller decides whether to start anyway).
        return AvailableResolution(theoretical_config, theoretical_key, False, down)
    ref, provider = chosen
    cfg = _apply_role_overrides(_apply_provider(config, ref, provider), role)
    return AvailableResolution(cfg, f"{provider.name}/{ref.model}", True, down)


async def resolve_available(
    config: Config, role: AgentRole, *, timeout: float = AVAILABILITY_PROBE_TIMEOUT
) -> AvailableResolution:
    """Resolve the role's model with availability taken into account —
    the async step the pipeline applies ONCE at every run start (never
    mid-run: there is no fallback inside a run, only between attempts).

    Semantics:

    1. `DRCODE_MODEL` / `DRCODE_BASE_URL` env — no probing, no fallback
       chain: the env layer wins over everything exactly as before.
    2. A "custom" project slot — the project pinned one exact connection;
       no alternatives are consulted (the user-level chain exists for
       availability portability, which a pinned custom connection
       deliberately opts out of).
    3. Otherwise the candidate chain (project reference first, then the
       user-level primary and fallbacks) is probed in parallel and the
       FIRST AVAILABLE candidate by priority wins; its provider name (not
       the reference's — a cross-provider id match may have been used)
       names the model key.
    4. RESEARCH/RESOLVER with nothing usable fall back to the CODING
       chain (the `resolve_effective` inheritance, availability edition).

    All probes run concurrently with a short per-probe timeout: the
    run-start delay is bounded by one timeout, not the chain length.
    Failures degrade (an `available=False` resolution), never raise.
    """
    if os.environ.get("DRCODE_MODEL") or os.environ.get("DRCODE_BASE_URL"):
        return AvailableResolution(
            resolve(config, role), resolve_model_key(config, role), True, []
        )
    result = await _resolve_available_chain(config, role, timeout=timeout)
    if role in (AgentRole.RESEARCH, AgentRole.RESOLVER) and not result.config.is_ready:
        return await resolve_available(config, AgentRole.CODING, timeout=timeout)
    return result


# -- one-time migrations of legacy flat configs --------------------------------


def migrate_user_config() -> bool:
    """Move the user-level flat model config into the catalog.

    When `~/.config/drcode/drcode.json` still carries the legacy flat
    model keys and `models.json` does not exist yet, a catalog with one
    provider/model built from the flat data is written, with every role
    default (coding, research, tester, compaction) pointed at it
    (otherwise the roles would end up without a model after the flat
    fallback was removed; research additionally falls back to the coding
    model via `resolve_effective`, so its default is a convenience, not
    a requirement). The flat model keys are
    then stripped from the user drcode.json (non-provider settings like
    compaction_threshold/request_limit stay).

    Idempotent (a second run is a no-op); every write failure is swallowed
    — the worst case is that the legacy keys stay and are simply ignored
    by the new resolution scheme.
    """
    data = _read_json(_config.USER_CONFIG_FILE)
    if "model" not in data:
        return False
    store = UserModelsStore()
    if not store.path.exists():
        flat = _provider_from_flat(data, carry_compaction=False)
        if flat is not None:
            catalog = UserModelsConfig(providers=[flat])
            ref = ModelRef(provider=flat.name, model=flat.models[0].id)
            catalog.defaults = {role: [ref] for role in AgentRole}
            try:
                store.save(catalog)
            except OSError:
                # Could not write the catalog: keep the flat keys so
                # nothing the user configured is lost.
                return False
    _strip_flat_model_keys(_config.USER_CONFIG_FILE, data)
    return True


def migrate_project_config(project_root: Path) -> bool:
    """Move the project's legacy flat model config into per-role custom
    models.

    When the project's drcode.json (or the legacy root-level one) carries
    the flat model keys, they are converted into `custom_models[role]` for
    every role (coding, research, tester, compaction) with
    `models[role] = "custom"` — independent
    copies. A slot that already holds a catalog reference stays active;
    the converted flat config is parked in `custom_models[role]` without
    activation. The flat keys are then removed from the json file
    (non-provider settings survive). The legacy root-level `drcode.json`
    is migrated into `.drcode/drcode.json` and deleted, as before.

    Idempotent; write failures are swallowed (never fatal at startup).
    """
    current = project_root / PROJECT_STATE_DIRNAME / PROJECT_CONFIG_FILENAME
    legacy = project_root / _LEGACY_PROJECT_CONFIG_FILENAME
    if current.is_file():
        path, legacy_only = current, False
    elif legacy.is_file():
        path, legacy_only = legacy, True
    else:
        return False
    data = _read_json(path)
    has_flat = "model" in data or "base_url" in data
    if has_flat:
        flat = _provider_from_flat(data)
        if flat is not None:
            entry = CustomModelEntry(
                model=flat.models[0].id,
                base_url=flat.base_url,
                api_key=flat.api_key,
                health_check_url=flat.health_check_url,
                max_context_tokens=flat.models[0].max_context_tokens,
                max_tokens=flat.models[0].max_tokens,
            )
            customs_raw = data.get("custom_models")
            customs = dict(customs_raw) if isinstance(customs_raw, dict) else {}
            models_raw = data.get("models")
            for role in AgentRole:
                customs[role.value] = json.loads(entry.model_dump_json())
                # A catalog reference already in the slot stays active;
                # only an unpinned slot activates the migrated custom
                # model.
                existing = (
                    models_raw.get(role.value) if isinstance(models_raw, dict) else None
                )
                if _parse_ref(existing) is None:
                    _apply_project_slot(data, role, "custom")
            data["custom_models"] = customs
        _strip_flat_model_keys(path, data)
    if legacy_only:
        # The legacy root-level file moves into `.drcode/` wholesale (the
        # same transfer the old save path performed), so the root copy
        # can't shadow anything.
        try:
            _write_project_json(project_root, data)
            legacy.unlink()
        except OSError:
            pass
    return has_flat or legacy_only


def _strip_flat_model_keys(path: Path, data: dict[str, object]) -> None:
    """Remove the legacy flat model keys from a drcode.json (best-effort;
    all other keys are preserved)."""
    changed = False
    for key in _FLAT_MODEL_KEYS:
        if key in data:
            del data[key]
            changed = True
    if not changed:
        return
    try:
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
    except OSError:
        pass  # never fatal: the flat keys are simply ignored by resolve
