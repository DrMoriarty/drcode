"""User-level model catalog and per-role resolution (drcode.providers)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

import drcode.config as config_module
from drcode.agent_model import ModelFetchError, fetch_models
from drcode.config import Config
from drcode.providers import (
    AgentRole,
    CustomModelEntry,
    ModelEntry,
    ModelRef,
    Provider,
    UserModelsConfig,
    UserModelsStore,
    load_project_custom_models,
    load_project_model_refs,
    load_project_model_slots,
    load_seeded_catalog,
    migrate_project_config,
    migrate_user_config,
    project_has_model_selection,
    resolve,
    set_project_custom_active,
    set_project_custom_model,
    set_project_model,
)


def make_config(project_root: Path, **data: object) -> Config:
    return Config(project_root=project_root, **data)


def write_user_catalog(cfg: UserModelsConfig, user_config_dir: Path) -> None:
    UserModelsStore().save(cfg)


@pytest.fixture
def catalog(user_config_dir: Path) -> UserModelsConfig:
    cfg = UserModelsConfig(
        providers=[
            Provider(
                name="local-swap",
                base_url="http://localhost:8080/v1",
                api_key="key-1",
                health_check_url="http://localhost:8080/v1/models",
                models=[
                    ModelEntry(
                        id="qwen3-coder-30b",
                        max_context_tokens=131072,
                        max_tokens=4096,
                    ),
                    ModelEntry(id="small"),
                ],
            ),
            Provider(name="cloud", base_url="https://api.example.com/v1"),
        ]
    )
    write_user_catalog(cfg, user_config_dir)
    return cfg


# -- resolve -----------------------------------------------------------------


def test_resolve_without_selection_is_not_set(project_root: Path) -> None:
    """No catalog/project/env selection: the role resolves to "model not
    set" — the flat config's model fields are NOT a fallback anymore."""
    config = make_config(project_root, model="m", base_url="http://x/v1")
    resolved = resolve(config, AgentRole.CODING)
    assert resolved.model == ""
    assert not resolved.is_ready


def test_resolve_project_slot_wins(
    project_root: Path, catalog: UserModelsConfig
) -> None:
    set_project_model(
        project_root,
        AgentRole.CODING,
        ModelRef(provider="local-swap", model="qwen3-coder-30b"),
    )
    config = make_config(
        project_root, model="flat-m", base_url="http://flat/v1", api_key="flat-key"
    )
    resolved = resolve(config, AgentRole.CODING)
    assert resolved.model == "qwen3-coder-30b"
    assert resolved.base_url == "http://localhost:8080/v1"
    assert resolved.api_key == "key-1"
    assert resolved.max_context_tokens == 131072
    assert resolved.max_tokens == 4096
    assert resolved.health_check_url == "http://localhost:8080/v1/models"
    # Other roles have no selection: "not set".
    tester = resolve(config, AgentRole.TESTER)
    assert tester.model == ""


def test_resolve_custom_slot_applies_all_six_fields(project_root: Path) -> None:
    set_project_custom_model(
        project_root,
        AgentRole.TESTER,
        CustomModelEntry(
            model="custom-m",
            base_url="http://custom/v1",
            api_key="custom-key",
            health_check_url="http://custom/v1/health",
            max_context_tokens=32000,
            max_tokens=2048,
        ),
    )
    config = make_config(project_root, model="flat-m", base_url="http://flat/v1")
    resolved = resolve(config, AgentRole.TESTER)
    assert resolved.model == "custom-m"
    assert resolved.base_url == "http://custom/v1"
    assert resolved.api_key == "custom-key"
    assert resolved.health_check_url == "http://custom/v1/health"
    assert resolved.max_context_tokens == 32000
    assert resolved.max_tokens == 2048
    # The coding role's custom model is a different, independent slot.
    coding = resolve(config, AgentRole.CODING)
    assert coding.model == ""


def test_resolve_custom_slot_without_entry_is_not_selected(
    project_root: Path,
) -> None:
    """`models[role]="custom"` without a valid custom_models[role] entry
    reads as "nothing selected" (no model, and the file is not rewritten)."""
    cfg_file = project_root / ".drcode" / "drcode.json"
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text(
        json.dumps(
            {
                "models": {"tester": "custom"},
                "custom_models": {"tester": {"model": "", "base_url": ""}},
            }
        )
    )
    config = make_config(project_root, model="flat-m")
    assert resolve(config, AgentRole.TESTER).model == ""
    data = json.loads(cfg_file.read_text())
    assert data["custom_models"]["tester"] == {"model": "", "base_url": ""}


def test_resolve_custom_slot_max_tokens_none_keeps_flat(
    project_root: Path,
) -> None:
    set_project_custom_model(
        project_root,
        AgentRole.COMPACTION,
        CustomModelEntry(model="m", base_url="http://c/v1"),
    )
    config = make_config(project_root, max_tokens=1234)
    resolved = resolve(config, AgentRole.COMPACTION)
    assert resolved.model == "m"
    assert resolved.max_tokens == 1234


def test_resolve_user_default_when_project_slot_absent(
    project_root: Path, catalog: UserModelsConfig
) -> None:
    catalog.defaults[AgentRole.COMPACTION] = ModelRef(
        provider="local-swap", model="small"
    )
    UserModelsStore().save(catalog)
    config = make_config(project_root, model="flat-m", base_url="http://flat/v1")
    resolved = resolve(config, AgentRole.COMPACTION)
    assert resolved.model == "small"
    assert resolved.max_context_tokens == 200_000


def test_resolve_dangling_project_ref_falls_through_to_default(
    project_root: Path, catalog: UserModelsConfig
) -> None:
    set_project_model(
        project_root, AgentRole.TESTER, ModelRef(provider="gone", model="x")
    )
    catalog.defaults[AgentRole.TESTER] = ModelRef(provider="local-swap", model="small")
    UserModelsStore().save(catalog)
    config = make_config(project_root, model="flat-m", base_url="http://flat/v1")
    assert resolve(config, AgentRole.TESTER).model == "small"


def test_resolve_dangling_refs_leave_role_not_set(
    project_root: Path, catalog: UserModelsConfig
) -> None:
    set_project_model(
        project_root, AgentRole.TESTER, ModelRef(provider="gone", model="x")
    )
    config = make_config(project_root, model="flat-m", base_url="http://flat/v1")
    assert resolve(config, AgentRole.TESTER).model == ""


def test_resolve_env_override_wins_for_all_roles(
    project_root: Path, catalog: UserModelsConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_project_model(
        project_root,
        AgentRole.CODING,
        ModelRef(provider="local-swap", model="qwen3-coder-30b"),
    )
    set_project_custom_model(
        project_root,
        AgentRole.TESTER,
        CustomModelEntry(model="custom-m", base_url="http://custom/v1"),
    )
    monkeypatch.setenv("DRCODE_MODEL", "env-model")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://env/v1")
    config = make_config(project_root)
    assert config.model == "env-model"
    for role in AgentRole:
        resolved = resolve(config, role)
        assert resolved.model == "env-model"
        assert resolved.base_url == "http://env/v1"


def test_resolve_carries_per_model_compaction_threshold(
    project_root: Path, catalog: UserModelsConfig
) -> None:
    set_project_model(
        project_root,
        AgentRole.CODING,
        ModelRef(provider="local-swap", model="qwen3-coder-30b"),
    )
    catalog.providers[0].models[0].compaction_threshold = 60
    UserModelsStore().save(catalog)
    config = make_config(project_root, model="flat-m", base_url="http://flat/v1")
    resolved = resolve(config, AgentRole.CODING)
    assert resolved.compaction_threshold == 60


def test_resolve_threshold_none_inherits_flat_config(
    project_root: Path, catalog: UserModelsConfig
) -> None:
    set_project_model(
        project_root,
        AgentRole.CODING,
        ModelRef(provider="local-swap", model="qwen3-coder-30b"),
    )
    # Model entry without a compaction override: the flat config's value
    # (default 80, or whatever the project/user files set) must survive.
    config = make_config(project_root, model="flat-m", base_url="http://flat/v1")
    assert resolve(config, AgentRole.CODING).compaction_threshold == 80
    config = make_config(
        project_root,
        model="flat-m",
        base_url="http://flat/v1",
        compaction_threshold=90,
    )
    assert resolve(config, AgentRole.CODING).compaction_threshold == 90


def test_resolve_does_not_mutate_original(
    project_root: Path, catalog: UserModelsConfig
) -> None:
    set_project_model(
        project_root, AgentRole.CODING, ModelRef(provider="local-swap", model="small")
    )
    config = make_config(project_root, model="flat-m", base_url="http://flat/v1")
    resolve(config, AgentRole.CODING)
    assert config.model == "flat-m"


# -- project models key --------------------------------------------------------


def test_set_project_model_roundtrip_and_preserves_other_keys(
    project_root: Path,
) -> None:
    cfg_file = project_root / ".drcode" / "drcode.json"
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text(
        json.dumps(
            {
                "model": "flat-m",
                "base_url": "http://flat/v1",
                "compaction_threshold": 90,
                "models": {"tester": "cloud/gpt-x"},
            }
        )
    )
    set_project_model(
        project_root, AgentRole.CODING, ModelRef(provider="local-swap", model="q")
    )
    data = json.loads(cfg_file.read_text())
    assert data["model"] == "flat-m"
    assert data["compaction_threshold"] == 90
    assert data["models"]["tester"] == "cloud/gpt-x"
    assert data["models"]["coding"] == "local-swap/q"
    refs = load_project_model_refs(project_root)
    assert refs[AgentRole.CODING] == ModelRef(provider="local-swap", model="q")
    assert refs[AgentRole.TESTER] == ModelRef(provider="cloud", model="gpt-x")

    # Clearing one slot keeps the other.
    set_project_model(project_root, AgentRole.CODING, None)
    data = json.loads(cfg_file.read_text())
    assert "coding" not in data["models"] or data["models"]["coding"] is None
    assert data["models"]["tester"] == "cloud/gpt-x"

    # Clearing the last slot removes the whole key.
    set_project_model(project_root, AgentRole.TESTER, None)
    data = json.loads(cfg_file.read_text())
    assert "models" not in data


def test_set_project_custom_model_roundtrip(project_root: Path) -> None:
    entry = CustomModelEntry(
        model="custom-m",
        base_url="http://custom/v1",
        api_key="k",
        health_check_url="http://custom/health",
        max_context_tokens=100000,
        max_tokens=512,
    )
    set_project_custom_model(project_root, AgentRole.TESTER, entry)
    data = json.loads((project_root / ".drcode" / "drcode.json").read_text())
    assert data["models"]["tester"] == "custom"
    assert data["custom_models"]["tester"]["model"] == "custom-m"
    loaded = load_project_custom_models(project_root)
    assert loaded[AgentRole.TESTER] == entry
    assert load_project_model_slots(project_root) == {AgentRole.TESTER: "custom"}

    # Overwriting one role's entry leaves the others untouched.
    set_project_custom_model(
        project_root,
        AgentRole.CODING,
        CustomModelEntry(model="coding-m", base_url="http://c2/v1"),
    )
    set_project_custom_active(project_root, AgentRole.CODING)
    data = json.loads((project_root / ".drcode" / "drcode.json").read_text())
    assert data["models"]["tester"] == "custom"
    assert data["models"]["coding"] == "custom"
    assert data["custom_models"]["tester"]["model"] == "custom-m"
    assert data["custom_models"]["coding"]["model"] == "coding-m"

    # set_project_model (catalog ref) preserves the custom entry ("parked").
    set_project_model(project_root, AgentRole.CODING, ModelRef(provider="p", model="m"))
    data = json.loads((project_root / ".drcode" / "drcode.json").read_text())
    assert data["models"]["coding"] == "p/m"
    assert data["custom_models"]["coding"]["model"] == "coding-m"

    # Clearing the catalog ref re-activates nothing automatically, but the
    # custom entry survives.
    set_project_model(project_root, AgentRole.CODING, None)
    data = json.loads((project_root / ".drcode" / "drcode.json").read_text())
    assert "coding" not in data["models"] or data["models"]["coding"] is None
    assert data["custom_models"]["coding"]["model"] == "coding-m"


def test_broken_custom_entries_ignored(project_root: Path) -> None:
    cfg_file = project_root / ".drcode" / "drcode.json"
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text(
        json.dumps(
            {
                "custom_models": {
                    "coding": {"base_url": "http://x/v1"},  # no model
                    "tester": {"model": "m"},  # no base_url
                    "compaction": "garbage",
                }
            }
        )
    )
    assert load_project_custom_models(project_root) == {}
    assert not project_has_model_selection(project_root)


def test_project_has_model_selection(project_root: Path) -> None:
    assert not project_has_model_selection(project_root)
    # A custom entry counts as a selection...
    set_project_custom_model(
        project_root,
        AgentRole.COMPACTION,
        CustomModelEntry(model="m", base_url="http://c/v1"),
    )
    assert project_has_model_selection(project_root)
    # ...but a broken one does not.
    cfg_file = project_root / ".drcode" / "drcode.json"
    cfg_file.write_text(json.dumps({"custom_models": {"coding": {"model": ""}}}))
    assert not project_has_model_selection(project_root)
    # A catalog reference does.
    set_project_model(project_root, AgentRole.CODING, ModelRef(provider="p", model="m"))
    assert project_has_model_selection(project_root)


# -- migrations ------------------------------------------------------------------


def _project_config_file(project_root: Path) -> Path:
    return project_root / ".drcode" / "drcode.json"


def test_migrate_user_config_creates_catalog_with_role_defaults(
    user_config_dir: Path,
) -> None:
    config_module.USER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config_module.USER_CONFIG_FILE.write_text(
        json.dumps(
            {
                "model": "qwen3-coder-30b",
                "base_url": "http://localhost:8080/v1",
                "api_key": "key-1",
                "health_check_url": "http://localhost:8080/v1/models",
                "max_context_tokens": 131072,
                "max_tokens": 4096,
                "compaction_threshold": 65,
                "request_limit": 100,
            }
        )
    )
    assert migrate_user_config() is True
    # The catalog file was created with one provider and ALL three role
    # defaults pointed at the migrated model...
    catalog = UserModelsStore().load()
    assert len(catalog.providers) == 1
    provider = catalog.providers[0]
    assert provider.name == "localhost"
    assert provider.base_url == "http://localhost:8080/v1"
    assert provider.api_key == "key-1"
    assert provider.models == [
        ModelEntry(id="qwen3-coder-30b", max_context_tokens=131072, max_tokens=4096)
    ]
    for role in AgentRole:
        assert catalog.defaults[role] == ModelRef(
            provider="localhost", model="qwen3-coder-30b"
        )
    # ...and the flat model keys are gone from the user drcode.json, while
    # the non-provider settings survive.
    data = json.loads(config_module.USER_CONFIG_FILE.read_text())
    for key in (
        "model",
        "base_url",
        "api_key",
        "health_check_url",
        "max_context_tokens",
        "max_tokens",
    ):
        assert key not in data
    assert data["compaction_threshold"] == 65
    assert data["request_limit"] == 100

    # Idempotent: a second run is a no-op.
    assert migrate_user_config() is False
    assert UserModelsStore().load() == catalog


def test_migrate_user_config_without_flat_model_is_noop(
    user_config_dir: Path,
) -> None:
    config_module.USER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config_module.USER_CONFIG_FILE.write_text(json.dumps({"compaction_threshold": 70}))
    assert migrate_user_config() is False
    assert not UserModelsStore().path.exists()
    assert json.loads(config_module.USER_CONFIG_FILE.read_text()) == {
        "compaction_threshold": 70
    }


def test_migrate_user_config_keeps_keys_when_catalog_write_fails(
    user_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwritable models.json keeps the flat keys (nothing is lost)."""
    config_module.USER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config_module.USER_CONFIG_FILE.write_text(
        json.dumps({"model": "m", "base_url": "http://x/v1"})
    )

    def fail_save(self: UserModelsStore, catalog: UserModelsConfig) -> Path:
        raise OSError("read-only")

    monkeypatch.setattr(UserModelsStore, "save", fail_save)
    assert migrate_user_config() is False
    data = json.loads(config_module.USER_CONFIG_FILE.read_text())
    assert data["model"] == "m"


def test_migrate_project_config_moves_flat_keys_to_custom_models(
    project_root: Path,
) -> None:
    cfg_file = _project_config_file(project_root)
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text(
        json.dumps(
            {
                "model": "flat-m",
                "base_url": "http://flat/v1",
                "api_key": "flat-key",
                "health_check_url": "http://flat/health",
                "max_context_tokens": 64000,
                "max_tokens": 2048,
                "compaction_threshold": 90,
                "models": {"tester": "cloud/gpt-x"},
            }
        )
    )
    assert migrate_project_config(project_root) is True
    data = json.loads(cfg_file.read_text())
    # Three independent per-role copies...
    for role in AgentRole:
        assert data["custom_models"][role.value]["model"] == "flat-m"
        assert data["custom_models"][role.value]["base_url"] == "http://flat/v1"
    assert data["models"]["coding"] == "custom"
    assert data["models"]["compaction"] == "custom"
    # ...but a slot that already held a catalog reference stays active,
    # with the flat config parked (not activated) in custom_models.
    assert data["models"]["tester"] == "cloud/gpt-x"
    # Flat keys removed, non-provider settings preserved.
    for key in ("model", "base_url", "api_key"):
        assert key not in data
    assert data["compaction_threshold"] == 90
    # The migrated custom models resolve for their roles.
    config = make_config(project_root)
    for role in (AgentRole.CODING, AgentRole.COMPACTION):
        resolved = resolve(config, role)
        assert resolved.model == "flat-m"
        assert resolved.base_url == "http://flat/v1"
    # The kept catalog reference is dangling (no user catalog here) and
    # there is no user default: the role reads as "not selected".
    assert resolve(config, AgentRole.TESTER).model == ""

    # Idempotent.
    assert migrate_project_config(project_root) is False
    assert json.loads(cfg_file.read_text()) == data


def test_migrate_project_config_legacy_root_file(project_root: Path) -> None:
    """A legacy root-level drcode.json is migrated into .drcode/ (flat
    model keys converted, the stale root copy removed)."""
    legacy = project_root / "drcode.json"
    legacy.write_text(json.dumps({"model": "old-m", "base_url": "http://old/v1"}))
    assert migrate_project_config(project_root) is True
    assert not legacy.exists()
    data = json.loads(_project_config_file(project_root).read_text())
    assert data["custom_models"]["coding"]["model"] == "old-m"
    assert data["models"]["coding"] == "custom"
    assert "model" not in data


def test_migrate_project_config_no_flat_keys_is_noop(project_root: Path) -> None:
    cfg_file = _project_config_file(project_root)
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text(json.dumps({"compaction_threshold": 55}))
    assert migrate_project_config(project_root) is False
    assert json.loads(cfg_file.read_text()) == {"compaction_threshold": 55}


def test_migrate_project_config_transfers_legacy_without_model_keys(
    project_root: Path,
) -> None:
    """A legacy root-level drcode.json with only non-provider settings is
    still transferred into .drcode/ (the old save-path behavior)."""
    legacy = project_root / "drcode.json"
    legacy.write_text(json.dumps({"compaction_threshold": 33}))
    assert migrate_project_config(project_root) is True
    assert not legacy.exists()
    assert json.loads(_project_config_file(project_root).read_text()) == {
        "compaction_threshold": 33
    }
    # Idempotent: nothing left to transfer.
    assert migrate_project_config(project_root) is False


def test_migrations_run_before_resolution(
    project_root: Path, user_config_dir: Path
) -> None:
    """Full legacy startup: user flat config → catalog defaults, project
    flat config → three custom models; resolution then works as before."""
    config_module.USER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config_module.USER_CONFIG_FILE.write_text(
        json.dumps({"model": "user-m", "base_url": "http://user/v1"})
    )
    cfg_file = _project_config_file(project_root)
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text(json.dumps({"model": "flat-m", "base_url": "http://flat/v1"}))
    migrate_user_config()
    migrate_project_config(project_root)
    config = make_config(project_root)
    # The project's custom models beat the fresh catalog defaults.
    for role in AgentRole:
        resolved = resolve(config, role)
        assert resolved.model == "flat-m"
        assert resolved.base_url == "http://flat/v1"
    # Without the project migration the catalog defaults would apply.
    cfg_file.unlink()
    config = make_config(project_root)
    for role in AgentRole:
        assert resolve(config, role).model == "user-m"


# -- seeding from the project custom models --------------------------------------


def test_seed_from_project_custom_models(project_root: Path) -> None:
    set_project_custom_model(
        project_root,
        AgentRole.CODING,
        CustomModelEntry(
            model="qwen3-coder-30b",
            base_url="http://localhost:8080/v1",
            api_key="key-1",
        ),
    )
    set_project_custom_model(
        project_root,
        AgentRole.TESTER,
        CustomModelEntry(
            model="tester-m",
            base_url="http://localhost:8080/v1",
            api_key="key-1",
        ),
    )
    set_project_custom_model(
        project_root,
        AgentRole.COMPACTION,
        CustomModelEntry(model="cloud-m", base_url="https://api.example.com/v1"),
    )
    store = UserModelsStore()
    catalog, note = load_seeded_catalog(store, project_root=project_root)
    assert not store.path.exists()  # nothing persisted by loading alone
    assert note is not None
    # Same connection → one provider with both models; the other connection
    # → its own provider.
    by_name = {p.name: p for p in catalog.providers}
    local = by_name["localhost"]
    assert {m.id for m in local.models} == {"qwen3-coder-30b", "tester-m"}
    assert local.base_url == "http://localhost:8080/v1"
    assert local.api_key == "key-1"
    cloud = by_name["api.example.com"]
    assert [m.id for m in cloud.models] == ["cloud-m"]
    # Role defaults are never derived from the seed.
    assert catalog.defaults == {}


def test_seed_name_collision_disambiguated(project_root: Path) -> None:
    set_project_custom_model(
        project_root,
        AgentRole.CODING,
        CustomModelEntry(model="m1", base_url="http://host/v1", api_key="a"),
    )
    set_project_custom_model(
        project_root,
        AgentRole.TESTER,
        CustomModelEntry(model="m2", base_url="http://host:9000/v1", api_key="b"),
    )
    catalog, _ = load_seeded_catalog(UserModelsStore(), project_root=project_root)
    names = {p.name for p in catalog.providers}
    assert len(names) == 2
    assert any(name.startswith("host") for name in names)
    # Distinct connections never merge.
    assert {p.api_key for p in catalog.providers} == {"a", "b"}


def test_seed_skipped_without_project_or_flat(project_root: Path) -> None:
    catalog, note = load_seeded_catalog(UserModelsStore())
    assert note is None
    assert catalog.providers == []


def test_provider_and_model_names_reject_slash() -> None:
    """Neither provider names nor model ids may contain '/': it is the
    separator of the "provider/model" reference format."""
    with pytest.raises(ValidationError):
        Provider(name="a/b", base_url="http://x/v1")
    with pytest.raises(ValidationError):
        ModelEntry(id="a/b")
    # A valid pair passes.
    Provider(
        name="openrouter", base_url="http://x/v1", models=[ModelEntry(id="llama-3")]
    )


# -- user catalog store ----------------------------------------------------------


def test_store_save_load_roundtrip(user_config_dir: Path) -> None:
    store = UserModelsStore()
    cfg = UserModelsConfig(
        providers=[
            Provider(name="p", base_url="http://p/v1", models=[ModelEntry(id="m")])
        ],
        defaults={AgentRole.CODING: ModelRef(provider="p", model="m")},
    )
    store.save(cfg)
    loaded = UserModelsStore().load()
    assert loaded == cfg
    assert loaded.defaults[AgentRole.CODING] == ModelRef(provider="p", model="m")
    assert UserModelsStore().last_warning is None


def test_store_defaults_saved_as_strings(user_config_dir: Path) -> None:
    """Each defaults slot is persisted as one "provider/model" string."""
    store = UserModelsStore()
    cfg = UserModelsConfig(
        providers=[
            Provider(name="p", base_url="http://p/v1", models=[ModelEntry(id="m")])
        ],
        defaults={AgentRole.CODING: ModelRef(provider="p", model="m")},
    )
    store.save(cfg)
    data = json.loads(store.path.read_text())
    assert data["defaults"]["coding"] == "p/m"


def test_store_loads_string_defaults(user_config_dir: Path) -> None:
    """The persisted string form parses back into ModelRef slots."""
    store = UserModelsStore()
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps(
            {
                "providers": [
                    {"name": "p", "base_url": "http://p/v1", "models": [{"id": "m"}]}
                ],
                "defaults": {"coding": "p/m", "tester": None},
            }
        )
    )
    loaded = UserModelsStore().load()
    assert loaded.defaults[AgentRole.CODING] == ModelRef(provider="p", model="m")
    assert loaded.defaults[AgentRole.TESTER] is None
    assert UserModelsStore().last_warning is None


def test_store_loads_legacy_object_defaults(user_config_dir: Path) -> None:
    """The pre-string format ({"provider": …, "model": …} objects) still
    loads, so an old models.json survives the format change."""
    store = UserModelsStore()
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps(
            {
                "providers": [
                    {"name": "p", "base_url": "http://p/v1", "models": [{"id": "m"}]}
                ],
                "defaults": {
                    "coding": {"provider": "p", "model": "m"},
                },
            }
        )
    )
    loaded = UserModelsStore().load()
    assert loaded.defaults[AgentRole.CODING] == ModelRef(provider="p", model="m")
    assert UserModelsStore().last_warning is None


def test_store_corrupt_falls_back_to_bak(user_config_dir: Path) -> None:
    store = UserModelsStore()
    cfg = UserModelsConfig(
        providers=[
            Provider(name="p", base_url="http://p/v1", models=[ModelEntry(id="m")])
        ]
    )
    store.save(cfg)
    # The .bak copy of the previous version only exists from the second
    # save on (nothing to back up on the very first one).
    store.save(cfg)
    store.path.write_text("{corrupted")
    loader = UserModelsStore()
    loaded = loader.load()
    assert loaded.providers[0].name == "p"
    assert loader.last_warning is not None


def test_store_corrupt_without_bak_returns_defaults(
    user_config_dir: Path,
) -> None:
    store = UserModelsStore()
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{corrupted")
    loader = UserModelsStore()
    loaded = loader.load()
    assert loaded.providers == []
    assert loaded.defaults == {}
    assert loader.last_warning is not None


def test_store_missing_file_is_empty(user_config_dir: Path) -> None:
    loader = UserModelsStore()
    loaded = loader.load()
    assert loaded.providers == []
    assert loaded.defaults == {}
    assert loader.last_warning is None


# -- seeding from the user-level flat config -------------------------------------


def _write_user_drcode_json(user_config_dir: Path, data: dict[str, object]) -> None:
    import drcode.config as config_module

    config_module.USER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config_module.USER_CONFIG_FILE.write_text(json.dumps(data))


def test_seed_from_user_config_when_models_json_missing(
    user_config_dir: Path,
) -> None:
    _write_user_drcode_json(
        user_config_dir,
        {
            "model": "qwen3-coder-30b",
            "base_url": "http://localhost:8080/v1",
            "api_key": "key-1",
            "health_check_url": "http://localhost:8080/v1/models",
            "max_context_tokens": 131072,
            "max_tokens": 4096,
        },
    )
    store = UserModelsStore()
    catalog, note = load_seeded_catalog(store)
    assert not store.path.exists()  # nothing persisted by loading alone
    assert note is not None
    assert len(catalog.providers) == 1
    provider = catalog.providers[0]
    assert provider.name == "localhost"
    assert provider.base_url == "http://localhost:8080/v1"
    assert provider.api_key == "key-1"
    assert provider.health_check_url == "http://localhost:8080/v1/models"
    assert provider.models == [
        ModelEntry(
            id="qwen3-coder-30b",
            max_context_tokens=131072,
            max_tokens=4096,
        )
    ]


def test_seed_fills_defaults_for_omitted_fields(user_config_dir: Path) -> None:
    _write_user_drcode_json(user_config_dir, {"model": "gpt-4o"})
    catalog, note = load_seeded_catalog(UserModelsStore())
    assert note is not None
    provider = catalog.providers[0]
    assert provider.name == "api.openai.com"
    assert provider.base_url == "https://api.openai.com/v1"
    assert provider.api_key == ""
    assert provider.models[0].max_context_tokens == 200_000
    assert provider.models[0].max_tokens is None


def test_seed_carries_explicit_compaction_threshold(user_config_dir: Path) -> None:
    _write_user_drcode_json(
        user_config_dir,
        {"model": "m1", "compaction_threshold": 65},
    )
    catalog, note = load_seeded_catalog(UserModelsStore())
    assert note is not None
    assert catalog.providers[0].models[0].compaction_threshold == 65


def test_seed_leaves_compaction_threshold_unset_when_absent(
    user_config_dir: Path,
) -> None:
    _write_user_drcode_json(user_config_dir, {"model": "m1"})
    catalog, _ = load_seeded_catalog(UserModelsStore())
    # None = inherit: the flat config keeps applying until the user pins
    # a per-model value in the catalog UI.
    assert catalog.providers[0].models[0].compaction_threshold is None


def test_seed_survives_save(user_config_dir: Path) -> None:
    _write_user_drcode_json(
        user_config_dir, {"model": "m1", "base_url": "http://x:9000/v1"}
    )
    store = UserModelsStore()
    catalog, _ = load_seeded_catalog(store)
    store.save(catalog)
    reloaded = UserModelsStore().load()
    assert reloaded == catalog
    # The seeded provider name has no "/" so it is a valid reference head.
    assert "/" not in reloaded.providers[0].name


def test_seed_skipped_when_models_json_exists(user_config_dir: Path) -> None:
    _write_user_drcode_json(user_config_dir, {"model": "flat-m"})
    store = UserModelsStore()
    # Even an empty-but-existing catalog file is the source of truth.
    store.save(UserModelsConfig())
    catalog, note = load_seeded_catalog(store)
    assert note is None
    assert catalog.providers == []


def test_seed_skipped_without_user_drcode_json(user_config_dir: Path) -> None:
    catalog, note = load_seeded_catalog(UserModelsStore())
    assert note is None
    assert catalog.providers == []


def test_seed_skipped_when_user_config_has_no_model(user_config_dir: Path) -> None:
    _write_user_drcode_json(user_config_dir, {"base_url": "http://x/v1"})
    catalog, note = load_seeded_catalog(UserModelsStore())
    assert note is None
    assert catalog.providers == []


# -- fetch_models ------------------------------------------------------------------


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


async def test_fetch_models_canonical_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        assert request.headers["Authorization"] == "Bearer k"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "a", "max_model_len": 4096},
                    {"id": "b", "context_window": "not-a-number"},
                    {"id": "c", "meta": {"n_ctx": 8192}},
                    {"id": "d"},
                    {"no_id": True},
                ]
            },
        )

    entries = await fetch_models("http://x/v1", "k", _transport=_transport(handler))
    assert [e.id for e in entries] == ["a", "b", "c", "d"]
    assert entries[0].max_context_tokens == 4096
    assert entries[1].max_context_tokens == 200_000
    assert entries[2].max_context_tokens == 8192
    assert entries[3].max_context_tokens == 200_000


async def test_fetch_models_v1_fallback() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/models":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    entries = await fetch_models("http://x", "", _transport=_transport(handler))
    assert [e.id for e in entries] == ["m"]
    assert seen == ["/models", "/v1/models"]


async def test_fetch_models_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    with pytest.raises(ModelFetchError, match="auth failed"):
        await fetch_models("http://x/v1", "bad", _transport=_transport(handler))


async def test_fetch_models_no_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    with pytest.raises(ModelFetchError):
        await fetch_models("http://x/v1", "", _transport=_transport(handler))
