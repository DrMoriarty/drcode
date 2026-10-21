"""User-level model catalog and per-role resolution (drcode.providers)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from drcode.agent_model import ModelFetchError, fetch_models
from drcode.config import Config
from drcode.providers import (
    AgentRole,
    ModelEntry,
    ModelRef,
    Provider,
    UserModelsConfig,
    UserModelsStore,
    load_project_model_refs,
    load_seeded_catalog,
    resolve,
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
                        context_reported=True,
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


def test_resolve_falls_back_to_flat_config(project_root: Path) -> None:
    config = make_config(project_root, model="m", base_url="http://x/v1")
    resolved = resolve(config, AgentRole.CODING)
    assert resolved.model == "m"
    assert resolved.base_url == "http://x/v1"


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
    # Other roles keep the flat config.
    tester = resolve(config, AgentRole.TESTER)
    assert tester.model == "flat-m"


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


def test_resolve_dangling_refs_use_flat_config(
    project_root: Path, catalog: UserModelsConfig
) -> None:
    set_project_model(
        project_root, AgentRole.TESTER, ModelRef(provider="gone", model="x")
    )
    config = make_config(project_root, model="flat-m", base_url="http://flat/v1")
    assert resolve(config, AgentRole.TESTER).model == "flat-m"


def test_resolve_env_override_wins_for_all_roles(
    project_root: Path, catalog: UserModelsConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_project_model(
        project_root,
        AgentRole.CODING,
        ModelRef(provider="local-swap", model="qwen3-coder-30b"),
    )
    # The flat config comes from the project file (the ordinary Config
    # layering); DRCODE_MODEL set afterwards must win over everything.
    cfg_file = project_root / ".drcode" / "drcode.json"
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(json.dumps({"model": "flat-m", "base_url": "http://flat/v1"}))
    monkeypatch.setenv("DRCODE_MODEL", "env-model")
    config = make_config(project_root)
    assert config.model == "env-model"
    for role in AgentRole:
        resolved = resolve(config, role)
        assert resolved.model == "env-model"
        assert resolved.base_url == "http://flat/v1"


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


def test_project_ref_with_slashed_model_id(project_root: Path) -> None:
    set_project_model(
        project_root,
        AgentRole.CODING,
        ModelRef(provider="openrouter", model="meta-llama/llama-3"),
    )
    refs = load_project_model_refs(project_root)
    assert refs[AgentRole.CODING].model == "meta-llama/llama-3"


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
            context_reported=False,
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
    assert provider.models[0].context_reported is False


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
    assert entries[0].context_reported is True
    assert entries[1].max_context_tokens == 200_000
    assert entries[1].context_reported is False
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
