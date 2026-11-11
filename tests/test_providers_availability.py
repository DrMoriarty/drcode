"""Ordered per-role defaults, cross-provider reference matching and the
availability-aware run-start resolution (drcode.providers)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Self

import httpx
import pytest
from pydantic import ValidationError

import drcode.providers as providers_module
from drcode.config import Config
from drcode.core.state import CoreState
from drcode.providers import (
    MAX_ROLE_DEFAULTS,
    AgentRole,
    AvailableResolution,
    ModelEntry,
    ModelRef,
    Provider,
    UserModelsConfig,
    UserModelsStore,
    migrate_user_config,
    provider_available,
    resolve,
    resolve_available,
    resolve_model_key,
    set_project_model,
)


def make_config(project_root: Path, **data: object) -> Config:
    return Config(project_root=project_root, **data)


def make_catalog() -> UserModelsConfig:
    """Two "machines": vpn-gate (behind a VPN) and direct-gate (reachable
    without one), each serving overlapping model ids."""
    return UserModelsConfig(
        providers=[
            Provider(
                name="vpn-gate",
                base_url="http://vpn.local/v1",
                api_key="vpn-key",
                health_check_url="http://vpn.local/v1/health",
                models=[ModelEntry(id="m1", max_context_tokens=131072)],
            ),
            Provider(
                name="direct-gate",
                base_url="http://direct.local/v1",
                api_key="direct-key",
                health_check_url="http://direct.local/v1/health",
                models=[
                    ModelEntry(id="m1", max_context_tokens=32000),
                    ModelEntry(id="m2"),
                ],
            ),
        ]
    )


def save_catalog(cfg: UserModelsConfig) -> None:
    UserModelsStore().save(cfg)


# -- ModelEntry.favorite -------------------------------------------------------


def test_favorite_defaults_to_zero_and_roundtrips(user_config_dir: Path) -> None:
    store = UserModelsStore()
    save_catalog(
        UserModelsConfig(
            providers=[
                Provider(
                    name="p",
                    base_url="http://p/v1",
                    models=[ModelEntry(id="m", favorite=2), ModelEntry(id="n")],
                )
            ]
        )
    )
    loaded = store.load()
    assert loaded.providers[0].models[0].favorite == 2
    assert loaded.providers[0].models[1].favorite == 0
    raw = json.loads(store.path.read_text())
    assert raw["providers"][0]["models"][0]["favorite"] == 2


def test_favorite_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelEntry(id="m", favorite=4)
    with pytest.raises(ValidationError):
        ModelEntry(id="m", favorite=-1)


# -- ordered per-role defaults -------------------------------------------------


def test_defaults_list_roundtrip_and_cap(user_config_dir: Path) -> None:
    store = UserModelsStore()
    refs = [ModelRef(provider="p", model=f"m{i}") for i in range(5)]
    save_catalog(
        UserModelsConfig(
            providers=[Provider(name="p", base_url="http://p/v1", models=[])],
            defaults={AgentRole.CODING: refs},
        )
    )
    loaded = store.load()
    # Capped at MAX_ROLE_DEFAULTS, order preserved.
    assert loaded.role_refs(AgentRole.CODING) == refs[:MAX_ROLE_DEFAULTS]
    raw = json.loads(store.path.read_text())
    assert raw["defaults"]["coding"] == [str(r) for r in refs[:MAX_ROLE_DEFAULTS]]


def test_defaults_legacy_single_string_loads_as_chain(user_config_dir: Path) -> None:
    store = UserModelsStore()
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps(
            {
                "providers": [
                    {"name": "p", "base_url": "http://p/v1", "models": [{"id": "m"}]}
                ],
                "defaults": {"coding": "p/m"},
            }
        )
    )
    loaded = store.load()
    assert loaded.default_ref(AgentRole.CODING) == ModelRef(provider="p", model="m")
    assert loaded.role_refs(AgentRole.CODING) == [ModelRef(provider="p", model="m")]
    assert store.last_warning is None


def test_defaults_mixed_list_drops_unparsable_and_dedupes(
    user_config_dir: Path,
) -> None:
    store = UserModelsStore()
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps(
            {
                "providers": [],
                "defaults": {
                    "coding": [
                        "p/m",
                        "garbage",
                        None,
                        "p/m",
                        {"provider": "q", "model": "n"},
                    ]
                },
            }
        )
    )
    loaded = store.load()
    assert loaded.role_refs(AgentRole.CODING) == [
        ModelRef(provider="p", model="m"),
        ModelRef(provider="q", model="n"),
    ]


def test_migrate_user_config_writes_ordered_lists(
    user_config_dir: Path,
) -> None:
    config_dir = user_config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "drcode.json").write_text(
        json.dumps({"model": "m", "base_url": "http://x/v1"})
    )
    assert migrate_user_config() is True
    catalog = UserModelsStore().load()
    for role in AgentRole:
        assert catalog.role_refs(role) == [ModelRef(provider="x", model="m")]


# -- resolve order: exact → cross-provider id-match → chain ---------------------


def test_resolve_cross_provider_id_match(project_root: Path) -> None:
    """A reference whose provider does not exist on this machine matches
    BY MODEL ID against any catalog provider — portable project configs
    (no notification, no crash). Tie-break: catalog order."""
    save_catalog(make_catalog())
    set_project_model(
        project_root, AgentRole.CODING, ModelRef(provider="gone", model="m1")
    )
    config = make_config(project_root, model="flat", base_url="http://flat/v1")
    resolved = resolve(config, AgentRole.CODING)
    assert resolved.model == "m1"
    assert resolved.base_url == "http://vpn.local/v1"
    assert resolved.api_key == "vpn-key"
    assert resolved.max_context_tokens == 131072
    # The usage key names the provider the reference ACTUALLY resolved to.
    assert resolve_model_key(config, AgentRole.CODING) == "vpn-gate/m1"


def test_resolve_exact_provider_beats_cross_match(project_root: Path) -> None:
    """When the named provider EXISTS, the exact match wins even if it
    does not list the model id (a hand-entered reference to an unlisted
    model keeps working; no cross-provider jump)."""
    save_catalog(make_catalog())
    set_project_model(
        project_root, AgentRole.CODING, ModelRef(provider="vpn-gate", model="m2")
    )
    config = make_config(project_root)
    resolved = resolve(config, AgentRole.CODING)
    assert resolved.model == "m2"
    assert resolved.base_url == "http://vpn.local/v1"
    assert resolve_model_key(config, AgentRole.CODING) == "vpn-gate/m2"


def test_resolve_user_chain_primary_first_then_fallback(
    project_root: Path,
) -> None:
    catalog = make_catalog()
    catalog.defaults[AgentRole.CODING] = [
        ModelRef(provider="vpn-gate", model="m1"),
        ModelRef(provider="direct-gate", model="m2"),
    ]
    save_catalog(catalog)
    config = make_config(project_root, model="flat", base_url="http://flat/v1")
    # The PRIMARY (first element) wins when it resolves — the project
    # slot is empty.
    resolved = resolve(config, AgentRole.CODING)
    assert resolved.model == "m1"
    assert resolved.base_url == "http://vpn.local/v1"
    assert resolved.max_context_tokens == 131072


def test_resolve_user_chain_skips_dangling(project_root: Path) -> None:
    """A chain entry resolving to NO provider (unknown provider name AND
    an id no catalog provider serves) is skipped in favor of the next."""
    catalog = make_catalog()
    catalog.defaults[AgentRole.CODING] = [
        ModelRef(provider="gone", model="zz"),
        ModelRef(provider="direct-gate", model="m2"),
    ]
    save_catalog(catalog)
    config = make_config(project_root, model="flat", base_url="http://flat/v1")
    resolved = resolve(config, AgentRole.CODING)
    assert resolved.model == "m2"
    assert resolved.base_url == "http://direct.local/v1"


def test_resolve_project_slot_beats_user_chain(project_root: Path) -> None:
    catalog = make_catalog()
    catalog.defaults[AgentRole.CODING] = [ModelRef(provider="vpn-gate", model="m1")]
    save_catalog(catalog)
    set_project_model(
        project_root, AgentRole.CODING, ModelRef(provider="direct-gate", model="m2")
    )
    config = make_config(project_root)
    assert resolve(config, AgentRole.CODING).model == "m2"


def test_resolve_model_key_chain_order(project_root: Path) -> None:
    catalog = make_catalog()
    catalog.defaults[AgentRole.CODING] = [
        ModelRef(provider="gone", model="zz"),
        ModelRef(provider="direct-gate", model="m1"),
    ]
    save_catalog(catalog)
    config = make_config(project_root)
    assert resolve_model_key(config, AgentRole.CODING) == "direct-gate/m1"


# -- provider probe -------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeAsyncClient:
    """httpx.AsyncClient stand-in: canned per-URL outcomes."""

    outcomes: ClassVar[dict[str, object]] = {}  # url → status int or Exception
    requested: ClassVar[list[str]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def get(self, url: str, headers: object = None) -> _FakeResponse:
        _FakeAsyncClient.requested.append(url)
        outcome = _FakeAsyncClient.outcomes.get(url, 200)
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(int(outcome))  # type: ignore[arg-type]


@pytest.fixture
def fake_http(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAsyncClient]:
    _FakeAsyncClient.outcomes = {}
    _FakeAsyncClient.requested = []
    monkeypatch.setattr(providers_module.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


@pytest.fixture
def probe_spy(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Records every provider probe (name, delay, verdict) — replaces
    `provider_available` inside resolve_available."""
    spy = SimpleNamespace(calls=[], delays={}, results={})

    async def fake(provider: Provider, *, timeout: float = 2.5) -> bool:
        spy.calls.append(provider.name)
        await asyncio.sleep(spy.delays.get(provider.name, 0.0))
        return bool(spy.results.get(provider.name, True))

    monkeypatch.setattr(providers_module, "provider_available", fake)
    return spy


VPN = "http://vpn.local/v1/health"
DIRECT = "http://direct.local/v1/health"


def test_provider_available_health_answer(fake_http: type[_FakeAsyncClient]) -> None:
    provider = Provider(
        name="p",
        base_url="http://p/v1",
        health_check_url="http://p/v1/health",
        api_key="k",
    )
    fake_http.outcomes = {"http://p/v1/health": 200}
    assert asyncio.run(provider_available(provider)) is True
    fake_http.outcomes = {"http://p/v1/health": 503}
    assert asyncio.run(provider_available(provider)) is False
    fake_http.outcomes = {"http://p/v1/health": 401}
    assert asyncio.run(provider_available(provider)) is False
    fake_http.outcomes = {"http://p/v1/health": httpx.ConnectError("no route")}
    # A provider WITH a health endpoint that does not answer: unavailable.
    assert asyncio.run(provider_available(provider)) is False


def test_provider_available_without_health_url_degrades(
    fake_http: type[_FakeAsyncClient],
) -> None:
    """No cached health endpoint: a dead /models probe is not proof of
    unavailability (gateways serving only /chat/completions exist) —
    assume available, let the run's retry layer deal with it."""
    provider = Provider(name="p", base_url="http://p/v1")
    fake_http.outcomes = {"http://p/v1/models": httpx.ConnectError("refused")}
    assert asyncio.run(provider_available(provider)) is True
    # ...but an answering probe is still a positive signal.
    fake_http.outcomes = {"http://p/v1/models": 200}
    assert asyncio.run(provider_available(provider)) is True
    # A provider without any base URL is simply unusable.
    assert asyncio.run(provider_available(Provider(name="q", base_url=""))) is False


# -- resolve_available ------------------------------------------------------------


def _catalog_for(
    catalog: UserModelsConfig, role: AgentRole, refs: list[ModelRef]
) -> None:
    catalog.defaults[role] = refs
    save_catalog(catalog)


def test_resolve_available_env_wins_and_skips_probes(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe_spy: SimpleNamespace,
) -> None:
    save_catalog(make_catalog())
    _catalog_for(
        make_catalog(),
        AgentRole.CODING,
        [ModelRef(provider="vpn-gate", model="m1")],
    )
    monkeypatch.setenv("DRCODE_MODEL", "env-model")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://env/v1")
    config = make_config(project_root)
    resolution = asyncio.run(resolve_available(config, AgentRole.CODING))
    assert probe_spy.calls == []  # env outranks everything, no probing
    assert resolution.available is True
    assert resolution.unavailable == []
    assert resolution.config.model == "env-model"
    assert resolution.model_key == "env-model"


def test_resolve_available_first_available_by_priority(
    project_root: Path, probe_spy: SimpleNamespace
) -> None:
    catalog = make_catalog()
    _catalog_for(
        catalog,
        AgentRole.CODING,
        [
            ModelRef(provider="vpn-gate", model="m1"),
            ModelRef(provider="direct-gate", model="m1"),
        ],
    )
    probe_spy.results = {"vpn-gate": False, "direct-gate": True}
    config = make_config(project_root, model="flat", base_url="http://flat/v1")
    resolution = asyncio.run(resolve_available(config, AgentRole.CODING))
    assert resolution.available is True
    assert resolution.config.model == "m1"
    assert resolution.config.base_url == "http://direct.local/v1"
    assert resolution.config.max_context_tokens == 32000
    assert resolution.model_key == "direct-gate/m1"
    assert resolution.unavailable == ["vpn-gate"]


def test_resolve_available_probes_candidates_in_parallel(
    project_root: Path, probe_spy: SimpleNamespace
) -> None:
    """ALL candidates are probed concurrently — a slow available primary
    must not prevent the fallback's probe from running (the total
    run-start delay is bounded by one timeout, not the chain length)."""
    catalog = make_catalog()
    _catalog_for(
        catalog,
        AgentRole.CODING,
        [
            ModelRef(provider="vpn-gate", model="m1"),
            ModelRef(provider="direct-gate", model="m1"),
        ],
    )
    probe_spy.delays = {"vpn-gate": 0.05}
    probe_spy.results = {"vpn-gate": True}
    config = make_config(project_root)
    resolution = asyncio.run(resolve_available(config, AgentRole.CODING))
    # The second candidate was probed even though the first (slower) one
    # ended up available — a sequential short-circuit would never have
    # reached it.
    assert sorted(probe_spy.calls) == ["direct-gate", "vpn-gate"]
    assert resolution.available is True
    assert resolution.model_key == "vpn-gate/m1"
    assert resolution.unavailable == []


def test_resolve_available_all_providers_down_degrades(
    project_root: Path, probe_spy: SimpleNamespace
) -> None:
    catalog = make_catalog()
    _catalog_for(
        catalog,
        AgentRole.CODING,
        [ModelRef(provider="vpn-gate", model="m1")],
    )
    probe_spy.results = {"vpn-gate": False}
    config = make_config(project_root, model="flat", base_url="http://flat/v1")
    resolution = asyncio.run(resolve_available(config, AgentRole.CODING))
    assert resolution.available is False
    assert resolution.unavailable == ["vpn-gate"]
    # The theoretical primary is handed back (the caller decides what to
    # do — degrade, never crash).
    assert resolution.config.model == "m1"
    assert resolution.model_key == "vpn-gate/m1"
    assert resolution.config.is_ready


def test_resolve_available_no_candidates_no_probe(
    project_root: Path, probe_spy: SimpleNamespace
) -> None:
    config = make_config(project_root, model="flat", base_url="http://flat/v1")
    resolution = asyncio.run(resolve_available(config, AgentRole.CODING))
    assert probe_spy.calls == []
    assert resolution.available is True
    assert not resolution.config.is_ready
    assert resolution.model_key == ""


def test_resolve_available_research_inherits_coding_chain(
    project_root: Path, probe_spy: SimpleNamespace
) -> None:
    """Research with no selection of its own falls back to the CODING
    chain — availability edition of `resolve_effective`."""
    catalog = make_catalog()
    _catalog_for(
        catalog,
        AgentRole.CODING,
        [ModelRef(provider="vpn-gate", model="m1")],
    )
    probe_spy.results = {"vpn-gate": True}
    config = make_config(project_root, model="flat", base_url="http://flat/v1")
    resolution = asyncio.run(resolve_available(config, AgentRole.RESEARCH))
    assert resolution.config.model == "m1"
    assert resolution.config.base_url == "http://vpn.local/v1"
    assert resolution.model_key == "vpn-gate/m1"


def test_resolve_available_custom_slot_no_probes(
    project_root: Path, probe_spy: SimpleNamespace
) -> None:
    from drcode.providers import CustomModelEntry, set_project_custom_model

    set_project_custom_model(
        project_root,
        AgentRole.CODING,
        CustomModelEntry(model="custom-m", base_url="http://custom/v1"),
    )
    save_catalog(make_catalog())
    config = make_config(project_root)
    resolution = asyncio.run(resolve_available(config, AgentRole.CODING))
    assert probe_spy.calls == []
    assert resolution.config.model == "custom-m"
    assert resolution.model_key == "custom-m"
    assert resolution.available is True


# -- resolution pinning (core.state) ----------------------------------------------


def test_pin_resolution_overrides_caches_until_invalidated(
    project_root: Path,
) -> None:
    state = CoreState()
    state._resolved_configs = {}
    state._resolved_model_keys = {}
    state._pinned_resolutions = {}
    state.config = make_config(project_root)
    role = AgentRole.CODING
    theoretical = AvailableResolution(
        config=state.config.model_copy(update={"model": "theoretical"}),
        model_key="prov/theoretical",
        available=True,
        unavailable=[],
    )
    state._resolved_configs[role] = theoretical.config
    state._resolved_model_keys[role] = "prov/theoretical"
    assert state.resolved_config(role).model == "theoretical"
    assert state.resolved_model_key(role) == "prov/theoretical"

    pinned = AvailableResolution(
        config=state.config.model_copy(update={"model": "fallback"}),
        model_key="other/fallback",
        available=True,
        unavailable=["prov"],
    )
    state._pin_resolution(role, pinned)
    # The factual choice outranks the cache.
    assert state.resolved_config(role).model == "fallback"
    assert state.resolved_model_key(role) == "other/fallback"

    state.invalidate_resolved_configs()
    assert state._pinned_resolutions == {}
    assert state._resolved_configs == {}
    assert state._resolved_model_keys == {}
