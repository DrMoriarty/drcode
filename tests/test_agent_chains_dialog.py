"""Tests for the agent model chains dialog (drcode.tui.agent_chains_dialog).

Runs the screen headlessly via textual's pilot: prefill from the stored
chains, the right-hand catalog list (favorites highlight + filter),
adding to / removing from the per-role chains (duplicate and 3-entry
limit included), the save path (ordered lists into the user models.json,
empty chains dropped) and the entry point from the provider catalog
screen's "Assign agents…" button.
"""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import App
from textual.widgets import Button, Input, OptionList, Static

from drcode.providers import (
    AgentRole,
    ModelEntry,
    ModelRef,
    Provider,
    UserModelsConfig,
    UserModelsStore,
)
from drcode.tui.agent_chains_dialog import AgentChainsScreen
from drcode.tui.model_setup_app import ModelCatalogScreen


class _Host(App):
    """Minimal host pushing the chains screen (the palette entry point)."""

    def __init__(self, store: UserModelsStore) -> None:
        super().__init__()
        self._store = store

    def on_mount(self) -> None:
        self.push_screen(AgentChainsScreen(store=self._store))


class _CatalogHost(App):
    """Host pushing the provider catalog screen (its entry point)."""

    def __init__(self, store: UserModelsStore) -> None:
        super().__init__()
        self._store = store

    def on_mount(self) -> None:
        self.push_screen(ModelCatalogScreen(store=self._store))


def _seed_store(path: Path) -> UserModelsStore:
    """A catalog with two providers / three models (m2 is a tier-1
    favorite) and no chains yet."""
    store = UserModelsStore(path)
    store.save(
        UserModelsConfig(
            providers=[
                Provider(
                    name="alpha",
                    base_url="http://a/v1",
                    models=[ModelEntry(id="m1"), ModelEntry(id="m2", favorite=1)],
                ),
                Provider(
                    name="beta", base_url="http://b/v1", models=[ModelEntry(id="m3")]
                ),
            ]
        )
    )
    return store


def _right_list(screen: AgentChainsScreen) -> OptionList:
    return screen.query_one("#ac-models", OptionList)


def _group_list(screen: AgentChainsScreen, role: AgentRole) -> OptionList:
    return screen.query_one(f"#ac-list-{role.value}", OptionList)


def _status(screen: AgentChainsScreen) -> str:
    return str(screen.query_one("#ac-status", Static).render())


async def test_prefill_and_favorites_order(tmp_path: Path) -> None:
    """Stored chains prefill the left groups in order; the right list
    shows every catalog model (favorites first, same tier glyphs as the
    catalog table)."""
    path = tmp_path / "models.json"
    store = _seed_store(path)
    catalog = store.load()
    catalog.defaults[AgentRole.CODING] = [
        ModelRef(provider="alpha", model="m1"),
        ModelRef(provider="beta", model="m3"),
    ]
    store.save(catalog)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, AgentChainsScreen)
        coding = _group_list(screen, AgentRole.CODING)
        texts = [str(option.prompt) for option in coding.options]
        assert texts == ["1. alpha/m1  ← primary", "2. beta/m3"]
        header = screen.query_one("#ac-role-coding", Static)
        assert "2/3" in str(header.render())
        # Other roles start empty.
        assert _group_list(screen, AgentRole.RESEARCH).option_count == 0
        # Right list: favorite m2 first, then the rest alphabetically by
        # model id; the favorite row carries the tier glyph.
        right = _right_list(screen)
        labels = [option.id for option in right.options]
        assert labels == ["ref:alpha/m2", "ref:alpha/m1", "ref:beta/m3"]
        first_prompt = str(right.options[0].prompt)
        assert "❶" in first_prompt and "alpha/m2" in first_prompt
        assert "❶" not in str(right.options[1].prompt)


async def test_filter_narrows_model_list(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    store = _seed_store(path)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, AgentChainsScreen)
        flt = screen.query_one("#ac-filter", Input)
        flt.value = "BET"  # case-insensitive provider-name match
        await pilot.pause()
        right = _right_list(screen)
        assert [option.id for option in right.options] == ["ref:beta/m3"]
        flt.value = "m1"  # model-name match
        await pilot.pause()
        assert [option.id for option in right.options] == ["ref:alpha/m1"]
        flt.value = "no-such-model"
        await pilot.pause()
        assert _right_list(screen).option_count == 0


async def test_add_to_chain_and_save(tmp_path: Path) -> None:
    """Highlighting a model on the right and pressing a group's Add
    appends it to that role's chain (insertion order); Save persists the
    ordered lists into the user models.json."""
    path = tmp_path / "models.json"
    store = _seed_store(path)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, AgentChainsScreen)
        right = _right_list(screen)
        right.highlighted = 1  # alpha/m1
        await pilot.pause()
        await pilot.click("#ac-add-coding")
        await pilot.pause()
        right.highlighted = 0  # alpha/m2 (favorite, first row)
        await pilot.pause()
        # Second press via Button.press(): two pilot.click() calls on the
        # same button back to back are flaky — Button._on_click ignores a
        # click while the previous press's `-active` class (a 0.2 s real-
        # time effect) is still set, and pilot.pause() passes no real time.
        screen.query_one("#ac-add-coding", Button).press()
        await pilot.pause()
        texts = [str(o.prompt) for o in _group_list(screen, AgentRole.CODING).options]
        assert texts == ["1. alpha/m1  ← primary", "2. alpha/m2"]
        assert "added to coding" in _status(screen)
        await pilot.click("#ac-save")
        await pilot.pause()
        data = json.loads(path.read_text())
        assert data["defaults"] == {
            "coding": ["alpha/m1", "alpha/m2"],
        }


async def test_add_requires_selected_model(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    store = _seed_store(path)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, AgentChainsScreen)
        await pilot.click("#ac-add-coding")
        await pilot.pause()
        assert "Select a model" in _status(screen)
        assert _group_list(screen, AgentRole.CODING).option_count == 0


async def test_duplicate_add_rejected(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    store = _seed_store(path)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, AgentChainsScreen)
        _right_list(screen).highlighted = 0
        await pilot.pause()
        await pilot.click("#ac-add-coding")
        await pilot.pause()
        screen.query_one("#ac-add-coding", Button).press()
        await pilot.pause()
        assert "already in the coding chain" in _status(screen)
        assert _group_list(screen, AgentRole.CODING).option_count == 1


async def test_limit_three_per_role(tmp_path: Path) -> None:
    """Adding a fourth model is forbidden with a clear message; the chain
    keeps exactly the first three entries in insertion order."""
    path = tmp_path / "models.json"
    store = UserModelsStore(path)
    store.save(
        UserModelsConfig(
            providers=[
                Provider(
                    name="alpha",
                    base_url="http://a/v1",
                    models=[ModelEntry(id="m1"), ModelEntry(id="m2", favorite=1)],
                ),
                Provider(
                    name="beta",
                    base_url="http://b/v1",
                    models=[ModelEntry(id="m3"), ModelEntry(id="m4")],
                ),
            ]
        )
    )
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, AgentChainsScreen)
        right = _right_list(screen)
        add = screen.query_one("#ac-add-tester", Button)
        for index in range(3):
            right.highlighted = index
            await pilot.pause()
            add.press()
            await pilot.pause()
        assert _group_list(screen, AgentRole.TESTER).option_count == 3
        # A fourth add (a model NOT yet in the chain) is rejected.
        right.highlighted = 3  # beta/m4
        await pilot.pause()
        add.press()
        await pilot.pause()
        assert "chain is full (3 models max)" in _status(screen)
        assert _group_list(screen, AgentRole.TESTER).option_count == 3
        await pilot.click("#ac-save")
        await pilot.pause()
        data = json.loads(path.read_text())
        assert data["defaults"]["tester"] == [
            "alpha/m2",
            "alpha/m1",
            "beta/m3",
        ]


async def test_remove_from_chain(tmp_path: Path) -> None:
    """Del removes the highlighted entry; the rest keep their insertion
    order and are renumbered."""
    path = tmp_path / "models.json"
    store = _seed_store(path)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, AgentChainsScreen)
        right = _right_list(screen)
        right.highlighted = 1  # alpha/m1
        await pilot.pause()
        await pilot.click("#ac-add-coding")
        await pilot.pause()
        right.highlighted = 0  # alpha/m2
        await pilot.pause()
        # Second press via Button.press() — same-button double pilot.click
        # can be swallowed inside the 0.2 s `-active` debounce (see
        # test_add_to_chain_and_save).
        screen.query_one("#ac-add-coding", Button).press()
        await pilot.pause()
        # Nothing highlighted in the group list yet → a hint, no removal.
        await pilot.click("#ac-del-coding")
        await pilot.pause()
        assert "Select an entry" in _status(screen)
        coding = _group_list(screen, AgentRole.CODING)
        assert coding.option_count == 2
        coding.highlighted = 0  # alpha/m1 (primary)
        await pilot.pause()
        screen.query_one("#ac-del-coding", Button).press()
        await pilot.pause()
        assert "removed" in _status(screen)
        texts = [str(o.prompt) for o in coding.options]
        assert texts == ["1. alpha/m2  ← primary"]


async def test_save_drops_emptied_chain(tmp_path: Path) -> None:
    """Removing every entry of a prefilled chain and saving drops the
    role's slot entirely (empty chain = absence)."""
    path = tmp_path / "models.json"
    store = _seed_store(path)
    catalog = store.load()
    catalog.defaults[AgentRole.RESEARCH] = [ModelRef(provider="beta", model="m3")]
    store.save(catalog)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, AgentChainsScreen)
        research = _group_list(screen, AgentRole.RESEARCH)
        research.highlighted = 0
        await pilot.pause()
        await pilot.click("#ac-del-research")
        await pilot.pause()
        await pilot.click("#ac-save")
        await pilot.pause()
        data = json.loads(path.read_text())
        assert data.get("defaults", {}) == {}


async def test_escape_closes_without_saving(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    store = _seed_store(path)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, AgentChainsScreen)
        _right_list(screen).highlighted = 0
        await pilot.pause()
        await pilot.click("#ac-add-coding")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        data = json.loads(path.read_text())
        assert data.get("defaults", {}) == {}


async def test_opened_from_catalog_screen_shares_catalog(tmp_path: Path) -> None:
    """The "Assign agents…" button opens the chains dialog over the
    catalog screen's in-memory catalog; the dialog's Save persists the
    chains (together with the shared catalog object), and the catalog
    screen's own Save keeps them."""
    path = tmp_path / "models.json"
    store = _seed_store(path)
    async with _CatalogHost(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        catalog_screen = pilot.app.screen
        assert isinstance(catalog_screen, ModelCatalogScreen)
        await pilot.click("#mc-assign-agents")
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, AgentChainsScreen)
        # The dialog edits the SAME catalog object the catalog screen holds.
        assert screen.catalog is catalog_screen.catalog
        _right_list(screen).highlighted = 2  # beta/m3
        await pilot.pause()
        await pilot.click("#ac-add-compaction")
        await pilot.pause()
        await pilot.click("#ac-save")
        await pilot.pause()
        # Back on the catalog screen after the dialog closed.
        assert isinstance(pilot.app.screen, ModelCatalogScreen)
        data = json.loads(path.read_text())
        assert data["defaults"] == {"compaction": ["beta/m3"]}
        # The catalog screen's Save must not clobber the saved chains.
        await pilot.click("#mc-save")
        await pilot.pause()
        data = json.loads(path.read_text())
        assert data["defaults"] == {"compaction": ["beta/m3"]}


async def test_add_button_ids_cover_all_roles(tmp_path: Path) -> None:
    """Every role group carries its Add/Del buttons (the ids the handlers
    dispatch on)."""
    path = tmp_path / "models.json"
    store = _seed_store(path)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, AgentChainsScreen)
        for role in AgentRole:
            assert screen.query_one(f"#ac-add-{role.value}", Button)
            assert screen.query_one(f"#ac-del-{role.value}", Button)
            assert screen.query_one(f"#ac-list-{role.value}", OptionList)
