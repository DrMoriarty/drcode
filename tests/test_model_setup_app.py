"""Smoke tests for the model-catalog screen (drcode.tui.model_setup_app).

Runs the screen headlessly via textual's pilot: provider add/edit, model
addition, context-size editing, role-default persistence and the save path.
"""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import App
from textual.widgets import Input, OptionList, Select

from drcode.providers import Provider, UserModelsConfig, UserModelsStore
from drcode.tui.model_setup_app import ModelCatalogScreen


class _Host(App):
    """Minimal host pushing the catalog screen (mirrors both entry points)."""

    def __init__(self, store: UserModelsStore) -> None:
        super().__init__()
        self._store = store

    def on_mount(self) -> None:
        self.push_screen(ModelCatalogScreen(store=self._store))


async def test_mount_empty_catalog(tmp_path: Path) -> None:
    store = UserModelsStore(tmp_path / "models.json")
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        options = pilot.app.screen.query_one("#mc-providers", OptionList)
        assert len(options.options) == 1  # only "＋ Add provider…"


async def test_add_provider_and_save(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    store = UserModelsStore(path)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        await pilot.click("#mc-add")
        await pilot.pause()
        pilot.app.screen.query_one("#mc-name", Input).value = "local-swap"
        pilot.app.screen.query_one("#mc-url", Input).value = "http://localhost:8080/v1"
        pilot.app.screen.query_one("#mc-key", Input).value = "secret"
        await pilot.click("#mc-save")
        await pilot.pause()
        data = json.loads(path.read_text())
        provider = data["providers"][0]
        assert provider["name"] == "local-swap"
        assert provider["base_url"] == "http://localhost:8080/v1"
        assert provider["api_key"] == "secret"


async def test_add_and_edit_model_context(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    store = UserModelsStore(path)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        await pilot.click("#mc-add")
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, ModelCatalogScreen)
        screen.query_one("#mc-name", Input).value = "p"
        screen.query_one("#mc-url", Input).value = "http://p/v1"
        # Add a model by hand.
        screen.query_one("#mc-new-model", Input).value = "my-model"
        await pilot.click("#mc-model-add")
        await pilot.pause()
        # Select its row and set the context size.
        screen._selected_model = "my-model"
        screen.query_one("#mc-ctx", Input).value = "131072"
        await pilot.click("#mc-ctx-set")
        await pilot.pause()
        # Save with the model picked as the coding default.
        select = screen.query_one("#mc-role-coding", Select)
        select.value = "p/my-model"
        await pilot.click("#mc-save")
        await pilot.pause()
        data = json.loads(path.read_text())
        entry = data["providers"][0]["models"][0]
        assert entry["id"] == "my-model"
        assert entry["max_context_tokens"] == 131072
        assert data["defaults"]["coding"] == {"provider": "p", "model": "my-model"}


async def test_duplicate_provider_name_rejected(tmp_path: Path) -> None:
    store = UserModelsStore(tmp_path / "models.json")
    store.save(
        UserModelsConfig(providers=[Provider(name="taken", base_url="http://t/v1")])
    )
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        await pilot.click("#mc-add")
        await pilot.pause()
        pilot.app.screen.query_one("#mc-name", Input).value = "taken"
        pilot.app.screen.query_one("#mc-url", Input).value = "http://x/v1"
        await pilot.click("#mc-save")
        await pilot.pause()
        status = pilot.app.screen.query_one("#mc-status")
        assert "already exists" in str(status.render())


async def test_escape_closes_without_saving(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    store = UserModelsStore(path)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        await pilot.click("#mc-add")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not path.exists()
