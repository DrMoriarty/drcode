"""Smoke tests for the model-catalog screen (drcode.tui.model_setup_app).

Runs the screen headlessly via textual's pilot: provider add/edit, model
addition, context-size editing, role-default persistence and the save path.
"""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import App
from textual.widgets import DataTable, Input, OptionList, Select

from drcode.providers import (
    AgentRole,
    ModelEntry,
    Provider,
    UserModelsConfig,
    UserModelsStore,
)
from drcode.tui.model_setup_app import ModelCatalogScreen


class _Host(App):
    """Minimal host pushing the catalog screen (mirrors both entry points)."""

    def __init__(
        self, store: UserModelsStore, project_root: Path | None = None
    ) -> None:
        super().__init__()
        self._store = store
        self._project_root = project_root

    def on_mount(self) -> None:
        self.push_screen(
            ModelCatalogScreen(store=self._store, project_root=self._project_root)
        )


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
        # Add a model by hand from the editor fields (id + context size).
        screen.query_one("#mc-model-id", Input).value = "my-model"
        screen.query_one("#mc-ctx", Input).value = "131072"
        await pilot.click("#mc-model-add")
        await pilot.pause()
        assert screen._selected_model == "my-model"
        # Save with the model picked as the coding default.
        select = screen.query_one("#mc-role-coding", Select)
        select.value = "p/my-model"
        await pilot.click("#mc-save")
        await pilot.pause()
        data = json.loads(path.read_text())
        entry = data["providers"][0]["models"][0]
        assert entry["id"] == "my-model"
        assert entry["max_context_tokens"] == 131072
        assert data["defaults"]["coding"] == "p/my-model"


async def test_add_and_edit_model_compaction_threshold(tmp_path: Path) -> None:
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
        screen.query_one("#mc-model-id", Input).value = "my-model"
        screen.query_one("#mc-ctx", Input).value = "131072"
        # Add a model by hand with a per-model compaction limit straight
        # from the editor fields.
        screen.query_one("#mc-pct", Input).value = "60"
        await pilot.click("#mc-model-add")
        await pilot.pause()
        assert screen.catalog.providers[0].models[0].compaction_threshold == 60
        # The model row stays selected: editing the limit via Update works.
        assert screen._selected_model == "my-model"
        pct_input = screen.query_one("#mc-pct", Input)
        pct_input.value = "60"
        await pilot.click("#mc-model-update")
        await pilot.pause()
        assert screen.catalog.providers[0].models[0].compaction_threshold == 60
        # Enter in the field applies the value too: clearing it removes the
        # override (inherit again).
        pct_input.focus()
        pct_input.value = ""
        await pilot.press("enter")
        await pilot.pause()
        assert screen.catalog.providers[0].models[0].compaction_threshold is None
        # Set it again and save.
        pct_input.value = "75"
        await pilot.press("enter")
        await pilot.pause()
        assert screen.catalog.providers[0].models[0].compaction_threshold == 75
        await pilot.click("#mc-save")
        await pilot.pause()
        entry = json.loads(path.read_text())["providers"][0]["models"][0]
        assert entry["compaction_threshold"] == 75


async def test_add_model_validation(tmp_path: Path) -> None:
    """Add model creates a new entry from the editor fields; missing id,
    duplicate id and a bad context size are rejected."""
    store = UserModelsStore(tmp_path / "models.json")
    store.save(
        UserModelsConfig(
            providers=[
                Provider(name="p", base_url="http://p/v1", models=[ModelEntry(id="m")])
            ]
        )
    )
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, ModelCatalogScreen)
        status = screen.query_one("#mc-status")
        # Empty id (clear the editor first — mounting the table
        # auto-fills it with the first row).
        screen.query_one("#mc-model-id", Input).value = ""
        await pilot.click("#mc-model-add")
        await pilot.pause()
        assert "Enter a model id" in str(status.render())
        # Duplicate id. (Consecutive clicks on the same button need a real
        # delay: the press animation swallows clicks for ~0.2s.)
        screen.query_one("#mc-model-id", Input).value = "m"
        await pilot.pause(0.3)
        await pilot.click("#mc-model-add")
        await pilot.pause()
        assert "already exists" in str(status.render())
        # Bad context size.
        screen.query_one("#mc-model-id", Input).value = "new-m"
        screen.query_one("#mc-ctx", Input).value = "zero"
        await pilot.pause(0.3)
        await pilot.click("#mc-model-add")
        await pilot.pause()
        assert "positive integer" in str(status.render())
        # Valid: the model is created with the entered caps.
        screen.query_one("#mc-ctx", Input).value = "8192"
        screen.query_one("#mc-pct", Input).value = "50"
        await pilot.pause(0.3)
        await pilot.click("#mc-model-add")
        await pilot.pause()
        provider = screen.catalog.providers[0]
        added = provider.models[-1]
        assert added.id == "new-m"
        assert added.max_context_tokens == 8192
        assert added.compaction_threshold == 50
        assert "added new-m" in str(status.render())


async def test_compaction_threshold_input_validation(tmp_path: Path) -> None:
    store = UserModelsStore(tmp_path / "models.json")
    store.save(
        UserModelsConfig(
            providers=[
                Provider(
                    name="p",
                    base_url="http://p/v1",
                    models=[ModelEntry(id="m")],
                )
            ]
        )
    )
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, ModelCatalogScreen)
        screen._selected_model = "m"
        pct_input = screen.query_one("#mc-pct", Input)
        pct_input.focus()
        screen.query_one("#mc-pct", Input).value = "not-a-number"
        await pilot.press("enter")
        await pilot.pause()
        status = screen.query_one("#mc-status")
        assert "between 1 and 100" in str(status.render())
        screen.query_one("#mc-pct", Input).value = "150"
        await pilot.press("enter")
        await pilot.pause()
        assert "between 1 and 100" in str(status.render())
        assert screen.catalog.providers[0].models[0].compaction_threshold is None


async def test_row_selection_fills_editor_and_update_remove(tmp_path: Path) -> None:
    store = UserModelsStore(tmp_path / "models.json")
    store.save(
        UserModelsConfig(
            providers=[
                Provider(
                    name="p",
                    base_url="http://p/v1",
                    models=[
                        ModelEntry(
                            id="m1",
                            max_context_tokens=100_000,
                            compaction_threshold=60,
                        ),
                        ModelEntry(id="m2"),
                    ],
                )
            ]
        )
    )
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, ModelCatalogScreen)
        # Single selection (arrow keys — no double click) fills all three
        # editor inputs with the selected model's actual values.
        table = screen.query_one("#mc-models", DataTable)
        table.focus()
        await pilot.press("down")
        await pilot.pause()
        # m2: default context, no per-model limit (inherit).
        assert screen.query_one("#mc-model-id", Input).value == "m2"
        assert screen.query_one("#mc-ctx", Input).value == "200000"
        assert screen.query_one("#mc-pct", Input).value == ""
        await pilot.press("up")
        await pilot.pause()
        # m1: its actual context size and compaction limit.
        assert screen.query_one("#mc-model-id", Input).value == "m1"
        assert screen.query_one("#mc-ctx", Input).value == "100000"
        assert screen.query_one("#mc-pct", Input).value == "60"
        # Edit the id and the limit, then Update: the model is renamed
        # (role-default references follow) and the limit is applied.
        screen.query_one("#mc-model-id", Input).value = "m1-renamed"
        screen.query_one("#mc-pct", Input).value = "70"
        await pilot.click("#mc-model-update")
        await pilot.pause()
        provider = screen.catalog.providers[0]
        assert [m.id for m in provider.models] == ["m1-renamed", "m2"]
        assert provider.models[0].compaction_threshold == 70
        assert provider.models[0].max_context_tokens == 100_000
        # Remove: the model is gone and the editor inputs are cleared.
        await pilot.click("#mc-model-del")
        await pilot.pause()
        assert [m.id for m in provider.models] == ["m2"]
        assert screen.query_one("#mc-model-id", Input).value == ""
        assert screen.query_one("#mc-ctx", Input).value == ""
        assert screen.query_one("#mc-pct", Input).value == ""


async def test_slash_rejected_in_provider_and_model_names(tmp_path: Path) -> None:
    """'/' is the provider/model reference separator: the catalog screen
    refuses it both in the provider name and in the model id."""
    store = UserModelsStore(tmp_path / "models.json")
    store.save(
        UserModelsConfig(
            providers=[
                Provider(name="p", base_url="http://p/v1", models=[ModelEntry(id="m")])
            ]
        )
    )
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, ModelCatalogScreen)
        status = screen.query_one("#mc-status")
        # Provider name with a slash.
        screen.query_one("#mc-name", Input).value = "a/b"
        await pilot.click("#mc-save")
        await pilot.pause()
        assert "provider name must not contain '/'" in str(status.render())
        # Model id with a slash (Add model path).
        screen.query_one("#mc-name", Input).value = "p"
        screen.query_one("#mc-model-id", Input).value = "x/y"
        await pilot.click("#mc-model-add")
        await pilot.pause()
        assert "model id must not contain '/'" in str(status.render())
        assert [m.id for m in screen.catalog.providers[0].models] == ["m"]


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


# -- seeding from the user-level drcode.json --------------------------------------


def _write_user_drcode_json(data: dict[str, object]) -> None:
    import drcode.config as config_module

    config_module.USER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config_module.USER_CONFIG_FILE.write_text(json.dumps(data))


async def test_screen_seeded_from_user_config_and_saved_explicitly(
    tmp_path: Path,
) -> None:
    _write_user_drcode_json(
        {"model": "qwen3-coder-30b", "base_url": "http://localhost:8080/v1"}
    )
    path = tmp_path / "models.json"
    store = UserModelsStore(path)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, ModelCatalogScreen)
        # The seeded provider is already in the list and shown in the form…
        options = screen.query_one("#mc-providers", OptionList)
        assert any((option.id or "") == "prov:localhost" for option in options.options)
        assert screen.query_one("#mc-name", Input).value == "localhost"
        assert screen.query_one("#mc-url", Input).value == "http://localhost:8080/v1"
        status = screen.query_one("#mc-status")
        assert "Pre-filled" in str(status.render())
        # …but nothing is written until the user saves.
        assert not path.exists()
        await pilot.click("#mc-save")
        await pilot.pause()
        assert path.exists()
        data = json.loads(path.read_text())
        provider = data["providers"][0]
        assert provider["name"] == "localhost"
        assert provider["models"][0]["id"] == "qwen3-coder-30b"


async def test_seeded_screen_escape_discards_seed(tmp_path: Path) -> None:
    _write_user_drcode_json({"model": "flat-m", "base_url": "http://x/v1"})
    path = tmp_path / "models.json"
    store = UserModelsStore(path)
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        # Quitting without saving: no models.json is created, the flat
        # config keeps working as before.
        assert not path.exists()


async def test_standalone_app_escape_quits() -> None:
    """The `drcode-models` utility has the catalog screen as its whole UI:
    pressing esc must exit the app, not leave an empty black terminal."""
    from drcode.tui.model_setup_app import ModelCatalogApp

    app = ModelCatalogApp()
    async with app.run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.return_code == 0


# -- seeding from the project custom models ---------------------------------------


async def test_screen_seeded_from_project_custom_models_only_on_save(
    tmp_path: Path, project_root: Path
) -> None:
    """No models.json, but the project has custom models: the catalog
    screen pre-fills itself from them — in memory only; the file appears
    solely after an explicit Save (and no role defaults are derived)."""
    from drcode.providers import CustomModelEntry, set_project_custom_model

    set_project_custom_model(
        project_root,
        AgentRole.CODING,
        CustomModelEntry(
            model="qwen3-coder-30b",
            base_url="http://localhost:8080/v1",
            api_key="key-1",
        ),
    )
    path = tmp_path / "models.json"
    store = UserModelsStore(path)
    async with _Host(store, project_root=project_root).run_test(
        size=(120, 56)
    ) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, ModelCatalogScreen)
        options = screen.query_one("#mc-providers", OptionList)
        assert any((option.id or "") == "prov:localhost" for option in options.options)
        assert screen.query_one("#mc-name", Input).value == "localhost"
        status = screen.query_one("#mc-status")
        assert "custom models" in str(status.render())
        # Nothing written yet.
        assert not path.exists()
        await pilot.click("#mc-save")
        await pilot.pause()
        assert path.exists()
        data = json.loads(path.read_text())
        provider = data["providers"][0]
        assert provider["name"] == "localhost"
        assert provider["base_url"] == "http://localhost:8080/v1"
        assert provider["api_key"] == "key-1"
        assert provider["models"][0]["id"] == "qwen3-coder-30b"
        # No role defaults derived from the seed.
        assert data.get("defaults", {}) == {}


async def test_screen_not_seeded_from_project_when_catalog_exists(
    tmp_path: Path, project_root: Path
) -> None:
    from drcode.providers import CustomModelEntry, set_project_custom_model

    set_project_custom_model(
        project_root,
        AgentRole.CODING,
        CustomModelEntry(model="m", base_url="http://x/v1"),
    )
    path = tmp_path / "models.json"
    store = UserModelsStore(path)
    store.save(UserModelsConfig())  # an existing catalog is the truth
    async with _Host(store).run_test(size=(120, 56)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, ModelCatalogScreen)
        assert screen.catalog.providers == []
