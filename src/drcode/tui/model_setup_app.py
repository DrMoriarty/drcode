"""Model-catalog editor: providers, auto-discovered models, role defaults.

Edits the user-level catalog (`~/.config/drcode/models.json`, see
`drcode.providers`) — the only UI that writes that file, so no manual JSON
editing is ever needed. Two entry points share this one screen:

1. the standalone `drcode-models` utility (a tiny App pushing the screen);
2. the main TUI's Settings palette ("Provider catalog…"), which pushes the
   same screen modally.

Per provider: name / base URL / API key editing with a live connection
check (`probe_model`), one-click model auto-discovery (`fetch_models`,
existing entries keep their manual edits, new ones are appended) and an
editable model table (context sizes; entries whose context size the
gateway did not report are flagged for manual confirmation). Per agent
role (coding / tester / compaction) a default model can be picked from the
catalog — used when the project does not pin its own.

When models.json does not exist yet but the user-level drcode.json already
describes a model, the screen starts pre-filled with one provider/model
seeded from that data (`providers.load_seeded_catalog`); the file is only
written when the user explicitly saves.
"""

from __future__ import annotations

from typing import ClassVar

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, OptionList, Select, Static
from textual.widgets.option_list import Option

from ..agent import ModelFetchError, fetch_models, probe_model
from ..providers import (
    USER_MODELS_FILENAME,
    AgentRole,
    ModelEntry,
    ModelRef,
    Provider,
    UserModelsConfig,
    UserModelsStore,
    load_seeded_catalog,
)

_PROBE_TIMEOUT = 15.0

_ROLES: tuple[AgentRole, ...] = (
    AgentRole.CODING,
    AgentRole.TESTER,
    AgentRole.COMPACTION,
)


class ModelCatalogScreen(ModalScreen[None]):
    """Provider/model catalog editor (modal inside the main TUI, or the
    whole UI of the standalone `drcode-models` utility)."""

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Close", priority=True),
    ]

    DEFAULT_CSS = """
    ModelCatalogScreen {
        align: center middle;
    }
    #mc-dialog {
        width: 100;
        max-width: 98%;
        height: 96%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
        overflow-y: auto;
    }
    .mc-title {
        color: $accent;
        text-style: bold;
    }
    #mc-main { height: auto; }
    #mc-left { width: 34; margin-right: 2; height: 22; }
    #mc-providers { height: 1fr; margin-top: 1; }
    #mc-right { width: 1fr; height: auto; }
    #mc-name, #mc-url, #mc-key { height: 3; }
    .mc-btnrow {
        height: auto;
        margin-top: 1;
    }
    .mc-btnrow Button { margin-right: 1; }
    #mc-status {
        height: auto;
        min-height: 1;
        margin-top: 1;
        color: $text-muted;
    }
    #mc-models {
        height: 8;
        margin-top: 1;
    }
    #mc-ctx, #mc-new-model { width: 16; }
    #mc-pct { width: 16; }
    #mc-roles { height: auto; margin-top: 1; }
    #mc-roles Select { width: 1fr; margin-right: 1; }
    #mc-footer {
        height: auto;
        margin-top: 1;
    }
    #mc-hint {
        width: 1fr;
        padding: 1 0 0 1;
        color: $text-disabled;
        text-align: right;
    }
    """

    def __init__(self, store: UserModelsStore | None = None) -> None:
        super().__init__()
        self.store = store or UserModelsStore()
        # Load the catalog, pre-filling it from the user-level drcode.json
        # when the catalog file does not exist yet (one-time migration aid;
        # in-memory only until the user presses Save).
        self.catalog: UserModelsConfig
        self.seed_note: str | None
        self.catalog, self.seed_note = load_seeded_catalog(self.store)
        self._current: str | None = None
        self._selected_model: str | None = None
        # Guards against re-entrant commits while a fetch/probe runs.
        self._busy = False

    # -- compose ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="mc-dialog"):
            yield Static(
                "Model catalog — OpenAI-compatible providers and their models. "
                f"Stored in [b]{self.store.path.name}[/b] (user level).",
                classes="mc-title",
            )
            with Horizontal(id="mc-main"):
                with Vertical(id="mc-left"):
                    yield Static("Providers", classes="mc-title")
                    yield OptionList(id="mc-providers")
                    with Horizontal(classes="mc-btnrow"):
                        yield Button("+ Add", id="mc-add")
                        yield Button("Delete", id="mc-del-provider")
                with Vertical(id="mc-right"):
                    yield Input(
                        id="mc-name", placeholder="provider name, e.g. local-swap"
                    )
                    yield Input(
                        id="mc-url",
                        placeholder="base URL, e.g. http://localhost:8080/v1",
                    )
                    yield Input(
                        id="mc-key",
                        password=True,
                        placeholder="API key (optional — leave empty if not needed)",
                    )
                    with Horizontal(classes="mc-btnrow"):
                        yield Button("Check connection", id="mc-check")
                        yield Button("Fetch models", id="mc-fetch")
                    yield Static("", id="mc-status")
                    yield DataTable(id="mc-models", cursor_type="row")
                    with Horizontal(classes="mc-btnrow"):
                        yield Input(id="mc-ctx", placeholder="context tokens")
                        yield Button("Set ctx", id="mc-ctx-set")
                        yield Button("Remove", id="mc-model-del")
                    with Horizontal(classes="mc-btnrow"):
                        yield Input(
                            id="mc-pct", placeholder="compact % (empty = inherit)"
                        )
                        yield Button("Set pct", id="mc-pct-set")
                    with Horizontal(classes="mc-btnrow"):
                        yield Input(id="mc-new-model", placeholder="add model id")
                        yield Button("Add model", id="mc-model-add")
            with Horizontal(id="mc-roles"):
                for role in _ROLES:
                    yield Select(
                        [(f"{role.value}: (none)", "")],
                        prompt=f"{role.value}:",
                        id=f"mc-role-{role.value}",
                        allow_blank=True,
                    )
            with Horizontal(id="mc-footer"):
                yield Button("Save", variant="primary", id="mc-save")
                yield Static(
                    "esc — close without saving · model table: Enter/click "
                    "selects a row, then set its context size and compaction "
                    "limit (compact %, empty = inherit the global setting)",
                    id="mc-hint",
                )

    # -- helpers ------------------------------------------------------------

    def _status(self, text: str) -> None:
        self.query_one("#mc-status", Static).update(text)

    def _provider(self) -> Provider | None:
        if self._current is None:
            return None
        return next(
            (p for p in self.catalog.providers if p.name == self._current), None
        )

    def _commit_form(self) -> bool:
        """Flush the provider form into the catalog. False on a validation
        error (the status line explains)."""
        provider = self._provider()
        if provider is None:
            return True
        name = self.query_one("#mc-name", Input).value.strip()
        url = self.query_one("#mc-url", Input).value.strip()
        key = self.query_one("#mc-key", Input).value.strip()
        if not name:
            self._status("✗ provider name must not be empty")
            return False
        if not url:
            self._status("✗ base URL must not be empty")
            return False
        if name != provider.name and any(
            p.name == name for p in self.catalog.providers
        ):
            self._status(f"✗ provider '{name}' already exists")
            return False
        if name != provider.name:
            # Rename: keep the role-default references pointing at the
            # same provider.
            old = provider.name
            provider.name = name
            for ref in self.catalog.defaults.values():
                if ref is not None and ref.provider == old:
                    ref.provider = name
            self._current = name
        provider.base_url = url
        provider.api_key = key
        return True

    def _refresh_providers(self) -> None:
        option_list = self.query_one("#mc-providers", OptionList)
        option_list.clear_options()
        for provider in self.catalog.providers:
            option_list.add_option(
                Option(
                    f"{provider.name}  ({len(provider.models)} models)",
                    id=f"prov:{provider.name}",
                )
            )
        option_list.add_option(Option("＋ Add provider…", id="add"))
        if self._current is not None:
            for index, option in enumerate(option_list.options):
                if option.id == f"prov:{self._current}":
                    option_list.highlighted = index
                    break

    def _refresh_models(self) -> None:
        table = self.query_one("#mc-models", DataTable)
        table.clear(columns=True)
        table.add_columns("model id", "context", "context size")
        provider = self._provider()
        if provider is None:
            return
        for entry in provider.models:
            table.add_row(
                entry.id,
                str(entry.max_context_tokens),
                "from gateway" if entry.context_reported else "default — confirm!",
                key=entry.id,
            )

    def _refresh_roles(self) -> None:
        model_refs = [
            (f"{provider.name}/{entry.id}", f"{provider.name}/{entry.id}")
            for provider in self.catalog.providers
            for entry in provider.models
        ]
        for role in _ROLES:
            select = self.query_one(f"#mc-role-{role.value}", Select)
            # The role is part of every label: one compact row of three
            # selects shows which slot is which even with a value picked.
            select.set_options([(f"{role.value}: (none)", "")] + model_refs)
            ref = self.catalog.defaults.get(role)
            select.value = f"{ref.provider}/{ref.model}" if ref else ""

    def _load_form(self, name: str | None) -> None:
        self._current = name
        self._selected_model = None
        provider = self._provider()
        self.query_one("#mc-name", Input).value = provider.name if provider else ""
        self.query_one("#mc-url", Input).value = provider.base_url if provider else ""
        self.query_one("#mc-key", Input).value = provider.api_key if provider else ""
        self.query_one("#mc-ctx", Input).value = ""
        self.query_one("#mc-new-model", Input).value = ""
        self._refresh_models()
        self._status("")

    # -- lifecycle ------------------------------------------------------------

    def on_mount(self) -> None:
        if self.store.last_warning:
            self._status(f"⚠ {self.store.last_warning}")
        if self.catalog.providers and self._current is None:
            self._current = self.catalog.providers[0].name
        self._refresh_providers()
        self._load_form(self._current)
        self._refresh_roles()
        if self.store.last_warning:
            # Re-assert after _load_form (it clears the status line).
            self._status(f"⚠ {self.store.last_warning}")
        elif self.seed_note:
            self._status(f"ℹ {self.seed_note}")
        self.query_one("#mc-providers", OptionList).focus()

    def action_close(self) -> None:
        self.dismiss(None)

    # -- provider list ----------------------------------------------------------

    @on(OptionList.OptionSelected, "#mc-providers")
    def _provider_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id or ""
        if option_id == "add":
            self._add_provider()
            return
        if option_id.startswith("prov:"):
            if not self._commit_form():
                return
            self._load_form(option_id[len("prov:") :])

    def _add_provider(self) -> None:
        if not self._commit_form():
            return
        base = "new-provider"
        name = base
        n = 2
        while any(p.name == name for p in self.catalog.providers):
            name = f"{base}-{n}"
            n += 1
        self.catalog.providers.append(Provider(name=name, base_url=""))
        self._refresh_providers()
        self._load_form(name)
        self.query_one("#mc-name", Input).focus()

    @on(Button.Pressed, "#mc-add")
    def _add_clicked(self) -> None:
        self._add_provider()

    @on(Button.Pressed, "#mc-del-provider")
    def _delete_provider(self) -> None:
        if not self._commit_form():
            return
        provider = self._provider()
        if provider is None:
            return
        self.catalog.providers = [
            p for p in self.catalog.providers if p.name != provider.name
        ]
        for role in _ROLES:
            ref = self.catalog.defaults.get(role)
            if ref is not None and ref.provider == provider.name:
                self.catalog.defaults.pop(role, None)
        self._current = (
            self.catalog.providers[0].name if self.catalog.providers else None
        )
        self._refresh_providers()
        self._refresh_roles()
        self._load_form(self._current)

    # -- connection check / model discovery --------------------------------------

    @on(Button.Pressed, "#mc-check")
    async def _check_connection(self) -> None:
        if self._busy or not self._commit_form():
            return
        provider = self._provider()
        if provider is None:
            return
        self._busy = True
        try:
            self._status("Checking connection…")
            if provider.models:
                # probe_model needs a model id: check with the first one.
                ok, detail, checked = await probe_model(
                    provider.models[0].id,
                    provider.base_url,
                    provider.api_key,
                    timeout=_PROBE_TIMEOUT,
                    health_check_url=provider.health_check_url,
                )
                if ok:
                    provider.health_check_url = checked
                    self._status(f"✓ connection ok (via {checked})")
                else:
                    self._status(f"✗ connection failed: {detail}")
            else:
                try:
                    entries = await fetch_models(provider.base_url, provider.api_key)
                except ModelFetchError as e:
                    self._status(f"✗ connection failed: {e}")
                else:
                    self._status(
                        f"✓ connection ok — {len(entries)} models available "
                        "(use “Fetch models” to import them)"
                    )
        finally:
            self._busy = False

    @on(Button.Pressed, "#mc-fetch")
    async def _fetch_models(self) -> None:
        if self._busy or not self._commit_form():
            return
        provider = self._provider()
        if provider is None:
            return
        self._busy = True
        try:
            self._status("Fetching models…")
            try:
                entries = await fetch_models(provider.base_url, provider.api_key)
            except ModelFetchError as e:
                self._status(f"✗ fetch failed: {e}")
                return
            existing = {m.id for m in provider.models}
            fresh = [e for e in entries if e.id not in existing]
            provider.models.extend(fresh)
            self._refresh_models()
            self._refresh_providers()
            unconfirmed = sum(1 for e in fresh if not e.context_reported)
            note = (
                f"; {unconfirmed} context size(s) not reported — confirm by hand"
                if unconfirmed
                else ""
            )
            self._status(f"✓ fetched {len(entries)} models ({len(fresh)} new){note}")
        finally:
            self._busy = False

    # -- model table -----------------------------------------------------------------

    @on(DataTable.RowSelected, "#mc-models")
    def _model_row_selected(self, event: DataTable.RowSelected) -> None:
        model_id = event.row_key.value
        if not model_id:
            return
        self._selected_model = model_id
        provider = self._provider()
        if provider is None:
            return
        entry = next((m for m in provider.models if m.id == model_id), None)
        if entry is not None:
            self.query_one("#mc-ctx", Input).value = str(entry.max_context_tokens)
            pct = entry.compaction_threshold
            self.query_one("#mc-pct", Input).value = "" if pct is None else str(pct)
            self.query_one("#mc-ctx", Input).focus()

    def _current_model_entry(self) -> ModelEntry | None:
        provider = self._provider()
        model_id = self._selected_model
        if provider is None or model_id is None:
            return None
        return next((m for m in provider.models if m.id == model_id), None)

    @on(Button.Pressed, "#mc-ctx-set")
    def _set_context(self) -> None:
        entry = self._current_model_entry()
        if entry is None:
            self._status("Select a model row first (Enter or click).")
            return
        raw = self.query_one("#mc-ctx", Input).value.strip()
        try:
            value = int(raw)
        except ValueError:
            self._status("✗ context tokens must be a positive integer")
            return
        if value <= 0:
            self._status("✗ context tokens must be a positive integer")
            return
        entry.max_context_tokens = value
        # A hand-entered value is by definition confirmed.
        entry.context_reported = True
        self._refresh_models()
        self._status(f"✓ {entry.id}: context set to {value}")

    @on(Button.Pressed, "#mc-pct-set")
    def _set_compaction_threshold(self) -> None:
        self._apply_compaction_threshold()

    @on(Input.Submitted, "#mc-pct")
    def _compaction_threshold_submitted(self) -> None:
        self._apply_compaction_threshold()

    def _apply_compaction_threshold(self) -> None:
        entry = self._current_model_entry()
        if entry is None:
            self._status("Select a model row first (Enter or click).")
            return
        raw = self.query_one("#mc-pct", Input).value.strip()
        if not raw:
            # Empty = inherit: drop the per-model override.
            entry.compaction_threshold = None
            self._refresh_models()
            self._status(f"✓ {entry.id}: compaction limit cleared (inherits)")
            return
        try:
            value = int(raw)
        except ValueError:
            self._status("✗ compact % must be an integer between 1 and 100")
            return
        if not 1 <= value <= 100:
            self._status("✗ compact % must be an integer between 1 and 100")
            return
        entry.compaction_threshold = value
        self._refresh_models()
        self._status(f"✓ {entry.id}: compaction at {value}% of context")

    @on(Button.Pressed, "#mc-model-del")
    def _remove_model(self) -> None:
        provider = self._provider()
        entry = self._current_model_entry()
        if provider is None or entry is None:
            self._status("Select a model row first (Enter or click).")
            return
        provider.models = [m for m in provider.models if m.id != entry.id]
        self._refresh_models()
        self._refresh_roles()
        self._status(f"✓ removed {entry.id}")

    @on(Button.Pressed, "#mc-model-add")
    def _add_model(self) -> None:
        if not self._commit_form():
            return
        provider = self._provider()
        if provider is None:
            return
        model_id = self.query_one("#mc-new-model", Input).value.strip()
        if not model_id:
            self._status("Enter a model id to add.")
            return
        if any(m.id == model_id for m in provider.models):
            self._status(f"✗ model '{model_id}' already exists")
            return
        provider.models.append(ModelEntry(id=model_id))
        self._refresh_models()
        self._refresh_providers()
        self._refresh_roles()
        self.query_one("#mc-new-model", Input).value = ""
        self._status(f"✓ added {model_id}")

    # -- role defaults & save -------------------------------------------------------

    @on(Button.Pressed, "#mc-save")
    def _save(self) -> None:
        if not self._commit_form():
            return
        for role in _ROLES:
            select = self.query_one(f"#mc-role-{role.value}", Select)
            value = str(select.value or "")
            if value:
                provider_name, _, model_id = value.partition("/")
                ref = ModelRef(provider=provider_name, model=model_id)
                self.catalog.defaults[role] = ref
            else:
                self.catalog.defaults.pop(role, None)
        try:
            self.store.save(self.catalog)
        except OSError as e:
            self._status(f"✗ could not write {USER_MODELS_FILENAME}: {e}")
            return
        self._status(f"✓ saved to {self.store.path}")


class ModelCatalogApp(App[None]):
    """Standalone `drcode-models` utility: just the catalog screen."""

    TITLE = "drcode models"

    def on_mount(self) -> None:
        # The catalog screen is the whole UI here: dismissing it (esc) is
        # the same as quitting — without this the app would linger on an
        # empty black screen, forcing the user to guess for ctrl+q.
        self.push_screen(ModelCatalogScreen(), callback=lambda _: self.exit())


def main() -> None:
    ModelCatalogApp().run()


if __name__ == "__main__":  # pragma: no cover
    main()
