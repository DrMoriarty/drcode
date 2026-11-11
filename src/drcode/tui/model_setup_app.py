"""Model-catalog editor: providers and their models.

Edits the user-level catalog (`~/.config/drcode/models.json`, see
`drcode.providers`) — the only UI that writes that file, so no manual JSON
editing is ever needed. Two entry points share this one screen:

1. the main TUI's Settings palette ("Provider catalog…"), which pushes the
   screen modally;
2. a standalone debug app (`python -m drcode.tui.model_setup_app`, a tiny
   App pushing the screen — deliberately not installed as a command).

Per provider: name / base URL / API key editing with a live connection
check (`probe_model`), one-click model auto-discovery (`fetch_models`,
existing entries keep their manual edits — favorites included — and new
ones are appended) and an editable model table: selecting a row (single
click or arrow keys — no double click needed) fills the editor line below
it with the model's id, context size and compaction limit; "Update" writes
the edited values back to the selected model, "Add model" creates a new
model from the same fields, "Remove" deletes it. A click on a row's
Favorites cell cycles its favorite level (empty → ❶ → ★ → ✚ → empty);
favorites sort to the top of the table, tier by tier. Binding models to
agent roles does NOT happen here: it lives in its own dialog, opened from
this screen's "Assign agents…" button.

When models.json does not exist yet but a legacy flat config (user-level
drcode.json) or the project's `custom_models` describe models, the screen
starts pre-filled with providers seeded from that data
(`providers.load_seeded_catalog`); the file is only written when the user
explicitly saves.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, OptionList, Static
from textual.widgets.option_list import Option

from ..agent import ModelFetchError, fetch_models, probe_model
from ..providers import (
    USER_MODELS_FILENAME,
    ModelEntry,
    Provider,
    UserModelsConfig,
    UserModelsStore,
    load_seeded_catalog,
)

_PROBE_TIMEOUT = 15.0

# Fallback width of the "model id" table column (used before the first
# layout pass): long ids are middle-truncated so the context /
# compaction-limit columns stay on screen. After layout the column is
# re-fitted so the table fills the FULL width of the modal — the id
# column absorbs whatever space the fixed columns leave free.
_MODEL_ID_COL_WIDTH = 42
_MIN_ID_COL_WIDTH = 16
_MAX_ID_COL_WIDTH = 80
# Cells always reserved for the (potential) vertical scrollbar, so the
# columns never overflow into a horizontal scrollbar.
_SCROLLBAR_ALLOWANCE = 2

# The Favorites column sits FIRST in the model table (index 0) — the
# click target must be visible without scrolling a wide id column.
_FAVORITE_COLUMN_INDEX = 0
_FAVORITE_COL_WIDTH = 3

# Favorite level → cell glyph. Each level is a DIFFERENT shape AND color so
# the tiers are tellable apart at a glance: ❶ (filled circle, red),
# ★ (five-pointed star, yellow), ✚ (cross, cyan). 0 = not a favorite
# (empty cell).
_FAVORITE_GLYPHS: dict[int, Text] = {
    1: Text("❶", style="bold red"),
    2: Text("★", style="bold yellow"),
    3: Text("✚", style="bold cyan"),
}

# Display-order rank of a favorite level: tiers 1 → 2 → 3 first, then
# everything else; alphabetical inside each group (the pre-favorites
# display order).
_FAVORITE_TIER_RANK = {1: 0, 2: 1, 3: 2}


def _model_sort_key(entry: ModelEntry) -> tuple[int, str]:
    return (_FAVORITE_TIER_RANK.get(entry.favorite, 3), entry.id)


def _truncate_middle(text: str, width: int = _MODEL_ID_COL_WIDTH) -> str:
    """Shorten a long model id by cutting out its middle (keeping the
    beginning and the end), so a fixed-width column still hints at both
    the namespace and the actual model name."""
    if len(text) <= width:
        return text
    keep = width - 1  # one cell for the ellipsis
    head = (keep + 1) // 2
    tail = keep - head
    return f"{text[:head]}…{text[len(text) - tail :]}"


def _column_widths(entries: list[ModelEntry]) -> tuple[int, int]:
    """Rendered widths of the fixed table columns: the widest of the
    header label and the cell values ("context", "compact limit")."""
    ctx_w = max([len("context")] + [len(str(e.max_context_tokens)) for e in entries])
    pct_w = max(
        [len("compact limit")]
        + [
            len(
                "inherit"
                if e.compaction_threshold is None
                else f"{e.compaction_threshold}%"
            )
            for e in entries
        ]
    )
    return ctx_w, pct_w


class _ModelTable(DataTable):
    """Model table whose Favorites column is clickable: a click on a cell
    of that column cycles the row's favorite level instead of only
    selecting the row. Everything else behaves like a plain DataTable with
    a row cursor."""

    class FavoriteClicked(Message):
        """Posted when a Favorites cell is clicked."""

        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            super().__init__()

    class RefitRequested(Message):
        """Posted when the table's own size changed (first layout or a
        terminal resize) and the id column should be re-fitted to the
        available width."""

    def on_resize(self, event: events.Resize) -> None:
        # events.Resize does NOT bubble, so the screen cannot observe the
        # table's layout directly — translate it into a bubbling message.
        self.post_message(self.RefitRequested())

    async def _on_click(self, event: events.Click) -> None:
        meta = event.style.meta
        row = meta.get("row")
        if (
            isinstance(row, int)
            and row >= 0  # -1 is the header row
            and meta.get("column") == _FAVORITE_COLUMN_INDEX
            and 0 <= row < len(self.ordered_rows)
        ):
            self.move_cursor(row=row, scroll=False)
            self.post_message(
                self.FavoriteClicked(self.ordered_rows[row].key.value or "")
            )
            event.stop()
            return
        await super()._on_click(event)


class ModelCatalogScreen(ModalScreen[None]):
    """Provider/model catalog editor (modal inside the main TUI, or the
    whole UI of the standalone debug app)."""

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
    #mc-main {
        height: 1fr;
    }
    #mc-left { width: 34; margin-right: 2; height: auto; }
    #mc-providers { height: 6; margin-top: 1; }
    #mc-right { width: 1fr; height: auto; }
    /* Compact form widgets are 1 line tall (see DrCodeApp.CSS) — the old
       border-inclusive height:3 would pad every field with 2 blank rows. */
    #mc-name, #mc-url, #mc-key { height: 1; }
    /* Push the provider fields down so the Check connection / Fetch models
       row sits on ONE height with the Add / Delete row of the providers
       pane: that row's top = title (1) + list margin (1) + list (6) +
       button-row margin (1) = 3 blank rows above it, mirrored here. */
    /* Blank lines between the provider fields so they don't run together
       (the key row needs no bottom margin — the button row below already
       carries margin-top). NOTE: margins must be declared in ONE rule per
       widget — a later longhand rule resets the earlier one's sides. */
    #mc-name { margin-top: 3; margin-bottom: 1; }
    #mc-url { margin-bottom: 1; }
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
    /* The model table spans the full width of the modal (it lives outside
       the two-column block) and grows with the modal height. */
    #mc-models {
        width: 100%;
        height: 1fr;
        min-height: 8;
        margin-top: 1;
    }
    /* Blank line between the model table and the editor fields. */
    #mc-model-row { height: auto; margin-top: 1; }
    #mc-model-row Input { height: 1; }
    #mc-model-id { width: 1fr; }
    #mc-ctx, #mc-pct { width: 14; }
    #mc-assign-agents { margin-top: 1; }
    #mc-footer {
        height: auto;
        margin-top: 1;
        align-horizontal: right;
    }
    #mc-hint {
        width: 100%;
        padding: 1 2 0 2;
        color: $text-disabled;
    }
    """

    def __init__(
        self,
        store: UserModelsStore | None = None,
        project_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.store = store or UserModelsStore()
        self.project_root = project_root
        # Load the catalog, pre-filling it from the legacy user-level flat
        # config and the project's custom models when the catalog file does
        # not exist yet (one-time migration aid; in-memory only until the
        # user presses Save).
        self.catalog: UserModelsConfig
        self.seed_note: str | None
        self.catalog, self.seed_note = load_seeded_catalog(
            self.store, project_root=project_root
        )
        self._current: str | None = None
        self._selected_model: str | None = None
        # Current width of the fitted "model id" column (see _fit_id_width).
        self._model_id_width: int = _MODEL_ID_COL_WIDTH
        # Set while removing a model: the table rebuild queues a
        # RowHighlighted event for the next row, which must not refill the
        # editor (Remove always ends with empty inputs).
        self._suppress_highlight = False
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
            # The model table and its editor live OUTSIDE the two-column
            # block: the table spans the full width of the modal.
            yield _ModelTable(id="mc-models", cursor_type="row")
            with Horizontal(id="mc-model-row"):
                yield Input(id="mc-model-id", placeholder="model id")
                yield Input(id="mc-ctx", placeholder="context tokens")
                yield Input(id="mc-pct", placeholder="compact % (empty = inherit)")
            with Horizontal(classes="mc-btnrow"):
                yield Button("Update", id="mc-model-update")
                yield Button("Remove", id="mc-model-del")
                yield Button("Add model", id="mc-model-add")
            # Just below the model buttons, its own row.
            yield Button("Assign agents…", id="mc-assign-agents")
            yield Static(
                "esc — close without saving · model table: click or arrow "
                "keys select a row and fill the editor; a click on the "
                "Favorites cell cycles the favorite level (empty → ❶ → "
                "★ → ✚), favorites sort to the top; Update writes the "
                "edited id/context/limit back, Add model creates a new "
                "model from the same fields, Remove deletes it",
                id="mc-hint",
            )
            # The very last row: Save pinned to the modal's bottom-right
            # corner (kept out of the wrapping hint block for that).
            with Horizontal(id="mc-footer"):
                yield Button("Save", variant="primary", id="mc-save")

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
        if "/" in name:
            self._status("✗ provider name must not contain '/'")
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
            # Rename: keep the stored role-default references (edited in
            # the separate agent-binding dialog, not here) pointing at the
            # same provider.
            old = provider.name
            provider.name = name
            for ref in self.catalog.all_refs():
                if ref.provider == old:
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

    def _refresh_models(self, keep_row: str | None = None) -> None:
        table = self.query_one("#mc-models", _ModelTable)
        provider = self._provider()
        entries = sorted(provider.models, key=_model_sort_key) if provider else []
        ctx_w, pct_w = _column_widths(entries)
        id_w = self._fit_id_width(table, ctx_w, pct_w)
        self._model_id_width = id_w
        table.clear(columns=True)
        table.add_column("Favorites", width=_FAVORITE_COL_WIDTH)
        table.add_column("model id", width=id_w)
        table.add_columns("context", "compact limit")
        # Display order: favorite tiers 1 → 2 → 3 first, then the rest;
        # alphabetical inside each group. This is the VIEW order only —
        # the stored catalog order (discovery order) is never rewritten.
        for entry in entries:
            pct = entry.compaction_threshold
            table.add_row(
                _FAVORITE_GLYPHS.get(entry.favorite, Text("")),
                _truncate_middle(entry.id, id_w),
                str(entry.max_context_tokens),
                "inherit" if pct is None else f"{pct}%",
                key=entry.id,
            )
        if (
            keep_row is not None
            and provider is not None
            and any(m.id == keep_row for m in provider.models)
        ):
            # Keep the toggled row under the cursor: swallow the reset
            # highlight queued by the rebuild, then re-highlight the row
            # (its RowHighlighted refills the editor with its values).
            self._suppress_highlight = True
            table.move_cursor(row=table.get_row_index(keep_row), scroll=False)

    def _fit_id_width(self, table: _ModelTable, ctx_w: int, pct_w: int) -> int:
        """Width of the "model id" column: whatever the modal width leaves
        free after the fixed columns, so the model list fills the FULL
        width of the modal. Falls back to the fixed default before the
        first layout pass (when the table has no size yet)."""
        available = table.size.width
        if not available:
            return _MODEL_ID_COL_WIDTH
        return max(
            _MIN_ID_COL_WIDTH,
            min(
                available - _SCROLLBAR_ALLOWANCE - _FAVORITE_COL_WIDTH - ctx_w - pct_w,
                _MAX_ID_COL_WIDTH,
            ),
        )

    @on(_ModelTable.RefitRequested)
    def _refit_models(self) -> None:
        """The table was (re)laid out: re-fit the model id column so the
        model list keeps filling the full modal width."""
        table = self.query_one("#mc-models", _ModelTable)
        if not table.size.width:
            return
        provider = self._provider()
        ctx_w, pct_w = _column_widths(provider.models if provider else [])
        if self._fit_id_width(table, ctx_w, pct_w) != self._model_id_width:
            self._refresh_models(keep_row=self._selected_model)

    def _load_form(self, name: str | None) -> None:
        self._current = name
        self._selected_model = None
        provider = self._provider()
        self.query_one("#mc-name", Input).value = provider.name if provider else ""
        self.query_one("#mc-url", Input).value = provider.base_url if provider else ""
        self.query_one("#mc-key", Input).value = provider.api_key if provider else ""
        self._clear_model_inputs()
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
        # Drop the role-default references (edited in the separate
        # agent-binding dialog, not here) that pointed at the deleted
        # provider — never leave dangling refs in the catalog.
        for role in list(self.catalog.defaults):
            refs = self.catalog.defaults.get(role)
            if refs:
                self.catalog.defaults[role] = [
                    r for r in refs if r.provider != provider.name
                ]
        self._current = (
            self.catalog.providers[0].name if self.catalog.providers else None
        )
        self._refresh_providers()
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
            # Existing entries are NEVER replaced (only fresh ids are
            # appended), so the user's manual edits — favorite levels
            # included — survive a re-discovery untouched.
            existing = {m.id for m in provider.models}
            fresh = [e for e in entries if e.id not in existing]
            provider.models.extend(fresh)
            self._refresh_models()
            self._refresh_providers()
            self._status(f"✓ fetched {len(entries)} models ({len(fresh)} new)")
        finally:
            self._busy = False

    # -- model table -----------------------------------------------------------------

    def _select_model(self, model_id: str | None) -> None:
        """Mark a table row as selected and fill the editor line with the
        model's actual values (called on single click / arrow keys too,
        not only on the double-click Enter selection)."""
        if not model_id:
            return
        provider = self._provider()
        entry = (
            next((m for m in provider.models if m.id == model_id), None)
            if provider is not None
            else None
        )
        if entry is None:
            return
        self._selected_model = model_id
        self.query_one("#mc-model-id", Input).value = entry.id
        self.query_one("#mc-ctx", Input).value = str(entry.max_context_tokens)
        pct = entry.compaction_threshold
        self.query_one("#mc-pct", Input).value = "" if pct is None else str(pct)

    @on(DataTable.RowHighlighted, "#mc-models")
    def _model_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self._suppress_highlight:
            # Swallow the re-highlight queued by the Remove rebuild.
            self._suppress_highlight = False
            return
        self._select_model(event.row_key.value)

    @on(DataTable.RowSelected, "#mc-models")
    def _model_row_selected(self, event: DataTable.RowSelected) -> None:
        self._select_model(event.row_key.value)
        if self._current_model_entry() is not None:
            self.query_one("#mc-model-id", Input).focus()

    @on(_ModelTable.FavoriteClicked)
    def _favorite_clicked(self, event: _ModelTable.FavoriteClicked) -> None:
        self._toggle_favorite(event.model_id)

    def _toggle_favorite(self, model_id: str) -> None:
        """Cycle the model's favorite level: 0 → 1 → 2 → 3 → 0, then
        re-render the table (favorites re-sort to the top)."""
        provider = self._provider()
        entry = (
            next((m for m in provider.models if m.id == model_id), None)
            if provider is not None
            else None
        )
        if entry is None:
            return
        entry.favorite = (entry.favorite + 1) % 4
        self._refresh_models(keep_row=entry.id)
        level = entry.favorite
        note = "not a favorite" if level == 0 else f"favorite level {level}"
        self._status(f"✓ {entry.id}: {note}")

    def _current_model_entry(self) -> ModelEntry | None:
        provider = self._provider()
        model_id = self._selected_model
        if provider is None or model_id is None:
            return None
        return next((m for m in provider.models if m.id == model_id), None)

    def _clear_model_inputs(self) -> None:
        for input_id in ("#mc-model-id", "#mc-ctx", "#mc-pct"):
            self.query_one(input_id, Input).value = ""

    @on(Button.Pressed, "#mc-model-update")
    def _update_clicked(self) -> None:
        self._update_model()

    @on(Input.Submitted, "#mc-model-id")
    @on(Input.Submitted, "#mc-ctx")
    @on(Input.Submitted, "#mc-pct")
    def _model_input_submitted(self) -> None:
        self._update_model()

    def _parse_caps(self) -> tuple[int, int | None] | None:
        """Parse the context / compaction-limit editor fields into
        (context_tokens, compaction_threshold); None on a validation error
        (the status line explains)."""
        raw_ctx = self.query_one("#mc-ctx", Input).value.strip()
        try:
            ctx = int(raw_ctx)
        except ValueError:
            self._status("✗ context tokens must be a positive integer")
            return None
        if ctx <= 0:
            self._status("✗ context tokens must be a positive integer")
            return None
        raw_pct = self.query_one("#mc-pct", Input).value.strip()
        pct: int | None = None
        if raw_pct:
            try:
                pct = int(raw_pct)
            except ValueError:
                self._status("✗ compact % must be an integer between 1 and 100")
                return None
            if not 1 <= pct <= 100:
                self._status("✗ compact % must be an integer between 1 and 100")
                return None
        return ctx, pct

    def _update_model(self) -> None:
        """Write the edited id / context / compaction limit back into the
        selected model."""
        entry = self._current_model_entry()
        if entry is None:
            self._status("Select a model row first (click or arrow keys).")
            return
        provider = self._provider()
        assert provider is not None
        new_id = self.query_one("#mc-model-id", Input).value.strip() or entry.id
        caps = self._parse_caps()
        if caps is None:
            return
        ctx, pct = caps
        if new_id != entry.id:
            if any(m.id == new_id for m in provider.models):
                self._status(f"✗ model '{new_id}' already exists")
                return
            # Rename: keep the role-default references pointing at the
            # same model.
            old = entry.id
            entry.id = new_id
            for ref in self.catalog.all_refs():
                if ref.provider == provider.name and ref.model == old:
                    ref.model = new_id
            self._selected_model = new_id
        if ctx != entry.max_context_tokens:
            entry.max_context_tokens = ctx
        entry.compaction_threshold = pct
        self._refresh_models()
        limit = f"{pct}% of context" if pct is not None else "inherits"
        self._status(f"✓ {entry.id}: context {ctx}, compaction {limit}")

    @on(Button.Pressed, "#mc-model-del")
    def _remove_model(self) -> None:
        provider = self._provider()
        entry = self._current_model_entry()
        if provider is None or entry is None:
            self._status("Select a model row first (click or arrow keys).")
            return
        provider.models = [m for m in provider.models if m.id != entry.id]
        self._refresh_models()
        self._selected_model = None
        # The rebuild queues a RowHighlighted for the next row — swallow
        # it, so Remove always ends with empty editor inputs.
        self._suppress_highlight = True
        self._clear_model_inputs()
        self._status(f"✓ removed {entry.id}")

    @on(Button.Pressed, "#mc-model-add")
    def _add_model(self) -> None:
        """Create a new model from the editor fields (id, context, limit) —
        the same inputs the row selection fills for editing."""
        if not self._commit_form():
            return
        provider = self._provider()
        if provider is None:
            return
        model_id = self.query_one("#mc-model-id", Input).value.strip()
        if not model_id:
            self._status("Enter a model id to add.")
            return
        if any(m.id == model_id for m in provider.models):
            self._status(f"✗ model '{model_id}' already exists")
            return
        caps = self._parse_caps()
        if caps is None:
            return
        ctx, pct = caps
        provider.models.append(
            ModelEntry(
                id=model_id,
                max_context_tokens=ctx,
                compaction_threshold=pct,
            )
        )
        self._selected_model = model_id
        self._refresh_models()
        self._refresh_providers()
        limit = f"{pct}% of context" if pct is not None else "inherits"
        self._status(f"✓ added {model_id}: context {ctx}, compaction {limit}")

    # -- agent binding & save ---------------------------------------------------------

    @on(Button.Pressed, "#mc-assign-agents")
    def _open_agent_binding(self) -> None:
        """Open the agent model chains dialog (its own modal screen): the
        ordered per-role fallback lists of the user catalog. The dialog
        edits THIS screen's in-memory catalog object, so its unsaved
        provider edits stay authoritative; its own Save persists both."""
        from .agent_chains_dialog import AgentChainsScreen

        if not self._commit_form():
            return
        self.app.push_screen(
            AgentChainsScreen(store=self.store, catalog=self.catalog),
            self._agent_chains_done,
        )

    def _agent_chains_done(self, _result: object) -> None:
        self._status(
            f"ℹ agent chains saved to {self.store.path}"
            if _result
            else "agent chains dialog closed without saving"
        )

    @on(Button.Pressed, "#mc-save")
    def _save(self) -> None:
        if not self._commit_form():
            return
        try:
            self.store.save(self.catalog)
        except OSError as e:
            self._status(f"✗ could not write {USER_MODELS_FILENAME}: {e}")
            return
        self._status(f"✓ saved to {self.store.path}")


class ModelCatalogApp(App[None]):
    """Standalone debug app: just the catalog screen."""

    TITLE = "drcode models"

    # Mirror the main TUI's compact form geometry (DrCodeApp.CSS): without
    # it the borderless 1-line Buttons/Inputs the screen's CSS assumes turn
    # back into 3-line bordered widgets and the button rows misalign.
    CSS = """
    Button, Input, Switch, Checkbox, RadioButton, RadioSet, OptionList,
    Select > SelectCurrent {
        border: none !important;
    }
    Input {
        height: 1;
        padding: 0 1;
    }
    OptionList {
        padding: 0;
    }
    """

    def on_mount(self) -> None:
        # The catalog screen is the whole UI here: dismissing it (esc) is
        # the same as quitting — without this the app would linger on an
        # empty black screen, forcing the user to guess for ctrl+q.
        self.push_screen(ModelCatalogScreen(), callback=lambda _: self.exit())


def main() -> None:
    ModelCatalogApp().run()


if __name__ == "__main__":  # pragma: no cover
    main()
