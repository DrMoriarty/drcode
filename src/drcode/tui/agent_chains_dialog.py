"""Agent model chains editor: the ordered per-role fallback lists.

Edits the user-level `defaults[role]` ordered chains of `models.json`
(see `drcode.providers`): up to `MAX_ROLE_DEFAULTS` catalog references
per agent role, in priority order — the first entry is the role's
PRIMARY model (the historical single default), the rest are availability
fallbacks tried in order (`resolve_available`). This dialog replaces the
single-slot role binding that used to live in the provider catalog
screen; it is opened from that screen's "Assign agents…" button and from
the "Model settings…" palette ("Agent model chains…").

Layout: RIGHT — every model of the whole catalog (all providers),
favorites highlighted with the same tier glyphs/colors as the catalog
table's Favorites column (the color on the GLYPH only — the model name
stays uncolored), labels middle-truncated to the list width so long ids
never wrap, with a case-insensitive substring filter above
(model and/or provider name). LEFT — one group per agent role, each
holding the role's selected references in chain order (long names
middle-truncated the same way; the primary is simply the first entry —
no extra marker). Order is the
insertion order on purpose: no internal re-sorting, reordering is done
by removing and re-adding (at most three entries).

Save writes the whole user catalog atomically through `UserModelsStore`
(the project's drcode.json is never touched here) and invalidates the
app's per-role resolution caches. No provider/availability editing
happens in this dialog — that stays in the catalog's provider editor.
"""

from __future__ import annotations

from typing import ClassVar, cast

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from ..providers import (
    MAX_ROLE_DEFAULTS,
    USER_MODELS_FILENAME,
    AgentRole,
    ModelEntry,
    ModelRef,
    Provider,
    UserModelsConfig,
    UserModelsStore,
)
from .model_setup_app import (
    _FAVORITE_GLYPHS,
    _model_sort_key,
    _truncate_middle,
)

# Cells reserved for the vertical scrollbar of an OptionList, so a label
# never wraps onto a second line (it is middle-truncated instead — same
# approach as the catalog table's model id column).
_SCROLLBAR_ALLOWANCE = 2
# Fallback label widths, used before the first layout pass (widget size
# is still 0): roughly the CSS width of the lists minus the prefix.
_FALLBACK_RIGHT_WIDTH = 50
_FALLBACK_GROUP_WIDTH = 42


class _FittingOptionList(OptionList):
    """OptionList that reports its own width changes to the screen:
    `events.Resize` does NOT bubble (see `_ModelTable` in
    `model_setup_app`), but the labels are middle-truncated to the list's
    current width, so a terminal resize must trigger a re-render."""

    class WidthChanged(Message):
        """Posted when the list's size changed and the labels it holds
        should be re-fitted."""

    def on_resize(self, event: events.Resize) -> None:
        self.post_message(self.WidthChanged())


def _label_width(option_list: OptionList, fallback: int) -> int:
    """Usable character width for one option line of `option_list`
    (minus the scrollbar allowance; `fallback` before the first layout)."""
    width = option_list.size.width
    if not width:
        return fallback
    return max(8, width - _SCROLLBAR_ALLOWANCE)


def _model_label(entry: ModelEntry, provider_name: str, name_width: int) -> Text:
    """One right-hand list line: the favorite tier glyph (same shapes and
    colors as the catalog table's Favorites column — единообразие with the
    catalog screen) followed by "provider/model". ONLY the glyph carries
    the tier color (a span, not the base style — otherwise Rich would
    paint the whole line); the name stays uncolored and is
    middle-truncated to `name_width` so long ids never wrap."""
    name = _truncate_middle(f"{provider_name}/{entry.id}", name_width)
    glyph = _FAVORITE_GLYPHS.get(entry.favorite)
    if glyph is None:
        return Text(f"  {name}")
    line = Text("  ")
    line.append(f"{glyph.plain} ", style=glyph.style)
    line.append(name)
    return line


class AgentChainsScreen(ModalScreen["bool | None"]):
    """Per-role ordered model chains editor (modal screen)."""

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Close", priority=True),
    ]

    DEFAULT_CSS = """
    AgentChainsScreen {
        align: center middle;
    }
    #ac-dialog {
        width: 110;
        max-width: 98%;
        height: 96%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    .ac-title {
        color: $accent;
        text-style: bold;
    }
    #ac-main {
        height: 1fr;
    }
    #ac-left {
        width: 48;
        max-width: 45%;
        margin-right: 2;
        height: 1fr;
        overflow-y: auto;
    }
    #ac-right {
        width: 1fr;
        height: 1fr;
    }
    .ac-group-head {
        height: auto;
        margin-top: 1;
    }
    .ac-role {
        width: 1fr;
        color: $accent;
        text-style: bold;
    }
    .ac-group-head Button {
        height: 1;
        min-width: 5;
        width: auto;
        padding: 0 1;
        margin-left: 1;
    }
    .ac-group-list {
        height: auto;
        max-height: 3;
        min-height: 1;
        scrollbar-size: 0 0;
    }
    #ac-filter {
        height: 1;
    }
    #ac-models {
        height: 1fr;
        margin-top: 1;
    }
    #ac-status {
        height: auto;
        min-height: 1;
        color: $text-muted;
    }
    #ac-footer {
        height: auto;
        margin-top: 1;
        align-horizontal: right;
    }
    #ac-hint {
        width: 1fr;
        padding: 1 2 1 2;
        color: $text-disabled;
    }
    """

    def __init__(
        self,
        store: UserModelsStore | None = None,
        catalog: UserModelsConfig | None = None,
    ) -> None:
        super().__init__()
        self.store = store or UserModelsStore()
        # `catalog` is the SHARED in-memory config when the dialog is opened
        # from the provider catalog screen (its unsaved provider edits stay
        # authoritative and its later Save persists the chains too); without
        # one, the dialog works on its own freshly loaded copy.
        self.catalog: UserModelsConfig
        if catalog is not None:
            self.catalog = catalog
        else:
            self.catalog = self.store.load()
        # Working copy of the chains (role → ordered refs), seeded from the
        # catalog; mutated live, flushed into `catalog.defaults` on Save.
        self.chains: dict[AgentRole, list[ModelRef]] = {
            role: self.catalog.role_refs(role) for role in AgentRole
        }
        self._filter_text: str = ""
        self._selected_ref: ModelRef | None = None

    # -- compose ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="ac-dialog"):
            yield Static(
                "Agent model chains — up to 3 models per agent role, in "
                "priority order (first = primary, the rest are availability "
                f"fallbacks). Stored in [b]{self.store.path.name}[/b] "
                "(user level).",
                classes="ac-title",
            )
            with Horizontal(id="ac-main"):
                with Vertical(id="ac-left"):
                    for role in AgentRole:
                        with Horizontal(classes="ac-group-head"):
                            yield Static(
                                f"{role.value} · 0/{MAX_ROLE_DEFAULTS}",
                                id=f"ac-role-{role.value}",
                                classes="ac-role",
                            )
                            yield Button(
                                "Add", id=f"ac-add-{role.value}", classes="ac-add"
                            )
                            yield Button(
                                "Del", id=f"ac-del-{role.value}", classes="ac-del"
                            )
                        yield _FittingOptionList(
                            id=f"ac-list-{role.value}", classes="ac-group-list"
                        )
                with Vertical(id="ac-right"):
                    yield Input(
                        id="ac-filter",
                        placeholder="filter models… (substring, matches model and/or provider name)",
                    )
                    yield _FittingOptionList(id="ac-models")
            yield Static("", id="ac-status")
            with Horizontal(id="ac-footer"):
                yield Static(
                    "esc — close without saving · pick a model on the right, "
                    "then its group's Add appends it to that agent's chain; "
                    "Del removes the highlighted entry of the group. No "
                    "reordering: remove and re-add to change the order.",
                    id="ac-hint",
                )
                yield Button("Cancel", id="ac-cancel")
                yield Button("Save", variant="primary", id="ac-save")

    # -- helpers ------------------------------------------------------------

    def _status(self, text: str) -> None:
        self.query_one("#ac-status", Static).update(text)

    def _all_models(self) -> list[tuple[Provider, ModelEntry]]:
        """Every catalog model as (provider, entry) pairs, in the catalog's
        favorite-tier display order (the same view order the catalog table
        uses — the stored order is never rewritten)."""
        pairs = [
            (provider, entry)
            for provider in self.catalog.providers
            for entry in provider.models
        ]
        return sorted(pairs, key=lambda pair: _model_sort_key(pair[1]))

    def _refresh_right(self) -> None:
        option_list = self.query_one("#ac-models", OptionList)
        # Remember the highlighted option (a refit-width rebuild — e.g. the
        # first layout pass or a terminal resize — must not silently drop
        # the user's picked model; a filtered-out one legitimately does).
        highlighted_id: str | None = None
        if option_list.highlighted is not None and option_list.highlighted < len(
            option_list.options
        ):
            highlighted_id = option_list.options[option_list.highlighted].id
        option_list.clear_options()
        needle = self._filter_text.casefold()
        # The glyph prefix ("❶ " / two spaces) occupies 2 cells before the
        # "provider/model" name.
        name_width = _label_width(option_list, _FALLBACK_RIGHT_WIDTH) - 2
        for provider, entry in self._all_models():
            if needle and needle not in f"{provider.name} {entry.id}".casefold():
                continue
            option_list.add_option(
                Option(
                    _model_label(entry, provider.name, name_width),
                    id=f"ref:{provider.name}/{entry.id}",
                )
            )
        # The rebuild drops the highlight — the picked model must not
        # survive invisibly (Add would use a stale, unlisted entry).
        self._selected_ref = None
        if highlighted_id is not None:
            for index, option in enumerate(option_list.options):
                if option.id == highlighted_id:
                    option_list.highlighted = index  # re-fires the handler
                    break

    def _refresh_group(self, role: AgentRole) -> None:
        refs = self.chains.get(role, [])
        option_list = self.query_one(f"#ac-list-{role.value}", OptionList)
        prev_highlight = option_list.highlighted
        option_list.clear_options()
        # "N. " occupies 3 cells before the "provider/model" name; the
        # name is middle-truncated so long ids never wrap.
        name_width = _label_width(option_list, _FALLBACK_GROUP_WIDTH) - 3
        for index, ref in enumerate(refs):
            name = _truncate_middle(str(ref), name_width)
            option_list.add_option(
                Option(f"{index + 1}. {name}", id=f"{role.value}:{index}")
            )
        if prev_highlight is not None and prev_highlight < len(refs):
            option_list.highlighted = prev_highlight
        self.query_one(f"#ac-role-{role.value}", Static).update(
            f"{role.value} · {len(refs)}/{MAX_ROLE_DEFAULTS}"
        )

    def _refresh_all_groups(self) -> None:
        for role in AgentRole:
            self._refresh_group(role)

    def _role_from_button(self, button_id: str | None, prefix: str) -> AgentRole | None:
        if not button_id or not button_id.startswith(prefix + "-"):
            return None
        role_value = button_id[len(prefix) + 1 :]
        return next((r for r in AgentRole if r.value == role_value), None)

    # -- lifecycle ------------------------------------------------------------

    def on_mount(self) -> None:
        if self.store.last_warning:
            self._status(f"⚠ {self.store.last_warning}")
        self._refresh_right()
        self._refresh_all_groups()
        self.query_one("#ac-filter", Input).focus()

    def action_close(self) -> None:
        self.dismiss(None)

    @on(_FittingOptionList.WidthChanged)
    def _lists_resized(self) -> None:
        """A list's width changed (e.g. terminal resize): re-fit the
        middle-truncated labels of every list."""
        self._refresh_right()
        self._refresh_all_groups()

    # -- right-hand model list --------------------------------------------------

    @on(Input.Changed, "#ac-filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        self._filter_text = event.value
        self._refresh_right()

    @on(OptionList.OptionHighlighted, "#ac-models")
    def _model_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        option_id = event.option.id or ""
        if not option_id.startswith("ref:"):
            self._selected_ref = None
            return
        raw = option_id[len("ref:") :]
        provider, _, model = raw.partition("/")
        self._selected_ref = ModelRef(provider=provider, model=model)

    # -- chain editing ------------------------------------------------------------

    @on(Button.Pressed, ".ac-add")
    def _add_clicked(self, event: Button.Pressed) -> None:
        role = self._role_from_button(event.button.id or "", "ac-add")
        if role is not None:
            self._add_to_role(role)

    def _add_to_role(self, role: AgentRole) -> None:
        ref = self._selected_ref
        if ref is None:
            self._status("Select a model on the right first (click or arrow keys).")
            return
        refs = self.chains.setdefault(role, [])
        if ref in refs:
            self._status(f"✗ {ref} is already in the {role.value} chain")
            return
        if len(refs) >= MAX_ROLE_DEFAULTS:
            self._status(
                f"✗ the {role.value} chain is full ({MAX_ROLE_DEFAULTS} models "
                "max) — remove one before adding another"
            )
            return
        refs.append(ref)
        self._refresh_group(role)
        position = len(refs)
        note = "primary" if position == 1 else f"fallback #{position}"
        self._status(f"✓ {ref} added to {role.value} as {note}")

    @on(Button.Pressed, ".ac-del")
    def _del_clicked(self, event: Button.Pressed) -> None:
        role = self._role_from_button(event.button.id or "", "ac-del")
        if role is not None:
            self._remove_from_role(role)

    def _remove_from_role(self, role: AgentRole) -> None:
        option_list = self.query_one(f"#ac-list-{role.value}", OptionList)
        index = option_list.highlighted
        refs = self.chains.get(role, [])
        if index is None or not 0 <= index < len(refs):
            self._status(f"Select an entry in the {role.value} list first.")
            return
        removed = refs.pop(index)
        self._refresh_group(role)
        self._status(f"✓ {removed} removed from {role.value}")

    # -- save -----------------------------------------------------------------

    def _flush_chains(self) -> None:
        """Write the working chains into the catalog object: empty chains
        are absence (the slot is dropped, matching the on-load normalizer)."""
        for role in AgentRole:
            refs = self.chains.get(role, [])
            if refs:
                self.catalog.defaults[role] = list(refs)
            else:
                self.catalog.defaults.pop(role, None)

    def _invalidate_app_caches(self) -> None:
        """Drop the app's per-role resolution caches after a save (same
        pattern as `_model_catalog_done` in actions_models.py). Duck-typed:
        the standalone debug app host has none of these methods."""
        app = self.app
        invalidate = getattr(app, "invalidate_resolved_configs", None)
        if not callable(invalidate):
            return
        before: dict[AgentRole, str] = {}
        snapshot = getattr(app, "_snapshot_model_keys", None)
        if callable(snapshot):
            before = cast("dict[AgentRole, str]", snapshot())
        invalidate()
        emit = getattr(app, "_emit_manual_model_change_events", None)
        if callable(emit):
            emit(before)
        refresh = getattr(app, "refresh_connection", None)
        if callable(refresh):
            refresh()
        status = getattr(app, "refresh_status", None)
        if callable(status):
            status()
        advance = getattr(app, "_advance", None)
        if callable(advance):
            advance()

    @on(Button.Pressed, "#ac-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#ac-save")
    def _save(self) -> None:
        self._flush_chains()
        try:
            self.store.save(self.catalog)
        except OSError as e:
            self._status(f"✗ could not write {USER_MODELS_FILENAME}: {e}")
            return
        self._invalidate_app_caches()
        self.dismiss(True)
