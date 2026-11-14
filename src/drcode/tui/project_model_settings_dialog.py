"""Project model settings dialog — the single modal form for the
per-role model selection and the per-role max_tokens overrides.

Opened from the "Model settings…" sub-menu's single "Project model
settings…" entry (it replaces the old per-role picker entries of the
sub-menu; the underlying per-role picker machinery
(``role_model_options_provider`` + ``action_change_*_model``) stays for
the actions layer and tests). The dialog mirrors the Task-settings
dialog:

- works on a DRAFT: model slots, per-role custom models and the
  per-role ``max_tokens`` overrides are persisted only on Apply (ESC
  dismisses with None and discards everything);
- one section per agent role (coding / research / tester / compaction /
  resolver / subagents) with a model Select (catalog models, the role's saved custom
  model as "Custom: <model-id>", "Custom…" to enter a free-form model
  via the existing ModelSetupDialog, and "User default" to drop the
  project slot — the reset is available for EVERY role, including those
  the old palette offered no reset for; for the SUBAGENTS section the
  reset reads "Inherit parent role" — an empty slot means the
  `delegate_research` subagents run on the parent task's role model).
  The user-level default model of
  a role is marked by a single ``▸`` glyph and a bold name in the row's
  marker gutter; the currently selected model carries no marker at all
  (the Select's own field shows it) — the old " (default)"/" (current)"
  suffixes made the narrow rows wrap;
- an "max_tokens" Input per role: empty = inherit (the catalog entry's
  cap or the flat-config default — the override key is removed), a
  value = a positive-integer override that beats the more general
  layers (``providers.resolve`` applies it last). A live hint warns
  when the value exceeds the role model's context window, and a static
  note explains why thinking models commonly need a raised budget;
- both widgets are tinted orange ("changed") as soon as their draft
  value differs from the persisted one — the same marker the
  Task-settings dialog uses;
- COMPACTION-only extras: a "request timeout" Input (seconds; empty =
  the OpenAI SDK default 600 s — a summarizer call prefills a
  multi-hundred-k-token transcript and may need a longer read window)
  and a Switch "split a giant summarizer prompt" (the
  ``compaction_split_giant_prompt`` project setting: when ON, a
  single-shot summarizer prompt past ~256k tokens is cut into ~equal
  map-reduce slices instead of one giant call). The underlying
  ``role_request_timeout`` plumbing is role-generic, the dialog edits
  the compaction slot only — a summarizer is the role whose requests
  legitimately outlast a coding turn.

On Apply the draft is handed back to the app layer
(``ModelSelectionLayer._project_model_settings_done``), which persists
the new custom models (``set_project_custom_model``), the slots
(``set_project_model``) and the settings overrides, and replays the
same invalidation side effects as the old pickers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, Switch

from ..providers import (
    AgentRole,
    CustomModelEntry,
    ModelRef,
    UserModelsStore,
    load_project_custom_models,
    load_project_model_slots,
)
from ..settings import ProjectSettingsStore
from .model_setup_app import _FAVORITE_GLYPHS
from .setup_dialog import ModelSetup, ModelSetupDialog

if TYPE_CHECKING:
    from ..config import Config

# Sentinel values of the model Select. "Custom…" is transient: picking it
# only opens the setup dialog, the select bounces back to the draft slot
# until a setup is returned. ModelRef instances and the "custom" string
# are real option values (mirroring the project slot forms).
_CUSTOM_NEW = "__custom__"
_USER_DEFAULT = "__user_default__"

# One-character marker of the user-level default model — it replaced the
# " (default)" suffix (long catalog names wrapped in the narrow Select
# rows). It sits in the two-column gutter every catalog row starts with,
# so no row grows wider than before.
DEFAULT_MARKER = "▸"

_ROLE_TITLES: dict[AgentRole, str] = {
    AgentRole.CODING: "Coding",
    AgentRole.RESEARCH: "Research",
    AgentRole.TESTER: "Tester",
    AgentRole.COMPACTION: "Compaction",
    AgentRole.RESOLVER: "Resolver",
    AgentRole.SUBAGENT: "Subagents (delegate_research)",
}

THINKING_HINT = (
    "Thinking models spend their reasoning tokens from the same output "
    "budget — for the tester/resolver roles set it noticeably higher "
    "(16–32k)."
)


@dataclass
class ProjectModelSettingsResult:
    """The dialog's accepted draft: the per-role model slots (a ModelRef,
    "custom", or None = "User default"), the per-role max_tokens
    overrides (only the roles with a value) and the NEW/changed custom
    model entries to persist."""

    slots: dict[AgentRole, ModelRef | str | None] = field(default_factory=dict)
    role_max_tokens: dict[AgentRole, int] = field(default_factory=dict)
    # The compaction role's request-timeout override (seconds; only this
    # role is editable in the dialog, so only its key is ever present) and
    # the giant-prompt split toggle — both applied by the caller.
    role_request_timeout: dict[AgentRole, int] = field(default_factory=dict)
    split_giant_prompt: bool = False
    customs: dict[AgentRole, CustomModelEntry] = field(default_factory=dict)


def custom_entry_from_setup(setup: ModelSetup) -> CustomModelEntry:
    """The setup-dialog result as a per-role custom model entry (a None
    cap from the dialog means "keep the default"/"no cap"). Mirrors
    ``ModelSelectionLayer.custom_entry_from_setup`` (kept here so the
    dialog does not import the actions layer)."""
    return CustomModelEntry(
        model=setup.model,
        base_url=setup.base_url,
        api_key=setup.api_key,
        health_check_url=setup.health_check_url,
        max_context_tokens=setup.max_context_tokens or 200_000,
        max_tokens=setup.max_tokens,
    )


class ProjectModelSettingsDialog(ModalScreen[ProjectModelSettingsResult | None]):
    """Modal form: one section per role (model Select + max_tokens
    Input), live validation hints, Apply / ESC-to-cancel. Returns the
    accepted draft or None on cancel."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Close", priority=True),
    ]

    DEFAULT_CSS = """
    ProjectModelSettingsDialog {
        align: center middle;
    }
    #pms-dialog {
        width: 88;
        max-width: 95%;
        height: auto;
        max-height: 92%;
        padding: 0 2 1 2;
        background: $panel;
        border: thick $accent;
    }
    #pms-title {
        color: $text;
        text-style: bold;
        margin-bottom: 1;
    }
    #pms-body {
        height: auto;
        max-height: 30;
        overflow-y: auto;
    }
    .pms-section {
        color: $text;
        text-style: bold;
        margin-top: 1;
        background: $panel;
    }
    .pms-row {
        height: auto;
    }
    .pms-model-col {
        width: 2fr;
        height: auto;
    }
    .pms-tokens-col {
        width: 1fr;
        height: auto;
        margin-left: 1;
    }
    .pms-timeout-col {
        width: 1fr;
        height: auto;
        margin-left: 1;
    }
    .pms-hint-col {
        width: 1fr;
        height: auto;
        margin-left: 1;
    }
    #pms-split-giant {
        height: auto;
    }
    #pms-split-giant-label {
        color: $text-muted;
        margin-top: 1;
    }
    .pms-label {
        width: auto;
        color: $text-muted;
        margin-top: 0; /* compact widgets need no border-spacing */
        height: 1;
    }
    Select {
        width: 1fr;
    }
    # "Changed" marker (a field differing from the saved value), same
    # orange tint as the Task-settings dialog: a border-tint is useless
    # under the compact form styles (the border the signal used to color
    # is gone), and a tint set on the Select itself is hidden behind the
    # opaque SelectCurrent — it goes on that child, plus $warning text for
    # terminals without truecolor. Inputs get the same treatment (they had
    # no marker at all here, unlike the Task-settings dialog's Switch).
    Select.changed > SelectCurrent {
        background-tint: $warning 25%;
        color: $warning;
    }
    Input.changed {
        background-tint: $warning 25%;
    }
    .pms-hint {
        height: auto;
        color: $text-disabled;
    }
    .pms-hint.warning {
        color: $warning;
    }
    .pms-hint.error {
        color: $error;
        text-style: bold;
    }
    #pms-thinking-hint {
        height: auto;
        margin-top: 1;
        color: $text-muted;
    }
    #pms-buttons {
        dock: bottom;
        height: auto;
        margin-top: 1;
    }
    #pms-apply {
        width: auto;
        margin-right: 1;
    }
    #pms-cancel {
        width: auto;
    }
    #pms-esc-note {
        width: 1fr;
        content-align: right middle;
        color: $text-disabled;
    }
    """

    def __init__(
        self,
        project_root: Path,
        resolved: dict[AgentRole, Config],
    ) -> None:
        super().__init__()
        self._project_root = project_root
        # The per-role effective configs (from providers.resolve_effective):
        # placeholders of the max_tokens inputs and the context-size
        # comparison of the over-context live hint.
        self._resolved = dict(resolved)
        # Draft state (see the module docstring): the per-role slot
        # (ModelRef / "custom" / None), the role's custom entry (saved or
        # freshly entered — persisted on Apply only) and the per-role
        # max_tokens override (None = inherit).
        self._slots: dict[AgentRole, ModelRef | str | None] = {
            role: load_project_model_slots(project_root).get(role) for role in AgentRole
        }
        self._customs: dict[AgentRole, CustomModelEntry] = load_project_custom_models(
            project_root
        )
        # The saved overrides prefill the inputs: an untouched field keeps
        # its override on Apply, an emptied one drops it (= inherit).
        saved_settings = ProjectSettingsStore(project_root).load()
        saved_overrides = saved_settings.role_max_tokens
        self._saved_overrides = dict(saved_overrides)
        self._max_tokens: dict[AgentRole, int | None] = {
            role: saved_overrides.get(role.value) for role in AgentRole
        }
        # The compaction extras (see the docstring): the per-role request
        # timeout (only the compaction slot is editable) and the
        # giant-prompt split toggle. Same untouched/emptied rule as the
        # token caps.
        saved_timeouts = saved_settings.role_request_timeout
        self._saved_timeouts = dict(saved_timeouts)
        self._timeouts: dict[AgentRole, int | None] = {
            role: saved_timeouts.get(role.value) for role in AgentRole
        }
        self._split_giant = saved_settings.compaction_split_giant_prompt
        self._saved_split_giant = self._split_giant

    # -- draft ---------------------------------------------------------------------

    def _model_options(self, role: AgentRole) -> list[tuple[Text | str, object]]:
        """The model Select options of one role.

        Catalog models (as "provider/model" prompts — Select values must be
        hashable and are the plain "provider/model" strings) are sorted
        favorites-first — level 1, then 2, then 3, then the non-favorites,
        the original order kept within each group. A row is built as a
        compact marker gutter followed by the name:

        - a favorite carries its tier glyph, same shapes and colors as the
          catalog table's Favorites column (``★ ``), non-favorites keep the
          same two-space indent so the names align;
        - the user-level default model of the role is marked with a single
          ``▸`` in the gutter and a BOLD name (the old " (default)" suffix
          made the narrow rows wrap); a favorite default gets both marks
          (``▸ ★ name``). No word is ever appended to a name.

        The effective slot is deliberately NOT marked: the Select's own
        field already shows the selected value (the old " (current)" suffix
        was pure noise). Then the saved/draft custom model, "Custom…" and
        the reset ("User default" — for the SUBAGENT pseudo-role
        "Inherit parent role"; always present)."""
        catalog = UserModelsStore().load()
        user_default = catalog.default_ref(role)
        slot = self._slots[role]
        options: list[tuple[Text | str, object]] = []
        # (tier rank, "provider/model", is user default) — 4 = not a
        # favorite, sorts after the 1..3 tiers.
        catalog_options: list[tuple[int, str, bool]] = []
        for provider in catalog.providers:
            for entry in provider.models:
                ref = f"{provider.name}/{entry.id}"
                is_default = user_default is not None and ref == str(user_default)
                catalog_options.append((entry.favorite or 4, ref, is_default))
        catalog_options.sort(key=lambda item: item[0])
        for favorite, ref, is_default in catalog_options:
            # Fixed two-column gutter before the name: the default marker
            # column ("▸ " or blank) and the favorite tier column ("★ " or
            # blank), so every name starts at the same offset — marking a
            # default widens nothing but the marker itself.
            prompt = Text(f"{DEFAULT_MARKER} " if is_default else "  ")
            glyph = _FAVORITE_GLYPHS.get(favorite)
            if glyph is None:
                prompt.append("  ")
            else:
                # ONLY the glyph carries the tier color (a span, not the
                # base style — same rule as the catalog table and the
                # chains dialog).
                prompt.append(f"{glyph.plain} ", style=glyph.style)
            # The default's name is bold on top of the arrow marker.
            prompt.append(ref, style="bold" if is_default else None)
            options.append((prompt, ref))
        if isinstance(slot, ModelRef) and str(slot) not in {
            value for _, value in options
        }:
            # A catalog reference whose provider/model is missing from THIS
            # machine's user catalog (project settings committed to git and
            # cloned elsewhere): without an option carrying the slot value
            # the Select constructor raises InvalidSelectValueError. Keep
            # the draft slot selectable instead of crashing.
            options.append((f"{slot} (not in user catalog)", str(slot)))
        custom = self._customs.get(role)
        if custom is not None:
            options.append((f"Custom: {custom.model}", "custom"))
        elif slot == "custom":
            # A "custom" slot whose saved custom_models[role] entry is
            # missing/invalid: same missing-value hazard as above.
            options.append(("Custom: <no saved model>", "custom"))
        options.append(("Custom…", _CUSTOM_NEW))
        # The reset's label: for the SUBAGENT pseudo-role an empty slot
        # means "run on the parent task's role model", not "use the
        # user-level default" (the subagent slot has no user-level
        # default chain — see providers.resolve_subagent).
        options.append(
            (
                "Inherit parent role" if role is AgentRole.SUBAGENT else "User default",
                _USER_DEFAULT,
            )
        )
        return options

    def _slot_value(self, role: AgentRole) -> object:
        """The Select value for the draft slot (None = the "User
        default" sentinel; a ModelRef becomes its "provider/model"
        string; "custom" stays itself)."""
        slot = self._slots[role]
        if slot is None:
            return _USER_DEFAULT
        return slot if isinstance(slot, str) else str(slot)

    def _select(self, role: AgentRole) -> Select:
        return self.query_one(f"#pms-model-{role.value}", Select)

    def _tokens_input(self, role: AgentRole) -> Input:
        return self.query_one(f"#pms-tokens-{role.value}", Input)

    def _hint(self, role: AgentRole) -> Static:
        return self.query_one(f"#pms-hint-{role.value}", Static)

    def _role_of_widget_id(
        self, widget_id: str | None, prefix: str
    ) -> AgentRole | None:
        if not widget_id or not widget_id.startswith(prefix):
            return None
        tail = widget_id[len(prefix) :]
        return next((r for r in AgentRole if r.value == tail), None)

    # -- composition -----------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="pms-dialog"):
            yield Static("Project model settings", id="pms-title")
            with VerticalScroll(id="pms-body"):
                for role in AgentRole:
                    yield Label(_ROLE_TITLES[role], classes="pms-section")
                    with Horizontal(classes="pms-row"):
                        with Vertical(classes="pms-model-col"):
                            yield Label("Model", classes="pms-label")
                            yield Select(
                                self._model_options(role),
                                value=self._slot_value(role),
                                allow_blank=False,
                                id=f"pms-model-{role.value}",
                            )
                        with Vertical(classes="pms-tokens-col"):
                            yield Label(
                                "max_tokens (empty = inherit)",
                                classes="pms-label",
                            )
                            yield Input(
                                placeholder=self._placeholder(role),
                                value=self._input_value(role),
                                id=f"pms-tokens-{role.value}",
                            )
                    if role is AgentRole.COMPACTION:
                        # Compaction-only extras (see the module
                        # docstring): the request-timeout override and the
                        # giant-prompt split toggle — both live in
                        # ProjectSettings, not in the model catalog.
                        with Horizontal(classes="pms-row"):
                            with Vertical(classes="pms-timeout-col"):
                                yield Label(
                                    "request timeout, s (empty = SDK default 600)",
                                    classes="pms-label",
                                )
                                yield Input(
                                    placeholder="600",
                                    value=self._timeout_value(role),
                                    id="pms-timeout-compaction",
                                )
                            with Vertical(classes="pms-hint-col"):
                                yield Label("", classes="pms-label")
                                yield Static(
                                    "",
                                    classes="pms-hint",
                                    id="pms-timeout-hint",
                                )
                        yield Label(
                            "Split a giant summarizer prompt — a single-shot "
                            "compaction call past ~256k tokens is cut into "
                            "~equal map-reduce slices instead of one huge "
                            "prefill (off: one call while it fits the "
                            "window)",
                            id="pms-split-giant-label",
                        )
                        yield Switch(
                            value=self._split_giant,
                            id="pms-split-giant",
                        )
                    yield Static("", classes="pms-hint", id=f"pms-hint-{role.value}")
            yield Static(THINKING_HINT, id="pms-thinking-hint")
            with Horizontal(id="pms-buttons"):
                yield Button("Apply", variant="primary", id="pms-apply")
                yield Button("Cancel", id="pms-cancel")
                yield Static("esc — discard all changes", id="pms-esc-note")

    def _placeholder(self, role: AgentRole) -> str:
        resolved = self._resolved.get(role)
        return str(resolved.max_tokens) if resolved is not None else ""

    def _input_value(self, role: AgentRole) -> str:
        value = self._max_tokens[role]
        return "" if value is None else str(value)

    def _timeout_input(self, role: AgentRole) -> Input:
        return self.query_one("#pms-timeout-compaction", Input)

    def _timeout_value(self, role: AgentRole) -> str:
        value = self._timeouts[role]
        return "" if value is None else str(value)

    def _split_switch(self) -> Switch:
        return self.query_one("#pms-split-giant", Switch)

    def on_mount(self) -> None:
        for role in AgentRole:
            self._refresh_role(role)

    # -- refresh -----------------------------------------------------------------------

    def _refresh_role(self, role: AgentRole) -> None:
        """Sync one role's select to the draft and render its live hint
        (invalid value / over-context warning / quiet)."""
        select = self._select(role)
        with self.prevent(Select.Changed):
            # set_options RESETS the selection and posts a Changed event —
            # prevented and the draft value re-applied explicitly (same
            # staleness rule as the Task-settings dialog).
            select.set_options(self._model_options(role))
            select.value = self._slot_value(role)
        original_slot = load_project_model_slots(self._project_root).get(role)
        select.set_class(self._slots[role] != original_slot, "changed")
        self._mark_tokens_changed(role)
        self._render_hint(role)

    def _mark_tokens_changed(self, role: AgentRole) -> None:
        """The max_tokens Input's "changed" class: an override differing
        from the saved one (set or cleared) is tinted, same marker as the
        model Select — it must be re-evaluated on every keystroke, not only
        with the model options. For the compaction role the request-timeout
        input and the split switch are marked the same way (and the
        timeout hint re-renders)."""
        self._tokens_input(role).set_class(
            self._max_tokens[role] != self._saved_overrides.get(role.value),
            "changed",
        )
        if role is AgentRole.COMPACTION:
            self._timeout_input(role).set_class(
                self._timeouts[role] != self._saved_timeouts.get(role.value),
                "changed",
            )
            self._split_switch().set_class(
                self._split_giant != self._saved_split_giant, "changed"
            )
            self._render_timeout_hint()

    def _render_hint(self, role: AgentRole) -> None:
        text = self._tokens_input(role).value.strip()
        value = self._max_tokens[role]
        resolved = self._resolved.get(role)
        context = resolved.max_context_tokens if resolved is not None else 200_000
        hint = self._hint(role)
        if not text:
            hint.remove_class("warning", "error")
            hint.update(
                f"empty = inherit ({self._placeholder(role)} — the effective value)"
            )
        elif value is None:
            hint.add_class("error")
            hint.remove_class("warning")
            hint.update("max_tokens must be a positive integer")
        elif value > context:
            hint.add_class("warning")
            hint.remove_class("error")
            hint.update(
                f"{value} exceeds the model's context window ({context}) "
                "— requests will likely be rejected"
            )
        else:
            hint.remove_class("warning", "error")
            hint.update(f"override: {value} (beats the catalog entry and the default)")

    # -- widgets -> draft ----------------------------------------------------------------

    @on(Select.Changed)
    def _model_changed(self, event: Select.Changed) -> None:
        role = self._role_of_widget_id(event.select.id, "pms-model-")
        if role is None:
            return
        if event.value == _CUSTOM_NEW:
            # Transient choice: bounce the select back to the draft slot
            # and open the existing free-form setup dialog (pre-filled
            # from the role's saved custom model). The result is held in
            # the draft and persisted on Apply only.
            with self.prevent(Select.Changed):
                event.select.value = self._slot_value(role)  # type: ignore[assignment]
            self._open_custom_setup(role)
            return
        value = event.value
        if value == _USER_DEFAULT:
            self._slots[role] = None
        elif value == "custom" or value == _CUSTOM_NEW:
            self._slots[role] = cast("ModelRef | str", value)
        elif isinstance(value, str) and "/" in value:
            # A "provider/model" catalog reference: parse back into a
            # ModelRef (the Select's value is the plain string — Select
            # values must be hashable, ModelRef is not).
            provider, _, model = value.partition("/")
            self._slots[role] = ModelRef(provider=provider, model=model)
        else:
            return
        self._refresh_role(role)

    def _open_custom_setup(self, role: AgentRole) -> None:
        custom = self._customs.get(role)
        self.app.push_screen(
            ModelSetupDialog(
                model=custom.model if custom else "",
                base_url=custom.base_url if custom else "",
                api_key=custom.api_key if custom else "",
                max_context_tokens=(custom.max_context_tokens if custom else 200_000),
                max_tokens=(
                    custom.max_tokens if custom and custom.max_tokens else 8192
                ),
            ),
            lambda setup, role=role: self._custom_setup_done(role, setup),
        )

    def _custom_setup_done(self, role: AgentRole, setup: ModelSetup | None) -> None:
        if setup is None:
            return  # cancelled in the setup dialog — draft untouched
        self._customs[role] = custom_entry_from_setup(setup)
        self._slots[role] = "custom"
        self._refresh_role(role)

    @on(Input.Changed)
    def _tokens_changed(self, event: Input.Changed) -> None:
        role = self._role_of_widget_id(event.input.id, "pms-tokens-")
        if role is not None:
            text = event.value.strip()
            if not text:
                self._max_tokens[role] = None  # empty = inherit
            elif text.isdigit() and int(text) > 0:
                self._max_tokens[role] = int(text)
            else:
                self._max_tokens[role] = None  # invalid: the hint says so
            self._mark_tokens_changed(role)
            self._render_hint(role)
            return
        role = self._role_of_widget_id(event.input.id, "pms-timeout-")
        if role is not None:
            text = event.value.strip()
            if not text:
                self._timeouts[role] = None  # empty = the SDK default
            elif text.isdigit() and int(text) > 0:
                self._timeouts[role] = int(text)
            else:
                self._timeouts[role] = None  # invalid: the hint says so
            self._render_timeout_hint()

    def _render_timeout_hint(self) -> None:
        """The live hint of the compaction request-timeout input (the
        same error/warning shape as the max_tokens hints)."""
        text = self._timeout_input(AgentRole.COMPACTION).value.strip()
        value = self._timeouts[AgentRole.COMPACTION]
        hint = self.query_one("#pms-timeout-hint", Static)
        if not text:
            hint.remove_class("warning", "error")
            hint.update(
                "empty = the OpenAI SDK default (600 s read/write) — "
                "set it higher when a big summarizer prefill dies at 600 s"
            )
        elif value is None:
            hint.add_class("error")
            hint.remove_class("warning")
            hint.update("request timeout must be a positive integer (seconds)")
        else:
            hint.remove_class("warning", "error")
            hint.update(
                f"override: {value} s (the compaction client's read/write "
                "timeout; other roles keep the SDK default)"
            )

    @on(Switch.Changed, "#pms-split-giant")
    def _split_giant_changed(self, event: Switch.Changed) -> None:
        self._split_giant = event.value
        self._split_switch().set_class(
            self._split_giant != self._saved_split_giant, "changed"
        )

    # -- apply / cancel --------------------------------------------------------------------

    def _first_invalid(self) -> str | None:
        """A description of the first invalid field (its override stays
        unset), or None. Checks every max_tokens input and the compaction
        request-timeout input."""
        for role in AgentRole:
            text = self._tokens_input(role).value.strip()
            if text and self._max_tokens[role] is None:
                return f"max_tokens for {role.value}"
        timeout_text = self._timeout_input(AgentRole.COMPACTION).value.strip()
        if timeout_text and self._timeouts[AgentRole.COMPACTION] is None:
            return "request timeout for compaction"
        return None

    @on(Button.Pressed, "#pms-apply")
    def _apply_pressed(self) -> None:
        invalid = self._first_invalid()
        if invalid is not None:
            self.notify(
                f"{invalid} must be a positive integer "
                "(max_tokens / seconds; empty = inherit)",
                title="Project model settings",
                severity="error",
                timeout=6,
            )
            return
        original_customs = load_project_custom_models(self._project_root)
        timeouts = {
            role: value for role, value in self._timeouts.items() if value is not None
        }
        self.dismiss(
            ProjectModelSettingsResult(
                slots=dict(self._slots),
                role_max_tokens={
                    role: value
                    for role, value in self._max_tokens.items()
                    if value is not None
                },
                role_request_timeout=timeouts,
                split_giant_prompt=self._split_giant,
                customs={
                    role: entry
                    for role, entry in self._customs.items()
                    if original_customs.get(role) != entry
                },
            )
        )

    @on(Button.Pressed, "#pms-cancel")
    def _cancel_pressed(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
