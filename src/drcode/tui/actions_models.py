"""Model-selection actions layer — per-role model picking.

The "Model settings…" sub-menu, the provider/model catalog screen, the
per-role pickers (catalog references and free-form custom models) and
the labels the Settings palette shows for each role.

Split out of :mod:`drcode.tui.actions` — see the composition in
``ActionsLayer``.
"""

from __future__ import annotations

from functools import partial

from textual.command import (
    CommandPalette,
)

from ..providers import (
    AgentRole,
    CustomModelEntry,
    ModelRef,
    UserModelsStore,
    load_project_custom_models,
    load_project_model_slots,
    set_project_custom_active,
    set_project_custom_model,
    set_project_model,
)
from .command_providers import (
    ModelCatalogCommandProvider,
    ModelSettingsPalette,
    ProjectModelSettingsCommandProvider,
    SettingsPalette,
    role_model_options_provider,
)
from .model_setup_app import ModelCatalogScreen
from .project_model_settings_dialog import (
    ProjectModelSettingsDialog,
    ProjectModelSettingsResult,
)
from .setup_dialog import ModelSetup, ModelSetupDialog
from .state import DrCodeState


@staticmethod
def custom_entry_from_setup(setup: ModelSetup) -> CustomModelEntry:
    """The dialog result as a per-role custom model entry (a None cap
    from the dialog means "keep the default"/"no cap")."""
    return CustomModelEntry(
        model=setup.model,
        base_url=setup.base_url,
        api_key=setup.api_key,
        health_check_url=setup.health_check_url,
        max_context_tokens=setup.max_context_tokens or 200_000,
        max_tokens=setup.max_tokens,
    )


class ModelSelectionLayer(DrCodeState):
    def action_open_model_settings(self) -> None:
        """Open the "Model settings…" sub-menu: the provider catalog
        editor and the single "Project model settings…" entry (the
        per-role model/max_tokens form). Runs as the callback of a
        palette entry — the palette has already dismissed itself by the
        time this executes, so pushing the sub-palette is safe."""
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            ModelSettingsPalette(
                id="--model-settings-palette",
                providers=[
                    ModelCatalogCommandProvider,
                    ProjectModelSettingsCommandProvider,
                ],
                placeholder="Model settings…",
            )
        )

    # -- project model settings dialog ----------------------------------------

    def action_open_project_model_settings(self) -> None:
        """Open the per-role "Project model settings…" modal form: model
        selection plus the max_tokens override for every agent role, in
        one dialog with atomic Apply. Runs as the callback of a palette
        entry — the palette has already dismissed itself by the time
        this executes, so pushing the dialog is safe."""
        if isinstance(self.screen, CommandPalette):
            return
        resolved = {role: self.resolved_config(role) for role in AgentRole}
        self.push_screen(
            ProjectModelSettingsDialog(self.project_root, resolved),
            self._project_model_settings_done,
        )

    def _project_model_settings_done(
        self, result: ProjectModelSettingsResult | None
    ) -> None:
        if result is None:
            return  # ESC / Cancel — nothing persisted.
        model_keys_before = self._snapshot_model_keys()
        try:
            # New/changed free-form custom models first: set_project_custom_
            # model stores the entry AND activates it ("custom" slot).
            for role, entry in result.customs.items():
                set_project_custom_model(self.project_root, role, entry)
            current_slots = load_project_model_slots(self.project_root)
            for role, slot in result.slots.items():
                if role in result.customs or slot == current_slots.get(role):
                    continue  # already activated / untouched
                if isinstance(slot, ModelRef):
                    set_project_model(self.project_root, role, slot)
                elif slot == "custom":
                    set_project_custom_active(self.project_root, role)
                else:
                    set_project_model(self.project_root, role, None)
        except OSError as e:
            self.notify(
                f"Could not write the project config: {e}",
                title="Project model settings",
                severity="error",
                timeout=8,
            )
            return
        # The per-role max_tokens overrides live in the project settings
        # (merged against the STORED file, not just the live object — the
        # live one may predate an external edit).
        stored = dict(self.settings_store.load().role_max_tokens)
        merged = dict(stored)
        for role in AgentRole:
            merged.pop(role.value, None)
        for role, value in result.role_max_tokens.items():
            merged[role.value] = value
        self.settings.role_max_tokens = merged
        if merged != stored:
            self.settings_store.save(self.settings)
        self.invalidate_resolved_configs()
        self._emit_manual_model_change_events(model_keys_before)
        self.refresh_connection()
        self.refresh_status()
        self._advance()
        overrides = (
            ", ".join(
                f"{role.value}={value}"
                for role, value in sorted(
                    result.role_max_tokens.items(), key=lambda kv: kv[0].value
                )
            )
            or "no max_tokens overrides"
        )
        self.notify(
            f"Project model settings applied ({overrides})",
            title="Project model settings",
            timeout=4,
        )

    # -- per-role model selection ---------------------------------------------

    def action_open_model_catalog(self) -> None:
        """Open the provider/model catalog editor (user-level models.json)
        as a modal screen. Also used by the standalone debug app; when no
        catalog exists yet, the screen pre-fills itself from this
        project's custom models (saved only on explicit Save).
        On close the per-role resolutions are recomputed."""
        if isinstance(self.screen, CommandPalette):
            return
        self.push_screen(
            ModelCatalogScreen(project_root=self.project_root),
            self._model_catalog_done,
        )

    def _model_catalog_done(self, _result: object) -> None:
        model_keys_before = self._snapshot_model_keys()
        self.invalidate_resolved_configs()
        self._emit_manual_model_change_events(model_keys_before)
        self.refresh_connection()

    def role_model_label(self, role: AgentRole) -> str:
        """Human-readable effective model of a role, for the Settings
        palette entries: the project slot (catalog reference or the role's
        custom model id), else the user-level default, else "not set".
        The research role falls back to the coding model when unpinned —
        the label says so instead of the bare "not set"."""
        slot = load_project_model_slots(self.project_root).get(role)
        if isinstance(slot, ModelRef):
            return f"{slot.provider}/{slot.model}"
        if slot == "custom":
            custom = load_project_custom_models(self.project_root).get(role)
            return custom.model if custom is not None else "not set"
        ref = UserModelsStore().load().default_ref(role)
        if ref is not None:
            return f"{ref.provider}/{ref.model}"
        if role in (AgentRole.RESEARCH, AgentRole.RESOLVER):
            return f"{self.role_model_label(AgentRole.CODING)} (inherits coding)"
        return "not set"

    def action_change_coding_model(self) -> None:
        self._open_role_model_picker(AgentRole.CODING)

    def action_change_research_model(self) -> None:
        self._open_role_model_picker(AgentRole.RESEARCH)

    def action_change_tester_model(self) -> None:
        self._open_role_model_picker(AgentRole.TESTER)

    def action_change_compaction_model(self) -> None:
        self._open_role_model_picker(AgentRole.COMPACTION)

    def action_change_resolver_model(self) -> None:
        self._open_role_model_picker(AgentRole.RESOLVER)

    def _open_role_model_picker(self, role: AgentRole) -> None:
        """Selection palette with the catalog models for a role (grouped by
        provider) plus the "Custom…" free-form entry."""
        if isinstance(self.screen, CommandPalette):
            return
        catalog = UserModelsStore().load()
        if not any(p.models for p in catalog.providers):
            # No catalog models to choose from: skip the (empty) list and
            # open the custom-model dialog right away. "Provider catalog…"
            # stays available from the main Settings menu.
            self.action_custom_role_model(role)
            return
        self.push_screen(
            SettingsPalette(
                id=f"--role-model-{role.value}",
                providers=[role_model_options_provider(role)],
                placeholder=f"Pick the {role.value} model…",
            )
        )

    def action_set_role_model(self, role: AgentRole, ref: ModelRef | None) -> None:
        """Persist one role's model choice into the project's
        drcode.json ("models" key) and apply it live."""
        model_keys_before = self._snapshot_model_keys()
        try:
            set_project_model(self.project_root, role, ref)
        except OSError as e:
            self.notify(
                f"Could not write drcode.json: {e}",
                title="Model selection",
                severity="error",
                timeout=8,
            )
            return
        self.invalidate_resolved_configs()
        self._emit_manual_model_change_events(model_keys_before)
        self.refresh_connection()
        label = f"{ref.provider}/{ref.model}" if ref else "user default"
        self.notify(
            f"{role.value} model: {label}",
            title="Model selection",
            timeout=4,
        )

    def action_activate_role_custom(self, role: AgentRole) -> None:
        """Re-activate the role's saved custom model (picker item
        "Custom: <model-id>"): only the `models[role]` slot flips back to
        "custom" — the stored custom model is reused as-is, no re-entry."""
        customs = load_project_custom_models(self.project_root)
        model_keys_before = self._snapshot_model_keys()
        if role not in customs:
            # Broken/missing custom_models[role] (a slot pointing at it is
            # treated as "nothing selected"): never silently rewritten —
            # tell the user instead.
            self.notify(
                f"No valid custom model saved for the {role.value} role — "
                "use “Custom…” to configure one.",
                title="Model selection",
                severity="warning",
                timeout=8,
            )
            return
        try:
            set_project_custom_active(self.project_root, role)
        except OSError as e:
            self.notify(
                f"Could not write drcode.json: {e}",
                title="Model selection",
                severity="error",
                timeout=8,
            )
            return
        self.invalidate_resolved_configs()
        self._emit_manual_model_change_events(model_keys_before)
        self.refresh_connection()
        self.notify(
            f"{role.value} model: {customs[role].model}",
            title="Model selection",
            timeout=4,
        )

    def action_custom_role_model(self, role: AgentRole) -> None:
        """Free-form model for a role: the ModelSetupDialog, pre-filled
        from the role's saved custom model (if any). The result is stored
        under `custom_models[role]` and activated (`models[role] =
        "custom"`) — custom models of different roles never mix."""
        if isinstance(self.screen, CommandPalette):
            return
        custom = load_project_custom_models(self.project_root).get(role)
        self.push_screen(
            ModelSetupDialog(
                model=custom.model if custom else "",
                base_url=custom.base_url if custom else "",
                api_key=custom.api_key if custom else "",
                max_context_tokens=(custom.max_context_tokens if custom else 200_000),
                max_tokens=(
                    custom.max_tokens if custom and custom.max_tokens else 8192
                ),
            ),
            partial(self._custom_role_model_done, role),
        )

    def _custom_role_model_done(
        self, role: AgentRole, setup: ModelSetup | None
    ) -> None:
        if setup is None:
            return
        model_keys_before = self._snapshot_model_keys()
        try:
            set_project_custom_model(
                self.project_root, role, custom_entry_from_setup(setup)
            )
        except OSError as e:
            self.notify(
                f"Could not write drcode.json: {e}",
                title="Model selection",
                severity="error",
                timeout=8,
            )
            return
        self.invalidate_resolved_configs()
        self._emit_manual_model_change_events(model_keys_before)
        self.refresh_connection()
        self.refresh_status()
        self._advance()
        self.notify(
            f"{role.value} model: {setup.model}",
            title="Model selection",
            timeout=4,
        )
