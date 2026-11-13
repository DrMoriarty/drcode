"""Per-role model selection providers of the command palettes."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

from ...providers import (
    AgentRole,
    ModelRef,
    UserModelsStore,
    load_project_custom_models,
    load_project_model_slots,
)
from .base import CommandEntry, CommandListProvider

if TYPE_CHECKING:
    from ..app import DrCodeApp


class ProjectModelSettingsCommandProvider(CommandListProvider):
    """The single "Project model settings…" entry of the "Model
    settings…" sub-menu (replaces the old per-role picker entries).

    Opens the per-role modal form: for every agent role the model
    selection (catalog / custom / user default) plus the per-role
    max_tokens override — in one dialog with atomic Apply.
    """

    def _commands(self) -> list[CommandEntry]:
        app = cast("DrCodeApp", self.app)
        return [
            CommandEntry(
                "Project model settings…",
                app.action_open_project_model_settings,
                "Model and max_tokens override per agent role (coding / "
                "research / tester / compaction / resolver / subagents) — "
                "the model slots go to the project's drcode.json, the "
                "max_tokens overrides to the project's settings.json",
            ),
        ]


def role_model_options_provider(role: AgentRole) -> type[CommandListProvider]:
    """Selection palette listing the catalog models available for one agent
    role ("provider/model" entries; the user-level default is marked
    "(default)", the effective one "(current)"), the saved custom model of
    the role as "Custom: <model-id>" (selecting it re-activates it without
    re-entering anything), "(default — user catalog)" to drop the project
    override (only when a user default exists) and "Custom…" to enter a
    free-form model via ModelSetupDialog."""

    class RoleModelOptions(CommandListProvider):
        def _commands(self) -> list[CommandEntry]:
            app = cast("DrCodeApp", self.app)
            catalog = UserModelsStore().load()
            slots = load_project_model_slots(app.project_root)
            slot = slots.get(role)
            current_ref = slot if isinstance(slot, ModelRef) else None
            current_custom = slot == "custom"
            user_default = catalog.default_ref(role)
            commands: list[CommandEntry] = []
            for provider in catalog.providers:
                for entry in provider.models:
                    ref = ModelRef(provider=provider.name, model=entry.id)
                    label = f"{provider.name}/{entry.id}"
                    if ref == user_default:
                        label += " (default)"
                    if ref == current_ref:
                        label += " (current)"
                    commands.append(
                        CommandEntry(
                            label, partial(app.action_set_role_model, role, ref)
                        )
                    )
            customs = load_project_custom_models(app.project_root)
            if role in customs:
                label = f"Custom: {customs[role].model}"
                if current_custom:
                    label += " (current)"
                commands.append(
                    CommandEntry(label, partial(app.action_activate_role_custom, role))
                )
            if user_default is not None:
                commands.append(
                    CommandEntry(
                        "(default — user catalog)",
                        partial(app.action_set_role_model, role, None),
                    )
                )
            commands.append(
                CommandEntry("Custom…", partial(app.action_custom_role_model, role))
            )
            return commands

    return RoleModelOptions
