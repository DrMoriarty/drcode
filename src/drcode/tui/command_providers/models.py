"""Per-role model selection providers of the command palettes."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, cast

from textual.command import DiscoveryHit, Hit, Hits, Provider

from ...providers import (
    AgentRole,
    ModelRef,
    UserModelsStore,
    load_project_custom_models,
    load_project_model_slots,
)

if TYPE_CHECKING:
    from ..app import DrCodeApp


class RoleModelsCommandsProvider(Provider):
    """Per-role model pickers of the "Model settings…" sub-menu.

    One entry per agent role; its current effective model is shown after
    the ":". Picking an entry opens the role's selection palette (same UX
    as the option pickers); Escape dismisses it and keeps the current
    model.
    """

    def _commands(self) -> list[tuple[str, str, Callable[[], object]]]:
        app = cast("DrCodeApp", self.app)
        return [
            (
                f"Coding model: {app.role_model_label(AgentRole.CODING)}",
                (
                    "Model the development agent runs on; pick from the "
                    "user-level provider catalog (saved to the project's "
                    "drcode.json)"
                ),
                app.action_change_coding_model,
            ),
            (
                f"Research model: {app.role_model_label(AgentRole.RESEARCH)}",
                (
                    "Model research tasks run on (analysis/docs profile: "
                    "reads a lot, writes little — a cheap-input "
                    "large-context model works well); falls back to the "
                    "coding model when not set; pick from the user-level "
                    "provider catalog (saved to the project's drcode.json)"
                ),
                app.action_change_research_model,
            ),
            (
                f"Tester model: {app.role_model_label(AgentRole.TESTER)}",
                (
                    "Model the QA tester runs on; pick from the user-level "
                    "provider catalog (saved to the project's drcode.json)"
                ),
                app.action_change_tester_model,
            ),
            (
                f"Compaction model: {app.role_model_label(AgentRole.COMPACTION)}",
                (
                    "Model that summarizes the history when the context "
                    "fill hits the threshold; pick from the user-level "
                    "provider catalog (saved to the project's drcode.json)"
                ),
                app.action_change_compaction_model,
            ),
        ]

    async def discover(self) -> Hits:
        for name, help_text, callback in self._commands():
            yield DiscoveryHit(name, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, help_text, callback in self._commands():
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name), callback, help=help_text)


def role_model_options_provider(role: AgentRole) -> type[Provider]:
    """Selection palette listing the catalog models available for one agent
    role ("provider/model" entries; the user-level default is marked
    "(default)", the effective one "(current)"), the saved custom model of
    the role as "Custom: <model-id>" (selecting it re-activates it without
    re-entering anything), "(default — user catalog)" to drop the project
    override (only when a user default exists) and "Custom…" to enter a
    free-form model via ModelSetupDialog."""

    class RoleModelOptions(Provider):
        def _commands(self) -> list[tuple[str, Callable[[], object]]]:
            app = cast("DrCodeApp", self.app)
            catalog = UserModelsStore().load()
            slots = load_project_model_slots(app.project_root)
            slot = slots.get(role)
            current_ref = slot if isinstance(slot, ModelRef) else None
            current_custom = slot == "custom"
            user_default = catalog.defaults.get(role)
            commands: list[tuple[str, Callable[[], object]]] = []
            for provider in catalog.providers:
                for entry in provider.models:
                    ref = ModelRef(provider=provider.name, model=entry.id)
                    label = f"{provider.name}/{entry.id}"
                    if ref == user_default:
                        label += " (default)"
                    if ref == current_ref:
                        label += " (current)"
                    commands.append(
                        (label, partial(app.action_set_role_model, role, ref))
                    )
            customs = load_project_custom_models(app.project_root)
            if role in customs:
                label = f"Custom: {customs[role].model}"
                if current_custom:
                    label += " (current)"
                commands.append((label, partial(app.action_activate_role_custom, role)))
            if user_default is not None:
                commands.append(
                    (
                        "(default — user catalog)",
                        partial(app.action_set_role_model, role, None),
                    )
                )
            commands.append(("Custom…", partial(app.action_custom_role_model, role)))
            return commands

        async def discover(self) -> Hits:
            for name, callback in self._commands():
                yield DiscoveryHit(name, callback)

        async def search(self, query: str) -> Hits:
            matcher = self.matcher(query)
            for name, callback in self._commands():
                if (match := matcher.match(name)) > 0:
                    yield Hit(match, matcher.highlight(name), callback)

    return RoleModelOptions
