"""Command-palette providers, split by domain.

Re-exports the full public surface of the former single-module
``tui/command_providers.py`` so existing imports
(``from drcode.tui.command_providers import X``) keep working unchanged.

Modules:
- ``base`` — SpacedCommandList and shared widgets;
- ``settings`` — the Settings palette, its sub-menus, main-menu entry
  providers and per-preference option providers;
- ``models`` — per-role model pickers;
- ``themes`` — the theme picker;
- ``deps`` — dependency-gate commands.
"""

from .base import SpacedCommandList
from .deps import DependencyCommandsProvider
from .models import RoleModelsCommandsProvider, role_model_options_provider
from .settings import (
    WORKFLOW_PRESETS,
    DevRoundsOptionsProvider,
    EscAbortOptionsProvider,
    GitResearchOptionsProvider,
    GitTesterOptionsProvider,
    IsolationOptionsProvider,
    ModelCatalogCommandProvider,
    ModelSettingsCommandProvider,
    ModelSettingsPalette,
    ParallelTasksOptionsProvider,
    ProjectSettingsCommandsProvider,
    RetentionOptionsProvider,
    ServiceCommandsProvider,
    SettingsPalette,
    StallNudgeOptionsProvider,
    TaskSettingsCommandProvider,
    TaskSettingsEntriesProvider,
    TaskSettingsPalette,
    VerificationOptionsProvider,
    WorkflowPreset,
    WorkflowPresetOptionsProvider,
    WorktreeRemovalOptionsProvider,
    workflow_preset_for,
    workflow_preset_label,
)
from .themes import ThemeOptionsProvider

__all__ = [
    "WORKFLOW_PRESETS",
    "DependencyCommandsProvider",
    "DevRoundsOptionsProvider",
    "EscAbortOptionsProvider",
    "GitResearchOptionsProvider",
    "GitTesterOptionsProvider",
    "IsolationOptionsProvider",
    "ModelCatalogCommandProvider",
    "ModelSettingsCommandProvider",
    "ModelSettingsPalette",
    "ParallelTasksOptionsProvider",
    "ProjectSettingsCommandsProvider",
    "RetentionOptionsProvider",
    "RoleModelsCommandsProvider",
    "ServiceCommandsProvider",
    "SettingsPalette",
    "SpacedCommandList",
    "StallNudgeOptionsProvider",
    "TaskSettingsCommandProvider",
    "TaskSettingsEntriesProvider",
    "TaskSettingsPalette",
    "ThemeOptionsProvider",
    "VerificationOptionsProvider",
    "WorkflowPreset",
    "WorkflowPresetOptionsProvider",
    "WorktreeRemovalOptionsProvider",
    "role_model_options_provider",
    "workflow_preset_for",
    "workflow_preset_label",
]
