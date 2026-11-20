"""Command-palette providers, split by domain.

Re-exports the full public surface of the former single-module
``tui/command_providers.py`` so existing imports
(``from drcode.tui.command_providers import X``) keep working unchanged.

Modules:
- ``base`` — SpacedCommandList, the shared widgets and the declarative
  ``CommandListProvider`` base (``CommandEntry`` lists);
- ``settings`` — the Settings palette, its sub-menus, main-menu entry
  providers and per-preference option providers;
- ``models`` — per-role model pickers;
- ``themes`` — the theme picker.
"""

from .base import CommandEntry, CommandListProvider, SpacedCommandList
from .models import ProjectModelSettingsCommandProvider, role_model_options_provider
from .settings import (
    WORKFLOW_PRESETS,
    AgentChainsCommandProvider,
    AgentRuntimeCommandProvider,
    DevRoundsOptionsProvider,
    EscAbortOptionsProvider,
    GitResearchOptionsProvider,
    GitTesterOptionsProvider,
    IsolationOptionsProvider,
    ModelCatalogCommandProvider,
    ModelSettingsCommandProvider,
    ModelSettingsPalette,
    ParallelTasksOptionsProvider,
    ProjectEnvironmentCommandProvider,
    ProjectSettingsCommandsProvider,
    RetentionOptionsProvider,
    ServiceCommandsProvider,
    SettingsPalette,
    StallNudgeOptionsProvider,
    StartBranchOptionsProvider,
    SubagentsCoderOptionsProvider,
    SubagentsResearcherOptionsProvider,
    VerificationOptionsProvider,
    WebFetchModeOptionsProvider,
    WorkflowPreset,
    WorkflowPresetOptionsProvider,
    WorktreeRemovalOptionsProvider,
    WorktreeSnapshotOptionsProvider,
    workflow_preset_for,
    workflow_preset_label,
)
from .themes import ThemeOptionsProvider

__all__ = [
    "WORKFLOW_PRESETS",
    "AgentChainsCommandProvider",
    "AgentRuntimeCommandProvider",
    "CommandEntry",
    "CommandListProvider",
    "DevRoundsOptionsProvider",
    "EscAbortOptionsProvider",
    "GitResearchOptionsProvider",
    "GitTesterOptionsProvider",
    "IsolationOptionsProvider",
    "ModelCatalogCommandProvider",
    "ModelSettingsCommandProvider",
    "ModelSettingsPalette",
    "ParallelTasksOptionsProvider",
    "ProjectEnvironmentCommandProvider",
    "ProjectModelSettingsCommandProvider",
    "ProjectSettingsCommandsProvider",
    "RetentionOptionsProvider",
    "ServiceCommandsProvider",
    "SettingsPalette",
    "SpacedCommandList",
    "StallNudgeOptionsProvider",
    "StartBranchOptionsProvider",
    "SubagentsCoderOptionsProvider",
    "SubagentsResearcherOptionsProvider",
    "ThemeOptionsProvider",
    "VerificationOptionsProvider",
    "WebFetchModeOptionsProvider",
    "WorkflowPreset",
    "WorkflowPresetOptionsProvider",
    "WorktreeRemovalOptionsProvider",
    "WorktreeSnapshotOptionsProvider",
    "role_model_options_provider",
    "workflow_preset_for",
    "workflow_preset_label",
]
