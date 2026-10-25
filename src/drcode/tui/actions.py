"""Actions layer — the composition of the user-action mixin layers.

Historically this module held one 1850-line ``ActionsLayer`` mixin with
every bound action of the main screen in a single class. It is split into
focused sub-mixins (each a :class:`~drcode.tui.state.DrCodeState` layer,
mirroring the ``ChatViewLayer`` style):

- :mod:`drcode.tui.actions_forms` — the new-task/edit-task editor, the
  first-start model-setup result and the AGENTS.md offer;
- :mod:`drcode.tui.actions_lifecycle` — user commands over tasks (done,
  cancel, ESC run-stop, nudge, queue ordering, dependencies,
  hold/resume/delete) and submitting user input to a task;
- :mod:`drcode.tui.actions_models` — per-role model selection, custom
  models and the provider catalog;
- :mod:`drcode.tui.actions_settings` — navigation, the project-settings
  palette entries, GitHub/notifications/voice and the theme picker.

``ActionsLayer`` stays the collective name (the MRO order of the
sub-layers matches the historical layout, so method resolution against
``ChatViewLayer``/``PipelineLayer`` is unchanged) —
:class:`~drcode.tui.app.DrCodeApp` and every external import keep working
untouched.
"""

from __future__ import annotations

from .actions_forms import TaskFormLayer
from .actions_lifecycle import TaskLifecycleLayer
from .actions_models import ModelSelectionLayer
from .actions_settings import SettingsLayer
from .state import DrCodeState


class ActionsLayer(
    TaskFormLayer,
    TaskLifecycleLayer,
    ModelSelectionLayer,
    SettingsLayer,
    DrCodeState,
):
    """Collective mixin: all user-action layers in one class."""
