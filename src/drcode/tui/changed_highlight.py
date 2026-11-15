"""The shared "changed" highlight for the settings dialogs.

The orange ($warning) "changed" signal — a field differing from its
saved value — is the reference behavior of the Task-settings dialog
(a saved-values snapshot beside a draft, re-synced on every change).
This module extracts the pieces every Save/Apply dialog shares so the
signal works uniformly instead of living in two dialogs only:

- ``CHANGED_CSS`` — the tint rules for the plain form controls. A tint
  set on the ``Select`` itself is invisible (its opaque ``SelectCurrent``
  child covers it), so the rule targets that child; ``Input`` tints
  directly. Dialogs append the constant to their ``DEFAULT_CSS``.
- ``apply_changed`` / ``ChangedHighlightMixin`` — the class-setting half:
  the ``changed`` class rides the widget that actually renders the
  signal. A ``ToggleRow`` (or its RadioButton, reached by the toggle id)
  marks the row's explanatory Label — the glyph renders identically on
  and off, so a tint there carries nothing (the ToggleRow's own CSS
  tints ``Label.changed``); every other control takes the plain class.
- ``ChangedTracker`` — the snapshot half: capture the saved values when
  the dialog opens, produce the per-field changed flags for the current
  values in the change handlers.

A dialog keeps its own snapshot (a ``model_copy(deep=True)`` draft like
the Task-settings dialog, scattered ``_initial_*`` attributes or a dict)
and calls ``_apply_changed`` with a ``{widget: differs}`` mapping from
its change handlers — the helper only decides WHERE the class lands.
"""

from __future__ import annotations

from textual.widget import Widget
from textual.widgets import RadioButton

from .toggle_row import ToggleRow

CHANGED_CSS = """
    /* The "changed" marker (a field differing from its saved value), the
       Task-settings dialog's orange tint: a border-tint is useless under
       the compact form styles, and a tint set on the Select itself is
       hidden behind the opaque SelectCurrent — it goes on that child,
       plus $warning text for terminals without truecolor. Inputs tint
       directly. */
    Select.changed > SelectCurrent {
        background-tint: $warning 25%;
        color: $warning;
    }
    Input.changed {
        background-tint: $warning 25%;
    }
"""


def apply_changed(widget: Widget, is_changed: bool) -> None:
    """Set or clear the ``changed`` class on the widget that carries the
    marker: a ``ToggleRow``'s explanatory Label (whether reached by the
    row or by its RadioButton), else the control itself."""
    if isinstance(widget, ToggleRow):
        widget.set_changed(is_changed)
    elif isinstance(widget, RadioButton) and isinstance(widget.parent, ToggleRow):
        widget.parent.set_changed(is_changed)
    else:
        widget.set_class(is_changed, "changed")


class ChangedHighlightMixin:
    """Mix into a dialog/screen (any ``Widget``) for the sync helper."""

    def _apply_changed(self, markers: dict[Widget, bool]) -> None:
        """Set/clear the ``changed`` class for a ``{widget: differs}``
        mapping (the dialog's change handlers rebuild it on every edit)."""
        for widget, is_changed in markers.items():
            apply_changed(widget, is_changed)


class ChangedTracker:
    """The saved-values snapshot: ``changed(**current)`` flags the fields
    whose live value differs from the snapshot (identity compare for
    unhashable objects — enums, lists — equality for the rest)."""

    def __init__(self, **initial: object) -> None:
        self._initial: dict[str, object] = dict(initial)

    def changed(self, **current: object) -> dict[str, bool]:
        return {key: self._initial.get(key) != value for key, value in current.items()}
