"""Shared flat on/off toggle row for the settings dialogs.

The boolean toggles of the Task-settings dialog (``.ts-toggle-row`` —
a RadioButton beside a wrapping label, styled flat: transparent
background, no border, no padding) are the reference look of the
settings forms. This module extracts that pattern into a reusable
widget so the other settings dialogs (project model settings, voice,
notifications, GitHub) present their booleans the same way — they used
the stock ``Switch``, which renders differently.

The widget owns the CSS (including the "changed" marker rule: the
``changed`` class set on the row tints the explanatory Label orange —
the same field-differs-from-saved signal the Task-settings dialog uses,
visible even under the compact borderless styling; the glyph itself
renders identically on and off, so a tint there is pointless). The row
never sets the marker itself; a dialog that tracks a draft marks its
rows exactly like its other fields (``row.set_changed(bool)`` —
a thin ``set_class(changed)`` on the row's Label).

``set_toggle`` is the programmatic-sync helper (an analogue of the Task
settings dialog's ``_set_toggle``): assigning ``value`` posts a
``RadioButton.Changed`` message that Textual delivers asynchronously, so
such an event can be STALE by the time its handler runs — the guard
suppresses it.

The whole row is the click target: the Label carries its own click
handler that flips the RadioButton — a click on the explanatory text
activates the toggle exactly like a click on the glyph (the reference
behavior of the GitHub dialog's interval radio group, where the text is
the RadioButton's own content). A label click is a genuine USER action,
so it POSTS ``RadioButton.Changed`` normally — the draft-tracking
handlers must see it; only ``set_toggle`` (programmatic sync) goes
through the ``prevent`` path.
"""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label, RadioButton


class _RowLabel(Label):
    """The row's explanatory label — clickable.

    A click flips the row's RadioButton (the sibling glyph toggles
    itself, this covers the text). The handler lives HERE, not on the
    row: a Click event bubbles, so a row-level handler would also fire
    for clicks on the RadioButton and toggle it twice.
    """

    def on_click(self) -> None:
        row = self.parent
        assert isinstance(row, ToggleRow)  # composed as ToggleRow's child
        row.toggle.toggle()


class ToggleRow(Horizontal):
    """A flat boolean toggle: ``RadioButton`` + wrapping ``Label``.

    ``toggle_id`` is the RadioButton's id (dialogs and tests query the
    toggle by it, keeping the ids the Switches used before); ``id`` — the
    row container's own id (only when a selector must reach the row).
    """

    DEFAULT_CSS: ClassVar = """
    ToggleRow {
        height: auto;
        margin-top: 1;
    }
    ToggleRow > RadioButton {
        width: auto;
        background: transparent;
        border: none;
        padding: 0;
        margin-right: 1;
    }
    ToggleRow > Label {
        width: 1fr; /* fill the row so long text WRAPS, never clips */
        height: auto; /* grow with the wrapped line count */
        color: $text-muted;
        margin-top: 0; /* compact widgets need no border-spacing */
    }
    /* The "changed" marker (a field differing from the saved value): the
       tint goes on the explanatory Label — the RadioButton glyph renders
       identically on and off, so a tint there carries nothing; plus
       $warning text for terminals without truecolor. */
    ToggleRow > Label.changed {
        background-tint: $warning 25%;
        color: $warning;
    }
    """

    def __init__(
        self,
        label: str | Text,
        *,
        value: bool = False,
        toggle_id: str | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._row_label = label
        self._initial_value = value
        self._toggle_id = toggle_id

    def compose(self) -> ComposeResult:
        yield RadioButton(value=self._initial_value, id=self._toggle_id)
        yield _RowLabel(self._row_label)

    @property
    def toggle(self) -> RadioButton:
        """The row's RadioButton (queried by type — the only one here)."""
        return self.query_one(RadioButton)

    @property
    def label(self) -> Label:
        """The row's explanatory Label."""
        return self.query_one(Label)

    def set_changed(self, is_changed: bool) -> None:
        """Mark the row as differing from its saved value: the "changed"
        class rides the explanatory Label (a legible tint on text — the
        glyph renders identically on and off), never the RadioButton.
        Draft-tracking dialogs call this instead of setting classes on
        the toggle."""
        self.label.set_class(is_changed, "changed")

    def set_toggle(self, value: bool) -> None:
        """Programmatic sync of the toggle (the Task-settings
        ``_set_toggle`` rule: a stale Changed event must never echo back
        into the draft)."""
        toggle = self.toggle
        if toggle.value != value:
            with self.prevent(RadioButton.Changed):
                toggle.value = value
