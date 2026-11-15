"""The shared ToggleRow: a click on the explanatory Label flips the
RadioButton — the glyph is no longer the only click target (parity with
the GitHub dialog's interval radio group, where the text is the
RadioButton's own content). A label click is a USER action, so it posts
``RadioButton.Changed``; the programmatic ``set_toggle`` stays
event-silent (the stale-event guard). The "changed" marker
(``set_changed``) rides the Label, never the glyph."""

from __future__ import annotations

from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Label, RadioButton

from drcode.tui.toggle_row import ToggleRow


class _ToggleApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.changed: list[bool] = []

    def compose(self) -> ComposeResult:
        yield ToggleRow("Enable the thing", value=False, toggle_id="x-toggle")

    @on(RadioButton.Changed)
    def _record(self, event: RadioButton.Changed) -> None:
        self.changed.append(event.radio_button.value)


async def test_label_click_toggles_and_posts_changed() -> None:
    """Clicking the Label flips the RadioButton and the Changed event
    reaches a handler (a draft-tracking dialog must see it)."""
    app = _ToggleApp()
    async with app.run_test() as pilot:
        row = app.query_one(ToggleRow)
        toggle = row.toggle
        assert toggle.value is False

        await pilot.click(row.query_one(Label))
        await pilot.pause()

        assert toggle.value is True
        assert app.changed == [True]

        # Clicking again flips it back.
        await pilot.click(row.query_one(Label))
        await pilot.pause()
        assert toggle.value is False
        assert app.changed == [True, False]


async def test_glyph_click_not_double_toggled() -> None:
    """The glyph still toggles exactly once per click (a row-level click
    handler would bubble a second toggle out of the RadioButton's own)."""
    app = _ToggleApp()
    async with app.run_test() as pilot:
        toggle = app.query_one("#x-toggle", RadioButton)
        await pilot.click(toggle)
        await pilot.pause()
        assert toggle.value is True
        assert app.changed == [True]


async def test_set_toggle_stays_event_silent() -> None:
    """Programmatic sync (set_toggle) flips the value without posting
    Changed — the stale-event rule of the draft dialogs."""
    app = _ToggleApp()
    async with app.run_test() as pilot:
        row = app.query_one(ToggleRow)
        row.set_toggle(True)
        await pilot.pause()
        assert row.toggle.value is True
        assert app.changed == []

        # Setting the same value is a no-op (no event either way).
        row.set_toggle(True)
        await pilot.pause()
        assert app.changed == []


async def test_set_changed_marks_the_label_not_the_glyph() -> None:
    """The changed marker rides the row's Label — the RadioButton glyph
    renders identically on and off, so the orange tint is on the
    explanatory text (label/set_changed, dialogs never query the Label)."""
    app = _ToggleApp()
    async with app.run_test() as pilot:
        row = app.query_one(ToggleRow)
        label = row.label

        row.set_changed(True)
        await pilot.pause()
        assert label.has_class("changed")
        assert not row.toggle.has_class("changed")

        row.set_changed(False)
        await pilot.pause()
        assert not label.has_class("changed")
        assert not row.toggle.has_class("changed")
