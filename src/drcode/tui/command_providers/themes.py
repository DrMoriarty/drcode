"""Theme-picker provider of the Settings palette."""

from __future__ import annotations

from functools import partial
from operator import attrgetter

from textual.color import Color
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.content import Content
from textual.theme import Theme


def _color_spec(color: Color) -> str:
    """Color as a textual style spec: hex for ordinary colors, the ANSI
    name ("ansi_blue", "ansi_default") for ANSI-theme colors — both parse
    as textual styles."""
    return color.hex


def _theme_preview(
    theme: Theme, *, current: bool, name: Content | None = None
) -> Content:
    """One-line preview of a theme for the picker: the theme's name followed
    by sample words ("text", "primary", "secondary", "accent") rendered in
    the theme's own colors on the theme's background, so it is immediately
    obvious whether the combination is e.g. pink-on-gray or green-on-black.
    """
    variables = theme.to_color_system().generate()

    def spec(key: str) -> str:
        return _color_spec(Color.parse(variables[key]))

    words = (
        ("text", "foreground"),
        ("primary", "primary"),
        ("secondary", "secondary"),
        ("accent", "accent"),
    )
    content = name if name is not None else Content(theme.name)
    if current:
        content = content.append_text(" (current)", "dim")
    content = content.append_text("  ")
    block_start = content.cell_length
    content = content.append_text("".join(f" {label} " for label, _ in words))
    # Word colors overlay the background so each sample word shows its
    # theme role ("primary" in $primary, …) on the theme's own background.
    content = content.stylize(f"on {spec('background')}", block_start)
    offset = block_start
    for label, key in words:
        content = content.stylize(spec(key), offset + 1, offset + 1 + len(label))
        offset += len(label) + 2
    return content


class ThemeOptionsProvider(Provider):
    """Options of the theme picker.

    Same selection-palette UX as the other option providers: pick a theme to
    apply it immediately, Escape keeps the current one. The currently active
    theme is marked with "(current)". Every entry carries a one-line color
    preview so themes can be told apart without applying them.
    """

    def _themes(self) -> list[tuple[Theme, bool, partial[None]]]:
        themes = self.app.available_themes

        def set_app_theme(name: str) -> None:
            self.app.theme = name

        return [
            (
                theme,
                theme.name == self.app.theme,
                partial(set_app_theme, theme.name),
            )
            for theme in sorted(themes.values(), key=attrgetter("name"))
        ]

    async def discover(self) -> Hits:
        for theme, current, callback in self._themes():
            yield DiscoveryHit(_theme_preview(theme, current=current), callback)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for theme, current, callback in self._themes():
            if (match := matcher.match(theme.name)) > 0:
                yield Hit(
                    match,
                    _theme_preview(
                        theme,
                        current=current,
                        name=matcher.highlight(theme.name),
                    ),
                    callback,
                )
