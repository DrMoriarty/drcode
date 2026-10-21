"""Notifications settings dialog (ntfy.sh push notifications).

Opened from the Settings palette ("Notifications: on/off"). The form
collects:

- Enabled (a Switch: toggled by mouse click, Enter or Space),
- the ntfy.sh topic name to publish to (a plain input).

Save persists the values into the project settings
(`<project>/.drcode/settings.json`); ESC — close without saving anything.
The dialog never probes the network (publishing a probe message would
push a real notification to every subscriber of the topic): instead, the
status line below the form live-renders whether the current combination
would actually deliver anything, warning about a missing topic when
Enabled is on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, Switch


@dataclass
class NotificationsSetup:
    """Values collected by the dialog (handed back to the app on Save)."""

    enabled: bool
    topic: str


class NotificationsDialog(ModalScreen[NotificationsSetup | None]):
    """Modal form: enable switch, ntfy.sh topic, Save / ESC-to-cancel."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Close", priority=True),
    ]

    DEFAULT_CSS = """
    NotificationsDialog {
        align: center middle;
    }
    #ntf-dialog {
        width: 64;
        max-width: 95%;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #ntf-title { color: $text; margin-bottom: 1; text-style: bold; }
    #ntf-note { color: $text-disabled; }
    .ntf-label {
        width: auto;
        margin-top: 1;
        color: $text-muted;
        text-style: bold;
    }
    #ntf-enabled-row { height: auto; }
    #ntf-enabled-row Switch { margin-right: 1; }
    #ntf-status {
        height: auto;
        min-height: 1;
        margin-top: 1;
        color: $text-muted;
    }
    #ntf-status.ok { color: $success; }
    #ntf-status.error { color: $error; }
    #ntf-buttons {
        dock: bottom;
        height: auto;
        margin-top: 1;
        background: $panel;
    }
    #ntf-save { width: auto; }
    #ntf-hint {
        width: 1fr;
        padding: 1 0 0 1;
        color: $text-disabled;
        text-align: right;
    }
    """

    def __init__(self, enabled: bool = False, topic: str = "") -> None:
        super().__init__()
        self._initial_enabled = enabled
        self._initial_topic = topic

    def compose(self) -> ComposeResult:
        with Vertical(id="ntf-dialog"):
            yield Static("Notifications", id="ntf-title")
            yield Static(
                "Push notifications about important events (task status "
                "changes, GitHub sync, an idle pipeline) published to the "
                "public [b]ntfy.sh[/b] service. Subscribe to the same topic "
                "in the ntfy app to receive them. Anyone who knows the "
                "topic can read its messages — pick an unguessable one.",
                id="ntf-note",
            )
            with Horizontal(id="ntf-enabled-row"):
                yield Switch(
                    value=self._initial_enabled,
                    id="ntf-enabled",
                )
                yield Static("Enabled", classes="ntf-label")
            yield Static("ntfy.sh topic", classes="ntf-label")
            yield Input(
                placeholder="my-secret-drcode-topic",
                value=self._initial_topic,
                id="ntf-topic",
            )
            yield Static("", id="ntf-status")
            with Horizontal(id="ntf-buttons"):
                yield Button("Save", variant="primary", id="ntf-save")
                yield Static("esc — close without saving", id="ntf-hint")

    def on_mount(self) -> None:
        (self.query_one("#ntf-topic", Input)).focus()
        self._refresh_status()

    def action_cancel(self) -> None:
        self.dismiss(None)

    # -- live delivery hint ----------------------------------------------------

    @on(Switch.Changed, "#ntf-enabled")
    def _enabled_changed(self, event: Switch.Changed) -> None:
        self._refresh_status()

    @on(Input.Changed, "#ntf-topic")
    def _topic_changed(self, event: Input.Changed) -> None:
        self._refresh_status()

    def _refresh_status(self) -> None:
        enabled = self.query_one("#ntf-enabled", Switch).value
        topic = self.query_one("#ntf-topic", Input).value.strip()
        status = self.query_one("#ntf-status", Static)
        if not enabled:
            status.remove_class("ok", "error")
            status.update("Notifications are off — nothing will be sent.")
            return
        if not topic:
            status.remove_class("ok")
            status.add_class("error")
            status.update("✗ Enabled, but no topic given: nothing will be sent.")
            return
        status.remove_class("error")
        status.add_class("ok")
        status.update(f"✓ Events will be published to ntfy.sh/{topic}.")

    # -- save ------------------------------------------------------------------

    @on(Button.Pressed, "#ntf-save")
    def _save_clicked(self) -> None:
        self.dismiss(
            NotificationsSetup(
                enabled=self.query_one("#ntf-enabled", Switch).value,
                topic=self.query_one("#ntf-topic", Input).value.strip(),
            )
        )
