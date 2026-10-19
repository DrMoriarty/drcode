"""GitHub integration settings dialog.

Opened from the Settings palette ("GitHub integration: on/off"). The
form collects:

- Enabled (a Switch: toggled by mouse click, Enter or Space),
- the personal access token (a masked input),
- the repository name "owner/repo" (a plain input; a pasted URL is
  normalized on save),
- the sync interval (1 / 2 / 4 hours, a RadioSet).

While the dialog is open, the status line at the bottom continuously
re-checks the connection: every change to the token or repository (and
each dialog open) triggers a live probe of GET /repos/{repo} a moment
later (debounced), and the line reports "checking…", the success or a
human-readable reason for the failure. The Save button is always
available — the user may enable the integration even when the
connection check fails or the fields are empty; the sync will simply
log its errors each cycle until the values are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Input, RadioButton, RadioSet, Static, Switch

from ..github import GitHubClient, normalize_repo
from ..settings import GITHUB_SYNC_INTERVAL_OPTIONS, GitHubSyncInterval

# How long after the last keystroke/click the live probe fires (the API
# must not be hit per character).
_PROBE_DEBOUNCE = 0.7
_PROBE_TIMEOUT = 15.0


@dataclass
class GitHubSetup:
    """Values collected by the dialog (handed back to the app on Save)."""

    enabled: bool
    token: str
    repo: str
    interval: GitHubSyncInterval


class GitHubSettingsDialog(ModalScreen[GitHubSetup | None]):
    """Modal form: enable switch, token, repository, sync interval, and a
    live connection-status line. ESC — close without saving."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Close", priority=True),
    ]

    DEFAULT_CSS = """
    GitHubSettingsDialog {
        align: center middle;
    }
    #gh-dialog {
        width: 68;
        max-width: 95%;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #gh-title { color: $text; margin-bottom: 1; text-style: bold; }
    .gh-label {
        width: auto;
        margin-top: 1;
        color: $text-muted;
        text-style: bold;
    }
    .gh-opt { color: $text-disabled; text-style: italic; }
    #gh-enabled-row { height: auto; margin-top: 1; }
    #gh-enabled-row Switch { margin-right: 1; }
    #gh-token { margin-bottom: 0; }
    #gh-interval { margin-top: 1; border: none; padding: 0; background: transparent; }
    #gh-status {
        height: auto;
        min-height: 1;
        margin-top: 1;
        color: $text-muted;
    }
    #gh-status.ok { color: $success; }
    #gh-status.error { color: $error; }
    #gh-buttons { height: auto; margin-top: 1; }
    #gh-save { width: auto; }
    #gh-hint {
        width: 1fr;
        padding: 1 0 0 1;
        color: $text-disabled;
        text-align: right;
    }
    """

    # Bumped on every form edit; the debounced probe only runs when the
    # counter still matches — stale timers from older edits are no-ops.
    _form_version: reactive[int] = reactive(0, init=False)

    def __init__(
        self,
        enabled: bool = False,
        token: str = "",
        repo: str = "",
        interval: GitHubSyncInterval = GitHubSyncInterval.HOUR_1,
    ) -> None:
        super().__init__()
        self._initial_enabled = enabled
        self._initial_token = token
        self._initial_repo = repo
        self._initial_interval = interval
        self._probe_timer = None

    def compose(self) -> ComposeResult:
        with Vertical(id="gh-dialog"):
            yield Static("GitHub integration", id="gh-title")
            yield Static(
                "Periodically imports open issues from a GitHub repository: "
                "[b]bug[/b] → bug report, [b]enhancement[/b] → feature "
                "request; [b]invalid[/b]/[b]wontfix[/b] are skipped.",
                id="gh-note",
            )
            with Horizontal(id="gh-enabled-row"):
                yield Switch(
                    value=self._initial_enabled,
                    id="gh-enabled",
                )
                yield Static("Enabled", classes="gh-label")
            yield Static("Token", classes="gh-label")
            yield Input(
                placeholder="personal access token (needs read access to the repo)",
                value=self._initial_token,
                password=True,
                id="gh-token",
            )
            yield Static("Repository", classes="gh-label")
            yield Input(
                placeholder="owner/repo (a pasted github.com URL works too)",
                value=self._initial_repo,
                id="gh-repo",
            )
            yield Static("Sync interval", classes="gh-label")
            with RadioSet(id="gh-interval"):
                for option in GITHUB_SYNC_INTERVAL_OPTIONS:
                    button = RadioButton(option.label)
                    button.value = option is self._initial_interval
                    yield button
            yield Static("", id="gh-status")
            with Horizontal(id="gh-buttons"):
                yield Button("Save", variant="primary", id="gh-save")
                yield Static("esc — close without saving", id="gh-hint")

    def on_mount(self) -> None:
        (self.query_one("#gh-token", Input)).focus()
        self._schedule_probe()

    def action_cancel(self) -> None:
        self.dismiss(None)

    # -- live connection status ---------------------------------------------

    @on(Switch.Changed, "#gh-enabled")
    def _enabled_changed(self, event: Switch.Changed) -> None:
        # Immediate visual feedback; the probe itself does not depend on
        # the switch, but the status line re-renders its prefix.
        self._form_version += 1

    @on(Input.Changed, "#gh-token")
    def _token_changed(self, event: Input.Changed) -> None:
        self._form_version += 1

    @on(Input.Changed, "#gh-repo")
    def _repo_changed(self, event: Input.Changed) -> None:
        self._form_version += 1

    def watch__form_version(self) -> None:
        self._schedule_probe()

    def _schedule_probe(self) -> None:
        status = self.query_one("#gh-status", Static)
        status.remove_class("ok", "error")
        status.update("…")
        if self._probe_timer is not None:
            self._probe_timer.stop()
        self._probe_timer = self.set_timer(_PROBE_DEBOUNCE, self._start_probe)

    def _start_probe(self) -> None:
        self._probe_timer = None
        self.run_worker(self._probe(), exclusive=True, group="gh-probe")

    async def _probe(self) -> None:
        token = self.query_one("#gh-token", Input).value.strip()
        repo = normalize_repo(self.query_one("#gh-repo", Input).value)
        status = self.query_one("#gh-status", Static)
        if not token and not repo:
            status.remove_class("ok", "error")
            status.update("Enter the token and the repository to check the connection.")
            return
        if not repo or "/" not in repo:
            status.remove_class("ok")
            status.add_class("error")
            status.update("✗ Repository name must be in the owner/repo form.")
            return
        status.remove_class("ok", "error")
        status.update(f"Checking {repo}…")
        ok, detail = await GitHubClient(token, repo).probe()
        # A later edit may have superseded this probe run (exclusive worker
        # usually prevents that, but a race at the timer boundary remains).
        current_repo = normalize_repo(self.query_one("#gh-repo", Input).value)
        if repo != current_repo:
            return
        if ok:
            status.remove_class("error")
            status.add_class("ok")
            status.update(f"✓ Connection OK: {repo} is accessible with this token.")
        else:
            status.remove_class("ok")
            status.add_class("error")
            status.update(f"✗ Connection failed: {detail}")

    # -- save ----------------------------------------------------------------

    def _selected_interval(self) -> GitHubSyncInterval:
        radio_set = self.query_one("#gh-interval", RadioSet)
        pressed = radio_set.pressed_button
        if pressed is not None:
            label = str(pressed.label)
            for option in GITHUB_SYNC_INTERVAL_OPTIONS:
                if option.label == label:
                    return option
        return self._initial_interval

    @on(Button.Pressed, "#gh-save")
    def _save_clicked(self) -> None:
        self.dismiss(
            GitHubSetup(
                enabled=self.query_one("#gh-enabled", Switch).value,
                token=self.query_one("#gh-token", Input).value.strip(),
                repo=normalize_repo(self.query_one("#gh-repo", Input).value),
                interval=self._selected_interval(),
            )
        )
