"""First-start model setup dialog.

Shown modally when the app starts with no model/base_url configured
anywhere (env vars, project drcode.json, user-level config). The user
must enter a model id and a base URL; the API key and the context-window
size are optional. ESC dismisses the dialog — the app stays usable
without a model (tasks can be created and browsed, nothing runs).

Save first validates the connection with a live probe request against
the endpoint; on failure the error is shown and the dialog stays open,
on success the values are handed back to the app, which persists them
into `<project>/drcode.json` and starts using them immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from ..agent import probe_model

# Probe timeout for an explicit user-initiated check: the header's
# background probe uses 5s, here a slow gateway deserves more patience
# before the error is shown.
_PROBE_TIMEOUT = 15.0


@dataclass
class ModelSetup:
    """Values collected by the dialog (handed to the app on success)."""

    model: str
    base_url: str
    api_key: str
    max_context_tokens: int | None  # None = keep the default


class ModelSetupDialog(ModalScreen[ModelSetup | None]):
    """Modal form: model (required), URL (required), API key and max
    context tokens (optional). ESC — skip, Save — validate & apply."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Skip", priority=True),
    ]

    DEFAULT_CSS = """
    ModelSetupDialog {
        align: center middle;
    }
    #setup-dialog {
        width: 68;
        max-width: 95%;
        height: auto;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #sd-note { color: $text; margin-bottom: 1; }
    #sd-note .req { color: $error; }
    .sd-label {
        width: auto;
        margin-top: 1;
        color: $text-muted;
        text-style: bold;
    }
    .sd-opt { color: $text-disabled; text-style: italic; }
    #sd-error {
        height: auto;
        min-height: 1;
        margin-top: 1;
        color: $error;
    }
    #sd-buttons {
        height: auto;
        margin-top: 1;
    }
    #sd-save { width: auto; }
    #sd-hint {
        width: 1fr;
        padding: 1 0 0 1;
        color: $text-disabled;
        text-align: right;
    }
    """

    def __init__(
        self,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        max_context_tokens: int = 200_000,
    ) -> None:
        super().__init__()
        self._initial_model = model
        self._initial_base_url = base_url
        self._initial_api_key = api_key
        self._initial_max_context_tokens = max_context_tokens
        # Re-entrancy guard: Enter in an input and the Save button must
        # not start two probes in parallel.
        self._checking = False

    def compose(self) -> ComposeResult:
        with Vertical(id="setup-dialog"):
            yield Static(
                "Model connection is not configured. Enter the settings for "
                "this project — they will be saved to [b]drcode.json[/b]. "
                "Fields marked [b]*[/b] are required.",
                id="sd-note",
            )
            yield Static("Model *", classes="sd-label")
            yield Input(
                placeholder="e.g. gpt-4o",
                value=self._initial_model,
                id="sd-model",
            )
            yield Static("Base URL *", classes="sd-label")
            yield Input(
                placeholder="https://api.openai.com/v1",
                value=self._initial_base_url,
                id="sd-url",
            )
            yield Static("API key (optional)", classes="sd-label sd-opt")
            yield Input(
                placeholder="leave empty if the endpoint needs no key",
                value=self._initial_api_key,
                password=True,
                id="sd-key",
            )
            yield Static("Max context tokens (optional)", classes="sd-label sd-opt")
            yield Input(
                placeholder=str(self._initial_max_context_tokens),
                value=str(self._initial_max_context_tokens),
                id="sd-ctx",
            )
            yield Static("", id="sd-error")
            with Horizontal(id="sd-buttons"):
                yield Button("Save", variant="primary", id="sd-save")
                yield Static("esc — skip and work without a model", id="sd-hint")

    def on_mount(self) -> None:
        self.query_one("#sd-model", Input).focus()

    def action_cancel(self) -> None:
        """ESC: keep working without a model (nothing is saved)."""
        self.dismiss(None)

    def _fail(self, message: str, focus_id: str) -> None:
        error = self.query_one("#sd-error", Static)
        error.update(f"✗ {message}")
        self.query_one(focus_id, Input).focus()

    async def _try_save(self) -> None:
        if self._checking:
            return
        model = self.query_one("#sd-model", Input).value.strip()
        base_url = self.query_one("#sd-url", Input).value.strip()
        api_key = self.query_one("#sd-key", Input).value.strip()
        ctx_raw = self.query_one("#sd-ctx", Input).value.strip()
        error = self.query_one("#sd-error", Static)

        if not model:
            self._fail("Model is required.", "#sd-model")
            return
        if not base_url:
            self._fail("Base URL is required.", "#sd-url")
            return
        max_context_tokens: int | None = None
        if ctx_raw:
            try:
                max_context_tokens = int(ctx_raw)
            except ValueError:
                self._fail("Max context tokens must be a positive integer.", "#sd-ctx")
                return
            if max_context_tokens <= 0:
                self._fail("Max context tokens must be a positive integer.", "#sd-ctx")
                return

        error.update("Checking connection…")
        button = self.query_one("#sd-save", Button)
        self._checking = True
        button.disabled = True
        try:
            ok, detail = await probe_model(
                model, base_url, api_key, timeout=_PROBE_TIMEOUT
            )
        finally:
            self._checking = False
            button.disabled = False
        if not ok:
            error.update(f"✗ Connection failed: {detail}")
            return
        self.dismiss(
            ModelSetup(
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_context_tokens=max_context_tokens,
            )
        )

    @on(Button.Pressed, "#sd-save")
    async def _save_clicked(self) -> None:
        await self._try_save()

    @on(Input.Submitted)
    async def _input_submitted(self, event: Input.Submitted) -> None:
        # Enter in any field = Save (the user filled the form top to
        # bottom and expects the last Enter to submit it).
        event.stop()
        await self._try_save()
