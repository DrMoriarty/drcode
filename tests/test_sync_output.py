"""Synchronized-output (DEC 2026) gating.

iTerm2 3.7.x buffers textual's DEC 2026 wrapped frames, showing up as a
constant 0.5–1.0 s keystroke lag. Priority matrix (see sync_output.py):

1. ``DRCODE_DISABLE_SYNC_OUTPUT`` — explicit opt-out, always wins.
2. ``DRCODE_ENABLE_SYNC_OUTPUT`` — explicit opt-in, overrides only the
   iTerm2 default (never #1).
3. iTerm2 (TERM_PROGRAM / LC_TERMINAL) — off by default.
4. Other terminals — stock behavior (probe sent, sync used when offered).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import textual.drivers.linux_driver as _linux_driver_module
import textual.drivers.linux_inline_driver as _linux_inline_driver_module
from textual import messages
from textual.app import App

from drcode.tui.app import DrCodeApp
from drcode.tui.sync_output import (
    ENABLE_ENV_VAR,
    ENV_VAR,
    apply_sync_output_override,
    is_iterm2,
    sync_output_disabled,
    sync_output_off_reason,
    sync_output_should_be_off,
)

_PROBE_CLASSES = (
    _linux_driver_module.LinuxDriver,
    _linux_inline_driver_module.LinuxInlineDriver,
)

_DISABLE_OFF = [None, "", "0", "false", "off", "junk"]
_DISABLE_ON = ["1", "true", "YES", " on "]


@pytest.fixture
def restore_probe_methods() -> Any:
    """Save/restore the drivers' DECRQM probe methods around a test that
    (possibly) lets apply_sync_output_override() patch them globally."""
    saved: dict[Any, Any] = {
        cls: cls._request_terminal_sync_mode_support for cls in _PROBE_CLASSES
    }
    yield
    for cls, method in saved.items():
        cls._request_terminal_sync_mode_support = method


@pytest.fixture(autouse=True)
def clean_terminal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Conftest strips only the provider-config DRCODE_* vars and never
    TERM_PROGRAM/LC_TERMINAL — drop those plus both sync-output switches so
    tests never inherit the host environment (this suite itself may run
    inside iTerm2 with the kill-switch set)."""
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("LC_TERMINAL", raising=False)
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)


def _set_terminal(monkeypatch: pytest.MonkeyPatch, how: str | None) -> None:
    """how: None (not iTerm2), 'term_program', 'lc_terminal'."""
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("LC_TERMINAL", raising=False)
    if how == "term_program":
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    elif how == "lc_terminal":
        monkeypatch.setenv("LC_TERMINAL", "iTerm2")


# ---------------------------------------------------------------------------
# Detection and the decision function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", [None, "term_program", "lc_terminal"])
def test_is_iterm2_detects_by_both_vars(
    how: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_terminal(monkeypatch, how)
    assert is_iterm2() is (how is not None)


def test_is_iterm2_other_term_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    monkeypatch.setenv("LC_TERMINAL", "iTerm.app")  # wrong value on purpose
    assert is_iterm2() is False


@pytest.mark.parametrize("value", _DISABLE_OFF)
@pytest.mark.parametrize("how", [None, "term_program", "lc_terminal"])
def test_decision_matrix_without_disable(
    value: str | None,
    how: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No explicit disable: iTerm2 defaults to off, other terminals don't."""
    _set_terminal(monkeypatch, how)
    if value is not None:
        monkeypatch.setenv(ENV_VAR, value)
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    assert sync_output_should_be_off() is (how is not None)


@pytest.mark.parametrize("value", _DISABLE_ON)
@pytest.mark.parametrize("how", [None, "term_program", "lc_terminal"])
def test_decision_matrix_disable_always_wins(
    value: str,
    how: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_terminal(monkeypatch, how)
    monkeypatch.setenv(ENV_VAR, value)
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    assert sync_output_should_be_off() is True
    assert sync_output_disabled() is True


@pytest.mark.parametrize("how", [None, "term_program", "lc_terminal"])
@pytest.mark.parametrize("enable", ["1", "true", "YES", " on "])
def test_enable_overrides_iterm2_default(
    how: str | None, enable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_terminal(monkeypatch, how)
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setenv(ENABLE_ENV_VAR, enable)
    assert sync_output_should_be_off() is False


@pytest.mark.parametrize("disable_value", _DISABLE_OFF)
def test_enable_beats_falsy_disable(
    disable_value: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A falsy/junk/empty disable is NOT an explicit opt-out: a truthy
    enable wins over both the junk value and the iTerm2 default."""
    _set_terminal(monkeypatch, "term_program")
    if disable_value is not None:
        monkeypatch.setenv(ENV_VAR, disable_value)
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    assert sync_output_should_be_off() is False


# ---------------------------------------------------------------------------
# Driver probe patching
# ---------------------------------------------------------------------------


def test_apply_override_noop_on_plain_terminal(
    monkeypatch: pytest.MonkeyPatch, restore_probe_methods: None
) -> None:
    """Without the switch and outside iTerm2 the drivers keep their stock
    probe methods."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    before = {cls: cls._request_terminal_sync_mode_support for cls in _PROBE_CLASSES}
    apply_sync_output_override()
    for cls, method in before.items():
        assert cls._request_terminal_sync_mode_support is method


def test_apply_override_noop_on_iterm2_with_enable(
    monkeypatch: pytest.MonkeyPatch, restore_probe_methods: None
) -> None:
    """An explicit enable restores the stock probe even on iTerm2."""
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    before = {cls: cls._request_terminal_sync_mode_support for cls in _PROBE_CLASSES}
    apply_sync_output_override()
    for cls, method in before.items():
        assert cls._request_terminal_sync_mode_support is method


@pytest.mark.parametrize("how", ["env", "term_program", "lc_terminal"])
def test_apply_override_skips_probe(
    how: str, monkeypatch: pytest.MonkeyPatch, restore_probe_methods: None
) -> None:
    """Explicit disable or iTerm2 default: the probe is replaced by a no-op
    that never writes."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    if how == "env":
        monkeypatch.setenv(ENV_VAR, "1")
    elif how == "term_program":
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    else:
        monkeypatch.setenv("LC_TERMINAL", "iTerm2")
    original = {cls: cls._request_terminal_sync_mode_support for cls in _PROBE_CLASSES}
    apply_sync_output_override()
    for cls, method in original.items():
        patched = cls._request_terminal_sync_mode_support
        assert patched is not method
        # The stub takes self, ignores it and never touches the writer.
        assert cast("Any", patched)(None) is None


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["env", "term_program", "lc_terminal"])
def test_app_constructor_applies_override(
    how: str,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_probe_methods: None,
) -> None:
    """Building the app is enough: with sync output forced off (explicitly
    or via the iTerm2 default) the drivers are patched before any driver
    instance could send the probe."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    if how == "env":
        monkeypatch.setenv(ENV_VAR, "1")
    elif how == "term_program":
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    else:
        monkeypatch.setenv("LC_TERMINAL", "iTerm2")
    original = {cls: cls._request_terminal_sync_mode_support for cls in _PROBE_CLASSES}
    DrCodeApp(project_root)
    for cls, method in original.items():
        assert cls._request_terminal_sync_mode_support is not method


def test_app_constructor_no_override_on_iterm2_with_enable(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_probe_methods: None,
) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    before = {cls: cls._request_terminal_sync_mode_support for cls in _PROBE_CLASSES}
    DrCodeApp(project_root)
    for cls, method in before.items():
        assert cls._request_terminal_sync_mode_support is method


@pytest.mark.parametrize("how", ["env", "term_program", "lc_terminal"])
def test_app_handler_gates_sync_support_message(
    how: str, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When sync output must stay off (explicit disable or iTerm2 default),
    a TerminalSupportsSynchronizedOutput message is swallowed (sync stays
    off); on a plain terminal the stock handling runs."""
    seen: list[messages.TerminalSupportsSynchronizedOutput] = []
    monkeypatch.setattr(
        App,
        "_on_terminal_supports_synchronized_output",
        lambda self, message: seen.append(message),
    )
    app = DrCodeApp(project_root)
    message = messages.TerminalSupportsSynchronizedOutput()

    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    _set_terminal(monkeypatch, None)
    app._on_terminal_supports_synchronized_output(message)
    assert seen == [message]

    if how == "env":
        monkeypatch.setenv(ENV_VAR, "1")
    elif how == "term_program":
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    else:
        monkeypatch.setenv("LC_TERMINAL", "iTerm2")
    app._on_terminal_supports_synchronized_output(message)
    assert seen == [message]
    assert app._sync_available is False


def test_app_handler_accepts_sync_support_on_iterm2_with_enable(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit enable lets the stock handling flip _sync_available even
    on iTerm2."""
    seen: list[messages.TerminalSupportsSynchronizedOutput] = []
    monkeypatch.setattr(
        App,
        "_on_terminal_supports_synchronized_output",
        lambda self, message: seen.append(message),
    )
    app = DrCodeApp(project_root)
    message = messages.TerminalSupportsSynchronizedOutput()

    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    app._on_terminal_supports_synchronized_output(message)
    # The gate let it through to the (stubbed) stock handling.
    assert seen == [message]
    assert app._sync_available is False  # the stub above never flips it


# ---------------------------------------------------------------------------
# Startup diagnostics (app log only — no toast)
# ---------------------------------------------------------------------------


class _Capture:
    """What the diagnostics emit, recorded."""

    def __init__(self) -> None:
        self.toasts: list[str] = []
        self.log_lines: list[str] = []


def _capture_diagnostics(monkeypatch: pytest.MonkeyPatch, app: DrCodeApp) -> _Capture:
    """Hook notify/log on an UNMOUNTED app so the diagnostics can run
    without a live event loop."""
    cap = _Capture()
    monkeypatch.setattr(app, "notify", lambda text, **kw: cap.toasts.append(text))
    monkeypatch.setattr(
        DrCodeApp,
        "log",
        property(lambda self: lambda text: cap.log_lines.append(text)),
    )
    return cap


def test_off_reason_from_disable_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "1")
    assert sync_output_off_reason() == "DRCODE_DISABLE_SYNC_OUTPUT=1"
    monkeypatch.setenv(ENV_VAR, " true ")
    assert sync_output_off_reason() == "DRCODE_DISABLE_SYNC_OUTPUT=true"


def test_off_reason_none_when_decided_by_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain terminal (or an explicit enable): the probe goes out and the
    effective state is only known from the DECRQM answer — no reason yet."""
    _set_terminal(monkeypatch, None)
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    assert sync_output_off_reason() is None
    _set_terminal(monkeypatch, "term_program")
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    assert sync_output_off_reason() is None


@pytest.mark.parametrize("how", ["term_program", "lc_terminal"])
def test_startup_log_off_iterm2_default(
    how: str, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The iTerm2 default-off is known at startup: an app-log line with
    the reason, never a toast."""
    _set_terminal(monkeypatch, how)
    app = DrCodeApp(project_root)
    cap = _capture_diagnostics(monkeypatch, app)
    app._log_sync_mode_state()
    assert cap.log_lines == ["Synchronized output (DEC 2026): off (iTerm2 default)"]
    assert cap.toasts == []


def test_startup_log_off_disable_env(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The explicit opt-out is logged with the actual variable value."""
    monkeypatch.setenv(ENV_VAR, "1")
    app = DrCodeApp(project_root)
    cap = _capture_diagnostics(monkeypatch, app)
    app._log_sync_mode_state()
    expected = "Synchronized output (DEC 2026): off (DRCODE_DISABLE_SYNC_OUTPUT=1)"
    assert cap.log_lines == [expected]
    assert cap.toasts == []


def test_startup_plain_terminal_logs_nothing(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probe sent (plain terminal): nothing is logged at startup — the
    effective state is only known from the DECRQM answer."""
    _set_terminal(monkeypatch, None)
    app = DrCodeApp(project_root)
    cap = _capture_diagnostics(monkeypatch, app)
    app._log_sync_mode_state()
    assert cap.log_lines == []
    assert cap.toasts == []


def test_answer_logs_on_when_terminal_answers(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probe sent, terminal answers "yes": the "on" line lands in the app
    log (no toast)."""
    _set_terminal(monkeypatch, None)
    monkeypatch.setattr(
        App,
        "_on_terminal_supports_synchronized_output",
        lambda self, message: None,
    )
    app = DrCodeApp(project_root)
    cap = _capture_diagnostics(monkeypatch, app)
    app._on_terminal_supports_synchronized_output(
        messages.TerminalSupportsSynchronizedOutput()
    )
    assert cap.log_lines == ["Synchronized output (DEC 2026): on"]
    assert cap.toasts == []


def test_gated_message_never_logs_on(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With sync forced off, a slipping-through support message changes
    nothing: no "on" line, and the startup diagnostic still says off."""
    monkeypatch.setenv(ENV_VAR, "1")
    app = DrCodeApp(project_root)
    cap = _capture_diagnostics(monkeypatch, app)
    app._on_terminal_supports_synchronized_output(
        messages.TerminalSupportsSynchronizedOutput()
    )
    assert cap.log_lines == []
    app._log_sync_mode_state()
    expected = "Synchronized output (DEC 2026): off (DRCODE_DISABLE_SYNC_OUTPUT=1)"
    assert cap.log_lines == [expected]
    assert cap.toasts == []
