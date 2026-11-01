"""DRCODE_DISABLE_SYNC_OUTPUT — the synchronized-output (DEC 2026) kill-switch.

iTerm2 3.7.x buffers textual's DEC 2026 wrapped frames, showing up as a
constant 0.5–1.0 s keystroke lag. The switch makes drcode never probe for
DEC 2026 support and ignore any "supported" answer; the default behavior
(probe sent, sync mode used when offered) must stay untouched.
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
    ENV_VAR,
    apply_sync_output_override,
    sync_output_disabled,
)

_PROBE_CLASSES = (
    _linux_driver_module.LinuxDriver,
    _linux_inline_driver_module.LinuxInlineDriver,
)


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("off", False),
        ("junk", False),
        ("1", True),
        ("true", True),
        ("YES", True),
        (" on ", True),
    ],
)
def test_sync_output_disabled_parses_env(
    value: str | None, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if value is None:
        monkeypatch.delenv(ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ENV_VAR, value)
    assert sync_output_disabled() is expected


def test_apply_override_noop_without_env(
    monkeypatch: pytest.MonkeyPatch, restore_probe_methods: None
) -> None:
    """Without the switch the drivers keep their stock probe methods."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    before = {cls: cls._request_terminal_sync_mode_support for cls in _PROBE_CLASSES}
    apply_sync_output_override()
    for cls, method in before.items():
        assert cls._request_terminal_sync_mode_support is method


def test_apply_override_skips_probe_with_env(
    monkeypatch: pytest.MonkeyPatch, restore_probe_methods: None
) -> None:
    """With the switch the probe is replaced by a no-op that never writes."""
    monkeypatch.setenv(ENV_VAR, "1")
    original = {cls: cls._request_terminal_sync_mode_support for cls in _PROBE_CLASSES}
    apply_sync_output_override()
    for cls, method in original.items():
        patched = cls._request_terminal_sync_mode_support
        assert patched is not method
        # The stub takes self, ignores it and never touches the writer.
        assert cast("Any", patched)(None) is None


def test_app_constructor_applies_override_with_env(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_probe_methods: None,
) -> None:
    """Building the app is enough: with the switch set the drivers are
    patched before any driver instance could send the probe."""
    monkeypatch.setenv(ENV_VAR, "1")
    original = {cls: cls._request_terminal_sync_mode_support for cls in _PROBE_CLASSES}
    DrCodeApp(project_root)
    for cls, method in original.items():
        assert cls._request_terminal_sync_mode_support is not method


def test_app_handler_gates_sync_support_message(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the switch set, a TerminalSupportsSynchronizedOutput message is
    swallowed (sync stays off); without it, the stock handling runs."""
    seen: list[messages.TerminalSupportsSynchronizedOutput] = []
    monkeypatch.setattr(
        App,
        "_on_terminal_supports_synchronized_output",
        lambda self, message: seen.append(message),
    )
    app = DrCodeApp(project_root)
    message = messages.TerminalSupportsSynchronizedOutput()

    monkeypatch.delenv(ENV_VAR, raising=False)
    app._on_terminal_supports_synchronized_output(message)
    assert seen == [message]

    monkeypatch.setenv(ENV_VAR, "1")
    app._on_terminal_supports_synchronized_output(message)
    assert seen == [message]
    assert app._sync_available is False
