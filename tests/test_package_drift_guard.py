"""The drift guard's repeated reminder toast.

After the one-time poll report (`_package_drift_reported = True`) the
guard keeps refusing the lazy screen imports SILENTLY — the real case
02.11: a missed 12-second toast, then every Ctrl+M / footer / ⎇-click
did nothing while the UI still advertised unmerged work. The guard must
re-toast a short reminder on every blocked action, throttled."""

from __future__ import annotations

from typing import Any

from drcode.tui.app import DrCodeApp


def _app_with_drift(project_root: Any, monkeypatch: Any) -> tuple[DrCodeApp, list[str]]:
    """A constructed app with an already-reported drift and a notify recorder."""
    app = DrCodeApp(project_root)
    app._package_drift_reported = True
    toasts: list[str] = []
    monkeypatch.setattr(app, "notify", lambda message, **_: toasts.append(message))
    return app, toasts


def test_guard_repeats_short_reminder_when_already_reported(
    project_root: Any, monkeypatch: Any
) -> None:
    app, toasts = _app_with_drift(project_root, monkeypatch)
    assert app.package_drift_guard() is True
    assert len(toasts) == 1
    assert "restart the app" in toasts[0]
    # The short reminder, not the full first-report explanation.
    assert "ImportError" not in toasts[0]


def test_guard_reminder_is_throttled(project_root: Any, monkeypatch: Any) -> None:
    app, toasts = _app_with_drift(project_root, monkeypatch)
    assert app.package_drift_guard() is True
    assert app.package_drift_guard() is True
    assert app.package_drift_guard() is True
    assert len(toasts) == 1  # bursts within the interval stay silent


def test_guard_reminder_fires_again_after_interval(
    project_root: Any, monkeypatch: Any
) -> None:
    app, toasts = _app_with_drift(project_root, monkeypatch)
    assert app.package_drift_guard() is True
    app._package_drift_remind_at -= app._PACKAGE_DRIFT_REMIND_INTERVAL + 1.0
    assert app.package_drift_guard() is True
    assert len(toasts) == 2


def test_first_drift_report_keeps_full_text(
    project_root: Any, monkeypatch: Any
) -> None:
    """An unreported drift goes through `_report_package_drift` — the full
    explanation toast, no duplicate reminder on the same call."""
    app = DrCodeApp(project_root)
    toasts: list[str] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **_: toasts.append(message),
    )
    monkeypatch.setattr(
        "drcode.tui.app.drift_since_baseline", lambda: "version 0.1 → 0.2"
    )
    assert app.package_drift_guard() is True
    assert len(toasts) == 1
    assert "ImportError" in toasts[0]
