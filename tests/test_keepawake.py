"""Unit tests for drcode.keepawake: busy-status detection, start/stop
idempotency, caffeinate respawn after a process death, missing-backend
degradation, the Windows backend and the pytest environment guards.

All subprocess-backed tests inject a fake caffeinate argv (a python child
sleeping for a while) — nothing here depends on the real caffeinate
binary or the host platform.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest

from drcode import keepawake as keepawake_module
from drcode.keepawake import (
    BUSY_STATUSES,
    ES_CONTINUOUS,
    ES_SYSTEM_REQUIRED,
    KeepAwake,
    statuses_hold_awake,
)
from drcode.models import TaskStatus

SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


def _keepawake(argv: list[str] | None) -> KeepAwake:
    """A macOS-backend KeepAwake with an injected caffeinate command and
    the environment guards lifted (the unit tests exercise the real
    start/stop paths; the two env-guard tests use guarded instances)."""
    ka = KeepAwake(argv_provider=lambda: argv, platform="darwin", guarded=False)
    return ka


# -- busy detection -------------------------------------------------------------


def test_busy_statuses_hold_awake() -> None:
    assert BUSY_STATUSES == frozenset(
        {TaskStatus.QUEUED, TaskStatus.ACTIVE, TaskStatus.TESTING}
    )
    assert statuses_hold_awake([TaskStatus.QUEUED])
    assert statuses_hold_awake([TaskStatus.ACTIVE])
    assert statuses_hold_awake([TaskStatus.TESTING])


def test_non_busy_statuses_do_not_hold_awake() -> None:
    assert not statuses_hold_awake([])
    assert not statuses_hold_awake([TaskStatus.HOLD])
    assert not statuses_hold_awake([TaskStatus.AWAITING_CLARIFICATION])
    assert not statuses_hold_awake([TaskStatus.COMPLETED])
    # A hold next to a queued task still counts: work exists in flight.
    assert statuses_hold_awake([TaskStatus.HOLD, TaskStatus.QUEUED])


# -- start / stop idempotency (macOS backend, injected command) -----------------


def test_start_stop_idempotent() -> None:
    ka = _keepawake(SLEEPER)
    try:
        assert ka.start("reason") is True
        assert ka.active
        first = ka._proc
        assert first is not None and first.poll() is None
        # Second start: no second process, the first one is kept.
        assert ka.start("again") is True
        assert ka._proc is first
        ka.stop()
        assert not ka.active
        # Idempotent stop (and after the child already exited).
        ka.stop()
        assert not ka.active
    finally:
        ka.stop()


def test_stop_without_start_is_noop() -> None:
    ka = _keepawake(SLEEPER)
    ka.stop()
    assert not ka.active
    assert ka.status_note is None


# -- respawn after process death -------------------------------------------------


def test_ping_respawns_dead_process() -> None:
    ka = _keepawake(SLEEPER)
    try:
        assert ka.start("reason") is True
        proc = ka._proc
        assert proc is not None and proc.poll() is None
        # The child dies (simulated crash) — ping notices and respawns.
        proc.terminate()
        proc.wait(timeout=5)
        ka.ping("tick")
        assert ka._proc is not proc
        assert ka.active and ka._proc is not None
        assert ka._proc.poll() is None
    finally:
        ka.stop()


def test_ping_when_inactive_starts() -> None:
    ka = _keepawake(SLEEPER)
    try:
        ka.ping("tick")
        assert ka.active
    finally:
        ka.stop()


# -- degradation -----------------------------------------------------------------


def test_missing_backend_degrades() -> None:
    ka = _keepawake(None)  # shutil.which("caffeinate") found nothing
    assert ka.start("reason") is False
    assert not ka.active
    assert ka.status_note == "keep-awake: caffeinate not found"
    # stop() on a degraded instance is a harmless no-op.
    ka.stop()


def test_failing_spawn_degrades() -> None:
    # A nonexistent binary: Popen raises (OSError) — deterministic.
    ka = _keepawake(["/nonexistent/drcode-no-such-caffeinate"])
    assert ka.start("reason") is False
    assert not ka.active
    assert ka.status_note is not None and "failed to start" in ka.status_note


def test_immediate_exit_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    """A spawned backend that dies at once is treated as a failure (the
    machine would stay unprotected while we believed the opposite). The
    child is faked: real exit timing is racy under parallel pytest load."""

    class _DeadProc:
        returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr(
        keepawake_module.subprocess, "Popen", lambda *a, **k: _DeadProc(), raising=False
    )
    ka = _keepawake(SLEEPER)
    assert ka.start("reason") is False
    assert not ka.active
    assert ka.status_note == "keep-awake: caffeinate exited immediately"


def test_unknown_platform_is_noop() -> None:
    ka = KeepAwake(platform="freebsd", guarded=False)
    assert ka.start("reason") is False
    assert ka.status_note == "keep-awake: no backend for this platform"
    ka.stop()


# -- Windows backend (ctypes mocked — the test host is macOS) --------------------


class _FakeKernel32:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def SetThreadExecutionState(self, flags: int) -> int:
        self.calls.append(flags)
        return 1 if flags else 0


@pytest.fixture
def fake_windows(monkeypatch: pytest.MonkeyPatch) -> _FakeKernel32:
    kernel32 = _FakeKernel32()
    fake_windll = SimpleNamespace(kernel32=kernel32)
    monkeypatch.setattr(keepawake_module.ctypes, "windll", fake_windll, raising=False)
    return kernel32


def test_windows_backend_holds_and_releases(fake_windows: _FakeKernel32) -> None:
    ka = KeepAwake(platform="win32", guarded=False)
    assert ka.start("reason") is True
    assert fake_windows.calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]
    ka.stop()
    assert fake_windows.calls[-1] == ES_CONTINUOUS
    assert not ka.active
    # Idempotent stop: the release call is not repeated.
    ka.stop()
    assert len(fake_windows.calls) == 2


def test_windows_backend_degrades_on_ctypes_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Broken:
        def __getattr__(self, name: str) -> Any:
            raise OSError("no kernel32")

    monkeypatch.setattr(keepawake_module.ctypes, "windll", _Broken(), raising=False)
    ka = KeepAwake(platform="win32")
    assert ka.start("reason") is False
    assert not ka.active
    assert ka.status_note is not None
    ka.stop()


# -- environment guards ------------------------------------------------------------


def test_pytest_env_disables_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests::test_x")
    monkeypatch.delenv("DRCODE_NO_KEEPAWAKE", raising=False)
    ka = KeepAwake(argv_provider=lambda: SLEEPER, platform="darwin")
    assert ka.start("reason") is False
    assert not ka.active
    assert ka._proc is None


def test_drcode_no_keepawake_disables_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("DRCODE_NO_KEEPAWAKE", "1")
    ka = KeepAwake(argv_provider=lambda: SLEEPER, platform="darwin")
    assert ka.start("reason") is False
    assert not ka.active
    # "0" is an explicit opt-out of the guard.
    monkeypatch.setenv("DRCODE_NO_KEEPAWAKE", "0")
    try:
        assert ka.start("reason") is True
    finally:
        ka.stop()


# -- threading safety (start/stop are called from ticks and workers) --------------


def test_concurrent_start_stop_is_thread_safe() -> None:
    ka = _keepawake(SLEEPER)
    try:
        import threading

        def churn() -> None:
            for _ in range(20):
                ka.start("t")
                time.sleep(0)
                ka.stop()

        threads = [threading.Thread(target=churn) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # The churn ends in stop(): nothing left held, no crash.
        assert not ka.active
    finally:
        ka.stop()
