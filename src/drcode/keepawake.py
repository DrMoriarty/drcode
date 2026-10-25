"""Keep-awake: prevent the OS from sleeping the machine while drcode has
work in flight.

Why: a desktop that dozes mid-run kills the network (macOS tears the
connections down on sleep), and every concurrent agent run then dies with
``ModelConnectionError`` — the retry budget (~2.5 min) never survives a
full sleep. While any task is QUEUED/ACTIVE/TESTING (or a run is in
flight) drcode therefore holds a system-sleep assertion; when the pipeline
settles, the assertion is released and the machine may sleep as usual.

Platform backends (degradation follows the :mod:`drcode.voice` pattern —
an unavailable backend degrades to a no-op + a status note, never an
exception):

- **macOS**: a ``caffeinate -i -s`` subprocess (the stock utility, zero
  dependencies). ``-i`` = ``PreventUserIdleSystemSleep`` — the system does
  not fall asleep on its own while the assertion is held (the display may
  still blank, the network stays up); ``-s`` additionally holds the system
  sleep on AC power. The child process is polled on every
  :meth:`KeepAwake.ping` and respawned if it died.
- **Windows**: ``ctypes`` → ``kernel32.SetThreadExecutionState(
  ES_CONTINUOUS | ES_SYSTEM_REQUIRED)``; releasing is the same call with
  bare ``ES_CONTINUOUS``.
- **Linux**: a deliberate no-op — desktop sleep is rare and servers do
  not sleep. TODO: a D-Bus inhibit (``org.freedesktop.PowerManagement.
  Inhibit``) could be added if a laptop distro ever needs it.

Test guards: under pytest (``PYTEST_CURRENT_TEST``) or with
``DRCODE_NO_KEEPAWAKE=1`` in the environment :meth:`KeepAwake.start`
degrades to a no-op — otherwise every ``uv run pytest`` on a macOS dev
machine would spawn real ``caffeinate`` children.

CAVEAT (documented behaviour): the assertion does NOT stop the clamshell
sleep — closing the lid on battery still sleeps a macOS laptop. That is a
separate macOS power mode no user-land assertion can override.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable

from .models import TaskStatus

# Windows SetThreadExecutionState flags.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

# Task statuses that count as "work in flight": the pipeline may start or
# resume them at any moment (dependency-blocked QUEUED tasks included — a
# manual `git merge drcode/<dep>` can unblock them at any time, which the
# 30 s _poll_dependency_unblock tick exists for). HOLD (parked by the
# user), AWAITING_CLARIFICATION (paused for a human answer) and COMPLETED
# do NOT keep the machine awake.
BUSY_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.QUEUED,
        TaskStatus.ACTIVE,
        TaskStatus.TESTING,
    }
)


def statuses_hold_awake(statuses: Iterable[TaskStatus]) -> bool:
    """Pure busy-detect half of the keep-awake decision: True when at
    least one task status counts as work in flight (see
    :data:`BUSY_STATUSES`)."""
    return any(status in BUSY_STATUSES for status in statuses)


def _env_disabled(env: dict[str, str] | os._Environ[str]) -> bool:
    """Keep-awake is force-disabled under pytest or via DRCODE_NO_KEEPAWAKE
    (set by tests/conftest.py) — test runs must never spawn caffeinate."""
    if "PYTEST_CURRENT_TEST" in env:
        return True
    return env.get("DRCODE_NO_KEEPAWAKE", "") not in ("", "0")


class KeepAwake:
    """One power-management assertion for the app's lifetime.

    :meth:`start` / :meth:`stop` are idempotent and safe to call from any
    thread (the pipeline re-evaluates the busy state on every
    ``_advance()`` and on a periodic tick); :meth:`ping` additionally
    respawns a dead ``caffeinate`` child. Everything degrades: a missing
    backend leaves :attr:`active` False and records the reason in
    :attr:`status_note` instead of raising.
    """

    def __init__(
        self,
        *,
        argv_provider: Callable[[], list[str] | None] | None = None,
        platform: str | None = None,
        guarded: bool = True,
    ) -> None:
        """``argv_provider`` returns the full command to run (macOS backend)
        or None when the backend is unavailable — injectable for tests;
        the default resolves ``caffeinate`` via ``shutil.which``.
        ``platform`` overrides ``sys.platform`` (tests). ``guarded=False``
        lifts the pytest / DRCODE_NO_KEEPAWAKE environment guards — used
        by the unit tests that exercise the real start/stop paths (pytest
        re-installs PYTEST_CURRENT_TEST after fixture setup, so it cannot
        be neutralized by a fixture)."""
        self._argv_provider = argv_provider or self._default_argv
        self._platform = platform if platform is not None else sys.platform
        self._guarded = guarded
        self._lock = threading.Lock()
        self._active = False
        self._proc: subprocess.Popen[bytes] | None = None
        self._es_held = False
        self.status_note: str | None = None

    # -- backend resolution --------------------------------------------------

    @staticmethod
    def _default_argv() -> list[str] | None:
        """macOS: the stock caffeinate; `-i` prevents idle system sleep,
        `-s` additionally holds system sleep on AC power."""
        caffeinate = shutil.which("caffeinate")
        if caffeinate is None:
            return None
        return [caffeinate, "-i", "-s"]

    # -- public API ------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether a sleep assertion is currently held (True also means the
        backend reported success — a degraded backend never activates)."""
        with self._lock:
            return self._active

    def start(self, reason: str = "") -> bool:
        """Hold the assertion (idempotent). Returns whether it is held
        afterwards. A disabled/missing backend is a no-op with a note,
        never an exception."""
        with self._lock:
            if self._active:
                return True
            self.status_note = None
            if self._guarded and _env_disabled(os.environ):
                self.status_note = "keep-awake disabled (test environment)"
                return False
            if self._platform.startswith("darwin"):
                return self._start_macos(reason)
            if self._platform.startswith("win"):
                return self._start_windows()
            # Linux (and anything else): deliberate no-op backend.
            self.status_note = "keep-awake: no backend for this platform"
            return False

    def stop(self) -> None:
        """Release the assertion (idempotent; safe when nothing is held)."""
        with self._lock:
            if self._platform.startswith("darwin"):
                self._stop_macos()
            elif self._platform.startswith("win"):
                self._stop_windows()
            else:
                self._active = False

    def ping(self, reason: str = "") -> None:
        """Periodic safety tick: respawn a dead caffeinate child, and
        (re)start when the caller says the machine should be kept awake
        but the assertion is not held."""
        with self._lock:
            if self._active and self._proc is not None:
                if self._proc.poll() is None:
                    return
                # caffeinate died (crash, killed by the system): respawn.
                self._stop_macos()
            # Release the re-entrant lock start() would otherwise block on:
            # do the work inline instead of calling start() under the lock.
        self.start(reason)

    # -- macOS backend -----------------------------------------------------------

    def _start_macos(self, reason: str) -> bool:
        argv = self._argv_provider()
        if not argv:
            self.status_note = "keep-awake: caffeinate not found"
            return False
        try:
            proc: subprocess.Popen[bytes] = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            self.status_note = f"keep-awake: caffeinate failed to start: {e}"
            return False
        if proc.poll() is not None:
            # Died immediately: a broken binary would leave the machine
            # unprotected while we believe the opposite.
            self.status_note = "keep-awake: caffeinate exited immediately"
            return False
        try:
            # Bounded liveness probe (races the poll above for fast
            # exits); start() transitions are rare busy-flips, a 0.1 s
            # block on each is acceptable. TimeoutExpired = still running.
            proc.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            pass
        else:
            self.status_note = "keep-awake: caffeinate exited immediately"
            return False
        self._proc = proc
        self._active = True
        return True

    def _stop_macos(self) -> None:
        proc, self._proc = self._proc, None
        self._active = False
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass  # best-effort: the child dies with the session anyway
        try:
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001 - teardown must never raise
            try:
                proc.kill()
            except OSError:
                pass

    # -- Windows backend -----------------------------------------------------------

    def _start_windows(self) -> bool:
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            result = kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            )
        except Exception as e:  # noqa: BLE001 - degradation, never a crash
            self.status_note = f"keep-awake: SetThreadExecutionState failed: {e}"
            return False
        if not result:
            self.status_note = "keep-awake: SetThreadExecutionState returned 0"
            return False
        self._es_held = True
        self._active = True
        return True

    def _stop_windows(self) -> None:
        if self._es_held:
            with contextlib.suppress(Exception):  # best-effort release
                ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
                    ES_CONTINUOUS
                )
            self._es_held = False
        self._active = False
