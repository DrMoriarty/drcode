"""App-level keep-awake integration: the real TUI pipeline (_advance
gate, on_mount wiring, on_unmount teardown) with the KeepAwake backend
replaced by a spy — no process is ever spawned.

Checks the two sides of the contract: the assertion is held while work
exists (a task in QUEUED/ACTIVE/TESTING or a run in flight), released
once the pipeline settles, never held when the setting is off, and
cleared at teardown.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drcode.models import TaskKind, TaskStatus
from drcode.tui.app import DrCodeApp


class SpyKeepAwake:
    """Record-only stand-in for KeepAwake (same surface the pipeline
    uses: start/stop/ping/active)."""

    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0
        self._held = False

    def start(self, reason: str = "") -> bool:
        self.starts += 1
        self._held = True
        return True

    def stop(self) -> None:
        self.stops += 1
        self._held = False

    def ping(self, reason: str = "") -> None:
        self.start(reason)

    @property
    def active(self) -> bool:
        return self._held


@pytest.fixture
def spy() -> SpyKeepAwake:
    return SpyKeepAwake()


async def test_assertion_held_while_queued_and_released_when_settled(
    project_root: Path, spy: SpyKeepAwake
) -> None:
    app = DrCodeApp(project_root)
    app.keepawake = spy  # type: ignore[assignment]
    task = app.manager.create("t", "brief", kind=TaskKind.RESEARCH)
    async with app.run_test() as pilot:
        await pilot.pause()
        # First evaluation: with no model configured the on_mount resume
        # keeps the pipeline idle, so the gate is fired by hand (same
        # pattern as the other pipeline tests).
        app._advance()
        await pilot.pause()
        # A queued/active task alone keeps the machine awake (no model
        # needed: busy is a status-level property — the task may be
        # promoted to ACTIVE with its initial brief any moment).
        assert task.status in (TaskStatus.QUEUED, TaskStatus.ACTIVE)
        assert spy.active and spy.starts >= 1
        held_at = spy.starts

        # The pipeline settles: nothing in QUEUED/ACTIVE/TESTING any more
        # → the assertion is released.
        app.manager.complete_without_verification(task.id)
        app._advance()
        await pilot.pause()
        assert not spy.active
        assert spy.stops >= 1
        assert spy.starts == held_at

        # Work appears again (comment → pending input → QUEUED): the
        # assertion is re-held by the same gate.
        app.manager.send(task.id, "one more round")
        app._advance()
        await pilot.pause()
        assert spy.active
        assert spy.starts > held_at


async def test_setting_off_never_holds(project_root: Path, spy: SpyKeepAwake) -> None:
    app = DrCodeApp(project_root)
    app.keepawake = spy  # type: ignore[assignment]
    app.settings.keep_awake = False
    task = app.manager.create("t", "brief", kind=TaskKind.RESEARCH)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert task.status is TaskStatus.QUEUED  # busy by status…
        assert spy.starts == 0  # …but the setting suppresses the assertion
        app._advance()
        await pilot.pause()
        assert spy.starts == 0


async def test_teardown_releases_assertion(
    project_root: Path, spy: SpyKeepAwake
) -> None:
    app = DrCodeApp(project_root)
    app.keepawake = spy  # type: ignore[assignment]
    app.manager.create("t", "brief", kind=TaskKind.RESEARCH)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._advance()
        await pilot.pause()
        assert spy.active
    # run_test has exited → on_unmount ran: the assertion is released
    # even though the queued task would otherwise keep holding it.
    assert not spy.active
    assert spy.stops >= 1


async def test_toggle_action_flips_setting_and_assertion(
    project_root: Path, spy: SpyKeepAwake
) -> None:
    app = DrCodeApp(project_root)
    app.keepawake = spy  # type: ignore[assignment]
    app.manager.create("t", "brief", kind=TaskKind.RESEARCH)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._advance()
        await pilot.pause()
        assert spy.active  # default keep_awake=True holds it
        app.action_toggle_keep_awake()
        assert app.settings.keep_awake is False
        assert not spy.active
        # The persisted project setting reflects the toggle.
        assert (
            DrCodeApp(project_root).settings.keep_awake is False
        )  # fresh store reads settings.json
        app.action_toggle_keep_awake()
        assert app.settings.keep_awake is True
        assert spy.active
        await pilot.pause()
