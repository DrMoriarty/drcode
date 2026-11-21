"""Deferred startup archive load + lazy COMPLETED sidebar section.

Large archives (drcode's own ~900-task dogfood archive) used to be read
synchronously in TaskManager.__init__ — inside DrCodeApp.__init__, BEFORE
the event loop starts: the first frame waited seconds (tens on a cold
file-system cache). The load now happens in a worker after the first
frame; the pipeline's first _advance() is gated on it, and the COMPLETED
sidebar section is collapsed by default with a lazily mounted row cap.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from drcode.models import TaskStatus
from drcode.task_manager import TaskManager
from drcode.tui import sidebar_view
from drcode.tui.app import DrCodeApp
from drcode.tui.sidebar import TaskItem

from .network_mocks import wait_until


def _make_tasks(project_root: Path, count: int) -> None:
    manager = TaskManager(project_root)
    for i in range(count):
        manager.create(f"task {i}", "prompt")
    # Distinct created_at order is guaranteed by create(); nothing else
    # needed — the tests below only care about file counts.


def test_small_archive_loads_eagerly_even_with_defer_load(
    project_root: Path,
) -> None:
    """Below the threshold defer_load changes nothing: the archive is
    read inside the constructor (the historic synchronous contract)."""
    _make_tasks(project_root, 3)
    manager = TaskManager(project_root, defer_load=True)
    assert manager.loaded
    assert len(manager.tasks) == 3


def test_large_archive_defers_and_load_completes(project_root: Path) -> None:
    """Above the threshold the archive stays unread until load(); the
    load then runs the full startup bookkeeping."""
    _make_tasks(project_root, 5)
    manager = _deferred_manager(project_root)
    assert not manager.loaded
    assert manager.tasks == {}
    manager.load()
    assert manager.loaded
    assert len(manager.tasks) == 5
    # Every task got a queue stamp (backfill ran).
    assert all(t.queue_seq is not None for t in manager.tasks.values())


def _deferred_manager(project_root: Path) -> TaskManager:
    """A TaskManager forced onto the deferred branch via the threshold
    (-1: even an empty archive counts as 'large')."""
    original = TaskManager.LARGE_ARCHIVE_FILE_THRESHOLD
    TaskManager.LARGE_ARCHIVE_FILE_THRESHOLD = -1
    try:
        return TaskManager(project_root, defer_load=True)
    finally:
        TaskManager.LARGE_ARCHIVE_FILE_THRESHOLD = original


def test_deferred_load_demotes_stale_active_and_backfills(
    project_root: Path,
) -> None:
    eager = TaskManager(project_root)
    task = eager.create("running when the app died", "prompt")
    task.status = TaskStatus.ACTIVE
    eager.storage.save(task)
    # A task WITHOUT a queue stamp (older build): the backfill must stamp it.
    task2 = eager.create("unstamped", "prompt")
    task2.queue_seq = None

    manager = _deferred_manager(project_root)
    assert not manager.loaded
    manager.load()
    loaded = {t.id: t for t in manager.tasks.values()}
    assert loaded[task.id].status is TaskStatus.QUEUED
    assert loaded[task.id].unfinished
    assert loaded[task2.id].queue_seq is not None


def test_task_created_during_load_window_survives(project_root: Path) -> None:
    """A task created in memory before load() finished is kept and
    re-stamped to the queue tail (its window stamp counted from a zero
    counter and would sort above the whole archive)."""
    manager = _deferred_manager(project_root)
    eager_stamp_source = TaskManager(project_root)
    archived = eager_stamp_source.create("archived", "prompt")
    del eager_stamp_source
    window_task = manager.create("typed during the load window", "prompt")
    manager.load()
    assert window_task.id in manager.tasks
    archived_seq = manager.tasks[archived.id].queue_seq
    window_seq = manager.tasks[window_task.id].queue_seq
    assert archived_seq is not None and window_seq is not None
    assert archived_seq < window_seq


def test_reload_new_tasks_before_load_is_noop(project_root: Path) -> None:
    manager = _deferred_manager(project_root)
    eager = TaskManager(project_root)
    eager.create("external", "prompt")
    del eager
    assert manager.reload_new_tasks() == []
    manager.load()
    assert "external" in {t.title for t in manager.tasks.values()}


def test_metadata_epoch_tracks_metadata_changes(project_root: Path) -> None:
    manager = TaskManager(project_root)
    storage = manager.storage
    task = manager.create("t", "prompt")
    epoch_after_create = storage.metadata_epoch

    task.title = "renamed"
    storage.save_metadata(task)
    assert storage.metadata_epoch > epoch_after_create

    # A transcript-only flush must NOT count: the sidebar poll stays
    # quiet while a run streams into a sidecar.
    task._transcripts.loaded = True
    task._transcripts.messages = []
    epoch_before_flush = storage.metadata_epoch
    storage._flush_transcript(task)
    assert storage.metadata_epoch == epoch_before_flush

    epoch_before_delete = storage.metadata_epoch
    storage.delete(task.id)
    assert storage.metadata_epoch > epoch_before_delete


def test_sidebar_fingerprint_is_cheap_and_change_sensitive(
    project_root: Path,
) -> None:
    """The fingerprint no longer sweeps the tasks: identical state gives
    an identical tuple; a saved mutation (epoch bump) changes it."""
    app = DrCodeApp(project_root)
    a = app.manager.create("one", "prompt")
    b = app.manager.create("two", "prompt")
    fp = app._sidebar_fingerprint()
    assert app._sidebar_fingerprint() == fp
    a.title = "changed"
    app.manager.storage.save(a)
    assert app._sidebar_fingerprint() != fp
    # Session state is carried explicitly: the viewed marker moves it.
    fp2 = app._sidebar_fingerprint()
    app.viewed_id = b.id
    assert app._sidebar_fingerprint() != fp2


async def test_completed_section_auto_collapses_and_lazily_mounts(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A COMPLETED section over the threshold starts collapsed; expanding
    mounts a capped slice plus a "show all" row; clicking it mounts the
    rest. Section DATA (the top-task pick) ignores the collapse."""
    monkeypatch.setattr(sidebar_view, "COMPLETED_COLLAPSE_THRESHOLD", 2)
    app = DrCodeApp(project_root)
    # The visibility cap the app captured at import time; shrink it here.
    app._completed_limit = 2
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    for i, name in enumerate(["old", "mid", "new"]):
        t = app.manager.create(name, "prompt")
        t.updated_at = base + timedelta(seconds=i)
        t.status = TaskStatus.COMPLETED
        app.manager.storage.save(t)
    queued = app.manager.create("queued", "prompt")

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")  # skip the setup dialog
        await pilot.pause()
        # Startup pick ignores the collapsed section: the queued task is
        # the only non-completed one.
        assert app.viewed_id == queued.id

        def completed_rows() -> list[TaskItem]:
            return [
                item
                for item in app.sidebar.children
                if isinstance(item, TaskItem) and item.section == "COMPLETED"
            ]

        def all_rows() -> list[TaskItem]:
            return [
                item
                for item in app.sidebar.children
                if isinstance(item, TaskItem) and item.task_id is not None
            ]

        # Collapsed by default: the header alone, no completed rows —
        # the queued task's row is the only task row mounted.
        assert len(completed_rows()) == 1
        assert [i.task_id for i in all_rows() if i.task_id] == [queued.id]

        # Expand: capped to 2 rows + the "show all" row.
        app.toggle_sidebar_section("COMPLETED")
        await pilot.pause()
        mounted = [i.task_id for i in all_rows() if i.task_id]
        more = [
            i
            for i in app.sidebar.children
            if isinstance(i, TaskItem) and i.more_section
        ]
        assert len(mounted) == 3  # queued + 2 completed
        assert len(more) == 1

        # "Show all": the cap drops for the session.
        app._show_all_section_rows("COMPLETED")
        await pilot.pause()
        mounted = [i.task_id for i in all_rows() if i.task_id]
        assert len(mounted) == 4  # queued + all 3 completed
        assert not any(
            isinstance(i, TaskItem) and i.more_section for i in app.sidebar.children
        )


async def test_deferred_app_startup_finishes_after_load(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full app path over a 'large' archive: the first frame paints with
    the load still in flight, the archive-load worker finishes it and the
    startup tail (initial view pick, sidebar fill) runs afterwards."""
    monkeypatch.setattr(TaskManager, "LARGE_ARCHIVE_FILE_THRESHOLD", 0)
    _make_tasks(project_root, 2)
    app = DrCodeApp(project_root)
    assert not app.manager.loaded
    assert app._archive_load_deferred
    async with app.run_test() as pilot:
        await wait_until(lambda: app._startup_finished, pilot=pilot)
        assert app.manager.loaded
        assert len(app.manager.tasks) == 2
        # The initial view pick ran against the LOADED archive.
        assert app.viewed_id is not None
        assert app.viewed_id in app.manager.tasks
