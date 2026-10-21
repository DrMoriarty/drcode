"""Notifications feature: project settings, the ntfy.sh notifier, the
status-change observer wiring of the TaskManager and the settled-pipeline
dedup."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest

from drcode.models import Task, TaskKind, TaskStatus
from drcode.notifications import Notifier, Priority
from drcode.settings import ProjectSettings, ProjectSettingsStore
from drcode.task_manager import TaskManager


def _capture_deliver(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace the HTTP delivery with an in-memory recorder."""
    sent: list[dict] = []

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

    def fake_post(url: str, json: dict, timeout: float) -> _FakeResponse:
        sent.append(json)
        return _FakeResponse()

    import drcode.notifications as mod

    monkeypatch.setattr(mod.httpx, "post", fake_post)
    return sent


def _run_notifier_now(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make notify() deliver synchronously (no daemon threads in tests)."""
    import drcode.notifications as mod

    monkeypatch.setattr(mod.threading, "Thread", _ImmediateThread)


class _ImmediateThread:
    """threading.Thread stand-in that runs the target synchronously."""

    def __init__(
        self,
        target: Callable[..., None],
        args: tuple = (),
        kwargs: dict | None = None,
        **_ignored: object,
    ) -> None:
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self) -> None:
        self._target(*self._args, **self._kwargs)


# -- project settings -------------------------------------------------------


def test_settings_defaults_are_off(tmp_path: Path) -> None:
    settings = ProjectSettingsStore(tmp_path).load()
    assert settings.notifications_enabled is False
    assert settings.ntfy_topic == ""


def test_settings_roundtrip(tmp_path: Path) -> None:
    store = ProjectSettingsStore(tmp_path)
    settings = store.load()
    settings.notifications_enabled = True
    settings.ntfy_topic = "my-secret-topic"
    store.save(settings)
    raw = json.loads((tmp_path / ".drcode" / "settings.json").read_text())
    assert raw["ntfy_topic"] == "my-secret-topic"
    reloaded = ProjectSettingsStore(tmp_path).load()
    assert reloaded.notifications_enabled is True
    assert reloaded.ntfy_topic == "my-secret-topic"


# -- notifier delivery -------------------------------------------------------


def test_disabled_or_topicless_notifier_stays_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_notifier_now(monkeypatch)
    sent = _capture_deliver(monkeypatch)
    settings = ProjectSettings()
    notifier = Notifier(lambda: settings)

    notifier.notify("t", "m")  # disabled
    settings.notifications_enabled = True
    notifier.notify("t", "m")  # enabled but no topic
    settings.ntfy_topic = "  "  # whitespace-only topic
    notifier.notify("t", "m")
    assert sent == []


def test_notifier_publishes_json_to_ntfy(monkeypatch: pytest.MonkeyPatch) -> None:
    _run_notifier_now(monkeypatch)
    sent = _capture_deliver(monkeypatch)
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="topic1")
    notifier = Notifier(lambda: settings, ntfy_base="https://ntfy.sh/")

    notifier.notify("title", "message", tags=["bell"], priority=Priority.HIGH)
    assert len(sent) == 1
    payload = sent[0]
    assert payload["topic"] == "topic1"
    assert payload["title"] == "title"
    assert payload["message"] == "message"
    assert payload["tags"] == ["bell"]
    assert payload["priority"] == 4  # ntfy JSON publish: priority must be an int


def test_notifier_delivery_failure_is_swallowed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_notifier_now(monkeypatch)
    import drcode.notifications as mod

    def failing_post(url: str, json: dict, timeout: float) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(mod.httpx, "post", failing_post)
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="t")
    errors: list[str] = []
    notifier = Notifier(lambda: settings, on_error=errors.append, retry_interval=3600)

    notifier.notify("t", "m")  # synchronous via _ImmediateThread
    assert len(errors) == 1 and "boom" in errors[0]
    # Fallback: never raises, reports to stderr. (The failed notification
    # itself stays pending — a second notify() would only queue behind it,
    # so the queue is emptied here to provoke a fresh delivery attempt.)
    notifier._on_error = None
    notifier._pending.clear()
    notifier.notify("t", "m")
    assert "drcode notifications" in capsys.readouterr().err


# -- deferred delivery (pending queue) ----------------------------------------

_sent: list[dict] = []


def _failing_post(url: str, json: dict, timeout: float) -> object:
    raise RuntimeError("network unreachable")


def _capture_post(url: str, json: dict, timeout: float) -> object:
    _sent.append(json)

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

    return _FakeResponse()


def _notifier(
    settings: ProjectSettings,
    project_root: Path | None = None,
    retry_interval: float = 3600.0,
) -> Notifier:
    return Notifier(
        lambda: settings,
        project_root=project_root,
        retry_interval=retry_interval,
    )


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll a condition set by a background thread (the retry waiter runs
    in a real thread even when delivery is made synchronous)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _read_pending_file(project_root: Path) -> list[dict]:
    path = project_root / ".drcode" / "pending_notifications.json"
    return json.loads(path.read_text())


def test_failed_send_is_persisted_with_timestamp(
    monkeypatch: pytest.MonkeyPatch, project_root: Path
) -> None:
    _run_notifier_now(monkeypatch)
    import drcode.notifications as mod

    def failing_post(url: str, json: dict, timeout: float) -> object:
        raise TimeoutError("network unreachable")

    monkeypatch.setattr(mod.httpx, "post", failing_post)
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="topic1")
    notifier = _notifier(settings, project_root)

    before = datetime.now().astimezone()
    notifier.notify("title", "message", tags=["bell"], priority=Priority.HIGH)

    raw = _read_pending_file(project_root)
    assert len(raw) == 1
    entry = raw[0]
    assert entry["topic"] == "topic1"
    assert entry["title"] == "title"
    assert entry["message"] == "message"
    assert entry["tags"] == ["bell"]
    assert entry["priority"] == 4
    # The timestamp of the moment the notification arose is stored.
    created = datetime.fromisoformat(entry["created_at"])
    assert abs((created - before).total_seconds()) < 60
    # The pending copy keeps the original text; the delay note is added
    # only when the notification is actually re-sent.
    assert "deferred" not in entry["message"]


def test_new_notifications_queue_behind_unsent_ones(
    monkeypatch: pytest.MonkeyPatch, project_root: Path
) -> None:
    """While an older notification is unsent, new ones are never attempted
    — they only join the tail of the queue, in arrival order."""
    _run_notifier_now(monkeypatch)
    import drcode.notifications as mod

    def failing_post(url: str, json: dict, timeout: float) -> object:
        raise RuntimeError("down")

    monkeypatch.setattr(mod.httpx, "post", failing_post)
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="t")
    notifier = _notifier(settings, project_root)

    notifier.notify("first", "m1")
    notifier.notify("second", "m2")
    notifier.notify("third", "m3")

    raw = _read_pending_file(project_root)
    assert [e["title"] for e in raw] == ["first", "second", "third"]


def test_retry_delivers_in_order_with_delay_note_then_clears_file(
    monkeypatch: pytest.MonkeyPatch, project_root: Path
) -> None:
    _run_notifier_now(monkeypatch)
    import drcode.notifications as mod

    monkeypatch.setattr(mod.httpx, "post", _failing_post)
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="t")
    notifier = _notifier(settings, project_root, retry_interval=0.05)

    notifier.notify("first", "m1")
    notifier.notify("second", "m2")

    # Network comes back: the waiter retries after retry_interval and both
    # notifications are delivered in their original order.
    monkeypatch.setattr(mod.httpx, "post", _capture_post)
    _sent.clear()
    assert _wait_until(lambda: len(notifier._pending) == 0)
    assert [e["title"] for e in _sent] == ["first", "second"]
    # Both re-sends say they are late and when each notification arose.
    assert "deferred" in _sent[0]["message"]
    assert "delivered late" in _sent[0]["message"]
    assert "deferred" in _sent[1]["message"]
    # Success: the file is removed.
    assert not (project_root / ".drcode" / "pending_notifications.json").exists()


def test_startup_flushes_saved_pending_file(
    monkeypatch: pytest.MonkeyPatch, project_root: Path
) -> None:
    """At startup a saved pending file is loaded and delivered in order."""
    _run_notifier_now(monkeypatch)
    import drcode.notifications as mod

    monkeypatch.setattr(mod.httpx, "post", _failing_post)
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="t")
    notifier = _notifier(settings, project_root)
    notifier.notify("first", "m1")
    notifier.notify("second", "m2")

    # "Restart": a fresh notifier over the same project root, healthy
    # network — the saved notifications must go out before anything new.
    monkeypatch.setattr(mod.httpx, "post", _capture_post)
    _sent.clear()
    restarted = _notifier(settings, project_root)
    assert [e["title"] for e in _sent] == ["first", "second"]
    assert restarted._pending == []
    assert not (project_root / ".drcode" / "pending_notifications.json").exists()

    # And a new notification is delivered right after the backlog.
    restarted.notify("third", "m3")
    assert [e["title"] for e in _sent] == ["first", "second", "third"]
    assert "deferred" not in _sent[2]["message"]  # it was not delayed


def test_startup_preserves_arisen_at_timestamp(
    monkeypatch: pytest.MonkeyPatch, project_root: Path
) -> None:
    _run_notifier_now(monkeypatch)
    import drcode.notifications as mod

    monkeypatch.setattr(mod.httpx, "post", _failing_post)
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="t")
    notifier = _notifier(settings, project_root)
    before = datetime.now().astimezone()
    notifier.notify("first", "m1")

    monkeypatch.setattr(mod.httpx, "post", _capture_post)
    _sent.clear()
    _notifier(settings, project_root)
    assert "deferred" in _sent[0]["message"]
    # The delay note carries the original arisen-at moment, not the
    # restart moment.
    stamp = _sent[0]["message"].split("arose at ")[1].split(",")[0]
    restored = datetime.strptime(stamp, "%d.%m.%Y %H:%M:%S").astimezone()
    assert abs((restored - before).total_seconds()) < 60


def test_corrupt_pending_file_moves_aside(
    monkeypatch: pytest.MonkeyPatch, project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pending_dir = project_root / ".drcode"
    pending_dir.mkdir()
    (pending_dir / "pending_notifications.json").write_text("{not json")

    _run_notifier_now(monkeypatch)
    import drcode.notifications as mod

    monkeypatch.setattr(mod.httpx, "post", _capture_post)
    _sent.clear()
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="t")
    notifier = _notifier(settings, project_root)
    assert notifier._pending == []
    assert (pending_dir / "pending_notifications.json.bak").exists()
    assert "drcode notifications" in capsys.readouterr().err
    # The app keeps working: a fresh notification is delivered normally.
    notifier.notify("t", "m")
    assert [e["title"] for e in _sent] == ["t"]


def test_in_memory_queue_without_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a project root the queue still works, only in memory."""
    _run_notifier_now(monkeypatch)
    import drcode.notifications as mod

    monkeypatch.setattr(mod.httpx, "post", _failing_post)
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="t")
    notifier = _notifier(settings, retry_interval=0.05)
    notifier.notify("first", "m1")
    notifier.notify("second", "m2")
    assert [e.title for e in notifier._pending] == ["first", "second"]

    monkeypatch.setattr(mod.httpx, "post", _capture_post)
    _sent.clear()
    assert _wait_until(lambda: len(notifier._pending) == 0)
    assert [e["title"] for e in _sent] == ["first", "second"]


def test_title_includes_project_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    _run_notifier_now(monkeypatch)
    sent = _capture_deliver(monkeypatch)
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="t")
    notifier = Notifier(lambda: settings, project="myproject")

    notifier.task_status(_task(), TaskStatus.QUEUED, TaskStatus.ACTIVE, "system")
    assert len(sent) == 1
    assert sent[0]["title"] == "DrCode (myproject): task active"

    # Without a project the plain "DrCode:" prefix is kept.
    notifier = Notifier(lambda: settings)
    notifier.task_status(_task(), TaskStatus.QUEUED, TaskStatus.ACTIVE, "system")
    assert sent[1]["title"] == "DrCode: task active"


# -- event semantics ----------------------------------------------------------


def _task(kind: TaskKind = TaskKind.FEATURE) -> Task:
    return Task(title="Do things", kind=kind)


def test_task_status_skips_user_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    _run_notifier_now(monkeypatch)
    sent = _capture_deliver(monkeypatch)
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="t")
    notifier = Notifier(lambda: settings)
    task = _task()

    # User-initiated transitions are not reported (the user just caused it).
    notifier.task_status(task, TaskStatus.ACTIVE, TaskStatus.COMPLETED, "user")
    assert sent == []
    # System transitions are.
    notifier.task_status(task, TaskStatus.ACTIVE, TaskStatus.COMPLETED, "system")
    assert len(sent) == 1
    assert "Do things" in sent[0]["message"]
    # Same-status "changes" are ignored.
    notifier.task_status(task, TaskStatus.QUEUED, TaskStatus.QUEUED, "system")
    assert len(sent) == 1


def test_completion_verdict_matches_verification_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the verdict in the completion notification must follow
    the actual verification label (the enum values are Russian strings —
    comparing them against "passed" silently marked every passed task as
    ПРОВАЛЕНО)."""
    from drcode.models import VerificationStatus

    _run_notifier_now(monkeypatch)
    sent = _capture_deliver(monkeypatch)
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="t")
    notifier = Notifier(lambda: settings)

    passed = _task()
    passed.verification = VerificationStatus.PASSED
    notifier.task_status(passed, TaskStatus.TESTING, TaskStatus.COMPLETED, "system")
    assert "testing passed" in sent[0]["message"]
    assert "FAILED" not in sent[0]["message"]

    failed = _task()
    failed.verification = VerificationStatus.FAILED
    notifier.task_status(failed, TaskStatus.TESTING, TaskStatus.COMPLETED, "system")
    assert "testing FAILED" in sent[1]["message"]

    # No verdict at all (completed without the tester): no verdict suffix.
    plain = _task()
    notifier.task_status(plain, TaskStatus.ACTIVE, TaskStatus.COMPLETED, "system")
    assert "testing" not in sent[2]["message"]


def test_github_sync_notification(monkeypatch: pytest.MonkeyPatch) -> None:
    _run_notifier_now(monkeypatch)
    sent = _capture_deliver(monkeypatch)
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="t")
    Notifier(lambda: settings).github_sync(imported=3, skipped=2)
    assert len(sent) == 1
    assert "3" in sent[0]["message"] and "2" in sent[0]["message"]


def test_pipeline_idle_dedup_and_priming(monkeypatch: pytest.MonkeyPatch) -> None:
    _run_notifier_now(monkeypatch)
    sent = _capture_deliver(monkeypatch)
    settings = ProjectSettings(notifications_enabled=True, ntfy_topic="t")
    notifier = Notifier(lambda: settings)

    # Priming call (first evaluation after startup): recorded, not sent.
    notifier.pipeline_idle(blocked=False, detail="", prime=True)
    assert sent == []
    # Same settled state again: still nothing.
    notifier.pipeline_idle(blocked=False, detail="")
    assert sent == []
    # Transition empty -> blocked -> empty: two notifications.
    notifier.pipeline_idle(blocked=True, detail="Reasons: on hold: 1.")
    notifier.pipeline_idle(blocked=False, detail="")
    assert len(sent) == 2
    assert "on hold" in sent[0]["message"]
    assert "empty" in sent[1]["title"]


# -- TaskManager observer wiring -----------------------------------------------


def test_manager_fires_status_events_with_origin(manager: TaskManager) -> None:
    events: list[tuple[TaskStatus, TaskStatus, str]] = []
    manager.add_status_listener(
        lambda t, old, new, origin: events.append((old, new, origin))
    )

    task = manager.create("", "build it")
    assert events == []  # creation is not a status transition

    manager.mark_done(task.id)  # default origin: system
    assert events == [(TaskStatus.QUEUED, TaskStatus.TESTING, "system")]
    events.clear()

    manager.send(task.id, "one more round")  # user-initiated re-queue
    assert events == [(TaskStatus.TESTING, TaskStatus.QUEUED, "user")]
    events.clear()

    promoted = manager.promote_next()
    assert promoted is not None and promoted.id == task.id
    assert events == [(TaskStatus.QUEUED, TaskStatus.ACTIVE, "system")]
    events.clear()

    manager.cancel(task.id)  # user-initiated completion
    assert events == [(TaskStatus.ACTIVE, TaskStatus.COMPLETED, "user")]
    events.clear()

    # The tester's verdict: a TESTING task completed by set_verification.
    verified = manager.create("", "verify me")
    manager.mark_done(verified.id)
    events.clear()
    manager.set_verification(verified.id, success=True)
    assert events == [(TaskStatus.TESTING, TaskStatus.COMPLETED, "system")]


def test_manager_set_verification(manager: TaskManager) -> None:
    from drcode.models import VerificationStatus

    task = manager.create("", "build it")
    manager.mark_done(task.id)
    manager.set_verification(task.id, success=False)
    stored = manager.get(task.id)
    assert stored.status is TaskStatus.COMPLETED
    assert stored.verification is VerificationStatus.FAILED
    assert stored.completed_at is not None


def test_broken_listener_never_breaks_the_manager(manager: TaskManager) -> None:
    def broken(*args: object) -> None:
        raise RuntimeError("listener exploded")

    manager.add_status_listener(broken)
    task = manager.create("", "build it")
    manager.mark_done(task.id)  # must not raise
    assert manager.get(task.id).status is TaskStatus.TESTING
