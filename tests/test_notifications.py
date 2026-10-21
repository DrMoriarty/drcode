"""Notifications feature: project settings, the ntfy.sh notifier, the
status-change observer wiring of the TaskManager and the settled-pipeline
dedup."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from drcode.models import Task, TaskKind, TaskStatus
from drcode.notifications import Notifier, Priority
from drcode.settings import ProjectSettings, ProjectSettingsStore
from drcode.task_manager import TaskManager


def _capture_deliver(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace the HTTP delivery with an in-memory recorder."""
    sent: list[dict] = []

    def fake_post(url: str, json: dict, timeout: float) -> object:
        sent.append(json)
        return object()

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

    notifier.notify("заголовок", "сообщение", tags=["bell"], priority=Priority.HIGH)
    assert len(sent) == 1
    payload = sent[0]
    assert payload["topic"] == "topic1"
    # Non-ASCII titles/messages must survive (JSON publish, not headers).
    assert payload["title"] == "заголовок"
    assert payload["message"] == "сообщение"
    assert payload["tags"] == ["bell"]
    assert payload["priority"] == "high"


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
    notifier = Notifier(lambda: settings, on_error=errors.append)

    notifier.notify("t", "m")  # synchronous via _ImmediateThread
    assert len(errors) == 1 and "boom" in errors[0]
    # Fallback: never raises, reports to stderr.
    notifier._on_error = None
    notifier.notify("t", "m")
    assert "drcode notifications" in capsys.readouterr().err


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
    notifier.pipeline_idle(blocked=True, detail="Причины: на удержании (hold): 1.")
    notifier.pipeline_idle(blocked=False, detail="")
    assert len(sent) == 2
    assert "удержании" in sent[0]["message"]
    assert "пуста" in sent[1]["title"]


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
