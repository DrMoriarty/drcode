"""Push notifications about important pipeline events via ntfy.sh.

A separate module on purpose: the TUI (and any future front-end) only has
to feed it events — the module decides *whether* anything is sent at all
(the feature is configured per project in `<project>/.drcode/settings.json`:
`notifications_enabled` + `ntfy_topic`), formats the message and delivers it
without ever disturbing the caller:

- delivery happens in a short-lived daemon thread (a hanging ntfy.sh must
  never stall the UI event loop or an agent run; the sender may be called
  from worker threads — e.g. from the tester's `set_verification` tool —
  where no event loop is reachable anyway),
- every failure is swallowed and logged (stderr / an optional callback):
  a broken notification channel must never break the pipeline.

Deferred delivery: when a notification cannot be sent right away (the
network is down, the request times out), it is not lost — it is persisted
in `<project>/.drcode/pending_notifications.json` together with the
timestamp of the moment it arose, and re-delivery is retried every
`RETRY_INTERVAL` (15 minutes). The strict rules of the pending queue:

- notifications are always delivered in the order they arose in the system
  (a single FIFO queue);
- while at least one older notification is unsent, no new notification is
  attempted — a new one can only join the tail of the queue;
- a notification re-sent after a delay says so in its message, along with
  the moment it originally arose;
- a successfully sent notification is removed from the file;
- at startup a saved pending file is loaded and its notifications are
  delivered (in order) before anything new.

Three kinds of events are reported:

1. task status changes — except those directly initiated by the user who
   is sitting at the computer right now (typing a message, cancelling,
   holding/resuming, pressing Done): the origin ("user"/"system") is
   supplied by the caller (see `TaskManager.add_status_listener`);
2. GitHub Issues sync cycles — with the counts of imported/skipped issues;
3. pipeline settled states — the queue is empty (everything processed),
   or tasks remain but none can be taken into development (all on hold,
   awaiting clarification or failed this session). These are deduplicated:
   a notification is sent only when the settled state *changes*, so a
   dozen `_advance()` calls never spam a dozen identical messages.
"""

from __future__ import annotations

import enum
import json
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, ValidationError

from .models import VerificationStatus

if TYPE_CHECKING:
    from .models import Task, TaskStatus
    from .settings import ProjectSettings

NTFY_BASE = "https://ntfy.sh"

# Delivery timeout: ntfy.sh answers in well under a second normally; the
# generous ceiling only matters for a degraded network — and even then the
# send runs in its own daemon thread.
SEND_TIMEOUT = 10.0

# How often delivery of the pending (deferred) notifications is retried.
RETRY_INTERVAL = 15 * 60.0

# Name of the deferred-notifications file inside `<project>/.drcode/`.
PENDING_FILENAME = "pending_notifications.json"

# Captured at import time on purpose: the retry waiter must stay a real
# background thread even in tests that make the delivery itself
# synchronous by monkeypatching `threading.Thread` — otherwise the
# 15-minute wait would run inline and hang the test.
_THREAD = threading.Thread


class Priority(int, enum.Enum):
    """ntfy.sh message priorities (the ones worth distinguishing here).

    Numeric on purpose: the JSON publish API accepts `priority` only as an
    int (1..5) — a string like "high" makes ntfy.sh answer 400
    "request body must be valid JSON"."""

    MIN = 1
    DEFAULT = 3
    HIGH = 4
    URGENT = 5


class PendingNotification(BaseModel):
    """A notification that could not be delivered immediately.

    `created_at` is the moment the event arose in the system (not the
    moment of the failed send) — it is stamped once, survives restarts in
    the pending file and is shown to the recipient when the notification
    is finally delivered with a delay. `topic` is snapshotted at creation
    time so a settings change between the failed attempt and the retry
    cannot silently redirect the old notification elsewhere."""

    topic: str
    title: str
    message: str
    tags: list[str] = []
    priority: int = Priority.DEFAULT.value
    created_at: datetime
    # Set after the first failed delivery attempt: the re-send must say
    # that it is arriving late (see _delivery_message).
    delayed: bool = False


def _status_text(status: TaskStatus, task: Task) -> str:
    """Human-readable one-liner for a status transition (English, matching
    the rest of the user-facing strings of the app)."""
    if status.value == "active":
        return "taken into development"
    if status.value == "testing":
        return "development finished — awaiting testing"
    if status.value == "completed":
        verdict = ""
        if task.verification is not None:
            verdict = (
                " (testing passed)"
                if task.verification is VerificationStatus.PASSED
                else " (testing FAILED)"
            )
        return f"completed{verdict}"
    if status.value == "clarification":
        return "awaiting an answer to a clarifying question"
    if status.value == "hold":
        return "on hold"
    if status.value == "queued":
        return "returned to the queue"
    return status.value


class Notifier:
    """Event hub: the app feeds pipeline events in, the module filters,
    formats and delivers them to ntfy.sh.

    `settings_getter` is read on every event, so a settings change made in
    the dialog applies immediately without rebuilding anything.

    `project_root` enables the deferred-delivery queue (see the module
    docstring): failed sends are persisted in `<project>/.drcode/` and
    retried every 15 minutes. Without it the queue still works, but only
    in memory (it dies with the process).
    """

    def __init__(
        self,
        settings_getter: Callable[[], ProjectSettings],
        *,
        project: str = "",
        project_root: Path | None = None,
        ntfy_base: str = NTFY_BASE,
        retry_interval: float = RETRY_INTERVAL,
        on_error: Callable[[str], None] | None = None,
        speaker: Callable[[str], None] | None = None,
    ) -> None:
        self._settings_getter = settings_getter
        # Name of the project folder the app works in: prepended to every
        # title ("DrCode (myproject): ...") so notifications from several
        # drcode instances sharing one ntfy topic stay distinguishable.
        self.project = project
        self.ntfy_base = ntfy_base.rstrip("/")
        self._retry_interval = retry_interval
        self._on_error = on_error
        # Voice channel (TTS announcements): a callback that synthesizes
        # and plays a phrase (drcode.voice.TtsSpeaker.speak). Independent
        # of ntfy: a broken/absent speaker never affects the pushes and
        # vice versa. Each event decides by its own project-settings
        # checkbox whether it is voiced at all.
        self._speaker = speaker
        # Deferred-delivery queue: strict FIFO, guarded by _lock. Every
        # delivery attempt (immediate or retried) goes through it, which
        # guarantees both the original ordering and that nothing new is
        # ever attempted while an older notification is unsent.
        self._lock = threading.Lock()
        self._pending: list[PendingNotification] = []
        self._flushing = False
        self._retry_waiter: _THREAD | None = None
        self._pending_path = (
            project_root / ".drcode" / PENDING_FILENAME if project_root else None
        )
        self._pending = self._load_pending()
        # Last settled pipeline state already reported (None = a run is
        # active / nothing settled yet): dedup of the empty/blocked
        # notifications — see pipeline_idle().
        self._settled_state: str | None = None
        # Startup: a saved pending file must be delivered before anything
        # new — the freshly loaded notifications are flushed right away.
        if self._pending:
            with self._lock:
                self._flushing = True
            self._spawn_flush()

    # -- low-level delivery ---------------------------------------------------

    def _title(self, text: str) -> str:
        """Title with the app prefix — and, when known, the project folder
        name, so instances working in different folders (but publishing to
        one topic) can be told apart."""
        if self.project:
            return f"DrCode ({self.project}): {text}"
        return f"DrCode: {text}"

    def _topic(self) -> str:
        """Configured ntfy topic, or "" when notifications are off."""
        settings = self._settings_getter()
        if not settings.notifications_enabled:
            return ""
        return settings.ntfy_topic.strip()

    def notify(
        self,
        title: str,
        message: str,
        *,
        tags: Sequence[str] = (),
        priority: Priority = Priority.DEFAULT,
    ) -> None:
        """Fire-and-forget send. Safe to call from any thread (worker
        threads included): the delivery runs in its own daemon thread.

        The notification always joins the tail of the FIFO queue: if older
        notifications are still unsent, this one is only stored (and
        persisted, when a project root is known) — never attempted before
        them."""
        topic = self._topic()
        if not topic:
            return
        entry = PendingNotification(
            topic=topic,
            title=title,
            message=message,
            tags=list(tags),
            priority=priority.value,
            created_at=datetime.now().astimezone(),
        )
        with self._lock:
            self._pending.append(entry)
            kick = not self._flushing
            if kick:
                self._flushing = True
            if self._pending[:-1]:
                # Older notifications are still unsent: persist the new
                # tail right away so nothing is lost, but do not attempt
                # any send — the running flush or the retry waiter owns
                # the delivery now.
                self._persist_locked()
        if kick:
            self._spawn_flush()

    def _delivery_message(self, entry: PendingNotification) -> str:
        """Message text for the wire: a deferred re-send announces the
        delay and the moment the notification originally arose."""
        if not entry.delayed:
            return entry.message
        return (
            f"{entry.message}\n"
            f"⏰ deferred notification: arose at "
            f"{entry.created_at:%d.%m.%Y %H:%M:%S}, delivered late"
        )

    def _attempt_send(self, entry: PendingNotification) -> bool:
        """One delivery attempt. Never raises: True = delivered, False =
        the notification stays pending (and will be retried)."""
        try:
            response = httpx.post(
                self.ntfy_base,
                # JSON publish (POST to the root): unlike header-based
                # publishing it carries non-ASCII titles/messages safely.
                json={
                    "topic": entry.topic,
                    "title": entry.title,
                    "message": self._delivery_message(entry),
                    "tags": entry.tags,
                    "priority": entry.priority,
                },
                timeout=SEND_TIMEOUT,
            )
            response.raise_for_status()
            return True
        except Exception as e:  # noqa: BLE001 - notifications must never crash anything
            self._report_error(f"{type(e).__name__}: {e}")
            return False

    def _spawn_flush(self) -> None:
        """Run the flush loop in a daemon thread (a hanging ntfy.sh must
        never stall the caller)."""
        threading.Thread(
            target=self._flush_owned, daemon=True, name="drcode-notify"
        ).start()

    def _flush_owned(self) -> None:
        """Deliver pending notifications strictly in order until the queue
        is empty or a send fails. Exactly one flush loop runs at a time
        (the `_flushing` flag): everyone else just queues and relies on
        the running loop — or the retry waiter — to pick the tail up."""
        retry_scheduled = False
        try:
            while True:
                with self._lock:
                    entry = self._pending[0] if self._pending else None
                if entry is None:
                    return
                if not self._attempt_send(entry):
                    # Keep the notification (with the arisen-at timestamp)
                    # and everything queued behind it; retry later. All of
                    # them become "delayed": nothing behind the stuck head
                    # can be delivered on time anymore — the queue is
                    # strict FIFO.
                    with self._lock:
                        for pending in self._pending:
                            pending.delayed = True
                        self._persist_locked()
                        self._schedule_retry_locked()
                    retry_scheduled = True
                    return
                with self._lock:
                    self._pending.pop(0)
                    # Success: the sent notification leaves the file.
                    self._persist_locked()
        finally:
            with self._lock:
                self._flushing = False
                # A notification may have been queued while this loop was
                # finishing; if nothing owns the delivery now, take it
                # again. After a failed send the retry waiter owns it.
                restart = bool(self._pending) and not retry_scheduled
                if restart:
                    self._flushing = True
            if restart:
                self._spawn_flush()

    def _schedule_retry_locked(self) -> None:
        """Arrange a retry of the whole pending queue in `retry_interval`
        seconds. Caller holds `_lock`. At most one waiter sleeps at a
        time — an existing one already covers everything queued now."""
        if self._retry_waiter is not None:
            return

        def _wait_and_flush() -> None:
            time.sleep(self._retry_interval)
            with self._lock:
                self._retry_waiter = None
                if self._flushing or not self._pending:
                    return
                self._flushing = True
            self._flush_owned()

        waiter = _THREAD(
            target=_wait_and_flush, daemon=True, name="drcode-notify-retry"
        )
        self._retry_waiter = waiter
        waiter.start()

    # -- pending-queue persistence ---------------------------------------------

    def _load_pending(self) -> list[PendingNotification]:
        """Read the deferred queue from disk (startup recovery). A corrupt
        file is moved aside (`.bak`), never fatal."""
        if self._pending_path is None:
            return []
        try:
            raw = self._pending_path.read_text()
        except FileNotFoundError:
            return []
        except OSError as e:
            self._report_error(f"pending notifications unreadable: {e}")
            return []
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise TypeError("pending notifications: expected a list")
            return [PendingNotification.model_validate(item) for item in data]
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            backup = self._pending_path.with_name(self._pending_path.name + ".bak")
            try:
                self._pending_path.replace(backup)
            except OSError:
                pass
            self._report_error(
                f"pending notifications file corrupted, moved to {backup.name}: {e}"
            )
            return []

    def _persist_locked(self) -> None:
        """Write the whole pending queue atomically. An empty queue
        removes the file. Caller holds `_lock`."""
        if self._pending_path is None:
            return
        try:
            if not self._pending:
                self._pending_path.unlink(missing_ok=True)
                return
            self._pending_path.parent.mkdir(parents=True, exist_ok=True)
            body = ",\n".join(e.model_dump_json() for e in self._pending)
            fd, tmp = tempfile.mkstemp(dir=self._pending_path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(f"[\n{body}\n]\n")
                os.replace(tmp, self._pending_path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except OSError as e:
            self._report_error(f"pending notifications not saved: {e}")

    def _report_error(self, text: str) -> None:
        if self._on_error is not None:
            try:
                self._on_error(text)
                return
            except Exception as cb_error:  # noqa: BLE001 - the callback is best-effort
                print(
                    f"drcode notifications: error callback failed: {cb_error}",
                    file=sys.stderr,
                )
        print(f"drcode notifications: {text}", file=sys.stderr)

    # -- voice announcements (TTS) ----------------------------------------------

    def _speak(self, enabled: bool, text: str) -> None:
        """Voice one event phrase through the speaker callback (when the
        event's checkbox is on). Best-effort: a failed/broken speaker is
        swallowed, exactly like a failed push."""
        if not enabled or self._speaker is None:
            return
        try:
            self._speaker(text)
        except Exception as e:  # noqa: BLE001 - announcements never break anything
            self._report_error(f"voice announcement failed: {e}")

    def bug_report_created(self, task: Task, origin: str) -> None:
        """A bug report task was created (TaskManager creation listener).
        Voiced under the "tester failure" checkbox — together with the
        re-queue phrase it covers both halves of the tester-failure event.
        Only agent-created reports (origin "system") are voiced: a bug the
        user typed by hand is something they just watched happen."""
        if origin != "user" and task.kind.value == "bug":
            self._speak(
                self._settings_getter().voice_tts_tester_failed,
                f"Заведён баг-репорт: {task.title}",
            )

    # -- pipeline events --------------------------------------------------------

    def task_status(
        self,
        task: Task,
        old: TaskStatus,
        new: TaskStatus,
        origin: str = "system",
    ) -> None:
        """A task changed its status. User-initiated transitions (the user
        is at the computer and has just pressed/typed whatever caused the
        change) are not reported — they already saw it happen."""
        if origin == "user" or old is new:
            return
        # Voice announcements (each behind its own project-settings
        # checkbox; they ride on the same system-origin transitions the
        # pushes use, so no pipeline logic is duplicated here):
        # - a completion; a FAILED verdict is voiced as a tester failure
        #   instead (the bug report, when the tester created one, is
        #   voiced separately through bug_report_created);
        # - TESTING → QUEUED with system origin: the tester failed the
        #   task and returned it for another development round
        #   (requeue_failed_fix / requeue_review_rejection — the user
        #   path re-queues with origin "user" via send()).
        if new.value == "completed":
            failed = task.verification is VerificationStatus.FAILED
            self._speak(
                (
                    self._settings_getter().voice_tts_tester_failed
                    if failed
                    else self._settings_getter().voice_tts_completed
                ),
                (
                    f"Задача «{task.title}» завершена с ошибкой"
                    if failed
                    else f"Задача «{task.title}» завершена"
                ),
            )
        elif new.value == "queued" and old.value == "testing":
            self._speak(
                self._settings_getter().voice_tts_tester_failed,
                f"Тестирование провалено — задача «{task.title}» "
                "возвращена в разработку",
            )
        # The special settled states below cover "everything finished";
        # a per-task completion message is still useful and is not deduped.
        # Текст уведомления: "задача <status>" / "#{id} «{title}» — <описание статуса>".
        self.notify(
            self._title(f"task {new.value}"),
            f'#{task.id} "{task.title}" — {_status_text(new, task)}',
            tags=["clipboard" if new.value == "completed" else "hammer_and_wrench"],
            priority=(
                Priority.HIGH
                if new.value in ("clarification", "queued")
                else Priority.DEFAULT
            ),
        )

    def github_sync(self, imported: int, skipped: int) -> None:
        """A completed GitHub Issues sync cycle with its outcome."""
        self.notify(
            self._title("GitHub sync"),
            (
                f"GitHub sync finished: tasks imported — "
                f"{imported}, skipped — {skipped}."
            ),
            tags=["github"],
        )

    def pipeline_idle(self, *, blocked: bool, detail: str, prime: bool = False) -> None:
        """The pipeline settled with nothing left to run.

        blocked=False — the queue is empty, all work is processed.
        blocked=True — tasks remain, but none can be taken into
        development right now (all held / awaiting clarification / failed
        this session); `detail` says why.

        Sent only when the settled state changes (and never on the priming
        call at app startup), so repeated _advance() calls don't spam.
        """
        state = "blocked" if blocked else "empty"
        if prime:
            self._settled_state = state
            return
        if state == self._settled_state:
            return
        self._settled_state = state
        if blocked:
            # The "blocked" settled state means tasks remain that cannot
            # run right now — a living backlog, not a finished queue: not
            # voiced (only the genuinely-empty state is, by its checkbox).
            self.notify(
                self._title("pipeline idle"),
                f"No tasks can be taken into development. {detail}",
                tags=["warning"],
                priority=Priority.HIGH,
            )
        else:
            self._speak(
                self._settings_getter().voice_tts_queue_empty,
                "Очередь пуста — все задачи обработаны",
            )
            self.notify(
                self._title("queue empty"),
                "All tasks processed — no new tasks in the queue.",
                tags=["white_check_mark"],
                priority=Priority.HIGH,
            )

    def failed_tasks_requeued(self, count: int) -> None:
        """The failed-queue deadlock escape fired: every queued task had
        failed earlier this session and none could be promoted, so the
        error exclusion was forcibly lifted from all of them and they are
        back in development by the ordinary priority rules."""
        self.notify(
            self._title("failed tasks back in development"),
            f"No error-free tasks left in the queue: failed tasks — "
            f"{count}. All of them are back in development.",
            tags=["arrows_counterclockwise"],
            priority=Priority.HIGH,
        )

    def task_context_overflow(self, task_title: str) -> None:
        """A task's run died with a deterministic context-overflow error:
        its transcript exceeds the model's context window, so retrying the
        same history is guaranteed to fail — the task needs manual
        intervention (delete it, or shrink its transcript)."""
        self.notify(
            self._title("task needs manual intervention"),
            f"Task “{task_title}”: the transcript exceeds the model's "
            f"context window — the model rejects it on every attempt, "
            f"retrying cannot help. Delete the task or shrink its "
            f"transcript.",
            tags=["warning"],
            priority=Priority.HIGH,
        )

    def pipeline_busy(self) -> None:
        """A run started (development or testing): reset the settled-state
        dedup so the next idle transition notifies again."""
        self._settled_state = None
