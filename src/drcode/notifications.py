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
import sys
import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import httpx

from .models import VerificationStatus

if TYPE_CHECKING:
    from .models import Task, TaskStatus
    from .settings import ProjectSettings

NTFY_BASE = "https://ntfy.sh"

# Delivery timeout: ntfy.sh answers in well under a second normally; the
# generous ceiling only matters for a degraded network — and even then the
# send runs in its own daemon thread.
SEND_TIMEOUT = 10.0


class Priority(int, enum.Enum):
    """ntfy.sh message priorities (the ones worth distinguishing here).

    Numeric on purpose: the JSON publish API accepts `priority` only as an
    int (1..5) — a string like "high" makes ntfy.sh answer 400
    "request body must be valid JSON"."""

    MIN = 1
    DEFAULT = 3
    HIGH = 4
    URGENT = 5


def _status_text(status: TaskStatus, task: Task) -> str:
    """Human-readable one-liner for a status transition (Russian, matching
    the rest of the user-facing strings of the app)."""
    if status.value == "active":
        return "взята в разработку"
    if status.value == "testing":
        return "разработка завершена — ожидает тестирования"
    if status.value == "completed":
        verdict = ""
        if task.verification is not None:
            verdict = (
                " (тестирование пройдено)"
                if task.verification is VerificationStatus.PASSED
                else " (тестирование ПРОВАЛЕНО)"
            )
        return f"завершена{verdict}"
    if status.value == "clarification":
        return "ждёт ответа на уточняющий вопрос"
    if status.value == "hold":
        return "приостановлена (hold)"
    if status.value == "queued":
        return "возвращена в очередь"
    return status.value


class Notifier:
    """Event hub: the app feeds pipeline events in, the module filters,
    formats and delivers them to ntfy.sh.

    `settings_getter` is read on every event, so a settings change made in
    the dialog applies immediately without rebuilding anything.
    """

    def __init__(
        self,
        settings_getter: Callable[[], ProjectSettings],
        *,
        project: str = "",
        ntfy_base: str = NTFY_BASE,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._settings_getter = settings_getter
        # Name of the project folder the app works in: prepended to every
        # title ("DrCode (myproject): ...") so notifications from several
        # drcode instances sharing one ntfy topic stay distinguishable.
        self.project = project
        self.ntfy_base = ntfy_base.rstrip("/")
        self._on_error = on_error
        # Last settled pipeline state already reported (None = a run is
        # active / nothing settled yet): dedup of the empty/blocked
        # notifications — see pipeline_idle().
        self._settled_state: str | None = None

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
        threads included): the delivery runs in its own daemon thread."""
        topic = self._topic()
        if not topic:
            return
        thread = threading.Thread(
            target=self._deliver,
            args=(topic, title, message, tuple(tags), priority.value),
            daemon=True,
            name="drcode-notify",
        )
        thread.start()

    def _deliver(
        self, topic: str, title: str, message: str, tags: tuple[str, ...], priority: str
    ) -> None:
        try:
            response = httpx.post(
                self.ntfy_base,
                # JSON publish (POST to the root): unlike header-based
                # publishing it carries non-ASCII titles/messages safely.
                json={
                    "topic": topic,
                    "title": title,
                    "message": message,
                    "tags": list(tags),
                    "priority": priority,
                },
                timeout=SEND_TIMEOUT,
            )
            response.raise_for_status()
        except Exception as e:  # noqa: BLE001 - notifications must never crash anything
            self._report_error(f"{type(e).__name__}: {e}")

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
        # The special settled states below cover "everything finished";
        # a per-task completion message is still useful and is not deduped.
        self.notify(
            self._title(f"задача {new.value}"),
            f"#{task.id} «{task.title}» — {_status_text(new, task)}",
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
            self._title("синхронизация GitHub"),
            (
                f"Синхронизация с GitHub завершена: импортировано задач — "
                f"{imported}, пропущено — {skipped}."
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
            self.notify(
                self._title("конвейер простаивает"),
                f"Нет задач, которые можно взять в разработку. {detail}",
                tags=["warning"],
                priority=Priority.HIGH,
            )
        else:
            self.notify(
                self._title("очередь пуста"),
                "Все задачи обработаны — новых задач в очереди нет.",
                tags=["white_check_mark"],
                priority=Priority.HIGH,
            )

    def failed_tasks_requeued(self, count: int) -> None:
        """The failed-queue deadlock escape fired: every queued task had
        failed earlier this session and none could be promoted, so the
        error exclusion was forcibly lifted from all of them and they are
        back in development by the ordinary priority rules."""
        self.notify(
            self._title("задачи с ошибкой вернулись в разработку"),
            f"В очереди не осталось задач без ошибок: задач с ошибкой — "
            f"{count}. Все они снова возвращены в разработку.",
            tags=["arrows_counterclockwise"],
            priority=Priority.HIGH,
        )

    def pipeline_busy(self) -> None:
        """A run started (development or testing): reset the settled-state
        dedup so the next idle transition notifies again."""
        self._settled_state = None
