"""UI-neutral worker runtime for the pipeline layers.

The pipeline's blocking work runs in workers, never on the UI event
loop. Under Textual the historical mechanism is the ``@work`` decorator
— Textual-specific, hence unusable from ``drcode.core``. This module
provides the neutral replacement:

- :func:`work` — the decorator the core pipeline layers use. It does
  NOT run anything itself: it dispatches through the host seam
  (``self.host.host_spawn_worker``), so each frontend keeps its own
  runtime and registry. The Textual host delegates to
  ``App.run_worker`` — the very call ``@work`` made internally — which
  preserves the exact semantics (exclusive groups keyed
  ``agent-<task-id>``, single ``tester`` / ``resolver`` / ``review``
  slots, cancellation of superseded runs, teardown on app exit, and the
  ``workers.cancel_group`` entries the abort/nudge paths rely on).

- :class:`AsyncioWorkerRuntime` — a standalone asyncio implementation
  of the same contract (registry keyed by group, exclusive groups
  cancel the previous occupant) for UI shells that have no worker
  machinery of their own.

The "run state counted, not singular" bookkeeping (``_agent_runs`` /
generations) is the pipeline's own, on top of either runtime — it is
not duplicated here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar


def work(
    *,
    group: str = "default",
    exclusive: bool = False,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Neutral replacement of textual's ``@work``.

    The decorated async method, when called, dispatches a worker through
    the host seam instead of running inline: ``self.host.host_spawn_worker``
    receives a factory bound to the call arguments plus the group /
    exclusivity flags. The host owns the runtime and the registry.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            async def runner() -> Any:
                return await fn(self, *args, **kwargs)

            return self.host.host_spawn_worker(
                runner,
                name=name or fn.__name__,
                group=group,
                exclusive=exclusive,
            )

        return wrapper

    return decorator


ResultT = TypeVar("ResultT")


class AsyncioWorkerRuntime:
    """Standalone asyncio implementation of the worker contract.

    Workers are plain ``asyncio.Task`` objects registered by group;
    starting an exclusive worker cancels the group's current one (and
    awaits its cancellation), mirroring textual's exclusive-group
    semantics. ``cancel_all`` is the teardown hook (the app shell calls
    it on close). Exceptions never escape the registry: they are logged
    through `on_error` — a worker crash must not take the frontend down
    (the textual runtime has the same property via its error handling).
    """

    def __init__(
        self,
        on_error: Callable[[str, BaseException], None] | None = None,
    ) -> None:
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._on_error = on_error

    def spawn(
        self,
        factory: Callable[[], Any],
        *,
        name: str = "",
        group: str = "default",
        exclusive: bool = False,
    ) -> asyncio.Task[None]:
        """Start `factory()` as a worker task in `group` (an exclusive
        start first cancels the group's current worker)."""
        if exclusive:
            self.cancel_group(group)

        async def runner() -> None:
            try:
                await factory()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - a worker crash is logged, not fatal
                if self._on_error is not None:
                    self._on_error(name or group, e)

        task = asyncio.create_task(runner(), name=name or f"worker-{group}")
        self._workers[group] = task
        task.add_done_callback(lambda t: self._reap(group, t))
        return task

    def _reap(self, group: str, task: asyncio.Task[None]) -> None:
        if self._workers.get(group) is task:
            del self._workers[group]

    def cancel_group(self, group: str) -> None:
        """Cancel the group's current worker (no-op when none)."""
        task = self._workers.pop(group, None)
        if task is not None and not task.done():
            task.cancel()

    async def wait_group(self, group: str) -> None:
        """Await the group's current worker's completion (tests)."""
        task = self._workers.get(group)
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass

    def cancel_all(self) -> None:
        """Teardown: cancel every registered worker."""
        for group in list(self._workers):
            self.cancel_group(group)
