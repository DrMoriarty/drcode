"""The host seam of the UI-neutral pipeline.

The pipeline layers (``core.pipeline_*``) own the state machine and the
run semantics but must not touch a concrete UI. Everything they need
FROM the frontend — toasts, deferred UI calls, timers, the worker
runtime, the streaming chat blocks — goes through the
:class:`PipelineHost` callback protocol declared here.

The Textual app implements the protocol in ``tui.host.TextualHost``
(a thin adapter over ``App.notify`` / ``App.set_timer`` /
``App.run_worker`` / the chat-block widgets); a future GUI implements
the same protocol its own way. The pipeline layers only ever hold a
``host`` reference — no import of any UI package, enforced by
``tests/test_core_purity.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class StreamBlock(Protocol):
    """Handle of a live-streamed chat block (reasoning / assistant).

    The pipeline grows these blocks while a model request streams; the
    host hands out the concrete widgets (Textual ``Static`` subclasses
    today). `stream_epoch` marks the chat-view generation a block was
    mounted in — a mid-node view switch bumps the app's epoch and
    orphans the block (the fresh one carries the already-streamed text
    over, see ``pipeline_stream``).
    """

    at: datetime
    stream_epoch: int

    def begin(self) -> None:
        """Streaming starts (spinner on, header rendered)."""
        ...

    def append(self, text: str) -> None:
        """Append one stream delta (content is accumulated immediately,
        the repaint is throttled by the implementation)."""
        ...

    def thinking(self, on: bool) -> None:
        """Spinner on/off while the stream is still running."""
        ...

    def finish(self) -> None:
        """The node's stream ended: stop spinners/timers."""
        ...

    def set_content(self, text: str) -> None:
        """Fill from history (not streaming)."""
        ...

    def get_content(self) -> str:
        """The accumulated content so far."""
        ...


class PipelineHost(Protocol):
    """Callback protocol the pipeline layers use instead of a UI base
    class. Implementations must be safe to call from the event loop
    thread (worker bodies run there, exactly like the textual ``@work``
    workers they replace)."""

    @property
    def host_active(self) -> bool:
        """False while the frontend is being dismantled (textual's
        ``_exit`` / ``is_running`` pair): deferred ticks and worker
        starts must be skipped."""
        ...

    def host_notify(
        self,
        message: str,
        *,
        title: str | None = None,
        severity: str = "information",
        timeout: float | None = None,
    ) -> None:
        """Show a toast (best-effort)."""
        ...

    def host_set_timer(self, delay: float, callback: Any) -> Any:
        """Schedule a deferred callback on the UI event loop."""
        ...

    def host_call_after_refresh(self, callback: Any, *args: Any) -> Any:
        """Run `callback` after the next UI refresh pass."""
        ...

    def host_spawn_worker(
        self,
        factory: Any,
        *,
        name: str = "",
        group: str = "default",
        exclusive: bool = False,
    ) -> Any:
        """Start a background worker running `factory()` (an awaitable
        factory). With `exclusive=True` a starting worker cancels the
        previous worker in the same `group` — the semantic base of the
        per-task run groups (`agent-<task-id>`) and the single
        tester/resolver/review slots."""
        ...

    def host_log(self, message: Any) -> None:
        """Diagnostics log (never fatal, user-invisible)."""
        ...

    # -- header connection indicator (model-status probe) -------------------

    def host_model_status_is_error(self) -> bool:
        """Whether the header label currently carries the error state
        (read before a probe round-trip)."""
        ...

    def host_publish_model_status(self, label: str, state: str) -> None:
        """Publish a probe result: `label` is the status text, `state`
        one of "ok" / "warn" / "error" (drives the widget's CSS class
        and the re-render)."""
        ...

    # -- live chat blocks ---------------------------------------------------

    def host_new_reasoning_block(self, at: datetime | None = None) -> Any:
        """Create a reasoning block (collapsible model-thinking widget)."""
        ...

    def host_new_assistant_block(self) -> Any:
        """Create an assistant answer block (streamed markdown)."""
        ...

    def host_new_tool_call_block(self, call_text: str) -> Any:
        """Create a tool-call block for `call_text` ("name(args)")."""
        ...
