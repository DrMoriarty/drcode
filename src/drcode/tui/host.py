"""The Textual implementation of the pipeline host seam.

A thin adapter (no pipeline logic): every method forwards to the
corresponding ``App``/widget API. The core pipeline layers call only
the :class:`drcode.core.host.PipelineHost` protocol — this class is the
one place where Textual specifics (``App.notify``, ``App.set_timer``,
``App.run_worker``, the chat-block widgets) meet the core.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from .blocks import AssistantBlock, ReasoningBlock, ToolCallBlock

if TYPE_CHECKING:
    from .app import DrCodeApp


class TextualHost:
    """Implements the pipeline host seam over ``DrCodeApp``."""

    def __init__(self, app: DrCodeApp) -> None:
        self._app = app

    @property
    def host_active(self) -> bool:
        """False while the app is being dismantled (textual's ``_exit``
        flag / ``is_running`` pump state): deferred ticks and worker
        starts must be skipped."""
        app = self._app
        return not app._exit and app.is_running

    def host_notify(
        self,
        message: str,
        *,
        title: str | None = None,
        severity: str = "information",
        timeout: float | None = None,
    ) -> None:
        self._app.notify(
            message,
            title=title or "",
            severity=cast(Any, severity),
            timeout=timeout,
        )

    def host_set_timer(self, delay: float, callback: Any) -> Any:
        return self._app.set_timer(delay, callback)

    def host_call_after_refresh(self, callback: Any, *args: Any) -> Any:
        return self._app.call_after_refresh(callback, *args)

    def host_spawn_worker(
        self,
        factory: Any,
        *,
        name: str = "",
        group: str = "default",
        exclusive: bool = False,
    ) -> Any:
        # The same dispatch the textual @work decorator made internally:
        # the app's worker registry stays the single source of truth, so
        # the abort/nudge paths (`workers.cancel_group(self, group)`) keep
        # working unchanged.
        return self._app.run_worker(
            factory,
            name=name,
            group=group,
            exclusive=exclusive,
        )

    def host_log(self, message: Any) -> None:
        self._app.log(message)

    # -- header connection indicator ----------------------------------------

    def host_model_status_is_error(self) -> bool:
        return self._app.model_status.has_class("error")

    def host_publish_model_status(self, label: str, state: str) -> None:
        widget = self._app.model_status
        widget.remove_class("ok", "warn", "error")
        widget.add_class(state)
        self._app._render_model_status()

    # -- live chat blocks ---------------------------------------------------

    def host_new_reasoning_block(self, at: datetime | None = None) -> Any:
        return ReasoningBlock(at)

    def host_new_assistant_block(self) -> Any:
        return AssistantBlock()

    def host_new_tool_call_block(self, call_text: str) -> Any:
        return ToolCallBlock(call_text)
