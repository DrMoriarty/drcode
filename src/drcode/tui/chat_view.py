"""Chat view layer — the composition of the chat-related UI mixin layers.

Historically this module held one 2000-line ``ChatViewLayer`` mixin with
everything the main screen renders. It is split into focused sub-mixins
(each a :class:`~drcode.tui.state.DrCodeState` layer, mirroring the
``ActionsLayer``/``PipelineLayer`` style):

- :mod:`drcode.tui.sidebar_view` — sidebar (project info, task sections,
  todo panel) and its pollers;
- :mod:`drcode.tui.header_view` — connection status, expanded header,
  voice-dictation indicator;
- :mod:`drcode.tui.activity_view` — run-activity indicator, git
  unmerged-work counter, stall watchdog and nudge;
- :mod:`drcode.tui.chat_timeline` — chat history rendering, batching,
  delta re-render, tool-call spinners, service events;
- :mod:`drcode.tui.task_status_view` — statusbar, task statusbar,
  context-fill counter.

``ChatViewLayer`` stays the collective name (the MRO order of the
sub-layers matches the historical layout, so method resolution against
``ActionsLayer``/``PipelineLayer`` is unchanged) — :class:`~drcode.tui.app.DrCodeApp`
and every external import keep working untouched.
"""

from __future__ import annotations

from .activity_view import ActivityLayer
from .chat_timeline import ChatTimelineLayer, _mention_markup  # noqa: F401
from .header_view import HeaderLayer
from .sidebar_view import SidebarLayer
from .state import DrCodeState
from .task_status_view import TaskStatusLayer


class ChatViewLayer(
    SidebarLayer,
    HeaderLayer,
    ActivityLayer,
    ChatTimelineLayer,
    TaskStatusLayer,
    DrCodeState,
):
    """Collective mixin: all chat-view UI layers in one class."""
