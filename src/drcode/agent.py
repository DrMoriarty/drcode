"""Coding and QA tester agents — pure re-export facade.

The implementation lives in split-out modules: `agent_build` (the deps
dataclasses and the `build_agent` / `build_tester_agent` wiring),
`agent_tools` (todo / dependency / commit / push tool helpers),
`agent_model` (model / network layer), `service_events` (service-event
logging), `tool_runtime` (worker-thread tool execution), `compaction`
(summarizer/compaction block), `transcript` (transcript / usage /
chat-rendering) and the bundled system prompts in `prompts.system`.
Everything is re-exported here so the test-suite and the TUI keep
importing from ``drcode.agent``; see `__all__`.
"""

from __future__ import annotations

from . import agent_model, task_search, tools
from .agent_build import (
    AgentDeps,
    TesterDeps,
    build_agent,
    build_tester_agent,
)
from .agent_model import (
    AsyncOpenAI,
    ModelFetchError,
    fetch_models,
    probe_model,
)
from .agent_tools import (
    TODO_MIN_ITEMS,
    _commit_tracked_files,
    _git_push,
    _retry_failed_here,
    _set_task_dependencies,
    _todo_create,
    _todo_update,
    _track_edited_file,
    load_agents_md,
    todo_dynamic_instructions,
    todo_instructions,
)
from .compaction import (
    MECHANICAL_HEADROOM_TOKENS,
    build_compaction_agent,
    build_compaction_prompt,
    build_reduce_prompt,
    chunk_history_for_summary,
    compacted_messages,
    extract_previous_summary,
    is_summary_message,
    mechanical_compaction_messages,
    prompt_fits_context,
    prompt_fits_window,
    render_history_for_summary,
    slice_budget_tokens,
    split_original_prompts,
)
from .prompts import (
    BUG_SYSTEM_PROMPT,
    FEATURE_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    TESTER_SYSTEM_PROMPT,
    TESTER_SYSTEM_PROMPT_NO_COMMIT,
    load_system_prompt,
    load_tester_prompt,
)
from .service_events import log_service_event
from .service_events import log_violation as _log_violation
from .task_manager import TaskManager
from .tool_runtime import (
    _run_bash_logged,
    _run_tool,
    _run_tool_logged,
)
from .transcript import (
    COMPACTION_SUMMARY_PREFIX,
    _context_status,
    all_user_prompts,
    backdate_user_prompt,
    ends_with_open_request,
    extract_display,
    first_user_prompt,
    is_pending_retry,
    last_context_tokens,
    last_response_usage,
    load_messages,
    resumable_history,
    run_context_tokens,
    sanitized_messages,
    serialize_messages,
)

__all__ = [
    "BUG_SYSTEM_PROMPT",
    "COMPACTION_SUMMARY_PREFIX",
    "FEATURE_SYSTEM_PROMPT",
    "MECHANICAL_HEADROOM_TOKENS",
    "RESEARCH_SYSTEM_PROMPT",
    "TESTER_SYSTEM_PROMPT",
    "TESTER_SYSTEM_PROMPT_NO_COMMIT",
    "TODO_MIN_ITEMS",
    "AgentDeps",
    "AsyncOpenAI",
    "ModelFetchError",
    "TaskManager",
    "TesterDeps",
    "_commit_tracked_files",
    "_context_status",
    "_git_push",
    "_log_violation",
    "_retry_failed_here",
    "_run_bash_logged",
    "_run_tool",
    "_run_tool_logged",
    "_set_task_dependencies",
    "_todo_create",
    "_todo_update",
    "_track_edited_file",
    "agent_model",
    "all_user_prompts",
    "backdate_user_prompt",
    "build_agent",
    "build_compaction_agent",
    "build_compaction_prompt",
    "build_reduce_prompt",
    "build_tester_agent",
    "chunk_history_for_summary",
    "compacted_messages",
    "ends_with_open_request",
    "extract_display",
    "extract_previous_summary",
    "fetch_models",
    "first_user_prompt",
    "is_pending_retry",
    "is_summary_message",
    "last_context_tokens",
    "last_response_usage",
    "load_agents_md",
    "load_messages",
    "load_system_prompt",
    "load_tester_prompt",
    "log_service_event",
    "mechanical_compaction_messages",
    "probe_model",
    "prompt_fits_context",
    "prompt_fits_window",
    "render_history_for_summary",
    "resumable_history",
    "run_context_tokens",
    "sanitized_messages",
    "serialize_messages",
    "slice_budget_tokens",
    "split_original_prompts",
    "task_search",
    "todo_dynamic_instructions",
    "todo_instructions",
    "tools",
]
