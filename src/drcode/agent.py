# pyright: reportUnsupportedDunderAll=false
"""Coding and QA tester agents — lazy re-export facade.

The implementation lives in split-out modules: `agent_build` (the deps
dataclasses and the `build_agent` / `build_tester_agent` wiring),
`agent_tools` (todo / dependency / commit / push tool helpers),
`goals` (the product-goal registry store), `agent_model` (model /
network layer), `service_events` (service-event logging), `tool_runtime`
(worker-thread tool execution), `compaction` (summarizer/compaction
block), `transcript` (transcript / usage / chat-rendering) and the
bundled system prompts in `prompts.system`.

Everything is re-exported here so the test-suite and the TUI keep
importing from ``drcode.agent`` — see `__all__` — but NOTHING is
imported eagerly: importing this module is nearly free, and each name
is resolved on first access through the module-level `__getattr__`
(PEP 562). This keeps the pydantic_ai/openai stack (via `agent_build` /
`agent_model`), the MCP SDK, the native PortAudio library and
tree-sitter out of the TUI's cold-start path; they load on the first
real use (agent run, model probe, code_* tool, …) instead.

Keep the `__all__` list in sync with the `_EXPORTS` mapping below.
"""

from __future__ import annotations

import importlib
from typing import Any

# exported name -> submodule that defines it
_EXPORTS: dict[str, str] = {
    # agent_build
    "AgentDeps": "agent_build",
    "ClarifyQuestion": "agent_build",
    "ResolutionVerdict": "agent_build",
    "ResolverDeps": "agent_build",
    "TesterDeps": "agent_build",
    "build_agent": "agent_build",
    "build_resolver_agent": "agent_build",
    "build_resolver_reviewer_agent": "agent_build",
    "build_tester_agent": "agent_build",
    "code_index_tools_for": "agent_build",
    # agent_model
    "AsyncOpenAI": "agent_model",
    "ModelFetchError": "agent_model",
    "TailInjectionModel": "agent_model",
    "fetch_models": "agent_model",
    "probe_model": "agent_model",
    # agent_tools
    "TODO_MIN_ITEMS": "agent_tools",
    "InputDeliveryHooks": "agent_tools",
    "_commit_tracked_files": "agent_tools",
    "_create_goal": "agent_tools",
    "_git_push": "agent_tools",
    "_list_goals": "agent_tools",
    "_mark_unneeded": "agent_tools",
    "_read_goal": "agent_tools",
    "_read_linked_goal": "agent_tools",
    "_retry_failed_here": "agent_tools",
    "_run_project_check": "agent_tools",
    "_send_task_input": "agent_tools",
    "_set_task_dependencies": "agent_tools",
    "_todo_create": "agent_tools",
    "_todo_update": "agent_tools",
    "_track_edited_file": "agent_tools",
    "_update_goal": "agent_tools",
    "eviction_dynamic_instructions": "agent_tools",
    "goals_dynamic_instructions": "agent_tools",
    "goals_instructions": "agent_tools",
    "load_agents_md": "agent_tools",
    "subagent_notice": "agent_tools",
    "task_goal_block": "agent_tools",
    "todo_instructions": "agent_tools",
    # compaction
    "GIANT_PROMPT_TOKENS": "compaction",
    "MECHANICAL_HEADROOM_TOKENS": "compaction",
    "build_compaction_agent": "compaction",
    "build_compaction_prompt": "compaction",
    "build_reduce_prompt": "compaction",
    "chunk_history_for_summary": "compaction",
    "compacted_messages": "compaction",
    "even_slice_budget": "compaction",
    "extract_previous_summary": "compaction",
    "is_summary_message": "compaction",
    "mechanical_compaction_messages": "compaction",
    "prompt_fits_context": "compaction",
    "prompt_fits_window": "compaction",
    "render_history_for_summary": "compaction",
    "slice_budget_tokens": "compaction",
    "split_original_prompts": "compaction",
    "tester_verdict_issued": "compaction",
    "unwrap_summary_envelope": "compaction",
    # prompts
    "BUG_SYSTEM_PROMPT": "prompts",
    "FEATURE_SYSTEM_PROMPT": "prompts",
    "RESEARCH_GIT_PROMPT_LOCAL": "prompts",
    "RESEARCH_GIT_PROMPT_PUSH": "prompts",
    "RESEARCH_SYSTEM_PROMPT": "prompts",
    "RESOLVER_REVIEWER_PROMPT": "prompts",
    "RESOLVER_SYSTEM_PROMPT": "prompts",
    "TESTER_SYSTEM_PROMPT": "prompts",
    "TESTER_SYSTEM_PROMPT_NO_COMMIT": "prompts",
    "load_resolver_prompt": "prompts",
    "load_resolver_reviewer_prompt": "prompts",
    "load_system_prompt": "prompts",
    "load_tester_prompt": "prompts",
    # service_events
    "log_service_event": "service_events",
    "_log_violation": "service_events",
    # subagents
    "SubagentDeps": "subagents",
    "SubagentGate": "subagents",
    "build_subagent_agent": "subagents",
    "run_delegate_tool": "subagents",
    "subagent_limit_for": "subagents",
    # task_manager
    "TaskManager": "task_manager",
    # title_gen
    "TITLE_MAX_LEN": "title_gen",
    "build_title_agent": "title_gen",
    "build_title_prompt": "title_gen",
    "clean_generated_title": "title_gen",
    # tool_runtime
    "_run_bash_logged": "tool_runtime",
    "_run_tool": "tool_runtime",
    "_run_tool_logged": "tool_runtime",
    "web_fetch_available": "tool_runtime",
    # transcript
    "COMPACTION_RESUME_PREFIX": "transcript",
    "COMPACTION_SUMMARY_PREFIX": "transcript",
    "MODEL_SWITCH_NOTE_PREFIX": "transcript",
    "_context_status": "transcript",
    "all_user_prompts": "transcript",
    "backdate_user_prompt": "transcript",
    "drop_model_switch_notes": "transcript",
    "ends_with_empty_response": "transcript",
    "ends_with_open_request": "transcript",
    "ends_with_truncated_response": "transcript",
    "extract_display": "transcript",
    "first_user_prompt": "transcript",
    "is_pending_retry": "transcript",
    "last_context_tokens": "transcript",
    "last_response_usage": "transcript",
    "load_messages": "transcript",
    "resumable_history": "transcript",
    "run_context_tokens": "transcript",
    "sanitized_messages": "transcript",
    "serialize_messages": "transcript",
    "stale_thinking_tokens": "transcript",
    "strip_stale_thinking": "transcript",
    "strip_thinking_parts": "transcript",
    # web_fetch
    "FETCHES_PER_RUN": "web_fetch",
    "MAX_BYTES": "web_fetch",
    "MAX_OUTPUT_CHARS": "web_fetch",
    "WebFetchError": "web_fetch",
    "WebFetchGate": "web_fetch",
    "WebFetchPolicy": "web_fetch",
    "fetch": "web_fetch",
    "format_page": "web_fetch",
    "html_to_text": "web_fetch",
    "resolve_addresses": "web_fetch",
    "run_web_fetch": "web_fetch",
    "web_fetch_policy": "web_fetch",
    "web_fetch_policy_for": "web_fetch",
}

# bare submodules exposed as attributes (``drcode.agent.tools`` etc.)
_MODULES: frozenset[str] = frozenset({"agent_model", "task_search", "tools"})

# facade name -> the name in the defining submodule (aliases only)
_ALIASES: dict[str, str] = {"_log_violation": "log_violation"}


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        module = importlib.import_module(f"drcode.{name}")
        globals()[name] = module
        return module
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"drcode.{target}")
    # Aliased re-exports (e.g. `_log_violation` = `service_events.log_violation`).
    value = getattr(module, _ALIASES.get(name, name))
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS) | _MODULES)


# The names resolve lazily through the module `__getattr__` above; the

__all__ = [  # noqa: F822
    "BUG_SYSTEM_PROMPT",
    "COMPACTION_RESUME_PREFIX",
    "COMPACTION_SUMMARY_PREFIX",
    "FEATURE_SYSTEM_PROMPT",
    "FETCHES_PER_RUN",
    "GIANT_PROMPT_TOKENS",
    "MAX_BYTES",
    "MAX_OUTPUT_CHARS",
    "MECHANICAL_HEADROOM_TOKENS",
    "MODEL_SWITCH_NOTE_PREFIX",
    "RESEARCH_GIT_PROMPT_LOCAL",
    "RESEARCH_GIT_PROMPT_PUSH",
    "RESEARCH_SYSTEM_PROMPT",
    "RESOLVER_REVIEWER_PROMPT",
    "RESOLVER_SYSTEM_PROMPT",
    "TESTER_SYSTEM_PROMPT",
    "TESTER_SYSTEM_PROMPT_NO_COMMIT",
    "TITLE_MAX_LEN",
    "TODO_MIN_ITEMS",
    "AgentDeps",
    "AsyncOpenAI",
    "ClarifyQuestion",
    "InputDeliveryHooks",
    "ModelFetchError",
    "ResolutionVerdict",
    "ResolverDeps",
    "SubagentDeps",
    "SubagentGate",
    "TailInjectionModel",
    "TaskManager",
    "TesterDeps",
    "WebFetchError",
    "WebFetchGate",
    "WebFetchPolicy",
    "_commit_tracked_files",
    "_context_status",
    "_create_goal",
    "_git_push",
    "_list_goals",
    "_log_violation",
    "_mark_unneeded",
    "_read_goal",
    "_read_linked_goal",
    "_retry_failed_here",
    "_run_bash_logged",
    "_run_project_check",
    "_run_tool",
    "_run_tool_logged",
    "_send_task_input",
    "_set_task_dependencies",
    "_todo_create",
    "_todo_update",
    "_track_edited_file",
    "_update_goal",
    "agent_model",
    "all_user_prompts",
    "backdate_user_prompt",
    "build_agent",
    "build_compaction_agent",
    "build_compaction_prompt",
    "build_reduce_prompt",
    "build_resolver_agent",
    "build_resolver_reviewer_agent",
    "build_subagent_agent",
    "build_tester_agent",
    "build_title_agent",
    "build_title_prompt",
    "chunk_history_for_summary",
    "clean_generated_title",
    "code_index_tools_for",
    "compacted_messages",
    "drop_model_switch_notes",
    "ends_with_empty_response",
    "ends_with_open_request",
    "ends_with_truncated_response",
    "even_slice_budget",
    "eviction_dynamic_instructions",
    "extract_display",
    "extract_previous_summary",
    "fetch",
    "fetch_models",
    "first_user_prompt",
    "format_page",
    "goals_dynamic_instructions",
    "goals_instructions",
    "html_to_text",
    "is_pending_retry",
    "is_summary_message",
    "last_context_tokens",
    "last_response_usage",
    "load_agents_md",
    "load_messages",
    "load_resolver_prompt",
    "load_resolver_reviewer_prompt",
    "load_system_prompt",
    "load_tester_prompt",
    "log_service_event",
    "mechanical_compaction_messages",
    "probe_model",
    "prompt_fits_context",
    "prompt_fits_window",
    "render_history_for_summary",
    "resolve_addresses",
    "resumable_history",
    "run_context_tokens",
    "run_delegate_tool",
    "run_web_fetch",
    "sanitized_messages",
    "serialize_messages",
    "slice_budget_tokens",
    "split_original_prompts",
    "stale_thinking_tokens",
    "strip_stale_thinking",
    "strip_thinking_parts",
    "subagent_limit_for",
    "subagent_notice",
    "task_goal_block",
    "task_search",
    "tester_verdict_issued",
    "todo_instructions",
    "tools",
    "unwrap_summary_envelope",
    "web_fetch_available",
    "web_fetch_policy",
    "web_fetch_policy_for",
]
