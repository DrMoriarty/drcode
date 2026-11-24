"""Tool runtime: run blocking tools in a worker thread and translate refusals.

`_run_tool` runs a tool in a thread (stalling the event loop otherwise, which
also drives the Textual UI); tools that raise a fixable error are converted to
`ModelRetry`. `_run_tool_logged` / `_run_bash_logged` additionally turn a
`PathViolation` into a logged service event plus a retry prompt. The deps types
are imported under `TYPE_CHECKING` to keep `agent.py` free of a circular import.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import tools
from .service_events import log_violation
from .tools import PathViolation

if TYPE_CHECKING:
    from .agent import AgentDeps, ResolverDeps, TesterDeps

# Errors tools raise for situations the model can fix (bad path, missing
# old_string, broken regex). pydantic-ai 2.x converts only ModelRetry and
# arg-validation failures into retry prompts; any other exception crashes
# the whole run, so these must be re-raised as ModelRetry.
TOOL_RETRY_ERRORS = (ValueError, OSError, re.error)


async def _run_tool(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    from pydantic_ai.exceptions import ModelRetry

    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except TOOL_RETRY_ERRORS as e:
        raise ModelRetry(str(e) or type(e).__name__) from e


async def _run_tool_logged(
    deps: AgentDeps | TesterDeps | ResolverDeps,
    tool: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a tool and turn a PathViolation into a retry prompt for the model,
    logging the refusal into the task history first."""
    from pydantic_ai.exceptions import ModelRetry

    try:
        return await _run_tool(fn, *args, **kwargs)
    except PathViolation as e:
        log_violation(deps, tool, str(e))
        raise ModelRetry(str(e)) from e


def web_fetch_available(deps: AgentDeps | TesterDeps | ResolverDeps) -> bool:
    """Whether the run that owns `deps` has the managed `web_fetch` tool.

    Only the RESEARCH agent ever gets it, and only when the project's
    web-fetch mode is not off — which the pipeline expresses as a present
    `web_fetch_gate` (see `agent_build.AgentDeps`). The coding, tester and
    resolver deps have no such field at all, so the flag is False there and
    the bash network guard's refusal says external access is closed instead
    of pointing at a tool the model does not have.
    """
    return getattr(deps, "web_fetch_gate", None) is not None


def network_unrestricted(deps: AgentDeps | TesterDeps | ResolverDeps) -> bool:
    """Whether this run's run_bash may reach external hosts.

    Mirrors the project's `unrestricted` web-fetch mode (the pipeline stamps
    it onto every dep shape — coding/tester/resolver included: the mode is a
    PROJECT-level network-access decision, not a property of one role). With
    the flag the bash network guard (`drcode.tools.check_command`) is lifted
    entirely — any host, redirect flags included; every other run_bash
    refusal (rm/mv/cp, git stash, filesystem scans) still applies. Off by
    default and for dep shapes that never carry the field.
    """
    return bool(getattr(deps, "network_unrestricted", False))


async def _run_bash_logged(
    deps: AgentDeps | TesterDeps | ResolverDeps,
    root: Path,
    command: str,
    timeout: float | None = None,
) -> str:
    from pydantic_ai.exceptions import ModelRetry

    try:
        return await tools.run_bash(
            root,
            command,
            timeout,
            extra_writable=deps.scratch,
            task_id=deps.task_id,
            # Only shapes the network refusal's wording (see `check_command`):
            # an agent WITH web_fetch is pointed at it, one without is told
            # that external access is closed in this run.
            web_fetch_available=web_fetch_available(deps),
            # The project's `unrestricted` web-fetch mode lifts the whole
            # network guard (any host, redirect flags included) — see
            # `network_unrestricted`.
            network_unrestricted=network_unrestricted(deps),
        )
    except PathViolation as e:
        log_violation(deps, "run_bash", str(e))
        raise ModelRetry(str(e)) from e


# Lazy re-export of the pydantic-ai-bound batch guard (see
# `drcode.tool_runtime_guard`): keeps historical
# `from .tool_runtime import attach_batch_guard` imports working while
# this module stays pydantic_ai-free at import time.
def __getattr__(name: str) -> Any:
    if name in ("BatchGuard", "BatchGuardToolset", "attach_batch_guard"):
        from . import tool_runtime_guard

        value = getattr(tool_runtime_guard, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
