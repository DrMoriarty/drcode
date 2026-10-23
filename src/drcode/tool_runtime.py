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

from pydantic_ai.exceptions import ModelRetry

from . import tools
from .service_events import log_violation
from .tools import PathViolation

if TYPE_CHECKING:
    from .agent import AgentDeps, TesterDeps

# Errors tools raise for situations the model can fix (bad path, missing
# old_string, broken regex). pydantic-ai 2.x converts only ModelRetry and
# arg-validation failures into retry prompts; any other exception crashes
# the whole run, so these must be re-raised as ModelRetry.
TOOL_RETRY_ERRORS = (ValueError, OSError, re.error)


async def _run_tool(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except TOOL_RETRY_ERRORS as e:
        raise ModelRetry(str(e) or type(e).__name__) from e


async def _run_tool_logged(
    deps: AgentDeps | TesterDeps,
    tool: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a tool and turn a PathViolation into a retry prompt for the model,
    logging the refusal into the task history first."""
    try:
        return await _run_tool(fn, *args, **kwargs)
    except PathViolation as e:
        log_violation(deps, tool, str(e))
        raise ModelRetry(str(e)) from e


async def _run_bash_logged(
    deps: AgentDeps | TesterDeps, root: Path, command: str, timeout: float | None = None
) -> str:
    try:
        return await tools.run_bash(
            root, command, timeout, extra_writable=deps.scratch, task_id=deps.task_id
        )
    except PathViolation as e:
        log_violation(deps, "run_bash", str(e))
        raise ModelRetry(str(e)) from e
