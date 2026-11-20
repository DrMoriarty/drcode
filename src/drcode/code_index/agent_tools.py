"""Registration of the code-index tools onto pydantic-ai agents (§5).

Shared by `agent_build.build_agent` (the coding role), the tester and
the resolver builders and `subagents.build_subagent_agent` (the
`delegate_research` subagents — read-only code researchers, the main
beneficiaries). Deliberately a separate module: both importers already
sit at the top of drcode's import graph and neither may import the
other.

The registration follows the web_fetch precedent exactly: the tool
functions are defined WITHOUT the `@agent.tool` decorator and get their
`__doc__` ASSIGNED afterwards — pydantic-ai parses the description out
of the source at registration time, so a decorator would register the
tool with an EMPTY description.

There is no per-run budget gate (§5): the calls are local, millisecond
reads; context bloat is bounded by the output caps in `queries` and the
existing mark_unneeded / auto-eviction machinery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai import Agent, RunContext

from drcode.service_events import log_service_event
from drcode.tool_runtime import _run_tool

from . import queries
from .store import take_build_notice

__all__ = ["register_code_index_tools"]


def _persist_for(ctx: RunContext[Any]) -> bool:
    """Whether the index may touch the disk for THIS call: only when the
    tree being indexed IS the main checkout (§7.1 — the physical index
    lives at the storage root, never inside a worktree, whose `.drcode/`
    must stay copy-skip clean). A worktree task (or a deps shape without
    a manager) gets the in-memory-only §6.2 overlay instead."""
    deps = ctx.deps
    root = getattr(deps, "root", None)
    manager = getattr(deps, "manager", None)
    if root is None or manager is None:
        return False
    try:
        return Path(root).resolve() == Path(manager.storage.root).resolve()
    except (OSError, ValueError):
        return False


def register_code_index_tools(agent: Agent[Any, Any], *, enabled: bool) -> None:
    """Register the code-index query tools (`code_symbols`, `code_outline`,
    `code_refs`, `code_callers`) on the agent. `enabled`
    is the resolved project decision (`agent_build.code_index_tools_for`)
    — when False the tools are not registered AT ALL: the model never
    sees them, which is itself the OFF switch (§5)."""
    if not enabled:
        return

    async def code_symbols(
        ctx: RunContext[Any], query: str, kind: str = "", limit: int = 20
    ) -> str:
        result = await _run_tool(
            queries.find_symbols,
            ctx.deps.root,
            query,
            kind=kind,
            limit=limit,
            persist=_persist_for(ctx),
        )
        _log_first_build(ctx)
        return result

    # The docstring is assigned, not written inline (web_fetch precedent:
    # pydantic-ai reads the description from the source).
    code_symbols.__doc__ = """Find symbol DEFINITIONS in the project's code index (functions, methods,
            classes, interfaces, structs, enums, …) by name substring — the cheap
            replacement for reading whole files just to locate a definition.
            `query` is a name substring (case-insensitive); `kind` optionally
            filters (`function`, `method`, `class`, `interface`, `struct`,
            `enum`, `trait`, `module`, `variable`); `limit` caps the results
            (default 20). Definitions are parsed from source (reliable), but
            coverage depends on the language being indexed; an empty result
            means "not indexed here", not "does not exist" — fall back to
            grep_search then."""
    agent.tool(code_symbols)

    async def code_outline(
        ctx: RunContext[Any], path: str, around: str = "", span: int = 0
    ) -> str:
        result = await _run_tool(
            queries.outline,
            ctx.deps.root,
            path,
            around=around,
            span=span,
            persist=_persist_for(ctx),
        )
        _log_first_build(ctx)
        return result

    code_outline.__doc__ = """Show a file's structure WITHOUT reading it whole: every definition with
            its line, kind and signature — a map, not the code. Use it to decide
            WHAT to read, then read_file the exact range. With `around` (a
            symbol name) — the neighbourhood of that one symbol: itself plus
            its sibling definitions on the same nesting level; `span` adds the
            symbol's actual body from disk, capped (max 400 lines). A path not
            in the index (unknown language) comes back as an explicit notice —
            read that file directly instead."""
    agent.tool(code_outline)

    async def code_refs(ctx: RunContext[Any], name: str, limit: int = 50) -> str:
        result = await _run_tool(
            queries.refs,
            ctx.deps.root,
            name,
            limit=limit,
            persist=_persist_for(ctx),
        )
        _log_first_build(ctx)
        return result

    code_refs.__doc__ = """List every occurrence of an identifier across indexed files — definitions
            marked [def], the rest are usages (import sites, calls, plain
            mentions). This is a TEXTUAL match: same-name locals, strings and
            comments appear too — read the returned line, not just the
            location, before treating a hit as a real usage. For "who calls
            this function" prefer code_callers; an empty result means the
            identifier lives in an unindexed file — fall back to grep_search."""
    agent.tool(code_refs)

    async def code_callers(ctx: RunContext[Any], name: str, limit: int = 30) -> str:
        result = await _run_tool(
            queries.callers,
            ctx.deps.root,
            name,
            limit=limit,
            persist=_persist_for(ctx),
        )
        _log_first_build(ctx)
        return result

    code_callers.__doc__ = """List functions that LIKELY call `name`, inferred from identifier matches
            inside function bodies (NOT a compiler-accurate call graph). Ranked
            by import proximity plus a lexical score; may include false
            positives for common names — every hit ships the matching source
            line so you can verify at a glance. Dynamic dispatch (getattr,
            reflection) is invisible by design. Use code_symbols to locate the
            definition first, then this to map its blast radius; an empty
            result means nothing indexed matched — fall back to grep_search."""
    agent.tool(code_callers)


def _log_first_build(ctx: RunContext[Any]) -> None:
    """Consume the pending first-build notice (§8.3: ONE service event
    when the index is (re)built in this process — chat visibility, never
    model context). The subagent deps have no own task_id — the notice is
    simply dropped there (the parent already logged its own)."""
    notice = take_build_notice(ctx.deps.root)
    deps = ctx.deps
    if notice is None or getattr(deps, "task_id", None) is None:
        return
    try:
        log_service_event(deps, "code_index", notice)
    except Exception:  # noqa: BLE001, S110 - visibility never crashes a tool
        pass
