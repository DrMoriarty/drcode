"""Query helpers behind the phase-1 code_* tools (§5.1, §5.4).

`find_symbols` — definition lookup by name substring across the whole
indexed project; `outline` — a file's skeleton or one symbol's
neighbourhood (optionally with the body read from disk). Both are
read-only, blocking (run them through `tool_runtime._run_tool`) and
CAPPED: the index is an accelerator, not a golden source — an empty
result always says "fall back to grep_search" (§6.3) and every output
is truncated honestly.

Self-healing (§6.4): when a query hits a stale record (the requested
body range is beyond the current file, the file vanished), the concrete
entry is invalidated and re-parsed once before the answer is shaped —
a wrong index entry can never survive a query that proves it wrong.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from .extract import FileIndex, Symbol
from .store import BuildInfo, load_index, reparse_entry

__all__ = [
    "OUTPUT_MAX_CHARS",
    "OUTPUT_MAX_LINES",
    "find_symbols",
    "outline",
]

# Output caps (§5): line-first, then characters — a runaway project (or a
# pathologically common name) must not be able to flood a context.
OUTPUT_MAX_LINES = 100
OUTPUT_MAX_CHARS = 8000

_EMPTY_HINT = (
    "No indexed definition matches. The index covers only files whose "
    "language is supported — this does NOT mean the symbol does not "
    "exist: fall back to grep_search / read_file."
)

_KINDS = {
    "function",
    "method",
    "class",
    "interface",
    "struct",
    "enum",
    "trait",
    "module",
    "variable",
}


def _norm_kind(kind: str) -> str:
    return kind.strip().lower()


def _rank(query: str, name: str) -> int:
    """0 = exact match, 1 = name starts with the query, 2 = substring —
    the §5.1 ordering (shortest path breaks ties later)."""
    lowered = name.lower()
    q = query.lower()
    if lowered == q:
        return 0
    if lowered.startswith(q):
        return 1
    return 2


def _truncate(lines: list[str]) -> str:
    """Join, cap by OUTPUT_MAX_LINES/OUTPUT_MAX_CHARS, mark truncation."""
    truncated = False
    if len(lines) > OUTPUT_MAX_LINES:
        lines = lines[:OUTPUT_MAX_LINES]
        truncated = True
    text = "\n".join(lines)
    if len(text) > OUTPUT_MAX_CHARS:
        text = text[:OUTPUT_MAX_CHARS]
        truncated = True
    if truncated:
        text += "\n… (output truncated — narrow the query or raise nothing: the cap is fixed)"
    return text


def _load(
    project_root: Path, *, persist: bool = True
) -> tuple[dict[str, FileIndex], BuildInfo]:
    return load_index(project_root, persist=persist)


def find_symbols(
    project_root: Path,
    query: str,
    kind: str = "",
    limit: int = 20,
    *,
    persist: bool = True,
) -> str:
    """Definitions whose name contains `query` (case-insensitive), ranked
    exact → prefix → substring, shortest path first (§5.1). `kind`
    optionally filters (function|method|class|interface|struct|enum|
    trait|module|variable). Returns `relpath:line  kind  scope.name`
    plus the signature (and the doc when there is one)."""
    root = Path(project_root)
    query = query.strip()
    if not query:
        return "Empty query — give a name substring (e.g. `run_agent`)."
    kind_filter = _norm_kind(kind) if kind else ""
    if kind_filter and kind_filter not in _KINDS:
        return f"Unknown kind `{kind}` — use one of: " + ", ".join(sorted(_KINDS)) + "."
    limit = max(1, min(limit, 100))
    entries, _info = _load(root, persist=persist)

    hits: list[tuple[int, int, str, Symbol]] = []
    for rel, record in entries.items():
        for symbol in record.symbols:
            if kind_filter and symbol.kind != kind_filter:
                continue
            if query.lower() in symbol.name.lower():
                hits.append((_rank(query, symbol.name), len(rel), rel, symbol))
    hits.sort(key=lambda item: (item[0], item[1], item[2], item[3].line))
    if not hits:
        return _EMPTY_HINT
    lines = [
        f"{len(hits)} match(es) for `{query}`"
        + (f", kind={kind_filter}" if kind_filter else "")
    ]
    for _rank_, _plen, rel, symbol in hits[:limit]:
        qualified = f"{symbol.scope}.{symbol.name}" if symbol.scope else symbol.name
        line = f"{rel}:{symbol.line}  {symbol.kind}  {qualified}  {symbol.signature}"
        if symbol.doc:
            line += f"  — {symbol.doc}"
        lines.append(line)
    if len(hits) > limit:
        lines.append(f"… and {len(hits) - limit} more (raise `limit` to see them)")
    return _truncate(lines)


def outline(
    project_root: Path,
    path: str,
    around: str = "",
    span: int = 0,
    *,
    persist: bool = True,
) -> str:
    """A file's skeleton: every definition with line, kind and signature
    (§5.4) — the map, not the code. With `around=<symbol name>`: the
    symbol plus its same-scope siblings; `span > 0` adds the symbol's
    real body from disk, capped at `span` lines (max 400)."""
    root = Path(project_root)
    rel = _relative(root, path)
    if rel is None:
        return f"Path `{path}` is outside the project — pass a project-relative path."
    span = max(0, min(span, 400))

    record = _record_for(root, rel, persist=persist)
    if record is None:
        return (
            f"`{rel}` is not in the index (unknown language or no grammar "
            "available) — use read_file / grep_search on it."
        )
    if not record.symbols:
        return f"`{rel}` ({record.language}): no definitions found."

    if not around:
        lines = [
            f"{rel} ({record.language}){', partial parse' if record.partial else ''}: {len(record.symbols)} definitions"
        ]
        lines.extend(_symbol_line(rel, s) for s in record.symbols)
        return _truncate(lines)

    target = None
    for symbol in record.symbols:
        if symbol.name == around:
            target = symbol
            break
    if target is None:
        lowered = around.lower()
        for symbol in record.symbols:
            if symbol.name.lower() == lowered:
                target = symbol
                break
    if target is None:
        return (
            f"`{around}` not found in the index for {rel}. Definitions: "
            + ", ".join(s.name for s in record.symbols[:50])
        )

    siblings = [s for s in record.symbols if s.scope == target.scope]
    lines = [
        f"{rel} — neighbourhood of {target.name}"
        + (f" (scope {target.scope})" if target.scope else " (top level)")
    ]
    lines.extend(_symbol_line(rel, s) for s in siblings)
    if span > 0:
        body = _body(root, rel, target, span)
        if body is None:
            # Stale record: the recorded range no longer fits the file —
            # a query PROVED its record wrong, so the entry is re-parsed
            # (cache + disk) and the answer is shaped from the fresh
            # record (§6.4 self-heal). One retry, then the honest hint.
            record = reparse_entry(root, rel, persist=persist)
            fresh = (
                next(
                    (s for s in record.symbols if s.name == target.name),
                    None,
                )
                if record is not None
                else None
            )
            if fresh is not None:
                target = fresh
                body = _body(root, rel, target, span)
        if body is None:
            lines.append(
                f"(body of {target.name} could not be read — the record went "
                "stale; use read_file on the line range)"
            )
        else:
            lines.append(body)
    return _truncate(lines)


def _symbol_line(rel: str, symbol: Symbol) -> str:
    qualified = f"{symbol.scope}.{symbol.name}" if symbol.scope else symbol.name
    line = f"{symbol.line:>5}  {symbol.kind:<9} {qualified}  {symbol.signature}"
    if symbol.doc:
        line += f"  — {symbol.doc}"
    return line


def _body(root: Path, rel: str, symbol: Symbol, span: int) -> str | None:
    """The symbol's body from disk (`[line, end_line]`, capped at `span`
    lines) — None when the file moved on (stale record, §6.4)."""
    try:
        full = (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = full.splitlines()
    start = symbol.line - 1
    if start >= len(lines) or symbol.end_line > len(lines):
        return None
    chunk = lines[start : start + span]
    header = f"--- body of {symbol.name} ({symbol.kind}, lines {symbol.line}-{symbol.end_line}"
    if symbol.end_line - symbol.line + 1 > span:
        header += f", first {span}"
    header += ") ---"
    return "\n".join([header, *chunk])


def _record_for(root: Path, rel: str, *, persist: bool = True) -> FileIndex | None:
    """The file's record through the normal stat gate. The query-level
    self-heal (a proved-stale record) goes through `store.reparse_entry`
    instead — see `outline`."""
    entries, _info = load_index(root, persist=persist)
    return entries.get(rel)


def _relative(root: Path, path: str) -> str | None:
    """Normalize a tool-supplied path into the index's relpath key
    (accepts relative and absolute paths; a symlink-free POSIX relpath
    is the canonical form)."""
    if not path.strip():
        return None
    candidate = Path(path.strip())
    if candidate.is_absolute():
        try:
            resolved = candidate.resolve()
            root_resolved = root.resolve()
            relative = resolved.relative_to(root_resolved)
        except (OSError, ValueError):
            return None
        return PurePosixPath(relative).as_posix()
    rel = PurePosixPath(candidate.as_posix())
    if rel.is_absolute() or ".." in rel.parts:
        return None
    # Normalize `./src/x.py` → `src/x.py`.
    parts = [p for p in rel.parts if p not in (".",)]
    if not parts:
        return None
    return PurePosixPath(*parts).as_posix()
