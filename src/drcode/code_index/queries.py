"""Query helpers behind the phase-1 code_* tools (§5.1, §5.4) and the
phase-2 ones (§5.2 `refs`, §5.3 `callers`).

`find_symbols` — definition lookup by name substring across the whole
indexed project; `outline` — a file's skeleton or one symbol's
neighbourhood (optionally with the body read from disk); `refs` — every
textual occurrence of an identifier across indexed files (definitions
marked); `callers` — the §4.3 INFERRED call heuristic: identifier
matches inside function bodies, ranked by import proximity (§4.2
family-based import graph) and a cheap BM25 over "caller name + context
line" (the #0dc7fc65 result: on identifiers, lexical scoring is the
ceiling — no models). Call edges are computed ON QUERY, never stored
(§4.3); every answer carries the honest INFERRED label.

All are read-only, blocking (run them through `tool_runtime._run_tool`)
and CAPPED: the index is an accelerator, not a golden source — an empty
result always says "fall back to grep_search" (§6.3) and every output
is truncated honestly.

Self-healing (§6.4): when a query hits a stale record (the requested
body range is beyond the current file, the file vanished), the concrete
entry is invalidated and re-parsed once before the answer is shaped —
a wrong index entry can never survive a query that proves it wrong.

The structured cores (`definitions_of`, `ref_hits`, `caller_hits`,
`ImportGraph`) are exported for non-tool consumers: the
fastcontext-scout bench adapter and `scripts/edge_precision.py`.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .extract import FileIndex, Symbol
from .store import BuildInfo, load_index, reparse_entry

__all__ = [
    "OUTPUT_MAX_CHARS",
    "OUTPUT_MAX_LINES",
    "CallerHit",
    "ImportGraph",
    "RefHit",
    "batched_callers",
    "batched_refs",
    "caller_hits",
    "callers",
    "definitions_of",
    "find_symbols",
    "load_entries",
    "outline",
    "ref_hits",
    "refs",
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
    project_root: Path, *, persist: bool | Path = True
) -> tuple[dict[str, FileIndex], BuildInfo]:
    return load_index(project_root, persist=persist)


def find_symbols(
    project_root: Path,
    query: str,
    kind: str = "",
    limit: int = 20,
    *,
    persist: bool | Path = True,
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
    persist: bool | Path = True,
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


def _record_for(
    root: Path, rel: str, *, persist: bool | Path = True
) -> FileIndex | None:
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


# ---------------------------------------------------------------------------
# Phase 2: the §4.2 import graph, §5.2 refs, §5.3 callers.
# ---------------------------------------------------------------------------

# How many files `caller_hits` scans, ordered by import proximity (the wide
# pool of §4.3 — the pool is the recall ceiling, so it stays generous; the
# cap only bounds the pathological case, ranking happens after the scan).
_CALLERS_MAX_FILES = 400


def _split_identifier(ident: str) -> list[str]:
    """snake_case / camelCase → lowercase parts (`_run_summarizer` →
    [run, summarizer]); the token vocabulary of the lexical ranking."""
    ident = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", ident)
    ident = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", ident)
    return [t.lower() for t in ident.split("_") if t]


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for part in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        out.extend(_split_identifier(part))
    return out


def _bm25(query_tokens: list[str], docs_tokens: list[list[str]]) -> list[float]:
    """Okapi BM25 (k1=1.2, b=0.75) — the same formula the scout's BM25
    baseline uses, in-process, stdlib only (§4.3: no models in the
    ranking — the ablation measured exactly zero gain from them)."""
    n_docs = len(docs_tokens)
    if n_docs == 0:
        return []
    avgdl = (sum(len(d) for d in docs_tokens) / n_docs) or 1.0
    tfs = [Counter(d) for d in docs_tokens]
    df: Counter[str] = Counter()
    for tf in tfs:
        df.update(tf.keys())
    scores: list[float] = []
    for i, tf in enumerate(tfs):
        dl = len(docs_tokens[i]) or 1
        score = 0.0
        for term in dict.fromkeys(query_tokens):
            f = tf.get(term)
            if not f:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (f * 2.2) / (f + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))
        scores.append(score)
    return scores


def _module_paths_of(entries: dict[str, FileIndex]) -> dict[str, str]:
    """relpath → extension-stripped module path (`src/drcode/x.py` →
    `src/drcode/x`), the matching universe of the python/java-family
    dotted specs. `__init__.py` folds to its package path."""
    out: dict[str, str] = {}
    for rel in entries:
        stem = rel.rsplit(".", 1)[0] if "." in rel.rsplit("/", 1)[-1] else rel
        stem = stem.removesuffix("/__init__")
        out[rel] = stem
    return out


_ES_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte")


def _match_module(stem: str, module: str) -> bool:
    """`module` == stem or a path-suffix of it (`drcode.tool_runtime`
    matches `src/drcode/tool_runtime`, never `src/drcode/not_tool_runtime`)."""
    return stem == module or stem.endswith("/" + module)


def _resolve_spec(
    spec: str,
    family: str,
    importer_rel: str,
    module_paths: dict[str, str],
) -> set[str]:
    """§4.2: one raw import spec → the indexed file(s) it names. Exact
    resolution where the family allows it, last-segment heuristics
    otherwise (marked as such in the design — a ranking signal, not a
    fact). Unresolvable specs contribute no edge."""
    targets: set[str] = set()
    if family == "python":
        level = len(spec) - len(spec.lstrip("."))
        module = spec.lstrip(".")
        if level:  # relative import — anchored at the importer's package
            parts = importer_rel.split("/")[:-level]
            base = "/".join(parts)
            prefix = f"{base}/" if base else ""
            module_path = prefix + module.replace(".", "/")
        else:
            module_path = module.replace(".", "/")
        if not module_path:
            # `from . import x` — the importer's own package.
            module_path = "/".join(importer_rel.split("/")[: -level or 1])
        for rel, stem in module_paths.items():
            if _match_module(stem, module_path) or _match_module(
                stem, module_path + "/__init__"
            ):
                targets.add(rel)
    elif family == "es-family":
        if spec.startswith(("./", "../", "@/", "~/", "/")):
            base = importer_rel.rsplit("/", 1)[0] if "/" in importer_rel else ""
            path = spec
            for prefix, strip in (("@/", "src/"), ("~/", "src/")):
                if path.startswith(prefix):
                    path = strip + path[len(prefix) :]
                    break
            while path.startswith("../"):
                path = path[3:]
                base = base.rsplit("/", 1)[0] if "/" in base else ""
            path = (base + "/" if base else "") + path.lstrip("./")
            path = path.split("?", 1)[0].split("#", 1)[0]
            for rel in module_paths:
                if rel == path or rel.startswith((path + ".", path + "/index.")):
                    targets.add(rel)
        else:
            # Bare specifier (a package name): last-segment heuristic.
            last = spec.split("/")[-1].lower()
            for rel in module_paths:
                parts = rel.split("/")
                stem = parts[-1].rsplit(".", 1)[0].lower()
                if stem == last or (len(parts) > 1 and parts[-2].lower() == last):
                    targets.add(rel)
    elif family == "go":
        last = spec.rstrip("/").split("/")[-1].lower()
        for rel in module_paths:
            parts = rel.split("/")
            if len(parts) > 1 and parts[-2].lower() == last:
                targets.add(rel)
    elif family == "java-family":
        module = spec.replace(".", "/")
        for rel, stem in module_paths.items():
            if _match_module(stem, module):
                targets.add(rel)
    elif family == "c-family":
        name = spec.split("/")[-1]
        for rel in module_paths:
            if rel == spec or rel.endswith("/" + spec):
                targets.add(rel)
            else:  # `#include "x.h"` also reaches x.c/x.hpp siblings
                stem = rel.rsplit(".", 1)[0]
                if stem.endswith("/" + name.rsplit(".", 1)[0]):
                    targets.add(rel)
    elif family == "rust":
        path = spec
        base = ""
        if path.startswith("crate::"):
            path = path[len("crate::") :]
        elif path.startswith("self::"):
            path = path[len("self::") :]
        elif path.startswith("super::"):
            base_parts = importer_rel.split("/")[:-1]
            while path.startswith("super::"):
                path = path[len("super::") :]
                base_parts = base_parts[:-1]
            base = "/".join(base_parts)
        path = path.split(" as ")[0].replace("::", "/").split(":")[0].strip()
        if path.startswith("{") or not path:
            return targets
        module = (base + "/" if base else "") + path
        for rel in module_paths:
            if _match_module(rel, module) or _match_module(rel, module + "/mod"):
                targets.add(rel)
    targets.discard(importer_rel)
    return targets


class ImportGraph:
    """The §4.2 file-level import graph, computed ON QUERY from the
    stored raw specs (the index stays dumb; heuristics evolve without
    re-indexing). Direction: `importers_of(rel)` — who imports this
    file, expanded to 2 hops (direct importers first, then their
    importers) — the proximity signal of `caller_hits`."""

    def __init__(self, entries: dict[str, FileIndex]):
        self._module_paths = _module_paths_of(entries)
        self._importers: dict[str, dict[str, None]] = {}
        for rel, record in entries.items():
            for imp in record.imports:
                for target in _resolve_spec(
                    imp.spec, imp.kind, rel, self._module_paths
                ):
                    self._importers.setdefault(target, {}).setdefault(rel, None)

    def importers_of(self, rel: str, *, hops: int = 2) -> dict[str, int]:
        """rel → import distance (1 = direct importer, 2 = importer of an
        importer). Closer wins when a file is reachable both ways."""
        out: dict[str, int] = {}
        frontier = {rel}
        for distance in range(1, hops + 1):
            nxt: set[str] = set()
            for current in frontier:
                for importer in self._importers.get(current, ()):
                    if importer != rel and importer not in out:
                        out[importer] = distance
                        nxt.add(importer)
            frontier = nxt
            if not frontier:
                break
        return out


# The graph is a pure function of the entries' import lists — it costs
# seconds to build (spec resolution over every file) and a run makes
# many caller queries over one entries snapshot. Cached per (root,
# entries identity); the strong ref pins the entries dict, so an id
# can never be reused behind the cache's back.
_GRAPH_CACHE: dict[tuple[str, int], ImportGraph] = {}
_GRAPH_CACHE_MAX = 4


def _graph_for(root: Path, entries: dict[str, FileIndex]) -> ImportGraph:
    key = (str(root), id(entries))
    graph = _GRAPH_CACHE.get(key)
    if graph is None:
        graph = ImportGraph(entries)
        if len(_GRAPH_CACHE) >= _GRAPH_CACHE_MAX:
            _GRAPH_CACHE.clear()
        _GRAPH_CACHE[key] = graph
    return graph


# -- refs (§5.2) --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefHit:
    """One textual occurrence of an identifier in an indexed file."""

    rel: str
    line: int
    text: str
    is_def: bool
    is_call: bool


def _match_regex(name: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(name)}\b")


def _scope_prefix(line_text: str, start: int, end: int) -> bool:
    """`name.foo(…)` — the match sits in a scope PREFIX: the edge belongs
    to `foo`, not to `name` (§4.3 step 4, positional distinction)."""
    rest = line_text[end:].lstrip()
    return (
        bool(rest[:1] == ".")
        and len(rest) > 1
        and (rest[1].isalpha() or rest[1] == "_")
    )


def _call_position(line_text: str, start: int, end: int) -> bool:
    """True when the identifier sits in CALL position: `name(` (direct or
    after an attribute hop, `bar.name(`)."""
    return line_text[end:].lstrip().startswith("(")


def ref_hits(
    project_root: Path,
    name: str,
    entries: dict[str, FileIndex] | None = None,
    *,
    persist: bool | Path = True,
) -> list[RefHit]:
    """Every textual occurrence of `\\bname\\b` across the indexed files
    (§5.2 — "managed grep": grep_search scans everything, this scans the
    files the index already knows about). The structured core behind
    `refs`; bodies are NOT consulted — the file's own lines are."""
    root = Path(project_root)
    name = name.strip()
    if not name:
        return []
    if entries is None:
        entries, _info = _load(root, persist=persist)
    regex = _match_regex(name)
    hits: list[RefHit] = []
    for rel in sorted(entries):
        record = entries[rel]
        def_lines = {s.line for s in record.symbols if s.name == name}
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue  # Vanished mid-query: the record self-heals next call.
        for lineno, line_text in enumerate(text.splitlines(), 1):
            match = regex.search(line_text)
            if match is None:
                continue
            hits.append(
                RefHit(
                    rel=rel,
                    line=lineno,
                    text=line_text.strip()[:200],
                    is_def=lineno in def_lines,
                    is_call=_call_position(line_text, match.start(), match.end()),
                )
            )
    return hits


def refs(
    project_root: Path, name: str, limit: int = 50, *, persist: bool | Path = True
) -> str:
    """The formatted §5.2 answer: occurrences grouped by file, definitions
    marked [def], a `N files, M hits` header, the honest textual-match
    caveat."""
    root = Path(project_root)
    name = name.strip()
    if not name:
        return "Empty name — give an identifier (e.g. `run_agent`)."
    limit = max(1, min(limit, 200))
    hits = ref_hits(root, name, persist=persist)
    if not hits:
        return _EMPTY_HINT
    by_file: dict[str, list[RefHit]] = {}
    for hit in hits:
        by_file.setdefault(hit.rel, []).append(hit)
    lines = [
        (
            f"{len(by_file)} file(s), {len(hits)} hit(s) for `{name}` — textual "
            "match: same-name locals, strings and comments appear too; read the "
            "line, not just the location. For likely callers use code_callers."
        )
    ]
    shown = 0
    truncated_files = 0
    for rel, file_hits in sorted(by_file.items()):
        if shown >= limit:
            truncated_files += 1
            continue
        lines.append(f"{rel}:")
        for hit in file_hits:
            if shown >= limit:
                truncated_files += 1
                break
            mark = "  [def]" if hit.is_def else ""
            lines.append(f"  {hit.line}: {hit.text}{mark}")
            shown += 1
    hidden = len(hits) - shown
    if hidden > 0:
        lines.append(f"… and {hidden} more hit(s) (raise `limit` to see them)")
    elif truncated_files:
        lines.append(f"… and {truncated_files} more file(s)")
    return _truncate(lines)


# -- callers (§4.3 / §5.3) ----------------------------------------------------

# The §4.3 noise filter (measured: comments and docstrings were the bulk
# of the false edges — scripts/edge_precision.py). A per-file mask of
# CODE lines, computed on query from the plain text: line comments,
# /* */ blocks and triple-quoted docstring bodies are excluded. Purely
# heuristic and family-agnostic — a wrong classification costs a missed
# or a spurious edge, never an error; languages without a style keep
# every line (None = no opinion).
_FAMILY_COMMENT_STYLE: dict[
    str, tuple[tuple[str, ...], str | None, str | None, bool]
] = {
    "python": (("#",), None, None, True),
    "es-family": (("//",), "/*", "*/", False),
    "go": (("//",), "/*", "*/", False),
    "java-family": (("//",), "/*", "*/", False),
    "c-family": (("//",), "/*", "*/", False),
    "rust": (("//",), "/*", "*/", False),
    "lua": (("--",), None, None, False),
    "sql": (("--",), "/*", "*/", False),
    "haskell": (("--",), "{-", "-}", False),
    "clojure": (("; ", ";"), None, None, False),
}
_LANGUAGE_COMMENT_STYLE: dict[
    str, tuple[tuple[str, ...], str | None, str | None, bool]
] = {
    "bash": (("#",), None, None, False),
    "ruby": (("#",), None, None, False),
    "elixir": (("#",), None, None, True),
    "nim": (("#",), None, None, False),
    "julia": (("#",), "#=", "=#", False),
}


def _comment_style(
    language: str,
) -> tuple[tuple[str, ...], str | None, str | None, bool] | None:
    from .extract import _IMPORT_FAMILY

    style = _LANGUAGE_COMMENT_STYLE.get(language)
    if style is None:
        style = _FAMILY_COMMENT_STYLE.get(_IMPORT_FAMILY.get(language, ""))
    return style


_STRING_LINE_RE = re.compile(r"^[frbFRB]{0,2}(['\"])(.*)\1[,)\]]*$")


def _string_only_line(stripped: str) -> bool:
    """A line that is ENTIRELY a string literal (multi-line string
    argument / concatenation line): `\"text\"` / `f\"…\"` / `'…',` —
    identifier mentions inside are strings, not code (measured FP class)."""
    return bool(_STRING_LINE_RE.match(stripped))


def _blank_strings(line: str) -> str:
    """The line with STRING-LITERAL CONTENTS blanked (delimiters and
    f-string `{…}` interpolations kept): `getattr(x, "_name")` must not
    become an edge for `_name` — a string mention is not a code
    reference (the largest measured FP class). Quote-aware char scan
    with escapes; unbalanced quotes degrade to "no blanking" for the
    tail — a heuristic, never a crash."""
    out: list[str] = []
    quote = ""
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if quote:
            if ch == "\\":
                out.append("  ")
                i += 2
                continue
            if ch == quote:
                quote = ""
                out.append(ch)
                i += 1
                continue
            out.append(" ")
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _comment_cut(line: str, marker: str) -> str:
    """The line truncated at the first UNQUOTED trailing comment marker
    (`#` or `//`, per the language's style) — `x = f()  # mentions
    spawn` must not become an edge. Quote-aware char scan; escapes
    honored. Other markers (`--`, `;`) are never cut mid-line
    (decrement / statement separators)."""
    quote = ""
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            continue
        if line.startswith(marker, i):
            return line[:i]
        i += 1
    return line


# Prepared caller-scan lines per file (masked + blanked), reused across
# queries within the process: the docstring/string/comment filter is a
# pure function of the file text, and a run asks for many names. Keyed
# (abs path, text hash); a small bounded map — eviction is wholesale
# clear, the next query repares what it needs.
_PREP_CACHE: dict[
    tuple[str, int, int], tuple[set[int] | None, list[str], list[str]]
] = {}
_PREP_CACHE_MAX = 1024


def _prepared_scan(
    root: Path,
    rel: str,
    language: str,
    cut_marker: str | None,
) -> tuple[set[int] | None, list[str], list[str]] | None:
    """The file's code-line mask, its searchable lines (string contents
    blanked, trailing comments cut) and its raw lines — keyed by the
    stat pair, so a warm pass skips the read entirely (the file's text
    is a pure function of the stat key within a pass; a same-size
    same-mtime touch is the §6.4 corner the HEAD guard and the
    self-heal already own). None when the file is unreadable."""
    path = root / rel
    try:
        stat = path.stat()
    except OSError:
        return None
    # Key by ABSOLUTE path: rel is project-relative and two different
    # test/scratch projects can legitimately produce the same rel with
    # the same size and mtime in one process — a collision would serve
    # one project the other's lines.
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _PREP_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    mask = _code_line_mask(text, language)
    lines: list[str] = []
    for raw in text.splitlines():
        prepared = _blank_strings(raw)
        if cut_marker:
            prepared = _comment_cut(prepared, cut_marker)
        lines.append(prepared)
    if len(_PREP_CACHE) >= _PREP_CACHE_MAX:
        _PREP_CACHE.clear()
    _PREP_CACHE[key] = (mask, lines, text.splitlines())
    return mask, lines, text.splitlines()


def _code_line_mask(text: str, language: str) -> set[int] | None:
    """1-based line numbers of lines that are CODE — not comment lines
    (leading or trailing), not `/* */` block bodies, not triple-quoted
    docstring lines (openers, bodies and closers), not whole-line string
    literals. None when the language has no comment style (keep every
    line)."""
    style = _comment_style(language)
    if style is None:
        return None
    line_prefixes, block_open, block_close, triple = style
    code: set[int] = set()
    open_delim = ""  # active multi-line region (triple quote or /* */)
    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if open_delim:
            if open_delim in stripped and stripped.count(open_delim) % 2 == 1:
                open_delim = ""
            continue  # closing line carries prose, not code (measured)
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in line_prefixes):
            continue
        if block_open and stripped.startswith(block_open):
            if not (block_close and block_close in stripped[len(block_open) :]):
                open_delim = block_close or ""
            continue
        if triple:
            engaged = False
            for delim in ('"""', "'''"):
                count = raw.count(delim)
                if count == 1:
                    open_delim = delim  # docstring opens/closes mid-line
                    engaged = True
                elif count >= 2:
                    engaged = True  # whole-line triple-quoted string
            if engaged:
                continue
        if _string_only_line(stripped):
            continue
        code.add(lineno)
    return code


@dataclass(frozen=True, slots=True)
class CallerHit:
    """One INFERRED caller edge: the enclosing function (innermost symbol
    containing the match), the matching line, and the cheap ranking
    signals that produced the score."""

    caller_rel: str
    caller_line: int
    caller_name: str
    context_line: int
    context_text: str
    is_call: bool
    import_distance: int  # 0 = same file, 1 = direct importer, …
    score: float


def load_entries(
    project_root: Path, *, persist: bool | Path = True
) -> dict[str, FileIndex]:
    """The project's per-file index records through the normal
    incremental cache — the public entry point for batch consumers (the
    scout bench adapter) that make MANY queries over one entries
    snapshot and pass it to `batched_*` explicitly."""
    entries, _info = _load(Path(project_root), persist=persist)
    return entries


def definitions_of(
    project_root: Path,
    name: str,
    entries: dict[str, FileIndex] | None = None,
    *,
    persist: bool | Path = True,
) -> list[tuple[str, Symbol]]:
    """Every indexed definition named exactly `name` (exact match, all
    files) — §4.3 step 1."""
    root = Path(project_root)
    name = name.strip()
    if not name:
        return []
    if entries is None:
        entries, _info = _load(root, persist=persist)
    out: list[tuple[str, Symbol]] = []
    for rel in sorted(entries):
        for symbol in entries[rel].symbols:
            if symbol.name == name:
                out.append((rel, symbol))
    return out


def batched_callers(
    project_root: Path,
    names: list[str],
    entries: dict[str, FileIndex] | None = None,
    graph: ImportGraph | None = None,
    *,
    persist: bool | Path = True,
) -> dict[str, list[CallerHit]]:
    """The §4.3 heuristic for MANY names in ONE file pass: identifier
    matches inside function/method bodies → INFERRED caller edges,
    ranked by a cheap BM25 over «caller name + context line» plus the
    import-proximity and call-position bonuses. Semantically identical
    to calling `caller_hits` per name — the single pass (one read, one
    alternation regex, one prepared-scan per file) exists because a
    bench query and an agent turn alike ask for several names at once.
    NOISE FILTERS (§4.3 step 4, tightened by measurement): nested
    definitions are not calls; scope prefixes (`name.foo(`) belong to
    `foo`; comments, docstring bodies and string literals are not code.
    Computed on query, never stored."""
    root = Path(project_root)
    wanted = [n.strip() for n in names if n.strip()]
    wanted = list(dict.fromkeys(wanted))
    if not wanted:
        return {}
    if entries is None:
        entries, _info = _load(root, persist=persist)
    if graph is None:
        graph = _graph_for(root, entries)

    defs_by_name = {n: definitions_of(root, n, entries) for n in wanted}
    proximity_by_name: dict[str, dict[str, int]] = {}
    def_lines_by_name: dict[str, dict[str, set[int]]] = {}
    for name, defs in defs_by_name.items():
        proximity: dict[str, int] = {}
        for rel, _s in defs:
            proximity[rel] = 0
        for rel, _s in defs:
            for importer, distance in graph.importers_of(rel).items():
                if distance < proximity.get(importer, 99):
                    proximity[importer] = distance
        proximity_by_name[name] = proximity
        def_lines: dict[str, set[int]] = {}
        for rel, symbol in defs:
            def_lines.setdefault(rel, set()).add(symbol.line)
        def_lines_by_name[name] = def_lines

    # raw[name] -> (rel, enclosing, context, lineno, is_call)
    raw: dict[str, list[tuple[str, Symbol, str, int, bool]]] = {
        name: [] for name in wanted
    }
    for rel in sorted(entries):
        record = entries[rel]
        symbols = record.symbols
        style = _comment_style(record.language)
        cut_marker: str | None = None
        if style is not None:
            if "#" in style[0]:
                cut_marker = "#"
            elif "//" in style[0]:
                cut_marker = "//"
        prepared = _prepared_scan(root, rel, record.language, cut_marker)
        if prepared is None:
            continue  # vanished / unreadable mid-pass
        code_mask, search_lines, file_lines = prepared
        # The per-line word-boundary filter is the §4.3 «managed grep».
        # Implemented over plain substring finds on the blanked text (C
        # speed) instead of a regex per line — 8 names × ~200k lines of
        # `finditer` measured at seconds per query; find scans are ~50×
        # cheaper and the boundaries are checked by hand. Positions are
        # 1:1 between the blanked and the raw line (blanking is
        # length-preserving), so columns transfer directly.
        joined = "\n".join(search_lines)
        if not any(name in joined for name in wanted):
            continue  # cheap per-file prefilter
        starts = [0]
        for searchable in search_lines:
            starts.append(starts[-1] + len(searchable) + 1)
        joined_len = len(joined)
        for name in wanted:
            skip_lines = def_lines_by_name[name].get(rel, ())
            pos = joined.find(name)
            while pos != -1:
                pos_end = pos + len(name)
                before = joined[pos - 1] if pos else ""
                after = joined[pos_end] if pos_end < joined_len else ""
                word_before = before.isalnum() or before == "_"
                word_after = after.isalnum() or after == "_"
                next_pos = joined.find(name, pos_end)
                if word_before or word_after:
                    pos = next_pos
                    continue  # \b boundary failed
                lineno = bisect_right(starts, pos)
                if code_mask is not None and lineno not in code_mask:
                    pos = next_pos
                    continue  # comment / docstring body — not an edge (§4.3)
                if lineno in skip_lines:
                    pos = next_pos
                    continue  # a nested definition, not a call
                line_text = file_lines[lineno - 1]
                # Offsets within the LINE (blanking is length-preserving).
                line_start = starts[lineno - 1]
                col_start = pos - line_start
                col_end = pos_end - line_start
                if _scope_prefix(line_text, col_start, col_end):
                    pos = next_pos
                    continue  # `name.foo(` — the edge belongs to `foo`
                # The INNERMOST function/method symbol containing the line.
                enclosing: Symbol | None = None
                for symbol in symbols:
                    if symbol.kind not in ("function", "method"):
                        continue
                    if symbol.line <= lineno <= symbol.end_line and (
                        enclosing is None
                        or symbol.end_line - symbol.line
                        < enclosing.end_line - enclosing.line
                    ):
                        enclosing = symbol
                if enclosing is None:
                    pos = next_pos
                    continue  # module-level mention — no calling function
                raw[name].append(
                    (
                        rel,
                        enclosing,
                        line_text.strip()[:200],
                        lineno,
                        _call_position(line_text, col_start, col_end),
                    )
                )
                pos = next_pos

    result: dict[str, list[CallerHit]] = {}
    for name in wanted:
        raw_hits = raw[name]
        if not raw_hits:
            result[name] = []
            continue
        defs = defs_by_name[name]
        proximity = proximity_by_name[name]
        # BM25 over «caller name + context line» (§4.3 ranking); the query
        # carries the name plus the definition's scope/module tokens, so
        # hits that actually mention the defining module outrank bare
        # name echoes.
        query_tokens = _tokens(name)
        for rel, symbol in defs:
            query_tokens.extend(_tokens(symbol.scope))
            query_tokens.extend(_tokens(rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]))
        docs = [
            _tokens(f"{symbol.scope}.{symbol.name} {context}")
            for _rel, symbol, context, _l, _c in raw_hits
        ]
        scores = _bm25(query_tokens, docs)
        hits: list[CallerHit] = []
        for (rel, symbol, context, lineno, is_call), lexical in zip(raw_hits, scores):
            distance = proximity.get(rel, 99)
            score = lexical + (2.0 if is_call else 0.0) + max(0.0, 2.0 - distance * 0.5)
            hits.append(
                CallerHit(
                    caller_rel=rel,
                    caller_line=symbol.line,
                    caller_name=(
                        f"{symbol.scope}.{symbol.name}" if symbol.scope else symbol.name
                    ),
                    context_line=lineno,
                    context_text=context,
                    is_call=is_call,
                    import_distance=distance,
                    score=round(score, 4),
                )
            )
        hits.sort(key=lambda h: (-h.score, h.caller_rel, h.context_line))
        result[name] = hits
    return result


def caller_hits(
    project_root: Path,
    name: str,
    entries: dict[str, FileIndex] | None = None,
    graph: ImportGraph | None = None,
    *,
    persist: bool | Path = True,
) -> list[CallerHit]:
    """The §4.3 heuristic for ONE name — the thin wrapper over
    `batched_callers` (the same single-pass scan, one identifier)."""
    stripped = name.strip()
    if not stripped:
        return []
    return batched_callers(
        project_root,
        [stripped],
        entries,
        graph,
        persist=persist,
    ).get(stripped, [])


def batched_refs(
    project_root: Path,
    names: list[str],
    entries: dict[str, FileIndex] | None = None,
    *,
    persist: bool | Path = True,
) -> dict[str, list[RefHit]]:
    """The §5.2 textual scan for MANY names in ONE file pass — the batch
    twin of `ref_hits` (semantically identical to calling it per name;
    one read, one prepare, one substring pass per file). The TEXTUAL
    contract is kept: strings and comments are scanned too — only the
    store's prepared line indexing is reused, no masking here."""
    root = Path(project_root)
    wanted = [n.strip() for n in names if n.strip()]
    wanted = list(dict.fromkeys(wanted))
    if not wanted:
        return {}
    if entries is None:
        entries, _info = _load(root, persist=persist)
    def_lines_by_name: dict[str, dict[str, set[int]]] = {}
    for name in wanted:
        by_file: dict[str, set[int]] = {}
        for rel in sorted(entries):
            lines = {s.line for s in entries[rel].symbols if s.name == name}
            if lines:
                by_file[rel] = lines
        def_lines_by_name[name] = by_file
    result: dict[str, list[RefHit]] = {name: [] for name in wanted}
    for rel in sorted(entries):
        record = entries[rel]
        style = _comment_style(record.language)
        cut_marker: str | None = None
        if style is not None:
            if "#" in style[0]:
                cut_marker = "#"
            elif "//" in style[0]:
                cut_marker = "//"
        prepared = _prepared_scan(root, rel, record.language, cut_marker)
        if prepared is None:
            continue
        _mask, _search_lines, file_lines = prepared
        joined = "\n".join(file_lines)  # the RAW lines — textual contract
        starts = [0]
        for line in file_lines:
            starts.append(starts[-1] + len(line) + 1)
        joined_len = len(joined)
        for name in wanted:
            if name not in joined:
                continue
            def_lines = def_lines_by_name[name].get(rel, ())
            pos = joined.find(name)
            while pos != -1:
                pos_end = pos + len(name)
                before = joined[pos - 1] if pos else ""
                after = joined[pos_end] if pos_end < joined_len else ""
                word_before = before.isalnum() or before == "_"
                word_after = after.isalnum() or after == "_"
                next_pos = joined.find(name, pos_end)
                if word_before or word_after:
                    pos = next_pos
                    continue
                lineno = bisect_right(starts, pos)
                line_text = file_lines[lineno - 1]
                line_start = starts[lineno - 1]
                result[name].append(
                    RefHit(
                        rel=rel,
                        line=lineno,
                        text=line_text.strip()[:200],
                        is_def=lineno in def_lines,
                        is_call=_call_position(
                            line_text, pos - line_start, pos_end - line_start
                        ),
                    )
                )
                pos = next_pos
    for hits in result.values():
        hits.sort(key=lambda h: (h.rel, h.line))
    return result


def callers(
    project_root: Path,
    name: str,
    limit: int = 30,
    *,
    persist: bool | Path = True,
) -> str:
    """The formatted §5.3 answer: the INFERRED service line, where the
    symbol is defined, then `caller (file:line) → matching line`, ranked
    by import proximity + lexical score."""
    root = Path(project_root)
    name = name.strip()
    if not name:
        return "Empty name — give an identifier (e.g. `run_agent`)."
    limit = max(1, min(limit, 200))
    entries, _info = _load(root, persist=persist)
    defs = definitions_of(root, name, entries)
    hits = caller_hits(root, name, entries, persist=persist)
    if not hits and not defs:
        return _EMPTY_HINT
    lines = [
        (
            "callers are INFERRED from identifier matches — may include false "
            "positives (same-name symbols, strings); verify the line"
        )
    ]
    if defs:
        where = ", ".join(f"{rel}:{s.line}" for rel, s in defs[:5])
        more = f" (+{len(defs) - 5} more)" if len(defs) > 5 else ""
        lines.append(f"defined at: {where}{more}")
    else:
        lines.append(
            f"no indexed definition of `{name}` — matches below are body "
            "mentions only; use code_symbols / grep_search to locate it"
        )
    if not hits:
        lines.append(f"No caller-body matches for `{name}` in the indexed files.")
        return _truncate(lines)
    lines.append(f"{len(hits)} candidate caller(s) of `{name}`:")
    for shown, hit in enumerate(hits[:limit]):
        distance = {0: "same file", 1: "importer", 2: "indirect"}.get(
            hit.import_distance, ""
        )
        suffix = f"  [{distance}]" if distance else ""
        lines.append(
            f"{hit.caller_name} ({hit.caller_rel}:{hit.context_line}) → "
            f"{hit.context_text}{suffix}"
        )
    if len(hits) > limit:
        lines.append(f"… and {len(hits) - limit} more (raise `limit` to see them)")
    return _truncate(lines)
