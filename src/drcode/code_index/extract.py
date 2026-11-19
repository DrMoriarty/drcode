"""Symbol extraction for the code index (docs/code-index-design.md §2).

The extractor is a tree-sitter adapter over `tree-sitter-language-pack`:
definitions come from the grammar's bundled `tags` query when one exists
(zero per-language code) and from a small node-kind table otherwise.
The language is resolved from the file extension via a self-contained
table — predictable, and an unknown extension simply means "not indexed"
(the rest of the tooling keeps honest coverage: grep_search still sees
such files).

Degradation contract (§2.4): a grammar that is not available locally
and cannot be downloaded comes back as `None` from `extract` — the file
is skipped, never fatal. `partial` marks files whose parse produced
error nodes (symbols may be missing).

BLOCKING: parsing and (in dev runs) the on-demand grammar download are
blocking calls — callers must run extraction in a worker thread
(`tool_runtime._run_tool` already does).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Language, Node, Parser, Query, QueryCursor
from tree_sitter_language_pack import (
    DownloadError,
    LanguageNotFoundError,
    get_parser,
    get_tags_query,
    has_language,
)

__all__ = [
    "MAX_INDEXED_FILE_BYTES",
    "SIGNATURE_MAX_CHARS",
    "FileIndex",
    "ImportEntry",
    "Symbol",
    "SymbolExtractor",
    "file_index_from_disk",
    "file_index_to_disk",
]

# Files larger than this are skipped with a mark (§3.4): not our
# audience, and a megabyte of minified JS would poison the index size.
MAX_INDEXED_FILE_BYTES = 1_000_000

SIGNATURE_MAX_CHARS = 200
DOC_MAX_CHARS = 160


@dataclass(frozen=True, slots=True)
class Symbol:
    """One extracted definition (§3.2): 1-based inclusive line range,
    a flat dotted scope ("AgentDeps.build_agent", '' on the top level),
    the first line of the definition (≤200 chars) and up to 2 docstring
    /comment lines (≤160 chars). Bodies are NEVER stored — queries read
    them from disk by the line range."""

    name: str
    kind: str
    scope: str
    signature: str
    line: int
    end_line: int
    doc: str = ""


@dataclass(frozen=True, slots=True)
class ImportEntry:
    """One import line of the file (§4.2): the language family the spec
    was matched with ("python"/"es-family"/"go"/"java-family"/"c-family"/
    "rust") and the raw import spec ("drcode.tool_runtime", "./utils",
    "fs"). Phase 1 only stores them; the import graph consumes them in
    phase 2."""

    kind: str
    spec: str


@dataclass(frozen=True, slots=True)
class FileIndex:
    """The per-file index record (§3.2): the resolved language, whether
    the parse was partial (error nodes — symbols may be missing), the
    definitions ordered by line and the file's import lines."""

    language: str
    partial: bool
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[ImportEntry] = field(default_factory=list)


def file_index_to_disk(index: FileIndex) -> dict[str, object]:
    """Serialize a FileIndex into the §3.2 JSON shape (the value half of
    an entry; the stat-key pair is the store's business)."""
    return {
        "language": index.language,
        "partial": index.partial,
        "symbols": [
            [
                s.name,
                s.kind,
                s.scope,
                s.signature,
                s.line,
                s.end_line,
                s.doc,
            ]
            for s in index.symbols
        ],
        "imports": [[i.kind, i.spec] for i in index.imports],
    }


def file_index_from_disk(raw: object) -> FileIndex | None:
    """Validate one disk entry into a FileIndex (None when the shape is
    foreign/broken — the entry is quietly rebuilt, never fatal)."""
    if not isinstance(raw, dict):
        return None
    language = raw.get("language")
    if not isinstance(language, str) or not language:
        return None
    raw_symbols = raw.get("symbols")
    raw_imports = raw.get("imports", [])
    if not isinstance(raw_symbols, list) or not isinstance(raw_imports, list):
        return None
    symbols: list[Symbol] = []
    for item in raw_symbols:
        if not (isinstance(item, list) and len(item) == 7):
            return None
        name, kind, scope, signature, line, end_line, doc = item
        if not (
            isinstance(name, str) and isinstance(kind, str) and isinstance(scope, str)
        ):
            return None
        if not (isinstance(signature, str) and isinstance(doc, str)):
            return None
        if not (isinstance(line, int) and isinstance(end_line, int) and line >= 1):
            return None
        symbols.append(Symbol(name, kind, scope, signature, line, end_line, doc))
    imports: list[ImportEntry] = []
    for item in raw_imports:
        if not (isinstance(item, list) and len(item) == 2):
            return None
        kind, spec = item
        if not (isinstance(kind, str) and isinstance(spec, str)):
            return None
        imports.append(ImportEntry(kind, spec))
    partial = raw.get("partial", False)
    if not isinstance(partial, bool):
        partial = False
    return FileIndex(language, partial, symbols, imports)


# -- language table -----------------------------------------------------------
# Extension → `tree-sitter-language-pack` language name (§2.3.3). Deliberately
# a self-contained mainstream table, not the pack's own detector: the mapping
# must be deterministic and testable, and ambiguous extensions (.h, .ts) must
# resolve to ONE language (a wrong guess yields a partially-useful parse at
# worst — the file stays greppable either way).

EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".cs": "csharp",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cppm": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".lua": "lua",
    ".zig": "zig",
    ".hs": "haskell",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".r": "r",
    ".jl": "julia",
    ".groovy": "groovy",
    ".v": "verilog",
    ".sql": "sql",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".vue": "vue",
    ".svelte": "svelte",
    ".proto": "proto",
    ".graphql": "graphql",
    ".nix": "nix",
    ".vim": "vim",
    ".tex": "latex",
    ".md": "markdown",
    ".dockerfile": "dockerfile",
}

# `definition.<suffix>` tags-query captures → Symbol kinds (§2.3's narrow
# scope: function/method/class/interface/struct/enum/trait/module/variable).
# Unknown suffixes degrade to "variable" — a named definition is still worth
# indexing under the honest weakest label.
_KIND_BY_CAPTURE: dict[str, str] = {
    "function": "function",
    "method": "method",
    "class": "class",
    "interface": "interface",
    "struct": "struct",
    "enum": "enum",
    "trait": "trait",
    "module": "module",
    "package": "module",
    "implementation": "class",
    "constant": "variable",
    "variable": "variable",
    "field": "variable",
    "component": "variable",
}

# Node-kind fallback for grammars whose bundled tags query is empty or
# missing (§2.3.2): {language: {tree-sitter node type: Symbol kind}}.
# Only mainstream languages pay this table (~5–10 lines each); everything
# else is honestly skipped.
NODE_KIND_TABLES: dict[str, dict[str, str]] = {
    "zig": {
        "function_declaration": "function",
        "test_declaration": "function",
    },
    "haskell": {
        "function": "function",
        "data_type": "struct",
        "type_synonym": "struct",
        "newtype": "struct",
    },
    "r": {
        "function_definition": "function",
    },
    "julia": {
        "function_definition": "function",
        "short_function_definition": "function",
        "struct_definition": "struct",
        "module_definition": "module",
        "abstract_definition": "interface",
    },
    "elixir": {
        "def": "function",
        "defp": "function",
        "defmodule": "module",
    },
    "ocaml": {
        "value_definition": "variable",
        "type_definition": "struct",
        "module_definition": "module",
    },
}


class _Compiled:
    """Per-language parse artifacts, built lazily (the first use of a
    language may download the grammar — dev-only, see §2.4) and cached
    for the process lifetime under `_COMPILE_LOCK`."""

    __slots__ = ("kinds", "language", "parser", "query")

    def __init__(
        self,
        parser: Parser,
        language: Language,
        query: Query | None,
        kinds: dict[str, str] | None,
    ) -> None:
        self.parser = parser
        self.language = language
        self.query = query
        self.kinds = kinds


_COMPILED: dict[str, _Compiled] = {}
_COMPILE_LOCK = threading.Lock()
_UNAVAILABLE: set[str] = set()
_CACHE_CONFIGURED = False


def _ensure_cache_configured() -> None:
    """Point the language pack at the FROZEN bundle's prefetched grammar
    cache when the entry stub exported one (§2.4.1/§2.4.2):
    `DRCODE_TS_PACK_CACHE` names the base directory the pack appends its
    versioned `libs` tree to. Dev runs never set it — the pack uses its
    own user cache and may download on demand. Once per process, under
    the compile lock."""
    global _CACHE_CONFIGURED
    if _CACHE_CONFIGURED:
        return
    _CACHE_CONFIGURED = True
    override = os.environ.get("DRCODE_TS_PACK_CACHE")
    if not override:
        return
    try:
        import tree_sitter_language_pack as pack

        pack.configure(pack.PackConfig(cache_dir=override))
    except Exception:  # noqa: BLE001, S110 - a broken override degrades to the default cache
        pass


def _compile(language: str) -> _Compiled | None:
    """Compile the language's parser + definition query (None when the
    grammar is unavailable — offline, no cache; the honest §2.4
    degradation). The result is memoized both ways: a working compile is
    reused, a failed one is remembered so a huge unindexed project does
    not retry the download on every file."""
    with _COMPILE_LOCK:
        _ensure_cache_configured()
        if language in _COMPILED:
            return _COMPILED[language]
        if language in _UNAVAILABLE:
            return None
        try:
            parser = get_parser(language)
            tree_language = parser.language
            if tree_language is None:  # pragma: no cover - get_parser sets it
                raise LanguageNotFoundError(language)
        except (DownloadError, LanguageNotFoundError, OSError, ValueError):
            _UNAVAILABLE.add(language)
            return None
        query: Query | None = None
        try:
            source = str(get_tags_query(language))
            # A trivial tags query (just a bare capture) extracts nothing
            # useful — treat it as missing and fall back to the kind table.
            query = Query(tree_language, source) if len(source) > 16 else None
        except Exception:  # noqa: BLE001 - a broken query is a per-language gap
            query = None
        compiled = _Compiled(
            parser, tree_language, query, NODE_KIND_TABLES.get(language)
        )
        _COMPILED[language] = compiled
        return compiled


def _doc_of(node: Node, src: bytes) -> str:
    """Best-effort doc: the first string/comment node within a shallow
    BFS under the definition (docstrings usually sit one statement deep
    in the body block) — its first ≤2 lines, ≤160 chars. Cheap and
    language-agnostic; misses degrade to ''."""
    _STRING_TYPES = {
        "string",
        "string_content",
        "comment",
        "line_comment",
        "block_comment",
        "documentation_comment",
    }
    seen = 0
    frontier = list(node.children)
    while frontier and seen < 24:
        candidate = frontier.pop(0)
        seen += 1
        ctype = candidate.type
        if ctype in _STRING_TYPES:
            # A docstring is the FIRST child of a block/suite-ish parent
            # (python: expression_statement(string) opening the body); a
            # string inside a return/print statement is content, not a
            # doc. Comments are accepted anywhere shallow.
            parent = candidate.parent
            if parent is None:
                continue
            blockish = parent.type in (
                "block",
                "body",
                "suite",
                "statement_block",
                "function_body",
                "class_body",
                "consequent",
            )
            first_child = bool(parent.children) and (
                parent.children[0].id == candidate.id
            )
            if (ctype == "comment" or blockish) and (
                first_child or parent.id == node.id
            ):
                text = src[candidate.start_byte : candidate.end_byte].decode(
                    "utf-8", errors="replace"
                )
                lines = [ln.strip().strip('"#*/ \t') for ln in text.splitlines()[:2]]
                doc = " ".join(ln for ln in lines if ln)[:DOC_MAX_CHARS]
                if doc:
                    return doc
        if len(frontier) < 8:
            frontier.extend(list(candidate.children)[:4])
    return ""


class SymbolExtractor:
    """The §2.2 extractor interface: `languages()` (the names this
    extractor can resolve) and `extract(path, text)` (the per-file
    record, or None when the file's language is not indexable)."""

    def languages(self) -> set[str]:
        """Language names the extension table can resolve (whether the
        grammar is locally available is checked per file by `extract`)."""
        return set(EXTENSION_LANGUAGES.values())

    def language_for(self, path: Path) -> str | None:
        """Resolve the language from the file extension (None = not
        indexed; the file is then honestly absent from every answer)."""
        return EXTENSION_LANGUAGES.get(path.suffix.lower())

    def extract(self, path: Path, text: str) -> FileIndex | None:
        """Extract the definitions and import lines of one file. Returns
        None when the extension is unknown or the grammar is unavailable
        (§2.4.3) — the file is skipped, never an error. A parse with
        error nodes yields a `partial=True` record (symbols may be
        missing); a byte-identical dedup of kinds keeps the order by
        line."""
        language = self.language_for(path)
        if language is None or not has_language(language):
            return None
        compiled = _compile(language)
        if compiled is None:
            return None
        src = text.encode("utf-8", errors="replace")
        tree = compiled.parser.parse(src)
        root = tree.root_node
        partial = root.has_error

        symbols: list[Symbol] = []
        if compiled.query is not None:
            symbols = self._from_tags_query(root, src, compiled.query)
        elif compiled.kinds:
            symbols = self._from_kind_table(root, src, compiled.kinds)
        symbols.sort(key=lambda s: (s.line, s.end_line))
        return FileIndex(
            language=language,
            partial=partial,
            symbols=symbols,
            imports=extract_imports(text, language),
        )

    def _from_tags_query(self, root: Node, src: bytes, query: Query) -> list[Symbol]:
        """The tags-query path (§2.3.1): walk the query matches, pair
        each `@definition.*` capture with its `@name`, and derive the
        flat dotted scope from the enclosing definitions' own names."""
        try:
            matches = QueryCursor(query).matches(root)
        except Exception:  # noqa: BLE001 - a query/runtime mismatch is a gap, not a crash
            return []
        name_by_node: dict[int, tuple[int, str]] = {}
        raw: list[tuple[Node, str, str]] = []  # (def_node, name, kind)
        for _pattern, captures in matches:
            def_nodes: list[Node] = []
            kind = ""
            for capture, nodes in captures.items():
                if capture == "name" or not nodes:
                    continue
                if capture.startswith("definition."):
                    kind = _KIND_BY_CAPTURE.get(
                        capture.removeprefix("definition."), "variable"
                    )
                    def_nodes.extend(nodes)
            names = captures.get("name") or []
            if not kind or not def_nodes or not names:
                continue
            node = def_nodes[0]
            name_node = names[0]
            name = src[name_node.start_byte : name_node.end_byte].decode(
                "utf-8", errors="replace"
            )
            if not name:
                continue
            raw.append((node, name, kind))
            name_by_node[node.id] = (node.start_point[0], name)
        symbols: list[Symbol] = []
        for node, name, kind in raw:
            scope_parts: list[str] = []
            parent = getattr(node, "parent", None)
            while parent is not None:
                entry = name_by_node.get(parent.id)
                if entry is not None:
                    scope_parts.append(entry[1])
                parent = getattr(parent, "parent", None)
            first_line = (
                src[node.start_byte : node.end_byte]
                .decode("utf-8", errors="replace")
                .splitlines()
            )
            signature = (first_line[0].strip() if first_line else "")[
                :SIGNATURE_MAX_CHARS
            ]
            symbols.append(
                Symbol(
                    name=name,
                    kind=kind,
                    scope=".".join(reversed(scope_parts)),
                    signature=signature,
                    line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    doc=_doc_of(node, src),
                )
            )
        return symbols

    def _from_kind_table(
        self, root: Node, src: bytes, kinds: dict[str, str]
    ) -> list[Symbol]:
        """The node-kind-table path (§2.3.2): walk the tree, every node
        whose type is in the table is a definition; the name is its
        first identifier-like child's text, the scope comes from the
        enclosing table matches the same way as the tags path."""
        name_by_node: dict[int, str] = {}
        raw: list[tuple[Node, str, str]] = []
        stack = [root]
        while stack:
            node = stack.pop()
            ntype = getattr(node, "type", "")
            kind = kinds.get(ntype)
            if kind is not None:
                name = ""
                for child in getattr(node, "children", []):
                    if getattr(child, "type", "") in (
                        "identifier",
                        "name",
                        "type_identifier",
                        "property_identifier",
                    ):
                        name = src[child.start_byte : child.end_byte].decode(
                            "utf-8", errors="replace"
                        )
                        break
                if name:
                    raw.append((node, name, kind))
                    name_by_node[node.id] = name
            stack.extend(getattr(node, "children", []))
        symbols: list[Symbol] = []
        for node, name, kind in raw:
            scope_parts: list[str] = []
            parent = getattr(node, "parent", None)
            while parent is not None:
                entry = name_by_node.get(parent.id)
                if entry is not None:
                    scope_parts.append(entry)
                parent = getattr(parent, "parent", None)
            text = src[node.start_byte : node.end_byte].decode(
                "utf-8", errors="replace"
            )
            first = text.splitlines()
            symbols.append(
                Symbol(
                    name=name,
                    kind=kind,
                    scope=".".join(reversed(scope_parts)),
                    signature=(first[0].strip() if first else "")[:SIGNATURE_MAX_CHARS],
                    line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    doc=_doc_of(node, src),
                )
            )
        return symbols


# -- import lines (§4.2: ~6 family templates, plain line scan, no AST) --------

_ES_IMPORT_PREFIXES = ("import ", "export ")


def extract_imports(text: str, language: str) -> list[ImportEntry]:
    """The file's import lines, matched against the language-family
    templates of §4.2. A crude line scan on purpose (no AST): dozens of
    entries per file, feeding the phase-2 import graph. Only the RAW
    spec is stored — resolution heuristics live in the queries layer."""
    family = _IMPORT_FAMILY.get(language)
    if family is None:
        return []
    imports: list[ImportEntry] = []
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        spec = _match_import(stripped, family)
        if spec and spec not in seen:
            seen.add(spec)
            imports.append(ImportEntry(family, spec))
    return imports


_IMPORT_FAMILY: dict[str, str] = {
    "python": "python",
    "typescript": "es-family",
    "tsx": "es-family",
    "javascript": "es-family",
    "go": "go",
    "java": "java-family",
    "kotlin": "java-family",
    "csharp": "java-family",
    "c": "c-family",
    "cpp": "c-family",
    "rust": "rust",
    "scala": "java-family",
}


def _match_import(stripped: str, family: str) -> str:
    """The raw spec of one import line (or '' when the line is not an
    import of the family)."""
    if family == "python":
        if stripped.startswith("import "):
            return stripped.removeprefix("import ").split(" as ")[0].strip()
        if stripped.startswith("from ") and " import " in stripped:
            return stripped.removeprefix("from ").split(" import ")[0].strip()
    elif family == "es-family":
        if stripped.startswith(_ES_IMPORT_PREFIXES) and "from" in stripped:
            part = stripped.rsplit("from", 1)[1].strip().strip("'\";")
            return part
        if stripped.startswith("import(") and stripped.endswith(")"):
            return stripped[len("import(") : -1].strip("'\"")
        if stripped.startswith("require(") and stripped.endswith(")"):
            return stripped[len("require(") : -1].strip("'\"")
    elif family == "go":
        if stripped.startswith("import ") and stripped.endswith('"'):
            return stripped.removeprefix("import ").strip().strip('"')
        if stripped.startswith('"') and stripped.endswith('"'):
            return stripped.strip('"')
    elif family == "java-family":
        if stripped.startswith("import ") and stripped.endswith(";"):
            return stripped.removeprefix("import ").rstrip(";").strip()
        if stripped.startswith("package ") and stripped.endswith(";"):
            return stripped.removeprefix("package ").rstrip(";").strip()
    elif family == "c-family":
        if stripped.startswith("#include"):
            part = stripped.removeprefix("#include").strip()
            return part.strip('"<> ')
    elif family == "rust":
        if stripped.startswith("use "):
            return stripped.removeprefix("use ").rstrip(";").strip()
        if stripped.startswith("mod ") and stripped.endswith(";"):
            return stripped.removeprefix("mod ").rstrip(";").strip()
    return ""
