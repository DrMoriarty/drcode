"""The optional universal-ctags extractor (§2.2 — "tree-sitter — ядро,
ctags — опциональный адаптер").

Covers languages OUTSIDE the tree-sitter pack's extension table: the
adapter is consulted by the store only when the primary extractor
returns no record (unknown extension or unavailable grammar), and it
deliberately refuses every extension tree-sitter already covers — it is
a coverage extension, never a second source of truth.

Auto-detection (§12 фаза 2): the binary must be found in PATH AND be a
universal-ctags (`--output-format=json` is a u-ctags feature; the BSD
/macOS legacy ctags rejects it). When unavailable the adapter is inert:
`language_for` returns None for everything, the store skips the file
honestly (§6.3 — grep_search keeps the coverage) and the feature works
without it by design.

A per-file subprocess is fine on purpose: ctags only ever sees the
out-of-pack tail of a project, a handful of files per index build.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .extract import (
    EXTENSION_LANGUAGES,
    SIGNATURE_MAX_CHARS,
    FileIndex,
    Symbol,
)

__all__ = ["CtagsExtractor"]


# Languages u-ctags knows that the tree-sitter extension table does not
# (§2.2: the adapter extends coverage). A static mainstream table first —
# deterministic and testable; anything else falls back to a cached
# `--print-language` probe.
CTAGS_LANGUAGE_BY_EXT: dict[str, str] = {
    ".sql": "SQL",
    ".pl": "Perl",
    ".pm": "Perl",
    ".t": "Perl",
    ".tcl": "Tcl",
    ".tk": "Tcl",
    ".vim": "Vim",
    ".groovy": "Groovy",
    ".gradle": "Gradle",
    ".jenkinsfile": "Jenkinsfile",
    ".d": "D",
    ".di": "D",
    ".nim": "Nim",
    ".hx": "Haxe",
    ".hxml": "Haxe",
    ".purs": "PureScript",
    ".vhd": "VHDL",
    ".vhdl": "VHDL",
    ".v": "Verilog",
    ".sv": "SystemVerilog",
    ".svh": "SystemVerilog",
    ".vh": "SystemVerilog",
    ".cob": "COBOL",
    ".cbl": "COBOL",
    ".f90": "Fortran",
    ".f95": "Fortran",
    ".f03": "Fortran",
    ".f08": "Fortran",
    ".m": "MatLab",
    ".mm": "ObjectiveC",
    ".raku": "Raku",
    ".rakumod": "Raku",
    ".p6": "Raku",
    ".pro": "Prolog",
    ".adb": "Ada",
    ".ads": "Ada",
    ".ada": "Ada",
    ".ps1": "PowerShell",
    ".psm1": "PowerShell",
    ".bat": "DosBatch",
    ".cmd": "DosBatch",
    ".asm": "Asm",
    ".s": "Asm",
    ".st": "Smalltalk",
    ".abap": "ABAP",
    ".4th": "Forth",
    ".fs": "Forth",
}


# u-ctags kind names / letters → the §3.2 Symbol kinds. Kinds outside the
# table (locals, labels, sections, …) are dropped — the extraction scope
# stays «function/method/class/…» by design (§2.3).
# Extensions never probed (docs/data/config — u-ctags HAS parsers for some
# of them, their "symbols" are headings and keys, and probing every such
# file would burn a subprocess for noise).
_NON_SOURCE_EXTS = frozenset(
    {
        ".md",
        ".rst",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".xml",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".less",
        ".svg",
        ".csv",
        ".tsv",
        ".log",
        ".lock",
        ".env",
        ".envrc",
        ".gitignore",
        ".license",
        ".ico",
        ".png",
        ".jpg",
        ".gif",
        ".pdf",
    }
)


_KIND_BY_CTAGS: dict[str, str] = {
    "function": "function",
    "func": "function",
    "f": "function",
    "method": "method",
    "m": "method",
    "class": "class",
    "c": "class",
    "interface": "interface",
    "i": "interface",
    "struct": "struct",
    "s": "struct",
    "enum": "enum",
    "g": "enum",
    "trait": "trait",
    "module": "module",
    "namespace": "module",
    "package": "module",
    "mod": "module",
    "n": "module",
    "variable": "variable",
    "var": "variable",
    "constant": "variable",
    "const": "variable",
    "member": "variable",
    "property": "variable",
    "typedef": "variable",
    "v": "variable",
    "t": "variable",
}


class CtagsExtractor:
    """The `SymbolExtractor`-shaped ctags fallback: `language_for(path)`
    (None when the extension is tree-sitter's business or the binary is
    unusable) and `extract(path, text)` (None on any failure — the file
    is skipped, never an error)."""

    def __init__(self, binary: str | None = None):
        self._binary = binary or shutil.which("ctags")
        self._capable: bool | None = None
        self._languages: dict[str, str | None] = {}

    @property
    def binary(self) -> str | None:
        return self._binary

    def _is_capable(self) -> bool:
        """universal-ctags only: the legacy BSD ctags that ships with
        macOS rejects `--output-format=json` (verified on this machine) —
        one cheap `--version` probe, cached."""
        if self._binary is None:
            return False
        if self._capable is None:
            try:
                proc = subprocess.run(
                    [self._binary, "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self._capable = "universal ctags" in proc.stdout.lower()
            except (OSError, subprocess.SubprocessError):
                self._capable = False
        return self._capable

    def language_for(self, path: Path) -> str | None:
        """The ctags language name for files OUTSIDE the tree-sitter
        table (None otherwise — including "no ctags / unknown file")."""
        if not self._is_capable():
            return None
        ext = path.suffix.lower()
        if ext in EXTENSION_LANGUAGES or ext in _NON_SOURCE_EXTS:
            return None  # tree-sitter's business / not source at all
        if ext in self._languages:
            return self._languages[ext]
        if ext in CTAGS_LANGUAGE_BY_EXT:
            self._languages[ext] = CTAGS_LANGUAGE_BY_EXT[ext]
            return self._languages[ext]
        # Unknown extension: one cached probe per extension.
        self._languages[ext] = self._probe_language(path)
        return self._languages[ext]

    def _probe_language(self, path: Path) -> str | None:
        assert self._binary is not None
        try:
            proc = subprocess.run(
                [self._binary, "--print-language", str(path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        language = proc.stdout.strip()
        if not language or "unknown" in language.lower():
            return None
        return language

    def extract(self, path: Path, text: str) -> FileIndex | None:
        """Run u-ctags over ONE file and map its JSON tags onto the §3.2
        record. Any failure (no binary, no language, subprocess error,
        junk output) → None: the file is honestly absent, never fatal.
        `partial` is always False — ctags exposes no error nodes."""
        language = self.language_for(path)
        if language is None:
            return None
        assert self._binary is not None
        try:
            proc = subprocess.run(
                [
                    self._binary,
                    f"--languages={language}",
                    "--output-format=json",
                    "--fields=+n+e+S+K",
                    "--kind-long",
                    "-f",
                    "-",
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        lines = text.splitlines()
        symbols: list[Symbol] = []
        for raw_line in proc.stdout.splitlines():
            raw_line = raw_line.strip()
            if not raw_line.startswith("{"):
                continue
            try:
                tag = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(tag, dict) or tag.get("_type") != "tag":
                continue
            name = tag.get("name")
            line = tag.get("line")
            if not isinstance(name, str) or not isinstance(line, int) or line < 1:
                continue
            kind = _KIND_BY_CTAGS.get(str(tag.get("kind", "")).lower())
            if kind is None:
                continue
            signature = tag.get("signature")
            if not isinstance(signature, str) or not signature:
                # Fall back to the definition's own first line.
                signature = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            scope = tag.get("scope")
            scope_str = scope if isinstance(scope, str) and scope else ""
            end = tag.get("end")
            end_line = end if isinstance(end, int) and end >= line else line
            symbols.append(
                Symbol(
                    name=name,
                    kind=kind,
                    scope=scope_str,
                    signature=signature[:SIGNATURE_MAX_CHARS],
                    line=line,
                    end_line=end_line,
                )
            )
        symbols.sort(key=lambda s: (s.line, s.end_line))
        return FileIndex(language=language.lower(), partial=False, symbols=symbols)
