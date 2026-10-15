"""Filesystem and shell tools for the coding agent, jailed to the project root."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

MAX_TOOL_OUTPUT = 10_000
BASH_TIMEOUT = 120

IGNORED_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    ".drcode",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
}


def resolve_path(root: Path, path: str) -> Path:
    p = (root / path).resolve()
    if not (p == root or root in p.parents):
        raise ValueError(f"path escapes project root: {path}")
    return p


def read_file(root: Path, path: str, offset: int = 1, limit: int = 2000) -> str:
    p = resolve_path(root, path)
    lines = p.read_text(errors="replace").splitlines()
    start = max(offset - 1, 0)
    chunk = lines[start : start + limit]
    numbered = [f"{start + i + 1}: {line}" for i, line in enumerate(chunk)]
    return "\n".join(numbered) or "(empty file)"


def write_file(root: Path, path: str, content: str) -> str:
    p = resolve_path(root, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"written {len(content)} chars to {path}"


def edit_file(root: Path, path: str, old_string: str, new_string: str) -> str:
    p = resolve_path(root, path)
    text = p.read_text(errors="replace")
    count = text.count(old_string)
    if count == 0:
        raise ValueError("old_string not found in file")
    if count > 1:
        raise ValueError(
            f"old_string matches {count} locations, provide more context to make it unique"
        )
    p.write_text(text.replace(old_string, new_string, 1))
    return f"edited {path}"


def glob_files(root: Path, pattern: str) -> list[str]:
    matches = [p for p in root.glob(pattern) if p.is_file()]
    return sorted(str(p.relative_to(root)) for p in matches)[:200]


def grep_search(root: Path, pattern: str, include: str = "") -> list[str]:
    regex = re.compile(pattern)
    include_regex = re.compile(include) if include else None
    results: list[str] = []
    for dirpath, dirnames, filenames in root.walk():
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for name in filenames:
            if include_regex and not include_regex.search(name):
                continue
            f = Path(dirpath) / name
            try:
                for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                    if regex.search(line):
                        results.append(
                            f"{f.relative_to(root)}:{i}: {line.strip()[:200]}"
                        )
                        if len(results) >= 100:
                            return results
            except OSError:
                continue
    return results


def run_bash(root: Path, command: str) -> str:
    proc = subprocess.run(
        command,
        shell=True,
        check=False,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=BASH_TIMEOUT,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    out = out.strip()[:MAX_TOOL_OUTPUT]
    return f"exit={proc.returncode}\n{out or '(no output)'}"
