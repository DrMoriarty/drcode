"""Keeps `.drcode/` out of version control.

`.drcode/` holds per-project state: the task transcripts (via
`drcode.storage`) and the per-project config `drcode.json` (via
`drcode.config`). None of it belongs in a repository, so whenever the
project directory contains a git repository a `.drcode/` entry is
appended to the project's `.gitignore` — creating the file when it does
not exist yet. Projects without a git repository are left untouched
(no stray `.gitignore` is created for them).
"""

from __future__ import annotations

from pathlib import Path

GITIGNORE_ENTRY = ".drcode/"


def _is_git_repo(project_root: Path) -> bool:
    # `.git` is a directory for normal clones and a plain file for
    # worktrees/submodules — `exists()` covers both shapes.
    return (project_root / ".git").exists()


def _already_ignored(lines: list[str]) -> bool:
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue  # blank lines and comments don't ignore anything
        # Match `.drcode`, `.drcode/` and anchored `/.drcode` spellings,
        # including negations (`!.drcode` still mentions the entry).
        if stripped.lstrip("!").rstrip("/") == ".drcode" or stripped.endswith(
            "/.drcode"
        ):
            return True
    return False


def ensure_drcode_gitignored(project_root: Path) -> None:
    """Add `.drcode/` to the project's `.gitignore` when it's a git repo.

    Idempotent: an existing (non-comment) mention of `.drcode` is never
    duplicated. A missing `.gitignore` is created. Never raises: an
    unwritable project directory just means the user has to ignore the
    directory by hand.
    """
    if not _is_git_repo(project_root):
        return
    gitignore = project_root / ".gitignore"
    lines: list[str] = []
    if gitignore.is_file():
        try:
            lines = gitignore.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return  # unreadable file — don't risk corrupting it further
    if _already_ignored(lines):
        return
    try:
        with gitignore.open("a", encoding="utf-8") as f:
            if lines and lines[-1].strip():
                f.write("\n")  # keep appended entries on their own line
            f.write(GITIGNORE_ENTRY + "\n")
    except OSError:
        pass
