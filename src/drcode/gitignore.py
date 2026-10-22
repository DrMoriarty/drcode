"""Keeps drcode's private per-project state out of version control.

`.drcode/` mixes private state with team-shared configuration: the task
transcripts (`drcode.storage`), the per-project settings with secrets
(`drcode.settings`) and the GitHub-sync bookkeeping never belong in a
repository, while the workflow files (`.drcode/git-workflow.md`,
`.drcode/workflow.py`) and the model selection (`.drcode/drcode.json`) are
deliberately committable — the team shares its branching/review rules and
model picks through them.

Whenever the project directory contains a git repository, the project's
`.gitignore` is therefore made to ignore exactly the private subset
(below). A legacy broad `.drcode/` entry written by older drcode builds is
replaced by the specific set, so the shared files can actually be
committed; any other pre-existing `.drcode` mention (a custom path, a
negation) is left untouched. Projects without a git repository are left
alone (no stray `.gitignore` is created for them).
"""

from __future__ import annotations

from pathlib import Path

# The private subset: per-task transcripts (may contain proprietary code
# and conversation), settings (tokens) and sync bookkeeping (machine state).
GITIGNORE_ENTRIES: tuple[str, ...] = (
    ".drcode/tasks/",
    ".drcode/settings.json",
    ".drcode/github_state.json",
    ".drcode/worktrees/",
)

# Spellings of the legacy whole-directory entry older drcode versions
# appended (bare, non-negated). These are migrated to GITIGNORE_ENTRIES.
_BROAD_ENTRIES = frozenset({".drcode", ".drcode/", "/.drcode", "/.drcode/"})


def _is_git_repo(project_root: Path) -> bool:
    # `.git` is a directory for normal clones and a plain file for
    # worktrees/submodules — `exists()` covers both shapes.
    return (project_root / ".git").exists()


def _is_broad_entry(line: str) -> bool:
    stripped = line.strip()
    return not stripped.startswith("!") and stripped in _BROAD_ENTRIES


def _has_entry(lines: list[str], entry: str) -> bool:
    return any(line.strip() == entry for line in lines)


def _write(gitignore: Path, lines: list[str]) -> None:
    gitignore.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_drcode_gitignored(project_root: Path) -> None:
    """Make the project's `.gitignore` ignore drcode's private state.

    Idempotent: entries already present are never duplicated. A missing
    `.gitignore` is created. Never raises: an unwritable project directory
    just means the user has to ignore the directory by hand.
    """
    if not _is_git_repo(project_root):
        return
    gitignore = project_root / ".gitignore"
    lines: list[str] = []
    ends_nl = True
    if gitignore.is_file():
        try:
            raw = gitignore.read_text(encoding="utf-8")
            lines = raw.splitlines()
            ends_nl = raw.endswith("\n") or not raw
        except (OSError, UnicodeDecodeError):
            return  # unreadable file — don't risk corrupting it further
    try:
        broad = [i for i, line in enumerate(lines) if _is_broad_entry(line)]
        if broad:
            # Legacy broad entry: replace it with the private subset, so
            # the team-shared workflow/model files stay committable.
            kept = [line for i, line in enumerate(lines) if i not in broad]
            while kept and not kept[-1].strip():
                kept.pop()  # collapse the whitespace the removal left
            missing = [e for e in GITIGNORE_ENTRIES if not _has_entry(kept, e)]
            if missing:
                if kept and kept[-1].strip():
                    kept.append("")
                kept.extend(missing)
            _write(gitignore, kept)
            return
        missing = [e for e in GITIGNORE_ENTRIES if not _has_entry(lines, e)]
        if not missing:
            return
        with gitignore.open("a", encoding="utf-8") as f:
            if lines and not ends_nl:
                # The file lacks a trailing newline: keep the appended
                # entries on their own line.
                f.write("\n")
            for entry in missing:
                f.write(entry + "\n")
    except OSError:
        pass
