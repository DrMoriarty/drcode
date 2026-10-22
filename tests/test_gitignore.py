"""Tests for drcode.gitignore: drcode's private state kept out of version
control, team-shared workflow files committable."""

from __future__ import annotations

from pathlib import Path

from drcode.gitignore import GITIGNORE_ENTRIES, ensure_drcode_gitignored

ENTRIES = "\n".join(GITIGNORE_ENTRIES) + "\n"


def make_git_repo(root: Path) -> None:
    (root / ".git").mkdir()


def test_creates_gitignore_in_git_repo(project_root: Path) -> None:
    make_git_repo(project_root)
    ensure_drcode_gitignored(project_root)
    assert (project_root / ".gitignore").read_text() == ENTRIES


def test_appends_to_existing_gitignore(project_root: Path) -> None:
    make_git_repo(project_root)
    (project_root / ".gitignore").write_text("*.pyc\nbuild/\n")
    ensure_drcode_gitignored(project_root)
    text = (project_root / ".gitignore").read_text()
    assert text.startswith("*.pyc\nbuild/\n")
    assert text.endswith("\n" + ENTRIES)


def test_no_trailing_blank_line_added(project_root: Path) -> None:
    make_git_repo(project_root)
    (project_root / ".gitignore").write_text("*.pyc")  # no trailing newline
    ensure_drcode_gitignored(project_root)
    assert (project_root / ".gitignore").read_text() == "*.pyc\n" + ENTRIES


def test_no_duplicate_entries(project_root: Path) -> None:
    make_git_repo(project_root)
    (project_root / ".gitignore").write_text(ENTRIES)
    ensure_drcode_gitignored(project_root)
    assert (project_root / ".gitignore").read_text() == ENTRIES


def test_partial_entries_completed(project_root: Path) -> None:
    make_git_repo(project_root)
    (project_root / ".gitignore").write_text(".drcode/tasks/\n")
    ensure_drcode_gitignored(project_root)
    assert (project_root / ".gitignore").read_text() == ENTRIES


def test_legacy_broad_entry_is_narrowed(project_root: Path) -> None:
    """Older drcode builds appended a whole-directory entry; it must be
    replaced so the team-shared workflow/model files can be committed."""
    make_git_repo(project_root)
    (project_root / ".gitignore").write_text("*.pyc\n.drcode/\n")
    ensure_drcode_gitignored(project_root)
    text = (project_root / ".gitignore").read_text()
    assert ".drcode/\n" not in text.replace(".drcode/tasks/\n", "")
    assert "*.pyc\n" in text
    for entry in GITIGNORE_ENTRIES:
        assert entry in text


def test_legacy_anchored_broad_entry_is_narrowed(project_root: Path) -> None:
    make_git_repo(project_root)
    (project_root / ".gitignore").write_text("/.drcode\n")
    ensure_drcode_gitignored(project_root)
    assert (project_root / ".gitignore").read_text() == ENTRIES


def test_negated_entry_left_alone(project_root: Path) -> None:
    """A custom negation means the user manages the entries by hand —
    never rewrite those."""
    make_git_repo(project_root)
    (project_root / ".gitignore").write_text("!.drcode\n")
    ensure_drcode_gitignored(project_root)
    assert (project_root / ".gitignore").read_text() == "!.drcode\n" + ENTRIES


def test_nested_drcode_mention_not_touched(project_root: Path) -> None:
    make_git_repo(project_root)
    (project_root / ".gitignore").write_text("sub/.drcode/\n")
    ensure_drcode_gitignored(project_root)
    text = (project_root / ".gitignore").read_text()
    assert "sub/.drcode/\n" in text
    for entry in GITIGNORE_ENTRIES:
        assert entry in text


def test_no_git_repo_untouched(project_root: Path) -> None:
    ensure_drcode_gitignored(project_root)
    assert not (project_root / ".gitignore").exists()


def test_git_worktree_dotgit_file_counts(project_root: Path) -> None:
    # In worktrees/submodules `.git` is a file pointing at the real dir.
    (project_root / ".git").write_text("gitdir: ../main/.git\n")
    ensure_drcode_gitignored(project_root)
    assert (project_root / ".gitignore").is_file()


def test_unreadable_gitignore_left_alone(project_root: Path) -> None:
    make_git_repo(project_root)
    gitignore = project_root / ".gitignore"
    gitignore.write_bytes(b"\xff\xfe\x00")  # not valid UTF-8
    ensure_drcode_gitignored(project_root)  # must not raise
    assert gitignore.read_bytes() == b"\xff\xfe\x00"
