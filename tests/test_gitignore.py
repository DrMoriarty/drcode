"""Tests for drcode.gitignore: `.drcode/` kept out of version control."""

from __future__ import annotations

from pathlib import Path

from drcode.gitignore import ensure_drcode_gitignored


def make_git_repo(root: Path) -> None:
    (root / ".git").mkdir()


def test_creates_gitignore_in_git_repo(project_root: Path) -> None:
    make_git_repo(project_root)
    ensure_drcode_gitignored(project_root)
    assert (project_root / ".gitignore").read_text() == ".drcode/\n"


def test_appends_to_existing_gitignore(project_root: Path) -> None:
    make_git_repo(project_root)
    (project_root / ".gitignore").write_text("*.pyc\nbuild/\n")
    ensure_drcode_gitignored(project_root)
    text = (project_root / ".gitignore").read_text()
    assert text.startswith("*.pyc\nbuild/\n")
    assert text.endswith("\n.drcode/\n")


def test_no_trailing_blank_line_added(project_root: Path) -> None:
    make_git_repo(project_root)
    (project_root / ".gitignore").write_text("*.pyc")  # no trailing newline
    ensure_drcode_gitignored(project_root)
    assert (project_root / ".gitignore").read_text() == "*.pyc\n.drcode/\n"


def test_no_duplicate_entry(project_root: Path) -> None:
    make_git_repo(project_root)
    (project_root / ".gitignore").write_text(".drcode/\n")
    ensure_drcode_gitignored(project_root)
    assert (project_root / ".gitignore").read_text().count(".drcode") == 1


def test_anchored_entry_counts(project_root: Path) -> None:
    make_git_repo(project_root)
    (project_root / ".gitignore").write_text("/.drcode\n")
    ensure_drcode_gitignored(project_root)
    assert (project_root / ".gitignore").read_text().count(".drcode") == 1


def test_comments_do_not_count(project_root: Path) -> None:
    make_git_repo(project_root)
    (project_root / ".gitignore").write_text("# .drcode\n")
    ensure_drcode_gitignored(project_root)
    assert ".drcode/" in (project_root / ".gitignore").read_text()


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
