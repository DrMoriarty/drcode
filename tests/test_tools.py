"""Tests for drcode.tools: path permissions, file tools, grep/glob, bash."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import pytest

from drcode import tools
from drcode.tools import (
    IGNORED_DIRS,
    PathViolation,
    check_command,
    edit_file,
    glob_files,
    grep_search,
    is_writable,
    read_file,
    resolve_edit_path,
    resolve_read_path,
    run_bash,
    write_file,
)

# -- path resolution & permissions ----------------------------------------------


def test_resolve_read_path_relative_against_root(project_root: Path) -> None:
    p = resolve_read_path(project_root, "src/main.py")
    assert p == (project_root / "src/main.py").resolve()


def test_resolve_read_path_absolute(project_root: Path) -> None:
    p = resolve_read_path(project_root, "/etc/passwd")
    assert p == Path("/etc/passwd").resolve()


def test_resolve_read_path_expands_user(project_root: Path) -> None:
    p = resolve_read_path(project_root, "~/notes.txt")
    assert not str(p).startswith("~")


def test_is_writable_inside_root(project_root: Path) -> None:
    assert is_writable(project_root, project_root)
    assert is_writable(project_root, project_root / "src" / "main.py")


def test_is_writable_tmp(project_root: Path) -> None:
    assert is_writable(project_root, Path("/tmp/scratch/x.txt"))


def test_is_writable_outside(project_root: Path) -> None:
    assert not is_writable(project_root, Path("/etc/passwd"))
    assert not is_writable(project_root, project_root.parent / "sneaky.txt")
    assert not is_writable(project_root, Path("/tmpfoo/x.txt"))


def test_resolve_edit_path_outside_raises(project_root: Path) -> None:
    with pytest.raises(PathViolation):
        resolve_edit_path(project_root, "/etc/hosts")


def test_resolve_edit_path_symlink_escape_raises(project_root: Path) -> None:
    outside = project_root.parent / "outside.txt"
    outside.write_text("secret")
    link = project_root / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(PathViolation):
        resolve_edit_path(project_root, "link.txt")


def test_resolve_edit_path_symlink_into_root_ok(project_root: Path) -> None:
    target = project_root / "real.txt"
    target.write_text("ok")
    link = project_root / "link.txt"
    link.symlink_to(target)
    assert resolve_edit_path(project_root, "link.txt") == target.resolve()


def test_resolve_edit_path_relative(project_root: Path) -> None:
    assert (
        resolve_edit_path(project_root, "a/b.txt")
        == (project_root / "a/b.txt").resolve()
    )


def test_resolve_edit_path_macos_tmp_symlink(project_root: Path) -> None:
    # /tmp may resolve elsewhere; both literal and resolved must be accepted.
    resolved = Path("/tmp").resolve()
    assert is_writable(project_root, resolved / "x.txt")


# -- read_file ------------------------------------------------------------------


def test_read_file_numbering(project_root: Path) -> None:
    (project_root / "f.txt").write_text("alpha\nbeta\ngamma\n")
    out = read_file(project_root, "f.txt")
    assert out == "1: alpha\n2: beta\n3: gamma"


def test_read_file_offset_and_limit(project_root: Path) -> None:
    (project_root / "f.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)))
    out = read_file(project_root, "f.txt", offset=3, limit=2)
    assert out == "3: line3\n4: line4"


def test_read_file_offset_beyond_end(project_root: Path) -> None:
    (project_root / "f.txt").write_text("one")
    out = read_file(project_root, "f.txt", offset=100)
    assert out == "(empty file)"


def test_read_file_empty(project_root: Path) -> None:
    (project_root / "f.txt").write_text("")
    assert read_file(project_root, "f.txt") == "(empty file)"


def test_read_file_anywhere_allowed(project_root: Path) -> None:
    outside = project_root.parent / "outside.txt"
    outside.write_text("readable")
    assert "readable" in read_file(project_root, str(outside))


def test_read_file_missing_raises(project_root: Path) -> None:
    with pytest.raises(OSError):
        read_file(project_root, "nope.txt")


# -- write_file ------------------------------------------------------------------


def test_write_file_creates_parents(project_root: Path) -> None:
    msg = write_file(project_root, "deep/nested/f.txt", "hello")
    assert (project_root / "deep/nested/f.txt").read_text() == "hello"
    assert "5 chars" in msg  # char count mentioned in the reply


def test_write_file_overwrites(project_root: Path) -> None:
    (project_root / "f.txt").write_text("old")
    write_file(project_root, "f.txt", "new")
    assert (project_root / "f.txt").read_text() == "new"


def test_write_file_to_tmp_allowed(project_root: Path, tmp_path: Path) -> None:
    # /tmp (with all subdirectories) is a writable scratch area.
    scratch = tools.TMP_ROOT / f"drcode_test_{os.getpid()}"
    try:
        write_file(project_root, str(scratch / "f.txt"), "data")
        assert (scratch / "f.txt").read_text() == "data"
    finally:
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)


def test_write_file_outside_refused(project_root: Path) -> None:
    with pytest.raises(PathViolation):
        write_file(project_root, "../evil.txt", "nope")


def test_write_file_home_refused(project_root: Path) -> None:
    with pytest.raises(PathViolation):
        write_file(project_root, "~/evil.txt", "nope")


# -- edit_file ------------------------------------------------------------------


def test_edit_file_replaces_unique(project_root: Path) -> None:
    (project_root / "f.txt").write_text("hello world")
    msg = edit_file(project_root, "f.txt", "world", "there")
    assert (project_root / "f.txt").read_text() == "hello there"
    assert "edited" in msg


def test_edit_file_not_found(project_root: Path) -> None:
    (project_root / "f.txt").write_text("hello")
    with pytest.raises(ValueError, match="not found"):
        edit_file(project_root, "f.txt", "missing", "x")


def test_edit_file_multiple_matches(project_root: Path) -> None:
    (project_root / "f.txt").write_text("a a")
    with pytest.raises(ValueError, match="2 locations"):
        edit_file(project_root, "f.txt", "a", "b")


def test_edit_file_outside_refused(project_root: Path) -> None:
    with pytest.raises(PathViolation):
        edit_file(project_root, "/etc/hosts", "x", "y")


# -- glob_files / grep_search ------------------------------------------------------


def test_glob_files_relative_sorted(project_root: Path) -> None:
    (project_root / "src").mkdir()
    (project_root / "src" / "b.py").write_text("x")
    (project_root / "src" / "a.py").write_text("x")
    (project_root / "src" / "sub").mkdir()
    (project_root / "src" / "sub" / "c.py").write_text("x")
    assert glob_files(project_root, "src/**/*.py") == [
        "src/a.py",
        "src/b.py",
        "src/sub/c.py",
    ]


def test_glob_files_dirs_excluded(project_root: Path) -> None:
    (project_root / "adir").mkdir()
    assert glob_files(project_root, "*") == []


def test_glob_files_cap_200(project_root: Path) -> None:
    for i in range(210):
        (project_root / f"f{i:03}.txt").write_text("")
    assert len(glob_files(project_root, "*.txt")) == 200


def test_grep_search_finds_lines(project_root: Path) -> None:
    (project_root / "code.py").write_text("def foo():\n    return 1\n# foo again\n")
    results = grep_search(project_root, r"foo")
    assert results == [
        "code.py:1: def foo():",
        "code.py:3: # foo again",
    ]


def test_grep_search_include_filter(project_root: Path) -> None:
    (project_root / "a.py").write_text("needle\n")
    (project_root / "b.txt").write_text("needle\n")
    results = grep_search(project_root, "needle", include=r"\.py$")
    assert results == ["a.py:1: needle"]


def test_grep_search_skips_ignored_dirs(project_root: Path) -> None:
    for d in IGNORED_DIRS:
        (project_root / d).mkdir()
        (project_root / d / "f.py").write_text("needle\n")
    (project_root / "ok.py").write_text("needle\n")
    assert grep_search(project_root, "needle") == ["ok.py:1: needle"]


def test_grep_search_cap_100(project_root: Path) -> None:
    (project_root / "big.txt").write_text("hit\n" * 150)
    assert len(grep_search(project_root, "hit")) == 100


def test_grep_search_bad_regex_raises(project_root: Path) -> None:
    with pytest.raises(re.error):
        grep_search(project_root, "[")


def test_grep_search_unreadable_file_skipped(project_root: Path) -> None:
    (project_root / "adir").mkdir()
    assert grep_search(project_root, "x") == []


# -- check_command -----------------------------------------------------------------


def test_check_command_allows_safe_commands(project_root: Path) -> None:
    check_command(project_root, "ls -la")
    check_command(project_root, "echo 'rm -rf / is just text in a string'")
    check_command(project_root, "pytest -k test_foo")
    check_command(project_root, "grep rm file.txt")


def test_check_command_allows_destructive_inside_root(project_root: Path) -> None:
    check_command(project_root, "rm -rf build")
    check_command(project_root, "mv a.txt b.txt")
    check_command(project_root, "cp src/dst.py dst.py")
    check_command(project_root, "rm /tmp/scratch/x")


def test_check_command_refuses_outside_paths(project_root: Path) -> None:
    with pytest.raises(PathViolation):
        check_command(project_root, "rm -rf /")
    with pytest.raises(PathViolation):
        check_command(project_root, "rm /etc/passwd")
    with pytest.raises(PathViolation):
        check_command(project_root, "mv /etc/hosts ./hosts")
    with pytest.raises(PathViolation):
        check_command(project_root, "cp ./f.txt /etc/evil")


def test_check_command_refuses_relative_escape(project_root: Path) -> None:
    with pytest.raises(PathViolation):
        check_command(project_root, "rm ../../outside.txt")


def test_check_command_full_paths_tolerated(project_root: Path) -> None:
    # /bin/rm removing something inside the project is fine.
    check_command(project_root, "/bin/rm build/out.o")


def test_check_command_wrappers(project_root: Path) -> None:
    with pytest.raises(PathViolation):
        check_command(project_root, "sudo rm -rf /")
    with pytest.raises(PathViolation):
        check_command(project_root, "nohup rm -rf /")
    with pytest.raises(PathViolation):
        check_command(project_root, "env -i nohup rm -rf /")
    with pytest.raises(PathViolation):
        check_command(project_root, "VAR=1 time nice rm -rf /")
    # Wrapper without destructive payload is fine.
    check_command(project_root, "sudo ls /etc")


def test_check_command_dashdash(project_root: Path) -> None:
    # `rm -- -file` inside the project: the name after -- is a path.
    check_command(project_root, "rm -- weird-file")
    with pytest.raises(PathViolation):
        check_command(project_root, "rm -- /etc/passwd")


def test_check_command_segments(project_root: Path) -> None:
    check_command(project_root, "echo hi && rm ./tmpfile")
    with pytest.raises(PathViolation):
        check_command(project_root, "echo hi && rm /etc/passwd")
    with pytest.raises(PathViolation):
        check_command(project_root, "echo hi; mv x /etc/evil")
    with pytest.raises(PathViolation):
        check_command(project_root, "echo hi || cp f /etc/evil")
    with pytest.raises(PathViolation):
        check_command(project_root, "echo hi\nrm /etc/passwd")


def test_check_command_flags_not_paths(project_root: Path) -> None:
    # Flags like -rf or --recursive must not be treated as paths.
    check_command(project_root, "rm -rf --recursive ./build")


def test_check_command_unparsable_quoting_conservative(project_root: Path) -> None:
    # Unbalanced quotes with rm inside → refuse rather than run blind.
    with pytest.raises(PathViolation):
        check_command(project_root, "rm 'unbalanced")
    # Unparsable but no destructive command → allowed.
    check_command(project_root, "echo 'unbalanced")


def test_check_command_tilde_expansion(project_root: Path) -> None:
    with pytest.raises(PathViolation):
        check_command(project_root, "rm ~/.bashrc")


# -- run_bash (async) ----------------------------------------------------------------


def test_run_bash_output_and_exit_code(project_root: Path) -> None:
    out = asyncio.run(run_bash(project_root, "echo hello; exit 3"))
    assert out.startswith("exit=3")
    assert "hello" in out


def test_run_bash_cwd_is_root(project_root: Path) -> None:
    (project_root / "marker.txt").write_text("here")
    out = asyncio.run(run_bash(project_root, "ls"))
    assert "marker.txt" in out


def test_run_bash_stderr_captured(project_root: Path) -> None:
    out = asyncio.run(run_bash(project_root, "echo oops >&2"))
    assert "oops" in out


def test_run_bash_no_output(project_root: Path) -> None:
    out = asyncio.run(run_bash(project_root, "true"))
    assert out == "exit=0\n(no output)"


def test_run_bash_refuses_destructive_outside(project_root: Path) -> None:
    with pytest.raises(PathViolation):
        asyncio.run(run_bash(project_root, "rm -rf /"))


def test_run_bash_timeout_message(project_root: Path, monkeypatch) -> None:
    monkeypatch.setattr(tools, "BASH_TIMEOUT", 1)
    out = asyncio.run(run_bash(project_root, "sleep 5"))
    assert "timed out" in out
    assert "exit=-1" in out


def test_run_bash_explicit_timeout(project_root: Path) -> None:
    # A small explicit timeout still cuts the command short.
    out = asyncio.run(run_bash(project_root, "sleep 5", timeout=1))
    assert "timed out after 1s" in out


def test_run_bash_timeout_clamped_to_cap(project_root: Path) -> None:
    # Values above BASH_MAX_TIMEOUT (60 min) are clamped down to the cap;
    # the command finishes long before, so only the message is checked.
    out = asyncio.run(run_bash(project_root, "echo hi", timeout=10**9))
    assert out == "exit=0\nhi"


def test_clamp_bash_timeout() -> None:
    assert tools.clamp_bash_timeout(None) == tools.BASH_TIMEOUT
    assert tools.clamp_bash_timeout(0) == tools.BASH_TIMEOUT
    assert tools.clamp_bash_timeout(-5) == tools.BASH_TIMEOUT
    assert tools.clamp_bash_timeout(30) == 30
    assert tools.clamp_bash_timeout(tools.BASH_MAX_TIMEOUT) == tools.BASH_MAX_TIMEOUT
    assert (
        tools.clamp_bash_timeout(tools.BASH_MAX_TIMEOUT + 1) == tools.BASH_MAX_TIMEOUT
    )


def test_run_bash_output_truncated(project_root: Path, monkeypatch) -> None:
    monkeypatch.setattr(tools, "MAX_TOOL_OUTPUT", 100)
    out = asyncio.run(run_bash(project_root, "echo " + "x" * 500))
    # "exit=0\n" prefix + exactly MAX_TOOL_OUTPUT characters of output.
    assert out == "exit=0\n" + "x" * 100


def test_ignored_dirs_contents() -> None:
    # Storage dir and venvs must never be grepped/globbed into noise.
    for d in (".git", ".venv", "node_modules", ".drcode", "__pycache__"):
        assert d in IGNORED_DIRS


def test_env_variable_not_picked_as_wrapper_word(project_root: Path) -> None:
    # A variable assignment alone (no command) must not crash the checker.
    check_command(project_root, "FOO=bar")
    check_command(project_root, "FOO=bar baz=qux ls")
