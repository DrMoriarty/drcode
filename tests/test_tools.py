"""Tests for drcode.tools: path permissions, file tools, grep/glob, bash."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
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


@pytest.fixture
def scratch_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the /tmp/drcode scratch base into the test's tmp dir, so the
    per-task scratch tests never touch the real /tmp."""
    root = tmp_path / "drcode-scratch"
    monkeypatch.setattr(tools, "SCRATCH_ROOT", root)
    return root


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


def test_is_writable_own_scratch(project_root: Path, scratch_root: Path) -> None:
    scratch = tools.scratch_dir("abc12345")
    assert is_writable(project_root, scratch, scratch)
    assert is_writable(project_root, scratch / "sub" / "x.txt", scratch)


def test_is_writable_tmp_outside_own_scratch_refused(
    project_root: Path, scratch_root: Path
) -> None:
    scratch = tools.scratch_dir("abc12345")
    # Foreign scratch directories and the rest of /tmp are NOT writable.
    assert not is_writable(project_root, tools.scratch_dir("other"), scratch)
    assert not is_writable(project_root, Path("/tmp/scratch/x.txt"), scratch)
    # Without a scratch area nothing under /tmp is writable at all.
    assert not is_writable(project_root, Path("/tmp/scratch/x.txt"))


def test_is_writable_outside(project_root: Path) -> None:
    assert not is_writable(project_root, Path("/etc/passwd"))
    assert not is_writable(project_root, project_root.parent / "sneaky.txt")
    assert not is_writable(project_root, Path("/tmpfoo/x.txt"))


def test_resolve_edit_path_outside_raises(
    project_root: Path, scratch_root: Path
) -> None:
    with pytest.raises(PathViolation):
        resolve_edit_path(
            project_root,
            "/etc/hosts",
            extra_writable=tools.scratch_dir("abc12345"),
        )


def test_resolve_edit_path_symlink_escape_raises(
    project_root: Path, scratch_root: Path
) -> None:
    outside = project_root.parent / "outside.txt"
    outside.write_text("secret")
    link = project_root / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(PathViolation):
        resolve_edit_path(
            project_root, "link.txt", extra_writable=tools.scratch_dir("abc12345")
        )


def test_resolve_edit_path_symlink_into_root_ok(
    project_root: Path, scratch_root: Path
) -> None:
    target = project_root / "real.txt"
    target.write_text("ok")
    link = project_root / "link.txt"
    link.symlink_to(target)
    assert (
        resolve_edit_path(
            project_root, "link.txt", extra_writable=tools.scratch_dir("abc12345")
        )
        == target.resolve()
    )


def test_resolve_edit_path_relative(project_root: Path, scratch_root: Path) -> None:
    assert (
        resolve_edit_path(
            project_root, "a/b.txt", extra_writable=tools.scratch_dir("abc12345")
        )
        == (project_root / "a/b.txt").resolve()
    )


def test_resolve_edit_path_macos_tmp_symlink(
    project_root: Path, scratch_root: Path
) -> None:
    # /tmp may resolve elsewhere (macOS: /tmp -> /private/tmp); both the
    # literal and the resolved scratch location must be accepted.
    scratch = tools.scratch_dir("abc12345")
    assert is_writable(project_root, scratch, scratch.resolve())
    assert is_writable(project_root, scratch.resolve(), scratch)


def test_resolve_edit_path_relative_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scratch_root: Path
) -> None:
    # An embedder may pass a relative project root (e.g. run_dir / "workspace");
    # the tool sandbox compares resolved paths against the root, so it must
    # accept both the literal and the resolved root.
    monkeypatch.chdir(tmp_path)
    root = Path("workspace")
    (tmp_path / "workspace" / "sub").mkdir(parents=True)
    p = resolve_edit_path(
        root, "sub/f.txt", extra_writable=tools.scratch_dir("abc12345")
    )
    assert p == (tmp_path / "workspace" / "sub" / "f.txt").resolve()


def test_resolve_edit_path_relative_root_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scratch_root: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PathViolation) as exc:
        resolve_edit_path(
            Path("workspace"),
            "/etc/evil",
            extra_writable=tools.scratch_dir("abc12345"),
        )
    # The hint interpolates the root as it was given (relative stays readable).
    assert "'workspace'" in str(exc.value)
    assert "/etc/evil" in str(exc.value)


def test_resolve_edit_path_symlinked_root(tmp_path: Path, scratch_root: Path) -> None:
    # A symlinked absolute root (e.g. /tmp on macOS) must be accepted for
    # files under its resolved location — mirror of the scratch-dir case.
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    scratch = tools.scratch_dir("abc12345")
    p = resolve_edit_path(link, "f.txt", extra_writable=scratch)
    assert p == (real / "f.txt").resolve()
    assert is_writable(link, p, scratch)
    with pytest.raises(PathViolation):
        resolve_edit_path(
            link, str(tmp_path / "elsewhere" / "f.txt"), extra_writable=scratch
        )


def test_check_command_relative_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scratch_root: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("workspace")
    (tmp_path / "workspace").mkdir()
    # In-project destructive commands pass; escapes still refused.
    check_command(root, "rm -rf build")
    check_command(root, "mv a.txt b.txt")
    with pytest.raises(PathViolation):
        check_command(root, "rm /etc/passwd")


def test_resolve_edit_path_foreign_scratch_raises(
    project_root: Path, scratch_root: Path
) -> None:
    own = tools.scratch_dir("mine")
    foreign = tools.scratch_dir("theirs")
    with pytest.raises(PathViolation):
        resolve_edit_path(project_root, str(foreign / "f.txt"), extra_writable=own)


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


def test_write_file_creates_parents(project_root: Path, scratch_root: Path) -> None:
    msg = write_file(
        project_root,
        "deep/nested/f.txt",
        "hello",
        extra_writable=tools.scratch_dir("t"),
    )
    assert (project_root / "deep/nested/f.txt").read_text() == "hello"
    assert "5 chars" in msg  # char count mentioned in the reply


def test_write_file_overwrites(project_root: Path, scratch_root: Path) -> None:
    (project_root / "f.txt").write_text("old")
    write_file(project_root, "f.txt", "new", extra_writable=tools.scratch_dir("t"))
    assert (project_root / "f.txt").read_text() == "new"


def test_write_file_to_own_scratch_allowed(
    project_root: Path, scratch_root: Path
) -> None:
    # The task's own scratch directory is writable and is created lazily
    # on the first write.
    scratch = tools.scratch_dir("abc12345")
    assert not scratch.exists()
    write_file(project_root, str(scratch / "f.txt"), "data", extra_writable=scratch)
    assert (scratch / "f.txt").read_text() == "data"


def test_write_file_to_foreign_scratch_refused(
    project_root: Path, scratch_root: Path
) -> None:
    # The soft isolation boundary: another task's scratch directory must not
    # be writable — two tasks cannot interfere through shared scratch files.
    own = tools.scratch_dir("mine")
    foreign = tools.scratch_dir("theirs")
    with pytest.raises(PathViolation):
        write_file(project_root, str(foreign / "f.txt"), "nope", extra_writable=own)
    assert not foreign.exists()
    # And plain /tmp is off-limits too.
    with pytest.raises(PathViolation):
        write_file(project_root, "/tmp/loose.txt", "nope", extra_writable=own)


def test_write_file_outside_refused(project_root: Path, scratch_root: Path) -> None:
    with pytest.raises(PathViolation):
        write_file(
            project_root, "../evil.txt", "nope", extra_writable=tools.scratch_dir("t")
        )


def test_write_file_home_refused(project_root: Path, scratch_root: Path) -> None:
    with pytest.raises(PathViolation):
        write_file(
            project_root, "~/evil.txt", "nope", extra_writable=tools.scratch_dir("t")
        )


# -- edit_file ------------------------------------------------------------------


def test_edit_file_replaces_unique(project_root: Path, scratch_root: Path) -> None:
    (project_root / "f.txt").write_text("hello world")
    msg = edit_file(
        project_root, "f.txt", "world", "there", extra_writable=tools.scratch_dir("t")
    )
    assert (project_root / "f.txt").read_text() == "hello there"
    assert "edited" in msg


def test_edit_file_not_found(project_root: Path, scratch_root: Path) -> None:
    (project_root / "f.txt").write_text("hello")
    with pytest.raises(ValueError, match="not found"):
        edit_file(
            project_root, "f.txt", "missing", "x", extra_writable=tools.scratch_dir("t")
        )


def test_edit_file_multiple_matches(project_root: Path, scratch_root: Path) -> None:
    (project_root / "f.txt").write_text("a a")
    with pytest.raises(ValueError, match="2 locations"):
        edit_file(
            project_root, "f.txt", "a", "b", extra_writable=tools.scratch_dir("t")
        )


def test_edit_file_outside_refused(project_root: Path, scratch_root: Path) -> None:
    with pytest.raises(PathViolation):
        edit_file(
            project_root, "/etc/hosts", "x", "y", extra_writable=tools.scratch_dir("t")
        )


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
    (project_root / "sub").mkdir()
    (project_root / "sub" / "c.py").write_text("needle\n")
    # `include` is a glob mask over the project-relative path.
    assert grep_search(project_root, "needle", include="*.py") == [
        "a.py:1: needle",
        "sub/c.py:1: needle",
    ]
    assert grep_search(project_root, "needle", include="src/**/*.py") == []


def test_grep_search_include_nested_glob(project_root: Path) -> None:
    (project_root / "a.txt").write_text("needle\n")
    (project_root / "src").mkdir()
    (project_root / "src" / "b.py").write_text("needle\n")
    (project_root / "src" / "deep").mkdir()
    (project_root / "src" / "deep" / "c.py").write_text("needle\n")
    results = grep_search(project_root, "needle", include="src/**/*.py")
    assert results == [
        "src/b.py:1: needle",
        "src/deep/c.py:1: needle",
    ]


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


def test_check_command_allows_destructive_inside_root(
    project_root: Path, scratch_root: Path
) -> None:
    own = tools.scratch_dir("abc12345")
    check_command(project_root, "rm -rf build", own)
    check_command(project_root, "mv a.txt b.txt", own)
    check_command(project_root, "cp src/dst.py dst.py", own)
    check_command(project_root, f"rm {own}/scratch/x", own)


def test_check_command_scratch_boundaries(
    project_root: Path, scratch_root: Path
) -> None:
    own = tools.scratch_dir("mine")
    foreign = tools.scratch_dir("theirs")
    # rm/mv/cp inside the task's own scratch directory are fine...
    check_command(project_root, f"rm -rf {own}/build", own)
    # ...touching another task's scratch (or the rest of /tmp) is not.
    with pytest.raises(PathViolation):
        check_command(project_root, f"rm -rf {foreign}/build", own)
    with pytest.raises(PathViolation):
        check_command(project_root, "rm -rf /tmp/loose", own)
    # Without a scratch area nothing under /tmp passes the check.
    with pytest.raises(PathViolation):
        check_command(project_root, "rm -rf /tmp/loose")


def test_check_command_protects_state_dir(project_root: Path) -> None:
    # `.drcode` sits inside the writable area but holds all task history
    # (gitignored, hence unrecoverable) — rm/mv/cp must refuse it and
    # anything that would take it down with it.
    with pytest.raises(PathViolation):
        check_command(project_root, "rm -rf .drcode")
    with pytest.raises(PathViolation):
        check_command(project_root, "rm -rf .drcode/tasks")
    with pytest.raises(PathViolation):
        check_command(project_root, f"rm -rf {project_root}/.drcode")
    with pytest.raises(PathViolation):
        check_command(project_root, "mv .drcode /tmp/loose")
    with pytest.raises(PathViolation):
        check_command(project_root, "cp f.txt .drcode/tasks/x.json")
    # Recursive parents that would swallow the state dir are refused too.
    with pytest.raises(PathViolation):
        check_command(project_root, "rm -rf .")
    with pytest.raises(PathViolation):
        check_command(project_root, f"rm -rf {project_root}")
    # Siblings and unrelated paths inside the root stay allowed.
    check_command(project_root, "rm -rf build")
    check_command(project_root, "ls .drcode")


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


def test_run_bash_exports_task_env(project_root: Path) -> None:
    """Parallel development slots: the child process can tell its own task
    (and its worktree) apart via the exported DRCODE_* variables."""
    out = asyncio.run(
        run_bash(
            project_root,
            'echo "$DRCODE_TASK_ID|$DRCODE_WORKTREE"',
            task_id="deadbeef",
        )
    )
    assert f"deadbeef|{project_root}" in out
    # Without a task id the variables stay unset (no empty-string noise).
    out = asyncio.run(run_bash(project_root, 'echo "[$DRCODE_TASK_ID]"'))
    assert "[]" in out


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


def test_run_bash_sets_task_tmpdir(project_root: Path, scratch_root: Path) -> None:
    # Child processes get TMPDIR pointed at the task's scratch directory,
    # which is created lazily on the first run_bash call.
    scratch = tools.scratch_dir("abc12345")
    out = asyncio.run(run_bash(project_root, "echo $TMPDIR", extra_writable=scratch))
    assert out == f"exit=0\n{scratch}"
    assert scratch.is_dir()


def test_run_bash_without_scratch_keeps_env(project_root: Path) -> None:
    # Internal calls without a scratch area leave TMPDIR untouched.
    out = asyncio.run(run_bash(project_root, "echo ${TMPDIR:-unset}"))
    assert out == f"exit=0\n{os.environ.get('TMPDIR', 'unset')}"


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


# -- scratch lifecycle ---------------------------------------------------------------


def test_cleanup_orphan_scratches_removes_unknown_ids(scratch_root: Path) -> None:
    (scratch_root / "live").mkdir(parents=True)
    (scratch_root / "dead").mkdir(parents=True)
    (scratch_root / "not-a-dir").write_text("x")
    removed = tools.cleanup_orphan_scratches({"live"})
    assert removed == ["dead"]
    assert (scratch_root / "live").is_dir()
    assert not (scratch_root / "dead").exists()


def test_cleanup_orphan_scratches_no_base(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tools, "SCRATCH_ROOT", tmp_path / "absent")
    assert tools.cleanup_orphan_scratches(set()) == []


def test_remove_scratch_is_best_effort(scratch_root: Path) -> None:
    tools.remove_scratch("never-created")  # no error when absent


# -- scratch owner marker (multi-instance safety) ------------------------------------


def _dead_pid() -> int:
    """A pid that is guaranteed not to exist anymore."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _live_foreign_process() -> subprocess.Popen[bytes]:
    """A live process simulating another running drcode instance."""
    return subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE)


def test_run_bash_stamps_owner_marker(project_root: Path, scratch_root: Path) -> None:
    scratch = tools.scratch_dir("abc12345")
    asyncio.run(run_bash(project_root, "true", extra_writable=scratch))
    assert (scratch / tools.SCRATCH_OWNER_FILE).read_text() == str(os.getpid())


def test_write_file_stamps_owner_marker(project_root: Path, scratch_root: Path) -> None:
    scratch = tools.scratch_dir("abc12345")
    write_file(
        project_root, str(scratch / "sub" / "x.txt"), "hi", extra_writable=scratch
    )
    assert (scratch / tools.SCRATCH_OWNER_FILE).read_text() == str(os.getpid())
    # A write into the project root does not restamp the marker.
    (scratch / tools.SCRATCH_OWNER_FILE).write_text("42")
    write_file(project_root, "root.txt", "hi", extra_writable=scratch)
    assert (scratch / tools.SCRATCH_OWNER_FILE).read_text() == "42"


def test_cleanup_keeps_live_foreign_scratch(scratch_root: Path) -> None:
    # Active task of ANOTHER drcode instance (marker names a live foreign
    # pid): the orphan sweep must keep it even though no local task matches.
    foreign = scratch_root / "foreign"
    foreign.mkdir(parents=True)
    other = _live_foreign_process()
    try:
        (foreign / tools.SCRATCH_OWNER_FILE).write_text(str(other.pid))
        assert tools.cleanup_orphan_scratches(set()) == []
        assert foreign.is_dir()
    finally:
        other.kill()
        other.wait()
    # Now the owner is dead: the same sweep removes the leftover.
    assert tools.cleanup_orphan_scratches(set()) == ["foreign"]
    assert not foreign.exists()


def test_cleanup_removes_dead_owner_scratch(scratch_root: Path) -> None:
    # Marker of a dead process: crashed session's leftover → removed.
    dead = scratch_root / "dead-owner"
    dead.mkdir(parents=True)
    (dead / tools.SCRATCH_OWNER_FILE).write_text(str(_dead_pid()))
    assert tools.cleanup_orphan_scratches(set()) == ["dead-owner"]
    assert not dead.exists()


def test_remove_scratch_skips_live_foreign_owner(scratch_root: Path) -> None:
    # Id collision across projects: this instance's task completes, but the
    # directory is currently owned by another live process — keep it.
    scratch = scratch_root / "abc12345"
    other = _live_foreign_process()
    try:
        scratch.mkdir(parents=True)
        (scratch / tools.SCRATCH_OWNER_FILE).write_text(str(other.pid))
        tools.remove_scratch("abc12345")
        assert scratch.is_dir()
    finally:
        other.kill()
        other.wait()
    tools.remove_scratch("abc12345")
    assert not scratch.exists()
