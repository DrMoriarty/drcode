"""Tests for drcode.tools: path permissions, file tools, grep/glob, bash."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import time
import types
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


def test_read_file_oversized_chunk_truncated_not_refused(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read above the per-result cap is truncated (head+tail), never refused:
    the model keeps a usable slice plus the offset/limit guidance."""
    monkeypatch.setattr(tools, "MAX_READ_RESULT_BYTES", 500)
    n = 200
    (project_root / "big.txt").write_text(
        "\n".join(f"line{i:04d} " + "x" * 20 for i in range(1, n + 1))
    )
    out = read_file(project_root, "big.txt")
    assert "output truncated" in out
    assert "adjusting `offset` and `limit`" in out
    # Head keeps the file's start, tail its end — original line numbers intact.
    assert "1: line0001" in out
    assert f"{n}: line{n:04d}" in out
    # The middle is gone.
    assert "line0100" not in out
    assert "Refusing to read" not in out


def test_read_file_truncation_respects_cap(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No single read result — slices AND the note — exceeds the cap."""
    monkeypatch.setattr(tools, "MAX_READ_RESULT_BYTES", 500)
    (project_root / "big.txt").write_text(
        "\n".join(f"line{i:04d} " + "y" * 30 for i in range(1, 301))
    )
    out = read_file(project_root, "big.txt")
    assert len(out.encode("utf-8")) <= tools.MAX_READ_RESULT_BYTES


def test_read_file_truncation_honors_offset(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Truncation of a mid-file chunk keeps the chunk's own numbering: the head
    starts at the requested offset, not at line 1."""
    monkeypatch.setattr(tools, "MAX_READ_RESULT_BYTES", 400)
    (project_root / "big.txt").write_text(
        "\n".join(f"line{i:04d} " + "z" * 20 for i in range(1, 401))
    )
    out = read_file(project_root, "big.txt", offset=100)
    assert out.startswith("100: line0100")
    assert "output truncated" in out


def test_read_file_truncation_single_giant_line_fallback(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single line longer than the whole budget defeats line-based slicing —
    fall back to a raw character head+tail so the slice is still usable."""
    monkeypatch.setattr(tools, "MAX_READ_RESULT_BYTES", 500)
    (project_root / "one.txt").write_text("A" * 2000 + "\n" + "B" * 2000)
    out = read_file(project_root, "one.txt")
    assert "output truncated" in out
    assert out.startswith("1: AAAA")
    assert out.rstrip().endswith("BBBB")
    assert len(out.encode("utf-8")) <= tools.MAX_READ_RESULT_BYTES


def test_read_file_budget_refusal_still_gates_after_truncation(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Truncation shrinks the piece, but the context-budget gate still refuses
    when even the truncated result would push the fill past the threshold."""
    monkeypatch.setattr(tools, "MAX_READ_RESULT_BYTES", 500)
    (project_root / "big.txt").write_text(
        "\n".join(f"line{i:04d} " + "w" * 30 for i in range(1, 301))
    )
    out = read_file(
        project_root,
        "big.txt",
        current_usage=790,
        max_context_tokens=1000,
        compaction_threshold=80,
    )
    assert "Refusing to read" in out
    assert "context budget" in out


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


def test_grep_search_per_file_cap_marked(project_root: Path) -> None:
    # One noisy file must not eat the whole result budget: it is cut at
    # GREP_PER_FILE_LIMIT lines and the cut is marked explicitly.
    (project_root / "big.txt").write_text("hit\n" * 150)
    results = grep_search(project_root, "hit")
    assert len(results) == 20 + 1
    assert results[0] == "big.txt:1: hit"
    assert "big.txt: ... (truncated: more than 20 matches in this file)" in results


def test_grep_search_truncation_summary(project_root: Path) -> None:
    # Global cap: 10 files × 10 matches = 100 total → 80 shown + summary;
    # counting is not interrupted early, so N and M are exact.
    for n in range(10):
        (project_root / f"f{n}.txt").write_text("hit\n" * 10)
    results = grep_search(project_root, "hit")
    assert len(results) == 80 + 1
    assert results[-1] == (
        "showing 80 of ≥100 matches in 10 files — "
        "narrow the include or refine the pattern"
    )


def test_grep_search_small_result_no_summary(project_root: Path) -> None:
    (project_root / "a.py").write_text("hit\n")
    assert grep_search(project_root, "hit") == ["a.py:1: hit"]


def test_grep_search_skips_binary_files(project_root: Path) -> None:
    (project_root / "bin.dat").write_bytes(b"needle\x00needle\n")
    (project_root / "ok.py").write_text("needle\n")
    assert grep_search(project_root, "needle") == ["ok.py:1: needle"]


def test_grep_search_skips_oversized_files(project_root: Path) -> None:
    (project_root / "big.log").write_text("needle\n" + "x" * 1_000_001)
    (project_root / "ok.py").write_text("needle\n")
    assert grep_search(project_root, "needle") == ["ok.py:1: needle"]


def test_grep_search_respects_gitignore(project_root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(project_root)], check=True)
    (project_root / ".gitignore").write_text("vendor/\n*.log\n")
    (project_root / "vendor").mkdir()
    (project_root / "vendor" / "lib.py").write_text("needle\n")
    (project_root / "noise.log").write_text("needle\n")
    (project_root / "keep.py").write_text("needle\n")
    assert grep_search(project_root, "needle") == ["keep.py:1: needle"]


def test_grep_search_nested_gitignore_honored(project_root: Path) -> None:
    # The project root is NOT a working tree, but a first-level subdir is one
    # on its own: its .gitignore must apply (git ls-files from that subdir).
    nested = project_root / "torch-vulkan"
    nested.mkdir()
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    (nested / ".gitignore").write_text("target/\n")
    (nested / "target").mkdir()
    (nested / "target" / "build_artifact.o").write_text("needle\n")
    (nested / "src").mkdir()
    (nested / "src" / "lib.py").write_text("needle\n")
    assert grep_search(project_root, "needle") == ["torch-vulkan/src/lib.py:1: needle"]


def test_grep_search_nested_repos_cover_their_subtree(project_root: Path) -> None:
    # A nested working tree replaces the plain walk under it entirely: even
    # files git would track-but-ignore via nested .gitignores stay consistent,
    # and the walk never descends into it for a second listing.
    nested = project_root / "repo"
    nested.mkdir()
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    (nested / "deep").mkdir()
    (nested / "deep" / "f.py").write_text("needle\n")
    (project_root / "plain").mkdir()
    (project_root / "plain" / "g.py").write_text("needle\n")
    (project_root / "root.py").write_text("needle\n")
    assert grep_search(project_root, "needle") == [
        "root.py:1: needle",
        "repo/deep/f.py:1: needle",
        "plain/g.py:1: needle",
    ]


def test_grep_search_non_repo_subdir_keeps_walk_filters(project_root: Path) -> None:
    # Subdirs that are not git repos keep the old behavior: plain walk with
    # IGNORED_DIRS pruning, late size/binary filters decide.
    (project_root / "build").mkdir()
    (project_root / "build" / "huge.bin").write_text("needle\n" + "x" * 1_000_001)
    (project_root / "build" / "ok.py").write_text("needle\n")
    (project_root / "build" / "node_modules").mkdir()
    (project_root / "build" / "node_modules" / "dep.py").write_text("needle\n")
    assert grep_search(project_root, "needle") == ["build/ok.py:1: needle"]


def test_grep_search_nested_repo_at_depths_2_and_3(project_root: Path) -> None:
    # The nested-repo probe reaches three levels below the root: a working
    # tree at root/a/repo (depth 2) and one at root/x/y/deep (depth 3) each
    # replace the plain walk under them — their .gitignore applies and the
    # walk never reads their ignored build artifacts.
    for rel, ignore, tracked in (
        ("a/repo", "build/\n", "src/lib.py"),
        ("x/y/deep", "dist/\n", "core.c"),
    ):
        nested = project_root / rel
        nested.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(nested)], check=True)
        (nested / ".gitignore").write_text(ignore)
        (nested / tracked).parent.mkdir(parents=True, exist_ok=True)
        (nested / tracked).write_text("needle\n")
        artifact = nested / ignore.strip("/\n") / "artifact"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("needle\n")
    (project_root / "keep.py").write_text("needle\n")
    (project_root / "plain").mkdir()
    (project_root / "plain" / "n.txt").write_text("needle\n")
    assert sorted(grep_search(project_root, "needle")) == [
        "a/repo/src/lib.py:1: needle",
        "keep.py:1: needle",
        "plain/n.txt:1: needle",
        "x/y/deep/core.c:1: needle",
    ]


def test_grep_search_probe_depth_capped_at_3(project_root: Path) -> None:
    # Beyond depth 3 no probe runs: a working tree at root/a/b/c/d keeps the
    # plain walk — including files its own .gitignore would exclude.
    deep = project_root / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(deep)], check=True)
    (deep / ".gitignore").write_text("gen/\n")
    (deep / "real.py").write_text("needle\n")
    (deep / "gen" / "x.py").parent.mkdir(parents=True)
    (deep / "gen" / "x.py").write_text("needle\n")
    (project_root / "keep.py").write_text("needle\n")
    assert sorted(grep_search(project_root, "needle")) == [
        "a/b/c/d/gen/x.py:1: needle",
        "a/b/c/d/real.py:1: needle",
        "keep.py:1: needle",
    ]


def test_grep_search_shallow_repo_wins_over_deeper(project_root: Path) -> None:
    # A working tree found at a shallower level replaces the walk under it
    # entirely — no deeper probe ever runs below it: outer's .gitignore hides
    # inner/ from outer's ls-files, and if inner were probed its i.py would
    # leak into the results.
    outer = project_root / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    (outer / ".gitignore").write_text("inner/\n")
    (outer / "o.py").write_text("needle\n")
    inner = outer / "inner"
    inner.mkdir()
    subprocess.run(["git", "init", "-q", str(inner)], check=True)
    (inner / "i.py").write_text("needle\n")
    assert grep_search(project_root, "needle") == ["outer/o.py:1: needle"]


def test_grep_search_prefix_parity_through_nested_repos(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Documented invariant of _grep_candidates: with a literal directory
    # prefix the candidate set equals the unprefixed one restricted to the
    # prefix — including when nested working trees sit on the prefix's
    # ancestor chain or below the prefix directory within the probe depth.
    repo = project_root / "a" / "b"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("junk/\n")
    for rel in ("lib.py", "c/deep.py", "c/sub/more.py", "junk/x.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("needle\n")
    (project_root / "a" / "plain.py").parent.mkdir(parents=True, exist_ok=True)
    (project_root / "a" / "plain.py").write_text("needle\n")
    (project_root / "a" / "other").mkdir()
    (project_root / "a" / "other" / "o.py").write_text("needle\n")
    (project_root / "z.py").write_text("needle\n")

    def rels(prefix: str = "") -> list[str]:
        return sorted(rel for rel, _ in tools._grep_candidates(project_root, prefix))

    full = rels()
    assert "a/b/lib.py" in full and "a/b/junk/x.py" not in full
    for prefix in ("a/", "a/b/", "a/b/c/", "a/b/c/sub/", "a/other/", "a/b/junk/"):
        assert rels(prefix) == [r for r in full if r.startswith(prefix)], prefix
    # End to end: the include-pruned run agrees with the unpruned one.
    unpruned = grep_search(project_root, "needle")
    for include in ("a/**", "a/b/**", "a/b/c/**", "a/other/**"):
        assert grep_search(project_root, "needle", include=include) == [
            line for line in unpruned if line.startswith(include[:-3])
        ], include


def test_grep_search_exact_include_parity_with_depth_3_repo(
    project_root: Path,
) -> None:
    # An exact include must agree with the unpruned run on which files
    # exist: a gitignored file inside a depth-3 nested working tree is
    # governed by the repo's ls-files (probe depth 3) — the exact include
    # probes the same ancestor chain and therefore finds nothing too.
    repo = project_root / "a" / "b" / "c"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("d.txt\n")
    (repo / "ok.txt").write_text("needle\n")
    (repo / "d.txt").write_text("needle\n")
    assert grep_search(project_root, "needle") == ["a/b/c/ok.txt:1: needle"]
    assert grep_search(project_root, "needle", include="a/b/c/d.txt") == []
    assert grep_search(project_root, "needle", include="a/b/c/*.txt") == [
        "a/b/c/ok.txt:1: needle"
    ]


def test_grep_search_prefix_chain_deeper_than_probe_not_honored(
    project_root: Path,
) -> None:
    # The unprefixed walk never probes beyond root/a/b/c, so a prefix whose
    # ancestor chain is deeper than the probe depth must not probe a depth-4
    # working tree either — its files stay plain-walk candidates (gitignore
    # NOT applied), keeping the prefix/unprefixed result sets identical.
    deep = project_root / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(deep)], check=True)
    (deep / ".gitignore").write_text("gen/\n")
    (deep / "real.py").write_text("needle\n")
    (deep / "gen" / "x.py").parent.mkdir(parents=True)
    (deep / "gen" / "x.py").write_text("needle\n")

    full = sorted(rel for rel, _ in tools._grep_candidates(project_root))
    # The depth-4 repo is not probed: plain walk, so its .gitignore file is
    # an ordinary candidate and "gen/" is NOT excluded.
    assert full == ["a/b/c/d/.gitignore", "a/b/c/d/gen/x.py", "a/b/c/d/real.py"]
    prefixed = sorted(
        rel for rel, _ in tools._grep_candidates(project_root, "a/b/c/d/")
    )
    assert prefixed == [r for r in full if r.startswith("a/b/c/d/")]
    assert grep_search(project_root, "needle", include="a/b/c/d/**") == [
        "a/b/c/d/real.py:1: needle",
        "a/b/c/d/gen/x.py:1: needle",
    ]


def test_grep_search_bad_regex_raises(project_root: Path) -> None:
    with pytest.raises(re.error):
        grep_search(project_root, "[")


def test_grep_search_unreadable_file_skipped(project_root: Path) -> None:
    (project_root / "adir").mkdir()
    assert grep_search(project_root, "x") == []


def _slow_read(monkeypatch: pytest.MonkeyPatch, delay: float) -> None:
    """Make every candidate read take `delay` seconds (the stand-in for a
    huge slow scan — the deadline check is per candidate, so a slowed
    _read_greppable is exactly what moves the clock)."""
    real_read = tools._read_greppable

    def slow(f: Path) -> str | None:
        time.sleep(delay)
        return real_read(f)

    monkeypatch.setattr(tools, "_read_greppable", slow)


def test_grep_search_default_timeout_partial_results(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The default time budget (GREP_DEFAULT_TIMEOUT_S) interrupts a slow wide
    # scan: the agent gets the partial hits in the usual format plus ONE
    # explicit interruption note — elapsed time, the candidate path the scan
    # stopped at, and the advice to narrow the include mask.
    for n in range(40):
        (project_root / f"f{n:02}.txt").write_text("needle\n")
    # Budget 0.5s vs 20ms per read: the lazy enumeration (one `git ls-files`
    # spawn) fits comfortably inside the budget, the 40 reads do not.
    _slow_read(monkeypatch, delay=0.02)
    monkeypatch.setattr(tools, "GREP_DEFAULT_TIMEOUT_S", 0.5)
    results = grep_search(project_root, "needle")
    hits = [r for r in results if r.endswith(": needle")]
    notes = [r for r in results if not r.endswith(": needle")]
    assert hits, "partial results must survive the interruption"
    assert len(hits) < 40, "the scan must have been cut short"
    assert len(notes) == 1
    note = notes[0]
    assert "interrupted after" in note
    assert "time budget expired" in note
    # Names the candidate where the scan stopped (not yet scanned).
    m = re.search(r"candidate '(f\d+\.txt)'", note)
    assert m is not None, note
    assert int(m.group(1)[1:3]) >= len(hits) - 1
    # Points at the escape hatches: a narrower mask or an explicit timeout.
    assert "include" in note and "timeout" in note


def test_grep_search_explicit_timeout_completes_full_scan(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An explicitly larger timeout lets the same slow scan run to completion:
    # every file's hit comes back, no interruption note.
    for n in range(40):
        (project_root / f"f{n:02}.txt").write_text("needle\n")
    _slow_read(monkeypatch, delay=0.005)
    results = grep_search(project_root, "needle", timeout=60.0)
    assert len(results) == 40
    assert all(": needle" in r for r in results)


def test_grep_search_zero_timeout_returns_only_note(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An already-expired budget interrupts before the first candidate: no
    # hits, only the interruption note (still actionable).
    (project_root / "a.py").write_text("needle\n")
    results = grep_search(project_root, "needle", timeout=0.0)
    assert len(results) == 1
    assert "interrupted after" in results[0]
    assert "candidate 'a.py'" in results[0]


# -- grep_search include pruning (candidate enumeration shortcut) ---------------


def test_grep_search_exact_include_never_enumerates(project_root: Path) -> None:
    # A magic-free include ("src/api/ops/binary.cpp") is one literal path: no
    # directory walk and no full `git ls-files` may run — the candidate comes
    # from a single stat (plus, inside a working tree, a pathspec-limited
    # ls-files). The real case this fixes: a huge non-repo directory made an
    # exact-path grep take minutes to read one file.
    huge = project_root / "experiments"
    huge.mkdir()
    for i in range(300):
        d = huge / f"run{i:03}"
        d.mkdir()
        (d / "data.bin").write_text("needle\n")
    target = project_root / "torchvulkan" / "src" / "api" / "ops"
    target.mkdir(parents=True)
    (target / "binary.cpp").write_text("needle\n")

    full_listing: list[Path] | None = [Path("experiments/run000/data.bin")]
    calls: list[str | None] = []

    def spy(root: Path, pathspec: str | None = None) -> list[Path] | None:
        calls.append(pathspec)
        return full_listing  # pretend root IS a repo with that one file

    monkey = pytest.MonkeyPatch()
    monkey.setattr(tools, "_git_ls_files", spy)
    try:
        assert grep_search(
            project_root, "needle", include="torchvulkan/src/api/ops/binary.cpp"
        ) == ["torchvulkan/src/api/ops/binary.cpp:1: needle"]
    finally:
        monkey.undo()
    # Only pathspec-limited calls — never the full enumeration.
    assert calls == ["torchvulkan/src/api/ops/binary.cpp"]


def test_grep_search_exact_include_missing_path_empty(project_root: Path) -> None:
    (project_root / "a.py").write_text("needle\n")
    assert grep_search(project_root, "needle", include="no/such/file.py") == []
    # A directory under the include mask is not a greppable candidate either.
    (project_root / "sub").mkdir()
    assert grep_search(project_root, "needle", include="sub") == []


def test_grep_search_exact_include_gitignored_invisible(project_root: Path) -> None:
    # Gitignore fidelity of the fast path: an exact-path include must not
    # surface a file the enumeration would skip.
    # Root is NOT a working tree, but the path's first-level directory is one
    # on its own: its ls-files governs (tracked file found, ignored not).
    nested = project_root / "vendor"
    nested.mkdir()
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    (nested / ".gitignore").write_text("*.o\n")
    (nested / "lib.py").write_text("needle\n")
    (nested / "build.o").write_text("needle\n")
    (project_root / "loose.py").write_text("needle\n")
    assert grep_search(project_root, "needle", include="vendor/lib.py") == [
        "vendor/lib.py:1: needle"
    ]
    assert grep_search(project_root, "needle", include="vendor/build.o") == []
    assert grep_search(project_root, "needle", include="loose.py") == [
        "loose.py:1: needle"
    ]
    # Now the root becomes a working tree: it governs every path, and git does
    # not descend into a nested repo — exactly like the ls-files enumeration.
    subprocess.run(["git", "init", "-q", str(project_root)], check=True)
    (project_root / ".gitignore").write_text("noise.log\n")
    (project_root / "noise.log").write_text("needle\n")
    assert grep_search(project_root, "needle", include="noise.log") == []
    assert grep_search(project_root, "needle", include="loose.py") == [
        "loose.py:1: needle"
    ]
    assert grep_search(project_root, "needle", include="vendor/lib.py") == []


def test_grep_search_include_prefix_and_magic_prunes(project_root: Path) -> None:
    # "src/**/*.py": the walk starts at src/ — everything outside the literal
    # prefix is never visited; results are unchanged either way.
    huge = project_root / "experiments"
    huge.mkdir()
    for i in range(200):
        (huge / f"f{i:03}.py").write_text("needle\n")
    (project_root / "src" / "deep").mkdir(parents=True)
    (project_root / "src" / "b.py").write_text("needle\n")
    (project_root / "src" / "deep" / "c.py").write_text("needle\n")
    (project_root / "top.py").write_text("needle\n")
    assert grep_search(project_root, "needle", include="src/**/*.py") == [
        "src/b.py:1: needle",
        "src/deep/c.py:1: needle",
    ]


def test_grep_search_include_prefix_file_siblings_safe(project_root: Path) -> None:
    # Magic right after a file name ("src/lib.p*"): the literal run is NOT cut
    # at the file name — the walk prunes only to the containing directory, so
    # siblings the trailing magic can still match ("lib.py2") are not lost.
    (project_root / "src").mkdir()
    (project_root / "src" / "lib.py").write_text("needle\n")
    (project_root / "src" / "lib.py2").write_text("needle\n")
    (project_root / "src" / "other.py").write_text("needle\n")
    assert grep_search(project_root, "needle", include="src/lib.p*") == [
        "src/lib.py:1: needle",
        "src/lib.py2:1: needle",
    ]


def test_grep_search_include_leading_globstar_unchanged(project_root: Path) -> None:
    # A mask starting with "**/" has an empty literal prefix — nothing to
    # prune, the full enumeration runs as before.
    (project_root / "a" / "x.py").mkdir(parents=True)
    (project_root / "a" / "x.py" / "f.py").write_text("needle\n")
    (project_root / "b.py").write_text("needle\n")
    assert grep_search(project_root, "needle", include="**/f.py") == [
        "a/x.py/f.py:1: needle"
    ]
    assert grep_search(project_root, "needle", include="**/*.py") == [
        "b.py:1: needle",
        "a/x.py/f.py:1: needle",
    ]


def test_grep_search_include_into_ignored_dir_empty(project_root: Path) -> None:
    # The enumeration prunes IGNORED_DIRS on every level from the root — the
    # pruned paths (exact and prefix) must not surface files the enumeration
    # would never enter (regression: the fast paths skipped that check).
    def build(root: Path) -> None:
        (root / "node_modules" / "inner").mkdir(parents=True)
        (root / "node_modules" / "direct.js").write_text("needle\n")
        (root / "node_modules" / "inner" / "dep.js").write_text("needle\n")
        (root / ".venv" / "lib").mkdir(parents=True)
        (root / ".venv" / "lib" / "venvlib.py").write_text("needle\n")
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "a.cache").write_text("needle\n")
        (root / "keep.py").write_text("needle\n")

    # Non-repo root.
    build(project_root)
    for mask in [
        "node_modules/direct.js",
        "node_modules/inner/dep.js",
        "node_modules/**/*",
        ".venv/lib/venvlib.py",
        "__pycache__/a.cache",
    ]:
        assert grep_search(project_root, "needle", include=mask) == [], mask
    # Repo root: git does not know IGNORED_DIRS — the Python-side filter must
    # apply on the exact-path fast path too.
    subprocess.run(["git", "init", "-q", str(project_root)], check=True)
    subprocess.run(["git", "-C", str(project_root), "add", "keep.py"], check=True)
    for mask in [
        "node_modules/direct.js",
        ".venv/lib/venvlib.py",
        "__pycache__/a.cache",
    ]:
        assert grep_search(project_root, "needle", include=mask) == [], mask
    assert grep_search(project_root, "needle", include="keep.py") == [
        "keep.py:1: needle"
    ]


def test_grep_search_include_prefix_through_ignored_dir_empty(
    project_root: Path,
) -> None:
    # A prefix that runs through an IGNORED_DIRS component: the old walk never
    # entered node_modules/, so the prefixed walk must yield nothing either.
    (project_root / "node_modules" / "pkg").mkdir(parents=True)
    (project_root / "node_modules" / "pkg" / "dep.js").write_text("needle\n")
    assert grep_search(project_root, "needle", include="node_modules/**/*.js") == []
    assert grep_search(project_root, "needle", include="node_modules/pkg/*.js") == []


def test_grep_search_include_absolute_path_empty(project_root: Path) -> None:
    # A leading "/" can never match a project-relative path (the include
    # regex is anchored there) — and the pruning must not let `root / "/abs"`
    # escape the project root and read host files.
    outside = project_root.parent / "grep-outside-probe"
    outside.mkdir(exist_ok=True)
    (outside / "real.py").write_text("needle\n")
    try:
        assert grep_search(project_root, "needle", include=f"{outside}/real.py") == []
        assert grep_search(project_root, "needle", include=f"{outside}/*.py") == []
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_grep_search_include_prefix_inside_nested_repo(project_root: Path) -> None:
    # ls-files branch of a first-level nested repo + prefix: candidates from
    # the nested repo's own gitignore-aware listing, cut to the prefix; a
    # sibling top-level dir outside the prefix is not even probed.
    nested = project_root / "torchvulkan"
    nested.mkdir()
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    (nested / ".gitignore").write_text("target/\n")
    api = nested / "src" / "api"
    api.mkdir(parents=True)
    (api / "binary.cpp").write_text("needle\n")
    (nested / "target" / "artifact.o").mkdir(parents=True)
    (nested / "target" / "artifact.o" / "x.cpp").write_text("needle\n")
    (nested / "readme.md").write_text("needle\n")
    probed: list[str] = []

    real_ls = tools._git_ls_files
    top_names = {project_root.name, "torchvulkan"}

    def spy(root: Path, pathspec: str | None = None) -> list[Path] | None:
        # Full listings of the project root and the prefix's own top-level dir
        # are legitimate; anything else would be a probe outside the prefix.
        if pathspec is None and root.name not in top_names:
            probed.append(root.name)
        return real_ls(root, pathspec)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(tools, "_git_ls_files", spy)
    try:
        assert grep_search(
            project_root, "needle", include="torchvulkan/src/**/*.cpp"
        ) == ["torchvulkan/src/api/binary.cpp:1: needle"]
    finally:
        monkey.undo()
    assert probed == []  # no top-level probe outside the prefix was needed


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


def test_check_command_multiline_quoted_argument(project_root: Path) -> None:
    # A newline inside a quoted argument does not split the segment:
    # multi-line commit messages that merely mention cp/rm/mv must run
    # (regression: the tester's `git commit -m '...\ncp mentions...'`).
    msg = "Live progress bar for cloudcp cp with --threads > 1\n\nmv and rm mentioned as plain words."
    check_command(project_root, f"git commit -m '{msg}'")
    check_command(project_root, f'git commit -m "{msg}"')
    check_command(project_root, f"printf '{msg}'")
    # The exact incident shape: shell-escaped apostrophe inside the message
    # (`don'"'"'t`) plus the standalone word `cp` in the body.
    incident = (
        "Live progress bar (%, speed, ETA) for cloudcp cp with --threads > 1\n\n"
        "Per-chunk thread-safe byte counting; Bar.update/_draw now run under a\n"
        "lock so worker threads don'\"'\"'t interleave stderr draws."
    )
    check_command(project_root, f"git commit -m '{incident}'")
    # Separators inside quotes are literal text, not command separators.
    check_command(project_root, "echo 'a; rm /etc/passwd' && echo done")
    # Real separators outside the quotes still split and are still checked.
    with pytest.raises(PathViolation):
        check_command(project_root, "echo 'text' && rm /etc/passwd")
    with pytest.raises(PathViolation):
        check_command(project_root, "echo 'text'\nrm /etc/passwd")


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


# -- check_command: git stash guard --------------------------------------------------


def test_check_command_refuses_git_stash(project_root: Path) -> None:
    # refs/stash is shared between all worktrees of the repository — parallel
    # task agents would push and pop each other's stashed changes.
    for cmd in (
        "git stash",
        "git stash pop",
        "git stash push -m wip",
        "git stash apply stash@{1}",
        "git stash drop",
        "git stash clear",
        "git -C . stash clear",
        "/usr/bin/git stash pop",
        "env -i git stash",
        "VAR=1 time nice git stash pop",
        "git stash -q && pytest -k test_foo",
        "echo hi; git stash drop",
        "echo hi || git stash",
    ):
        with pytest.raises(PathViolation):
            check_command(project_root, cmd)


def test_check_command_stash_refusal_suggests_worktree(project_root: Path) -> None:
    with pytest.raises(PathViolation, match=r"git worktree add"):
        check_command(project_root, "git stash pop")


def test_check_command_allows_git_stash_list_and_safe_git(project_root: Path) -> None:
    check_command(project_root, "git stash list")
    check_command(project_root, "git status")
    check_command(project_root, "git diff")
    check_command(project_root, "git -C . stash list")
    check_command(project_root, "git worktree add /tmp/clean HEAD")


def test_check_command_git_stash_unparsable_quoting_conservative(
    project_root: Path,
) -> None:
    # Unbalanced quotes with `git stash` inside → refuse rather than run blind.
    with pytest.raises(PathViolation, match=r"git worktree add"):
        check_command(project_root, "git stash 'unbalanced")
    # Unparsable but no stash in there → allowed.
    check_command(project_root, "git 'unbalanced")


def test_run_bash_refuses_git_stash(project_root: Path) -> None:
    # PathViolation (not a crash): tool_runtime converts it to a ModelRetry.
    with pytest.raises(PathViolation, match=r"git worktree add"):
        asyncio.run(run_bash(project_root, "git stash pop"))


# -- check_command: network guard ----------------------------------------------------
#
# External access belongs to the managed web_fetch tool; run_bash may only
# reach the loopback interface (a dev server / test fixture started inside
# the project). The resolver is faked throughout: the guard's decision must
# be provable without real DNS.


@pytest.fixture
def fake_dns(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Replace the guard's DNS seam: host → the addresses it "resolves" to.

    A host missing from the map resolves to nothing (refused as unprovable).
    `localhost` is handled before the resolver by the guard itself (RFC 6761).
    """
    table: dict[str, list[str]] = {}
    monkeypatch.setattr(
        tools, "_resolve_addresses", lambda host, port: list(table.get(host, []))
    )
    return table


def test_network_guard_allows_loopback(project_root: Path, fake_dns) -> None:
    # A local server (dev server, pytest fixture) is exactly what run_bash
    # stays for, so every loopback spelling must pass.
    for command in (
        "curl http://localhost:8000/health",
        "curl localhost:8000",
        "curl 127.0.0.1:5000/x",
        "curl http://127.1:8000/",
        "curl 'http://[::1]:8000/x'",
        "curl http://2130706433:8000/",  # decimal 127.0.0.1
        "curl http://[::ffff:127.0.0.1]:8000/",
        "curl http://api.localhost:9000/x",
        "wget -qO- http://127.0.0.1:5000/health",
        "http :8000/api",
        "http GET localhost:8000/api",
        "https localhost:8443/api",
        "ftp localhost",
        (
            "curl -s -X POST -H 'Content-Type: application/json' "
            "-d @payload.json localhost:8000/api/items"
        ),
        "curl -o out.json localhost:8000/data",
        'curl -H "Referer: https://example.com/" localhost:8000/x',
        "curl --max-time 5 --retry 2 localhost:8000/x",
        "curl -sSD - localhost:8000/x > out.json",
        "curl localhost:8000/x 2>&1",
    ):
        check_command(project_root, command)


def test_network_guard_refuses_external_hosts(project_root: Path, fake_dns) -> None:
    fake_dns["example.com"] = ["93.184.216.34"]
    for command in (
        "curl https://example.com/page",
        "curl example.com",
        "wget http://example.com/file.tgz",
        "http GET https://example.com/api",
        "https example.com/api",
        "ftp example.com",
        "curl --url https://example.com/",
        "curl -sL --url example.com",
        "/usr/bin/curl https://example.com",
    ):
        with pytest.raises(PathViolation, match="Command refused"):
            check_command(project_root, command)


def test_network_guard_refuses_private_and_metadata_addresses(
    project_root: Path, fake_dns
) -> None:
    # A literal is judged without the resolver — a lying resolver must not
    # launder a local/LAN/metadata address into something allowed.
    for target in (
        "169.254.169.254",  # cloud metadata
        "http://169.254.169.254/latest/meta-data/",
        "10.0.0.5",
        "http://192.168.1.1/admin",
        "172.16.0.1",
        "http://[fd00::1]/",
    ):
        with pytest.raises(PathViolation, match="Command refused"):
            check_command(project_root, f"curl {target}")


def test_network_guard_refuses_names_resolving_away(
    project_root: Path, fake_dns
) -> None:
    # A name resolving to a public address, a LAN box or a MIX of loopback and
    # non-loopback addresses is refused: only "every address is loopback"
    # counts as local.
    fake_dns["public.test"] = ["93.184.216.34"]
    fake_dns["lan.test"] = ["192.168.0.10"]
    fake_dns["metadata.test"] = ["169.254.169.254"]
    fake_dns["mixed.test"] = ["127.0.0.1", "93.184.216.34"]
    fake_dns["loopback.test"] = ["127.0.0.1", "::1"]
    for host in ("public.test", "lan.test", "metadata.test", "mixed.test"):
        with pytest.raises(PathViolation):
            check_command(project_root, f"curl http://{host}:8000/x")
    # A name that really resolves to loopback only is local — allowed.
    check_command(project_root, "curl http://loopback.test:8000/x")


def test_network_guard_refuses_unresolvable_names(project_root: Path, fake_dns) -> None:
    # Cannot be PROVEN local → refused (the client could not reach it either).
    with pytest.raises(PathViolation):
        check_command(project_root, "curl http://nonexistent.test/x")


def test_network_guard_refuses_redirect_flags(project_root: Path, fake_dns) -> None:
    # Redirect targets are never checked, so following one is refused even
    # for a loopback URL — the model is told to drop the flag.
    for command in (
        "curl -L localhost:8000/x",
        "curl -sSL localhost:8000/x",
        "curl --location localhost:8000/x",
        "curl --location-trusted localhost:8000/x",
        "curl --max-redirs 3 localhost:8000/x",
        "wget --max-redirect=2 http://127.0.0.1:5000/x",
    ):
        with pytest.raises(PathViolation, match="follows redirects"):
            check_command(project_root, command)
    # wget's -L is `--relative`, not a redirect flag: a loopback URL passes.
    check_command(project_root, "wget -L http://127.0.0.1:5000/")


def test_network_guard_only_leading_command_of_a_segment(
    project_root: Path, fake_dns
) -> None:
    fake_dns["example.com"] = ["93.184.216.34"]
    # Leading position after env assignments / wrappers / separators, as with
    # the rm/mv/cp checks — same strictness, same scope.
    for command in (
        "VAR=1 curl https://example.com",
        "nohup curl https://example.com",
        "env -i sudo curl https://example.com",
        "echo hi && curl https://example.com",
        "echo hi; wget https://example.com/f",
        "echo hi || http GET https://example.com",
        "echo hi | curl -d @- https://example.com",
        "echo hi\ncurl https://example.com",
    ):
        with pytest.raises(PathViolation):
            check_command(project_root, command)
    # Nested command substitution is NOT analysed (documented, same as the
    # rm/mv/cp guard) — and neither are interpreters (python, package
    # managers) nor git remotes: those are out of scope on purpose.
    for command in (
        "echo $(curl https://example.com)",
        "python -c 'import urllib.request; urllib.request.urlopen(url)'",
        "uv pip install requests",
        "git push origin main",
        "git clone https://example.com/repo.git",
    ):
        check_command(project_root, command)


def test_network_guard_values_are_not_targets(project_root: Path, fake_dns) -> None:
    # Headers, payloads, output files and credentials never reach the target
    # check: a URL living in a header value is data, not a request target.
    fake_dns["example.com"] = ["93.184.216.34"]
    for command in (
        "curl -H 'Referer: https://example.com/' localhost:8000/x",
        'curl -d \'{"from": "https://example.com"}\' localhost:8000/x',
        "curl --data-binary @body.json localhost:8000/x",
        "curl -o example.com.html localhost:8000/x",
        "curl -u user:pass localhost:8000/x",
        "http --json :8000/api q==example.com",
        "http :8000/api Authorization:'Bearer x'",
    ):
        check_command(project_root, command)
    # The target of the same invocation is still checked.
    with pytest.raises(PathViolation):
        check_command(project_root, "curl -d @body.json https://example.com")


def test_network_guard_unparsable_quoting_conservative(project_root: Path) -> None:
    # Unbalanced quoting with a network client in it → refuse rather than run
    # blind (the target could not be checked at all).
    with pytest.raises(PathViolation):
        check_command(project_root, "curl 'https://example.com")
    # A URL in an unrelated segment is not an invocation, though.
    check_command(project_root, "echo 'https://example.com is a URL'")


def test_network_guard_refusal_mentions_web_fetch(project_root: Path, fake_dns) -> None:
    fake_dns["example.com"] = ["93.184.216.34"]
    with pytest.raises(PathViolation, match="web_fetch"):
        check_command(
            project_root, "curl https://example.com", None, web_fetch_available=True
        )
    # Without the tool in this run the message says external access is closed
    # instead of pointing at a tool the model does not have.
    with pytest.raises(PathViolation, match="closed for this agent") as excinfo:
        check_command(project_root, "curl https://example.com")
    assert "web_fetch tool" not in str(excinfo.value)
    # Both variants name the loopback exception, so the model can act on it.
    assert "localhost" in str(excinfo.value)


def test_network_guard_via_run_bash(project_root: Path, fake_dns) -> None:
    # The guard runs through the real run_bash entry point (PathViolation, not
    # a crash: tool_runtime turns it into a ModelRetry prompt).
    fake_dns["example.com"] = ["93.184.216.34"]
    with pytest.raises(PathViolation):
        asyncio.run(run_bash(project_root, "curl https://example.com"))
    # Local commands still run — the guard only covers the network clients.
    assert "exit=0" in asyncio.run(run_bash(project_root, "echo hello"))


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


def test_run_bash_sandboxes_user_config(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_bash(task_id=...) redirects the child's drcode user config into a
    seeded per-task sandbox copy (DRCODE_USER_CONFIG_DIR): a diagnostic
    snippet's default-constructed store reads/writes the copy, never the
    real ~/.config/drcode (incident ebe6ba15). XDG itself stays untouched."""
    real = tmp_path / "real-xdg" / "drcode"
    real.mkdir(parents=True)
    (real / "models.json").write_text('{"providers": ["real"]}')
    monkeypatch.setattr(tools._config, "_USER_CONFIG_DIR", real)
    scratch = tmp_path / "scratch"

    out = asyncio.run(
        run_bash(
            project_root,
            'echo "$DRCODE_USER_CONFIG_DIR"; cat "$DRCODE_USER_CONFIG_DIR/models.json"',
            extra_writable=scratch,
            task_id="deadbeef",
        )
    )
    sandbox = scratch / "user-config"
    # The child sees the sandbox, seeded with a copy of the real config.
    assert str(sandbox) in out
    assert '{"providers": ["real"]}' in out
    assert (sandbox / "models.json").read_text() == '{"providers": ["real"]}'

    # Writes from the child land in the sandbox only.
    out = asyncio.run(
        run_bash(
            project_root,
            'echo agent > "$DRCODE_USER_CONFIG_DIR/agent-wrote"',
            extra_writable=scratch,
            task_id="deadbeef",
        )
    )
    assert out.startswith("exit=0")
    assert (sandbox / "agent-wrote").read_text().strip() == "agent"
    assert not (real / "agent-wrote").exists()
    assert (real / "models.json").read_text() == '{"providers": ["real"]}'


def test_run_bash_user_config_sandbox_without_extra_writable(
    project_root: Path,
    scratch_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a task id but no explicit extra_writable the sandbox lives under
    the task's own scratch directory (SCRATCH_ROOT/<task-id>/user-config)."""
    real = tmp_path / "real-xdg" / "drcode"
    real.mkdir(parents=True)
    monkeypatch.setattr(tools._config, "_USER_CONFIG_DIR", real)
    out = asyncio.run(
        run_bash(project_root, 'echo "$DRCODE_USER_CONFIG_DIR"', task_id="feedface")
    )
    sandbox = scratch_root / "feedface" / "user-config"
    assert str(sandbox) in out
    assert sandbox.is_dir()
    # No real config to seed from → the sandbox exists but stays empty.
    assert list(sandbox.iterdir()) == []


def test_run_bash_user_config_sandbox_persisted_across_calls(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sandbox is seeded once, not on every run_bash call: the agent's
    own writes persist for the task's lifetime (plausible-looking data)."""
    real = tmp_path / "real-xdg" / "drcode"
    real.mkdir(parents=True)
    (real / "models.json").write_text("v1")
    monkeypatch.setattr(tools._config, "_USER_CONFIG_DIR", real)
    scratch = tmp_path / "scratch"
    asyncio.run(
        run_bash(
            project_root,
            'echo agent >> "$DRCODE_USER_CONFIG_DIR/notes"',
            extra_writable=scratch,
            task_id="deadbeef",
        )
    )
    (real / "models.json").write_text("v2")
    out = asyncio.run(
        run_bash(
            project_root,
            'cat "$DRCODE_USER_CONFIG_DIR/models.json"; '
            'cat "$DRCODE_USER_CONFIG_DIR/notes"',
            extra_writable=scratch,
            task_id="deadbeef",
        )
    )
    # Still the seeded v1 (not the changed real catalog), plus the note.
    assert "v1" in out
    assert "v2" not in out
    assert "agent" in out


def test_run_bash_without_task_id_leaves_user_config_unset(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task-less child (drcode's own git plumbing) never sees a config
    redirect — not even one inherited from the parent environment."""
    monkeypatch.setenv("DRCODE_USER_CONFIG_DIR", "/tmp/foreign-sandbox")
    out = asyncio.run(run_bash(project_root, 'echo "[$DRCODE_USER_CONFIG_DIR]"'))
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


# -- filesystem-scan guard (run_bash) ----------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # The 2026-11-23 incident command, verbatim.
        (
            "find / -maxdepth 6 -name 'torchvulkan-blockers.md' 2>/dev/null"
            " | grep -v build-torch | head"
        ),
        "find ~ -name x",
        "find $HOME -name x",
        "find /Users/x -name y",
        "find ../",
        "sudo find /",
        "du -sh ~",
        "ls -R /Users/x",
        "rg pattern /Volumes/Backup",
        "grep -r pattern /Users/x",
        "grep -rn pattern /Users/x",
        "fd pattern /",
        "fd --search-path /",
    ],
)
def test_run_bash_refuses_scan_outside_project(
    project_root: Path, command: str
) -> None:
    # A filesystem scan rooted outside the writable area walks the user's
    # personal data and fires macOS TCC prompts attributed to DrCode.
    with pytest.raises(PathViolation):
        check_command(project_root, command)


def test_run_bash_refuses_scan_outside_via_run_bash(project_root: Path) -> None:
    with pytest.raises(PathViolation):
        asyncio.run(run_bash(project_root, "du -sh ~"))


@pytest.mark.parametrize(
    "command",
    [
        "find . -name x",
        "find -maxdepth 6 -name x",  # no path: defaults to `.`
        "find",
        "du -sh .",
        "ls -R",  # no path: defaults to `.`
        "ls -la ~",  # not recursive: one dir, no walk
        "grep -n pattern file.txt",  # not recursive
        "grep -r pattern src",
        "rg -e pat --type py",
        "mdfind -name torchvulkan-blockers",  # Spotlight, not a walk — allowed
    ],
)
def test_run_bash_allows_scans_inside_project(project_root: Path, command: str) -> None:
    check_command(project_root, command)


def test_run_bash_allows_scan_of_project_and_scratch(
    project_root: Path, scratch_root: Path
) -> None:
    check_command(project_root, f"find {project_root} -name x")
    check_command(
        project_root, f"find {scratch_root} -name x", extra_writable=scratch_root
    )


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


# -- child process priority -------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX preexec_fn nice")
def test_child_priority_kwargs_posix() -> None:
    # POSIX: a preexec_fn that nices the child plus start_new_session — the
    # child gets its own process group (grandchildren die with it on
    # timeout/cancel) and no controlling tty.
    kwargs = tools.child_priority_kwargs()
    assert set(kwargs) == {"preexec_fn", "start_new_session"}
    assert callable(kwargs["preexec_fn"])
    assert kwargs["start_new_session"] is True


def test_child_priority_kwargs_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    # Windows: BELOW_NORMAL_PRIORITY_CLASS, no preexec_fn (not supported
    # there). subprocess lacks the constant on POSIX, so stub the module.
    stub = types.SimpleNamespace(BELOW_NORMAL_PRIORITY_CLASS=0x00004000)
    monkeypatch.setattr(tools, "subprocess", stub)
    monkeypatch.setattr(tools.os, "name", "nt")
    kwargs = tools.child_priority_kwargs()
    assert kwargs == {"creationflags": 0x00004000}


def test_child_priority_kwargs_unknown_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools.os, "name", "java")
    assert tools.child_priority_kwargs() == {}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX nice inheritance")
def test_run_bash_child_runs_below_parent_priority(project_root: Path) -> None:
    # The whole child tree (shell + its children) is niced below the drcode
    # process, so a CPU-saturating child cannot starve the TUI render loop.
    # nice clamps at 19 without error, hence the min().
    parent_nice = os.nice(0)
    out = asyncio.run(
        run_bash(project_root, f'"{sys.executable}" -c "import os; print(os.nice(0))"')
    )
    child_nice = int(out.splitlines()[-1])
    assert child_nice == min(parent_nice + 10, 19)


@pytest.mark.skipif(sys.platform == "win32", reason="pgrep-based check")
def test_run_bash_cancel_still_kills_niced_child(project_root: Path) -> None:
    # preexec_fn must not break the CancelledError path: cancelling the run
    # kills the child, nothing outlives the tool call.
    marker = f"drcode-cancel-{os.getpid()}"
    asyncio.run(_cancel_run_bash(project_root, marker))
    leftover = subprocess.run(
        ["pgrep", "-f", marker], check=False, capture_output=True, text=True
    )
    assert leftover.returncode != 0  # no surviving process


async def _cancel_run_bash(project_root: Path, marker: str) -> None:
    src = f"import time; time.sleep(30)  # {marker}"
    task = asyncio.create_task(run_bash(project_root, f'"{sys.executable}" -c "{src}"'))
    await asyncio.sleep(1.0)
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_run_bash_kill_tolerates_already_reaped_process() -> None:
    # The kill can race the child's own exit: communicate() has reaped the
    # process and closed the transport, and a late kill() (timeout/cancel
    # path) used to raise ProcessLookupError and crash the caller.
    async def _scenario() -> None:
        proc = await asyncio.create_subprocess_shell("true")
        await proc.communicate()
        tools._kill_proc(proc)  # must not raise

    asyncio.run(_scenario())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_run_bash_timeout_kills_grandchildren(project_root: Path) -> None:
    # The timeout kill must take down the WHOLE process group: a grandchild
    # surviving the shell's SIGKILL reparents to init and keeps the TUI's
    # controlling tty open as a competing stdin reader (the input-stealing
    # incident). `sh -c 'sleep 30 …' & wait` leaves exactly such a grandchild.
    marker = f"drcode-grandchild-{os.getpid()}"
    out = asyncio.run(
        run_bash(
            project_root,
            f"sh -c 'sleep 30 # {marker}' & wait",
            timeout=1,
        )
    )
    assert "timed out" in out
    leftover = subprocess.run(
        ["pgrep", "-f", marker], check=False, capture_output=True, text=True
    )
    assert leftover.returncode != 0  # no surviving grandchild


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_run_bash_normal_exit_kills_grandchildren(project_root: Path) -> None:
    # NORMAL completion needs the same group sweep as timeout/cancel: a
    # grandchild surviving the shell's exit reparents to init and lives on
    # (incident 2026-11-02: 10 orphaned processes at ~80% CPU for hours).
    # The outer shell backgrounds a python grandchild holding the marker and
    # exits immediately — before the fix that grandchild outlived run_bash by
    # the full 30 seconds. (A plain `sh -c 'sleep 30 # marker'` is no good:
    # the shell exec-optimizes the single command away and the marker
    # vanishes from every cmdline, so pgrep has nothing to find.) The
    # grandchild's output is redirected so it does NOT hold run_bash's pipes
    # (a detached daemon/pty harness wouldn't either): otherwise
    # communicate() would wait for the pipes to close and the normal-exit
    # path would never fire while it lived.
    marker = f"drcode-orphan-{os.getpid()}"
    out = asyncio.run(
        run_bash(
            project_root,
            f'"{sys.executable}" -c "import time; time.sleep(30)  # {marker}"'
            " >/dev/null 2>&1 &",
        )
    )
    assert out.startswith("exit=0")
    leftover = subprocess.run(
        ["pgrep", "-f", marker], check=False, capture_output=True, text=True
    )
    assert leftover.returncode != 0  # no surviving grandchild


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX tty semantics")
def test_run_bash_child_stdin_is_not_a_tty(project_root: Path) -> None:
    # The child's stdin is DEVNULL, never the inherited tty: an interactive
    # prompt (npm exec's "Ok to proceed? (y)") gets EOF instead of hanging
    # forever with the TUI's keystrokes on the line.
    out = asyncio.run(run_bash(project_root, "[ -t 0 ] && echo tty || echo notty"))
    assert out == "exit=0\nnotty"


def test_run_bash_child_stdin_reads_eof(project_root: Path) -> None:
    # A stdin-reading child (cat) must exit immediately on EOF — with the
    # inherited tty stdin it would hang until the timeout instead.
    out = asyncio.run(
        run_bash(project_root, f'"{sys.executable}" -c "import sys; sys.stdin.read()"')
    )
    assert out == "exit=0\n(no output)"


def test_run_bash_output_truncated(project_root: Path, monkeypatch) -> None:
    monkeypatch.setattr(tools, "MAX_TOOL_OUTPUT", 1000)
    out = asyncio.run(run_bash(project_root, "echo " + "x" * 2000))
    prefix = "exit=0\n"
    assert out.startswith(prefix)
    body = out[len(prefix) :]
    # Truncation is VISIBLE: head + note + tail, the note names the omitted
    # span and the whole body stays within MAX_TOOL_OUTPUT.
    assert "[output truncated:" in body
    assert "2000 characters omitted" in body
    assert len(body) <= tools.MAX_TOOL_OUTPUT
    # Both ends survive: head (75%) and tail (25%) slices of the x-run.
    head, note, rest = body.partition("[output truncated:")
    assert note
    assert head.strip("x\n") == ""
    tail = rest.split("]\n", 1)[1]
    assert len(head) > len(tail)
    assert tail.strip("x\n") == ""
    # Short output is untouched.
    ok = asyncio.run(run_bash(project_root, "echo hi"))
    assert ok == "exit=0\nhi"


def test_run_bash_output_truncated_keeps_last_line(
    project_root: Path, monkeypatch
) -> None:
    monkeypatch.setattr(tools, "MAX_TOOL_OUTPUT", 1000)
    script = "echo banner; seq 1 300; echo DONE"
    out = asyncio.run(run_bash(project_root, script))
    # The tail slice keeps the closing summary line (snapped to whole lines,
    # no mid-line cut of "DONE").
    assert out.rstrip().endswith("DONE")
    assert "banner" in out


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


def test_remove_scratch_kills_scratch_dwellers(scratch_root: Path) -> None:
    # A process fully detached from run_bash's process group (own session,
    # like a forkpty harness) but with its cwd inside the scratch must not
    # survive the cleanup: deleting the directory alone leaves it running
    # (incident 2026-11-02: ten orphaned ~80%-CPU processes).
    scratch = scratch_root / "abc12345"
    scratch.mkdir(parents=True)
    (scratch / "pty_geom.log").write_text("x")
    dweller = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        cwd=str(scratch),
        start_new_session=True,
    )
    try:
        assert dweller.poll() is None
        tools.remove_scratch("abc12345")
        assert not scratch.exists()
        assert dweller.wait(timeout=15) != 0  # terminated by the echelon
    finally:
        if dweller.poll() is None:
            dweller.kill()
            dweller.wait()


def test_remove_scratch_dweller_scan_is_best_effort(
    scratch_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A broken/absent lsof must not break the removal: degrade to rmtree-only.
    monkeypatch.setattr(
        tools, "_scratch_dweller_pids", lambda scratch: (_ for _ in ()).throw(OSError)
    )
    scratch = scratch_root / "abc12345"
    scratch.mkdir(parents=True)
    tools.remove_scratch("abc12345")
    assert not scratch.exists()
