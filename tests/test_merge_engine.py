"""Tests for the algorithmic merge engine (`drcode.merge_engine`): the
reversed merge (base INTO the task's worktree), the structured conflict
triage report, the provably-safe auto-resolvers (identical stages,
zdiff3 markers, rerere replay) and the finish/abort conclusion helpers.

Everything runs against real throwaway git repositories — no network,
no model stubs needed (the engine is deliberately LLM-free)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from drcode import merge_engine as me

# -- helpers --------------------------------------------------------------------


def _run(
    root: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if check:
        assert proc.returncode == 0, f"git {args}: {proc.stderr}"
    return proc


@pytest.fixture
def repo(project_root: Path) -> Path:
    """A git repository on `main` with one commit (the merge base)."""
    _run(project_root, "init", "-q", "-b", "main")
    _run(project_root, "config", "user.email", "test@example.com")
    _run(project_root, "config", "user.name", "Test")
    (project_root / "app.py").write_text("a\nb\n")
    _run(project_root, "add", ".")
    _run(project_root, "commit", "-qm", "base")
    return project_root


def _task_worktree(repo: Path, task_id: str, start: str = "HEAD") -> Path:
    """Create the task's worktree the way `drcode.task_env` does."""
    wt = repo / ".drcode" / "worktrees" / task_id
    _run(repo, "worktree", "add", str(wt), "-b", f"drcode/{task_id}", start)
    return wt


def _head(repo: Path) -> str:
    return _run(repo, "rev-parse", "HEAD").stdout.strip()


def _stage_equal_versions(wt: Path, path: str, oid: str) -> None:
    """Fabricate an UU conflict whose two stages are byte-identical (the
    only way git produces one is via exotic rename/mode collisions, but
    the resolver must handle it wherever it appears)."""
    proc = subprocess.run(
        ["git", "update-index", "--index-info"],
        cwd=wt,
        input=f"100644 {oid} 2\t{path}\n100644 {oid} 3\t{path}\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0


# -- the reversed merge -----------------------------------------------------------


def test_clean_reverse_merge_makes_base_fast_forward(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    (wt / "task.txt").write_text("task work\n")
    _run(wt, "add", ".")
    _run(wt, "commit", "-qm", "task work")
    (repo / "main.txt").write_text("main work\n")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "main work")

    report = me.merge_base_into_worktree(repo, "t1", wt, "main")

    assert report.ok and not report.conflicted
    assert report.fast_forward is True
    # The merge commit lives on the TASK branch, the checkout is clean.
    assert "Merge branch 'main'" in _run(wt, "log", "--oneline", "-1").stdout
    assert (repo / "task.txt").exists() is False
    # And the merge into the user's checkout is a pure fast-forward.
    assert me.is_fast_forward(repo, "drcode/t1", "main")
    _run(repo, "merge", "--ff-only", "drcode/t1")
    assert (repo / "task.txt").read_text(encoding="utf-8") == "task work\n"


def test_task_branch_already_contains_base(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    (repo / "app.py").write_text("changed\n")
    _run(repo, "commit", "-qam", "main edit")
    # The task branch is behind — merging base INTO it fast-forwards it,
    # and the report says up-to-date only when it already contains base.
    report = me.merge_base_into_worktree(repo, "t1", wt, "main")
    assert report.ok and not report.conflicted
    assert _run(wt, "rev-parse", "HEAD").stdout.strip() == _head(repo)

    report2 = me.merge_base_into_worktree(repo, "t1", wt, "main")
    assert report2.ok and report2.up_to_date


def test_resumed_staged_merge_is_not_ff_ready(repo: Path) -> None:
    """A merge resolved and staged but never committed (the screen was
    closed mid-resolution): the re-run refuses to conclude anything —
    ok + not conflicted, but NO merge commit exists, so `fast_forward`
    stays None and the summary must not claim a clean merge."""
    wt = _task_worktree(repo, "t1")
    (wt / "app.py").write_text("task\nb\n")
    _run(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("main\nb\n")
    _run(repo, "commit", "-qam", "main edit")
    report = me.merge_base_into_worktree(repo, "t1", wt, "main")
    assert report.conflicted

    # Resolve and stage by hand, do NOT commit, re-run the prepare.
    (wt / "app.py").write_text("resolved\nb\n")
    _run(wt, "add", "app.py")
    report2 = me.merge_base_into_worktree(repo, "t1", wt, "main")

    assert report2.ok and not report2.conflicted
    assert report2.fast_forward is None  # nothing concluded — no ff claim
    assert me.in_progress(wt)  # the merge is still half-done
    _head_before = _head(wt)
    summary = report2.summary()
    assert "NOT committed" in summary
    assert "--ff-only" in summary
    assert "cleanly" not in summary
    # The branch tip did not move: only a commit concludes it.
    assert _head(wt) == _head_before


# -- the triage report -------------------------------------------------------------


def test_content_conflict_parks_in_worktree_with_zdiff3_report(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    (wt / "app.py").write_text("task\nb\n")
    _run(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("main\nb\n")
    _run(repo, "commit", "-qam", "main edit")

    report = me.merge_base_into_worktree(repo, "t1", wt, "main", save=True)

    assert report.ok and report.conflicted
    assert report.fast_forward is None
    [conflict] = report.remaining
    assert conflict.path == "app.py"
    assert conflict.kind == me.CONTENT
    assert conflict.base and conflict.ours and conflict.theirs
    assert conflict.base_oid and conflict.ours_oid and conflict.theirs_oid
    assert conflict.hunks == 1
    assert conflict.conflict_lines is not None
    assert conflict.conflict_lines == {"ours": 1, "base": 1, "theirs": 1}
    assert conflict.resolution is None
    # zdiff3 markers: the common ancestor section is embedded.
    markers = (wt / "app.py").read_text(encoding="utf-8")
    assert "<<<<<<<" in markers and ">>>>>>>" in markers
    assert "|||||||" in markers  # the zdiff3 base section
    # The USER'S CHECKOUT is untouched.
    assert (repo / "app.py").read_text(encoding="utf-8") == "main\nb\n"
    _run(repo, "status", "--porcelain", "-uno")  # exits 0 either way
    status = _run(repo, "status", "--porcelain", "-uno", check=False)
    assert "UU" not in status.stdout and "AA" not in status.stdout
    # The report is persisted and round-trips.
    persisted = me.load_report(repo, "t1")
    assert persisted is not None
    assert persisted.to_dict() == report.to_dict()
    assert me.report_path(repo, "t1").is_file()


def test_modify_delete_conflict_classified(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    (wt / "app.py").write_text("edited\n")
    _run(wt, "commit", "-qam", "task edit")
    _run(repo, "rm", "-q", "app.py")
    _run(repo, "commit", "-qm", "main deletes")

    report = me.merge_base_into_worktree(repo, "t1", wt, "main")

    assert report.conflicted
    [conflict] = report.remaining
    assert conflict.kind == me.MODIFY_DELETE
    assert conflict.hunks is None  # nothing to parse — one side is gone
    # The merge is rollable: abort restores the task branch's own state.
    ok, _ = me.abort(wt)
    assert ok
    assert (wt / "app.py").read_text(encoding="utf-8") == "edited\n"
    assert not me.in_progress(wt)


def test_rename_delete_conflict_classified(repo: Path) -> None:
    (repo / "keep.txt").write_text("k\n")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "more base")
    wt = _task_worktree(repo, "t1")
    _run(wt, "mv", "app.py", "renamed.py")
    _run(wt, "commit", "-qam", "task renames")
    _run(repo, "rm", "-q", "app.py")
    _run(repo, "commit", "-qm", "main deletes")

    report = me.merge_base_into_worktree(repo, "t1", wt, "main")

    assert report.conflicted
    [conflict] = report.remaining
    assert conflict.path == "renamed.py"
    assert conflict.kind == me.RENAME_DELETE


def test_rename_rename_conflict_classified(repo: Path) -> None:
    base = _head(repo)
    wt = _task_worktree(repo, "t1")
    _run(wt, "mv", "app.py", "from-task.py")
    _run(wt, "commit", "-qam", "task renames")
    # A second branch (from the same base) renames the SAME file to a
    # different name; main first takes t1's rename, then the engine
    # merges main into t2 — the two renames collide.
    wt2 = _task_worktree(repo, "t2", base)
    _run(wt2, "mv", "app.py", "from-t2.py")
    _run(wt2, "commit", "-qam", "t2 renames")
    _run(repo, "merge", "-q", "--no-edit", "drcode/t1")

    report = me.merge_base_into_worktree(repo, "t2", wt2, "main")

    assert report.conflicted
    kinds = {f.path: f.kind for f in report.remaining}
    # Both renamed paths collide (RENAME_RENAME) and the original path is
    # deleted by both sides (DELETE_DELETE — part of the same collision).
    assert me.RENAME_RENAME in kinds.values()
    assert set(kinds.values()) <= {me.RENAME_RENAME, me.DELETE_DELETE}


def test_add_add_conflict_classified(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    (wt / "new.txt").write_text("task version\n")
    _run(wt, "add", ".")
    _run(wt, "commit", "-qm", "task adds")
    (repo / "new.txt").write_text("main version\n")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "main adds")

    report = me.merge_base_into_worktree(repo, "t1", wt, "main")

    assert report.conflicted
    [conflict] = report.remaining
    assert conflict.kind == me.ADD_ADD
    assert conflict.base is False and conflict.ours and conflict.theirs
    assert conflict.hunks == 1


# -- the safe auto-resolvers ---------------------------------------------------------


def test_identical_stages_auto_resolved(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    (wt / "app.py").write_text("task\nb\n")
    _run(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("main\nb\n")
    _run(repo, "commit", "-qam", "main edit")
    report = me.merge_base_into_worktree(repo, "t1", wt, "main")
    assert report.conflicted
    ours_oid = report.remaining[0].ours_oid
    assert ours_oid
    # Fabricate the identical-both-sides conflict: both stages carry the
    # task's blob, the working tree already holds that content.
    _stage_equal_versions(wt, "app.py", ours_oid)
    (wt / "app.py").write_text("task\nb\n")

    resolved = me.resolve_identical(wt)

    assert resolved == ["app.py"]
    assert me.unmerged_paths(wt) == []
    # The staged content IS the shared blob — nothing was invented.
    assert (wt / "app.py").read_text(encoding="utf-8") == "task\nb\n"


def test_identical_resolver_ignores_differing_content(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    (wt / "app.py").write_text("task\nb\n")
    _run(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("main\nb\n")
    _run(repo, "commit", "-qam", "main edit")
    report = me.merge_base_into_worktree(repo, "t1", wt, "main")
    assert report.conflicted

    assert me.resolve_identical(wt) == []
    assert me.unmerged_paths(wt) == ["app.py"]


def test_rerere_replays_recorded_resolution(repo: Path) -> None:
    base = _head(repo)
    # First task branch: the conflict is resolved BY HAND, recorded.
    wt1 = _task_worktree(repo, "t1")
    (wt1 / "app.py").write_text("TASK\nb\n")
    _run(wt1, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("MAIN\nb\n")
    _run(repo, "commit", "-qam", "main edit")
    report1 = me.merge_base_into_worktree(repo, "t1", wt1, "main")
    assert report1.conflicted
    (wt1 / "app.py").write_text("TASK\nb\n")
    _run(wt1, "add", "app.py")
    me.rerere_record(wt1)
    ok, _ = me.finish(wt1)
    assert ok

    # Second task branch, the SAME edit against the same base: the
    # identical conflict shape must resolve itself.
    wt2 = _task_worktree(repo, "t2", base)
    (wt2 / "app.py").write_text("TASK\nb\n")
    _run(wt2, "commit", "-qam", "same edit again")

    report2 = me.merge_base_into_worktree(repo, "t2", wt2, "main")

    assert report2.ok
    assert report2.conflicted is False
    assert report2.rerere["replayed"] == ["app.py"]
    resolved = report2.resolved
    assert [f.path for f in resolved] == ["app.py"]
    assert resolved[0].resolution == "rerere"
    # The merge was concluded — the branch fast-forwards into main.
    assert report2.fast_forward is True
    assert (wt2 / "app.py").read_text(encoding="utf-8") == "TASK\nb\n"
    assert me.unmerged_paths(wt2) == []


def test_rerere_replay_never_stages_markers(repo: Path) -> None:
    # A conflict NOT seen before: rerere has nothing recorded, the file
    # keeps its markers and must stay unmerged.
    wt = _task_worktree(repo, "t1")
    (wt / "app.py").write_text("task\nb\n")
    _run(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("main\nb\n")
    _run(repo, "commit", "-qam", "main edit")
    report = me.merge_base_into_worktree(repo, "t1", wt, "main")
    assert report.conflicted
    assert report.rerere["replayed"] == []
    assert me.unmerged_paths(wt) == ["app.py"]


# -- finish / abort / refusals -------------------------------------------------------


def test_finish_refuses_while_conflicts_remain(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    (wt / "app.py").write_text("task\nb\n")
    _run(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("main\nb\n")
    _run(repo, "commit", "-qam", "main edit")
    me.merge_base_into_worktree(repo, "t1", wt, "main")

    ok, message = me.finish(wt)

    assert not ok
    assert "app.py" in message
    assert me.in_progress(wt)


def test_finish_concludes_and_fast_forwards(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    (wt / "app.py").write_text("resolved\nb\n")
    _run(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("main\nb\n")
    _run(repo, "commit", "-qam", "main edit")
    report = me.merge_base_into_worktree(repo, "t1", wt, "main")
    assert report.conflicted
    (wt / "app.py").write_text("resolved\nb\n")
    _run(wt, "add", "app.py")

    ok, _ = me.finish(wt)

    assert ok
    assert not me.in_progress(wt)
    assert me.is_fast_forward(repo, "drcode/t1", "main")
    assert report.fast_forward is None  # the earlier report is frozen


def test_finish_without_merge_in_progress(repo: Path) -> None:
    ok, message = me.finish(repo)
    assert not ok and "no merge" in message


def test_abort_without_merge_in_progress(repo: Path) -> None:
    ok, message = me.abort(repo)
    assert not ok and "no merge" in message


def test_dirty_worktree_refused_without_merge(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    (repo / "main.txt").write_text("main work\n")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "main work")
    (wt / "app.py").write_text("uncommitted\n")
    # A TRACKED file modified in the worktree (untracked scratch files
    # never block a merge — git merges around them just fine).
    (wt / "app.py").write_text("uncommitted\n")

    report = me.merge_base_into_worktree(repo, "t1", wt, "main")

    assert not report.ok
    assert "snapshot" in report.message
    # Git state untouched: no merge started.
    assert not me.in_progress(wt)
    assert _head(repo) == _run(wt, "rev-parse", "main").stdout.strip()


def test_wrong_branch_refused(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    report = me.merge_base_into_worktree(repo, "t1", wt, "main", branch="drcode/other")
    assert not report.ok
    assert "expected drcode/other" in report.message


def test_missing_base_refused(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    report = me.merge_base_into_worktree(repo, "t1", wt, "no/such/branch")
    assert not report.ok
    assert "does not exist" in report.message


def test_missing_worktree_refused(repo: Path) -> None:
    report = me.merge_base_into_worktree(
        repo, "t1", repo / ".drcode" / "worktrees" / "gone", "main"
    )
    assert not report.ok
    assert "does not exist" in report.message


def test_dirty_worktree_resumes_in_progress_merge(repo: Path) -> None:
    """A dirty tree is only a refusal BEFORE a merge starts — an already
    conflicted (dirty by definition) worktree is re-triaged, not refused:
    the engine must be callable twice on the same conflict."""
    wt = _task_worktree(repo, "t1")
    (wt / "app.py").write_text("task\nb\n")
    _run(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("main\nb\n")
    _run(repo, "commit", "-qam", "main edit")
    first = me.merge_base_into_worktree(repo, "t1", wt, "main")
    assert first.conflicted
    second = me.merge_base_into_worktree(repo, "t1", wt, "main")
    assert second.ok and second.conflicted
    assert [f.path for f in second.remaining] == ["app.py"]


def test_report_round_trip_all_fields(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    (wt / "app.py").write_text("task\nb\n")
    _run(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("main\nb\n")
    _run(repo, "commit", "-qam", "main edit")
    report = me.merge_base_into_worktree(repo, "t1", wt, "main", save=True)
    loaded = me.load_report(repo, "t1")
    assert loaded is not None
    assert loaded.to_dict() == report.to_dict()
    # from_dict reconstructs the dataclass faithfully.
    restored = me.ConflictFile.from_dict(report.files[0].to_dict())
    assert restored == report.files[0]
    # A corrupt/absent report is None, never a crash.
    assert me.load_report(repo, "absent") is None


def test_load_report_corrupt_file_is_none(repo: Path) -> None:
    path = me.report_path(repo, "t1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert me.load_report(repo, "t1") is None


def test_summary_shapes(repo: Path) -> None:
    wt = _task_worktree(repo, "t1")
    (wt / "app.py").write_text("task\nb\n")
    _run(wt, "commit", "-qam", "task edit")
    (repo / "app.py").write_text("main\nb\n")
    _run(repo, "commit", "-qam", "main edit")
    report = me.merge_base_into_worktree(repo, "t1", wt, "main")
    summary = report.summary()
    assert "1 conflicted file(s) (1 content)" in summary
    assert "--ff-only" in summary
    refused = me.MergeReport(
        task_id="t1", branch="drcode/t1", base_branch="main", worktree=str(wt)
    )
    refused.ok = False
    refused.message = "refused"
    assert refused.summary() == "refused"
    # ok + not conflicted but the ff flag says NO: the branch holds
    # nothing beyond the base — no "fast-forward" claim.
    behind = me.MergeReport(
        task_id="t1",
        branch="drcode/t1",
        base_branch="main",
        worktree=str(wt),
        ok=True,
        fast_forward=False,
        message="merged main into drcode/t1 cleanly",
    )
    behind_summary = behind.summary()
    assert "no commits beyond main" in behind_summary
    assert "fast-forward" not in behind_summary
