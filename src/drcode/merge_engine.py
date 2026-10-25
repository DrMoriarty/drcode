"""Algorithmic merge engine for task branches (`drcode.merge_engine`).

The problem it solves: merging a finished task's branch `drcode/<id>` into
the user's base branch CONFLICTS — and the old behaviour (abort the merge
in the user's checkout, "sort it out by hand") hands a human zero context.
Real work was lost this way (task #6c065401: map-reduce wiring destroyed
during a manual conflict resolution of merge 10ad6ab).

The engine turns the merge around into a SAFE PLACE. Instead of merging
`drcode/<id>` into the base branch in the user's checkout, it merges the
BASE BRANCH INTO the task's worktree, on the task's own branch:

    git -c rerere.enabled=true -c merge.conflictstyle=zdiff3 \\
        -C <task worktree> merge --no-edit <base-branch>

Consequences of the reversal:

- a conflict now lives in the task's ISOLATED worktree, never in the
  user's checkout (which stays clean the whole time);
- once the conflicts are resolved there (by hand, by the agent resolver —
  the next layer on top of this module — or automatically, see below) and
  committed, merging `drcode/<id>` into the base branch is a
  FAST-FORWARD: no second round of conflicts is possible;
- any verification (tests, lint) runs in the isolated tree before the
  fast-forward, not in the user's checkout.

The engine is deliberately ALGORITHMIC — no LLM anywhere. It does three
things on its own, each of them provably safe (no freedom of choice):

1. TRIAGE: a structured JSON report of every conflicted path — conflict
   kind (content / add-add / modify-delete / rename-rename /
   rename-delete / delete-delete), the three stage blobs, conflict-hunk
   sizes, whether a merge-base version exists. The report feeds both the
   upcoming agent resolver and the TUI ConflictScreen; it is persisted
   under `.drcode/merge_reports/<task-id>.json`.
2. SAFE AUTO-RESOLVERS, applied immediately after the merge:
   - both stages byte-identical (same blob oid, same mode) → take it —
     there is literally nothing to decide;
   - zdiff3 conflict markers (`git merge.conflictstyle=zdiff3`, the
     `git merge-file --zdiff3` format): the conflict hunks embed the
     COMMON ANCESTOR section and factor out lines both sides changed the
     same way — strictly more information for a human or an agent, zero
     resolution bias;
   - git rerere (enabled per-merge via `-c`, never written into the
     repository config): a conflict shape a human/agent has already
     resolved once is replayed automatically — task branches repeatedly
     hit the same files (`tui/app.py`, `pipeline.py`, tests), so the
     second and every later occurrence resolves itself. A replayed file
     is staged only when NO conflict markers remain in it.
3. CONCLUSION HELPERS: `finish` (commit the resolved merge) and `abort`
   (roll back to the pre-merge state); `is_fast_forward` tells the caller
   whether the resulting branch can now fast-forward into the base.

Contract (unchanged): **drcode never merges on its own.** Every entry
point here is invoked by an explicit user action (the future
ConflictScreen) or from the user's own `.drcode/workflow.py` hook — the
workflow template's `_merge_task_branch` is the reference caller: it runs
the reversed merge and only on success fast-forwards the user's checkout;
on any refusal/conflict it reports and leaves the checkout untouched.
Manual (hook-less) merges are unaffected.

Sync module (plain subprocesses, mirroring `drcode.task_env`): callers on
the event loop wrap the calls in `asyncio.to_thread`.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ADD_ADD",
    "CONTENT",
    "DELETE_DELETE",
    "MODIFY_DELETE",
    "RENAME_DELETE",
    "RENAME_RENAME",
    "ConflictFile",
    "MergeReport",
    "abort",
    "apply_zdiff3_markers",
    "finish",
    "in_progress",
    "is_fast_forward",
    "load_report",
    "merge_base_into_worktree",
    "report_path",
    "rerere_replay",
    "rerere_status",
    "resolve_identical",
    "save_report",
    "triage",
    "unmerged_paths",
]

# Conflict kinds (`ConflictFile.kind`). Derived from the porcelain v2
# unmerged codes plus stage-blob analysis — see `triage`.
CONTENT = "content"  # both sides modified (UU)
ADD_ADD = "add-add"  # both sides added (AA / AU / UA)
MODIFY_DELETE = "modify-delete"  # one side edited, the other deleted (DU/UD)
RENAME_RENAME = "rename-rename"  # both sides renamed the same file apart
RENAME_DELETE = "rename-delete"  # one side renamed, the other deleted
DELETE_DELETE = "delete-delete"  # both deleted (DD; only via renames)

# Zero oid git reports for an absent stage in `status --porcelain=v2`.
_ZERO_OID = "0" * 40

# Where the engine persists triage reports (inside the already-gitignored
# `.drcode/` state directory).
_REPORTS_SUBDIR = Path(".drcode") / "merge_reports"

# Longest a single git call may run (mirrors task_env).
_TIMEOUT = 120


# -- data model --------------------------------------------------------------


@dataclass
class ConflictFile:
    """One conflicted path of an in-progress merge.

    `stages`/`oids`/`modes` mirror `git ls-files -u`: `base` is the merge
    ancestor (absent for add-add), `ours` the task-branch version, `theirs`
    the base-branch version (the merge runs INSIDE the task's worktree, so
    "ours" is the task). `hunks`/`conflict_lines` describe the conflict
    markers in the working tree (None when there is nothing to parse, e.g.
    a modify-delete). `resolution` is set when an auto-resolver cleared
    the path: "identical-both-sides" or "rerere".
    """

    path: str
    kind: str
    base: bool = False
    ours: bool = False
    theirs: bool = False
    base_oid: str | None = None
    ours_oid: str | None = None
    theirs_oid: str | None = None
    base_mode: str | None = None
    ours_mode: str | None = None
    theirs_mode: str | None = None
    hunks: int | None = None
    conflict_lines: dict[str, int] | None = None
    resolution: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind,
            "stages": {"base": self.base, "ours": self.ours, "theirs": self.theirs},
            "oids": {
                "base": self.base_oid,
                "ours": self.ours_oid,
                "theirs": self.theirs_oid,
            },
            "modes": {
                "base": self.base_mode,
                "ours": self.ours_mode,
                "theirs": self.theirs_mode,
            },
            "hunks": self.hunks,
            "conflict_lines": self.conflict_lines,
            "resolution": self.resolution,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ConflictFile:
        stages = data.get("stages") or {}
        oids = data.get("oids") or {}
        modes = data.get("modes") or {}
        assert isinstance(stages, dict) and isinstance(oids, dict)
        assert isinstance(modes, dict)
        return cls(
            path=str(data.get("path", "")),
            kind=str(data.get("kind", CONTENT)),
            base=bool(stages.get("base")),
            ours=bool(stages.get("ours")),
            theirs=bool(stages.get("theirs")),
            base_oid=oids.get("base"),  # type: ignore[arg-type]
            ours_oid=oids.get("ours"),  # type: ignore[arg-type]
            theirs_oid=oids.get("theirs"),  # type: ignore[arg-type]
            base_mode=modes.get("base"),  # type: ignore[arg-type]
            ours_mode=modes.get("ours"),  # type: ignore[arg-type]
            theirs_mode=modes.get("theirs"),  # type: ignore[arg-type]
            hunks=data.get("hunks"),  # type: ignore[arg-type]
            conflict_lines=data.get("conflict_lines"),  # type: ignore[arg-type]
            resolution=data.get("resolution"),  # type: ignore[arg-type]
        )


@dataclass
class MergeReport:
    """The structured outcome of one engine run (the triage JSON).

    `ok` — the merge was attempted and reached a well-defined state (clean
    merge, up-to-date, or a triaged conflict). False means the engine
    REFUSED (dirty worktree, wrong branch, no base, …) and git state is
    untouched. `conflicted` — conflicts remain for a human/agent. `files`
    lists every conflicted path as first triaged, `resolved` the subset an
    auto-resolver cleared, `remaining` what still needs a decision.
    `fast_forward` (True/False once the merge concluded or was clean, None
    while conflicts stand): whether merging the task branch into the base
    is now a fast-forward.
    """

    task_id: str
    branch: str
    base_branch: str
    worktree: str
    ok: bool = False
    conflicted: bool = False
    up_to_date: bool = False
    fast_forward: bool | None = None
    message: str = ""
    files: list[ConflictFile] = field(default_factory=list)
    rerere: dict[str, object] = field(default_factory=dict)
    conflict_style: str = "zdiff3"

    @property
    def resolved(self) -> list[ConflictFile]:
        return [f for f in self.files if f.resolution is not None]

    @property
    def remaining(self) -> list[ConflictFile]:
        return [f for f in self.files if f.resolution is None]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": 1,
            "task_id": self.task_id,
            "branch": self.branch,
            "base_branch": self.base_branch,
            "worktree": self.worktree,
            "ok": self.ok,
            "conflicted": self.conflicted,
            "up_to_date": self.up_to_date,
            "fast_forward": self.fast_forward,
            "message": self.message,
            "conflict_style": self.conflict_style,
            "rerere": self.rerere,
            "files": [f.to_dict() for f in self.files],
            "resolved": [f.to_dict() for f in self.resolved],
            "remaining": [f.to_dict() for f in self.remaining],
        }

    def summary(self) -> str:
        """One human-readable line for service events / messages."""
        if not self.ok:
            return self.message
        if self.up_to_date:
            return f"{self.branch} already contains {self.base_branch}"
        if not self.conflicted:
            return (
                f"merged {self.base_branch} into {self.branch} cleanly "
                f"({self.worktree}); the merge into the base is now a "
                f"fast-forward"
            )
        remaining = self.remaining
        if not remaining:
            return (
                f"the merge of {self.base_branch} into {self.branch} "
                f"conflicted, but no triageable paths remain in "
                f"{self.worktree} — inspect the worktree by hand"
            )
        kinds: dict[str, int] = {}
        for f in remaining:
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
        detail = ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
        return (
            f"{len(remaining)} conflicted file(s) ({detail}) left in "
            f"{self.worktree} on {self.branch}; resolve, commit, then "
            f"`git merge --ff-only {self.branch}` from the base checkout"
        )


# -- git plumbing -------------------------------------------------------------


def _git(*args: str, cwd: Path) -> tuple[int, str]:
    """Run git, return (returncode, combined output). Never raises."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, f"{type(e).__name__}: {e}"
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _git_bytes(*args: str, cwd: Path) -> tuple[int, bytes]:
    """Run git capturing raw bytes (`cat-file blob`). Never raises."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, f"{type(e).__name__}: {e}".encode()
    return proc.returncode, proc.stdout


def in_progress(worktree: Path) -> bool:
    """Is a merge currently half-done in this worktree (MERGE_HEAD set)?"""
    code, _ = _git("rev-parse", "-q", "--verify", "MERGE_HEAD", cwd=worktree)
    return code == 0


def _dirty(worktree: Path) -> bool:
    code, out = _git("status", "--porcelain", "-uno", cwd=worktree)
    return code == 0 and bool(out)


def _current_branch(worktree: Path) -> str | None:
    code, out = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=worktree)
    if code != 0 or out in ("", "HEAD"):
        return None
    return out


def _branch_exists(branch: str, cwd: Path) -> bool:
    code, _ = _git("rev-parse", "--verify", "--quiet", branch, cwd=cwd)
    return code == 0


def is_fast_forward(project_root: Path, branch: str, base_branch: str) -> bool:
    """Would merging `branch` into `base_branch` be a fast-forward (the
    base tip is an ancestor of the branch tip)?"""
    if not _branch_exists(branch, cwd=project_root) or not _branch_exists(
        base_branch, cwd=project_root
    ):
        return False
    code, _ = _git("merge-base", "--is-ancestor", base_branch, branch, cwd=project_root)
    return code == 0


# -- triage -------------------------------------------------------------------


def unmerged_paths(worktree: Path) -> list[str]:
    """Every path with unmerged index entries of the in-progress merge."""
    paths: list[str] = []
    for _, _xy, _sub, _m1, _m2, _m3, _mw, _h1, _h2, _h3, path in _u_records(worktree):
        paths.append(path)
    return paths


def _u_records(
    worktree: Path,
) -> list[tuple[str, str, str, str, str, str, str, str, str, str, str]]:
    """The `u` records of `git status --porcelain=v2 -z`, split into
    (marker, XY, sub, m1, m2, m3, mW, h1, h2, h3, path) — the three stage
    oids and modes come straight from the record, no `ls-files -u`
    round-trip. With -z the path is never quoted; `split(" ", 10)` keeps
    inner spaces on the path, where they belong."""
    code, out = _git("status", "--porcelain=v2", "-z", cwd=worktree)
    if code != 0:
        return []
    records: list[tuple[str, str, str, str, str, str, str, str, str, str, str]] = []
    for record in out.split("\0"):
        if not record.startswith("u "):
            continue
        fields = record.split(" ", 10)
        if len(fields) != 11:
            continue
        records.append(tuple(fields))  # type: ignore[arg-type]
    return records


def _parse_conflict_markers(text: str) -> tuple[int, dict[str, int]] | None:
    """Count conflict hunks in working-tree content and the line totals of
    each marker section (ours / base / theirs — the base section only
    exists in zdiff3/diff3 markers). None when no markers are present."""
    ours = base = theirs = 0
    hunks = 0
    section = ""  # "", "ours", "base", "theirs"
    found = False
    for line in text.splitlines():
        if line.startswith("<<<<<<<"):
            hunks += 1
            found = True
            section = "ours"
            continue
        if line.startswith("|||||||") and section == "ours":
            section = "base"
            continue
        if line.startswith("=======") and section in ("ours", "base"):
            section = "theirs"
            continue
        if line.startswith(">>>>>>>") and section == "theirs":
            section = ""
            continue
        if section == "ours":
            ours += 1
        elif section == "base":
            base += 1
        elif section == "theirs":
            theirs += 1
    if not found:
        return None
    return hunks, {"ours": ours, "base": base, "theirs": theirs}


def _classify(
    entries: list[tuple[str, str, str, str, str, str, str, str]],
) -> list[ConflictFile]:
    """Derive `ConflictFile`s from porcelain v2 unmerged records.

    Each record tuple: (path, XY, h1, h2, h3, m1, m2, m3) — the three
    stage oids and modes of the path.
    """
    files: list[ConflictFile] = []
    for path, xy, h1, h2, h3, m1, m2, m3 in entries:
        ours_oid, theirs_oid, base_oid = (
            h2 if h2 != _ZERO_OID else None,
            h3 if h3 != _ZERO_OID else None,
            h1 if h1 != _ZERO_OID else None,
        )
        ours_mode = m2 if m2 != "000000" else None
        theirs_mode = m3 if m3 != "000000" else None
        base_mode = m1 if m1 != "000000" else None
        has_base, has_ours, has_theirs = (
            base_oid is not None,
            ours_oid is not None,
            theirs_oid is not None,
        )
        kind = _kind_for(
            xy, has_base, has_ours, has_theirs, ours_oid, theirs_oid, base_oid
        )
        files.append(
            ConflictFile(
                path=path,
                kind=kind,
                base=has_base,
                ours=has_ours,
                theirs=has_theirs,
                base_oid=base_oid,
                ours_oid=ours_oid,
                theirs_oid=theirs_oid,
                base_mode=base_mode,
                ours_mode=ours_mode,
                theirs_mode=theirs_mode,
            )
        )
    return _mark_rename_renames(files)


def _kind_for(
    xy: str,
    has_base: bool,
    has_ours: bool,
    has_theirs: bool,
    ours_oid: str | None,
    theirs_oid: str | None,
    base_oid: str | None,
) -> str:
    if xy == "UU":
        return CONTENT
    if xy == "DD":
        return DELETE_DELETE
    if xy in ("DU", "UD"):
        # Exactly one side's blob survives: DU = deleted by us (ours is
        # gone, theirs is there), UD = deleted by them. When the
        # surviving content equals the merge-base content, the surviving
        # side did not edit the file — a pure rename against the other
        # side's deletion (git already folded the rename into the
        # stages). Otherwise one side really edited what the other
        # removed.
        survivor = theirs_oid if xy == "DU" else ours_oid
        if has_base and survivor is not None and survivor == base_oid:
            return RENAME_DELETE
        return MODIFY_DELETE
    if xy in ("AA", "AU", "UA"):
        return ADD_ADD
    # Fallback from the stage pattern itself.
    if has_base and has_ours and has_theirs:
        return CONTENT
    if not has_base and has_ours and has_theirs:
        return ADD_ADD
    if has_ours != has_theirs:
        return MODIFY_DELETE
    return CONTENT


def _mark_rename_renames(files: list[ConflictFile]) -> list[ConflictFile]:
    """Two paths fighting over the SAME blob from opposite sides is a
    rename-rename: side A renamed the file to path1, side B to path2."""
    by_ours = {f.ours_oid: f for f in files if f.ours_oid}
    for f in files:
        if f.theirs_oid and f.ours_oid is None:
            other = by_ours.get(f.theirs_oid)
            if other is not None and other.path != f.path:
                other.kind = f.kind = RENAME_RENAME
    return files


def triage(worktree: Path) -> list[ConflictFile]:
    """Structured triage of the in-progress merge's conflicts.

    Reads `git status --porcelain=v2` unmerged records (with the three
    stage oids/modes per path — no extra `ls-files -u` round-trip), classifies
    each conflict kind and measures the conflict hunks in the working tree.
    """
    entries: list[tuple[str, str, str, str, str, str, str, str]] = []
    for _u, xy, _sub, m1, m2, m3, _mw, h1, h2, h3, path in _u_records(worktree):
        entries.append((path, xy, h1, h2, h3, m1, m2, m3))
    files = _classify(entries)
    for f in files:
        parsed = _hunks_for(worktree, f)
        if parsed is not None:
            f.hunks, f.conflict_lines = parsed
    return files


def _hunks_for(worktree: Path, f: ConflictFile) -> tuple[int, dict[str, int]] | None:
    """Conflict-marker stats of the working-tree file, when parseable."""
    if f.kind not in (CONTENT, ADD_ADD):
        return None
    try:
        text = (worktree / f.path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return _parse_conflict_markers(text)


# -- safe auto-resolvers -------------------------------------------------------


def _materialize(worktree: Path, oid: str, path: str, mode: str | None) -> bool:
    """Write the blob `oid` into the working-tree file (and restore the
    executable bit from `mode`)."""
    code, blob = _git_bytes("cat-file", "blob", oid, cwd=worktree)
    if code != 0:
        return False
    target = worktree / path
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_bytes(blob)
        if mode == "100755":
            target.chmod(0o755)
    except OSError:
        return False
    return True


def resolve_identical(
    worktree: Path, files: list[ConflictFile] | None = None
) -> list[str]:
    """Auto-resolve paths whose BOTH stages are byte-identical (same blob
    oid, same mode) — there is nothing to decide, take it. Returns the
    resolved paths."""
    if files is None:
        files = triage(worktree)
    resolved: list[str] = []
    for f in files:
        if (
            f.ours_oid is not None
            and f.ours_oid == f.theirs_oid
            and f.ours_mode == f.theirs_mode
        ):
            if not _materialize(worktree, f.ours_oid, f.path, f.ours_mode):
                continue
            code, _ = _git("add", "--", f.path, cwd=worktree)
            if code == 0:
                f.resolution = "identical-both-sides"
                resolved.append(f.path)
    return resolved


def _zdiff3_supported() -> bool:
    """Does this git know the zdiff3 conflict style (>= 2.35)? Probed once
    with a real `git merge-file --zdiff3` invocation."""
    global _ZDIFF3_OK
    if _ZDIFF3_OK is None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ours = tmp_path / "o"
            base = tmp_path / "b"
            theirs = tmp_path / "t"
            ours.write_text("a\n")
            base.write_text("a\n")
            theirs.write_text("a\n")
            code, out = _git(
                "merge-file",
                "--zdiff3",
                "-L",
                "ours",
                "-L",
                "base",
                "-L",
                "theirs",
                str(ours),
                str(base),
                str(theirs),
                cwd=tmp_path,
            )
            _ZDIFF3_OK = code != 129 and "unknown option" not in out
    return _ZDIFF3_OK


_ZDIFF3_OK: bool | None = None


def apply_zdiff3_markers(
    worktree: Path, files: list[ConflictFile] | None = None
) -> list[str]:
    """Re-render the conflict markers of a live (possibly older-style)
    merge in the zdiff3 format (`git merge-file --zdiff3`): every conflict
    hunk gains the COMMON ANCESTOR section, and lines both sides changed
    identically are factored out of the hunk. Marker rendering only — it
    never resolves anything, so it is safe to run on any in-progress
    merge. Returns the re-rendered paths."""
    if not _zdiff3_supported():
        return []
    if files is None:
        files = triage(worktree)
    done: list[str] = []
    for f in files:
        if f.kind not in (CONTENT, ADD_ADD) or f.resolution:
            continue
        if not (f.ours_oid and f.theirs_oid):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ours = tmp_path / "ours"
            theirs = tmp_path / "theirs"
            base = tmp_path / "base"
            ok = True
            for oid, target in (
                (f.ours_oid, ours),
                (f.theirs_oid, theirs),
            ):
                code, blob = _git_bytes("cat-file", "blob", oid, cwd=worktree)
                if code != 0:
                    ok = False
                    break
                target.write_bytes(blob)
            if not ok:
                continue
            if f.base_oid:
                code, blob = _git_bytes("cat-file", "blob", f.base_oid, cwd=worktree)
                if code != 0:
                    continue
                base.write_bytes(blob)
            else:
                base.write_bytes(b"")
            labels = [
                "-L",
                f"ours: {f.path}",
                "-L",
                f"merge base: {f.path}",
                "-L",
                f"theirs: {f.path}",
            ]
            code, _out = _git(
                "merge-file",
                "--zdiff3",
                *labels,
                str(ours),
                str(base),
                str(theirs),
                cwd=tmp_path,
            )
            if code < 0:
                continue
            try:
                (worktree / f.path).write_bytes(ours.read_bytes())
            except OSError:
                continue
            done.append(f.path)
    return done


def rerere_status(worktree: Path) -> list[str]:
    """Paths whose conflict shape rerere has recorded for the current
    merge (empty when rerere saw nothing or is disabled)."""
    code, out = _git("rerere", "status", cwd=worktree)
    if code != 0:
        return []
    return [line for line in out.splitlines() if line.strip()]


def _has_markers(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True  # unreadable: do not stage
    return _parse_conflict_markers(text) is not None


def rerere_replay(worktree: Path, files: list[ConflictFile] | None = None) -> list[str]:
    """Replay previously recorded resolutions (`git rerere`) and STAGE
    every cleared file.

    With rerere enabled the merge itself already APPLIES a recorded
    resolution to the working tree (the file content is the replayed one,
    the index entry stays unmerged) — this function runs `git rerere` once
    more for the record-then-apply edge cases and then stages exactly
    those conflicted paths whose working-tree content no longer contains
    conflict markers: content that rerere has resolved. Rerere only ever
    replays a resolution a human/agent has ALREADY approved for the same
    conflict shape, so staging it is safe; the marker check keeps a
    partial/failed replay from being staged (an unresolved conflict ALWAYS
    still carries markers). Returns the staged paths; `files` (the triage
    list) gets `resolution="rerere"` stamped on the cleared entries."""
    if files is None:
        files = triage(worktree)
    _git("rerere", cwd=worktree)  # best-effort: applies pending replays
    staged: list[str] = []
    for f in files:
        if f.kind not in (CONTENT, ADD_ADD) or f.resolution:
            continue
        if _has_markers(worktree / f.path):
            continue
        code, _ = _git("add", "--", f.path, cwd=worktree)
        if code != 0:
            continue
        f.resolution = "rerere"
        staged.append(f.path)
    return staged


def rerere_record(worktree: Path) -> int:
    """Record the CURRENT resolution state into the rerere cache so the
    next occurrence of the same conflict shape replays automatically
    (`git rerere`; the cache lives in $GIT_DIR/rr-cache — shared by every
    worktree of the repository, which is the point: task branches keep
    hitting the same files). Returns how many conflict shapes still wait
    for a resolution (0 once everything is recorded)."""
    code, _ = _git("rerere", cwd=worktree)
    if code != 0:
        return 0
    return len(rerere_status(worktree))


# -- the engine entry point ----------------------------------------------------


def merge_base_into_worktree(
    project_root: Path,
    task_id: str,
    worktree: Path,
    base_branch: str,
    *,
    branch: str | None = None,
    rerere: bool = True,
    save: bool = False,
) -> MergeReport:
    """THE REVERSED MERGE: merge `base_branch` INTO the task's worktree
    (branch `branch`, default: the worktree's checked-out branch) instead
    of merging the task branch into the user's checkout.

    Never touches anything outside the task's worktree — the user's
    checkout is not even read beyond `project_root` (used only to persist
    the report when `save`). On conflict the merge is left half-done in
    the worktree (that is the point): resolve there, then `finish`, then
    the merge into the base is a fast-forward.

    Refusals (report.ok is False, git state untouched): the worktree is
    not a usable git tree, is dirty without a merge in progress (snapshot
    first — see `drcode.task_env.snapshot`), is checked out on a different
    branch than requested, or the base branch does not exist.
    """
    report = MergeReport(
        task_id=task_id,
        branch=branch or "?",
        base_branch=base_branch,
        worktree=str(worktree),
    )
    wt = worktree
    if not wt.is_dir():
        report.message = f"worktree {wt} does not exist"
        return _maybe_save(project_root, task_id, report, save)
    code, out = _git("rev-parse", "--is-inside-work-tree", cwd=wt)
    if code != 0 or out != "true":
        report.message = f"{wt} is not a usable git worktree"
        return _maybe_save(project_root, task_id, report, save)
    current = _current_branch(wt)
    if branch is None:
        branch = current or "?"
        report.branch = branch
    if current != branch:
        report.message = f"{wt} is on {current or '(detached HEAD)'}, expected {branch}"
        return _maybe_save(project_root, task_id, report, save)
    if not _branch_exists(base_branch, cwd=project_root):
        report.message = f"base branch {base_branch} does not exist"
        return _maybe_save(project_root, task_id, report, save)

    resuming = in_progress(wt)
    if not resuming:
        if _dirty(wt):
            report.message = (
                f"the worktree has uncommitted changes; snapshot them onto "
                f"{branch} first (drcode.task_env.snapshot), then retry"
            )
            return _maybe_save(project_root, task_id, report, save)
        # Already up to date? The task branch containing the base tip
        # means there is nothing to merge (and the base→task merge would
        # be a no-op / a fast-forward of the task branch).
        code, _ = _git("merge-base", "--is-ancestor", base_branch, "HEAD", cwd=wt)
        if code == 0:
            report.ok = True
            report.up_to_date = True
            report.fast_forward = True
            report.message = f"{branch} already contains {base_branch}"
            return _maybe_save(project_root, task_id, report, save)

    # The merge itself, with the safe-resolver machinery switched on for
    # THIS invocation only (`-c`, never the repository config):
    # - rerere replays previously recorded resolutions;
    # - zdiff3 markers give every conflict hunk the common-ancestor view.
    args: list[str] = ["merge", "--no-edit", base_branch]
    if _zdiff3_supported():
        args = ["-c", "merge.conflictstyle=zdiff3", *args]
    if rerere:
        args = ["-c", "rerere.enabled=true", "-c", "rerere.autoUpdate=false", *args]
    code, out = _git(*args, cwd=wt)
    if code != 0 and not in_progress(wt):
        # The merge never started (unrelated histories, index.lock, …):
        # report the refusal, git state untouched.
        report.message = f"git merge failed without starting: {out[:1000]}"
        return _maybe_save(project_root, task_id, report, save)

    report.ok = True
    if code == 0:
        report.message = f"merged {base_branch} into {branch} cleanly"
        report.fast_forward = is_fast_forward(project_root, branch, base_branch)
        return _maybe_save(project_root, task_id, report, save)

    # Conflicts: triage, then the provably-safe auto-resolvers. Both
    # resolvers stamp `resolution` onto the entries of `files` in place —
    # that list stays the report's `files` even when the post-resolver
    # index is clean (re-triaging then would lose what was resolved).
    files = triage(wt)
    rerere_staged = rerere_replay(wt, files) if rerere else []
    identical = resolve_identical(wt, files)
    report.files = files
    report.rerere = {
        "recorded_shapes": rerere_status(wt) if rerere else [],
        "replayed": rerere_staged,
    }
    remaining_paths = unmerged_paths(wt)
    report.conflicted = bool(remaining_paths)
    if not report.conflicted and (rerere_staged or identical):
        # The safe resolvers cleared EVERYTHING (all of it pre-approved
        # content: identical blobs or replayed rerere resolutions) —
        # conclude the merge right here, the caller gets a finished
        # branch instead of an empty conflict list.
        ok, _msg = finish(wt)
        if ok:
            report.fast_forward = is_fast_forward(project_root, branch, base_branch)
            report.message = (
                f"merged {base_branch} into {branch}; all "
                f"{len(rerere_staged) + len(identical)} conflict(s) "
                f"auto-resolved (rerere: {len(rerere_staged)}, "
                f"identical: {len(identical)})"
            )
            return _maybe_save(project_root, task_id, report, save)
    report.message = report.summary()
    return _maybe_save(project_root, task_id, report, save)


# -- conclusion -----------------------------------------------------------------


def finish(worktree: Path, message: str | None = None) -> tuple[bool, str]:
    """Commit the resolved merge (all conflicts cleared and staged).
    Returns (ok, message). Refuses while unmerged paths remain."""
    if not in_progress(worktree):
        return False, "no merge is in progress"
    remaining = unmerged_paths(worktree)
    if remaining:
        return False, f"{len(remaining)} conflicted file(s) remain: {remaining[:20]}"
    args = ["commit", "--no-edit"] if message is None else ["commit", "-m", message]
    code, out = _git(*args, cwd=worktree)
    if code != 0:
        return False, out[:1000]
    return True, out[:1000]


def abort(worktree: Path) -> tuple[bool, str]:
    """Roll the worktree back to the pre-merge state (`git merge --abort`)."""
    if not in_progress(worktree):
        return False, "no merge is in progress"
    code, out = _git("merge", "--abort", cwd=worktree)
    return code == 0, out[:1000]


# -- report persistence ----------------------------------------------------------


def report_path(project_root: Path, task_id: str) -> Path:
    """Where the triage report of `task_id` is persisted (the TUI
    ConflictScreen and the agent resolver read it from here)."""
    return project_root / _REPORTS_SUBDIR / f"{task_id}.json"


def save_report(project_root: Path, task_id: str, report: MergeReport) -> Path | None:
    """Persist the report atomically; best-effort (None on failure)."""
    path = report_path(project_root, task_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)
        return path
    except OSError:
        return None


def load_report(project_root: Path, task_id: str) -> MergeReport | None:
    """The persisted report of `task_id`, or None when absent/corrupt."""
    try:
        data = json.loads(
            report_path(project_root, task_id).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    files = [
        ConflictFile.from_dict(f) for f in data.get("files", []) if isinstance(f, dict)
    ]
    rerere = data.get("rerere")
    return MergeReport(
        task_id=str(data.get("task_id", "")),
        branch=str(data.get("branch", "")),
        base_branch=str(data.get("base_branch", "")),
        worktree=str(data.get("worktree", "")),
        ok=bool(data.get("ok")),
        conflicted=bool(data.get("conflicted")),
        up_to_date=bool(data.get("up_to_date")),
        fast_forward=data.get("fast_forward"),  # type: ignore[arg-type]
        message=str(data.get("message", "")),
        files=files,
        rerere=rerere if isinstance(rerere, dict) else {},
        conflict_style=str(data.get("conflict_style", "zdiff3")),
    )


def _maybe_save(
    project_root: Path, task_id: str, report: MergeReport, save: bool
) -> MergeReport:
    if save and report.ok and report.conflicted:
        save_report(project_root, task_id, report)
    return report
