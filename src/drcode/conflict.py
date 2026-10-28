"""Conflict-resolution model for the TUI ConflictScreen (`drcode.conflict`).

The third layer of the conflict-resolving stack, under the algorithmic
merge engine (`drcode.merge_engine`, #d5841a9f) and the agent resolver
(`drcode.merge_resolver`, #59d85021): everything the interactive screen
needs to let a HUMAN resolve the conflicts the machine could not — and
it is deliberately LLM-free and sync, exactly like the two layers below.

The unit of work here is the zdiff3 conflict hunk. The engine's reversed
merge parks conflicts in the task's worktree with `<<<<<<< / ||||||| /
======= / >>>>>>>` markers, where the middle section is the COMMON
ANCESTOR (the merge runs with `merge.conflictstyle=zdiff3`, and
`apply_zdiff3_markers` re-renders older-style files into it). This module
parses that shape into structured hunks and applies the three
user-facing decisions to a hunk:

- ``ours``   — keep the task branch's lines (HEAD of the worktree),
- ``theirs`` — take the base branch's lines (MERGE_HEAD of the merge),
- ``both``   — ours first, theirs right after (the classic "keep both").

Nothing here decides anything by itself: every operation is a direct
mechanical translation of the user's key press. After an edit the file is
staged ONLY when no conflict markers remain in it (the same rule the
engine's rerere replay uses) — a partially resolved file can never sneak
into the merge commit.

For the kinds without content markers (modify-delete and the rename
family) the hunk model does not apply; file-level decisions materialize a
side straight from the triage report's stage blobs (`git cat-file blob`
of the recorded oids — no re-derivation) or record the deletion with
`git rm`.

The commit/abort conclusions are the engine's own `finish`/`abort`, so
the screen can never commit while unmerged paths remain, and everything
it writes lands on the task's branch — the merge into the user's
checkout stays a fast-forward the hook/user performs afterwards (drcode
still never merges on its own).

Sync module (plain file/subprocess calls, mirroring `drcode.task_env` /
`drcode.merge_engine`): callers on the event loop wrap the calls in
`asyncio.to_thread`.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import merge_engine
from .merge_engine import MergeReport
from .tools import child_priority_kwargs

__all__ = [
    "CHOICES",
    "ConflictWorktree",
    "FileState",
    "Hunk",
    "parse_hunks",
]

# The hunk decisions (the o / t / b keys of the screen).
OURS = "ours"
THEIRS = "theirs"
BOTH = "both"
CHOICES = (OURS, THEIRS, BOTH)

# The marker prefixes of a conflict block. zdiff3/diff3 files carry the
# `|||||||` ancestor separator; plain-style conflicts do not (base = ()).
_START = "<<<<<<<"
_SEP = "======="
_BASE = "|||||||"
_END = ">>>>>>>"

# Where the split of a conflict block's sections sits: section content is
# every line between the markers, exclusive.


@dataclass
class Hunk:
    """One conflict block of a file's working-tree content.

    ``start``/``end`` are 0-based line indexes into the file, ``end``
    exclusive — the full marker block, markers included. ``ours``,
    ``base`` and ``theirs`` hold the CONTENT lines of the respective
    sections (markers stripped); ``base`` is empty for plain-style
    conflicts. ``head_labels``/``tail_labels`` carry the marker lines'
    labels (`<<<<<<< ours: app.py`) for display purposes only.
    """

    start: int
    end: int
    ours: list[str] = field(default_factory=list)
    base: list[str] = field(default_factory=list)
    theirs: list[str] = field(default_factory=list)
    head_label: str = ""
    tail_label: str = ""

    def size(self) -> int:
        return self.end - self.start


def parse_hunks(text: str) -> list[Hunk]:
    """Parse every conflict block out of working-tree content.

    Tolerant of style: zdiff3 (ours/base/theirs), diff3 (the same,
    usually an empty base section) and the default two-section style
    (ours/theirs, base empty). Unterminated blocks (a file saved mid-edit)
    yield no hunk — the trailing partial block is garbage, not a decision
    the screen should offer."""

    lines = text.split("\n")
    hunks: list[Hunk] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.startswith(_START):
            i += 1
            continue
        # Scan forward for the end marker; the ======= line must be
        # inside the block (it is a legitimate content line otherwise).
        end = None
        sep = None
        base_sep = None
        for j in range(i + 1, n):
            if lines[j].startswith(_END):
                end = j
                break
            if lines[j].startswith(_BASE) and sep is None:
                base_sep = j
            elif lines[j].startswith(_SEP) and sep is None:
                sep = j
        if end is None or sep is None:
            break  # unterminated garbage tail — stop scanning
        hunk = Hunk(
            start=i,
            end=end + 1,
            ours=lines[i + 1 : (base_sep if base_sep is not None else sep)],
            base=lines[base_sep + 1 : sep] if base_sep is not None else [],
            theirs=lines[sep + 1 : end],
            head_label=line.removeprefix(_START).strip(),
            tail_label=lines[end].removeprefix(_END).strip(),
        )
        hunks.append(hunk)
        i = end + 1
    return hunks


def _apply_choice(hunk: Hunk, choice: str) -> list[str]:
    """The replacement lines for a hunk under the user's decision."""
    if choice == OURS:
        return list(hunk.ours)
    if choice == THEIRS:
        return list(hunk.theirs)
    if choice == BOTH:
        return [*hunk.ours, *hunk.theirs]
    raise ValueError(f"unknown resolution choice: {choice!r}")


@dataclass
class FileState:
    """The live state of one triaged conflict path (the file list row).

    ``unmerged`` — the index still holds unmerged entries for the path;
    ``has_markers`` — the working-tree file still carries conflict
    markers (a resolved-but-undecided file: content staged by an auto-
    resolver, or edited outside and already `git add`-ed, shows
    ``resolved`` here). ``hunks`` is the current marker count of the
    working-tree file (None: no markers / not a content conflict).
    """

    path: str
    kind: str
    unmerged: bool
    has_markers: bool
    hunks: int | None

    @property
    def resolved(self) -> bool:
        return not self.unmerged and not self.has_markers

    def badge(self) -> str:
        if self.resolved:
            return "[green]✓ resolved[/]"
        if self.unmerged and self.hunks:
            return f"[yellow]{self.hunks} hunk(s)[/]"
        if self.unmerged:
            return "[red]untouched[/]"
        return "[yellow]edited, unstaged[/]"


class ConflictWorktree:
    """The parked conflict of one task's worktree, as the screen sees it.

    Wraps the engine's report with the live working-tree state: the
    triage report is the STATIC picture at merge time, this object reads
    the DYNAMIC one (what the user has already decided) on every refresh.
    All methods are sync and blocking; the screen runs them in worker
    threads.
    """

    def __init__(
        self,
        project_root: Path,
        task_id: str,
        worktree: Path,
        report: MergeReport | None = None,
    ) -> None:
        self.project_root = project_root
        self.task_id = task_id
        self.worktree = worktree
        # A report handed in beats the persisted one (the caller may have
        # just re-prepared the merge); otherwise load what the engine saved.
        self.report = report or merge_engine.load_report(project_root, task_id)

    # -- git plumbing -----------------------------------------------------------

    def _git(self, *args: str) -> tuple[int, str]:
        proc = subprocess.run(
            ["git", *args],
            cwd=self.worktree,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            **child_priority_kwargs(),
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()

    # -- live state ---------------------------------------------------------------

    def in_progress(self) -> bool:
        return merge_engine.in_progress(self.worktree)

    def file_states(self) -> list[FileState]:
        """Every triaged path with its CURRENT status (resolved files
        included — the user watches their own progress in the list)."""
        unmerged = set(merge_engine.unmerged_paths(self.worktree))
        kinds = {f.path: f.kind for f in (self.report.files if self.report else [])}
        states: list[FileState] = []
        for path in sorted(set(unmerged) | set(kinds)):
            content = self._read(path)
            parsed = parse_hunks(content) if content is not None else []
            states.append(
                FileState(
                    path=path,
                    kind=kinds.get(path, merge_engine.CONTENT),
                    unmerged=path in unmerged,
                    has_markers=bool(parsed),
                    hunks=len(parsed) or None,
                )
            )
        return states

    def _read(self, path: str) -> str | None:
        try:
            return (self.worktree / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def read_file(self, path: str) -> str | None:
        return self._read(path)

    # -- hunk operations ------------------------------------------------------------

    def apply_hunk(self, path: str, hunk: Hunk, choice: str) -> tuple[bool, str]:
        """Apply the user's decision to one hunk and (when the file is
        then marker-free) stage it. Returns (ok, message)."""
        replacement = _apply_choice(hunk, choice)
        try:
            text = (self.worktree / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return False, f"{type(e).__name__}: {e}"
        lines = text.split("\n")
        if not lines[hunk.start].startswith(_START) or not lines[
            hunk.end - 1
        ].startswith(_END):
            # The file changed under the cursor (an external editor, a
            # rerere replay): the stale hunk must not hit wrong lines.
            return False, "the file changed on disk — refresh (r) and retry"
        new_lines = lines[: hunk.start] + replacement + lines[hunk.end :]
        new_text = "\n".join(new_lines)
        return self._write_and_stage(path, new_text)

    def _write_and_stage(self, path: str, new_text: str) -> tuple[bool, str]:
        target = self.worktree / path
        try:
            target.write_text(new_text, encoding="utf-8")
        except OSError as e:
            return False, f"{type(e).__name__}: {e}"
        parsed = parse_hunks(new_text)
        if not parsed:
            code, out = self._git("add", "--", path)
            if code != 0:
                return False, f"git add failed: {out[:300]}"
            return True, "resolved and staged"
        return True, f"{len(parsed)} hunk(s) left"

    # -- file-level operations (markerless kinds) --------------------------------

    def take_side(self, path: str, choice: str) -> tuple[bool, str]:
        """A whole-file ours/theirs decision for the kinds the hunk model
        does not cover (modify-delete, rename family). ``both`` is not
        defined there (ValueError). The side is materialized from the
        triage report's recorded stage blobs — never re-derived."""
        if choice == BOTH:
            raise ValueError("whole-file 'both' is not defined for this kind")
        entry = next(
            (f for f in (self.report.files if self.report else []) if f.path == path),
            None,
        )
        if entry is None:
            return False, "no triage entry for this path"
        if choice == OURS:
            if entry.ours_oid is None:
                # Ours deleted the file (modify-delete / delete-delete).
                code, out = self._git("rm", "-f", "--", path)
                return (code == 0, out[:300] if code else "ours: file removed")
            return self._materialize(path, entry.ours_oid, entry.ours_mode)
        if entry.theirs_oid is None:
            code, out = self._git("rm", "-f", "--", path)
            return (code == 0, out[:300] if code else "theirs: file removed")
        return self._materialize(path, entry.theirs_oid, entry.theirs_mode)

    def _materialize(self, path: str, oid: str, mode: str | None) -> tuple[bool, str]:
        proc = subprocess.run(
            ["git", "cat-file", "blob", oid],
            cwd=self.worktree,
            capture_output=True,
            timeout=120,
            check=False,
            **child_priority_kwargs(),
        )
        if proc.returncode != 0:
            return False, f"git cat-file failed: {proc.stderr.decode()[:300]}"
        target = self.worktree / path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(proc.stdout)
        except OSError as e:
            return False, f"{type(e).__name__}: {e}"
        code, out = self._git("add", "--", path)
        if code != 0:
            return False, f"git add failed: {out[:300]}"
        return True, "resolved and staged"

    # -- refresh / conclusion ------------------------------------------------------

    def refresh(self) -> None:
        """Re-triage the live merge in place (cheap, keeps resolutions) —
        keeps the report's file list in sync with an externally edited
        worktree without redeploying anything."""
        if not self.in_progress():
            return
        files = merge_engine.triage(self.worktree)
        if self.report is None:
            self.report = MergeReport(
                task_id=self.task_id,
                branch="?",
                base_branch="?",
                worktree=str(self.worktree),
            )
        if files:
            self.report.files = files

    def stage_clean(self, path: str) -> tuple[bool, str]:
        """Stage a path the user resolved outside (an editor run with no
        markers left). Refuses while markers remain."""
        content = self._read(path)
        if content is not None and parse_hunks(content):
            return False, "conflict markers remain — resolve them first"
        code, out = self._git("add", "--", path)
        return (code == 0, out[:300] if code else "staged")

    def commit(self, message: str) -> tuple[bool, str]:
        """Commit the fully resolved merge (`merge_engine.finish`) — the
        engine refuses while unmerged paths remain, which is the last
        line of defense behind the screen's own checks."""
        return merge_engine.finish(self.worktree, message)

    def abort(self) -> tuple[bool, str]:
        """Roll the worktree back to the pre-merge state."""
        return merge_engine.abort(self.worktree)
