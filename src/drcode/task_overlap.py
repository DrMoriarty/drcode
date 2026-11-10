"""File-overlap warning at task creation (the #260aa4b8 incident).

A research/agent run can file several feature/bug reports off one
incident; when the descriptions all mention the same files, the tasks
race in parallel and their branches collide at the merge — the
conflict-resolver rounds then cost more than the development itself.
Noticing the skew after the tasks are ACTIVE is too late:
`set_task_dependencies` only edits QUEUED/HOLD/DRAFT tasks, and the
queue promotes a freshly created report within seconds.

So the creation paths (the agents' create_bug_report /
create_feature_request and the headless CLI) run this mechanical check:
path-like tokens are extracted from the new task's description and
compared against the descriptions (and the tracked `edited_files`) of
every other unfinished task. A non-empty overlap is a WARNING, never a
block — the task is always created — but the tool result carries the
warning text so the agent can still call `depends_on` at the very
moment of the tool call (a service event alone would be read too
late, if at all).

Deliberately LLM-free: a fixed regex for path-like tokens and a fixed
matching rule, exactly like the other mechanical guards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Task, TaskStatus

# --- path-token extraction ---------------------------------------------------
#
# Two token shapes:
#   * a slashed path (`src/drcode/core/pipeline_resolver.py`,
#     `tests/test_x.py`) — any run of path characters with at least one `/`;
#   * a bare `name.ext` (`merge_resolver.py`, `README.md`) — the extension
#     must START WITH A LETTER, which keeps version numbers ("1.2.3",
#     "v2.0"), decimals ("3.5") and bare line numbers out.
# An optional `:123` / `:12-34` line suffix is consumed but stripped —
# `file.py:123` in one description must match `file.py` in another.
_PATH_TOKEN_RE = re.compile(
    r"(?<![\w-])"  # never the tail of a longer word (a leading `/` is fine —
    r"("  # absolute paths must match too, so it is NOT in the lookbehind)
    r"(?:[\w.-]+/)+[\w.-]+"  # contains at least one `/`
    r"|"
    r"[\w-]+\.[A-Za-z][\w]{0,9}"  # bare name.ext, letter-first extension
    r")"
    r"(?::\d+(?:-\d+)?)?"  # optional line / line-range suffix, dropped
)

# Tokens shorter than this carry no signal ("e.g", "i.e" and other dotted
# abbreviations fall under the limit together with 1-char base names).
_MIN_TOKEN_LEN = 4

# Abbreviations that survive the length limit but are not paths.
_ABBREVIATIONS = frozenset({"e.g.", "i.e.", "etc.", "vs.", "cf."})


def normalize_path_token(raw: str) -> str:
    """The comparison/display form of a path token: backslashes to
    slashes, leading `/` (absolute path), a Windows drive prefix (`c:`)
    and `./` dropped, trailing slash and dots stripped, lowercased. What
    remains is the path SUFFIX `paths_overlap` relates across
    differently-quoted mentions of the same file."""
    token = raw.replace("\\", "/").strip().lower().lstrip("/")
    token = re.sub(r"^[a-z]:", "", token).lstrip("/")
    while token.startswith("./"):
        token = token[2:]
    return token.rstrip("/").rstrip(".")


def extract_path_tokens(text: str) -> list[str]:
    """The normalized, deduplicated path-like tokens of a text, in first
    occurrence order. Windows backslash paths are folded to slashes
    first, so `C:\\Users\\foo\\bar.py` yields the full segment chain
    (minus the drive letter's colon noise — see normalize)."""
    seen: dict[str, None] = {}
    for match in _PATH_TOKEN_RE.finditer(text.replace("\\", "/")):
        token = normalize_path_token(match.group(1))
        if len(token) < _MIN_TOKEN_LEN:
            continue
        if token in _ABBREVIATIONS or token.rstrip(".") in _ABBREVIATIONS:
            continue
        seen.setdefault(token, None)
    return list(seen)


def paths_overlap(a: str, b: str) -> bool:
    """Whether two normalized paths denote the same file: either equal
    or one a path-suffix of the other (`core/pipeline_resolver.py`
    matches `src/drcode/core/pipeline_resolver.py` — descriptions quote
    the same file with different amounts of leading context)."""
    if a == b:
        return True
    return a.endswith("/" + b) or b.endswith("/" + a)


# --- the scan -----------------------------------------------------------------

# The overlap report is bounded so a service event never bloats the chat.
MAX_WARNED_TASKS = 3
MAX_WARNED_FILES = 5


@dataclass
class TaskFileOverlap:
    """One unfinished task whose description (or tracked edits) shares
    path tokens with the new task's description."""

    task_id: str
    files: list[str] = field(default_factory=list)


def _task_description_texts(task: Task) -> list[str]:
    """The description-side texts of an existing task, WITHOUT touching
    the lazy transcripts: the title, whatever is still queued in
    `pending_input`, and the skeleton's stored user prompts (the prompt
    of an ACTIVE task is consumed out of pending_input — the cheap
    metadata-side skeleton is where it stays readable)."""
    texts = [task.title]
    texts.extend(entry.text for entry in task.pending_input)
    prompts = task.skeleton.get("prompts")
    if isinstance(prompts, list):
        texts.extend(p for p in prompts if isinstance(p, str))
    return texts


def find_file_overlaps(
    tasks: dict[str, Task],
    description: str,
    *,
    exclude_id: str,
    depends_on: list[str] | None = None,
) -> list[TaskFileOverlap]:
    """Unfinished tasks (any status but COMPLETED) whose description or
    tracked `edited_files` share path tokens with `description`.

    The new task never matches itself (`exclude_id`), tasks it already
    declared as `depends_on` are skipped (a declared dependency is
    exactly the mitigation this scan recommends), and the result is
    ordered deterministically: most overlapping files first, then the
    task id."""
    new_tokens = extract_path_tokens(description)
    if not new_tokens:
        return []
    excluded = {exclude_id, *(depends_on or [])}
    overlaps: list[TaskFileOverlap] = []
    for task_id, task in tasks.items():
        if task_id in excluded or task.status is TaskStatus.COMPLETED:
            continue
        existing = set(extract_path_tokens(" \n".join(_task_description_texts(task))))
        existing.update(normalize_path_token(p) for p in task.edited_files)
        files = sorted(
            {
                new_token
                for new_token in new_tokens
                if any(paths_overlap(new_token, old) for old in existing)
            }
        )
        if files:
            overlaps.append(TaskFileOverlap(task_id=task_id, files=files))
    overlaps.sort(key=lambda o: (-len(o.files), o.task_id))
    return overlaps[:MAX_WARNED_TASKS]


def _capped_files(files: list[str]) -> list[str]:
    shown = files[:MAX_WARNED_FILES]
    if len(files) > MAX_WARNED_FILES:
        shown.append(f"… (+{len(files) - MAX_WARNED_FILES} more)")
    return shown


def build_overlap_warning(overlaps: list[TaskFileOverlap]) -> str:
    """The one-line warning appended to the create_* tool result — the
    text the creating agent sees IMMEDIATELY, while the depends_on
    parameter of the very next call is still an option."""
    ids = ", ".join(f"#{o.task_id}" for o in overlaps)
    files = ", ".join(_capped_files(sorted({f for o in overlaps for f in o.files})))
    return (
        f"Warning: files {files} also appear in unfinished "
        f"{'task' if len(overlaps) == 1 else 'tasks'} {ids} — consider "
        "depends_on to serialize the merge."
    )


def build_overlap_event_text(overlaps: list[TaskFileOverlap]) -> str:
    """The human-facing TaskEvent text: a per-task breakdown of the
    overlap (bounded to MAX_WARNED_TASKS / MAX_WARNED_FILES) and the
    dependency advice."""
    lines = [
        (
            "Warning: the description of this task overlaps by files with "
            "unfinished task(s) — they may conflict at the merge if run in "
            "parallel:"
        ),
    ]
    for overlap in overlaps:
        files = ", ".join(_capped_files(overlap.files))
        lines.append(f"- #{overlap.task_id}: {files}")
    lines.append(
        "Consider depends_on (or set_task_dependencies while the task is "
        "still queued) to serialize the merge."
    )
    return "\n".join(lines)
