"""Read-only keyword search over the project's task archive.

The research agent needs to assess the state of development around a topic —
across ALL statuses (queued, hold, active, testing, completed, clarification)
— without reading the raw task JSONs: a task file is `Task.model_dump_json`
with the full transcript (`messages`, `tester_messages`, `events`), and
reading several of those burns tens of thousands of tokens. The plain file
tools cannot help either: `.drcode/` is in `tools.IGNORED_DIRS`, so
grep/glob never see it, and `run_bash` + `grep`/`jq` is a fragile workaround
(jq may be absent, the JSON is multi-line).

This module gives the model a deliberately COMPACT view instead:

- a light parse — `json.loads` plus a hand-picked subset of keys, NEVER
  `Task.model_validate` (validating giant `messages` lists for a search is
  pure waste);
- indexing the cheap, high-signal fields: the task's short `id` (so a task
  is findable by its id alone — a never-run task with no events would
  otherwise surface only by accident), `title`, every `user-prompt` part
  of the transcript, `pending_input` texts, `edited_files` paths (so a file
  name finds the task that touched it), the agent's FINAL text response
  (the closing summary — often bilingual, bridging a Russian prompt and an
  English title) and the `events` texts (where outcomes like "merge FAILED"
  live). The bulk of model responses and tool results stay unindexed —
  noise and volume, not signal;
- matching is case-insensitive substring with a light stemming heuristic:
  besides the keyword itself, a variant with one common trailing ending
  stripped (ru: -ых/-ена/..., en: -ing/-ed/-s/...) is matched too, so
  "смерженных" also finds "смерженный"/"смерженные" without a real stemmer;
- AND semantics first (every keyword must occur); when NOTHING matches
  AND, a second OR pass runs (any keyword) and is returned under an
  explicit "no exact AND match" preamble — better a ranked fuzzy answer
  than a bare refusal;
- ranking: by the number of keyword occurrences, ties broken newest-first;
- one output line per task: id, status/kind, verification + unmerged
  markers, date, title and a snippet around the most informative keyword
  occurrence (capped); never a transcript excerpt. The mergedness marker
  (branch `drcode/<id>` still holds unmerged commits — via
  `TaskEnvironment.unmerged_ids`, blocking git calls) is computed ONLY for
  short result lists, so a big archive scan stays cheap;
- the caller's own task (`exclude_task_id`) is excluded from the results,
  but when it matches the keywords a one-line notice says so — twice now a
  real investigation's answer lived in the searcher's own transcript;
- resilience follows the `Storage.load_one` contract (`storage.py`): a
  corrupted or unreadable file is silently skipped, never fatal. Races need
  no handling — `Storage.save` is atomic (`os.replace`), the worst case is
  missing one task created mid-scan.

Worktree isolation note: under `task_isolation: worktree` the task files
live in the MAIN checkout's `.drcode/tasks/`, not in the task's worktree —
callers must pass the main project root (the pipeline's `manager.storage.root`),
never the agent's cwd/worktree root.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_STATE_DIRNAME
from .models import TaskKind, TaskStatus, VerificationStatus

# Output budget: the tool exists to SAVE context, so the result is hard-capped.
# No more than MAX_RESULTS lines per call; when the archive holds more
# matches, the first MAX_RESULTS are returned plus a "...and N more" tail
# telling the model to refine the query.
MAX_RESULTS = 50
# Snippet geometry: ±SNIPPET_CONTEXT characters around the chosen keyword
# occurrence, hard-capped at SNIPPET_MAX characters.
SNIPPET_CONTEXT = 60
SNIPPET_MAX = 120
# Mergedness checks are blocking git calls (a couple per task id) — run them
# only when the displayed result list is short enough that the added latency
# is justified. Longer lists silently skip the marker.
MERGEDNESS_MAX_IDS = 10
# Indexing caps for the extended fields: they keep the in-memory haystack
# (and the occurrence scoring) bounded no matter how chatty the agent was.
FINAL_RESPONSE_CAP = 4000
EVENT_TEXT_CAP = 500

# Light stemming: ONE common trailing ending is stripped from a keyword to
# produce an extra match variant (the full keyword is always tried too).
# Deliberately NOT a stemmer — just the frequent tails of Russian and
# English inflection. A variant is kept only when at least 4 characters
# remain, so short words are never mangled.
_STEM_SUFFIXES = (
    # Russian plural/adjectival/case endings first (longest match wins).
    "иями",
    "ениями",
    "ами",
    "ями",
    "ого",
    "его",
    "ому",
    "ему",
    "ыми",
    "ими",
    "енных",
    "ых",
    "их",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ена",
    "ено",
    "ены",
    "ой",
    "ей",
    "ый",
    "ий",
    "ом",
    "ем",
    "ах",
    "ях",
    "ов",
    "ев",
    "ну",
    "на",
    "но",
    "ны",
    "ть",
    "ет",
    "ут",
    "ют",
    "ат",
    "ят",
    "ла",
    "ло",
    "ли",
    # English inflection.
    "ing",
    "ed",
    "es",
    "s",
)

# Short display forms for the verification label on a result line.
_VERIFICATION_SHORT = {
    VerificationStatus.PASSED.value: "passed",
    VerificationStatus.FAILED.value: "failed",
}


def _validate_choice(value: str | None, choices: type, name: str) -> str | None:
    """Validate an optional status/kind filter against its enum.

    Raises ValueError (→ ModelRetry via _run_tool) for an unknown value so
    the model can correct itself instead of silently searching unfiltered.
    """
    if not value:
        return None
    allowed = [member.value for member in choices]  # type: ignore[attr-defined]
    if value not in allowed:
        raise ValueError(
            f"Unknown {name} filter {value!r}; allowed values: "
            + ", ".join(allowed)
            + " (omit the argument to search all of them)."
        )
    return value


def _prompt_texts(messages: list[dict[str, Any]]) -> list[str]:
    """The user-authored texts of a serialized transcript.

    Serialized `ModelRequest`s carry `parts`, and only the parts with
    `part_kind == "user-prompt"` are indexed. Content is usually a plain
    string; multimodal requests carry a list of content items (strings or
    {"text": ...} dicts) — their text is joined. Model responses, tool calls
    and tool results are deliberately ignored (the agent's FINAL response is
    indexed separately — see `_final_response_text`).
    """
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        for part in message.get("parts") or []:
            if not isinstance(part, dict):
                continue
            if part.get("part_kind") != "user-prompt":
                continue
            content = part.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        texts.append(item["text"])
    return texts


def _final_response_text(messages: list[dict[str, Any]]) -> str:
    """The agent's closing summary: the LAST text part of the LAST model
    response in the transcript (the final answer, not the tool chatter).

    Serialized `ModelResponse`s carry `parts` with `part_kind == "text"`;
    tool calls / tool returns use other part kinds and are skipped. Returns
    "" when the transcript has no response text at all.
    """
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        text_parts = [
            part.get("content")
            for part in message.get("parts") or []
            if isinstance(part, dict) and part.get("part_kind") == "text"
        ]
        if not text_parts:
            continue
        content = text_parts[-1]
        if isinstance(content, str):
            return content[:FINAL_RESPONSE_CAP]
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            if chunks:
                return " ".join(chunks)[:FINAL_RESPONSE_CAP]
    return ""


def _pending_texts(pending_input: list[Any]) -> list[str]:
    """`pending_input` texts (queued user messages not yet consumed).

    Tolerates both shapes the storage can hold: current `PendingInput`
    dicts and the legacy plain-string entries (see Task's migration).
    """
    texts: list[str] = []
    for entry in pending_input:
        if isinstance(entry, str):
            texts.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("text"), str):
            texts.append(entry["text"])
    return texts


def _indexed_fields(task_id: str, data: dict[str, Any]) -> list[tuple[str, str]]:
    """The (label, text) pairs that make up a task's search index.

    Labels are cheap provenance tags used to decorate snippets ("files:",
    "summary:", "events:") so the model can tell WHERE a match came from.
    The id comes first as its own field ("id") so the task is findable by
    its short id alone — a never-run task without events would otherwise
    be reachable only by accident. It is deliberately NOT snippet-decorated
    (the id is already shown in `#{task_id}` on every result line).
    """
    fields: list[tuple[str, str]] = [
        ("id", task_id),
        ("title", str(data.get("title") or "")),
    ]
    for text in _prompt_texts(data.get("messages") or []):
        fields.append(("prompt", text))
    for text in _pending_texts(data.get("pending_input") or []):
        fields.append(("prompt", text))
    for path in data.get("edited_files") or []:
        if isinstance(path, str):
            fields.append(("files", path))
    final = _final_response_text(data.get("messages") or [])
    if final:
        fields.append(("summary", final))
    for event in data.get("events") or []:
        if isinstance(event, dict) and isinstance(event.get("text"), str):
            fields.append(("events", event["text"][:EVENT_TEXT_CAP]))
    return fields


def _variants(keyword: str) -> list[str]:
    """Match variants for one keyword: itself plus, when it ends in a known
    inflection tail, the stem with that tail stripped (at most ONE strip)."""
    for suffix in _STEM_SUFFIXES:
        if keyword.endswith(suffix) and len(keyword) - len(suffix) >= 4:
            return [keyword, keyword[: -len(suffix)]]
    return [keyword]


def _count_matches(hay: str, variants: list[str]) -> int:
    """Total case-insensitive occurrences of any variant in `hay`."""
    return sum(hay.count(v) for v in variants)


def _build_snippet(
    fields: list[tuple[str, str]], keyword_variants: list[list[str]]
) -> str:
    """A snippet around the most informative keyword occurrence.

    Every keyword's first occurrence in every field is a candidate; the
    winner is the one whose window holds the MOST distinct keywords (so a
    spot where title- and prompt-keywords coexist beats a lone match), then
    the earliest field / position. Non-prompt fields get a provenance prefix.
    """
    best_candidate: tuple[int, int, int] | None = None
    chosen: tuple[str, str, int, int] | None = None  # (label, flat, start, end)
    for field_index, (label, text) in enumerate(fields):
        flat = " ".join(text.split())
        lower = flat.lower()
        for keyword in keyword_variants:
            pos = -1
            for variant in keyword:
                pos = lower.find(variant)
                if pos >= 0:
                    break
            if pos < 0:
                continue
            start = max(0, pos - SNIPPET_CONTEXT)
            end = min(len(flat), pos + len(keyword[0]) + SNIPPET_CONTEXT)
            window = lower[start:end]
            distinct = sum(
                1 for group in keyword_variants if any(v in window for v in group)
            )
            # Rank: most distinct keywords, then earliest field, then the
            # longest matched keyword (full wordforms beat stems), then the
            # earliest position.
            key = (distinct, -field_index, len(keyword[0]))
            if best_candidate is None or key > best_candidate:
                best_candidate = key
                chosen = (label, flat, start, end)
    if chosen is None:
        return ""
    label, flat, start, end = chosen
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(flat) else ""
    body = (prefix + flat[start:end] + suffix)[:SNIPPET_MAX]
    if label not in ("title", "prompt", "id"):
        body = f"{label}: {body}"
    return body


def _date(data: dict[str, Any]) -> str:
    """The task's most relevant date: completion when it has one, else
    creation — as a bare ISO date (`YYYY-MM-DD`)."""
    stamp = data.get("completed_at") or data.get("created_at") or ""
    return str(stamp)[:10]


def _recency(data: dict[str, Any]) -> float:
    """Timestamp used for the newest-first tie-break (0 when unparsable)."""
    stamp = data.get("completed_at") or data.get("created_at") or ""
    try:
        return datetime.fromisoformat(str(stamp)).timestamp()
    except ValueError:
        return 0.0


def _verification_marker(data: dict[str, Any]) -> str:
    """Short verification label for a result line ("" when unverified)."""
    value = data.get("verification")
    if not isinstance(value, str):
        return ""
    return _VERIFICATION_SHORT.get(value, value)


def _unmerged_ids(project_root: Path, task_ids: list[str]) -> set[str]:
    """Which of `task_ids` still have unmerged commits on `drcode/<id>`.

    Reuses `TaskEnvironment.unmerged_ids` with isolation FORCED ON: the
    mergedness question is about branch existence, not about the current
    setting (a user may have toggled isolation off after the work landed on
    its branch). Every git call here is blocking, so the caller invokes this
    only for short result lists; any failure degrades to "no markers".
    """
    if not task_ids:
        return set()
    try:
        from .settings import ProjectSettings, TaskIsolation
        from .task_env import TaskEnvironment

        env = TaskEnvironment(
            project_root, ProjectSettings(task_isolation=TaskIsolation.WORKTREE)
        )
        return env.unmerged_ids(task_ids)
    except (OSError, subprocess.SubprocessError):
        # No git, a repo so broken it cannot answer, a timeout — the marker
        # is cosmetic, silence is the right degradation.
        return set()


def _stem_hint(keywords: list[list[str]]) -> str:
    """A "try a shorter stem" suggestion for the no-result message, or ""
    when nothing can be shortened further.

    The suggestion goes ONE step beyond what `_variants` already tried —
    the one-strip variants were matched by the search itself, so they must
    not be offered back as advice.
    """
    for variants in keywords:
        base = variants[-1]
        if len(base) >= 7:
            shorter = base[: max(4, len(base) - 2)]
            return (
                f' Also try a shorter stem, e.g. "{shorter}" '
                f'instead of "{variants[0]}".'
            )
    return ""


@dataclass
class TaskHit:
    """One search result as structured data (the UI-facing view).

    `search()` formats these into the agent's one-line strings; the TUI's
    search dialog renders the same fields as list rows. `unmerged` is left
    None by the matching core — checking it needs BLOCKING git calls, so
    the caller decides when it is worth it (`annotate_unmerged`, short
    result lists only).
    """

    task_id: str
    title: str
    status: str
    kind: str
    # Start of the INITIAL user input: the first user-prompt of the
    # transcript, or — for a task never run (empty/no-prompt transcript) —
    # the first queued pending_input text (TaskManager.create puts the
    # first message exactly there).
    first_prompt: str
    # Snippet around the most informative keyword occurrence
    # (_build_snippet; "" when no keyword occurrence survived flattening).
    snippet: str
    # Relevance: total keyword occurrences (ties broken newest-first).
    score: int
    # Completion date when present, else creation (YYYY-MM-DD).
    date: str
    # Short verification label ("passed"/"failed"), "" when unverified.
    verification: str
    # Error markers (the "something went wrong" combination): an
    # interrupted attempt, a FAILED verdict, a failed workflow-hook merge,
    # and the text of the last error-looking service event.
    unfinished: bool
    hook_merge_failed: bool
    last_error_event: str
    unmerged: bool | None = None


def _first_prompt(data: dict[str, Any]) -> str:
    """The beginning of the task's initial user input (see TaskHit)."""
    prompts = _prompt_texts(data.get("messages") or [])
    if prompts:
        text = prompts[0]
    else:
        pending = _pending_texts(data.get("pending_input") or [])
        text = pending[0] if pending else ""
    return " ".join(text.split())


# Substrings (lowercase) that mark a service event as error-looking. Events
# carry no dedicated error kind — failures are logged with the default
# "system"/"testing"/"workflow" kinds and human texts ("tester failed: …",
# "merge FAILED: …", "testing failed — returned …"), so the marker is a
# deliberately coarse keyword filter over the event texts.
_ERROR_EVENT_HINTS = (
    "fail",
    "error",
    "ошибк",
    "прерван",
    "не удал",
    "aborted",
)


def _last_error_event(data: dict[str, Any]) -> str:
    """The (flattened) text of the LAST error-looking service event, or ""
    when the log holds none."""
    for event in reversed(data.get("events") or []):
        if not isinstance(event, dict):
            continue
        text = event.get("text")
        if not isinstance(text, str):
            continue
        low = text.lower()
        if any(hint in low for hint in _ERROR_EVENT_HINTS):
            return " ".join(text.split())
    return ""


def first_prompt_snippet(text: str, cap: int = SNIPPET_MAX) -> str:
    """A display snippet of a first prompt: flattened and hard-capped."""
    flat = " ".join(text.split())
    return flat if len(flat) <= cap else flat[:cap] + "…"


def load_archive(
    project_root: Path,
) -> list[tuple[str, dict[str, Any], list[tuple[str, str]]]]:
    """Read the whole task archive into memory.

    Returns (task_id, raw task dict, indexed fields) for every readable
    task file; corrupted/unreadable files are silently skipped (the
    load_one contract). The id comes from the file stem — the filename is
    the archive's identity, `data["id"]` is only a copy of it. BLOCKING
    file I/O: callers on an event loop must run this in a worker thread
    (the TUI dialog reads it once per session and filters in memory).
    """
    tasks_dir = Path(project_root) / PROJECT_STATE_DIRNAME / "tasks"
    try:
        paths = sorted(tasks_dir.glob("*.json"))
    except OSError:
        return []
    records: list[tuple[str, dict[str, Any], list[tuple[str, str]]]] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue  # Corrupted/unreadable: skip silently (load_one contract).
        if not isinstance(data, dict):
            continue
        records.append((path.stem, data, _indexed_fields(path.stem, data)))
    return records


def _match_records(
    records: list[tuple[str, dict[str, Any], list[tuple[str, str]]]],
    variants_per_keyword: list[list[str]],
    *,
    status: str | None = None,
    kind: str | None = None,
) -> tuple[
    list[tuple[str, dict[str, Any], list[tuple[str, str]], int]],
    list[tuple[str, dict[str, Any], list[tuple[str, str]], int]],
]:
    """Match preloaded archive records against the keyword variants.

    Returns (and_matches, or_matches) — a record lands in the AND list when
    EVERY keyword occurs, otherwise in the OR list when ANY does. Pure
    in-memory computation (no I/O), cheap enough to re-run per keystroke.
    """
    and_matches: list[tuple[str, dict[str, Any], list[tuple[str, str]], int]] = []
    or_matches: list[tuple[str, dict[str, Any], list[tuple[str, str]], int]] = []
    for task_id, data, fields in records:
        if status is not None and data.get("status") != status:
            continue
        if kind is not None and data.get("kind") != kind:
            continue
        haystack = "\n".join(text for _, text in fields).lower()
        scores = [_count_matches(haystack, vs) for vs in variants_per_keyword]
        if all(score > 0 for score in scores):
            and_matches.append((task_id, data, fields, sum(scores)))
        elif any(score > 0 for score in scores):
            or_matches.append((task_id, data, fields, sum(scores)))
    return and_matches, or_matches


def _prepare(
    project_root: Path,
    keywords: list[str],
    *,
    status: str | None = None,
    kind: str | None = None,
) -> tuple[
    list[str],
    list[list[str]],
    list[tuple[str, dict[str, Any], list[tuple[str, str]], int]],
    list[tuple[str, dict[str, Any], list[tuple[str, str]], int]],
]:
    """Shared front half of the archive-loading entry points: validate the
    keywords and filters, build the stem variants, load the archive and
    match it.

    Returns (cleaned keywords, variants, and_matches, or_matches) — the
    variants are needed downstream to build snippets.
    """
    return _match_prepared(
        load_archive(project_root), keywords, status=status, kind=kind
    )


def _hits_from(
    matches: list[tuple[str, dict[str, Any], list[tuple[str, str]], int]],
    variants_per_keyword: list[list[str]],
    limit: int,
) -> list[TaskHit]:
    """Rank matches (occurrences first, ties newest-first), cap to `limit`
    and shape them into TaskHits (with `unmerged` left to the caller)."""
    matches.sort(key=lambda m: (-m[3], -_recency(m[1])))
    hits: list[TaskHit] = []
    for task_id, data, fields, score in matches[:limit]:
        hits.append(
            TaskHit(
                task_id=task_id,
                title=" ".join(str(data.get("title") or "").split()),
                status=str(data.get("status") or ""),
                kind=str(data.get("kind") or ""),
                first_prompt=_first_prompt(data),
                snippet=_build_snippet(fields, variants_per_keyword),
                score=score,
                date=_date(data),
                verification=_verification_marker(data),
                unfinished=bool(data.get("unfinished")),
                hook_merge_failed=bool(data.get("hook_merge_failed")),
                last_error_event=_last_error_event(data),
            )
        )
    return hits


def annotate_unmerged(project_root: Path, hits: list[TaskHit]) -> None:
    """Fill `hit.unmerged` in place for COMPLETED hits with unmerged commits
    still on their `drcode/<id>` branches.

    BLOCKING git calls (TaskEnvironment.unmerged_ids) — run only for short
    result lists and OFF the event loop; any failure degrades to "no
    markers" (hits keep unmerged=None).
    """
    completed = [
        hit.task_id for hit in hits if hit.status == TaskStatus.COMPLETED.value
    ]
    if not 0 < len(completed) <= MERGEDNESS_MAX_IDS:
        return
    unmerged = _unmerged_ids(Path(project_root), completed)
    for hit in hits:
        if hit.task_id in unmerged:
            hit.unmerged = True


def search_records(
    records: list[tuple[str, dict[str, Any], list[tuple[str, str]]]],
    keywords: list[str],
    *,
    status: str | None = None,
    kind: str | None = None,
    limit: int = MAX_RESULTS,
) -> list[TaskHit]:
    """Match keywords against PRELOADED archive records (`load_archive`).

    Same matching/ranking as `search_tasks`, but without any file I/O —
    the TUI search dialog loads the archive once per session and calls
    this per keystroke. Raises ValueError on empty keywords; `unmerged`
    stays None (the caller annotates when it can afford the git calls).
    """
    _cleaned, variants, and_matches, or_matches = _match_prepared(
        records, keywords, status=status, kind=kind
    )
    matches = and_matches if and_matches else or_matches
    return _hits_from(matches, variants, limit)


def _match_prepared(
    records: list[tuple[str, dict[str, Any], list[tuple[str, str]]]],
    keywords: list[str],
    *,
    status: str | None = None,
    kind: str | None = None,
) -> tuple[
    list[str],
    list[list[str]],
    list[tuple[str, dict[str, Any], list[tuple[str, str]], int]],
    list[tuple[str, dict[str, Any], list[tuple[str, str]], int]],
]:
    """`_prepare` over preloaded records (no archive reading) — shared by
    `search_records` and the archive-loading variants below."""
    cleaned = [k.strip().lower() for k in keywords if k and k.strip()]
    if not cleaned:
        raise ValueError(
            "search_task_history needs at least one non-empty keyword; "
            "pass the words you expect in a task's title, its user prompt, "
            "edited file paths or the agent's final summary."
        )
    status = _validate_choice(status, TaskStatus, "status")
    kind = _validate_choice(kind, TaskKind, "kind")
    variants_per_keyword = [_variants(k) for k in cleaned]
    and_matches, or_matches = _match_records(
        records, variants_per_keyword, status=status, kind=kind
    )
    return cleaned, variants_per_keyword, and_matches, or_matches


def search_tasks(
    project_root: Path,
    keywords: list[str],
    *,
    status: str | None = None,
    kind: str | None = None,
    limit: int = MAX_RESULTS,
) -> list[TaskHit]:
    """Search the project's task archive by keywords, as structured hits.

    Same matching, ranking and filters as `search` (the agent tool), but
    returns TaskHit records instead of formatted lines — the building block
    for UI surfaces. `unmerged` is NOT filled here: the marker needs
    blocking git calls, so callers opt in via `annotate_unmerged` (short
    lists, off the event loop). Raises ValueError on empty keywords or an
    unknown status/kind filter.
    """
    _cleaned, variants, and_matches, or_matches = _prepare(
        project_root, keywords, status=status, kind=kind
    )
    matches = and_matches if and_matches else or_matches
    return _hits_from(matches, variants, limit)


def search(
    project_root: Path,
    keywords: list[str],
    *,
    exclude_task_id: str | None = None,
    status: str | None = None,
    kind: str | None = None,
    limit: int = MAX_RESULTS,
) -> str:
    """Search the project's task archive by keywords (all statuses).

    Matching is a case-insensitive substring over each task's id, title,
    user prompts, pending input, edited file paths, final agent response
    and event texts; besides each keyword a lightly-stemmed variant (one
    common trailing ending stripped) is matched too. Multiple keywords are
    combined
    with AND (every keyword must occur); when NOTHING matches the AND, an
    OR pass runs and its results are returned under an explicit "no exact
    AND match" preamble. Results are ranked by keyword occurrences (ties:
    newest first). Optional filters: `status` (queued/hold/active/testing/
    completed/clarification) and `kind` (bug/feature/research). Returns at
    most `limit` (default MAX_RESULTS = 50) one-line results — id,
    status/kind, verification and unmerged-branch markers (the latter only
    for short result lists: blocking git calls), date, title and a snippet —
    truncated with a "refine the query" tail when longer. Corrupted task
    files are skipped silently. The caller's own task (`exclude_task_id`) is
    excluded from the results, but a one-line notice reports when it matches.
    """
    cleaned, variants, and_matches, or_matches = _prepare(
        project_root, keywords, status=status, kind=kind
    )
    # The exclude_task_id/notice logic is the agent tool's own concern: a
    # task id can be in the AND list only when it matched EVERY keyword and
    # in the OR list only when it matched SOME-but-not-all — exactly the
    # own_matched_and / own_matched_or distinction.
    own_matched_and = exclude_task_id is not None and any(
        task_id == exclude_task_id for task_id, *_ in and_matches
    )
    own_matched_or = exclude_task_id is not None and any(
        task_id == exclude_task_id for task_id, *_ in or_matches
    )
    if exclude_task_id is not None:
        and_matches = [m for m in and_matches if m[0] != exclude_task_id]
        or_matches = [m for m in or_matches if m[0] != exclude_task_id]

    fallback = not and_matches
    matches = and_matches if and_matches else or_matches
    lines: list[str] = []
    total = len(matches)
    if fallback and total:
        lines.append(
            "No exact AND match — showing OR matches (any keyword occurs), "
            "ranked by occurrences:"
        )
    hits = _hits_from(matches, variants, limit)
    # Mergedness needs blocking git calls — only worth it for short outputs.
    annotate_unmerged(project_root, hits)
    for hit in hits:
        title = hit.title
        # A long title is as much a context hazard as a long snippet —
        # cap it to keep every line within the compact-output budget.
        if len(title) > SNIPPET_MAX:
            title = title[:SNIPPET_MAX] + "…"
        markers = [m for m in (hit.verification,) if m]
        if hit.status == TaskStatus.COMPLETED.value and hit.unmerged:
            markers.append("unmerged")
        marker_text = (" " + " ".join(markers)) if markers else ""
        lines.append(
            f"#{hit.task_id} [{hit.status}/{hit.kind}]"
            f"{marker_text} {hit.date} {title} — {hit.snippet}"
        )
    if total > len(hits):
        lines.append(f"...and {total - len(hits)} more — refine the query.")
    if fallback and not hits:
        # No AND match, no OR match either — say so with guidance; the
        # own-task notice (below) may still point at the searcher's own
        # transcript.
        lines.append(
            f"No tasks matched: {', '.join(cleaned)}"
            + (f" (status={status})" if status else "")
            + (f" (kind={kind})" if kind else "")
            + f".{_stem_hint(variants)}"
            " Try fewer or shorter keywords, drop the status/kind filters, "
            "or search for wording from a file path or the agent's summary."
        )
    if own_matched_and or (fallback and own_matched_or):
        lines.append(
            f"Note: the current task {exclude_task_id} also matches these "
            "keywords — read its transcript directly "
            f"(.drcode/tasks/{exclude_task_id}.json)."
        )
    return "\n".join(lines)
