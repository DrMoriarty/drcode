"""Tests for the research agent's task-archive search (drcode.task_search)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from drcode import task_search
from drcode.models import Task, TaskKind, TaskStatus, VerificationStatus


def _write_task(root: Path, task: Task) -> None:
    path = root / ".drcode" / "tasks" / f"{task.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(task.model_dump_json(indent=2))


def _task(**kwargs) -> Task:
    defaults = {"title": "t", "kind": TaskKind.FEATURE}
    defaults.update(kwargs)
    return Task(**defaults)


def test_search_matches_title_and_user_prompt(project_root: Path) -> None:
    by_title = _task(
        title="Fix login timeout",
        status=TaskStatus.COMPLETED,
        completed_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    by_prompt = _task(
        title="unrelated",
        messages=[
            {"parts": [{"part_kind": "user-prompt", "content": "Add login retry"}]}
        ],
    )
    no_match = _task(title="something else")
    for t in (by_title, by_prompt, no_match):
        _write_task(project_root, t)
    result = task_search.search(project_root, ["login"])
    assert f"#{by_title.id}" in result
    assert f"#{by_prompt.id}" in result
    assert f"#{no_match.id}" not in result
    # The completed task's line shows its completion date.
    assert "2024-01-02" in result


def test_search_is_case_insensitive_and_ands_keywords(project_root: Path) -> None:
    both = _task(title="Retry BACKOFF policy", status=TaskStatus.QUEUED)
    one = _task(title="Retry logic only", status=TaskStatus.QUEUED)
    other = _task(title="backoff handling", status=TaskStatus.QUEUED)
    for t in (both, one, other):
        _write_task(project_root, t)
    # AND: every keyword must occur; case does not matter on either side.
    result = task_search.search(project_root, ["retry", "BACKOFF"])
    assert f"#{both.id}" in result
    assert f"#{one.id}" not in result
    assert f"#{other.id}" not in result
    result = task_search.search(project_root, ["RETRY", "backoff"])
    assert f"#{both.id}" in result


def test_search_indexes_pending_input_but_not_old_model_parts(
    project_root: Path,
) -> None:
    # The keyword lives in queued pending input AND in an OLDER model
    # response; only the pending input must match — model chatter is not
    # indexed (the FINAL response is, see the dedicated test below).
    task = _task(
        title="pending carrier",
        pending_input=[{"text": "check the websocket reconnect"}],
        messages=[
            {"parts": [{"part_kind": "user-prompt", "content": "hello"}]},
            {
                "parts": [
                    {
                        "part_kind": "text",
                        "content": "websocket reconnect is handled here",
                    }
                ]
            },
            {"parts": [{"part_kind": "text", "content": "all done"}]},
        ],
    )
    _write_task(project_root, task)
    result = task_search.search(project_root, ["websocket", "reconnect"])
    assert f"#{task.id}" in result


def test_search_filters_status_and_kind(project_root: Path) -> None:
    bug = _task(title="crash on export", kind=TaskKind.BUG)
    feature = _task(title="crash course docs", kind=TaskKind.FEATURE)
    for t in (bug, feature):
        _write_task(project_root, t)
    result = task_search.search(project_root, ["crash"], kind="bug")
    assert f"#{bug.id}" in result
    assert f"#{feature.id}" not in result
    # A status filter can exclude the very same task.
    held = _task(title="crash again", status=TaskStatus.HOLD)
    _write_task(project_root, held)
    by_status = task_search.search(project_root, ["crash"], status="hold")
    assert f"#{held.id}" in by_status
    assert f"#{bug.id}" not in by_status  # still queued


def test_search_unknown_filter_raises_valueerror(project_root: Path) -> None:
    with pytest.raises(ValueError, match="status"):
        task_search.search(project_root, ["x"], status="done")
    with pytest.raises(ValueError, match="kind"):
        task_search.search(project_root, ["x"], kind="spike")


def test_search_limit_and_more_tail(project_root: Path) -> None:
    for i in range(task_search.MAX_RESULTS + 5):
        _write_task(project_root, _task(title=f"keyword hit {i}"))
    result = task_search.search(project_root, ["keyword"])
    lines = result.splitlines()
    assert len(lines) == task_search.MAX_RESULTS + 1
    assert "more" in lines[-1] and "refine" in lines[-1]


def test_search_snippet_is_compact(project_root: Path) -> None:
    filler = "word " * 200
    task = _task(title=filler + "needle " + filler)
    _write_task(project_root, task)
    result = task_search.search(project_root, ["needle"])
    body_line = next(line for line in result.splitlines() if "needle" in line)
    # Line = header + capped title + capped snippet (both ≤ SNIPPET_MAX).
    assert len(body_line) < task_search.SNIPPET_MAX * 2 + 80
    # No transcript leakage: the long filler is truncated away.
    assert body_line.count("word") < 60


def test_search_excludes_own_task_but_notices_the_match(
    project_root: Path,
) -> None:
    own = _task(title="research the flaky tests")
    _write_task(project_root, own)
    result = task_search.search(project_root, ["flaky"], exclude_task_id=own.id)
    # No result LINE for the own task…
    own_lines = [ln for ln in result.splitlines() if ln.startswith(f"#{own.id}")]
    assert not own_lines
    # …but a notice that it matches (the answer may live in the searcher's
    # own transcript).
    assert own.id in result
    assert "also matches" in result
    # Without exclusion it is found as a regular result.
    assert f"#{own.id}" in task_search.search(project_root, ["flaky"])


def test_search_own_task_notice_also_on_or_fallback(
    project_root: Path,
) -> None:
    own = _task(title="counter of unmerged tasks")
    _write_task(project_root, own)
    result = task_search.search(
        project_root, ["unmerged", "quantum"], exclude_task_id=own.id
    )
    # AND fails (no "quantum" anywhere) and no other task matches even one
    # keyword — but the own task DOES match one keyword, so besides the
    # no-result guidance the notice must point at the searcher's own
    # transcript.
    assert "No tasks matched" in result
    assert "also matches" in result
    assert own.id in result


def test_search_or_fallback_when_and_is_empty(project_root: Path) -> None:
    left = _task(title="alpha processor")
    right = _task(title="beta processor")
    _write_task(project_root, left)
    _write_task(project_root, right)
    result = task_search.search(project_root, ["alpha", "beta"])
    # No task has BOTH keywords → the AND pass is empty, OR kicks in with
    # an explicit preamble.
    assert "No exact AND match" in result
    assert f"#{left.id}" in result
    assert f"#{right.id}" in result
    # When AND matches exist, no fallback happens: an AND-only task list
    # must not drag in single-keyword tasks.
    both = _task(title="alpha beta")
    _write_task(project_root, both)
    strict = task_search.search(project_root, ["alpha", "beta"])
    assert "No exact AND match" not in strict
    assert f"#{both.id}" in strict
    assert f"#{left.id}" not in strict


def test_search_or_fallback_ranks_by_occurrences_then_recency(
    project_root: Path,
) -> None:
    rare = _task(title="zebra once")  # 1 occurrence
    common = _task(title="zebra zebra zebra")  # 3 occurrences
    _write_task(project_root, rare)
    _write_task(project_root, common)
    result = task_search.search(project_root, ["zebra", "quantum"])
    lines = [ln for ln in result.splitlines() if ln.startswith("#")]
    assert len(lines) == 2
    assert lines[0].startswith(f"#{common.id}")  # more occurrences first
    assert lines[1].startswith(f"#{rare.id}")

    # Equal occurrence counts → the newer task first.
    old = _task(
        title="equal term",
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    new = _task(title="equal term", created_at=datetime(2025, 1, 1, tzinfo=UTC))
    _write_task(project_root, old)
    _write_task(project_root, new)
    ranked = task_search.search(project_root, ["equal"])
    order = [ln.split()[0] for ln in ranked.splitlines() if ln.startswith("#")]
    assert order == [f"#{new.id}", f"#{old.id}"]


def test_search_ranks_and_results_by_occurrences(project_root: Path) -> None:
    weak = _task(title="alpha with beta")  # each keyword once
    strong = _task(title="alpha alpha with beta beta beta")
    _write_task(project_root, weak)
    _write_task(project_root, strong)
    result = task_search.search(project_root, ["alpha", "beta"])
    order = [ln.split()[0] for ln in result.splitlines() if ln.startswith("#")]
    assert order == [f"#{strong.id}", f"#{weak.id}"]


def test_search_indexes_edited_files(project_root: Path) -> None:
    task = _task(
        title="Add header git indicator",
        edited_files=["src/drcode/tui/header.py", "docs/workflow.md"],
    )
    _write_task(project_root, task)
    # Searching by a file path finds the task that touched that file.
    result = task_search.search(project_root, ["tui/header.py"])
    assert f"#{task.id}" in result
    # The snippet carries a provenance prefix for file matches.
    assert "files:" in result


def test_search_indexes_final_response(project_root: Path) -> None:
    # The Russian prompt / English title gap: the keyword exists ONLY in the
    # agent's closing summary — still enough to find the task.
    task = _task(
        title="Add header git indicator counting unmerged completed tasks",
        messages=[
            {"parts": [{"part_kind": "user-prompt", "content": "сделай индикатор"}]},
            {"parts": [{"part_kind": "text", "content": "промежуточный текст"}]},
            {
                "parts": [
                    {
                        "part_kind": "text",
                        "content": "Готово: добавлен счётчик выполненных "
                        "в ветках и не смерженных задач.",
                    }
                ]
            },
        ],
    )
    _write_task(project_root, task)
    result = task_search.search(project_root, ["счётчик"])
    assert f"#{task.id}" in result
    assert "summary:" in result  # provenance prefix for the final response


def test_search_indexes_events(project_root: Path) -> None:
    task = _task(
        title="merge investigation",
        events=[
            {"kind": "system", "text": "merge FAILED: conflict in app.py"},
            {"kind": "system", "text": "retry scheduled"},
        ],
    )
    _write_task(project_root, task)
    result = task_search.search(project_root, ["merge failed"])
    assert f"#{task.id}" in result
    assert "events:" in result


def test_search_finds_task_by_its_short_id(project_root: Path) -> None:
    """The task id is an indexed field: searching by the exact short id
    finds exactly that task, no matter its status/events."""
    target = _task(id="3717c406", title="rework the cache layer")
    other = _task(id="deadbeef", title="unrelated work")
    for t in (target, other):
        _write_task(project_root, t)
    hits = task_search.search_tasks(project_root, ["3717c406"])
    assert [h.task_id for h in hits] == ["3717c406"]
    # Same via the formatted agent tool output.
    result = task_search.search(project_root, ["3717c406"])
    assert "#3717c406" in result
    assert "#deadbeef" not in result


def test_search_by_id_finds_never_run_task_with_no_events(
    project_root: Path,
) -> None:
    """A task that never ran (no prompts, no events, no files) was
    previously invisible to the search — the id field alone must find it."""
    task = _task(id="1234abcd", title="just an idea")
    _write_task(project_root, task)
    assert "#1234abcd" in task_search.search(project_root, ["1234abcd"])


def test_search_by_id_prefix_substring(project_root: Path) -> None:
    """A partial id (e.g. a 4-char prefix) matches too — matching is a plain
    case-insensitive substring over the id field."""
    target = _task(id="3717c406", title="worktree isolation")
    decoy = _task(id="ffff0000", title="another task")
    for t in (target, decoy):
        _write_task(project_root, t)
    result = task_search.search(project_root, ["3717"])
    assert "#3717c406" in result
    assert "#ffff0000" not in result


def test_search_id_match_snippet_has_no_provenance_prefix(
    project_root: Path,
) -> None:
    """The id is already shown in `#{task_id}` on the result line, so an
    id-match snippet carries no "id:" prefix (unlike files:/summary:/…)."""
    task = _task(id="3717c406", title="quiet title")
    _write_task(project_root, task)
    result = task_search.search(project_root, ["3717c406"])
    (line,) = [ln for ln in result.splitlines() if ln.startswith("#3717c406")]
    assert "id:" not in line


def test_search_own_task_notice_on_own_id(project_root: Path) -> None:
    """Searching for the searcher's own id excludes the result line but
    keeps the "your own task also matches" notice."""
    own = _task(id="3717c406", title="the searcher itself")
    other = _task(id="deadbeef", title="unrelated work")
    for t in (own, other):
        _write_task(project_root, t)
    result = task_search.search(project_root, ["3717c406"], exclude_task_id="3717c406")
    assert not [ln for ln in result.splitlines() if ln.startswith("#3717c406")]
    assert "also matches" in result


def test_search_id_keyword_combines_with_other_keywords(project_root: Path) -> None:
    """AND semantics are unchanged: an id keyword must combine with other
    keywords like any other field."""
    both = _task(id="3717c406", title="cache layer")
    id_only = _task(id="3717c407", title="unrelated")
    text_only = _task(title="cache layer elsewhere")
    for t in (both, id_only, text_only):
        _write_task(project_root, t)
    result = task_search.search(project_root, ["3717c406", "cache"])
    assert "#3717c406" in result
    assert "#3717c407" not in result
    assert f"#{text_only.id}" not in result


def test_search_stem_variant_matches_inflected_forms(
    project_root: Path,
) -> None:
    task = _task(title="показать счётчик смерженных задач")
    _write_task(project_root, task)
    # Exact wordforms differ ("смерженных" in the task vs "смерженный" in
    # the query) — the stripped- ending variant bridges them.
    result = task_search.search(project_root, ["смерженный"])
    assert f"#{task.id}" in result


def test_search_no_results_message_hints_loosening(project_root: Path) -> None:
    result = task_search.search(project_root, ["nothing", "here"])
    assert "No tasks matched" in result
    # Guidance for widening the net (and a stem example when one applies).
    assert "fewer" in result
    stemmed = task_search.search(project_root, ["квантующихся"])
    assert "shorter stem" in stemmed


def test_search_verification_marker_on_result_line(
    project_root: Path,
) -> None:
    passed = _task(
        title="verified fix",
        status=TaskStatus.COMPLETED,
        verification=VerificationStatus.PASSED,
    )
    failed = _task(
        title="broken fix",
        status=TaskStatus.COMPLETED,
        verification=VerificationStatus.FAILED,
    )
    _write_task(project_root, passed)
    _write_task(project_root, failed)
    result = task_search.search(project_root, ["fix"])
    passed_line = next(
        ln for ln in result.splitlines() if ln.startswith(f"#{passed.id}")
    )
    failed_line = next(
        ln for ln in result.splitlines() if ln.startswith(f"#{failed.id}")
    )
    assert "passed" in passed_line
    assert "failed" in failed_line


def test_search_unmerged_marker_for_completed_branch(
    project_root: Path,
) -> None:
    """A COMPLETED task whose drcode/<id> branch still holds commits not in
    the base branch gets an "unmerged" marker (short result list only)."""
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
        )

    merged = _task(title="already merged fix", status=TaskStatus.COMPLETED)
    unmerged_task = _task(
        title="floating fix",
        status=TaskStatus.COMPLETED,
    )
    _write_task(project_root, merged)
    _write_task(project_root, unmerged_task)
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (project_root / "base.txt").write_text("base")
    git("add", "base.txt")
    git("commit", "-q", "-m", "base")
    # The unmerged task's branch carries an extra commit; the merged one has
    # no branch at all (work already integrated, branch cleaned up).
    git("branch", f"drcode/{unmerged_task.id}")
    git(
        "update-ref",
        f"refs/heads/drcode/{unmerged_task.id}",
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    (project_root / "extra.txt").write_text("work")
    git("add", "extra.txt")
    git("commit", "-q", "-m", "task work")
    git("update-ref", f"refs/heads/drcode/{unmerged_task.id}", "HEAD")
    git("reset", "-q", "--hard", "HEAD~1")

    result = task_search.search(project_root, ["fix"])
    unmerged_line = next(
        ln for ln in result.splitlines() if ln.startswith(f"#{unmerged_task.id}")
    )
    merged_line = next(
        ln for ln in result.splitlines() if ln.startswith(f"#{merged.id}")
    )
    assert "unmerged" in unmerged_line
    assert "unmerged" not in merged_line


def test_search_skips_corrupted_files(project_root: Path) -> None:
    good = _task(title="pagination bug")
    _write_task(project_root, good)
    bad = project_root / ".drcode" / "tasks" / "broken.json"
    bad.write_text("{not json")
    (project_root / ".drcode" / "tasks" / "empty.json").write_text("[]")
    result = task_search.search(project_root, ["pagination"])
    assert f"#{good.id}" in result
    assert "broken" not in result


def test_search_empty_keywords_raises(project_root: Path) -> None:
    with pytest.raises(ValueError, match="keyword"):
        task_search.search(project_root, ["  "])


def test_search_no_results_message(project_root: Path) -> None:
    result = task_search.search(project_root, ["nothing", "here"])
    assert "No tasks matched" in result


def test_search_works_against_main_root_not_worktree(
    project_root: Path, tmp_path: Path
) -> None:
    """Worktree isolation: task files live in the MAIN checkout — the search
    must be pointed at the main project root, never at the agent's cwd."""
    task = _task(title="isolation check")
    _write_task(project_root, task)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    found = task_search.search(project_root, ["isolation"])
    assert f"#{task.id}" in found
    assert "No tasks matched" in task_search.search(worktree, ["isolation"])


# -- structural core (TaskHit / search_tasks) -------------------------------------


def test_search_tasks_returns_hits_with_first_prompt(
    project_root: Path,
) -> None:
    """first_prompt: the first transcript user-prompt when the task ran at
    least once, the first pending_input text when it never did."""
    from drcode.task_search import TaskHit, search_tasks

    ran = _task(
        title="ran once",
        messages=[
            {"parts": [{"part_kind": "user-prompt", "content": "first ask"}]},
            {"parts": [{"part_kind": "user-prompt", "content": "second ask"}]},
        ],
        pending_input=[{"text": "queued comment"}],
    )
    never_ran = _task(
        title="never ran",
        pending_input=[{"text": "initial message"}, {"text": "later comment"}],
    )
    for t in (ran, never_ran):
        _write_task(project_root, t)
    hits = {h.task_id: h for h in search_tasks(project_root, ["once", "never"])}
    assert isinstance(hits[ran.id], TaskHit)
    assert hits[ran.id].first_prompt == "first ask"
    # A never-run task takes its first prompt from pending_input[0] — that
    # is where TaskManager.create puts the first message.
    assert hits[never_ran.id].first_prompt == "initial message"
    # Structural fields ride along.
    assert hits[ran.id].status == "queued"
    assert hits[ran.id].kind == "feature"
    assert len(hits[ran.id].date) == 10  # bare YYYY-MM-DD
    assert hits[ran.id].verification == ""
    assert hits[ran.id].snippet  # keyword "once" is in the title


def test_search_tasks_first_prompt_falls_back_to_pending_input(
    project_root: Path,
) -> None:
    """A transcript WITHOUT user-prompt parts (e.g. compaction artefacts)
    still yields pending_input[0]."""
    from drcode.task_search import search_tasks

    task = _task(
        title="odd transcript",
        messages=[{"parts": [{"part_kind": "text", "content": "agent text"}]}],
        pending_input=[{"text": "the real first message"}],
    )
    _write_task(project_root, task)
    (hit,) = search_tasks(project_root, ["odd"])
    assert hit.first_prompt == "the real first message"


def test_search_tasks_error_markers(project_root: Path) -> None:
    """The error combination: unfinished, FAILED verification,
    hook_merge_failed, and the last error-looking event text."""
    from drcode.task_search import search_tasks

    clean = _task(title="clean task")
    flagged = _task(
        title="troubled task",
        unfinished=True,
        hook_merge_failed=True,
        verification=VerificationStatus.FAILED,
        events=[
            {"kind": "system", "text": "retry scheduled"},  # not an error
            {"kind": "testing", "text": "testing failed — round 2"},
            {"kind": "system", "text": "merge FAILED: conflict in app.py"},
        ],
    )
    for t in (clean, flagged):
        _write_task(project_root, t)
    hits = {h.task_id: h for h in search_tasks(project_root, ["task"])}
    bad = hits[flagged.id]
    assert bad.unfinished is True
    assert bad.hook_merge_failed is True
    assert bad.verification == "failed"
    # The LAST error-looking event wins (the "round 2" testing failure is
    # skipped in favour of the later merge failure).
    assert bad.last_error_event == "merge FAILED: conflict in app.py"
    good = hits[clean.id]
    assert good.unfinished is False
    assert good.hook_merge_failed is False
    assert good.verification == ""
    assert good.last_error_event == ""


def test_search_tasks_unmerged_left_to_caller_and_annotated(
    project_root: Path,
) -> None:
    """search_tasks never computes mergedness itself (blocking git);
    annotate_unmerged fills it for short lists of completed hits."""
    from drcode.task_search import annotate_unmerged, search_tasks

    task = _task(title="floating fix", status=TaskStatus.COMPLETED)
    _write_task(project_root, task)
    # No git repo here: the annotation degrades to "no markers".
    (hit,) = search_tasks(project_root, ["fix"])
    assert hit.unmerged is None
    annotate_unmerged(project_root, [hit])
    assert hit.unmerged is None


def test_search_tasks_rejects_bad_input(project_root: Path) -> None:
    """Same validation contract as the agent tool: empty keywords and
    unknown filters raise ValueError."""
    import pytest

    from drcode.task_search import search_tasks

    with pytest.raises(ValueError, match="keyword"):
        search_tasks(project_root, ["  "])
    with pytest.raises(ValueError, match="status"):
        search_tasks(project_root, ["x"], status="done")
    with pytest.raises(ValueError, match="kind"):
        search_tasks(project_root, ["x"], kind="spike")


def test_search_tasks_ranking_matches_search(project_root: Path) -> None:
    """search_tasks ranks exactly like the formatter: occurrences first,
    ties newest-first."""
    from drcode.task_search import search_tasks

    weak = _task(title="alpha with beta")
    strong = _task(title="alpha alpha with beta beta beta")
    _write_task(project_root, weak)
    _write_task(project_root, strong)
    hits = search_tasks(project_root, ["alpha", "beta"])
    assert [h.task_id for h in hits] == [strong.id, weak.id]
    assert hits[0].score > hits[1].score


# -- incremental archive cache (load_archive) ---------------------------------


def test_load_archive_is_incremental(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warm searches reuse cached records; only files whose (mtime_ns, size)
    changed get re-read and re-parsed."""
    import os

    from drcode.task_search import _extract_record

    a = _task(title="alpha one")
    b = _task(title="beta two")
    _write_task(project_root, a)
    _write_task(project_root, b)

    calls: list[str] = []
    real = _extract_record
    monkeypatch.setattr(
        task_search,
        "_extract_record",
        lambda task_id, data: (calls.append(task_id), real(task_id, data))[1],
    )

    first = task_search.load_archive(project_root)
    assert sorted(task_id for task_id, _, _ in first) == sorted([a.id, b.id])
    assert sorted(calls) == sorted([a.id, b.id])  # cold: both parsed

    calls.clear()
    second = task_search.load_archive(project_root)
    assert calls == []  # warm: nothing re-parsed
    assert sorted(task_id for task_id, _, _ in second) == sorted([a.id, b.id])

    # A mtime-only touch re-parses exactly that file (a +2s bump is coarse
    # enough for any filesystem's timestamp granularity).
    path = project_root / ".drcode" / "tasks" / f"{a.id}.json"
    stat = path.stat()
    bumped = stat.st_mtime_ns + 2_000_000_000
    os.utime(path, ns=(bumped, bumped))
    calls.clear()
    task_search.load_archive(project_root)
    assert calls == [a.id]

    # Changed content: the file is re-parsed and its reduced data refreshed.
    _write_task(project_root, _task(id=a.id, title="alpha renamed"))
    calls.clear()
    records = task_search.load_archive(project_root)
    assert calls == [a.id]
    data = next(d for task_id, d, _ in records if task_id == a.id)
    assert data["title"] == "alpha renamed"

    # A new file: only the newcomer is parsed.
    c = _task(title="gamma three")
    _write_task(project_root, c)
    calls.clear()
    records = task_search.load_archive(project_root)
    assert calls == [c.id]
    assert sorted(task_id for task_id, _, _ in records) == sorted([a.id, b.id, c.id])

    # A deleted file drops out of the (re)cached results without a parse.
    (project_root / ".drcode" / "tasks" / f"{b.id}.json").unlink()
    calls.clear()
    records = task_search.load_archive(project_root)
    assert calls == []
    assert sorted(task_id for task_id, _, _ in records) == sorted([a.id, c.id])


def test_load_archive_records_are_reduced(project_root: Path) -> None:
    """Cached records hold the search-relevant fields only — the raw
    transcript is extracted at build time and never kept in memory."""
    task = _task(
        title="reduced check",
        messages=[
            {"parts": [{"part_kind": "user-prompt", "content": "hello index"}]},
            {"parts": [{"part_kind": "text", "content": "final answer"}]},
        ],
    )
    _write_task(project_root, task)
    records = task_search.load_archive(project_root)
    assert len(records) == 1
    task_id, data, fields = records[0]
    assert task_id == task.id
    assert "messages" not in data
    assert "tester_messages" not in data
    assert data["prompts"] == ["hello index"]
    assert ("prompt", "hello index") in fields
    assert ("summary", "final answer") in fields
    # A warm reload reuses the SAME reduced record (no transcript anywhere).
    again = task_search.load_archive(project_root)
    assert again[0] is records[0]


def test_load_archive_survives_broken_cache(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupted/foreign cache entry is quietly rebuilt from disk — the
    load_one contract, never fatal."""
    task = _task(title="cache casualty")
    _write_task(project_root, task)
    task_search.load_archive(project_root)  # warm the real cache first

    monkeypatch.setitem(
        task_search._ARCHIVE_CACHES, str(project_root), {"junk": object()}
    )
    records = task_search.load_archive(project_root)
    assert [task_id for task_id, _, _ in records] == [task.id]
    assert records[0][1]["title"] == "cache casualty"
    # The cache is healed: the next call is warm and identical.
    assert task_search.load_archive(project_root)[0] is records[0]


def test_load_archive_skips_corrupted_files(project_root: Path) -> None:
    """A corrupted task file is skipped silently and never poisons the
    cache — fixing the file makes it visible again without a restart."""
    good = _task(title="good task")
    _write_task(project_root, good)
    bad_path = project_root / ".drcode" / "tasks" / "badjob.json"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not json", encoding="utf-8")

    records = task_search.load_archive(project_root)
    assert [task_id for task_id, _, _ in records] == [good.id]

    records = task_search.load_archive(project_root)
    assert [task_id for task_id, _, _ in records] == [good.id]  # warm, still

    bad_path.write_text(
        _task(id="badjob", title="repaired task").model_dump_json(),
        encoding="utf-8",
    )
    records = task_search.load_archive(project_root)
    assert sorted(task_id for task_id, _, _ in records) == sorted([good.id, "badjob"])
