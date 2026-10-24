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
