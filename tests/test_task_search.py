"""Tests for the research agent's task-archive search (drcode.task_search)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from drcode import task_search
from drcode.models import Task, TaskKind, TaskStatus


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


def test_search_indexes_pending_input_and_skips_model_parts(
    project_root: Path,
) -> None:
    # The keyword lives in queued pending input AND in a model RESPONSE;
    # only the pending input must match (responses are not indexed).
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


def test_search_excludes_own_task(project_root: Path) -> None:
    own = _task(title="research the flaky tests")
    _write_task(project_root, own)
    result = task_search.search(project_root, ["flaky"], exclude_task_id=own.id)
    assert f"#{own.id}" not in result
    # Without exclusion it is found.
    assert f"#{own.id}" in task_search.search(project_root, ["flaky"])


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
