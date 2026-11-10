"""File-overlap warning at task creation (src/drcode/task_overlap.py).

The #260aa4b8 incident: one research run filed three feature requests
off one incident, all touching the same files, no dependencies — the
branches conflicted at the merge and the resolver rounds cost more than
the development. The creation paths must WARN (never block) when the
new task's description shares path tokens with unfinished tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from drcode import agent_tools, cli
from drcode.models import TaskKind, TaskStatus
from drcode.task_manager import TaskManager
from drcode.task_overlap import (
    MAX_WARNED_FILES,
    MAX_WARNED_TASKS,
    TaskFileOverlap,
    build_overlap_event_text,
    build_overlap_warning,
    extract_path_tokens,
    find_file_overlaps,
    normalize_path_token,
    paths_overlap,
)

# --- the path-token extractor -------------------------------------------------


def test_extracts_backticked_slashed_path() -> None:
    assert extract_path_tokens("`src/drcode/core/pipeline_resolver.py`") == [
        "src/drcode/core/pipeline_resolver.py"
    ]


def test_extracts_bare_slashed_path() -> None:
    assert extract_path_tokens("see src/drcode/core/pipeline_resolver.py for it") == [
        "src/drcode/core/pipeline_resolver.py"
    ]


def test_extracts_path_with_line_suffix() -> None:
    # The :line suffix is consumed and stripped — the file is the token.
    assert extract_path_tokens("`tests/test_x.py:123` fails") == ["tests/test_x.py"]
    assert extract_path_tokens("agent.py:45-60 handles it") == ["agent.py"]


def test_extracts_bare_file_name() -> None:
    assert extract_path_tokens("the fix belongs in merge_resolver.py") == [
        "merge_resolver.py"
    ]
    assert "readme.md" == extract_path_tokens("update README.md")[0]


def test_extracts_absolute_posix_path() -> None:
    # Absolute paths must yield the same token shape as relative ones —
    # the leading `/` is stripped by normalization, the suffix matching
    # in paths_overlap relates it to the relative mention.
    assert extract_path_tokens(
        "bug in /Users/vasiliy/pr/drcode/src/drcode/core/pipeline_resolver.py"
    ) == ["users/vasiliy/pr/drcode/src/drcode/core/pipeline_resolver.py"]


def test_extracts_backticked_absolute_path_with_line() -> None:
    assert extract_path_tokens("crash at `/home/u/repo/src/app.py:42`") == [
        "home/u/repo/src/app.py"
    ]


def test_extracts_windows_backslash_path() -> None:
    # The full segment chain (the `C:` drive prefix is not part of the
    # path chars and is stripped in normalization — the chain is what
    # suffix-matches a differently-rooted mention of the same file).
    assert extract_path_tokens(r"see C:\Users\foo\bar.py here") == ["users/foo/bar.py"]


def test_absolute_and_relative_mentions_overlap(manager: TaskManager) -> None:
    active = manager.create(
        "Feature one", "Rework src/drcode/core/pipeline_resolver.py"
    )
    overlaps = find_file_overlaps(
        manager.tasks,
        "Patch /Users/vasiliy/pr/drcode/src/drcode/core/pipeline_resolver.py:42",
        exclude_id="nonexistent",
    )
    assert [o.task_id for o in overlaps] == [active.id]
    # The reported file is the NEW task's quoting (the absolute form);
    # it must still relate to the relative mention via suffix matching.
    assert len(overlaps[0].files) == 1
    assert paths_overlap("src/drcode/core/pipeline_resolver.py", overlaps[0].files[0])


def test_normalizes_leading_dot_slash_and_case() -> None:
    assert normalize_path_token("./Src/App.PY") == "src/app.py"
    assert normalize_path_token("/Abs/Path.py") == "abs/path.py"


def test_drops_numbers_versions_and_abbreviations() -> None:
    tokens = extract_path_tokens(
        "line 123 of it, version 1.2.3, ratio 3.5:1, see e.g. the docs"
    )
    assert tokens == []


def test_deduplicates_and_keeps_order() -> None:
    tokens = extract_path_tokens("a.py then src/a.py then a.py again")
    assert tokens == ["a.py", "src/a.py"]


def test_suffix_matching_relates_short_and_long_quotes() -> None:
    assert paths_overlap(
        "core/pipeline_resolver.py", "src/drcode/core/pipeline_resolver.py"
    )
    assert paths_overlap(
        "src/drcode/core/pipeline_resolver.py", "core/pipeline_resolver.py"
    )
    assert not paths_overlap("core/pipeline_resolver.py", "core/other_resolver.py")


# --- the scan -----------------------------------------------------------------


def test_find_overlaps_by_description_tokens(manager: TaskManager) -> None:
    active = manager.create(
        "Fix resolver", "Rework src/drcode/core/pipeline_resolver.py"
    )
    overlap_tasks = find_file_overlaps(
        manager.tasks,
        "Refactor `src/drcode/core/pipeline_resolver.py` and merge_resolver.py",
        exclude_id="nonexistent",
    )
    assert [o.task_id for o in overlap_tasks] == [active.id]
    assert overlap_tasks[0].files == ["src/drcode/core/pipeline_resolver.py"]


def test_find_overlaps_uses_edited_files_of_running_tasks(manager: TaskManager) -> None:
    active = manager.create("Rework resolver", "Rework the retry logic")
    active.status = TaskStatus.ACTIVE
    active.edited_files.append("src/drcode/merge_resolver.py")
    overlaps = find_file_overlaps(
        manager.tasks, "Patch merge_resolver.py:123", exclude_id="nonexistent"
    )
    assert [o.task_id for o in overlaps] == [active.id]


def test_running_task_never_matches_itself(manager: TaskManager) -> None:
    active = manager.create("Rework resolver", "Patch merge_resolver.py:123")
    active.status = TaskStatus.ACTIVE
    active.edited_files.append("src/drcode/merge_resolver.py")
    assert (
        find_file_overlaps(
            manager.tasks, "Patch merge_resolver.py:123", exclude_id=active.id
        )
        == []
    )


def test_no_overlap_no_warning(manager: TaskManager) -> None:
    manager.create("Docs pass", "Rewrite docs/guide.md and README.md")
    assert (
        find_file_overlaps(
            manager.tasks,
            "Add tests for tests/test_task_overlap_warning.py",
            exclude_id="nonexistent",
        )
        == []
    )


def test_completed_tasks_are_ignored(manager: TaskManager) -> None:
    done = manager.create("Old work", "Touched src/drcode/core/pipeline_resolver.py")
    done.status = TaskStatus.COMPLETED
    assert (
        find_file_overlaps(
            manager.tasks,
            "Fix src/drcode/core/pipeline_resolver.py",
            exclude_id="nonexistent",
        )
        == []
    )


def test_depends_on_tasks_are_excluded(manager: TaskManager) -> None:
    other = manager.create("First feature", "Edit src/drcode/merge_resolver.py")
    overlaps = find_file_overlaps(
        manager.tasks,
        "Second feature editing src/drcode/merge_resolver.py",
        exclude_id="nonexistent",
        depends_on=[other.id],
    )
    assert overlaps == []


def test_report_is_capped_to_top_tasks_and_files(manager: TaskManager) -> None:
    file_count = MAX_WARNED_FILES + 5
    many_files = " ".join(f"src/mod_{i}.py" for i in range(file_count))
    for i in range(MAX_WARNED_TASKS + 2):
        manager.create(f"Task {i}", many_files)
    overlaps = find_file_overlaps(manager.tasks, many_files, exclude_id="nonexistent")
    assert len(overlaps) == MAX_WARNED_TASKS
    assert len(overlaps[0].files) == file_count
    assert f"+{file_count - MAX_WARNED_FILES} more" in build_overlap_event_text(
        overlaps
    )


def test_texts_are_actionable() -> None:
    overlaps = [TaskFileOverlap(task_id="260aa4b8", files=["merge_resolver.py"])]
    warning = build_overlap_warning(overlaps)
    assert "Warning:" in warning
    assert "#260aa4b8" in warning
    assert "depends_on" in warning
    event = build_overlap_event_text(overlaps)
    assert "#260aa4b8" in event
    assert "set_task_dependencies" in event


# --- integration: the agent's create tools -------------------------------------


def _tester_deps(manager: TaskManager) -> agent_tools.TesterDeps:
    from drcode.agent_build import TesterDeps
    from drcode.tools import scratch_dir

    verifier = manager.create("verify", "x")
    return TesterDeps(
        root=manager.storage.root,
        manager=manager,
        task_id=verifier.id,
        scratch=scratch_dir(verifier.id),
    )


def test_create_report_task_warns_in_result_and_event(
    manager: TaskManager,
) -> None:
    running = manager.create(
        "Feature one", "Rework src/drcode/core/pipeline_resolver.py"
    )
    deps = _tester_deps(manager)
    task, text = agent_tools._create_report_task(
        deps,
        "Feature two",
        "Also reworks `src/drcode/core/pipeline_resolver.py`",
        kind=TaskKind.FEATURE,
        creator="tester",
    )
    assert "Warning:" in text
    assert f"#{running.id}" in text
    assert "depends_on" in text
    assert any(e.kind == "system" and f"#{running.id}" in e.text for e in task.events)


def test_create_report_task_silent_without_overlap(
    manager: TaskManager,
) -> None:
    manager.create("Docs pass", "Rewrite docs/guide.md")
    deps = _tester_deps(manager)
    task, text = agent_tools._create_report_task(
        deps,
        "Unrelated fix",
        "Fix the flaky header test in tests/test_header.py — it fails on CI",
        kind=TaskKind.BUG,
        creator="tester",
    )
    assert "Warning:" not in text
    assert task.events == []


def test_create_report_task_draft_also_warns(manager: TaskManager) -> None:
    running = manager.create(
        "Feature one", "Rework src/drcode/core/pipeline_resolver.py"
    )

    @dataclass
    class DraftDeps:
        """Minimal deps stand-in with agent_creates_drafts=True."""

        root: Path
        manager: TaskManager
        task_id: str = "whatever"
        scratch: Path = Path("/tmp/drcode/whatever")
        agent_creates_drafts: bool = True

    task, text = agent_tools._create_report_task(
        DraftDeps(root=manager.storage.root, manager=manager),  # type: ignore[arg-type]
        "Feature two",
        "Also reworks src/drcode/core/pipeline_resolver.py",
        kind=TaskKind.FEATURE,
        creator="research agent",
    )
    assert task.status is TaskStatus.DRAFT
    assert "Warning:" in text
    assert f"#{running.id}" in text


# --- integration: the headless CLI ---------------------------------------------


def test_cli_create_warns_on_overlap(project_root: Path, capsys) -> None:
    manager = TaskManager(project_root)
    running = manager.create(
        "Feature one", "Rework src/drcode/core/pipeline_resolver.py"
    )
    code = cli.create_task_from_cli(
        project_root,
        TaskKind.FEATURE,
        "Feature two: also reworks src/drcode/core/pipeline_resolver.py",
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Warning:" in out
    assert f"#{running.id}" in out
    reloaded = TaskManager(project_root)
    assert any(
        f"#{running.id}" in e.text for t in reloaded.tasks.values() for e in t.events
    )


def test_cli_create_silent_without_overlap(project_root: Path, capsys) -> None:
    manager = TaskManager(project_root)
    manager.create("Docs pass", "Rewrite docs/guide.md")
    code = cli.create_task_from_cli(
        project_root, TaskKind.FEATURE, "Add tests for tests/test_header.py"
    )
    assert code == 0
    assert "Warning:" not in capsys.readouterr().out
