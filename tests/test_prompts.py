"""Tests for drcode.prompts: bundled template loading and substitution."""

from __future__ import annotations

from pathlib import Path

from drcode.models import TaskKind
from drcode.prompts import (
    PROJECT_PATH_PLACEHOLDER,
    load_compaction_prompts,
    load_init_repo_prompt,
    load_system_prompt,
    load_tester_prompt,
)


def test_load_compaction_prompts_returns_two_nonempty_strings() -> None:
    system, template = load_compaction_prompts()
    assert system.strip()
    assert template.strip()


def test_compaction_prompts_are_distinct() -> None:
    system, template = load_compaction_prompts()
    assert system != template


def test_init_repo_prompt_substitutes_project_path(tmp_path: Path) -> None:
    prompt = load_init_repo_prompt(tmp_path / "my repo")
    assert str((tmp_path / "my repo").resolve()) in prompt
    assert PROJECT_PATH_PLACEHOLDER not in prompt


def test_init_repo_prompt_not_empty(tmp_path: Path) -> None:
    assert load_init_repo_prompt(tmp_path).strip()


def test_system_prompt_substitutes_project_path(tmp_path: Path) -> None:
    prompt = load_system_prompt(TaskKind.FEATURE, project_root=tmp_path)
    assert str(tmp_path) in prompt
    assert PROJECT_PATH_PLACEHOLDER not in prompt


def test_system_prompt_without_root_substitutes_neutral_marker() -> None:
    prompt = load_system_prompt(TaskKind.BUG)
    assert "(unknown)" in prompt
    assert PROJECT_PATH_PLACEHOLDER not in prompt


def test_tester_prompt_substitutes_project_path(tmp_path: Path) -> None:
    for allow_commit in (True, False):
        prompt = load_tester_prompt(allow_commit, project_root=tmp_path)
        assert str(tmp_path) in prompt
        assert PROJECT_PATH_PLACEHOLDER not in prompt


def test_system_prompt_templates_carry_placeholder_header() -> None:
    from drcode.prompts import (
        BUG_SYSTEM_PROMPT,
        FEATURE_SYSTEM_PROMPT,
        RESEARCH_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT_NO_COMMIT,
    )

    for template in (
        FEATURE_SYSTEM_PROMPT,
        BUG_SYSTEM_PROMPT,
        RESEARCH_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT_NO_COMMIT,
    ):
        assert template.startswith(
            "Working directory (project root): {project_path}\n"
            "Relative tool paths resolve against it; run_bash starts there.\n"
        )


def test_templates_exist_in_package() -> None:
    from drcode.prompts import _PROMPTS_DIR

    for name in ("compaction.txt", "compaction.md", "init_repo.md"):
        assert (_PROMPTS_DIR / name).is_file()
