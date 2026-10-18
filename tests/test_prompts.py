"""Tests for drcode.prompts: bundled template loading and substitution."""

from __future__ import annotations

from pathlib import Path

from drcode.prompts import (
    PROJECT_PATH_PLACEHOLDER,
    load_compaction_prompts,
    load_init_repo_prompt,
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


def test_templates_exist_in_package() -> None:
    from drcode.prompts import _PROMPTS_DIR

    for name in ("compaction.txt", "compaction.md", "init_repo.md"):
        assert (_PROMPTS_DIR / name).is_file()
