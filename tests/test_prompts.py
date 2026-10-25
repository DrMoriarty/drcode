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


def test_tester_prompt_no_push_variant(tmp_path: Path) -> None:
    """The commit-without-push finalization selects the dedicated template:
    it teaches the commit flow but never promises a push — and the default
    (pushing) template does."""
    from drcode.prompts import TESTER_SYSTEM_PROMPT_COMMIT_NO_PUSH

    pushy = load_tester_prompt(True, tmp_path)
    no_push = load_tester_prompt(True, tmp_path, allow_push=False)
    no_commit = load_tester_prompt(False, tmp_path, allow_push=False)
    # The no-push prompt is its own template, distinct from both others.
    assert no_push == TESTER_SYSTEM_PROMPT_COMMIT_NO_PUSH.replace(
        "{project_path}", str(tmp_path)
    )
    assert no_push != pushy and no_push != no_commit
    assert "git_commit" in no_push
    assert "never push" in no_push
    # The no-push template must not promise a push anywhere.
    assert "it also pushes the branch" not in no_push
    # The default template keeps its push promise.
    assert "pushes the branch" in pushy


def test_tester_prompt_no_push_project_override(tmp_path: Path) -> None:
    """`.drcode/prompts/tester_commit_no_push.md` overrides the bundled
    no-push template, like the other tester variants."""
    override_dir = tmp_path / ".drcode" / "prompts"
    override_dir.mkdir(parents=True)
    (override_dir / "tester_commit_no_push.md").write_text(
        "Working directory (project root): {project_path}\nCustom no-push prompt."
    )
    prompt = load_tester_prompt(True, tmp_path, allow_push=False)
    assert "Custom no-push prompt." in prompt
    assert str(tmp_path) in prompt


def test_system_prompt_templates_carry_placeholder_header() -> None:
    from drcode.prompts import (
        BUG_SYSTEM_PROMPT,
        FEATURE_SYSTEM_PROMPT,
        RESEARCH_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT_COMMIT_NO_PUSH,
        TESTER_SYSTEM_PROMPT_NO_COMMIT,
    )

    for template in (
        FEATURE_SYSTEM_PROMPT,
        BUG_SYSTEM_PROMPT,
        RESEARCH_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT_NO_COMMIT,
        TESTER_SYSTEM_PROMPT_COMMIT_NO_PUSH,
    ):
        assert template.startswith(
            "Working directory (project root): {project_path}\n"
            "Relative tool paths resolve against it; run_bash starts there.\n"
        )


def test_templates_exist_in_package() -> None:
    from drcode.prompts import _PROMPTS_DIR

    for name in ("compaction.txt", "compaction.md", "init_repo.md"):
        assert (_PROMPTS_DIR / name).is_file()
