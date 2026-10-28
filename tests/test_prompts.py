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
from drcode.settings import GitPolicy


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


def test_tester_prompts_teach_project_environment_checks() -> None:
    """All three tester templates must route Python tooling through the
    run_check tool (the project's own environment) and forbid falling back
    to a bare PATH binary — a PATH ruff of a different version reports
    phantom failures the project's own gate never produces (3ea596cb)."""
    from drcode.prompts import (
        TESTER_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT_COMMIT_NO_PUSH,
        TESTER_SYSTEM_PROMPT_NO_COMMIT,
    )

    for template in (
        TESTER_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT_NO_COMMIT,
        TESTER_SYSTEM_PROMPT_COMMIT_NO_PUSH,
    ):
        assert "run_check" in template
        assert "PATH binary" in template
        assert "phantom" in template
        # The old wording that never said whose tool runs the checks.
        assert "Run each detected check via run_bash" not in template


def test_tester_prompt_no_push_variant(tmp_path: Path) -> None:
    """The commit-without-push tester git policy selects the dedicated
    template: it teaches the commit flow but never promises a push — and
    the default (pushing) template does."""
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


def test_research_prompt_git_policy_push_default(tmp_path: Path) -> None:
    """The default research prompt (git policy push) appends the push
    fragment: git_commit_docs is taught with the push promise — and the
    base template itself carries no git section (so the OFF policy can
    drop it entirely)."""
    from drcode.prompts import RESEARCH_SYSTEM_PROMPT

    push = load_system_prompt(
        TaskKind.RESEARCH, project_root=tmp_path, research_git=GitPolicy.PUSH
    )
    base = load_system_prompt(
        TaskKind.RESEARCH, project_root=tmp_path, research_git=GitPolicy.OFF
    )
    # The base template has no commit section at all.
    assert "git_commit_docs" not in base
    assert "Documentation assignments" not in base
    assert "Never commit to git unless explicitly asked" in base
    # The push fragment rides after the base template.
    assert "git_commit_docs" in push
    assert "MUST finish the task by calling git_commit_docs" in push
    assert "pushes the branch when a remote is configured" in push
    assert "Never commit to git unless explicitly asked" in push
    assert push.startswith(base.rstrip("\n"))
    # The module constant is the base template; the push fragment is its
    # own bundled file.
    assert RESEARCH_SYSTEM_PROMPT == base.replace(str(tmp_path), "{project_path}")


def test_research_prompt_git_policy_local(tmp_path: Path) -> None:
    """The local research prompt teaches git_commit_docs but never
    promises a push — it explicitly forbids pushing instead."""
    local = load_system_prompt(
        TaskKind.RESEARCH, project_root=tmp_path, research_git=GitPolicy.LOCAL
    )
    push = load_system_prompt(
        TaskKind.RESEARCH, project_root=tmp_path, research_git=GitPolicy.PUSH
    )
    assert "git_commit_docs" in local
    assert "MUST finish the task by calling git_commit_docs" in local
    assert "never push" in local
    # The local template must not promise a push anywhere.
    assert "pushes the branch" not in local
    assert local != push


def test_research_prompt_git_fragment_project_override(tmp_path: Path) -> None:
    """`.drcode/prompts/research_git_local.md` overrides the bundled
    local fragment, mirroring the tester variants."""
    override_dir = tmp_path / ".drcode" / "prompts"
    override_dir.mkdir(parents=True)
    (override_dir / "research_git_local.md").write_text("Custom local fragment.")
    prompt = load_system_prompt(
        TaskKind.RESEARCH, project_root=tmp_path, research_git=GitPolicy.LOCAL
    )
    assert "Custom local fragment." in prompt
    # The base template (and its project override) is still there.
    assert str(tmp_path) in prompt


def test_research_prompt_initiative_boundary() -> None:
    """The research prompt carries the initiative-boundary section: answers
    separate what was asked from agent's proposals, ask_clarifying_question
    options never bundle initiative into package opt-ins, and filed task
    descriptions record only the user-confirmed scope."""
    from drcode.prompts import RESEARCH_SYSTEM_PROMPT

    assert "# Initiative boundary" in RESEARCH_SYSTEM_PROMPT
    assert "(agent's suggestion, not from your request)" in RESEARCH_SYSTEM_PROMPT
    assert "SEPARATE opt-in" in RESEARCH_SYSTEM_PROMPT
    assert "create_bug_report / create_feature_request" in RESEARCH_SYSTEM_PROMPT
    assert "at the implementer's discretion" in RESEARCH_SYSTEM_PROMPT
    # The information-vs-decision fork: FYI notes stay in prose, verdict-
    # requiring proposals must go through ask_clarifying_question — with the
    # explicit anti-example against "say the word and I'll do it" endings.
    assert "REQUIRES a user verdict" in RESEARCH_SYSTEM_PROMPT
    assert "must NEVER be left as prose awaiting" in RESEARCH_SYSTEM_PROMPT
    assert "BAD example" in RESEARCH_SYSTEM_PROMPT
    assert "word and I'll create the task." in RESEARCH_SYSTEM_PROMPT
    assert "GOOD: the same question inside" in RESEARCH_SYSTEM_PROMPT


def test_bug_and_feature_prompts_mirror_options_discipline() -> None:
    """The coding agents' prompts (bug/feature) carry the same rule next to
    their ask_clarifying_question mention: options keep requested work
    separate from initiative, initiative rides in its own marked opt-in."""
    from drcode.prompts import BUG_SYSTEM_PROMPT, FEATURE_SYSTEM_PROMPT

    for template in (BUG_SYSTEM_PROMPT, FEATURE_SYSTEM_PROMPT):
        assert "keep the user's request separate from your" in template
        assert "(agent's suggestion, not from your request)" in template
        # The mirror of the information-vs-decision fork: verdict-requiring
        # proposals ride the tool, never prose.
        assert "needs the user's verdict" in template
        assert "never as prose in the answer" in template


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
