"""Bundled prompt templates for service tasks (package data).

Templates live in `src/drcode/prompts/` and are read from disk at import time
(`load_system_prompt` / `load_tester_prompt` read them at call time; the
module-level string constants hold the raw, unsubstituted templates) or at
call time (`load_compaction_prompts`, `load_init_repo_prompt`). Reading them
as text — rather than importing the
`.md` files as Python modules — keeps the templates plain data the packager
can ship as `package-data`.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import TaskKind
from .settings import GitPolicy
from .workflow import load_git_workflow

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
# System prompts for the coding/tester agents live here (one `.md` file each).
SYSTEM_PROMPTS_DIR = _PROMPTS_DIR / "system"

# System prompts are read from disk and bound here as RAW templates (the
# `{project_path}` placeholder is NOT substituted) so consumers can import the
# text directly (e.g. the test-suite asserts on their content).
FEATURE_SYSTEM_PROMPT = (SYSTEM_PROMPTS_DIR / "feature.md").read_text(encoding="utf-8")
BUG_SYSTEM_PROMPT = (SYSTEM_PROMPTS_DIR / "bug.md").read_text(encoding="utf-8")
RESEARCH_SYSTEM_PROMPT = (SYSTEM_PROMPTS_DIR / "research.md").read_text(
    encoding="utf-8"
)
TESTER_SYSTEM_PROMPT = (SYSTEM_PROMPTS_DIR / "tester.md").read_text(encoding="utf-8")
TESTER_SYSTEM_PROMPT_NO_COMMIT = (SYSTEM_PROMPTS_DIR / "tester_no_commit.md").read_text(
    encoding="utf-8"
)
TESTER_SYSTEM_PROMPT_COMMIT_NO_PUSH = (
    SYSTEM_PROMPTS_DIR / "tester_commit_no_push.md"
).read_text(encoding="utf-8")
# Git-policy fragments of the research prompt (appended to the base
# research.md template by the policy — see load_system_prompt_components).
# The base template deliberately carries no git section at all: the OFF
# policy (research never touches git) gets neither fragment nor tool.
RESEARCH_GIT_PROMPT_PUSH = (SYSTEM_PROMPTS_DIR / "research_git_push.md").read_text(
    encoding="utf-8"
)
RESEARCH_GIT_PROMPT_LOCAL = (SYSTEM_PROMPTS_DIR / "research_git_local.md").read_text(
    encoding="utf-8"
)
RESOLVER_SYSTEM_PROMPT = (SYSTEM_PROMPTS_DIR / "resolver.md").read_text(
    encoding="utf-8"
)
RESOLVER_REVIEWER_PROMPT = (SYSTEM_PROMPTS_DIR / "resolver_reviewer.md").read_text(
    encoding="utf-8"
)

# Placeholder in the templates, replaced with the actual project directory.
PROJECT_PATH_PLACEHOLDER = "{project_path}"
# Neutral substitution used when no project root is known — the raw
# placeholder must never leak into a prompt handed to the model.
_UNKNOWN_PROJECT_PATH = "(unknown)"


def _substitute_project_path(template: str, project_root: Path | None) -> str:
    """Replace ``{project_path}`` with the actual project directory.

    ``None`` substitutes the neutral ``(unknown)`` marker instead.
    """
    replacement = (
        str(project_root) if project_root is not None else _UNKNOWN_PROJECT_PATH
    )
    return template.replace(PROJECT_PATH_PLACEHOLDER, replacement)


# The "no commits" sentence of the bundled coding prompts (spelled across a
# line break in feature.md). When the project ships its own git workflow,
# this sentence is removed: the workflow file decides when committing is
# allowed, and its text takes over entirely.
_NEVER_COMMIT_RE = re.compile(r"\s*Never commit to\s+git unless explicitly asked\.?")

# Header under which the project's `.drcode/git-workflow.md` rules are
# appended to a system prompt.
GIT_WORKFLOW_HEADER = "Git workflow (project rules)"


def _project_prompt_override(project_root: Path | None, name: str) -> str | None:
    """A project-local override of a bundled system prompt:
    `<root>/.drcode/prompts/<name>.md`, or None when absent."""
    if project_root is None:
        return None
    path = project_root / ".drcode" / "prompts" / name
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _join_components(base: str, git_workflow: str | None) -> str:
    """Append the git-workflow rules to a base prompt (the single place the
    two components of any system prompt are composed)."""
    if git_workflow is None:
        return base
    return f"{base}\n\n# {GIT_WORKFLOW_HEADER}\n\n{git_workflow}"


def load_system_prompt_components(
    kind: TaskKind,
    project_root: Path | None = None,
    research_git: GitPolicy = GitPolicy.PUSH,
) -> tuple[str, str | None]:
    """(base prompt, git workflow or None) for a coding task of the given kind.

    The base prompt is the (possibly project-overridden) template with the
    ``{project_path}`` placeholder substituted and — when the project ships
    `.drcode/git-workflow.md` — the bundled "Never commit to git" sentence
    removed. For research tasks the project's git policy (`research_git`,
    `GitPolicy`) selects the git fragment appended after the base template:
    PUSH/LOCAL add the git_commit_docs section (`research_git_push.md` /
    `research_git_local.md`, each overridable project-locally by the same
    name in `.drcode/prompts/`), OFF adds nothing — the research agent then
    has no commit step at all. The second element is the workflow text to
    append under `GIT_WORKFLOW_HEADER` (None when there is none). Note the
    base prompt does NOT include the project's AGENTS.md: that appendage is
    assembled by `drcode.agent.build_agent` (see `load_agents_md`).
    `load_system_prompt` joins the two components; consumers that must
    account for the parts separately (the context-layout estimate in the
    TUI) use this directly.
    """
    name = {
        TaskKind.FEATURE: "feature.md",
        TaskKind.BUG: "bug.md",
        TaskKind.RESEARCH: "research.md",
    }[kind]
    template = _project_prompt_override(project_root, name) or (
        SYSTEM_PROMPTS_DIR / name
    ).read_text(encoding="utf-8")
    if kind is TaskKind.RESEARCH:
        fragment_name = {
            GitPolicy.PUSH: "research_git_push.md",
            GitPolicy.LOCAL: "research_git_local.md",
            GitPolicy.OFF: None,
        }[research_git]
        if fragment_name is not None:
            fragment = _project_prompt_override(project_root, fragment_name) or (
                SYSTEM_PROMPTS_DIR / fragment_name
            ).read_text(encoding="utf-8")
            template = f"{template}\n\n{fragment}"
    git_workflow = load_git_workflow(project_root)
    if git_workflow is not None:
        template = _NEVER_COMMIT_RE.sub("", template)
    return _substitute_project_path(template, project_root), git_workflow


def load_tester_prompt_components(
    allow_commit: bool = True,
    project_root: Path | None = None,
    allow_push: bool = True,
) -> tuple[str, str | None]:
    """(base prompt, git workflow or None) for the QA tester — the component
    counterpart of `load_system_prompt_components`."""
    if not allow_commit:
        name = "tester_no_commit.md"
    elif not allow_push:
        name = "tester_commit_no_push.md"
    else:
        name = "tester.md"
    template = _project_prompt_override(project_root, name) or (
        SYSTEM_PROMPTS_DIR / name
    ).read_text(encoding="utf-8")
    git_workflow = load_git_workflow(project_root)
    return _substitute_project_path(template, project_root), git_workflow


def load_system_prompt(
    kind: TaskKind,
    project_root: Path | None = None,
    research_git: GitPolicy = GitPolicy.PUSH,
) -> str:
    """Bundled system prompt for a coding task of the given kind.

    Each kind gets its own `.md` template in `prompts/system/` (feature tasks
    encourage refactoring, bug tasks demand minimal changes, research tasks
    answer questions / update docs). A project may override the template with
    `<root>/.drcode/prompts/<name>.md`. The ``{project_path}`` placeholder in
    the template header is replaced with the actual project directory; without
    a root a neutral ``(unknown)`` marker is substituted. For research tasks
    `research_git` (the project's git policy for the role) picks the
    git_commit_docs fragment appended after the base template — none under
    the OFF policy.

    When the project defines `.drcode/git-workflow.md` (team branching /
    commit / review rules), the bundled "Never commit to git" sentence is
    removed (the workflow file governs committing instead) and the workflow
    text is appended under a dedicated header.
    """
    base, git_workflow = load_system_prompt_components(kind, project_root, research_git)
    return _join_components(base, git_workflow)


def load_tester_prompt(
    allow_commit: bool = True,
    project_root: Path | None = None,
    allow_push: bool = True,
) -> str:
    """Bundled QA tester system prompt.

    With ``allow_commit=False`` (tester git policy: off) the commit-free
    template is returned; otherwise the default one that
    mentions ``git_commit``. With ``allow_push=False`` (tester git policy:
    local) the commit-locally template is returned — it never promises a
    push. A project may override the template
    with `<root>/.drcode/prompts/tester.md` (or `tester_no_commit.md` /
    `tester_commit_no_push.md`). The ``{project_path}`` placeholder in the
    template header is replaced with the actual project directory; without a
    root a neutral ``(unknown)`` marker is substituted.

    When the project defines `.drcode/git-workflow.md`, its rules are
    appended: the tester must finalize the task according to the same team
    process the coding agent follows.
    """
    base, git_workflow = load_tester_prompt_components(
        allow_commit, project_root, allow_push
    )
    return _join_components(base, git_workflow)


def load_resolver_prompt(project_root: Path | None = None) -> str:
    """Bundled merge-conflict resolver system prompt.

    The ``{project_path}`` placeholder in the template header is replaced
    with the RESOLVED worktree's path (the isolated tree the conflicted
    merge is parked in), not the project root. A project may override the
    template with ``<root>/.drcode/prompts/resolver.md``.
    """
    template = _project_prompt_override(project_root, "resolver.md") or (
        SYSTEM_PROMPTS_DIR / "resolver.md"
    ).read_text(encoding="utf-8")
    return _substitute_project_path(template, project_root)


def load_resolver_reviewer_prompt(project_root: Path | None = None) -> str:
    """Bundled prompt of the intent-preservation reviewer: the pass that
    receives diff(resolved, ours) and diff(resolved, theirs) and answers
    whether a piece of either side was lost (the #6c065401 incident
    class). A project may override it with
    ``<root>/.drcode/prompts/resolver_reviewer.md``."""
    return _project_prompt_override(project_root, "resolver_reviewer.md") or (
        SYSTEM_PROMPTS_DIR / "resolver_reviewer.md"
    ).read_text(encoding="utf-8")


def load_compaction_prompts() -> tuple[str, str]:
    """(system prompt, output template) for the context-compaction agent.

    The system prompt (compaction.txt) instructs an anchored summarizer: it
    updates the current summary instead of writing a new one from scratch.
    The template (compaction.md) is the exact output structure the
    summarizer must follow.
    """
    system = (_PROMPTS_DIR / "compaction.txt").read_text(encoding="utf-8")
    template = (_PROMPTS_DIR / "compaction.md").read_text(encoding="utf-8")
    return system, template


def load_init_repo_prompt(project_root: Path) -> str:
    """Prompt for the "Update/create AGENTS.md" research task.

    The bundled template ends with a paragraph that names the repository
    location; the `{project_path}` placeholder is replaced with the actual
    project directory the TUI is operating on.
    """
    template = (_PROMPTS_DIR / "init_repo.md").read_text(encoding="utf-8")
    return template.replace(PROJECT_PATH_PLACEHOLDER, str(project_root.resolve()))
