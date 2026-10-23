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
    kind: TaskKind, project_root: Path | None = None
) -> tuple[str, str | None]:
    """(base prompt, git workflow or None) for a coding task of the given kind.

    The base prompt is the (possibly project-overridden) template with the
    ``{project_path}`` placeholder substituted and — when the project ships
    `.drcode/git-workflow.md` — the bundled "Never commit to git" sentence
    removed. The second element is the workflow text to append under
    `GIT_WORKFLOW_HEADER` (None when there is none). Note the base prompt does
    NOT include the project's AGENTS.md: that appendage is assembled by
    `drcode.agent.build_agent` (see `load_agents_md`). `load_system_prompt`
    joins the two components; consumers that must account for the parts
    separately (the context-layout estimate in the TUI) use this directly.
    """
    name = {
        TaskKind.FEATURE: "feature.md",
        TaskKind.BUG: "bug.md",
        TaskKind.RESEARCH: "research.md",
    }[kind]
    template = _project_prompt_override(project_root, name) or (
        SYSTEM_PROMPTS_DIR / name
    ).read_text(encoding="utf-8")
    git_workflow = load_git_workflow(project_root)
    if git_workflow is not None:
        template = _NEVER_COMMIT_RE.sub("", template)
    return _substitute_project_path(template, project_root), git_workflow


def load_tester_prompt_components(
    allow_commit: bool = True, project_root: Path | None = None
) -> tuple[str, str | None]:
    """(base prompt, git workflow or None) for the QA tester — the component
    counterpart of `load_system_prompt_components`."""
    name = "tester_no_commit.md" if not allow_commit else "tester.md"
    template = _project_prompt_override(project_root, name) or (
        SYSTEM_PROMPTS_DIR / name
    ).read_text(encoding="utf-8")
    git_workflow = load_git_workflow(project_root)
    return _substitute_project_path(template, project_root), git_workflow


def load_system_prompt(kind: TaskKind, project_root: Path | None = None) -> str:
    """Bundled system prompt for a coding task of the given kind.

    Each kind gets its own `.md` template in `prompts/system/` (feature tasks
    encourage refactoring, bug tasks demand minimal changes, research tasks
    answer questions / update docs). A project may override the template with
    `<root>/.drcode/prompts/<name>.md`. The ``{project_path}`` placeholder in
    the template header is replaced with the actual project directory; without
    a root a neutral ``(unknown)`` marker is substituted.

    When the project defines `.drcode/git-workflow.md` (team branching /
    commit / review rules), the bundled "Never commit to git" sentence is
    removed (the workflow file governs committing instead) and the workflow
    text is appended under a dedicated header.
    """
    base, git_workflow = load_system_prompt_components(kind, project_root)
    return _join_components(base, git_workflow)


def load_tester_prompt(
    allow_commit: bool = True, project_root: Path | None = None
) -> str:
    """Bundled QA tester system prompt.

    With ``allow_commit=False`` (project setting "Task finalization: Testing")
    the commit-free template is returned; otherwise the default one that
    mentions ``git_commit``. A project may override the template with
    `<root>/.drcode/prompts/tester.md` (or `tester_no_commit.md`). The
    ``{project_path}`` placeholder in the template header is replaced with the
    actual project directory; without a root a neutral ``(unknown)`` marker is
    substituted.

    When the project defines `.drcode/git-workflow.md`, its rules are
    appended: the tester must finalize the task according to the same team
    process the coding agent follows.
    """
    base, git_workflow = load_tester_prompt_components(allow_commit, project_root)
    return _join_components(base, git_workflow)


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
