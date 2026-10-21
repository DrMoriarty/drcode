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

from pathlib import Path

from .models import TaskKind

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


def load_system_prompt(kind: TaskKind, project_root: Path | None = None) -> str:
    """Bundled system prompt for a coding task of the given kind.

    Each kind gets its own `.md` template in `prompts/system/` (feature tasks
    encourage refactoring, bug tasks demand minimal changes, research tasks
    answer questions / update docs). The ``{project_path}`` placeholder in the
    template header is replaced with the actual project directory; without a
    root a neutral ``(unknown)`` marker is substituted.
    """
    name = {
        TaskKind.FEATURE: "feature.md",
        TaskKind.BUG: "bug.md",
        TaskKind.RESEARCH: "research.md",
    }[kind]
    template = (SYSTEM_PROMPTS_DIR / name).read_text(encoding="utf-8")
    return _substitute_project_path(template, project_root)


def load_tester_prompt(
    allow_commit: bool = True, project_root: Path | None = None
) -> str:
    """Bundled QA tester system prompt.

    With ``allow_commit=False`` (project setting "Task finalization: Testing")
    the commit-free template is returned; otherwise the default one that
    mentions ``git_commit``. The ``{project_path}`` placeholder in the template
    header is replaced with the actual project directory; without a root a
    neutral ``(unknown)`` marker is substituted.
    """
    name = "tester_no_commit.md" if not allow_commit else "tester.md"
    template = (SYSTEM_PROMPTS_DIR / name).read_text(encoding="utf-8")
    return _substitute_project_path(template, project_root)


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
