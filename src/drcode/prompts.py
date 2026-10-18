"""Bundled prompt templates for service tasks (package data).

Templates live in `src/drcode/prompts/`; placeholders inside them are
substituted at load time with values from the running app.
"""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Placeholder in the templates, replaced with the actual project directory.
PROJECT_PATH_PLACEHOLDER = "{project_path}"


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
    """Prompt for the "Update/create AGENTS.md" documentation task.

    The bundled template ends with a paragraph that names the repository
    location; the `{project_path}` placeholder is replaced with the actual
    project directory the TUI is operating on.
    """
    template = (_PROMPTS_DIR / "init_repo.md").read_text(encoding="utf-8")
    return template.replace(PROJECT_PATH_PLACEHOLDER, str(project_root.resolve()))
