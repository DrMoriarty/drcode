"""Bundled prompt templates for service tasks (package data).

Templates live in `src/drcode/prompts/`; placeholders inside them are
substituted at load time with values from the running app.
"""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Placeholder in the templates, replaced with the actual project directory.
PROJECT_PATH_PLACEHOLDER = "{project_path}"


def load_init_repo_prompt(project_root: Path) -> str:
    """Prompt for the "Обновить/создать AGENTS.md" documentation task.

    The bundled template ends with a paragraph that names the repository
    location; the `{project_path}` placeholder is replaced with the actual
    project directory the TUI is operating on.
    """
    template = (_PROMPTS_DIR / "init_repo.md").read_text(encoding="utf-8")
    return template.replace(PROJECT_PATH_PLACEHOLDER, str(project_root.resolve()))
