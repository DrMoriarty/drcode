from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from .models import Task


class Storage:
    """Persists tasks as JSON files under `<project>/.drcode/tasks/`."""

    def __init__(self, project_root: Path) -> None:
        self.root = project_root
        self.tasks_dir = project_root / ".drcode" / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_gitignored(project_root)

    @staticmethod
    def _ensure_gitignored(project_root: Path) -> None:
        """Add `.drcode/` to the project's .gitignore if one exists."""
        gitignore = project_root / ".gitignore"
        if not gitignore.is_file():
            return
        lines = gitignore.read_text().splitlines()
        if any(
            line.strip().lstrip("!").rstrip("/") == ".drcode"
            or line.strip().endswith("/.drcode")
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ):
            return
        with gitignore.open("a") as f:
            if lines and lines[-1].strip():
                f.write("\n")
            f.write(".drcode/\n")

    def load_all(self) -> list[Task]:
        tasks: list[Task] = []
        for path in sorted(self.tasks_dir.glob("*.json")):
            try:
                tasks.append(Task.model_validate(json.loads(path.read_text())))
            except (json.JSONDecodeError, ValidationError):
                # Corrupted task file: keep a backup, don't crash the app.
                backup = path.with_suffix(".json.bak")
                os.replace(path, backup)
        tasks.sort(key=lambda t: t.created_at)
        return tasks

    def save(self, task: Task) -> None:
        path = self.tasks_dir / f"{task.id}.json"
        data = task.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(dir=self.tasks_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def delete(self, task_id: str) -> None:
        path = self.tasks_dir / f"{task_id}.json"
        path.unlink(missing_ok=True)
