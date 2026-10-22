from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from .config import PROJECT_STATE_DIRNAME
from .gitignore import ensure_drcode_gitignored
from .models import Task


class Storage:
    """Persists tasks as JSON files under `<project>/.drcode/tasks/`."""

    def __init__(self, project_root: Path) -> None:
        self.root = project_root
        self.tasks_dir = project_root / PROJECT_STATE_DIRNAME / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        # `.drcode/` holds transcripts and the project config — never
        # commit it (the helper is a no-op outside a git repository).
        ensure_drcode_gitignored(project_root)

    def load_one(self, path: Path) -> Task | None:
        """Parse a single task file; None on failure.

        A corrupted file is backed up (`.json.bak`), never fatal; a file
        that vanished between the glob and the read (removed externally
        mid-scan) is silently skipped. Never raises on unreadable input —
        callers run inside the pipeline's hot paths.
        """
        try:
            return Task.model_validate(json.loads(path.read_text()))
        except (json.JSONDecodeError, ValidationError):
            # Corrupted task file: keep a backup, don't crash the app.
            try:
                os.replace(path, path.with_suffix(".json.bak"))
            except OSError:
                pass
            return None
        except OSError:
            # Unreadable / disappeared between the glob and the read.
            return None

    def load_all(self) -> list[Task]:
        tasks = [
            t
            for t in (self.load_one(p) for p in sorted(self.tasks_dir.glob("*.json")))
            if t is not None
        ]
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
