"""Persistence of long-lived product goals (see drcode.models.Goal).

One JSON file per goal under `<project>/.drcode/goals/`, written atomically —
the same pattern as task storage (`drcode.storage.Storage`): a corrupted file
is backed up (`.json.bak`), never deleted and never fatal. There are
deliberately few goals (months-long roadmap items, a handful per project),
so no indexing or in-memory caching layer is needed: the store keeps them in
a plain dict loaded at startup, mirroring `TaskManager.tasks`.

The store knows nothing about tasks or the agent layer — linking is the
TaskManager's job (`create(goal_id=...)` appends to `linked_task_ids`);
the store only persists and looks up goals.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from .config import PROJECT_STATE_DIRNAME
from .gitignore import ensure_drcode_gitignored
from .models import Goal, GoalStatus, GoalType


class GoalStore:
    """Persists goals as JSON files under `<project>/.drcode/goals/`."""

    def __init__(self, project_root: Path) -> None:
        self.root = project_root
        self.goals_dir = project_root / PROJECT_STATE_DIRNAME / "goals"
        self.goals_dir.mkdir(parents=True, exist_ok=True)
        # `.drcode/` holds transcripts and the project config — never commit
        # it (a no-op outside a git repository). Usually already ensured by
        # Storage, but goals can be used without tasks ever existing.
        ensure_drcode_gitignored(project_root)
        self.goals: dict[str, Goal] = {g.id: g for g in self.load_all()}

    def _load_one(self, path: Path) -> Goal | None:
        """Parse a single goal file; None on failure. Same resilience
        contract as Storage.load_one: a corrupted file is renamed to
        `.json.bak`, an unreadable/vanished file is silently skipped —
        startup and the agent tools run on the UI/pipeline hot paths."""
        try:
            return Goal.model_validate(json.loads(path.read_text()))
        except (json.JSONDecodeError, ValidationError):
            try:
                os.replace(path, path.with_suffix(".json.bak"))
            except OSError:
                pass
            return None
        except OSError:
            return None

    def load_all(self) -> list[Goal]:
        goals = [
            g
            for g in (self._load_one(p) for p in sorted(self.goals_dir.glob("*.json")))
            if g is not None
        ]
        goals.sort(key=lambda g: g.created_at)
        return goals

    def save(self, goal: Goal) -> None:
        path = self.goals_dir / f"{goal.id}.json"
        data = goal.model_dump_json(indent=2)
        # The state dir can vanish mid-session (deleted externally):
        # recreate instead of crashing the caller.
        self.goals_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.goals_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def delete(self, goal_id: str) -> None:
        self.goals.pop(goal_id, None)
        (self.goals_dir / f"{goal_id}.json").unlink(missing_ok=True)

    # -- lookups -------------------------------------------------------------

    def get(self, goal_id: str) -> Goal:
        """The goal by id; KeyError when absent — callers turn it into a
        user/agent-facing message (an unknown goal id is a fixable mistake,
        never a crash)."""
        return self.goals[goal_id]

    def __contains__(self, goal_id: object) -> bool:
        return goal_id in self.goals

    def active(self) -> list[Goal]:
        """All ACTIVE goals, creation order — the set the research agent's
        system prompt carries (done/archived stay out of the context)."""
        return [g for g in self.goals.values() if g.status is GoalStatus.ACTIVE]

    def children(self, goal_id: str) -> list[Goal]:
        return [g for g in self.goals.values() if g.parent_id == goal_id]

    # -- mutations -----------------------------------------------------------

    def create(
        self,
        title: str,
        *,
        description: str = "",
        parent_id: str | None = None,
        origin: str = "user",
    ) -> Goal:
        """Create and persist a goal. `parent_id`, when given, must point at
        an existing goal (ValueError otherwise — the TUI form and the agent
        tool both surface it). The type is not taken here: a freshly
        captured item starts as an IDEA (the capture form asks for the
        title only; classification is refined later via update_goal)."""
        cleaned_parent = parent_id.strip() if parent_id else None
        if cleaned_parent and cleaned_parent not in self.goals:
            raise ValueError(f"unknown parent goal id: {cleaned_parent}")
        goal = Goal(
            title=title.strip() or "Untitled goal", description=description.strip()
        )
        goal.parent_id = cleaned_parent or None
        self.goals[goal.id] = goal
        self.save(goal)
        return goal

    def update(self, goal_id: str, **fields: object) -> Goal:
        """Apply partial updates to a goal and persist it. Only the model's
        own fields are accepted (a typo raises immediately instead of
        silently doing nothing); invalid enum values and unknown parent ids
        raise ValueError → the agent tool converts to a retry prompt. The
        enum checks are explicit: pydantic does not validate on assignment
        by default, and a raw string would otherwise be stored verbatim
        (breaking the model_dump_json round-trip silently)."""
        goal = self.get(goal_id)
        allowed = set(Goal.model_fields)
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown goal fields: {', '.join(sorted(unknown))}")
        if "type" in fields:
            fields["type"] = self._validated_enum(fields["type"], GoalType, "type")
        if "status" in fields:
            fields["status"] = self._validated_enum(
                fields["status"], GoalStatus, "status"
            )
        for plain in ("title", "description"):
            if plain in fields and not isinstance(fields[plain], str):
                raise ValueError(f"goal {plain} must be a string")
        if "parent_id" in fields:
            parent = fields["parent_id"]
            parent = (
                parent.strip() if isinstance(parent, str) and parent.strip() else None
            )
            if parent is not None and parent not in self.goals:
                raise ValueError(f"unknown parent goal id: {parent}")
            if parent is not None:
                self._ensure_acyclic(goal_id, parent)
            fields["parent_id"] = parent
        for key, value in fields.items():
            setattr(goal, key, value)
        goal.touch()
        self.save(goal)
        return goal

    @staticmethod
    def _validated_enum(
        value: object, enum_cls: type[GoalType | GoalStatus], what: str
    ) -> GoalType | GoalStatus:
        """Coerce an enum field value: an enum member passes through, a
        string is matched case-insensitively against the values, anything
        else (or an unknown value) raises ValueError."""
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, str):
            try:
                return enum_cls(value.strip().lower())
            except ValueError:
                pass
        valid = ", ".join(member.value for member in enum_cls)
        raise ValueError(f"invalid goal {what}: {value!r} — expected one of: {valid}")

    def _ensure_acyclic(self, goal_id: str, new_parent: str) -> None:
        """A sub-goal must not become its own ancestor: walking UP the
        parent chain from `new_parent` must never reach `goal_id`."""
        seen: set[str] = set()
        node: str | None = new_parent
        while node is not None and node not in seen:
            if node == goal_id:
                raise ValueError(
                    f"parent {new_parent} is a descendant of {goal_id} — "
                    "the decomposition must stay a tree, not a cycle"
                )
            seen.add(node)
            node = self.goals[node].parent_id if node in self.goals else None
