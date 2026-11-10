"""Persistence of long-lived product goals (see drcode.models.Goal).

One JSON file per goal under `<project>/.drcode/goals/`, written atomically —
the same pattern as task storage (`drcode.storage.Storage`): a corrupted file
is backed up (`.json.bak`), never deleted and never fatal. There are
deliberately few goals (months-long roadmap items, a handful per project),
so no indexing or in-memory caching layer is needed: the store keeps them in
a plain dict loaded at startup, mirroring `TaskManager.tasks`.

Two files per goal, split by mutability:

- `<id>.json` — the roadmap content (title/description/notes/type/
  parent/order). Team projects track it in git, so it changes ONLY on an
  explicit roadmap edit (create, the user's/agent's update, ordering).
- `<id>.state.json` — the volatile lifecycle state (`status`,
  `updated_at`), gitignored (see GITIGNORE_ENTRIES). Auto-transitions
  (the task-completion auto-done, the done-goal reactivation) and any
  other status/touch mutation land here, so the tracked file never goes
  dirty on its own. On load the sidecar overrides the main file; a
  missing sidecar is seeded from whatever the main file carries.

The store knows nothing about tasks or the agent layer — the link to
tasks lives solely on the task side (`task.goal_id`); the store only
persists and looks up goals.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from .config import PROJECT_STATE_DIRNAME
from .gitignore import ensure_drcode_gitignored
from .models import Goal, GoalStatus, GoalType

# Suffix of the volatile-state sidecar (gitignored, see GITIGNORE_ENTRIES).
_STATE_SUFFIX = ".state.json"
# Fields of Goal that never enter the (possibly git-tracked) main file:
# the lifecycle state lives in the sidecar only.
_VOLATILE_FIELDS: set[str] = {"status", "updated_at"}
# Keys old goal files carried that this build no longer writes into the
# main file. Their presence in a loaded file marks it for the one-time
# migration: the sidecar is seeded from them and they are stripped.
_LEGACY_MAIN_KEYS = frozenset({"linked_task_ids", "updated_at", "status"})


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

    def _state_path(self, goal_id: str) -> Path:
        return self.goals_dir / f"{goal_id}{_STATE_SUFFIX}"

    def _overlay_state(self, goal: Goal) -> None:
        """Apply the sidecar's lifecycle state over the goal freshly
        parsed from the main file (the sidecar wins — it is the write
        target of every status/touch mutation). A missing or corrupted
        sidecar is silently ignored: the main file's snapshot applies."""
        try:
            raw = json.loads(self._state_path(goal.id).read_text())
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        status = raw.get("status")
        if isinstance(status, str):
            try:
                goal.status = GoalStatus(status)
            except ValueError:
                pass
        updated = raw.get("updated_at")
        if isinstance(updated, str):
            try:
                goal.updated_at = datetime.fromisoformat(updated)
            except ValueError:
                pass

    def _load_one(self, path: Path) -> Goal | None:
        """Parse a single goal file; None on failure. Same resilience
        contract as Storage.load_one: a corrupted file is renamed to
        `.json.bak`, an unreadable/vanished file is silently skipped —
        startup and the agent tools run on the UI/pipeline hot paths.

        Also the migration point: a file carrying legacy keys
        (`linked_task_ids`, `updated_at`, `status` — written by older
        builds that kept the whole lifecycle state in the tracked file)
        loads fine (the model ignores unknown fields), gets its sidecar
        seeded from the parsed state and is rewritten ONCE without the
        stale keys — an expected one-time diff the user commits."""
        try:
            raw = json.loads(path.read_text())
            goal = Goal.model_validate(raw)
        except (json.JSONDecodeError, ValidationError):
            try:
                os.replace(path, path.with_suffix(".json.bak"))
            except OSError:
                pass
            return None
        except OSError:
            return None
        if not isinstance(raw, dict):
            raw = {}
        self._overlay_state(goal)
        legacy = _LEGACY_MAIN_KEYS.intersection(raw)
        if legacy or not self._state_path(goal.id).exists():
            # Seed the sidecar (missing, or taking over the legacy state).
            self._write_state(goal)
        if legacy:
            self._rewrite_main(path, goal)
        return goal

    def load_all(self) -> list[Goal]:
        paths = [
            p
            for p in sorted(self.goals_dir.glob("*.json"))
            if not p.name.endswith(_STATE_SUFFIX)
        ]
        goals = [g for g in (self._load_one(p) for p in paths) if g is not None]
        goals.sort(key=lambda g: g.created_at)
        return goals

    def _atomic_write(self, path: Path, data: str) -> None:
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

    @staticmethod
    def _main_text(goal: Goal) -> str:
        """The tracked file's content: the roadmap fields only — the
        lifecycle state (status/updated_at) never lands here, so an
        auto-transition never dirties a team's checkout."""
        payload = goal.model_dump(mode="json", exclude=_VOLATILE_FIELDS)
        return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    def _write_state(self, goal: Goal) -> None:
        payload = {
            "status": goal.status.value,
            "updated_at": goal.updated_at.isoformat(),
        }
        self._atomic_write(
            self._state_path(goal.id),
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        )

    def _rewrite_main(self, path: Path, goal: Goal) -> None:
        self._atomic_write(path, self._main_text(goal))

    def save(self, goal: Goal) -> None:
        """Persist a goal in full: the roadmap part into the tracked
        `<id>.json`, the lifecycle state into the gitignored sidecar.
        The explicit-edit path (create/update/move/reparent) — an
        auto-transition must go through `update_status`/`save_state`
        and leave the tracked file untouched."""
        self._atomic_write(self.goals_dir / f"{goal.id}.json", self._main_text(goal))
        self._write_state(goal)

    def save_state(self, goal: Goal) -> None:
        """Persist ONLY the volatile lifecycle state (the sidecar) — the
        tracked goal file is not touched."""
        self._write_state(goal)

    def update_status(self, goal_id: str, status: GoalStatus) -> Goal:
        """Apply a status change as a VOLATILE mutation: the in-memory
        status and `updated_at` are stamped and only the sidecar is
        rewritten. This is the auto-transition path (task-completion
        auto-done, done-goal reactivation) — a tracked goal file never
        changes here."""
        goal = self.get(goal_id)
        goal.status = status
        goal.touch()
        self.save_state(goal)
        return goal

    def delete(self, goal_id: str) -> None:
        self.goals.pop(goal_id, None)
        (self.goals_dir / f"{goal_id}.json").unlink(missing_ok=True)
        self._state_path(goal_id).unlink(missing_ok=True)

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

    def children(self, goal_id: str | None) -> list[Goal]:
        """Goals whose parent is `goal_id`; `None` yields the top-level
        set (the tree roots). Unsorted — see display_children."""
        return [g for g in self.goals.values() if g.parent_id == goal_id]

    # -- ordering / tree lookups ------------------------------------------------

    @staticmethod
    def _order_key(goal: Goal) -> tuple[int, float, str]:
        """Display sort key among siblings: an explicit `order` stamp wins
        (the GoalScreen's Ctrl+Up/Down writes it), legacy goals without one
        (written before the field existed) trail in creation order."""
        if goal.order is not None:
            return (0, float(goal.order), goal.id)
        return (1, goal.created_at.timestamp(), goal.id)

    def top_level(self) -> list[Goal]:
        """Goals without a parent, in display order (sidebar tree roots)."""
        return sorted(
            (g for g in self.goals.values() if not g.parent_id),
            key=self._order_key,
        )

    def display_children(self, goal_id: str) -> list[Goal]:
        """Children of `goal_id` in display order (the tree's next level)."""
        return sorted(self.children(goal_id), key=self._order_key)

    def descendant_ids(self, goal_id: str) -> set[str]:
        """All strict descendants of `goal_id` (the whole subtree below
        it, any depth). Used by the reparent picker (a goal cannot become
        its own descendant) and the milestone progress sum."""
        result: set[str] = set()
        stack = [child.id for child in self.children(goal_id)]
        while stack:
            node = stack.pop()
            if node in result:
                continue  # defensive: a manual parent_id cycle cannot loop us
            result.add(node)
            stack.extend(child.id for child in self.children(node))
        return result

    def _next_order(self, parent_id: str | None) -> int:
        """The next free `order` stamp among `parent_id`'s children (0 for
        an empty sibling group) — a new goal lands at the end of its
        sibling list. `children(None)` is exactly the top-level set, so
        both cases share one expression."""
        orders = [g.order for g in self.children(parent_id) if g.order is not None]
        return max(orders, default=-1) + 1

    # -- mutations -----------------------------------------------------------

    def create(
        self,
        title: str,
        *,
        description: str = "",
        notes: str = "",
        parent_id: str | None = None,
        origin: str = "user",
    ) -> Goal:
        """Create and persist a goal. `parent_id`, when given, must point at
        an existing goal (ValueError otherwise — the TUI form and the agent
        tool both surface it). The type is not taken here: a freshly
        captured item starts as an IDEA (the capture form asks for the
        title only; classification is refined later via update_goal).
        `description` is the short 1–2-line summary injected into the
        system prompt; `notes` is the detailed free-form description the
        agent reads via read_goal — it never enters the prompt."""
        cleaned_parent = parent_id.strip() if parent_id else None
        if cleaned_parent and cleaned_parent not in self.goals:
            raise ValueError(f"unknown parent goal id: {cleaned_parent}")
        goal = Goal(
            title=title.strip() or "Untitled goal",
            description=description.strip(),
            notes=notes.strip(),
        )
        goal.parent_id = cleaned_parent or None
        goal.order = self._next_order(goal.parent_id)
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
        for plain in ("title", "description", "notes"):
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
        # A status-only update rewrites the tracked file byte-for-byte
        # (status lives in the sidecar) — no git dirt from a lifecycle
        # click; a real roadmap edit is the user's explicit change.
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

    # -- ordering mutations -----------------------------------------------------

    def _normalize_orders(self, parent_id: str | None) -> list[Goal]:
        """Re-stamp `order` 0..n-1 over the sibling group in display order
        and persist the changed goals. Legacy goals without a stamp (and
        any gaps left by deletions/re-parenting) dissolve into a dense
        sequence, so `move` below only ever swaps two clean integers."""
        ordered = self.display_children(parent_id) if parent_id else self.top_level()
        for index, goal in enumerate(ordered):
            if goal.order != index:
                goal.order = index
                goal.touch()
                self.save(goal)
        return ordered

    def move(self, goal_id: str, *, up: bool) -> Goal:
        """Move `goal_id` one position up/down among its siblings
        (Ctrl+Up/Down on the GoalScreen). Returns the goal; moving past
        the end of the sibling list is a no-op. Like queue_seq re-stamps,
        the sibling group is normalized first so the swap is exact."""
        goal = self.get(goal_id)
        siblings = self._normalize_orders(goal.parent_id)
        index = next(i for i, g in enumerate(siblings) if g.id == goal_id)
        other = index - 1 if up else index + 1
        if not 0 <= other < len(siblings):
            return goal
        siblings[index].order, siblings[other].order = (
            siblings[other].order,
            siblings[index].order,
        )
        for moved in (siblings[index], siblings[other]):
            moved.touch()
            self.save(moved)
        return goal

    def reparent(self, goal_id: str, new_parent: str | None) -> Goal:
        """Move `goal_id` under `new_parent` (None = make top-level). The
        parent must exist and must not be inside the goal's own subtree
        (the decomposition stays a tree); the goal lands at the end of the
        new sibling group. ValueError on violation — the picker surfaces
        it; the acyclicity check is the same _ensure_acyclic update uses."""
        goal = self.get(goal_id)
        cleaned = new_parent.strip() if new_parent else None
        if cleaned == goal.parent_id:
            return goal  # no-op: already there
        if cleaned is not None:
            if cleaned not in self.goals:
                raise ValueError(f"unknown parent goal id: {cleaned}")
            self._ensure_acyclic(goal_id, cleaned)
        goal.parent_id = cleaned or None
        goal.order = self._next_order(goal.parent_id)
        goal.touch()
        self.save(goal)
        return goal
