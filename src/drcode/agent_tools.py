"""Tool bodies and service helpers behind the coding/tester agents.

This module holds the *logic* of the agent-facing tools that are too
substantial to inline in the builders: the self-managed todo checklist
(``todo_create`` / ``todo_update`` plus the dynamic-instructions hook),
dependency editing for the research agent, edited-file tracking and the
commit/push helpers, and the tester's retry-here verdict handler. The
builders in `agent_build` register thin pydantic-ai tool wrappers that
delegate here (usually via `_run_tool`, so fixable ValueErrors become
ModelRetry prompts for the model).

The deps types are imported only for typing: the helpers receive them at
runtime from the tool wrappers and never construct or isinstance-check
them, so a module-level import would be a needless cycle.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import RunContext

from . import tools
from .merge_resolver import _project_python, _venv_env, _worktree_pythonpath_env
from .models import (
    Goal,
    GoalStatus,
    GoalType,
    Task,
    TaskEvent,
    TaskKind,
    TaskStatus,
    TodoItem,
    VerificationStatus,
    render_todos,
)
from .service_events import log_service_event
from .task_env import WORKTREES_SUBDIR, branch_for

if TYPE_CHECKING:
    from .agent_build import AgentDeps, TesterDeps

__all__ = [
    "REPORT_DESC_MIN_CHARS",
    "TODO_MIN_ITEMS",
    "_commit_tracked_files",
    "_create_goal",
    "_create_report_task",
    "_create_tester_bug_report",
    "_deps_suffix",
    "_git_push",
    "_list_goals",
    "_read_goal",
    "_retry_failed_here",
    "_run_project_check",
    "_set_task_dependencies",
    "_todo_create",
    "_todo_update",
    "_track_edited_file",
    "_update_goal",
    "goals_dynamic_instructions",
    "goals_instructions",
    "load_agents_md",
    "todo_dynamic_instructions",
    "todo_instructions",
]


# Minimum checklist size: a shorter list is the degenerate anti-pattern
# (OpenCode-style one-item todos on trivial work) — the guard rejects the
# call and the ModelRetry corrects the model in-session.
TODO_MIN_ITEMS = 3


def _todo_create(deps: AgentDeps, items: list[str]) -> str:
    """Create/replace the task's todo checklist (feature tasks only).

    Raises ValueError (→ ModelRetry via _run_tool) for a degenerate list:
    the agent must not maintain a checklist for single-step work."""
    cleaned = [item.strip() for item in items if item and item.strip()]
    if len(cleaned) < TODO_MIN_ITEMS:
        raise ValueError(
            f"todo_create needs at least {TODO_MIN_ITEMS} items, got "
            f"{len(cleaned)}: for single-step work do not use the checklist, "
            "just work."
        )
    task = deps.manager.get(deps.task_id)
    replaced = bool(task.todos)
    task.todos = [TodoItem(text=text) for text in cleaned]
    task.touch()
    deps.manager.storage.save(task)
    head = "Checklist replaced" if replaced else "Checklist created"
    return (
        f"{head} with {len(cleaned)} items:\n{render_todos(task.todos)}\n\n"
        "Keep the list accurate: call todo_update immediately after "
        "finishing each item, before starting the next one."
    )


def _todo_update(deps: AgentDeps, targets: list[int | str], done: bool) -> str:
    """Check/uncheck checklist items by 1-based index or text fragment.

    Raises ValueError (→ ModelRetry via _run_tool) when a target matches
    nothing — the model then corrects itself instead of silently losing an
    item."""
    task = deps.manager.get(deps.task_id)
    if not task.todos:
        raise ValueError("This task has no checklist — call todo_create first.")
    matched: set[int] = set()
    for target in targets:
        if isinstance(target, int):
            index = target
        elif isinstance(target, str) and target.strip().isdigit():
            index = int(target.strip())
        else:
            fragment = str(target).strip().lower()
            hits = [
                i for i, item in enumerate(task.todos) if fragment in item.text.lower()
            ]
            if not hits:
                raise ValueError(
                    f"No checklist item matches {target!r}; the current "
                    f"items are:\n{render_todos(task.todos)}"
                )
            matched.update(hits)
            continue
        if not 1 <= index <= len(task.todos):
            raise ValueError(
                f"todo index {index} is out of range 1..{len(task.todos)}."
            )
        matched.add(index - 1)
    for i in sorted(matched):
        task.todos[i].done = done
    task.touch()
    deps.manager.storage.save(task)
    return (
        f"Checklist updated ({len(matched)} item(s) marked "
        f"{'done' if done else 'not done'}):\n{render_todos(task.todos)}"
    )


def _set_task_dependencies(deps: AgentDeps, task_id: str, dep_ids: list[str]) -> str:
    """Replace a task's depends_on list (research agent only).

    Raises ValueError (→ ModelRetry via _run_tool) when the task does not
    exist, is already under way (only QUEUED/HOLD/DRAFT tasks are editable —
    the same rule as the command palette; a draft is not started either, the
    user is still composing it) or a dependency id is unknown /
    would form a cycle."""
    if task_id not in deps.manager.tasks:
        raise ValueError(f"unknown task id: {task_id}")
    task = deps.manager.get(task_id)
    if task.status not in (TaskStatus.QUEUED, TaskStatus.HOLD, TaskStatus.DRAFT):
        raise ValueError(
            f"task #{task_id} is {task.status.value} — dependencies can only "
            "be set on queued, held or draft tasks (before development starts)."
        )
    deps.manager.set_dependencies(task_id, dep_ids)
    if task.depends_on:
        return (
            f"Task #{task_id} now waits for: "
            + ", ".join(f"#{d}" for d in task.depends_on)
            + " — it is not promoted until every listed task is COMPLETED "
            "and (under worktree isolation) merged."
        )
    return f"Task #{task_id} has no dependencies — it is freely promotable."


def _deps_suffix(task: Task) -> str:
    """The dependency note appended to a create_* tool result."""
    if not task.depends_on:
        return ""
    return (
        " It waits for: "
        + ", ".join(f"#{d}" for d in task.depends_on)
        + " (not promoted until all are COMPLETED and merged)."
    )


def _create_report_task(
    deps: AgentDeps | TesterDeps,
    title: str,
    description: str,
    *,
    kind: TaskKind,
    creator: str,
    depends_on: list[str] | None = None,
    goal_id: str | None = None,
) -> tuple[Task, str]:
    """Shared body of the agents' task-creation tools (research:
    create_bug_report / create_feature_request; tester: create_bug_report).

    Honors `deps.agent_creates_drafts`: the new task is then created in
    the DRAFT status — it never enters the queue by itself, the user must
    open it in the new-task editor and finish it (a service event on the
    draft tells them so). The headless CLI and the GitHub Issues import
    do not run through here: they create full tasks unconditionally.
    origin="system" — the tool (not the user) created the task, so
    creation listeners (e.g. the voice announcements) can tell
    agent-created reports from hand-typed ones. Returns the created task
    and the leading part of the tool-result text (the callers append
    their dependency/goal suffixes)."""
    draft = deps.agent_creates_drafts
    task = deps.manager.create(
        title,
        description,
        kind=kind,
        origin="system",
        draft=draft,
        depends_on=depends_on,
        goal_id=goal_id,
    )
    label = "Bug report" if kind is TaskKind.BUG else "Feature request"
    if draft:
        task.events.append(
            TaskEvent(
                kind="system",
                text=(
                    f"Created as a draft by the {creator} — open the task "
                    "and finish it to put it into the queue."
                ),
            )
        )
        task.touch()
        deps.manager.storage.save(task)
        return (
            task,
            (
                f"{label} #{task.id} '{task.title}' created as a draft — the "
                "user must finish it before it enters the queue."
            ),
        )
    placement = (
        "at the head of the queue"
        if kind is TaskKind.BUG
        else "at the tail of the queue"
    )
    return task, f"{label} #{task.id} '{task.title}' created {placement}."


# Minimum meaningful bug-report description (chars): anything shorter cannot
# say what is wrong, where and how it manifests. A degraded model at the end
# of a long testing round emits stub calls; a length floor is the mechanical
# line of defense no prompt edit can provide at the context limit.
REPORT_DESC_MIN_CHARS = 40

# Template stubs a degraded call uses as the WHOLE description, matched after
# normalization (lowercase, whitespace collapsed, surrounding punctuation
# stripped). Only full-match stubs are rejected: a title "placeholder" with a
# real description stays allowed, and so does a genuine description that
# merely mentions the word (e.g. a bug about placeholder rendering).
_REPORT_DESC_STUBS = frozenset(
    {
        "placeholder",
        "placeholder report",
        "placeholder bug report",
        "placeholder will not file",
        "placeholder will not be filed",
        "stub",
        "todo",
        "tbd",
        "n/a",
        "na",
        "none",
        "no description",
        "description",
        "empty",
        "test",
        "testing",
        "bug",
        "bug report",
        "issue",
        "report",
        "описание",
        "нет описания",
        "заглушка",
        "тест",
        "баг",
        "багрепорт",
    }
)


def _validate_bug_report_args(title: str, description: str) -> None:
    """Mechanical guard on the tester's create_bug_report arguments.

    Raises ValueError (→ ModelRetry via _run_tool) when the arguments are
    empty, degenerately short or a template stub: a degraded-model call
    must never become a real task (the #f668795c incident — title=
    "placeholder", description="placeholder - will not file" reached the
    queue as a phantom bug report). The retry tells the model to re-call
    the tool with actual content; nothing is created in between."""
    if not title.strip():
        raise ValueError(
            "create_bug_report: title is empty — repeat the call with a "
            "short imperative summary of the defect."
        )
    desc = description.strip()
    if not desc:
        raise ValueError(
            "create_bug_report: description is empty — repeat the call "
            "with a description of what is wrong, where (files/lines) and "
            "how it manifests."
        )
    normalized = re.sub(r"\s+", " ", desc.lower()).strip(" .!?,;:'\"`()[]<>-*—–…~")
    # A second view with punctuation between words turned into spaces, so
    # "placeholder - will not file" matches its dash-free stub entry (n/a
    # keeps its own entry in both spellings).
    collapsed = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", normalized)).strip()
    if normalized in _REPORT_DESC_STUBS or collapsed in _REPORT_DESC_STUBS:
        raise ValueError(
            f"create_bug_report: description {description!r} is a "
            "placeholder, not a bug report — repeat the call with a real "
            "description of the defect, or do not file a report at all."
        )
    if len(desc) < REPORT_DESC_MIN_CHARS:
        raise ValueError(
            f"create_bug_report: description is too short ({len(desc)} "
            f"chars, minimum {REPORT_DESC_MIN_CHARS}) — describe what is "
            "wrong, where (files/lines) and how it manifests."
        )


def _create_tester_bug_report(deps: TesterDeps, title: str, description: str) -> str:
    """Body of the tester's create_bug_report tool (the builder registers
    a thin wrapper that runs this via `_run_tool`, so a ValueError from the
    validation below becomes a ModelRetry prompt).

    Validates the arguments mechanically first — a degraded-model stub call
    is refused before anything is created — then files the report with a
    dependency on the verified task: it is not promoted until that task is
    COMPLETED and (under worktree isolation) merged, so the fix agent gets a
    worktree where the observed defect actually exists (the c8623e0c
    incident: a fix agent editing pyproject "blindly"). A task completed as
    FAILED after the fix-round cap is COMPLETED too, so the gate resolves on
    that path as well."""
    _validate_bug_report_args(title, description)
    task_id = deps.task_id
    origin = (
        f"\n\nOrigin: filed by the tester of task #{task_id} — the defect"
        " was observed in that task's verification environment"
    )
    if (
        deps.root.resolve()
        == (deps.manager.storage.root / WORKTREES_SUBDIR / task_id).resolve()
    ):
        origin += f" (its worktree, branch {branch_for(task_id)})"
    origin += "."
    bug, text = _create_report_task(
        deps,
        title,
        description + origin,
        kind=TaskKind.BUG,
        creator="tester",
        depends_on=[task_id],
    )
    return text + _deps_suffix(bug)


# Context discipline of the goal injection: the system prompt carries only
# the title and the first ~160 chars of each description (1–2 lines) — the
# full text stays behind read_goal, like todo lists stay behind their own
# compact rendering. There are deliberately few active goals, but a roadmap
# description can be a couple of paragraphs; never pour it into the prompt.
_GOAL_PROMPT_CAP = 160


def goals_instructions(active: list[Goal]) -> str | None:
    """The compact "active goals" block for the research agent's dynamic
    instructions (None when there are no active goals — the block is then
    excluded from the request entirely, same contract as
    todo_instructions).

    Registered as DYNAMIC instructions (evaluated per model request), so a
    goal created or updated mid-run by this very agent (create_goal /
    update_goal) is visible to the next request of the same activation —
    run-level instructions freeze at `agent.iter` start and would miss it.
    Only status=active goals are listed: done/archived ones leave the
    context (and the agent's planning surface) as soon as they are closed.
    """
    if not active:
        return None
    lines = [
        "# Product goals (current roadmap)",
        "",
        "Long-lived goals the user is developing this project towards. Plan",
        "any roadmap-adjacent work AGAINST them; details via read_goal:",
        "",
    ]
    lines.extend(f"- {g.summary_line(_GOAL_PROMPT_CAP)}" for g in active)
    lines.append(
        "\nWhen your work serves one of these goals, say so explicitly in "
        "the answer (a reference to a specific goal) and link any task you "
        "create to that goal via its goal_id."
    )
    return "\n".join(lines)


def goals_dynamic_instructions(ctx: RunContext[AgentDeps]) -> str | None:
    """Dynamic instructions hook for the research agent: recompute the
    active-goals block from the manager's CURRENT goal state for every
    model request (see goals_instructions). Registered on the coding agent
    for all kinds — only research tasks get goals, everything else yields
    None (structurally: coding agents and the tester are never goal-aware
    on this stage)."""
    task = ctx.deps.manager.get(ctx.deps.task_id)
    if task.kind is not TaskKind.RESEARCH:
        return None
    return goals_instructions(ctx.deps.manager.goals.active())


def _list_goals(deps: AgentDeps, include_done: bool = False) -> str:
    """Compact listing of the goal registry for the research agent.

    Active goals first; done/archived ones only on explicit request (the
    caller usually plans forward, not backward). One line per goal — the
    same summary_line primitive the system prompt uses, so the listing
    stays cheap even when the registry grows."""
    goals = sorted(deps.manager.goals.goals.values(), key=lambda g: g.created_at)
    if not include_done:
        goals = [g for g in goals if g.status is GoalStatus.ACTIVE]
    if not goals:
        return "No goals in the registry yet (create one with create_goal)."
    lines = [g.summary_line(160) for g in goals]
    if include_done:
        return "All goals:\n" + "\n".join(lines)
    hidden = len(deps.manager.goals.goals) - len(goals)
    suffix = (
        f"\n(+{hidden} done/archived not shown; call list_goals again with include_done=True.)"
        if hidden
        else ""
    )
    return "\n".join(lines) + suffix


def _read_goal(deps: AgentDeps, goal_id: str) -> str:
    """Full goal details for the research agent: description (untruncated),
    decomposition (parent/children), linked tasks with their live status.
    Raises ValueError (→ ModelRetry via _run_tool) on an unknown id."""
    try:
        goal = deps.manager.goals.get(goal_id.strip())
    except KeyError:
        known = ", ".join(sorted(deps.manager.goals.goals)) or "(none)"
        raise ValueError(f"unknown goal id: {goal_id}. Known goals: {known}") from None
    store = deps.manager.goals
    lines = [
        f"Goal #{goal.id} ({goal.type.value}, {goal.status.value}): {goal.title}",
        f"Created {goal.created_at:%Y-%m-%d}, updated {goal.updated_at:%Y-%m-%d}",
    ]
    if goal.description.strip():
        lines += ["", "Description:", goal.description.strip()]
    if goal.parent_id is not None:
        parent = store.goals.get(goal.parent_id)
        label = (
            f"#{parent.id} {parent.title_display}"
            if parent
            else f"#{goal.parent_id} (deleted)"
        )
        lines.append(f"\nParent goal: {label}")
    children = store.children(goal.id)
    if children:
        lines.append("\nSub-goals:")
        lines.extend(f"- {c.summary_line()}" for c in children)
    if goal.linked_task_ids:
        lines.append("\nLinked tasks:")
        for tid in goal.linked_task_ids:
            t = deps.manager.tasks.get(tid)
            if t is None:
                lines.append(f"- #{tid} (deleted)")
            else:
                lines.append(
                    f"- #{tid} ({t.kind.value}, {t.status.value}): {t.title_display}"
                )
    return "\n".join(lines)


_GOAL_TYPES = ", ".join(t.value for t in GoalType)
_GOAL_STATUSES = ", ".join(s.value for s in GoalStatus)


def _parse_goal_type(value: str) -> GoalType:
    normalized = value.strip().lower()
    try:
        return GoalType(normalized)
    except ValueError:
        raise ValueError(
            f"invalid goal type: {value!r} — expected one of: {_GOAL_TYPES}"
        ) from None


def _parse_goal_status(value: str) -> GoalStatus:
    normalized = value.strip().lower()
    try:
        return GoalStatus(normalized)
    except ValueError:
        raise ValueError(
            f"invalid goal status: {value!r} — expected one of: {_GOAL_STATUSES}"
        ) from None


def _create_goal(
    deps: AgentDeps,
    title: str,
    description: str = "",
    goal_type: str = GoalType.IDEA.value,
    parent_id: str | None = None,
) -> str:
    """Agent-side goal creation (research role only).

    Raises ValueError (→ ModelRetry via _run_tool) for a bad type or an
    unknown parent id. Every agent-made change is logged as a service event
    in the task's chat — never silent."""
    parsed_type = _parse_goal_type(goal_type)
    goal = deps.manager.goals.create(
        title,
        description=description,
        parent_id=parent_id,
        origin="agent",
    )
    goal.type = parsed_type
    goal.touch()
    deps.manager.goals.save(goal)
    note = f"цель {goal.title_display} ({goal.id}) создана агентом: {goal.type.value}"
    if parent_id:
        note += f", подцель #{parent_id}"
    log_service_event(deps, "goals", note)
    return (
        f"Goal #{goal.id} ({goal.type.value}) created: {goal.title_display}.\n"
        "Link the tasks that serve it via create_bug_report / "
        "create_feature_request with goal_id."
    )


def _update_goal(
    deps: AgentDeps,
    goal_id: str,
    *,
    title: str | None = None,
    description: str | None = None,
    goal_type: str | None = None,
    status: str | None = None,
    parent_id: str | None = None,
) -> str:
    """Agent-side partial goal update (research role only).

    Only the explicitly passed fields change; `parent_id=""` detaches the
    goal from its parent. Raises ValueError (→ ModelRetry via _run_tool)
    for an unknown goal id, bad enum values, unknown parent or a cycle in
    the decomposition. The change is logged as a service event — never
    silent."""
    goal_id = goal_id.strip()
    if goal_id not in deps.manager.goals:
        raise ValueError(f"unknown goal id: {goal_id}")
    fields: dict[str, object] = {}
    changes: list[str] = []
    if title is not None and title.strip():
        fields["title"] = title.strip()
        changes.append("название")
    if description is not None:
        fields["description"] = description.strip()
        changes.append("описание")
    if goal_type is not None:
        parsed_type = _parse_goal_type(goal_type)
        fields["type"] = parsed_type
        changes.append(f"тип → {parsed_type.value}")
    if status is not None:
        parsed_status = _parse_goal_status(status)
        fields["status"] = parsed_status
        changes.append(f"статус → {parsed_status.value}")
    if parent_id is not None:
        fields["parent_id"] = parent_id
        changes.append(
            "привязана к родителю #" + parent_id.strip()
            if parent_id.strip()
            else "отделена от родителя"
        )
    if not fields:
        raise ValueError(
            "update_goal: nothing to change — pass at least one field "
            "(title, description, goal_type, status, parent_id)."
        )
    goal = deps.manager.goals.update(goal_id, **fields)
    log_service_event(
        deps,
        "goals",
        f"цель {goal.title_display} ({goal.id}) уточнена агентом: {', '.join(changes)}",
    )
    return (
        f"Goal #{goal.id} updated ({', '.join(changes)}). "
        f"Current state: ({goal.type.value}, {goal.status.value}) "
        f"{goal.title_display}."
    )


def todo_instructions(task: Task) -> str | None:
    """The task's current checklist state as an instruction block (feature
    tasks with a non-empty list only — nothing else has todo tools or a
    checklist).

    This is the key anti-"forgot the list" measure. It is injected as
    DYNAMIC instructions (see `todo_dynamic_instructions`), so it is
    recomputed for EVERY model request of every activation: the model sees
    the live list state on the very first request after a `todo_create`
    call within the same uninterrupted run, not only after a restart or a
    compaction. A synthetic user message would pile up one stale copy per
    activation in the persisted transcript; instructions never touch the
    transcript. The list itself lives in `task.todos`; if it were only
    prompt material, an interrupted attempt or a compaction could drop it."""
    if task.kind is not TaskKind.FEATURE or not task.todos:
        return None
    unchecked = len(task.unchecked_todos)
    lines = [
        "# Task checklist (current state)",
        "",
        "Your todo checklist for this task, as it stands right now:",
        "",
        render_todos(task.todos),
        "",
        (
            f"{unchecked} of {len(task.todos)} items are unchecked. Keep the "
            "list accurate with todo_update: check an item off immediately "
            "after finishing it, before starting the next one. Do not finish "
            "the task while items remain unchecked unless the user "
            "explicitly dropped them."
        ),
    ]
    return "\n".join(lines)


def todo_dynamic_instructions(ctx: RunContext[AgentDeps]) -> str | None:
    """Dynamic instructions hook: recompute the checklist block from the
    task's CURRENT state for every model request. Registered on the coding
    agent in `build_agent` — pydantic-ai evaluates it per request, so a
    checklist created (or updated) mid-activation is visible to the next
    request of the same run, which run-level `instructions=` (frozen at
    `agent.iter` start) could never achieve."""
    return todo_instructions(ctx.deps.manager.get(ctx.deps.task_id))


def load_agents_md(root: Path) -> str:
    """Return the project's AGENTS.md content, or an empty string if absent."""
    try:
        return (root / "AGENTS.md").read_text(encoding="utf-8")
    except OSError:
        return ""


def _retry_failed_here(deps: TesterDeps, report: str) -> str:
    """Apply the tester's `retry_here` failure verdict (set_verification with
    success=False, retry_here=True): requeue the task into development with
    the failure report as its next input — unless the fix-round cap
    (project setting `max_dev_rounds`) is reached, in which case the task
    completes as FAILED and the tester must create the bug report. A stale
    verdict (the task is no longer TESTING — it already returned to
    development or was cancelled/re-queued while the call was in flight)
    changes nothing and is reported as ignored. The applied outcomes are
    logged into the task's service-event history by the manager
    (requeue_failed_fix); returns the tool-result text telling the
    model which path was taken."""
    task = deps.manager.get(deps.task_id)
    if task.status is not TaskStatus.TESTING:
        return (
            "Verification label ignored: the task is no longer being "
            "tested (it already returned to development) — do NOT call "
            "create_bug_report."
        )
    rounds_done = task.fix_rounds + 1
    max_rounds = max(1, deps.max_dev_rounds)
    deps.manager.requeue_failed_fix(deps.task_id, report, max_rounds=max_rounds)
    task = deps.manager.get(deps.task_id)
    if task.status is TaskStatus.QUEUED and task.fix_rounds == rounds_done:
        return (
            "Verification label set: failed. The failure is caused by this "
            "task's own changes: the task returns to development in its own "
            f"worktree (round {task.fix_rounds + 1}/{max_rounds} of its fix "
            "loop). The report was queued as the task's next input — do NOT "
            "call create_bug_report."
        )
    if (
        task.status is TaskStatus.COMPLETED
        and task.verification is VerificationStatus.FAILED
    ):
        return (
            "Verification label set: failed; task completed. The bug "
            f"survived {max_rounds} dedicated development rounds — call "
            "create_bug_report now with a clear title and a description of "
            "what still fails (the exact command and the error output)."
        )
    # Defensive fallback (e.g. the user cancelled the task in the tiny
    # window between the TESTING check above and the requeue): the stale
    # verdict changed nothing.
    return "Verification label ignored: the task is no longer being tested."


# Module names run_check accepts: dotted Python identifiers only — the name
# is spliced into `python -m <module>` for a resolved interpreter, never into
# a shell, so the check keeps the tool-selection out of the model's hands.
_MODULE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


def _run_project_check(
    root: Path, module: str, args: list[str], timeout: float | None = None
) -> str:
    """Run a check tool from the project's OWN environment: `python -m
    <module> <args>` with the interpreter whose environment actually holds
    the project's tooling (the plan of `merge_resolver._project_python`:
    the worktree's synced .venv → `uv run` → the interpreter drcode runs
    under, the module must really import there).

    The tester is an LLM with a free-form run_bash, so nothing structural
    stops it from invoking a bare `ruff`/`pytest` and getting whichever
    binary PATH supplies — a different version with a different default
    ruleset, reporting phantom failures the project's own gate never
    produces (the 3ea596cb class: DTZ005 from an old PATH ruff). This
    helper takes the environment choice away from the shell entirely.
    Returns "exit=<code>" plus the output; when the project has no such
    tool the check is reported as unavailable — the model must treat it
    as skipped, never fall back to a PATH binary."""
    module = module.strip()
    if not _MODULE_NAME_RE.fullmatch(module):
        raise ValueError(
            f"Invalid module name {module!r}: expected a Python module "
            "(e.g. 'pytest', 'ruff'), run non-Python tooling via run_bash."
        )
    plan = _project_python(root, module)
    if plan is None:
        return (
            f"'{module}' is not available in the project's own environment "
            "(no .venv/uv environment there imports it) — the project does "
            "not have this tool installed. Treat the check as skipped; do "
            "NOT fall back to a bare PATH binary: a different version or "
            "default ruleset would report phantom failures."
        )
    cmd, fix_path = plan
    try:
        proc = subprocess.run(
            [*cmd, "-m", module, *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=tools.clamp_bash_timeout(timeout),
            # Some tools do their OWN environment discovery despite being
            # launched as `-m <tool>`: the pip pyright package is a node
            # wrapper that probes VIRTUAL_ENV/PATH and ignores the
            # interpreter it ran under — without the pin it lands on a
            # foreign python and reports phantom reportMissingImports
            # (the 9d29e63d class). Pin the plan's venv defensively for
            # every module.
            env=_venv_env(cmd, _worktree_pythonpath_env(root) if fix_path else None),
            check=False,
            **tools.child_priority_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"exit=-1\n({type(e).__name__}: {e})"
    out = tools.strip_malloc_stack_logging_noise(
        proc.stdout + "\n" + proc.stderr
    ).strip()[: tools.MAX_TOOL_OUTPUT]
    return f"exit={proc.returncode}\n{out or '(no output)'}"


async def _git_push(deps: AgentDeps | TesterDeps, root: Path) -> str:
    """Best-effort `git push` after a successful commit.

    A push failure never fails the caller's flow: the error is written into
    the task's service-event log and reported back to the model as a note, so
    the tester still passes the task and the research agent still finishes
    its documentation assignment with the commit safe locally."""
    remotes = await tools.run_bash(root, "git remote", extra_writable=deps.scratch)
    names = remotes.removeprefix("exit=0").strip()
    if not remotes.startswith("exit=0") or not names or names == "(no output)":
        log_service_event(deps, "git", "git push skipped: no remote configured.")
        return "\ngit push skipped: no remote configured."
    push = await tools.run_bash(root, "git push", extra_writable=deps.scratch)
    if push.startswith("exit=0"):
        return "\nPushed successfully."
    if "no upstream" in push:
        # Fresh branch without tracking: retry once, setting the upstream.
        push = await tools.run_bash(
            root, "git push --set-upstream origin HEAD", extra_writable=deps.scratch
        )
        if push.startswith("exit=0"):
            return "\nPushed successfully (set upstream to origin/HEAD)."
    detail = push.removeprefix("exit=0").strip() or "(no output)"
    log_service_event(deps, "git", f"git push failed:\n{detail}")
    return (
        "\nPush failed (the commit is safe locally):\n"
        f"{detail}\n"
        "This is not a verification failure — proceed with "
        "set_verification(success=True) and mention the push problem "
        "in your summary."
    )


def _track_edited_file(deps: AgentDeps, resolved: Path) -> None:
    """Record a file the research agent created/edited on the task.

    Only research tasks are tracked: the coding agent's feature/bug changes
    are committed by the tester, not by the agent itself. Paths outside the
    project root (scratch files under /tmp/drcode/<task-id>) are ignored —
    they can never become commit candidates. The list persists with the
    task, so tracking survives
    restarts and the multi-turn loop of one run (a new AgentDeps is built per
    `agent.iter`, the task field is the shared accumulator). Duplicates are
    collapsed, keeping the first-seen order.
    """
    task = deps.manager.get(deps.task_id)
    if task.kind is not TaskKind.RESEARCH:
        return
    try:
        rel = resolved.relative_to(deps.root.resolve())
    except ValueError:
        return
    rel_posix = rel.as_posix()
    if rel_posix not in task.edited_files:
        task.edited_files.append(rel_posix)
        task.touch()
        deps.manager.storage.save(task)


async def _commit_tracked_files(
    deps: AgentDeps, root: Path, message: str, allow_push: bool = True
) -> str:
    """Commit EXACTLY the files the research agent created/edited itself.

    The stage list comes from `task.edited_files` (see `_track_edited_file`)
    — the model cannot widen it, so unrelated working-tree changes are never
    committed. Pushes via `_git_push` when `allow_push` (the project's
    research git policy: push) and a remote are configured; with
    `allow_push=False` the commit stays local and the tool result says so.
    The tracked list is cleared after a successful commit so a later call
    does not re-stage already-committed files.
    """
    message = message.strip().strip("'\"")
    if not message:
        return "Empty commit message — nothing committed."
    task = deps.manager.get(deps.task_id)
    files = list(dict.fromkeys(task.edited_files))
    if not files:
        return (
            "Nothing to commit — you have not created or edited any files in "
            "this task yourself. Only files you changed are ever committed; "
            "other project changes are never staged."
        )
    inside = await tools.run_bash(
        root, "git rev-parse --is-inside-work-tree", extra_writable=deps.scratch
    )
    if not inside.startswith("exit=0") or "true" not in inside:
        return "Not a git repository — nothing committed."
    quoted = " ".join(shlex.quote(f) for f in files)
    add = await tools.run_bash(
        root, f"git add -A -- {quoted}", extra_writable=deps.scratch
    )
    if not add.startswith("exit=0"):
        return f"Staging failed:\n{add}"
    staged = await tools.run_bash(
        root, "git diff --cached --name-only", extra_writable=deps.scratch
    )
    if staged.startswith("exit=0") and not staged.removeprefix("exit=0").strip():
        return (
            "Nothing to commit — the files you edited carry no changes "
            "against HEAD (already committed or untouched)."
        )
    commit = await tools.run_bash(
        root, f"git commit -m {shlex.quote(message)}", extra_writable=deps.scratch
    )
    if not commit.startswith("exit=0"):
        return f"Commit failed:\n{commit}"
    head = await tools.run_bash(
        root, "git log -1 --oneline", extra_writable=deps.scratch
    )
    task.edited_files.clear()
    task.touch()
    deps.manager.storage.save(task)
    result = f"Committed successfully.\n{head}"
    if allow_push:
        return result + await _git_push(deps, root)
    return result + "\nCommit is local — this project never pushes."
