"""Mutating MCP tools: the §4 gating matrix and the §6 re-advance.

The principle of §4: **MCP cannot do what a human cannot do in the TUI** —
the admissibility of every mutation repeats `actions_lifecycle.py`, it does
not invent new rules. Two mechanisms carry that principle:

**Gating.** Each body checks the task's live status against the matrix
before touching the manager. A denied mutation is NOT an exception in the
protocol sense — it is a normal MCP `isError` result carrying the current
status and a piece of advice ("cancel it first or wait"), exactly the way
the TUI explains an impossible keypress instead of crashing. Only a
genuinely unexpected failure (a corrupted store, a failed `call_from_thread`
on the *mutation itself*) surfaces as a crash, and the SDK sanitizes that.

**Re-advance (§6).** Every successful mutating call after which the TUI
would have pressed Enter schedules the app's `_advance()` — the single
pipeline gate — through the `MutationHooks` the embedding app provides.
Everything lands on the ONE application event loop (`call_from_thread`),
so races with the pipeline are impossible by construction. The hooks are
injected, never imported: the tests pass recorders, the TUI passes
`call_from_thread` wrappers.

Thread model: the handlers are sync (the SDK runs them in a worker thread
of the server loop) and hop into the app thread ONLY through the hooks —
the manager calls themselves are direct (TaskManager is thread-tolerant;
§0). A hook failure after a successful mutation is logged, never raised:
the mutation already happened, and the next pipeline event re-advances.

Deliberately NOT published (§4, "Не публикуются"): the tester's verdict
tools, the hook-requeue internals, `complete_without_verification`, any
transcript/skeleton writing, model configuration, `purge_expired_completed`
and `link_goal`/`unlink_goal` (v1: `goal_id` only at `create_task`).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from mcp.server.mcpserver.exceptions import ToolError

from ..models import Task, TaskEvent, TaskKind, TaskStatus
from ..task_overlap import (
    build_overlap_event_text,
    build_overlap_warning,
    find_file_overlaps,
)
from .reads import ProjectView

#: Tool names — constants so tests and registration can never disagree.
CREATE_TASK = "create_task"
EDIT_TASK = "edit_task"
SEND_INPUT = "send_input"
ANSWER_CLARIFICATION = "answer_clarification"
SET_DEPENDENCIES = "set_dependencies"
SET_KIND = "set_kind"
SET_HOLD = "set_hold"
MOVE_IN_QUEUE = "move_in_queue"
CANCEL_TASK = "cancel"
DELETE_TASK = "delete"
TRIGGER_ADVANCE = "trigger_advance"

#: The whole mutation surface, in registration order (tests assert it in
#: one place, after the nine read tools).
MUTATION_TOOLS: tuple[str, ...] = (
    CREATE_TASK,
    EDIT_TASK,
    SEND_INPUT,
    ANSWER_CLARIFICATION,
    SET_DEPENDENCIES,
    SET_KIND,
    SET_HOLD,
    MOVE_IN_QUEUE,
    CANCEL_TASK,
    DELETE_TASK,
    TRIGGER_ADVANCE,
)

#: The statuses the not-started-only tools (set_dependencies, set_kind,
#: set_hold(on), delete) accept — the §4 row "queued/draft/hold".
_NOT_STARTED: tuple[TaskStatus, ...] = (
    TaskStatus.QUEUED,
    TaskStatus.DRAFT,
    TaskStatus.HOLD,
)

#: The advice of the §4 ACTIVE/TESTING cell, verbatim in spirit:
#: «задача ACTIVE/TESTING — отмените (cancel) или дождитесь».
_BUSY_ADVICE = "cancel it first (cancel) or wait for it to finish"


# -- hooks ---------------------------------------------------------------------


@dataclass
class MutationHooks:
    """The app-side callbacks every mutation needs AFTER the manager call.

    `advance` is the §6 re-advance: schedule the app's `_advance()` on the
    application event loop. `abort_runs` is the worker-group half of the
    Ctrl+X cancel (stop the task's agent/tester workers); `restart_run`
    supersedes an ACTIVE task's in-flight run when a comment arrives — the
    TUI's `run_agent` call for a mid-flight comment. All three are
    optional so a test surface can inject only what it asserts on;
    `None` degrades to a no-op (a mutation still applies — a missing
    post-step must never fail an already-successful mutation).
    """

    advance: Callable[[], None] | None = None
    abort_runs: Callable[[str], None] | None = None
    restart_run: Callable[[str], None] | None = None

    def schedule_advance(self) -> None:
        """Plan the pipeline re-advance on the app loop (§6). Never raises:
        the mutation has already been applied when this runs."""
        if self.advance is not None:
            try:
                self.advance()
            except Exception as e:  # noqa: BLE001 - post-step, never fatal
                _hook_failed("advance", e)

    def abort_task_runs(self, task_id: str) -> None:
        """Stop the task's in-flight workers (the cancel power-action)."""
        if self.abort_runs is not None:
            try:
                self.abort_runs(task_id)
            except Exception as e:  # noqa: BLE001 - post-step, never fatal
                _hook_failed("abort_runs", e)

    def restart_task_run(self, task_id: str) -> None:
        """Supersede the ACTIVE task's run with a fresh one (comment)."""
        if self.restart_run is not None:
            try:
                self.restart_run(task_id)
            except Exception as e:  # noqa: BLE001 - post-step, never fatal
                _hook_failed("restart_run", e)


def _hook_failed(name: str, e: Exception) -> None:
    print(
        f"drcode: mcp mutation hook {name} failed: {type(e).__name__}: {e}",
        file=sys.stderr,
    )


def app_hooks(app) -> MutationHooks:
    """The production hooks: hop into the application thread.

    `app` is the live Textual app (duck-typed: `call_from_thread`,
    `_on_app_thread`, `_advance`, `run_agent`, `_abort_task_runs`). Every
    callback lands on the ONE application event loop — the §6 guarantee
    that mutation post-steps cannot race the pipeline. A failing hop
    (app shutting down) is reported, never raised.
    """

    def on_app_thread(fn: Callable[..., object], *args: object) -> None:
        try:
            if app._on_app_thread():
                fn(*args)
            else:
                app.call_from_thread(fn, *args)
        except Exception as e:  # noqa: BLE001 - post-step, never fatal
            _hook_failed(getattr(fn, "__name__", "call"), e)

    return MutationHooks(
        advance=lambda: on_app_thread(app._advance),
        abort_runs=lambda task_id: on_app_thread(app._abort_task_runs, task_id),
        restart_run=lambda task_id: on_app_thread(app.run_agent, task_id),
    )


# -- gating --------------------------------------------------------------------


def _deny(task: Task, advice: str) -> ToolError:
    """The §4 isError: the current status plus a way out — never a bare no."""
    return ToolError(f"task #{task.id} is {task.status.value} — {advice}")


def _require_known(view: ProjectView, task_id: str) -> Task:
    """The task or the not-found isError (with real ids, like get_task)."""
    try:
        return view.manager.get(task_id)
    except KeyError:
        known = ", ".join(f"#{t.id}" for t in view.manager.all_tasks()[-5:])
        hint = f" Known task ids include: {known}." if known else ""
        raise ToolError(f"unknown task id {task_id!r}.{hint}") from None


def _require_not_started(task: Task) -> None:
    """The shared §4 row of set_dependencies/set_kind/set_hold(on)/delete:
    queued/draft/hold pass; clarification and completed are denied with
    their own advice; ACTIVE/TESTING gets the matrix's verbatim advice."""
    status = task.status
    if status in _NOT_STARTED:
        return
    if status in (TaskStatus.ACTIVE, TaskStatus.TESTING):
        raise _deny(task, _BUSY_ADVICE)
    if status is TaskStatus.AWAITING_CLARIFICATION:
        raise _deny(
            task, "answer the pending clarification first (answer_clarification)"
        )
    raise _deny(
        task,
        "a completed task cannot be changed — send_input takes it back into development",
    )


def _parse_kind(value: str) -> TaskKind:
    try:
        return TaskKind(str(value))
    except ValueError:
        allowed = ", ".join(member.value for member in TaskKind)
        raise ToolError(f"unknown kind {value!r}; expected one of: {allowed}") from None


# -- mutation bodies -----------------------------------------------------------


def create_task(
    view: ProjectView,
    hooks: MutationHooks,
    *,
    title: str | None,
    text: str,
    kind: str,
    depends_on: list[str] | None,
    goal_id: str | None,
    hold: bool,
) -> str:
    """File a new task — always allowed (creation touches no live run).

    Mirrors the CLI's `--feature-request`: the overlap pre-flight runs
    after the create and lands as a service event + a warning in the
    answer (a WARNING, never a block — `depends_on` at creation is the
    only real prevention of the parallel merge race).
    """
    if not text or not text.strip():
        raise ToolError("text must not be empty — it is the task's condition")
    task_kind = _parse_kind(kind)
    try:
        task = view.manager.create(
            title or "",
            text.strip(),
            kind=task_kind,
            depends_on=list(depends_on or []),
            goal_id=goal_id,
            hold=hold,
            origin="mcp",
        )
    except ValueError as e:
        raise ToolError(str(e)) from None
    hooks.schedule_advance()
    head = (
        f"created #{task.id} [{task.kind.value}] {task.title} — "
        f"status {task.status.value}"
    )
    lines = [head]
    if hold:
        lines.append("the task is parked on hold — set_hold(on=false) resumes it")
    overlaps = find_file_overlaps(
        view.manager.tasks, text, exclude_id=task.id, depends_on=task.depends_on
    )
    if overlaps:
        task.events.append(
            TaskEvent(kind="system", text=build_overlap_event_text(overlaps))
        )
        view.manager.storage.save(task)
        lines.append(build_overlap_warning(overlaps))
    return "\n".join(lines)


def edit_task(
    view: ProjectView,
    hooks: MutationHooks,
    task_id: str,
    text: str,
    kind: str | None,
) -> str:
    """Rewrite the condition of a not-yet-started task (§4: by
    `editable_condition`, the reason lives in the error).

    Editable means: the task has not started (queued/hold/draft) AND still
    has queued user text (`editable_condition` non-empty — an empty
    editable condition means the original prompt is already consumed into
    the transcript and cannot be rewritten without breaking the history).
    No re-advance: a held/draft task is never promotable, so the TUI does
    not press Enter here either (§6)."""
    task = _require_known(view, task_id)
    _require_not_started(task)
    editable = view.manager.editable_condition(task_id)
    if not editable:
        raise ToolError(
            f"task #{task.id} has nothing editable left — every queued text "
            "was already consumed into the transcript; send_input a comment "
            "instead"
        )
    if not text or not text.strip():
        raise ToolError("text must not be empty — it replaces the whole condition")
    new_kind = _parse_kind(kind) if kind else None
    task = view.manager.edit_task(task_id, text.strip(), kind=new_kind)
    return (
        f"#{task.id} condition rewritten ({len(text.strip())} chars, title "
        f"re-derived: {task.title!r}) — status {task.status.value}"
    )


def send_input(
    view: ProjectView,
    hooks: MutationHooks,
    task_id: str,
    text: str,
) -> str:
    """Queue user input — allowed in EVERY status (§4, "везде да").

    The status decides what the input MEANS, and the TUI's Enter behavior
    is replicated exactly (§4/§6):
    - queued/hold/draft: the text accumulates in pending_input;
    - clarification: the text is the ANSWER (see answer_clarification for
      the per-question form);
    - active/testing: a mid-flight comment — the in-flight run is
      superseded (`restart_run`, the TUI's run_agent call) and restarts
      with the comment; for TESTING the manager re-queues the task first;
    - completed: the task re-enters the queue for a fresh round (the old
      verification label, the cancel marker and the review state are
      dropped by `manager.send`).
    """
    task = _require_known(view, task_id)
    if not text or not text.strip():
        raise ToolError("text must not be empty")
    was_active = task.status is TaskStatus.ACTIVE
    view.manager.send(task_id, text.strip())
    if was_active:
        # The comment interrupts the run: supersede it like the TUI's
        # _submit_user_text does (run_agent on the exclusive per-task
        # group) — and then re-advance like after every mutation.
        hooks.restart_task_run(task_id)
    hooks.schedule_advance()
    fresh = view.manager.get(task_id)
    if was_active:
        detail = "comment interrupts the running agent — the run restarts with it"
    elif task.status is TaskStatus.TESTING:
        detail = "the testing round is re-queued — the task will run again"
    elif task.status is TaskStatus.COMPLETED:
        detail = "the completed task re-enters the queue for a fresh round"
    else:
        detail = f"status {fresh.status.value}"
    return f"input queued for #{task_id} — {detail}"


def answer_clarification(
    view: ProjectView,
    hooks: MutationHooks,
    task_id: str,
    answers: list[str],
) -> str:
    """Answer the pending clarification package.

    `answers` must contain EXACTLY one answer per question of
    `task.clarification` (the count is validated — a mismatched answer
    batch is refused before anything is sent). The answers are joined one
    line per question — exactly what the TUI submits on Enter over the
    per-question draft lines; a question with options is answered with the
    option's TEXT. `manager.send` then does the TUI's transition: the
    answer re-queues the task at the head and clears the package."""
    task = _require_known(view, task_id)
    clarification = task.clarification
    if clarification is None:
        raise _deny(task, "it has no pending clarification to answer (see get_task)")
    questions = clarification.questions
    cleaned = [str(answer) for answer in answers]
    if len(cleaned) != len(questions):
        listing = "\n".join(
            f"  {i + 1}. {item.question}" for i, item in enumerate(questions)
        )
        raise ToolError(
            f"expected {len(questions)} answer(s), got {len(cleaned)} — "
            f"one answer per question, in order:\n{listing}"
        )
    view.manager.send(task_id, "\n".join(cleaned))
    hooks.schedule_advance()
    fresh = view.manager.get(task_id)
    return (
        f"answers sent to #{task_id} ({len(cleaned)} question(s)) — "
        f"clarification answered, status {fresh.status.value}"
    )


def set_dependencies(
    view: ProjectView,
    hooks: MutationHooks,
    task_id: str,
    deps: list[str],
) -> str:
    """Replace `depends_on` of a not-yet-started task (§4). The manager
    validates ids/cycles (ValueError → isError). Re-advances afterwards:
    dropping a blocker is the TUI's force-start (`action_remove_dependency`
    advances), and adding one makes the advance a harmless no-op."""
    task = _require_known(view, task_id)
    _require_not_started(task)
    try:
        task = view.manager.set_dependencies(task_id, [str(d) for d in deps])
    except ValueError as e:
        raise ToolError(str(e)) from None
    hooks.schedule_advance()
    shown = ", ".join(f"#{d}" for d in task.depends_on) or "(none)"
    return f"#{task.id} depends_on set to: {shown}"


def set_kind(
    view: ProjectView,
    hooks: MutationHooks,
    task_id: str,
    kind: str,
) -> str:
    """Re-type a not-yet-started task (§4): bug → head of the queue,
    feature → tail. No re-advance — the TUI's Ctrl+T does not advance
    either (the task only changes its queue position)."""
    task = _require_known(view, task_id)
    _require_not_started(task)
    task_kind = _parse_kind(kind)
    task = view.manager.set_kind(task_id, task_kind)
    return f"#{task.id} kind set to {task.kind.value}"


def set_hold(
    view: ProjectView,
    hooks: MutationHooks,
    task_id: str,
    on: bool,
) -> str:
    """Park (`on=true`) or resume (`on=false`) a task (§4).

    `on=true` follows the matrix row: queued/draft/hold only — a task
    awaiting clarification is answered (or the TUI user parks it there via
    Ctrl+H, which the matrix deliberately keeps out of the MCP surface).
    `on=false` is the resume: HOLD only, and it is one of the §6
    re-advance triggers — a resumed task may be promotable right away."""
    task = _require_known(view, task_id)
    if on:
        _require_not_started(task)
        task = view.manager.set_hold(task_id, on=True)
        return f"#{task.id} parked on hold" + (
            "" if task.status is TaskStatus.HOLD else " (status unchanged)"
        )
    if task.status is TaskStatus.HOLD:
        task = view.manager.set_hold(task_id, on=False)
        hooks.schedule_advance()
        return f"#{task.id} resumed — status {task.status.value}"
    if task.status in (TaskStatus.ACTIVE, TaskStatus.TESTING):
        raise _deny(task, "a running task cannot be held — " + _BUSY_ADVICE)
    raise _deny(task, "it is not on hold")


def move_in_queue(
    view: ProjectView,
    hooks: MutationHooks,
    task_id: str,
    up: bool,
) -> str:
    """Move a QUEUED task one step in the backlog (§4: queued only).

    An impossible move (queue edge, the bug block, the dependency tier)
    is an isError with the same reasons the TUI toasts — a silent no-op
    would look like the tool ignored the call."""
    task = _require_known(view, task_id)
    if task.status is not TaskStatus.QUEUED:
        raise _deny(task, "only a queued task can be moved in the queue")
    moved = view.manager.move_in_queue(task_id, up=up)
    if moved is None:
        raise ToolError(_move_refusal(view, task_id, up=up))
    hooks.schedule_advance()
    return f"#{task_id} moved {'up (away from execution)' if up else 'down (closer to execution)'} in the queue"


def _move_refusal(view: ProjectView, task_id: str, *, up: bool) -> str:
    """Why `move_in_queue` returned None — the TUI toast's three reasons."""
    order = view.manager.queued()
    task = view.manager.get(task_id)
    i = order.index(task) if task in order else -1
    j = i + 1 if up else i - 1
    if i == -1 or not 0 <= j < len(order):
        return (
            "the task is already farthest from execution"
            if up
            else "the task is already next to run"
        )
    if (order[j].kind is TaskKind.BUG) != (task.kind is TaskKind.BUG):
        return (
            "bug reports always run first — the swap would cross the bug "
            "block (hold the bug with set_hold to postpone it)"
        )
    return (
        "the task waits for its dependencies — blocked tasks run only "
        "after every runnable one"
    )


def cancel(
    view: ProjectView,
    hooks: MutationHooks,
    task_id: str,
) -> str:
    """The power-action of Ctrl+X (§4, "да" everywhere except completed).

    This is a POWER-ACTION, and the tool description says so in plain
    words: it ABORTS any in-flight run of the task (the agent workers and
    the task's tester round — through the `abort_runs` hook, the same
    worker-group cancellation Ctrl+X performs), drops all queued input
    and moves the task straight to COMPLETED — the tester never runs, the
    work is not verified. There is no undo."""
    task = _require_known(view, task_id)
    if task.status is TaskStatus.COMPLETED:
        raise _deny(task, "it is already completed")
    # The worker half first (the in-flight run must not persist over the
    # transition), then the transition, then the §6 re-advance.
    hooks.abort_task_runs(task_id)
    view.manager.cancel(task_id)
    hooks.schedule_advance()
    return f"#{task_id} cancelled — completed without testing (power-action: the in-flight run was aborted, the tester never ran)"


def delete(
    view: ProjectView,
    hooks: MutationHooks,
    task_id: str,
) -> str:
    """Remove a parked task entirely — HOLD/DRAFT only (§4: the contract
    of `manager.delete`; a queued task is cancelled or held first). One
    of the §6 re-advance triggers (the queue may have been blocked on
    nothing anymore — cheap no-op when it was not)."""
    task = _require_known(view, task_id)
    if task.status not in (TaskStatus.HOLD, TaskStatus.DRAFT):
        if task.status in (TaskStatus.ACTIVE, TaskStatus.TESTING):
            raise _deny(task, _BUSY_ADVICE)
        raise _deny(
            task,
            "only a held or draft task can be deleted — hold it (set_hold) "
            "or cancel it (cancel) first",
        )
    title = task.title
    view.manager.delete(task_id)
    hooks.schedule_advance()
    return f"#{task_id} ({title!r}) deleted"


def trigger_advance(
    view: ProjectView,
    hooks: MutationHooks,
) -> str:
    """The explicit «nudge»: schedule the app's `_advance()` (§4).

    Usually unnecessary — every mutation re-advances anyway — but it is
    the external agent's manual kick when the pipeline looks stalled on
    something the MCP surface cannot see (a merged dependency branch, a
    retry backoff that has just expired)."""
    hooks.schedule_advance()
    return "pipeline advance scheduled"


# -- registration --------------------------------------------------------------


def register_mutation_tools(server, view: ProjectView, hooks: MutationHooks) -> None:
    """Register the eleven mutation tools of §4/§7 on the SDK server.

    Same shape as `tools.register_tools`: thin typed wrappers over the
    bodies above; every anticipated failure is already a `ToolError` (an
    MCP `isError`), and the SDK sanitizes anything that is not.
    """

    @server.tool(
        name=CREATE_TASK,
        title="Create a drcode task",
        description=(
            "File a NEW task into the live drcode pipeline. `text` is the "
            "task condition (the first prompt the coding agent will "
            "receive); `kind` is required: bug (jumps to the queue head), "
            "feature (queue tail) or research (no tester round). An empty "
            "`title` is derived from the text's first line (and may later "
            "be renamed by the title generator); an explicit one is kept "
            "verbatim. `depends_on` lists task ids that must COMPLETE "
            "(and merge) before this one can start — the only way to "
            "prevent a parallel file-overlap merge race; `goal_id` links "
            "the task to a roadmap goal (v1: only at creation); "
            "`hold=true` parks the task instead of queueing it. The "
            "overlap pre-flight runs automatically and comes back as a "
            "warning in the answer. The queue is re-advanced afterwards "
            "(the task may go ACTIVE immediately when a slot is free)."
        ),
    )
    def _create_task(
        title: str | None = None,
        text: str = "",
        kind: str = "feature",
        depends_on: list[str] | None = None,
        goal_id: str | None = None,
        hold: bool = False,
    ) -> str:
        return create_task(
            view,
            hooks,
            title=title,
            text=text,
            kind=kind,
            depends_on=depends_on,
            goal_id=goal_id,
            hold=hold,
        )

    @server.tool(
        name=EDIT_TASK,
        title="Edit a drcode task's condition",
        description=(
            "Rewrite the condition (the queued prompt text) of a task that "
            "has NOT started yet (queued/hold/draft) and still has its "
            "queued text — the merged editable text is replaced by one new "
            "text, the title is re-derived from its first line and "
            "`kind` (optional) re-types it like at creation. The old "
            "pending comments are gone (their content should be merged "
            "into the new text). Editing is refused for a started, "
            "completed or clarification-awaiting task — everything already "
            "consumed into the transcript is history, not a form field."
        ),
    )
    def _edit_task(task_id: str, text: str, kind: str | None = None) -> str:
        return edit_task(view, hooks, task_id, text, kind)

    @server.tool(
        name=SEND_INPUT,
        title="Send input to a drcode task",
        description=(
            "Queue a user message to a task — allowed in EVERY status, "
            "and the status decides what it means: queued/hold/draft — "
            "the text accumulates as the task's next prompt; a pending "
            "clarification — the text is the answer; ACTIVE — a mid-"
            "flight comment that INTERRUPTS the running agent and "
            "restarts it with the comment; TESTING — the task is "
            "re-queued and will run again; completed — the task re-enters "
            "the queue for a fresh development round (dropping the old "
            "verification label). The pipeline re-advances afterwards. "
            "For a per-question clarification answer use "
            "answer_clarification."
        ),
    )
    def _send_input(task_id: str, text: str) -> str:
        return send_input(view, hooks, task_id, text)

    @server.tool(
        name=ANSWER_CLARIFICATION,
        title="Answer a drcode clarification",
        description=(
            "Answer the task's pending clarification package: `answers` "
            "must contain EXACTLY one answer per question, in question "
            "order (the count is validated and a mismatch is an error "
            "listing the questions). The answers are submitted one line "
            "per question — exactly what the TUI sends on Enter. A "
            "question that offered options is answered with the option's "
            "text (free text is equally valid). The task re-enters the "
            "queue at the HEAD and the pipeline re-advances — the agent "
            "continues with the answers in context."
        ),
    )
    def _answer_clarification(task_id: str, answers: list[str]) -> str:
        return answer_clarification(view, hooks, task_id, answers)

    @server.tool(
        name=SET_DEPENDENCIES,
        title="Set a drcode task's dependencies",
        description=(
            "REPLACE the task's `depends_on` list (ids of tasks that must "
            "COMPLETE and merge before this one may start — the "
            "dependency gate). Only a not-yet-started task (queued/hold/"
            "draft) accepts it; an ACTIVE/TESTING task is refused (cancel "
            "or wait). The list is validated (unknown ids, cycles) and an "
            "invalid one is an error. Removing a blocker takes effect at "
            "the re-advance that follows."
        ),
    )
    def _set_dependencies(task_id: str, deps: list[str]) -> str:
        return set_dependencies(view, hooks, task_id, deps)

    @server.tool(
        name=SET_KIND,
        title="Set a drcode task's kind",
        description=(
            "Re-type a not-yet-started task (queued/hold/draft): bug — "
            "moves it to the queue head, feature — to the tail, research "
            "— keeps the position (and never runs the tester). "
            "ACTIVE/TESTING tasks are refused (cancel or wait)."
        ),
    )
    def _set_kind(task_id: str, kind: str) -> str:
        return set_kind(view, hooks, task_id, kind)

    @server.tool(
        name=SET_HOLD,
        title="Hold or resume a drcode task",
        description=(
            "`on=true` parks a not-yet-started task (queued/hold/draft) "
            "out of the backlog: never promoted while held, input keeps "
            "accumulating. `on=false` resumes a held task back into the "
            "queue (or back to its unanswered clarification) and "
            "re-advances the pipeline — the task may start immediately. "
            "A running (ACTIVE/TESTING) task cannot be held; a task "
            "awaiting clarification should be answered instead."
        ),
    )
    def _set_hold(task_id: str, on: bool) -> str:
        return set_hold(view, hooks, task_id, on)

    @server.tool(
        name=MOVE_IN_QUEUE,
        title="Move a drcode task in the queue",
        description=(
            "Move a QUEUED task one step in the backlog: `up=true` — away "
            "from execution, `up=false` — one step closer (it swaps with "
            "the task the pipeline would pick before it). An impossible "
            "move is an error saying why: the queue edge, the bug block "
            "(bug reports always run first) or the dependency tier. The "
            "new position is persisted and survives restarts."
        ),
    )
    def _move_in_queue(task_id: str, up: bool) -> str:
        return move_in_queue(view, hooks, task_id, up)

    @server.tool(
        name=CANCEL_TASK,
        title="Cancel a drcode task (power-action)",
        description=(
            "POWER-ACTION (the Ctrl+X of the TUI) — use with care, there "
            "is no undo. It ABORTS the task's in-flight agent run (and "
            "its tester round, if one is running), drops all queued "
            "input and moves the task straight to COMPLETED: the tester "
            "NEVER runs and the work is NOT verified. Allowed for a task "
            "in any status except completed. If the task is merely not "
            "relevant anymore, this is the right tool; to stop a run "
            "while KEEPING the task in the pipeline there is no MCP tool "
            "by design — a human presses ESC in the TUI."
        ),
    )
    def _cancel(task_id: str) -> str:
        return cancel(view, hooks, task_id)

    @server.tool(
        name=DELETE_TASK,
        title="Delete a drcode task",
        description=(
            "Remove a parked task entirely — from the board and from "
            "disk (.drcode/tasks/<id>.json). Only HOLD and DRAFT tasks "
            "are deletable (nothing runs for them, no worker state is "
            "left behind); a queued task must be held or cancelled "
            "first, a started task is refused. The pipeline re-advances "
            "afterwards (dependants re-evaluate their gate)."
        ),
    )
    def _delete(task_id: str) -> str:
        return delete(view, hooks, task_id)

    @server.tool(
        name=TRIGGER_ADVANCE,
        title="Advance the drcode pipeline",
        description=(
            "Explicitly schedule the pipeline's single advance gate — the "
            "external agent's «nudge» button. Usually unnecessary: every "
            "mutation re-advances anyway. Useful when the pipeline looks "
            "stalled on something invisible to the MCP surface — e.g. a "
            "dependency branch was merged by hand and the blocked tasks "
            "should be re-evaluated now. Never starts a second run on its "
            "own: the gate respects slots, the tester backlog and the "
            "dependency gates."
        ),
    )
    def _trigger_advance() -> str:
        return trigger_advance(view, hooks)


__all__ = [
    "ANSWER_CLARIFICATION",
    "CANCEL_TASK",
    "CREATE_TASK",
    "DELETE_TASK",
    "EDIT_TASK",
    "MOVE_IN_QUEUE",
    "MUTATION_TOOLS",
    "SEND_INPUT",
    "SET_DEPENDENCIES",
    "SET_HOLD",
    "SET_KIND",
    "TRIGGER_ADVANCE",
    "MutationHooks",
    "answer_clarification",
    "app_hooks",
    "cancel",
    "create_task",
    "delete",
    "edit_task",
    "move_in_queue",
    "register_mutation_tools",
    "send_input",
    "set_dependencies",
    "set_hold",
    "set_kind",
    "trigger_advance",
]
