"""Custom team workflow: `.drcode/workflow.py` hooks and `.drcode/git-workflow.md` rules.

Two independent layers, each enabled by the mere presence of the file in
`<project>/.drcode/` (template scaffolds `workflow.py.example` and
`git-workflow.md.example` are written on every app start and never
overwritten — see `ensure_workflow_examples`):

- **`.drcode/git-workflow.md`** — declarative process rules written by the
  team in plain prose. When present, it replaces the "Never commit to git"
  line in the coding agents' system prompts (commits become part of the
  workflow) and is appended to the tester's prompt (see `drcode.prompts`).
- **`.drcode/workflow.py`** — hard gates: an optional Python script with
  hook functions the pipeline calls at fixed lifecycle points. Each call
  runs in a **subprocess** (`python -c <bootstrap> <script>`, cwd = project
  root): the bootstrap imports the script, passes one JSON payload to the
  hook function and prints the returned dict as JSON — the same isolation
  model as the agent's `run_bash` tool, so a hanging or crashing script can
  never take the TUI down. The script only defines the functions; the
  dispatch is done for it.

Hook contract (every function is optional; an absent function simply
skips its step; `task` is the payload dict described below):

    def on_task_start(task) -> dict | None
        Called before the FIRST activation of a task (e.g. to check out a
        feature branch). MUST be idempotent: an abort leaves a dirty tree
        on the branch (the resume continues in it), a crash leaves the
        checkout on the old branch, and the previous task may be unmerged.
        Return {"instructions": "..."} to inject text into the coding
        agent's first prompt.
    def on_development_done(task) -> dict | None
        Development finished, before testing (push, `gh pr create`, …).
        Return {"text": "..."} to append context to the tester's prompt.
    def on_tester_passed(task) -> dict | None
        Called right after the tester stamped the task PASSED and the task
        is finalized (also when a review-mode task is accepted later by the
        external review — an install/deploy step must run on every path to
        COMPLETED-after-PASSED). The unconditional post-verification hook:
        it needs no on_review gate and never changes pipeline behaviour.
        Research tasks never trigger it (they skip testing). Return
        {"text": "..."} for the task's service-event log.
    def on_review(task) -> dict
        External code review after the tester passed the task. Return
        {"verdict": "accept"} or {"verdict": "reject", "comment": "..."}.
        Any other verdict (None / "pending" — e.g. CI still running) keeps
        the task waiting; the pipeline retries.
    def on_accepted(task) -> dict | None
        After an accept (merge the PR, delete the branch). Return
        {"text": "..."} for the task's service-event log.

The `task` argument is one JSON object:
    {"hook": "<name>",
     "task": {"id", "title", "kind", "github_issue"},
     "git":  {"branch": "<current branch or null>", "dirty": <bool>}}

A typical script looks like:

    import subprocess

    def on_task_start(task):
        subprocess.run(["git", "checkout", "-B", f"task/{task['task']['id']}"], check=True)
        return {"instructions": f"Commit your work to branch task/{task['task']['id']}."}

    def on_review(task):
        ci = subprocess.run(["gh", "pr", "checks"], capture_output=True)
        if ci.returncode != 0:
            return {"verdict": "pending"}
        return {"verdict": "accept"}

    def on_tester_passed(task):
        r = subprocess.run(["uv", "tool", "install", ".", "--force"],
                           capture_output=True, text=True)
        return {"text": r.stdout or r.stderr}

A hook failure (non-zero exit, malformed output, timeout) is NEVER fatal:
it is written into the task's service-event log and the pipeline moves on
— with one exception: `on_review` fails **closed** (the task waits and the
review is retried with backoff), because an unreachable reviewer must not
wave a change through.

Review flow (on_review defined):
- right after the tester passes the task, the pipeline calls on_review;
  **accept** completes the task (and runs on_accepted), **reject** re-queues
  it with the comment as new user input (the coding agent reworks it in the
  same transcript);
- a **pending** verdict (None / "pending" — CI still running, a human
  review in progress) releases the pipeline: the task completes with its
  PASSED label and `review_pending` set, and the periodic review poll
  re-asks on_review (every 60 s) until a final verdict arrives — so a
  long human review never blocks new development. "Awaiting review" =
  COMPLETED with an open PR; such tasks are excluded from the startup
  auto-cleanup, so a verdict that arrives days later still finds its task.

Loop protection: after `MAX_REVIEW_REJECTIONS` consecutive rejects the
task is parked on hold ("requires human attention") instead of ping-ponging
between the agent and the reviewer forever.

SECURITY: `.drcode/workflow.py` is arbitrary code executed with the user's
full permissions. This is a deliberate design decision (the agent's own
`run_bash` is no safer), but it must be stated plainly: a workflow file
from an UNTRUSTED repository is NOT safe. Review it before running drcode
in a project you did not author.
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Task

WORKFLOW_SCRIPT = "workflow.py"
GIT_WORKFLOW_FILE = "git-workflow.md"

# Template files scaffolded on app start (never overwritten): the user
# activates a workflow layer by renaming the template to the real name.
WORKFLOW_EXAMPLE_FILE = "workflow.py.example"
GIT_WORKFLOW_EXAMPLE_FILE = "git-workflow.md.example"

# The workflow.py template: the hook contract in miniature, every hook
# commented out so a renamed file defines no hooks until edited.
WORKFLOW_EXAMPLE_CONTENT = '''\
"""drcode workflow hooks (template).

ACTIVATION: rename this file to `workflow.py` (keep it in the project's
`.drcode/` directory) and uncomment the hooks you need. drcode scaffolds
this template on every start, but NEVER overwrites it — edit it in place.

What it gives you: hard gates at fixed lifecycle points of the task
pipeline. Each defined hook runs in an isolated subprocess (cwd = project
root) and receives one JSON payload:

    {"hook": "<name>",
     "task": {"id", "title", "kind", "github_issue"},
     "git":  {"branch": "<current branch or null>", "dirty": <bool>}}

Every hook is optional — an absent function simply skips its step. A hook
failure (non-zero exit, timeout, malformed output) is NEVER fatal: it is
written into the task's service-event log and the pipeline moves on — with
one exception: on_review fails CLOSED (the task waits, the review is
retried with backoff), because an unreachable reviewer must not wave a
change through.

SECURITY: workflow.py is arbitrary code executed with your full
permissions. Only activate a workflow file you have read and trust.
"""

# import subprocess
#
#
# def on_task_start(task):
#     """Before the FIRST activation of a task (e.g. check out a feature
#     branch). MUST be idempotent: an abort leaves a dirty tree on the
#     branch. Return {"instructions": "..."} to inject text into the coding
#     agent's first prompt."""
#     subprocess.run(
#         ["git", "checkout", "-B", f"task/{task['task']['id']}"], check=True
#     )
#     return {
#         "instructions": f"Commit your work to branch task/{task['task']['id']}."
#     }
#
#
# def on_development_done(task):
#     """Development finished, before testing (push, `gh pr create`, ...).
#     Return {"text": "..."} to append context to the tester's prompt."""
#
#
# def on_tester_passed(task):
#     """Right after the tester stamped the task PASSED and it is finalized
#     (runs on every path to COMPLETED-after-PASSED) — the unconditional
#     post-verification hook: install / deploy here. Research tasks never
#     trigger it (they skip testing). Return {"text": "..."} for the
#     task's service-event log."""
#
#
# def on_review(task):
#     """External code review after the tester passed. Return
#     {"verdict": "accept"} — the task completes (and on_accepted runs);
#     {"verdict": "reject", "comment": "..."} — the task is re-queued with
#     the comment as new user input (3 consecutive rejects park it on
#     hold); {"verdict": "pending"} (or None) — the task completes with
#     review_pending set and this hook is re-polled every 60 s until a
#     final verdict arrives (a long human review never blocks the queue)."""
#     if subprocess.run(["gh", "pr", "checks"]).returncode != 0:
#         return {"verdict": "pending"}
#     return {"verdict": "accept"}
#
#
# def on_accepted(task):
#     """After an accept (merge the PR, delete the branch). Return
#     {"text": "..."} for the task's service-event log."""
'''

# The git-workflow.md template: prose rules replacing the built-in
# "never commit to git" agent instruction once activated.
GIT_WORKFLOW_EXAMPLE_CONTENT = """\
<!-- drcode git workflow rules (template).

     ACTIVATION: rename this file to `git-workflow.md` (keep it in the
     project's `.drcode/` directory) and edit the rules below. drcode
     scaffolds this template on every start, but NEVER overwrites it.

     What it gives you: when present, these rules REPLACE the built-in
     "never commit to git" instruction in the coding agents' system
     prompts (commits become part of the workflow) and are appended to
     the tester's prompt. Write plain prose — branching model, commit
     conventions, PR rules — the agents follow it literally. -->

# Git workflow rules

- Work on a feature branch named `task/<id>`; never commit directly to `main`.
- Commit messages: imperative mood, reference the task id — `feat: add X (#<id>)`.
- Keep commits atomic: no unrelated changes in one commit.
- After development, push the branch and open a PR (`gh pr create`).
"""


def ensure_workflow_examples(project_root: Path) -> None:
    """Scaffold the two workflow templates in `<project>/.drcode/`.

    On every app start the project's `.drcode/` gets `workflow.py.example`
    and `git-workflow.md.example` — short, ready-to-edit templates that
    document the user-activatable workflow features (hooks / git rules).
    Existing files are NEVER overwritten (a user's edited template or an
    already-renamed real file stays untouched; a real `workflow.py` /
    `git-workflow.md` does not suppress the template — the pair is
    independent documentation). Never raises: an unwritable project
    directory just means no templates.
    """
    drcode_dir = project_root / ".drcode"
    try:
        drcode_dir.mkdir(parents=True, exist_ok=True)
        for name, content in (
            (WORKFLOW_EXAMPLE_FILE, WORKFLOW_EXAMPLE_CONTENT),
            (GIT_WORKFLOW_EXAMPLE_FILE, GIT_WORKFLOW_EXAMPLE_CONTENT),
        ):
            path = drcode_dir / name
            if path.exists():
                continue
            path.write_text(content, encoding="utf-8")
    except OSError:
        pass


# Every hook the pipeline may call (in lifecycle order).
HOOK_NAMES: tuple[str, ...] = (
    "on_task_start",
    "on_development_done",
    "on_tester_passed",
    "on_review",
    "on_accepted",
)

# Hard cap on a single hook invocation: a hook may push and wait for CI,
# but it must not stall the pipeline forever (the sync review blocks it).
HOOK_TIMEOUT = 600.0

# Consecutive review rejections before the task is parked on hold.
MAX_REVIEW_REJECTIONS = 3

# How long a def function definition may be looked for — the hook scan is
# a plain AST walk; a pathological script is capped by HOOK_TIMEOUT anyway.


class _HookScan(ast.NodeVisitor):
    """Collects the names of top-level function definitions."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)


# The subprocess entry: imports the workflow script, reads the JSON payload
# from stdin, calls the requested hook function with it and prints the
# returned dict as JSON (a None/absent hook yields an empty object).
_BOOTSTRAP = """\
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("drcode_workflow_hook", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
payload = json.load(sys.stdin)
hook = getattr(module, payload["hook"], None)
if hook is None:
    json.dump({}, sys.stdout)
else:
    result = hook(payload)
    json.dump(result if result is not None else {}, sys.stdout)
"""


@dataclass
class HookResult:
    """Outcome of one subprocess hook call.

    `ok` is False for any failure (non-zero exit, timeout, malformed
    output); `error` carries a human-readable reason for the service-event
    log. `output` is the parsed JSON object on success (empty dict when
    the hook legitimately returned nothing).
    """

    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def _hook_output_text(output: dict[str, Any], *keys: str) -> str:
    """Extract a free-text result ("instructions" / "text") from a hook's
    JSON output. Accepts the documented keys (first match wins); any
    non-string value is ignored."""
    for key in keys:
        value = output.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class WorkflowHooks:
    """The loaded `.drcode/workflow.py` of one project.

    `load_error` is set when the script exists but cannot be parsed — in
    that case no hooks are reported (the workflow silently degrades to the
    default behaviour; `describe()` surfaces the reason for the Settings
    line). The script is never imported into this process: every call is
    a fresh subprocess, so a `sys.exit`, an infinite loop or a memory bomb
    in the script cannot take the app with it.
    """

    def __init__(self, project_root: Path, script: Path, load_error: str = "") -> None:
        self.project_root = project_root
        self.script = script
        self.load_error = load_error
        self._hooks: frozenset[str] = frozenset() if load_error else self._scan()

    def _scan(self) -> frozenset[str]:
        try:
            tree = ast.parse(
                self.script.read_text(encoding="utf-8"), filename=str(self.script)
            )
        except (OSError, SyntaxError, ValueError) as e:
            self.load_error = f"{type(e).__name__}: {e}"
            return frozenset()
        visitor = _HookScan()
        visitor.visit(tree)
        return frozenset(visitor.names & set(HOOK_NAMES))

    def has(self, hook: str) -> bool:
        """Whether the script defines the named hook (False without the
        script, for a broken script, or for an unknown name)."""
        return hook in self._hooks

    def defined_hooks(self) -> list[str]:
        """The defined hooks, in lifecycle order (for the Settings line)."""
        return [h for h in HOOK_NAMES if h in self._hooks]

    def describe(self) -> str:
        """One-line summary for the Settings entry ("custom (…)" / error)."""
        if self.load_error:
            return (
                f"custom (.drcode/{WORKFLOW_SCRIPT} — unparseable: {self.load_error})"
            )
        hooks = ", ".join(self.defined_hooks()) or "no hooks"
        return f"custom (.drcode/{WORKFLOW_SCRIPT}: {hooks})"

    async def call(
        self, hook: str, task: Task, timeout: float = HOOK_TIMEOUT
    ) -> HookResult:
        """Run one hook: `python -c <bootstrap> <script>` in a subprocess
        with cwd = project root; the bootstrap imports the script, feeds it
        the JSON payload on stdin and prints the hook's return value as a
        JSON object. The script itself never has to write dispatch code —
        it just defines the hook functions (see the module docstring).

        Never raises (except cancellation — an aborted run must kill the
        child and unwind); every failure comes back as
        ``HookResult(ok=False)`` so the callers only branch on `ok`.
        """
        if not self.has(hook):
            return HookResult(ok=False, error=f"hook {hook} is not defined")
        payload = {
            "hook": hook,
            "task": {
                "id": task.id,
                "title": task.title,
                "kind": task.kind.value,
                "github_issue": task.github_issue,
            },
            "git": await self._git_state(),
        }
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _BOOTSTRAP,
                str(self.script),
                cwd=self.project_root,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as e:
            return HookResult(ok=False, error=f"cannot start {WORKFLOW_SCRIPT}: {e}")
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=json.dumps(payload).encode()),
                timeout=timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return HookResult(
                ok=False, error=f"hook {hook} timed out after {int(timeout)} s"
            )
        except asyncio.CancelledError:
            # The run was aborted or the app is quitting: the child must
            # not outlive the call (mirrors tools.run_bash).
            proc.kill()
            try:
                await asyncio.shield(proc.wait())
            except asyncio.CancelledError:
                pass
            raise
        if proc.returncode != 0:
            detail = (stderr or stdout).decode(errors="replace").strip()
            return HookResult(
                ok=False,
                error=(
                    f"hook {hook} exited with code {proc.returncode}: "
                    f"{detail[:2000] or '(no output)'}"
                ),
            )
        try:
            output = json.loads(stdout.decode(errors="replace"))
        except ValueError:
            return HookResult(
                ok=False,
                error=f"hook {hook} printed a non-JSON response",
            )
        if not isinstance(output, dict):
            return HookResult(
                ok=False,
                error=f"hook {hook} must print a JSON object, got {type(output).__name__}",
            )
        return HookResult(ok=True, output=output)

    async def _git_state(self) -> dict[str, Any]:
        """Current git state of the project (branch, dirty or not) for the
        hook payload. Best-effort: outside a repository the branch is None
        and the tree counts as clean."""
        from . import tools  # local import: keeps the module import graph flat

        head = await tools.run_bash(
            self.project_root, "git rev-parse --abbrev-ref HEAD"
        )
        branch: str | None = None
        if head.startswith("exit=0"):
            name = head.removeprefix("exit=0").strip()
            branch = name or None
        status = await tools.run_bash(self.project_root, "git status --porcelain")
        dirty = bool(status.removeprefix("exit=0").strip())
        return {"branch": branch, "dirty": dirty}


def load_hooks(project_root: Path) -> WorkflowHooks | None:
    """The project's workflow hooks, or None when `.drcode/workflow.py`
    does not exist (the default single-developer setup: no hooks, the
    pipeline behaves exactly as before)."""
    script = project_root / ".drcode" / WORKFLOW_SCRIPT
    if not script.is_file():
        return None
    return WorkflowHooks(project_root, script)


def load_git_workflow(project_root: Path | None) -> str | None:
    """The project's `.drcode/git-workflow.md` content (stripped), or None
    when the file is absent or blank."""
    if project_root is None:
        return None
    path = project_root / ".drcode" / GIT_WORKFLOW_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    stripped = text.strip()
    return stripped or None


def describe_project_workflow(project_root: Path) -> str:
    """One-line status of both workflow layers for the Settings entry:
    "default" without the files, otherwise what is active."""
    parts: list[str] = []
    if load_git_workflow(project_root) is not None:
        parts.append(f".drcode/{GIT_WORKFLOW_FILE}")
    hooks = load_hooks(project_root)
    if hooks is not None:
        parts.append(hooks.describe())
    if not parts:
        return "default"
    return "custom (" + "; ".join(parts) + ")"
