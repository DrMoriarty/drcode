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
    def on_research_accepted(task) -> dict | None
        Right after a RESEARCH task completes (research skips testing —
        this is the research counterpart of on_accepted: merge the docs
        branch `drcode/<task-id>`, push, open a PR). The payload also
        carries "project": {"path", "base_branch"} — with worktree
        isolation the hook's cwd is the task's worktree, so merging into
        the user's checkout means operating on project.path, e.g.
        `git -C <project.path> merge drcode/<task-id>`. Return
        {"text": "..."} for the task's service-event log.

The `task` argument is one JSON object:
    {"hook": "<name>",
     "task": {"id", "title", "kind", "github_issue"},
     "git":  {"branch": "<current branch or null>", "dirty": <bool>},
     "worktree": {"path": "...", "branch": "drcode/<task-id>"},
     "project": {"path": "...", "base_branch": "<branch>"}}

(`project` is present only for `on_research_accepted` — the base branch
is resolved best-effort from the main checkout; outside a repository it
is absent.)

`worktree` is present only when per-task worktree isolation is on for the
project (`.drcode/settings.json` `task_isolation: worktree`, see
`drcode.task_env`): every hook then runs INSIDE the task's worktree (the
subprocess cwd, and the tree `git.branch`/`git.dirty` above describe) —
`on_task_start`'s branch checkout operates on the task's dedicated branch
`drcode/<task-id>`, not on the user's main checkout. Without isolation
the hooks run in the project directory as before and the key is absent.

With isolation on, drcode itself never merges the task's branch back into
the user's base branch: the commits stay on `drcode/<task-id>` and
merging/PR-management is the team's decision — do it in `on_accepted`
(merge the branch, delete it), which runs on every accept path. Research
tasks never reach that path (they skip testing): their completed work is
handed over by `on_research_accepted` instead — when the script does not
define it, drcode reminds the user about the unmerged docs branch, same
as for development tasks.

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
from .tools import sanitized_env

WORKFLOW_SCRIPT = "workflow.py"
GIT_WORKFLOW_FILE = "git-workflow.md"

# Template files scaffolded on app start (never overwritten): the user
# activates a workflow layer by renaming the template to the real name.
WORKFLOW_EXAMPLE_FILE = "workflow.py.example"
GIT_WORKFLOW_EXAMPLE_FILE = "git-workflow.md.example"

# The workflow.py template: ready-made hook implementations, every hook
# commented out (renaming the file without editing defines no hooks and
# changes nothing). Uncommenting the three degenerate-accept hooks gives
# worktree isolation an out-of-the-box auto-merge — see the template's
# module docstring for which hooks to uncomment in which setup. Every
# commented line is exactly "# " over real code, so an editor's uncomment
# makes the hooks live verbatim.
WORKFLOW_EXAMPLE_CONTENT = '''\
"""drcode workflow hooks (template).

ACTIVATION: rename this file to `workflow.py` (keep it in the project's
`.drcode/` directory) and uncomment the hooks you need. drcode scaffolds
this template on every start, but NEVER overwrites it — edit it in place.

Each defined hook runs in an isolated subprocess (cwd = project root, or
the task's worktree when per-task worktree isolation is on) and receives
one JSON payload:

    {"hook": "<name>",
     "task": {"id", "title", "kind", "github_issue"},
     "git":  {"branch": "<current branch or null>", "dirty": <bool>},
     "worktree": {"path", "branch"}        <- only with task isolation on,
     "project": {"path", "base_branch"}}   <- only on_research_accepted}

Every hook is optional — an absent function simply skips its step. A hook
failure (non-zero exit, timeout, malformed output) is NEVER fatal: it is
written into the task's service-event log and the pipeline moves on — with
one exception: on_review fails CLOSED (the task waits, the review is
retried with backoff), because an unreachable reviewer must not wave a
change through.

============================================================================
WHICH HOOKS TO UNCOMMENT (the three common setups)
============================================================================

1.  TASK ISOLATION + AUTOMATIC MERGE — the out-of-the-box degenerate
    accept. You turned `task_isolation: worktree` on (project settings)
    and want a working loop with no external review: every task develops
    inside its own worktree on branch `drcode/<task-id>`, and when the
    tester passes it the finished work is merged into your checkout
    automatically. Uncomment exactly THREE hooks below: on_review (the
    degenerate reviewer — accepts everything), on_accepted and
    on_research_accepted (the mergers). on_review MUST be defined for the
    merges to happen at all: the pipeline reaches on_accepted only
    through an accepted review, so the always-accept reviewer unlocks
    them. The helper functions between this docstring and the hooks are
    already active — no other edits are needed. Afterwards: task done and
    tester-passed means its commits are in your checkout's branch; each
    merge is recorded in the task's service-event log ("accepted: merged
    drcode/<id> into <branch>").

2.  TASK ISOLATION, MANUAL MERGE. Tasks work in worktrees, but you merge
    `drcode/<task-id>` into your branch by hand when you are ready.
    Uncomment NOTHING: without hooks the pipeline simply reminds you
    about every unmerged task branch.

3.  EXTERNAL REVIEW (CI, a human, a PR gate) instead of the degenerate
    one. Uncomment on_review and replace its body with real checks
    ({"verdict": "pending"} while checks run, {"verdict": "reject",
    "comment": "..."} sends the task back to development); keep
    on_accepted / on_research_accepted uncommented too if the merge
    itself should stay automatic — the real verdict then only decides
    WHEN it happens.

Optional extras (rarely needed):
    on_task_start — check out a branch / inject instructions into the
                    first prompt; only meaningful WITHOUT worktree
                    isolation (with isolation the task already works on
                    its own branch `drcode/<task-id>` in its own tree).
    on_development_done — push / open a PR before testing starts.
    on_tester_passed — install / deploy after every PASSED finalization.

SECURITY: workflow.py is arbitrary code executed with your full
permissions. Only activate a workflow file you have read and trust.
"""

import subprocess
from pathlib import Path


def _git(*args, cwd=None):
    """Shared helper: run git, return (returncode, combined output).
    Never raises."""
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=120
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _main_checkout(task):
    """Shared helper: the user's main checkout — the tree whose
    checked-out branch the task's work must finally land on.
    on_research_accepted carries it in the payload ("project"."path");
    for every other hook it is derived from git itself: the first entry
    of `git worktree list` is always the main working tree, and the
    hook's cwd (the task's worktree) belongs to the same repository."""
    path = (task.get("project") or {}).get("path")
    if path:
        return Path(path)
    code, out = _git("worktree", "list", "--porcelain")
    if code == 0 and out:
        return Path(out.splitlines()[0].removeprefix("worktree "))
    return Path.cwd()


def _merge_task_branch(task, what):
    """Shared helper: THE DEGENERATE ACCEPT. Merge the finished task's
    branch `drcode/<task-id>` into whatever branch the main checkout has
    checked out right now (a fast-forward when possible, a merge commit
    otherwise). A checkout with modified tracked files is left alone —
    the branch is reported for a manual merge instead of fighting the
    user's uncommitted work (untracked files, e.g. this very script, do
    not block the merge). With isolation off the work already IS in the
    shared tree: nothing to merge, returns None."""
    worktree = task.get("worktree")
    if not worktree:
        return None
    branch = worktree["branch"]
    main = _main_checkout(task)
    code, head = _git("-C", str(main), "rev-parse", "--abbrev-ref", "HEAD")
    base = head if code == 0 and head else "(detached HEAD)"
    code, status = _git("-C", str(main), "status", "--porcelain", "-uno")
    if code != 0:
        return {
            "text": f"{what}: branch {branch} NOT merged — {main} is not a "
            f"usable git checkout; merge by hand: git merge {branch}"
        }
    if status:
        return {
            "text": f"{what}: branch {branch} NOT merged into {base} — the "
            f"checkout has uncommitted changes; merge by hand when ready: "
            f"git merge {branch}"
        }
    code, out = _git("-C", str(main), "merge", "--no-edit", branch)
    if code != 0:
        # A conflicted merge must not stay half-done in the shared
        # checkout: every later hook-merge would cascade ("the checkout
        # has uncommitted changes"). Abort it — the tree returns to the
        # pre-merge state and the branch is reported for a manual merge
        # instead. The abort is best-effort: when the merge failed
        # without starting (e.g. an unknown branch) it is a harmless
        # no-op, and its failure must not mask the original one.
        _git("-C", str(main), "merge", "--abort")
        return {
            "text": f"{what}: merge of {branch} into {base} FAILED — the "
            f"conflicted merge was aborted ({main} is clean on {base} "
            f"again); sort it out by hand (git merge {branch}):\\n{out}"
        }
    return {"text": f"{what}: merged {branch} into {base}:\\n{out}"}


# def on_review(task):
#     """Degenerate accept, part 1 of 3: the always-yes reviewer. Needed
#     whenever you want the automatic merge (below) but have no external
#     review to gate it — the pipeline runs on_accepted only after an
#     accepted review, so this hook simply waves everything through.
#     Replace the body later with real checks (setup 3 in the module
#     docstring above)."""
#     return {"verdict": "accept"}


# def on_accepted(task):
#     """Degenerate accept, part 2 of 3: auto-merge for development tasks.
#     Runs right after the review accepted the tester-passed task (the
#     pipeline's only path to on_accepted). Merges `drcode/<task-id>`
#     into the main checkout's current branch. Return {"text": "..."} for
#     the task's service-event log."""
#     return _merge_task_branch(task, "accepted")


# def on_research_accepted(task):
#     """Degenerate accept, part 3 of 3: auto-merge for RESEARCH tasks
#     (they skip testing entirely and never reach on_accepted). Merges
#     the docs branch `drcode/<task-id>` the same way; the payload also
#     carries "project": {"path", "base_branch"} — _merge_task_branch
#     prefers it when present. Return {"text": "..."} for the task's
#     service-event log."""
#     return _merge_task_branch(task, "research accepted")


# def on_task_start(task):
#     """Optional extra: before the FIRST activation of a task. Only
#     meaningful WITHOUT worktree isolation (with isolation the task
#     already works on its own branch `drcode/<task-id>` in its own
#     worktree). MUST be idempotent: an abort leaves a dirty tree on the
#     branch. Return {"instructions": "..."} to inject text into the
#     coding agent's first prompt."""
#     subprocess.run(
#         ["git", "checkout", "-B", f"task/{task['task']['id']}"], check=True
#     )
#     return {
#         "instructions": f"Commit your work to branch task/{task['task']['id']}."
#     }


# def on_development_done(task):
#     """Optional extra: development finished, before testing (push,
#     `gh pr create`, ...). Return {"text": "..."} to append context to
#     the tester's prompt."""


# def on_tester_passed(task):
#     """Optional extra: right after the tester stamped the task PASSED
#     and it is finalized (runs on every path to COMPLETED-after-PASSED) —
#     the unconditional post-verification hook: install / deploy here.
#     Research tasks never trigger it (they skip testing). Return
#     {"text": "..."} for the task's service-event log."""
#     proc = subprocess.run(
#         ["uv", "tool", "install", ".", "--force"], capture_output=True, text=True
#     )
#     return {"text": proc.stdout or proc.stderr}
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
    "on_research_accepted",
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
        self,
        hook: str,
        task: Task,
        timeout: float = HOOK_TIMEOUT,
        *,
        cwd: Path | None = None,
        worktree: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> HookResult:
        """Run one hook: `python -c <bootstrap> <script>` in a subprocess
        with cwd = project root (or the task's worktree — see below); the
        bootstrap imports the script, feeds it the JSON payload on stdin
        and prints the hook's return value as a JSON object. The script
        itself never has to write dispatch code — it just defines the hook
        functions (see the module docstring).

        ``cwd`` (the pipeline passes the task's worktree root when
        per-task worktree isolation is on) is where the hook subprocess
        runs and what `git.branch`/`git.dirty` in the payload describe.
        ``worktree`` ({"path", "branch"}, when the task is isolated) is
        copied into the payload as the `worktree` key so the script can
        address the task's tree/branch explicitly. ``extra`` (optional)
        adds caller-provided top-level payload keys on top (e.g. the
        `project` entry of `on_research_accepted`).

        Never raises (except cancellation — an aborted run must kill the
        child and unwind); every failure comes back as
        ``HookResult(ok=False)`` so the callers only branch on `ok`.
        """
        if not self.has(hook):
            return HookResult(ok=False, error=f"hook {hook} is not defined")
        payload: dict[str, Any] = {
            "hook": hook,
            "task": {
                "id": task.id,
                "title": task.title,
                "kind": task.kind.value,
                "github_issue": task.github_issue,
            },
            "git": await self._git_state(cwd or self.project_root),
        }
        if worktree is not None:
            payload["worktree"] = worktree
        if extra:
            payload.update(extra)
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _BOOTSTRAP,
                str(self.script),
                cwd=cwd or self.project_root,
                env=sanitized_env(),
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

    async def _git_state(self, cwd: Path) -> dict[str, Any]:
        """Current git state of `cwd` (the project root, or the task's
        worktree under worktree isolation — the tree the hook runs in) for
        the hook payload: branch, dirty or not. Best-effort: outside a
        repository the branch is None and the tree counts as clean."""
        from . import tools  # local import: keeps the module import graph flat

        head = await tools.run_bash(cwd, "git rev-parse --abbrev-ref HEAD")
        branch: str | None = None
        if head.startswith("exit=0"):
            name = head.removeprefix("exit=0").strip()
            branch = name or None
        status = await tools.run_bash(cwd, "git status --porcelain")
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
