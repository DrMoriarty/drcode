# Custom team workflow

By default drcode works the same for everyone: the coding agent works in the
current branch and never commits, the tester commits and pushes after a
successful verification. A team that needs its own branching / merge / review
process can enable a **custom workflow** — without changing any UI and without
new task statuses — by adding one or both files to the project:

| File | Layer | Effect |
|---|---|---|
| `.drcode/git-workflow.md` | declarative | prose rules injected into the agent prompts |
| `.drcode/workflow.py` | hard gates | lifecycle hooks the pipeline calls |

Both layers are enabled by the **mere presence of the file**; there is nothing
to toggle in the UI (the Settings menu shows a "Workflow:" line describing
what is active). A single developer without these files keeps the default
behaviour unchanged.

## Layer 1 — `.drcode/git-workflow.md`

The team's process rules in plain language. When the file exists:

- in the system prompts of the coding agents (feature / bug / research) it
  **replaces** the built-in "Never commit to git" sentence — committing
  becomes part of the workflow, governed by the file's text;
- it is **appended** to the tester's system prompt, so the finalization
  (commit / push / PR) follows the same team rules.

A project may also override any bundled system prompt entirely by placing a
file with the same name into `.drcode/prompts/` (e.g.
`.drcode/prompts/feature.md`, `.drcode/prompts/tester.md`).

## Layer 2 — `.drcode/workflow.py`

A Python script with optional hook functions the pipeline calls at fixed
lifecycle points. Every function is optional — an absent function simply
skips its step. Each call runs in an isolated **subprocess** (the same
isolation model as the agent's `run_bash`): the hook receives one JSON
payload and returns a JSON-serializable dict; a crashing or hanging script is
logged into the task's service events and is never fatal (with one
exception — see `on_review`).

```python
import subprocess

def on_task_start(task):
    """Before the FIRST activation of a task (branch checkout etc.).
    MUST be idempotent: an abort leaves a dirty tree on the branch (the
    resume continues in it), a crash leaves the checkout on the old branch,
    and the previous task may be unmerged."""
    branch = f"task/{task['task']['id']}"
    subprocess.run(["git", "checkout", "-B", branch], check=True)
    return {"instructions": f"Commit your work to branch {branch}."}

def on_development_done(task):
    """Development finished, before testing (push, open a PR, ...).
    The returned text is appended to the tester's context."""
    subprocess.run(["git", "push", "-u", "origin", "HEAD"], check=True)
    url = subprocess.run(["gh", "pr", "create", "--fill"], capture_output=True, text=True).stdout
    return {"text": f"Pull request: {url}"}

def on_review(task):
    """External code review after the tester passed the task.
    accept  -> {"verdict": "accept"}
    reject  -> {"verdict": "reject", "comment": "what to fix"}
    pending -> None or {"verdict": "pending"} (CI still running, a human
               has not reviewed yet) — the task is released and the verdict
               is polled later."""

def on_accepted(task):
    """After an accept (merge the PR, delete the branch, ...). The
    returned text goes to the task's service-event log."""
    subprocess.run(["gh", "pr", "merge", "--squash", "--delete-branch"], check=True)
    return {"text": "PR merged, branch deleted."}
```

The payload passed to every hook:

```json
{
  "hook": "<name>",
  "task": {"id": "...", "title": "...", "kind": "feature|bug|research",
           "github_issue": 42},
  "git":  {"branch": "main", "dirty": false}
}
```

Behaviour per hook:

- **`on_task_start`** — runs once, before the task's first activation.
  `{"instructions": "..."}` is injected at the top of the coding agent's
  first prompt. A failure is logged and the run proceeds.
- **`on_development_done`** — runs when a development round finishes and
  before the tester round. Its `{"text": ...}` reaches the tester's prompt.
  Research tasks skip it. A failure is logged and testing proceeds.
- **`on_review`** — the external review gate. After the tester stamps the
  task PASSED, the pipeline calls `on_review` (the task stays in TESTING —
  no new development starts while a verdict is outstanding, exactly like an
  in-flight tester round):
  - `accept` → the task completes (with the "Verified successfully" label)
    and `on_accepted` runs;
  - `reject` → the reviewer's `comment` becomes the task's new user input
    and the task re-enters the queue — the coding agent reworks it **in the
    same transcript**, seeing the full history plus the review remarks;
  - anything else (`None`, `"pending"`) → the task is **released** (it
    completes with its PASSED label and an "awaiting review" marker), the
    pipeline continues with other work, and a periodic poll re-asks
    `on_review` (every 60 s) until a final verdict arrives. "Awaiting
    review" is therefore a COMPLETED task with an open PR — no new status.
  - a hook **failure is fail-closed**: the task keeps waiting and the call
    is retried with backoff (30 s doubling up to 10 min). An unreachable
    reviewer must never wave a change through.
- **`on_accepted`** — runs after every accept (both right after a
  synchronous review and from the async poll), in the sync mode immediately
  and in the async mode when the poll detects the merge/accept.

Loop protection: after **3 consecutive rejects** the task is put on hold
("requires human attention" is written to its service events; Ctrl+P
resumes it). Any human input to the task or its cancellation resets the
counter.

Tasks awaiting an async review verdict are excluded from the
auto-cleanup of completed tasks (the retention setting), so a verdict that
arrives days later still finds its task.

## Git commit behaviour

The tester's `git_commit` tool is **not** disabled by a custom workflow: a
hook that checked out a feature branch receives the tester's commit and push
on that branch (`git push` falls back to `git push --set-upstream origin
HEAD`), leaving the default branch untouched. Disabling commits entirely
remains the existing per-project setting "Task finalization: Testing".

## Sharing with the team

`.drcode/git-workflow.md`, `.drcode/workflow.py` and `.drcode/drcode.json`
(the model picks) are **committable** — drcode's `.gitignore` management
ignores only the private state (`.drcode/tasks/`, `.drcode/settings.json`,
`.drcode/github_state.json`), and older projects whose `.gitignore` carries
the legacy whole-directory `.drcode/` entry are migrated to the specific
set automatically.

## Security

`.drcode/workflow.py` is **arbitrary code executed with your full user
permissions** — the same trust level as the shell commands the agent itself
runs. This is a deliberate design decision, but it must be stated plainly:
a workflow file coming from an untrusted repository is NOT safe. Always
review `.drcode/workflow.py` before running drcode in a project you did not
author (the same advice as for pre-commit hooks or Makefiles).
