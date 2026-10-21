# drcode

Console AI coding agent (pydantic-ai + textual), inspired by opencode.
The center is not a chat session, but a **task** (feature request, bug report, or research question):

- one task is *active* at a time — work on it looks like a dialog that starts with the task description;
- while the agent works, you can create new tasks and put them in a **queue**;
- when the active task is marked done, the next queued task starts **automatically**;
- you can browse completed tasks and write into one — it re-enters the queue (or becomes active immediately if nothing is running).

## Appearance

![drcode TUI](screenshot.png)

## Install & run

From source (development):

```bash
uv sync
uv run drcode
```

Globally available from any project directory:

```bash
uv tool install .        # installs the `drcode` command into ~/.local/bin
# or, equivalently, with pipx:
pipx install .
# or plain pip (preferably into the user environment):
pip install .
```

Update after code changes: `uv tool install . --force` (pipx: `pipx upgrade .` or reinstall).
Uninstall: `uv tool uninstall drcode`.

The TUI operates on the **current directory** (`Path.cwd()`).

## Standalone binary

A self-contained bundle that ships its own Python interpreter and all
dependencies — no Python installation and no library versions required on the
target machine. Build it on each target OS/architecture (cross-compilation is
not supported):

```bash
bash scripts/build-binary.sh
# → dist/drcode/            (onedir bundle, run it as dist/drcode/drcode)
# → dist/drcode-<version>-<os>-<arch>.tar.gz   (distributable archive)
```

Install from a release archive:

```bash
tar -xzf drcode-<version>-<os>-<arch>.tar.gz
./drcode/drcode --version
```

The build produces an **onedir** bundle rather than a single-file executable
on purpose: a one-file PyInstaller binary unpacks its whole payload into a
fresh temp directory on every start, and on hardened macOS setups each
freshly written binary/dylib is then re-verified by the OS (hundreds of ms
per file), which made even `drcode --version` take 10+ seconds. With onedir
the payload is written once at unpack time and subsequent starts are fast
(~0.4s). Set `DRCODE_ONEFILE=1` to build a legacy single-file executable
anyway.

PyInstaller is provisioned automatically via the `build` dependency group
(`uv sync --group build`). The binary reads the same configuration (env vars,
`~/.config/drcode/drcode.json`, per-project `.env` / `.drcode/drcode.json`) as the
installed package.

## Configuration

Point it to any OpenAI-compatible API via env vars (or a `.env` file in the project root):

```bash
DRCODE_MODEL=gpt-4o
DRCODE_BASE_URL=https://api.openai.com/v1   # or your own endpoint
DRCODE_API_KEY=sk-...
DRCODE_MAX_CONTEXT_TOKENS=200000            # context window of the model
DRCODE_MAX_TOKENS=8192                      # per-response output cap
DRCODE_COMPACTION_THRESHOLD=80              # % fill at which history is compacted
```

If nothing anywhere provides a model and base URL, the TUI opens a **first-start setup dialog**: model and endpoint are required, the API key and context size are optional; Save probes the endpoint live (the verified availability endpoint is cached for cheaper later checks) and the values are persisted into `<project>/.drcode/drcode.json`. ESC skips it — the app stays usable without a model (tasks can be created and browsed, nothing runs).

Without `DRCODE_MODEL` / `DRCODE_API_KEY` every message fails with "Model is not configured". If the project root contains an **AGENTS.md**, its content is appended to the agent's instructions — and if it doesn't, the app offers on startup to create one with a pre-filled research task (the same one the Settings palette's "Update/create AGENTS.md" entry runs).

## Task pipeline

Tasks flow through a pipeline, mirrored by the sidebar: QUEUE → NEEDS CLARIFICATION → ACTIVE → TESTING → COMPLETED, plus ON HOLD.

- **Kinds** — every task is a `feature` (F), `bug` (B) or `research` (R). Each kind has a dedicated system prompt: feature tasks are encouraged to refactor and restructure, bug tasks must keep changes minimal, research tasks answer questions or update docs.
- **Auto-completion** — when the agent finishes a run with nothing pending, the task moves on by itself (as if you pressed done). Documentation tasks skip testing and complete right away.
- **Tester agent** — after every completed task a separate QA agent verifies the project: it detects available linters/tests/build commands, runs them, and labels the task ✅ passed or ❌ failed (only the tester can complete a task after testing). It receives the whole conversation — the original task plus any mid-flight clarifications — not just the first message. On success, in a git repository, it stages the changes, commits them with a message paraphrasing the original task (appending `Fixes #N` for tasks imported from GitHub issues) and best-effort pushes them to the remote. On failure it creates a **bug report task** at the head of the queue. Failed tester runs are retried automatically with capped exponential backoff.
- **Context compaction** — when the conversation fill reaches `DRCODE_COMPACTION_THRESHOLD` (80 % of the context window by default), a tool-less summarizer rewrites the transcript into a structured summary before the next turn; the original user prompts stay verbatim, so long-running tasks never hit the context ceiling.
- **Pipeline gate** — while any task is awaiting verification, no new development starts: testing always has priority over the queue.
- **Clarifying questions** — the agent can pause a task with a question (optionally with suggested answers rendered as clickable options). An answered task re-enters the pipeline at the head of the queue.
- **Interrupted work is never lost** — aborting (Esc) or an agent error leaves the partial work in the transcript; the next activation resumes the attempt without re-sending the prompt. A task whose run crashed is demoted to the queue and retried, while a dead gateway can't loop the whole queue through the error.
- **Stall watchdog** — a run that stops showing any sign of life (no stream deltas, no tool events) is aborted and restarted automatically after the configured stall limit (15/30 min, 1/2 hours, or never; 30 min by default, per project in `.drcode/settings.json`): the agent resumes via a continuation turn, the tester gets a fresh round. The same nudge is available manually with `ctrl+r` on an active or testing task. Every automatic nudge is logged into the task's service events.
- **Restart-safe** — on startup, stale active tasks are re-queued, and tasks stuck in TESTING are re-tested first, before any queued task is taken into development.
- **Failed-queue rescue** — a task whose run crashed (network outage, model error) is excluded from promotion for the rest of the session, so one broken task can't loop the queue. But when *every* queued task is failed and nothing is running, the exclusion is forcibly lifted and the backlog re-runs (with capped exponential backoff between rescue rounds, reset by any successful model call) — the queue keeps working instead of dead-ending until restart.
- **Queue ordering** — bug reports always come first (newest at the head, so tester-created bugs are picked up immediately); answered clarifications follow, then creation order. The kind of a still-queued task can be changed: re-typing to bug moves it to the head, to feature — to the tail.
- **Hold / delete / cancel** — park a queued task out of the backlog (input keeps accumulating while held, and an unanswered clarification is preserved), delete a held task entirely, or cancel a task (queued or already in TESTING): pending input is dropped and it completes immediately, skipping the tester.
- **Auto-cleanup** — completed tasks older than the retention period are deleted at startup. The setting (7/30/90 days or never) is per project, stored in `.drcode/settings.json`.

## GitHub Issues integration

drcode can turn a repository's GitHub issues into local tasks automatically. Configure it in the TUI (Settings → "GitHub integration"): enable switch, personal access token (needs repo read access), repository name (a pasted URL is normalized to `owner/repo`), and the sync interval (1/2/4 hours). The first cycle runs shortly after startup, then on schedule.

- Open issues labeled `bug` become local **bug reports**, labeled `enhancement` — **feature requests**.
- Issues labeled `invalid`/`wontfix`, or with neither actionable label, are dismissed: recorded as seen and never reconsidered (an unlabeled issue is picked up if it gets a label later).
- Every imported task carries the issue number; on success the tester appends `Fixes #N` to its commit message so GitHub links the commit back to the issue.
- Very long issue bodies are capped (8000 chars) so one pathological issue can't blow the agent's context.
- Sync results (including failures) are logged to the task history / service events and reported via push notifications.

A sync cycle can also be run headlessly, e.g. from cron: `drcode --git-hub-sync` (see below).

## Push notifications (ntfy.sh)

Per-project push notifications about pipeline events via the public [ntfy.sh](https://ntfy.sh) service — handy for overnight runs. Enable in the TUI (Settings → "Notifications") by picking an unguessable topic and subscribing to it in the ntfy app; messages carry the project folder name in the title.

Reported events:

- task status changes — except those directly caused by the user sitting at the computer (typing, cancelling, holding, pressing Done);
- GitHub sync cycles, with imported/skipped counts;
- settled pipeline states — the queue is empty, or tasks remain but none can be developed (all on hold, awaiting clarification or failed); these are deduplicated, so repeated pipeline re-evaluations never spam identical messages.

Delivery is fire-and-forget and can never stall the UI or an agent run: it happens in a short-lived daemon thread and every failure is swallowed (surfaced as an in-app toast). A notification that couldn't be sent (network down, timeout) is **not lost**: it is persisted in `.drcode/pending_notifications.json` and re-delivered in strict FIFO order, retrying every 15 minutes; a re-sent notification says so, with the moment it originally arose.

## Headless CLI

One-shot commands without the TUI (the pipeline itself is left to the next interactive start):

```bash
drcode --bug-report DESC          # create a bug report task
drcode --feature-request DESC     # create a feature request task
drcode --git-hub-sync             # run one GitHub Issues sync cycle and exit
```

`--bug-report` / `--feature-request` store the description as the task's first queued message (the title is derived from its first line) and print the created task's id. `--git-hub-sync` prints the cycle summary; exit codes: 0 success, 1 integration not configured, 2 sync failed.

## Tools & safety

The agent has filesystem and shell tools: `read_file`, `write_file`, `edit_file`, `glob_files`, `grep_search`, `run_bash`, `ask_clarifying_question`.

- Reading and searching work anywhere; writing is jailed to the project root and `/tmp` (symlink-aware). Violations are refused, logged into the task's service-event history and returned to the model as a retry prompt.
- `run_bash` inspects the command first and refuses `rm`/`mv`/`cp` whose path arguments escape the writable area.
- The tester runs with read-only tools; its single write exception is `git_commit`, allowed only after all checks passed.

## TUI

- Streaming chat: assistant answers and model reasoning render live in the chat log (markdown for answers, collapsible gray blocks for reasoning and tool calls, showing the tool name plus its primary argument).
- Sidebar grouped by pipeline stage with per-task markers: kind letter, status icon, verification label, viewed highlight. Sections collapse/expand via click, Enter or Space on the section header.
- Task status bar (top) with kind, id, creation date, status — and the **context fill** in the right corner: tokens and percent of `DRCODE_MAX_CONTEXT_TOKENS` (amber near the limit, red at it; the final value is restored from the saved transcript for finished tasks).
- Activity indicator: whether the agent/tester is making progress or has gone silent (hung request, long tool), with the currently running tool and its age.
- Model connection status, checked every 30 s; recovery from an outage un-blocks the tasks whose runs died with it.
- Working-agent badge in the terminal tab title (`F`/`B`/`R`/`T` for feature/bug/research/tester) — visible at a glance when drcode sits in a background tab.
- Copy on select: selecting text in the chat copies it to the clipboard (with a toast; `pbcopy` fallback on macOS).
- Multi-line chat input (Enter sends, Shift/Alt+Enter inserts a newline) that grows with the draft; the new-task form (Tab toggles the type, Ctrl+Enter creates).
- Settings palette (`ctrl+s`): system commands, themes, "Update/create AGENTS.md" service task (a pre-filled research task that writes or refreshes AGENTS.md), and the "Auto-delete completed tasks" / "Task finalization" / "Nudge stalled agent after" entries (current value shown after the colon; picking one opens the option-selection palette, Escape dismisses). "Task finalization" controls what happens after development: "Testing + Git commit + push" (default — the tester verifies, commits the verified changes and pushes them), "Testing" (the tester verifies but the `git_commit` tool is not registered, changes stay uncommitted) or "None" (no tester run — tasks go straight to Completed). The palette also holds the "GitHub integration: on/off" entry (token, repository, sync interval) and "Notifications: on/off" (ntfy.sh topic).
- Chat history is one chronological timeline: agent messages, tester rounds behind a "testing" separator, and service events (failures, skips, permission violations) with timestamps. None of the service lines ever reach the model context.

## Keys

| Key | Action |
| --- | --- |
| `ctrl+n` | New task (title + description, Tab toggles the type) |
| `ctrl+d` | Mark viewed task done → next queued starts |
| `ctrl+x` | Cancel viewed task (queued or testing): drop pending input, complete without testing |
| `ctrl+t` | Change type of a queued task (bug → head, feature → tail) |
| `ctrl+up` | Move a queued task one step up the backlog (away from execution) |
| `ctrl+down` | Move a queued task one step down the backlog (closer to execution) |
| `ctrl+h` | Hold / resume viewed task |
| `ctrl+delete` | Delete viewed held task (from the sidebar and disk) |
| `ctrl+s` | Settings palette (themes, AGENTS.md refresh, auto-cleanup, GitHub integration, notifications) |
| `ctrl+o` | Focus task sidebar (↑/↓ + Enter to switch tasks) |
| `ctrl+i` | Focus chat input |
| `escape` | Abort the running agent/tester |
| `ctrl+r` | Nudge viewed task: abort its hung run and restart it (agent continues, tester re-verifies) |
| `ctrl+q` | Quit |

## Storage

Tasks and full transcripts (including tool calls) are stored as JSON under
`.drcode/tasks/` in the project root (atomic writes; corrupted files are
moved to `.json.bak`, not deleted). Tester runs are persisted separately from
the coding agent's transcript, so they never leak into its model history.
`.drcode/` is gitignored. Alongside the tasks it holds the per-project
`settings.json` (retention, finalization, stall nudge, GitHub integration,
notifications), `github_state.json` (sync bookkeeping: last cycle, seen
issues — imported issues are never reconsidered) and
`pending_notifications.json` (deferred ntfy.sh deliveries).
