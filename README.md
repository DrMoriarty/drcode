# drcode

Console AI coding agent (pydantic-ai + textual), inspired by opencode.
The center is not a chat session, but a **task** (feature request, bug report, or documentation question):

- one task is *active* at a time — work on it looks like a dialog that starts with the task description;
- while the agent works, you can create new tasks and put them in a **queue**;
- when the active task is marked done, the next queued task starts **automatically**;
- you can browse completed tasks and write into one — it re-enters the queue (or becomes active immediately if nothing is running).

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

A self-contained executable that bundles its own Python interpreter and all
dependencies — no Python installation and no library versions required on the
target machine. Build it on each target OS/architecture (cross-compilation is
not supported):

```bash
bash scripts/build-binary.sh
# → dist/drcode  (one-file Mach-O / ELF executable, ~45 MB)
```

PyInstaller is provisioned automatically via the `build` dependency group
(`uv sync --group build`). The binary reads the same configuration (env vars,
`~/.config/drcode/drcode.json`, per-project `.env` / `drcode.json`) as the
installed package.

## Configuration

Point it to any OpenAI-compatible API via env vars (or a `.env` file in the project root):

```bash
DRCODE_MODEL=gpt-4o
DRCODE_BASE_URL=https://api.openai.com/v1   # or your own endpoint
DRCODE_API_KEY=sk-...
DRCODE_MAX_CONTEXT_TOKENS=200000            # context window of the model
```

Without `DRCODE_MODEL` / `DRCODE_API_KEY` the TUI starts fine, but every message fails with "Model is not configured". If the project root contains an **AGENTS.md**, its content is appended to the agent's instructions.

## Task pipeline

Tasks flow through a pipeline, mirrored by the sidebar: QUEUE → NEEDS CLARIFICATION → ACTIVE → TESTING → COMPLETED, plus ON HOLD.

- **Kinds** — every task is a `feature` (F), `bug` (B) or `documentation` (D). Each kind has a dedicated system prompt: feature tasks are encouraged to refactor and restructure, bug tasks must keep changes minimal, documentation tasks answer questions or update docs.
- **Auto-completion** — when the agent finishes a run with nothing pending, the task moves on by itself (as if you pressed done). Documentation tasks skip testing and complete right away.
- **Tester agent** — after every completed task a separate QA agent verifies the project: it detects available linters/tests/build commands, runs them, and labels the task ✅ passed or ❌ failed (only the tester can complete a task after testing). On success, in a git repository, it stages the changes and commits them with a message paraphrasing the original task. On failure it creates a **bug report task** at the head of the queue. Failed tester runs are retried automatically with capped exponential backoff.
- **Pipeline gate** — while any task is awaiting verification, no new development starts: testing always has priority over the queue.
- **Clarifying questions** — the agent can pause a task with a question (optionally with suggested answers rendered as clickable options). An answered task re-enters the pipeline at the head of the queue.
- **Interrupted work is never lost** — aborting (Esc) or an agent error leaves the partial work in the transcript; the next activation resumes the attempt without re-sending the prompt. A task whose run crashed is demoted to the queue and retried, while a dead gateway can't loop the whole queue through the error.
- **Restart-safe** — on startup, stale active tasks are re-queued, and tasks stuck in TESTING are re-tested first, before any queued task is taken into development.
- **Queue ordering** — bug reports always come first (newest at the head, so tester-created bugs are picked up immediately); answered clarifications follow, then creation order. The kind of a still-queued task can be changed: re-typing to bug moves it to the head, to feature — to the tail.
- **Hold / delete / cancel** — park a queued task out of the backlog (input keeps accumulating while held, and an unanswered clarification is preserved), delete a held task entirely, or cancel a task: pending input is dropped and it completes immediately, skipping the tester.
- **Auto-cleanup** — completed tasks older than the retention period are deleted at startup. The setting (7/30/90 days or never) is per project, stored in `.drcode/settings.json`.

## Tools & safety

The agent has filesystem and shell tools: `read_file`, `write_file`, `edit_file`, `glob_files`, `grep_search`, `run_bash`, `ask_clarifying_question`.

- Reading and searching work anywhere; writing is jailed to the project root and `/tmp` (symlink-aware). Violations are refused, logged into the task's service-event history and returned to the model as a retry prompt.
- `run_bash` inspects the command first and refuses `rm`/`mv`/`cp` whose path arguments escape the writable area.
- The tester runs with read-only tools; its single write exception is `git_commit`, allowed only after all checks passed.

## TUI

- Streaming chat: assistant answers and model reasoning render live in the chat log (markdown for answers, collapsible gray blocks for reasoning and tool calls, showing the tool name plus its primary argument).
- Sidebar grouped by pipeline stage with per-task markers: kind letter, status icon, verification label, viewed highlight.
- Task status bar (top) with kind, id, creation date, status — and the **context fill** in the right corner: tokens and percent of `DRCODE_MAX_CONTEXT_TOKENS` (amber near the limit, red at it; the final value is restored from the saved transcript for finished tasks).
- Activity indicator: whether the agent/tester is making progress or has gone silent (hung request, long tool), with the currently running tool and its age.
- Model connection status, checked every 30 s; recovery from an outage un-blocks the tasks whose runs died with it.
- Copy on select: selecting text in the chat copies it to the clipboard (with a toast; `pbcopy` fallback on macOS).
- Multi-line chat input (Enter sends, Shift/Alt+Enter inserts a newline) that grows with the draft; the new-task form (Tab toggles the type, Ctrl+Enter creates).
- Settings palette (`ctrl+s`): system commands, themes, "Обновить/создать AGENTS.md" service task (a pre-filled documentation task that writes or refreshes AGENTS.md), and the "Auto-delete completed tasks" entry (current value shown after the colon; picking it opens the option-selection palette, Escape dismisses).
- Chat history is one chronological timeline: agent messages, tester rounds behind a "testing" separator, and service events (failures, skips, permission violations) with timestamps. None of the service lines ever reach the model context.

## Keys

| Key | Action |
| --- | --- |
| `ctrl+n` | New task (title + description, Tab toggles the type) |
| `ctrl+d` | Mark viewed task done → next queued starts |
| `ctrl+x` | Cancel viewed task: drop pending input, complete without testing |
| `ctrl+t` | Change type of a queued task (bug → head, feature → tail) |
| `ctrl+h` | Hold / resume viewed task |
| `ctrl+delete` | Delete viewed held task (from the sidebar and disk) |
| `ctrl+s` | Settings palette (themes, AGENTS.md refresh, auto-cleanup) |
| `ctrl+o` | Focus task sidebar (↑/↓ + Enter to switch tasks) |
| `ctrl+i` | Focus chat input |
| `escape` | Abort the running agent/tester |
| `ctrl+q` | Quit |

## Storage

Tasks and full transcripts (including tool calls) are stored as JSON under
`.drcode/tasks/` in the project root (atomic writes; corrupted files are
moved to `.json.bak`, not deleted). Tester runs are persisted separately from
the coding agent's transcript, so they never leak into its model history.
`.drcode/` is gitignored.
