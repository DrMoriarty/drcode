# AGENTS.md

## Commands

```bash
uv sync                  # install deps into .venv (Python 3.12, src layout)
uv run drcode            # run the TUI — operates on the CURRENT directory (Path.cwd())
uv run ruff check src    # lint (no config; ruff defaults)
uv run ruff format src
uv run pyright           # typecheck (dev dep; no config)
```

No tests, CI, or pre-commit hooks exist — verify changes manually with ruff + pyright, and exercise the TUI by launching `uv run drcode` in a scratch directory.

## Configuration gotcha

The agent needs `DRCODE_MODEL` and `DRCODE_API_KEY` (env vars or `.env` in the repo root; see `src/drcode/config.py`). Without them the TUI starts fine but every message fails with "Model is not configured". `DRCODE_BASE_URL` defaults to the OpenAI API; any OpenAI-compatible endpoint works.

## Architecture

Task-centric coding agent (pydantic-ai + textual), inspired by opencode. Single package under `src/drcode/`:

- `task_manager.py` — state machine: queued → active → testing → completed, plus awaiting-clarification: the agent's `ask_clarifying_question` tool pauses the task on a question and the pipeline advances; when the user answers, the task re-enters the pipeline at the head of the queue (activated once no testing is pending and nothing is active — promotion is up to the app, see `tui/app.py`). At most one ACTIVE task; when it's marked done (development finished) it moves to TESTING — only the tester moves it to COMPLETED via `set_verification` (which also stamps the passed/failed label). A task also auto-completes (goes to TESTING) when the agent finishes its run with no pending input, no clarification and no unfinished attempt; aborts and agent errors keep the task paused via `task.unfinished` instead. Sending input to a COMPLETED/TESTING task re-queues it (the old verification label is cleared). On startup any stale ACTIVE task is demoted to queued, and all tasks stuck in TESTING are re-tested FIRST, before any queued task is taken into development (awaiting-clarification tasks survive restarts). `promote_next` activates a queued task that has pending input OR an unfinished attempt (auto-resume of interrupted work). `cancel` (user: Ctrl+X) drops all pending input and completes the task immediately — the tester is skipped everywhere a finished task would run it. `documentation` tasks never go to TESTING: `mark_done` completes them right away (the tester is skipped everywhere a finished task would run it).
- `models.py` — `Task` holds the full pydantic-ai transcript as serialized `ModelMessage` dicts (`task.messages`); queued user input lives in `task.pending_input` until the agent run consumes it. `task.unfinished` marks an interrupted attempt (abort/agent error): the consumed prompt is NEVER re-queued — it stays in the transcript at its chronological position, and the next activation resumes the attempt without re-sending text. Tester runs are persisted separately in `task.tester_messages` (one serialized transcript per testing round) so they never leak into the coding agent's model history; the chat renders each round behind a "testing" separator. `task.events` (`TaskEvent`: timestamp + text) is the service-event log (tester/agent failures, skips) — shown in the chat under a "service events" separator, but never fed to the model, since it lives outside both transcripts.
- `storage.py` — persists each task as `<project>/.drcode/tasks/<id>.json` (atomic tmp-file write; corrupted files are moved to `.json.bak`, not deleted). `.drcode/` is gitignored.
- `agent.py` — builds the pydantic-ai `Agent` with tools registered inline; `deps` is just the project root (`Path`). `serialize_messages`/`load_messages` round-trip transcripts; `resumable_history` trims an incomplete tail (unanswered tool calls / bare tool results) before a transcript is resumed; `sanitized_messages` applies the same trim at interrupt time so the stored transcript stays API-valid no matter in which order later messages are appended; `ends_with_open_request` tells whether a continuation turn has something to resume; `extract_display` renders them for the chat UI.
- `tools.py` — filesystem/shell tools for the agent. All paths go through `resolve_path`, which rejects anything escaping the project root. Grep/glob skip `IGNORED_DIRS` (add new cache dirs there).
- `tui/app.py` — textual app. `run_agent` is a `@work(exclusive=True, group="agent")` worker whose loop processes messages strictly chronologically: pending input in arrival order, then at most one continuation turn (`agent.iter(None, …)`) when `task.unfinished` and the transcript ends with an unanswered request; the interrupted prompt is never re-queued, and both the cancel and error handlers sanitize the transcript instead of re-queueing text. Agent errors push the failed attempt to `unfinished` + demote the task to QUEUED so nothing is lost. Pipeline advancement goes through the single `_advance()` gate: while any task is TESTING, no new development ever starts — the oldest TESTING task is (re)tested first, and a failed tester run is retried with capped exponential backoff (the pipeline waits; the retry timer re-advances). Cancelled workers (abort, superseded runs) do not restart the pipeline. Ctrl+X cancels the viewed task (`manager.cancel`): pending input dropped, completed immediately, no testing. Service events are persisted via `_log_event`/`_show_event` regardless of which task is open in the chat.
