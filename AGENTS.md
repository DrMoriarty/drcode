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

- `task_manager.py` — state machine: queued → active → testing → completed, plus awaiting-clarification: the agent's `ask_clarifying_question` tool pauses the task on a question and the next queued task starts; when the user answers, the task re-enters the pipeline at the head of the queue (or is activated immediately if nothing is active). At most one ACTIVE task; when it's marked done (development finished) it moves to TESTING — only the tester moves it to COMPLETED via `set_verification` (which also stamps the passed/failed label). A task also auto-completes (goes to TESTING) when the agent finishes its run with no pending input and no clarification (the tester runs next and promotes the queue); aborts and agent errors keep the message in `pending_input`, so those tasks do not advance. Sending input to a COMPLETED/TESTING task re-queues it (the old verification label is cleared). On startup any stale ACTIVE task is demoted to queued, and all tasks stuck in TESTING are re-tested FIRST, before any queued task is taken into development (awaiting-clarification tasks survive restarts).
- `models.py` — `Task` holds the full pydantic-ai transcript as serialized `ModelMessage` dicts (`task.messages`); queued user input lives in `task.pending_input` until the agent run consumes it.
- `storage.py` — persists each task as `<project>/.drcode/tasks/<id>.json` (atomic tmp-file write; corrupted files are moved to `.json.bak`, not deleted). `.drcode/` is gitignored.
- `agent.py` — builds the pydantic-ai `Agent` with tools registered inline; `deps` is just the project root (`Path`). `serialize_messages`/`load_messages` round-trip transcripts; `extract_display` renders them for the chat UI.
- `tools.py` — filesystem/shell tools for the agent. All paths go through `resolve_path`, which rejects anything escaping the project root. Grep/glob skip `IGNORED_DIRS` (add new cache dirs there).
- `tui/app.py` — textual app. `run_agent` is a `@work(exclusive=True, group="agent")` worker; abort/done cancel that worker group. Agent errors push the failed message back onto `pending_input` so nothing is lost.
