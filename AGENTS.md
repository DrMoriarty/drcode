# AGENTS.md

## Commands

```bash
uv sync                                    # install deps into .venv (Python 3.12, src layout)
uv run drcode                              # run the TUI — operates on the CURRENT directory (Path.cwd())
uv run pytest                              # whole suite — parallel by default (pytest-xdist, -n auto via addopts), ~2.5–3 min
uv run pytest tests/test_task_manager.py -k hold   # focused run (also parallel)
uv run pytest -n 0                         # serial run — for debugging (-s/--pdb; -p no:xdist needs -o addopts="" too)
uv run ruff check src                      # lint (no config; ruff defaults)
uv run ruff format src
uv run pyright                             # typecheck (dev dep; no config)
```

- No CI runs tests and there are no pre-commit hooks — local `ruff` + `pyright` + `pytest` is the only gate. Exercise the TUI by launching `uv run drcode` in a scratch directory.

## Version & packaging

- The bare `version` file at the repo root is the single source of truth for the app version (hatchling custom pattern; `src/drcode/version.py` exposes `APP_VERSION`). Bump it there — never in `pyproject.toml` (`version` is dynamic).
- System/compaction/init prompts are package data in `src/drcode/prompts/`, read as text at import time (`prompts.py`) and shipped in the wheel via the pyproject `include`.
- Standalone binary: `bash scripts/build-binary.sh` (provisions PyInstaller via `uv sync --group build`; platform-specific, no cross-compilation). Onedir on purpose — see README; `DRCODE_ONEFILE=1` builds the legacy single-file. `drcode_entry.py` is the PyInstaller stub; it sets `PYDANTIC_DISABLE_PLUGINS=logfire-plugin,logfire` because the logfire pydantic plugin crashes in a frozen app.
- Headless CLI (entry point `drcode:main`): `drcode --bug-report DESC` / `--feature-request DESC` create a task without the TUI; `--git-hub-sync` runs one GitHub Issues sync cycle. `drcode-models` (entry point `drcode.tui.model_setup_app:main`) edits the user catalog.
- Releases are tag-driven: pushing a `v*` tag runs `.github/workflows/build-binaries.yml` (onedir bundles for macOS arm64, Linux x86_64/arm64, Windows, attached to a GitHub Release).

## Custom team workflow (see docs/workflow.md)

Two optional layers, enabled by the **mere presence** of the file in the user's project — nothing to toggle:

- `.drcode/git-workflow.md` — prose rules injected into the agent prompts; **replaces** the built-in "Never commit to git" sentence in the coding agents' system prompts and is **appended** to the tester's.
- `.drcode/workflow.py` — lifecycle hooks the pipeline calls (each in an isolated subprocess, crashing is logged, never fatal). Default pipeline `on_accepted` lives in `src/drcode/workflow.py` (`_run_accepted_hook`/`_run_research_accepted_hook` in `tui/pipeline.py` run it under an `asyncio.Lock` — two tasks finalizing in parallel must not race a `git merge` into the shared checkout).
- Any bundled prompt can be overridden by a same-named file in `.drcode/prompts/` (e.g. `feature.md`).

## Configuration gotcha

Model selection is two-level (see `src/drcode/providers.py`):

- **User level** (`~/.config/drcode/`, XDG_CONFIG_HOME-aware) — the provider/model catalog `models.json` (`UserModelsConfig`: providers with base_url/api_key/health_check_url, their `ModelEntry`s and per-role `defaults`) plus non-provider settings in `drcode.json`. The catalog is written ONLY by the `drcode-models` utility or the main TUI's "Provider catalog…" screen (same `ModelCatalogScreen`, always visible in Settings); models are auto-discovered via `agent_model.fetch_models`.
- **Project level** (`.drcode/drcode.json`) — per-role picks: `models[role]` is either `"provider/model-id"` (a catalog reference) or `"custom"` (use `custom_models[role]`, a full connection stored per role — custom models of different roles never mix). null/absent slot = user default. Selecting a catalog model does NOT delete the parked `custom_models[role]`; the picker offers it back as "Custom: <model-id>".

Facts an agent is likely to get wrong:

- `Config` (`src/drcode/config.py`) keeps the model fields as an env-driven container (`DRCODE_*` vars > project `.env`), but its json layers deliberately NEVER provide them (`_PROVIDER_FIELDS`). Legacy flat model keys in drcode.json files are consumed by idempotent startup migrations (`migrate_user_config`, `migrate_project_config`, run from `DrCodeApp.__init__` before the Config is built); after that the flat keys are ignored.
- Non-provider json settings (compaction_threshold, request_limit) layer: env > project json > user json > defaults; a corrupted json file is skipped, never fatal. On first start the user config dir is scaffolded with `drcode.json.example` — when adding a new non-provider user-configurable field without a sensible default, add a placeholder to `_EXAMPLE_PLACEHOLDERS` in `config.py`.
- Project-scoped user preferences (retention, stall nudge, task isolation, parallel tasks, GitHub sync, ntfy notifications) live separately in `<project>/.drcode/settings.json` (`src/drcode/settings.py`, `ProjectSettings`; `parallel_tasks` mirrors the `DEV_ROUNDS_OPTIONS`/`dev_rounds_label` pattern with `PARALLEL_TASKS_OPTIONS`/`parallel_tasks_label`).
- Effective per-role connection: `providers.resolve(config, role)` — precedence `DRCODE_MODEL`/`DRCODE_BASE_URL` env (all roles) > project `models[role]` (ref or custom) > catalog default > "not set" (empty model: pipeline idles — there is NO flat-config fallback). The TUI resolves through `providers.resolve_effective`, which adds one exception: the RESEARCH role with nothing selected anywhere falls back to the resolved CODING model.
- Each `ModelEntry` carries the model's token caps — `max_context_tokens`, optional `max_tokens`, optional per-model `compaction_threshold` (null = inherit the layered config's value) — so model–context–compaction-limit is one unit. Compaction gates (`pipeline._fill_over_threshold`) always take a role and use that role's resolved config (context size AND threshold): a research transcript is measured against the research model's window, the tester round against the tester model's.
- First-start dialog (`ModelSetupDialog`): shown only when env is unset AND the user catalog has no providers AND the project has no active model entry (`project_has_model_selection`). Save stores the entered model as a CUSTOM model for coding/tester/compaction (independent copies); research is deliberately left unpinned so it inherits via the fallback. ESC skips — the app runs without a model (tasks still work, pipeline idles).

## Architecture

Task-centric coding agent (pydantic-ai + textual), inspired by opencode. Single package under `src/drcode/`:

- `task_manager.py` — state machine: queued → active → testing → completed, plus awaiting-clarification (the agent's `ask_clarifying_question` tool pauses the task; answering re-enters it at the head of the queue). Key rules:
  - Only the tester moves a task to COMPLETED (via its `set_verification` tool). A task auto-moves to TESTING when the agent finishes its run with no pending input, no clarification and no unfinished attempt; aborts/errors keep it paused via `task.unfinished` instead. Sending input to a COMPLETED/TESTING task re-queues it (verification label cleared).
  - `research` tasks never go to TESTING (`mark_done` completes them right away).
  - On startup: stale ACTIVE → queued; tasks stuck in TESTING are re-tested FIRST, before any queued task enters development.
  - `promote_next` activates a queued task that has pending input OR an unfinished attempt (auto-resume) — never a held one (`set_hold`, Ctrl+H; `delete` only works on held tasks). `cancel` (Ctrl+X) drops pending input and completes immediately, skipping the tester.
  - Queue order is `(dependency sink, bugs newest-first, queue_head, queue_seq)` — tester-created bug reports land at the very head. Manual reordering (`move_in_queue`, Ctrl+Up/Down) re-stamps `queue_seq` and never crosses the bug block; `set_kind` (Ctrl+T) re-positions too (bug → head, feature → tail).
  - Dependencies (`task.depends_on`, edited only via `validate_dependencies`-gated `set_dependencies`): promotion skips blocked tasks; the gate is dep COMPLETED **and merged** — mergedness is checked dynamically at every attempt via `task_env.unmerged_ids`, so a manual `git merge drcode/<dep>` unblocks dependents by itself, and a deleted dep task is judged by its surviving branch alone.
- `models.py` — `Task` holds the full pydantic-ai transcript as serialized `ModelMessage` dicts; queued input lives in `task.pending_input` until consumed. `task.unfinished` marks an interrupted attempt: the consumed prompt is NEVER re-queued — it stays in the transcript at its chronological position, and the next activation resumes without re-sending text. Tester transcripts persist separately in `task.tester_messages` (they never leak into the coding agent's history); `task.events` is a service-event log shown in chat but never fed to the model. `task.github_issue` (imported issues) makes the tester append "Fixes #N" to its commit message.
- `storage.py` — each task is `<project>/.drcode/tasks/<id>.json` (atomic write; corrupted files → `.json.bak`, never deleted). `.drcode/` is gitignored (enforced by `gitignore.py`).
- `agent.py` — thin orchestrator only; re-exports the split-out modules so the TUI and tests import everything from `drcode.agent`:
  - `agent_model.py` — model/network layer: `OpenAIChatModel` builder, `probe_model`, `fetch_models` (context size probed from optional gateway keys, else 200k default + "needs confirmation" flag).
  - `tool_runtime.py` — runs blocking tools in a worker thread (a stall would freeze the Textual UI); fixable errors (`ValueError`/`OSError`/`re.error`/`PathViolation`) become `ModelRetry`, everything else crashes the run.
  - `transcript.py` — transcript round-trip; `resumable_history`/`sanitized_messages` trim an incomplete tail (unanswered tool calls / bare tool results) so the stored transcript stays API-valid no matter the append order; usage/context-token accounting.
  - `compaction.py` — at `compaction_threshold` a tool-less summarizer rewrites the transcript into an anchored summary; user prompts stay verbatim, the summary rides in a synthetic message prefixed `COMPACTION_SUMMARY_PREFIX` (fed back on the next compaction).
- `tools.py` — reading works anywhere on the filesystem; **writing is jailed** to the project root and the task's scratch dir `/tmp/drcode/<task-id>` (symlink-aware — `/private/tmp` on macOS counts too); violations raise `PathViolation`. `run_bash` refuses `rm`/`mv`/`cp` whose path arguments escape the writable area and sets the child's `TMPDIR` to the scratch dir. Scratch dir is created lazily, removed on COMPLETED/delete, orphans cleaned at startup. Grep/glob skip `IGNORED_DIRS` (add new cache dirs there).
- `github.py` — Issues sync (project settings `github_*`): open issues labeled `bug`/`enhancement` become local tasks; `invalid`/`wontfix` and unlabeled ones are dismissed (recorded as seen, never reconsidered).
- `task_env.py` — optional per-task worktree isolation (`task_isolation: worktree`): each task gets `.drcode/worktrees/<id>` on branch `drcode/<id>`; **drcode never merges on its own** (merging is the workflow hook's or the user's job). `unmerged_ids(task_ids)` is the mergedness half of the dependency gate; with isolation off it degenerates to "COMPLETED". Blocking git calls run via `asyncio.to_thread`; `ensure_root` serializes its git-touching body behind a module-level `threading.Lock` so parallel slot activations don't race (fast-path checks stay outside the lock).
- `tui/` — `app.py` is the textual app; agent/pipeline logic lives in `pipeline.py` as a mixin (`run_agent`, the single `_advance()` gate), interface in `state.py`. Parallel development slots:
  - `parallel_tasks` (options 1–4, default 1) is effective only under `task_isolation: worktree` in a git repo (`parallel_limit` property; one-time startup warning when the persisted value is dormant).
  - `run_agent` is a plain dispatcher: `@work(exclusive=True, group="agent-<task_id>")` per task — exclusivity is WITHIN a task, other tasks' workers are untouched. Run state is **counted, not singular**: `_agent_runs: dict[task_id → generation]`, `agent_running = bool(self._agent_runs)`, and activity/watchdog state is keyed by (task, generation). The tester stays SINGLE (`group="tester"`), never occupies a development slot (it frees at `mark_done`), and while any task is TESTING no new development starts.
  - Pipeline advancement goes through the single `_advance()` gate, whose tail `_dispatch_promotions(blocked)` fills free slots. Agent errors push the attempt to `unfinished` + demote the task to QUEUED so nothing is lost. A failed tester run is retried with capped exponential backoff (the retry timer re-advances).
  - Stall watchdog (project setting): each running task has its OWN stall deadline; `_nudge_run` aborts/restarts only that task's group — same nudge is manual (Ctrl+R). ESC (`action_abort_run`) is a global stop only when `parallel_limit == 1`; above 1 it stops only the viewed task's run.
  - When any queued task has `depends_on`, the promotion tail moves to the `@work(group="promote")` worker `_promote_with_dependencies` (computes the blocked set off the event loop; block/unblock service events deduplicated per session); a 30s `_poll_dependency_unblock` tick re-fires `_advance()` so a manual `git merge` unblocks dependents even while the pipeline is idle. Dependency-blocked tasks are never rescued by `_promote_failed_queue`.

## Testing quirks

`tests/conftest.py` has an autouse fixture that sandboxes every test: the user config dir (`config._USER_CONFIG_DIR` and friends) is monkeypatched into a tmp path, the dotenv source is disabled (the repo's own `.env` must not leak), and all `DRCODE_*` variables are stripped — keep this in mind when adding config-layering tests. Common fixtures: `project_root` (fresh tmp dir, the agent's writable root), `manager` (TaskManager over it), `user_config_dir`. The `test_network_*` suite exercises model/network failures (outages, retries, recovery) exclusively through the stubs in `tests/network_mocks.py` — no real API access is needed anywhere.

Parallel runs: pytest-xdist is a dev dependency and enabled by default via `addopts = "-n auto"` in `[tool.pytest.ini_options]` — every `uv run pytest` (whole suite, focused `-k`/file picks alike) gets its own worker processes and per-test `tmp_path`, no known cross-test conflicts. Serial runs (debugging, `-s`/`--pdb`) go with `uv run pytest -n 0`; a bare `-p no:xdist` alone errors out (unrecognized `-n` from addopts) — combine it with `-o addopts=""` if you really need the plugin unloaded. This is a LOCAL option only (no CI runs tests). On a slow/loaded machine the async helper `wait_until` (`tests/network_mocks.py`) takes its default deadline from `DRCODE_TEST_WAIT_TIMEOUT` (float seconds, default 30, read via `os.environ` directly so the conftest sandbox doesn't strip it) — raise it there instead of editing timeouts per test.
