### DrCode vs. OpenCode: Feature Availability

| Feature (DrCode, out of the box) | What it does | OpenCode status |
|---|---|---|
| **Task queue with auto-advancement** | Keeps a queue of tasks and automatically starts the next one when the active task finishes | **Via plugin** |
| **Gated pipeline with testing stage** | Enforces `QUEUE → ACTIVE → TESTING → COMPLETED`; new code does not start while testing is in progress | **Via plugin** |
| **Parallel task isolation via git worktrees** | Each task runs in its own worktree/branch (`drcode/…`), configurable parallel slots | **Via plugin** |
| **Per-agent model assignment, including a dedicated compaction model** | Lets you assign a cheaper/faster model specifically for context compaction (summarization), while keeping a stronger model for development — directly saving tokens on the expensive primary model | **Via plugin** — dedicated compaction model requires a plugin (e.g. `@madsoftwaredev/opencode-compaction-model`); native `agents.compaction.model` is ignored in current v2 builds |
| **Voice input and output** | Push-to-talk dictation and spoken responses work across platforms without extra setup | **Via plugin** |
| **Stall watchdog** | Detects stalled agents (no streaming / no tool events) and automatically aborts/restarts them | **Via plugin** |
| **Automatic transport retries** | Retries on network errors, timeouts, and HTTP 429 with exponential backoff | **Via plugin** |
| **Restart-safe interrupted work** | Partial transcript survives Esc/errors; next activation continues from the same point. Stuck `ACTIVE` tasks return to queue on restart | **Via plugin** |
| **Failed-queue rescue** | If every task in the queue is excluded, DrCode resets the exclusions and retries instead of staying idle | **No** |
| **GitHub Issues → local tasks sync** | Converts `bug`/`enhancement` issues into local tasks; adds `Fixes #N` to commits; configurable sync interval | **Via plugin** |
| **ntfy.sh push notifications** | Sends pipeline events to phone/desktop via ntfy.sh | **Via plugin** |
| **Python lifecycle workflow hooks** | `.drcode/workflow.py` defines hooks such as `on_accepted` for auto-merge or chained tasks | **Via plugin** |
| **Custom system-prompt override** | Any built-in prompt can be replaced by dropping a file into `.drcode/prompts/` | **Via plugin** |

- “Via plugin” means it is not available out of the box. In OpenCode these capabilities depend on installing and configuring third-party plugins; DrCode exposes them without additional plugin setup.
- **What DrCode does natively.** You can configure a primary (heavyweight) model for coding and separately point the compaction step to a smaller, cheaper model. This keeps the expensive model focused on development while routine summarization runs on a budget model.
- **What OpenCode does.** OpenCode **does** support per-agent model assignment in general (you can set a different `model` for `build`, `plan`, etc.). However, for the **compaction agent specifically**, the situation is different:
  - In current v2 beta builds, the `agents.compaction.model` configuration key is accepted but **silently ignored** — compaction always uses the session’s current model.
  - This is a known regression/feature gap in OpenCode, and users who want a dedicated compaction model must rely on community plugins such as `@madsoftwaredev/opencode-compaction-model`.
- **Why this matters.** Without per-compaction model control, every summarization call is billed at the primary model’s rate, which can noticeably inflate token costs in long development sessions. DrCode exposes this optimization as a first-class setting.