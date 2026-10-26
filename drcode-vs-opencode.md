### DrCode vs. OpenCode — Feature Availability

| Feature (DrCode, out of the box) | What it does | OpenCode status |
|---|---|---|
| **Tester agent verification with automatic bug filing** | After every completed task, a separate QA agent detects and runs linters/tests/build commands. On failure, it **creates a bug report task at the head of the queue**; only the tester can mark a task as passed after verification | **Via plugin** — `code-ensemble` adds remediation tasks via the `plan` action `add`; `OpencodePlugins` implements a **Coder↔Tester fix loop** (tester fails → coder fixes → re-verify) |
| **Research agent creates tasks for other agents** | The research agent can **file new tasks on the user’s behalf** (`create_bug_report` / `create_feature_request`), which are placed by ordinary queue rules. It can also search task history and commit documentation edits | **Via plugin** — `opencode-agent-teams` enables parallel agent execution with shared task boards; `opencode-orchestrator` uses a Commander agent that delegates work to other agents; `agent-tdd` turns research findings into specced GitHub issues (RootIssue + SubIssues) |
| **Task dependencies** | A task waits for other tasks; a dependency is satisfied only when it is **completed AND merged** (dynamic git-state check, 30 s re-poll) | **Via plugin** — `opencode-agent-teams`, `opencode-ensemble` |
| **Merge conflict resolution** | On a failed automated merge: LLM-free triage (byte-identical, zdiff3, rerere), then a dedicated resolver agent with intent-preservation review; falls back to manual per-hunk screen (`ctrl+m`) | **Via plugin** — `opencode-mad` (dedicated merger agent), `git-reconcile` |
| **Task archive search** | `ctrl+f` keyword search across titles, prompts, paths, summaries and events in every task status; the same index powers the research agent’s `search_task_history` tool | **Via plugin** — `opencode-history-mcp`, `agent-historian`, `opencode-session-recall` |
| **Per-agent model assignment, incl. dedicated compaction model** | A model per role: coding / research / tester / compaction / resolver; the compaction summarizer can run on a cheaper model, saving tokens on the primary development model | **Via plugin / partially broken** — `agents.compaction.model` is silently ignored in current v2 beta builds; a dedicated compaction model requires a plugin |
| **Task queue with auto-advancement** | Queue tasks; the next starts automatically when the active one finishes | **Via plugin** |
| **Gated pipeline with testing stage** | `QUEUE → ACTIVE → TESTING → COMPLETED`; no new development while testing is pending | **Via plugin** |
| **Parallel development via git worktrees** | Each task gets its own worktree and `drcode/` branch; 1–4 parallel slots | **Via plugin** |
| **Voice input and output** | Push-to-talk dictation and spoken pipeline announcements, built in across platforms | **Via plugin** |
| **Stall watchdog** | Aborts and restarts a run with no stream/tool events after a configurable stall limit | **Via plugin** |
| **Automatic transport retries** | Retries connection errors / HTTP 429 with doubling backoff (5 s → 60 s, up to 5 times) | **Via plugin** |
| **Restart-safe interrupted work** | Partial transcript survives `Esc`/errors; the next activation resumes without re-sending the prompt | **Via plugin** |
| **Failed-queue rescue** | If every queued task failed, the exclusion is forcibly lifted and the backlog re-runs — the queue never dead-ends | **No** — no dedicated plugin identified for this specific recovery loop |
| **GitHub Issues → local tasks** | `bug`/`enhancement` issues become local tasks; `Fixes #N` appended to commits; configurable sync interval | **Via plugin** |
| **ntfy.sh push notifications** | Pipeline events pushed to ntfy.sh; undelivered messages persisted and retried | **Via plugin** |
| **Custom team workflow (Python hooks + git-workflow.md)** | `.drcode/workflow.py` lifecycle hooks (`on_review`, `on_accepted`, `on_tester_passed`) and `.drcode/git-workflow.md` replace built-in commit rules | **Via plugin** |
| **Custom system-prompt override** | Any built-in prompt can be replaced by dropping a file into `.drcode/prompts/` | **Via plugin** |
| **Keep computer awake** | Holds a system sleep assertion while the pipeline has work | **Via plugin** |
| **Auto-cleanup of completed tasks** | Deletes completed tasks older than the retention period (7/30/90 days or never) at startup | **Via plugin** |

### Key takeaways

- **Dependencies, merge-conflict resolution, and history search** are now first-class DrCode features, but all three are achievable in OpenCode through plugins — they are **not** OpenCode gaps in absolute terms, only in the “out of the box” sense.
- **Per-agent compaction model** remains the sharpest differentiator: OpenCode’s native `agents.compaction.model` is silently ignored in current v2 beta builds, so token-saving on compaction requires a workaround or plugin.
- **Failed-queue rescue** and the **built-in merge-conflict funnel** (LLM-free triage → resolver agent → manual screen) are the features with **no direct OpenCode equivalent** identified in this search.
- **Tester → bug filing** is a structural part of DrCode’s pipeline: the tester is the **only** agent that can complete a task after verification, and a failed run **automatically injects a bug report at the head of the queue**, so fixes are picked up immediately. In OpenCode, this requires a multi-agent plugin such as `code-ensemble` or `OpencodePlugins`, where remediation or fix loops are configured per workflow.
- **Research → task creation** in DrCode is exposed as dedicated tools (`create_bug_report` / `create_feature_request`) available to the research role, with queue placement handled by the standard rules. OpenCode plugins achieve similar outcomes through shared task boards (`opencode-agent-teams`), Commander delegation (`opencode-orchestrator`), or issue generation (`agent-tdd`), but none of these are native core features.
- **What DrCode does natively.** You can configure a primary (heavyweight) model for coding and separately point the compaction step to a smaller, cheaper model. This keeps the expensive model focused on development while routine summarization runs on a budget model.
- **What OpenCode does.** OpenCode **does** support per-agent model assignment in general (you can set a different `model` for `build`, `plan`, etc.). However, for the **compaction agent specifically**, the situation is different:
  - In current v2 beta builds, the `agents.compaction.model` configuration key is accepted but **silently ignored** — compaction always uses the session’s current model.
  - This is a known regression/feature gap in OpenCode, and users who want a dedicated compaction model must rely on community plugins such as `@madsoftwaredev/opencode-compaction-model`.
- **Why this matters.** Without per-compaction model control, every summarization call is billed at the primary model’s rate, which can noticeably inflate token costs in long development sessions. DrCode exposes this optimization as a first-class setting.