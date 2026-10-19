"""pydantic-ai agents wired to an OpenAI-compatible endpoint, with coding tools."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.agent import AgentRun
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RunUsage

from . import tools
from .config import Config
from .models import (
    Clarification,
    TaskEvent,
    TaskKind,
    TaskStatus,
    VerificationStatus,
    _now,
)
from .prompts import load_compaction_prompts
from .task_manager import TaskManager
from .tools import PathViolation
from .version import APP_VERSION

FEATURE_SYSTEM_PROMPT = """\
You are drcode, an expert software engineering agent working directly in the user's repository.
You can read, create and edit files, search the codebase and run shell commands.
Work autonomously: explore the code, implement the requested change, run tests or
builds when available, and report a concise summary of what you did.
You are implementing a FEATURE. Do not shy away from large changes: refactoring
the existing codebase, introducing new classes and modules, and restructuring
existing code are all normal and expected parts of building a feature — do them
when they make the result better, cleaner or more maintainable. Never commit to
git unless explicitly asked.
If the request is ambiguous or missing a detail you cannot resolve from the code,
call ask_clarifying_question instead of guessing. After calling it the task is
paused awaiting the user's answer: end your turn immediately and do not call any
more tools.\
"""

BUG_SYSTEM_PROMPT = """\
You are drcode, an expert software engineering agent working directly in the user's repository.
You can read, create and edit files, search the codebase and run shell commands.
Work autonomously: explore the code, implement the requested change, run tests or
builds when available, and report a concise summary of what you did.
You are fixing a BUG. Your goal is to fix the bug with MINIMAL changes — touch
only what is necessary to make the bug go away. Large changes, refactoring,
creating new classes or modules and other restructuring must happen ONLY when
the bug cannot be fixed otherwise; if a minimal fix is possible, always choose it.
Never commit to git unless explicitly asked.
If the request is ambiguous or missing a detail you cannot resolve from the code,
call ask_clarifying_question instead of guessing. After calling it the task is
paused awaiting the user's answer: end your turn immediately and do not call any
more tools.\
"""

RESEARCH_SYSTEM_PROMPT = """\
You are drcode, a research assistant working directly in the user's repository.
Your task is to answer the user's questions by studying the source code and the
project's documentation, or to update the documentation in accordance with the
user's assignment. All tools are available to you: read and search the codebase,
create and edit files, and run shell commands. Explore the code thoroughly before
answering or editing so your answer and changes reflect the actual implementation.
Prefer minimal, focused changes. Never commit to git unless explicitly asked.

HARD LIMIT: you NEVER change source code. write_file / edit_file / run_bash may
only touch documentation (*.md files, AGENTS.md, docs/); using them on source
code is forbidden. If the user asks you to fix a problem or implement an
improvement — including picking one of the solution variants you proposed — do
NOT implement it yourself: hand it over to a coding agent via create_bug_report
or create_feature_request, putting the chosen solution and the relevant
file/line references into the task description, then end your turn.

You can also create new tasks in the backlog with create_bug_report (a bug report)
and create_feature_request (a feature request). STRICT RULE: create a task only
when the user explicitly asks for it ("file a bug", "create a task for this", …)
or when they pick one of the solution variants you proposed. If you notice a
potential problem or an improvement opportunity on your own, do NOT
create anything silently — ask the user first via ask_clarifying_question whether
a task should be filed, describe what you found, and act only on their answer.

If the request is ambiguous or missing a detail you cannot resolve from the code,
call ask_clarifying_question instead of guessing. After calling it the task is
paused awaiting the user's answer: end your turn immediately and do not call any
more tools.\
"""


@dataclass
class AgentDeps:
    """Dependencies for the coding agent: project root, task manager, task id."""

    root: Path
    manager: TaskManager
    task_id: str


# Гейтвей api-copilot.x5.ru блокирует запросы с UA известных CLI/SDK
# (curl, OpenAI/Python, pydantic-ai, ...) — политика против автоматизации.
# Нейтральный UA проходит; ставим его на уровне клиента и в model_settings
# (pydantic-ai перебивает client-заголовки per-request заголовком pydantic-ai/x.y).
USER_AGENT = f"drcode/{APP_VERSION}"
EXTRA_HEADERS = {"User-Agent": USER_AGENT}


async def probe_model(
    model: str, base_url: str, api_key: str, *, timeout: float = 5.0
) -> tuple[bool, str]:
    """Check that an OpenAI-compatible endpoint accepts the given settings.

    Probes `POST {base_url}/chat/completions` with empty messages: the
    backend answers 400 — which means TLS, key, user-agent and the model
    id were all accepted — and that counts as a working connection.
    Returns (ok, human-readable detail); `detail` is empty on success.
    """
    # Системное хранилище сертификатов: корпоративные CA отсутствуют в
    # certifi, из-за чего httpx даёт CERTIFICATE_VERIFY_FAILED, который
    # оборачивается в ConnectError.
    try:
        import truststore

        verify: ssl.SSLContext | bool = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        verify = True
    headers: dict[str, str] = {"User-Agent": USER_AGENT}
    if api_key:
        # An empty key must be omitted entirely: "Bearer " (trailing
        # space) is an illegal header value for h11/httpx.
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=verify) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json={"model": model, "messages": []},
            )
    except httpx.HTTPError as e:
        return False, f"unavailable ({type(e).__name__})"
    if resp.status_code in (401, 403):
        return False, f"auth failed (HTTP {resp.status_code})"
    if resp.status_code >= 500:
        return False, f"unavailable (HTTP {resp.status_code})"
    return True, ""


# Errors tools raise for situations the model can fix (bad path, missing
# old_string, broken regex). pydantic-ai 2.x converts only ModelRetry and
# arg-validation failures into retry prompts; any other exception crashes
# the whole run, so these must be re-raised as ModelRetry.
TOOL_RETRY_ERRORS = (ValueError, OSError, re.error)


async def _run_tool(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return await asyncio.to_thread(fn, *args)
    except TOOL_RETRY_ERRORS as e:
        raise ModelRetry(str(e) or type(e).__name__) from e


def _log_service_event(deps: AgentDeps | TesterDeps, kind: str, text: str) -> None:
    """Persist a service event into the task's event log.

    Like all TaskEvents, it is shown in the chat under the "service events"
    separator but never reaches the model context.
    """
    task = deps.manager.get(deps.task_id)
    task.events.append(TaskEvent(kind=kind, text=text))
    task.touch()
    deps.manager.storage.save(task)


def _log_violation(deps: AgentDeps | TesterDeps, tool: str, error: str) -> None:
    """Persist a permission violation into the task's service-event log."""
    _log_service_event(deps, "permission", f"{tool}: {error}")


async def _run_tool_logged(
    deps: AgentDeps | TesterDeps, tool: str, fn: Callable[..., Any], *args: Any
) -> Any:
    """Run a tool and turn a PathViolation into a retry prompt for the model,
    logging the refusal into the task history first."""
    try:
        return await _run_tool(fn, *args)
    except PathViolation as e:
        _log_violation(deps, tool, str(e))
        raise ModelRetry(str(e)) from e


async def _run_bash_logged(
    deps: AgentDeps | TesterDeps, root: Path, command: str, timeout: float | None = None
) -> str:
    try:
        return await tools.run_bash(root, command, timeout)
    except PathViolation as e:
        _log_violation(deps, "run_bash", str(e))
        raise ModelRetry(str(e)) from e


def build_model(config: Config):
    client = AsyncOpenAI(
        base_url=config.base_url,
        # Keyless gateways (local servers) still need a syntactically
        # valid Authorization header: "Bearer " is rejected by httpx.
        api_key=config.api_key or "no-key",
        default_headers=EXTRA_HEADERS,
    )
    return OpenAIChatModel(config.model, provider=OpenAIProvider(openai_client=client))


def load_agents_md(root: Path) -> str:
    """Return the project's AGENTS.md content, or an empty string if absent."""
    try:
        return (root / "AGENTS.md").read_text(encoding="utf-8")
    except OSError:
        return ""


def build_agent(
    config: Config, kind: TaskKind = TaskKind.FEATURE, root: Path | None = None
) -> Agent[AgentDeps, str]:
    """Build the coding agent. Every task kind gets a dedicated system prompt:
    bug tasks instruct minimal changes, feature tasks encourage refactoring and
    new classes, research tasks answer questions / update docs. The research
    agent additionally gets the task-creation tools (bug reports and feature
    requests). When the project root contains AGENTS.md, its content is
    appended to the instructions."""
    model = build_model(config)
    base_prompt = {
        TaskKind.BUG: BUG_SYSTEM_PROMPT,
        TaskKind.FEATURE: FEATURE_SYSTEM_PROMPT,
        TaskKind.RESEARCH: RESEARCH_SYSTEM_PROMPT,
    }[kind]
    instructions = base_prompt
    if root is not None:
        agents_md = load_agents_md(root)
        if agents_md.strip():
            instructions = (
                f"{base_prompt}\n\n# Project instructions (AGENTS.md)\n\n{agents_md}"
            )
    agent = Agent(
        model,
        instructions=instructions,
        deps_type=AgentDeps,
        model_settings={"extra_headers": EXTRA_HEADERS},
        retries=10,
    )

    # All blocking tools run in a worker thread (or as asyncio subprocesses
    # for run_bash): called inline they would stall the event loop, which
    # also drives the Textual UI — long commands froze keyboard input.
    @agent.tool
    async def read_file(
        ctx: RunContext[AgentDeps], path: str, offset: int = 1, limit: int = 2000
    ) -> str:
        """Read a text file from anywhere on the filesystem (relative paths resolve
        against the project root). Lines are numbered, starting from `offset`
        (1-based), up to `limit` lines."""
        return await _run_tool_logged(
            ctx.deps, "read_file", tools.read_file, ctx.deps.root, path, offset, limit
        )

    @agent.tool
    async def write_file(ctx: RunContext[AgentDeps], path: str, content: str) -> str:
        """Create or overwrite a file. Only the project root (with all subdirectories)
        and /tmp (with all subdirectories) are writable; use /tmp to test hypotheses."""
        return await _run_tool_logged(
            ctx.deps, "write_file", tools.write_file, ctx.deps.root, path, content
        )

    @agent.tool
    async def edit_file(
        ctx: RunContext[AgentDeps], path: str, old_string: str, new_string: str
    ) -> str:
        """Replace an exact unique substring in a file. old_string must match exactly once.
        Only the project root (with all subdirectories) and /tmp (with all subdirectories)
        are writable; use /tmp to test hypotheses."""
        return await _run_tool_logged(
            ctx.deps,
            "edit_file",
            tools.edit_file,
            ctx.deps.root,
            path,
            old_string,
            new_string,
        )

    @agent.tool
    async def glob_files(ctx: RunContext[AgentDeps], pattern: str) -> list[str]:
        """Find files by glob pattern relative to project root, e.g. 'src/**/*.py'."""
        return await _run_tool(tools.glob_files, ctx.deps.root, pattern)

    @agent.tool
    async def grep_search(
        ctx: RunContext[AgentDeps], pattern: str, include: str = ""
    ) -> list[str]:
        """Regex search in file contents. `include` is an optional glob mask
        filtering files by their project-relative path, e.g. '*.py' or 'src/**/*.py'."""
        return await _run_tool(tools.grep_search, ctx.deps.root, pattern, include)

    @agent.tool
    async def run_bash(
        ctx: RunContext[AgentDeps], command: str, timeout: int | None = None
    ) -> str:
        """Run a shell command in the project root and return its output.
        The command is killed after `timeout` seconds (default 120); pass a
        larger timeout for long-running commands (builds, test suites) —
        values above 3600 (60 minutes) are clamped to 3600.
        rm/mv/cp are only allowed with paths inside the project root or /tmp."""
        return await _run_bash_logged(ctx.deps, ctx.deps.root, command, timeout)

    # Task-creation tools exist only for the research agent: coding
    # agents (feature/bug) work on their own task and must not spawn more;
    # the research agent can file bug reports and feature requests on
    # the user's behalf. Placement follows the ordinary queue rules via
    # TaskManager.create (bugs join the head of the bug block, everything
    # else the tail of the queue).
    if kind is TaskKind.RESEARCH:

        @agent.tool
        def create_bug_report(
            ctx: RunContext[AgentDeps], title: str, description: str
        ) -> str:
            """Create a bug report task in the backlog. Use ONLY when the user
            explicitly asked for it — if you found a problem on your own, ask
            via ask_clarifying_question whether a task should be filed.
            `title` is a short imperative summary; `description` carries the
            details: what is wrong, where (files/lines), and how it manifests."""
            bug = ctx.deps.manager.create(title, description, kind=TaskKind.BUG)
            return (
                f"Bug report #{bug.id} '{bug.title}' created at the head of the queue."
            )

        @agent.tool
        def create_feature_request(
            ctx: RunContext[AgentDeps], title: str, description: str
        ) -> str:
            """Create a feature request task in the backlog. Use ONLY when the
            user explicitly asked for it — if you spotted an improvement on
            your own, ask via ask_clarifying_question whether a task should be
            filed. `title` is a short summary of the desired change;
            `description` details what should be done and why."""
            feature = ctx.deps.manager.create(title, description, kind=TaskKind.FEATURE)
            return (
                f"Feature request #{feature.id} '{feature.title}' created at "
                "the tail of the queue."
            )

    @agent.tool
    def ask_clarifying_question(
        ctx: RunContext[AgentDeps], question: str, options: list[str] | None = None
    ) -> str:
        """Ask the user a clarifying question and pause the task until they answer.
        Use when the request is ambiguous or a required detail is missing.
        `options` is an optional list of suggested answers; pass it when a small
        set of choices covers the reasonable answers, otherwise omit it."""
        task = ctx.deps.manager.get(ctx.deps.task_id)
        task.clarification = Clarification(
            question=question.strip(),
            options=[o.strip() for o in (options or []) if o.strip()],
        )
        task.status = TaskStatus.AWAITING_CLARIFICATION
        task.touch()
        ctx.deps.manager.storage.save(task)
        return (
            "Question sent to the user. The task is paused awaiting clarification; "
            "end your turn now and do not call any more tools."
        )

    return agent


TESTER_SYSTEM_PROMPT = """\
You are drcode's QA tester agent. A coding task has just been completed and your \
job is to verify that it did not break the project.

Steps:
1. Inspect the project to detect available tooling: linters (ruff, eslint, mypy, …), \
tests (pytest, npm test, cargo test, …) and a build/compile command.
2. Run each detected check via run_bash. If there are no tests or linters configured, \
say so and still try to build/compile the project.
3. If everything passes: before labelling the task, check whether the project root is \
a git repository (git_commit does this itself). If it is, write a short commit message \
that concisely paraphrases the original task (imperative mood, e.g. "Add retry logic \
to the tester pipeline") and call git_commit with it; it also pushes the branch to \
the remote. If the commit fails (e.g. git identity is not configured), still pass \
verification and mention the failure in your summary. A failed push (no network, \
diverged remote, no upstream, …) is NOT a verification failure: the error is logged \
into the task's event log automatically — still pass verification and mention the \
push problem in your summary. If the project is not a git repository, skip the \
commit entirely.
4. Call set_verification(success=True) and give a short summary of what you ran \
(and whether a commit was made and pushed).
5. If anything fails, call set_verification(success=False), then call create_bug_report \
with a clear title and a description containing what failed, the exact command and the \
error output, and give a short summary.

Your filesystem tools are read-only and you must not modify any files; run_bash is for \
running checks only. The single exception is git_commit, used only after all checks \
passed: it stages the changes and commits them.\
"""

# Same tester, but the "Task finalization" project setting is "Testing": the
# verified changes are left uncommitted — there is no git_commit tool at all,
# and the prompt must not mention one.
TESTER_SYSTEM_PROMPT_NO_COMMIT = """\
You are drcode's QA tester agent. A coding task has just been completed and your \
job is to verify that it did not break the project.

Steps:
1. Inspect the project to detect available tooling: linters (ruff, eslint, mypy, …), \
tests (pytest, npm test, cargo test, …) and a build/compile command.
2. Run each detected check via run_bash. If there are no tests or linters configured, \
say so and still try to build/compile the project.
3. Call set_verification(success=True) and give a short summary of what you ran. \
Do NOT create any git commits — committing verified changes is not part of your job.
4. If anything fails, call set_verification(success=False), then call create_bug_report \
with a clear title and a description containing what failed, the exact command and the \
error output, and give a short summary.

Your filesystem tools are read-only and you must not modify any files; run_bash is for \
running checks only. Never stage, commit or otherwise modify the repository state.\
"""


@dataclass
class TesterDeps:
    """Dependencies for the tester agent: project root, task manager, verified task."""

    root: Path
    manager: TaskManager
    task_id: str


async def _git_push(deps: TesterDeps, root: Path) -> str:
    """Best-effort `git push` after a successful commit.

    A push failure never fails verification: the error is written into the
    task's service-event log and reported back to the model as a note, so the
    tester still passes the task."""
    remotes = await tools.run_bash(root, "git remote")
    names = remotes.removeprefix("exit=0").strip()
    if not remotes.startswith("exit=0") or not names or names == "(no output)":
        _log_service_event(deps, "git", "git push skipped: no remote configured.")
        return "\ngit push skipped: no remote configured."
    push = await tools.run_bash(root, "git push")
    if push.startswith("exit=0"):
        return "\nPushed successfully."
    if "no upstream" in push:
        # Fresh branch without tracking: retry once, setting the upstream.
        push = await tools.run_bash(root, "git push --set-upstream origin HEAD")
        if push.startswith("exit=0"):
            return "\nPushed successfully (set upstream to origin/HEAD)."
    detail = push.removeprefix("exit=0").strip() or "(no output)"
    _log_service_event(deps, "git", f"git push failed:\n{detail}")
    return (
        "\nPush failed (the commit is safe locally):\n"
        f"{detail}\n"
        "This is not a verification failure — proceed with "
        "set_verification(success=True) and mention the push problem "
        "in your summary."
    )


def build_tester_agent(
    config: Config, *, allow_commit: bool = True
) -> Agent[TesterDeps, str]:
    """Build the QA tester agent.

    `allow_commit=False` (project setting "Task finalization: Testing")
    selects the commit-free system prompt and does not register the
    git_commit tool, so verified changes are left uncommitted.
    """
    model = build_model(config)
    agent = Agent(
        model,
        instructions=(
            TESTER_SYSTEM_PROMPT if allow_commit else TESTER_SYSTEM_PROMPT_NO_COMMIT
        ),
        deps_type=TesterDeps,
        model_settings={"extra_headers": EXTRA_HEADERS},
        retries=10,
    )

    @agent.tool
    async def read_file(
        ctx: RunContext[TesterDeps], path: str, offset: int = 1, limit: int = 2000
    ) -> str:
        """Read a text file from anywhere on the filesystem (relative paths resolve
        against the project root). Lines are numbered, starting from `offset`
        (1-based), up to `limit` lines."""
        return await _run_tool_logged(
            ctx.deps, "read_file", tools.read_file, ctx.deps.root, path, offset, limit
        )

    @agent.tool
    async def glob_files(ctx: RunContext[TesterDeps], pattern: str) -> list[str]:
        """Find files by glob pattern relative to project root, e.g. 'src/**/*.py'."""
        return await _run_tool(tools.glob_files, ctx.deps.root, pattern)

    @agent.tool
    async def grep_search(
        ctx: RunContext[TesterDeps], pattern: str, include: str = ""
    ) -> list[str]:
        """Regex search in file contents. `include` is an optional glob mask
        filtering files by their project-relative path, e.g. '*.py' or 'src/**/*.py'."""
        return await _run_tool(tools.grep_search, ctx.deps.root, pattern, include)

    @agent.tool
    async def run_bash(
        ctx: RunContext[TesterDeps], command: str, timeout: int | None = None
    ) -> str:
        """Run a shell command in the project root and return its output.
        The command is killed after `timeout` seconds (default 120); pass a
        larger timeout for long-running checks (full test suites, builds) —
        values above 3600 (60 minutes) are clamped to 3600.
        rm/mv/cp are only allowed with paths inside the project root or /tmp."""
        return await _run_bash_logged(ctx.deps, ctx.deps.root, command, timeout)

    @agent.tool
    def create_bug_report(
        ctx: RunContext[TesterDeps], title: str, description: str
    ) -> str:
        """Create a bug report task. It is placed at the head of the execution queue."""
        bug = ctx.deps.manager.create(title, description, kind=TaskKind.BUG)
        return f"Bug report #{bug.id} '{bug.title}' created at the head of the queue."

    # The commit tool exists only when the project setting allows it: with
    # "Task finalization: Testing" the model cannot commit even if it tries.
    if allow_commit:

        @agent.tool
        async def git_commit(ctx: RunContext[TesterDeps], message: str) -> str:
            """Stage all changed files in the project root and create a git commit
            with the given message, then push the branch to its remote. Use only
            after all checks passed, with a short message paraphrasing the
            original task. Skips gracefully when the project root is not a git
            repository or there is nothing to commit. A failed push (no network,
            diverged remote, no upstream, …) is NOT a failure: the push error is
            recorded in the task's event log and the commit stays safe locally —
            still proceed with set_verification(success=True)."""
            root = ctx.deps.root
            message = message.strip().strip("'\"")
            if not message:
                return "Empty commit message — nothing committed."
            inside = await tools.run_bash(root, "git rev-parse --is-inside-work-tree")
            if not inside.startswith("exit=0") or "true" not in inside:
                return "Not a git repository — nothing committed."
            await tools.run_bash(root, "git add -A -- . ':(exclude).drcode'")
            staged = await tools.run_bash(root, "git diff --cached --name-only")
            if (
                staged.startswith("exit=0")
                and not staged.removeprefix("exit=0").strip()
            ):
                return "Nothing to commit — the working tree is clean."
            commit = await tools.run_bash(root, f"git commit -m {shlex.quote(message)}")
            if not commit.startswith("exit=0"):
                return f"Commit failed:\n{commit}"
            head = await tools.run_bash(root, "git log -1 --oneline")
            result = f"Committed successfully.\n{head}"
            return result + await _git_push(ctx.deps, root)

    @agent.tool
    def set_verification(ctx: RunContext[TesterDeps], success: bool) -> str:
        """Set the verification label on the verified task (passed or failed)
        and finish its testing: the task becomes completed."""
        task = ctx.deps.manager.get(ctx.deps.task_id)
        task.verification = (
            VerificationStatus.PASSED if success else VerificationStatus.FAILED
        )
        task.status = TaskStatus.COMPLETED
        task.completed_at = _now()
        task.touch()
        ctx.deps.manager.storage.save(task)
        return f"Verification label set: {task.verification.value}; task completed."

    return agent


# Marker prefix of the synthetic user message that carries the compaction
# summary inside the transcript. Lets us (a) feed the previous summary back
# to the summarizer as <previous-summary> on the next compaction and
# (b) render it in the chat as a "context compacted" block instead of a
# regular user message.
COMPACTION_SUMMARY_PREFIX = "[context summary]"

# Tool results carry most of the working context (file reads, command
# output), but a single pathological result must not blow up the
# summarizer's own request — everything above the cap is cut.
_COMPACTION_TOOL_RESULT_CAP = 20_000


def build_compaction_agent(config: Config) -> Agent[None, str]:
    """Tool-less summarizer agent used to compact a task's history.

    System prompt and output structure come from the bundled compaction.txt
    / compaction.md templates; the conversation itself is passed as a
    plain-text user prompt (see build_compaction_prompt). No tools, no deps:
    the agent only reads the rendered transcript and returns the summary.
    """
    system_prompt, _ = load_compaction_prompts()
    return Agent(
        build_model(config),
        instructions=system_prompt,
        model_settings={"extra_headers": EXTRA_HEADERS},
    )


def _is_request(msg: dict[str, Any]) -> bool:
    return msg.get("kind") == "request" or msg.get("role") == "request"


def _user_prompt_content(msg: dict[str, Any]) -> str | None:
    """First string user-prompt content of a serialized request, if any."""
    for part in msg.get("parts", []):
        if not isinstance(part, dict):
            continue
        if part.get("part_kind") == "user-prompt":
            content = part.get("content")
            if isinstance(content, str) and content:
                return content
    return None


def is_summary_message(msg: dict[str, Any]) -> bool:
    """True for the synthetic user message carrying the compaction summary."""
    if not _is_request(msg):
        return False
    content = _user_prompt_content(msg)
    return content is not None and content.startswith(COMPACTION_SUMMARY_PREFIX)


def split_original_prompts(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the transcript into (original user prompts, rest).

    The leading consecutive user requests are the task's original
    assignment(s) — they are kept verbatim across compactions (the summary
    message, if a previous compaction left one, ends this prefix: it lives
    in `rest` so the summarizer sees it as <previous-summary>).
    """
    i = 0
    while i < len(messages):
        msg = messages[i]
        content = _user_prompt_content(msg) if _is_request(msg) else None
        if content is None or content.startswith(COMPACTION_SUMMARY_PREFIX):
            break
        i += 1
    return messages[:i], messages[i:]


def extract_previous_summary(messages: list[dict[str, Any]]) -> str | None:
    """Body of the latest compaction summary in the transcript (None if the
    history was never compacted)."""
    found: str | None = None
    for msg in messages:
        if not _is_request(msg):
            continue
        content = _user_prompt_content(msg)
        if content is not None and content.startswith(COMPACTION_SUMMARY_PREFIX):
            found = content[len(COMPACTION_SUMMARY_PREFIX) :].strip()
    return found


def render_history_for_summary(messages: list[dict[str, Any]]) -> str:
    """Render the serialized transcript as readable text for the summarizer.

    User prompts, assistant text, tool calls and tool results are kept
    (tool results are the bulk of the working context — file contents and
    command output the summary must capture); thinking parts are dropped;
    an existing compaction summary is skipped (it is passed separately as
    <previous-summary>).
    """
    lines: list[str] = []
    for msg in messages:
        if is_summary_message(msg):
            continue
        is_request = _is_request(msg)
        for part in msg.get("parts", []):
            if not isinstance(part, dict):
                continue
            kind = part.get("part_kind")
            if kind == "user-prompt":
                content = part.get("content")
                if isinstance(content, str) and content:
                    lines.append(f"[user]\n{content}")
            elif kind == "tool-return":
                # Tool returns live in request messages (ModelRequest with a
                # ToolReturnPart) — they must be rendered BEFORE the generic
                # "skip everything else in a request" branch below, or the
                # summarizer loses the bulk of the working context.
                content = part.get("content")
                if isinstance(content, str):
                    text = content
                else:
                    try:
                        text = json.dumps(content, ensure_ascii=False, indent=2)
                    except (TypeError, ValueError):
                        text = str(content)
                if len(text) > _COMPACTION_TOOL_RESULT_CAP:
                    text = text[:_COMPACTION_TOOL_RESULT_CAP] + "\n…[truncated]"
                lines.append(f"[tool result for {part.get('tool_name', '?')}]\n{text}")
            elif is_request:
                continue
            elif kind == "text":
                content = part.get("content", "")
                if content:
                    lines.append(f"[assistant]\n{content}")
            elif kind == "tool-call":
                args = part.get("args")
                args_str = (
                    args
                    if isinstance(args, str)
                    else json.dumps(args or {}, ensure_ascii=False)
                )
                lines.append(f"[tool call] {part.get('tool_name', '?')}({args_str})")
            elif kind == "tool-return":
                content = part.get("content")
                if isinstance(content, str):
                    text = content
                else:
                    try:
                        text = json.dumps(content, ensure_ascii=False, indent=2)
                    except (TypeError, ValueError):
                        text = str(content)
                if len(text) > _COMPACTION_TOOL_RESULT_CAP:
                    text = text[:_COMPACTION_TOOL_RESULT_CAP] + "\n…[truncated]"
                lines.append(f"[tool result for {part.get('tool_name', '?')}]\n{text}")
    return "\n\n".join(lines)


def build_compaction_prompt(
    previous_summary: str | None, transcript: str, template: str
) -> str:
    """User prompt for the summarizer: the conversation, the anchored
    previous summary (if any) and the required output structure."""
    parts = [
        "Below is the conversation history of a coding session.",
        "",
        "<conversation>",
        transcript,
        "</conversation>",
        "",
        "<previous-summary>",
        previous_summary or "(none)",
        "</previous-summary>",
        "",
        "Produce the updated summary following exactly this structure:",
        "",
        template,
    ]
    return "\n".join(parts)


def compacted_messages(
    messages: list[dict[str, Any]], summary: str
) -> list[dict[str, Any]]:
    """New transcript after compaction: the original user prompts verbatim
    plus one synthetic request carrying the summary."""
    preserved, _ = split_original_prompts(messages)
    request = ModelRequest(
        parts=[
            UserPromptPart(content=f"{COMPACTION_SUMMARY_PREFIX}\n\n{summary.strip()}")
        ]
    )
    return preserved + serialize_messages([request])


def serialize_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    return ModelMessagesTypeAdapter.dump_python(messages, mode="json")  # type: ignore[return-value]


def sanitized_messages(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transcript truncated to its API-valid prefix (resumable_history rule).

    Applied when a run is interrupted: the invalid tail (a response with
    tool calls whose results never arrived, the empty interrupted request
    pydantic-ai appends) is dropped so the stored transcript stays valid no
    matter in which order later user messages get appended to it.
    """
    return serialize_messages(resumable_history(data))


def ends_with_open_request(data: list[dict[str, Any]]) -> bool:
    """True when the transcript ends with an unanswered user request —
    i.e. there is an interrupted attempt that a continuation turn can pick
    up (the transcript is expected to be sanitized at that point)."""
    if not data:
        return False
    last = data[-1]
    return last.get("kind") == "request" or last.get("role") == "request"


def _usage_context_tokens(input_tokens: int, cache_read: int, cache_write: int) -> int:
    """Tokens the model actually saw in that request: uncached prompt +
    cache reads + cache writes (Anthropic-style accounting splits the prompt
    across the three fields)."""
    return input_tokens + cache_read + cache_write


def last_context_tokens(data: list[dict[str, Any]]) -> int | None:
    """Context fill (tokens) at the end of a serialized transcript.

    Each ModelResponse carries the usage of the request that produced it,
    and its input side is the full context the model saw (history + tools +
    prompt). The last response with non-zero usage therefore gives the
    final context fill of the task — for finished tasks as well, since the
    transcript with usage is persisted in the task history.
    """
    for msg in reversed(data):
        if msg.get("kind") != "response" and msg.get("role") != "response":
            continue
        usage = msg.get("usage") or {}
        tokens = _usage_context_tokens(
            usage.get("input_tokens") or 0,
            usage.get("cache_read_tokens") or 0,
            usage.get("cache_write_tokens") or 0,
        )
        if tokens:
            return tokens
    return None


def run_context_tokens(run: AgentRun[Any, Any]) -> int | None:
    """Context fill (tokens) after the latest completed model request of a
    live run (usage of the last ModelResponse in the run's transcript)."""
    try:
        messages = run.all_messages()
    except Exception:  # noqa: BLE001 - state may be mid-teardown after cancel
        return None
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            usage = msg.usage
            tokens = _usage_context_tokens(
                usage.input_tokens, usage.cache_read_tokens, usage.cache_write_tokens
            )
            if tokens:
                return tokens
    return None


def last_response_usage(run: AgentRun[Any, Any]) -> RunUsage | None:
    """Exact token usage of the latest completed model request of a live run.

    Each ModelResponse carries the usage of the request that produced it;
    the last response with non-zero usage therefore gives the per-request
    input/cache/output figures for the accumulator in the TUI (distinct
    from run_context_tokens, which only sums the input side). None when
    the run has no completed model request yet.
    """
    try:
        messages = run.all_messages()
    except Exception:  # noqa: BLE001 - state may be mid-teardown after cancel
        return None
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            u = msg.usage
            if (
                u.input_tokens
                or u.output_tokens
                or u.cache_read_tokens
                or u.cache_write_tokens
            ):
                return RunUsage(
                    input_tokens=u.input_tokens,
                    cache_read_tokens=u.cache_read_tokens,
                    cache_write_tokens=u.cache_write_tokens,
                    output_tokens=u.output_tokens,
                )
    return None


def load_messages(data: list[dict[str, Any]]) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_python(data)


def resumable_history(data: list[dict[str, Any]]) -> list[ModelMessage]:
    """Load persisted messages for resuming an interrupted/previous run.

    The only tail a model API cannot accept is a half-done exchange: a
    model response with tool calls whose results never arrived, or a
    response the stream cut off before any part arrived (an empty
    assistant message). The transcript is truncated right before such a
    response. Everything else — user prompts, tool returns, retry prompts,
    completed text answers — is a valid point to continue from and is
    kept, so a retried run resumes the attempt where it stopped instead of
    redoing it from scratch.
    """
    messages = list(load_messages(data))
    cut = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest) or (
            isinstance(msg, ModelResponse)
            and any(isinstance(p, TextPart) and p.content for p in msg.parts)
            and not any(isinstance(p, ToolCallPart) for p in msg.parts)
        ):
            cut = i + 1
    return messages[:cut]


def is_pending_retry(messages: list[dict[str, Any]], text: str) -> bool:
    """True when `text` is already the prompt of an unfinished attempt in the
    serialized transcript (possible only for legacy tasks whose interrupted
    prompt was re-queued into `pending_input` by older builds). The chat
    renders the prompt and the partial work from the transcript itself;
    showing the pending copy too would draw the same message twice.
    """
    idx = None
    for i, msg in enumerate(messages):
        if msg.get("kind") != "request" and msg.get("role") != "request":
            continue
        for part in msg.get("parts", []):
            if part.get("part_kind") == "user-prompt" and part.get("content") == text:
                idx = i
    if idx is None:
        return False
    for msg in reversed(messages[idx + 1 :]):
        if msg.get("kind") == "response" or msg.get("role") == "response":
            parts = msg.get("parts", [])
            has_text = any(
                p.get("part_kind") == "text" and p.get("content") for p in parts
            )
            has_tool_call = any(p.get("part_kind") == "tool-call" for p in parts)
            return not (has_text and not has_tool_call)
    return True


def backdate_user_prompt(messages: list[ModelMessage], text: str, at: datetime) -> None:
    """Stamp the user request carrying `text` with `at` — the moment the
    user typed the prompt — in place.

    pydantic-ai timestamps the ModelRequest it builds from a consumed
    pending prompt with the consumption time, which can be minutes after
    the message was actually sent (the prompt may sit in pending_input
    while other work runs). The chat thread is ordered by these stamps, so
    the request is backdated to keep the transcript chronological.
    """
    for msg in reversed(messages):
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if isinstance(part, UserPromptPart) and part.content == text:
                part.timestamp = at
                msg.timestamp = at
                return


def first_user_prompt(messages: list[dict[str, Any]]) -> str:
    """First user prompt of a serialized transcript — the original task text.

    Used to hand the tester the task's initial assignment (the title alone is
    often too terse to paraphrase into a commit message).
    """
    for msg in messages:
        if msg.get("kind") != "request" and msg.get("role") != "request":
            continue
        for part in msg.get("parts", []):
            if part.get("part_kind") == "user-prompt":
                content = part.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return ""


def _message_time(msg: dict[str, Any]) -> datetime | None:
    """Timestamp of a serialized ModelMessage (None when not recorded).

    pydantic-ai 2.x puts a top-level `timestamp` on both ModelRequest and
    ModelResponse; older transcripts only carried per-part timestamps on
    requests (UserPromptPart / ToolReturnPart), so those are used as a
    fallback (the latest part stamp)."""
    raw = msg.get("timestamp")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            raw = None
    if isinstance(raw, datetime):
        return raw
    stamps: list[datetime] = []
    for part in msg.get("parts", []):
        if not isinstance(part, dict):
            continue
        part_raw = part.get("timestamp")
        if isinstance(part_raw, str):
            try:
                stamps.append(datetime.fromisoformat(part_raw))
            except ValueError:
                pass
        elif isinstance(part_raw, datetime):
            stamps.append(part_raw)
    return max(stamps) if stamps else None


def extract_display(
    messages: list[dict[str, Any]],
) -> list[tuple[datetime | None, str, str]]:
    """Turn serialized ModelMessages into (timestamp, role, text) triples
    for the chat UI.

    Roles: "user", "reasoning" (model thinking), "assistant" (text),
    "tool-call" (name(args) line; the TUI shows the name as the collapsed
    header and the args in the expandable body), rendered by the TUI in
    message order. The timestamp (message creation time, None when the
    transcript format didn't record one) lets the TUI interleave these
    entries with other timestamped streams (service events, tester rounds)
    into a single chronological timeline.
    """
    out: list[tuple[datetime | None, str, str]] = []
    reasoning: list[str] = []
    chunks: list[str] = []
    for msg in messages:
        # pydantic-ai 2.x serializes ModelRequest with "kind": "request"
        # (older versions used "role"); support both.
        at = _message_time(msg)
        if msg.get("kind") == "request" or msg.get("role") == "request":
            for part in msg.get("parts", []):
                if part.get("part_kind") == "user-prompt":
                    content = part.get("content")
                    if isinstance(content, str) and content:
                        if content.startswith(COMPACTION_SUMMARY_PREFIX):
                            # Synthetic compaction summary, not a user message:
                            # rendered as its own "context compacted" block.
                            out.append(
                                (
                                    at,
                                    "summary",
                                    content[len(COMPACTION_SUMMARY_PREFIX) :].strip(),
                                )
                            )
                        else:
                            out.append((at, "user", content))
            continue
        for part in msg.get("parts", []):
            kind = part.get("part_kind")
            if kind == "thinking":
                content = part.get("content", "")
                if content:
                    if chunks:
                        out.append((at, "assistant", "\n".join(chunks)))
                        chunks = []
                    reasoning.append(content)
            elif kind == "text":
                content = part.get("content", "")
                if content:
                    if reasoning:
                        out.append((at, "reasoning", "\n\n".join(reasoning)))
                        reasoning = []
                    chunks.append(content)
            elif kind == "tool-call":
                args = part.get("args")
                args_str = (
                    args
                    if isinstance(args, str)
                    else json.dumps(args or {}, ensure_ascii=False)
                )
                if reasoning:
                    out.append((at, "reasoning", "\n\n".join(reasoning)))
                    reasoning = []
                if chunks:
                    out.append((at, "assistant", "\n".join(chunks)))
                    chunks = []
                out.append(
                    (at, "tool-call", f"{part.get('tool_name', '?')}({args_str})")
                )
        if reasoning:
            out.append((at, "reasoning", "\n\n".join(reasoning)))
            reasoning = []
        if chunks:
            out.append((at, "assistant", "\n".join(chunks)))
            chunks = []
    return out


__all__ = [
    "BUG_SYSTEM_PROMPT",
    "COMPACTION_SUMMARY_PREFIX",
    "FEATURE_SYSTEM_PROMPT",
    "RESEARCH_SYSTEM_PROMPT",
    "AgentDeps",
    "ModelRequest",
    "ModelResponse",
    "TesterDeps",
    "TextPart",
    "ToolCallPart",
    "ToolReturnPart",
    "UserPromptPart",
    "build_agent",
    "build_compaction_agent",
    "build_compaction_prompt",
    "build_tester_agent",
    "compacted_messages",
    "ends_with_open_request",
    "extract_display",
    "extract_previous_summary",
    "first_user_prompt",
    "is_pending_retry",
    "is_summary_message",
    "last_context_tokens",
    "last_response_usage",
    "load_agents_md",
    "load_messages",
    "probe_model",
    "render_history_for_summary",
    "resumable_history",
    "run_context_tokens",
    "sanitized_messages",
    "serialize_messages",
    "split_original_prompts",
]
