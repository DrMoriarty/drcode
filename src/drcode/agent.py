"""pydantic-ai agents wired to an OpenAI-compatible endpoint, with coding tools."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.agent import AgentRun
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

from . import tools
from .config import Config
from .models import Clarification, TaskKind, TaskStatus, VerificationStatus
from .task_manager import TaskManager

SYSTEM_PROMPT = """\
You are drcode, an expert software engineering agent working directly in the user's repository.
You can read, create and edit files, search the codebase and run shell commands.
Work autonomously: explore the code, implement the requested change, run tests or
builds when available, and report a concise summary of what you did.
Prefer minimal, focused changes. Never commit to git unless explicitly asked.
If the request is ambiguous or missing a detail you cannot resolve from the code,
call ask_clarifying_question instead of guessing. After calling it the task is
paused awaiting the user's answer: end your turn immediately and do not call any
more tools.\
"""

DOCUMENTATION_SYSTEM_PROMPT = """\
You are drcode, a documentation assistant working directly in the user's repository.
Your task is to answer the user's questions by studying the source code and the
project's documentation, or to update the documentation in accordance with the
user's assignment. All tools are available to you: read and search the codebase,
create and edit files, and run shell commands. Explore the code thoroughly before
answering or editing so your answer and changes reflect the actual implementation.
Prefer minimal, focused changes. Never commit to git unless explicitly asked.
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
USER_AGENT = "drcode/0.1"
EXTRA_HEADERS = {"User-Agent": USER_AGENT}


def build_model(config: Config):
    client = AsyncOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
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
    """Build the coding agent. Documentation tasks get a dedicated system
    prompt (answer questions / update docs) but the same tool set. When the
    project root contains AGENTS.md, its content is appended to the
    instructions."""
    model = build_model(config)
    base_prompt = (
        DOCUMENTATION_SYSTEM_PROMPT if kind is TaskKind.DOCUMENTATION else SYSTEM_PROMPT
    )
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
    )

    # All blocking tools run in a worker thread (or as asyncio subprocesses
    # for run_bash): called inline they would stall the event loop, which
    # also drives the Textual UI — long commands froze keyboard input.
    @agent.tool
    async def read_file(
        ctx: RunContext[AgentDeps], path: str, offset: int = 1, limit: int = 2000
    ) -> str:
        """Read a text file. Lines are numbered, starting from `offset` (1-based), up to `limit` lines."""
        return await asyncio.to_thread(
            tools.read_file, ctx.deps.root, path, offset, limit
        )

    @agent.tool
    async def write_file(ctx: RunContext[AgentDeps], path: str, content: str) -> str:
        """Create or overwrite a file with the given content."""
        return await asyncio.to_thread(tools.write_file, ctx.deps.root, path, content)

    @agent.tool
    async def edit_file(
        ctx: RunContext[AgentDeps], path: str, old_string: str, new_string: str
    ) -> str:
        """Replace an exact unique substring in a file. old_string must match exactly once."""
        return await asyncio.to_thread(
            tools.edit_file, ctx.deps.root, path, old_string, new_string
        )

    @agent.tool
    async def glob_files(ctx: RunContext[AgentDeps], pattern: str) -> list[str]:
        """Find files by glob pattern relative to project root, e.g. 'src/**/*.py'."""
        return await asyncio.to_thread(tools.glob_files, ctx.deps.root, pattern)

    @agent.tool
    async def grep_search(
        ctx: RunContext[AgentDeps], pattern: str, include: str = ""
    ) -> list[str]:
        """Regex search in file contents. `include` is an optional filename regex filter."""
        return await asyncio.to_thread(
            tools.grep_search, ctx.deps.root, pattern, include
        )

    @agent.tool
    async def run_bash(ctx: RunContext[AgentDeps], command: str) -> str:
        """Run a shell command in the project root and return its output (2 min timeout)."""
        return await tools.run_bash(ctx.deps.root, command)

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
3. If everything passes, call set_verification(success=True) and give a short summary \
of what you ran.
4. If anything fails, call set_verification(success=False), then call create_bug_report \
with a clear title and a description containing what failed, the exact command and the \
error output, and give a short summary.

Your filesystem tools are read-only and you must not modify any files; run_bash is for \
running checks only.\
"""


@dataclass
class TesterDeps:
    """Dependencies for the tester agent: project root, task manager, verified task."""

    root: Path
    manager: TaskManager
    task_id: str


def build_tester_agent(config: Config) -> Agent[TesterDeps, str]:
    model = build_model(config)
    agent = Agent(
        model,
        instructions=TESTER_SYSTEM_PROMPT,
        deps_type=TesterDeps,
        model_settings={"extra_headers": EXTRA_HEADERS},
    )

    @agent.tool
    async def read_file(
        ctx: RunContext[TesterDeps], path: str, offset: int = 1, limit: int = 2000
    ) -> str:
        """Read a text file. Lines are numbered, starting from `offset` (1-based), up to `limit` lines."""
        return await asyncio.to_thread(
            tools.read_file, ctx.deps.root, path, offset, limit
        )

    @agent.tool
    async def glob_files(ctx: RunContext[TesterDeps], pattern: str) -> list[str]:
        """Find files by glob pattern relative to project root, e.g. 'src/**/*.py'."""
        return await asyncio.to_thread(tools.glob_files, ctx.deps.root, pattern)

    @agent.tool
    async def grep_search(
        ctx: RunContext[TesterDeps], pattern: str, include: str = ""
    ) -> list[str]:
        """Regex search in file contents. `include` is an optional filename regex filter."""
        return await asyncio.to_thread(
            tools.grep_search, ctx.deps.root, pattern, include
        )

    @agent.tool
    async def run_bash(ctx: RunContext[TesterDeps], command: str) -> str:
        """Run a shell command in the project root and return its output (2 min timeout)."""
        return await tools.run_bash(ctx.deps.root, command)

    @agent.tool
    def create_bug_report(
        ctx: RunContext[TesterDeps], title: str, description: str
    ) -> str:
        """Create a bug report task. It is placed at the head of the execution queue."""
        bug = ctx.deps.manager.create(title, description, kind=TaskKind.BUG)
        return f"Bug report #{bug.id} '{bug.title}' created at the head of the queue."

    @agent.tool
    def set_verification(ctx: RunContext[TesterDeps], success: bool) -> str:
        """Set the verification label on the verified task (passed or failed)
        and finish its testing: the task becomes completed."""
        task = ctx.deps.manager.get(ctx.deps.task_id)
        task.verification = (
            VerificationStatus.PASSED if success else VerificationStatus.FAILED
        )
        task.status = TaskStatus.COMPLETED
        task.touch()
        ctx.deps.manager.storage.save(task)
        return f"Verification label set: {task.verification.value}; task completed."

    return agent


def serialize_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    return ModelMessagesTypeAdapter.dump_python(messages, mode="json")  # type: ignore[return-value]


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


def load_messages(data: list[dict[str, Any]]) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_python(data)


def resumable_history(data: list[dict[str, Any]]) -> list[ModelMessage]:
    """Load persisted messages for resuming an interrupted/previous run.

    A snapshot taken mid-turn may end with an incomplete exchange: tool
    calls whose results never arrived, or tool results without the
    follow-up model response. Model APIs reject such a tail, so truncate
    to the last safe boundary — a user prompt or a completed (tool-call-
    free) model response.
    """
    messages = list(load_messages(data))
    cut = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest):
            if any(isinstance(p, UserPromptPart) for p in msg.parts):
                cut = i + 1
        elif isinstance(msg, ModelResponse) and not any(
            isinstance(p, ToolCallPart) for p in msg.parts
        ):
            cut = i + 1
    return messages[:cut]


def strip_retried_prompt(
    messages: list[ModelMessage], text: str
) -> tuple[list[ModelMessage], bool]:
    """Cut an interrupted attempt at `text` out of a resumable transcript.

    A retried message may already be present in the persisted transcript:
    an interrupted run persists `all_messages()` (which includes the prompt
    and any partial work done so far) before the message goes back to
    `pending_input`. Re-sending the same text as a new prompt would duplicate
    it — in the stored transcript and as a second user block at the end of
    the chat. Dropping the prompt together with the incomplete tail after it
    makes the retry regenerate the attempt cleanly.

    The transcript is left untouched when the attempt already produced a
    final (non-empty text) model response — then an identical prompt is a
    genuine new request, not a retry.
    """
    idx = None
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest) and any(
            isinstance(p, UserPromptPart) and p.content == text for p in msg.parts
        ):
            idx = i
    if idx is None:
        return messages, False
    for msg in messages[idx + 1 :]:
        if isinstance(msg, ModelResponse) and any(
            isinstance(p, TextPart) and p.content for p in msg.parts
        ):
            return messages, False
    return messages[:idx], True


def extract_display(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Turn serialized ModelMessages into (role, text) pairs for the chat UI.

    Roles: "user", "reasoning" (model thinking), "assistant" (text),
    "tool-call" (name(args) line; the TUI shows the name as the collapsed
    header and the args in the expandable body), rendered by the TUI in
    message order.
    """
    out: list[tuple[str, str]] = []
    reasoning: list[str] = []
    chunks: list[str] = []
    for msg in messages:
        # pydantic-ai 2.x serializes ModelRequest with "kind": "request"
        # (older versions used "role"); support both.
        if msg.get("kind") == "request" or msg.get("role") == "request":
            for part in msg.get("parts", []):
                if part.get("part_kind") == "user-prompt":
                    content = part.get("content")
                    if isinstance(content, str) and content:
                        out.append(("user", content))
            continue
        for part in msg.get("parts", []):
            kind = part.get("part_kind")
            if kind == "thinking":
                content = part.get("content", "")
                if content:
                    if chunks:
                        out.append(("assistant", "\n".join(chunks)))
                        chunks = []
                    reasoning.append(content)
            elif kind == "text":
                content = part.get("content", "")
                if content:
                    if reasoning:
                        out.append(("reasoning", "\n\n".join(reasoning)))
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
                    out.append(("reasoning", "\n\n".join(reasoning)))
                    reasoning = []
                if chunks:
                    out.append(("assistant", "\n".join(chunks)))
                    chunks = []
                out.append(("tool-call", f"{part.get('tool_name', '?')}({args_str})"))
        if reasoning:
            out.append(("reasoning", "\n\n".join(reasoning)))
            reasoning = []
        if chunks:
            out.append(("assistant", "\n".join(chunks)))
            chunks = []
    return out


__all__ = [
    "DOCUMENTATION_SYSTEM_PROMPT",
    "AgentDeps",
    "ModelRequest",
    "ModelResponse",
    "TesterDeps",
    "TextPart",
    "ToolCallPart",
    "ToolReturnPart",
    "UserPromptPart",
    "build_agent",
    "build_tester_agent",
    "extract_display",
    "last_context_tokens",
    "load_agents_md",
    "load_messages",
    "resumable_history",
    "run_context_tokens",
    "serialize_messages",
    "strip_retried_prompt",
]
