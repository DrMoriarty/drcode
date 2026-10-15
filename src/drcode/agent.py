"""pydantic-ai agents wired to an OpenAI-compatible endpoint, with coding tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
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


def build_agent(config: Config) -> Agent[AgentDeps, str]:
    model = build_model(config)
    agent = Agent(
        model,
        instructions=SYSTEM_PROMPT,
        deps_type=AgentDeps,
        model_settings={"extra_headers": EXTRA_HEADERS},
    )

    @agent.tool
    def read_file(
        ctx: RunContext[AgentDeps], path: str, offset: int = 1, limit: int = 2000
    ) -> str:
        """Read a text file. Lines are numbered, starting from `offset` (1-based), up to `limit` lines."""
        return tools.read_file(ctx.deps.root, path, offset, limit)

    @agent.tool
    def write_file(ctx: RunContext[AgentDeps], path: str, content: str) -> str:
        """Create or overwrite a file with the given content."""
        return tools.write_file(ctx.deps.root, path, content)

    @agent.tool
    def edit_file(
        ctx: RunContext[AgentDeps], path: str, old_string: str, new_string: str
    ) -> str:
        """Replace an exact unique substring in a file. old_string must match exactly once."""
        return tools.edit_file(ctx.deps.root, path, old_string, new_string)

    @agent.tool
    def glob_files(ctx: RunContext[AgentDeps], pattern: str) -> list[str]:
        """Find files by glob pattern relative to project root, e.g. 'src/**/*.py'."""
        return tools.glob_files(ctx.deps.root, pattern)

    @agent.tool
    def grep_search(
        ctx: RunContext[AgentDeps], pattern: str, include: str = ""
    ) -> list[str]:
        """Regex search in file contents. `include` is an optional filename regex filter."""
        return tools.grep_search(ctx.deps.root, pattern, include)

    @agent.tool
    def run_bash(ctx: RunContext[AgentDeps], command: str) -> str:
        """Run a shell command in the project root and return its output (2 min timeout)."""
        return tools.run_bash(ctx.deps.root, command)

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
    def read_file(
        ctx: RunContext[TesterDeps], path: str, offset: int = 1, limit: int = 2000
    ) -> str:
        """Read a text file. Lines are numbered, starting from `offset` (1-based), up to `limit` lines."""
        return tools.read_file(ctx.deps.root, path, offset, limit)

    @agent.tool
    def glob_files(ctx: RunContext[TesterDeps], pattern: str) -> list[str]:
        """Find files by glob pattern relative to project root, e.g. 'src/**/*.py'."""
        return tools.glob_files(ctx.deps.root, pattern)

    @agent.tool
    def grep_search(
        ctx: RunContext[TesterDeps], pattern: str, include: str = ""
    ) -> list[str]:
        """Regex search in file contents. `include` is an optional filename regex filter."""
        return tools.grep_search(ctx.deps.root, pattern, include)

    @agent.tool
    def run_bash(ctx: RunContext[TesterDeps], command: str) -> str:
        """Run a shell command in the project root and return its output (2 min timeout)."""
        return tools.run_bash(ctx.deps.root, command)

    @agent.tool
    def create_bug_report(
        ctx: RunContext[TesterDeps], title: str, description: str
    ) -> str:
        """Create a bug report task. It is placed at the head of the execution queue."""
        bug = ctx.deps.manager.create(title, description, kind=TaskKind.BUG)
        return f"Bug report #{bug.id} '{bug.title}' created at the head of the queue."

    @agent.tool
    def set_verification(ctx: RunContext[TesterDeps], success: bool) -> str:
        """Set the verification label on the verified task: passed or failed."""
        task = ctx.deps.manager.get(ctx.deps.task_id)
        task.verification = (
            VerificationStatus.PASSED if success else VerificationStatus.FAILED
        )
        task.touch()
        ctx.deps.manager.storage.save(task)
        return f"Verification label set: {task.verification.value}"

    return agent


def serialize_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    return ModelMessagesTypeAdapter.dump_python(messages, mode="json")  # type: ignore[return-value]


def load_messages(data: list[dict[str, Any]]) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_python(data)


def extract_display(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Turn serialized ModelMessages into (role, text) pairs for the chat UI.

    Roles: "user", "reasoning" (model thinking), "assistant" (text),
    "tool-call" (name + args as one line), rendered by the TUI in message
    order.
    """
    out: list[tuple[str, str]] = []
    reasoning: list[str] = []
    chunks: list[str] = []
    for msg in messages:
        if msg.get("role") == "request":
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
    "load_messages",
    "serialize_messages",
]
