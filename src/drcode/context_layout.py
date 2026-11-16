"""Context-layout estimation for a task's model context (≈, no tokenizer).

Answers "what is actually inside the context slot of this task's model
requests": the system prompt (base template + AGENTS.md + git workflow, plus
the DYNAMIC goals instructions — the roadmap block for research tasks, the
per-task "Task goal" block for tasks with a linked goal — reconstructed by
`goals_system_component` and estimated as their own "system prompt" line,
since pydantic-ai renders dynamic instructions into a separate leading
system message of every request), the tool descriptions (docstrings + JSON
schemas), and the serialized transcript by part kind (user prompts, retry
prompts, compaction summary, model generation, tool returns).

There is deliberately NO tokenizer here — the model behind drcode is any
OpenAI-compatible endpoint, so every figure is the conventional
``len(text) / 4`` heuristic, marked with "≈" everywhere it is shown. The
exact numbers (the API's own usage counters) are reported separately and are
NOT used to normalize the estimate: the sum of the estimated sections will
not equal the exact context fill (chat template glue, tool-call JSON
round-trips, per-message framing are all invisible to chars/4).

The system prompt and tool schemas never appear in the transcript — they are
reconstructed by reusing the exact wiring of `drcode.agent`
(`build_agent` / `build_tester_agent` / `load_agents_md`) and
`drcode.prompts` (`load_*_components`), so the tool set matches the role and
kind (todo — FEATURE only, create_* — RESEARCH only, the tester's
git_commit — only when the tester's git policy allows commits,
git_commit_docs — only under a non-off research git policy) and the
prompt matches the `.drcode/prompts/` overrides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.usage import RunUsage

from .agent import (
    build_agent,
    build_resolver_agent,
    build_tester_agent,
    goals_instructions,
    load_agents_md,
    task_goal_block,
)
from .config import Config
from .models import Goal, Task, TaskKind
from .prompts import (
    load_resolver_prompt,
    load_system_prompt_components,
    load_tester_prompt_components,
)
from .settings import GitPolicy
from .transcript import (  # noqa: F401 - CHARS_PER_TOKEN is re-exported
    CHARS_PER_TOKEN,
    COMPACTION_SUMMARY_PREFIX,
    _message_time,
    estimate_tokens,
    last_context_tokens,
)
from .web_fetch import WebFetchPolicy

# `estimate_tokens` and `CHARS_PER_TOKEN` live in `transcript` now (shared
# with the compaction module) and are re-exported here — the existing
# `from context_layout import estimate_tokens` consumers keep working.

# How many of the heaviest tool results the report lists in detail.
TOP_TOOL_RESULTS = 5

# Cap of one tool-result label in the top-list (chars).
_TOOL_LABEL_CAP = 60

# ≈-token stand-in for a URL-referenced image in a multimodal user prompt
# (the gateway fetches the bytes server-side, the size is unknowable here —
# the point is to not claim 0). Base64-inline payloads are measured by their
# own length instead, which is the honest number.
IMAGE_URL_ESTIMATE_TOKENS = 1000


def format_tokens(n: int) -> str:
    """Compact human token count (e.g. 45k, 1.2M, 980) — shared by the
    context dialog and the status-bar counter."""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}".rstrip("0").rstrip(".") + "M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.1f}".rstrip("0").rstrip(".") + "k"
    return str(n)


# -- system prompt / tool-schema reconstruction (reuses agent wiring) --------


def system_components_coding(
    kind: TaskKind,
    project_root: Path | None,
    research_git: GitPolicy = GitPolicy.PUSH,
) -> list[tuple[str, str | None]]:
    """Labeled system-prompt pieces of a coding agent run: the base template
    (project override honored), the project's AGENTS.md, the git-workflow
    rules and the per-role instructions fragment (None when the project
    ships none) — exactly what `build_agent` assembles into the
    instructions. For research tasks `research_git` selects the git
    fragment (or none under the OFF policy), matching
    `build_agent(research_git=...)`."""
    base, git_workflow, role_instructions = load_system_prompt_components(
        kind, project_root, research_git
    )
    agents_md = load_agents_md(project_root) if project_root is not None else ""
    return [
        ("base template", base),
        ("AGENTS.md", agents_md.strip() or None),
        ("git workflow", git_workflow),
        ("role instructions", role_instructions),
    ]


def goals_system_component(
    kind: TaskKind,
    goal_id: str | None,
    active_goals: list[Goal],
    linked_goal: Goal | None,
) -> tuple[str, str] | None:
    """Reconstruct the DYNAMIC goals block the coding agent injects into the
    leading system messages of every request (`goals_dynamic_instructions`,
    registered in `agent_build` on the coding agent for all kinds): the
    exact text pydantic-ai renders as a separate leading system message,
    which the static system-components reconstruction above never sees.

    Mirrors the hook's branching exactly — RESEARCH tasks get the roadmap
    block over `active_goals` (the task's own goal is visible there
    already), any other kind with a linked goal gets the compact per-task
    block; empty registry / no link / deleted goal → None (the hook yields
    None and the block is excluded from the request entirely — there is no
    "absent" placeholder to estimate)."""
    if kind is TaskKind.RESEARCH:
        block = goals_instructions(active_goals)
        label = "product goals"
    elif goal_id is not None and linked_goal is not None:
        block = task_goal_block(linked_goal)
        label = "task goal (linked)"
    else:
        return None
    if block is None:
        return None
    return (label, block)


def system_components_tester(
    allow_commit: bool, project_root: Path | None, allow_push: bool = True
) -> list[tuple[str, str | None]]:
    """Labeled system-prompt pieces of the tester run: the base template
    (project override honored), the project's AGENTS.md, the git-workflow
    rules and the per-role instructions fragment (None when the project
    ships none) — exactly what `build_tester_agent` assembles into the
    instructions."""
    base, git_workflow, role_instructions = load_tester_prompt_components(
        allow_commit, project_root, allow_push
    )
    agents_md = load_agents_md(project_root) if project_root is not None else ""
    return [
        ("base template", base),
        ("AGENTS.md", agents_md.strip() or None),
        ("git workflow", git_workflow),
        ("role instructions", role_instructions),
    ]


def _tool_schema_text(name: str, description: str, schema: Any) -> str:
    """The tool's wire representation as sent to the model: name,
    docstring and the pydantic-generated parameter schema, serialized the
    way the OpenAI chat-completions tool definition is built."""
    return json.dumps(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description or "",
                "parameters": schema or {},
            },
        },
        ensure_ascii=False,
    )


def tool_schemas(agent: Any) -> list[tuple[str, str]]:
    """(name, wire-text) of every function tool registered on a built
    agent. Reads the agent's own function-toolset, so the set matches the
    run exactly (todo tools — FEATURE only, create_* — RESEARCH only, the
    tester's git_commit — only with commits allowed)."""
    tools: list[tuple[str, str]] = []
    toolset = getattr(agent, "_function_toolset", None)
    for tool in getattr(toolset, "tools", {}).values():
        defn = tool.tool_def
        tools.append(
            (
                defn.name,
                _tool_schema_text(
                    defn.name, defn.description, defn.parameters_json_schema
                ),
            )
        )
    tools.sort(key=lambda item: item[0])
    return tools


def coding_tool_schemas(
    config: Config,
    kind: TaskKind,
    root: Path | None = None,
    research_git: GitPolicy = GitPolicy.PUSH,
    subagent_limit: int = 0,
    research_web_fetch: WebFetchPolicy | None = None,
) -> list[tuple[str, str]]:
    """Tool wire-texts of the coding agent for `kind` (builds the agent —
    no network, pure wiring). `research_git`, `subagent_limit` and
    `research_web_fetch` match `build_agent` — pass the real setting, or
    the estimate misses (or invents) the delegate_research / web_fetch
    schemas. `research_web_fetch` is the RESEARCH policy (None for every
    other kind — see `web_fetch.web_fetch_policy_for`)."""
    return tool_schemas(
        build_agent(
            config,
            kind=kind,
            root=root,
            research_git=research_git,
            subagent_limit=subagent_limit,
            research_web_fetch=research_web_fetch,
        )
    )


def tester_tool_schemas(
    config: Config,
    *,
    allow_commit: bool = True,
    allow_push: bool = True,
    root: Path | None = None,
) -> list[tuple[str, str]]:
    """Tool wire-texts of the tester agent."""
    return tool_schemas(
        build_tester_agent(
            config, allow_commit=allow_commit, allow_push=allow_push, root=root
        )
    )


def system_components_resolver(
    project_root: Path | None,
) -> list[tuple[str, str | None]]:
    """Labeled system-prompt pieces of the merge-conflict resolver run: the
    base template (project override honored) and the project's AGENTS.md —
    exactly what `build_resolver_agent` assembles into the instructions
    (the resolver has no git-workflow fragment; see its builder)."""
    base = load_resolver_prompt(project_root)
    agents_md = load_agents_md(project_root) if project_root is not None else ""
    return [
        ("base template", base),
        ("AGENTS.md", agents_md.strip() or None),
    ]


def resolver_tool_schemas(
    config: Config,
    *,
    root: Path | None = None,
) -> list[tuple[str, str]]:
    """Tool wire-texts of the resolver agent (read/inspect tools + the
    edit pair + set_resolution — whatever `build_resolver_agent`
    registers; no network, pure wiring)."""
    return tool_schemas(build_resolver_agent(config, root=root))


# -- per-agent selection (shared by the status-bar counter and the dialog) ----

# The three transcripts/agents of a task: development (task.messages, served
# by the task's role), a testing round (task.tester_messages) and a
# conflict-resolution round (task.resolver_messages).
AGENT_DEV = "dev"
AGENT_TESTER = "tester"
AGENT_RESOLVER = "resolver"


def agent_transcript(task: Task, agent: str) -> list[dict[str, Any]]:
    """The transcript whose context the given agent's figures describe: the
    development transcript, the LAST testing round or the LAST resolver
    round (each round is a separate context). Empty when the agent never
    ran."""
    if agent == AGENT_DEV:
        return task.messages
    if agent == AGENT_TESTER:
        return task.tester_messages[-1] if task.tester_messages else []
    if agent == AGENT_RESOLVER:
        return task.resolver_messages[-1] if task.resolver_messages else []
    raise ValueError(f"unknown agent: {agent}")


def last_agent(task: Task) -> str:
    """Which agent worked on `task` LAST (heuristic over the persisted
    rounds, no Task model changes).

    The resolver counts as last when it has rounds and no NEW testing
    round followed its last one (round order decided by the message
    timestamps pydantic-ai already serializes into the parts); otherwise
    the tester when it has rounds; otherwise the developer. Missing
    timestamps read as "no tester round after" — the resolver is the
    conservative pick (its rounds are the rarest and the most recent by
    construction). Interrupted development after a tester round is a
    known, accepted imprecision of the heuristic."""
    if task.resolver_messages and task.resolver_messages[-1]:
        tester_time = (
            _message_time(task.tester_messages[-1][-1])
            if task.tester_messages and task.tester_messages[-1]
            else None
        )
        resolver_time = _message_time(task.resolver_messages[-1][-1])
        if tester_time is None or resolver_time is None or tester_time <= resolver_time:
            return AGENT_RESOLVER
        return AGENT_TESTER
    if task.tester_messages and task.tester_messages[-1]:
        return AGENT_TESTER
    return AGENT_DEV


# -- transcript analysis ------------------------------------------------------


@dataclass
class ToolResult:
    """One tool result of the transcript with its ≈ token weight and a
    human label (file / command where reconstructable from the call args)."""

    tool_name: str
    label: str
    tokens: int


@dataclass
class TranscriptStats:
    """≈ token layout of one serialized transcript, by part kind."""

    user_prompts: int = 0
    user_prompt_count: int = 0
    retry_prompts: int = 0
    retry_prompt_count: int = 0
    summary: int = 0
    generation: int = 0
    tool_results: int = 0
    tool_result_count: int = 0
    heaviest: list[ToolResult] = field(default_factory=list)
    compacted: bool = False
    message_count: int = 0
    tool_calls: int = 0
    # Cumulative exact usage over every model response of the transcript
    # (the billing-grade counters the API reports per request).
    usage: RunUsage = field(default_factory=RunUsage)
    # Number of model responses that reported usage (≈ request count).
    requests: int = 0
    # Exact usage of the LAST model response (cache split lives here only).
    last_cache_read: int = 0
    last_cache_write: int = 0


def _usage_of_response(msg: dict[str, Any]) -> RunUsage | None:
    u = msg.get("usage") or {}
    usage = RunUsage(
        input_tokens=u.get("input_tokens") or 0,
        cache_read_tokens=u.get("cache_read_tokens") or 0,
        cache_write_tokens=u.get("cache_write_tokens") or 0,
        output_tokens=u.get("output_tokens") or 0,
    )
    if (
        usage.input_tokens
        or usage.output_tokens
        or usage.cache_read_tokens
        or usage.cache_write_tokens
    ):
        return usage
    return None


def _part_text(part: dict[str, Any]) -> str:
    """Rendered text of a part content (tool results may be structured)."""
    content = part.get("content")
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(content)


def _short(value: str, cap: int = _TOOL_LABEL_CAP) -> str:
    value = " ".join(value.split())
    if len(value) > cap:
        return value[: cap - 1] + "…"
    return value


def _result_label(tool_name: str, args: Any) -> str:
    """Label of a tool result: the file/command the call targeted where the
    args allow it, otherwise a short args excerpt."""
    args_dict: dict[str, Any] = {}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            args_dict = parsed if isinstance(parsed, dict) else {}
        except ValueError:
            args_dict = {}
    elif isinstance(args, dict):
        args_dict = args
    for key in ("path", "command"):
        value = args_dict.get(key)
        if isinstance(value, str) and value:
            return f"{tool_name}: {_short(value)}"
    excerpt = _short(json.dumps(args_dict, ensure_ascii=False)) if args_dict else ""
    return f"{tool_name}: {excerpt}" if excerpt else tool_name


def analyze_transcript(messages: list[dict[str, Any]]) -> TranscriptStats:
    """Walk one serialized transcript and bucket every part's ≈ tokens."""

    def bump(bucket: str, text: str, count_attr: str | None = None) -> None:
        setattr(stats, bucket, getattr(stats, bucket) + estimate_tokens(text))
        if count_attr:
            setattr(stats, count_attr, getattr(stats, count_attr) + 1)

    stats = TranscriptStats(message_count=len(messages))
    # tool_call_id -> (tool name, args) of the calls seen so far, to label
    # their results (the result part itself carries only the call id).
    calls: dict[str, tuple[str, Any]] = {}
    results: list[ToolResult] = []
    for msg in messages:
        is_request = msg.get("kind") == "request" or msg.get("role") == "request"
        for part in msg.get("parts", []):
            if not isinstance(part, dict):
                continue
            kind = part.get("part_kind")
            if kind == "user-prompt":
                content = part.get("content")
                if isinstance(content, str):
                    if not content:
                        continue
                    if content.startswith(COMPACTION_SUMMARY_PREFIX):
                        stats.compacted = True
                        bump("summary", content[len(COMPACTION_SUMMARY_PREFIX) :])
                    else:
                        bump("user_prompts", content, "user_prompt_count")
                elif isinstance(content, list) and content:
                    # Multimodal prompt (text element + image items): the
                    # images are NOT free — a serialized BinaryContent rides
                    # in the transcript as base64 and is re-sent to the API
                    # on every request. Count the text items by the usual
                    # chars/4 and the base64 payload by its own length (the
                    # gateway bills it as input tokens the same way).
                    total = 0
                    for item in content:
                        if isinstance(item, str):
                            total += estimate_tokens(item)
                        elif isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                total += estimate_tokens(text)
                                continue
                            data = item.get("data")
                            if isinstance(data, str):
                                total += estimate_tokens(data)
                            elif isinstance(item.get("url"), str):
                                # A URL reference: the gateway fetches the
                                # bytes server-side, the size is unknowable
                                # here — a modest stand-in per image keeps
                                # the estimate from claiming 0.
                                total += IMAGE_URL_ESTIMATE_TOKENS
                    if total:
                        stats.user_prompts += total
                        stats.user_prompt_count += 1
            elif kind == "retry-prompt":
                # ModelRetry prompts: model-facing corrections, NOT user
                # messages — reported as their own line.
                bump("retry_prompts", _part_text(part), "retry_prompt_count")
            elif kind == "tool-return":
                text = _part_text(part)
                call_id = str(part.get("tool_call_id") or "")
                name, args = calls.get(
                    call_id, (str(part.get("tool_name") or "?"), None)
                )
                results.append(
                    ToolResult(
                        tool_name=name,
                        label=_result_label(name, args),
                        tokens=estimate_tokens(text),
                    )
                )
                bump("tool_results", text, "tool_result_count")
            elif kind == "tool-call":
                stats.tool_calls += 1
                call_id = part.get("tool_call_id")
                if call_id:
                    calls[call_id] = (part.get("tool_name", "?"), part.get("args"))
            elif kind in ("text", "thinking"):
                # Model generation: visible answers and thinking alike.
                bump("generation", part.get("content", "") or "")
        if not is_request:
            usage = _usage_of_response(msg)
            if usage is not None:
                stats.usage = stats.usage + usage
                stats.requests += 1
                if usage.cache_read_tokens or usage.cache_write_tokens:
                    stats.last_cache_read = usage.cache_read_tokens
                    stats.last_cache_write = usage.cache_write_tokens
    stats.heaviest = sorted(results, key=lambda r: r.tokens, reverse=True)[
        :TOP_TOOL_RESULTS
    ]
    return stats


def cumulative_usage(transcripts: list[list[dict[str, Any]]]) -> tuple[RunUsage, int]:
    """Exact cumulative (RunUsage, request count) over every model response
    of the given transcripts — the whole-task spend."""
    total = RunUsage()
    requests = 0
    for messages in transcripts:
        for msg in messages:
            if msg.get("kind") == "response" or msg.get("role") == "response":
                usage = _usage_of_response(msg)
                if usage is not None:
                    total = total + usage
                    requests += 1
    return total, requests


# -- report model (consumed by the TUI dialog) --------------------------------


@dataclass
class CtxLine:
    """One row of the report: a label, its ≈ tokens and an optional note."""

    label: str
    tokens: int
    note: str = ""


@dataclass
class CtxSection:
    """A titled group of lines (system prompt, tools, transcript) with the
    sum of its lines' estimates."""

    title: str
    lines: list[CtxLine]
    total: int


@dataclass
class CtxReport:
    """The full ≈ layout of ONE transcript (development or one testing
    round), plus the exact figures shown next to it."""

    title: str
    model: str
    max_context_tokens: int
    threshold_pct: int
    max_output_tokens: int
    exact_tokens: int | None
    estimate_total: int
    compacted: bool
    message_count: int
    tool_calls: int
    sections: list[CtxSection]
    heaviest: list[ToolResult]
    last_cache_read: int
    last_cache_write: int

    @property
    def exact_pct(self) -> float | None:
        return self._pct(self.exact_tokens)

    @property
    def estimate_pct(self) -> float | None:
        return self._pct(self.estimate_total)

    @property
    def threshold_tokens(self) -> int | None:
        if self.max_context_tokens <= 0:
            return None
        return round(self.threshold_pct * self.max_context_tokens / 100)

    @property
    def headroom_tokens(self) -> int | None:
        """Tokens left before the compaction threshold (from the exact
        reading when there is one, from the estimate otherwise)."""
        if self.threshold_tokens is None:
            return None
        tokens = self.exact_tokens if self.exact_tokens is not None else None
        if tokens is None:
            return None
        return max(0, self.threshold_tokens - tokens)

    def _pct(self, tokens: int | None) -> float | None:
        if self.max_context_tokens <= 0 or tokens is None:
            return None
        return round(tokens * 100 / self.max_context_tokens, 1)


def build_report(
    title: str,
    messages: list[dict[str, Any]],
    config: Config,
    *,
    system_components: list[tuple[str, str | None]],
    tools: list[tuple[str, str]],
    goals_component: tuple[str, str] | None = None,
    exact_tokens: int | None = None,
) -> CtxReport:
    """Assemble the report of one transcript: exact + estimated figures,
    the three sections and the heaviest tool results.

    `goals_component` is the (label, text) reconstruction of the coding
    agent's DYNAMIC goals instructions (see `goals_system_component`) —
    pydantic-ai renders them into a separate leading system message of
    every request, invisible to the static `system_components`, so it is
    estimated as its own line of the "system prompt" section (None when
    the agent carries no goals block; only the development report ever
    passes it — the tester and resolver agents register no goals hook)."""
    stats = analyze_transcript(messages)
    if exact_tokens is None:
        exact_tokens = last_context_tokens(messages)

    system_lines = [
        CtxLine(label, estimate_tokens(text) if text else 0, "" if text else "absent")
        for label, text in system_components
    ]
    if goals_component is not None:
        goals_label, goals_text = goals_component
        system_lines.append(
            CtxLine(
                goals_label,
                estimate_tokens(goals_text),
                "dynamic instructions — in every request",
            )
        )
    tool_lines = [
        CtxLine(name, estimate_tokens(text), "docstring + schema")
        for name, text in tools
    ]
    transcript_lines = [
        CtxLine(
            "user prompts",
            stats.user_prompts,
            f"{stats.user_prompt_count} message(s)",
        ),
    ]
    if stats.retry_prompt_count:
        transcript_lines.append(
            CtxLine(
                "retry prompts (ModelRetry)",
                stats.retry_prompts,
                f"{stats.retry_prompt_count} correction(s), not user messages",
            )
        )
    if stats.compacted:
        transcript_lines.append(
            CtxLine("compaction summary", stats.summary, "synthetic, not a user prompt")
        )
    transcript_lines.extend(
        [
            CtxLine("model generation", stats.generation, "text + thinking parts"),
            CtxLine(
                "tool results",
                stats.tool_results,
                f"{stats.tool_result_count} result(s)",
            ),
        ]
    )
    sections = [
        CtxSection(
            "system prompt",
            system_lines,
            sum(line.tokens for line in system_lines),
        ),
        CtxSection(
            "tools (descriptions + schemas)",
            tool_lines,
            sum(line.tokens for line in tool_lines),
        ),
        CtxSection(
            "transcript",
            transcript_lines,
            sum(line.tokens for line in transcript_lines),
        ),
    ]
    threshold = min(max(config.compaction_threshold, 1), 100)
    return CtxReport(
        title=title,
        model=config.model,
        max_context_tokens=config.max_context_tokens,
        threshold_pct=threshold,
        max_output_tokens=config.max_tokens,
        exact_tokens=exact_tokens,
        estimate_total=sum(section.total for section in sections),
        compacted=stats.compacted,
        message_count=stats.message_count,
        tool_calls=stats.tool_calls,
        sections=sections,
        heaviest=stats.heaviest,
        last_cache_read=stats.last_cache_read,
        last_cache_write=stats.last_cache_write,
    )


def task_reports(
    task: Task,
    dev_config: Config,
    tester_config: Config,
    resolver_config: Config,
    dev_role_label: str,
    project_root: Path,
    *,
    live_tokens: int | None = None,
    tester_live_tokens: int | None = None,
    resolver_live_tokens: int | None = None,
    tester_allow_commit: bool = True,
    tester_allow_push: bool = True,
    research_git: GitPolicy = GitPolicy.PUSH,
    subagent_limit: int = 0,
    research_web_fetch: WebFetchPolicy | None = None,
    active_goals: list[Goal] | None = None,
    linked_goal: Goal | None = None,
) -> tuple[CtxReport | None, CtxReport | None, CtxReport | None, RunUsage, int]:
    """The per-agent transcript reports of one task plus its whole-task
    cumulative exact usage.

    Development: `task.messages` measured against `dev_config` — the resolved
    config of the role serving the task (`_task_role`: research tasks run on
    the research model, everything else on coding); `research_git` mirrors
    the project's research git policy so prompt and tool set match the real
    wiring. Testing: the latest round
    of `task.tester_messages` against `tester_config`; Resolver: the latest
    round of `task.resolver_messages` against `resolver_config` (the rounds
    are separate contexts — neither the tester nor the resolver ever sees
    the development transcript). `research_web_fetch` is the RESEARCH web
    fetch policy (None when the role has no web_fetch tool), so the tool
    schema is counted exactly when the run really carries it. `active_goals`
    (the GoalStore's active list) and `linked_goal` (the goal behind
    `task.goal_id`, None when absent or deleted) feed the reconstruction of
    the coding agent's DYNAMIC goals instructions — they ride every dev
    request as leading system messages and are estimated as their own line
    of the development report's "system prompt" section (absent there when
    the hook yields None); the tester and resolver never carry goals, so
    their reports are unaffected. `*_live_tokens`
    carry the live in-run
    readings of the respective agents where one is running (they win over
    the persisted transcript, like the dev counter does).
    """
    dev: CtxReport | None = None
    if task.messages:
        dev = build_report(
            f"Development ({dev_role_label})",
            task.messages,
            dev_config,
            system_components=system_components_coding(
                task.kind, project_root, research_git
            ),
            tools=coding_tool_schemas(
                dev_config,
                task.kind,
                root=project_root,
                research_git=research_git,
                subagent_limit=subagent_limit,
                research_web_fetch=research_web_fetch,
            ),
            goals_component=goals_system_component(
                task.kind, task.goal_id, active_goals or [], linked_goal
            ),
            exact_tokens=live_tokens,
        )
    tester: CtxReport | None = None
    round_messages = task.tester_messages[-1] if task.tester_messages else []
    if round_messages:
        tester = build_report(
            f"Testing (tester, round {len(task.tester_messages)})",
            round_messages,
            tester_config,
            system_components=system_components_tester(
                tester_allow_commit, project_root, tester_allow_push
            ),
            tools=tester_tool_schemas(
                tester_config,
                allow_commit=tester_allow_commit,
                allow_push=tester_allow_push,
                root=project_root,
            ),
            exact_tokens=tester_live_tokens,
        )
    resolver: CtxReport | None = None
    resolver_messages = task.resolver_messages[-1] if task.resolver_messages else []
    if resolver_messages:
        resolver = build_report(
            f"Resolver (conflict resolution, round {len(task.resolver_messages)})",
            resolver_messages,
            resolver_config,
            system_components=system_components_resolver(project_root),
            tools=resolver_tool_schemas(resolver_config, root=project_root),
            exact_tokens=resolver_live_tokens,
        )
    usage, requests = cumulative_usage(
        [task.messages, *task.tester_messages, *task.resolver_messages]
    )
    return dev, tester, resolver, usage, requests
