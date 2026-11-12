"""Project-scoped settings, persisted in `<project>/.drcode/settings.json`.

Deliberately separate from `config.Config` (env-driven, model/API settings):
these are user preferences of the current project, so they live next to its
tasks and survive restarts regardless of the environment.
"""

from __future__ import annotations

import enum
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator


class CompletedRetention(str, enum.Enum):
    """How long successfully completed tasks are kept before automatic
    removal at startup."""

    DAYS_7 = "7"
    DAYS_30 = "30"
    DAYS_90 = "90"
    NEVER = "never"

    @property
    def days(self) -> int | None:
        """Retention period in days, or None when tasks are never removed."""
        if self is CompletedRetention.NEVER:
            return None
        return int(self.value)

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is CompletedRetention.NEVER:
            return "never"
        return f"after {self.value} days"


RETENTION_OPTIONS: tuple[CompletedRetention, ...] = (
    CompletedRetention.DAYS_7,
    CompletedRetention.DAYS_30,
    CompletedRetention.DAYS_90,
    CompletedRetention.NEVER,
)


class Verification(str, enum.Enum):
    """Whether the QA tester round runs at all (project setting).

    TESTER (the default): after development the tester verifies the
    project before the task completes. NONE: no tester at all —
    development tasks go straight to COMPLETED when the agent finishes
    (the same completion path research tasks have always used).
    """

    TESTER = "tester"
    NONE = "none"

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is Verification.TESTER:
            return "tester"
        return "none"


VERIFICATION_OPTIONS: tuple[Verification, ...] = (
    Verification.TESTER,
    Verification.NONE,
)


class GitPolicy(str, enum.Enum):
    """What git actions a role may take on the task's branch (project
    setting, one per role — see `GitPolicySetting`).

    OFF: the role never touches git — no tool is registered and the
    system prompt promises no commit step (for the tester that means
    verification only; for the research agent the edited files are still
    tracked, but only for the "edited files" report). LOCAL: the role
    commits to the task's branch `drcode/<id>` but never pushes — the
    branch stays local. PUSH (the default): commit + push to the remote
    (the historical behavior; a push failure is never fatal — logged
    into the task's event log only).
    """

    OFF = "off"
    LOCAL = "local"
    PUSH = "push"

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is GitPolicy.OFF:
            return "off"
        if self is GitPolicy.LOCAL:
            return "commit locally"
        return "commit + push"

    @property
    def commits(self) -> bool:
        """Whether the role is allowed to commit at all."""
        return self is not GitPolicy.OFF

    @property
    def pushes(self) -> bool:
        """Whether the role pushes the branch after committing."""
        return self is GitPolicy.PUSH


GIT_POLICY_OPTIONS: tuple[GitPolicy, ...] = (
    GitPolicy.PUSH,
    GitPolicy.LOCAL,
    GitPolicy.OFF,
)


class GitPolicySetting(BaseModel):
    """Per-role git policies (persisted as a nested `git_policy` object
    in settings.json). `tester` governs the QA tester's final commit;
    `research` the research agent's git_commit_docs. The tester policy is
    DORMANT while `verification` is NONE (no tester runs to commit
    anything) — the value stays persisted (the user's choice is never
    silently rewritten) and the app warns about the dormant mode at
    startup, mirroring `parallel_tasks` under isolation off."""

    tester: GitPolicy = GitPolicy.PUSH
    research: GitPolicy = GitPolicy.PUSH


# Legacy `task_finalization` values → (verification, git_policy.tester).
# The research policy is NOT part of the legacy setting: it keeps its
# default (push — the historical behavior) on migration.
_LEGACY_FINALIZATION_MAP: dict[str, tuple[Verification, GitPolicy]] = {
    "testing_commit": (Verification.TESTER, GitPolicy.PUSH),
    "testing_commit_no_push": (Verification.TESTER, GitPolicy.LOCAL),
    "testing": (Verification.TESTER, GitPolicy.OFF),
    "none": (Verification.NONE, GitPolicy.PUSH),
}


# Options of the `max_dev_rounds` setting (the fix-loop cap), offered in the
# Settings picker.
DEV_ROUNDS_OPTIONS: tuple[int, ...] = (1, 2, 3, 5)


def dev_rounds_label(rounds: int) -> str:
    """Human-readable option label for the Settings UI."""
    return "1 round" if rounds == 1 else f"{rounds} rounds"


# Options of the `parallel_tasks` setting (the development-slot limit),
# offered in the Settings picker. Parallelism is only meaningful with
# worktree isolation: without it the effective limit stays 1.
PARALLEL_TASKS_OPTIONS: tuple[int, ...] = (1, 2, 3, 4)


def parallel_tasks_label(count: int) -> str:
    """Human-readable option label for the Settings UI."""
    return "1 task" if count == 1 else f"{count} tasks"


# Options of the `subagents_coder` / `subagents_researcher` settings (the
# per-role cap on concurrently running read-only subagents the coding /
# research agent may spawn via delegate_research). 0 (the default) = the
# delegation tool is not registered at all — the agent never sees it.
SUBAGENTS_OPTIONS: tuple[int, ...] = (0, 1, 2, 3, 4)


def subagents_label(count: int) -> str:
    """Human-readable option label for the Settings UI."""
    if count == 0:
        return "off"
    return "1 subagent" if count == 1 else f"{count} subagents"


class WebFetchMode(str, enum.Enum):
    """Whether the RESEARCH agent may read web pages (project setting).

    The research agent gets a structured `web_fetch` tool — the only place
    an outbound HTTP request is policy-checked, SSRF-guarded, size-capped
    and wrapped in an explicit "this is untrusted content" frame (see
    `drcode.web_fetch`). run_bash is local-only regardless of this setting:
    the bash network guard (`drcode.tools.check_command`) refuses external
    requests of shell clients even when no `web_fetch` tool exists. The
    coding agent, the tester and the resolver never get the tool.

    OFF (the default): the tool is not registered at all — the agent never
    sees it. ALLOWLIST: only hosts from `web_fetch_allow_hosts` (and their
    subdomains) may be fetched; an EMPTY allow list makes the tool useless
    (every fetch is refused with a hint to configure the hosts) — that is
    on purpose, the safe mode must never silently allow everything.
    ALL: every public host is fetchable except `web_fetch_block_hosts`
    (which wins over the allow list too) — the SSRF guard always applies,
    in every mode.
    """

    OFF = "off"
    ALLOWLIST = "allowlist"
    ALL = "all"

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is WebFetchMode.ALLOWLIST:
            return "allowlist"
        if self is WebFetchMode.ALL:
            return "all hosts"
        return "off"

    @property
    def enabled(self) -> bool:
        """Whether the research agent gets the web_fetch tool at all."""
        return self is not WebFetchMode.OFF

    @property
    def description(self) -> str:
        """One-line explanation shown in the Settings palette."""
        if self is WebFetchMode.ALLOWLIST:
            return (
                "The research agent may read ONLY the hosts from the web-fetch "
                "allow list (and their subdomains)."
            )
        if self is WebFetchMode.ALL:
            return (
                "The research agent may read any public web host (the block "
                "list and the private-network guard still apply)."
            )
        return "The research agent gets no web access tool at all."


WEB_FETCH_MODE_OPTIONS: tuple[WebFetchMode, ...] = (
    WebFetchMode.OFF,
    WebFetchMode.ALLOWLIST,
    WebFetchMode.ALL,
)


def normalize_host_entry(value: str) -> str:
    """Normalize one host entry of `web_fetch_allow_hosts`/`_block_hosts`.

    Accepts a bare host ("example.com"), a host with a port
    ("example.com:8080"), a pasted URL ("https://docs.example.com/guide")
    and an IPv6 literal ("[::1]"); returns the lowercased bare host
    without a trailing dot, or "" for junk. Deliberately shared with
    `drcode.web_fetch`, which matches URL hosts against these entries the
    same way (settings.py must not import web_fetch — that would be a
    cycle).
    """
    text = value.strip()
    if not text:
        return ""
    # A pasted URL: keep the authority part only.
    if "://" in text:
        text = text.split("://", 1)[1]
    text = text.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    text = text.rsplit("@", 1)[-1]  # drop userinfo
    if text.startswith("["):  # IPv6 literal (optionally with a port)
        text = text[1:].split("]", 1)[0]
    elif text.count(":") == 1:  # host:port
        text = text.split(":", 1)[0]
    text = text.strip().lower().removesuffix(".")
    if not text or any(c.isspace() for c in text):
        return ""
    return text


def _normalized_hosts(value: object) -> object:
    """Field validator body of the web-fetch host lists (mode="before"):
    accepts a JSON list, a comma/space/newline-separated string, drops
    junk entries and duplicates while keeping the user's order. Junk must
    never invalidate the WHOLE settings file (a typo there would silently
    reset every other preference — see `ProjectSettingsStore.load`)."""
    if isinstance(value, str):
        raw = value.replace(",", " ").replace("\n", " ").split()
    elif isinstance(value, list | tuple):
        raw = [str(item) for item in value]
    else:
        return value
    cleaned: list[str] = []
    for item in raw:
        host = normalize_host_entry(item)
        if host and host not in cleaned:
            cleaned.append(host)
    return cleaned


class EscAbort(str, enum.Enum):
    """How ESC stops a running agent/tester (project setting).

    DOUBLE_PRESS (the default): the first ESC only ARMS the stop — a toast
    asks for a confirmation press within a short window — and only the
    second ESC actually aborts. This guards against the accidental
    "closed a modal dialog with ESC, pressed ESC again out of inertia"
    scenario that silently stopped a working agent. INSTANT: the
    historical behaviour — a single ESC stops the run immediately.
    """

    DOUBLE_PRESS = "double_press"
    INSTANT = "instant"

    @property
    def label(self) -> str:
        """Human-readable option label for the Settings UI."""
        if self is EscAbort.INSTANT:
            return "instant (one ESC)"
        return "double-press (confirm)"

    @property
    def description(self) -> str:
        """One-line explanation shown in the Settings palette."""
        if self is EscAbort.INSTANT:
            return "A single ESC stops the running agent/tester immediately."
        return (
            "The first ESC only arms the stop (a toast asks for a second "
            "press within 2 seconds); the second ESC stops the run."
        )


ESC_ABORT_OPTIONS: tuple[EscAbort, ...] = (
    EscAbort.DOUBLE_PRESS,
    EscAbort.INSTANT,
)


class TaskIsolation(str, enum.Enum):
    """Where a task's development happens.

    OFF (the default): every task works directly in the project directory
    (the historical "everything in one tree" behaviour — an interrupted
    task leaves its changes behind for the next one). WORKTREE: each task
    gets its own git worktree under `.drcode/worktrees/<task-id>` on a
    dedicated branch `drcode/<task-id>` (see `drcode.task_env`), so an
    interrupted (unfinished) task never hands a dirty tree to the next
    task in the queue.

    WORKTREE needs a git repository in the project. The setting is still
    saved when the project has none — the app warns at every startup (and
    when the value is picked) that the mode stays dormant until a
    repository appears (or the setting is turned off again); without git
    tasks simply keep running in the project directory.
    """

    OFF = "off"
    WORKTREE = "worktree"

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is TaskIsolation.WORKTREE:
            return "worktree (git)"
        return "off (shared directory)"


ISOLATION_OPTIONS: tuple[TaskIsolation, ...] = (
    TaskIsolation.OFF,
    TaskIsolation.WORKTREE,
)


class WorktreeRemoval(str, enum.Enum):
    """When a task's worktree directory `.drcode/worktrees/<task-id>`
    (see `drcode.task_env`) is removed automatically (project setting).

    ON_DELETE (the default, the historical behaviour): the directory
    lives as long as the task — it is torn down on cancel (Ctrl+X),
    delete (Ctrl+Delete) and the retention cleanup. ON_COMPLETION: the
    directory is removed as soon as the task reaches COMPLETED
    (research right in `mark_done`, dev tasks at the tester's
    `set_verification` verdict — and the no-tester / review-accept
    completions alike); the branch `drcode/<task-id>` with its
    committed work always survives (RULE 2), and re-activating a
    completed task (new input) recreates the worktree from the branch.
    NEVER: the directory is never removed automatically — `remove()`
    then only snapshots a dirty tree onto the branch (RULE 1); dead
    tasks' leftover directories are still swept at startup
    (`cleanup_orphans` ignores this setting), otherwise the mode would
    grow the worktrees directory without bound.

    The setting governs ONLY the worktree directory — never the branch:
    a branch holding unmerged commits is never deleted automatically
    (see the `drcode.task_env` module docstring).
    """

    ON_DELETE = "on_delete"
    ON_COMPLETION = "on_completion"
    NEVER = "never"

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is WorktreeRemoval.ON_DELETE:
            return "on task deletion"
        if self is WorktreeRemoval.ON_COMPLETION:
            return "on task completion"
        return "never"


WORKTREE_REMOVAL_OPTIONS: tuple[WorktreeRemoval, ...] = (
    WorktreeRemoval.ON_DELETE,
    WorktreeRemoval.ON_COMPLETION,
    WorktreeRemoval.NEVER,
)


class WorktreeSnapshot(str, enum.Enum):
    """When the worktree of a settling task gets the automatic snapshot
    commit (`git add -A` + "drcode: snapshot of <task-id>") — RULE 1 of
    the worktree contract, project setting.

    ALWAYS (the historic behaviour): every transition out of active work
    commits whatever is dirty onto the branch `drcode/<id>` — pure
    insurance, but it also commits a tail the coding agent left
    uncommitted ON PURPOSE (WIP, drafts) when it already made its own
    commits, littering the branch history with a mechanical commit.
    ONLY_IF_NO_COMMITS (the default): the snapshot runs only when the
    branch tip is where the activation left it — no run-authored commits
    since the worktree was created, so the snapshot is still the only
    anchoring of the work. Once the run made its own commits, the dirty
    tail is presumed deliberate: the snapshot is skipped with a
    service-event note and the WORKTREE DIRECTORY IS KEPT on any removal
    path (deleting it would destroy the unanchored tail — never done
    automatically; commit or clean the worktree by hand).
    NEVER: the automatic snapshot is off everywhere except the last-resort
    protections that run immediately before a worktree is destroyed
    anyway (the startup orphan sweep) — there the commit is the only way
    to keep the work, and skipping it would contradict RULE 1 outright.

    The setting never touches the agents' own commit tools (`git_commit`
    of the tester, the research agent's docs commit) — those are governed
    by `git_policy`.
    """

    ALWAYS = "always"
    ONLY_IF_NO_COMMITS = "only-if-no-commits"
    NEVER = "never"

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is WorktreeSnapshot.ALWAYS:
            return "always"
        if self is WorktreeSnapshot.ONLY_IF_NO_COMMITS:
            return "only if no commits"
        return "never"


WORKTREE_SNAPSHOT_OPTIONS: tuple[WorktreeSnapshot, ...] = (
    WorktreeSnapshot.ALWAYS,
    WorktreeSnapshot.ONLY_IF_NO_COMMITS,
    WorktreeSnapshot.NEVER,
)


class StallNudge(str, enum.Enum):
    """How long an agent/tester run may stay without any sign of life
    (no stream deltas, no tool events) before it is automatically aborted
    and restarted ("nudged"). A stalled run blocks its task — and via the
    pipeline gate, often the whole backlog — so the watchdog keeps an
    overnight run going when the model silently hangs."""

    MIN_15 = "15"
    MIN_30 = "30"
    HOUR_1 = "60"
    HOUR_2 = "120"
    NEVER = "never"

    @property
    def minutes(self) -> int | None:
        """Stall limit in minutes, or None when runs are never nudged."""
        if self is StallNudge.NEVER:
            return None
        return int(self.value)

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        if self is StallNudge.NEVER:
            return "never"
        if self is StallNudge.HOUR_1:
            return "1 hour"
        if self is StallNudge.HOUR_2:
            return "2 hours"
        return f"{self.value} min"


STALL_NUDGE_OPTIONS: tuple[StallNudge, ...] = (
    StallNudge.MIN_15,
    StallNudge.MIN_30,
    StallNudge.HOUR_1,
    StallNudge.HOUR_2,
    StallNudge.NEVER,
)


class GitHubSyncInterval(str, enum.Enum):
    """How often the GitHub Issues sync runs (project setting)."""

    HOUR_1 = "1"
    HOUR_2 = "2"
    HOUR_4 = "4"

    @property
    def hours(self) -> int:
        return int(self.value)

    @property
    def label(self) -> str:
        """Human-readable option label for the settings UI."""
        return f"{self.value} hour" + ("s" if self.hours > 1 else "")


GITHUB_SYNC_INTERVAL_OPTIONS: tuple[GitHubSyncInterval, ...] = (
    GitHubSyncInterval.HOUR_1,
    GitHubSyncInterval.HOUR_2,
    GitHubSyncInterval.HOUR_4,
)


class ProjectSettings(BaseModel):
    completed_retention: CompletedRetention = CompletedRetention.DAYS_7
    # Whether the QA tester round runs at all (Verification docstring).
    verification: Verification = Verification.TESTER
    # Per-role git policies (GitPolicySetting docstring): what the tester
    # and the research agent may do to git — nothing / local commit /
    # commit + push. The tester slot is dormant while verification is NONE.
    git_policy: GitPolicySetting = GitPolicySetting()
    # Fix-loop cap: how many consecutive development rounds the tester may
    # fail with a "the bug is this task's own change" verdict (retry_here)
    # before the failure completes the task as FAILED and the bug report is
    # created (branching from clean HEAD). Reset by a PASSED verdict or by
    # the user taking the task back by hand.
    max_dev_rounds: int = 3
    stall_nudge: StallNudge = StallNudge.MIN_30
    # Per-task worktree isolation (TaskIsolation docstring): a value that
    # the project cannot currently honour (worktree without a git repo) is
    # still persisted — the app warns about the dormant mode at every
    # startup instead of silently rewriting the user's choice.
    task_isolation: TaskIsolation = TaskIsolation.OFF
    # How many tasks may develop in parallel (development slots). Only
    # effective under worktree isolation — each parallel task needs its
    # own worktree, so without it the effective limit is 1 (the value is
    # still persisted; the app warns and behaves as 1). The tester stays
    # single: testing never occupies a development slot and never runs
    # concurrently with another tester round.
    parallel_tasks: int = 1
    # How many read-only research subagents the CODING agent may run in
    # parallel at once (spawned via its delegate_research tool; each
    # returns only its final answer, so the parent's context stays small).
    # 0 (the default) = delegation is off: the tool is not registered and
    # the agent neither knows about nor can use subagents.
    subagents_coder: int = 0
    # The same cap for the RESEARCH agent's runs (its resolved model
    # serves the subagents; the counter is per parent run/generation).
    subagents_researcher: int = 0
    # Web access of the RESEARCH agent (WebFetchMode docstring): off (no
    # tool), allowlist (only web_fetch_allow_hosts + their subdomains) or
    # all (every public host). The structured `web_fetch` tool is the only
    # way an agent reads external pages: it is GET-only, SSRF-guarded
    # (private/loopback/link-local/metadata addresses are refused), size-
    # and time-capped, and every result is framed as untrusted content.
    # run_bash is local-only regardless of this mode — the bash network
    # guard refuses external requests of curl/wget/http/https/ftp even
    # here, where the tool itself is unavailable. Only the research role
    # gets the tool — the coding agent, the tester and the resolver never
    # do.
    web_fetch_mode: WebFetchMode = WebFetchMode.OFF
    # Hosts the research agent may fetch in allowlist mode (allowlist mode
    # MEANS an entry here — with an empty list every fetch is refused with
    # a hint to configure the hosts). Entries are normalized
    # (normalize_host_entry): a bare host or a URL is accepted; a listed
    # host also matches its subdomains.
    web_fetch_allow_hosts: list[str] = Field(default_factory=list)
    # Hosts the research agent must NEVER fetch (wins over the allow list
    # and over the "all hosts" mode). The private-network / metadata-address
    # guard of `drcode.web_fetch` applies in every mode regardless.
    web_fetch_block_hosts: list[str] = Field(default_factory=list)
    # Agent-created task drafts: when on, the agents' task-creation tools
    # (research: create_bug_report / create_feature_request, tester:
    # create_bug_report) do NOT queue the new task — it is created in the
    # DRAFT status and waits until the user opens it in the new-task
    # editor and finishes it (into the queue or on hold). The headless
    # CLI (drcode --bug-report/--feature-request) and the GitHub Issues
    # import are deliberately NOT affected: they are already driven by an
    # explicit human request. The tester's fix-loop end (a FAILED task
    # after max_dev_rounds) also runs through the same tool — its bug
    # report becomes a draft too, which does not disturb the loop itself
    # (the failed task has already completed by then).
    agent_creates_drafts: bool = False
    # Agent-generated task titles: when on, the FIRST activation of a task
    # asks the model serving the task (the same role config the run itself
    # uses) to write a short title from the task's description — before the
    # main run starts, so the user sees the new name on the active task
    # from the very beginning. Off (the default) keeps the historical
    # first-line-of-the-prompt title untouched. Generation is exactly-once
    # per task lifetime (the `Task.title_generated` stamp, set on success
    # only): a re-activation never re-generates, a failed one keeps the old
    # title and lets the NEXT activation retry. Explicit author titles
    # (`Task.title_explicit` — the agents' report tools, the GitHub import,
    # a goal's title) are skipped entirely. Tasks that never reach the
    # agent (hold / draft / queued) keep the derived title — the generator
    # runs only in the agent-run worker.
    generate_task_titles: bool = False
    # Context-eviction hint (advisory nudge about mark_unneeded): when on,
    # the coding agent gets a dynamic instruction that, once the live
    # context fill passes ~half the model window, reminds the model of the
    # mark_unneeded tool with the real figures (see
    # agent_tools.eviction_dynamic_instructions). Off (the default): the
    # instruction is not registered at all — nothing rides the requests
    # and nothing scans the transcript. Off by default because the nudge
    # is useful only on some setups (e.g. llama.cpp prefix-cache-friendly
    # frozen texts, no big old tool returns); leaving it out keeps the
    # request framing minimal everywhere else.
    eviction_hint: bool = False
    # Live todo tail (agent_model.TodoTailModel): when on, every model
    # request of a feature-task run carries a synthetic TAIL message with
    # the task's current checklist state (recomputed per request, see
    # agent_build.build_agent). Off (the default): the model is NOT wrapped
    # at all — no request modification whatsoever; the checklist still
    # lives in task.todos and is visible in the TUI, the model just never
    # receives it. Opt-in because the tail changes the request framing and
    # costs tokens on every request, which some setups don't want. Read at
    # activation, so a toggle applies to the next activation of each task.
    todo_tail: bool = False
    # When the per-task worktree directory (.drcode/worktrees/<id>) is
    # removed automatically (WorktreeRemoval docstring). Only the
    # directory is governed: the branch `drcode/<id>` lives by RULE 2,
    # whatever this says.
    worktree_removal: WorktreeRemoval = WorktreeRemoval.ON_DELETE
    # When the settling task's worktree gets the automatic RULE-1 snapshot
    # commit (WorktreeSnapshot docstring). only-if-no-commits skips it when
    # the run made its own commits on the branch (a deliberately uncommitted
    # tail is then kept in the worktree, never auto-removed); never disables
    # it except before an actual worktree destruction.
    worktree_snapshot: WorktreeSnapshot = WorktreeSnapshot.ONLY_IF_NO_COMMITS
    # Start point of a NEW task branch `drcode/<id>` under worktree
    # isolation (see drcode.task_env): empty (the default) = the branch
    # the main checkout's HEAD is on — and a detached HEAD (mid-rebase,
    # mid-checkout) refuses the isolation for that activation instead of
    # forking from an unknown commit. A non-empty value = a named local
    # branch (e.g. "main"): every task branch is forked from THAT branch's
    # tip, even when the main checkout is detached or on another branch.
    # A configured branch that does not exist degrades to the auto mode
    # with a warning in the task's service-event log — never a crash.
    isolation_start_branch: str = ""
    # How ESC stops a running agent/tester (EscAbort docstring): the safe
    # double-press default guards the "closed a dialog with ESC, pressed
    # ESC once more by inertia" accident; "instant" restores the
    # historical one-press stop.
    esc_abort: EscAbort = EscAbort.DOUBLE_PRESS
    # Keep the computer awake (a system-sleep assertion) while the pipeline
    # has work: any task QUEUED/ACTIVE/TESTING or a run in flight. macOS
    # runs `caffeinate -i -s`, Windows holds SetThreadExecutionState, Linux
    # is a no-op. A sleeping machine tears the network down and kills every
    # concurrent run with ModelConnectionError — the retry budget does not
    # survive a full sleep. Does NOT stop the clamshell (lid closed on
    # battery) sleep mode.
    keep_awake: bool = True
    # -- per-role max_tokens overrides ----------------------------------------
    # Per-role output-budget overrides: keys are AgentRole values
    # ("coding" / "research" / "tester" / "compaction" / "resolver"),
    # values are positive ints (the role's per-response max_tokens).
    # Absent / empty = inherit (the catalog entry's max_tokens, else the
    # flat config default / DRCODE_MAX_TOKENS). The most specific layer —
    # the role — beats the more general ones (providers.resolve applies
    # it last): a thinking model burns the default 8192 budget on
    # reasoning alone, so the tester/resolver roles commonly need 16–32k.
    role_max_tokens: dict[str, int] = Field(default_factory=dict)

    # -- GitHub Issues integration ------------------------------------------
    # Enabled may be switched on even with no token/repo set (or when the
    # connection check fails): the sync simply logs an error each cycle
    # until the values are fixed.
    github_enabled: bool = False
    # Personal access token (fine-grained or classic, needs repo read access).
    github_token: str = ""
    # Full repository name "owner/repo" (a pasted URL is normalized on save).
    github_repo: str = ""
    github_sync_interval: GitHubSyncInterval = GitHubSyncInterval.HOUR_1
    # -- ntfy.sh push notifications -------------------------------------------
    # Enabled may be switched on with no topic set: the notifier simply
    # stays silent (and the settings dialog warns about the missing topic)
    # until a topic is provided.
    notifications_enabled: bool = False
    # Public ntfy.sh topic name to publish to (see https://ntfy.sh). Anyone
    # who knows the topic can read its messages — pick an unguessable one.
    ntfy_topic: str = ""
    # -- Voice I/O (ASR dictation + TTS announcements) -----------------------
    # OpenAI-compatible audio API endpoint (WITH the /v1 suffix; the audio
    # routes are /audio/transcriptions and /audio/speech below it).
    # Deliberately NOT the chat provider's base_url: gateways commonly serve
    # audio on a different route than the chat models. Empty (the default)
    # = every voice feature is off.
    voice_api_url: str = ""
    # Bearer token for the audio API. Empty = reuse the api_key of the
    # user-catalog provider whose base_url host matches voice_api_url's
    # host (no second token to manage for the common single-gateway case).
    voice_api_key: str = ""
    # Full ASR model id, e.g. "gigaam_rnnt_v3_e2e". Model ids may contain
    # "/" (unlike the agent-catalog ids), so they are stored here directly
    # instead of the model catalog.
    voice_asr_model: str = ""
    # TTS model id (e.g. "higgs-tts-3-4b") and the speaker voice (e.g.
    # "alloy"); both required for the announcements.
    voice_tts_model: str = ""
    voice_tts_voice: str = ""
    # Which events the TTS announces (each independent):
    voice_tts_completed: bool = False
    voice_tts_tester_failed: bool = False
    voice_tts_queue_empty: bool = False
    voice_tts_clarification: bool = False
    # Language of the spoken announcement phrases: "ru" or "en". Empty =
    # auto-derive from the terminal locale (LC_ALL/LC_MESSAGES/LANG, falling
    # back to the Python locale; anything non-Russian reads as English).
    voice_language: str = ""
    # Pre-run compaction of a cold-resume transcript (project setting):
    # a THRESHOLD IN TOKENS (not a percent). When a task is re-activated
    # after its previous run ended ≥ 3 minutes ago (retry_here after the
    # tester, out of awaiting-clarification, an unfinished resume — any
    # path where the provider's prompt cache has long burned out) and the
    # estimated context fill of its dev/research transcript EXCEEDS this
    # value, the transcript is force-compacted BEFORE the first turn, so
    # the cold re-read happens over a compact history instead of the full
    # one (see core/pipeline_compaction._maybe_compact_after_idle).
    # Values below 20000 (including the 0 default) keep the feature off —
    # a threshold low enough to fire on healthy sessions would compact
    # transcripts the ordinary threshold gate never touches, trading
    # cache warmth for summarizer calls for nothing.
    compaction_resume_min_tokens: int = 0

    @field_validator("role_max_tokens", mode="before")
    @classmethod
    def _clean_role_max_tokens(cls, value: object) -> object:
        """Drop entries that are not a valid role key or not a positive
        int (a stray key or a typo in settings.json must not invalidate
        the WHOLE file — the wholesale default fallback of load() is
        reserved for structural corruption)."""
        if not isinstance(value, dict):
            return value
        return {
            key: int(v)
            for key, v in value.items()
            if key in _ROLE_MAX_TOKENS_KEYS
            and isinstance(v, int | float)
            and not isinstance(v, bool)
            and int(v) > 0
        }

    @field_validator("web_fetch_allow_hosts", "web_fetch_block_hosts", mode="before")
    @classmethod
    def _clean_host_lists(cls, value: object) -> object:
        """Normalize the web-fetch host lists (`_normalized_hosts`): a
        hand-edited settings.json may carry a comma-separated string, a
        pasted URL or junk — none of it may invalidate the WHOLE file."""
        return _normalized_hosts(value)

    @field_validator("compaction_resume_min_tokens", mode="before")
    @classmethod
    def _clean_compaction_resume_min_tokens(cls, value: object) -> object:
        """Coerce junk to the disabled default instead of invalidating the
        WHOLE settings file (a typo in settings.json must never strip the
        other preferences — the wholesale default fallback of load() is
        reserved for structural corruption). Negative values clamp to 0
        (off); anything below 20000 stays persisted but means off."""
        if isinstance(value, bool) or not isinstance(value, int | float):
            return 0
        return max(int(value), 0)


# Valid keys of ProjectSettings.role_max_tokens: the AgentRole values
# (duplicated here as literals — settings.py must not import providers.py,
# which reads this module).
_ROLE_MAX_TOKENS_KEYS: frozenset[str] = frozenset(
    {"coding", "research", "tester", "compaction", "resolver"}
)


@dataclass
class SettingsHint:
    """One live hint about the current task-settings combination, shown
    in the Task-settings dialog (and reusable by the startup warnings).

    severity:
    - "dormant" — a persisted value that is currently INERT: some other
      setting (the governing field) keeps it from having any effect. The
      user's choice is parked, never rewritten.
    - "conflict" — a combination that makes no sense at all (it cannot
      work here even in principle, e.g. worktrees without a git repo).
    - "info" — everything else worth a line.

    `fix` is a mutation of a ProjectSettings draft that resolves the
    hint. It runs ONLY when the user presses the matching fix button in
    the dialog — never automatically (the settings philosophy: the
    user's choice is never silently rewritten).
    """

    severity: Literal["dormant", "conflict", "info"]
    text: str
    fix: Callable[[ProjectSettings], None] | None = None
    # Button label for the fix (shown only when fix is set).
    fix_label: str = ""


def task_settings_hints(
    draft: ProjectSettings, *, git_available: bool
) -> list[SettingsHint]:
    """The single source of the coupling rules between the task-management
    settings — every combination the UI/startup warnings must tell the
    user about, in one function, so the dialog and future warning wiring
    can never diverge.

    Pure: reads `draft` (and the git availability of the project), never
    mutates it; fixes are separate callables the user opts into.
    """
    hints: list[SettingsHint] = []
    isolated = draft.task_isolation is TaskIsolation.WORKTREE

    # -- parallel slots need worktree isolation -------------------------------
    if draft.parallel_tasks > 1 and not isolated:
        hints.append(
            SettingsHint(
                severity="dormant",
                text=(
                    f"Parallel tasks: {parallel_tasks_label(draft.parallel_tasks)} "
                    "is dormant without worktree isolation — every parallel "
                    "task needs its own git worktree, so the effective limit "
                    "is 1."
                ),
                fix=lambda s: setattr(s, "task_isolation", TaskIsolation.WORKTREE),
                fix_label="Enable isolation",
            )
        )

    # -- worktree mode without a git repository -------------------------------
    if isolated and not git_available:
        hints.append(
            SettingsHint(
                severity="conflict",
                text=(
                    "Task isolation: worktree cannot work — this project has "
                    "no git repository, so tasks keep running in the shared "
                    "project directory. Init git (git init) or set isolation "
                    "to off."
                ),
            )
        )

    # -- the tester's git policy is pointless without a tester ----------------
    if draft.verification is Verification.NONE and (
        draft.git_policy.tester is not GitPolicy.OFF
    ):
        hints.append(
            SettingsHint(
                severity="dormant",
                text=(
                    f"Git (tester): {draft.git_policy.tester.label} is dormant "
                    "— verification is none, no tester runs to commit "
                    "anything."
                ),
                fix=lambda s: setattr(s.git_policy, "tester", GitPolicy.OFF),
                fix_label="Set Git (tester): off",
            )
        )

    # -- web fetch of the research agent --------------------------------------
    if (
        draft.web_fetch_mode is WebFetchMode.ALLOWLIST
        and not draft.web_fetch_allow_hosts
    ):
        hints.append(
            SettingsHint(
                severity="conflict",
                text=(
                    "Web fetch: allowlist mode with an EMPTY allow list — "
                    "every web_fetch call of the research agent is refused "
                    "until at least one host is allowed."
                ),
                fix=lambda s: setattr(s, "web_fetch_mode", WebFetchMode.ALL),
                fix_label="Allow all hosts",
            )
        )
    if (
        draft.web_fetch_mode is not WebFetchMode.OFF
        and draft.web_fetch_allow_hosts
        and draft.web_fetch_mode is WebFetchMode.ALL
    ):
        hints.append(
            SettingsHint(
                severity="dormant",
                text=(
                    "Web fetch: the allow list is unused while the mode is "
                    f"{WebFetchMode.ALL.label} — every public host is "
                    "fetchable; the list takes effect in allowlist mode."
                ),
            )
        )
    if draft.web_fetch_mode is WebFetchMode.OFF and (
        draft.web_fetch_allow_hosts or draft.web_fetch_block_hosts
    ):
        hints.append(
            SettingsHint(
                severity="dormant",
                text=(
                    "Web fetch: the host lists are parked — web fetch is off, "
                    "so the research agent has no web tool at all."
                ),
            )
        )

    # -- parked fields under isolation off ------------------------------------
    if not isolated and draft.isolation_start_branch:
        hints.append(
            SettingsHint(
                severity="dormant",
                text=(
                    f"Isolation start branch: {draft.isolation_start_branch} "
                    "is parked — it governs which branch NEW task branches "
                    "are forked from, which only exists under worktree "
                    "isolation."
                ),
            )
        )
    if not isolated and draft.worktree_removal is not WorktreeRemoval.ON_DELETE:
        hints.append(
            SettingsHint(
                severity="dormant",
                text=(
                    f"Worktree removal: {draft.worktree_removal.label} is "
                    "parked — there are no task worktrees without "
                    "isolation."
                ),
            )
        )
    if (
        not isolated
        and draft.worktree_snapshot is not WorktreeSnapshot.ONLY_IF_NO_COMMITS
    ):
        hints.append(
            SettingsHint(
                severity="dormant",
                text=(
                    f"Worktree snapshot: {draft.worktree_snapshot.label} is "
                    "parked — the snapshot anchors a task worktree, and "
                    "there are none without isolation."
                ),
            )
        )

    return hints


def _migrate_legacy_finalization(data: dict) -> dict:
    """Map the legacy flat `task_finalization` key onto the orthogonal
    `verification` + `git_policy` fields (in place; the legacy key is
    dropped so the next save writes only the new shape).

    New keys win when both are present (a settings file already saved by
    the new code carries no legacy key anyway). The research policy is
    not part of the legacy setting and stays at its default.
    """
    legacy = data.pop("task_finalization", None)
    if legacy is None:
        return data
    mapping = _LEGACY_FINALIZATION_MAP.get(legacy)
    if mapping is None:
        # Unknown legacy value: leave the new fields untouched (a stale
        # key must not corrupt the current settings).
        return data
    verification, tester_policy = mapping
    if "verification" not in data:
        data["verification"] = verification.value
    if "git_policy" not in data:
        data["git_policy"] = {"tester": tester_policy.value}
    return data


class ProjectSettingsStore:
    """Loads/saves project settings from `<project>/.drcode/settings.json`."""

    def __init__(self, project_root: Path) -> None:
        self.path = project_root / ".drcode" / "settings.json"

    def load(self) -> ProjectSettings:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return ProjectSettings()
        except (OSError, json.JSONDecodeError):
            # Corrupt settings must never keep the app from starting.
            return ProjectSettings()
        if isinstance(data, dict):
            data = _migrate_legacy_finalization(data)
        try:
            return ProjectSettings.model_validate(data)
        except ValidationError:
            return ProjectSettings()

    def save(self, settings: ProjectSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(settings.model_dump_json(indent=2))
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
