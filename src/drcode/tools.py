"""Filesystem and shell tools for the coding agent.

Permission model:
- reading (read_file) works anywhere on the filesystem;
- writing (write_file / edit_file) is allowed only inside the project root
  (with all subdirectories) and inside the task's personal scratch directory
  /tmp/drcode/<task-id> — /tmp outside that directory is NOT writable, so two
  tasks cannot interfere through shared scratch files;
- run_bash inspects the command and refuses rm/mv/cp whose path arguments
  escape that writable area, points the child process's TMPDIR at the
  task's scratch directory, and exports DRCODE_TASK_ID / DRCODE_WORKTREE
  so a parallel task's child processes can pick per-task resources
  (ports, caches) — see the parallel development slots. It also redirects
  the child's drcode user config into a per-task sandbox via
  DRCODE_USER_CONFIG_DIR: diagnostic python snippets that construct
  default-constructed stores (`UserModelsStore()`, `Config()`) then read
  and write the sandbox copy, never the real `~/.config/drcode` (incident
  ebe6ba15: such a snippet wiped the user's model catalog);
- run_bash is also LOCAL-ONLY on the network: a shell command may talk to
  the loopback interface (localhost, 127.0.0.0/8, ::1 — a dev server or a
  test fixture started inside the project), while an outbound request of a
  network client (curl, wget, http/https, ftp, and any redirect-following
  flag) is refused. External network access belongs to the managed
  `web_fetch` tool — see the "bash network guard" section below for the
  exact scope (interpreters, package managers and git remotes are not
  analysed: this is a policy guard, not an OS-level network sandbox).

Every refusal raises PathViolation with a model-friendly message; the agent
layer logs it into the task's service-event history before returning it to
the model as a retry prompt.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import config as _config
from .config import PROJECT_STATE_DIRNAME

MAX_TOOL_OUTPUT = 10_000
BASH_TIMEOUT = 120
# Hard cap for a run_bash call: the agent may request a longer timeout than
# the default, but never more than an hour — values above are clamped to it.
BASH_MAX_TIMEOUT = 60 * 60
# Per-result cap for a read_file result: a read chunk larger than this is not
# refused but returned truncated (head + tail) with a note. The cap keeps one
# tool result well inside a single turn's growth budget relative to the
# compaction threshold — ~25k chars ≈ ~6k tokens, so one or two big reads can
# no longer push the context from below the threshold past 100% before the
# turn-boundary compaction gate fires. The model always gets a usable slice
# plus instructions on how to read the rest via `offset`/`limit`.
MAX_READ_RESULT_BYTES = 25_000
# Shares of the truncation budget given to the head and the tail slices.
_READ_HEAD_SHARE = 0.75
# Bytes reserved for the truncation note, so the WHOLE result (slices + note)
# stays within MAX_READ_RESULT_BYTES.
_READ_NOTE_RESERVE = 256
# Nice increment for spawned background children (see child_priority_kwargs).
_CHILD_NICE = 10
# Defaults for the context-budget check (overridden per run by the caller):
# a fresh agent starts with empty context, and the default compaction threshold
# matches Config.compaction_threshold (80).
DEFAULT_CURRENT_USAGE = 0
DEFAULT_MAX_CONTEXT_TOKENS = 200_000
DEFAULT_COMPACTION_THRESHOLD = 80

# Extra scratch area: every task gets a PERSONAL scratch directory
# /tmp/drcode/<task-id> — the model may create/edit files there even though
# it is outside the project, a sandbox for testing hypotheses. Directories
# are created lazily on the first write and removed when the task completes
# or is deleted. /tmp outside the task's own scratch directory is NOT
# writable: a shared /tmp would let two tasks interfere through scratch
# files, and closing that hole needs no OS-level sandbox — only a narrower
# writable area.
#
# /tmp/drcode is GLOBAL while task ids are only unique per project, so every
# scratch directory carries an owner marker (SCRATCH_OWNER_FILE below) with
# the owning process's pid: the startup orphan sweep and the completion-time
# cleanup must never touch a directory whose owner is another LIVE drcode
# instance.
SCRATCH_ROOT = Path("/tmp") / "drcode"


def scratch_dir(task_id: str) -> Path:
    """The per-task scratch directory under /tmp/drcode (not created here:
    it appears lazily on the first write or run_bash call)."""
    return SCRATCH_ROOT / task_id


IGNORED_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    ".drcode",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
}


class PathViolation(Exception):
    """A tool tried to touch a path outside the allowed area."""


def _writable_hint(root: Path, extra_writable: Path | None) -> str:
    scratch = (
        f", and your scratch directory '{extra_writable}'"
        if extra_writable is not None
        else ""
    )
    tip = (
        " (tip: the scratch directory is yours alone — test hypotheses there; "
        "it is removed when the task completes)"
        if extra_writable is not None
        else ""
    )
    return (
        f"Allowed locations: the project root ('{root}') with all its "
        f"subdirectories{scratch}{tip}."
    )


def _write_violation(
    root: Path, path: str, action: str, extra_writable: Path | None
) -> PathViolation:
    return PathViolation(
        f"Permission denied: cannot {action} '{path}' — it is outside the "
        f"allowed locations. {_writable_hint(root, extra_writable)}"
    )


def _bash_violation(
    root: Path,
    command: str,
    cmd: str,
    path: str | None,
    extra_writable: Path | None,
) -> PathViolation:
    target = f" with path '{path}'" if path else ""
    return PathViolation(
        f"Command refused: '{command}': {cmd}{target} would touch files outside "
        f"the allowed locations. {_writable_hint(root, extra_writable)}"
    )


def _stash_violation(command: str) -> PathViolation:
    return PathViolation(
        f"Command refused: '{command}': 'git stash' (any subcommand except list) "
        f"touches the shared stash stack (refs/stash), which is common to ALL "
        f"worktrees of the repository — parallel drcode tasks would push and pop "
        f"each other's stashed changes. To run a check on a clean tree, create a "
        f"throwaway worktree instead: `git worktree add <tmp-dir> HEAD && "
        f"cd <tmp-dir> && <check>` — a worktree has its own tree/index, so there "
        f"is no shared state to race on. Place <tmp-dir> inside your scratch "
        f"directory."
    )


def _state_dir_violation(command: str, cmd: str, path: str | None) -> PathViolation:
    target = f" with path '{path}'" if path else ""
    return PathViolation(
        f"Command refused: '{command}': {cmd}{target} would touch the project "
        f"state directory '{PROJECT_STATE_DIRNAME}/' — it holds drcode's task "
        f"history, settings and worktrees, is not recoverable (it is gitignored), "
        f"and must not be deleted, moved or overwritten."
    )


def resolve_read_path(root: Path, path: str) -> Path:
    """Reading is unrestricted: any path on the filesystem works.
    Relative paths are resolved against the project root."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def is_writable(root: Path, p: Path, extra_writable: Path | None = None) -> bool:
    # `p` is always resolved (resolve_edit_path / the command guard resolve
    # first), but `root` may arrive relative or through a symlink — accept
    # both the literal and the resolved location.
    for r in {root, root.resolve()}:
        if p == r or r in p.parents:
            return True
    if extra_writable is not None:
        # /tmp may be a symlink (macOS: /tmp -> /private/tmp) — accept both
        # the literal and the resolved location.
        for t in {extra_writable, extra_writable.resolve()}:
            if p == t or t in p.parents:
                return True
    return False


def resolve_edit_path(
    root: Path, path: str, action: str = "write to", *, extra_writable: Path
) -> Path:
    """Resolve `path` for a modifying operation; refuse anything outside
    the project root and the task's scratch directory `extra_writable`
    (symlinks are resolved first, so a symlink inside the project pointing
    outside is refused too)."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if is_writable(root, p, extra_writable):
        return p
    raise _write_violation(root, path, action, extra_writable)


def _fmt_bytes(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}".rstrip("0").rstrip(".") + "Kb"
    return f"{n}b"


def _token_estimate(text_bytes: int) -> int:
    """Rough chars→tokens heuristic (~4 bytes/token, the common rule of thumb
    for OpenAI-compatible models). Deliberately imprecise: it only warns about
    oversized reads, it never counts exactly."""
    return max(1, text_bytes // 4)


def _truncate_read(numbered: list[str]) -> str:
    """Cut an oversized read chunk to a head+tail slice within
    MAX_READ_RESULT_BYTES, keeping the original line numbers intact.

    The head gets `_READ_HEAD_SHARE` of the budget (the file's start is what
    the model usually navigates from), the tail the rest (the end often holds
    the current state). The middle is replaced by a note carrying the omitted
    span and the `offset`/`limit` guidance from the old refusal message."""
    total_bytes = sum(len(line.encode("utf-8")) + 1 for line in numbered)
    content_budget = MAX_READ_RESULT_BYTES - _READ_NOTE_RESERVE
    head_budget = int(content_budget * _READ_HEAD_SHARE)
    tail_budget = content_budget - head_budget

    head: list[str] = []
    taken = 0
    cut = len(numbered)
    for i, line in enumerate(numbered):
        size = len(line.encode("utf-8")) + 1
        if taken + size > head_budget:
            cut = i
            break
        head.append(line)
        taken += size

    tail: list[str] = []
    taken = 0
    tail_start = len(numbered)
    for i in range(len(numbered) - 1, cut - 1, -1):
        size = len(numbered[i].encode("utf-8")) + 1
        if taken + size > tail_budget:
            tail_start = i + 1
            break
        tail.append(numbered[i])
        taken += size
    tail.reverse()

    if not head and not tail:
        # Pathological chunk (single line(s) bigger than the budgets): line
        # slicing yields nothing, so fall back to a raw character head+tail —
        # the model still gets a usable slice, just without line numbers.
        enc = "\n".join(numbered).encode("utf-8")
        head_text = enc[:head_budget].decode("utf-8", errors="ignore")
        tail_text = enc[-tail_budget:].decode("utf-8", errors="ignore")
        note = (
            f"[output truncated: raw head+tail of {_fmt_bytes(len(enc))} — "
            f"read the file in smaller pieces by adjusting `offset` and `limit`]"
        )
        return f"{head_text}\n{note}\n{tail_text}"

    # total_bytes > MAX_READ_RESULT_BYTES >= head+tail, so something is always
    # omitted — the note is never a lie.
    omitted = tail_start - cut
    note = (
        f"\n[output truncated: {omitted} of {len(numbered)} lines omitted "
        f"({_fmt_bytes(total_bytes)} total) — read the file in smaller pieces "
        f"by adjusting `offset` and `limit`]"
    )
    return "\n".join(head + tail) + note


def read_file(
    root: Path,
    path: str,
    offset: int = 1,
    limit: int = 2000,
    *,
    current_usage: int = DEFAULT_CURRENT_USAGE,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    compaction_threshold: int = DEFAULT_COMPACTION_THRESHOLD,
) -> str:
    """Read a text file from anywhere on the filesystem (relative paths resolve
    against the project root). Lines are numbered, starting from `offset`
    (1-based), up to `limit` lines.

    A read that risks blowing up the model context is truncated, never refused:
    a single result larger than MAX_READ_RESULT_BYTES is cut to a head+tail
    slice with an explicit note (read the file in smaller pieces via
    `offset`/`limit`), and a read that would still push the context fill past
    the compaction threshold is refused — the token estimate of the (possibly
    truncated) piece is checked against the current context usage and the
    current compaction threshold (default 80% of the context window)."""
    p = resolve_read_path(root, path)
    lines = p.read_text(errors="replace").splitlines()
    start = max(offset - 1, 0)
    chunk = lines[start : start + limit]
    numbered = [f"{start + i + 1}: {line}" for i, line in enumerate(chunk)]
    text = "\n".join(numbered)
    chunk_bytes = len(text.encode("utf-8"))
    # Hard cap: a single read result larger than MAX_READ_RESULT_BYTES is
    # truncated to a head+tail slice — the model keeps a usable piece of the
    # file plus instructions on how to read the rest.
    if chunk_bytes > MAX_READ_RESULT_BYTES:
        text = _truncate_read(numbered)
        chunk_bytes = len(text.encode("utf-8"))
    # Context-budget check: does this read risk exceeding the compaction
    # threshold? `current_usage` is the tokens the model already holds.
    threshold = min(max(compaction_threshold, 1), 100)
    if (
        max_context_tokens > 0
        and current_usage + _token_estimate(chunk_bytes)
        > threshold * max_context_tokens // 100
    ):
        # The refusal stays neutral on purpose: exposing the absolute token
        # count, the current fill and the compaction threshold would feed the
        # model a "context is nearly full" signal it then worries about. The
        # gate still protects the window — the read is refused — but the message
        # only tells the model *that* the budget is taken, not *how much*.
        return (
            f"Refusing to read '{path}': this read would exceed the model's "
            f"current context budget. Read the file in smaller pieces by lowering "
            f"`limit` or raising `offset`, then retry."
        )
    return text or "(empty file)"


def write_file(root: Path, path: str, content: str, *, extra_writable: Path) -> str:
    p = resolve_edit_path(root, path, "create/overwrite", extra_writable=extra_writable)
    # Also creates the scratch directory lazily on the first write into it.
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    _stamp_owner_if_scratch(extra_writable, p)
    return f"written {len(content)} chars to {path}"


def edit_file(
    root: Path, path: str, old_string: str, new_string: str, *, extra_writable: Path
) -> str:
    p = resolve_edit_path(root, path, "edit", extra_writable=extra_writable)
    text = p.read_text(errors="replace")
    count = text.count(old_string)
    if count == 0:
        raise ValueError("old_string not found in file")
    if count > 1:
        raise ValueError(
            f"old_string matches {count} locations, provide more context to make it unique"
        )
    p.write_text(text.replace(old_string, new_string, 1))
    _stamp_owner_if_scratch(extra_writable, p)
    return f"edited {path}"


def glob_files(root: Path, pattern: str) -> list[str]:
    matches = [p for p in root.glob(pattern) if p.is_file()]
    return sorted(str(p.relative_to(root)) for p in matches)[:200]


def _include_matcher(glob: str) -> re.Pattern[str]:
    """Translate a glob mask into a regex over the project-relative posix path.

    `*` matches any characters including directory separators, so '*.py'
    selects files at any depth; '**/' optionally spans directories, so
    'src/**/*.py' selects every file under src/ at any depth. A glob cannot
    produce an invalid regex (metacharacters are escaped), unlike a raw one.
    """
    out: list[str] = []
    i = 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out.append(r"(?:.*/)?")
            i += 3
        elif glob[i] == "*":
            out.append(".*")
            i += 1
        elif glob[i] == "?":
            out.append(".")
            i += 1
        else:
            out.append(re.escape(glob[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def grep_search(root: Path, pattern: str, include: str = "") -> list[str]:
    regex = re.compile(pattern)
    include_re = _include_matcher(include) if include else None
    results: list[str] = []
    for dirpath, dirnames, filenames in root.walk():
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for name in filenames:
            f = Path(dirpath) / name
            # `include` is a glob mask over the project-relative path (models
            # naturally send "*.py" / "src/**/*.py", not regexes).
            if include_re and not include_re.match(f.relative_to(root).as_posix()):
                continue
            try:
                for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                    if regex.search(line):
                        results.append(
                            f"{f.relative_to(root)}:{i}: {line.strip()[:200]}"
                        )
                        if len(results) >= 100:
                            return results
            except OSError:
                continue
    return results


# -- child-process environment sanitation ------------------------------------

# Environment variables that leak the shell's "foreign project" context into
# child processes: when drcode is launched from a shell with another project's
# venv/conda activated, these point outside the current project. They make
# tools like `uv` print foreign paths into tool-returns and — worst case — let
# `uv run --active` delete and recreate the foreign venv. They are stripped
# from the environment of every subprocess the agent spawns. Everything else
# (PATH, HOME, …) is passed through unchanged, except DRCODE_TASK_ID /
# DRCODE_WORKTREE: run_bash manages those itself (exported when a task id is
# given, removed otherwise — a task-less child must never see a foreign id).
STRIPPED_ENV_VARS = (
    "VIRTUAL_ENV",
    "PYTHONHOME",
    "PYTHONPATH",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
)


def sanitized_env() -> dict[str, str]:
    """os.environ without variables identifying a foreign project's env."""
    env = dict(os.environ)
    for var in STRIPPED_ENV_VARS:
        env.pop(var, None)
    return env


# -- run_bash command inspection ---------------------------------------------

# Destructive/overwriting commands whose path arguments must stay inside the
# writable area (project root + /tmp).
DESTRUCTIVE_COMMANDS = {"rm", "mv", "cp"}
# Common wrappers that may prefix the actual command.
COMMAND_WRAPPERS = {"sudo", "nohup", "nice", "env", "command", "time"}
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _split_segments(command: str) -> list[str]:
    """Split a command line on shell separators (`&&`, `||`, `;`, `|`, `&`,
    newline) while honoring quoting: `'...'`, `"..."` and backslash escapes
    keep their content intact, so a newline or `;` inside a quoted argument
    (a multi-line commit message, say) never tears a segment into an
    unparsable half and feeds literal text to the destructive-command
    fallback."""
    segments: list[str] = []
    start = 0
    i = 0
    n = len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
        if quote:
            # A backslash escapes the next char only inside double quotes.
            if c == "\\" and quote == '"':
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
        elif c == "\\":
            i += 2
        elif c in "'\"":
            quote = c
            i += 1
        elif c in ";\n":
            segments.append(command[start:i])
            i += 1
            start = i
        elif c in "&|":
            segments.append(command[start:i])
            if command[i + 1 : i + 2] == c:
                i += 1
            i += 1
            start = i
        else:
            i += 1
    segments.append(command[start:])
    return segments


_DESTRUCTIVE_WORD = re.compile(r"(?:^|[\s;&|(])(sudo\s+)?(rm|mv|cp)\b")
# Conservative fallback for unparsable segments: any `git … stash …` in there.
_GIT_STASH_WORD = re.compile(r"\bgit\b[^;&|]*\bstash\b")

# `git stash` subcommands allowed through the guard (read-only over the stack).
_ALLOWED_STASH_SUBCOMMANDS = {"list"}


def _git_leading_words(words: list[str], i: int, n: int = 2) -> list[str]:
    """Return up to `n` non-flag words following the `git` at index `i`,
    skipping git's global flags (including their values, e.g. `-C <path>`,
    `-c <key>=<value>`)."""
    out: list[str] = []
    j = i + 1
    while j < len(words) and len(out) < n:
        w = words[j]
        if w == "--":
            break
        if w.startswith("-"):
            if w in ("-C", "-c") and j + 1 < len(words):
                j += 1  # these flags take a value
            j += 1
            continue
        out.append(w)
        j += 1
    return out


def _is_forbidden_git_stash(words: list[str], i: int) -> bool:
    """True if the `git` invocation at index `i` mutates the shared stash stack.

    `git stash` with no subcommand defaults to `stash push`, so only the
    read-only `list` is permitted.
    """
    sub = _git_leading_words(words, i)
    if not sub or sub[0] != "stash":
        return False
    return len(sub) < 2 or sub[1] not in _ALLOWED_STASH_SUBCOMMANDS


# -- bash network guard -------------------------------------------------------
#
# External network access of an agent goes through the managed `web_fetch`
# tool (drcode.web_fetch: GET-only, SSRF-guarded, size-capped, every result
# framed as untrusted content). run_bash stays for LOCAL work — an agent may
# start a dev server or a test fixture inside the project and poke its
# endpoints with `curl localhost:8000/health` — but an arbitrary outbound
# request from a shell command is refused: an unmanaged shell client bypasses
# every check `web_fetch` exists for.
#
# Permitted: the LOOPBACK interface only — `localhost` (and *.localhost,
# RFC 6761), 127.0.0.0/8 and ::1. Everything else is refused, INCLUDING names
# that resolve to public or to private-but-not-loopback addresses: the host is
# resolved through getaddrinfo and EVERY address is judged with `ipaddress`,
# which also closes the cloud metadata service (169.254.169.254). An address
# LITERAL is judged without the resolver (a lying resolver must not launder
# `http://127.0.0.1/`), and the numeric forms (`127.1`, `2130706433`,
# `::ffff:127.0.0.1`) parse as loopback — so they are allowed exactly as what
# they are, loopback, and nothing else.
#
# Redirect-following flags (`curl -L`/`--location`, `--max-redirs`, wget's
# `--max-redirect`) are REFUSED rather than under-checked: only the URL on the
# command line is validated, so a followed redirect could leave the loopback
# interface without any check. The refusal tells the model to drop the flag.
#
# Deliberately the SAME strictness as the rm/mv/cp and git-stash checks above:
# only the LEADING command of each shell segment is inspected (after env
# assignments and COMMAND_WRAPPERS), so a nested command substitution
# (`echo $(curl …)`) is not analysed. NOT analysed either — out of scope for
# what is a policy guard against casual shell fetching, not an OS-level
# network sandbox: interpreters and package managers (python, node, pip, uv,
# npm install), git remotes (push/fetch/clone), a URL list read from a file
# (`curl -i urls.txt`, `curl -K cfg`) and a proxy/tunnel the agent sets up
# itself.

# Command-line network clients: a leading command from this set may only talk
# to the loopback interface.
NETWORK_CLIENTS = frozenset({"curl", "wget", "http", "https", "ftp"})
# HTTPie (`http`/`https`) takes `[METHOD] URL [REQUEST_ITEM…]`: only the first
# non-method positional is a target — request items (`key=value`,
# `Header:value`) must never be read as hosts.
_HTTPIE_CLIENTS = frozenset({"http", "https"})
_HTTPIE_METHODS = frozenset(
    {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE", "CONNECT"}
)

# Long flags taking a VALUE — data, headers, output files, credentials — per
# client, so `curl -d @payload.json …` never has the payload read as a host
# and `http --json :8000/api` (where `--json` is a BOOLEAN) never has its URL
# swallowed as a flag value. The lists are deliberately explicit: a flag
# wrongly listed here would SKIP its target, i.e. open the hole this guard
# closes, while a missing one only makes a value look like a target (a false
# refusal, never a bypass).
_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "curl": frozenset(
        {
            "--output",
            "--data",
            "--data-ascii",
            "--data-binary",
            "--data-raw",
            "--data-urlencode",
            "--json",
            "--header",
            "--request",
            "--request-target",
            "--user",
            "--user-agent",
            "--referer",
            "--cookie",
            "--cookie-jar",
            "--form",
            "--form-string",
            "--upload-file",
            "--proxy",
            "--proxy-user",
            "--proxy-header",
            "--connect-timeout",
            "--max-time",
            "--max-redirs",
            "--max-filesize",
            "--range",
            "--write-out",
            "--config",
            "--cert",
            "--cacert",
            "--key",
            "--pass",
            "--ciphers",
            "--retry",
            "--retry-delay",
            "--retry-max-time",
            "--dump-header",
            "--stderr",
            "--trace",
            "--trace-ascii",
            "--limit-rate",
            "--speed-limit",
            "--speed-time",
            "--time-cond",
            "--continue-at",
            "--resolve",
            "--connect-to",
            "--interface",
            "--unix-socket",
            "--netrc-file",
            "--oauth2-bearer",
            "--local-port",
            "--dns-servers",
            "--keepalive-time",
            "--expect100-timeout",
            "--telnet-option",
            "--pinnedpubkey",
        }
    ),
    "wget": frozenset(
        {
            "--output-document",
            "--output-file",
            "--append-output",
            "--directory-prefix",
            "--post-data",
            "--post-file",
            "--body-data",
            "--body-file",
            "--method",
            "--timeout",
            "--tries",
            "--wait",
            "--waitretry",
            "--quota",
            "--input-file",
            "--base",
            "--level",
            "--accept",
            "--reject",
            "--domains",
            "--exclude-domains",
            "--include-directories",
            "--exclude-directories",
            "--load-cookies",
            "--save-cookies",
            "--ca-certificate",
            "--certificate",
            "--private-key",
            "--dns-timeout",
            "--read-timeout",
            "--connect-timeout",
            "--http-user",
            "--http-password",
            "--user",
            "--password",
            "--header",
            "--referer",
            "--user-agent",
            "--progress",
            "--bind-address",
            "--restrict-file-names",
            "--secure-protocol",
            "--max-redirect",
            "--limit-rate",
        }
    ),
    "ftp": frozenset(),
    "http": frozenset(
        {
            "--auth",
            "--auth-type",
            "--session",
            "--session-read-only",
            "--style",
            "--print",
            "--pretty",
            "--verify",
            "--cert",
            "--cert-key",
            "--timeout",
            "--proxy",
            "--default-scheme",
            "--output",
            "--format-options",
            "--response-charset",
            "--response-mime",
            "--max-redirects",
        }
    ),
    "https": frozenset(
        {
            "--auth",
            "--auth-type",
            "--session",
            "--session-read-only",
            "--style",
            "--print",
            "--pretty",
            "--verify",
            "--cert",
            "--cert-key",
            "--timeout",
            "--proxy",
            "--default-scheme",
            "--output",
            "--format-options",
            "--response-charset",
            "--response-mime",
            "--max-redirects",
        }
    ),
}

# Short flags whose value may be glued to the flag or follow it as the next
# token (`-oFILE`, `-sSo FILE`). Per client: the same letter differs (curl's
# `-O` is a boolean "keep the remote name", wget's `-O` takes the output file,
# wget's `-L` is `--relative` while curl's `-L` follows redirects).
_SHORT_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "curl": frozenset("odHXAuAebcFTxmrEwKDzCUyY"),
    "wget": frozenset("OPoaeUTtwlARDIXBiQ"),
    "ftp": frozenset(),
    "http": frozenset("opsaA"),
    "https": frozenset("opsaA"),
}

# Flags whose VALUE is itself a fetch target (curl's `--url http://…`).
_TARGET_VALUE_FLAGS = frozenset({"--url"})

# Flags following HTTP redirects, and the max-redirs knobs of both clients.
_REDIRECT_FLAGS = frozenset(
    {
        "--location",
        "--location-trusted",
        "--max-redirs",
        "--max-redirect",
        "--max-redirects",
    }
)

# Conservative fallback for unparsable segments (bad quoting): a leading
# network client in there is refused instead of executed blind. The
# `(?![:/])` keeps URLs (`http://…`, `ftp://…`) out of the match — they are
# data, not an invocation.
_NETWORK_WORD = re.compile(
    r"(?:^|[\s;&|(])(?:sudo\s+)?(?:curl|wget|https?|ftp)\b(?![:/])"
)

# A shell redirection token as shlex leaves it in a segment: `>`, `>>`, `2>`,
# `<`, `<<EOF`, `>file`, `>&2`, `2>&1`.
_REDIRECTION = re.compile(r"^(?P<fd>[0-9]*)(?P<op>>>|>>|<<|<|>)(?P<rest>.*)$")


def _redirection_end(token: str, next_index: int) -> int | None:
    """Index after a shell redirection token, or None when it is not one.

    run_bash's segment split does not treat `<`/`>` as separators, so they
    stay in the segment; the operator and its FILE operand are never fetch
    targets and must not be read as hosts. An inline operand (`>out.json`,
    `2>err.log`, `>&2`) is inside the token itself — nothing to skip; a bare
    operator (`>`, `2>`) takes the NEXT token as its operand.
    """
    match = _REDIRECTION.match(token)
    if match is None:
        return None
    rest = match.group("rest")
    if rest:
        return next_index  # `>&2` / `>out.json`: the operand is inline
    return next_index + 1  # `>` / `2>`: the operand is the next token


def _redirect_flag(cmd: str, args: list[str]) -> str | None:
    """The first redirect-following flag in `args`, if any (see the section
    comment: following a redirect would escape the loopback check)."""
    for token in args:
        if token.startswith("--"):
            name = token.partition("=")[0]
            if name in _REDIRECT_FLAGS:
                return name
            continue
        if cmd == "curl" and token.startswith("-") and "L" in token[1:]:
            # Also inside a cluster (`-sSL`), where it is easy to miss.
            return "-L"
    return None


def _redirect_violation(command: str, cmd: str, flag: str) -> PathViolation:
    return PathViolation(
        f"Command refused: '{command}': {cmd} {flag} follows redirects, and a "
        f"redirect target is never checked — run_bash may only reach the local "
        f"machine (loopback), so following a redirect could leave it. Drop the "
        f"flag and fetch the local endpoint directly; public pages are read "
        f"with the web_fetch tool, never with a shell client."
    )


def _network_violation(
    command: str,
    cmd: str,
    target: str,
    reason: str,
    *,
    web_fetch_available: bool,
) -> PathViolation:
    """The refusal of an outbound request from a shell command.

    `web_fetch_available` picks the actionable half of the message: when the
    running role has the `web_fetch` tool the model is pointed at it,
    otherwise it must know that no external access exists in this run at all
    (the coding, tester and resolver agents never get the tool).
    """
    if web_fetch_available:
        instead = (
            "outbound network requests from run_bash are disabled — read the "
            "page with the web_fetch tool instead."
        )
    else:
        instead = (
            "external network access is closed for this agent: there is no "
            "way to reach a public page from this run at all."
        )
    return PathViolation(
        f"Command refused: '{command}': {cmd} would talk to '{target}' "
        f"({reason}), but {instead} A shell command may only reach the LOCAL "
        f"machine — loopback addresses (localhost, 127.0.0.0/8, ::1), e.g. a "
        f"dev server or a test fixture started inside this project. Public "
        f"and LAN hosts are not available to run_bash."
    )


def _strip_url_head(raw: str) -> str:
    """The authority of a bare target — `host[:port]` out of
    `[user[:pass]@]host[:port]/path?query#fragment`."""
    head = raw.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if "@" in head:  # credentials before the host
        head = head.rsplit("@", 1)[1]
    return head


def _host_of_target(text: str) -> str | None:
    """The host a fetch target talks to, or None when the argument is not a
    network target at all (a local file, a bare path, stdin).

    Handles what the clients really accept: a full URL
    (`http://host:port/x`), a scheme-relative one (`//host/x`), a bare host
    (`example.com`, `localhost:8000/x`), an IPv6 literal (`[::1]:8000`) and
    HTTPie's `:3000/path` shorthand for localhost.
    """
    raw = (text or "").strip()
    if not raw or raw == "-" or raw.startswith("@"):
        # `-` is stdin, `@file` is a data/upload reference — never a host.
        return None
    if "://" in raw:
        try:
            return urlsplit(raw).hostname or None
        except ValueError:
            return _strip_url_head(raw) or None
    if raw.startswith("//"):
        try:
            return urlsplit(f"http:{raw}").hostname or None
        except ValueError:
            return None
    if raw.startswith("/"):
        # A bare path: there is no host to talk to (the client itself will
        # complain), so nothing to check.
        return None
    head = _strip_url_head(raw)
    if head.startswith(":"):
        return "localhost"  # HTTPie's `:3000/path`
    if head.startswith("["):  # [::1] or [::1]:8000
        end = head.find("]")
        return head[1:end] if end > 0 else head[1:]
    if head.count(":") >= 2:
        return head  # a bare IPv6 literal, no port to strip
    host, _, port = head.rpartition(":")
    if host and port.isdigit():
        return host
    return head or None


def _port_of_target(text: str) -> int | None:
    """The port a bare/URL target names (None when it names none)."""
    raw = (text or "").strip()
    try:
        return urlsplit(raw if "://" in raw else f"//{raw}").port
    except ValueError:
        return None


def _legacy_ipv4(text: str) -> str | None:
    """The canonical dotted form of a LEGACY numeric IPv4 spelling, or None.

    `inet_aton` is what the C library — and therefore every client — accepts:
    `127.1`, `0x7f.1`, `017700000001` and the integer form `2130706433` all
    mean 127.0.0.1. Judging them here keeps the decision independent of DNS
    for anything numeric, which is exactly what the `http://2130706433/`
    trick relies on (a resolver may or may not resolve it, the client always
    connects to 127.0.0.1).
    """
    try:
        packed = socket.inet_aton(text)
    except (OSError, ValueError):
        return None
    return str(ipaddress.IPv4Address(packed))


def _is_loopback_address(address: str) -> bool:
    """Whether this IP literal is loopback, parsed with `ipaddress` — the only
    reliable way to judge the numeric forms (`127.1`, `2130706433`,
    `::ffff:127.0.0.1`) a command can smuggle in."""
    text = address.split("%", 1)[0]  # drop an IPv6 scope id
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        legacy = _legacy_ipv4(text)
        return legacy is not None and ipaddress.ip_address(legacy).is_loopback
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None and mapped != ip:
        return _is_loopback_address(str(mapped))
    return ip.is_loopback


def _resolve_addresses(host: str, port: int | None) -> list[str]:
    """Every address `host` resolves to (the guard's single DNS seam).

    Kept as a module function so tests can inject a fake resolver instead of
    depending on real DNS: the guard's decision must be provable offline
    (and a CI box's resolver is not part of the contract). Any resolution
    failure is an empty list — the caller treats "could not resolve" as
    "cannot be proven local".
    """
    try:
        infos = socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
    except (socket.gaierror, UnicodeError, OSError):
        return []
    addresses: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr and str(sockaddr[0]) not in addresses:
            addresses.append(str(sockaddr[0]))
    return addresses


def _non_loopback_reason(host: str, port: int | None = None) -> str | None:
    """Why `host` may NOT be used, or None when it reaches only loopback.

    An address LITERAL is judged on the spot, without the resolver (the check
    must not depend on getaddrinfo for `127.0.0.1` to be recognized as local).
    Otherwise the name is resolved and EVERY returned address must be
    loopback: a name resolving anywhere else — a public host, a LAN box, the
    cloud metadata service — is refused. A name that does not resolve is
    refused too: it cannot be PROVEN local.
    """
    text = host.split("%", 1)[0].strip().strip("[]").strip(".")
    if not text:
        return "the target has an empty host"
    lower = text.lower()
    if lower == "localhost" or lower.endswith(".localhost"):
        return None  # RFC 6761: *.localhost is loopback by definition
    try:
        ipaddress.ip_address(text)
    except ValueError:
        # Not a canonical literal — but it may still be a LEGACY numeric IPv4
        # (`127.1`, `0x7f.1`, `2130706433`): the client's C library accepts
        # those and never asks DNS, so neither do we.
        legacy = _legacy_ipv4(text)
        if legacy is not None:
            if ipaddress.ip_address(legacy).is_loopback:
                return None
            return f"{legacy} is not a loopback address"
    else:
        if _is_loopback_address(text):
            return None
        return f"{text} is not a loopback address"
    addresses = _resolve_addresses(text, port)
    if not addresses:
        return f"{text} does not resolve, so it cannot be shown to be local"
    foreign = sorted(a for a in addresses if not _is_loopback_address(a))
    if foreign:
        return f"{text} resolves to {', '.join(foreign)} — not loopback"
    return None


def _network_targets(cmd: str, args: list[str]) -> list[str]:
    """The fetch targets (URLs / hosts / IP literals) among a client's args.

    Flags, shell redirections and the values of value-taking flags are not
    targets: headers, payloads and output file names never reach the check.
    For HTTPie only the first non-method positional counts.
    """
    value_flags = _VALUE_FLAGS.get(cmd, frozenset())
    short_value = _SHORT_VALUE_FLAGS.get(cmd, frozenset())
    httpie = cmd in _HTTPIE_CLIENTS
    targets: list[str] = []
    i = 0
    n = len(args)
    while i < n:
        token = args[i]
        i += 1
        if not token or token == "--":
            continue
        redirect_end = _redirection_end(token, i)
        if redirect_end is not None:
            i = redirect_end
            continue
        if token.startswith("--"):
            name, glued, value = token.partition("=")
            if name in _TARGET_VALUE_FLAGS:
                if glued:
                    if value:
                        targets.append(value)
                elif i < n:
                    targets.append(args[i])
                    i += 1
                continue
            if name in value_flags and not glued and i < n:
                i += 1  # the flag's value: a header, a payload, a file, …
            continue
        if token.startswith("-") and len(token) > 1:
            # A short-flag cluster: walk it, because a value-taking flag
            # hides its value inside or after the cluster (`-oFILE`,
            # `-sSo FILE`) and the rest of the cluster is not a target.
            for index, ch in enumerate(token[1:], start=1):
                if ch in short_value:
                    if index == len(token) - 1 and i < n:
                        i += 1  # the value follows as the next token
                    break
            continue
        targets.append(token)
        if httpie:
            break  # HTTPie's URL comes first; the rest are request items
    return targets


def _check_network_command(
    command: str,
    cmd: str,
    args: list[str],
    *,
    web_fetch_available: bool = False,
) -> None:
    """Refuse an outbound request of a network client unless it is loopback.

    Raises PathViolation — a plain refusal (the agent layer logs it and hands
    the text back as a retry prompt), never a crash.
    """
    flag = _redirect_flag(cmd, args)
    if flag is not None:
        raise _redirect_violation(command, cmd, flag)
    if cmd in _HTTPIE_CLIENTS:
        args = [a for a in args if a not in _HTTPIE_METHODS]
    for target in _network_targets(cmd, args):
        host = _host_of_target(target)
        if host is None:
            continue  # not a network target (a local file, an empty host)
        reason = _non_loopback_reason(host, _port_of_target(target))
        if reason is not None:
            raise _network_violation(
                command,
                cmd,
                target,
                reason,
                web_fetch_available=web_fetch_available,
            )


def check_command(
    root: Path,
    command: str,
    extra_writable: Path | None = None,
    *,
    web_fetch_available: bool = False,
) -> None:
    """Refuse rm/mv/cp whose path arguments escape the writable area.

    The command line is split into shell segments (`;`, `&&`, `||`, `|`,
    newlines) and the leading command of each is inspected — after env
    assignments (VAR=…) and common wrappers (sudo, nohup, …). For a
    destructive command every non-flag argument is treated as a path and
    checked against the writable area (project root + `extra_writable`).
    The project state directory (`<root>/.drcode`) is protected even though
    it sits inside the writable area: a path resolving into it — or into an
    ancestor of it (`rm -rf .`, `rm -rf <root>`) — is refused, since deleting
    it destroys all task history irrecoverably. Nested command substitution
    (`echo $(rm …)`) is not analysed; only top-level invocations are.

    Additionally, any `git stash` invocation that mutates the stash stack
    (`push`/`pop`/`apply`/`drop`/`clear`/bare `stash`) is refused outright:
    `refs/stash` is shared between all worktrees of the repository, so
    parallel task agents would otherwise push and pop each other's stashed
    changes. `git stash list` (read-only) stays allowed.

    Finally, an outbound request of a network client (`curl`, `wget`, `http`,
    `https`, `ftp`) is refused unless it targets the loopback interface:
    external network access belongs to the managed `web_fetch` tool, while
    run_bash stays for local work (a dev server or a test fixture started
    inside the project is reachable at localhost/127.0.0.1/::1). See the
    "bash network guard" section above for what is and is not analysed.

    `web_fetch_available` only shapes the refusal text: when the running role
    carries the `web_fetch` tool the model is pointed at it, otherwise it is
    told that external access is closed in this run.
    """
    state_dir = (root / PROJECT_STATE_DIRNAME).resolve()
    for segment in _split_segments(command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            words = shlex.split(segment)
        except ValueError:
            # Unparsable quoting: if a destructive command seems to be in
            # there, refuse conservatively rather than execute blind.
            if _DESTRUCTIVE_WORD.search(segment):
                raise _bash_violation(
                    root, command, "rm/mv/cp", None, extra_writable
                ) from None
            if _GIT_STASH_WORD.search(segment):
                raise _stash_violation(command) from None
            if _NETWORK_WORD.search(segment):
                raise _network_violation(
                    command,
                    "a network client",
                    "an unparsable command line",
                    "the command line could not be parsed, so its target "
                    "could not be checked",
                    web_fetch_available=web_fetch_available,
                ) from None
            continue
        i = 0
        # Strip leading env assignments and wrappers in any order
        # (`VAR=1 env -i sudo rm …`).
        while i < len(words):
            if _ENV_ASSIGN.match(words[i]):
                i += 1
            elif words[i] in COMMAND_WRAPPERS:
                i += 1
                # skip the wrapper's own flags (e.g. sudo -u user …)
                while i < len(words) and words[i].startswith("-"):
                    i += 1
            else:
                break
        if i >= len(words):
            continue
        base = Path(words[i]).name  # tolerate /bin/rm, /usr/bin/mv, …
        if base == "git" and _is_forbidden_git_stash(words, i):
            raise _stash_violation(command)
        if base in NETWORK_CLIENTS:
            _check_network_command(
                command,
                base,
                words[i + 1 :],
                web_fetch_available=web_fetch_available,
            )
            continue
        if base not in DESTRUCTIVE_COMMANDS:
            continue
        paths: list[str] = []
        after_dashdash = False
        for w in words[i + 1 :]:
            if not after_dashdash and w == "--":
                after_dashdash = True
            elif after_dashdash or not w.startswith("-"):
                paths.append(w)
        for raw in paths:
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = root / p
            p = p.resolve()
            if p == state_dir or state_dir in p.parents or p in state_dir.parents:
                raise _state_dir_violation(command, base, raw)
            if not is_writable(root, p, extra_writable):
                raise _bash_violation(root, command, base, raw, extra_writable)


# -- child process priority ----------------------------------------------------
#
# Children spawned for background work (run_bash, the git plumbing, workflow
# hooks, the resolver's verification funnel) must not compete with the TUI at
# equal priority: a `pytest -n auto` child tree can hold a dozen xdist workers,
# saturate the CPU and starve the Textual render loop (a CPU-bound Python
# process), so keystrokes lag seconds behind. The fix is inversion — lower the
# children instead of raising the parent (a negative nice needs root): they
# inherit the lowered nice into their own children, and the TUI stays
# responsive. Cost for the children is minimal — they still get the CPU as
# soon as the TUI blocks or idles.


def _nice_child() -> None:
    """preexec_fn body: runs in the forked child right before exec.

    os.nice is one simple syscall (no allocation, no locks) — safe in the
    pre-exec window. A failure must never fail the spawn itself: a scheduling
    preference is not worth an error, so it is swallowed.
    """
    try:
        os.nice(_CHILD_NICE)
    except OSError:
        pass


def child_priority_kwargs() -> dict[str, Any]:
    """subprocess kwargs that spawn the child BELOW the parent's priority.

    Merge into every subprocess spawn of background work (both the sync
    `subprocess.run`/`Popen` and the async `asyncio.create_subprocess_*`,
    which forwards **kwds to `subprocess.Popen`). POSIX uses preexec_fn with
    os.nice plus start_new_session (own process group, detached from the
    TUI's tty — see _kill_proc) — deliberately NOT a `nice …` wrapper in a
    command string: the
    run_bash command line is inspected verbatim (COMMAND_WRAPPERS) and a
    wrapper would distort that inspection. Windows uses
    BELOW_NORMAL_PRIORITY_CLASS. Unknown platforms get no kwargs.
    """
    if os.name == "nt":
        # The constant only exists in subprocess on Windows; the fallback
        # value is its documented stable value (keeps pyright/POSIX happy).
        return {
            "creationflags": getattr(
                subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000
            )
        }
    if os.name == "posix":  # Linux, macOS (darwin), the BSDs, ...
        # start_new_session (setsid) detaches the child from the TUI's
        # controlling terminal and makes it a process-group leader of its
        # own — that is what lets a timeout/cancel kill take down the whole
        # child tree with one killpg (see _kill_proc): without it grandchildren
        # survive the shell's death, reparent to init and keep the TUI's tty
        # open as competing stdin readers (the input-stealing incident).
        return {"preexec_fn": _nice_child, "start_new_session": True}
    return {}


# -- macOS malloc-stack-logging noise -------------------------------------------
#
# Corporate monitoring daemons (siagent/SearchInform, Check Point cpard) toggle
# malloc stack logging REMOTELY in live processes — including the TUI's python.
# libmalloc then makes every fork()ed child of such a process print a
# `Name(pid) MallocStackLogging: ...` report line to the INHERITED stderr in
# the window between fork() and exec() — before Python dup2's the capture
# pipes, so `capture_output` cannot intercept it (on the TUI that is the
# terminal; run_bash's own `2>` shell filter mirrors
# _install_stderr_filter). For a child that gets toggled ON while running
# (long verification runs), the same lines land in its CAPTURED stderr and
# from there into service events and agent transcripts — never the tool's
# own output. CPython 3.14 on macOS cannot avoid the fork path (close_fds
# defaults to True and os.POSIX_SPAWN_CLOSEFROM is missing, so posix_spawn is
# ineligible for nearly every spawn), hence the lines are stripped at the
# capture boundary instead of being prevented at the source.

_MALLOC_STACK_LOGGING_LINE_RE = re.compile(
    r"(?m)^[^(\n]+\(\d+\) MallocStackLogging: .*(?:\n|$)"
)


def strip_malloc_stack_logging_noise(text: str) -> str:
    """Remove libmalloc `Name(pid) MallocStackLogging: ...` report lines.

    Applied to every captured child output that is shown to the model or
    stored in transcripts/service events. Normal lines — including lines that
    merely QUOTE the word in some other context — are kept intact.
    """
    return _MALLOC_STACK_LOGGING_LINE_RE.sub("", text)


def clamp_bash_timeout(timeout: float | None) -> float:
    """Resolve the effective timeout of a run_bash call.

    None (or a non-positive value) falls back to the default BASH_TIMEOUT;
    anything above BASH_MAX_TIMEOUT is clamped down to the cap.
    """
    if timeout is None or timeout <= 0:
        return BASH_TIMEOUT
    return min(timeout, BASH_MAX_TIMEOUT)


# -- child-process user-config sandbox ----------------------------------------

# Name of the per-task user-config sandbox inside the scratch directory,
# exported to the child as DRCODE_USER_CONFIG_DIR (config._resolve_user_config_dir
# gives it priority over XDG_CONFIG_HOME — XDG itself is NOT redirected, so the
# child's git/gh and other XDG-reading tools keep working against the real
# config tree).
USER_CONFIG_SANDBOX_DIRNAME = "user-config"


def _seed_user_config_sandbox(sandbox: Path) -> None:
    """Fill a freshly created sandbox with a copy of the real user config.

    Diagnostic snippets the agent runs should see plausible data (the user's
    actual provider catalog etc.), so the first sandbox creation copies the
    real `~/.config/drcode` contents in. Later run_bash calls reuse the
    existing sandbox as-is — the agent's own writes persist for the task's
    lifetime, and the real catalog is never touched again. Failures are
    swallowed: an empty sandbox is still a safe sandbox.
    """
    real = _config._USER_CONFIG_DIR
    try:
        if real.is_dir():
            shutil.copytree(
                real,
                sandbox,
                dirs_exist_ok=True,
                ignore_dangling_symlinks=True,
            )
    except OSError:
        pass


def _user_config_sandbox_env(
    env: dict[str, str], task_id: str, extra_writable: Path | None
) -> None:
    """Point the child's drcode user config at the task's sandbox copy.

    The sandbox lives inside the task's scratch directory (created lazily,
    seeded once from the real config) and is removed with the scratch on
    task completion/delete, so writes from agent-spawned drcode processes —
    `UserModelsStore()` in a verification snippet, a nested `uv run drcode` —
    physically cannot reach the real user catalog. Only set when the export
    succeeds; on a failure to create the directory the variable stays unset
    (degraded to the old behavior) rather than pointing at a missing path.
    """
    base = extra_writable if extra_writable is not None else scratch_dir(task_id)
    sandbox = base / USER_CONFIG_SANDBOX_DIRNAME
    try:
        if not sandbox.is_dir():
            sandbox.mkdir(parents=True, exist_ok=True, mode=0o700)
            _seed_user_config_sandbox(sandbox)
    except OSError:
        return
    env["DRCODE_USER_CONFIG_DIR"] = str(sandbox)


# How long the guard's DNS lookups may take before the command is refused.
# `getaddrinfo` has no timeout of its own and a wedged resolver can sit on it
# for minutes; the decision is fail-closed anyway (an unresolvable target
# cannot be PROVEN loopback), so the bound only shortens a refusal that was
# already coming.
_NETWORK_CHECK_TIMEOUT = 30.0


async def _check_command_async(
    root: Path,
    command: str,
    extra_writable: Path | None,
    *,
    web_fetch_available: bool,
) -> None:
    """`check_command` off the event loop.

    The path half of the checks is pure and instant, but the bash network
    guard resolves host names (`socket.getaddrinfo` — see `_resolve_addresses`)
    and a slow resolver would otherwise stall the loop that drives the Textual
    UI: the same reason every tool body runs through `tool_runtime._run_tool`.
    """
    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                check_command,
                root,
                command,
                extra_writable,
                web_fetch_available=web_fetch_available,
            ),
            timeout=_NETWORK_CHECK_TIMEOUT,
        )
    except TimeoutError:
        raise PathViolation(
            f"Command refused: '{command}': checking the target's address took "
            f"longer than {_NETWORK_CHECK_TIMEOUT:.0f}s (a stalled name "
            f"resolver), so the command cannot be shown to reach only the LOCAL "
            f"machine. run_bash may only talk to loopback (localhost, "
            f"127.0.0.0/8, ::1); retry with a loopback address or an IP literal."
        ) from None


async def run_bash(
    root: Path,
    command: str,
    timeout: float | None = None,
    *,
    extra_writable: Path | None = None,
    task_id: str | None = None,
    web_fetch_available: bool = False,
) -> str:
    await _check_command_async(
        root,
        command,
        extra_writable,
        web_fetch_available=web_fetch_available,
    )
    effective = clamp_bash_timeout(timeout)
    env = sanitized_env()
    if extra_writable is not None:
        # The child process's temporary files land in the task's scratch
        # directory too (created lazily here on the first run_bash call) —
        # third-party tools then cannot clash with another task's temps.
        extra_writable.mkdir(parents=True, exist_ok=True, mode=0o700)
        _stamp_scratch_owner(extra_writable)
        env["TMPDIR"] = str(extra_writable)
    if task_id is not None:
        # Parallel development slots run several tasks at once: the child
        # process can tell its own task apart (pick per-task ports, write
        # per-task caches). Under worktree isolation the cwd IS the task's
        # worktree, so DRCODE_WORKTREE distinguishes it from the project
        # root; a plain task still reports the project directory.
        env["DRCODE_TASK_ID"] = task_id
        env["DRCODE_WORKTREE"] = str(root)
        # Sandbox the child's drcode user config: a diagnostic snippet's
        # default-constructed store reads/writes the scratch copy, never
        # the real ~/.config/drcode (incident ebe6ba15).
        _user_config_sandbox_env(env, task_id, extra_writable)
    else:
        # Without a task id the variables stay unset (no empty-string noise):
        # strip any inherited values so a task-less child never sees a
        # foreign task's identity — nor a foreign user-config sandbox.
        env.pop("DRCODE_TASK_ID", None)
        env.pop("DRCODE_WORKTREE", None)
        env.pop("DRCODE_USER_CONFIG_DIR", None)
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=root,
        env=env,
        # stdin is DEVNULL, never the inherited tty: run_bash offers no
        # interactive input anyway, and a child that would block on a prompt
        # (npm exec's "Ok to proceed? (y)") instead gets EOF and dies fast —
        # it must not sit forever reading the TUI's keystrokes (the
        # input-stealing incident).
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **child_priority_kwargs(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=effective)
    except TimeoutError:
        _kill_proc(proc)
        await proc.wait()
        return f"exit=-1\n(command timed out after {effective}s)"
    except asyncio.CancelledError:
        # The run was aborted or the app is quitting: the child process must
        # not outlive the tool call. Kill it and let the event loop reap it
        # and close its transport — otherwise the transport's __del__ later
        # tries to close pipes on an already-closed loop and prints
        # "RuntimeError: Event loop is closed" to the console.
        _kill_proc(proc)
        try:
            await asyncio.shield(proc.wait())
        except asyncio.CancelledError:
            pass
        raise
    # Normal completion: the shell exited, but grandchildren that outlive it
    # (a backgrounded daemon, a pty harness) stay in its process group and
    # reparent to init — the group still exists while any member lives, so a
    # single best-effort killpg takes the leftovers down (incident
    # 2026-11-02: orphaned pty-harness processes eating ~80% CPU for hours
    # after run_bash had returned). When the group is already empty this is
    # a no-op (ProcessLookupError). Processes that fully detached into a NEW
    # session (forkpty/setsid) are unreachable this way — that is the
    # scratch-cleanup report's second echelon.
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass  # group already gone / partial permission
    raw = stdout.decode(errors="replace") + stderr.decode(errors="replace")
    out = strip_malloc_stack_logging_noise(raw).strip()[:MAX_TOOL_OUTPUT]
    return f"exit={proc.returncode}\n{out or '(no output)'}"


def _kill_proc(proc: asyncio.subprocess.Process) -> None:
    """Kill the child — and its whole process group — if still alive.

    run_bash spawns the child in its own session (start_new_session), so the
    shell's pid is also its process-group id and one killpg takes down the
    grandchildren (npm/node/…) that would otherwise survive the shell's
    SIGKILL, reparent to init and keep the TUI's tty open as competing stdin
    readers. The fallback to a plain kill covers a child that is not a group
    leader (spawned elsewhere) or a group that has already gone away.

    The kill races the child's own exit: a cancellation can land after
    communicate() has already reaped the process and closed its transport,
    and then the signal raises ProcessLookupError — which is exactly the
    "already dead" outcome the kill was aiming for, never a real failure.
    """
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass  # not a group leader / already dead / partial permission
    try:
        proc.kill()
    except ProcessLookupError:
        pass


# -- scratch lifecycle ---------------------------------------------------------


# Marker file inside every scratch directory recording the pid of the drcode
# instance that owns it. Task ids are only unique per project (short 8-hex
# ids), but /tmp/drcode is global — so both the startup orphan sweep and the
# completion-time cleanup must not trust directory names alone: a directory
# with a live foreign owner belongs to an active task of ANOTHER running
# drcode instance and must be kept.
SCRATCH_OWNER_FILE = ".owner"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _stamp_scratch_owner(scratch: Path) -> None:
    """Record this process's pid as the scratch directory's owner."""
    try:
        (scratch / SCRATCH_OWNER_FILE).write_text(str(os.getpid()))
    except OSError:
        pass


def _stamp_owner_if_scratch(extra_writable: Path, p: Path) -> None:
    """Stamp the owner marker when the just-resolved `p` lies inside the
    task's scratch directory (write_file/edit_file create it lazily)."""
    try:
        resolved = extra_writable.resolve()
    except OSError:
        return
    if resolved in p.parents:
        _stamp_scratch_owner(resolved)


def _foreign_live_owner(scratch: Path) -> bool:
    """True when `scratch` carries an owner marker of another LIVE process."""
    try:
        pid = int((scratch / SCRATCH_OWNER_FILE).read_text().strip())
    except (OSError, ValueError):
        return False
    return pid != os.getpid() and _pid_alive(pid)


# Seconds between the SIGTERM and the SIGKILL for processes found living in
# a scratch directory: enough for a well-behaved TUI/server to shut down.
_SCRATCH_KILL_GRACE = 2.0
# Hard cap on the lsof scan itself: scratch trees are small, but a stray
# mount/symlink fan-out must not stall the cleanup (it runs on the UI thread
# via _remove_scratch_background's thread, still — keep it bounded).
_LSOF_TIMEOUT = 15.0


def _scratch_dweller_pids(scratch: Path) -> set[int]:
    """PIDs of THIS user's processes whose cwd or an open file lies under
    `scratch` (recursive lsof scan, best-effort).

    A missing/broken/timed-out lsof yields an empty set — the removal then
    proceeds rmtree-only, exactly as before this echelon existed. Processes
    of other users are filtered out even if lsof ever reports them (only
    root's lsof could): a path coincidence inside /tmp/drcode is ownership
    proof for the task, never a licence to kill someone else's process.
    """
    if os.name != "posix":
        return set()
    try:
        proc = subprocess.run(
            ["lsof", "-t", "-w", "+D", str(scratch)],
            check=False,
            capture_output=True,
            timeout=_LSOF_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    pids: set[int] = set()
    for token in proc.stdout.decode(errors="replace").split():
        try:
            pid = int(token)
        except ValueError:
            continue
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            continue  # already gone / foreign process — not ours to signal
        pids.add(pid)
    pids.discard(os.getpid())
    return pids


def _kill_scratch_dwellers(scratch: Path) -> None:
    """SIGTERM → short wait → SIGKILL the processes living in `scratch`.

    The only defense against processes that fully detached from run_bash's
    process group (forkpty/setsid) — killpg cannot reach them, but their cwd
    or open files still sit in the directory being removed.
    """
    try:
        pids = _scratch_dweller_pids(scratch)
    except OSError:
        return  # scan failed entirely — degrade to rmtree-only
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    deadline = time.monotonic() + _SCRATCH_KILL_GRACE
    while time.monotonic() < deadline and any(_pid_alive(p) for p in pids):
        time.sleep(0.1)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def remove_scratch(task_id: str) -> None:
    """Best-effort removal of a task's scratch directory.

    Called when the task reaches COMPLETED or is deleted. It is NOT called
    on an interrupted/re-queued task: the next activation resumes the work
    and may still need the scratch files from the interrupted attempt.

    Processes the agent left running from inside the scratch (pty harnesses,
    TUIs, servers) are terminated first — deleting a directory does not
    stop the processes whose cwd or open files live in it (incident
    2026-11-02: ten orphaned ~80%-CPU processes survived the rmtree).

    A directory owned by another live drcode process is left alone: task
    ids are per-project, so another instance's active task may share this
    id (collision) — its owner removes the directory when its own task
    completes or is deleted.
    """
    scratch = scratch_dir(task_id)
    if _foreign_live_owner(scratch):
        return
    _kill_scratch_dwellers(scratch)
    shutil.rmtree(scratch, ignore_errors=True)


def cleanup_orphan_scratches(live_task_ids: set[str]) -> list[str]:
    """Remove scratch directories of tasks that no longer exist.

    Runs once at app startup. A directory whose name matches no live task id
    of THIS project may still belong to an active task of ANOTHER running
    drcode instance: such directories carry an owner marker (see
    SCRATCH_OWNER_FILE) naming a live pid and are kept. Directories without
    a marker, or with a dead owner, are leftovers from a crashed session
    and removed. Returns the removed directory names.
    """
    if not SCRATCH_ROOT.is_dir():
        return []
    removed: list[str] = []
    for entry in SCRATCH_ROOT.iterdir():
        if not entry.is_dir() or entry.name in live_task_ids:
            continue
        if _foreign_live_owner(entry):
            continue
        shutil.rmtree(entry, ignore_errors=True)
        removed.append(entry.name)
    return removed
