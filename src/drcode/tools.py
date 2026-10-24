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
  (ports, caches) — see the parallel development slots.

Every refusal raises PathViolation with a model-friendly message; the agent
layer logs it into the task's service-event history before returning it to
the model as a retry prompt.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
from pathlib import Path

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
# Shell separators that split a command line into independent segments.
_SEGMENT_SPLIT = re.compile(r"&&|\|\||[;|\n&]")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
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


def check_command(root: Path, command: str, extra_writable: Path | None = None) -> None:
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
    """
    state_dir = (root / PROJECT_STATE_DIRNAME).resolve()
    for segment in _SEGMENT_SPLIT.split(command):
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


def clamp_bash_timeout(timeout: float | None) -> float:
    """Resolve the effective timeout of a run_bash call.

    None (or a non-positive value) falls back to the default BASH_TIMEOUT;
    anything above BASH_MAX_TIMEOUT is clamped down to the cap.
    """
    if timeout is None or timeout <= 0:
        return BASH_TIMEOUT
    return min(timeout, BASH_MAX_TIMEOUT)


async def run_bash(
    root: Path,
    command: str,
    timeout: float | None = None,
    *,
    extra_writable: Path | None = None,
    task_id: str | None = None,
) -> str:
    check_command(root, command, extra_writable)
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
    else:
        # Without a task id the variables stay unset (no empty-string noise):
        # strip any inherited values so a task-less child never sees a
        # foreign task's identity.
        env.pop("DRCODE_TASK_ID", None)
        env.pop("DRCODE_WORKTREE", None)
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=root,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=effective)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return f"exit=-1\n(command timed out after {effective}s)"
    except asyncio.CancelledError:
        # The run was aborted or the app is quitting: the child process must
        # not outlive the tool call. Kill it and let the event loop reap it
        # and close its transport — otherwise the transport's __del__ later
        # tries to close pipes on an already-closed loop and prints
        # "RuntimeError: Event loop is closed" to the console.
        proc.kill()
        try:
            await asyncio.shield(proc.wait())
        except asyncio.CancelledError:
            pass
        raise
    out = (stdout.decode(errors="replace") + stderr.decode(errors="replace")).strip()[
        :MAX_TOOL_OUTPUT
    ]
    return f"exit={proc.returncode}\n{out or '(no output)'}"


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


def remove_scratch(task_id: str) -> None:
    """Best-effort removal of a task's scratch directory.

    Called when the task reaches COMPLETED or is deleted. It is NOT called
    on an interrupted/re-queued task: the next activation resumes the work
    and may still need the scratch files from the interrupted attempt.

    A directory owned by another live drcode process is left alone: task
    ids are per-project, so another instance's active task may share this
    id (collision) — its owner removes the directory when its own task
    completes or is deleted.
    """
    scratch = scratch_dir(task_id)
    if _foreign_live_owner(scratch):
        return
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
