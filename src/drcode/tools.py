"""Filesystem and shell tools for the coding agent.

Permission model:
- reading (read_file) works anywhere on the filesystem;
- writing (write_file / edit_file) is allowed only inside the project root
  (with all subdirectories) and inside /tmp (with all subdirectories);
- run_bash inspects the command and refuses rm/mv/cp whose path arguments
  escape that writable area.

Every refusal raises PathViolation with a model-friendly message; the agent
layer logs it into the task's service-event history before returning it to
the model as a retry prompt.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from pathlib import Path

MAX_TOOL_OUTPUT = 10_000
BASH_TIMEOUT = 120
# Hard cap for a run_bash call: the agent may request a longer timeout than
# the default, but never more than an hour — values above are clamped to it.
BASH_MAX_TIMEOUT = 60 * 60

# Extra scratch area: the model may create/edit files here even though it is
# outside the project — a sandbox for testing hypotheses.
TMP_ROOT = Path("/tmp")

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


def _writable_hint(root: Path) -> str:
    return (
        f"Allowed locations: the project root ('{root}') with all its "
        f"subdirectories, and '/tmp' with all its subdirectories "
        f"(tip: /tmp is available for scratch work — test hypotheses there)."
    )


def _write_violation(root: Path, path: str, action: str) -> PathViolation:
    return PathViolation(
        f"Permission denied: cannot {action} '{path}' — it is outside the "
        f"allowed locations. {_writable_hint(root)}"
    )


def _bash_violation(
    root: Path, command: str, cmd: str, path: str | None
) -> PathViolation:
    target = f" with path '{path}'" if path else ""
    return PathViolation(
        f"Command refused: '{command}': {cmd}{target} would touch files outside "
        f"the allowed locations. {_writable_hint(root)}"
    )


def resolve_read_path(root: Path, path: str) -> Path:
    """Reading is unrestricted: any path on the filesystem works.
    Relative paths are resolved against the project root."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def is_writable(root: Path, p: Path) -> bool:
    if p == root or root in p.parents:
        return True
    # /tmp may be a symlink (macOS: /tmp -> /private/tmp) — accept both the
    # literal and the resolved location.
    for t in {TMP_ROOT, TMP_ROOT.resolve()}:
        if p == t or t in p.parents:
            return True
    return False


def resolve_edit_path(root: Path, path: str, action: str = "write to") -> Path:
    """Resolve `path` for a modifying operation; refuse anything outside
    the project root and /tmp (symlinks are resolved first, so a symlink
    inside the project pointing outside is refused too)."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if is_writable(root, p):
        return p
    raise _write_violation(root, path, action)


def read_file(root: Path, path: str, offset: int = 1, limit: int = 2000) -> str:
    p = resolve_read_path(root, path)
    lines = p.read_text(errors="replace").splitlines()
    start = max(offset - 1, 0)
    chunk = lines[start : start + limit]
    numbered = [f"{start + i + 1}: {line}" for i, line in enumerate(chunk)]
    return "\n".join(numbered) or "(empty file)"


def write_file(root: Path, path: str, content: str) -> str:
    p = resolve_edit_path(root, path, "create/overwrite")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"written {len(content)} chars to {path}"


def edit_file(root: Path, path: str, old_string: str, new_string: str) -> str:
    p = resolve_edit_path(root, path, "edit")
    text = p.read_text(errors="replace")
    count = text.count(old_string)
    if count == 0:
        raise ValueError("old_string not found in file")
    if count > 1:
        raise ValueError(
            f"old_string matches {count} locations, provide more context to make it unique"
        )
    p.write_text(text.replace(old_string, new_string, 1))
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


def check_command(root: Path, command: str) -> None:
    """Refuse rm/mv/cp whose path arguments escape the writable area.

    The command line is split into shell segments (`;`, `&&`, `||`, `|`,
    newlines) and the leading command of each is inspected — after env
    assignments (VAR=…) and common wrappers (sudo, nohup, …). For a
    destructive command every non-flag argument is treated as a path and
    checked against the writable area. Nested command substitution
    (`echo $(rm …)`) is not analysed; only top-level invocations are.
    """
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
                raise _bash_violation(root, command, "rm/mv/cp", None) from None
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
            if not is_writable(root, p):
                raise _bash_violation(root, command, base, raw)


def clamp_bash_timeout(timeout: float | None) -> float:
    """Resolve the effective timeout of a run_bash call.

    None (or a non-positive value) falls back to the default BASH_TIMEOUT;
    anything above BASH_MAX_TIMEOUT is clamped down to the cap.
    """
    if timeout is None or timeout <= 0:
        return BASH_TIMEOUT
    return min(timeout, BASH_MAX_TIMEOUT)


async def run_bash(root: Path, command: str, timeout: float | None = None) -> str:
    check_command(root, command)
    effective = clamp_bash_timeout(timeout)
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=root,
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
