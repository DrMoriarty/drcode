"""One-way secret redaction at the COMPLETED transition.

The user routinely pastes access tokens into a task description / dialog
(the agent needs them while the task is alive — that is fine), but once
the task reaches COMPLETED nothing reads the transcripts anymore, and the
secrets must not sit on disk in plaintext. A completion listener
(`TaskManager.add_completion_listener`, registered in TaskManager.__init__)
runs THIS module's scrubber on every transition into COMPLETED:

- the full dev transcript (`messages`) plus the tester/resolver rounds
  (`tester_messages` / `resolver_messages`) — the sidecar is force-loaded
  first (the LRU may have evicted it), every STRING VALUE inside every
  message part is redacted in place, the message STRUCTURE is untouched
  (the transcript stays API-valid for `resumable_history`);
- the metadata copies of the same secrets: `title` (the first line of a
  description becomes the title), `events[].text`, `pending_input[].text`,
  `todos[].text`; the `skeleton` is rebuilt from the scrubbed messages by
  the following `Storage.save`;
- the append-only compaction archive (`<id>.transcript.archive.jsonl`) —
  line by line, the same patterns (a token could have been archived by a
  pre-completion compaction);
- the task's entry in the persistent search index
  (`.drcode/search_index.json` holds prompt/event copies — see
  `task_search.invalidate_task_entry`).

Redaction is IRREVERSIBLE on purpose — no recovery, no key, no entropy
guessing. Only explicit deterministic patterns (agreed with the user):

- prefixed provider tokens: ``ghp_``/``gho_``/``ghu_``/``ghs_``/``ghr_``,
  ``github_pat_…``, ``sk-…``, ``AKIA…``;
- ``Bearer <token>`` (any scheme, case-insensitive keyword);
- JWT (three base64url segments joined by dots, ``eyJ…`` header);
- GUIDs (8-4-4-4-12 hex, case-insensitive).

Known, accepted trade-off: GUIDs also appear in ordinary code/logs —
their occurrences inside a completed task's transcript are masked too.
Re-sending input into a COMPLETED task re-queues it with the ALREADY
scrubbed history — the secret is gone from the model's context as well;
for a re-validation the user pastes the token anew. Not a bug.
Unfinished tasks (hold / unfinished attempt / clarification) keep their
secrets in plaintext — that is the requirement ("the agent needs the
information"): scrubbing happens strictly on the entry into COMPLETED.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from .models import Task

# The placeholder every match is replaced with (fixed shape, irreversible).
REDACTED = "XXXXXXX"

# (name, compiled pattern) — applied to every string value in order. All
# patterns match only characters that can appear inside a JSON string
# value (no quotes/backslashes/braces), so the raw-line redaction of the
# JSONL archive can never break the JSON structure of a line.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # GitHub fine-grained personal access token.
    (
        "github_pat",
        re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),
    ),
    # GitHub classic tokens: ghp_ (PAT), gho_ (OAuth), ghu_ (user-to-server),
    # ghs_ (server-to-server), ghr_ (refresh).
    (
        "github_token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ),
    # OpenAI-style secret keys (sk-... / sk-proj-...): the sk- prefix is
    # word-bounded, so ordinary hyphenated words ("task-...", "risk-...")
    # never match.
    (
        "secret_key",
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    ),
    # AWS access key id.
    (
        "aws_key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    # Bearer credentials: the keyword is kept, only the token is masked
    # (the `tok` group carries the credential — see redact_text).
    (
        "bearer",
        re.compile(r"(?i)\bbearer\s+(?P<tok>[A-Za-z0-9._~+/=-]{8,})"),
    ),
    # JWT: three base64url segments (header starts with "eyJ", the base64
    # of '{"'). Minimum lengths keep ordinary dotted abbreviations out.
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    # GUID / UUID (the accepted trade-off: masked everywhere, not only
    # where a secret is plausible).
    (
        "guid",
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
    ),
]


def redact_text(text: str, hits: dict[str, int] | None = None) -> str:
    """Mask every pattern occurrence in `text` (a new string; `text` is
    never mutated). A Bearer keyword is preserved, the credential after
    it is replaced (the pattern's `tok` group). When `hits` is given,
    the number of replacements per pattern name is accumulated there."""
    for name, pattern in PATTERNS:
        if "tok" in pattern.groupindex:
            new, n = pattern.subn(
                lambda m: m.group(0)[: -len(m.group("tok"))] + REDACTED, text
            )
        else:
            new, n = pattern.subn(REDACTED, text)
        if n and hits is not None:
            hits[name] = hits.get(name, 0) + n
        text = new
    return text


def contains_secret(text: str) -> bool:
    """True when at least one pattern matches `text` (used by the tests
    and by the scrubber's service-event decision)."""
    return any(pattern.search(text) for _name, pattern in PATTERNS)


class ScrubReport(NamedTuple):
    """What one scrub pass found. `hits` counts the replacements per
    pattern name; `credentials_masked` EXCLUDES the GUID pattern — GUIDs
    are the accepted trade-off and appear in ordinary code/logs of
    practically every task, so only real credentials may produce the
    chat-visible "secret redaction" event (otherwise every completed
    task's timeline grows a noise line)."""

    changed: bool
    hits: dict[str, int]

    @property
    def credentials_masked(self) -> int:
        return sum(n for name, n in self.hits.items() if name != "guid")


_NOTHING = ScrubReport(False, {})


def redact_strings_in(obj: Any, hits: dict[str, int] | None = None) -> bool:
    """Redact every string VALUE inside a nested dict/list structure IN
    PLACE (dict KEYS are never touched — the part structure of a
    serialized ModelMessage must stay byte-compatible). Returns True when
    at least one value changed."""
    changed = False
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                red = redact_text(value, hits)
                if red != value:
                    obj[key] = red
                    changed = True
            elif isinstance(value, (dict, list)):
                changed = redact_strings_in(value, hits) or changed
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            if isinstance(value, str):
                red = redact_text(value, hits)
                if red != value:
                    obj[index] = red
                    changed = True
            elif isinstance(value, (dict, list)):
                changed = redact_strings_in(value, hits) or changed
    return changed


def scrub_task(task: Task) -> ScrubReport:
    """Redact the secrets of a COMPLETED-bound task IN MEMORY: the three
    transcripts (force-loaded — the lazy sidecar may not be in memory
    yet), the title, the service events, the pending input and the todo
    texts. Message structure is untouched; the caller persists with
    `Storage.save` (which also rebuilds the skeleton from the scrubbed
    messages)."""
    hits: dict[str, int] = {}
    changed = False
    # Force-load the sidecar through the lazy properties (an LRU eviction
    # between the accesses just reloads from disk — after the scrubbing
    # save below the disk copy is clean anyway).
    for key in ("messages", "tester_messages", "resolver_messages"):
        transcript = getattr(task, key)
        for message in transcript:
            if redact_strings_in(message, hits):
                changed = True
    task.title = redact_text(task.title, hits)
    for event in task.events:
        red = redact_text(event.text, hits)
        if red != event.text:
            event.text = red
            changed = True
    for pending in task.pending_input:
        red = redact_text(pending.text, hits)
        if red != pending.text:
            pending.text = red
            changed = True
    for todo in task.todos:
        red = redact_text(todo.text, hits)
        if red != todo.text:
            todo.text = red
            changed = True
    return ScrubReport(changed, hits)


def scrub_jsonl_file(path: Path) -> ScrubReport:
    """Redact every line of a JSONL file in place (the compaction
    archive `<id>.transcript.archive.jsonl`). The patterns never span
    JSON structural characters, so a redacted line stays valid JSON."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return _NOTHING
    lines = text.splitlines(keepends=True)
    hits: dict[str, int] = {}
    changed = False
    for index, line in enumerate(lines):
        red = redact_text(line, hits)
        if red != line:
            lines[index] = red
            changed = True
    if not changed:
        return ScrubReport(False, hits)
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        os.replace(tmp, path)
        return ScrubReport(True, hits)
    except OSError:
        if tmp is not None and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return _NOTHING
