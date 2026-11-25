"""UI-neutral rich markup builders shared by the chat blocks (and reusable
by any future frontend).

Split out of the former monolithic `tui/blocks.py`: everything here is
plain rich (Text / Console rendering) with no Textual dependency — the
Textual widget classes that USE these builders stay in `drcode.tui.blocks`.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from collections.abc import Mapping
from datetime import datetime

from rich.cells import cell_len
from rich.console import Console
from rich.markdown import Markdown
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

# Spinner frame sets for the "work in progress" animation shown in a
# reasoning block while the model is thinking, in a tool-call block while
# the tool is executing and in the merge screens' busy/progress lines.
# `SPINNER_FRAMES` (below) picks the set for the ACTUAL terminal at import
# time — the braille glyphs only render when the session's character set
# carries them (see spinner_frames), single-byte terminals get the plain
# `/ - \ |` wheel.
BRAILLE_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
ASCII_SPINNER_FRAMES = "/-\\|"


def _locale_is_utf8(locale: str) -> bool:
    """Whether the locale's character set can carry the braille pattern
    block (U+2800–U+28FF): only the UTF-8 family can — C/POSIX, KOI8-*,
    ISO-8859-*, CP125x and the like are single-byte charsets.

    The character set is the trailing component after the last '.' or '@'
    (en_US.UTF-8, C.utf8, de_DE@UTF-8); a bare 'C'/'POSIX' has none at all.
    """
    for sep in (".", "@"):
        head, _, tail = locale.rpartition(sep)
        if head and tail.strip().lower().replace("-", "").replace("_", "") == "utf8":
            return True
    return False


def _braille_encodable(encoding: str) -> bool:
    """Whether a stream with `encoding` can carry the braille frames
    ("" or an unknown codec fails closed to the ASCII set)."""
    if not encoding:
        return False
    try:
        BRAILLE_SPINNER_FRAMES.encode(encoding)
    except (LookupError, UnicodeEncodeError, ValueError):
        return False
    return True


def spinner_frames(
    environ: Mapping[str, str] | None = None,
    output_encoding: str | None = None,
) -> str:
    """Pick the spinner frame set for the ACTUAL terminal.

    CPython forces UTF-8 mode itself when the locale is C/POSIX, so the
    stream encodings are NOT a reliable signal that the terminal renders
    Unicode — the session's locale environment is. Order:

    1. `DRCODE_SPINNER=ascii|braille` forces either side explicitly (the
       escape hatch for a UTF-8 session whose terminal font lacks the
       braille block — it would show empty boxes);
    2. the POSIX resolution of LC_CTYPE — `LC_ALL`, then `LC_CTYPE`, then
       `LANG`, the first non-empty wins: a UTF-8 character set means the
       terminal speaks Unicode; anything else (C/POSIX, KOI8-*,
       ISO-8859-*, CP125x, …) means a single-byte charset, where the
       braille bytes render as stray letters and marks — the plain
       `/ - \\ |` wheel is used there;
    3. with no locale information at all (Windows), the encoding of the
       stream Textual renders its frames into (`sys.__stderr__`) decides:
       braille only when it can encode them.

    The safe direction is the ASCII set: it renders on every terminal,
    while a wrong braille choice is the visible garble.
    """
    env = os.environ if environ is None else environ
    override = (env.get("DRCODE_SPINNER") or "").strip().lower()
    if override in ("ascii", "plain"):
        return ASCII_SPINNER_FRAMES
    if override == "braille":
        return BRAILLE_SPINNER_FRAMES
    for var in ("LC_ALL", "LC_CTYPE", "LANG"):
        value = (env.get(var) or "").strip()
        if value:
            return (
                BRAILLE_SPINNER_FRAMES
                if _locale_is_utf8(value)
                else ASCII_SPINNER_FRAMES
            )
    if output_encoding is None:
        stream = sys.__stderr__
        output_encoding = str(getattr(stream, "encoding", "") or "")
    return (
        BRAILLE_SPINNER_FRAMES
        if _braille_encodable(output_encoding)
        else ASCII_SPINNER_FRAMES
    )


# The set the live TUI uses — resolved for this process's terminal at
# import time (a session's character set does not change mid-run).
SPINNER_FRAMES = spinner_frames()
SPINNER_INTERVAL = 0.12

# Task-id mention: '#' + 8 hex chars (models._short_id() = uuid4().hex[:8]).
# The lookarounds reject ids inside longer hex tokens: a 9+-char run
# (#abcdef123 — the regex can't backtrack, 8 hex are mandatory), a 6- or
# 12-digit CSS color (#aabbcc, #AABBCCDD) and an id glued to a preceding
# hex char or another '#' (##69c8bd30). A non-hex word char before the '#'
# (e.g. "x#69c8bd30") still matches — the mention is unambiguous there.
TASK_MENTION_RE = re.compile(r"(?<![0-9A-Fa-f#])#([0-9A-Fa-f]{8})(?![0-9A-Fa-f])")


def _mention_click_style(task_id: str) -> Style:
    """Style for a clickable task-id mention.

    The click action must ride in the style's META (a `Style` object): rich's
    markup parser rejects `@click=…` as a style name, and Textual's
    `App._broker_event` dispatches the action from `event.style.meta` — a
    plain-colored span would render but never fire. Namespaced `app.` — the
    default namespace would resolve against the widget, not the app.
    """
    return Style.parse("cyan underline") + Style.from_meta(
        {"@click": f"app.open_task('{task_id}')"}
    )


def linkify(text: Text) -> Text:
    """Stylize task-id mentions (#69c8bd30) in a rendered Text as clickable
    links that open the mentioned task (app.open_task).

    Applied to the RENDERED text (after markdown/escape), never to the raw
    source. Membership in `manager.tasks` is deliberately not checked here:
    the text may have been rendered (or streamed) before the mentioned task
    existed (tester-created bug reports) and ids survive task deletion in
    old transcripts — unknown ids are caught at click time with a toast.
    """
    for match in TASK_MENTION_RE.finditer(text.plain):
        task_id = match.group(1).lower()
        text.stylize(_mention_click_style(task_id), match.start(), match.end())
    return text


def stamp_label(at: datetime | None) -> str:
    """Right-side header stamp for a block, "" when no time is known."""
    if at is None:
        return ""
    return at.astimezone().strftime("%H:%M:%S")


def append_stamp(header: Text, stamp: str, width: int) -> None:
    """Push `stamp` to the right edge of a header line `width` cells wide.

    Just enough padding is inserted between the header text and the stamp;
    when the terminal is too narrow for both, the stamp still renders,
    separated by the minimum gap (it may wrap/clip, which beats vanishing).
    """
    if not stamp:
        return
    pad = width - cell_len(header.plain) - len(stamp) - 1
    header.append(" " * max(pad, 1), "dim")
    header.append(stamp, "dim italic")


def markdown_text(text: str, width: int) -> Text:
    """Render markdown to a styled Text at a fixed width.

    A rich Markdown renderable can't take part in Textual's mouse selection:
    RichVisual strips carry no "offset" meta, so a drag inside the widget
    degrades to a select-all of the whole block, and no selection highlight
    is ever painted. A Text is promoted to Content, which supports both.
    """
    console = Console(width=width, legacy_windows=False, file=io.StringIO())
    options = console.options.update(width=width, highlight=False)
    result = Text()
    for segment in console.render(Markdown(text, code_theme="ansi_dark"), options):
        result.append(segment.text, segment.style)
    return result


def streamable_markdown(text: str, width: int) -> Text:
    """Render markdown on the longest complete prefix of *text*.

    Rich parses markdown as a whole document, so it can't be streamed token
    by token. Instead, format only the portion of *text* that ends at a
    newline and sits outside any *unclosed* fenced code block (```/~~~), and
    leave the partial remainder as raw text. Already-received complete
    fragments get live highlighting; the still-producing tail stays plain —
    an acceptable trade-off while the model is still streaming.
    """
    # A fenced code block is tracked via its opening/closing fences (runs of
    # three or more backticks or tildes): while the stack is non-empty we are
    # inside an open block, so a newline there is not a safe formatting
    # boundary. Inline code spans (1-2 backticks) are ignored here and left
    # to Rich (an unterminated one renders as literal text).
    fences: list[str] = []
    i, n = 0, len(text)
    last_safe = -1
    while i < n:
        c = text[i]
        if c in ("`", "~"):
            run = 1
            while i + run < n and text[i + run] == c:
                run += 1
            if run >= 3:
                if fences and fences[0] == c:
                    fences.pop()
                elif not fences:
                    fences.append(c)
                # A mismatched-char fence inside a block is literal code.
        elif c == "\n":
            if not fences:
                last_safe = i
        i += 1

    if last_safe == -1:
        return Text(text)

    result = markdown_text(text[: last_safe + 1], width)
    result.append(text[last_safe + 1 :])
    return result


# Width the JSON is lexed at: large enough that pygments never hard-wraps a
# line (the widget soft-wraps instead), so the highlighted Text is
# width-independent and can be cached for the block's lifetime.
JSON_RENDER_WIDTH = 10_000


def json_text(text: str) -> Text:
    """Syntax-highlight a JSON snippet (tool-call arguments).

    Valid JSON is re-indented for readability (display only — the stored
    transcript keeps the original bytes) and colored via pygments: keys,
    strings, numbers and punctuation each get their own ansi color.
    Anything unparsable (a truncated or non-JSON argument string) falls
    back to plain text. Rendered through a detached Console into a plain
    Text — the same trick as markdown_text — so Textual's mouse selection
    still works over the block.
    """
    try:
        parsed = json.loads(text)
    except ValueError:
        return Text(text)
    if isinstance(parsed, (dict, list)):
        text = json.dumps(parsed, indent=2, ensure_ascii=False)
    console = Console(width=JSON_RENDER_WIDTH, legacy_windows=False, file=io.StringIO())
    options = console.options.update(width=JSON_RENDER_WIDTH)
    result = Text()
    for segment in console.render(Syntax(text, "json", theme="ansi_dark"), options):
        result.append(segment.text, segment.style)
    if result.plain.endswith("\n"):
        result.right_crop(1)
    return result
