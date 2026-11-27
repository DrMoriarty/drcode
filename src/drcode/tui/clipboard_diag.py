"""Clipboard delivery diagnostics for remote (ssh / tmux) sessions.

A select-to-copy in the TUI writes OSC 52 (``copy_to_clipboard``), but on
the common ssh+tmux path the chain tmux → ssh → local terminal is easily
broken (tmux ``set-clipboard off``, a local terminal without OSC 52
support such as macOS Terminal.app). The old toast said "✓ copied"
unconditionally — a promise the TUI cannot keep over a remote hop.

This module answers ONE question as well as it can from inside the
session: will an OSC 52 write actually reach the local clipboard?

- **tmux check** — ``tmux show-options -gv set-clipboard`` (socket from the
  ``TMUX`` env, 2 s timeout, never fatal): ``off`` is a definite break, the
  result is cached for the session (no tmux call per selection).
- **Positive probe** — a one-shot OSC 52 *read* query (``\\x1b]52;c;?`` +
  BEL) written to the terminal, with a ~300 ms wait for the reply. A reply
  proves the whole write chain works (a terminal that can read back can
  also write). Silence is UNKNOWN — most terminals (kitty by default,
  xterm) refuse clipboard reads but accept writes, so silence must never
  be reported as breakage.

Everything here is best-effort: no diagnostic may block the UI for more
than the explicit timeouts, and none may ever crash a copy.

The reply capture hooks Textual's input parsing: Textual's ``XTermParser``
has no OSC handling and would reissue an OSC 52 reply as garbage key
events. ``install_parser_hook`` wraps the parser classes the drivers
instantiate so replies are diverted into a queue while a probe is armed;
the hook is inert otherwise (data passes through untouched).
"""

from __future__ import annotations

import importlib
import os
import queue
import re
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from textual.app import App

__all__ = [
    "DiagState",
    "arm_probe",
    "check_tmux_set_clipboard",
    "detect_remote",
    "disarm_probe",
    "install_parser_hook",
    "poll_probe_reply",
    "send_osc52_query",
    "toast_message",
]

PROBE_WINDOW_SECONDS = 0.3
"""How long the probe waits for a terminal reply (short — the reply, when
it comes at all, arrives within milliseconds of the query)."""

_TMUX_TIMEOUT_SECONDS = 2.0

OSC52_QUERY = "\x1b]52;c;?\x07"
"""OSC 52 clipboard READ query: "give me the (base64) system clipboard"."""


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def detect_remote(env: Mapping[str, str] | None = None) -> bool:
    """Whether this session sits behind ssh and/or tmux — i.e. whether the
    terminal the user reads is a network hop away from this process.

    ``TERM_PROGRAM`` is deliberately NOT consulted: it does not survive
    ssh (same caveat as ``tui/sync_output.py``), so a local terminal
    cannot be identified from inside the session either way.
    """
    env = os.environ if env is None else env
    return bool(env.get("TMUX") or env.get("SSH_CONNECTION") or env.get("SSH_TTY"))


def check_tmux_set_clipboard(env: Mapping[str, str] | None = None) -> str | None:
    """The tmux server's ``set-clipboard`` value ("on"/"off"/"external").

    The socket comes from the ``TMUX`` env (tmux resolves it itself); the
    call is time-boxed and never fatal — a missing/defunct tmux binary or
    a dead server means UNKNOWN (None), not an error. Returns None
    outside tmux as well.
    """
    env = os.environ if env is None else env
    if not env.get("TMUX"):
        return None
    try:
        proc = subprocess.run(
            ["tmux", "show-options", "-gv", "set-clipboard"],
            capture_output=True,
            text=True,
            timeout=_TMUX_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip().lower()
    return value if value in ("on", "off", "external") else None


# ---------------------------------------------------------------------------
# OSC 52 reply capture (Textual parser hook)
# ---------------------------------------------------------------------------

_OSC52_REPLY_RE = re.compile(r"\x1b\]52;c;([A-Za-z0-9+/=]*)(?:\x07|\x1b\\)")
"""A terminal's reply to the read query: OSC 52 with the base64 payload
(possibly empty), terminated by BEL or ST."""

_OSC52_START = "\x1b]52;c;"

_reply_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
_armed_until = 0.0
"""monotonic time until which replies are captured; the hook is inert
outside the window, so ordinary input streams pass through untouched."""

_hook_installed = False


def arm_probe(window: float = 5.0) -> None:
    """Arm reply capture for ``window`` seconds (covers the ~300 ms verdict
    window plus slack for a slow last-hop reply)."""
    global _reply_queue, _armed_until
    _reply_queue = queue.SimpleQueue()
    _armed_until = time.monotonic() + window


def disarm_probe() -> None:
    """Close the capture window (late replies then reach the parser as
    they would without the hook — harmless, the probe is already
    resolved)."""
    global _armed_until
    _armed_until = 0.0


def poll_probe_reply() -> str | None:
    """The captured reply's base64 payload (possibly empty string), or None
    while nothing has arrived yet. Thread-safe (the parser feeds from the
    driver's input thread, the probe polls from the app loop)."""
    try:
        return _reply_queue.get_nowait()
    except queue.Empty:
        return None


def _capture_active() -> bool:
    return time.monotonic() < _armed_until


def _extract_osc52_replies(data: str) -> str:
    """Strip complete OSC 52 replies out of ``data``, queueing their
    payloads; hold an unterminated reply prefix at the end (a reply split
    across os.read chunks) back for the next feed. Returns the cleaned
    data to hand to the wrapped parser."""
    data = _pending_tail[0] + data
    _pending_tail[0] = ""
    cleaned: list[str] = []
    pos = 0
    while True:
        match = _OSC52_REPLY_RE.search(data, pos)
        if match is None:
            break
        _reply_queue.put(match.group(1))
        cleaned.append(data[pos : match.start()])
        pos = match.end()
    cleaned.append(data[pos:])
    result = "".join(cleaned)

    # A reply cut mid-payload: keep the tail so the terminator (next feed)
    # completes it instead of leaking half a sequence into the key parser.
    start = result.rfind(_OSC52_START)
    if start != -1 and start + len(_OSC52_START) < len(result):
        rest = result[start + len(_OSC52_START) :]
        if "\x07" not in rest and "\x1b" not in rest and len(rest) <= 4096:
            _pending_tail[0] = result[start:]
            return result[:start]
    _pending_tail[0] = ""
    return result


_pending_tail: list[str] = [""]
"""Reassembly slot for a reply split across feeds (single probe at a
time, one slot is enough). Set/cleared only from the input thread."""


def install_parser_hook() -> None:
    """Wrap the XTermParser classes Textual's drivers instantiate so OSC 52
    replies are diverted instead of being reissued as garbage key events.
    Idempotent; drivers that fail to import (wrong platform) are skipped.

    Must run BEFORE the driver's input thread creates its parser — i.e.
    from ``DrCodeApp.__init__``, before ``run()`` starts application mode.
    """
    global _hook_installed
    if _hook_installed:
        return
    _hook_installed = True
    for module_name in (
        "textual.drivers.linux_driver",
        "textual.drivers.linux_inline_driver",
        "textual.drivers.web_driver",
        "textual.drivers.win32",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001, S112 - best-effort hook, never fatal
            continue
        base = getattr(module, "XTermParser", None)
        if base is None or getattr(base, "_osc52_hook", False):
            continue
        setattr(module, "XTermParser", _make_hooked_parser(base))  # noqa: B010


def _make_hooked_parser(base: type[Any]) -> type[Any]:
    class HookedParser(base):  # type: ignore[misc, valid-type]
        _osc52_hook = True

        def feed(self, data: str) -> Any:
            if not _capture_active():
                # Inert outside the probe window: any reply fragment held
                # from the window is flushed verbatim, data untouched.
                data = _pending_tail[0] + data
                _pending_tail[0] = ""
                return super().feed(data)
            return super().feed(_extract_osc52_replies(data))

    HookedParser.__name__ = f"OSC52Hooked{base.__name__}"
    return HookedParser


def send_osc52_query(app: App) -> bool:
    """Write the OSC 52 read query to the terminal (same driver write path
    ``copy_to_clipboard`` uses). Returns False when there is no live
    driver."""
    driver = app._driver
    if driver is None:
        return False
    try:
        driver.write(OSC52_QUERY)
    except Exception:  # noqa: BLE001 - best-effort probe, never fatal
        return False
    return True


# ---------------------------------------------------------------------------
# Session state / verdict
# ---------------------------------------------------------------------------


@dataclass
class DiagState:
    """Per-session clipboard diagnostics cache (one per DrCodeApp).

    The tmux check runs once (in a worker — never on the UI thread), the
    probe is one-shot per session, and both results are cached so a text
    selection never pays for a diagnostic.
    """

    remote: bool = field(default_factory=detect_remote)
    tmux_value: str | None = None
    tmux_checked: bool = False
    probe_started: bool = False
    probe_done: bool = False
    probe_ok: bool = False

    def run_tmux_check(self) -> None:
        """Run the (blocking, time-boxed) tmux check — call from a worker."""
        self.tmux_value = check_tmux_set_clipboard()
        self.tmux_checked = True

    def verdict(self) -> str:
        """One of:

        - ``"local"`` — no ssh/tmux: delivery is direct, toast as always.
        - ``"tmux-off"`` — known breakage: tmux swallows OSC 52.
        - ``"confirmed"`` — the probe reply proved the chain works.
        - ``"unknown"`` — remote, but nothing proved or disproved
          delivery (silence is never reported as breakage).
        """
        if not self.remote:
            return "local"
        if self.tmux_checked and self.tmux_value == "off":
            return "tmux-off"
        if self.probe_ok:
            return "confirmed"
        return "unknown"

    @property
    def tmux_off(self) -> bool:
        return self.tmux_checked and self.tmux_value == "off"


def toast_message(verdict: str, preview: str) -> str:
    """The copy toast text for a diagnostic verdict. Pure so tests can pin
    the wording; ``preview`` is already markup-escaped by the caller."""
    if verdict == "tmux-off":
        return (
            "⚠ копирование из TUI не дойдёт до локального буфера: "
            "tmux set-clipboard off — добавьте set -g set-clipboard on "
            "в ~/.tmux.conf и перезагрузите конфиг"
        )
    if verdict == "unknown":
        return (
            f"✓ скопировано через OSC 52: {preview}\n"
            "если локальный буфер пуст — проверьте tmux set-clipboard "
            "и поддержку OSC 52 в терминале"
        )
    # "local" and "confirmed": delivery is direct or proven — confident.
    return f"✓ copied: {preview}"
