"""Tests for the DrCode Desktop shell (desktop/).

Headless checks of the pieces the shell owns: recent-projects storage,
PTY session semantics (spawn/resize/exit), and the aiohttp ws/pty bridge.
The pywebview window and the JS terminal client are exercised manually
(see desktop/README.md — smoke run) since they need a GUI session.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import aiohttp
import pytest

# The desktop shell's modules live in desktop/ and are written as plain
# top-level modules (they are also loaded as scripts by PyInstaller), so
# put that directory on sys.path for the import.
DESKTOP_DIR = Path(__file__).resolve().parent.parent / "desktop"
if str(DESKTOP_DIR) not in sys.path:
    sys.path.insert(0, str(DESKTOP_DIR))

import projects  # type: ignore[import-not-found]
from pty_session import (  # type: ignore[import-not-found]
    PtySession,
    child_environment,
    payload_argv,
)
from server import ShellContext, run_server  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# projects.py — recents storage
# ---------------------------------------------------------------------------


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(d))
    return d


def test_recents_roundtrip(config_dir: Path) -> None:
    assert projects.load_recents() == []
    recents = projects.add_recent("/tmp/alpha")
    assert recents[0]["project"] == "/tmp/alpha"
    assert (config_dir / "drcode-desktop.json").is_file()

    projects.add_recent("/tmp/beta")
    projects.add_recent("/tmp/alpha")  # moves to front, deduped
    recents = projects.load_recents()
    assert [r["project"] for r in recents] == ["/tmp/alpha", "/tmp/beta"]


def test_recents_capped(config_dir: Path) -> None:
    for i in range(20):
        projects.add_recent(f"/tmp/p{i}")
    recents = projects.load_recents()
    assert len(recents) == projects.MAX_RECENTS
    assert recents[0]["project"] == "/tmp/p19"


def test_recents_corrupt_file(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "drcode-desktop.json").write_text("{not json", encoding="utf-8")
    assert projects.load_recents() == []


def test_recents_drops_bad_entries(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "drcode-desktop.json").write_text(
        json.dumps(
            {"recents": [{"project": ""}, {"nope": 1}, "str", {"project": "/ok"}]}
        ),
        encoding="utf-8",
    )
    recents = projects.load_recents()
    assert [r["project"] for r in recents] == ["/ok"]


# ---------------------------------------------------------------------------
# pty_session.py
# ---------------------------------------------------------------------------


def test_child_environment_forces_terminal_caps() -> None:
    env = child_environment({"EXTRA": "1"})
    assert env["TERM"] == "xterm-256color"
    assert env["COLORTERM"] == "truecolor"
    assert env["EXTRA"] == "1"
    assert "COLUMNS" not in env and "LINES" not in env


def test_payload_argv() -> None:
    assert payload_argv("/bin/drcode") == ["/bin/drcode"]
    assert payload_argv("/bin/drcode", ["--flag"]) == ["/bin/drcode", "--flag"]


async def test_pty_session_echo_and_exit(tmp_path: Path) -> None:
    """Run a python child through the PTY: input flows in, output flows out,
    the exit code is delivered exactly once."""
    chunks: list[bytes] = []
    exits: list[int] = []
    done = asyncio.Event()

    session = PtySession(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; sys.stdout.write('ready\\n'); sys.stdout.flush(); sys.exit(7)",
        ],
        tmp_path,
        on_output=chunks.append,
        on_exit=lambda code: (exits.append(code), done.set()),
    )
    await session.start(initial_size=(24, 80))
    await asyncio.wait_for(done.wait(), timeout=10)

    assert exits == [7]
    assert session.exit_code == 7
    assert b"ready" in b"".join(chunks)


async def test_pty_session_resize(tmp_path: Path) -> None:
    """ioctl(TIOCSWINSZ) reaches the child: a size-reporting loop observes
    the initial 30x100 window become 40x120 after session.resize()."""
    chunks: list[bytes] = []
    done = asyncio.Event()

    session = PtySession(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import os, sys, time\n"
                "for _ in range(40):\n"
                "    size = os.get_terminal_size()\n"
                "    sys.stdout.write(f'size:{size.columns}x{size.lines}\\n')\n"
                "    sys.stdout.flush()\n"
                "    time.sleep(0.1)\n"
            ),
        ],
        tmp_path,
        on_output=chunks.append,
        on_exit=lambda _code: done.set(),
    )
    await session.start(initial_size=(30, 100))

    deadline = asyncio.get_event_loop().time() + 15
    while b"size:100x30" not in b"".join(chunks):
        assert asyncio.get_event_loop().time() < deadline, "initial size not reported"
        await asyncio.sleep(0.05)

    session.resize(40, 120)
    deadline = asyncio.get_event_loop().time() + 15
    while b"size:120x40" not in b"".join(chunks):
        assert asyncio.get_event_loop().time() < deadline, "resized size not reported"
        await asyncio.sleep(0.05)

    session.terminate()
    await asyncio.wait_for(done.wait(), timeout=10)


async def test_pty_session_write_keeps_streaming(tmp_path: Path) -> None:
    """Bytes written before/while the child reads them all arrive (cat echoes)."""
    chunks: list[bytes] = []
    done = asyncio.Event()

    session = PtySession(
        ["cat"],
        tmp_path,
        on_output=chunks.append,
        on_exit=lambda _code: done.set(),
    )
    await session.start(initial_size=(24, 80))
    payload = "привёт мир — cyrillic IME payload ok\n".encode()
    session.write(payload)
    session.write(payload)

    # The pty line discipline echoes LF as CRLF (ONLCR) — normalize.
    deadline = asyncio.get_event_loop().time() + 10
    echoed = lambda: b"".join(chunks).replace(b"\r\n", b"\n")
    while echoed().count(payload) < 2:
        assert asyncio.get_event_loop().time() < deadline, "echo incomplete"
        await asyncio.sleep(0.05)

    # EOF for cat: Ctrl-D (0x04) in canonical mode.
    session.write(b"\x04\x04")
    await asyncio.wait_for(done.wait(), timeout=10)
    assert session.exit_code == 0


# ---------------------------------------------------------------------------
# server.py — the ws/pty bridge
# ---------------------------------------------------------------------------


async def _start_server(payload: str, tmp_path: Path):
    ctx = ShellContext(payload)
    runner, base_url = await run_server(ctx)
    return ctx, runner, base_url


async def test_server_serves_web_and_guards_ws(tmp_path: Path) -> None:
    _ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    try:
        async with aiohttp.ClientSession() as http, http.get(f"{base_url}/") as resp:
            assert resp.status == 200
            body = await resp.text()
            assert "term.js" in body
        async with (
            aiohttp.ClientSession() as http,
            http.get(f"{base_url}/static/term.js") as resp,
        ):
            assert resp.status == 200
        async with (
            aiohttp.ClientSession() as http,
            http.get(f"{base_url}/ws?token=WRONG") as resp,
        ):
            assert resp.status == 403
    finally:
        await runner.cleanup()


async def test_server_ws_bridge_roundtrip(tmp_path: Path) -> None:
    """Full bridge: start a python child over ws, see its PTY output as
    binary frames, then its exit as a JSON event."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    project = tmp_path / "proj"
    project.mkdir()

    # The "payload" is a plain python one-shot printing a marker (spawned
    # directly, so it needs a shebang + exec bit, like the real bundle).
    script = tmp_path / "payload.py"
    script.write_text(
        "#!/usr/bin/env python3\nprint('BRIDGE-OK-9137')\n", encoding="utf-8"
    )
    script.chmod(0o755)
    ctx.payload = str(script)

    try:
        async with (
            aiohttp.ClientSession() as http,
            http.ws_connect(f"{base_url}/ws?token={ctx.token}") as ws,
        ):
            await ws.send_json(
                {"type": "start", "project": str(project), "rows": 24, "cols": 80}
            )

            saw_marker = False
            exit_code = None
            for _ in range(200):
                msg = await asyncio.wait_for(ws.receive(), timeout=5)
                if msg.type == aiohttp.WSMsgType.BINARY:
                    if b"BRIDGE-OK-9137" in msg.data:
                        saw_marker = True
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(msg.data)
                    if event.get("event") == "exited":
                        exit_code = event["code"]
                        break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
            assert saw_marker, "child PTY output never reached the ws"
            assert exit_code == 0
    finally:
        await runner.cleanup()


# ---------------------------------------------------------------------------
# web/term.js — welcome-screen pick-project handler (static, headless)
# ---------------------------------------------------------------------------

# pywebview injects its JS bridge asynchronously AFTER the DOM loads, so an
# early click on "choose project" used to hit api("choose_project") == null
# and report the native folder dialog as unavailable even though the bridge
# was about to appear. The handler must retry for the bridge (like the
# recents get_info loop) and show the error only after the budget is
# exhausted. The JS itself needs a GUI session — checked statically here.


def test_term_js_pick_project_retries_before_error() -> None:
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")
    start = js.index('elPick.addEventListener("click"')
    end = js.index('document.getElementById("btn-restart")', start)
    handler = js[start:end]

    # The unavailability message is only reachable past the retry loop.
    msg_at = handler.index("Native folder dialog is unavailable")
    assert handler.index("setInterval") < msg_at
    assert handler.index("attempts > 40") < msg_at
    # Retry polls the same bridge call the click originally missed.
    assert handler.count('api("choose_project")') == 2
    # A second click while a retry is running does not stack another loop.
    assert "dataset.waiting" in handler


async def test_server_start_rejects_missing_project(tmp_path: Path) -> None:
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    try:
        async with (
            aiohttp.ClientSession() as http,
            http.ws_connect(f"{base_url}/ws?token={ctx.token}") as ws,
        ):
            await ws.send_json(
                {
                    "type": "start",
                    "project": str(tmp_path / "nope"),
                    "rows": 24,
                    "cols": 80,
                }
            )
            msg = await asyncio.wait_for(ws.receive(), timeout=5)
            assert msg.type == aiohttp.WSMsgType.TEXT
            event = json.loads(msg.data)
            assert event["event"] == "error"
            assert "not a directory" in event["message"]
    finally:
        await runner.cleanup()
