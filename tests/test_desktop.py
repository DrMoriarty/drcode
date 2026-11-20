"""Tests for the DrCode Desktop shell (desktop/).

Headless checks of the pieces the shell owns: recent-projects storage,
PTY session semantics (spawn/resize/exit), and the aiohttp ws/pty bridge.
The pywebview window and the JS terminal client are exercised manually
(see desktop/README.md — smoke run) since they need a GUI session.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import pytest

# The desktop shell's modules live in desktop/ and are written as plain
# top-level modules (they are also loaded as scripts by PyInstaller), so
# put that directory on sys.path for the import.
DESKTOP_DIR = Path(__file__).resolve().parent.parent / "desktop"
if str(DESKTOP_DIR) not in sys.path:
    sys.path.insert(0, str(DESKTOP_DIR))

import fonts  # type: ignore[import-not-found]
import projects  # type: ignore[import-not-found]
from pty_session import (  # type: ignore[import-not-found]
    PtySession,
    child_environment,
    payload_argv,
)
from server import (  # type: ignore[import-not-found]
    COALESCE_WINDOW_SECONDS,
    FRAME_MAX_BYTES,
    ShellContext,
    run_server,
)

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


def test_recents_remove(config_dir: Path) -> None:
    projects.add_recent("/tmp/alpha", title="alpha")
    projects.add_recent("/tmp/beta")
    recents = projects.remove_recent("/tmp/alpha")
    assert [r["project"] for r in recents] == ["/tmp/beta"]
    assert [r["project"] for r in projects.load_recents()] == ["/tmp/beta"]
    # Removing an unknown path is a no-op; removing everything empties.
    assert [r["project"] for r in projects.remove_recent("/nope")] == ["/tmp/beta"]
    assert projects.remove_recent("/tmp/beta") == []
    assert projects.load_recents() == []
    # The other config keys survive the removal write.
    projects.save_font_settings("Fira Code", 14)
    projects.add_recent("/tmp/alpha")
    projects.remove_recent("/tmp/alpha")
    assert projects.load_font_settings() == {
        "font_family": "Fira Code",
        "font_size": 14,
    }


def test_desktop_api_remove_recent_bridge(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DesktopApi.remove_recent delegates to projects.remove_recent."""
    try:
        import app as app_mod  # type: ignore[import-not-found]
        import webview  # noqa: F401
    except Exception:  # noqa: BLE001 - webview missing / no GUI libs
        pytest.skip("pywebview desktop group not installed")

    projects.add_recent("/tmp/alpha")
    projects.add_recent("/tmp/beta")
    api = app_mod.DesktopApi({}, None, Path("/dev/null"))
    recents = api.remove_recent("/tmp/alpha")
    assert [r["project"] for r in recents] == ["/tmp/beta"]
    monkeypatch.setattr(
        app_mod.projects_mod, "remove_recent", lambda project: [{"project": "stub"}]
    )
    assert api.remove_recent("whatever") == [{"project": "stub"}]


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
# packaging.py — seed lib alternatives (WebKitGTK 4.0 vs 4.1)
# ---------------------------------------------------------------------------


def _load_packaging(monkeypatch: pytest.MonkeyPatch, lib_dir: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "drcode_desktop_packaging",
        Path(__file__).resolve().parent.parent / "desktop" / "packaging.py",
    )
    assert spec is not None and spec.loader is not None
    packaging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packaging)
    monkeypatch.setattr(packaging, "SEED_SEARCH_DIRS", [lib_dir])
    return packaging


def _make_seed_dir(tmp_path: Path, webkit: str, jsc: str) -> Path:
    lib_dir = tmp_path / "libs"
    lib_dir.mkdir()
    for name in (
        webkit,
        jsc,
        "libgtk-3.so.0",
        "libsoup-3.0.so.0",
        "libgirepository-1.0.so.1",
    ):
        (lib_dir / name).touch()
    return lib_dir


def test_seed_libs_accept_webkit41(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packaging = _load_packaging(
        monkeypatch,
        _make_seed_dir(
            tmp_path, "libwebkit2gtk-4.1.so.0", "libjavascriptcoregtk-4.1.so.0"
        ),
    )
    names = {p.name for p in packaging.find_seed_libs()}
    assert "libwebkit2gtk-4.1.so.0" in names


def test_seed_libs_accept_webkit40(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packaging = _load_packaging(
        monkeypatch,
        _make_seed_dir(
            tmp_path, "libwebkit2gtk-4.0.so.0", "libjavascriptcoregtk-4.0.so.0"
        ),
    )
    names = {p.name for p in packaging.find_seed_libs()}
    assert "libwebkit2gtk-4.0.so.0" in names


def test_seed_libs_missing_all_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packaging = _load_packaging(monkeypatch, tmp_path)  # empty dir
    with pytest.raises(SystemExit, match="WebKitGTK seed libs not found"):
        packaging.find_seed_libs()


# ---------------------------------------------------------------------------
# projects.py — font settings storage
# ---------------------------------------------------------------------------


def test_font_settings_defaults_when_missing(config_dir: Path) -> None:
    assert projects.load_font_settings() == {"font_family": None, "font_size": None}


def test_font_settings_roundtrip(config_dir: Path) -> None:
    saved = projects.save_font_settings('"Fira Code"', 17)
    assert saved == {"font_family": '"Fira Code"', "font_size": 17}
    assert projects.load_font_settings() == saved
    assert (config_dir / "drcode-desktop.json").is_file()


def test_font_settings_none_resets_to_defaults(config_dir: Path) -> None:
    projects.save_font_settings('"Fira Code"', 17)
    saved = projects.save_font_settings(None, None)
    assert saved == {"font_family": None, "font_size": None}
    assert projects.load_font_settings() == saved


def test_font_settings_size_clamped(config_dir: Path) -> None:
    assert projects.save_font_settings("X", 100)["font_size"] == projects.MAX_FONT_SIZE
    assert projects.load_font_settings()["font_size"] == projects.MAX_FONT_SIZE
    assert projects.save_font_settings("X", 1)["font_size"] == projects.MIN_FONT_SIZE
    # Junk strings / bools / NaN-ish garbage → null, never a crash.
    assert projects.save_font_settings("X", "big")["font_size"] is None
    assert projects.save_font_settings("X", True)["font_size"] is None
    assert projects.save_font_settings("X", None)["font_size"] is None


def test_font_settings_blank_family_becomes_null(config_dir: Path) -> None:
    assert projects.save_font_settings("   ", 13)["font_family"] is None
    assert projects.load_font_settings()["font_family"] is None


def test_font_settings_invalid_stored_values_dropped(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "drcode-desktop.json").write_text(
        json.dumps(
            {
                "font_family": 42,
                "font_size": "13",
                "recents": [{"project": "/tmp/alpha", "last_used": 1}],
            }
        ),
        encoding="utf-8",
    )
    assert projects.load_font_settings() == {"font_family": None, "font_size": None}


def test_font_settings_survive_recents_write_and_vice_versa(config_dir: Path) -> None:
    """The two halves of the desktop config coexist: writing one never
    clobbers the other."""
    projects.save_font_settings('"SF Mono"', 15)
    projects.add_recent("/tmp/alpha")
    settings = projects.load_font_settings()
    assert settings == {"font_family": '"SF Mono"', "font_size": 15}
    assert [r["project"] for r in projects.load_recents()] == ["/tmp/alpha"]

    projects.save_font_settings("Consolas", 9)
    recents = projects.load_recents()
    assert [r["project"] for r in recents] == ["/tmp/alpha"]


# ---------------------------------------------------------------------------
# fonts.py — system font enumeration for the font dialog
# ---------------------------------------------------------------------------


@pytest.fixture
def fonts_fresh():
    """Isolate the module-level font cache between tests."""
    fonts.refresh_cache()
    yield fonts
    fonts.refresh_cache()


def test_linux_font_list_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """fc-list output: one font per line, family aliases comma-separated."""

    class FakeProc:
        returncode = 0
        stdout = b"Fira Code\nUbuntu Mono,Ubuntu Sans Mono\nLiberation Mono\n"

    monkeypatch.setattr(fonts.subprocess, "run", lambda *a, **kw: FakeProc())
    names = fonts._linux_family_names()
    assert names is not None
    for family in ("Fira Code", "Ubuntu Mono", "Ubuntu Sans Mono", "Liberation Mono"):
        assert family in names


def test_linux_missing_fc_list_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("fc-list")

    monkeypatch.setattr(fonts.subprocess, "run", fake_run)
    assert fonts._linux_family_names() is None


def test_available_fonts_fail_open_on_probe_error(
    monkeypatch: pytest.MonkeyPatch, fonts_fresh
) -> None:
    """A broken platform probe must yield None (unknown), never an empty
    set — the JS side treats unknown as 'no filtering' (fail-open)."""
    monkeypatch.setattr(fonts.sys, "platform", "linux")

    def boom() -> list[str]:
        raise RuntimeError("no fonts for you")

    monkeypatch.setattr(fonts, "_linux_family_names", boom)
    assert fonts.available_font_families() is None
    # The failed probe is cached too: no re-running on every dialog open.
    calls: list[int] = []
    monkeypatch.setattr(fonts, "_linux_family_names", lambda: calls.append(1) or ["X"])
    assert fonts.available_font_families() is None
    assert calls == []


def test_available_fonts_success_is_cached(
    monkeypatch: pytest.MonkeyPatch, fonts_fresh
) -> None:
    monkeypatch.setattr(fonts.sys, "platform", "linux")
    calls: list[int] = []

    def probe() -> list[str]:
        calls.append(1)
        return ["Menlo", "Menlo ", ""]  # blanks/empties are stripped

    monkeypatch.setattr(fonts, "_linux_family_names", probe)
    assert fonts.available_font_families() == frozenset({"Menlo"})
    assert fonts.available_font_families() == frozenset({"Menlo"})
    assert calls == [1]  # computed once per process


def test_desktop_api_available_fonts_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DesktopApi.get_available_fonts sorts the set and passes 'unknown'
    (None) through untouched. Needs the desktop dependency group."""
    try:
        import app as app_mod  # type: ignore[import-not-found]
        import webview  # noqa: F401
    except Exception:  # noqa: BLE001 - webview missing / no GUI libs
        pytest.skip("pywebview desktop group not installed")

    api = app_mod.DesktopApi({}, None, Path("/dev/null"))
    monkeypatch.setattr(
        app_mod.fonts_mod, "available_font_families", lambda: frozenset({"B", "A"})
    )
    assert api.get_available_fonts() == ["A", "B"]
    monkeypatch.setattr(app_mod.fonts_mod, "available_font_families", lambda: None)
    assert api.get_available_fonts() is None


# ---------------------------------------------------------------------------
# pty_session.py
# ---------------------------------------------------------------------------


def test_child_environment_forces_terminal_caps() -> None:
    env = child_environment({"EXTRA": "1"})
    assert env["TERM"] == "xterm-256color"
    assert env["COLORTERM"] == "truecolor"
    assert env["EXTRA"] == "1"
    assert "COLUMNS" not in env and "LINES" not in env


def test_child_environment_augments_gui_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GUI launches (Finder/Dock) give the process the bare OS PATH with
    no shell profile sourced; child_environment() must append the standard
    tool directories without reordering or duplicating what's there."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    env = child_environment()
    entries = env["PATH"].split(os.pathsep)
    # The user's entries survive, in order, at the front.
    assert entries[:4] == ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    # Standard directories that were missing are appended.
    assert "/opt/homebrew/bin" in entries
    assert "/usr/local/bin" in entries
    # No entry appears twice.
    assert len(entries) == len(set(entries))


def test_child_environment_path_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Already-present directories are never duplicated."""
    monkeypatch.setenv("HOME", "/home/tester")
    rich = os.pathsep.join(
        [
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/home/tester/.local/bin",
            "/usr/bin",
            "/bin",
        ]
    )
    monkeypatch.setenv("PATH", rich)
    env = child_environment()
    entries = env["PATH"].split(os.pathsep)
    # Existing entries keep their order and their head position.
    assert entries[:5] == rich.split(os.pathsep)
    # No entry appears twice.
    assert len(entries) == len(set(entries))


def test_child_environment_path_expands_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/home/tester")
    env = child_environment()
    entries = env["PATH"].split(os.pathsep)
    assert entries[:2] == ["/usr/bin", "/bin"]
    assert "/home/tester/.local/bin" in entries
    assert "/home/tester/.cargo/bin" in entries


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
# server.py — pump(): output coalescing
# ---------------------------------------------------------------------------


async def _drain_frames(
    ws: aiohttp.ClientWebSocketResponse, quiet: float = 0.4, timeout: float = 5.0
) -> list[bytes | dict]:
    """Read frames until the stream is quiet for ``quiet`` seconds."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    frames: list[bytes | dict] = []
    while loop.time() < deadline:
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=quiet)
        except TimeoutError:
            break
        if msg.type == aiohttp.WSMsgType.BINARY:
            frames.append(msg.data)
        elif msg.type == aiohttp.WSMsgType.TEXT:
            frames.append(json.loads(msg.data))
        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
            break
    return frames


async def _connect_with_sinks(ctx: ShellContext, base_url: str):
    """Open a ws connection and wait until the server side registered the
    sinks for the default tab (the handshake alone is not a guarantee)."""
    http = aiohttp.ClientSession()
    ws = await http.ws_connect(f"{base_url}/ws?token={ctx.token}")
    deadline = asyncio.get_event_loop().time() + 5
    state = ctx.tab("default")
    while state.frame_sink is None or state.event_sink is None:
        assert asyncio.get_event_loop().time() < deadline, "sinks never registered"
        await asyncio.sleep(0.02)
    return http, ws, state


async def test_server_ws_coalesces_output_burst(tmp_path: Path) -> None:
    """60 tiny chunks queued in one synchronous burst reach the client as
    ONE (or a few) coalesced binary frame(s), byte-exact and in order."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    try:
        http, ws, state = await _connect_with_sinks(ctx, base_url)
        try:
            chunks = [f"<chunk-{i:03d}>".encode() for i in range(60)]
            sink = state.frame_sink
            assert sink is not None
            # One synchronous stretch: all 60 puts land on the loop before
            # the pump task gets to run, i.e. they are all "already queued"
            # at the moment of the first send.
            for chunk in chunks:
                sink(chunk)
            frames = await _drain_frames(ws)
        finally:
            await http.close()

        binary = [f for f in frames if isinstance(f, bytes)]
        assert binary, "no output frames reached the client"
        # Order and content survive the coalescing exactly.
        assert b"".join(binary) == b"".join(chunks)
        # The burst was collapsed instead of forwarded chunk by chunk.
        assert len(binary) < len(chunks)
        assert len(binary) <= 5
        # Nothing else (no json events) was emitted for pure byte traffic.
        assert not [f for f in frames if not isinstance(f, bytes)]
    finally:
        await runner.cleanup()


async def test_server_ws_bytes_and_events_keep_queue_order(tmp_path: Path) -> None:
    """A json event queued between byte chunks goes out as its own text
    frame BETWEEN the bytes: bytes-before, event, bytes-after."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    try:
        http, ws, state = await _connect_with_sinks(ctx, base_url)
        try:
            sink = state.frame_sink
            event_sink = state.event_sink
            assert sink is not None and event_sink is not None
            sink(b"AAA-before")
            sink(b"BBB-before")
            marker = {"event": "error", "message": "order-marker", "tab": "default"}
            event_sink(marker)
            # The bytes sink enqueues DIRECTLY (it only ever fires on the
            # loop thread via add_reader), while the event sink schedules
            # its put through call_soon_threadsafe (the reaper calls it
            # from a worker thread). Give the loop a tick so the scheduled
            # event put lands in the queue before the trailing bytes —
            # exactly the ordering production relies on for exit events.
            await asyncio.sleep(0)
            sink(b"CCC-after")
            frames = await _drain_frames(ws)
        finally:
            await http.close()

        assert frames == [
            b"AAA-beforeBBB-before",  # bytes before the event, coalesced
            marker,  # the event as its own text frame
            b"CCC-after",  # bytes after the event, in their own frame
        ]
    finally:
        await runner.cleanup()


async def test_server_ws_coalesced_frames_capped(tmp_path: Path) -> None:
    """A firehose burst larger than FRAME_MAX_BYTES is split into several
    frames, each bounded by the cap (plus one read chunk), never merged
    into one unbounded frame, order intact."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    try:
        http, ws, state = await _connect_with_sinks(ctx, base_url)
        try:
            chunk = b"x" * 100_000
            chunks = [bytes([ord("0") + i]) + chunk for i in range(12)]
            sink = state.frame_sink
            assert sink is not None
            for c in chunks:
                sink(c)
            frames = await _drain_frames(ws)
        finally:
            await http.close()

        binary = [f for f in frames if isinstance(f, bytes)]
        assert len(binary) > 1, "cap did not split the burst"
        # Bound: FRAME_MAX_BYTES plus at most one chunk that over-shot.
        assert all(len(f) <= FRAME_MAX_BYTES + len(chunk) for f in binary)
        assert b"".join(binary) == b"".join(chunks)
    finally:
        await runner.cleanup()


async def test_server_ws_steady_stream_coalesced(tmp_path: Path) -> None:
    """An EVEN (non-bursty) byte stream — one tiny chunk every few ms, so
    each lands while the pump is asleep in out_queue.get() — does not ship
    one ws frame per chunk: the latency window lets the stream pile up and
    the pump sends it in far fewer frames, byte-exact and in order."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    try:
        http, ws, state = await _connect_with_sinks(ctx, base_url)
        try:
            chunks = [f"<t-{i:03d}>".encode() for i in range(40)]
            sink = state.frame_sink
            assert sink is not None

            async def _trickle() -> None:
                # 2 ms apart — well inside COALESCE_WINDOW_SECONDS, so the
                # window (not the drain pass) must do the collapsing.
                for chunk in chunks:
                    sink(chunk)
                    await asyncio.sleep(0.002)

            task = asyncio.ensure_future(_trickle())
            frames = await _drain_frames(ws, quiet=0.3, timeout=5.0)
            await task
        finally:
            await http.close()

        binary = [f for f in frames if isinstance(f, bytes)]
        assert binary, "no output frames reached the client"
        assert b"".join(binary) == b"".join(chunks)
        # The even stream was collapsed, not forwarded chunk by chunk:
        # with an 8 ms window and 2 ms spacing a frame covers ~4 chunks.
        assert len(binary) <= len(chunks) // 2
        # No json events were emitted for pure byte traffic.
        assert not [f for f in frames if not isinstance(f, bytes)]
    finally:
        await runner.cleanup()


async def test_server_ws_single_input_latency_bounded(tmp_path: Path) -> None:
    """The latency window never delays a lone chunk past the window: an
    interactive keystroke (a single sunk chunk on an idle connection) goes
    out within COALESCE_WINDOW_SECONDS plus loop/test slack, intact."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    try:
        http, ws, state = await _connect_with_sinks(ctx, base_url)
        try:
            sink = state.frame_sink
            assert sink is not None
            await asyncio.sleep(0.1)  # let the pump settle in get()

            started = time.monotonic()
            sink(b"keystroke")
            while True:
                msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                if msg.type == aiohttp.WSMsgType.BINARY:
                    break
                assert msg.type not in (
                    aiohttp.WSMsgType.ERROR,
                    aiohttp.WSMsgType.CLOSE,
                )
            latency = time.monotonic() - started
            assert msg.data == b"keystroke"
            # Window (8 ms) plus generous slack for loop scheduling/CI.
            assert latency < COALESCE_WINDOW_SECONDS + 0.5
        finally:
            await http.close()
    finally:
        await runner.cleanup()


# ---------------------------------------------------------------------------
# server.py — pump(): client-side flow control (pause/resume)
# ---------------------------------------------------------------------------


async def test_server_ws_flow_control_pause_resume(tmp_path: Path) -> None:
    """pause stops binary frames (JSON events still flow); resume releases
    the withheld bytes with neither loss nor reordering — including bytes
    queued between the pause and the resume."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    try:
        http, ws, state = await _connect_with_sinks(ctx, base_url)
        try:
            sink = state.frame_sink
            assert sink is not None

            await ws.send_json({"type": "pause"})
            # Give the pump time to apply the pause (the frame must be
            # received and queued before the test-side bytes are).
            await asyncio.sleep(0.3)

            sink(b"held-A")
            sink(b"held-B")
            marker = {"event": "error", "message": "flow-probe", "tab": "default"}
            state.event_sink(marker)

            # While paused: the event arrives, the bytes do not.
            frames = await _drain_frames(ws, quiet=0.4, timeout=1.5)
            assert frames == [marker], "paused bytes leaked or event lost"

            # Still paused: more output piles up server-side.
            sink(b"held-C")

            await ws.send_json({"type": "resume"})
            sink(b"after-resume")
            frames2 = await _drain_frames(ws)

            binary = [f for f in frames2 if isinstance(f, bytes)]
            # Nothing lost, nothing reordered: the client sees exactly the
            # sunk byte stream, in sink order.
            assert b"".join(binary) == b"held-Aheld-Bheld-C" + b"after-resume"
            # No stray json events mixed in.
            assert not [f for f in frames2 if not isinstance(f, bytes)]
        finally:
            await http.close()
    finally:
        await runner.cleanup()


async def test_server_ws_flow_control_json_not_blocked(tmp_path: Path) -> None:
    """A paused connection still delivers json events arriving AFTER the
    pause — start/exit/error must never wait on the byte stream."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    try:
        http, ws, state = await _connect_with_sinks(ctx, base_url)
        try:
            await ws.send_json({"type": "pause"})
            await asyncio.sleep(0.3)
            state.frame_sink(b"held")  # type: ignore[union-attr]

            sink_event = state.event_sink
            assert sink_event is not None
            first = {"event": "error", "message": "e1", "tab": "default"}
            second = {"event": "error", "message": "e2", "tab": "default"}
            sink_event(first)
            sink_event(second)

            frames = await _drain_frames(ws, quiet=0.4, timeout=1.5)
            assert frames == [first, second]
        finally:
            await http.close()
    finally:
        await runner.cleanup()


async def test_server_ws_flow_control_resets_on_reconnect(tmp_path: Path) -> None:
    """The pause lives on the connection, not the tab: a fresh ws attach to
    the same tab is never stuck paused."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    try:
        http1, ws1, _state = await _connect_with_sinks(ctx, base_url)
        await ws1.send_json({"type": "pause"})
        await asyncio.sleep(0.3)
        await http1.close()

        http2, ws2, state2 = await _connect_with_sinks(ctx, base_url)
        try:
            sink = state2.frame_sink
            assert sink is not None
            sink(b"fresh-connection")
            frames = await _drain_frames(ws2, quiet=0.4, timeout=1.5)
            binary = b"".join(f for f in frames if isinstance(f, bytes))
            assert b"fresh-connection" in binary
        finally:
            await http2.close()
    finally:
        await runner.cleanup()


async def test_server_ws_flow_control_cap_overflow_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past PAUSE_BUFFER_MAX_BYTES the pause is given up: output flows
    again (bytes never dropped), and the client's later resume is a no-op."""
    import server as server_module  # type: ignore[import-not-found]

    monkeypatch.setattr(server_module, "PAUSE_BUFFER_MAX_BYTES", 1000)
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    try:
        http, ws, state = await _connect_with_sinks(ctx, base_url)
        try:
            sink = state.frame_sink
            assert sink is not None

            await ws.send_json({"type": "pause"})
            await asyncio.sleep(0.3)

            chunk_a = b"a" * 600
            chunk_b = b"b" * 600  # 1200 held > 1000 cap → overflow → resume
            sink(chunk_a)
            sink(chunk_b)
            frames = await _drain_frames(ws)
            binary = [f for f in frames if isinstance(f, bytes)]
            assert b"".join(binary) == chunk_a + chunk_b

            # Still (client-side) paused: the server keeps sending anyway.
            sink(b"flows-despite-pause")
            frames2 = await _drain_frames(ws, quiet=0.4, timeout=1.5)
            binary2 = b"".join(f for f in frames2 if isinstance(f, bytes))
            assert b"flows-despite-pause" in binary2

            # The client's eventual resume is harmless.
            await ws.send_json({"type": "resume"})
            sink(b"post-resume")
            frames3 = await _drain_frames(ws, quiet=0.4, timeout=1.5)
            binary3 = b"".join(f for f in frames3 if isinstance(f, bytes))
            assert b"post-resume" in binary3
        finally:
            await http.close()
    finally:
        await runner.cleanup()


# ---------------------------------------------------------------------------
# server.py — tabs: one session per tab, tab-marked events
# ---------------------------------------------------------------------------


def _make_loop_payload(tmp_path: Path) -> str:
    """Payload printing its cwd every 0.1 s (easy stream assertions)."""
    script = tmp_path / "payload.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, time\n"
        "while True:\n"
        "    sys.stdout.write(os.getcwd() + '\\n')\n"
        "    sys.stdout.flush()\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return str(script)


async def _wait_for(
    ws: aiohttp.ClientWebSocketResponse, predicate, timeout: float = 10
):
    """Read frames until ``predicate`` matches; return (match, all frames)."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    seen = []
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise AssertionError(f"condition not met in time; seen: {seen!r}")
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
        except TimeoutError:
            # Deadline expiry must surface as a diagnostic assertion, not a
            # bare TimeoutError (xdist-load flakes are timing, not protocol).
            raise AssertionError(f"condition not met in time; seen: {seen!r}")
        if msg.type == aiohttp.WSMsgType.TEXT:
            event = json.loads(msg.data)
            seen.append(event)
            if predicate(event):
                return event, seen
        elif msg.type == aiohttp.WSMsgType.BINARY:
            seen.append(msg.data)
            if predicate(msg.data):
                return msg.data, seen
        elif msg.type == aiohttp.WSMsgType.ERROR:
            raise AssertionError(f"ws error; seen: {seen!r}")


async def test_server_tabs_are_independent(tmp_path: Path) -> None:
    """Two tabs = two PTY sessions: separate output streams, tab-marked
    events, and killing one tab leaves the other running."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    ctx.payload = _make_loop_payload(tmp_path)
    proj_a = tmp_path / "proj-alpha"
    proj_b = tmp_path / "proj-beta"
    proj_a.mkdir()
    proj_b.mkdir()

    try:
        async with aiohttp.ClientSession() as http:
            ws_a = await http.ws_connect(f"{base_url}/ws?token={ctx.token}&tab=ta")
            ws_b = await http.ws_connect(f"{base_url}/ws?token={ctx.token}&tab=tb")
            await ws_a.send_json(
                {"type": "start", "project": str(proj_a), "rows": 24, "cols": 80}
            )
            await ws_b.send_json(
                {"type": "start", "project": str(proj_b), "rows": 24, "cols": 80}
            )

            event_a, _ = await _wait_for(
                ws_a, lambda e: isinstance(e, dict) and e.get("event") == "started"
            )
            assert event_a["tab"] == "ta"
            event_b, _ = await _wait_for(
                ws_b, lambda e: isinstance(e, dict) and e.get("event") == "started"
            )
            assert event_b["tab"] == "tb"

            # Each stream carries only its own project's output.
            data_a, _ = await _wait_for(
                ws_a, lambda d: isinstance(d, bytes) and b"proj-alpha" in d
            )
            assert b"proj-beta" not in data_a
            data_b, _ = await _wait_for(
                ws_b, lambda d: isinstance(d, bytes) and b"proj-beta" in d
            )
            assert b"proj-alpha" not in data_b

            # Killing tab ta does not touch tb.
            await ws_a.send_json({"type": "kill", "tab": "ta"})
            event, _ = await _wait_for(
                ws_a,
                lambda e: isinstance(e, dict) and e.get("event") == "exited",
            )
            assert event["tab"] == "ta"
            assert ctx.tabs["ta"].session is None
            assert ctx.tabs["tb"].session is not None
            # tb keeps streaming after the neighbour died.
            await _wait_for(
                ws_b,
                lambda d: isinstance(d, bytes) and b"proj-beta" in d,
                timeout=10,
            )

            await ws_b.send_json({"type": "kill", "tab": "tb"})
            await _wait_for(
                ws_b, lambda e: isinstance(e, dict) and e.get("event") == "exited"
            )
            await ws_a.close()
            await ws_b.close()
    finally:
        await ctx.stop_all()
        await runner.cleanup()


async def test_server_stop_all_kills_every_tab(tmp_path: Path) -> None:
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    ctx.payload = _make_loop_payload(tmp_path)
    projects = []
    for name in ("p1", "p2", "p3"):
        p = tmp_path / name
        p.mkdir()
        projects.append(p)

    try:
        async with aiohttp.ClientSession() as http:
            sockets = []
            for i, proj in enumerate(projects):
                ws = await http.ws_connect(f"{base_url}/ws?token={ctx.token}&tab=t{i}")
                await ws.send_json(
                    {"type": "start", "project": str(proj), "rows": 24, "cols": 80}
                )
                sockets.append(ws)
            for i, ws in enumerate(sockets):
                await _wait_for(
                    ws, lambda e: isinstance(e, dict) and e.get("event") == "started"
                )
            assert ctx.live_project() == str(projects[0])
            await ctx.stop_all()
            for i in range(3):
                assert ctx.tabs[f"t{i}"].session is None
            assert ctx.live_project() is None
            for ws in sockets:
                await ws.close()
    finally:
        await runner.cleanup()


async def test_server_clean_exit_reports_tab_to_hook(tmp_path: Path) -> None:
    """A clean exit (code 0) invokes on_clean_exit with the tab id — the
    shell closes THAT tab, not the whole window."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    closed: list[str] = []
    ctx.on_clean_exit = closed.append
    script = tmp_path / "payload.py"
    script.write_text("#!/usr/bin/env python3\nprint('bye')\n", encoding="utf-8")
    script.chmod(0o755)
    ctx.payload = str(script)
    proj = tmp_path / "proj"
    proj.mkdir()

    try:
        async with (
            aiohttp.ClientSession() as http,
            http.ws_connect(f"{base_url}/ws?token={ctx.token}&tab=solo") as ws,
        ):
            await ws.send_json(
                {"type": "start", "project": str(proj), "rows": 24, "cols": 80}
            )
            event, _ = await _wait_for(
                ws, lambda e: isinstance(e, dict) and e.get("event") == "exited"
            )
            assert event["code"] == 0
            assert event["tab"] == "solo"
            # The hook ran before the event reached the wire.
            assert closed == ["solo"]
    finally:
        await ctx.stop_all()
        await runner.cleanup()


# ---------------------------------------------------------------------------
# server.py — one live session per project (duplicate-project guard)
# ---------------------------------------------------------------------------


async def _start_project(
    http: aiohttp.ClientSession,
    base_url: str,
    ctx: ShellContext,
    tab: str,
    project: Path,
):
    """Connect a tab and start ``project``; return (ws, all events up to
    and including the first error-or-started event)."""
    ws = await http.ws_connect(f"{base_url}/ws?token={ctx.token}&tab={tab}")
    await ws.send_json(
        {"type": "start", "project": str(project), "rows": 24, "cols": 80}
    )
    event, _ = await _wait_for(
        ws,
        lambda e: isinstance(e, dict) and e.get("event") in ("started", "error"),
    )
    return ws, event


async def test_server_duplicate_project_blocked_in_other_tab(tmp_path: Path) -> None:
    """The second tab over the same project gets an error event instead of
    a session; the first tab keeps running."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    ctx.payload = _make_loop_payload(tmp_path)
    proj = tmp_path / "proj"
    proj.mkdir()

    try:
        async with aiohttp.ClientSession() as http:
            ws_a, event_a = await _start_project(http, base_url, ctx, "ta", proj)
            assert event_a["event"] == "started"

            ws_b, event_b = await _start_project(http, base_url, ctx, "tb", proj)
            assert event_b["event"] == "error"
            assert "already open in another tab" in event_b["message"]
            assert "proj" in event_b["message"]
            assert ctx.tabs["tb"].session is None
            # The first tab's session is untouched.
            assert ctx.tabs["ta"].session is not None
            await _wait_for(
                ws_a, lambda d: isinstance(d, bytes) and b"proj" in d, timeout=10
            )
            await ws_a.close()
            await ws_b.close()
    finally:
        await ctx.stop_all()
        await runner.cleanup()


async def test_server_duplicate_project_symlink_same_path(tmp_path: Path) -> None:
    """A symlink to the project resolves to the same realpath: opening it
    from another tab is blocked too (macOS /private/tmp aliases included)."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    ctx.payload = _make_loop_payload(tmp_path)
    proj = tmp_path / "real-proj"
    proj.mkdir()
    link = tmp_path / "link-proj"
    link.symlink_to(proj)
    assert os.path.realpath(link) == os.path.realpath(proj)

    try:
        async with aiohttp.ClientSession() as http:
            ws_a, event_a = await _start_project(http, base_url, ctx, "ta", proj)
            assert event_a["event"] == "started"

            ws_b, event_b = await _start_project(http, base_url, ctx, "tb", link)
            assert event_b["event"] == "error"
            assert "already open in another tab" in event_b["message"]
            assert ctx.tabs["tb"].session is None
            await ws_a.close()
            await ws_b.close()
    finally:
        await ctx.stop_all()
        await runner.cleanup()


async def test_server_restart_own_project_allowed(tmp_path: Path) -> None:
    """Restarting a tab's OWN project never trips the guard (the holder
    is the tab itself)."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    ctx.payload = _make_loop_payload(tmp_path)
    proj = tmp_path / "proj"
    proj.mkdir()

    try:
        async with aiohttp.ClientSession() as http:
            ws, event = await _start_project(http, base_url, ctx, "ta", proj)
            assert event["event"] == "started"

            await ws.send_json(
                {"type": "restart", "project": str(proj), "rows": 24, "cols": 80}
            )
            event, _ = await _wait_for(
                ws,
                lambda e: (
                    isinstance(e, dict) and e.get("event") in ("started", "error")
                ),
            )
            assert event["event"] == "started"
            assert ctx.tabs["ta"].session is not None
            await ws.close()
    finally:
        await ctx.stop_all()
        await runner.cleanup()


async def test_server_project_free_after_tab_closed(tmp_path: Path) -> None:
    """Killing the holding tab frees the project: another tab can open it."""
    ctx, runner, base_url = await _start_server(str(tmp_path / "unused"), tmp_path)
    ctx.payload = _make_loop_payload(tmp_path)
    proj = tmp_path / "proj"
    proj.mkdir()

    try:
        async with aiohttp.ClientSession() as http:
            ws_a, event_a = await _start_project(http, base_url, ctx, "ta", proj)
            assert event_a["event"] == "started"

            await ws_a.send_json({"type": "kill", "tab": "ta"})
            await _wait_for(
                ws_a, lambda e: isinstance(e, dict) and e.get("event") == "exited"
            )
            assert ctx.tabs["ta"].session is None

            ws_b, event_b = await _start_project(http, base_url, ctx, "tb", proj)
            assert event_b["event"] == "started"
            assert ctx.tabs["tb"].session is not None
            await ws_a.close()
            await ws_b.close()
    finally:
        await ctx.stop_all()
        await runner.cleanup()


# ---------------------------------------------------------------------------
# web/term.js — tab model (static, headless)
# ---------------------------------------------------------------------------


def test_term_js_tab_model_wiring() -> None:
    """The client is multi-tab: one ws per tab (tab id in the connect
    URL), activation re-fits the active terminal, closing the last tab
    closes the window, and a clean exit closes only its own tab."""
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")

    # One ws per tab, addressed by tab id.
    assert '"&tab=" + encodeURIComponent(tab.id)' in js
    # Tab switching re-fits the active terminal and resizes its PTY.
    activate = js[js.index("function activateTab") :]
    assert "tab.fit.fit()" in activate
    assert "sendResize(tab)" in activate
    # Per-tab key handler + data sink (each xterm talks to its own ws).
    assert "function terminalKeyHandler(tab, ev)" in js
    assert "tab.ws.send(new TextEncoder().encode(data))" in js

    # Closing the last tab closes the window.
    close = js[js.index("function closeTab") : js.index("/* Python-side hook")]
    assert 'api("close_window")' in close
    # A clean exit (code 0) closes ITS tab, not the window; a crash opens
    # the restart dialog for that tab.
    exited = js[js.index("function onTabExited") : js.index("function hideDialog")]
    assert "closeTab(tab)" in exited
    assert "dialogTab = tab" in exited


def test_term_js_python_close_tab_hook() -> None:
    """app.py closes a tab via evaluate_js on window.__drcodeDesktop —
    the JS hook must exist and be idempotent (lookup in tabs)."""
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")
    assert "window.__drcodeDesktop" in js
    hook = js[js.index("window.__drcodeDesktop") :]
    assert "closeTab: function (id)" in hook
    assert "if (tab) closeTab(tab);" in hook

    src = (DESKTOP_DIR / "app.py").read_text(encoding="utf-8")
    assert "window.__drcodeDesktop.closeTab" in src
    assert "window.evaluate_js(" in src


def test_index_html_has_tab_bar() -> None:
    html = (DESKTOP_DIR / "web" / "index.html").read_text(encoding="utf-8")
    for el_id in ("tabbar", "tabs", "new-tab-btn", "panes"):
        assert f'id="{el_id}"' in html
    css = (DESKTOP_DIR / "web" / "style.css").read_text(encoding="utf-8")
    for selector in ("#tabbar", ".tab ", ".tab-pane", "#panes"):
        assert selector in css


def test_term_js_new_tab_button_and_restart_targeting() -> None:
    """The "+" next to the tab bar opens a fresh welcome tab; Ctrl+Shift+R
    restarts the ACTIVE tab's project; the crash dialog buttons act on the
    tab the dialog is about."""
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")

    new_tab = js[js.index("if (elNewTab)") : js.index("// Settings dialog wiring")]
    assert 'createTab("welcome", null)' in new_tab

    keys = js[
        js.index('window.addEventListener("keydown"') : js.index(
            "// Keep every PTY size"
        )
    ]
    restart_at = keys.index('ev.key === "r" || ev.key === "R"')
    target = keys.index("launchProject(activeTab, activeTab.project)")
    assert restart_at < target

    restart_btn = js[
        js.index('document.getElementById("btn-restart")') : js.index(
            'document.getElementById("btn-close")'
        )
    ]
    assert "launchProject(tab, tab.project)" in restart_btn
    close_btn = js[
        js.index('document.getElementById("btn-close")') : js.index("if (elNewTab)")
    ]
    assert "closeTab(tab)" in close_btn


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


def test_term_js_modified_enter_sends_csi_u() -> None:
    """xterm.js encodes every Enter as plain "\r", so Shift+Enter would
    submit the chat input instead of inserting a newline. The key handler
    must intercept modified Enter variants and forward them as kitty-style
    CSI u sequences (which Textual's parser decodes back into
    shift+enter / ctrl+enter / alt+enter), leaving plain Enter untouched."""
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")
    handler = js[js.index("function encodeEnter") :]

    # Modified Enter is intercepted BEFORE the font-size shortcut gate and
    # before the ctrl-only filter that would otherwise let it through.
    gate = handler.index("var code = ev.code")
    enter_at = handler.index('ev.key === "Enter"')
    assert enter_at < gate

    # Only while a live session exists (welcome screen has no PTY).
    assert handler.index("ws.readyState !== WebSocket.OPEN") < gate
    # Plain Enter is never intercepted: no branch matches unmodified Enter.
    assert "ev.shiftKey || ev.ctrlKey || ev.altKey" in handler
    # The payload is the kitty CSI u encoding Textual understands.
    assert '"\\x1b[13;" + mods + "u"' in handler
    # Only keydown acts: xterm.js also invokes the custom handler on
    # keyup/keypress — a keyup-matching Enter branch would send a second
    # CSI u and duplicate the newline (double font-size step likewise).
    guard = handler.index('ev.type !== "keydown"')
    assert guard < enter_at


def test_term_js_font_settings_wiring() -> None:
    """Static check of the font-settings flow: boot-time load through the
    bridge, constructor pickup, family application, and Ctrl+, interception."""
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")

    # Settings load with the bridge retry pattern, and the shell start
    # (first Terminal constructor included) consumes them (constructor
    # bakes the metrics into the renderer).
    assert 'api("get_settings")' in js
    boot = js[js.index("function boot") :]
    # The first terminal must be created only after settings resolve.
    assert "startShell();" in boot
    assert "fontFamily: fontStackFor(currentFontFamily)" in js
    assert "fontSize: fontSize" in js

    # Family change reuses the WebGL-recreation/refit path — family swaps
    # change cell metrics exactly like size changes.
    apply_family = js[js.index("function applyFontFamily") :]
    apply_size = js[
        js.index("function applyFontSize") : js.index("function applyFontFamily")
    ]
    for fn in (apply_family, apply_size):
        assert "recreateWebgl(tab)" in fn
        assert "refitAfterFontChange(tab)" in fn
    assert "fontStackFor" in apply_family

    # Ctrl+, is intercepted inside the custom key handler AFTER the
    # ctrl-only gate — otherwise the combo reaches the PTY via onData.
    handler = js[js.index("function terminalKeyHandler") :]
    gate = handler.index("if (!ev.ctrlKey")
    comma = handler.index('ev.key === ","')
    assert gate < comma < handler.index("openSettings()")
    assert "ev.preventDefault();\n      openSettings();\n      return false;" in handler


def test_term_js_hotkey_font_size_persists() -> None:
    """Hotkey size changes (Ctrl+= / Ctrl+- / Ctrl+0) must persist like the
    settings dialog: applyFontSize has to reach set_font_settings (debounced
    — a held key emits a burst of keydowns) and MUST pass the current font
    family along — the python side rewrites both fields, so a null family
    would wipe the stored font."""
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")
    apply_size = js[
        js.index("function persistFontSize") : js.index("function applyFontFamily")
    ]

    # The bridge save is debounced, not per-keydown.
    assert "setTimeout" in apply_size
    assert "clearTimeout" in apply_size
    # The saved family rides along on every persist call.
    assert "setFont(currentFontFamily" in apply_size
    # Bridge-less browser mode stays local (no persistence attempt).
    assert 'api("set_font_settings")' in apply_size
    # applyFontSize actually triggers the persistence after its change.
    assert apply_size.index("persistFontSize();") > apply_size.index(
        "term.options.fontSize = clamped;"
    )
    # The dialog's view of the stored settings is kept current.
    assert "savedSettings" in apply_size


def test_term_js_font_list_curated_and_filtered() -> None:
    """The font dialog offers a wide curated set (so filtering still
    leaves real choices on any platform) and hides the families the OS
    probe reports as absent. The apply path (fontStackFor) is NOT
    touched — filtering is a <select>-only concern."""
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")

    choices = js[js.index("var FONT_CHOICES") : js.index("var FONT_CUSTOM_VALUE")]
    for family in (
        "Menlo",
        "Monaco",
        "SF Mono",
        "Courier New",
        "Consolas",
        "Cascadia Code",
        "Fira Code",
        "Source Code Pro",
        "DejaVu Sans Mono",
        "Ubuntu Mono",
    ):
        assert family in choices

    # Availability probe at boot: same retry pattern as the other bridge
    # calls, absent bridge / failure → availableFonts stays null.
    assert "availableFonts = null;" in js
    assert 'api("get_available_fonts")' in js
    boot = js[js.index("function boot") :]
    assert "fetchAvailableFonts();" in boot

    # Matching is fail-open (null → true) and Default (empty) is always
    # offered regardless of the probe result.
    matcher = js[js.index("function fontIsAvailable") :]
    assert "if (availableFonts === null) return true;" in matcher
    assert "if (!want) return true;" in matcher

    populate = js[js.index("function populateFontSelect") :]
    assert "fontIsAvailable(choice.value)" in populate
    # Custom… is appended unconditionally — the only way to enter a rare
    # family the curated list (or the probe) does not know.
    assert 'custom.textContent = "Custom…"' in populate
    # The "known" check runs against the FILTERED list: a saved font that
    # is no longer offered falls into the Custom… branch (shown selected,
    # never reset) instead of silently losing the selection.
    assert "var known = choices.some(" in populate
    # Apply path unchanged: chosen family first, bundled stack trailing.
    assert 'return primary ? primary + ", " + FONT_STACK : FONT_STACK;' in js


def test_app_py_exposes_available_fonts_bridge() -> None:
    """The bridge method must exist and delegate to the fonts module —
    app.py is not importable without the desktop group, so this stays a
    source-level check (the behavioral test skips without webview)."""
    src = (DESKTOP_DIR / "app.py").read_text(encoding="utf-8")
    assert "def get_available_fonts(self)" in src
    assert "fonts_mod.available_font_families()" in src
    # The bridge returns None (unknown) rather than an empty list on a
    # failed probe — the JS side fail-opens only on None.
    assert "return sorted(fonts) if fonts is not None else None" in src


def test_index_html_has_settings_dialog() -> None:
    html = (DESKTOP_DIR / "web" / "index.html").read_text(encoding="utf-8")
    for el_id in (
        "settings-dialog",
        "settings-font-family",
        "settings-font-custom",
        "settings-font-size",
        "settings-font-preview",
        "btn-settings-save",
        "btn-settings-cancel",
        "open-settings",
    ):
        assert f'id="{el_id}"' in html
    # The preview shows the glyphs the TUI actually draws: ASCII plus
    # box-drawing characters.
    assert "settings-preview" in html
    assert "┌" in html


def test_index_html_has_floating_font_button() -> None:
    """After a project opens, the font dialog needs a VISIBLE entry point —
    the welcome screen's button and its Ctrl+, hint disappear with it. A
    small floating button overlays the terminal (initially hidden: the
    welcome screen has its own entry point)."""
    html = (DESKTOP_DIR / "web" / "index.html").read_text(encoding="utf-8")
    assert 'id="font-settings-btn"' in html
    assert "font-btn hidden" in html  # session-only: revealed on "started"
    # It overlays the tab panes, not the welcome card.
    assert html.index('id="panes"') < html.index('id="font-settings-btn"')

    css = (DESKTOP_DIR / "web" / "style.css").read_text(encoding="utf-8")
    btn = css[css.index(".font-btn {") : css.index(".font-btn:hover")]
    # Absolute overlay: the terminal's box must not change when the button
    # appears (fit/ResizeObserver-driven PTY resize stay exact).
    assert "position: absolute;" in btn
    assert "z-index" in btn
    overlay = css[css.index(".overlay {") : css.index(".dialog {")]
    assert (
        btn[btn.index("z-index: 15") + len("z-index: ") :].split(";")[0]
        < (overlay[overlay.index("z-index: ") + len("z-index: ") :].split(";")[0])
    ), "dialogs must cover the floating button"


def test_term_js_floating_font_button_wiring() -> None:
    """The floating button opens the SAME openSettings() dialog, must not
    steal keyboard focus from the terminal, and appears only while a
    session is live."""
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")
    assert 'document.getElementById("font-settings-btn")' in js

    wiring = js[js.index("if (elFontBtn) {") : js.index("if (elSettingsFont) {")]
    # Clicking never moves focus out of the terminal (mousedown default is
    # the focus change; the click still fires and opens the dialog).
    assert wiring.index('addEventListener("mousedown"') < wiring.index(
        'addEventListener("click", openSettings)'
    )
    assert "ev.preventDefault();" in wiring

    # Session-only visibility: revealed in the "started" event handler.
    started = js[
        js.index('event.event === "started"') : js.index('event.event === "exited"')
    ]
    assert 'elFontBtn.classList.remove("hidden")' in started

    # Handing focus back to the terminal when the dialog closes — opened
    # from any of the three entry points (hotkey / welcome / button).
    close = js[
        js.index("function closeSettings") : js.index("function showSettingsError")
    ]
    assert (
        "if (activeTab && activeTab.term && activeTab.started) activeTab.term.focus();"
        in close
    )


def test_term_js_font_preview_wiring() -> None:
    """The settings dialog's preview block must mirror the real terminal:
    the same fontStackFor() chain (chosen family first, bundled FONT_STACK
    trailing), the same size clamp, and live updates on every input the
    user can change (select, custom field, size)."""
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")
    assert 'document.getElementById("settings-font-preview")' in js

    # The preview update uses the terminal's own family chain, not a
    # ad-hoc stack — otherwise it could diverge from the real rendering.
    update = js[js.index("function updateFontPreview") :]
    assert "fontStackFor(dialogFamily())" in update
    assert "MAX_FONT_SIZE, Math.max(MIN_FONT_SIZE" in update

    # Custom… is resolved from the free-text field.
    dialog_family = js[js.index("function dialogFamily") :]
    assert "FONT_CUSTOM_VALUE" in dialog_family
    assert "elSettingsCustom.value.trim()" in dialog_family

    # Live updates: open (initial state) + select change + custom input +
    # size input all reach updateFontPreview.
    for fn in ("function openSettings", "function wireUi"):
        assert "updateFontPreview()" in js[js.index(fn) :]


def test_term_js_duplicate_project_guard() -> None:
    """One drcode instance per project, enforced in the client too: the
    launch paths pre-check the open tabs, the picker blocks before any
    tab is created, and occupied recents are marked and inert."""
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")

    # The occupancy helper and the shared message text exist.
    occupied = js[
        js.index("function projectOccupied") : js.index("function duplicateMessage")
    ]
    assert "t.started" in occupied  # only a live session holds the project
    assert "t !== excludeTab" in occupied  # a tab may restart its own project
    dup = js[js.index("function duplicateMessage") : js.index("function launchProject")]
    assert "already open in another tab" in dup

    # launchProject blocks (welcome error on an unstarted tab, boot error
    # on a live one) instead of opening a duplicate.
    launch = js[
        js.index("function launchProject") : js.index("function fetchAvailableFonts")
    ]
    assert "projectOccupied(project, tab)" in launch
    assert "showWelcomeError(duplicateMessage(project))" in launch
    assert "bootError(duplicateMessage(project))" in launch

    # The picker blocks BEFORE creating a welcome tab.
    picker = js[
        js.index("function launchFromPicker") : js.index("function loadRecents")
    ]
    assert "projectOccupied(project, null)" in picker
    assert picker.index("projectOccupied") < picker.index("createTab")

    # Occupied recents: "open" mark, not-allowed cursor, and the click
    # explains instead of launching.
    recents = js[js.index("function renderRecents") : js.index("/* Projects picked")]
    assert 'mark.className = "open-mark"' in recents
    assert 'li.className = "occupied"' in recents
    assert "duplicateMessage(item.project)" in recents
    # The list re-renders on welcome activation so the marks stay fresh.
    assert "renderRecents(lastRecents)" in js


def test_term_js_remove_recent_button() -> None:
    """Every recents entry carries a remove button: it stops the row's
    launch handler, calls the bridge and re-renders from the returned
    list; without the bridge it is a no-op."""
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")
    recents = js[js.index("function renderRecents") : js.index("/* Projects picked")]
    assert 'remove.className = "remove"' in recents
    assert '"Remove from recent projects"' in recents
    # The click must not launch the project being removed.
    assert "ev.stopPropagation()" in recents
    assert 'api("remove_recent")' in recents
    assert "doRemove(item.project)" in recents
    assert "renderRecents(recents)" in recents


def test_term_js_started_refreshes_recents() -> None:
    """A successful start must re-fetch the recents: the server records
    the project on launch, but the list is only read once at page boot —
    without the refresh a newly opened project appears in the welcome
    screen's recents only after an app restart."""
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")
    started = js[
        js.index('event.event === "started"') : js.index('event.event === "exited"')
    ]
    assert "loadRecents()" in started


def test_term_js_failed_start_reverts_tab() -> None:
    """A failed start (e.g. a recents entry whose directory was deleted)
    must not leave a phantom tab behind: launchProject renames the tab
    BEFORE the server's verdict, so the error handler has to revert the
    never-started tab to the welcome picker — otherwise the tab stays
    named after the missing project and a second picker click stacks
    yet another tab."""
    js = (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")

    revert = js[
        js.index("function revertTabToWelcome") : js.index("function projectOccupied")
    ]
    assert 'tab.kind = "welcome"' in revert
    assert "tab.project = null" in revert
    assert "refreshTabLabels()" in revert

    handler = js[js.index("function handleTabEvent") : js.index("function onTabExited")]
    err = handler[handler.index('event.event === "error"') :]
    assert "revertTabToWelcome(tab)" in err
    # Revert happens before the error is shown, and for a started tab
    # (crash dialog path) it must NOT fire.
    assert err.index("revertTabToWelcome") < err.index("showWelcomeError")
    started_branch = err[err.index("activeTab === tab") :]
    assert "revertTabToWelcome" not in started_branch


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


# ---------------------------------------------------------------------------
# app.py — frozen pywebview JS-asset shim (get_js_dir)
# ---------------------------------------------------------------------------


def test_frozen_js_dir_shim_prefers_meipass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frozen onedir layout: webview_js under sys._MEIPASS is picked."""
    import app as app_mod  # type: ignore[import-not-found]
    import webview.util

    bundled = tmp_path / "webview_js"
    bundled.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    original = webview.util.get_js_dir
    try:
        app_mod._fix_frozen_pywebview_js()
        assert webview.util.get_js_dir() == str(bundled)
    finally:
        webview.util.get_js_dir = original


def test_frozen_js_dir_shim_finds_macos_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """macOS .app layout: datas live in Contents/Resources, _MEIPASS in
    Contents/Frameworks — the shim must find them via the executable path."""
    import app as app_mod  # type: ignore[import-not-found]
    import webview.util

    fake_app = tmp_path / "DrCode.app" / "Contents"
    (fake_app / "MacOS").mkdir(parents=True)
    resources_js = fake_app / "Resources" / "webview_js"
    resources_js.mkdir(parents=True)
    fake_exe = fake_app / "MacOS" / "DrCode"
    fake_exe.write_text("")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(fake_app / "Frameworks"), raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))
    original = webview.util.get_js_dir
    try:
        app_mod._fix_frozen_pywebview_js()
        assert webview.util.get_js_dir() == str(resources_js)
    finally:
        webview.util.get_js_dir = original


def test_frozen_js_dir_shim_falls_back_without_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frozen but webview_js missing everywhere: fall back to original logic."""
    import app as app_mod  # type: ignore[import-not-found]
    import webview.util

    empty = tmp_path / "nowhere"
    empty.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(empty), raising=False)
    original = webview.util.get_js_dir
    try:
        app_mod._fix_frozen_pywebview_js()
        # In the test venv the original logic resolves the real webview/js.
        assert webview.util.get_js_dir() == original()
    finally:
        webview.util.get_js_dir = original


def test_js_dir_shim_noop_when_not_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dev mode: the shim must not touch webview.util.get_js_dir."""
    import app as app_mod  # type: ignore[import-not-found]
    import webview.util

    original = webview.util.get_js_dir
    monkeypatch.delattr(sys, "frozen", raising=False)
    app_mod._fix_frozen_pywebview_js()
    assert webview.util.get_js_dir is original


# ---------------------------------------------------------------------------
# app.py — process exit (last tab closed → the app must really terminate)
# ---------------------------------------------------------------------------
#
# Closing the last tab destroys the window while pywebview is still
# answering that very js_api call with a blocking evaluate_js. The window's
# completion handler never fires, so the (NON-daemon) bridge thread waits
# forever and interpreter shutdown — which joins non-daemon threads — hung:
# the GUI loop ended but the process stayed alive, window-less, ignoring the
# OS quit request until force-killed. Both entry points therefore end
# through _exit_process, which must not depend on those threads.


def test_exit_process_does_not_join_blocked_threads() -> None:
    """_exit_process leaves immediately even with a non-daemon thread
    parked forever (the pywebview bridge-thread situation)."""
    import subprocess
    import textwrap

    script = textwrap.dedent(
        """
        import sys, threading, time
        sys.path.insert(0, {desktop!r})
        import app as app_mod

        stuck = threading.Event()
        def block():
            stuck.wait()          # never set: a thread shutdown would join
        threading.Thread(target=block, name="stuck-bridge", daemon=False).start()
        time.sleep(0.2)

        app_mod._exit_process(7)
        print("UNREACHABLE", flush=True)
        """
    ).format(desktop=str(DESKTOP_DIR))

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 7, proc.stderr
    assert "UNREACHABLE" not in proc.stdout


def test_entry_points_exit_hard() -> None:
    """Both entry points go through _exit_process — a plain
    `SystemExit(main())` returns into interpreter shutdown, which joins the
    blocked pywebview threads and hangs the app."""
    app_src = (DESKTOP_DIR / "app.py").read_text(encoding="utf-8")
    assert "def _exit_process(code: int) -> NoReturn:" in app_src
    assert "os._exit(code)" in app_src
    tail = app_src[app_src.index('if __name__ == "__main__":') :]
    assert "_exit_process(main())" in tail

    main_src = (DESKTOP_DIR / "__main__.py").read_text(encoding="utf-8")
    assert "from .app import _exit_process, main" in main_src
    assert "_exit_process(main())" in main_src
    # The hanging construct must be gone from this entry point.
    assert "raise SystemExit(main())" not in main_src


def test_exit_process_flushes_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Output written right before the close must reach the terminal."""
    import io

    import app as app_mod  # type: ignore[import-not-found]

    fake_out, fake_err = io.StringIO(), io.StringIO()
    fake_out.write("last words")
    fake_err.write("last warning")
    calls: list[int] = []
    monkeypatch.setattr(sys, "stdout", fake_out)
    monkeypatch.setattr(sys, "stderr", fake_err)
    monkeypatch.setattr(app_mod.os, "_exit", lambda code: calls.append(code))

    app_mod._exit_process(3)  # os._exit stubbed: no interpreter teardown here
    assert calls == [3]
    assert fake_out.getvalue() == "last words"
    assert fake_err.getvalue() == "last warning"


# ---------------------------------------------------------------------------
# web/term.js — tab-switching shortcuts (Cmd+{ / Cmd+}, Cmd+Shift+[ / ])
# ---------------------------------------------------------------------------
#
# The JS terminal client is exercised manually (it needs a GUI session),
# so the shortcut wiring is checked statically: both delivery points —
# xterm's custom key handler (terminal focused) and the window keydown
# listener (focus anywhere else) — must route the combo through the shared,
# timestamp-deduped maybeTabSwitchShortcut.


def _term_js() -> str:
    return (DESKTOP_DIR / "web" / "term.js").read_text(encoding="utf-8")


def test_term_js_flow_control_wiring() -> None:
    """The client reports the xterm.js backlog to the shell server: every
    write goes through a processed-callback, pause/resume frames are
    edge-triggered with a hysteresis gap, and a fresh connection resets
    the accounting (the server's pause state is per-connection too)."""
    src = _term_js()
    # The single term.write path counts the chunk and releases it in the
    # write callback (fired when xterm.js has actually processed the data).
    assert "tab.flowOutstanding += bytes.length" in src
    assert "tab.term.write(bytes, function () {" in src
    assert "tab.flowOutstanding -= bytes.length" in src

    # Thresholds exist and the resume threshold is below the pause one
    # (either literal form: "1 * 1024 * 1024" or "1024 * 1024").
    def _flow_bytes(match: re.Match[str]) -> int:
        value = int(match.group(1))
        if match.group(2):
            value *= 1024 * 1024
        return value

    pause = re.search(r"FLOW_PAUSE_BYTES = (\d+)( \* 1024 \* 1024)?", src)
    resume = re.search(r"FLOW_RESUME_BYTES = (\d+)( \* 1024 \* 1024)?", src)
    assert pause is not None and resume is not None
    assert _flow_bytes(pause) > _flow_bytes(resume)
    # Both control frames are sent, gated by the flowPaused edge flag.
    assert '"type": "pause"' in src
    assert '"type": "resume"' in src
    assert "tab.flowPaused = true" in src
    assert "tab.flowPaused = false" in src
    # A fresh ws connection starts with a clean slate.
    assert "tab.flowOutstanding = 0" in src
    # The counter is only clamped at zero, never trimmed (no byte loss).
    assert "if (tab.flowOutstanding < 0) tab.flowOutstanding = 0;" in src


def test_tab_switch_shortcut_matched_by_physical_key() -> None:
    """The combo is matched by ev.code (layout-independent), never by
    ev.key: on a Cyrillic layout the same physical key yields «х»/«ъ»
    and the brace keycaps never appear."""
    src = _term_js()
    assert '"BracketLeft"' in src
    assert '"BracketRight"' in src
    # No ev.key-based matching of the brackets/braces crept in.
    assert 'ev.key === "[' not in src
    assert 'ev.key === "{"' not in src
    assert 'ev.key === "]"' not in src
    assert 'ev.key === "}"' not in src


def test_tab_switch_shortcut_in_both_handlers() -> None:
    src = _term_js()
    terminal_handler = src[src.index("function terminalKeyHandler") :]
    window_block = src[src.index('window.addEventListener("keydown"') :]
    assert "maybeTabSwitchShortcut" in terminal_handler
    assert "maybeTabSwitchShortcut" in window_block
    # Inside terminalKeyHandler the interception sits BEFORE the metaKey
    # pass-through gate — that gate deliberately lets every metaKey event
    # through to xterm.js, and the combo would reach the PTY.
    assert terminal_handler.index("maybeTabSwitchShortcut(") < terminal_handler.index(
        "if (!ev.ctrlKey || ev.metaKey"
    )


def test_tab_switch_shortcut_cyclic_and_deduped() -> None:
    """Cyclic wrap (last tab → first, macOS-style) over the tab order,
    and a shared handler deduping the double delivery: xterm's handler
    and the window listener receive the SAME DOM event (identical
    timeStamp), while a held-key repeat is a new event with a new stamp."""
    src = _term_js()
    assert "% tabs.length" in src
    assert "timeStamp" in src
    assert src.count("maybeTabSwitchShortcut(") >= 3  # definition + 2 call sites


def test_index_html_documents_tab_switch_shortcut() -> None:
    html = (DESKTOP_DIR / "web" / "index.html").read_text(encoding="utf-8")
    assert "switch tabs" in html
    assert "[" in html and "]" in html


# web/term.js — Cmd-W closes the active tab
# ---------------------------------------------------------------------------
#
# Same static-check pattern as the tab-switch shortcuts above: the JS
# terminal client needs a GUI session, so the Cmd-W wiring is verified
# structurally — both delivery points (xterm's custom key handler and the
# window keydown listener) must route the combo through the shared,
# timestamp-deduped maybeCloseTabShortcut, which never closes a tab under
# an open dialog.


def test_close_tab_shortcut_matched_by_physical_key() -> None:
    """Cmd-W is matched by ev.code ("KeyW") first — layout-independent —
    with the ev.key spelling kept as a fallback, mirroring the Cmd+T
    pattern."""
    src = _term_js()
    assert '"KeyW"' in src
    assert 'ev.key !== "w"' in src


def test_close_tab_shortcut_in_both_handlers() -> None:
    src = _term_js()
    terminal_handler = src[src.index("function terminalKeyHandler") :]
    window_block = src[src.index('window.addEventListener("keydown"') :]
    assert "maybeCloseTabShortcut" in terminal_handler
    assert "maybeCloseTabShortcut" in window_block
    # Inside terminalKeyHandler the interception sits BEFORE the metaKey
    # pass-through gate — that gate lets every metaKey event through to
    # xterm.js, and the combo would reach the PTY.
    assert terminal_handler.index("maybeCloseTabShortcut(") < terminal_handler.index(
        "if (!ev.ctrlKey || ev.metaKey"
    )


def test_close_tab_shortcut_closes_active_tab_and_dedupes() -> None:
    """The shared handler dedupes the double delivery (xterm's handler
    and the window listener see the SAME DOM event, identical
    timeStamp) and drives closeTab(activeTab) — the same logic as the
    tab "×" button, PTY kill included."""
    src = _term_js()
    handler = src[src.index("function maybeCloseTabShortcut") :]
    handler = handler[: handler.index("\n  function ", 1)]
    assert "lastCloseTabStamp" in src
    assert "ev.timeStamp === lastCloseTabStamp" in handler
    assert "closeTab(activeTab)" in handler


def test_close_tab_shortcut_defers_to_open_dialogs() -> None:
    """Over the settings dialog Cmd-W dismisses it (its Escape twin);
    over the crash dialog it is swallowed — closing the crashed tab out
    from under the dialog would drop the restart offer."""
    src = _term_js()
    handler = src[src.index("function maybeCloseTabShortcut") :]
    handler = handler[: handler.index("\n  function ", 1)]
    assert "elSettingsDialog" in handler
    assert "closeSettings()" in handler
    assert "elExitDialog" in handler
    # The crash-dialog branch returns BEFORE closeTab(activeTab).
    assert handler.index("elExitDialog") < handler.index("closeTab(activeTab)")


def test_index_html_documents_close_tab_shortcut() -> None:
    html = (DESKTOP_DIR / "web" / "index.html").read_text(encoding="utf-8")
    assert "close tab" in html


# desktop/app.py — _fix_cocoa_cmd_w (native Cmd-W must reach the page)
# ---------------------------------------------------------------------------


def _fake_cocoa_module(monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Stand-in for webview.platforms.cocoa with a patchable WebKitHost.

    The real class would need a live NSView instance; the fake mirrors
    the shape _fix_cocoa_cmd_w relies on: a BrowserView.WebKitHost class
    whose keyDown_ consumes some combos itself, a base class playing the
    super() responder chain, and an AppKit namespace with the mask.
    """
    import types

    super_events: list[object] = []
    original_events: list[object] = []

    class _FakeResponder:
        def keyDown_(self, event: object) -> None:
            super_events.append(event)

    class _FakeWebKitHost(_FakeResponder):
        def keyDown_(self, event: object) -> None:
            original_events.append(event)

    class _FakeBrowserView:
        WebKitHost = _FakeWebKitHost

    fake = types.ModuleType("webview.platforms.cocoa")
    fake.BrowserView = _FakeBrowserView  # type: ignore[attr-defined]
    fake.AppKit = types.SimpleNamespace(NSCommandKeyMask=1 << 20)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "webview.platforms.cocoa", fake)
    return fake, original_events, super_events


class _SimpleEvent:
    def __init__(self, char: str | None, cmd: bool) -> None:
        self._char = char
        self._cmd = cmd

    def modifierFlags(self) -> int:
        return (1 << 20) if self._cmd else 0

    def characters(self) -> str | None:
        return self._char


def test_cocoa_cmd_w_reaches_the_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain Cmd-W must NOT hit the original handler (native
    performClose_) — it goes through super() to the WKWebView responder
    chain, exactly what the unintercepted Cmd-T path does."""
    import app as app_mod  # type: ignore[import-not-found]

    fake, original_events, super_events = _fake_cocoa_module(monkeypatch)
    app_mod._fix_cocoa_cmd_w()

    event = _SimpleEvent("w", cmd=True)
    fake.BrowserView.WebKitHost.keyDown_(
        object.__new__(fake.BrowserView.WebKitHost), event
    )
    assert not original_events
    assert super_events == [event]


def test_cocoa_cmd_q_stays_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cmd-Q (and every other intercepted combo) is handed to the
    original keyDown_ untouched — app quit keeps its native behavior."""
    import app as app_mod  # type: ignore[import-not-found]

    fake, original_events, super_events = _fake_cocoa_module(monkeypatch)
    app_mod._fix_cocoa_cmd_w()

    event = _SimpleEvent("q", cmd=True)
    fake.BrowserView.WebKitHost.keyDown_(
        object.__new__(fake.BrowserView.WebKitHost), event
    )
    assert original_events == [event]
    assert not super_events


def test_cocoa_cmd_w_only_without_other_modifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper matches exactly the branch the original would have
    closed the window on (CommandKeyMask + 'w'); anything else — e.g. a
    non-command key event — falls through to the original."""
    import app as app_mod  # type: ignore[import-not-found]

    fake, original_events, super_events = _fake_cocoa_module(monkeypatch)
    app_mod._fix_cocoa_cmd_w()

    host = fake.BrowserView.WebKitHost
    self_ = object.__new__(host)
    plain = _SimpleEvent("w", cmd=False)
    host.keyDown_(self_, plain)
    assert original_events == [plain]
    assert not super_events


def test_cocoa_cmd_w_noop_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off macOS the patch must not touch anything (the cocoa import
    would fail there anyway — the guard comes first)."""
    import app as app_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(sys, "platform", "linux")
    fake, original_events, super_events = _fake_cocoa_module(monkeypatch)
    app_mod._fix_cocoa_cmd_w()

    event = _SimpleEvent("w", cmd=True)
    fake.BrowserView.WebKitHost.keyDown_(
        object.__new__(fake.BrowserView.WebKitHost), event
    )
    # Unpatched: the original handler consumed it, super never sees it.
    assert original_events == [event]
    assert not super_events


# ---------------------------------------------------------------------------
# web/term.js — WebGL fallback toast (static wiring + node-run logic)
# ---------------------------------------------------------------------------


def _toast_region(js: str) -> str:
    """The self-contained toast block, bounded by its section comment and
    the next section marker — executed verbatim by the node logic test."""
    start = js.index("/* ---------------- WebGL fallback toast")
    end = js.index("/* ---------------- terminal setup (per tab) ---------------- */")
    return js[start:end]


def test_term_js_webgl_fallback_toast_wiring() -> None:
    """The WebGL fallback paths — context loss and the failed addon
    load/creation (the catch also covers a machine without WebGL
    entirely) — must surface the once-per-tab toast; normal operation and
    the font-change recreate path must not toast on their own."""
    js = _term_js()
    region = _toast_region(js)

    # Helper trio exists; the tab-level wrapper dedups by tab id.
    assert "function showToast(message)" in region
    assert "function showToastOnce(key, message)" in region
    wrapper = region[region.index("function webglFallbackToast") :]
    assert "showToastOnce(tab.id" in wrapper

    # Initial setup and the font-change recreate path: each toasts ONLY
    # when the addon attach fails (nothing unconditional — a mere font
    # change must never re-toast).
    setup = js[
        js.index("function createTerminalFor") : js.index(
            "/* ---------------- font settings"
        )
    ]
    recreate = js[
        js.index("function recreateWebgl") : js.index("function refitAfterFontChange")
    ]
    for block in (setup, recreate):
        assert block.count("webglFallbackToast(tab)") == 1
        assert "if (!attachWebgl(tab))" in block

    # The recovery section owns the context-loss wiring: attachWebgl
    # registers onContextLoss → handleWebglContextLoss, which disposes the
    # dead addon (DOM fallback kicks in at once) and schedules a
    # budgeted recreate.
    recovery = js[
        js.index("/* ---------------- WebGL context-loss recovery") : js.index(
            "/* ---------------- terminal setup (per tab) ---------------- */"
        )
    ]
    loss_at = recovery.index("onContextLoss")
    handler_at = recovery.index("handleWebglContextLoss(tab)", loss_at)
    loss_handler = recovery[
        recovery.index("function handleWebglContextLoss") : recovery.index(
            "function scheduleWebglRecovery"
        )
    ]
    assert handler_at > loss_at
    assert loss_handler.index("disposeWebgl(tab)") < loss_handler.index(
        "webglFallbackToast(tab)"
    )
    assert "scheduleWebglRecovery(tab)" in loss_handler
    assert "function recoverWebgl" in recovery
    # Recovery is budgeted, never a page reload (a reload would drop the
    # client tab model with its live PTY sessions).
    assert "WEBGL_RECOVER_MAX_ATTEMPTS" in recovery
    assert "location.reload" not in js

    css = (DESKTOP_DIR / "web" / "style.css").read_text(encoding="utf-8")
    toast = css[css.index(".toast {") : css.index(".boot-error")]
    assert "position: absolute;" in toast
    # Above .font-btn (z-index 15), below .overlay (20): a toast must not
    # block a dialog the user is interacting with.
    assert "z-index: 16;" in toast


TOAST_NODE_DRIVER = """
const fs = require("fs");
const region = fs.readFileSync(process.argv[2], "utf8");

// Minimal DOM stub: enough surface for the toast helper (createElement /
// appendChild / classList add+remove+contains). Timers are recorded, not
// run — the test fires the auto-hide callback explicitly.
function makeEl() {
  const classes = new Set();
  return {
    className: "",
    textContent: "",
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      contains: (c) => classes.has(c),
    },
  };
}
const created = [];
const appended = [];
const documentStub = {
  createElement: () => { const el = makeEl(); created.push(el); return el; },
  body: { appendChild: (el) => appended.push(el) },
};
const timers = [];
const setTimeoutStub = (fn, ms) => { timers.push({ fn, ms }); return timers.length; };
const clearTimeoutStub = () => {};

const holder = {};
const src = region +
  "\\n;holder.api = { showToast: showToast, showToastOnce: showToastOnce," +
  " webglFallbackToast: webglFallbackToast, visibleMs: TOAST_VISIBLE_MS };";
new Function(
  "document", "setTimeout", "clearTimeout", "holder", src
)(documentStub, setTimeoutStub, clearTimeoutStub, holder);
const api = holder.api;

const el = () => created[0];

api.showToast("first");
const state1 = {
  created: created.length,
  appended: appended.length,
  text: el().textContent,
  hidden: el().classList.contains("hidden"),
  timers: timers.length,
  timerMs: timers[0].ms,
};
timers[0].fn(); // the scheduled auto-hide fires
const state2 = { hidden: el().classList.contains("hidden") };

api.showToastOnce("k1", "m1");
const state3 = {
  created: created.length,
  text: el().textContent,
  hidden: el().classList.contains("hidden"),
};
api.showToastOnce("k1", "m2-ignored");
const state4 = { text: el().textContent };
timers[1].fn();
api.showToastOnce("k2", "m3");
const state5 = { text: el().textContent };

console.log(JSON.stringify({
  visibleMs: api.visibleMs,
  hasWebglToast: typeof api.webglFallbackToast,
  state1, state2, state3, state4, state5,
}));
"""


def _run_js_driver(driver: str, region: str, tmp_path: Path) -> dict:
    """Run a node driver script against a JS region and return its JSON
    output. If the node binary itself cannot load (broken dylib — e.g. a
    half-upgraded Homebrew leaves node linked against a removed
    libsimdutf), fall back to bun, which runs the same CommonJS driver.
    A genuine script failure is never masked: the fallback triggers only
    on a dyld loader error in the runner's stderr, and if EVERY available
    runner is loader-broken the test is skipped — a broken interpreter is
    an environment defect, not a regression in the shipped JS."""
    if shutil.which("node") is None and shutil.which("bun") is None:
        pytest.skip("neither node nor bun is available")
    region_file = tmp_path / "js_region.js"
    region_file.write_text(region, encoding="utf-8")
    driver_file = tmp_path / "js_driver.js"
    driver_file.write_text(driver, encoding="utf-8")

    last: subprocess.CompletedProcess[str] | None = None
    loader_broken = True  # stays True only if EVERY failure is a loader error
    for runner in filter(None, (shutil.which("node"), shutil.which("bun"))):
        last = subprocess.run(
            [runner, str(driver_file), str(region_file)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if last.returncode == 0:
            return json.loads(last.stdout)
        if "dyld[" not in last.stderr:
            loader_broken = False
            break  # the script itself failed — don't mask it with another runner
    if loader_broken and last is not None:
        pytest.skip(f"JS runner cannot load (broken dylib): {last.stderr.strip()}")
    assert last is not None and last.returncode == 0, (
        last.stderr if last else "no JS runner available"
    )
    raise AssertionError("unreachable")  # the assert above always raises


def test_term_js_toast_logic(tmp_path: Path) -> None:
    """The extracted toast helper actually runs (node): it shows a single
    element, hides it via the scheduled timer, dedups repeated keys, and
    re-shows for a new key."""
    result = _run_js_driver(TOAST_NODE_DRIVER, _toast_region(_term_js()), tmp_path)

    assert result["visibleMs"] > 0
    assert result["hasWebglToast"] == "function"

    # First show: exactly one element, created AND appended, visible, one
    # auto-hide timer scheduled with the module's visibility window.
    s1 = result["state1"]
    assert s1["created"] == s1["appended"] == 1
    assert s1["text"] == "first"
    assert s1["hidden"] is False
    assert s1["timers"] == 1
    assert s1["timerMs"] == result["visibleMs"]

    # The timer hides the toast...
    assert result["state2"]["hidden"] is True

    # ...and a later showToastOnce reuses the same element (no stacking).
    s3 = result["state3"]
    assert s3["created"] == 1
    assert s3["text"] == "m1"
    assert s3["hidden"] is False

    # A repeated key is a no-op: no re-toast, no extra timer churn.
    assert result["state4"]["text"] == "m1"

    # A new key toasts again (once-per-key, not once-per-session).
    assert result["state5"]["text"] == "m3"


# web/term.js — WebGL context-loss recovery (node-run logic)
# ---------------------------------------------------------------------------


def _recovery_region(js: str) -> str:
    """The self-contained WebGL recovery block, bounded by its section
    comment and the terminal-setup marker (concatenated with the toast
    region it depends on, then executed verbatim by the node test)."""
    start = js.index("/* ---------------- WebGL context-loss recovery")
    end = js.index("/* ---------------- terminal setup (per tab) ---------------- */")
    return js[start:end]


WEBGL_RECOVERY_NODE_DRIVER = """
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");

// Controllable state: the fake addon's failure mode and the clock the
// recovery budget's rolling window reads.
const state = { throwMode: null, now: 1000000 };

class FakeWebglAddon {
  constructor() {
    if (state.throwMode === "construct") throw new Error("no WebGL context");
    this.disposed = false;
    this.lossCb = null;
  }
  onContextLoss(cb) { this.lossCb = cb; }
  dispose() {
    if (state.throwMode === "dispose") throw new Error("context gone");
    this.disposed = true;
  }
  fireLoss() { if (this.lossCb) this.lossCb(); }
}

// Timers are recorded, not run — the test fires them explicitly by delay.
const timers = [];
let timerSeq = 0;
const setTimeoutStub = (fn, ms) => {
  const id = ++timerSeq;
  timers.push({ id, fn, ms });
  return id;
};
const clearTimeoutStub = (id) => {
  const i = timers.findIndex((t) => t.id === id);
  if (i >= 0) timers.splice(i, 1);
};
function fireTimers(ms) {
  for (let i = timers.length - 1; i >= 0; i--) {
    if (timers[i].ms === ms) timers.splice(i, 1)[0].fn();
  }
}

// Minimal DOM stub for the toast helpers (same shape as the toast driver).
function makeEl() {
  const classes = new Set();
  return {
    className: "",
    textContent: "",
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      contains: (c) => classes.has(c),
    },
  };
}
const created = [];
const documentStub = {
  createElement: () => { const el = makeEl(); created.push(el); return el; },
  body: { appendChild: () => {} },
};
const DateStub = { now: () => state.now };

function makeTab(id) {
  return {
    id,
    term: { addons: [], loadAddon(a) { this.addons.push(a); } },
    elTerm: { parentNode: {} },
    webgl: null,
    webglRecoverTimer: null,
    webglRecoverAttempts: [],
  };
}
const windowStub = { WebglAddon: { WebglAddon: FakeWebglAddon } };

const holder = {};
const full = src +
  "\\n;holder.api = { attachWebgl, handleWebglContextLoss, scheduleWebglRecovery," +
  " recoverWebgl, recreateWebgl, delay: WEBGL_RECOVER_DELAY_MS," +
  " restoredMessage: WEBGL_RESTORED_MESSAGE };";
new Function(
  "document", "setTimeout", "clearTimeout", "Date", "window", "holder", full
)(documentStub, setTimeoutStub, clearTimeoutStub, DateStub, windowStub, holder);
const api = holder.api;

const tab = makeTab("t1");
const toastText = () => (created.length ? created[0].textContent : null);
const toastVisible = () => (created.length ? !created[0].classList.contains("hidden") : null);
const out = {};

// 1. A healthy attach loads exactly one addon, schedules nothing.
out.attach = {
  ok: api.attachWebgl(tab),
  addonCount: tab.term.addons.length,
  timers: timers.length,
};

// 2. Context loss: the dead addon is disposed (xterm drops to the DOM
//    renderer at once), the once-per-tab fallback toast shows, and ONE
//    delayed recovery timer is armed.
tab.webgl.fireLoss();
out.loss = {
  webglGone: tab.webgl === null,
  addonDisposed: tab.term.addons[0].disposed,
  toastText: toastText(),
  toastVisible: toastVisible(),
  delayTimerScheduled: timers.some((t) => t.ms === api.delay),
};

// 3. The recovery timer fires with a cooperative webview: a fresh addon
//    (fresh glyph atlas) is attached, the budget is renewed and the
//    restored toast shows under its own dedup key.
fireTimers(api.delay);
out.recovered = {
  webglBack: tab.webgl !== null,
  freshAddon: tab.term.addons.length === 2 && !tab.term.addons[1].disposed,
  attempts: tab.webglRecoverAttempts.length,
  toastText: toastText(),
};

// 4. Repeated losses with a refusing webview (constructor throws) burn
//    the budget one attempt per loss, never crash.
state.throwMode = "construct";
for (let i = 0; i < 3; i++) {
  api.handleWebglContextLoss(tab);
  fireTimers(api.delay);
}
out.budget = {
  attempts: tab.webglRecoverAttempts.length,
  webglStillNull: tab.webgl === null,
};

// 5. Over budget: a further recovery is a no-op (attempt not recorded).
api.scheduleWebglRecovery(tab);
fireTimers(api.delay);
out.exhausted = { attempts: tab.webglRecoverAttempts.length };

// 6. Beyond the rolling window the budget is fresh again (the attempt
//    is recorded once more, the constructor still refuses).
state.now += 60000;
api.scheduleWebglRecovery(tab);
fireTimers(api.delay);
out.afterWindow = { attempts: tab.webglRecoverAttempts.length };

// 7. A user-driven font change renews the budget on success.
state.throwMode = null;
api.recreateWebgl(tab);
out.recreate = {
  webglBack: tab.webgl !== null,
  attempts: tab.webglRecoverAttempts.length,
};

// 8. recoverWebgl is a no-op while the addon is already healthy.
const addonsBefore = tab.term.addons.length;
const timersBefore = timers.length;
api.recoverWebgl(tab);
out.guard = {
  noNewAddon: tab.term.addons.length === addonsBefore,
  noNewTimer: timers.length === timersBefore,
};

console.log(JSON.stringify(out));
"""


def test_term_js_webgl_recovery_logic(tmp_path: Path) -> None:
    """The context-loss recovery path actually runs (node): a loss
    disposes the addon and arms a delayed recreate; a successful recreate
    restores WebGL and renews the budget; a refusing webview exhausts the
    budget into a stable DOM fallback, which the rolling window and a
    font change both renew."""
    js = _term_js()
    # recreateWebgl lives in the font-settings section — append its body so
    # the budget-renewal step runs against the real implementation.
    recreate = js[
        js.index("function recreateWebgl") : js.index("function refitAfterFontChange")
    ]
    region = _toast_region(js) + "\n;" + _recovery_region(js) + "\n;" + recreate
    out = _run_js_driver(WEBGL_RECOVERY_NODE_DRIVER, region, tmp_path)

    assert out["attach"] == {"ok": True, "addonCount": 1, "timers": 0}

    loss = out["loss"]
    assert loss["webglGone"] and loss["addonDisposed"]
    assert (
        loss["toastText"]
        == "WebGL renderer unavailable — slow DOM fallback is active (the terminal may feel sluggish)."
    )
    assert loss["toastVisible"]
    assert loss["delayTimerScheduled"]

    recovered = out["recovered"]
    assert recovered["webglBack"] and recovered["freshAddon"]
    assert recovered["attempts"] == 0  # budget renewed on success
    assert recovered["toastText"] == "WebGL renderer restored."

    assert out["budget"] == {"attempts": 3, "webglStillNull": True}
    assert out["exhausted"] == {"attempts": 3}
    assert out["afterWindow"] == {"attempts": 1}
    assert out["recreate"] == {"webglBack": True, "attempts": 0}
    assert out["guard"] == {"noNewAddon": True, "noNewTimer": True}
