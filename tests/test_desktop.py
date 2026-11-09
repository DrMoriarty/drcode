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

import fonts  # type: ignore[import-not-found]
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


async def _wait_for(ws: aiohttp.ClientWebSocketResponse, predicate, timeout: float = 5):
    """Read frames until ``predicate`` matches; return (match, all frames)."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    seen = []
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise AssertionError(f"condition not met in time; seen: {seen!r}")
        msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
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
                timeout=2,
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
    http: aiohttp.ClientSession, base_url: str, ctx: ShellContext,
    tab: str, project: Path,
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
                ws_a, lambda d: isinstance(d, bytes) and b"proj" in d, timeout=2
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
                lambda e: isinstance(e, dict) and e.get("event") in ("started", "error"),
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
    occupied = js[js.index("function projectOccupied") : js.index("function duplicateMessage")]
    assert "t.started" in occupied  # only a live session holds the project
    assert "t !== excludeTab" in occupied  # a tab may restart its own project
    dup = js[js.index("function duplicateMessage") : js.index("function launchProject")]
    assert "already open in another tab" in dup

    # launchProject blocks (welcome error on an unstarted tab, boot error
    # on a live one) instead of opening a duplicate.
    launch = js[js.index("function launchProject") : js.index("function fetchAvailableFonts")]
    assert "projectOccupied(project, tab)" in launch
    assert "showWelcomeError(duplicateMessage(project))" in launch
    assert "bootError(duplicateMessage(project))" in launch

    # The picker blocks BEFORE creating a welcome tab.
    picker = js[js.index("function launchFromPicker") : js.index("function loadRecents")]
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

    handler = js[
        js.index("function handleTabEvent") : js.index("function onTabExited")
    ]
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
