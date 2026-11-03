"""DrCode Desktop — native shell around the drcode TUI.

pywebview window + the aiohttp server lifecycle. The shell contains zero
drcode logic: it spawns the unmodified onedir bundle
(``scripts/build-binary.sh`` → ``dist/drcode/drcode``) inside a PTY with
the user's chosen project as ``cwd`` and relays bytes between the PTY
and xterm.js.

Run from the repo root (dev mode)::

    uv run --group desktop python desktop/app.py --dev --project /path/to/project

Frozen builds are produced by ``desktop/packaging.py`` (see that module).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path

try:  # package import (tests) or script import (PyInstaller / dev run)
    from . import projects as projects_mod
    from .pty_session import resolve_payload
    from .server import ShellContext, run_server
except ImportError:  # pragma: no cover - script/frozen entry
    import projects as projects_mod  # type: ignore[no-redef]
    from pty_session import resolve_payload  # type: ignore[no-redef]
    from server import ShellContext, run_server  # type: ignore[no-redef]

try:
    import webview
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pywebview is required: run with `uv run --group desktop ...` "
        f"(desktop dependency group in pyproject.toml). Import error: {exc}"
    ) from exc

APP_TITLE = "DrCode"
WINDOW_DEFAULTS = {"width": 1100, "height": 760, "min_size": (480, 320)}


def _frozen_js_dir_candidates() -> list[str]:
    """Where desktop.spec may have placed the webview_js data directory.

    PyInstaller's data placement differs per layout: onedir (Linux
    AppImage/.deb, Windows) puts datas under ``sys._MEIPASS``; the macOS
    .app bundle puts data-only directories in ``Contents/Resources``
    while ``sys._MEIPASS`` points at ``Contents/Frameworks``.
    """
    meipass = getattr(sys, "_MEIPASS", "")
    exe_dir = os.path.dirname(sys.executable)
    candidates = []
    if meipass:
        candidates.append(os.path.join(meipass, "webview_js"))
    candidates.append(os.path.join(exe_dir, "..", "Resources", "webview_js"))
    candidates.append(os.path.join(exe_dir, "webview_js"))
    return candidates


def _fix_frozen_pywebview_js() -> None:
    """Point pywebview's get_js_dir at the bundled JS assets when frozen.

    In a PyInstaller bundle pywebview's pure modules live in the PYZ and
    ``webview.util.get_js_dir`` looks for a ``webview/js`` directory next
    to a ``__file__`` that does not exist on disk; its only fallback is
    the py2app layout, so it raises FileNotFoundError. The cocoa/gtk
    backends swallow that exception inside ``inject_pywebview`` and the
    ``window.pywebview`` bridge is never injected (no native dialogs,
    recents, clipboard fallback, fullscreen, set_title). desktop.spec
    ships the assets to ``webview_js``; ``load_js_files`` resolves
    ``get_js_dir`` through the module globals at call time, so replacing
    the attribute is enough. Dev mode is untouched.
    """
    if not getattr(sys, "frozen", False):
        return
    import webview.util

    original = webview.util.get_js_dir
    bundled = next(
        (os.path.abspath(c) for c in _frozen_js_dir_candidates() if os.path.isdir(c)),
        None,
    )

    def _get_js_dir() -> str:
        if bundled is not None:
            return bundled
        return original()

    webview.util.get_js_dir = _get_js_dir


class DesktopApi:
    """Methods exposed to the web page as ``window.pywebview.api.*``.

    All methods return JSON-serializable values and are called from
    pywebview's own threads.
    """

    def __init__(self, window_holder: dict, ctx: ShellContext, payload: Path) -> None:
        self._window_holder = window_holder
        self._ctx = ctx
        self._payload = payload

    # -- project selection -------------------------------------------------

    def choose_project(self) -> str | None:
        """Native folder dialog; returns the picked path or None."""
        window = self._window_holder.get("window")
        if window is None:
            return None
        result = window.create_file_dialog(webview.FOLDER_DIALOG)
        if not result:
            return None
        if isinstance(result, str):
            path = result
        else:
            path = result[0]
        return path or None

    def get_info(self) -> dict:
        return {
            "recents": projects_mod.load_recents(),
            "payload": str(self._payload),
            "project": self._ctx.current_project,
        }

    def open_recent(self, project: str) -> str | None:
        """Record-free convenience for the welcome screen (kept for parity)."""
        return project if Path(project).is_dir() else None

    # -- window controls ---------------------------------------------------

    def toggle_fullscreen(self) -> bool:
        window = self._window_holder.get("window")
        if window is None:
            return False
        window.toggle_fullscreen()
        return True

    def set_title(self, title: str) -> bool:
        window = self._window_holder.get("window")
        if window is None:
            return False
        window.set_title(title or APP_TITLE)
        return True

    def close_window(self) -> bool:
        window = self._window_holder.get("window")
        if window is None:
            return False
        window.destroy()
        return True

    # -- clipboard fallback -------------------------------------------------

    def set_clipboard(self, text: str) -> bool:
        """System clipboard from the Python side.

        The JS side prefers navigator.clipboard; this bridge covers
        WKWebView gesture restrictions and WebKitGTK quirks.
        """
        return set_system_clipboard(text)


def set_system_clipboard(text: str) -> bool:
    if not text:
        return False
    if sys.platform == "darwin":
        try:
            from AppKit import NSPasteboard  # type: ignore[import-not-found]  # pyobjc ships with pywebview on macOS

            pb = NSPasteboard.generalPasteboard()
            pb.clearContents()
            return bool(pb.setString_forType_(text, "public.utf8-plain-text"))
        except Exception:  # noqa: BLE001, S110 - best-effort clipboard, next fallback tried
            pass
    # Linux/other: try the usual CLI clipboard utilities.
    for cmd in (
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
    ):
        try:
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def _serve(ctx: ShellContext, ready: threading.Event) -> None:
    """Background thread: aiohttp server on its own loop, runs until app exit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ctx.loop = loop

    async def _start() -> str:
        _runner, base_url = await run_server(ctx)
        return base_url

    try:
        base_url = loop.run_until_complete(_start())
        ctx.base_url = base_url
    except Exception as exc:  # pragma: no cover  # noqa: BLE001 - report any server failure
        ctx.startup_error = f"shell server failed to start: {exc}"
        ready.set()
        return
    ready.set()
    try:
        loop.run_forever()
    finally:
        loop.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DrCode Desktop shell")
    parser.add_argument(
        "--payload",
        help="path to the drcode onedir bundle executable (default: auto-resolve)",
    )
    parser.add_argument(
        "--project",
        help="open this project directory immediately, skipping the welcome screen",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="dev mode: resolve the payload from the repo (dist/ or desktop/resources/)",
    )
    args = parser.parse_args(argv)

    try:
        payload = Path(args.payload) if args.payload else resolve_payload()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    if not payload.is_file():
        raise SystemExit(f"payload not found: {payload}")

    ctx = ShellContext(str(payload))

    ready = threading.Event()
    server_thread = threading.Thread(
        target=_serve, args=(ctx, ready), daemon=True, name="drcode-desktop-server"
    )
    server_thread.start()
    if not ready.wait(timeout=10) or getattr(ctx, "startup_error", None):
        raise SystemExit(getattr(ctx, "startup_error", "shell server did not start"))

    url = f"{ctx.base_url}/?token={ctx.token}"
    if args.project:
        url += f"&project={args.project}"
    print(f"[desktop] serving on {ctx.base_url}, window url ready", flush=True)

    _fix_frozen_pywebview_js()

    window_holder: dict = {}
    api = DesktopApi(window_holder, ctx, payload)

    # Recents: record on every successful launch (called in the server loop).
    def _record(project: str) -> None:
        projects_mod.add_recent(project, title=projects_mod.project_title(project))

    ctx.on_project_launched = _record

    window = webview.create_window(
        APP_TITLE,
        url=url,
        js_api=api,
        width=WINDOW_DEFAULTS["width"],
        height=WINDOW_DEFAULTS["height"],
        min_size=WINDOW_DEFAULTS["min_size"],
        text_select=False,
    )
    window_holder["window"] = window
    print("[desktop] window created, starting GUI loop", flush=True)

    # Clean exit of drcode (code 0) closes the whole shell window.
    def _close() -> None:
        try:
            window.destroy()
        except Exception:  # noqa: BLE001, S110 - window may already be gone
            pass

    ctx.on_clean_exit = _close

    try:
        webview.start(debug=os.environ.get("DRCODE_DESKTOP_DEBUG") == "1")
    except Exception as exc:
        print(f"[desktop] GUI loop crashed: {exc!r}", flush=True)
        raise
    finally:
        loop = ctx.loop
        if loop is not None and loop.is_running():
            try:
                stop = asyncio.run_coroutine_threadsafe(ctx.stop_session(), loop)
                stop.result(timeout=3)
            except Exception:  # noqa: BLE001, S110 - shutdown must not raise
                pass
            loop.call_soon_threadsafe(loop.stop)
        server_thread.join(timeout=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
