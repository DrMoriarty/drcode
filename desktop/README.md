# DrCode Desktop

Native shell around the drcode TUI: a pywebview window that spawns the
existing `drcode` onedir bundle (see `scripts/build-binary.sh`) inside a
PTY and relays bytes to an embedded `xterm.js` terminal.

**The shell contains zero drcode logic.** Functional parity with the TUI
holds by construction: the payload is the unmodified onedir bundle, taken
as `build-binary.sh` produced it, and run with `cwd` = the user's chosen
project folder (drcode operates on `Path.cwd()`). The bundle's version
inside a desktop build is the repo root `version` file (same `v*` tag).

## Layout

| file | role |
|---|---|
| `app.py` | pywebview window + lifecycle: starts the aiohttp server thread, native folder dialog, clipboard bridge, fullscreen/title/close |
| `server.py` | aiohttp on `127.0.0.1:<ephemeral>`; serves `web/`; `/ws?token=<random>` is the WebSocket ↔ PTY bridge (pywebview's own HTTP helper has no WebSocket support — not used) |
| `pty_session.py` | `os.openpty()` + spawn; asyncio read/write pumps; window resize → `ioctl(TIOCSWINSZ)` (Textual reacts to SIGWINCH itself); child env `TERM=xterm-256color`, `COLORTERM=truecolor` |
| `projects.py` | recent projects (`~/.config/drcode-desktop.json`, XDG-aware) |
| `packaging.py` | `.app` / AppImage / `.deb` assembly (see below) |
| `desktop.spec` | PyInstaller spec for the SHELL; copies the pre-built payload in as `resources/drcode` |
| `web/` | `index.html`, `term.js`, styles, vendored xterm.js UMD bundles + fit/WebGL addons (no npm/webpack), bundled Nerd font |

One pty session per window — no tabs, by design. The window closes when
drcode exits with code 0; a crash (non-zero exit) opens a restart dialog.

## Terminal compatibility

- **OSC 52** — `term.js` intercepts `ESC]52;…` in the output byte stream
  *before* xterm.js and routes it to the system clipboard:
  `navigator.clipboard` first, then the pywebview bridge
  (`api.set_clipboard` → `NSPasteboard` on macOS, `wl-copy`/`xclip`/`xsel`
  on Linux). Without this handler, copying in the TUI would not work.
- **Bracketed paste** — handled by xterm.js out of the box (DECSET 2004).
- **Nerd font** — bundled in `web/fonts/` via `@font-face`.
- **Mouse** — SGR escapes, covered by xterm.js; kitty protocol not
  required (graceful fallback).
- **Cyrillic / IME** — UTF-8 bytes flow to xterm.js unmodified (it keeps
  its own UTF-8 state across chunks); composition works natively on
  WebKit. Verified in the macOS spike.
- **Alternate screen** — scrollback is disabled in xterm.js; Textual
  scrolls its own content.
- **Voice** — sounddevice lives inside the drcode process; the shell
  needs nothing (the macOS bundle declares `NSMicrophoneUsageDescription`
  so the child's push-to-talk prompts correctly).

## Building

```bash
bash scripts/build-binary.sh     # payload first: dist/drcode/
uv sync --group desktop --group build
uv run --group desktop --group build python -m desktop.packaging --target macos     # DrCode.app + zip
uv run --group desktop --group build python -m desktop.packaging --target appimage  # .AppImage (Linux)
uv run --group desktop --group build python -m desktop.packaging --target deb       # .deb (Linux)
```

- **macOS** — PyInstaller BUNDLE with Info.plist + generated icon;
  ad-hoc codesigned locally (first launch: right-click → Open). Release
  builds: set `DRCODE_CODESIGN_IDENTITY` to a Developer ID Application
  certificate (then notarize the zip externally) — an external condition,
  not automated here.
- **AppImage** — bundles WebKitGTK: `packaging.py` copies the seed libs
  (`libwebkit2gtk-4.1`, `libgtk-3`, …) plus their `ldd` closure into
  `AppDir/usr/lib`, bundles all gobject-introspection typelibs, and wraps
  it in an `AppRun` that extends `LD_LIBRARY_PATH`/`GI_TYPELIB_PATH`.
  Build-time needs: `libwebkit2gtk-4.1-dev libgtk-3-dev`.
- **.deb** — installs to `/opt/drcode-desktop/` with
  `/usr/bin/drcode-desktop` and a desktop entry; `Depends:` on the
  distro's WebKitGTK instead of bundling it.

The shell is built into `dist-desktop/`/`build-desktop/` (NOT the repo
root `dist/`): on macOS's case-insensitive filesystem, a `DrCode` output
would collide with the payload's `drcode` directory and corrupt it.

## Dev mode / smoke run

```bash
uv sync --group desktop --group build
bash scripts/build-binary.sh                 # or place a bundle in dist/
uv run --group desktop python desktop/app.py --project /path/to/project
```

Useful switches: `--payload <bundle-executable>` overrides the payload
path. `DRCODE_DESKTOP_DEBUG=1` turns on pywebview's JS console and
server-side control-message logging.

CI (`.github/workflows/build-binaries.yml`, `desktop` job) builds the
macOS zip and the Linux AppImage/.deb on `v*` tags and attaches them to
the GitHub Release.
