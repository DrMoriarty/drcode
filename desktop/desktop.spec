# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the DrCode Desktop SHELL (not the drcode payload).

The shell is a small pywebview app (desktop/app.py + sibling modules).
The drcode TUI bundle itself is NOT rebuilt here — an already-built
onedir bundle (scripts/build-binary.sh → dist/drcode/) is copied into
the app as a data directory, untouched:

  - macos:  dist/DrCode.app  (BUNDLE, Contents/Frameworks/resources/drcode)
  - linux:  dist/DrCode/     (onedir, then AppImage/.deb via packaging.py)

Build via `python -m desktop.packaging --target <macos|appimage|deb>`,
which also generates the icon and runs codesign. The payload location can
be overridden with DRCODE_PAYLOAD, the icns with DRCODE_DESKTOP_ICNS.

Why the shell is windowed (console=False): pywebview owns the GUI; the
terminal lives in xterm.js, not in stdout. Any Python traceback still
reaches the user through the page's boot-error surface / crash dialog.
"""
import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent  # the spec lives in desktop/, repo root is its parent
DESKTOP = ROOT / "desktop"
VERSION = (ROOT / "version").read_text(encoding="utf-8").strip()

payload = Path(os.environ.get("DRCODE_PAYLOAD", ROOT / "dist" / "drcode")).resolve()
exe_name = "drcode.exe" if sys.platform == "win32" else "drcode"

datas = [
    # The web UI (index/term.js/vendor xterm bundles/fonts) — served by
    # desktop/server.py from web/ next to the frozen modules.
    (str(DESKTOP / "web"), "web"),
]
if (payload / exe_name).is_file():
    # The drcode onedir bundle, byte-for-byte as scripts/build-binary.sh
    # produced it: the shell never modifies its payload.
    datas.append((str(payload), "resources/drcode"))
else:
    print(
        f"WARNING: drcode payload not found at {payload} — the shell will "
        "not be able to start a session (dev builds resolve it externally).",
        file=sys.stderr,
    )

hiddenimports = [
    # pywebview imports its platform backend dynamically at runtime
    # (importlib.import_module('webview.platforms.<gui>')) — static
    # analysis cannot see it, name it explicitly per platform:
    "webview.platforms.cocoa" if sys.platform == "darwin" else "webview.platforms.gtk",
]

a = Analysis(
    [str(DESKTOP / "app.py")],
    pathex=[str(DESKTOP)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DrCode",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DrCode",
)

if sys.platform == "darwin":
    icns = os.environ.get("DRCODE_DESKTOP_ICNS", str(DESKTOP / "build" / "icon.icns"))
    app = BUNDLE(
        coll,
        name="DrCode.app",
        icon=icns if Path(icns).is_file() else None,
        info_plist={
            "CFBundleName": "DrCode",
            "CFBundleDisplayName": "DrCode",
            "CFBundleIdentifier": "com.drmoriarty.drcode-desktop",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "CFBundlePackageType": "APPL",
            "LSMinimumSystemVersion": "11.0",
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
            "NSSupportsAutomaticGraphicsSwitching": True,
            # The shell embeds a terminal; no microphone/camera keys here —
            # voice lives INSIDE the drcode process (sounddevice), which
            # inherits the entitlement prompt of this bundle automatically.
            "NSMicrophoneUsageDescription":
                "drcode uses the microphone for push-to-talk dictation.",
        },
    )
