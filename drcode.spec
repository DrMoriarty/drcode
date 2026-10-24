# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: standalone `drcode` binary.

Build with `bash scripts/build-binary.sh` (or `.venv/bin/pyinstaller
--noconfirm --clean drcode.spec` after `uv sync --group build`).

The result is a self-contained `dist/drcode/` folder (onedir bundle) with
the `drcode` executable inside — it bundles its own Python interpreter and
all dependencies, no system Python or site-packages required. It is
platform-specific (build on each target OS/arch).

Why onedir and not one-file: a one-file executable unpacks its whole
payload into a fresh temp directory on EVERY start, and on hardened macOS
setups each freshly written binary/dylib then costs hundreds of
milliseconds to load (per-file signature verification by the OS) —
`drcode --version` measured 10-20s, and even a hello-world one-file build
took 3-5s. With onedir the files are written once at unpack time and
subsequent starts load them straight from disk: ~0.4s cold binary start.
Set DRCODE_ONEFILE=1 to build a legacy single-file executable instead.
"""
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

ROOT = Path(SPECPATH).resolve()
ONEFILE = os.environ.get("DRCODE_ONEFILE", "") not in ("", "0", "false")

a = Analysis(
    [str(ROOT / "drcode_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[
        # Bundled prompt templates: read via `Path(__file__).parent / "prompts"`,
        # which inside a frozen app resolves to <_MEIPASS>/drcode/prompts.
        (str(ROOT / "src" / "drcode" / "prompts"), "drcode/prompts"),
        # Single source of the app version (see src/drcode/version.py).
        (str(ROOT / "version"), "."),
        # genai_prices (via pydantic-ai) and pydantic_ai look up their own
        # dist metadata at import time (importlib.metadata.version); PyInstaller
        # does not bundle *.dist-info by default, so copy it in explicitly.
        *copy_metadata("genai_prices"),
        *copy_metadata("pydantic_ai_slim"),
        # Voice extra (drcode.voice → sounddevice): the wheel ships the
        # PortAudio binary library inside `_sounddevice_data` — without
        # collecting it the frozen app imports sounddevice but fails at
        # the first actual device call with "OSError: PortAudio library
        # not found". The voice extra must be installed into the build
        # venv (build-binary.sh does `uv sync --extra voice`).
        *collect_data_files("sounddevice"),
    ],
    # anyio imports its backend dynamically — PyInstaller can't see it
    # from static analysis alone; sounddevice is the optional voice extra
    # (imported lazily by drcode.voice; numpy-free via raw streams).
    hiddenimports=["anyio._backends._asyncio", "sounddevice"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="drcode",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        exclude_binaries=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="drcode",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
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
        name="drcode",
    )
