# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: standalone one-file `drcode` binary.

Build with `bash scripts/build-binary.sh` (or `.venv/bin/pyinstaller
--noconfirm --clean drcode.spec` after `uv sync --group build`).

The result is a self-contained executable that bundles its own Python
interpreter and all dependencies — no system Python or site-packages
required. It is platform-specific (build on each target OS/arch).
"""
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata

ROOT = Path(SPECPATH).resolve()

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
    ],
    # anyio imports its backend dynamically — PyInstaller can't see it
    # from static analysis alone.
    hiddenimports=["anyio._backends._asyncio"],
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
