"""Application version, single source of truth: the `version` file.

The file lives at the repository root and contains nothing but the version
string ("1.2.3"). This module reads it once at import and exposes it as
`APP_VERSION` — every place that displays the version must use it.

When the package is installed (and no `version` file is discoverable next
to the source tree), the version falls back to the package metadata.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path


def _frozen_version() -> str | None:
    """Version from the `version` file bundled into a PyInstaller build.

    Frozen binaries carry the file next to the unpacked payload
    (`sys._MEIPASS`); no `pyproject.toml` exists there, so the regular
    discovery below would never see it.
    """
    if not getattr(sys, "frozen", False):
        return None
    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    try:
        text = (base / "version").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


@lru_cache(maxsize=1)
def app_version() -> str:
    frozen = _frozen_version()
    if frozen:
        return frozen
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "version"
        # The pyproject.toml guard keeps us from picking up a random
        # `version` file somewhere above an installed copy.
        if candidate.is_file() and (parent / "pyproject.toml").is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text
    try:
        return _pkg_version("drcode")
    except PackageNotFoundError:
        return "0.0.0"


APP_VERSION = app_version()
