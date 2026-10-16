"""Application version, single source of truth: the `version` file.

The file lives at the repository root and contains nothing but the version
string ("1.2.3"). This module reads it once at import and exposes it as
`APP_VERSION` — every place that displays the version must use it.

When the package is installed (and no `version` file is discoverable next
to the source tree), the version falls back to the package metadata.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path


@lru_cache(maxsize=1)
def app_version() -> str:
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
