"""On-disk package drift detection for a live TUI session.

When the installed drcode package is replaced while the TUI is running
(``uv tool upgrade``, a re-install, or a developer editing the source
tree), the already-imported modules keep serving the OLD bytecode from
``sys.modules`` while lazy imports (the widget→app screen edges, mostly)
read the NEW files from disk. The result is the notorious half-upgraded
zoo: ``ImportError: cannot import name ... from 'drcode.tools'`` deep
inside a screen import — the function exists on disk, it just does not
exist in the stale in-memory module (real case 31.10→01.11: Ctrl+M →
``merge_screen`` → ``conflict`` crashed exactly this way).

The fix is detection, not recovery: at TUI startup a cheap fingerprint of
the installed package is captured (the distribution version + mtime/size
of a handful of core files) and re-checked periodically — and right
before the risky lazy screen imports. On drift the user gets a toast
("the package was upgraded on disk, restart the app") and a service
event in the viewed task's log, so the eventual ImportError reads as an
environment mismatch, not as a code bug.

Deliberately LLM-free, allocation-free and stat-only: the poll runs every
30 s on the UI thread, so it must stay in the microseconds. There is no
attempt to hot-reload modules — unloading a live pydantic/textual model
graph safely is not a thing.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path

#: Files fingerprinted relative to the package root. A small, stable set
#: spanning the layers whose lazy-import edges produced the real-world
#: crashes: the tool/agent core (imported eagerly — the stale side of
#: every mismatch) and the lazily imported TUI screens (the fresh side).
#: Every entry is one ``stat`` pair — the list can grow, the cost stays
#: trivial.
WATCHED_FILES: tuple[Path, ...] = (
    Path("tools.py"),
    Path("conflict.py"),
    Path("agent.py"),
    Path("task_manager.py"),
    Path("merge_engine.py"),
    # The conflict screen: a thin re-export shim over the tui/merge/
    # package — a stale submodule behind a fresh shim (or vice versa)
    # crashes exactly like the pre-split monolith did, so both sides of
    # the edge are fingerprinted.
    Path("tui") / "merge_screen.py",
    Path("tui") / "merge" / "screen.py",
    Path("tui") / "merge" / "views.py",
    Path("tui") / "merge" / "stages.py",
    Path("tui") / "merge" / "check_details.py",
    Path("tui") / "merge" / "widgets.py",
    Path("tui") / "unmerged_screen.py",
)

#: Drift reason carries the nature of the change for the toast / event.
_MISSING = "missing"


def package_root() -> Path | None:
    """Directory of the installed ``drcode`` package, or None.

    None under a frozen PyInstaller build (the payload is unpacked once
    per launch — an on-disk upgrade does not apply to a running binary)
    or when the package layout makes file-level fingerprinting
    meaningless.
    """
    if getattr(sys, "frozen", False):
        return None
    import drcode

    file = getattr(drcode, "__file__", None)
    if file is None:  # pragma: no cover - namespace-package corner
        return None
    return Path(file).resolve().parent


def capture(root: Path | None = None) -> dict[str, object] | None:
    """Fingerprint the installed package: version + stat pairs.

    Returns None when fingerprinting is not applicable (frozen build).
    Individual failures degrade: a missing dist metadata yields a None
    version, a vanished/unreadable file a ``"missing"`` marker — the
    comparison below treats any change, including to missing, as drift.
    """
    if root is None:
        root = package_root()
        if root is None:
            return None
    try:
        dist_version: str | None = _dist_version("drcode")
    except PackageNotFoundError:
        dist_version = None
    files: dict[str, tuple[int, int] | str] = {}
    for rel in WATCHED_FILES:
        try:
            st = (root / rel).stat()
            files[rel.as_posix()] = (st.st_mtime_ns, st.st_size)
        except OSError:
            files[rel.as_posix()] = _MISSING
    return {"version": dist_version, "files": files}


def drift(baseline: dict[str, object], current: dict[str, object]) -> str | None:
    """Human-readable reason the fingerprints differ, or None.

    The version part is only reported when it actually changed — a
    same-version reinstall is caught by the file stats instead.
    """
    if baseline == current:
        return None
    parts: list[str] = []
    old_version = baseline.get("version")
    new_version = current.get("version")
    if old_version != new_version and new_version is not None:
        parts.append(f"version {old_version} → {new_version}")
    old_files = baseline.get("files")
    new_files = current.get("files")
    if isinstance(old_files, dict) and isinstance(new_files, dict):
        changed = sorted(
            name
            for name in set(old_files) | set(new_files)
            if old_files.get(name) != new_files.get(name)
        )
        if changed:
            shown = ", ".join(changed[:3])
            if len(changed) > 3:
                shown += f" +{len(changed) - 3} more"
            parts.append(f"changed on disk: {shown}")
    return "; ".join(parts) or "package files changed"


# The baseline is captured at import time of this module — which the app
# pulls in at startup, before any lazy screen import can run. One capture
# per process; nothing rewrites it.
_BASELINE: dict[str, object] | None = None


def capture_baseline() -> dict[str, object] | None:
    """Pin the startup fingerprint explicitly (idempotent).

    Called from the app constructor so the baseline predates the first
    lazy screen import; the first poll at +30 s would otherwise become
    the baseline and mask an upgrade landing in that first half minute.
    """
    global _BASELINE
    if _BASELINE is None:
        _BASELINE = capture()
    return _BASELINE


def _baseline() -> dict[str, object] | None:
    return capture_baseline()


def drift_since_baseline() -> str | None:
    """Drift reason against the startup capture, or None.

    Always None when fingerprinting was not applicable at startup (no
    baseline) — a frozen binary or an exotic layout simply opts out.
    """
    baseline = _baseline()
    if baseline is None:
        return None
    try:
        current = capture()
    except OSError:  # pragma: no cover - stat failures are folded in
        return None
    if current is None:
        return None
    return drift(baseline, current)
