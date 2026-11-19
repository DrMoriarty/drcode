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
the running installation is captured (the distribution version + mtime/size
of a handful of core files) and re-checked periodically — and right
before the risky lazy screen imports. On drift the user gets a toast
("the package was upgraded on disk, restart the app") and a service
event in the viewed task's log, so the eventual ImportError reads as an
environment mismatch, not as a code bug.

The frozen (PyInstaller onedir) build is NOT an opt-out, despite the
payload living outside site-packages: in onedir the lazy imports read
``PYZ-00.pyz`` and ``_internal`` from disk on every call, so an on-disk
rebuild under a live session applies in full (real case 17.11:
``build-binary.sh`` run while DrCode Desktop kept the bundle's drcode
alive — the next lazy import died with ``zlib.error: incorrect header
check`` from ``pyimod01_archive.extract``, cascading on shutdown). The
fingerprint is simply taken over a different root there: the executable
itself plus a few stable files under its ``_internal`` contents
directory (see ``bundle_root``). Onefile builds still opt out — their
payload is unpacked once per launch, an on-disk upgrade does not apply
to the running process.

Deliberately LLM-free, allocation-free and stat-only: the poll runs every
30 s on the UI thread, so it must stay in the microseconds. There is no
attempt to hot-reload modules — unloading a live pydantic/textual model
graph safely is not a thing.
"""

from __future__ import annotations

import os
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

#: Files fingerprinted inside the onedir bundle's ``_internal`` contents
#: directory (PyInstaller ≥ 6 layout: the PYZ archive and the frozen
#: stdlib live there, and it is exactly what lazy imports re-read from
#: disk). A small, stable set — every entry is one ``stat`` pair.
FROZEN_WATCHED: tuple[Path, ...] = (
    Path("PYZ-00.pyz"),
    Path("base_library.zip"),
)

#: Native extension libraries in ``_internal`` join the fingerprint too
#: (a rebuild swaps them all), but their exact names are version- and
#: platform-dependent — they are discovered by a deterministic sorted
#: scan, capped so the fingerprint stays a handful of stat pairs.
_NATIVE_SUFFIXES = frozenset({".so", ".dylib", ".dll"})
_NATIVE_LIB_CAP = 3


def bundle_root() -> Path | None:
    """Onedir bundle root — the directory holding ``sys.executable``.

    None outside a frozen PyInstaller build, without an executable path,
    or when that path cannot be resolved: fingerprinting simply opts
    out, never raises.
    """
    if not getattr(sys, "frozen", False):
        return None
    executable = getattr(sys, "executable", None)
    if not executable:
        return None
    try:
        return Path(executable).resolve().parent
    except OSError:  # pragma: no cover - defensive, resolve rarely raises
        return None


def _native_libraries(internal: Path) -> list[Path]:
    """A few native libraries from ``_internal``, sorted, capped.

    The sorted scan keeps the fingerprint stable between captures; the
    cap keeps it at a handful of stat pairs. An unreadable directory
    yields [] — the caller records the watched entries as "missing".
    """
    try:
        names = sorted(
            entry.name
            for entry in os.scandir(internal)
            if entry.is_file() and Path(entry.name).suffix in _NATIVE_SUFFIXES
        )
    except OSError:
        return []
    return [Path(name) for name in names[:_NATIVE_LIB_CAP]]


def _capture_bundle(root: Path) -> dict[str, object] | None:
    """Fingerprint a frozen onedir bundle (see ``capture``).

    The executable is the anchor: when it is unstatable there is no
    meaningful bundle and the capture degrades to None. The ``_internal``
    entries degrade individually to the "missing" marker — a rebuild
    that removes a file must read as drift — but when NOTHING under
    ``_internal`` stats (an onefile layout, an exotic bundle), the
    fingerprint would be blind, so it returns None instead.
    """
    try:
        dist_version: str | None = _dist_version("drcode")
    except PackageNotFoundError:
        dist_version = None
    files: dict[str, tuple[int, int] | str] = {}
    try:
        st = Path(sys.executable).stat()
        files["<executable>"] = (st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    internal = root / "_internal"
    watched = [*FROZEN_WATCHED, *_native_libraries(internal)]
    for rel in watched:
        try:
            st = (internal / rel).stat()
            files[rel.as_posix()] = (st.st_mtime_ns, st.st_size)
        except OSError:
            files[rel.as_posix()] = _MISSING
    if all(value == _MISSING for key, value in files.items() if key != "<executable>"):
        # Nothing under _internal stats (or nothing was watched at all):
        # the fingerprint would be blind — degrade to None.
        return None
    return {"version": dist_version, "files": files}


def package_root() -> Path | None:
    """Directory of the installed ``drcode`` package, or None.

    None under a frozen PyInstaller build (the bundle is fingerprinted
    through ``bundle_root`` instead — see ``capture``) or when the
    package layout makes file-level fingerprinting meaningless.
    """
    if getattr(sys, "frozen", False):
        return None
    import drcode

    file = getattr(drcode, "__file__", None)
    if file is None:  # pragma: no cover - namespace-package corner
        return None
    return Path(file).resolve().parent


def capture(root: Path | None = None) -> dict[str, object] | None:
    """Fingerprint the running installation: version + stat pairs.

    Non-frozen, the fingerprint covers the installed package; frozen
    (onedir), it covers the bundle — ``sys.executable`` plus a few
    stable files under its ``_internal`` contents directory (see
    ``_capture_bundle``), because lazy imports there read the PyInstaller
    archive from disk on every call. Either way, returns None when
    fingerprinting is not applicable (onefile build, exotic layout).
    Individual failures degrade: a missing dist metadata yields a None
    version, a vanished/unreadable file a ``"missing"`` marker — the
    comparison below treats any change, including to missing, as drift.
    """
    if getattr(sys, "frozen", False):
        if root is None:
            root = bundle_root()
        return None if root is None else _capture_bundle(root)
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
    baseline) — a onefile build or an exotic layout simply opts out.
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
