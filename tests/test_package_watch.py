"""Tests for the on-disk package drift detector (`drcode.package_watch`).

The scenario this guards: the installed drcode package is upgraded while
the TUI runs — stale modules stay in sys.modules, lazy screen imports
read fresh files, the next Ctrl+M dies with a cryptic ImportError. The
detector captures a startup fingerprint (dist version + stat pairs of
core files) and reports any later change. These tests exercise the pure
fingerprint/diff layer against a synthetic package directory — no real
metadata or repo files are touched.
"""

from __future__ import annotations

import os
from pathlib import Path

import drcode
from drcode import package_watch


def _fake_package(tmp_path: Path) -> Path:
    """A synthetic package dir carrying every watched file."""
    root = tmp_path / "pkg"
    for rel in package_watch.WATCHED_FILES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel}\n", encoding="utf-8")
    return root


def _capture(root: Path) -> dict[str, object]:
    captured = package_watch.capture(root)
    assert captured is not None
    return captured


def test_capture_fingerprints_all_watched_files(tmp_path: Path) -> None:
    root = _fake_package(tmp_path)
    fingerprint = _capture(root)
    files = fingerprint["files"]
    assert isinstance(files, dict)
    assert set(files) == {rel.as_posix() for rel in package_watch.WATCHED_FILES}
    # Real stat pairs, not the missing-marker.
    assert all(value != "missing" for value in files.values())
    # Version comes from the real installed dist in the test env —
    # present but irrelevant to the file-level assertions here.
    assert isinstance(fingerprint["version"], str | None)


def test_drift_none_for_identical_fingerprints(tmp_path: Path) -> None:
    root = _fake_package(tmp_path)
    baseline = _capture(root)
    assert package_watch.drift(baseline, _capture(root)) is None


def test_drift_detects_touched_file(tmp_path: Path) -> None:
    root = _fake_package(tmp_path)
    baseline = _capture(root)
    target = root / "tools.py"
    os.utime(target, ns=(1_000_000_000, 1_000_000_000))
    reason = package_watch.drift(baseline, _capture(root))
    assert reason is not None
    assert "tools.py" in reason


def test_drift_detects_size_change(tmp_path: Path) -> None:
    root = _fake_package(tmp_path)
    baseline = _capture(root)
    (root / "conflict.py").write_text("# rewritten\n", encoding="utf-8")
    reason = package_watch.drift(baseline, _capture(root))
    assert reason is not None
    assert "conflict.py" in reason


def test_drift_detects_missing_file(tmp_path: Path) -> None:
    root = _fake_package(tmp_path)
    baseline = _capture(root)
    (root / "tui" / "merge_screen.py").unlink()
    reason = package_watch.drift(baseline, _capture(root))
    assert reason is not None
    assert "tui/merge_screen.py" in reason


def test_drift_reports_version_change_with_new_version(tmp_path: Path) -> None:
    old = {"version": "1.2.3", "files": {}}
    new = {"version": "1.3.0", "files": {}}
    reason = package_watch.drift(old, new)
    assert reason is not None
    assert "1.2.3 → 1.3.0" in reason


def test_drift_with_missing_files_component(tmp_path: Path) -> None:
    # Defensive: a malformed fingerprint never crashes the diff.
    reason = package_watch.drift({"version": None}, {"version": None, "files": {}})
    assert reason is not None


def test_drift_since_baseline_uses_baseline_capture(
    tmp_path: Path, monkeypatch
) -> None:
    root = _fake_package(tmp_path)
    monkeypatch.setattr(package_watch, "package_root", lambda: root)
    monkeypatch.setattr(package_watch, "_BASELINE", None)
    assert package_watch.drift_since_baseline() is None
    # An upgrade lands: a watched file changes on disk.
    (root / "tools.py").write_text("# upgraded\n", encoding="utf-8")
    reason = package_watch.drift_since_baseline()
    assert reason is not None
    assert "tools.py" in reason


def test_drift_since_baseline_none_without_package(tmp_path: Path, monkeypatch) -> None:
    # Frozen build / exotic layout: opt-out, never a false positive.
    monkeypatch.setattr(package_watch, "package_root", lambda: None)
    monkeypatch.setattr(package_watch, "_BASELINE", None)
    assert package_watch.drift_since_baseline() is None


def test_capture_baseline_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    root = _fake_package(tmp_path)
    monkeypatch.setattr(package_watch, "package_root", lambda: root)
    monkeypatch.setattr(package_watch, "_BASELINE", None)
    first = package_watch.capture_baseline()
    # The package "upgrades" after the baseline was pinned — a re-call
    # must keep the ORIGINAL fingerprint (it pins startup, not "now").
    (root / "tools.py").write_text("# upgraded\n", encoding="utf-8")
    assert package_watch.capture_baseline() == first
    assert first is not None
    files = first["files"]
    assert isinstance(files, dict)
    assert files.get("tools.py") != "missing"


def test_watched_files_exist_in_the_real_package() -> None:
    # Sanity: the watched set tracks real files of the shipped layout —
    # a renamed module would otherwise silently fingerprint "missing"
    # forever and the detector would still work, but blind.
    root = Path(drcode.__file__).resolve().parent
    for rel in package_watch.WATCHED_FILES:
        assert (root / rel).is_file(), rel
