"""The conflict-resolution screen package (`drcode.tui.merge`).

The visual (human) layer of the three-layer conflict stack, one module
per UI unit:

- ``stages`` — stage dataclasses and the ETA/percent arithmetic of the
  progress panel (widget-free, unit-testable);
- ``views`` — the line-API scroll views (`ConflictView`, the zdiff3
  hunk viewer with its cursor, and `CheckOutputView`);
- ``check_details`` — the `CheckDetailsScreen` modal (one check's full
  output, ``[`` / ``]`` step through the checks);
- ``widgets`` — the file-pane list items (`FileItem`, `DiffStatItem`);
- ``screen`` — the `ConflictScreen` modal itself (the ~1600-line
  orchestrator: prepare, funnel, commit/abort/ff, resolver watch).

`tui/merge_screen.py` stays as a thin re-export shim so the lazy
importers (`tui/actions_merge.py`, the tests) keep their import paths —
and the stale-module drift guard in `package_watch.py` keeps watching
both the shim and the submodules.
"""

from .check_details import CheckDetailsScreen
from .screen import ConflictScreen
from .stages import (
    _CheckStage,
    _elide_middle,
    _fmt_seconds,
    _PrepareStage,
    _stages_estimate,
    _stages_eta,
    _stages_percent,
)
from .views import CheckOutputView, ConflictView
from .widgets import DiffStatItem, FileItem, _diffstat_row_text

__all__ = [
    "CheckDetailsScreen",
    "CheckOutputView",
    "ConflictScreen",
    "ConflictView",
    "DiffStatItem",
    "FileItem",
    "_CheckStage",
    "_PrepareStage",
    "_diffstat_row_text",
    "_elide_middle",
    "_fmt_seconds",
    "_stages_estimate",
    "_stages_eta",
    "_stages_percent",
]
