"""Compat re-export shim for the conflict screen package.

The conflict-resolution UI lives in the :mod:`drcode.tui.merge` package
(one module per UI unit: ``stages``/``views``/``check_details``/
``widgets``/``screen`` — see its docstring for the layout). This module
keeps the historical import path working for the lazy importers that
deliberately stay untouched:

- ``tui/actions_merge.py`` (``from .merge_screen import ConflictScreen``
  behind the package-drift guard — the widget→app screen edge);
- the tests (``tests/test_conflict_screen.py``,
  ``tests/test_unmerged_list_screen.py``), which also reach the private
  helpers (`_elide_middle`, `_diffstat_row_text`);
- the stale-module detection of `package_watch.py`, which fingerprints
  this file (and the submodules) on disk.

Every public and historically-imported private name is re-exported
unchanged; new code should import from ``drcode.tui.merge`` directly.
"""

from .merge import (
    CheckDetailsScreen,
    CheckOutputView,
    ConflictScreen,
    ConflictView,
    DiffStatItem,
    FileItem,
    _CheckStage,
    _diffstat_row_text,
    _elide_middle,
    _fmt_seconds,
    _PrepareStage,
    _stages_estimate,
    _stages_eta,
    _stages_percent,
)

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
