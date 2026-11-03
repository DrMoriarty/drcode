"""The `drcode.core` package must stay UI-neutral.

Two independent guards:

1. A source-level scan: no module under ``src/drcode/core`` may import
   ``textual`` (or anything from ``drcode.tui``) — not even under
   ``TYPE_CHECKING``, where the import is still a real coupling.
2. A runtime check in a fresh interpreter: importing every core module
   must not pull ``textual`` into ``sys.modules`` (catches transitive
   imports the AST scan could miss).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent.parent / "src" / "drcode" / "core"


def _iter_core_modules() -> list[Path]:
    return sorted(CORE_DIR.rglob("*.py"))


def _import_targets(node: ast.AST) -> list[str]:
    """Absolute module names imported by an import statement."""
    targets: list[str] = []
    if isinstance(node, ast.Import):
        targets.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        if node.level == 0 and node.module:
            targets.append(node.module)
        elif node.level > 0:
            # Relative import: resolve against the file's package to
            # catch `from ..tui.blocks import ...` style escapes.
            targets.append(f"<relative+{node.level}>{node.module or ''}")
    return targets


def test_core_sources_never_import_textual_or_tui() -> None:
    offenders: list[str] = []
    for path in _iter_core_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for target in _import_targets(node):
                if target == "textual" or target.startswith("textual."):
                    offenders.append(f"{path.name}: {target}")
                if target.startswith("<relative+"):
                    level = int(target.split("+")[1].split(">")[0])
                    # level 1 = drcode.core, level 2 = drcode — both fine;
                    # level >= 3 would climb into a sibling package (tui).
                    if level >= 3:
                        offenders.append(f"{path.name}: {target}")
    assert offenders == [], "core/ imports a UI package:\n" + "\n".join(offenders)


def test_core_imports_do_not_pull_textual_at_runtime() -> None:
    modules = sorted(
        f"drcode.core.{p.relative_to(CORE_DIR).with_suffix('')}".replace("/", ".")
        for p in _iter_core_modules()
        if p.name != "__init__.py"
    )
    code = (
        "import importlib, sys\n"
        f"for m in {modules!r}:\n"
        "    importlib.import_module(m)\n"
        "assert 'textual' not in sys.modules, sorted("
        "m for m in sys.modules if m.startswith('textual'))\n"
        "print('core runtime import: clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "core runtime import: clean" in result.stdout


def test_core_runtime_imports_clean_in_subprocess() -> None:
    """The tui facade keeps working while core stays import-parallel."""
    code = (
        "import sys\n"
        "from drcode.core.pipeline import PipelineLayer\n"
        "from drcode.core.state import CoreState\n"
        "assert 'textual' not in sys.modules\n"
        "print('facade-free core ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
