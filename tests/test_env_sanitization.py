"""Environment sanitation for child processes (run_bash and workflow hooks).

When drcode is launched from a shell with another project's venv/conda
activated, VIRTUAL_ENV and friends leak into every subprocess the agent
spawns: `uv` prints the foreign path into tool-returns, and `uv run
--active` can delete and recreate the foreign venv. These tests pin the
contract: the listed variables never reach a child process, while
everything else (PATH, HOME, DRCODE_*) passes through unchanged.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import pytest

from drcode.tools import STRIPPED_ENV_VARS, run_bash, sanitized_env
from tests.test_workflow import make_hooks, make_script

FOREIGN_ENV = {
    "VIRTUAL_ENV": "/Users/someone/pr/ru-finance-mcp/.venv",
    "PYTHONHOME": "/opt/some-python",
    "PYTHONPATH": "/Users/someone/pr/ru-finance-mcp/src",
    "CONDA_PREFIX": "/opt/miniconda3/envs/other",
    "CONDA_DEFAULT_ENV": "other",
}


def test_sanitized_env_strips_foreign_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    for var, value in FOREIGN_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv("DRCODE_PROBE", "kept")
    env = sanitized_env()
    for var in FOREIGN_ENV:
        assert var not in env
    assert env["DRCODE_PROBE"] == "kept"


def test_sanitized_env_keeps_rest_of_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/foreign/.venv")
    env = sanitized_env()
    assert env["PATH"] == os.environ["PATH"]
    assert env["HOME"] == os.environ["HOME"]
    assert env.get("VIRTUAL_ENV") is None


def test_stripped_env_vars_are_exactly_documented_set() -> None:
    assert STRIPPED_ENV_VARS == (
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
    )


async def test_run_bash_child_env_is_sanitized(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var, value in FOREIGN_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv("DRCODE_PROBE", "kept")
    prog = (
        "import os, json; "
        f"print(json.dumps({{v: os.environ.get(v) for v in {list(FOREIGN_ENV)!r}}}), "
        "os.environ.get('DRCODE_PROBE'), "
        "bool(os.environ.get('PATH')), "
        "bool(os.environ.get('HOME')))"
    )
    out = await run_bash(project_root, f"{shlex.quote(sys.executable)} -c {prog!r}")
    body = out.split("\n", 1)[1]
    # The child saw none of the foreign variables…
    assert "/Users/someone" not in body
    assert "/opt/miniconda3" not in body
    for var in FOREIGN_ENV:
        assert f"'{var}': null" in body or f'"{var}": null' in body
    # …but kept the sanctioned ones (probe value, then PATH/HOME booleans).
    assert "kept" in body
    assert " True True" in body


async def test_workflow_hook_child_env_is_sanitized(
    project_root: Path,
    manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var, value in FOREIGN_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv("DRCODE_PROBE", "kept")
    make_script(
        project_root,
        "import os\n\n"
        "def on_task_start(task):\n"
        "    return {\n"
        "        'virtual_env': os.environ.get('VIRTUAL_ENV'),\n"
        "        'conda_prefix': os.environ.get('CONDA_PREFIX'),\n"
        "        'probe': os.environ.get('DRCODE_PROBE'),\n"
        "        'path': bool(os.environ.get('PATH')),\n"
        "    }\n",
    )
    hooks = make_hooks(project_root)
    task = manager.create("t", "")
    result = await hooks.call("on_task_start", task)
    assert result.ok, result.error
    out = result.output
    assert out["virtual_env"] is None
    assert out["conda_prefix"] is None
    assert out["probe"] == "kept"
    assert out["path"] is True
