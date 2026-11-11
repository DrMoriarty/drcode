"""In-binary runner for `.drcode/workflow.py` hooks in a frozen bundle.

In a frozen (PyInstaller) app `sys.executable` is the drcode binary itself,
not a python interpreter, so `WorkflowHooks.call` cannot spawn hooks with
the usual `python -c <bootstrap>` — the binary would ignore `-c`, fall
through `drcode.main()` into a second TUI instance and the hook would die
by `HOOK_TIMEOUT`. Instead the frozen path spawns
``<drcode binary> --run-hook <workflow.py>``: `drcode.main()` recognizes the
hidden flag before any heavy import (and before the TUI fallback) and hands
over to `run()` below, which performs exactly what the `_BOOTSTRAP` source
in `drcode.workflow` does — imports the workflow script, reads the JSON
payload from stdin, calls the requested hook function and prints the
returned dict as JSON.

Deliberately stdlib-only and free of drcode-internal imports: the hook
runner must not pay for the application's import graph.
"""

from __future__ import annotations

import importlib.util
import json
import sys

_FLAG = "--run-hook"


def run() -> int:
    """Execute one hook call; argv is ``… --run-hook <workflow.py>``.

    The JSON payload ({"hook": name, ...}) arrives on stdin, the hook's
    return value is printed as a JSON object on stdout. Errors (missing
    argument, failing import, a raising hook, an unserializable result)
    surface as a traceback and a non-zero exit — the parent turns that
    into ``HookResult(ok=False)`` exactly as for the `-c` bootstrap.
    """
    argv = sys.argv
    try:
        script = argv[argv.index(_FLAG) + 1]
    except (ValueError, IndexError):
        print(f"error: {_FLAG} requires a workflow script argument.", file=sys.stderr)
        return 2
    spec = importlib.util.spec_from_file_location("drcode_workflow_hook", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.load(sys.stdin)
    hook = getattr(module, payload["hook"], None)
    if hook is None:
        json.dump({}, sys.stdout)
    else:
        result = hook(payload)
        json.dump(result if result is not None else {}, sys.stdout)
    sys.stdout.flush()
    return 0
