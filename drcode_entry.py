# PyInstaller entry stub: a plain script that imports the installed/source
# package and starts the TUI on the current working directory.
import os
import sys

# The logfire pydantic plugin (transitive dep of pydantic-ai) calls
# inspect.getsource() on pydantic's compiled PluggableSchemaValidator at
# import time, which fails in a frozen app (no source files). Disable it
# before pydantic is first imported. Disabled by entry-point name
# (`logfire-plugin`); `logfire` is matched too for older logfire versions.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "logfire-plugin,logfire")

# The code index's tree-sitter grammars are prefetched at BUILD time into
# the payload (scripts/build-binary.sh runs drcode.code_index.prefetch,
# drcode.spec ships the cache as drcode/ts_parsers — the on-demand
# download is impossible in a frozen, network-free app; §2.4 of
# docs/code-index-design.md). Point the language pack at the bundled
# cache before drcode (and thus the pack) is first imported.
if getattr(sys, "frozen", False):
    bundled = os.path.join(getattr(sys, "_MEIPASS", ""), "drcode", "ts_parsers")
    if os.path.isdir(bundled):
        os.environ.setdefault("DRCODE_TS_PACK_CACHE", bundled)

# Frozen mode must also behave like an interpreter for the classic `-c`
# one-liners external tooling spawns with sys.executable as the python
# command — notably merge_resolver's `_module_importable` probe
# (`<binary> -c "import <module>"`). Ignoring the argument would launch a
# FULL second drcode instance (live event loop, holding the parent's TTY
# on fd 0) instead of a one-shot import — two readers split the
# keyboard input of the real TUI. Unfrozen runs are untouched: there
# sys.executable is a real python and `-c` works natively.
if getattr(sys, "frozen", False) and "-c" in sys.argv:
    _idx = sys.argv.index("-c")
    if _idx + 1 >= len(sys.argv):
        # Mirror CPython's own message/exit code for `python -c` with no code.
        print("Argument expected for the -c option", file=sys.stderr)
        sys.exit(2)
    _code = sys.argv[_idx + 1]
    try:
        exec(compile(_code, "<string>", "exec"), {"__name__": "__main__"})  # noqa: S102 — the whole point: frozen interpreter emulation
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — mirror CPython: ANY error in -c code exits 1 with a traceback
        import traceback

        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)

from drcode import main

main()
