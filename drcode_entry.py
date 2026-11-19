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

from drcode import main

main()
