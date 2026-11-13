"""`python -m drcode.mcp_proxy` — the stdio shim entry point (design doc §1).

The module exists only so the console script (`drcode-mcp` in
`[project.scripts]`, pyproject.toml) and `python -m drcode.mcp_proxy`
resolve to the same callable. All logic lives in `drcode.mcp.proxy`,
next to the rest of the MCP server — this file is the executable shim
of the import, nothing more.
"""

from __future__ import annotations

import sys

from .mcp.proxy import main

if __name__ == "__main__":
    sys.exit(main())
