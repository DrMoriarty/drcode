# PyInstaller entry stub: a plain script that imports the installed/source
# package and starts the TUI on the current working directory.
import os

# The logfire pydantic plugin (transitive dep of pydantic-ai) calls
# inspect.getsource() on pydantic's compiled PluggableSchemaValidator at
# import time, which fails in a frozen app (no source files). Disable it
# before pydantic is first imported. Disabled by entry-point name
# (`logfire-plugin`); `logfire` is matched too for older logfire versions.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "logfire-plugin,logfire")

from drcode import main

main()
