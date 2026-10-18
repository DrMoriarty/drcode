#!/usr/bin/env bash
# Build a standalone one-file `drcode` binary into dist/drcode.
#
# Requires uv (https://docs.astral.sh/uv/); everything else (Python,
# dependencies, PyInstaller) is provisioned into the project venv
# automatically. The binary is platform-specific: build it on each
# target OS/architecture you want to ship.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --group build
.venv/bin/pyinstaller --noconfirm --clean drcode.spec

echo
echo "Binary ready: $(pwd)/dist/drcode"
