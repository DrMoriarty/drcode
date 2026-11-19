#!/usr/bin/env bash
# Build a standalone `drcode` binary into dist/drcode/ (onedir bundle),
# then pack it into a single distributable archive dist/drcode-<version>-<os>-<arch>.tar.gz.
#
# Requires uv (https://docs.astral.sh/uv/); everything else (Python,
# dependencies, PyInstaller) is provisioned into the project venv
# automatically. The bundle is platform-specific: build it on each
# target OS/architecture you want to ship.
#
# Set DRCODE_ONEFILE=1 to build a legacy single-file executable instead
# (much slower to start: the whole payload is unpacked to a temp dir on
# every launch, and each freshly written binary is re-verified by the OS).
set -euo pipefail
cd "$(dirname "$0")/.."

version="$(tr -d '[:space:]' < version)"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  target=macos-arm64; ext="" ;;
  Darwin-x86_64) target=macos-x86_64; ext="" ;;
  Linux-x86_64)  target=linux-x86_64; ext="" ;;
  Linux-aarch64) target=linux-arm64; ext="" ;;
  MINGW*|MSYS*|CYGWIN*) target=windows-x86_64; ext=".exe" ;;
  *) echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

# sounddevice (PortAudio) is a core dependency, so a plain `uv sync` puts
# it into the build venv; its wheel carries the platform libportaudio
# binary inside _sounddevice_data, which drcode.spec collects into the app
# (numpy-free: the voice PCM path runs on stdlib array via sounddevice's
# raw streams).
uv sync --group build

# Code-index grammars (drcode.code_index, §2.4 of docs/code-index-design.md):
# the language pack downloads parsers on demand — impossible in the frozen,
# network-free bundle. Prefetch the mainstream set into build/ts-pack-cache
# (drcode.spec ships it as drcode/ts_parsers; drcode_entry.py points the
# pack at it). Best effort: a failed download degrades at runtime to
# "file not indexed" and never breaks the build.
.venv/bin/python -m drcode.code_index.prefetch --dest build/ts-pack-cache || \
  echo "warning: code-index grammar prefetch failed — the bundle ships without it" >&2

.venv/bin/pyinstaller --noconfirm --clean drcode.spec

mkdir -p dist
if [ -d dist/drcode ]; then
  # onedir bundle
  name="drcode-${version}-${target}"
  tar -C dist -czf "dist/${name}.tar.gz" drcode
  echo
  echo "Bundle ready: $(pwd)/dist/drcode/drcode${ext}"
  echo "Archive:      $(pwd)/dist/${name}.tar.gz"
  echo "Install:      tar -xzf ${name}.tar.gz && ./drcode/drcode${ext}"
else
  # legacy one-file build (DRCODE_ONEFILE=1)
  echo
  echo "Binary ready: $(pwd)/dist/drcode${ext}"
fi
