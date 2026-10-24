#!/usr/bin/env bash
# Stage a self-contained libportaudio.so.2 for the frozen Linux bundle into
# build/portaudio/ (picked up by drcode.spec). No-op on non-Linux hosts and
# when the library is already staged.
#
# Why a custom build: python-sounddevice ships the PortAudio binary inside
# its wheels for macOS and Windows only — there is no manylinux wheel, so on
# Linux `pip install sounddevice` yields a pure-Python package (no
# _sounddevice_data) and a frozen drcode would depend on the user's system
# libportaudio2. The distro's libportaudio2 in turn hard-links JACK
# (libjack.so.0), which clean machines don't have, so we build PortAudio
# ourselves with ALSA only — the sole remaining runtime dependency is
# libasound.so.2, which every Linux with audio has.
#
# Requires: gcc/make and the ALSA headers (Debian/Ubuntu: libasound2-dev,
# Fedora: alsa-lib-devel). Network access to github.com.
set -euo pipefail
cd "$(dirname "$0")/.."

[[ "$(uname -s)" == "Linux" ]] || exit 0

OUT="build/portaudio/libportaudio.so.2"
if [[ -f "$OUT" ]]; then
  echo "PortAudio already staged: $OUT"
  exit 0
fi

if ! printf '#include <alsa/asoundlib.h>\n' | cc -E -x c - -o /dev/null 2>/dev/null; then
  echo "error: ALSA headers not found — install libasound2-dev (Debian/Ubuntu)" \
       "or alsa-lib-devel (Fedora); the bundled PortAudio cannot be built." >&2
  exit 1
fi

VERSION="${PORTAUDIO_VERSION:-19.7.0}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -fsSL "https://github.com/PortAudio/portaudio/archive/refs/tags/v${VERSION}.tar.gz" \
  | tar -xz -C "$tmp"

(cd "$tmp/portaudio-${VERSION}" &&
  ./configure --with-alsa --without-jack --disable-static --silent &&
  # build the shared library only (the default target also compiles
  # examples/tests)
  make --silent -j"$(nproc)" lib/libportaudio.la
)

lib="$tmp/portaudio-${VERSION}/lib/.libs/libportaudio.so.2"
[[ -f "$lib" ]] || { echo "error: PortAudio build produced no libportaudio.so.2" >&2; exit 1; }
# A backend-less build is useless — fail loudly instead of shipping a
# library that cannot open any device.
if ! strings "$lib" | grep -q 'libasound\.so'; then
  echo "error: built libportaudio has no ALSA backend (libasound.so.2 not linked)" >&2
  exit 1
fi

mkdir -p build/portaudio
cp "$lib" "$OUT"
echo "Staged $OUT ($(uname -m), PortAudio v${VERSION}, ALSA-only)"
