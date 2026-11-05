"""Packaging for DrCode Desktop.

Builds the shell with PyInstaller (desktop.spec) around the pre-built
drcode onedir payload (``scripts/build-binary.sh`` → ``dist/drcode/``)
and produces per-platform distributables:

  macos      DrCode.app (inside a .zip) — Info.plist + icon + ad-hoc
             codesign (real Developer ID signing is an external
             condition: set DRCODE_CODESIGN_IDENTITY, otherwise ad-hoc)
  appimage   DrCode-<version>-linux-<arch>.AppImage — bundles WebKitGTK
             (ldd-closure of seed libs) so the runtime WebKit version
             stops mattering
  deb        DrCode-<version>-linux-<arch>.deb — system package that
             *depends* on distro-provided WebKitGTK instead of bundling

Usage (from the repo root)::

    uv run --group desktop --group build python -m desktop.packaging --target macos
    uv run --group desktop --group build python -m desktop.packaging --target appimage
    uv run --group desktop --group build python -m desktop.packaging --target deb

The drcode payload is NOT rebuilt by default; pass ``--build-payload``
to run ``scripts/build-binary.sh`` first when it is missing.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DESKTOP = Path(__file__).resolve().parent
VERSION_FILE = REPO / "version"

APP_NAME = "DrCode"
BUNDLE_ID = "com.drmoriarty.drcode-desktop"

# Shared libraries the AppImage must bundle even though ldd cannot trace
# them from PyInstaller output (pywebview loads the GTK/WebKit stack
# dynamically through PyGObject). Everything each seed links against is
# copied recursively; glibc & friends are whitelisted out.
# Each entry is either a single required library name or a list of
# mutually-exclusive alternatives of the same API (WebKitGTK 4.0 vs 4.1 —
# a distro ships exactly one of the two; e.g. libwebkit2gtk-4.0-dev was
# removed from Ubuntu starting 23.04). At least ONE member of every
# entry must be present; on a 4.0-only system the ldd closure pulls
# libsoup-2.4 in automatically.
LIB_SEED_GROUPS: list[str | list[str]] = [
    ["libwebkit2gtk-4.1.so.0", "libwebkit2gtk-4.0.so.0"],
    "libgtk-3.so.0",
    ["libjavascriptcoregtk-4.1.so.0", "libjavascriptcoregtk-4.0.so.0"],
    "libsoup-3.0.so.0",
    "libgirepository-1.0.so.1",
]

# System libs that must stay external (they exist everywhere and bundling
# them is the classic AppImage breaker).
GLIBC_WHITELIST_PREFIXES = (
    "ld-linux",
    "libc.so",
    "libm.so",
    "libdl",
    "libpthread",
    "librt",
    "libnsl",
    "libresolv",
    "libutil",
    "libcrypt",
    "libgcc_s",
    "libstdc++",
    "libnss_",
    "libnsssystemd",
    "libz.so",
    "libzstd",
    "liblzma",
    "libbz2",
    "liblz4",
    "libcap",
    "libselinux",
    "libmount",
    "libblkid",
    "libuuid",
    "libsystemd",
    "libgcrypt",
    "libgpg-error",
    "libexpat",
    "libGL",
    "libEGL",
    "libglapi",
    "libdrm",
    "libgbm",
    "libxcb",
    "libX11",
)


def repo_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def run(cmd: list[str | Path], **kwargs: object) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kwargs)  # type: ignore[arg-type]


def payload_executable(payload_dir: Path) -> Path:
    exe = "drcode.exe" if sys.platform == "win32" else "drcode"
    return payload_dir / exe


def ensure_payload(payload_dir: Path | None, build_if_missing: bool) -> Path:
    candidate = payload_dir or (REPO / "dist" / "drcode")
    if not payload_executable(candidate).is_file():
        if not build_if_missing:
            raise SystemExit(
                f"drcode payload not found at {candidate} — run "
                "`bash scripts/build-binary.sh` first, or pass --build-payload"
            )
        run(["bash", "scripts/build-binary.sh"], cwd=REPO)
    return candidate


# ---------------------------------------------------------------------------
# icon (generated, pure build-time artifact)
# ---------------------------------------------------------------------------


def generate_icon(path: Path, size: int = 512) -> Path:
    """Render the placeholder app icon with Pillow (terminal glyph on tile)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required for icon generation (desktop group)"
        ) from exc

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = size // 5
    # rounded dark tile with an accent prompt mark
    draw.rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=radius, fill=(22, 24, 29, 255)
    )
    draw.rounded_rectangle(
        [0, 0, size - 1, size - 1],
        radius=radius,
        outline=(79, 156, 249, 255),
        width=size // 40,
    )
    # ">_" glyph drawn as thick lines (no font dependency)
    accent = (79, 156, 249, 255)
    w = size // 14  # stroke width
    pad = size // 4
    # ">" as two strokes
    x0, y0 = pad, size // 3
    x1, y1 = pad + size // 5, size // 2
    draw.line([(x0, y0), (x1, y1)], fill=accent, width=w)
    draw.line([(x1, y1), (x0, y0 + size // 6 + size // 6)], fill=accent, width=w)
    # "_" as a horizontal stroke
    ux0, ux1 = pad + size // 3, size - pad
    uy = size - pad - size // 10
    draw.line([(ux0, uy), (ux1, uy)], fill=accent, width=w)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def ensure_icns(png_path: Path, icns_path: Path) -> Path:
    """macOS icns via iconutil (ships with macOS)."""
    if icns_path.exists():
        return icns_path
    iconset = icns_path.parent / "DrCode.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        img = Image.open(png_path)
        for size in (16, 32, 128, 256, 512):
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            resized.save(iconset / f"icon_{size}x{size}.png")
            if size <= 256:
                img.resize((size * 2, size * 2), Image.Resampling.LANCZOS).save(
                    iconset / f"icon_{size}x{size}@2x.png"
                )
        run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns_path)])
    finally:
        shutil.rmtree(iconset, ignore_errors=True)
    return icns_path


# ---------------------------------------------------------------------------
# shell build (PyInstaller, desktop.spec)
# ---------------------------------------------------------------------------


def build_shell(payload_dir: Path, dist_dir: Path) -> Path:
    """Run PyInstaller for the shell; returns the build output path.

    The shell gets its OWN dist/work directories (`dist-desktop/`,
    `build-desktop/`): the repo's `dist/` and `build/` belong to the
    drcode payload build, and on macOS's case-insensitive filesystem a
    shell output named `DrCode` would collide with the payload's
    `drcode` dir, corrupting the payload.
    """
    env = dict(os.environ)
    env["DRCODE_PAYLOAD"] = str(payload_dir)
    env["DRCODE_DESKTOP_VERSION"] = repo_version()

    png = DESKTOP / "build" / "icon.png"
    generate_icon(png)
    if sys.platform == "darwin":
        icns = DESKTOP / "build" / "icon.icns"
        ensure_icns(png, icns)
        env["DRCODE_DESKTOP_ICNS"] = str(icns)

    run(
        [
            ".venv/bin/pyinstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            REPO / "dist-desktop",
            "--workpath",
            REPO / "build-desktop",
            str(DESKTOP / "desktop.spec"),
        ],
        cwd=REPO,
        env=env,
    )
    out = dist_dir / (f"{APP_NAME}.app" if sys.platform == "darwin" else APP_NAME)
    if not out.exists():
        raise SystemExit(f"expected build output missing: {out}")
    return out


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------


def codesign_app(app_path: Path) -> None:
    identity = os.environ.get("DRCODE_CODESIGN_IDENTITY", "")
    if identity:
        run(
            [
                "codesign",
                "--deep",
                "--force",
                "--options",
                "runtime",
                "-s",
                identity,
                app_path,
            ]
        )
        print(f"Signed with identity: {identity}")
    else:
        run(["codesign", "--deep", "--force", "-s", "-", app_path])
        print(
            "Ad-hoc signed (no DRCODE_CODESIGN_IDENTITY set). "
            "For distribution: Developer ID Application identity + notarization."
        )


def zip_app(app_path: Path, out: Path) -> Path:
    """ditto preserves the mach-o signatures and symlinks a plain zip mangles."""
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["ditto", "-c", "-k", "--keepParent", str(app_path), str(out)])
    return out


# ---------------------------------------------------------------------------
# Linux: shared helpers
# ---------------------------------------------------------------------------


def linux_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    raise SystemExit(f"Unsupported Linux architecture: {machine}")


def _is_whitelisted(soname: str) -> bool:
    return any(soname.startswith(prefix) for prefix in GLIBC_WHITELIST_PREFIXES)


def ldd_closure(path: Path) -> list[Path]:
    """All non-whitelisted shared libs ``path`` (a lib or an ELF) links to."""
    out: dict[str, Path] = {}
    try:
        proc = subprocess.run(
            ["ldd", str(path)], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return []
    for line in proc.stdout.splitlines():
        # "<name> => <path> (0x...)" or "<path> (0x...)" (static ld.so line)
        if " => " in line:
            soname, _, target = line.partition(" => ")
            target = target.split(" (")[0].strip()
        else:
            continue
        if not target or not target.startswith("/"):
            continue
        soname = soname.strip()
        if _is_whitelisted(soname):
            continue
        candidate = Path(target)
        if candidate.is_file() and soname not in out:
            out[soname] = candidate
    return list(out.values())


def copy_lib_closure(seeds: list[Path], lib_dir: Path) -> list[Path]:
    """Copy seed libs + their ldd closure into ``lib_dir`` (symlinks kept)."""
    lib_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    queue = list(seeds)
    seen: set[str] = set()
    while queue:
        src = queue.pop(0)
        resolved = src.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        # Preserve the original symlink name (libwebkit2gtk-4.1.so.0 etc.)
        dest = lib_dir / src.name
        if not dest.exists():
            shutil.copy2(resolved, dest, follow_symlinks=True)
            copied.append(dest)
        for dep in ldd_closure(resolved):
            if str(dep.resolve()) not in seen:
                queue.append(dep)
    return copied


def copy_typelibs(lib_dir: Path) -> Path:
    """Bundle ALL gobject-introspection typelibs (cheap, removes guesswork)."""
    candidates = [
        Path("/usr/lib/x86_64-linux-gnu/girepository-1.0"),
        Path("/usr/lib/aarch64-linux-gnu/girepository-1.0"),
        Path("/usr/lib/girepository-1.0"),
        Path("/usr/lib64/girepository-1.0"),
    ]
    for source in candidates:
        if source.is_dir():
            dest = lib_dir / "girepository-1.0"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)
            return dest
    raise SystemExit(
        "girepository-1.0 typelibs not found — install libgirepository1.0-dev"
    )


SEED_SEARCH_DIRS = [
    Path("/usr/lib/x86_64-linux-gnu"),
    Path("/usr/lib/aarch64-linux-gnu"),
    Path("/usr/lib"),
    Path("/usr/lib64"),
]


def find_seed_libs() -> list[Path]:
    found: list[Path] = []
    missing: list[str] = []
    for group in LIB_SEED_GROUPS:
        names = [group] if isinstance(group, str) else group
        group_found = False
        for name in names:
            for d in SEED_SEARCH_DIRS:
                candidate = d / name
                if candidate.exists():
                    found.append(candidate)
                    group_found = True
                    break
            if group_found:
                break
        if not group_found:
            missing.append("/".join(names))
    if missing:
        raise SystemExit(
            "WebKitGTK seed libs not found: "
            + ", ".join(missing)
            + " — install libwebkit2gtk-4.1-dev libgtk-3-dev"
        )
    return found


def write_desktop_file(path: Path, exec_cmd: str, icon_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "[Desktop Entry]",
                "Type=Application",
                f"Name={APP_NAME}",
                "Comment=Native shell around the drcode coding agent",
                f"Exec={exec_cmd}",
                f"Icon={icon_name}",
                "Terminal=false",
                "Categories=Development;IDE;Utility;",
                "StartupWMClass=drcode-desktop",
                "NoDisplay=false",
                "",
            ]
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Linux: AppImage
# ---------------------------------------------------------------------------


def build_appimage(shell_dir: Path, dist_out: Path) -> Path:
    arch = linux_arch()
    app_dir = DESKTOP / "build" / f"{APP_NAME}.AppDir"
    if app_dir.exists():
        shutil.rmtree(app_dir)
    (app_dir / "usr" / "bin").mkdir(parents=True)
    (app_dir / "usr" / "lib").mkdir(parents=True)

    # The PyInstaller onedir output becomes the app payload.
    shutil.copytree(shell_dir, app_dir / "usr" / "bin" / APP_NAME, symlinks=True)

    libs_dir = app_dir / "usr" / "lib"
    seeds = find_seed_libs()
    copied = copy_lib_closure(seeds, libs_dir)
    copy_typelibs(libs_dir)
    print(f"AppImage: bundled {len(copied)} shared libraries + typelibs")

    generate_icon(app_dir / f"{APP_NAME}.png", 512)
    write_desktop_file(
        app_dir / f"{APP_NAME}.desktop",
        exec_cmd=f"{APP_NAME}.AppDir/AppRun",
        icon_name=APP_NAME,
    )

    apprun = app_dir / "AppRun"
    apprun.write_text(
        "\n".join(  # noqa: FLY002 - readability over micro-optimization
            [
                "#!/bin/sh",
                'HERE="$(dirname "$(readlink -f "$0")")"',
                'export LD_LIBRARY_PATH="$HERE/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
                'export GI_TYPELIB_PATH="$HERE/usr/lib/girepository-1.0${GI_TYPELIB_PATH:+:$GI_TYPELIB_PATH}"',
                'export GDK_PIXBUF_MODULEDIR="$HERE/usr/lib/gdk-pixbuf-2.0/2.10.0/loaders"',
                'exec "$HERE/usr/bin/DrCode/DrCode" "$@"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    apprun.chmod(0o755)

    dist_out.mkdir(parents=True, exist_ok=True)
    tool = _appimage_tool(arch)
    out_file = dist_out / f"{APP_NAME}-{repo_version()}-linux-{arch}.AppImage"
    run(
        [tool, "--appimage-extract-and-run", str(app_dir), str(out_file)],
        cwd=REPO,
    )
    return out_file


def _appimage_tool(arch: str) -> Path:
    """Download (and cache) appimagetool for the current arch."""
    tools_dir = DESKTOP / "build" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    tool = tools_dir / f"appimagetool-{arch}.AppImage"
    if not tool.exists():
        url = (
            "https://github.com/AppImage/appimagetool/releases/download/"
            f"continuous/appimagetool-{arch}.AppImage"
        )
        run(["curl", "-sfL", "--retry", "3", "-o", str(tool), url])
        tool.chmod(0o755)
    return tool


# ---------------------------------------------------------------------------
# Linux: .deb
# ---------------------------------------------------------------------------

DEB_DEPENDS = "libgtk-3-0, libwebkit2gtk-4.1-0, libgirepository-1.0-1"


def build_deb(shell_dir: Path, dist_out: Path) -> Path:
    arch_map = {"x86_64": "amd64", "aarch64": "arm64"}
    arch, deb_arch = linux_arch(), arch_map[linux_arch()]
    version = repo_version()
    root = DESKTOP / "build" / "deb-root"
    if root.exists():
        shutil.rmtree(root)

    opt = root / "opt" / "drcode-desktop"
    shutil.copytree(shell_dir, opt / APP_NAME, symlinks=True)

    bin_link = root / "usr" / "bin"
    bin_link.mkdir(parents=True, exist_ok=True)
    (bin_link / "drcode-desktop").symlink_to(
        f"/opt/drcode-desktop/{APP_NAME}/{APP_NAME}"
    )

    generate_icon(
        root
        / "usr"
        / "share"
        / "icons"
        / "hicolor"
        / "512x512"
        / "apps"
        / "drcode.png",
        512,
    )
    write_desktop_file(
        root / "usr" / "share" / "applications" / "drcode.desktop",
        exec_cmd="/usr/bin/drcode-desktop",
        icon_name="drcode",
    )

    control = root / "DEBIAN" / "control"
    control.parent.mkdir(parents=True, exist_ok=True)
    control.write_text(
        "\n".join(
            [
                "Package: drcode-desktop",
                f"Version: {version}",
                f"Architecture: {deb_arch}",
                "Maintainer: DrMoriarty <drmoriarty.0@gmail.com>",
                f"Depends: {DEB_DEPENDS}",
                f"Installed-Size: {int(_tree_size_kb(opt / APP_NAME))}",
                "Section: devel",
                "Priority: optional",
                f"Description: {APP_NAME} — native shell around the drcode coding agent",
                " Runs the existing drcode TUI bundle in an embedded terminal",
                " (pywebview + xterm.js + PTY bridge).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    dist_out.mkdir(parents=True, exist_ok=True)
    out_file = dist_out / f"{APP_NAME}-{version}-linux-{arch}.deb"
    run(["dpkg-deb", "--build", "--root-owner-group", str(root), str(out_file)])
    return out_file


def _tree_size_kb(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total // 1024


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------


def macos_arch() -> str:
    machine = platform.machine().lower()
    return "macos-arm64" if machine == "arm64" else "macos-x86_64"


def build_macos(payload_dir: Path, dist_out: Path) -> Path:
    app_path = build_shell(payload_dir, REPO / "dist-desktop")
    codesign_app(app_path)
    return zip_app(
        app_path, dist_out / f"{APP_NAME}-{repo_version()}-{macos_arch()}.zip"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build DrCode Desktop distributables")
    parser.add_argument("--target", choices=["macos", "appimage", "deb"], required=True)
    parser.add_argument(
        "--payload", help="drcode onedir bundle dir (default dist/drcode)"
    )
    parser.add_argument(
        "--build-payload",
        action="store_true",
        help="run scripts/build-binary.sh if the payload is missing",
    )
    parser.add_argument(
        "--out", default=str(REPO / "dist"), help="output dir for distributables"
    )
    args = parser.parse_args(argv)

    payload_dir = ensure_payload(
        Path(args.payload) if args.payload else None, args.build_payload
    )
    dist_out = Path(args.out)
    if args.target == "macos":
        if sys.platform != "darwin":
            raise SystemExit("target macos must run on macOS")
        out = build_macos(payload_dir, dist_out)
    elif args.target == "appimage":
        if not sys.platform.startswith("linux"):
            raise SystemExit("target appimage must run on Linux")
        shell = build_shell(payload_dir, REPO / "dist-desktop")
        out = build_appimage(shell, dist_out)
    else:
        if not sys.platform.startswith("linux"):
            raise SystemExit("target deb must run on Linux")
        shell = build_shell(payload_dir, REPO / "dist-desktop")
        out = build_deb(shell, dist_out)
    print(f"\nReady: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
