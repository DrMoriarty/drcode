"""System font enumeration for the DrCode Desktop shell.

The webview cannot list the fonts it can render, so the Python side asks
the OS once per process and exposes the family-name set through the
bridge (``DesktopApi.get_available_fonts``). The terminal's font dialog
uses it to hide curated families that do not exist on this machine.

Design rules (mirroring the shell's general posture):

- **Best-effort, never fatal.** Any failure returns ``None`` — the JS
  side treats "unknown" as "no filtering" (fail-open: the full curated
  list is shown, exactly like before this module existed).
- **Families, not files.** The webview resolves ``font-family`` names,
  so we collect family/full-name strings, whatever the platform API
  calls them; loose matching happens in JS.
- **Cached.** Font sets do not change mid-session (installing a font
  would require a webview restart anyway), so the result is computed
  once and reused.
"""

from __future__ import annotations

import re
import subprocess
import sys

__all__ = ["available_font_families", "refresh_cache"]

# Two-part cache: the bool distinguishes "not computed yet" from
# "computed, unknown" (a failed probe must stay cached as None, not
# re-run on every dialog open).
_cache_loaded = False
_cache_value: frozenset[str] | None = None


def refresh_cache() -> None:
    """Drop the memoized enumeration (tests / explicit reloads only)."""
    global _cache_loaded
    _cache_loaded = False


def available_font_families() -> frozenset[str] | None:
    """Family names the OS can render, or ``None`` if undetectable.

    ``None`` is a deliberate signal (unknown ≠ empty): the JS side must
    not filter anything when the platform probe fails.
    """
    global _cache_loaded, _cache_value
    if _cache_loaded:
        return _cache_value
    result: frozenset[str] | None = None
    try:
        if sys.platform == "darwin":
            names = _macos_family_names()
        elif sys.platform == "win32":
            names = _windows_family_names()
        else:
            names = _linux_family_names()
        if names:
            result = frozenset(n.strip() for n in names if n and n.strip())
    except Exception:  # noqa: BLE001 - enumeration is strictly best-effort
        result = None
    _cache_loaded = True
    _cache_value = result
    return result


# -- macOS ------------------------------------------------------------------
# CoreText is the exact answer (its family names are what the WKWebView
# resolves font-family against), but pywebview's pyobjc only ships
# pyobjc-framework-Cocoa — CoreText bindings are often absent. AppKit
# (guaranteed present) is the fallback: NSFontManager returns POSTSCRIPT
# names ("Menlo-Regular", "CourierNewPSMT"), so each is mapped back to
# its real family via NSFont.familyName before use.


def _macos_family_names() -> list[str] | None:
    try:
        from CoreText import (  # type: ignore[import-not-found]
            CTFontCollectionCopyAvailableFontFamilyNames,
        )

        return [str(n) for n in CTFontCollectionCopyAvailableFontFamilyNames()]
    except Exception:  # noqa: BLE001, S110 - bindings not installed, try AppKit
        pass
    try:
        from AppKit import NSFont, NSFontManager  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    try:
        ps_names = [str(n) for n in NSFontManager.sharedFontManager().availableFonts()]
    except Exception:  # noqa: BLE001
        return None
    families: list[str] = []
    for ps_name in ps_names:
        try:
            font = NSFont.fontWithName_size_(ps_name, 12.0)
            family = (
                str(font.familyName()) if font is not None and font.familyName() else ""
            )
        except Exception:  # noqa: BLE001 - a single dead font name must not kill the probe
            family = ""
        families.append(family or ps_name)
    return families


# -- Linux ------------------------------------------------------------------
# fontconfig is the authority on installed fonts (and what the GTK
# WebKit backend resolves against); headless machines without it get
# the fail-open treatment.


def _linux_family_names() -> list[str] | None:
    try:
        proc = subprocess.run(
            ["fc-list", "--format", "%{family}\n"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out: list[str] = []
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        # One font can expose several family aliases, comma-separated.
        out.extend(part.strip() for part in line.split(","))
    return out


# -- Windows ----------------------------------------------------------------
# GDI enumeration covers everything a classic app can render; the
# registry (machine + per-user installs) is the fallback when GDI is
# unavailable in the current session. Both give *name-ish* strings
# ("Consolas", "Consolas Bold (TrueType)") — the JS matcher is prefix-
# tolerant exactly because of that.


def _windows_family_names() -> list[str] | None:
    names = _windows_gdi_family_names()
    if not names:
        names = _windows_registry_family_names()
    return names or None


def _windows_gdi_family_names() -> list[str]:
    import ctypes
    from ctypes import wintypes

    if sys.platform != "win32":  # pragma: no cover - platform guard for pyright
        return []

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    hdc = user32.GetDC(0)
    if not hdc:
        return []

    LF_FACESIZE = 32

    class LOGFONTW(ctypes.Structure):
        _fields_ = [
            ("lfHeight", wintypes.LONG),
            ("lfWidth", wintypes.LONG),
            ("lfEscapement", wintypes.LONG),
            ("lfOrientation", wintypes.LONG),
            ("lfWeight", wintypes.LONG),
            ("lfItalic", wintypes.BYTE),
            ("lfUnderline", wintypes.BYTE),
            ("lfStrikeOut", wintypes.BYTE),
            ("lfCharSet", wintypes.BYTE),
            ("lfOutPrecision", wintypes.BYTE),
            ("lfClipPrecision", wintypes.BYTE),
            ("lfQuality", wintypes.BYTE),
            ("lfPitchAndFamily", wintypes.BYTE),
            ("lfFaceName", wintypes.WCHAR * LF_FACESIZE),
        ]

    class ENUMLOGFONTEXW(ctypes.Structure):
        _fields_ = [
            ("elfLogFont", LOGFONTW),
            ("elfFullName", wintypes.WCHAR * LF_FACESIZE),
            ("elfStyle", wintypes.WCHAR * LF_FACESIZE),
            ("elfScript", wintypes.WCHAR * LF_FACESIZE),
        ]

    FONTENUMPROC = ctypes.WINFUNCTYPE(
        wintypes.INT,
        ctypes.c_void_p,  # const ENUMLOGFONTEXW* (untyped: varlayout varies)
        ctypes.c_void_p,  # const TEXTMETRICW*
        wintypes.DWORD,
        wintypes.LPARAM,
    )

    names: list[str] = []
    ex_ptr = ctypes.POINTER(ENUMLOGFONTEXW)

    def on_font(
        lpelfe: ctypes.c_void_p, _lpntme: ctypes.c_void_p, _ftype: int, _lparam: int
    ) -> int:
        try:
            entry = ctypes.cast(lpelfe, ex_ptr).contents
            for value in (entry.elfLogFont.lfFaceName, entry.elfFullName):
                name = str(value).strip()
                if name:
                    names.append(name)
        except Exception:  # noqa: BLE001, S110 - skip malformed entries, keep enumerating
            pass
        return 1  # continue enumeration

    enumproc = FONTENUMPROC(on_font)  # keep alive for the duration of the call
    logfont = LOGFONTW()
    logfont.lfCharSet = 1  # DEFAULT_CHARSET: all charsets, all fonts
    logfont.lfFaceName = ""
    try:
        gdi32.EnumFontFamiliesExW(hdc, ctypes.byref(logfont), enumproc, 0, 0)
    finally:
        user32.ReleaseDC(0, hdc)
    return names


_FONT_FILE_SUFFIX_RE = re.compile(r"\s*\((?:TrueType|OpenType|Bitmap|Type1)[^)]*\)\s*$")


def _windows_registry_family_names() -> list[str]:
    # Dynamic import: the winreg stub is Windows-only and pyright on
    # macOS/Linux would flag every attribute access on a static import.
    import importlib

    winreg = importlib.import_module("winreg")

    names: list[str] = []
    for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            key = winreg.OpenKey(
                root, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"
            )
        except OSError:
            continue
        with key:
            index = 0
            while True:
                try:
                    value_name, _data, _type = winreg.EnumValue(key, index)
                except OSError:
                    break
                index += 1
                # "Consolas Bold (TrueType)" → "Consolas Bold"
                clean = _FONT_FILE_SUFFIX_RE.sub("", value_name).strip()
                if clean:
                    names.append(clean)
    return names
