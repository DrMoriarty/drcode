"""UI-side staging of image attachments (screenshots) for the chat editor
and the new-task form.

Sits strictly on top of the FR #407e1959 contract: ``PendingInput.attachments``
(list of :class:`~drcode.models.ImageAttachment` references), byte storage via
``Storage.save_attachment`` (``.drcode/tasks/<id>/attachments/``) and the
multimodal consumption in the pipeline. The bytes live in memory only while a
message is being composed — on submit they are written to the task's
attachment directory through ``Storage.save_attachment`` and only the
references ride into ``manager.send`` / ``manager.create``.

The pipeline's per-prompt caps (``MAX_PROMPT_IMAGES`` / ``MAX_IMAGE_BYTES``
in ``core/pipeline_agent_run.py``) are mirrored here as staging limits, so an
oversized image is refused with a toast at attach time instead of being
silently skipped at consumption time.
"""

from __future__ import annotations

import base64
import contextlib
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape

from ..models import ImageAttachment
from ..storage import Storage

# Media types the attachment storage accepts (mirrors Storage's
# _ATTACHMENT_EXTENSIONS); anything else is refused at staging time.
SUPPORTED_MEDIA_TYPES: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# Mirrors the pipeline's consumption caps (MAX_PROMPT_IMAGES /
# MAX_IMAGE_BYTES in core/pipeline_agent_run.py): refusing at staging time
# keeps the "attach succeeded" contract honest — a staged image always
# reaches the model, never a "Skipped attached image(s)" event.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_STAGED_IMAGES = 6

_CLIPBOARD_TIMEOUT = 5.0


@dataclass(frozen=True)
class StagedImage:
    """An image staged for the next send: bytes in memory plus a display
    name and media type. The persisted ``ImageAttachment`` reference (with
    its on-disk uuid filename) is only created on submit."""

    filename: str  # display name — source file name or clipboard origin
    media_type: str
    data: bytes

    @property
    def size_label(self) -> str:
        return _format_size(len(self.data))


def _format_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


def load_staged_file(path: str) -> StagedImage:
    """Load an image file for staging; ``ValueError`` with a user-facing
    message on anything unusable (missing, wrong type, too big)."""
    clean = path.strip().strip("'\"")
    if not clean:
        raise ValueError("No path given")
    file = Path(clean).expanduser()
    if not file.is_file():
        raise ValueError(f"No such file: {clean}")
    media = SUPPORTED_EXTENSIONS.get(file.suffix.lower())
    if media is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported image type '{file.suffix or '?'}' — supported: {supported}"
        )
    data = file.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image is too large ({_format_size(len(data))}) — "
            f"the limit is {_format_size(MAX_IMAGE_BYTES)}"
        )
    return StagedImage(filename=file.name, media_type=media, data=data)


def load_staged_from_path_text(text: str) -> StagedImage | None:
    """A pasted text that is exactly a path to an existing image file
    becomes a staged image; everything else (prose, code, a path with no
    image extension) is normal paste. ``None`` = not an image path."""
    clean = text.strip().strip("'\"")
    if not clean or any(c.isspace() for c in clean):
        return None
    media = SUPPORTED_EXTENSIONS.get(Path(clean).suffix.lower())
    if media is None:
        return None
    file = Path(clean).expanduser()
    if not file.is_file():
        return None
    try:
        return load_staged_file(str(file))
    except ValueError:
        return None


def save_staged(
    storage: Storage, task_id: str, staged: Sequence[StagedImage]
) -> list[ImageAttachment]:
    """Persist staged bytes into the task's attachment directory and return
    the references for ``manager.send`` / ``manager.create``."""
    return [
        storage.save_attachment(task_id, image.data, image.media_type)
        for image in staged
    ]


def render_chip_line(images: Sequence[StagedImage]) -> str:
    """The attachment indicator line: one ``[image/<type>]`` chip per staged
    image (rich-escaped — ``[image/png]`` would parse as markup). Empty for
    no images."""
    chips = [
        f"\\[image/{image.media_type.split('/', 1)[-1]}] "
        f"{escape(image.filename)} ({image.size_label})"
        for image in images
    ]
    return " · ".join(chips)


# -- system clipboard readers --------------------------------------------------
#
# A terminal TUI never receives image bytes through bracketed paste (a
# Cmd+V paste event carries text only), so the primary screenshot scenario
# — the image already sitting in the clipboard — is served by an explicit
# reader: ctrl+v in the editor stages the clipboard image instead of a
# plain paste. Every reader degrades to None (no image), never raises.


def read_clipboard_image() -> StagedImage | None:
    """The clipboard's image content as a staged image, or None when the
    clipboard holds no readable image (or the platform tooling is absent)."""
    try:
        if _is_darwin():
            return _clipboard_image_darwin()
        if _is_windows():
            return _clipboard_image_windows()
        return _clipboard_image_linux()
    except Exception:  # noqa: BLE001 — any clipboard oddity degrades to None
        return None


def read_clipboard_text() -> str | None:
    """Plain text clipboard fallback for ctrl+v when no image is present
    (keeps the key's paste semantics); None when unavailable."""
    try:
        if _is_darwin():
            return _read_text(["pbpaste"])
        if _is_windows():
            return _read_text(["powershell", "-NoProfile", "-Command", "Get-Clipboard"])
        return _read_text(["xclip", "-selection", "clipboard", "-o"])
    except Exception:  # noqa: BLE001
        return None


def _is_darwin() -> bool:
    import sys

    return sys.platform == "darwin"


def _is_windows() -> bool:
    import sys

    return sys.platform == "win32"


def _run(args: list[str], *, binary: bool = False) -> bytes:
    result = subprocess.run(
        args,
        capture_output=True,
        timeout=_CLIPBOARD_TIMEOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout if binary else result.stdout.strip()


def _read_text(args: list[str]) -> str | None:
    out = _run(args).decode(errors="replace")
    return out or None


def _clipboard_image_darwin() -> StagedImage | None:
    # macOS screenshots put (on-demand convertible) PNG on the pasteboard;
    # ask for it as «class PNGf» — the reply is «data PNGf<hex>».
    out = _run(["osascript", "-e", "the clipboard as «class PNGf»"]).decode(
        errors="replace"
    )
    match = re.search(r"PNGf([0-9A-Fa-f]+)", out)
    if match:
        data = bytes.fromhex(match.group(1))
        if data:
            return StagedImage(
                filename="Clipboard.png", media_type="image/png", data=data
            )
    return _clipboard_image_darwin_tiff()


def _clipboard_image_darwin_tiff() -> StagedImage | None:
    # Fallback: raw TIFF on the pasteboard (older paste flows), converted
    # to PNG via macOS's built-in sips.
    out = _run(["osascript", "-e", "the clipboard as «class TIFF»"]).decode(
        errors="replace"
    )
    match = re.search(r"TIFF([0-9A-Fa-f]+)", out)
    if not match:
        return None
    tiff = bytes.fromhex(match.group(1))
    if not tiff:
        return None
    with tempfile.TemporaryDirectory(prefix="drcode-clip-") as tmp:
        src = Path(tmp) / "clip.tiff"
        dst = Path(tmp) / "clip.png"
        src.write_bytes(tiff)
        _run(["sips", "-s", "format", "png", str(src), "--out", str(dst)])
        data = dst.read_bytes()
    if not data:
        return None
    return StagedImage(filename="Clipboard.png", media_type="image/png", data=data)


def _clipboard_image_windows() -> StagedImage | None:
    # Get-Clipboard's image, encoded to PNG in-process and handed back as
    # base64 (no temp files, no extra tooling beyond the always-present
    # .NET assemblies).
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "Add-Type -AssemblyName System.Drawing; "
        "$i=[System.Windows.Forms.Clipboard]::GetImage(); "
        "if ($i) { "
        "$m=New-Object System.IO.MemoryStream; "
        "$i.Save($m,[System.Drawing.Imaging.ImageFormat]::Png); "
        "[Convert]::ToBase64String($m.ToArray()) }"
    )
    out = _run(["powershell", "-NoProfile", "-Command", script]).decode(
        errors="replace"
    )
    data = base64.b64decode(out) if out else b""
    if not data:
        return None
    return StagedImage(filename="Clipboard.png", media_type="image/png", data=data)


def _clipboard_image_linux() -> StagedImage | None:
    # X11 (xclip) first, then Wayland (wl-paste); png and jpeg are the
    # only types the attachment storage accepts anyway.
    for tool, media, ext in (
        ("xclip", "image/png", "png"),
        ("xclip", "image/jpeg", "jpg"),
        ("wl-paste", "image/png", "png"),
        ("wl-paste", "image/jpeg", "jpg"),
    ):
        args = (
            [tool, "-selection", "clipboard", "-t", media, "-o"]
            if tool == "xclip"
            else [tool, "--type", media]
        )
        data: bytes | None = None
        with contextlib.suppress(Exception):  # try the next reader
            data = _run(args, binary=True)
        if data:
            return StagedImage(filename=f"Clipboard.{ext}", media_type=media, data=data)
    return None
