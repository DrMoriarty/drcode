"""Prefetch the code-index grammars (docs/code-index-design.md §2.4).

The language pack downloads its tree-sitter grammars on first use into a
local cache directory. That is fine for a dev run (`uv run drcode`) but
forbidden in the frozen bundle: drcode ships without network at startup
and the parsers must be in the payload. The build script runs this
module BEFORE PyInstaller packs the bundle:

    .venv/bin/python -m drcode.code_index.prefetch --dest build/ts-pack-cache

The spec then collects the directory as `drcode/ts_parsers`, and
`drcode_entry.py` points `DRCODE_TS_PACK_CACHE` at it — `extract`
configures the pack with that cache directory at first use.

Exit code 0 even when individual downloads fail (best effort): a
missing grammar degrades at runtime to "file not indexed", never to a
broken build.
"""

from __future__ import annotations

import argparse
import sys

from .extract import EXTENSION_LANGUAGES

__all__ = ["PREFETCH_LANGUAGES", "prefetch"]

# The mainstream set worth shipping in the bundle (§2.3.2): everything
# the extension table resolves to for code files, minus the config/data
# formats that never carry symbols worth the payload. Deterministic —
# the same list on every platform.
_SKIP_FOR_PAYLOAD = {
    "json",
    "yaml",
    "toml",
    "html",
    "css",
    "scss",
    "markdown",
    "sql",
    "latex",
    "vim",
    "vue",
    "svelte",
    "graphql",
    "proto",
    "dockerfile",
    "bash",
}

PREFETCH_LANGUAGES: list[str] = sorted(
    set(EXTENSION_LANGUAGES.values()) - _SKIP_FOR_PAYLOAD
)


def prefetch(dest: str | None = None) -> list[str]:
    """Download the bundle's grammar set into `dest` (None = the user
    cache default). Returns the languages that FAILED (best effort —
    the caller decides whether that is fatal; the build treats it as a
    warning)."""
    import tree_sitter_language_pack as pack

    if dest is not None:
        pack.configure(pack.PackConfig(cache_dir=dest))
    failed: list[str] = []
    for language in PREFETCH_LANGUAGES:
        ok = False
        for _attempt in range(2):  # one retry: the CDN hiccups sometimes
            try:
                pack.prefetch([language])
                ok = True
                break
            except Exception:  # noqa: BLE001, S112 - a missing grammar is a degradation
                continue
        if not ok:
            failed.append(language)
    return failed


def main() -> None:  # pragma: no cover - build-time helper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        default=None,
        help="cache directory to fill (default: the user cache)",
    )
    args = parser.parse_args()
    failed = prefetch(args.dest)
    if failed:
        print(
            f"prefetch: {len(failed)} grammar(s) failed: {', '.join(failed)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
