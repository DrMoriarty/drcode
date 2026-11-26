"""USD costing of model calls, priced from the user-level catalog.

The per-token prices live on `ModelEntry` (`price_prompt` /
`price_completion` / `price_cache_read`, USD per ONE token — see
`providers.py`; auto-discovered from the gateway's pricing block by
`agent_model.fetch_models` or entered by hand in models.json).

THE FROZEN-PRICE CONTRACT: the cost of a request is computed ONCE, at
the moment the response arrives, from the catalog prices AS THEY ARE
THEN — and never recomputed. Historical costs do not follow later price
edits (otherwise "yesterday model A, today model B" comparisons lose
their meaning). The frozen number is what lands in the usage store and
in the task's cumulative `cost_usd`.

A model without prices (custom / env models, locally served free
endpoints) has NO cost, deliberately not zero: its sessions are priced
``None`` and contribute nothing to the money aggregates, while the
token accounting is untouched (mixed setups — a free compaction model
beside paid coding/research — keep working end to end).

The formula mirrors the long-standing `_row_cost` of the statistics
dialog: ``input`` already INCLUDES cache reads and writes (pydantic-ai
normalizes usage that way), so cache-read tokens are priced separately
and subtracted from the prompt-priced remainder; cache writes stay
inside the prompt term (no separate per-token price exists for them,
nor for reasoning tokens — the provider reports none).

Deliberately pydantic-ai-free and Textual-free: the module sits on the
per-request recording hot path AND is imported by the statistics
dialog, and the core pipeline layers must be able to reach the pricing
rules without dragging a UI stack in.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .providers import UserModelsStore, user_models_file

# Per-model prices of one usage key: (prompt, completion, cache_read) per
# token in USD; cache_read may be None (unknown) while the other two are
# always present — the index only holds entries whose cost CAN be computed
# (with the caveat that a nonzero cache_read still needs price_cache_read).
Prices = tuple[float, float, float | None]

# Cache of the last-built price index, keyed by (resolved catalog path,
# mtime_ns, size) so an edited models.json is picked up on the next
# request while steady-state requests cost no file read at all. The
# lookup runs on the event loop (inside `_record_session_usage`), where
# the persistent usage store already pays a full-file read per record —
# one small stat()-guarded read on catalog edits is noise next to that.
# The path is part of the key: the conftest sandbox and the per-task
# run_bash redirect repoint `config._USER_CONFIG_DIR`, and two different
# sandboxes can host identical (mtime, size) pairs.
_CATALOG_STAT: tuple[str, int, int] | None = None
_INDEX_CACHE: dict[str, Prices] | None = None


def _catalog_path() -> Path:
    return user_models_file()


def price_index() -> dict[str, Prices]:
    """Model reference → per-token prices, from the user-level catalog.

    The usage keys are "<role>/<model>" where <model> is the catalog's
    "<provider>/<model-id>" reference (or the bare model id for custom /
    env models), so the lookup mirrors `providers._match_ref`: the full
    "<provider>/<model-id>" form first, then the bare model id across
    the catalog. Only entries with BOTH the prompt and the completion
    price are indexed — without either the cost is never computable;
    the cache-read price stays optional (a request with zero cache
    reads doesn't need it).

    The result is cached against the catalog file's (mtime_ns, size):
    a repeated call between edits is a pure stat(). A missing catalog
    yields an empty index — everything is then unpriced, which is the
    honest reading of "no catalog, no prices".
    """
    global _CATALOG_STAT, _INDEX_CACHE
    path = _catalog_path()
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        # No catalog file (yet): cache the emptiness under a path-only
        # key so a later first save (new mtime) invalidates it.
        key = (str(path), -1, -1)
    if _INDEX_CACHE is not None and _CATALOG_STAT == key:
        return _INDEX_CACHE
    catalog = UserModelsStore(path=path).load()
    index: dict[str, Prices] = {}
    for provider in catalog.providers:
        for entry in provider.models:
            if entry.price_prompt is None or entry.price_completion is None:
                continue
            prices: Prices = (
                entry.price_prompt,
                entry.price_completion,
                entry.price_cache_read,
            )
            index.setdefault(f"{provider.name}/{entry.id}", prices)
    for provider in catalog.providers:
        for entry in provider.models:
            if entry.price_prompt is None or entry.price_completion is None:
                continue
            index.setdefault(
                entry.id,
                (entry.price_prompt, entry.price_completion, entry.price_cache_read),
            )
    _CATALOG_STAT = key
    _INDEX_CACHE = index
    return index


def prices_for(model_key: str) -> Prices | None:
    """The prices of one resolved model key ("<provider>/<model-id>" or
    the bare id), or None when the model has no computable price —
    the "session without a cost" marker, not an error."""
    return price_index().get(model_key)


def row_cost(counters: Mapping[str, int], prices: Prices) -> float | None:
    """The USD cost of one counter row (a per-request usage or a
    day/model aggregate — the shape is the same field names).

    `input` already INCLUDES cache reads/writes (pydantic-ai normalizes
    usage that way — see `transcript._usage_context_tokens`), so the
    cache-read tokens are priced separately and subtracted from the
    prompt-priced remainder; cache writes stay inside the prompt term
    (no separate per-token price exists for them). None = not
    computable: a needed price is missing (the cache-read price only
    once the row actually has cache reads)."""
    price_prompt, price_completion, price_cache_read = prices
    cache_read = counters.get("cache_read", 0) or 0
    if cache_read and price_cache_read is None:
        return None
    plain_input = max(0, (counters.get("input", 0) or 0) - cache_read)
    return (
        plain_input * price_prompt
        + cache_read * (price_cache_read or 0.0)
        + (counters.get("output", 0) or 0) * price_completion
    )


def fmt_cost(cost: float) -> str:
    """Human-readable USD amount: costs are usually cents or less, so
    the precision scales down with the value; trailing zeros are trimmed
    ("$0.023", "$0.000735")."""
    if cost >= 1:
        text = f"{cost:,.2f}"
    elif cost >= 0.01:
        text = f"{cost:.4f}"
    else:
        text = f"{cost:.6f}"
    text = text.rstrip("0").rstrip(".")
    return f"${text or '0'}"


def reset_price_cache() -> None:
    """Drop the module-level price cache. The cache is stat-guarded, so
    production never needs this; it exists for tests that swap catalogs
    in ways a (mtime_ns, size) key cannot distinguish."""
    global _CATALOG_STAT, _INDEX_CACHE
    _CATALOG_STAT = None
    _INDEX_CACHE = None
