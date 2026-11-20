"""The language-agnostic symbol code index (docs/code-index-design.md).

Phase 1 (MVP): the tree-sitter extractor (`extract`), the per-project
JSON store (`store`, §3) and the two query helpers behind the
`code_symbols` / `code_outline` tools (`queries`). Phase 2 adds
`code_refs` / `code_callers`, the ctags adapter and overlay refinements.
"""
