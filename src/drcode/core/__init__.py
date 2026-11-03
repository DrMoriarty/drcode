"""UI-neutral core of drcode.

Everything in this package is independent of the Textual TUI: the
pipeline layers (over the host seam), the network-failure policies,
the push-to-talk dictation state machine and the rich markup builders.
Importing `textual` here is forbidden (enforced by tests/test_core_purity.py) —
a future GUI shell implements the same host protocols without Textual.
"""
