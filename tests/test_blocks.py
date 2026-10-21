"""Tests for TUI block rendering helpers."""

from __future__ import annotations

from textual.app import App, ComposeResult

from drcode.tui.blocks import ToolCallBlock, _json_text


def test_json_text_keeps_full_body() -> None:
    """The trailing newline from the Syntax render is cropped by one
    character, not by a cell-length computation that would cut off
    almost the whole multi-line body."""
    text = _json_text('{"command": "ls"}')
    assert text.plain == '{\n  "command": "ls"\n}'


def test_json_text_invalid_json_falls_back_to_plain() -> None:
    text = _json_text("{truncated")
    assert text.plain == "{truncated"


async def test_tool_call_block_body_is_full_json() -> None:
    """Expanding a tool-call block shows the whole re-indented JSON body."""

    class _Host(App[None]):
        def compose(self) -> ComposeResult:
            yield ToolCallBlock('run_bash({"command": "ls"})')

    async with _Host().run_test() as pilot:
        block = pilot.app.query_one(ToolCallBlock)
        block.remove_class("collapsed")
        block._redraw()
        assert block._body is not None
        assert block._body.plain == '{\n  "command": "ls"\n}'
