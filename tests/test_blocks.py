"""Tests for TUI block rendering helpers."""

from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.content import Content

from drcode.tui.blocks import (
    AssistantBlock,
    ToolCallBlock,
    UserBlock,
    _json_text,
    linkify,
)


def _clicks(text: Text) -> list[tuple[str, str]]:
    """(plain substring, task id) for every span carrying an @click meta."""
    clicks: list[tuple[str, str]] = []
    for span in text.spans:
        style = span.style
        if isinstance(
            style, str
        ):  # pragma: no cover — linkify always sets Style objects
            style = Style.parse(style)
        if "@click" in style.meta:
            clicks.append(
                (text.plain[span.start : span.end], str(style.meta["@click"]))
            )
    return clicks


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


async def test_tool_call_block_elapsed_counter() -> None:
    """A running call's header shows the elapsed time, fed by the spinner
    tick; stop_spinner() freezes the last value instead of clearing it."""

    class _Host(App[None]):
        def compose(self) -> ComposeResult:
            yield ToolCallBlock('search_task_history({"query": "compaction"})')

    async with _Host().run_test() as pilot:
        block = pilot.app.query_one(ToolCallBlock)
        # A call that never ran shows no counter.
        assert block._elapsed_text() == ""
        block.start_spinner()
        assert block._run_started is not None
        block._run_started -= 12.5  # simulate a 12.5 s call without sleeping
        block._redraw()
        header = block.render()
        assert isinstance(header, Content)
        assert "· 12s" in header.plain
        assert "· 12s" in block._elapsed_text()
        block.stop_spinner()
        # Frozen: the value survives the stop, marker back to idle.
        assert block._elapsed_text() == " · 12s"
        frozen = block.render()
        assert isinstance(frozen, Content)
        assert "· 12s" in frozen.plain
        # No wrap for an hour-long (exaggerated) call.
        block.start_spinner()
        block._run_started -= 3725
        block._redraw()
        assert block._elapsed_text() == " · 62m05s"
        block.stop_spinner()


async def test_tool_call_block_counter_restarts_on_reattach() -> None:
    """A fresh block handed the spinner after a view re-render starts the
    counter from zero (its own _run_started), with no stale frozen value."""

    class _Host(App[None]):
        def compose(self) -> ComposeResult:
            yield ToolCallBlock('run_bash({"command": "ls"})')

    async with _Host().run_test() as pilot:
        block = pilot.app.query_one(ToolCallBlock)
        block.start_spinner()
        assert block._run_started is not None
        block._run_started -= 20
        block.stop_spinner()
        assert block._elapsed_text() == " · 20s"
        # Fresh block (as _reattach_tool_spinner would get): counter at zero.
        fresh = ToolCallBlock('run_bash({"command": "ls"})')
        await pilot.app.mount(fresh)
        fresh.start_spinner()
        assert fresh._elapsed_frozen is None
        assert fresh._elapsed_text() == " · 0s"
        fresh.stop_spinner()


def test_linkify_matches_task_mention() -> None:
    text = linkify(Text("see #69c8bd30 please"))
    assert _clicks(text) == [("#69c8bd30", "app.open_task('69c8bd30')")]


def test_linkify_normalizes_uppercase() -> None:
    text = linkify(Text("#ABCDEF12"))
    assert _clicks(text) == [("#ABCDEF12", "app.open_task('abcdef12')")]


def test_linkify_multiple_mentions() -> None:
    text = linkify(Text("#69c8bd30 then #0badf00d"))
    assert [(s, c) for s, c in _clicks(text)] == [
        ("#69c8bd30", "app.open_task('69c8bd30')"),
        ("#0badf00d", "app.open_task('0badf00d')"),
    ]


def test_linkify_rejects_non_ids() -> None:
    # 9 hex chars, 6 (CSS color), 4, 12, and an id after another '#':
    # none of these is a bare 8-hex mention.
    for source in (
        "#abcdef123",
        "#aabbcc",
        "#abcd",
        "#aabbccddeeff",
        "##69c8bd30",
        "#69c8bd3",  # 7 chars
        "no mention here",
    ):
        assert _clicks(linkify(Text(source))) == [], source


def test_linkify_hex_prefix_boundary() -> None:
    # A hex char right before the '#' is part of a longer token, not a
    # mention; a non-hex word char before '#' leaves the mention intact.
    assert _clicks(linkify(Text("a#69c8bd30"))) == []
    assert _clicks(linkify(Text("x#69c8bd30"))) == [
        ("#69c8bd30", "app.open_task('69c8bd30')")
    ]


def test_user_block_links_mentions() -> None:
    # UserBlock linkifies in __init__: the stored content Text already
    # carries the click spans (no app needed to check them).
    block = UserBlock("check #69C8BD30")
    content = block.content
    assert isinstance(content, Text)
    assert _clicks(content) == [("#69C8BD30", "app.open_task('69c8bd30')")]


def test_user_block_no_mentions_no_spans() -> None:
    block = UserBlock("nothing to see")
    content = block.content
    assert isinstance(content, Text)
    assert _clicks(content) == []


async def test_assistant_block_links_mentions_in_rendered_markdown() -> None:
    class _Host(App[None]):
        def compose(self) -> ComposeResult:
            yield AssistantBlock("Mentioned #69c8bd30 in the answer.")

    async with _Host().run_test() as pilot:
        rendered = pilot.app.query_one(AssistantBlock).render()
    assert "#69c8bd30" in rendered.plain
    assert _clicks(rendered) == [("#69c8bd30", "app.open_task('69c8bd30')")]


async def test_assistant_block_streaming_tail_links_mentions() -> None:
    class _Host(App[None]):
        def compose(self) -> ComposeResult:
            yield AssistantBlock()

    async with _Host().run_test() as pilot:
        block = pilot.app.query_one(AssistantBlock)
        block.append("partial line without newline #69c8bd30")
        block._flush()
        rendered = block.render()
    # The raw (unformatted) streaming tail is linkified too.
    assert _clicks(rendered) == [("#69c8bd30", "app.open_task('69c8bd30')")]
