"""Regression test: a streaming AssistantBlock must grow its height.

The block's height is auto (computed at layout time from the renderable).
`refresh()` alone repaints into the stale region and clips the message to
the height measured at the last layout pass — the text looked truncated
until the view was rebuilt. `_flush`/`finish` must request a layout.
"""

from __future__ import annotations

from drcode.tui.app import DrCodeApp
from drcode.tui.blocks import AssistantBlock
from tests.network_mocks import wait_until


def test_streaming_assistant_block_grows(project_root) -> None:
    app = DrCodeApp(project_root)
    app.manager.create("watched task", "the prompt")

    async def run() -> None:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            block = AssistantBlock()
            app._write_chat(block)
            await pilot.pause()
            await pilot.pause()

            long_line = "word " * 60  # wraps into several lines at any width
            block.append(long_line)
            # _flush fires 0.1 s after the first append; let it run and settle.
            await wait_until(lambda: block.size.height >= 3, pilot, timeout=2.0)

            block.append("\n\n" + long_line)
            height_before = block.size.height
            await wait_until(
                lambda: block.size.height > height_before, pilot, timeout=2.0
            )

    import asyncio

    asyncio.run(run())


def test_reasoning_block_applies_markdown() -> None:
    """Reasoning content is rendered as markdown (bold, code, links),
    not as plain text."""
    from rich.text import Text

    from drcode.tui.blocks import ReasoningBlock

    def _styled_spans(body: Text) -> list[str]:
        return [str(s) for s in body.spans]

    # Finished: full markdown render — Rich parses bold / code / links.
    block = ReasoningBlock()
    block._content = "**bold** and `code` and [url]link[/]"
    block._streaming = False
    body = block._render_body()
    assert body is not None
    styles = _styled_spans(body)
    assert any("bold=True" in s for s in styles), "bold markup not styled"
    assert any("cyan" in s for s in styles), "code markup not styled"

    # Streaming: complete fragments highlighted live, partial tail stays raw.
    block2 = ReasoningBlock()
    block2._content = "**bold** end\nstill producing **tail**"
    block2._streaming = True
    body2 = block2._render_body()
    assert body2 is not None
    rendered2 = str(body2)
    # The complete first line is highlighted (markdown applied): the bold
    # markers are consumed, so it renders as the plain "bold end". The
    # still-producing tail stays raw: its "**tail**" markers are preserved.
    assert "bold end" in rendered2
    assert "**tail**" in rendered2
    assert any("bold=True" in s for s in _styled_spans(body2))

