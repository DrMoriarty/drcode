"""Regression test: a streaming AssistantBlock must grow its height.

The block's height is auto (computed at layout time from the renderable).
`refresh()` alone repaints into the stale region and clips the message to
the height measured at the last layout pass — the text looked truncated
until the view was rebuilt. `_flush`/`finish` must request a layout.
"""

from __future__ import annotations

from drcode.tui.app import AssistantBlock, DrCodeApp
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
