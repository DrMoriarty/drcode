"""Tests for the per-run read_file dedup cache (read_cache.py).

Conventions match tests/test_tools.py: direct function calls against the
`project_root` fixture, only module constants monkeypatched. The
transcript side is a minimal fake task exposing serialized messages in
the same shape pydantic-ai produces (see eviction.py's scanners)."""

from __future__ import annotations

from typing import Any

import pytest

from drcode import tools
from drcode.eviction import evicted_stub
from drcode.read_cache import ReadCache, read_file_cached


def _transcript(tool_call_id: str, content: str) -> list[dict[str, Any]]:
    """A minimal serialized transcript carrying one read_file result."""
    return [
        {
            "kind": "request",
            "parts": [
                {
                    "part_kind": "tool-call",
                    "tool_call_id": tool_call_id,
                    "tool_name": "read_file",
                    "args": {"path": "f.txt"},
                }
            ],
        },
        {
            "kind": "response",
            "parts": [
                {
                    "part_kind": "tool-return",
                    "tool_call_id": tool_call_id,
                    "tool_name": "read_file",
                    "content": content,
                }
            ],
        },
    ]


class FakeTask:
    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self.messages = messages or []


def _provider(task: FakeTask | None):
    return (lambda: task) if task is not None else None


def _read(root, path="f.txt", **kwargs) -> str:
    kwargs.setdefault("task_provider", _provider(FakeTask()))
    return read_file_cached(ReadCache(), root, path, **kwargs)


def test_first_read_returns_full_content(project_root) -> None:
    (project_root / "f.txt").write_text("a\nb\nc\n")
    result = _read(project_root)
    assert result == "1: a\n2: b\n3: c"


def test_unchanged_reread_returns_stub(project_root) -> None:
    (project_root / "f.txt").write_text(
        "\n".join(f"line {i}" for i in range(200)) + "\n"
    )
    cache = ReadCache()
    task = FakeTask()
    first = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-1",
        task_provider=_provider(task),
    )
    task.messages = _transcript("call-1", first)
    second = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-2",
        task_provider=_provider(task),
    )
    assert "unchanged since read #1" in second
    assert "1: line 0" not in second
    # The full content is NOT re-sent: the stub is far smaller.
    assert len(second) < len(first) // 4


def test_unchanged_reread_full_when_result_missing_from_transcript(
    project_root,
) -> None:
    (project_root / "f.txt").write_text("a\nb\nc\n")
    cache = ReadCache()
    read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-1",
        task_provider=_provider(FakeTask()),  # empty transcript
    )
    second = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-2",
        task_provider=_provider(FakeTask()),
    )
    assert second == "1: a\n2: b\n3: c"
    # The dead entry was dropped and the fresh full read recorded anew —
    # the next re-read can stub again off the new tool result.
    entry = cache.get(str((project_root / "f.txt").resolve()))
    assert entry is not None and entry.tool_call_id == "call-2"


def test_evicted_stub_return_falls_back_to_full(project_root) -> None:
    (project_root / "f.txt").write_text("a\nb\nc\n")
    cache = ReadCache()
    task = FakeTask()
    first = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-1",
        task_provider=_provider(task),
    )
    # mark_unneeded + the eviction funnel collapsed the result to a stub.
    task.messages = _transcript("call-1", evicted_stub("read_file", {"path": "f.txt"}))
    second = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-2",
        task_provider=_provider(task),
    )
    assert second == first


def test_changed_file_returns_diff(project_root) -> None:
    (project_root / "f.txt").write_text(
        "\n".join(f"line {i}" for i in range(40)) + "\n"
    )
    cache = ReadCache()
    task = FakeTask()
    first = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-1",
        task_provider=_provider(task),
    )
    task.messages = _transcript("call-1", first)
    (project_root / "f.txt").write_text(
        "\n".join(f"line {i}" for i in range(39)) + "\nCHANGED\n"
    )
    second = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-2",
        task_provider=_provider(task),
    )
    assert "changed since read #1" in second
    assert "unified diff" in second
    assert "-40: line 39" in second
    assert "+40: CHANGED" in second
    # Unchanged lines are not re-sent.
    assert "\n1: line 0" not in second


def test_changed_file_diff_carries_full_new_state_in_cache(project_root) -> None:
    """After a diff answer the entry stores the FULL new text, so the next
    diff is against the reconstructed file state, not the diff."""
    (project_root / "f.txt").write_text(
        "\n".join(f"line {i}" for i in range(40)) + "\n"
    )
    cache = ReadCache()
    task = FakeTask()
    first = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-1",
        task_provider=_provider(task),
    )
    task.messages = _transcript("call-1", first)
    (project_root / "f.txt").write_text(
        "\n".join(f"line {i}" for i in range(39)) + "\nCHANGED\n"
    )
    second = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-2",
        task_provider=_provider(task),
    )
    task.messages = _transcript("call-2", second)
    (project_root / "f.txt").write_text(
        "\n".join(f"line {i}" for i in range(39)) + "\nREWROTE\n"
    )
    third = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-3",
        task_provider=_provider(task),
    )
    assert "-40: CHANGED" in third
    assert "+40: REWROTE" in third
    # No stale hunks from two reads ago.
    assert "-40: line 39" not in third


def test_changed_file_beyond_share_returns_full_text(project_root) -> None:
    (project_root / "f.txt").write_text("\n".join(f"line {i}" for i in range(40)))
    cache = ReadCache()
    task = FakeTask()
    first = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-1",
        task_provider=_provider(task),
    )
    task.messages = _transcript("call-1", first)
    (project_root / "f.txt").write_text("\n".join(f"CHANGED {i}" for i in range(40)))
    second = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-2",
        task_provider=_provider(task),
    )
    # A near-total rewrite: the diff is comparable to the full text, so
    # the full text comes back instead.
    assert second == tools.read_file(project_root, "f.txt")
    assert "unified diff" not in second


def test_different_slice_returns_full_text(project_root) -> None:
    (project_root / "f.txt").write_text("a\nb\nc\n")
    cache = ReadCache()
    task = FakeTask()
    first = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-1",
        task_provider=_provider(task),
    )
    task.messages = _transcript("call-1", first)
    second = read_file_cached(
        cache,
        project_root,
        "f.txt",
        offset=2,
        limit=1,
        tool_call_id="call-2",
        task_provider=_provider(task),
    )
    assert second == "2: b"


def test_change_outside_slice_reports_slice_current(project_root) -> None:
    (project_root / "f.txt").write_text(
        "\n".join(f"line {i}" for i in range(50)) + "\n"
    )
    cache = ReadCache()
    task = FakeTask()
    first = read_file_cached(
        cache,
        project_root,
        "f.txt",
        limit=10,
        tool_call_id="call-1",
        task_provider=_provider(task),
    )
    task.messages = _transcript("call-1", first)
    (project_root / "f.txt").write_text(
        "\n".join(f"line {i}" for i in range(50)) + "\nappended\n"
    )
    second = read_file_cached(
        cache,
        project_root,
        "f.txt",
        limit=10,
        tool_call_id="call-2",
        task_provider=_provider(task),
    )
    assert "still current" in second
    assert "unified diff" not in second


def test_truncated_reads_never_diff(project_root, monkeypatch) -> None:
    monkeypatch.setattr(tools, "MAX_READ_RESULT_BYTES", 60)
    (project_root / "f.txt").write_text(
        "\n".join(f"line {i}" for i in range(40)) + "\n"
    )
    cache = ReadCache()
    task = FakeTask()
    first = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-1",
        task_provider=_provider(task),
    )
    task.messages = _transcript("call-1", first)
    (project_root / "f.txt").write_text(
        "\n".join(f"CHANGED {i}" for i in range(40)) + "\n"
    )
    second = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-2",
        task_provider=_provider(task),
    )
    # Both slices are head+tail cuts, not file states — no diff theater,
    # the (truncated) full text comes back.
    assert second == tools.read_file(project_root, "f.txt")
    assert "unified diff" not in second


def test_budget_refusal_is_not_cached(project_root) -> None:
    (project_root / "f.txt").write_text("a\nb\nc\n")
    cache = ReadCache()
    refused = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-1",
        current_usage=1000,
        max_context_tokens=100,
    )
    assert refused.startswith(tools.READ_REFUSAL_PREFIX)
    assert not cache.entries


def test_no_task_provider_falls_back_to_full(project_root) -> None:
    (project_root / "f.txt").write_text("a\nb\nc\n")
    cache = ReadCache()
    read_file_cached(cache, project_root, "f.txt", tool_call_id="call-1")
    second = read_file_cached(cache, project_root, "f.txt", tool_call_id="call-2")
    assert second == "1: a\n2: b\n3: c"


def test_none_tool_call_id_falls_back_to_full(project_root) -> None:
    (project_root / "f.txt").write_text("a\nb\nc\n")
    cache = ReadCache()
    read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id=None,
        task_provider=_provider(FakeTask()),
    )
    second = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-2",
        task_provider=_provider(FakeTask()),
    )
    assert second == "1: a\n2: b\n3: c"


def test_failing_task_provider_falls_back_to_full(project_root) -> None:
    def _boom():
        raise RuntimeError("task vanished")

    (project_root / "f.txt").write_text("a\nb\nc\n")
    cache = ReadCache()
    read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-1",
        task_provider=_boom,
    )
    second = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-2",
        task_provider=_boom,
    )
    assert second == "1: a\n2: b\n3: c"


def test_same_content_after_delete_recreate_is_unchanged(project_root) -> None:
    """Byte-identical content IS unchanged, however the file got there —
    the sha test, not mtime, decides."""
    (project_root / "f.txt").write_text("a\nb\nc\n")
    cache = ReadCache()
    task = FakeTask()
    first = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-1",
        task_provider=_provider(task),
    )
    task.messages = _transcript("call-1", first)
    (project_root / "f.txt").write_text("x\ny\n")  # different size/mtime...
    (project_root / "f.txt").write_text("a\nb\nc\n")  # ...then back
    second = read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-2",
        task_provider=_provider(task),
    )
    assert "unchanged since read #1" in second


def test_deleted_file_raises_oserror(project_root) -> None:
    (project_root / "f.txt").write_text("a\n")
    cache = ReadCache()
    task = FakeTask()
    read_file_cached(
        cache,
        project_root,
        "f.txt",
        tool_call_id="call-1",
        task_provider=_provider(task),
    )
    task.messages = _transcript("call-1", "1: a")
    (project_root / "f.txt").unlink()
    with pytest.raises(OSError):
        read_file_cached(
            cache,
            project_root,
            "f.txt",
            tool_call_id="call-2",
            task_provider=_provider(task),
        )


def test_none_cache_is_plain_read(project_root) -> None:
    (project_root / "f.txt").write_text("a\nb\nc\n")
    for _ in range(2):
        assert read_file_cached(None, project_root, "f.txt") == "1: a\n2: b\n3: c"
