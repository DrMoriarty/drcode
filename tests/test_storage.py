"""Tests for drcode.storage: persistence, corruption handling, gitignore."""

from __future__ import annotations

import json
import shutil
from datetime import timedelta
from pathlib import Path

from drcode.models import PendingInput, Task, TaskStatus, VerificationStatus, _now
from drcode.storage import Storage


def make_task(title: str, **kwargs) -> Task:
    return Task(title=title, **kwargs)


def test_save_and_load_roundtrip(project_root: Path) -> None:
    storage = Storage(project_root)
    task = make_task(
        "persisted",
        pending_input=[PendingInput(text="do it", at=_now())],
        queue_seq=3,
    )
    storage.save(task)

    assert (project_root / ".drcode" / "tasks" / f"{task.id}.json").is_file()

    storage2 = Storage(project_root)
    loaded = storage2.load_all()
    assert [t.id for t in loaded] == [task.id]
    assert loaded[0].title == "persisted"
    assert [e.text for e in loaded[0].pending_input] == ["do it"]
    assert loaded[0].queue_seq == 3


def test_tasks_dir_created_on_init(project_root: Path) -> None:
    assert not (project_root / ".drcode").exists()
    Storage(project_root)
    assert (project_root / ".drcode" / "tasks").is_dir()


def test_load_all_sorted_by_created_at(project_root: Path) -> None:
    storage = Storage(project_root)
    old = make_task("old")
    old.created_at = _now() - timedelta(hours=1)
    new = make_task("new")
    storage.save(new)
    storage.save(old)  # saved in reverse order on purpose

    loaded = storage.load_all()
    assert [t.title for t in loaded] == ["old", "new"]


def test_corrupted_file_backed_up_not_deleted(project_root: Path) -> None:
    storage = Storage(project_root)
    bad = storage.tasks_dir / "badtask.json"
    bad.write_text("{not json at all")
    good = make_task("good")
    storage.save(good)

    loaded = storage.load_all()
    assert [t.title for t in loaded] == ["good"]
    assert bad.with_suffix(".json.bak").is_file()
    assert not bad.exists()


def test_invalid_schema_backed_up(project_root: Path) -> None:
    storage = Storage(project_root)
    bad = storage.tasks_dir / "wrong.json"
    bad.write_text(json.dumps({"title": 42, "kind": "nonsense"}))

    loaded = storage.load_all()
    assert loaded == []
    assert bad.with_suffix(".json.bak").is_file()


def test_legacy_russian_verification_labels_load(project_root: Path) -> None:
    # Before the Russian→English translation the enum values were Russian
    # strings; persisted tasks carrying them must not be treated as corrupted.
    storage = Storage(project_root)
    for name, legacy in (
        ("passed", "Проверено успешно"),
        ("failed", "Проверено с ошибками"),
    ):
        path = storage.tasks_dir / f"{name}.json"
        path.write_text(json.dumps({"id": name, "title": name, "verification": legacy}))

    loaded = {t.id: t for t in storage.load_all()}
    assert loaded["passed"].verification is VerificationStatus.PASSED
    assert loaded["failed"].verification is VerificationStatus.FAILED
    # And the round-trip persists the current (English) values.
    storage.save(loaded["passed"])
    assert (
        json.loads((storage.tasks_dir / "passed.json").read_text())["verification"]
        == "Verified successfully"
    )


def test_delete_removes_file(project_root: Path) -> None:
    storage = Storage(project_root)
    task = make_task("doomed")
    storage.save(task)
    assert (storage.tasks_dir / f"{task.id}.json").is_file()

    storage.delete(task.id)
    assert not (storage.tasks_dir / f"{task.id}.json").exists()


def test_delete_missing_file_is_noop(project_root: Path) -> None:
    Storage(project_root).delete("nonexistent")


def test_save_is_atomic_no_tmp_leftovers(project_root: Path) -> None:
    storage = Storage(project_root)
    task = make_task("atomic")
    storage.save(task)
    leftovers = list(storage.tasks_dir.glob("*.tmp"))
    assert leftovers == []


def test_save_recreates_vanished_state_dir(project_root: Path) -> None:
    # If .drcode/tasks is removed externally mid-session (e.g. by an `rm -rf`
    # that predates the guard), save() must recreate the dir instead of
    # raising FileNotFoundError and killing the pipeline worker.
    storage = Storage(project_root)
    task = make_task("survivor")
    shutil.rmtree(project_root / ".drcode")
    storage.save(task)
    assert (storage.tasks_dir / f"{task.id}.json").is_file()
    assert storage.load_all()[0].title == "survivor"


def test_gitignore_appended_when_missing(project_root: Path) -> None:
    (project_root / ".git").mkdir()
    (project_root / ".gitignore").write_text("*.pyc\n")
    Storage(project_root)
    text = (project_root / ".gitignore").read_text()
    assert ".drcode/tasks/" in text


def test_gitignore_created_when_absent(project_root: Path) -> None:
    # A git repo without a .gitignore gets one, with the entries inside.
    (project_root / ".git").mkdir()
    Storage(project_root)
    assert (project_root / ".gitignore").is_file()
    assert ".drcode/tasks/" in (project_root / ".gitignore").read_text()


def test_no_git_repo_no_gitignore_created(project_root: Path) -> None:
    # Outside a git repository drcode never drops a .gitignore.
    Storage(project_root)
    assert not (project_root / ".gitignore").exists()


def test_gitignore_not_duplicated(project_root: Path) -> None:
    (project_root / ".git").mkdir()
    (project_root / ".gitignore").write_text(".drcode/tasks/\n")
    Storage(project_root)
    first = (project_root / ".gitignore").read_text()
    Storage(project_root)  # second pass must change nothing
    assert (project_root / ".gitignore").read_text() == first
    assert first.count(".drcode/tasks/") == 1


def test_gitignore_respects_comment_lines(project_root: Path) -> None:
    (project_root / ".git").mkdir()
    (project_root / ".gitignore").write_text("# .drcode\n")
    Storage(project_root)
    # The commented mention doesn't count: the real entries are added.
    assert ".drcode/tasks/" in (project_root / ".gitignore").read_text()
    first = (project_root / ".gitignore").read_text()
    Storage(project_root)  # second pass must change nothing
    assert (project_root / ".gitignore").read_text() == first
    assert first.count(".drcode/tasks/") == 1


# -- split storage: metadata + sidecar transcript ---------------------------------


def _user_msg(text: str) -> dict:
    return {
        "kind": "request",
        "parts": [{"part_kind": "user-prompt", "content": text}],
    }


def _assistant_msg(text: str) -> dict:
    return {"kind": "response", "parts": [{"part_kind": "text", "content": text}]}


def test_transcripts_split_into_sidecar(project_root: Path) -> None:
    storage = Storage(project_root)
    task = make_task("split")
    task.messages = [_user_msg("hello"), _assistant_msg("final answer")]
    task.tester_messages = [[_user_msg("verify")]]
    task.resolver_messages = []
    storage.save(task)

    metadata_path = storage.tasks_dir / f"{task.id}.json"
    sidecar_path = storage.tasks_dir / f"{task.id}.transcript.json"
    assert metadata_path.is_file() and sidecar_path.is_file()
    metadata = json.loads(metadata_path.read_text())
    # The metadata file carries the skeleton, never the bulky transcripts.
    assert "messages" not in metadata
    assert "tester_messages" not in metadata
    assert metadata["skeleton"]["prompts"] == ["hello"]
    assert metadata["skeleton"]["final_response"] == "final answer"
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["messages"] == task.messages
    assert sidecar["tester_messages"] == task.tester_messages
    assert sidecar["resolver_messages"] == []


def test_transcripts_load_lazily_on_first_access(project_root: Path) -> None:
    storage = Storage(project_root)
    task = make_task("lazy")
    task.messages = [_user_msg("hello")]
    task.tester_messages = [[_user_msg("verify")]]
    storage.save(task)

    loaded = Storage(project_root).load_all()[0]
    # Startup parsed metadata only: the sidecar is untouched in memory.
    assert loaded.transcripts_loaded is False
    assert loaded.messages == task.messages
    # One access loads the whole sidecar — all three lists at once.
    assert loaded.transcripts_loaded is True
    assert loaded.tester_messages == task.tester_messages
    assert loaded.resolver_messages == []


def test_unloaded_save_never_rewrites_sidecar(project_root: Path) -> None:
    storage = Storage(project_root)
    task = make_task("quiet")
    task.messages = [_user_msg("hello")]
    storage.save(task)
    sidecar_path = storage.tasks_dir / f"{task.id}.transcript.json"
    sidecar_before = sidecar_path.read_text()

    # A fresh storage (startup shape: transcripts on disk, not in memory):
    # a metadata-only save (status change etc.) must not rewrite megabytes.
    reloaded = Storage(project_root).load_all()[0]
    assert reloaded.transcripts_loaded is False
    reloaded.status = TaskStatus.COMPLETED
    Storage(project_root).save(reloaded)
    assert sidecar_path.read_text() == sidecar_before


def test_loaded_save_flushes_sidecar(project_root: Path) -> None:
    storage = Storage(project_root)
    task = make_task("flush")
    task.messages = [_user_msg("first")]
    storage.save(task)

    reloaded = Storage(project_root).load_all()[0]
    reloaded.messages = [*reloaded.messages, _assistant_msg("second")]
    reloaded.skeleton = {}
    Storage(project_root).save(reloaded)
    # The mutation is durable: a new storage sees it, and the skeleton was
    # rebuilt from the live transcript on flush.
    again = Storage(project_root).load_all()[0]
    assert again.messages == task.messages + [_assistant_msg("second")]
    metadata = json.loads((storage.tasks_dir / f"{task.id}.json").read_text())
    assert metadata["skeleton"]["prompts"] == ["first"]


def test_missing_sidecar_degrades_to_empty(project_root: Path) -> None:
    storage = Storage(project_root)
    task = make_task("orphan")
    task.messages = [_user_msg("hello")]
    storage.save(task)
    storage.transcript_path(task.id).unlink()

    loaded = Storage(project_root).load_all()[0]
    assert loaded.messages == []  # no crash — metadata is the source of truth


def test_corrupted_sidecar_degrades_to_empty(project_root: Path) -> None:
    storage = Storage(project_root)
    task = make_task("broken sidecar")
    task.messages = [_user_msg("hello")]
    storage.save(task)
    storage.transcript_path(task.id).write_text("{not json")

    loaded = Storage(project_root).load_all()[0]
    assert loaded.messages == []


def test_load_all_skips_transcript_files(project_root: Path) -> None:
    storage = Storage(project_root)
    task = make_task("real")
    task.messages = [_user_msg("hello")]
    storage.save(task)
    # A sidecar in the tasks dir must never be parsed as task metadata.
    (storage.tasks_dir / f"{task.id}.transcript.json").write_text('{"messages": []}')

    loaded = storage.load_all()
    assert [t.id for t in loaded] == [task.id]
    assert storage.load_one(storage.transcript_path(task.id)) is None


def test_save_recreates_vanished_dir_with_sidecar(project_root: Path) -> None:
    # Same contract as the metadata test above, for the sidecar: a save of
    # a LOADED task recreates both files after an external rm -rf.
    storage = Storage(project_root)
    task = make_task("survivor")
    task.messages = [_user_msg("hello")]
    storage.save(task)
    shutil.rmtree(project_root / ".drcode")
    storage.save(task)
    assert storage.task_path(task.id).is_file()
    assert storage.transcript_path(task.id).is_file()
    assert Storage(project_root).load_all()[0].messages == task.messages


def test_delete_removes_sidecar_too(project_root: Path) -> None:
    storage = Storage(project_root)
    task = make_task("doomed")
    task.messages = [_user_msg("hello")]
    storage.save(task)
    storage.delete(task.id)
    assert not storage.task_path(task.id).exists()
    assert not storage.transcript_path(task.id).exists()


# -- compaction archive (append-only forensics) --------------------------------------


def test_append_transcript_archive_creates_and_appends(project_root: Path) -> None:
    """The archive file appears on the first appended record and grows by
    exactly one JSON line per record — existing lines are never touched."""
    storage = Storage(project_root)
    task = make_task("archived")
    storage.save(task)
    archive = storage.archive_path(task.id)
    assert not archive.exists()  # created lazily, on the first compaction

    storage.append_transcript_archive(
        task.id,
        {"timestamp": "t1", "level": "threshold", "compaction": "llm", "messages": []},
    )
    storage.append_transcript_archive(
        task.id,
        {
            "timestamp": "t2",
            "level": "hard",
            "compaction": "mechanical",
            "messages": [],
        },
    )
    lines = archive.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["timestamp"] == "t1"
    assert json.loads(lines[1])["compaction"] == "mechanical"


def test_archive_not_read_at_load_all(project_root: Path) -> None:
    """Startup stays metadata-only: load_all neither parses nor requires
    the archive (a corrupted archive line must not affect anything), the
    sidecar stays lazy, and the archive is excluded from the metadata
    globs."""
    storage = Storage(project_root)
    task = make_task("lazy")
    task.messages = [_user_msg("hello")]
    storage.save(task)
    storage.append_transcript_archive(
        task.id, {"timestamp": "t", "messages": [_user_msg("evicted")]}
    )
    # Corrupt the archive: load_all must not care in the slightest.
    storage.archive_path(task.id).write_text('{"broken": tr')

    loaded = Storage(project_root).load_all()
    assert [t.id for t in loaded] == [task.id]
    # The transcript sidecar was NOT loaded at startup (lazy) — and the
    # archive played no role either way.
    assert not loaded[0].transcripts_loaded
    assert loaded[0].messages == [_user_msg("hello")]
    assert storage.archive_path(task.id).read_text(encoding="utf-8") == '{"broken": tr'
    # Metadata globs never pick the archive up as a task file.
    assert storage.task_file_paths() == [storage.task_path(task.id)]


def test_delete_removes_archive_too(project_root: Path) -> None:
    storage = Storage(project_root)
    task = make_task("doomed")
    storage.save(task)
    storage.append_transcript_archive(task.id, {"messages": []})
    assert storage.archive_path(task.id).is_file()

    storage.delete(task.id)
    assert not storage.archive_path(task.id).exists()


# -- legacy monolithic migration --------------------------------------------------


def test_legacy_monolithic_file_migrated_on_load(project_root: Path) -> None:
    """Pre-split archives carry the full transcripts inside <id>.json;
    loading splits them into the pair, derives the skeleton and rewrites
    the metadata file."""
    storage = Storage(project_root)
    legacy = {
        "id": "old1",
        "title": "old format",
        "messages": [_user_msg("legacy prompt"), _assistant_msg("legacy answer")],
        "tester_messages": [[_user_msg("legacy verify")]],
        "resolver_messages": [],
    }
    path = storage.tasks_dir / "old1.json"
    path.write_text(json.dumps(legacy))

    task = storage.load_one(path)
    assert task is not None and task.title == "old format"
    # The pair is on disk; the metadata file no longer carries transcripts.
    sidecar = json.loads(storage.transcript_path("old1").read_text())
    assert sidecar["messages"] == legacy["messages"]
    metadata = json.loads(path.read_text())
    assert "messages" not in metadata
    assert metadata["skeleton"]["prompts"] == ["legacy prompt"]
    assert metadata["skeleton"]["final_response"] == "legacy answer"
    # The returned task is disk-backed: the transcripts reload lazily.
    assert task.transcripts_loaded is False
    assert task.messages == legacy["messages"]
    assert task.tester_messages == legacy["tester_messages"]


def test_legacy_migration_survives_crash_between_replaces(
    project_root: Path,
) -> None:
    """Crash mid-migration (sidecar replaced, legacy metadata not): the
    next load sees the legacy file again and reruns the migration — the
    result is the same pair, no data loss, no duplication."""
    storage = Storage(project_root)
    legacy = {
        "id": "old2",
        "title": "crash test",
        "messages": [_user_msg("crash prompt")],
    }
    path = storage.tasks_dir / "old2.json"
    path.write_text(json.dumps(legacy))
    first = storage.load_one(path)
    assert first is not None
    sidecar_after_first = storage.transcript_path("old2").read_text()

    # Simulate the crash state: legacy metadata back, sidecar present.
    path.write_text(json.dumps(legacy))
    second = storage.load_one(path)
    assert second is not None
    assert second.messages == legacy["messages"]
    assert storage.transcript_path("old2").read_text() == sidecar_after_first
    assert "messages" not in json.loads(path.read_text())
    # And an already-migrated archive reloads idempotently.
    third = storage.load_one(path)
    assert third is not None
    assert third.messages == legacy["messages"]


def test_legacy_migration_invalid_metadata_backed_up(project_root: Path) -> None:
    storage = Storage(project_root)
    path = storage.tasks_dir / "bad_legacy.json"
    path.write_text(json.dumps({"id": "bad", "title": 42, "messages": []}))

    assert storage.load_one(path) is None
    assert path.with_suffix(".json.bak").is_file()
    assert not path.exists()
