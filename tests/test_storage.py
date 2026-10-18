"""Tests for drcode.storage: persistence, corruption handling, gitignore."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from drcode.models import PendingInput, Task, _now
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


def test_gitignore_appended_when_missing(project_root: Path) -> None:
    (project_root / ".gitignore").write_text("*.pyc\n")
    Storage(project_root)
    text = (project_root / ".gitignore").read_text()
    assert ".drcode/" in text


def test_gitignore_not_duplicated(project_root: Path) -> None:
    (project_root / ".gitignore").write_text(".drcode\n")
    Storage(project_root)
    assert (project_root / ".gitignore").read_text().count(".drcode") == 1


def test_gitignore_respects_comment_lines(project_root: Path) -> None:
    (project_root / ".gitignore").write_text("# .drcode\n")
    Storage(project_root)
    # The commented mention doesn't count: the real entry is added.
    assert ".drcode/" in (project_root / ".gitignore").read_text()


def test_no_gitignore_no_file_created(project_root: Path) -> None:
    Storage(project_root)
    assert not (project_root / ".gitignore").exists()
