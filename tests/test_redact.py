"""Secret redaction on the COMPLETED transition (drcode.redact)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from drcode.models import Task, TaskEvent, TaskKind, TaskStatus
from drcode.notifications import Notifier, PendingNotification
from drcode.redact import (
    REDACTED,
    contains_secret,
    redact_strings_in,
    redact_text,
    scrub_jsonl_file,
    scrub_task,
)
from drcode.settings import ProjectSettings, ProjectSettingsStore
from drcode.task_manager import TaskManager
from drcode.task_search import load_archive

GH_TOKEN = "ghp_0123456789abcdefghijklmnopqrstuv"
GH_PAT = "github_pat_0123456789ABCDEFGHIJKLMNOPQRSTUVWX"
SK_TOKEN = "sk-0123456789abcdef0123456789abcdef"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
BEARER_LINE = "Authorization: Bearer eyJhbGciOi.somecredentialvalue123456"
JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
    "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)
GUID = "838c66ff-8107-4137-9e33-ebcfd828594f"


def _user_message(text: str) -> dict:
    """A minimal serialized ModelRequest carrying one user prompt."""
    return {
        "parts": [
            {"part_kind": "user-prompt", "content": text, "timestamp": 0},
        ]
    }


# -- redact_text -----------------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    [GH_TOKEN, GH_PAT, SK_TOKEN, AWS_KEY, JWT, GUID],
)
def test_redact_text_masks_known_patterns(secret: str) -> None:
    red = redact_text(f"before {secret} after")
    assert secret not in red
    assert REDACTED in red
    assert red.startswith("before ") and red.endswith(" after")


def test_redact_text_keeps_bearer_keyword() -> None:
    red = redact_text(BEARER_LINE)
    assert "Bearer" in red
    assert "somecredentialvalue123456" not in red
    assert REDACTED in red


def test_redact_text_leaves_ordinary_text_alone() -> None:
    text = "fix the parser in src/drcode/tools.py, see task-1234567890abcdef"
    assert redact_text(text) == text


def test_contains_secret() -> None:
    assert contains_secret(f"token {SK_TOKEN}")
    assert not contains_secret("nothing here")


def test_guid_tradeoff_masks_code_contexts_too() -> None:
    # The accepted trade-off: GUIDs also appear in ordinary code/logs and
    # are masked there as well.
    red = redact_text(f"commit {GUID} touched tests")
    assert GUID not in red
    assert REDACTED in red


# -- redact_strings_in / scrub_task -----------------------------------------


def test_redact_strings_in_mutates_values_not_keys() -> None:
    message = {
        "parts": [
            {
                "part_kind": "user-prompt",
                "content": f"use {GH_TOKEN} please",
            },
            {
                "part_kind": "tool-call",
                "tool_name": "run_bash",
                "args": {"command": f"curl -H 'Bearer {SK_TOKEN}'"},
            },
        ]
    }
    assert redact_strings_in(message)
    part = message["parts"][0]
    assert part["part_kind"] == "user-prompt"  # keys untouched
    assert GH_TOKEN not in part["content"]
    tool_call = message["parts"][1]
    assert tool_call["tool_name"] == "run_bash"
    assert SK_TOKEN not in tool_call["args"]["command"]


def test_scrub_task_covers_transcripts_and_metadata() -> None:
    task = Task(title=f"bug {GUID}")
    task.messages = [_user_message(f"prompt with {GH_TOKEN}")]
    task.tester_messages = [[_user_message(f"tester saw {SK_TOKEN}")]]
    task.resolver_messages = [[_user_message(f"resolver saw {AWS_KEY}")]]
    task.events.append(TaskEvent(kind="system", text=f"note with {JWT}"))
    report = scrub_task(task)
    assert report.changed is True
    assert report.credentials_masked > 0
    assert GH_TOKEN not in json.dumps(task.messages)
    assert SK_TOKEN not in json.dumps(task.tester_messages)
    assert AWS_KEY not in json.dumps(task.resolver_messages)
    assert JWT not in task.events[0].text
    assert GUID not in task.title
    # Structure preserved: the API shape of the transcript is intact.
    assert task.messages[0]["parts"][0]["part_kind"] == "user-prompt"
    # Idempotent: a second run finds nothing to change.
    assert scrub_task(task).changed is False


def test_scrub_task_guid_only_is_not_a_credential() -> None:
    # The GUID trade-off fires on practically every task (UUIDs in
    # pydantic-ai run ids, logs, code) — it must not read as a
    # credential (no chat-visible event) while still being masked.
    task = Task(title="plain")
    task.messages = [_user_message(f"run {GUID} finished")]
    report = scrub_task(task)
    assert report.changed is True
    assert report.credentials_masked == 0
    assert GUID not in json.dumps(task.messages)


# -- completion listener integration ----------------------------------------


def _seed_archive_line(manager: TaskManager, task_id: str) -> None:
    manager.storage.append_transcript_archive(
        task_id,
        {"messages": [_user_message(f"pre-compaction {GH_TOKEN}")]},
    )


def test_completion_scrubs_all_files_on_disk(
    manager: TaskManager, tmp_path: Path
) -> None:
    task = manager.create(
        f"secret task {GUID}",
        f"please use {SK_TOKEN} for the API",
        kind=TaskKind.RESEARCH,
    )
    task.messages = [_user_message(f"dialog: {GH_TOKEN}")]
    manager.storage.save(task)
    _seed_archive_line(manager, task.id)
    # Prime the persistent search index with the pre-scrub metadata.
    load_archive(manager.storage.root)

    manager.mark_done(task.id)

    assert task.status is TaskStatus.COMPLETED
    metadata_text = manager.storage.task_path(task.id).read_text(encoding="utf-8")
    sidecar_text = manager.storage.transcript_path(task.id).read_text(encoding="utf-8")
    archive_text = manager.storage.archive_path(task.id).read_text(encoding="utf-8")
    index_text = (manager.storage.root / ".drcode" / "search_index.json").read_text(
        encoding="utf-8"
    )
    for secret in (GH_TOKEN, SK_TOKEN, GUID):
        assert secret not in metadata_text, secret
        assert secret not in sidecar_text, secret
        assert secret not in archive_text, secret
        assert secret not in index_text, secret
    assert REDACTED in sidecar_text
    assert REDACTED in archive_text
    # The service event is appended and persisted, and the skeleton (rebuilt
    # by the save inside the scrubber) is clean too.
    assert task.events[-1].text.startswith("secret redaction")
    assert SK_TOKEN not in metadata_text
    # A follow-up search load re-parses the scrubbed metadata — no secret
    # comes back through the index.
    for _task_id, _reduced, fields in load_archive(manager.storage.root):
        assert not any(contains_secret(text) for _label, text in fields)


def test_completion_without_setting_keeps_secrets(
    manager: TaskManager,
) -> None:
    store = ProjectSettingsStore(manager.storage.root)
    settings = ProjectSettings()
    settings.redact_secrets_on_completion = False
    store.save(settings)

    task = manager.create("t", f"prompt {SK_TOKEN}", kind=TaskKind.RESEARCH)
    task.messages = [_user_message(f"dialog: {GH_TOKEN}")]
    manager.storage.save(task)
    manager.mark_done(task.id)

    sidecar_text = manager.storage.transcript_path(task.id).read_text(encoding="utf-8")
    assert GH_TOKEN in sidecar_text
    assert not any(event.text.startswith("secret redaction") for event in task.events)


def test_guid_only_completion_scrubs_silently(
    manager: TaskManager,
) -> None:
    # UUIDs are everywhere (run ids, logs) — a GUID-only scrub must mask
    # the files but NOT grow a chat event on every completed task.
    task = manager.create("t", f"see {GUID}", kind=TaskKind.RESEARCH)
    task.messages = [_user_message(f"run id {GUID}")]
    manager.storage.save(task)
    manager.mark_done(task.id)
    sidecar_text = manager.storage.transcript_path(task.id).read_text(encoding="utf-8")
    assert GUID not in sidecar_text
    assert not any(event.text.startswith("secret redaction") for event in task.events)


def test_archive_only_secret_is_scrubbed_too(
    manager: TaskManager,
) -> None:
    # After a compaction the original prompts live ONLY in the append-only
    # archive — the live transcripts may be clean while the archive still
    # carries the token. The scrubber must not skip the archive in that
    # case (and the event fires — a real credential was masked).
    task = manager.create("t", "prompt", kind=TaskKind.RESEARCH)
    task.messages = [_user_message("summary only")]
    manager.storage.save(task)
    _seed_archive_line(manager, task.id)
    manager.mark_done(task.id)
    archive_text = manager.storage.archive_path(task.id).read_text(encoding="utf-8")
    assert GH_TOKEN not in archive_text
    assert REDACTED in archive_text
    assert any(event.text.startswith("secret redaction") for event in task.events)


def test_scrub_listener_survives_internal_error(
    manager: TaskManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = manager.create("t", "x", kind=TaskKind.RESEARCH)
    monkeypatch.setattr(
        "drcode.task_manager.scrub_task",
        lambda _task: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    # The listener guard must swallow the failure: the task still completes.
    manager.mark_done(task.id)
    assert task.status is TaskStatus.COMPLETED


def test_unfinished_task_is_never_scrubbed(manager: TaskManager) -> None:
    task = manager.create(f"held {GUID}", "prompt", hold=True)
    task.messages = [_user_message(f"dialog: {GH_TOKEN}")]
    manager.storage.save(task)
    # Held tasks never enter COMPLETED — nothing runs.
    assert task.status is TaskStatus.HOLD
    sidecar_text = manager.storage.transcript_path(task.id).read_text(encoding="utf-8")
    assert GH_TOKEN in sidecar_text


def test_scrubbed_history_stays_on_requeue(manager: TaskManager) -> None:
    task = manager.create("t", f"use {SK_TOKEN}", kind=TaskKind.RESEARCH)
    task.messages = [_user_message(f"dialog: {GH_TOKEN}")]
    manager.storage.save(task)
    manager.mark_done(task.id)
    # Re-queue by sending new input: the transcript stays scrubbed (the
    # secret is gone from the model's context too — by design).
    manager.send(task.id, "one more round")
    assert task.status is TaskStatus.QUEUED
    sidecar_text = manager.storage.transcript_path(task.id).read_text(encoding="utf-8")
    assert GH_TOKEN not in sidecar_text


def test_scrub_jsonl_file_is_atomic_and_selective(tmp_path: Path) -> None:
    path = tmp_path / "a.transcript.archive.jsonl"
    path.write_text(
        json.dumps({"messages": [_user_message(f"x {GH_TOKEN}")]})
        + "\n"
        + json.dumps({"messages": [_user_message("clean line")]})
        + "\n",
        encoding="utf-8",
    )
    report = scrub_jsonl_file(path)
    assert report.changed is True
    assert report.credentials_masked > 0
    lines = path.read_text(encoding="utf-8").splitlines()
    assert GH_TOKEN not in lines[0]
    assert REDACTED in lines[0]
    assert "clean line" in lines[1]
    # No leftovers (tmp files cleaned, valid JSON lines).
    assert not list(tmp_path.glob("*.tmp"))
    json.loads(lines[0])
    assert scrub_jsonl_file(path).changed is False


def test_scrub_jsonl_file_missing_is_noop(tmp_path: Path) -> None:
    assert scrub_jsonl_file(tmp_path / "nope.jsonl").changed is False


# -- notifications -----------------------------------------------------------


def test_notifier_redacts_outgoing_push(project_root: Path) -> None:
    captured: list[tuple[str, str]] = []
    notifier = Notifier(
        lambda: ProjectSettings(notifications_enabled=False),
        project_root=project_root,
    )
    # notifications_enabled=False keeps the queue from delivering; the
    # redaction check runs at task_status time — verify via notify capture.
    notifier.notify = (  # type: ignore[method-assign]
        lambda title, message, **kwargs: captured.append((title, message))
    )
    task = Task(title=f"leaky {GUID}")
    notifier.task_status(task, TaskStatus.ACTIVE, TaskStatus.COMPLETED)
    title, message = captured[0]
    assert GUID not in title
    assert GUID not in message
    assert REDACTED in message


def test_notifier_redact_pending_sweeps_queue(project_root: Path) -> None:
    settings = ProjectSettings(notifications_enabled=False)
    notifier = Notifier(lambda: settings, project_root=project_root)
    # Seed the pending queue directly (notify() with notifications off
    # drops the entry before it ever reaches the queue, and a real send
    # would hit the network).
    notifier._pending.append(
        PendingNotification(
            topic="x",
            title="t",
            message=f"undelivered {GH_TOKEN}",
            created_at=datetime.now().astimezone(),
        )
    )
    notifier.redact_pending()
    pending = notifier._pending
    assert pending and GH_TOKEN not in pending[0].message
    assert REDACTED in pending[0].message
    # The pending FILE was rewritten clean too.
    pending_file = project_root / ".drcode" / "pending_notifications.json"
    assert GH_TOKEN not in pending_file.read_text(encoding="utf-8")
