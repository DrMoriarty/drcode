"""Tests for the GitHub Issues integration: settings fields, the API
client (against httpx.MockTransport), issue classification, the sync
cycle (task creation + dedup) and the settings dialog.

No network access happens — every client is built with a MockTransport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import httpx
import pytest

from drcode.github import (
    GitHubClient,
    GitHubError,
    GitHubSyncState,
    GitHubSyncStateStore,
    classify_issue,
    issue_commit_hint,
    issue_task_description,
    normalize_repo,
    sync_issues,
)
from drcode.models import TaskKind
from drcode.settings import (
    GITHUB_SYNC_INTERVAL_OPTIONS,
    GitHubSyncInterval,
    ProjectSettings,
    ProjectSettingsStore,
)
from drcode.task_manager import TaskManager

# -- repo name normalization -----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("owner/repo", "owner/repo"),
        ("  owner/repo  ", "owner/repo"),
        ("https://github.com/owner/repo", "owner/repo"),
        ("http://github.com/owner/repo/", "owner/repo"),
        ("github.com/owner/repo.git", "owner/repo"),
        ("https://github.com/owner/repo/issues/42", "owner/repo"),
        ("owner/repo.git", "owner/repo"),
        ("", ""),
    ],
)
def test_normalize_repo(raw: str, expected: str) -> None:
    assert normalize_repo(raw) == expected


# -- issue classification --------------------------------------------------------


def test_classify_bug() -> None:
    assert classify_issue(["bug", "help wanted"]) is TaskKind.BUG


def test_classify_enhancement() -> None:
    assert classify_issue(["enhancement"]) is TaskKind.FEATURE


def test_classify_case_insensitive() -> None:
    assert classify_issue(["Bug"]) is TaskKind.BUG
    assert classify_issue(["Enhancement"]) is TaskKind.FEATURE


@pytest.mark.parametrize("labels", [["invalid"], ["wontfix"], ["invalid", "bug"]])
def test_classify_skip_labels_wins(labels: list[str]) -> None:
    assert classify_issue(labels) is None


def test_classify_unlabeled_issue_is_not_a_task() -> None:
    assert classify_issue([]) is None
    assert classify_issue(["documentation"]) is None


# -- descriptions and commit hint -------------------------------------------------


def test_issue_description_contains_issue_number() -> None:
    from drcode.github import GitHubIssue

    issue = GitHubIssue(
        number=42,
        title="Crash on start",
        body="Stack trace follows…",
        url="https://github.com/owner/repo/issues/42",
        labels=["bug"],
    )
    text = issue_task_description(issue)
    assert "#42" in text
    assert issue.url in text
    assert "bug" in text
    assert "Stack trace follows…" in text


def test_issue_description_body_capped() -> None:
    from drcode.github import ISSUE_BODY_CAP, GitHubIssue

    issue = GitHubIssue(number=1, title="t", body="x" * (ISSUE_BODY_CAP + 500))
    text = issue_task_description(issue)
    assert "[truncated]" in text
    assert len(text) < ISSUE_BODY_CAP + 500


def test_commit_hint() -> None:
    hint = issue_commit_hint(7)
    assert "Fixes #7" in hint


# -- settings ----------------------------------------------------------------------


def test_sync_interval_enum() -> None:
    assert GitHubSyncInterval.HOUR_1.hours == 1
    assert GitHubSyncInterval.HOUR_2.hours == 2
    assert GitHubSyncInterval.HOUR_4.hours == 4
    assert [i.label for i in GITHUB_SYNC_INTERVAL_OPTIONS] == [
        "1 hour",
        "2 hours",
        "4 hours",
    ]
    # Persisted in settings.json — must not change silently.
    assert GitHubSyncInterval.HOUR_4.value == "4"


def test_default_settings_github_off(project_root: Path) -> None:
    settings = ProjectSettingsStore(project_root).load()
    assert settings.github_enabled is False
    assert settings.github_token == ""
    assert settings.github_repo == ""
    assert settings.github_sync_interval is GitHubSyncInterval.HOUR_1


def test_settings_github_roundtrip(project_root: Path) -> None:
    store = ProjectSettingsStore(project_root)
    settings = ProjectSettings(
        github_enabled=True,
        github_token="ghp_test",
        github_repo="owner/repo",
        github_sync_interval=GitHubSyncInterval.HOUR_4,
    )
    store.save(settings)
    assert store.load() == settings


def test_settings_old_file_without_github_keys(project_root: Path) -> None:
    path = project_root / ".drcode" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"completed_retention": "30"}))
    settings = ProjectSettingsStore(project_root).load()
    assert settings.completed_retention.value == "30"
    assert settings.github_enabled is False


# -- sync state store ---------------------------------------------------------------


def test_state_store_roundtrip(project_root: Path) -> None:
    store = GitHubSyncStateStore(project_root)
    state = GitHubSyncState(imported={"1": "abc", "2": "skipped"})
    store.save(state)
    assert (project_root / ".drcode" / "github_state.json").is_file()
    loaded = store.load()
    assert loaded.imported == {"1": "abc", "2": "skipped"}
    assert loaded.last_sync is None


def test_state_store_missing_and_corrupt(project_root: Path) -> None:
    store = GitHubSyncStateStore(project_root)
    assert store.load() == GitHubSyncState()
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{broken")
    assert store.load() == GitHubSyncState()


# -- client --------------------------------------------------------------------------


def _mock_client(
    handler: httpx.MockTransport, token: str = "tok", repo: str = "owner/repo"
) -> GitHubClient:
    return GitHubClient(token, repo, transport=handler)


async def test_probe_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/owner/repo"
        assert request.headers["Authorization"] == "Bearer tok"
        return httpx.Response(200, json={"full_name": "owner/repo"})

    ok, detail = await _mock_client(httpx.MockTransport(handler)).probe()
    assert ok
    assert detail == ""


async def test_probe_without_token_sends_no_auth_header() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization", "")
        return httpx.Response(200, json={})

    ok, _ = await _mock_client(httpx.MockTransport(handler), token="").probe()
    assert ok
    assert seen["auth"] == ""


async def test_probe_bad_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={})

    ok, detail = await _mock_client(httpx.MockTransport(handler)).probe()
    assert not ok
    assert "401" in detail


async def test_probe_missing_repo() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={})

    ok, detail = await _mock_client(httpx.MockTransport(handler)).probe()
    assert not ok
    assert "404" in detail


async def test_probe_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    ok, detail = await _mock_client(httpx.MockTransport(handler)).probe()
    assert not ok
    assert "network error" in detail


def _issue_payload(number: int, labels: list[str | dict], *, pr: bool = False) -> dict:
    item: dict = {
        "number": number,
        "title": f"Issue {number}",
        "body": f"Body of {number}",
        "html_url": f"https://github.com/owner/repo/issues/{number}",
        "labels": labels,
    }
    if pr:
        item["pull_request"] = {"url": "…"}
    return item


async def test_fetch_issues_parses_and_skips_prs() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/owner/repo/issues"
        assert request.url.params["state"] == "open"
        return httpx.Response(
            200,
            json=[
                _issue_payload(1, [{"name": "bug"}]),
                _issue_payload(2, ["enhancement"], pr=True),
                _issue_payload(3, []),
            ],
        )

    issues = await _mock_client(httpx.MockTransport(handler)).fetch_issues()
    assert [i.number for i in issues] == [1, 3]
    assert issues[0].labels == ["bug"]


async def test_fetch_issues_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    with pytest.raises(GitHubError):
        await _mock_client(httpx.MockTransport(handler)).fetch_issues()


# -- sync cycle -----------------------------------------------------------------------


@pytest.fixture
def manager(project_root: Path) -> TaskManager:
    from drcode.task_manager import TaskManager as TM

    return TM(project_root)


def _issues_transport(payload: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


async def test_sync_creates_bug_and_feature_tasks(manager: TaskManager) -> None:
    transport = _issues_transport(
        [
            _issue_payload(1, ["bug"]),
            _issue_payload(2, ["enhancement"]),
        ]
    )
    state = GitHubSyncState()
    imported, skipped = await sync_issues(manager, _mock_client(transport), state)
    assert (imported, skipped) == (2, 0)
    tasks = manager.all_tasks()
    kinds = sorted(t.kind for t in tasks)
    assert kinds == [TaskKind.BUG, TaskKind.FEATURE]
    bug = next(t for t in tasks if t.kind is TaskKind.BUG)
    # The issue number is mandatory in the description…
    assert "#1" in bug.pending_input[0].text
    # …and stored structurally for the tester's commit message.
    assert bug.github_issue == 1
    assert state.imported["1"] == bug.id
    assert state.last_sync is not None


async def test_sync_skips_dismissed_and_unlabeled(manager: TaskManager) -> None:
    transport = _issues_transport(
        [
            _issue_payload(1, ["invalid"]),
            _issue_payload(2, ["wontfix", "bug"]),
            _issue_payload(3, []),
        ]
    )
    state = GitHubSyncState()
    imported, skipped = await sync_issues(manager, _mock_client(transport), state)
    assert (imported, skipped) == (0, 3)
    assert manager.all_tasks() == []
    # Dismissed issues are recorded as seen — never reconsidered.
    assert state.imported == {"1": "skipped", "2": "skipped", "3": "skipped"}


async def test_sync_is_idempotent(manager: TaskManager) -> None:
    transport = _issues_transport([_issue_payload(1, ["bug"])])
    client = _mock_client(transport)
    state = GitHubSyncState()
    assert (await sync_issues(manager, client, state))[0] == 1
    # Second cycle: the same issue is not imported twice.
    imported, skipped = await sync_issues(manager, client, state)
    assert (imported, skipped) == (0, 0)
    assert len(manager.all_tasks()) == 1


async def test_sync_failure_leaves_state_untouched(
    manager: TaskManager,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={})

    state = GitHubSyncState()
    with pytest.raises(GitHubError):
        await sync_issues(manager, _mock_client(httpx.MockTransport(handler)), state)
    assert state.last_sync is None
    assert state.imported == {}
    assert manager.all_tasks() == []


# -- settings dialog (TUI) -------------------------------------------------------------


async def test_dialog_save_persists_settings(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time as time_module

    from textual.widgets import Button, Input, RadioButton, Switch

    from drcode.tui.app import DrCodeApp
    from drcode.tui.github_dialog import GitHubSettingsDialog

    # No real network from the scheduled sync cycle.
    async def _fake_sync(self: DrCodeApp) -> None:
        self._github_synced = True  # type: ignore[attr-defined]

    monkeypatch.setattr(DrCodeApp, "_github_sync", _fake_sync)
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        await pilot.press("escape")  # skip the first-start setup dialog
        await pilot.pause()
        app.action_change_github()
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, GitHubSettingsDialog)

        dialog.query_one("#gh-token", Input).value = "ghp_live_token"
        dialog.query_one("#gh-repo", Input).value = "https://github.com/owner/repo"
        dialog.query_one("#gh-enabled", Switch).value = True
        # Pick the last interval option (4 hours).
        buttons = list(dialog.query(RadioButton))
        buttons[-1].value = True
        await pilot.pause()

        dialog.query_one("#gh-save", Button).press()
        await pilot.pause()

    settings = ProjectSettingsStore(project_root).load()
    assert settings.github_enabled is True
    assert settings.github_token == "ghp_live_token"
    # A pasted URL is normalized into owner/repo.
    assert settings.github_repo == "owner/repo"
    assert settings.github_sync_interval is GitHubSyncInterval.HOUR_4
    # Enabled + repo set: the app scheduled an immediate sync cycle (the
    # scheduler already re-stamped the next slot to interval + now).
    assert app._github_next_sync > time_module.monotonic()
    assert getattr(app, "_github_synced", False)


async def test_dialog_escape_saves_nothing(project_root: Path) -> None:
    from drcode.tui.app import DrCodeApp

    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
        app.action_change_github()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not None

    settings = ProjectSettingsStore(project_root).load()
    assert settings.github_enabled is False
    assert settings.github_token == ""


async def test_dialog_status_line_updates_live(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status line at the bottom of the dialog re-probes on every
    edit of the token/repository and reports the result."""
    from textual.widgets import Input, Static

    import drcode.tui.github_dialog as dialog_module
    from drcode.tui.app import DrCodeApp
    from drcode.tui.github_dialog import GitHubSettingsDialog

    class FakeClient:
        probed: ClassVar[list[str]] = []

        def __init__(self, token: str, repo: str, transport=None) -> None:
            self.repo = repo

        async def probe(self) -> tuple[bool, str]:
            FakeClient.probed.append(self.repo)
            if self.repo == "owner/good":
                return True, ""
            return False, "repository not found (HTTP 404)"

    monkeypatch.setattr(dialog_module, "GitHubClient", FakeClient)
    app = DrCodeApp(project_root)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
        app.action_change_github()
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, GitHubSettingsDialog)
        status = dialog.query_one("#gh-status", Static)

        # Empty fields: no probe, just a hint.
        await pilot.pause(1.0)
        assert "Enter the token" in str(status.render())

        # A good repo: the status flips to OK after the debounced probe.
        dialog.query_one("#gh-token", Input).value = "tok"
        dialog.query_one("#gh-repo", Input).value = "owner/good"
        await pilot.pause(1.2)
        assert "✓" in str(status.render())

        # An inaccessible repo: the status flips to the failure reason.
        dialog.query_one("#gh-repo", Input).value = "owner/bad"
        await pilot.pause(1.2)
        assert "✗" in str(status.render())
        assert "404" in str(status.render())

        assert FakeClient.probed == ["owner/good", "owner/bad"]
