"""GitHub Issues integration: API client, issue classification and sync.

One sync cycle fetches the repository's open issues and turns the ones
labeled `bug` into local bug reports and the ones labeled `enhancement`
into local feature requests. Issues labeled `invalid` and/or `wontfix`
are dismissed permanently (never reconsidered), as are issues with
neither of the actionable labels (until they get one, they are not
tasks — but they are recorded as seen, so a cycle only ever looks at
issues it has not seen before).

Every imported task carries the originating issue number twice: in the
description (human-readable, "GitHub issue #N: <url>") and in the
structured `Task.github_issue` field, which the tester uses to append
the "Fixes #N" keyword to its commit message so GitHub links the commit
back to the issue.
"""

from __future__ import annotations

import json
import os
import ssl
import tempfile
from datetime import datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from .models import TaskKind, _now
from .task_manager import TaskManager

GITHUB_API = "https://api.github.com"

# Labels that dismiss an issue: never imported, never reconsidered.
GITHUB_SKIP_LABELS = frozenset({"invalid", "wontfix"})
# Label -> local task kind mapping. Issues with neither label are ignored
# (recorded as seen, not imported).
GITHUB_BUG_LABEL = "bug"
GITHUB_ENHANCEMENT_LABEL = "enhancement"

# Issue bodies are pasted into the task prompt verbatim; a pathological
# 200k-character issue must not blow the coding agent's context.
ISSUE_BODY_CAP = 8000

# Per-page size for the issues listing; a single page (100) covers the
# "new issues since the last cycle" use case comfortably.
ISSUES_PER_PAGE = 100


class GitHubError(Exception):
    """A user-readable failure of a GitHub API call."""


class GitHubIssue(BaseModel):
    """The subset of the GitHub issue payload the integration needs."""

    number: int
    title: str
    body: str = ""
    url: str = ""
    labels: list[str] = Field(default_factory=list)


def normalize_repo(text: str) -> str:
    """Normalize user input into the "owner/repo" form.

    Accepts a pasted URL (https://github.com/owner/repo, with or without
    a trailing .git or extra path segments), "owner/repo.git" and the
    plain "owner/repo". Returns the cleaned name ("" for empty input).
    """
    repo = text.strip()
    for prefix in ("https://github.com/", "http://github.com/", "github.com/"):
        if repo.lower().startswith(prefix.lower()):
            repo = repo[len(prefix) :]
            break
    repo = repo.strip("/")
    repo = repo.removesuffix(".git")
    # A pasted issue/PR URL may carry extra segments — keep owner/repo only.
    parts = [p for p in repo.split("/") if p]
    if len(parts) >= 2:
        repo = f"{parts[0]}/{parts[1]}"
    else:
        repo = "/".join(parts)
    return repo


def classify_issue(labels: list[str]) -> TaskKind | None:
    """Map GitHub labels to a local task kind.

    `bug` -> TaskKind.BUG, `enhancement` -> TaskKind.FEATURE, labels
    `invalid`/`wontfix` (in any combination) dismiss the issue (None).
    Case-insensitive. An issue with no actionable label is not a task.
    """
    lowered = {label.strip().lower() for label in labels}
    if lowered & GITHUB_SKIP_LABELS:
        return None
    if GITHUB_BUG_LABEL in lowered:
        return TaskKind.BUG
    if GITHUB_ENHANCEMENT_LABEL in lowered:
        return TaskKind.FEATURE
    return None


def issue_commit_hint(issue: int) -> str:
    """Instruction appended to the tester's prompt for a GitHub-originated
    task: the commit message must end with the "Fixes #N" keyword so
    GitHub links the commit to the originating issue."""
    return (
        f"This task originated from GitHub issue #{issue}. When you create "
        f"the commit, the commit message MUST end with the line 'Fixes "
        f"#{issue}' (on its own line after the summary) — GitHub uses this "
        "keyword to link the commit to the issue and close it."
    )


def issue_task_description(issue: GitHubIssue) -> str:
    """First message of a task created from a GitHub issue.

    The issue number is mandatory in the description (the tester repeats
    it in the commit message via issue_commit_hint); the URL, labels and
    the original body follow.
    """
    labels = ", ".join(issue.labels) if issue.labels else "—"
    body = (issue.body or "(no description)").strip()
    if len(body) > ISSUE_BODY_CAP:
        body = body[:ISSUE_BODY_CAP] + "\n…[truncated]"
    return f"GitHub issue #{issue.number}: {issue.url}\nLabels: {labels}\n\n{body}"


class GitHubClient:
    """Minimal async GitHub REST client (issues read-only).

    `transport` is injectable for tests (httpx.MockTransport); production
    calls construct the client without it. Like probe_model, the system
    certificate store is preferred when truststore is available —
    corporate CA bundles are missing from certifi.
    """

    def __init__(
        self,
        token: str,
        repo: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.token = token.strip()
        self.repo = normalize_repo(repo)
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "drcode",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def _client(self) -> httpx.AsyncClient:
        if self._transport is not None:
            return httpx.AsyncClient(transport=self._transport, timeout=20.0)
        try:
            import truststore

            verify: ssl.SSLContext | bool = truststore.SSLContext(
                ssl.PROTOCOL_TLS_CLIENT
            )
        except ImportError:
            verify = True
        return httpx.AsyncClient(timeout=20.0, verify=verify)

    async def _request(
        self, method: str, path: str, **kwargs: object
    ) -> httpx.Response:
        async with await self._client() as client:
            try:
                return await client.request(
                    method,
                    f"{GITHUB_API}{path}",
                    headers=self._headers(),
                    **kwargs,  # type: ignore[arg-type]
                )
            except httpx.HTTPError as e:
                raise GitHubError(f"network error ({type(e).__name__})") from e

    @staticmethod
    def _status_detail(status: int) -> str:
        if status in (401,):
            return "token rejected (HTTP 401) — check the token"
        if status == 403:
            return (
                "access forbidden (HTTP 403) — the token lacks access, "
                "or the rate limit is exhausted"
            )
        if status == 404:
            return (
                "repository not found (HTTP 404) — check the name "
                "(owner/repo) and that the token can see it"
            )
        return f"GitHub answered HTTP {status}"

    async def probe(self) -> tuple[bool, str]:
        """Check that the token grants access to the repository.

        Used live by the settings dialog: a 200 on GET /repos/{repo}
        means token, repository name and access rights are all good.
        """
        if not self.repo:
            return False, "repository name is empty"
        if "/" not in self.repo:
            return False, "repository name must be in the owner/repo form"
        try:
            resp = await self._request("GET", f"/repos/{self.repo}")
        except GitHubError as e:
            return False, str(e)
        if resp.status_code == 200:
            return True, ""
        return False, self._status_detail(resp.status_code)

    async def fetch_issues(self) -> list[GitHubIssue]:
        """All open, non-PR issues of the repository (one page of 100).

        Raises GitHubError on any transport/API failure.
        """
        if "/" not in self.repo:
            raise GitHubError("repository name must be in the owner/repo form")
        resp = await self._request(
            "GET",
            f"/repos/{self.repo}/issues",
            params={
                "state": "open",
                "per_page": str(ISSUES_PER_PAGE),
                "sort": "created",
                "direction": "desc",
            },
        )
        if resp.status_code != 200:
            raise GitHubError(self._status_detail(resp.status_code))
        try:
            data = resp.json()
        except ValueError as e:
            raise GitHubError("GitHub returned a non-JSON response") from e
        if not isinstance(data, list):
            raise GitHubError("unexpected GitHub response shape")
        issues: list[GitHubIssue] = []
        for item in data:
            if not isinstance(item, dict) or "pull_request" in item:
                continue  # PRs also live in the issues endpoint — skip them
            raw_labels = item.get("labels") or []
            labels = [
                entry.get("name", "") if isinstance(entry, dict) else str(entry)
                for entry in raw_labels
            ]
            issues.append(
                GitHubIssue(
                    number=int(item.get("number", 0)),
                    title=str(item.get("title") or ""),
                    body=str(item.get("body") or ""),
                    url=str(item.get("html_url") or ""),
                    labels=[label for label in labels if label],
                )
            )
        return issues


class GitHubSyncState(BaseModel):
    """Sync bookkeeping, persisted in `<project>/.drcode/github_state.json`.

    `imported` maps every issue number ever SEEN (keys are strings —
    JSON object keys) to the local task id it created, or to "skipped"
    for dismissed/unclassified issues. A cycle therefore only imports
    issues never seen before, and a dismissed issue is never
    reconsidered even if its labels change later.
    """

    last_sync: datetime | None = None
    imported: dict[str, str] = Field(default_factory=dict)


class GitHubSyncStateStore:
    """Loads/saves the sync state from `<project>/.drcode/github_state.json`."""

    def __init__(self, project_root: Path) -> None:
        self.path = project_root / ".drcode" / "github_state.json"

    def load(self) -> GitHubSyncState:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return GitHubSyncState()
        except (OSError, json.JSONDecodeError):
            return GitHubSyncState()
        try:
            return GitHubSyncState.model_validate(data)
        except Exception:  # noqa: BLE001 - corrupt state must not crash the app
            return GitHubSyncState()

    def save(self, state: GitHubSyncState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(state.model_dump_json(indent=2))
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


async def sync_issues(
    manager: TaskManager,
    client: GitHubClient,
    state: GitHubSyncState,
) -> tuple[int, int]:
    """One sync cycle: fetch open issues and create local tasks.

    Returns (imported, skipped) — skipped counts issues dismissed by
    their labels in this cycle. Issues already present in
    `state.imported` are ignored; every processed issue number is
    recorded there. `state.last_sync` is stamped on success; on a
    GitHubError the state is left untouched and the exception propagates
    to the caller (the app shows it as a notification and retries on the
    next scheduled cycle).

    The caller is responsible for persisting the state afterwards.
    """
    issues = await client.fetch_issues()
    imported = 0
    skipped = 0
    for issue in issues:
        key = str(issue.number)
        if key in state.imported:
            continue
        kind = classify_issue(issue.labels)
        if kind is None:
            # Dismissed (invalid/wontfix) or not actionable (no bug /
            # enhancement label): record as seen, never reconsider.
            state.imported[key] = "skipped"
            skipped += 1
            continue
        title = issue.title.strip() or f"GitHub issue #{issue.number}"
        task = manager.create(title, issue_task_description(issue), kind=kind)
        task.github_issue = issue.number
        task.touch()
        manager.storage.save(task)
        state.imported[key] = task.id
        imported += 1
    state.last_sync = _now()
    return imported, skipped
