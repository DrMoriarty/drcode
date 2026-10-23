"""Tests for the custom team workflow: `.drcode/workflow.py` hooks
(subprocess JSON contract), `.drcode/git-workflow.md` prompt rules, the
external-review state machine (reject loop, accept, fail-closed), the
review-rejection loop protection and the retention exception for tasks
awaiting an async review. Git operations are mocked; the hook scripts
themselves run as real subprocesses; the model layer is stubbed (no
network anywhere) — same style as tests/network_mocks.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from drcode import agent_model, tools
from drcode.agent import all_user_prompts
from drcode.models import TaskKind, TaskStatus, VerificationStatus
from drcode.prompts import (
    GIT_WORKFLOW_HEADER,
    load_system_prompt,
    load_tester_prompt,
)
from drcode.task_manager import TaskManager
from drcode.tui.app import DrCodeApp
from drcode.workflow import (
    GIT_WORKFLOW_EXAMPLE_FILE,
    HOOK_TIMEOUT,
    MAX_REVIEW_REJECTIONS,
    WORKFLOW_EXAMPLE_FILE,
    WorkflowHooks,
    describe_project_workflow,
    ensure_workflow_examples,
    load_git_workflow,
    load_hooks,
)
from tests.network_mocks import (
    ScriptedState,
    event_texts,
    text_response,
    tool_response,
    wait_until,
)

# -- shared fixtures (mirroring tests/test_network_app.py) --------------------------


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured (but never contacted) model endpoint."""
    monkeypatch.setenv("DRCODE_MODEL", "mock-model")
    monkeypatch.setenv("DRCODE_API_KEY", "mock-key")
    monkeypatch.setenv("DRCODE_BASE_URL", "http://127.0.0.1:1/v1")


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedState:
    """Every agent drcode builds is a FunctionModel consuming `steps`."""
    state = ScriptedState()
    monkeypatch.setattr(agent_model, "build_model", lambda cfg: state.model())
    return state


# -- hook script sources (the documented format: plain hook functions; the
# runner's bootstrap does the stdin/stdout dispatch) ------------------------------

REVIEW_ACCEPT_SCRIPT = """
def on_review(task):
    return {"verdict": "accept"}
"""

PASSED_HOOK_SCRIPT = """
def on_tester_passed(task):
    return {"text": "installed with uv tool install"}
"""

PASSED_HOOK_CRASH_SCRIPT = """
def on_tester_passed(task):
    raise RuntimeError("uv install exploded")
"""

REVIEW_ACCEPT_AND_PASSED_SCRIPT = """
def on_review(task):
    return {"verdict": "accept"}

def on_tester_passed(task):
    return {"text": "installed with uv tool install"}

def on_accepted(task):
    return {"text": "merged the PR"}
"""

ECHO_SCRIPT = """
def on_task_start(task):
    return {"seen": task}
"""

REVIEW_DISPATCH_SCRIPT = """
def on_review(task):
    return {"verdict": "accept"}

def on_accepted(task):
    return {"text": "merged the PR"}
"""


def make_script(root: Path, source: str) -> Path:
    drcode = root / ".drcode"
    drcode.mkdir(parents=True, exist_ok=True)
    script = drcode / "workflow.py"
    script.write_text(source)
    return script


def make_hooks(project_root: Path) -> WorkflowHooks:
    hooks = load_hooks(project_root)
    assert hooks is not None
    return hooks


# -- hook discovery / loading ---------------------------------------------------------


def test_load_hooks_none_without_file(project_root: Path) -> None:
    assert load_hooks(project_root) is None
    assert describe_project_workflow(project_root) == "default"


# -- example scaffolding --------------------------------------------------------------


def test_examples_scaffolded_on_manager_init(project_root: Path) -> None:
    TaskManager(project_root)
    workflow_ex = project_root / ".drcode" / WORKFLOW_EXAMPLE_FILE
    git_ex = project_root / ".drcode" / GIT_WORKFLOW_EXAMPLE_FILE
    assert workflow_ex.is_file()
    assert git_ex.is_file()
    # The templates document activation (rename) and mention every hook.
    text = workflow_ex.read_text(encoding="utf-8")
    for hook in (
        "on_task_start",
        "on_development_done",
        "on_tester_passed",
        "on_review",
        "on_accepted",
        "on_research_accepted",
    ):
        assert hook in text
    assert "workflow.py" in text
    assert "git-workflow.md" in git_ex.read_text(encoding="utf-8")


def test_examples_not_overwritten(project_root: Path) -> None:
    drcode = project_root / ".drcode"
    drcode.mkdir(parents=True, exist_ok=True)
    workflow_ex = drcode / WORKFLOW_EXAMPLE_FILE
    git_ex = drcode / GIT_WORKFLOW_EXAMPLE_FILE
    workflow_ex.write_text("# my edits\n")
    git_ex.write_text("my rules\n")
    TaskManager(project_root)
    ensure_workflow_examples(project_root)  # direct repeat: still untouched
    assert workflow_ex.read_text(encoding="utf-8") == "# my edits\n"
    assert git_ex.read_text(encoding="utf-8") == "my rules\n"


def test_examples_do_not_activate_workflow(project_root: Path) -> None:
    ensure_workflow_examples(project_root)
    assert load_hooks(project_root) is None
    assert load_git_workflow(project_root) is None
    assert describe_project_workflow(project_root) == "default"


def test_describe_lists_defined_hooks(project_root: Path) -> None:
    make_script(project_root, "def on_review(task):\n    return None\n")
    hooks = make_hooks(project_root)
    assert hooks.has("on_review")
    assert not hooks.has("on_task_start")
    assert hooks.defined_hooks() == ["on_review"]
    assert "on_review" in describe_project_workflow(project_root)


def test_broken_script_yields_no_hooks(project_root: Path) -> None:
    make_script(project_root, "def broken(:\n")
    hooks = make_hooks(project_root)
    assert not hooks.has("on_review")
    assert hooks.load_error
    assert "unparseable" in describe_project_workflow(project_root)


def test_git_workflow_rules_loading(project_root: Path) -> None:
    assert load_git_workflow(project_root) is None
    drcode = project_root / ".drcode"
    drcode.mkdir()
    (drcode / "git-workflow.md").write_text("  \nBranch per task. \n")
    assert load_git_workflow(project_root) == "Branch per task."
    assert load_git_workflow(None) is None


# -- subprocess hook contract ---------------------------------------------------------


async def test_hook_roundtrip_accept(project_root: Path, manager: TaskManager) -> None:
    make_script(project_root, REVIEW_ACCEPT_SCRIPT)
    hooks = make_hooks(project_root)
    task = manager.create("Add logging", "add logging everywhere")
    result = await hooks.call("on_review", task)
    assert result.ok
    assert result.output == {"verdict": "accept"}


async def test_hook_receives_task_and_git_payload(
    project_root: Path, manager: TaskManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_run_bash(
        root: Path, command: str, timeout: float | None = None
    ) -> str:
        if "abbrev-ref" in command:
            return "exit=0\nfeature/login"
        return "exit=0\n M src/x.py\n"

    monkeypatch.setattr(tools, "run_bash", fake_run_bash)
    make_script(project_root, ECHO_SCRIPT)
    hooks = make_hooks(project_root)
    task = manager.create("Fix crash", "it crashes", kind=TaskKind.BUG)
    task.github_issue = 42
    result = await hooks.call("on_task_start", task)
    assert result.ok
    seen = result.output["seen"]
    assert seen["hook"] == "on_task_start"
    assert seen["task"]["id"] == task.id
    assert seen["task"]["title"] == "Fix crash"
    assert seen["task"]["kind"] == "bug"
    assert seen["task"]["github_issue"] == 42
    assert seen["git"] == {"branch": "feature/login", "dirty": True}


async def test_hook_crash_is_error(project_root: Path, manager: TaskManager) -> None:
    make_script(
        project_root,
        "raise RuntimeError('boom')\n\ndef on_review(task):\n    return {'verdict': 'accept'}\n",
    )
    hooks = make_hooks(project_root)
    result = await hooks.call("on_review", manager.create("t", ""))
    assert not result.ok
    assert "boom" in result.error


async def test_hook_unserializable_result_is_error(
    project_root: Path, manager: TaskManager
) -> None:
    make_script(project_root, "def on_review(task):\n    return object()\n")
    hooks = make_hooks(project_root)
    result = await hooks.call("on_review", manager.create("t", ""))
    assert not result.ok


async def test_hook_non_object_json_is_error(
    project_root: Path, manager: TaskManager
) -> None:
    make_script(project_root, "def on_review(task):\n    return [1]\n")
    hooks = make_hooks(project_root)
    result = await hooks.call("on_review", manager.create("t", ""))
    assert not result.ok
    assert "object" in result.error


async def test_hook_timeout_kills_script(
    project_root: Path, manager: TaskManager
) -> None:
    make_script(project_root, "import time\ndef on_review(task):\n    time.sleep(30)\n")
    hooks = make_hooks(project_root)
    result = await hooks.call("on_review", manager.create("t", ""), timeout=0.5)
    assert not result.ok
    assert "timed out" in result.error


async def test_hook_undefined_is_error(
    project_root: Path, manager: TaskManager
) -> None:
    make_script(project_root, REVIEW_ACCEPT_SCRIPT)
    hooks = make_hooks(project_root)
    result = await hooks.call("on_accepted", manager.create("t", ""))
    assert not result.ok
    assert "not defined" in result.error


def test_hook_timeout_constant_is_bounded() -> None:
    # A hook call blocks the pipeline (sync review) — it must never hang
    # forever, even though CI waits are legitimate.
    assert 0 < HOOK_TIMEOUT <= 3600


# -- prompts: git-workflow rules -------------------------------------------------------


def test_system_prompt_default_has_never_commit(project_root: Path) -> None:
    prompt = load_system_prompt(TaskKind.FEATURE, project_root=project_root)
    assert "Never commit to" in prompt
    assert GIT_WORKFLOW_HEADER not in prompt


def test_system_prompt_with_git_workflow_replaces_never_commit(
    project_root: Path,
) -> None:
    drcode = project_root / ".drcode"
    drcode.mkdir()
    (drcode / "git-workflow.md").write_text("Commit only on a feature branch.")
    for kind in (TaskKind.FEATURE, TaskKind.BUG, TaskKind.RESEARCH):
        prompt = load_system_prompt(kind, project_root=project_root)
        assert "Never commit to" not in prompt
        assert GIT_WORKFLOW_HEADER in prompt
        assert "Commit only on a feature branch." in prompt


def test_system_prompt_without_root_keeps_never_commit() -> None:
    assert "Never commit to" in load_system_prompt(TaskKind.FEATURE)


def test_tester_prompt_with_git_workflow_appends_rules(project_root: Path) -> None:
    drcode = project_root / ".drcode"
    drcode.mkdir()
    (drcode / "git-workflow.md").write_text("Always rebase before push.")
    for allow_commit in (True, False):
        prompt = load_tester_prompt(allow_commit, project_root=project_root)
        assert "Always rebase before push." in prompt
        assert GIT_WORKFLOW_HEADER in prompt


def test_project_prompt_override_wins(project_root: Path) -> None:
    drcode = project_root / ".drcode"
    (drcode / "prompts").mkdir(parents=True)
    (drcode / "prompts" / "feature.md").write_text(
        "Working directory (project root): {project_path}\nCustom feature prompt."
    )
    prompt = load_system_prompt(TaskKind.FEATURE, project_root=project_root)
    assert "Custom feature prompt." in prompt
    assert "implementing a FEATURE" not in prompt
    assert str(project_root) in prompt
    # Other kinds keep the bundled templates.
    assert "expert software engineering agent" in load_system_prompt(
        TaskKind.BUG, project_root=project_root
    )


# -- task manager: review-gate primitives ----------------------------------------------


def test_set_verification_review_gate_holds_task(manager: TaskManager) -> None:
    task = manager.create("t", "")
    task.status = TaskStatus.TESTING
    manager.storage.save(task)
    manager.set_verification(task.id, True, review_gate=True)
    assert task.status is TaskStatus.TESTING
    assert task.verification is VerificationStatus.PASSED


def test_set_verification_review_gate_failure_completes(manager: TaskManager) -> None:
    task = manager.create("t", "")
    task.status = TaskStatus.TESTING
    manager.storage.save(task)
    manager.set_verification(task.id, False, review_gate=True)
    assert task.status is TaskStatus.COMPLETED
    assert task.verification is VerificationStatus.FAILED


def test_set_verification_without_gate_completes(manager: TaskManager) -> None:
    task = manager.create("t", "")
    task.status = TaskStatus.TESTING
    manager.storage.save(task)
    manager.set_verification(task.id, True)
    assert task.status is TaskStatus.COMPLETED


def test_finalize_verified_completes_and_resets_rejections(
    manager: TaskManager,
) -> None:
    task = manager.create("t", "")
    task.status = TaskStatus.TESTING
    task.verification = VerificationStatus.PASSED
    task.review_rejections = 2
    manager.storage.save(task)
    manager.finalize_verified(task.id)
    assert task.status is TaskStatus.COMPLETED
    assert task.verification is VerificationStatus.PASSED
    assert task.completed_at is not None
    assert task.review_rejections == 0


def test_requeue_review_rejection(manager: TaskManager) -> None:
    task = manager.create("t", "")
    task.status = TaskStatus.TESTING
    task.verification = VerificationStatus.PASSED
    manager.storage.save(task)
    manager.requeue_review_rejection(task.id, "fix the style")
    assert task.status is TaskStatus.QUEUED
    assert task.verification is None
    assert task.review_rejections == 1
    assert [e.text for e in task.pending_input] == ["fix the style"]
    # The re-queued task is promotable (the agent gets the comment).
    assert manager.promote_next() is task


def test_requeue_review_rejection_ignores_active_task(manager: TaskManager) -> None:
    task = manager.create("t", "")
    task.status = TaskStatus.ACTIVE
    manager.storage.save(task)
    manager.requeue_review_rejection(task.id, "nope")
    assert task.status is TaskStatus.ACTIVE
    assert task.review_rejections == 0


def test_purge_skips_review_pending_tasks(manager: TaskManager) -> None:
    from datetime import UTC, datetime, timedelta

    task = manager.create("t", "")
    task.status = TaskStatus.COMPLETED
    task.completed_at = datetime.now(UTC) - timedelta(days=30)
    task.review_pending = True
    manager.storage.save(task)
    assert manager.purge_expired_completed(7) == []
    assert task.id in manager.tasks
    task.review_pending = False
    manager.storage.save(task)
    assert [t.id for t in manager.purge_expired_completed(7)] == [task.id]


def test_send_to_review_pending_clears_flag(manager: TaskManager) -> None:
    task = manager.create("t", "")
    task.status = TaskStatus.COMPLETED
    task.review_pending = True
    task.review_rejections = 2
    manager.storage.save(task)
    manager.send(task.id, "actually change the approach")
    assert task.status is TaskStatus.QUEUED
    assert task.review_pending is False
    assert task.review_rejections == 0


def test_cancel_clears_review_pending(manager: TaskManager) -> None:
    task = manager.create("t", "")
    task.status = TaskStatus.TESTING
    task.verification = VerificationStatus.PASSED
    task.review_pending = True
    manager.storage.save(task)
    manager.cancel(task.id)
    assert task.review_pending is False


# -- app-level pipeline: the external-review cycle ---------------------------------------


@pytest.fixture
def review_app(project_root: Path, configured_env: None) -> DrCodeApp:
    return DrCodeApp(project_root)


def passed_testing_task(app: DrCodeApp, title: str = "Add logging"):
    task = app.manager.create(title, "add logging")
    task.status = TaskStatus.TESTING
    task.verification = VerificationStatus.PASSED
    app.manager.storage.save(task)
    return task


REJECT_SCRIPT = """
def on_review(task):
    return {"verdict": "reject", "comment": "fix the style"}
"""

PENDING_SCRIPT = """
def on_review(task):
    return {"verdict": "pending"}
"""

CRASH_ON_REVIEW_SCRIPT = """
def on_review(task):
    return {"verdict": "accept"}
raise RuntimeError("boom")
"""


async def test_sync_review_reject_then_accept(
    review_app: DrCodeApp, scripted: ScriptedState
) -> None:
    """Reject: the reviewer's comment becomes the task's new input and the
    task re-enters the queue (the agent reworks it in the same transcript).
    Accept: the task completes and on_accepted is logged."""
    make_script(review_app.project_root, REVIEW_DISPATCH_SCRIPT)
    review_app.workflow = load_hooks(review_app.project_root)
    task = passed_testing_task(review_app)
    async with review_app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.COMPLETED,
            pilot,
            what="review accept completes the task",
        )
        assert task.verification is VerificationStatus.PASSED
        assert task.review_rejections == 0
        await wait_until(
            lambda: "on_accepted:\nmerged the PR" in "\n".join(event_texts(task)),
            pilot,
            what="on_accepted logged",
        )

        # Now the reviewer rejects.
        make_script(review_app.project_root, REJECT_SCRIPT)
        review_app.workflow = load_hooks(review_app.project_root)
        task.status = TaskStatus.TESTING
        task.verification = VerificationStatus.PASSED
        review_app.manager.storage.save(task)
        review_app._advance()
        await wait_until(
            lambda: task.status is TaskStatus.QUEUED,
            pilot,
            what="review reject requeues the task",
        )
        assert task.verification is None
        assert task.review_rejections == 1
        # The comment reached the model context of the SAME transcript (the
        # pipeline promoted the task immediately; the rework continues in
        # place, no new task).
        assert any("fix the style" in p for p in all_user_prompts(task.messages))
        assert "External review: rejected" in "\n".join(event_texts(task))


async def test_reject_loop_parks_task_on_hold(
    review_app: DrCodeApp, scripted: ScriptedState
) -> None:
    """After MAX_REVIEW_REJECTIONS consecutive rejects the task is held
    for a human instead of looping agent ⇄ reviewer forever. Each cycle:
    dev round (text response) → tester passes (set_verification tool) →
    review rejects."""
    make_script(
        review_app.project_root,
        "def on_review(task):\n"
        "    return {'verdict': 'reject', 'comment': 'still not good'}\n",
    )
    review_app.workflow = load_hooks(review_app.project_root)
    steps: list[Any] = []
    for _ in range(MAX_REVIEW_REJECTIONS):
        # Dev round (one text response), then the tester round: the
        # set_verification tool call plus the model's closing response.
        steps.append(text_response("reworked"))
        steps.append(tool_response("set_verification", {"success": True}))
        steps.append(text_response("all checks passed"))
    scripted.steps = list(steps)
    task = review_app.manager.create("Add logging", "add logging")
    async with review_app.run_test() as pilot:
        await pilot.pause()
        review_app._advance()
        await wait_until(
            lambda: (
                task.status is TaskStatus.HOLD
                and not review_app.agent_running
                and not review_app.tester_running
            ),
            pilot,
            what="task held after the rejection cap",
            timeout=30.0,
        )
        assert task.review_rejections == MAX_REVIEW_REJECTIONS
        assert "human attention" in "\n".join(event_texts(task))
        # Each rejection round appended the reviewer's comment to the same
        # transcript (the rework happens in place, not in a new task).
        prompts = all_user_prompts(task.messages)
        # The LAST rejection parks the task on hold before its rework round
        # starts, so the transcript carries the comment one time fewer than
        # the rejection count.
        assert sum("still not good" in p for p in prompts) >= MAX_REVIEW_REJECTIONS - 1


async def test_review_failure_is_fail_closed(
    review_app: DrCodeApp, scripted: ScriptedState
) -> None:
    make_script(
        review_app.project_root,
        "def on_review(task):\n    raise RuntimeError('boom')\n",
    )
    review_app.workflow = load_hooks(review_app.project_root)
    task = passed_testing_task(review_app)
    async with review_app.run_test() as pilot:
        await pilot.pause()
        review_app._advance()
        await wait_until(
            lambda: (
                "External review (on_review) failed" in "\n".join(event_texts(task))
            ),
            pilot,
            what="review failure logged",
        )
        # Fail-closed: the task keeps waiting for the next attempt.
        assert task.status is TaskStatus.TESTING
        assert task.verification is VerificationStatus.PASSED
        assert task.id in review_app._review_retry_at


async def test_pending_verdict_keeps_task_waiting(
    review_app: DrCodeApp, scripted: ScriptedState
) -> None:
    make_script(review_app.project_root, PENDING_SCRIPT)
    review_app.workflow = load_hooks(review_app.project_root)
    task = passed_testing_task(review_app)
    async with review_app.run_test() as pilot:
        await pilot.pause()
        review_app._advance()
        await wait_until(
            lambda: "no final verdict" in "\n".join(event_texts(task)),
            pilot,
            what="pending verdict logged",
        )
        # A pending verdict RELEASES the pipeline: the task completes with
        # its PASSED label and the poll re-asks for the verdict.
        assert task.status is TaskStatus.COMPLETED
        assert task.verification is VerificationStatus.PASSED
        assert task.review_pending is True


async def test_async_review_poll_accepts_completed_task(
    review_app: DrCodeApp, scripted: ScriptedState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async mode: a task completed with an open PR (review_pending) gets
    its verdict from the periodic poll; accept clears the marker (the
    retention auto-cleanup applies again) and runs on_accepted."""
    make_script(review_app.project_root, REVIEW_DISPATCH_SCRIPT)
    review_app.workflow = load_hooks(review_app.project_root)
    task = review_app.manager.create("Add logging", "add logging")
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    task.review_pending = True
    review_app.manager.storage.save(task)
    review_app._review_poll_at = 0.0
    async with review_app.run_test() as pilot:
        await pilot.pause()
        review_app._poll_workflow_review()
        await wait_until(
            lambda: (
                task.review_pending is False
                and "on_accepted:\nmerged the PR" in "\n".join(event_texts(task))
            ),
            pilot,
            what="async review accept",
        )
        assert task.status is TaskStatus.COMPLETED
        assert task.review_pending is False
        assert task.review_rejections == 0


async def test_task_start_instructions_injected(
    review_app: DrCodeApp, scripted: ScriptedState
) -> None:
    make_script(
        review_app.project_root,
        "def on_task_start(task):\n"
        "    return {'instructions': 'work on branch feature/x'}\n",
    )
    review_app.workflow = load_hooks(review_app.project_root)
    task = review_app.manager.create("Add logging", "add logging everywhere")
    scripted.steps = [text_response("done")]
    async with review_app.run_test() as pilot:
        await pilot.pause()
        await wait_until(
            lambda: task.status is TaskStatus.TESTING,
            pilot,
            what="task reached testing",
        )
    # The instructions ride in the FIRST prompt, before the user's text.
    assert "work on branch feature/x" in scripted.calls[0]
    assert scripted.calls[0].rstrip().endswith("add logging everywhere")
    assert "Workflow on_task_start" in "\n".join(event_texts(task))


async def test_dev_done_note_reaches_tester_prompt(
    review_app: DrCodeApp, scripted: ScriptedState
) -> None:
    make_script(
        review_app.project_root,
        "def on_development_done(task):\n    return {'text': 'PR #7 opened'}\n",
    )
    review_app.workflow = load_hooks(review_app.project_root)
    task = review_app.manager.create("Add logging", "add logging everywhere")
    scripted.steps = [
        text_response("done"),
        tool_response("set_verification", {"success": True}),
        text_response("all checks passed"),
    ]
    async with review_app.run_test() as pilot:
        await pilot.pause()
        review_app._advance()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and not review_app.agent_running
                and not review_app.tester_running
            ),
            pilot,
            what="task completed and pipeline settled",
        )
    # The tester round received the dev-done note (PR #7 opened).
    assert any("PR #7 opened" in call for call in scripted.calls[1:])
    # Without on_review the task simply completed — no review markers.
    assert task.review_pending is False


# -- on_tester_passed: the unconditional post-verification hook -----------------------


def passed_hook_events(task) -> str:
    return "\n".join(
        e.text
        for e in task.events
        if e.kind == "workflow" and "tester_passed" in e.text
    )


async def test_tester_passed_hook_fires_without_review(
    review_app: DrCodeApp, scripted: ScriptedState
) -> None:
    """No on_review defined: the tester passes the task, it completes with
    the PASSED label and on_tester_passed runs — no fake review gate
    needed."""
    make_script(review_app.project_root, PASSED_HOOK_SCRIPT)
    review_app.workflow = load_hooks(review_app.project_root)
    task = review_app.manager.create("Add logging", "add logging everywhere")
    scripted.steps = [
        text_response("done"),
        tool_response("set_verification", {"success": True}),
        text_response("all checks passed"),
    ]
    async with review_app.run_test() as pilot:
        await pilot.pause()
        review_app._advance()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and "installed with uv tool install" in passed_hook_events(task)
            ),
            pilot,
            what="on_tester_passed logged after the PASSED finalization",
        )
    assert task.verification is VerificationStatus.PASSED


async def test_tester_passed_hook_failure_is_non_fatal(
    review_app: DrCodeApp, scripted: ScriptedState
) -> None:
    """Fail-open: a crashing install step is logged into the service events
    and the task stays completed — it must not park the pipeline the way
    the fail-closed on_review does."""
    make_script(review_app.project_root, PASSED_HOOK_CRASH_SCRIPT)
    review_app.workflow = load_hooks(review_app.project_root)
    task = review_app.manager.create("Add logging", "add logging everywhere")
    scripted.steps = [
        text_response("done"),
        tool_response("set_verification", {"success": True}),
        text_response("all checks passed"),
    ]
    async with review_app.run_test() as pilot:
        await pilot.pause()
        review_app._advance()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED
                and "on_tester_passed failed (non-fatal)" in passed_hook_events(task)
            ),
            pilot,
            what="hook failure logged",
        )
    assert task.verification is VerificationStatus.PASSED
    assert "uv install exploded" in passed_hook_events(task)


async def test_tester_passed_hook_skipped_on_failed_verification(
    review_app: DrCodeApp, scripted: ScriptedState
) -> None:
    make_script(review_app.project_root, PASSED_HOOK_SCRIPT)
    review_app.workflow = load_hooks(review_app.project_root)
    task = review_app.manager.create("Add logging", "add logging everywhere")
    scripted.steps = [
        text_response("done"),
        # A foreign failure (retry_here=False): the task completes FAILED
        # and the passed-hook must not fire. The default retry_here=True
        # would requeue the task into another development round instead.
        tool_response("set_verification", {"success": False, "retry_here": False}),
        text_response("tests fail"),
    ]
    async with review_app.run_test() as pilot:
        await pilot.pause()
        review_app._advance()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED and not review_app.tester_running
            ),
            pilot,
            what="task completed with the FAILED label",
        )
    assert task.verification is VerificationStatus.FAILED
    assert passed_hook_events(task) == ""


async def test_tester_passed_hook_never_fires_for_research(
    review_app: DrCodeApp, scripted: ScriptedState
) -> None:
    """Research tasks skip testing entirely (mark_done completes them with
    no verification label) — the hook must not run."""
    make_script(review_app.project_root, PASSED_HOOK_SCRIPT)
    review_app.workflow = load_hooks(review_app.project_root)
    task = review_app.manager.create(
        "Research options", "compare the options", kind=TaskKind.RESEARCH
    )
    scripted.steps = [text_response("research done")]
    async with review_app.run_test() as pilot:
        await pilot.pause()
        review_app._advance()
        await wait_until(
            lambda: (
                task.status is TaskStatus.COMPLETED and not review_app.agent_running
            ),
            pilot,
            what="research task completed",
        )
    assert passed_hook_events(task) == ""


async def test_tester_passed_hook_fires_on_async_review_accept(
    review_app: DrCodeApp, scripted: ScriptedState
) -> None:
    """Review mode: a task completed while awaiting the async verdict gets
    on_tester_passed when the poll finally accepts it (before on_accepted)
    — an install step runs on every path to COMPLETED-after-PASSED."""
    make_script(review_app.project_root, REVIEW_ACCEPT_AND_PASSED_SCRIPT)
    review_app.workflow = load_hooks(review_app.project_root)
    task = review_app.manager.create("Add logging", "add logging")
    task.status = TaskStatus.COMPLETED
    task.verification = VerificationStatus.PASSED
    task.review_pending = True
    review_app.manager.storage.save(task)
    review_app._review_poll_at = 0.0
    async with review_app.run_test() as pilot:
        await pilot.pause()
        review_app._poll_workflow_review()
        await wait_until(
            lambda: (
                task.review_pending is False
                and "installed with uv tool install" in passed_hook_events(task)
                and "on_accepted:" in "\n".join(e.text for e in task.events)
            ),
            pilot,
            what="async review accept runs on_tester_passed",
        )
    assert task.status is TaskStatus.COMPLETED
    events = "\n".join(e.text for e in task.events)
    assert events.index("on_tester_passed:") < events.index("on_accepted:")


def test_tester_passed_hook_in_lifecycle_order(project_root: Path) -> None:
    from drcode.workflow import HOOK_NAMES

    assert HOOK_NAMES == (
        "on_task_start",
        "on_development_done",
        "on_tester_passed",
        "on_review",
        "on_accepted",
        "on_research_accepted",
    )
    make_script(project_root, PASSED_HOOK_SCRIPT)
    hooks = make_hooks(project_root)
    assert hooks.has("on_tester_passed")
    assert hooks.defined_hooks() == ["on_tester_passed"]
    assert "on_tester_passed" in describe_project_workflow(project_root)
