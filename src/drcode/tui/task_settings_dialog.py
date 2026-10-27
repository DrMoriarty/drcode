"""Task settings dialog — the single modal form for the task-management
project preferences (variant A of the settings redesign).

Opened from the Settings palette's "Task settings…" entry (it replaces
the old TaskSettingsPalette sub-menu; the per-field palettes and their
``action_set_*`` actions stay for quick toggles). The dialog:

- works on a DEEP COPY of the saved settings: nothing is persisted until
  Apply (ESC dismisses with None and discards everything);
- offers the workflow presets as a two-column radio grid on top — picking
  one applies it to the draft and every field widget updates in place;
  "Custom" is only the display state of a draft matching no preset
  (mirrors ``action_workflow_preset_custom``);
- groups the 11 field widgets into the three sections of the old sub-menu
  (Workflow / Testing / Git) — the closely-related fields sit side by side
  in two-column rows — marking fields that differ from the saved
  values and dimming the currently dormant ones (their value parks and
  activates when the governing field turns on — the user's choice is
  never silently rewritten);
- re-renders a hint panel on every draft change from
  :func:`drcode.settings.task_settings_hints` — one shared source of the
  coupling rules, with an optional fix button per hint the user must
  press explicitly.

On Apply the draft is handed back to the app layer
(``SettingsLayer._task_settings_done`` → ``action_apply_task_settings``)
which persists it once and replays the same side effects as the
per-field actions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Label,
    RadioButton,
    RadioSet,
    Select,
    Static,
    Switch,
)

from ..settings import (
    DEV_ROUNDS_OPTIONS,
    GIT_POLICY_OPTIONS,
    ISOLATION_OPTIONS,
    PARALLEL_TASKS_OPTIONS,
    RETENTION_OPTIONS,
    STALL_NUDGE_OPTIONS,
    SUBAGENTS_OPTIONS,
    VERIFICATION_OPTIONS,
    WORKTREE_REMOVAL_OPTIONS,
    CompletedRetention,
    GitPolicy,
    ProjectSettings,
    StallNudge,
    TaskIsolation,
    Verification,
    WorktreeRemoval,
    dev_rounds_label,
    parallel_tasks_label,
    subagents_label,
    task_settings_hints,
)
from .command_providers import WORKFLOW_PRESETS, workflow_preset_for

if TYPE_CHECKING:
    from .app import DrCodeApp


def copy_task_settings(dst: ProjectSettings, src: ProjectSettings) -> None:
    """Copy the task-management fields of `src` into `dst` IN PLACE.

    The app's live settings object is shared (the TaskEnvironment holds
    the same instance so isolation toggles propagate to the next
    activation), so an accepted draft is merged field-by-field instead of
    replacing the object. Deliberately only the fields the dialog edits —
    the rest of ProjectSettings (GitHub, voice, …) is untouched.
    """
    dst.completed_retention = src.completed_retention
    dst.verification = src.verification
    dst.git_policy.tester = src.git_policy.tester
    dst.git_policy.research = src.git_policy.research
    dst.max_dev_rounds = src.max_dev_rounds
    dst.stall_nudge = src.stall_nudge
    dst.task_isolation = src.task_isolation
    dst.parallel_tasks = src.parallel_tasks
    dst.subagents_coder = src.subagents_coder
    dst.subagents_researcher = src.subagents_researcher
    dst.worktree_removal = src.worktree_removal
    dst.isolation_start_branch = src.isolation_start_branch
    dst.keep_awake = src.keep_awake


class TaskSettingsDialog(ModalScreen[ProjectSettings | None]):
    """Modal form: preset radio row, three sections of field widgets,
    live hints, Apply / ESC-to-cancel. Returns the accepted draft (the
    same object it mutated) or None on cancel."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Close", priority=True),
    ]

    AUTO_LABEL = "auto (HEAD branch)"

    DEFAULT_CSS = """
    TaskSettingsDialog {
        align: center middle;
    }
    #ts-dialog {
        width: 96;
        max-width: 95%;
        height: auto;
        max-height: 92%;
        padding: 0 2 1 2;
        background: $panel;
        border: thick $accent;
    }
    #ts-title {
        color: $text;
        text-style: bold;
        margin-bottom: 1;
    }
    #ts-body {
        height: auto;
        max-height: 24;
        overflow-y: auto;
    }
    .ts-section {
        color: $text;
        text-style: bold;
        margin-top: 1;
        background: $panel;
    }
    .ts-label {
        width: auto;
        color: $text-muted;
        margin-top: 1;
    }
    #ts-preset {
        margin-top: 1;
        layout: grid;
        grid-size: 2;
    }
    .ts-row {
        height: auto;
    }
    .ts-col {
        width: 1fr;
        height: auto;
    }
    .ts-col > .ts-label {
        height: 1;
        overflow: hidden;
    }
    .ts-row .ts-col:first-child {
        margin-right: 1;
    }
    Select {
        width: 1fr;
    }
    Select.changed {
        border: tall $warning;
    }
    Switch.changed {
        border: tall $warning;
    }
    .ts-dormant {
        opacity: 0.55;
    }
    #ts-hints {
        height: auto;
        max-height: 8;
        overflow-y: auto;
        margin-top: 1;
    }
    .ts-hint {
        height: auto;
        margin-bottom: 1;
    }
    .ts-hint.dormant {
        color: $warning;
    }
    .ts-hint.conflict {
        color: $error;
        text-style: bold;
    }
    .ts-hint.info {
        color: $text-muted;
    }
    .ts-hint-empty {
        color: $text-disabled;
    }
    .ts-hint-text {
        width: 1fr;
    }
    .ts-fix {
        width: auto;
        margin-left: 1;
        height: auto;
    }
    #ts-buttons {
        dock: bottom;
        height: auto;
        margin-top: 1;
    }
    #ts-apply {
        width: auto;
        margin-right: 1;
    }
    #ts-cancel {
        width: auto;
    }
    #ts-esc-note {
        width: 1fr;
        content-align: right middle;
        color: $text-disabled;
    }
    """

    def __init__(
        self,
        settings: ProjectSettings,
        *,
        git_available: bool | None = None,
    ) -> None:
        super().__init__()
        self._original = settings.model_copy(deep=True)
        self._draft = settings.model_copy(deep=True)
        # None = derive from the app's TaskEnvironment at mount time.
        self._git_available = git_available
        self._branches: list[str] = []
        # fix-button id -> the hint's fix callable (rebuilt per render).
        self._fixes: dict[str, Callable[[ProjectSettings], None]] = {}

    # -- composition -----------------------------------------------------------

    def _branch_options(self) -> list[tuple[str, str]]:
        """The isolation-start-branch options: auto + every local branch,
        plus the draft's current value when git does not list it (a
        parked branch name must stay selectable — never rewritten)."""
        options: list[tuple[str, str]] = [(self.AUTO_LABEL, "")]
        options.extend((branch, branch) for branch in self._branches)
        current = self._draft.isolation_start_branch
        if current and current not in {value for _, value in options}:
            options.append((current, current))
        return options

    def compose(self) -> ComposeResult:
        d = self._draft
        matched = workflow_preset_for(d)
        with Vertical(id="ts-dialog"):
            yield Static("Task settings", id="ts-title")
            with VerticalScroll(id="ts-body"):
                yield Label("Workflow preset", classes="ts-section")
                with RadioSet(id="ts-preset"):
                    for preset in WORKFLOW_PRESETS:
                        yield RadioButton(
                            preset.label,
                            value=matched is preset,
                            id=f"ts-preset-{preset.key}",
                        )
                    yield RadioButton(
                        "Custom", value=matched is None, id="ts-preset-custom"
                    )
                yield Label("Workflow", classes="ts-section")
                with Horizontal(classes="ts-row"):
                    with Vertical(classes="ts-col"):
                        yield Label("Task isolation", classes="ts-label")
                        yield Select(
                            [(o.label, o) for o in ISOLATION_OPTIONS],
                            value=d.task_isolation,
                            allow_blank=False,
                            id="ts-isolation",
                        )
                    with Vertical(classes="ts-col"):
                        yield Label("Isolation start branch", classes="ts-label")
                        yield Select(
                            self._branch_options(),
                            value=d.isolation_start_branch,
                            allow_blank=False,
                            id="ts-start-branch",
                        )
                with Horizontal(classes="ts-row"):
                    with Vertical(classes="ts-col"):
                        yield Label("Parallel tasks", classes="ts-label")
                        yield Select(
                            [
                                (parallel_tasks_label(n), n)
                                for n in PARALLEL_TASKS_OPTIONS
                            ],
                            value=d.parallel_tasks,
                            allow_blank=False,
                            id="ts-parallel",
                        )
                    with Vertical(classes="ts-col"):
                        yield Label("Worktree removal", classes="ts-label")
                        yield Select(
                            [(o.label, o) for o in WORKTREE_REMOVAL_OPTIONS],
                            value=d.worktree_removal,
                            allow_blank=False,
                            id="ts-removal",
                        )
                with Horizontal(classes="ts-row"):
                    with Vertical(classes="ts-col"):
                        yield Label("Subagents (coder)", classes="ts-label")
                        yield Select(
                            [(subagents_label(n), n) for n in SUBAGENTS_OPTIONS],
                            value=d.subagents_coder,
                            allow_blank=False,
                            id="ts-subagents-coder",
                        )
                    with Vertical(classes="ts-col"):
                        yield Label("Subagents (researcher)", classes="ts-label")
                        yield Select(
                            [(subagents_label(n), n) for n in SUBAGENTS_OPTIONS],
                            value=d.subagents_researcher,
                            allow_blank=False,
                            id="ts-subagents-researcher",
                        )
                yield Label("Testing", classes="ts-section")
                with Horizontal(classes="ts-row"):
                    with Vertical(classes="ts-col"):
                        yield Label("Verification", classes="ts-label")
                        yield Select(
                            [(o.label, o) for o in VERIFICATION_OPTIONS],
                            value=d.verification,
                            allow_blank=False,
                            id="ts-verification",
                        )
                    with Vertical(classes="ts-col"):
                        yield Label("Max development rounds", classes="ts-label")
                        yield Select(
                            [(dev_rounds_label(n), n) for n in DEV_ROUNDS_OPTIONS],
                            value=d.max_dev_rounds,
                            allow_blank=False,
                            id="ts-dev-rounds",
                        )
                yield Label("Nudge stalled agent after", classes="ts-label")
                yield Select(
                    [(o.label, o) for o in STALL_NUDGE_OPTIONS],
                    value=d.stall_nudge,
                    allow_blank=False,
                    id="ts-nudge",
                )
                yield Label("Git", classes="ts-section")
                with Horizontal(classes="ts-row"):
                    with Vertical(classes="ts-col"):
                        yield Label("Git (tester)", classes="ts-label")
                        yield Select(
                            [(o.label, o) for o in GIT_POLICY_OPTIONS],
                            value=d.git_policy.tester,
                            allow_blank=False,
                            id="ts-git-tester",
                        )
                    with Vertical(classes="ts-col"):
                        yield Label("Git (research)", classes="ts-label")
                        yield Select(
                            [(o.label, o) for o in GIT_POLICY_OPTIONS],
                            value=d.git_policy.research,
                            allow_blank=False,
                            id="ts-git-research",
                        )
                yield Label("Auto-delete completed tasks", classes="ts-label")
                yield Select(
                    [(o.label, o) for o in RETENTION_OPTIONS],
                    value=d.completed_retention,
                    allow_blank=False,
                    id="ts-retention",
                )
                yield Label("Keep computer awake", classes="ts-label")
                yield Switch(value=d.keep_awake, id="ts-keep-awake")
            yield Vertical(id="ts-hints")
            with Horizontal(id="ts-buttons"):
                yield Button("Apply", variant="primary", id="ts-apply")
                yield Button("Cancel", id="ts-cancel")
                yield Static("esc — discard all changes", id="ts-esc-note")

    def on_mount(self) -> None:
        if self._git_available is None:
            app = self._drcode_app()
            self._git_available = app.task_env.available
        self._refresh()
        self.run_worker(self._load_branches(), exclusive=True, group="ts-branches")

    def _drcode_app(self) -> DrCodeApp:
        from .app import DrCodeApp

        app = self.app
        assert isinstance(app, DrCodeApp)
        return app

    async def _load_branches(self) -> None:
        """Fetch the repository's local branches (off the event loop) and
        refresh the start-branch options; without git the select keeps
        just the auto entry."""
        try:
            app = self._drcode_app()
            if app.task_env.available:
                self._branches = await asyncio.to_thread(app.task_env.local_branches)
        except Exception:  # noqa: BLE001 — a broken git must not kill the dialog
            self._branches = []
        if not self.is_mounted:  # dismissed while git ran
            return
        select = self.query_one("#ts-start-branch", Select)
        if select.value != self._draft.isolation_start_branch:
            select.value = self._draft.isolation_start_branch
        # set_options re-applies the current value; do it last so the
        # freshly fetched branches are all selectable.
        select.set_options(self._branch_options())
        self._refresh()

    # -- draft -> widgets ------------------------------------------------------

    def _set_select(self, selector: str, value: object) -> None:
        select = self.query_one(selector, Select)
        if select.value != value:
            # Programmatic sync must never post Select.Changed: Textual
            # delivers messages asynchronously, so such an event can be
            # STALE by the time its handler runs (the user has already
            # moved the widget on) — the handler would re-apply the stale
            # value into the draft and _refresh() would answer by writing
            # another widget, regenerating events forever (the
            # both-subagents-selects flicker loop).
            with self.prevent(Select.Changed):
                select.value = value  # type: ignore[assignment]

    def _sync_preset(self) -> None:
        """Press the radio of the preset the draft currently matches, or
        "Custom" when it matches none (the display-state rule)."""
        matched = workflow_preset_for(self._draft)
        for preset in WORKFLOW_PRESETS:
            radio = self.query_one(f"#ts-preset-{preset.key}", RadioButton)
            should = matched is preset
            if radio.value != should:
                radio.value = should
        custom_radio = self.query_one("#ts-preset-custom", RadioButton)
        if custom_radio.value != (matched is None):
            custom_radio.value = matched is None

    def _sync_fields(self) -> None:
        """Push the draft into the field widgets (after a preset pick) and
        mark/dim them."""
        d = self._draft
        o = self._original
        self._set_select("#ts-isolation", d.task_isolation)
        self._set_select("#ts-start-branch", d.isolation_start_branch)
        self._set_select("#ts-parallel", d.parallel_tasks)
        self._set_select("#ts-removal", d.worktree_removal)
        self._set_select("#ts-subagents-coder", d.subagents_coder)
        self._set_select("#ts-subagents-researcher", d.subagents_researcher)
        self._set_select("#ts-verification", d.verification)
        self._set_select("#ts-dev-rounds", d.max_dev_rounds)
        self._set_select("#ts-nudge", d.stall_nudge)
        self._set_select("#ts-git-tester", d.git_policy.tester)
        self._set_select("#ts-git-research", d.git_policy.research)
        self._set_select("#ts-retention", d.completed_retention)
        switch = self.query_one("#ts-keep-awake", Switch)
        if switch.value != d.keep_awake:
            with self.prevent(Switch.Changed):  # same staleness rule
                switch.value = d.keep_awake
        changed = {
            "#ts-isolation": d.task_isolation is not o.task_isolation,
            "#ts-start-branch": d.isolation_start_branch != o.isolation_start_branch,
            "#ts-parallel": d.parallel_tasks != o.parallel_tasks,
            "#ts-removal": d.worktree_removal is not o.worktree_removal,
            "#ts-subagents-coder": d.subagents_coder != o.subagents_coder,
            "#ts-subagents-researcher": (
                d.subagents_researcher != o.subagents_researcher
            ),
            "#ts-verification": d.verification is not o.verification,
            "#ts-dev-rounds": d.max_dev_rounds != o.max_dev_rounds,
            "#ts-nudge": d.stall_nudge is not o.stall_nudge,
            "#ts-git-tester": d.git_policy.tester is not o.git_policy.tester,
            "#ts-git-research": d.git_policy.research is not o.git_policy.research,
            "#ts-retention": d.completed_retention is not o.completed_retention,
            "#ts-keep-awake": d.keep_awake != o.keep_awake,
        }
        for selector, is_changed in changed.items():
            widget = self.query_one(selector)
            widget.set_class(is_changed, "changed")
        dormant: set[str] = set()
        if d.task_isolation is not TaskIsolation.WORKTREE:
            dormant |= {"#ts-start-branch", "#ts-parallel", "#ts-removal"}
        if d.verification is Verification.NONE:
            dormant.add("#ts-git-tester")
        for selector in (
            "#ts-start-branch",
            "#ts-parallel",
            "#ts-removal",
            "#ts-git-tester",
        ):
            self.query_one(selector).set_class(selector in dormant, "ts-dormant")

    def _sync_hints(self) -> None:
        hints = task_settings_hints(
            self._draft, git_available=bool(self._git_available)
        )
        panel = self.query_one("#ts-hints", Vertical)
        self._fixes = {}
        rows: list[Static | Horizontal] = []
        for i, hint in enumerate(hints):
            children: list[Static | Button] = [
                Static(hint.text, classes="ts-hint-text")
            ]
            if hint.fix is not None:
                button_id = f"ts-fix-{i}"
                self._fixes[button_id] = hint.fix
                children.append(
                    Button(
                        hint.fix_label or "Fix",
                        id=button_id,
                        classes="ts-fix",
                    )
                )
            rows.append(Horizontal(*children, classes=f"ts-hint {hint.severity}"))
        panel.remove_children()
        if rows:
            panel.mount(*rows)
        else:
            panel.mount(
                Static(
                    "No conflicts — every setting takes effect.",
                    classes="ts-hint-empty",
                )
            )

    def _refresh(self) -> None:
        self._sync_preset()
        self._sync_fields()
        self._sync_hints()

    # -- widgets -> draft ------------------------------------------------------
    # Only genuine user edits reach these handlers: the programmatic
    # widget writes in _refresh are wrapped in self.prevent(...) so they
    # never post Changed messages at all (Textual delivers messages
    # asynchronously — an un-prevented sync write can come back as a
    # STALE event and re-enter its handler after the user has already
    # moved the widget on). The handlers keep the "does the event value
    # already equal the draft" guard as the second half of the loop
    # protection.

    @on(RadioSet.Changed, "#ts-preset")
    def _preset_changed(self, event: RadioSet.Changed) -> None:
        index = event.index
        matched = workflow_preset_for(self._draft)
        matched_index = (
            WORKFLOW_PRESETS.index(matched)
            if matched is not None
            else len(WORKFLOW_PRESETS)
        )
        if index == matched_index:
            # The pressed radio is already the truth (programmatic sync,
            # or a "Custom" pick on a draft matching no preset).
            return
        if index >= len(WORKFLOW_PRESETS):
            # "Custom" picked: a display state, never an applier.
            self._refresh()
            return
        WORKFLOW_PRESETS[index].apply(self._draft)
        self._refresh()

    @on(Select.Changed, "#ts-isolation")
    def _isolation_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, TaskIsolation) and (
            event.value is not self._draft.task_isolation
        ):
            self._draft.task_isolation = event.value
            self._refresh()

    @on(Select.Changed, "#ts-start-branch")
    def _start_branch_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, str) and (
            event.value != self._draft.isolation_start_branch
        ):
            self._draft.isolation_start_branch = event.value
            self._refresh()

    @on(Select.Changed, "#ts-parallel")
    def _parallel_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, int) and event.value != self._draft.parallel_tasks:
            self._draft.parallel_tasks = event.value
            self._refresh()

    @on(Select.Changed, "#ts-removal")
    def _removal_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, WorktreeRemoval) and (
            event.value is not self._draft.worktree_removal
        ):
            self._draft.worktree_removal = event.value
            self._refresh()

    @on(Select.Changed, "#ts-subagents-coder")
    def _subagents_coder_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, int) and (
            event.value != self._draft.subagents_coder
        ):
            self._draft.subagents_coder = event.value
            self._refresh()

    @on(Select.Changed, "#ts-subagents-researcher")
    def _subagents_researcher_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, int) and (
            event.value != self._draft.subagents_researcher
        ):
            self._draft.subagents_researcher = event.value
            self._refresh()

    @on(Select.Changed, "#ts-verification")
    def _verification_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, Verification) and (
            event.value is not self._draft.verification
        ):
            self._draft.verification = event.value
            self._refresh()

    @on(Select.Changed, "#ts-dev-rounds")
    def _dev_rounds_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, int) and event.value != self._draft.max_dev_rounds:
            self._draft.max_dev_rounds = event.value
            self._refresh()

    @on(Select.Changed, "#ts-nudge")
    def _nudge_changed(self, event: Select.Changed) -> None:
        if (
            isinstance(event.value, StallNudge)
            and event.value is not self._draft.stall_nudge
        ):
            self._draft.stall_nudge = event.value
            self._refresh()

    @on(Select.Changed, "#ts-git-tester")
    def _git_tester_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, GitPolicy) and (
            event.value is not self._draft.git_policy.tester
        ):
            self._draft.git_policy.tester = event.value
            self._refresh()

    @on(Select.Changed, "#ts-git-research")
    def _git_research_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, GitPolicy) and (
            event.value is not self._draft.git_policy.research
        ):
            self._draft.git_policy.research = event.value
            self._refresh()

    @on(Select.Changed, "#ts-retention")
    def _retention_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, CompletedRetention) and (
            event.value is not self._draft.completed_retention
        ):
            self._draft.completed_retention = event.value
            self._refresh()

    @on(Switch.Changed, "#ts-keep-awake")
    def _keep_awake_changed(self, event: Switch.Changed) -> None:
        if event.value != self._draft.keep_awake:
            self._draft.keep_awake = event.value
            self._refresh()

    @on(Button.Pressed, ".ts-fix")
    def _fix_pressed(self, event: Button.Pressed) -> None:
        fix = self._fixes.get(event.button.id or "")
        if fix is not None:
            fix(self._draft)
            self._refresh()

    # -- apply / cancel ----------------------------------------------------------

    @on(Button.Pressed, "#ts-apply")
    def _apply_pressed(self) -> None:
        self.dismiss(self._draft)

    @on(Button.Pressed, "#ts-cancel")
    def _cancel_pressed(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
