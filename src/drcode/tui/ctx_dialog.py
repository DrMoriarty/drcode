"""Modal dialog: the ≈ context layout of the viewed task.

Opened by clicking the context counter in the task status bar (the 🧠
token figure at its right corner). Shows, for BOTH transcripts of the task
— development (`task.messages`) and the latest testing round
(`task.tester_messages[-1]`) — the estimated breakdown by section
(system prompt parts, tool descriptions, transcript parts) plus the exact
figures the API reported (last context fill, cache split, whole-task
cumulative usage).

Every estimated figure carries "≈": there is no tokenizer (the model is any
OpenAI-compatible endpoint), the estimate is chars/4 — and its sum is NOT
normalized to 100% and will not match the exact usage.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic_ai.usage import RunUsage
from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from ..context_layout import CHARS_PER_TOKEN, CtxReport, format_tokens


def _pct(tokens: int | None, max_tokens: int) -> str:
    if max_tokens <= 0 or tokens is None:
        return ""
    return f" · {round(tokens * 100 / max_tokens)}%"


def _report_lines(report: CtxReport) -> list[str]:
    """Render one report as markup lines."""
    max_tokens = report.max_context_tokens
    out: list[str] = [
        (
            f"[bold $accent]{escape(report.title)}[/]"
            f" [dim]model: {escape(report.model) or '—'}[/]"
        )
    ]
    if report.exact_tokens is not None:
        exact = (
            f"exact (API): {format_tokens(report.exact_tokens)} tok"
            f"{_pct(report.exact_tokens, max_tokens)} of "
            f"{format_tokens(max_tokens)}"
        )
        headroom = report.headroom_tokens
        if headroom is not None:
            exact += (
                f" · headroom to compaction {report.threshold_pct}% "
                f"({format_tokens(report.threshold_tokens or 0)}): "
                f"{format_tokens(headroom)}"
            )
        out.append(f"  {escape(exact)}")
    else:
        out.append("  [dim]exact (API): no model request yet — estimate only[/]")
    if report.last_cache_read or report.last_cache_write:
        out.append(
            "  last request cache: read "
            f"{format_tokens(report.last_cache_read)} / write "
            f"{format_tokens(report.last_cache_write)}"
        )
    estimate = (
        f"estimate: ≈{format_tokens(report.estimate_total)} tok"
        f"{_pct(report.estimate_total, max_tokens)}"
        f" — chars/{CHARS_PER_TOKEN} heuristic, no tokenizer; the sum of"
        " sections will NOT match the exact usage"
    )
    out.append(f"  [dim italic]{escape(estimate)}[/]")
    for section in report.sections:
        out.append(
            f"  [bold]{escape(section.title)}[/]"
            f" [dim]≈{format_tokens(section.total)}{_pct(section.total, max_tokens)}[/]"
        )
        for line in section.lines:
            if line.note == "absent":
                out.append(f"    [dim]{escape(line.label)} — absent[/]")
                continue
            row = (
                f"    {escape(line.label)} [dim]≈{format_tokens(line.tokens)}"
                f"{_pct(line.tokens, max_tokens)}[/]"
            )
            if line.note:
                row += f" [dim italic]({escape(line.note)})[/]"
            out.append(row)
    if report.heaviest:
        out.append("  [bold]top-5 heaviest tool results[/]")
        for result in report.heaviest:
            out.append(
                f"    {escape(result.label)} [dim]≈{format_tokens(result.tokens)}[/]"
            )
    service: list[str] = []
    if report.compacted:
        service.append("AFTER COMPACTION (transcript rewritten)")
    service.append(f"{report.message_count} message(s)")
    service.append(f"{report.tool_calls} tool call(s)")
    service.append(f"max_tokens {format_tokens(report.max_output_tokens)}")
    out.append(f"  [dim]{escape(' · '.join(service))}[/]")
    return out


class ContextDialog(ModalScreen[None]):
    """Read-only modal: the ≈ context layout of the viewed task. ESC or the
    Close button dismisses; nothing in it is interactive beyond that."""

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Close", priority=True),
    ]

    DEFAULT_CSS = """
    ContextDialog {
        align: center middle;
    }
    #ctx-dialog {
        width: 84;
        max-width: 95%;
        height: 32;
        max-height: 90%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    #ctx-scroll { height: 1fr; }
    #ctx-body { width: 1fr; }
    #ctx-close-row {
        height: auto;
        margin-top: 1;
        align-horizontal: right;
    }
    """

    def __init__(
        self,
        reports: list[CtxReport],
        cumulative: RunUsage,
        requests: int,
    ) -> None:
        super().__init__()
        self._reports = reports
        self._cumulative = cumulative
        self._requests = requests

    def compose(self) -> ComposeResult:
        with Vertical(id="ctx-dialog"):
            yield VerticalScroll(
                Static(self._body_text(), id="ctx-body"), id="ctx-scroll"
            )
            with Vertical(id="ctx-close-row"):
                yield Static("[dim]esc — close[/]")

    def _body_text(self) -> str:
        lines: list[str] = [
            (
                "[bold]Task context layout[/] "
                f"[dim](≈ estimates: chars/{CHARS_PER_TOKEN}, no tokenizer — "
                "arbitrary OpenAI-compatible model)[/]"
            )
        ]
        for report in self._reports:
            lines.append("")
            lines.extend(_report_lines(report))
        if self._requests:
            u = self._cumulative
            parts: list[str] = [f"in {format_tokens(u.input_tokens)}"]
            if u.cache_read_tokens or u.cache_write_tokens:
                parts.append(
                    f"cache {format_tokens(u.cache_read_tokens)}"
                    f"/{format_tokens(u.cache_write_tokens)}"
                )
            parts.append(f"out {format_tokens(u.output_tokens)}")
            parts.append(f"{self._requests} request(s)")
            lines.append("")
            lines.append(
                f"[bold]Whole task[/] [dim](exact, all transcripts): "
                f"{escape(' · '.join(parts))}[/]"
            )
        return "\n".join(lines)

    def action_close(self) -> None:
        self.dismiss(None)
