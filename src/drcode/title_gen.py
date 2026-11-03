"""Agent-generated task titles (project setting ``generate_task_titles``).

On a task's FIRST activation the pipeline asks the model serving the task
(the same resolved role config the run itself uses) to write a short title
from the task's description — before the main run starts, so the user sees
the new name on the active task from the very beginning. Kept apart from
the coding/tester agents exactly like the compaction summarizer: a
tool-less mini-agent that reads the description and returns one line.

Generation is exactly-once per task lifetime: the ``Task.title_generated``
stamp is set only on SUCCESS, so a failed attempt keeps the historical
first-line title and the NEXT activation may retry (one bounded attempt
per activation — never a loop within a run). Tasks that never reach the
agent (hold / draft / queued) keep their derived titles: the generator
runs only in the agent-run worker (see drcode.core.pipeline_agent_run).
"""

from __future__ import annotations

from pydantic_ai import Agent

from . import agent_model
from .agent_model import EXTRA_HEADERS
from .config import Config
from .prompts import load_title_prompt

# The stored-title cap: the same 60 characters TaskManager.create /
# edit_task derive titles with (title_display then truncates to 40 for the
# sidebar, so a 60-char title still survives storage round-trips).
TITLE_MAX_LEN = 60

# The generator only ever needs the shape of the task, not every detail:
# a description past this cap is cut (the head of the text carries the
# intent — and the model would otherwise spend the run's window on it).
_DESCRIPTION_CAP = 6_000


def build_title_agent(config: Config) -> Agent[None, str]:
    """Tool-less title generator over the task's serving role config.

    System prompt comes from the bundled ``title.txt`` template; the
    description is passed as a plain-text user prompt (see
    :func:`build_title_prompt`). No tools, no deps: the agent only reads
    the description and returns the title line.
    """
    return Agent(
        agent_model.build_model(config),
        instructions=load_title_prompt(),
        model_settings={
            "extra_headers": EXTRA_HEADERS,
            "max_tokens": config.max_tokens,
        },
    )


def build_title_prompt(description: str) -> str:
    """User prompt for the title generator: the (head-truncated) task
    description and the required output shape."""
    trimmed = description.strip()
    if len(trimmed) > _DESCRIPTION_CAP:
        trimmed = trimmed[:_DESCRIPTION_CAP] + "\n…[truncated]"
    return (
        "Below is the description of one software-development task.\n\n"
        "<task-description>\n"
        f"{trimmed}\n"
        "</task-description>\n\n"
        "Reply with the short title only: no quotes, no trailing period, "
        "no explanation."
    )


def clean_generated_title(raw: str | None) -> str | None:
    """Normalize a model reply into a storable title (None when empty).

    Collapses all whitespace (newlines/tabs included — the sidebar renders
    single lines only), strips the typical wrapper punctuation the models
    like to add (quotes, backticks, one trailing period), and caps the
    length at :data:`TITLE_MAX_LEN`.
    """
    if not raw:
        return None
    title = " ".join(raw.split())
    title = title.strip("\"'`“”‘’«» ")
    if title.endswith("."):
        title = title[:-1].rstrip()
    title = title.strip("\"'`“”‘’«» ")
    if not title:
        return None
    return title[:TITLE_MAX_LEN]
