"""run_bash tool-description wiring in the agent builders: the network
half of the description must match the run's actual bash network policy —
the project's `unrestricted` web-fetch mode lifts the guard, and the
model must be TOLD so (a description that still claims "external requests
are REFUSED" keeps the model from using the network it may use)."""

from __future__ import annotations

from pathlib import Path

import pytest

from drcode.agent import build_agent, build_resolver_agent, build_tester_agent
from drcode.config import Config
from drcode.models import TaskKind

RESTRICTED_SNIPPET = "external requests from bash are REFUSED"
UNRESTRICTED_SNIPPET = "external requests from bash are ALLOWED"


@pytest.fixture
def fake_config() -> Config:
    return Config(project_root=Path.cwd(), model="test-model", api_key="sk-test")


def _run_bash_desc(agent, name: str = "run_bash") -> str:
    for tool in agent._function_toolset.tools.values():
        if tool.tool_def.name == name:
            return tool.tool_def.description or ""
    raise AssertionError(f"tool {name} is not registered")


def test_restricted_by_default(fake_config: Config) -> None:
    """Without the flag (the default) every role's run_bash description
    says the bash network is loopback-only."""
    ag = build_agent(fake_config, kind=TaskKind.RESEARCH)
    assert RESTRICTED_SNIPPET in _run_bash_desc(ag)

    tester = build_tester_agent(fake_config)
    assert RESTRICTED_SNIPPET in _run_bash_desc(tester)

    resolver = build_resolver_agent(fake_config)
    assert RESTRICTED_SNIPPET in _run_bash_desc(resolver)


def test_unrestricted_promises_network_access(fake_config: Config) -> None:
    """With `network_unrestricted` the description stops claiming the
    local-only guard: any host and redirect flags are permitted — for the
    coding/research builder, the tester and the resolver alike."""
    for agent in (
        build_agent(fake_config, kind=TaskKind.RESEARCH, network_unrestricted=True),
        build_tester_agent(fake_config, network_unrestricted=True),
        build_resolver_agent(fake_config, network_unrestricted=True),
    ):
        desc = _run_bash_desc(agent)
        assert UNRESTRICTED_SNIPPET in desc
        assert "redirect-following flags are permitted" in desc
        assert RESTRICTED_SNIPPET not in desc


def test_unrestricted_research_keeps_web_fetch_hint(fake_config: Config) -> None:
    """In unrestricted mode the research agent's description still points
    at web_fetch for page reading — the guard is lifted, the managed tool
    stays the preferred way to read pages."""
    ag = build_agent(fake_config, kind=TaskKind.RESEARCH, network_unrestricted=True)
    assert "web_fetch" in _run_bash_desc(ag)
