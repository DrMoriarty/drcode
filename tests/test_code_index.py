"""Tests for the symbolic code index (drcode.code_index, phase 1).

Modeled on tests/test_task_search.py: incremental re-parsing by stat
keys, corrupted-index rebuild, stale-entry self-heal, the role gating
of the code_* tools and the "three accounting mirrors agree" invariant
(docs/code-index-design.md §8.2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from drcode.agent_build import (
    build_agent,
    build_resolver_agent,
    build_tester_agent,
    code_index_tools_for,
)
from drcode.code_index import queries
from drcode.code_index.store import (
    clear_code_index_cache,
    load_index,
    take_build_notice,
)
from drcode.config import Config
from drcode.models import TaskKind
from drcode.providers import AgentRole
from drcode.settings import CodeIndexMode, ProjectSettings
from drcode.subagents import build_subagent_agent

# The grammar for the tests' language (python) may need a one-time
# on-demand download (dev runs only — the frozen bundle prefetches it);
# when neither cache nor network is available the extraction tests skip
# honestly instead of failing.
try:
    from tree_sitter_language_pack import get_parser

    get_parser("python")
    _PYTHON_READY = True
except Exception:  # noqa: BLE001 - offline clean machine
    _PYTHON_READY = False

_PY_SAMPLE = '''"""Sample module."""
import os


class Widget:
    """A widget."""

    def render(self, indent: int) -> str:
        return " " * indent


def render_all(widgets, indent=0):
    return "\\n".join(w.render(indent) for w in widgets)
'''


@pytest.fixture(autouse=True)
def _fresh_index_cache():
    """Every test starts with a cold in-memory cache: the store is
    process-global (one entry per project root), the conftest tmp roots
    differ per test but the notices/caches must never leak."""
    clear_code_index_cache()
    yield
    clear_code_index_cache()


@pytest.fixture
def py_project(project_root: Path) -> Path:
    (project_root / "sample.py").write_text(_PY_SAMPLE)
    return project_root


def _counting_extractor(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Spy on the extraction path: returns the list of relpaths the
    extractor actually parsed (the incremental contract: only changed
    files pay the parse)."""
    from drcode.code_index import store as store_mod
    from drcode.code_index.extract import SymbolExtractor

    calls: list[str] = []
    original = store_mod._DEFAULT_EXTRACTOR.extract

    def spy_extract(path: Path, text: str) -> Any:
        calls.append(path.name)
        return original(path, text)

    monkeypatch.setattr(store_mod._DEFAULT_EXTRACTOR, "extract", spy_extract)
    assert isinstance(store_mod._DEFAULT_EXTRACTOR, SymbolExtractor)
    return calls


@pytest.mark.skipif(not _PYTHON_READY, reason="python grammar unavailable")
class TestStore:
    def test_extract_and_incremental_reparse(
        self, py_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _counting_extractor(monkeypatch)
        entries, info = load_index(py_project)
        assert sorted(calls) == ["sample.py"]
        assert info.built_now and info.files == 1
        record = entries["sample.py"]
        assert record.language == "python" and not record.partial
        names = {(s.scope, s.name, s.kind) for s in record.symbols}
        assert ("", "Widget", "class") in names
        assert ("Widget", "render", "function") in names
        assert ("", "render_all", "function") in names
        # The imports (§4.2) are extracted too.
        assert [i.spec for i in record.imports] == ["os"]

        # A warm call re-parses NOTHING.
        calls.clear()
        entries, info = load_index(py_project)
        assert calls == [] and not info.built_now

        # Change ONE file (content + fresh mtime): exactly one re-parse.
        calls.clear()
        other = py_project / "other.py"
        other.write_text("def helper():\n    return 1\n")
        entries, _ = load_index(py_project)
        assert calls == ["other.py"]
        assert "other.py" in entries and "sample.py" in entries

        # A vanished file drops out without an error.
        (py_project / "other.py").unlink()
        entries, _ = load_index(py_project)
        assert "other.py" not in entries

    def test_index_persists_and_survives_a_restart(
        self, py_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        load_index(py_project)
        index_path = py_project / ".drcode" / "code_index.json"
        data = json.loads(index_path.read_text())
        assert data["version"] == 1
        assert data["generator"].startswith("drcode-")
        assert "sample.py" in data["entries"]

        # A fresh process (cache cleared) is served from the DISK index:
        # zero re-parses, warm cache rebuilt.
        calls = _counting_extractor(monkeypatch)
        clear_code_index_cache()
        entries, info = load_index(py_project)
        assert calls == [] and not info.built_now
        assert "sample.py" in entries

    def test_corrupted_index_is_rebuilt(self, py_project: Path) -> None:
        load_index(py_project)
        index_path = py_project / ".drcode" / "code_index.json"
        index_path.write_text("{corrupted index")
        clear_code_index_cache()
        entries, _ = load_index(py_project)
        assert "sample.py" in entries
        data = json.loads(index_path.read_text())
        assert "sample.py" in data["entries"]

    def test_foreign_index_shape_is_rebuilt(self, py_project: Path) -> None:
        index_path = py_project / ".drcode" / "code_index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps({"version": 99, "entries": {}}))
        entries, _ = load_index(py_project)
        assert "sample.py" in entries

    def test_stale_index_entry_self_heals(self, py_project: Path) -> None:
        load_index(py_project)
        index_path = py_project / ".drcode" / "code_index.json"
        data = json.loads(index_path.read_text())
        entry_name = next(iter(data["entries"]))
        data["entries"][entry_name][0] = [1, 1]  # corrupt the stat key
        index_path.write_text(json.dumps(data))
        clear_code_index_cache()
        entries, _ = load_index(py_project)
        assert entry_name in entries
        fixed = json.loads(index_path.read_text())
        stat = (py_project / entry_name).stat()
        assert fixed["entries"][entry_name][0] == [stat.st_mtime_ns, stat.st_size]

    def test_worktree_index_never_touches_the_disk(self, tmp_path: Path) -> None:
        """§7.1/§7.3: a worktree-isolated task indexes ITS tree but must
        never create `.drcode/code_index.json` inside the worktree
        (persist=False — the in-memory cache is the overlay)."""
        (tmp_path / "wt.py").write_text("def wt_only():\n    pass\n")
        entries, _ = load_index(tmp_path, persist=False)
        assert "wt.py" in entries
        assert not (tmp_path / ".drcode" / "code_index.json").exists()

    def test_first_build_notice_is_consumed_once(self, py_project: Path) -> None:
        _entries, info = load_index(py_project)
        assert info.built_now
        notice = take_build_notice(py_project)
        assert notice is not None and notice.startswith("code index:")
        assert take_build_notice(py_project) is None

    def test_disk_served_restart_is_not_a_build(self, py_project: Path) -> None:
        """A fresh process served entirely from the on-disk index did no
        parsing — no build event (§8.3: the event marks a BUILD)."""
        load_index(py_project)
        clear_code_index_cache()
        _entries, info = load_index(py_project)
        assert not info.built_now
        assert take_build_notice(py_project) is None


@pytest.mark.skipif(not _PYTHON_READY, reason="python grammar unavailable")
class TestQueries:
    def test_find_symbols_ranking_and_kind(self, py_project: Path) -> None:
        text = queries.find_symbols(py_project, "render")
        assert "sample.py:8" in text  # Widget.render line
        assert "sample.py:12" in text  # render_all line
        assert "Widget.render" in text and "render_all" in text
        # kind filter and unknown kind
        class_only = queries.find_symbols(py_project, "widget", kind="class")
        assert "Widget" in class_only and "render_all" not in class_only
        assert "Unknown kind" in queries.find_symbols(py_project, "x", kind="nope")
        # Empty result → the honest grep hint (§6.3).
        assert "grep_search" in queries.find_symbols(py_project, "definitely_not_here")

    def test_outline_skeleton_and_neighbourhood(self, py_project: Path) -> None:
        skeleton = queries.outline(py_project, "sample.py")
        assert "3 definitions" in skeleton
        assert "Widget" in skeleton and "render_all" in skeleton
        # `around` — the same-scope neighbourhood; span adds the body.
        nbhd = queries.outline(py_project, "sample.py", around="render", span=5)
        assert "neighbourhood of render" in nbhd
        assert "body of render" in nbhd
        assert '" " * indent' in nbhd
        # A file outside the index is named honestly.
        assert "not in the index" in queries.outline(py_project, "nope.rs")

    def test_outline_stale_body_self_heals(
        self, py_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§6.4: a record whose line range no longer fits the file is
        re-parsed on the query that proves it stale — the answer comes
        from the fresh record, never from the stale range."""
        load_index(py_project)
        calls = _counting_extractor(monkeypatch)
        # Inject a stale record into the warm in-memory cache (stat key
        # untouched — the file on disk is still "current" as far as the
        # store knows): the recorded range points past the file's end.
        from drcode.code_index import store as store_mod
        from drcode.code_index.extract import FileIndex, Symbol

        cached = store_mod._INDEX_CACHES[str(py_project)]
        cached.entries["sample.py"] = FileIndex(
            language="python",
            partial=False,
            symbols=[
                Symbol(
                    name="render",
                    kind="function",
                    scope="Widget",
                    signature="def render(self):",
                    line=9999,
                    end_line=10005,
                    doc="",
                )
            ],
        )
        calls.clear()
        result = queries.outline(py_project, "sample.py", around="render", span=4)
        # The forced re-parse ran and the healed answer carries the REAL
        # body from disk, not the stale range.
        assert "sample.py" in calls
        assert "body of render" in result
        assert '" " * indent' in result


class TestRoleGating:
    def test_helper_gates_by_role_and_mode(self) -> None:
        off = ProjectSettings()
        assert off.code_index_mode is CodeIndexMode.OFF
        for role in (
            AgentRole.CODING,
            AgentRole.TESTER,
            AgentRole.RESOLVER,
            AgentRole.RESEARCH,
        ):
            assert code_index_tools_for(off, role) is False
        on = ProjectSettings(code_index_mode=CodeIndexMode.ON)
        for role in (
            AgentRole.CODING,
            AgentRole.TESTER,
            AgentRole.RESOLVER,
            AgentRole.RESEARCH,
        ):
            assert code_index_tools_for(on, role) is True

    @pytest.fixture
    def fake_config(self) -> Config:
        return Config(project_root=Path.cwd(), model="test-model", api_key="sk-test")

    @staticmethod
    def _tool_names(agent: Any) -> set[str]:
        return {t.name for t in agent._function_toolset.tools.values()}

    def test_off_registers_nothing(
        self, fake_config: Config, project_root: Path
    ) -> None:
        coding = build_agent(fake_config, kind=TaskKind.FEATURE, root=project_root)
        research = build_agent(fake_config, kind=TaskKind.RESEARCH, root=project_root)
        tester = build_tester_agent(fake_config, root=project_root)
        resolver = build_resolver_agent(fake_config, root=project_root)
        subagent = build_subagent_agent(fake_config, root=project_root)
        for agent in (coding, research, tester, resolver, subagent):
            names = self._tool_names(agent)
            assert "code_symbols" not in names and "code_outline" not in names

    def test_on_registers_for_coding_tester_resolver_research_subagent(
        self, fake_config: Config, project_root: Path
    ) -> None:
        coding = build_agent(
            fake_config, kind=TaskKind.FEATURE, root=project_root, code_index=True
        )
        tester = build_tester_agent(fake_config, root=project_root, code_index=True)
        resolver = build_resolver_agent(fake_config, root=project_root, code_index=True)
        subagent = build_subagent_agent(fake_config, root=project_root, code_index=True)
        for agent in (coding, tester, resolver, subagent):
            names = self._tool_names(agent)
            assert {
                "code_symbols",
                "code_outline",
                "code_refs",
                "code_callers",
            } <= names
        # The RESEARCH kind gets the tools on the same terms as coding —
        # its elementary lookups no longer force a delegate_research slot.
        research = build_agent(
            fake_config, kind=TaskKind.RESEARCH, root=project_root, code_index=True
        )
        names = self._tool_names(research)
        assert {
            "code_symbols",
            "code_outline",
            "code_refs",
            "code_callers",
        } <= names

    def test_tool_descriptions_are_assigned(
        self, fake_config: Config, project_root: Path
    ) -> None:
        """The web_fetch precedent: the docstring must be ASSIGNED (an
        inline f-string/absent doc would register an empty description)."""
        agent = build_agent(
            fake_config, kind=TaskKind.FEATURE, root=project_root, code_index=True
        )
        docs = {
            t.name: t.tool_def.description or ""
            for t in agent._function_toolset.tools.values()
        }
        assert "DEFINITIONS" in docs["code_symbols"]
        assert "grep_search" in docs["code_symbols"]  # the §6.3 fallback hint
        assert "WITHOUT reading it whole" in docs["code_outline"]


class TestMirrorsAgree:
    """§8.2's invariant: the counted tool schemas (the mirrors) show the
    tools exactly when the real wiring registers them."""

    @pytest.fixture
    def fake_config(self) -> Config:
        return Config(project_root=Path.cwd(), model="test-model", api_key="sk-test")

    def test_coding_schemas_match_real_wiring(
        self, fake_config: Config, project_root: Path
    ) -> None:
        from drcode.context_layout import coding_tool_schemas

        real = {
            t.name
            for t in build_agent(
                fake_config, kind=TaskKind.FEATURE, root=project_root, code_index=True
            )._function_toolset.tools.values()
        }
        counted = {
            name: text
            for name, text in coding_tool_schemas(
                fake_config,
                TaskKind.FEATURE,
                root=project_root,
                code_index=True,
            )
        }
        assert set(counted) == real
        assert "code_symbols" in counted and "code_outline" in counted
        assert "DEFINITIONS" in counted["code_symbols"]
        # OFF: neither the wiring nor the estimate knows the tools.
        off_real = build_agent(fake_config, kind=TaskKind.FEATURE, root=project_root)
        off_counted = dict(
            coding_tool_schemas(fake_config, TaskKind.FEATURE, root=project_root)
        )
        assert "code_symbols" not in {
            t.name for t in off_real._function_toolset.tools.values()
        }
        assert "code_symbols" not in off_counted

    def test_tester_and_resolver_schemas_match(
        self, fake_config: Config, project_root: Path
    ) -> None:
        from drcode.context_layout import (
            resolver_tool_schemas,
            tester_tool_schemas,
        )

        tester_counted = dict(
            tester_tool_schemas(fake_config, root=project_root, code_index=True)
        )
        tester_real = {
            t.name
            for t in build_tester_agent(
                fake_config, root=project_root, code_index=True
            )._function_toolset.tools.values()
        }
        assert set(tester_counted) == tester_real
        assert "code_symbols" in tester_counted

        resolver_counted = dict(
            resolver_tool_schemas(fake_config, root=project_root, code_index=True)
        )
        resolver_real = {
            t.name
            for t in build_resolver_agent(
                fake_config, root=project_root, code_index=True
            )._function_toolset.tools.values()
        }
        assert set(resolver_counted) == resolver_real
        assert "code_outline" in resolver_counted

    def test_research_schemas_match_real_wiring(
        self, fake_config: Config, project_root: Path
    ) -> None:
        """RESEARCH joins the gated roles: the counted schemas (mirror 1)
        must show the tools exactly when the research run registers them
        (the pipeline resolves the flag per role via code_index_tools_for)."""
        from drcode.context_layout import coding_tool_schemas

        real = {
            t.name
            for t in build_agent(
                fake_config,
                kind=TaskKind.RESEARCH,
                root=project_root,
                code_index=True,
            )._function_toolset.tools.values()
        }
        counted = {
            name: text
            for name, text in coding_tool_schemas(
                fake_config,
                TaskKind.RESEARCH,
                root=project_root,
                code_index=True,
            )
        }
        assert set(counted) == real
        assert {
            "code_symbols",
            "code_outline",
            "code_refs",
            "code_callers",
        } <= counted.keys()
        # OFF: neither side knows the tools.
        off_real = build_agent(fake_config, kind=TaskKind.RESEARCH, root=project_root)
        off_counted = dict(
            coding_tool_schemas(fake_config, TaskKind.RESEARCH, root=project_root)
        )
        assert "code_symbols" not in {
            t.name for t in off_real._function_toolset.tools.values()
        }
        assert "code_symbols" not in off_counted

    def test_task_reports_forward_the_mirror(
        self, fake_config: Config, project_root: Path
    ) -> None:
        """The context dialog's task_reports must count the schemas the
        moment the tools are registered (mirror 2's path)."""
        from drcode.context_layout import task_reports
        from drcode.models import Task

        task = Task(title="t", kind=TaskKind.FEATURE)
        task.messages = [{"parts": [{"part_kind": "user-prompt", "content": "do it"}]}]
        task.tester_messages = [[{"parts": [{"part_kind": "text", "content": "ok"}]}]]
        _dev, tester, _resolver, _usage, _req = task_reports(
            task,
            fake_config,
            fake_config,
            fake_config,
            "CODING",
            project_root,
            code_index=True,
        )
        assert tester is not None
        counted_names = {
            line.label for section in tester.sections for line in section.lines
        }
        assert "code_symbols" in counted_names
        assert "code_outline" in counted_names


class TestSettings:
    def test_default_off_and_options(self, tmp_path: Path) -> None:
        from drcode.settings import CODE_INDEX_MODE_OPTIONS, ProjectSettingsStore

        assert ProjectSettings().code_index_mode is CodeIndexMode.OFF
        assert CODE_INDEX_MODE_OPTIONS == (CodeIndexMode.OFF, CodeIndexMode.ON)
        assert CodeIndexMode.ON.label == "on"
        assert CodeIndexMode.OFF.description
        # Round-trips through the store like any other preference; a
        # bogus value in settings.json degrades to the default (never
        # invalidates the whole file).
        store = ProjectSettingsStore(tmp_path)
        store.save(ProjectSettings(code_index_mode=CodeIndexMode.ON))
        assert store.load().code_index_mode is CodeIndexMode.ON
        store.path.write_text(json.dumps({"code_index_mode": "bogus"}))
        assert store.load().code_index_mode is CodeIndexMode.OFF


# ---------------------------------------------------------------------------
# Phase 2: refs / callers / import graph (§4.2, §4.3, §5.2, §5.3), the
# ctags coverage adapter (§2.2) and the §6.4 HEAD guard.
# ---------------------------------------------------------------------------

_CALLS_SAMPLE_APP = """import helpers


def run_app():
    helpers.setup()
    return setup()
"""

_CALLS_SAMPLE_HELPERS = """COUNT = 1


def setup():
    return COUNT
"""

_CALLS_SAMPLE_LOOSE = """def loose_caller():
    return setup() + 1
"""


def _calls_project(project_root: Path) -> Path:
    (project_root / "app.py").write_text(_CALLS_SAMPLE_APP)
    (project_root / "helpers.py").write_text(_CALLS_SAMPLE_HELPERS)
    (project_root / "loose.py").write_text(_CALLS_SAMPLE_LOOSE)
    return project_root


@pytest.mark.skipif(not _PYTHON_READY, reason="python grammar unavailable")
class TestRefs:
    def test_refs_groups_and_marks_definitions(self, project_root: Path) -> None:
        _calls_project(project_root)
        out = queries.refs(project_root, "setup", limit=50)
        assert "textual" in out  # the honest noise caveat (§5.2)
        assert "[def]" in out
        assert "app.py:" in out and "helpers.py:" in out
        hits = queries.ref_hits(project_root, "setup")
        by_file = {h.rel for h in hits}
        assert by_file == {"app.py", "helpers.py", "loose.py"}
        defs = [h for h in hits if h.is_def]
        assert {(h.rel, h.line) for h in defs} == {("helpers.py", 4)}
        calls = [h for h in hits if h.is_call and not h.is_def]
        assert {(h.rel, h.line) for h in calls} == {
            ("app.py", 5),
            ("app.py", 6),
            ("loose.py", 2),
        }
        # Scope prefixes (`helpers.setup`) stay textual hits in refs but
        # are NOT calls of `helpers`.
        helper_hits = [h for h in queries.ref_hits(project_root, "helpers")]
        assert helper_hits and not any(h.is_call for h in helper_hits)

    def test_refs_empty_and_capped(self, project_root: Path) -> None:
        _calls_project(project_root)
        assert "grep_search" in queries.refs(project_root, "no_such_name_here")
        out = queries.refs(project_root, "setup", limit=1)
        assert "more hit(s)" in out


@pytest.mark.skipif(not _PYTHON_READY, reason="python grammar unavailable")
class TestCallers:
    def test_callers_found_ranked_and_labeled(self, project_root: Path) -> None:
        _calls_project(project_root)
        out = queries.callers(project_root, "setup", limit=10)
        assert out.startswith("callers are INFERRED")  # the honest label
        assert "defined at: helpers.py:4" in out
        assert "run_app (app.py:" in out
        # Import proximity ranking: the importer of helpers.py outranks
        # the non-importing loose caller.
        assert out.index("run_app") < out.index("loose_caller")

    def test_scope_prefix_and_nested_def_are_not_edges(
        self, project_root: Path
    ) -> None:
        _calls_project(project_root)
        # `helpers.setup()` creates the edge for `setup`, never for
        # `helpers` (§4.3 step 4) — and `helpers` has no definition hits
        # in any body here at all.
        assert queries.caller_hits(project_root, "helpers") == []
        hits = queries.caller_hits(project_root, "setup")
        assert hits, "the direct `setup()` call must be found"
        # Definition lines never come back as caller hits.
        assert all((h.caller_rel, h.context_line) != ("helpers.py", 4) for h in hits)

    def test_no_definition_still_reports_mentions(self, project_root: Path) -> None:
        (project_root / "ghost.py").write_text("def user():\n    return ghost_fn()\n")
        out = queries.callers(project_root, "ghost_fn")
        assert "no indexed definition" in out
        assert "user (ghost.py:" in out

    def test_structured_hits_carry_ranking_signals(self, project_root: Path) -> None:
        _calls_project(project_root)
        hits = queries.caller_hits(project_root, "setup")
        top = hits[0]
        assert top.caller_name == "run_app"
        assert top.is_call
        assert top.import_distance == 1  # app.py imports helpers.py
        assert top.score > 0


@pytest.mark.skipif(not _PYTHON_READY, reason="python grammar unavailable")
class TestImportGraph:
    def test_family_resolution_and_hops(self, project_root: Path) -> None:
        _calls_project(project_root)
        (project_root / "chain.py").write_text(
            "import app\n\n\ndef go():\n    return 1\n"
        )
        entries, _ = load_index(project_root)
        graph = queries.ImportGraph(entries)
        importers = graph.importers_of("helpers.py")
        assert importers.get("app.py") == 1
        assert importers.get("chain.py") == 2  # importer of an importer

    def test_python_suffix_resolution_with_src_prefix(self, project_root: Path) -> None:
        pkg = project_root / "src" / "mypkg"
        pkg.mkdir(parents=True)
        (pkg / "x.py").write_text("VALUE = 1\n")
        (project_root / "src" / "main.py").write_text(
            "from mypkg.x import VALUE\n\n\ndef main():\n    return VALUE\n"
        )
        entries, _ = load_index(project_root)
        graph = queries.ImportGraph(entries)
        assert graph.importers_of("src/mypkg/x.py").get("src/main.py") == 1


class TestCtagsAdapter:
    """The §2.2 coverage adapter, exercised through a stub u-ctags
    binary (real universal-ctags is optional and never required)."""

    @staticmethod
    def _stub_binary(tmp_path: Path) -> Path:
        stub = tmp_path / "ctags-stub"
        stub.write_text(
            """#!/bin/sh
case "$1" in
  --version) echo "Universal Ctags 6.1.0(p5.9.20240616.0)";;
  --print-language)
    case "$2" in
      *.nim) echo "Nim";;
      *) echo "unknown";;
    esac;;
  *)
    echo '{"_type":"tag","name":"do_stuff","kind":"function","line":3,"end":7,"signature":"proc do_stuff(a: int)"}'
    echo '{"_type":"tag","name":"Helper","kind":"class","line":10}'
    echo '{"_type":"ptag"}'
    echo '{"_type":"tag","name":"localvar","kind":"local","line":12}'
    ;;
esac
"""
        )
        stub.chmod(0o755)
        return stub

    def test_kind_mapping_and_scope_of_unknowns(self, tmp_path: Path) -> None:
        from drcode.code_index.ctags_adapter import CtagsExtractor

        ext = CtagsExtractor(binary=str(self._stub_binary(tmp_path)))
        source = tmp_path / "m.nim"
        source.write_text(
            "proc do_stuff(a: int) =\n"
            "  discard\n"
            "\n"
            "# padding\n"
            "# padding\n"
            "# padding\n"
            "# padding\n"
            "# padding\n"
            "# padding\n"
            "type Helper = object\n"
        )
        assert ext.language_for(source) == "Nim"
        record = ext.extract(source, source.read_text())
        assert record is not None and record.language == "nim"
        assert [(s.name, s.kind) for s in record.symbols] == [
            ("do_stuff", "function"),
            ("Helper", "class"),  # no `end` field → end_line = line
        ]
        assert record.symbols[0].end_line == 7
        assert record.symbols[0].signature == "proc do_stuff(a: int)"
        # No signature field → the definition's own file line (§3.2).
        assert record.symbols[1].signature == "type Helper = object"
        # The un-mappable `local` kind is dropped (§2.3 scope).
        assert all(s.kind != "local" for s in record.symbols)

    def test_never_touches_tree_sitter_or_non_source_extensions(
        self, tmp_path: Path
    ) -> None:
        from drcode.code_index.ctags_adapter import CtagsExtractor

        ext = CtagsExtractor(binary=str(self._stub_binary(tmp_path)))
        assert ext.language_for(tmp_path / "a.py") is None
        assert ext.language_for(tmp_path / "README.md") is None
        assert ext.language_for(tmp_path / "data.json") is None

    def test_legacy_bsd_ctags_is_rejected(self) -> None:
        # The machine's real ctags (BSD on macOS, if present) must not be
        # mistaken for universal-ctags: an incapable binary means "no
        # ctags coverage at all", never a crash.
        import shutil

        from drcode.code_index.ctags_adapter import CtagsExtractor

        binary = shutil.which("ctags")
        if binary is None:
            pytest.skip("no ctags on PATH")
        ext = CtagsExtractor(binary=binary)
        if not ext._is_capable():
            assert ext.language_for(Path("m.nim")) is None
        # Either way: a capable binary answers, an incapable one refuses.

    def test_store_falls_back_to_ctags_for_unknown_extensions(
        self, project_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from drcode.code_index import store as store_mod
        from drcode.code_index.ctags_adapter import CtagsExtractor

        monkeypatch.setattr(
            store_mod,
            "_FALLBACK_EXTRACTOR",
            CtagsExtractor(binary=str(self._stub_binary(project_root))),
        )
        (project_root / "app.nim").write_text("proc main() =\n  discard\n")
        # .rst is neither tree-sitter's business nor a probed ctags
        # extension (docs are excluded from probing on purpose).
        (project_root / "notes.rst").write_text("hi\n====\n")
        entries, info = load_index(project_root, persist=False)
        assert info.files == 1
        record = entries["app.nim"]
        assert record.language == "nim"
        assert [s.name for s in record.symbols] == ["do_stuff", "Helper"]
        assert "notes.rst" not in entries


class TestHeadGuard:
    """§6.4: the persisted index records the main checkout's HEAD; a
    changed HEAD invalidates the whole base layer (a branch switch that
    happened while the process was down is invisible to stat keys)."""

    def _git(self, *args: str) -> None:
        import subprocess

        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
            check=True,
            capture_output=True,
        )

    def test_head_change_rebuilds_the_base_layer(self, project_root: Path) -> None:
        from drcode.code_index.store import read_main_head

        (project_root / "a.py").write_text("def f():\n    return 1\n")
        self._git("init", "-q", str(project_root))
        self._git("-C", str(project_root), "add", ".")
        self._git("-C", str(project_root), "commit", "-qm", "one")
        _entries, info = load_index(project_root)
        assert info.built_now
        head = read_main_head(project_root)
        assert head and store_head(project_root) == head

        # A branch switch to the SAME commit keeps the same HEAD sha —
        # no rebuild (the guard is about content, not branch names).
        self._git("-C", str(project_root), "checkout", "-qb", "other")
        clear_code_index_cache()
        _, warm = load_index(project_root)
        assert not warm.built_now

        # A real HEAD change (new commit) drops the base layer.
        (project_root / "a.py").write_text("def f():\n    return 2\n")
        self._git("-C", str(project_root), "add", ".")
        self._git("-C", str(project_root), "commit", "-qm", "two")
        assert read_main_head(project_root) != head
        clear_code_index_cache()
        _, rebuilt = load_index(project_root)
        assert rebuilt.built_now
        assert store_head(project_root) == read_main_head(project_root)

        # Not a git checkout → the guard stays silent.
        plain = project_root.parent / "plain"
        plain.mkdir()
        assert read_main_head(plain) is None


def store_head(project_root: Path) -> str | None:
    from drcode.code_index import store as store_mod

    return store_mod._load_disk_index(project_root)[1]


class TestSubagentPromptNote:
    def test_note_only_when_enabled(self, project_root: Path) -> None:
        from drcode.subagents import _subagent_instructions

        off = _subagent_instructions(project_root, None, False)
        assert "code_callers" not in off
        assert "{code_index_note}" not in off
        on = _subagent_instructions(project_root, None, True)
        assert "code_callers" in on and "code_symbols" in on
        assert "{" not in on.replace("{project_path}", "")
