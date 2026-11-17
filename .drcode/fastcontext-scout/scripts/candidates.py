#!/usr/bin/env python3
"""fastcontext-scout, этап 2: общий интерфейс кандидатов + детерминированные бейслайны.

Интерфейс: query (str) -> список кандидатов
    {"path": str, "start_line": int, "end_line": int, "score": float, "source": str}
отсортированных по убыванию score. Путь — относительно корня исходников (src/).

Режимы (бейслайны):
    rg              — ripgrep по ключевым словам, извлечённым из запроса
    bm25            — Okapi BM25 по чанкам (функции/классы/шапка модуля)
    rg+bm25         — reciprocal rank fusion двух лексических бейслайнов
    bm25+struct     — bm25 + структурный буст по графу импортов
    rg+bm25+struct  — rrf + структурный буст

Чанкинг: одна верхнеуровневая сущность (def/класс) = один чанк, плюс
"шапка модуля" (docstring + импорты + код верхнего уровня до первого def).
Планировался tree-sitter, но сеть недоступна и пакета нет в кэше — чанкер
реализован на stdlib `ast` (индексируем только python, для src/drcode этого
достаточно). Точка замены — функция `chunk_file` (см. TODO(tree-sitter)).

CLI:
    python3 scripts/candidates.py --mode bm25 --query "..." [--limit 10]
    python3 scripts/candidates.py --stats

Без внешних зависимостей (python 3.9+), всё локально, без сети.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import subprocess
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------- чанкинг

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CYR_RE = re.compile(r"[А-Яа-яЁё]{4,}")


@dataclass
class Chunk:
    path: str          # rel path от корня исходников
    start_line: int    # 1-based, включительно
    end_line: int      # 1-based, включительно
    name: str          # "сигнатура" чанка (name/шапка)
    text: str          # исходник чанка (без строки-заголовка)

    @property
    def header(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}, {self.name}"

    @property
    def tokens(self) -> int:
        """≈токены по эвристике chars/4 (заголовок + тело)."""
        return (len(self.header) + len(self.text)) // 4


def _node_start(node: ast.AST) -> int:
    """Строка начала узла с учётом декораторов."""
    start = node.lineno
    for dec in getattr(node, "decorator_list", []):
        start = min(start, dec.lineno)
    return start


def _signature(node: ast.AST, lines: List[str]) -> str:
    """Первая физическая строка сущности, обрезанная — как 'сигнатура'."""
    s = lines[_node_start(node) - 1].strip()
    return (s[:110] + "…") if len(s) > 110 else s


def chunk_file(rel_path: str, source: str) -> List[Chunk]:
    """Разбить один python-файл на чанки.

    TODO(tree-sitter): здесь единственное место, привязанное к парсеру.
    Для не-python языков этапа 3 нужно заменить реализацию на tree-sitter
    (контракт: список чанков с (path, start_line, end_line, name, text)).
    """
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # битый файл — весь файл одним чанком, чтобы не терять из индекса
        return [Chunk(rel_path, 1, len(lines) or 1, "<module> (unparsed)", source)]

    top = [n for n in tree.body]
    defs = [n for n in top if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    chunks: List[Chunk] = []

    # шапка модуля: всё до первой сущности
    if defs:
        header_end = _node_start(defs[0]) - 1
    else:
        header_end = len(lines)
    header_text = "\n".join(lines[:header_end])
    if header_text.strip():
        chunks.append(Chunk(rel_path, 1, max(header_end, 1), "<module>", header_text))
    elif not defs:
        chunks.append(Chunk(rel_path, 1, max(len(lines), 1), "<module>", source))

    for node in defs:
        start, end = _node_start(node), getattr(node, "end_lineno", node.lineno)
        text = "\n".join(lines[start - 1:end])
        name = getattr(node, "name", None)
        sig = _signature(node, lines)
        if name:
            sig = f"{name}: {sig}" if not sig.startswith(name) else sig
        chunks.append(Chunk(rel_path, start, end, sig, text))
        # большие классы дополнительно режем на методы (клас целиком тоже остаётся)
        if isinstance(node, ast.ClassDef) and end - start > 120:
            methods = [n for n in node.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and getattr(n, "end_lineno", 0) - n.lineno > 15]
            for m in methods:
                ms, me = _node_start(m), getattr(m, "end_lineno", m.lineno)
                chunks.append(Chunk(rel_path, ms, me,
                                    f"{node.name}.{m.name}: "
                                    + lines[_node_start(m) - 1].strip()[:100],
                                    "\n".join(lines[ms - 1:me])))
    return chunks


def iter_py_files(src_root: Path) -> List[Path]:
    return sorted(p for p in src_root.rglob("*.py") if p.is_file())


# ---------------------------------------------------------------- токенизация

_CAMEL1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL2 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CODE_STOP = {"self", "none", "true", "false", "def", "class", "return", "import", "from", "else"}


def split_identifier(tok: str) -> List[str]:
    """snake_case / camelCase / kebab -> части в нижнем регистре."""
    tok = tok.replace("-", "_")
    if "_" in tok:
        parts = [p for p in tok.split("_") if p]
    else:
        s = _CAMEL1.sub(r"\1 \2", tok)
        s = _CAMEL2.sub(r"\1 \2", s)
        parts = s.split()
    return [p.lower() for p in parts if p]


def tokenize_code(text: str) -> List[str]:
    out: List[str] = []
    for w in WORD_RE.findall(text):
        out.extend(split_identifier(w))
    return [t for t in out if len(t) >= 3 and t not in _CODE_STOP]


# ------------------------------------------------- извлечение ключей запроса

_PATHLIKE_RE = re.compile(r"[\w\-.]+(?:/[\w\-.]+)*\.(?:py|md|json|toml|jsonl|txt|sh|yml|yaml)\b")


@dataclass
class QueryTerms:
    rg_patterns: List[Tuple[str, float]]          # (литерал для rg -F, вес)
    bm25_tokens: Dict[str, float]                  # токен -> вес запроса


def extract_query_terms(query: str) -> QueryTerms:
    rg: Dict[str, float] = {}
    bm: Dict[str, float] = {}

    def add_pat(pat: str, w: float) -> None:
        if len(pat) < 3:
            return
        rg[pat] = max(rg.get(pat, 0.0), w)

    def add_bm(tok: str, w: float) -> None:
        if len(tok) >= 3:
            bm[tok] = max(bm.get(tok, 0.0), w)

    # 1) пути и имена файлов, упомянутые в запросе (самый сильный сигнал)
    for raw in _PATHLIKE_RE.findall(query):
        p = raw.rstrip(".,:;)")
        parts = [x for x in re.split(r"[./\\-]", p) if x and x != p.split(".")[-1]]
        stem = parts[-1] if parts else p
        for i, part in enumerate(parts):
            if len(part) < 3:
                continue
            w = 3.0 if part == stem else (1.0 if i < len(parts) - 1 else 2.0)
            add_pat(part, w)
            add_bm(part.lower(), w)
            for t in split_identifier(part):
                add_pat(t, w * 0.7)
                add_bm(t, w * 0.7)

    # 2) ascii-слова: идентификаторы (snake/camel) — сильные, обычные слова — слабые
    for w0 in WORD_RE.findall(query):
        if len(w0) < 4 and "_" not in w0:
            continue
        parts = split_identifier(w0)
        if "_" in w0 or any(c.isupper() for c in w0):
            add_pat(w0, 3.0)
            add_bm(w0.lower(), 3.0)
            for t in parts:
                add_pat(t, 1.5)
                add_bm(t, 1.5)
        else:
            add_pat(w0, 1.5)
            add_bm(w0.lower(), 1.5)

    # 3) кириллица: в коде бывает в комментариях/строках — слабый сигнал
    #    (BM25-корпус лексический её не увидит — задокументированная слабость)
    for w0 in CYR_RE.findall(query):
        add_pat(w0, 0.4)

    patterns = sorted(rg.items(), key=lambda kv: -kv[1])[:32]
    return QueryTerms(rg_patterns=patterns, bm25_tokens=dict(list(bm.items())[:80]))


# ---------------------------------------------------------------- граф импортов

class ImportGraph:
    """Файл -> импортируемые файлы (с символами) и обратно. Парсинг ast."""

    def __init__(self, src_root: Path, rel_files: List[str]):
        self.src_root = src_root
        self.module_to_file: Dict[str, str] = {}
        for rel in rel_files:
            mod = rel[:-3].replace("/", ".")
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            self.module_to_file[mod] = rel
        # алиасы пакетов: drcode.tui.inputs -> пакет drcode.tui тоже резолвим
        self.out_edges: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        for rel in rel_files:
            self._scan(rel)

    def _resolve(self, dotted: str) -> Optional[str]:
        parts = dotted.split(".")
        while parts:
            mod = ".".join(parts)
            if mod in self.module_to_file:
                return self.module_to_file[mod]
            parts.pop()
        return None

    def _scan(self, rel: str) -> None:
        try:
            tree = ast.parse((self.src_root / rel).read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            return
        is_init = rel.endswith("__init__.py")
        mod_parts = rel[:-3].replace("/", ".").split(".")
        # пакет текущего файла: для inputs.py это drcode.tui; для __init__.py пакета — сам пакет
        pkg_parts = mod_parts if is_init else mod_parts[:-1]
        for node in ast.walk(tree):
            targets: List[Tuple[str, List[str]]] = []  # (dotted module, symbols)
            if isinstance(node, ast.Import):
                for a in node.names:
                    targets.append((a.name, []))
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base_parts = pkg_parts[: len(pkg_parts) - (node.level - 1)] if node.level > 1 else pkg_parts
                    base = ".".join(base_parts + ([node.module] if node.module else []))
                syms = [a.name for a in node.names if a.name != "*"]
                targets.append((base, syms))
            else:
                continue
            for dotted, syms in targets:
                if not dotted:
                    continue
                dep = self._resolve(dotted)
                if dep and dep != rel:
                    self.out_edges[rel][dep].update(syms)
                # from a.b import symbol, где symbol — подмодуль
                for s in syms:
                    sub = self._resolve(dotted + "." + s)
                    if sub and sub != rel:
                        self.out_edges[rel][sub].add(s)

    def neighbors(self, rel: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """(импортируемые файлом, кто импортирует файл) с символами."""
        return self.out_edges.get(rel, {}), {
            src: syms for src, m in self.out_edges.items() for dep, syms in m.items() if dep == rel
        }


# ---------------------------------------------------------------- BM25

class BM25Index:
    """Okapi BM25 по чанкам. Собственная реализация (rank_bm25 недоступен offline)."""

    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.tfs: List[Counter] = [Counter(toks) for toks in docs_tokens]
        self.dls = [sum(c.values()) for c in self.tfs]
        self.avgdl = (sum(self.dls) / self.N) if self.N else 0.0
        df: Counter = Counter()
        for c in self.tfs:
            df.update(c.keys())
        self.idf = {t: math.log(1.0 + (self.N - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, c in enumerate(self.tfs):
            for t, f in c.items():
                self.postings[t].append((i, f))

    def score(self, qweights: Dict[str, float]) -> List[float]:
        scores = [0.0] * self.N
        for qt, qw in qweights.items():
            for i, tf in self.postings.get(qt, ()):
                dl = self.dls[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += qw * self.idf.get(qt, 0.0) * tf * (self.k1 + 1) / denom
        return scores


# ---------------------------------------------------------------- индекс

@dataclass
class CodeIndex:
    src_root: Path
    chunks: List[Chunk] = field(default_factory=list)
    chunks_by_file: Dict[str, List[Chunk]] = field(default_factory=dict)
    chunk_by_key: Dict[Tuple[str, int, int], Chunk] = field(default_factory=dict)
    graph: Optional[ImportGraph] = None
    bm25: Optional[BM25Index] = None

    @classmethod
    def build(cls, src_root: Path) -> "CodeIndex":
        idx = cls(src_root=src_root)
        files = iter_py_files(src_root)
        rels = [str(p.relative_to(src_root)) for p in files]
        docs_tokens: List[List[str]] = []
        for p, rel in zip(files, rels):
            text = p.read_text(encoding="utf-8", errors="replace")
            cs = chunk_file(rel, text)
            idx.chunks_by_file[rel] = cs
            for c in cs:
                idx.chunk_by_key[(c.path, c.start_line, c.end_line)] = c
                # в документ BM25 добавляем и токены пути (имя файла ×2 — сильный сигнал)
                path_toks = tokenize_code(rel)
                docs_tokens.append(tokenize_code(c.text) + path_toks + path_toks[-2:])
            idx.chunks.extend(cs)
        idx.graph = ImportGraph(src_root, rels)
        idx.bm25 = BM25Index(docs_tokens)
        return idx

    def chunk_of(self, cand: Dict) -> Optional[Chunk]:
        return self.chunk_by_key.get((cand["path"], cand["start_line"], cand["end_line"]))

    def hits_to_chunk_scores(self, rel: str, line_weights: Dict[int, float],
                             file_bonus: float) -> Dict[Tuple[str, int, int], float]:
        out: Dict[Tuple[str, int, int], float] = {}
        cs = self.chunks_by_file.get(rel, [])
        starts = [c.start_line for c in cs]
        for line, w in line_weights.items():
            i = bisect_right(starts, line) - 1
            if 0 <= i < len(cs) and cs[i].end_line >= line:
                key = (cs[i].path, cs[i].start_line, cs[i].end_line)
                out[key] = out.get(key, 0.0) + w
        if file_bonus:
            for c in cs:
                key = (c.path, c.start_line, c.end_line)
                out[key] = out.get(key, 0.0) + file_bonus
        return out


def candidate(chunk: Chunk, score: float, source: str) -> Dict:
    return {
        "path": chunk.path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "score": round(score, 4),
        "source": source,
    }


# ---------------------------------------------------------------- бейслайны

class RipgrepProvider:
    """Бейслайн 'ripgrep-ключи': примитивное извлечение ключевых слов из запроса."""

    name = "rg"

    def __init__(self, index: CodeIndex):
        self.index = index

    def search(self, query: str, limit: int = 300) -> List[Dict]:
        terms = extract_query_terms(query)
        pats = [p for p, _ in terms.rg_patterns]
        if not pats:
            return []
        cmd = ["rg", "--json", "-F", "-g", "*.py"]
        for p in pats:
            cmd += ["-e", p]
        cmd.append(str(self.index.src_root))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (subprocess.TimeoutExpired, OSError):
            return []
        weight = dict(terms.rg_patterns)
        weight_l = {k.lower(): v for k, v in weight.items()}
        hits: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for line in proc.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "match":
                continue
            path = ev["data"]["path"]["text"]
            try:
                rel = str(Path(path).relative_to(self.index.src_root))
            except ValueError:
                continue
            ln = ev["data"]["line_number"]
            seen_terms: Set[str] = set()
            for sm in ev["data"].get("submatches", []):
                t = sm["match"]["text"]
                key = t if t in weight else weight_l.get(t.lower())
                if key is None:
                    continue
                norm = t if t in weight else t.lower()
                if norm in seen_terms:
                    continue
                seen_terms.add(norm)
                hits[rel][ln] += weight[key] if key in weight else weight_l.get(key.lower(), 0)
        out: List[Tuple[float, Chunk, str]] = []
        for rel, lws in hits.items():
            # бонус за совпадение паттерна с самим путём файла
            low = rel.lower()
            bonus = sum(w * 0.5 for p, w in terms.rg_patterns if p.lower() in low)
            for key, sc in self.index.hits_to_chunk_scores(rel, lws, bonus).items():
                ch = self.index.chunk_by_key[key]
                out.append((sc, ch, "rg"))
        out.sort(key=lambda t: (-t[0], t[1].path, t[1].start_line))
        return [candidate(ch, sc, src) for sc, ch, src in out[:limit]]


class BM25Provider:
    name = "bm25"

    def __init__(self, index: CodeIndex):
        self.index = index

    def search(self, query: str, limit: int = 300) -> List[Dict]:
        terms = extract_query_terms(query)
        if not terms.bm25_tokens:
            return []
        scores = self.index.bm25.score(terms.bm25_tokens)
        out = [(scores[i], ch, "bm25") for i, ch in enumerate(self.index.chunks) if scores[i] > 0]
        out.sort(key=lambda t: (-t[0], t[1].path, t[1].start_line))
        return [candidate(ch, sc, src) for sc, ch, src in out[:limit]]


class RRFProvider:
    """Reciprocal rank fusion двух бейслайнов (файловый уровень)."""

    name = "rg+bm25"
    K = 60

    def __init__(self, index: CodeIndex, providers: List[Tuple[str, "object"]]):
        self.index = index
        self.providers = providers

    def search(self, query: str, limit: int = 300) -> List[Dict]:
        file_rank: Dict[str, float] = defaultdict(float)
        best_chunk: Dict[str, Tuple[float, Chunk, str]] = {}
        for pname, prov in self.providers:
            cands = prov.search(query, limit=200)
            seen_rank: Dict[str, int] = {}
            for c in cands:
                f = c["path"]
                if f not in seen_rank:
                    seen_rank[f] = len(seen_rank) + 1
                key = (c["path"], c["start_line"], c["end_line"])
                ch = self.index.chunk_by_key[key]
                if f not in best_chunk or c["score"] > best_chunk[f][0]:
                    best_chunk[f] = (c["score"], ch, pname)
            for f, r in seen_rank.items():
                file_rank[f] += 1.0 / (self.K + r)
        ranked = sorted(file_rank.items(), key=lambda kv: (-kv[1], kv[0]))
        out: List[Dict] = []
        for f, rrf in ranked[:limit]:
            sc, ch, src = best_chunk[f]
            out.append(candidate(ch, rrf * 100.0, f"rrf({src})"))
        return out


class StructBoostProvider:
    """Структурный буст: соседи по графу импортов от уже найденных релевантных.

    Оборачивает внутренний провайдер: топ-N файлов внутреннего ранжирования
    тянут за собой импортируемые/импортирующие файлы с понижающим весом.
    Для `from pkg.mod import sym` кандидатом становится чанк с именем sym,
    иначе шапка модуля соседа.
    """

    name = "struct"
    IN_FACTOR = 0.5   # файл, который импортирует найденный
    OUT_FACTOR = 0.3  # файл, который импортируется найденным
    TOP_FILES = 8

    def __init__(self, index: CodeIndex, inner):
        self.index = index
        self.inner = inner
        self.name = f"{inner.name}+struct"

    def _symbol_chunk(self, rel: str, syms: Set[str]) -> Optional[Chunk]:
        cs = self.index.chunks_by_file.get(rel, [])
        for c in cs:
            base = c.name.split(":")[0].split(".")[-1]
            if base in syms:
                return c
        return cs[0] if cs else None

    def search(self, query: str, limit: int = 300) -> List[Dict]:
        cands = self.inner.search(query, limit=200)
        if not cands:
            return cands
        file_best: Dict[str, Tuple[float, Dict]] = {}
        for c in cands:
            f = c["path"]
            if f not in file_best:
                file_best[f] = (c["score"], c)
        out = list(cands)
        top = [f for f, (sc, _) in sorted(file_best.items(), key=lambda kv: -kv[1][0])
               if sc > 0][: self.TOP_FILES]
        assert self.index.graph is not None
        for f in top:
            ref_score = file_best[f][0]
            out_edges, in_edges = self.index.graph.neighbors(f)
            for dep, syms in out_edges.items():
                if dep in file_best:
                    continue
                ch = self._symbol_chunk(dep, syms)
                if ch is not None:
                    out.append(candidate(ch, self.OUT_FACTOR * ref_score, "struct(import)"))
            for src_f, syms in in_edges.items():
                if src_f in file_best:
                    continue
                ch = self._symbol_chunk(src_f, syms)
                if ch is not None:
                    out.append(candidate(ch, self.IN_FACTOR * ref_score, "struct(imported-by)"))
        out.sort(key=lambda c: (-c["score"], c["path"], c["start_line"]))
        return out[:limit]


# ---------------------------------------------------------------- сборка

def build_providers(src_root: Path, modes: Optional[List[str]] = None) -> Dict[str, object]:
    """Построить провайдеров над одним индексом (общая предобработка — один раз)."""
    index = CodeIndex.build(src_root)
    rg = RipgrepProvider(index)
    bm = BM25Provider(index)
    all_modes = {
        "rg": rg,
        "bm25": bm,
        "rg+bm25": RRFProvider(index, [("rg", rg), ("bm25", bm)]),
        "bm25+struct": StructBoostProvider(index, bm),
        "rg+bm25+struct": StructBoostProvider(index, RRFProvider(index, [("rg", rg), ("bm25", bm)])),
    }
    if modes:
        unknown = set(modes) - set(all_modes)
        if unknown:
            raise SystemExit(f"unknown modes: {unknown}; available: {sorted(all_modes)}")
        return {m: all_modes[m] for m in modes}
    return all_modes


def _print_cands(cands: List[Dict], index: CodeIndex, limit: int) -> None:
    for c in cands[:limit]:
        ch = index.chunk_of(c)
        print(f"{c['score']:8.3f}  {ch.header if ch else c['path']}  [{c['source']}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="candidate baselines (fastcontext-scout phase 2)")
    ap.add_argument("--src-root", default=str(Path(__file__).resolve().parents[3] / "src"),
                    help="корень исходников для индекса (по умолчанию <project>/src)")
    ap.add_argument("--mode", default="bm25")
    ap.add_argument("--query", default="")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--stats", action="store_true", help="показать статистику индекса")
    args = ap.parse_args()

    src_root = Path(args.src_root).resolve()
    provs = build_providers(src_root, [args.mode])
    prov = provs[args.mode]
    index = getattr(prov, "index", None) or getattr(prov, "inner").index  # type: ignore[union-attr]
    if args.stats:
        n_files = len(index.chunks_by_file)
        sizes = [c.end_line - c.start_line + 1 for c in index.chunks]
        med = sorted(sizes)[len(sizes) // 2] if sizes else 0
        tok = sum(c.tokens for c in index.chunks)
        print(f"files: {n_files}, chunks: {len(index.chunks)}, "
              f"median chunk lines: {med}, total ~tokens (chars/4): {tok}")
        return
    if not args.query:
        ap.error("--query required (or --stats)")
    cands = prov.search(args.query)  # type: ignore[attr-defined]
    _print_cands(cands, index, args.limit)


if __name__ == "__main__":
    main()
