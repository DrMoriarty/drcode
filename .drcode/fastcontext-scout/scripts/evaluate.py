#!/usr/bin/env python3
"""fastcontext-scout, этап 2: метрики бейслайнов против golden-датасета.

Метрики (файловый уровень; релевантность — файлы из датасета):
  - recall@k: доля релевантных файлов, покрытых топ-k кандидатами
    (кандидаты — чанки; файл учитывается один раз по первому вхождению);
  - recall@budget: жадная упаковка чанков по убыванию score в пакет ≤ B токенов
    (токены = chars/4 по выданным чанкам: заголовок + тело; чанки файлов,
    уже попавших в пакет, пропускаются и токенов не добавляют);
  - MRR: 1/ранг первого релевантного файла в ранжировании.

Считается отдельно по label=strong и по strong+weak (все релевантные).

CLI:
  python3 scripts/evaluate.py --dataset dev                 # таблица по всем режимам
  python3 scripts/evaluate.py --dataset test --modes bm25,rg
  python3 scripts/evaluate.py --dataset dev --errors --errors-mode bm25+struct --errors-n 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import candidates as C

K_VALUES = (5, 10, 20)
BUDGETS = (2000, 4000, 8000)


def load_records(path: Path) -> List[Dict]:
    recs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def norm_path(p: str) -> str:
    """Пути датасета нормализованы к корню проекта; индекс — к src/."""
    if p.startswith("src/"):
        p = p[len("src/"):]
    return p


def relevant_sets(rec: Dict, index_files: Optional[set] = None) -> Dict[str, set]:
    strong = {norm_path(r["path"]) for r in rec["relevant"] if r["label"] == "strong"}
    all_rel = {norm_path(r["path"]) for r in rec["relevant"]}
    if index_files is not None:
        return {"strong": strong & index_files, "all": all_rel & index_files,
                "strong_full": strong, "all_full": all_rel}
    return {"strong": strong, "all": all_rel, "strong_full": strong, "all_full": all_rel}


def eval_one(cands: List[Dict], rel_strong: set, rel_all: set,
             rel_strong_full: Optional[set] = None,
             rel_all_full: Optional[set] = None) -> Dict:
    res: Dict[str, Optional[float]] = {}

    # файлы в порядке первого вхождения чанка
    files_ranked: List[str] = []
    seen = set()
    for c in cands:
        if c["path"] not in seen:
            seen.add(c["path"])
            files_ranked.append(c["path"])

    def recall_at_k(subset: set, k: int, denom: Optional[set] = None) -> Optional[float]:
        if not denom:
            denom = subset
        if not denom:
            return None
        hit = len(subset & set(files_ranked[:k]))
        return hit / len(denom)

    for k in K_VALUES:
        # по всем релевантным (в знаменателе и недостижимые для индекса файлы)
        res[f"r@{k}"] = recall_at_k(rel_all, k, rel_all_full)
        res[f"r{k}_strong"] = recall_at_k(rel_strong, k, rel_strong_full)

    # MRR по первому вхождению релевантного файла
    def mrr(subset: set) -> Optional[float]:
        if not subset:
            return None
        for i, f in enumerate(files_ranked, 1):
            if f in subset:
                return 1.0 / i
        return 0.0

    res["mrr"] = mrr(rel_all)
    res["mrr_strong"] = mrr(rel_strong)

    # recall@budget: жадная упаковка
    def budget_recall(subset: set, budget: int, index: C.CodeIndex,
                      denom: Optional[set] = None) -> Optional[float]:
        if not denom:
            denom = subset
        if not denom:
            return None
        packed: set = set()
        tokens = 0
        for c in cands:
            if c["path"] in packed:
                continue
            ch = index.chunk_of(c)
            t = ch.tokens if ch else 200
            if tokens + t > budget:
                continue
            tokens += t
            packed.add(c["path"])
            if denom <= packed:
                break
        return len(subset & packed) / len(denom)

    for b in BUDGETS:
        res[f"rb@{b}"] = budget_recall(rel_all, b, _INDEX[0], rel_all_full)
        res[f"rb{b}_strong"] = budget_recall(rel_strong, b, _INDEX[0], rel_strong_full)
    return res


_INDEX: List[C.CodeIndex] = []  # для budget_recall без протаскивания аргумента


def run(dataset_path: Path, provs: Dict, index: C.CodeIndex, errors_mode: Optional[str],
        errors_n: int) -> Dict:
    _INDEX.clear()
    _INDEX.append(index)
    records = load_records(dataset_path)
    index_files = {c.path for c in index.chunks}
    ceil = []
    for rec in records:
        all_full = relevant_sets(rec)["all_full"]
        if all_full:
            ceil.append(len(all_full & index_files) / len(all_full))
    print(f"[{dataset_path.stem}] mean share of relevant files inside the index "
          f"(recall ceiling): {sum(ceil)/len(ceil)*100:.1f}%")
    results: Dict[str, Dict] = {"_ceiling": {"mean_reachable": sum(ceil) / len(ceil)}}
    per_mode_cands: Dict[str, List] = {}
    n_live = sum(1 for rec in records if relevant_sets(rec, index_files)["strong"])
    print(f"[{dataset_path.stem}] live subset (≥1 существующий strong-файл): {n_live}/{len(records)}")
    for mode, prov in provs.items():
        t0 = time.time()
        agg: Dict[str, List[float]] = {}
        live_agg: Dict[str, List[float]] = {}
        per_record = []
        for rec in records:
            sets = relevant_sets(rec, index_files)
            cands = prov.search(rec["query"], limit=1000)
            r = eval_one(cands, sets["strong"], sets["all"],
                         sets["strong_full"], sets["all_full"])
            per_record.append((rec, r, cands))
            for key, v in r.items():
                if v is not None:
                    agg.setdefault(key, []).append(v)
            # "живая" подвыборка: хотя бы один strong-файл существует в индексе
            if sets["strong"]:
                for key in ("r5_strong", "r10_strong", "r20_strong",
                            "rb2000_strong", "rb4000_strong", "rb8000_strong", "mrr_strong"):
                    if r.get(key) is not None:
                        live_agg.setdefault(key, []).append(r[key])
        summary = {k: sum(v) / len(v) for k, v in agg.items()}
        results[mode] = summary
        results.setdefault("_live", {})[mode] = {k: sum(v) / len(v) for k, v in live_agg.items()}
        per_mode_cands[mode] = per_record
        print(f"[{dataset_path.stem}] {mode}: {len(records)} queries in {time.time()-t0:.1f}s")
    if errors_mode and errors_mode in per_mode_cands:
        dump_errors(per_mode_cands[errors_mode], errors_n,
                    Path(__file__).resolve().parent.parent / "reports"
                    / f"errors_{dataset_path.stem}_{errors_mode.replace('+', '_')}.md")
    return results


def dump_errors(per_record: List, n: int, out_path: Path) -> None:
    """Худшие записи по recall@10 (strong) — разбор ошибок."""
    scored = []
    for rec, r, cands in per_record:
        s = r.get("r10_strong")
        if s is None:
            continue
        scored.append((s, rec, r, cands))
    scored.sort(key=lambda t: (t[0], -(t[2].get("mrr_strong") or 0)))
    worst = scored[:n]
    lines = ["# Разбор ошибок (худшие по recall@10 strong)\n"]
    for s, rec, r, cands in worst:
        sets = relevant_sets(rec)
        found: List[str] = []
        seen = set()
        for c in cands[:10]:
            if c["path"] not in seen:
                seen.add(c["path"])
                found.append(c["path"])
        missed = sorted(sets["strong"] - seen)
        lines.append(f"## {rec['task_id']} — recall@10 strong={s:.2f}, MRR strong={r.get('mrr_strong')}")
        lines.append(f"- kind: {rec['kind']}, lang: {rec['lang']}, n_strong: {rec.get('n_strong')}")
        lines.append(f"- **query**:\n\n```\n{rec['query'][:900]}\n```\n")
        lines.append(f"- **релевантные (strong)**: {', '.join(sorted(sets['strong']))}")
        weak = sorted(sets['all'] - sets['strong'])
        if weak:
            lines.append(f"- слабые: {', '.join(weak)}")
        lines.append(f"- **нашлось в топ-10**: {', '.join(found)}")
        lines.append(f"- **не нашлось (strong)**: {', '.join(missed) if missed else '—'}")
        lines.append(f"- метрики: {json.dumps({k: v for k, v in r.items()}, ensure_ascii=False)}")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"errors dumped: {out_path}")


def fmt_table(results: Dict[str, Dict]) -> str:
    cols = ([f"r@{k}" for k in K_VALUES] + [f"r{k}_strong" for k in K_VALUES]
            + [f"rb@{b}" for b in BUDGETS] + [f"rb{b}_strong" for b in BUDGETS]
            + ["mrr", "mrr_strong"])
    header = "| mode | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = [header, sep]
    for mode, summary in results.items():
        if mode.startswith("_"):
            continue
        vals = []
        for c in cols:
            v = summary.get(c)
            vals.append("—" if v is None else f"{v*100:.1f}")
        rows.append(f"| {mode} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="evaluate baselines (fastcontext-scout phase 2)")
    default_root = Path("/Users/vasiliy.makarov/pr/drcode/.drcode/fastcontext-scout")
    ap.add_argument("--src-root", default=str(Path(__file__).resolve().parents[3] / "src"))
    ap.add_argument("--dataset-dir", default=str(default_root / "dataset"))
    ap.add_argument("--dataset", default="dev", choices=["dev", "test", "golden"])
    ap.add_argument("--modes", default="rg,bm25,rg+bm25,bm25+struct,rg+bm25+struct")
    ap.add_argument("--out", default="", help="куда сохранить json результатов")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--errors-mode", default="bm25+struct")
    ap.add_argument("--errors-n", type=int, default=5)
    args = ap.parse_args()

    src_root = Path(args.src_root).resolve()
    modes = [m for m in args.modes.split(",") if m]
    provs = C.build_providers(src_root, modes)

    def index_of(p) -> C.CodeIndex:
        return p.index if hasattr(p, "index") else p.inner.index  # type: ignore[attr-defined]

    index = index_of(next(iter(provs.values())))
    ds_path = Path(args.dataset_dir) / f"{args.dataset}.jsonl"
    results = run(ds_path, provs, index,
                  args.errors_mode if args.errors else None, args.errors_n)
    print(f"\n## {args.dataset} ({ds_path})\n")
    print(fmt_table(results))
    live = {m: v for m, v in results.get("_live", {}).items()}
    if live:
        print("\n### Живая подвыборка (задачи с ≥1 существующим strong-файлом), strong-метрики\n")
        cols = ["r5_strong", "r10_strong", "r20_strong", "rb2000_strong",
                "rb4000_strong", "rb8000_strong", "mrr_strong"]
        print("| mode | " + " | ".join(cols) + " |")
        print("|" + "---|" * (len(cols) + 1))
        for mode, summary in live.items():
            print(f"| {mode} | " + " | ".join(
                "—" if summary.get(c) is None else f"{summary[c]*100:.1f}" for c in cols) + " |")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
