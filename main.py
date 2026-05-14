"""
mRAG experiment harness — CLI dispatcher.

Subcommands:
  smoke         End-to-end sanity run on 5 questions with TF-IDF, no API calls.
  run           Run the experiment grid (cells are skipped when results
                already exist in the parquet — caches do the rest).
  analyze       Build the HTML report from results.parquet.
  cache-stats   Inspect the on-disk cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from analyze import build_report
from benchmarks import load_benchmark
from cache import cache_stats
from distances import METRICS
from embeddings import build_embedder
from experiment import CellConfig, run_cell, run_grid, write_records


DEFAULT_DATASETS = ["hotpotqa", "multihoprag", "2wiki"]
# gemini-embedding-001 is the production text model with batch support.
# gemini-embedding-2 (multimodal) silently accepts only 1 item per call; if you
# pass it via --embedders, the harness detects the short response and falls
# back to one-at-a-time, but that is slower and costlier.
DEFAULT_EMBEDDERS = ["tfidf", "all-MiniLM-L6-v2", "bge-small-en-v1.5", "gemini-embedding-001"]
DEFAULT_METRICS = METRICS
DEFAULT_STRATEGIES = ["decay", "ppr"]


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_smoke(args: argparse.Namespace) -> int:
    """End-to-end on 5 HotpotQA distractor items with TF-IDF, no Gemini.

    Sanity-checks every retrieval mode and every Z-distance metric.
    """
    print("[smoke] loading 5 HotpotQA distractor items...")
    questions = load_benchmark("hotpotqa", n=5, seed=42)

    print("[smoke] building TF-IDF embedder...")
    fit_corpus = [c for q in questions for c in q.corpus]
    embedder = build_embedder("tfidf", fit_corpus=fit_corpus, target_dim=128)

    out_path = Path("results/smoke.parquet")
    if out_path.exists():
        out_path.unlink()

    cells = []
    # naive RAG + no_context once.
    cells.append(CellConfig(dataset="hotpotqa", embedder="tfidf", metric="cosine_x",
                             strategy="naive", d=16, n=5, gen_model="none"))
    cells.append(CellConfig(dataset="hotpotqa", embedder="none", metric="none",
                             strategy="no_context", d=16, n=5, gen_model="none"))
    # mRAG ablation
    for metric in METRICS:
        for strategy in ["decay", "ppr"]:
            cells.append(CellConfig(dataset="hotpotqa", embedder="tfidf", metric=metric,
                                     strategy=strategy, d=16, n=5, gen_model="none"))

    # Monkey-patch generate_answer + llm_judge to avoid API calls in smoke.
    import llm
    def _fake_gen(prompt: str, model: str = "none") -> str:
        # Pretend we read the first chunk of the prompt and reuse it.
        for line in prompt.splitlines():
            if line.startswith("- "):
                return line[2:].split(".")[0][:50]
        return "unknown"
    def _fake_judge(*args, **kwargs):
        return {"correct": False, "rationale": "smoke mode"}
    llm.generate_answer = _fake_gen
    llm.llm_judge = _fake_judge
    import experiment
    experiment.generate_answer = _fake_gen
    experiment.llm_judge = _fake_judge

    import time as _t
    t0 = _t.time()
    for i, cfg in enumerate(cells, start=1):
        records = run_cell(cfg, questions, embedder, do_judge=False,
                            cell_index=i, total_cells=len(cells))
        write_records(records, out_path)
    print(f"[smoke] wrote {out_path} in {_t.time() - t0:.1f}s")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    datasets = args.datasets or DEFAULT_DATASETS
    embedders = args.embedders or DEFAULT_EMBEDDERS
    metrics = args.metrics or DEFAULT_METRICS
    strategies = args.strategies or DEFAULT_STRATEGIES

    print(f"[run] datasets={datasets} embedders={embedders} metrics={metrics} strategies={strategies}")
    print(f"[run] n={args.n} d={args.d} judge={args.judge}")

    out = run_grid(
        datasets=datasets,
        embedders=embedders,
        metrics=metrics,
        strategies=strategies,
        n=args.n,
        d=args.d,
        out_path=args.out,
        judge=args.judge,
    )
    print(f"[run] results -> {out}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    out = build_report(results_path=args.results, out_path=args.report)
    print(f"[analyze] report -> {out.resolve()}")
    return 0


def cmd_cache_stats(args: argparse.Namespace) -> int:
    stats = cache_stats()
    print(json.dumps(stats, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Argparser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mrag")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("smoke", help="end-to-end sanity run (no API)")
    s.set_defaults(func=cmd_smoke)

    r = sub.add_parser("run", help="run experiment grid")
    r.add_argument("--datasets", nargs="*", default=None)
    r.add_argument("--embedders", nargs="*", default=None)
    r.add_argument("--metrics", nargs="*", default=None)
    r.add_argument("--strategies", nargs="*", default=None)
    r.add_argument("--n", type=int, default=500)
    r.add_argument("--d", type=int, default=32)
    r.add_argument("--out", type=str, default="results/results.parquet")
    r.add_argument("--no-judge", dest="judge", action="store_false")
    r.add_argument("--judge", dest="judge", action="store_true")
    r.set_defaults(judge=True)
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("analyze", help="build HTML report")
    a.add_argument("--results", type=str, default="results/results.parquet")
    a.add_argument("--report", type=str, default="results/report.html")
    a.set_defaults(func=cmd_analyze)

    cs = sub.add_parser("cache-stats", help="show cache contents")
    cs.set_defaults(func=cmd_cache_stats)

    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
