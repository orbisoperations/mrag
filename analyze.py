"""
Aggregate results.parquet into a single self-contained HTML report at
results/report.html, plus individual csv/png artifacts under results/.

Tables:
  - Answer accuracy (EM / F1 / judge-acc) by dataset × embedder × metric × strategy
  - Retrieval (recall, precision)
  - Dimensionality ablation
  - Wall-clock

Plots:
  - Heatmap of F1 across (metric × embedder) per dataset
  - Bar chart of F1 by metric, grouped by strategy, per dataset
  - Scatter of context-recall vs answer-F1
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _save_fig(fig, name: str) -> str:
    p = RESULTS_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=120, bbox_inches="tight")
    b = _fig_to_b64(fig)
    return b


def _table_answer_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["dataset", "embedder", "metric", "strategy"])
        .agg(em=("em", "mean"), f1=("f1", "mean"), judge=("judge", "mean"), n=("qid", "count"))
        .reset_index()
        .sort_values(["dataset", "embedder", "f1"], ascending=[True, True, False])
    )
    return g


def _table_retrieval(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["dataset", "embedder", "metric", "strategy"])
        .agg(recall=("recall", "mean"), precision=("precision", "mean"),
             retrieved=("n_retrieved", "mean"), n=("qid", "count"))
        .reset_index()
        .sort_values(["dataset", "embedder", "recall"], ascending=[True, True, False])
    )
    return g


def _table_wallclock(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "embedder", "metric", "strategy"])
        .agg(mean_wall_s=("wall_s", "mean"), total_wall_s=("wall_s", "sum"))
        .reset_index()
        .sort_values("total_wall_s", ascending=False)
    )


def _plot_heatmap_f1_per_dataset(df: pd.DataFrame) -> dict:
    """One heatmap per dataset: rows=metric, cols=embedder, values=F1 (best strategy)."""
    out: dict = {}
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        agg = sub.groupby(["metric", "embedder"]).agg(f1=("f1", "mean")).reset_index()
        pivot = agg.pivot(index="metric", columns="embedder", values="f1")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(2 + 1.2 * len(pivot.columns), 1.5 + 0.6 * len(pivot.index)))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if val < 0.5 else "black", fontsize=8)
        ax.set_title(f"F1 — {ds} (metric × embedder)")
        fig.colorbar(im, ax=ax)
        out[ds] = _save_fig(fig, f"heatmap_f1_{ds}.png")
    return out


def _plot_bar_f1_by_metric(df: pd.DataFrame) -> dict:
    """For each dataset, bar chart of F1 by metric, hue=strategy, averaged over embedders."""
    out: dict = {}
    for ds in sorted(df["dataset"].unique()):
        sub = df[(df["dataset"] == ds) & (df["strategy"].isin(["decay", "ppr", "naive", "no_context"]))]
        agg = sub.groupby(["metric", "strategy"]).agg(f1=("f1", "mean")).reset_index()
        if agg.empty:
            continue
        metrics = sorted(agg["metric"].unique())
        strategies = sorted(agg["strategy"].unique())
        x = np.arange(len(metrics))
        width = 0.8 / max(1, len(strategies))
        fig, ax = plt.subplots(figsize=(2 + 1.0 * len(metrics), 4))
        for i, s in enumerate(strategies):
            vals = [agg[(agg["metric"] == m) & (agg["strategy"] == s)]["f1"].mean() for m in metrics]
            ax.bar(x + i * width, vals, width, label=s)
        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=20, ha="right")
        ax.set_ylabel("F1 (mean over embedders + questions)")
        ax.set_title(f"F1 by metric and strategy — {ds}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        out[ds] = _save_fig(fig, f"bar_f1_{ds}.png")
    return out


def _plot_recall_vs_f1(df: pd.DataFrame) -> Optional[str]:
    sub = df[df["strategy"].isin(["decay", "ppr", "naive"])]
    if sub.empty:
        return None
    g = sub.groupby(["dataset", "embedder", "metric", "strategy"]).agg(
        recall=("recall", "mean"), f1=("f1", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"decay": "o", "ppr": "s", "naive": "^"}
    for s in g["strategy"].unique():
        ssub = g[g["strategy"] == s]
        ax.scatter(ssub["recall"], ssub["f1"], marker=markers.get(s, "x"),
                   label=s, alpha=0.7, s=40)
    ax.set_xlabel("Context recall")
    ax.set_ylabel("Answer F1")
    ax.set_title("Retrieval recall vs answer F1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return _save_fig(fig, "scatter_recall_vs_f1.png")


def _plot_dimensionality(df: pd.DataFrame) -> Optional[str]:
    if df["d"].nunique() <= 1:
        return None
    sub = df[df["strategy"].isin(["decay", "ppr"])]
    g = sub.groupby(["d", "embedder", "metric"]).agg(f1=("f1", "mean")).reset_index()
    if g.empty:
        return None
    # Plot best (embedder, metric) per d.
    best = g.loc[g.groupby("d")["f1"].idxmax()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(best["d"], best["f1"], marker="o")
    for _, r in best.iterrows():
        ax.annotate(f"{r['embedder']}\n{r['metric']}",
                    (r["d"], r["f1"]),
                    fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("PCA dimension d")
    ax.set_ylabel("Best F1 (over embedders × metrics)")
    ax.set_title("Dimensionality ablation")
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, "dim_ablation.png")


def _executive_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No data.</p>"
    summary_rows = []
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        g = sub.groupby(["embedder", "metric", "strategy"]).agg(f1=("f1", "mean"),
                                                                  judge=("judge", "mean"),
                                                                  em=("em", "mean"),
                                                                  n=("qid", "count")).reset_index()
        best_f1 = g.loc[g["f1"].idxmax()]
        best_judge = g.loc[g["judge"].idxmax()]
        naive_row = sub[sub["strategy"] == "naive"].agg({"f1": "mean", "judge": "mean"})
        no_ctx_row = sub[sub["strategy"] == "no_context"].agg({"f1": "mean", "judge": "mean"})
        summary_rows.append({
            "dataset": ds,
            "best_by_f1": f"{best_f1['embedder']} + {best_f1['metric']} + {best_f1['strategy']} (F1={best_f1['f1']:.3f}, n={int(best_f1['n'])})",
            "best_by_judge": f"{best_judge['embedder']} + {best_judge['metric']} + {best_judge['strategy']} (judge={best_judge['judge']:.3f})",
            "naive_baseline_f1": f"{naive_row['f1']:.3f}" if not naive_row.empty else "—",
            "no_context_baseline_f1": f"{no_ctx_row['f1']:.3f}" if not no_ctx_row.empty else "—",
        })
    df_s = pd.DataFrame(summary_rows)
    return df_s.to_html(index=False, classes="summary", border=0)


def _df_to_html(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df.empty:
        return "<p><i>(no data)</i></p>"
    fmt: dict[str, str] = {}
    for col in df.columns:
        if df[col].dtype.kind == "f":
            fmt[col] = floatfmt
    styled = df.style.format(fmt).hide(axis="index")
    # Conditional formatting for known metric columns.
    for col in ("em", "f1", "judge", "recall", "precision"):
        if col in df.columns and df[col].dtype.kind == "f":
            styled = styled.background_gradient(subset=[col], cmap="Greens", vmin=0, vmax=1)
    return styled.to_html(table_attributes='class="results sortable"')


def _example_gallery(df: pd.DataFrame, n: int = 10, seed: int = 0) -> str:
    sample_qids = (
        df[df["strategy"].isin(["decay", "ppr"])]
        .drop_duplicates("qid")
        .sample(n=min(n, df["qid"].nunique()), random_state=seed)["qid"].tolist()
    )
    blocks: List[str] = []
    for qid in sample_qids:
        sub = df[df["qid"] == qid]
        if sub.empty:
            continue
        first = sub.iloc[0]
        q = html.escape(str(first["query"]))
        gold = html.escape(str(first["gold_answer"]))
        rows = []
        for _, r in sub.iterrows():
            label = f"{r['embedder']} | {r['metric']} | {r['strategy']}"
            pred = html.escape(str(r.get("predicted", ""))[:300])
            rationale = html.escape(str(r.get("judge_rationale", ""))[:200])
            mark = "✓" if r.get("judge") else "✗"
            rows.append(
                f"<tr><td>{html.escape(label)}</td><td>{pred}</td>"
                f"<td>F1={r['f1']:.2f}, EM={r['em']:.0f}, judge={mark}</td>"
                f"<td><i>{rationale}</i></td></tr>"
            )
        block = (
            f"<details><summary><b>Q:</b> {q} &mdash; <b>gold:</b> {gold}</summary>"
            f"<table class='results'><thead><tr><th>cell</th><th>predicted</th><th>scores</th><th>judge says</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></details>"
        )
        blocks.append(block)
    return "\n".join(blocks)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ManifoldRAG experiment report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 2em auto; max-width: 1200px; color: #222; padding: 0 1em; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
  h2 {{ margin-top: 2.2em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
  table.results {{ border-collapse: collapse; margin: 1em 0; font-size: 0.9em;
                   width: 100%; }}
  table.results th, table.results td {{ border: 1px solid #ddd; padding: 0.4em 0.6em; }}
  table.results th {{ background: #f4f4f4; position: sticky; top: 0; }}
  table.results tr:nth-child(even) {{ background: #fafafa; }}
  table.summary {{ border-collapse: collapse; }}
  table.summary th, table.summary td {{ padding: 0.4em 0.8em; border: 1px solid #ccc; }}
  details {{ margin: 0.5em 0; padding: 0.5em 1em; background: #f9f9f9;
             border-left: 3px solid #6c8ebf; border-radius: 3px; }}
  details summary {{ cursor: pointer; font-weight: 500; }}
  img.plot {{ max-width: 100%; height: auto; margin: 1em 0;
              border: 1px solid #eee; padding: 4px; background: white; }}
  .meta {{ color: #666; font-size: 0.9em; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>ManifoldRAG experiment report</h1>
<p class="meta">{meta}</p>

<h2>1. Executive summary</h2>
{summary}

<h2>2. Answer accuracy (EM / F1 / LLM-judge)</h2>
{table_acc}

<h2>3. Retrieval (context recall / precision)</h2>
{table_retr}

<h2>4. F1 heatmaps (metric × embedder, per dataset)</h2>
{heatmaps}

<h2>5. F1 by metric and strategy (per dataset)</h2>
{bars}

<h2>6. Retrieval recall vs answer F1</h2>
{scatter}

<h2>7. Dimensionality ablation</h2>
{dim}

<h2>8. Wall-clock</h2>
{table_wall}

<h2>9. Example gallery</h2>
{gallery}

</body>
</html>
"""


def build_report(results_path: str = "results/results.parquet", out_path: str = "results/report.html") -> Path:
    rp = Path(results_path)
    if not rp.exists():
        raise FileNotFoundError(f"results parquet not found at {rp}")
    df = pd.read_parquet(rp)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tab_acc = _table_answer_accuracy(df)
    tab_retr = _table_retrieval(df)
    tab_wall = _table_wallclock(df)

    tab_acc.to_csv(RESULTS_DIR / "answer_accuracy.csv", index=False)
    tab_retr.to_csv(RESULTS_DIR / "retrieval.csv", index=False)
    tab_wall.to_csv(RESULTS_DIR / "wallclock.csv", index=False)

    heatmaps = _plot_heatmap_f1_per_dataset(df)
    bars = _plot_bar_f1_by_metric(df)
    scatter_b64 = _plot_recall_vs_f1(df)
    dim_b64 = _plot_dimensionality(df)

    def _img(b64: Optional[str]) -> str:
        if not b64:
            return "<p><i>(no plot)</i></p>"
        return f'<img class="plot" src="data:image/png;base64,{b64}">'

    heatmaps_html = "\n".join(
        f"<h3>{html.escape(ds)}</h3>{_img(b64)}" for ds, b64 in sorted(heatmaps.items())
    ) or "<p><i>(no heatmaps)</i></p>"
    bars_html = "\n".join(
        f"<h3>{html.escape(ds)}</h3>{_img(b64)}" for ds, b64 in sorted(bars.items())
    ) or "<p><i>(no bar charts)</i></p>"

    meta = (
        f"questions: {df['qid'].nunique()}, rows: {len(df)}, "
        f"datasets: {sorted(df['dataset'].unique())}, "
        f"embedders: {sorted(df['embedder'].unique())}, "
        f"metrics: {sorted(df['metric'].unique())}, "
        f"strategies: {sorted(df['strategy'].unique())}, "
        f"dims: {sorted(df['d'].unique())}"
    )

    html_doc = HTML_TEMPLATE.format(
        meta=html.escape(meta),
        summary=_executive_summary(df),
        table_acc=_df_to_html(tab_acc),
        table_retr=_df_to_html(tab_retr),
        heatmaps=heatmaps_html,
        bars=bars_html,
        scatter=_img(scatter_b64),
        dim=_img(dim_b64),
        table_wall=_df_to_html(tab_wall, floatfmt="{:.2f}"),
        gallery=_example_gallery(df),
    )

    out = Path(out_path)
    out.write_text(html_doc, encoding="utf-8")
    return out
