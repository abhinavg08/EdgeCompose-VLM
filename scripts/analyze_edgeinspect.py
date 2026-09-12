"""EdgeInspect stages A7/A8: aggregate raw runs, tables, Pareto fronts and figures.

    python scripts/analyze_edgeinspect.py --tag dev
    python scripts/analyze_edgeinspect.py --tag final --main     # also writes edgeinspect_results.csv for the optimizer

Outputs: results/edgeinspect/aggregate/<tag>_*.csv|md, plots/edgeinspect/<tag>_fig*.png
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

import edgeinspect  # noqa: F401
from analyze import md
from edgecompose.utils import PLOTS_DIR, RESULTS_DIR, setup_logging
from edgeinspect.analysis.fewshot import across_seeds, aufc_table, load_rows, macro, pareto_configs, per_config
from edgeinspect.datasets.mvtec_loco import CATEGORIES

logger = logging.getLogger("analyze_edgeinspect")
AGG = RESULTS_DIR / "edgeinspect" / "aggregate"
PLOTS = PLOTS_DIR / "edgeinspect"
DEVICE_MB = 8188.0
K_COLORS = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100", 12: "#e87ba4", 16: "#4a3aa7", 24: "#e34948"}
R_COLORS = {1.0: "#2a78d6", 0.75: "#eb6834", 0.5: "#1baf7a", 0.25: "#eda100"}
MARKERS = {1: "o", 2: "s", 4: "D", 8: "^", 12: "v", 16: "P", 24: "X", 1.0: "o", 0.75: "s", 0.5: "D", 0.25: "^"}


def memory_table() -> pd.DataFrame:
    files = sorted(glob.glob(str(AGG / "memory_scaling_*.csv")))
    return pd.concat([pd.read_csv(f) for f in files]) if files else pd.DataFrame()


def fig_quality_vs_retention(tab: pd.DataFrame, out: Path, metric: str) -> None:
    from edgecompose.analysis.plots import _style, INK2
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, g in tab.groupby("k"):
        g = g.sort_values("retention")
        x = g["retention"] * 100
        ax.plot(x, g[metric], color=K_COLORS.get(k, INK2), marker=MARKERS.get(k, "o"), markeredgecolor="#fcfcfb",
                markeredgewidth=1.3, label=f"{k}-shot")
        if f"{metric}_std" in g and g[f"{metric}_std"].notna().any():
            ax.fill_between(x, g[metric] - g[f"{metric}_std"], g[metric] + g[f"{metric}_std"],
                            color=K_COLORS.get(k, INK2), alpha=0.12, linewidth=0)
    ax.set_xticks([25, 50, 75, 100])
    ax.set_xlabel("visual tokens retained per image (%)")
    ax.set_ylabel(f"macro {metric.replace('_', '-').upper()} over categories")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, title="normal references", title_fontsize=8)
    ax.set_title(f"Figure 1 - {metric.replace('_', '-').upper()} vs visual-token retention", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_quality_vs_cost(tab: pd.DataFrame, cost: str, xlabel: str, out: Path, title: str, metric: str,
                        vline: float = None, vline_label: str = "", front: set = frozenset()) -> None:
    from edgecompose.analysis.plots import _style, INK, INK2
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for k, g in tab.groupby("k"):
        ax.scatter(g[cost], g[metric], color=K_COLORS.get(k, INK2), marker=MARKERS.get(k, "o"), s=48,
                   edgecolor="#fcfcfb", linewidth=1.3, label=f"{k}-shot", zorder=3)
        for _, r in g.iterrows():
            ax.annotate(f"{int(r['retention'] * 100)}%", (r[cost], r[metric]), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=INK2)
    if front:
        fr = tab.loc[sorted(front)].sort_values(cost)
        ax.scatter(fr[cost], fr[metric], s=170, facecolors="none", edgecolors=INK, linewidths=1.1, zorder=4,
                   label="Pareto-efficient")
        ax.step(fr[cost], fr[metric], where="post", color=INK, lw=0.8, ls="--", alpha=0.6)
    if vline:
        ax.axvline(vline, color="#d03b3b", lw=1.2, ls="--")
        ax.text(vline, ax.get_ylim()[0] + 0.02, f" {vline_label}", color="#d03b3b", fontsize=7.5, rotation=90,
                va="bottom", ha="right")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"macro {metric.replace('_', '-').upper()}")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_memory_scaling(mem: pd.DataFrame, out: Path) -> None:
    from edgecompose.analysis.plots import _style, INK2
    import matplotlib.pyplot as plt
    if mem.empty:
        return
    _style()
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    sub = mem[mem["attention_backend"] == "sdpa"] if "attention_backend" in mem else mem
    top = max(DEVICE_MB * 1.08, float(sub["device_footprint_mb"].max()) * 1.05)
    for r, g in sub.groupby("retention"):
        g = g.sort_values("k")
        ok = g[g["status"].isin(["ok", "over_vram"])]
        ax.plot(ok["k"], ok["device_footprint_mb"], color=R_COLORS.get(r, INK2), marker=MARKERS.get(r, "o"),
                markeredgecolor="#fcfcfb", label=f"{int(r * 100)}% tokens")
        over = g[g["status"] == "over_vram"]
        if len(over):
            ax.scatter(over["k"], over["device_footprint_mb"], s=150, facecolors="none", edgecolors="#d03b3b",
                       linewidths=1.2, zorder=5)
        oom = g[g["status"] == "OOM"]
        if len(oom):
            ax.scatter(oom["k"], [top * 0.97] * len(oom), marker="x", s=60, color=R_COLORS.get(r, INK2), zorder=5)
    ax.axhline(DEVICE_MB, color="#d03b3b", lw=1.2, ls="--")
    ax.text(ax.get_xlim()[0], DEVICE_MB, " 8 GB device (red ring = exceeds VRAM, spilled to system RAM; x = OOM)",
            color="#d03b3b", fontsize=7, va="bottom")
    ax.set_ylim(0, top)
    ax.set_xlabel("number of normal reference images k")
    ax.set_ylabel("device VRAM footprint (MB)\npeak reserved + CUDA context/other processes")
    ax.legend(fontsize=8)
    ax.set_title("Figure 4 - Reference count vs peak VRAM (isolated runs)", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_struct_vs_logical(tab: pd.DataFrame, out: Path) -> None:
    from edgecompose.analysis.plots import _style, INK2
    import matplotlib.pyplot as plt
    _style()
    ks = sorted(tab["k"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    for ax, kind in zip(axes, ("structural", "logical")):
        col = f"f1_{kind}"
        for k in ks:
            g = tab[tab["k"] == k].sort_values("retention")
            ax.plot(g["retention"] * 100, g[col], color=K_COLORS.get(k, INK2), marker=MARKERS.get(k, "o"),
                    markeredgecolor="#fcfcfb", label=f"{k}-shot")
        ax.set_xticks([25, 50, 75, 100])
        ax.set_ylim(0, 1)
        ax.set_xlabel("visual tokens retained per image (%)")
        ax.set_title(f"{kind} anomalies (vs normal)", loc="left")
    axes[0].set_ylabel("macro F1")
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure 5 - Structural vs logical anomaly F1 under compression", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_category_heatmap(pc_mean: pd.DataFrame, out: Path, metric: str) -> None:
    from edgecompose.analysis.plots import _style
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _style()
    pc_mean = pc_mean.copy()
    pc_mean["cfg"] = pc_mean.apply(lambda r: f"k={int(r['k'])}\nr={int(r['retention'] * 100)}%", axis=1)
    order = pc_mean.sort_values(["k", "retention"], ascending=[True, False])["cfg"].unique()
    pv = pc_mean.pivot_table(index="category", columns="cfg", values=metric).reindex(
        index=[c for c in CATEGORIES if c in pc_mean["category"].unique()], columns=order)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#86b6ef", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * pv.shape[1], 1.2 + 0.55 * pv.shape[0]))
    im = ax.imshow(pv.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(pv.shape[1]))
    ax.set_xticklabels(pv.columns, fontsize=7)
    ax.set_yticks(range(pv.shape[0]))
    ax.set_yticklabels(pv.index, fontsize=8)
    ax.grid(False)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if v > 0.6 else "#0b0b0b")
            else:
                ax.text(j, i, "OOM", ha="center", va="center", fontsize=6.5, color="#d03b3b")
    fig.colorbar(im, ax=ax, shrink=0.8, label=metric.replace("_", "-").upper())
    ax.set_title(f"Figure 6 - Per-category {metric.replace('_', '-').upper()} (mean over seeds)", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--main", action="store_true")
    ap.add_argument("--metric", default="f1", choices=["f1", "f1_max", "auroc"])
    args = ap.parse_args()
    setup_logging()
    AGG.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = load_rows(RESULTS_DIR / "edgeinspect" / "raw", args.tag)
    if df.empty:
        raise SystemExit(f"no rows for tag {args.tag}")
    logger.info("%d rows; status counts: %s", len(df), df["status"].value_counts().to_dict())
    df.to_csv(AGG / f"{args.tag}_per_query.csv", index=False)
    pc = per_config(df)
    pc.to_csv(AGG / f"{args.tag}_per_category.csv", index=False)
    mac = macro(pc)
    mac.to_csv(AGG / f"{args.tag}_macro_by_seed.csv", index=False)
    tab = across_seeds(mac)
    # memory: isolated device footprint when measured for (k, r), else per-query peak allocated max
    mem = memory_table()
    tab["vram_mb"] = tab["peak_allocated_mb_max"]
    tab["vram_source"] = "peak allocated (interleaved run)"
    if not mem.empty:
        m = mem[(mem["status"].isin(["ok", "over_vram"])) & (mem.get("attention_backend", "sdpa") == "sdpa")]
        m = m.groupby(["k", "retention"])["device_footprint_mb"].max().rename("device_footprint_mb").reset_index()
        tab = tab.merge(m, on=["k", "retention"], how="left")
        has = tab["device_footprint_mb"].notna()
        tab.loc[has, "vram_mb"] = tab.loc[has, "device_footprint_mb"]
        tab.loc[has, "vram_source"] = "isolated device footprint"
    tab["attention_backend"] = "sdpa"
    tab.to_csv(AGG / f"{args.tag}_results.csv", index=False)
    if args.main:
        tab.to_csv(AGG / "edgeinspect_results.csv", index=False)

    q = args.metric
    tol = {q: 0.01, "total_latency_ms_p50": 0.03 * float(tab["total_latency_ms_p50"].min()),
           "vram_mb": 0.01 * float(tab["vram_mb"].min())}
    feas = tab[tab["feasible_all"]].reset_index(drop=True)
    f_lat = set(pareto_configs(feas, q, ["total_latency_ms_p50"], tol))
    f_mem = set(pareto_configs(feas, q, ["vram_mb"], tol))
    f_all = set(pareto_configs(feas, q, ["total_latency_ms_p50", "vram_mb"], tol))
    f_tok = set(pareto_configs(feas, q, ["visual_tokens_after"], {q: 0.01, "visual_tokens_after": 1.0}))
    par = feas.loc[sorted(f_all | f_lat | f_mem | f_tok),
                   ["k", "retention", q, "total_latency_ms_p50", "vram_mb", "visual_tokens_after"]].copy()
    par["front_q_lat"] = [i in f_lat for i in par.index]
    par["front_q_vram"] = [i in f_mem for i in par.index]
    par["front_q_tokens"] = [i in f_tok for i in par.index]
    par["front_q_lat_vram"] = [i in f_all for i in par.index]
    par.to_csv(AGG / f"{args.tag}_pareto.csv", index=False)

    au = aufc_table(mac.groupby(["k", "retention"])[["f1_max", "f1"]].mean().reset_index(), "f1_max")
    pc_mean = pc.groupby(["category", "k", "retention"])[[c for c in ("f1", "f1_max", "auroc") if c in pc]].mean().reset_index()
    oom = df.groupby(["category", "reference_count", "token_retention"])["status"].apply(
        lambda s: f"{(s == 'OOM').sum()}/{len(s)}").unstack("category") if (df["status"] == "OOM").any() else pd.DataFrame()

    show = ["k", "retention", "n_seeds", "f1", "f1_std", "precision", "recall", "f1_max", "auroc", "f1_structural",
            "f1_logical", "auroc_structural", "auroc_logical", "pred_anomalous_ratio", "ttft_ms_p50",
            "total_latency_ms_p50", "vram_mb", "visual_tokens_after", "feasible_all"]
    with open(AGG / f"{args.tag}_tables.md", "w", encoding="utf-8") as f:
        f.write(f"# EdgeInspect results ({args.tag})\n\n## Macro over categories (mean over reference seeds)\n\n")
        f.write(md(tab[[c for c in show if c in tab.columns]], ".3f") + "\n\n")
        f.write(f"## Pareto-efficient (feasible) configurations on {q}\n\n" + md(par, ".3f") + "\n\n")
        if not au.empty:
            f.write("## AUFC (area under F1-max vs log2 k)\n\n" + md(au, ".3f") + "\n\n")
        f.write("## Per category (mean over seeds)\n\n" + md(pc_mean, ".3f") + "\n\n")
        if not oom.empty:
            f.write("## OOM counts (OOM / attempted queries)\n\n" + md(oom.reset_index().fillna("-")) + "\n")
        if not mem.empty:
            f.write("\n\n## Isolated memory scaling\n\n" + md(mem, ".1f") + "\n")
    logger.info("wrote %s", AGG / f"{args.tag}_tables.md")

    t = args.tag
    fig_quality_vs_retention(tab, PLOTS / f"{t}_fig1_{q}_vs_retention.png", q)
    if "f1_max" in tab and q != "f1_max":
        fig_quality_vs_retention(tab, PLOTS / f"{t}_fig1b_f1max_vs_retention.png", "f1_max")
    fig_quality_vs_cost(feas, "vram_mb", "peak VRAM (MB; isolated device footprint where measured)",
                        PLOTS / f"{t}_fig2_{q}_vs_vram.png", "Figure 2 - Quality vs peak VRAM", q,
                        vline=DEVICE_MB, vline_label="8 GB device", front=f_mem)
    fig_quality_vs_cost(feas, "total_latency_ms_p50", "median end-to-end latency (ms)",
                        PLOTS / f"{t}_fig3_{q}_vs_latency.png", "Figure 3 - Quality vs latency (ringed = Pareto)", q,
                        front=f_lat)
    fig_memory_scaling(mem, PLOTS / f"{t}_fig4_references_vs_vram.png")
    fig_struct_vs_logical(tab, PLOTS / f"{t}_fig5_structural_vs_logical.png")
    fig_category_heatmap(pc_mean, PLOTS / f"{t}_fig6_category_heatmap.png", q)
    logger.info("figures in %s", PLOTS)


if __name__ == "__main__":
    main()
