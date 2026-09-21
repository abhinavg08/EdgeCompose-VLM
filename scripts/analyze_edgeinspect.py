"""EdgeInspect final analysis: tables, residency labels, seed variability, Pareto, figures.

    python scripts/analyze_edgeinspect.py --tag final --main

Reads results/edgeinspect/raw/<tag>__*.jsonl (never modified) and the isolated memory probe
(results/edgeinspect/aggregate/memory_scaling_*.csv). Writes
results/edgeinspect/aggregate/<tag>_*.csv|md (and edgeinspect_results.csv for the optimizer
with --main) and plots/edgeinspect/fig{1..6}_*.png.

Residency (per k x r configuration, from the grid's per-query peak reserved memory, which is
attributable because the runner releases the allocator cache before every query):
  footprint = max peak reserved over all queries + CUDA context / other GPU processes
  VRAM-resident : footprint <= device memory
  spill/degraded: footprint  > device memory (Windows sysmem fallback pages to system RAM)
  OOM/failure   : any query failed with CUDA OOM
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
from edgeinspect.analysis.fewshot import (
    load_rows, macro, macro_auroc_ci, paired_macro_auroc_diff, pareto_configs, per_config,
)
from edgeinspect.datasets.mvtec_loco import CATEGORIES

logger = logging.getLogger("analyze_edgeinspect")
AGG = RESULTS_DIR / "edgeinspect" / "aggregate"
PLOTS = PLOTS_DIR / "edgeinspect"
DEVICE_MB = 8188.0
R_COLORS = {1.0: "#2a78d6", 0.75: "#eb6834", 0.5: "#1baf7a", 0.25: "#eda100"}
R_MARKERS = {1.0: "o", 0.75: "s", 0.5: "D", 0.25: "^"}
K_MARKERS = {1: "o", 2: "s", 4: "D", 8: "^"}
RES_STYLE = {"VRAM-resident": dict(alpha=1.0), "spill/degraded": dict(alpha=1.0), "OOM/failure": dict(alpha=1.0)}


def memory_table() -> pd.DataFrame:
    files = sorted(glob.glob(str(AGG / "memory_scaling_*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _style():
    from edgecompose.analysis.plots import _style as s
    s()


def fig1_auroc_vs_k(tab: pd.DataFrame, seed1: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for i, (r, g) in enumerate(sorted(tab.groupby("retention"), key=lambda x: -x[0])):
        g = g.sort_values("k")
        x = np.log2(g["k"]) + (i - 1.5) * 0.04
        ax.errorbar(x, g["auroc"], yerr=[g["auroc"] - g["auroc_ci_low"], g["auroc_ci_high"] - g["auroc"]],
                    color=R_COLORS[r], marker=R_MARKERS[r], markersize=6.5, markeredgecolor="#fcfcfb", capsize=2.5,
                    lw=2, elinewidth=1, label=f"{int(r * 100)}% visual tokens")
        s1 = seed1[seed1["retention"] == r].sort_values("k")
        if len(s1):
            ax.scatter(np.log2(s1["k"]) + (i - 1.5) * 0.04, s1["auroc"], facecolors="none",
                       edgecolors=R_COLORS[r], marker=R_MARKERS[r], s=55, linewidths=1.3, zorder=4)
    ax.axhline(0.5, color="#c3c2b7", lw=1, ls=":")
    ax.text(0.02, 0.505, "chance", color="#898781", fontsize=7.5)
    ax.set_xticks(np.log2([1, 2, 4, 8]))
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel("number of normal reference images k")
    ax.set_ylabel("AUROC (mean over 5 categories)")
    ax.set_ylim(0.45, 0.85)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Figure 1 - Anomaly discrimination vs reference count", loc="left")
    fig.text(0.01, -0.03, "Filled = reference seed 0 with 95% stratified-bootstrap CI (200 test queries); hollow = "
             "reference seed 1 (variability check). y-axis spans 0.45-0.85.", fontsize=7.2, color="#898781")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _res_scatter(ax, tab, xcol, label_pts=True):
    for _, r in tab.iterrows():
        status = r["residency"]
        color = R_COLORS[r["retention"]]
        mk = K_MARKERS.get(int(r["k"]), "o")
        if status == "VRAM-resident":
            ax.scatter(r[xcol], r["auroc"], color=color, marker=mk, s=60, edgecolor="#fcfcfb", linewidth=1.3, zorder=3)
        elif status == "spill/degraded":
            ax.scatter(r[xcol], r["auroc"], facecolors="none", edgecolors="#d03b3b", marker=mk, s=90, linewidths=1.6,
                       zorder=4)
            ax.scatter(r[xcol], r["auroc"], color=color, marker=mk, s=22, zorder=5)
        key = int(r["k"]) == 8 or (int(r["k"]) == 4 and r["retention"] == 1.0) or status != "VRAM-resident"
        if label_pts and key:
            ax.annotate(f"k={int(r['k'])}, {int(r['retention'] * 100)}%", (r[xcol], r["auroc"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7, color="#52514e")


def _legend_handles():
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=R_COLORS[r], marker="o", lw=0, label=f"{int(r * 100)}% tokens") for r in (1.0, 0.75, 0.5, 0.25)]
    h += [Line2D([], [], color="#52514e", marker=K_MARKERS[k], lw=0, label=f"k = {k}") for k in (1, 2, 4, 8)]
    h += [Line2D([], [], color="#d03b3b", marker="o", markerfacecolor="none", lw=0, markersize=9, label="spill/degraded")]
    return h


def fig2_auroc_vs_vram(tab: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _res_scatter(ax, tab, "vram_mb")
    ax.axvline(DEVICE_MB, color="#d03b3b", lw=1.2, ls="--")
    ax.text(DEVICE_MB - 40, 0.47, "8 GB device memory", color="#d03b3b", fontsize=7.5, ha="right")
    ax.set_xlim(0, max(DEVICE_MB * 1.12, float(tab["vram_mb"].max()) * 1.05))
    ax.set_ylim(0.45, 0.85)
    ax.set_xlabel("peak device VRAM footprint (MB) = peak reserved + CUDA context/other processes")
    ax.set_ylabel("AUROC (mean over 5 categories)")
    ax.legend(handles=_legend_handles(), fontsize=7, loc="lower left", ncol=3)
    ax.set_title("Figure 2 - Anomaly discrimination vs peak VRAM", loc="left")
    fig.text(0.01, -0.02, "Each point = one (k, retention) configuration, seed 0, 200 test queries. y-axis spans "
             "0.45-0.85 (0.5 = chance). Red ring = footprint exceeds the 8 GB card (spills to system RAM).",
             fontsize=7.2, color="#898781")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig3_capacity(cap: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    _style()
    c = cap[cap["attention"] == "sdpa"].sort_values("retention", ascending=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(c))
    bars = ax.bar(x, c["max_k_within_vram"], color=[R_COLORS[r] for r in c["retention"]], width=0.62,
                  edgecolor="#fcfcfb", linewidth=2)
    for xi, (_, r) in zip(x, c.iterrows()):
        ax.text(xi, r["max_k_within_vram"] + 0.3, f"k = {int(r['max_k_within_vram'])}\n{r['footprint_at_max_k_mb']:.0f} MB, "
                f"{r['latency_at_max_k_ms'] / 1e3:.1f} s", ha="center", fontsize=7.5, color="#0b0b0b")
    eager = cap[cap["attention"] == "eager"]
    for _, r in eager.iterrows():
        xi = list(c["retention"]).index(r["retention"]) if r["retention"] in list(c["retention"]) else None
        if xi is not None:
            ax.scatter(xi, r["max_k_within_vram"], marker="_", s=900, color="#0b0b0b", zorder=5, linewidths=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r * 100)}%" for r in c["retention"]])
    ax.set_xlabel("visual-token retention (VisionZip, per image)")
    ax.set_ylabel("max VRAM-resident reference count k")
    ax.set_ylim(0, c["max_k_within_vram"].max() + 4)
    ax.set_yticks(range(0, int(c["max_k_within_vram"].max()) + 5, 4))
    ax.set_title("Figure 3 - Maximum VRAM-resident reference count, RTX 4060 8 GB", loc="left")
    note = "k tested: 1, 2, 4, 8, 12, 16, 24 (pushpins, SDPA; isolated runs incl. ~1.1 GB context/other processes)."
    if len(eager):
        note += " Black tick = eager attention."
    fig.text(0.01, -0.03, note, fontsize=7.2, color="#898781")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig4_types(tab: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharey=True)
    k_colors = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100"}
    for ax, kind in zip(axes, ("structural", "logical")):
        for k, g in tab.groupby("k"):
            g = g.sort_values("retention")
            ax.errorbar(g["retention"] * 100, g[f"auroc_{kind}"],
                        yerr=[g[f"auroc_{kind}"] - g[f"auroc_{kind}_ci_low"], g[f"auroc_{kind}_ci_high"] - g[f"auroc_{kind}"]],
                        color=k_colors[int(k)], marker=K_MARKERS[int(k)], markeredgecolor="#fcfcfb", capsize=2, lw=1.8,
                        elinewidth=0.8, label=f"k = {int(k)}")
        ax.axhline(0.5, color="#c3c2b7", lw=1, ls=":")
        ax.set_xticks([25, 50, 75, 100])
        ax.set_ylim(0.3, 1.0)
        ax.set_xlabel("visual tokens retained per image (%)")
        ax.set_title(f"{kind} anomalies vs normal (n = {int(tab[f'n_{kind}'].iloc[0] * 5)} + 100 normal)", loc="left")
    axes[0].set_ylabel("AUROC (mean over 5 categories)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Figure 4 - Structural vs logical anomalies under compression", x=0.01, ha="left", fontsize=11.5,
                 fontweight="bold")
    fig.text(0.01, -0.03, "95% stratified-bootstrap CIs. y-axis starts at 0.3; dotted line = chance.", fontsize=7.2,
             color="#898781")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig5_heatmap(pc: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _style()
    sel = [(1, 1.0), (1, 0.5), (2, 1.0), (4, 1.0), (4, 0.75), (4, 0.5), (8, 0.75), (8, 0.5), (8, 0.25)]
    cols = [f"k={k}\n{int(r * 100)}%" for k, r in sel]
    d = pc[pc["seed"] == 0]
    mat = np.full((len(CATEGORIES), len(sel)), np.nan)
    for i, c in enumerate(CATEGORIES):
        for j, (k, r) in enumerate(sel):
            v = d[(d["category"] == c) & (d["k"] == k) & (d["retention"] == r)]["auroc"]
            if len(v):
                mat[i, j] = float(v.iloc[0])
    from matplotlib.colors import TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(1.3 + 0.78 * len(sel), 3.6))
    im = ax.imshow(mat, cmap=cmap, norm=TwoSlopeNorm(vmin=0.3, vcenter=0.5, vmax=1.0), aspect="auto")
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels(cols, fontsize=7.5)
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_yticklabels(CATEGORIES, fontsize=8)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, "-" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="#0b0b0b")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("AUROC (neutral = 0.5 chance; orange below, blue above)", fontsize=8)
    ax.set_title("Figure 5 - Per-category AUROC for selected configurations (seed 0, 40 queries each)", loc="left",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig6_latency(tab: pd.DataFrame, front: set, out: Path) -> None:
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _res_scatter(ax, tab, "total_latency_ms_p50")
    if front:
        fr = tab.loc[sorted(front)].sort_values("total_latency_ms_p50")
        ax.scatter(fr["total_latency_ms_p50"], fr["auroc"], s=200, facecolors="none", edgecolors="#0b0b0b",
                   linewidths=1.1, zorder=6)
        ax.step(fr["total_latency_ms_p50"], fr["auroc"], where="post", color="#0b0b0b", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlim(left=0)
    ax.set_ylim(0.45, 0.85)
    ax.set_xlabel("median end-to-end latency per query (ms)")
    ax.set_ylabel("AUROC (mean over 5 categories)")
    ax.legend(handles=_legend_handles(), fontsize=7, loc="lower right", ncol=3)
    ax.set_title("Figure 6 - Quality vs latency (black ring = Pareto-efficient, VRAM-resident only)", loc="left",
                 fontsize=10)
    fig.text(0.01, -0.02, "Pareto front computed among VRAM-resident configurations (ties within 0.01 AUROC / 3% "
             "latency). y-axis spans 0.45-0.85.", fontsize=7.2, color="#898781")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--main", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    setup_logging()
    AGG.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    t = args.tag

    df = load_rows(RESULTS_DIR / "edgeinspect" / "raw", t)
    if df.empty:
        raise SystemExit(f"no rows for tag {t}")
    df["role"] = df.get("role", "test").fillna("test")
    df.to_csv(AGG / f"{t}_per_query.csv", index=False)
    pc = per_config(df)
    pc.to_csv(AGG / f"{t}_per_category.csv", index=False)
    mac = macro(pc)
    mac.to_csv(AGG / f"{t}_macro_by_seed.csv", index=False)

    mem = memory_table()
    context_mb = float(mem["context_overhead_mb"].median()) if not mem.empty else 1100.0

    tab = mac[mac["seed"] == 0].copy().reset_index(drop=True)
    for kind in ("all", "structural", "logical"):
        cis = [macro_auroc_ci(df, 0, int(r["k"]), float(r["retention"]), kind, args.n_boot) for _, r in tab.iterrows()]
        suf = "" if kind == "all" else f"_{kind}"
        tab[f"auroc{suf}_ci_low"] = [c[1] for c in cis]
        tab[f"auroc{suf}_ci_high"] = [c[2] for c in cis]
    # residency from per-config peak reserved (max over categories) + context
    res_max = pc[pc["seed"] == 0].groupby(["k", "retention"])["peak_reserved_mb_max"].max().rename("peak_reserved_grid_mb")
    tab = tab.merge(res_max.reset_index(), on=["k", "retention"], how="left")
    tab["context_overhead_mb"] = context_mb
    tab["vram_mb"] = tab["peak_reserved_grid_mb"] + context_mb
    tab["residency"] = np.where(tab["n_oom"] > 0, "OOM/failure",
                                np.where(tab["vram_mb"] > DEVICE_MB, "spill/degraded", "VRAM-resident"))
    tab["attention_backend"] = "sdpa"
    tab.to_csv(AGG / f"{t}_results.csv", index=False)
    if args.main:
        tab.to_csv(AGG / "edgeinspect_results.csv", index=False)

    # seed variability (configs present for seed 1)
    s1 = mac[mac["seed"] == 1][["k", "retention", "auroc", "f1_cal", "auroc_structural", "auroc_logical"]]
    var = pd.DataFrame()
    if len(s1):
        var = tab[["k", "retention", "auroc", "f1_cal", "auroc_structural", "auroc_logical"]].merge(
            s1, on=["k", "retention"], suffixes=("_seed0", "_seed1"))
        for m in ("auroc", "f1_cal", "auroc_structural", "auroc_logical"):
            var[f"{m}_mean"] = (var[f"{m}_seed0"] + var[f"{m}_seed1"]) / 2
            var[f"{m}_diff"] = var[f"{m}_seed1"] - var[f"{m}_seed0"]
        var["auroc_sd_2seeds"] = var["auroc_diff"].abs() / np.sqrt(2)
        pcs = pc.pivot_table(index=["category", "k", "retention"], columns="seed", values="auroc").dropna()
        if 1 in pcs.columns:
            pcs["diff"] = pcs[1] - pcs[0]
            per_cat = pcs.reset_index().groupby(["k", "retention"])["diff"].agg(
                per_category_abs_diff_mean=lambda s: s.abs().mean(), per_category_abs_diff_max=lambda s: s.abs().max())
            var = var.merge(per_cat.reset_index(), on=["k", "retention"], how="left")
        var.to_csv(AGG / f"{t}_seed_variability.csv", index=False)

    # capacity from isolated probe
    cap = pd.DataFrame()
    if not mem.empty:
        rows = []
        for (attn, r), g in mem.groupby(["attention_backend", "retention"]):
            okk, bad = g[g["status"] == "ok"], g[g["status"] != "ok"]
            best = okk.loc[okk["k"].idxmax()] if len(okk) else None
            fail = bad.loc[bad["k"].idxmin()] if len(bad) else None
            rows.append({"attention": attn, "retention": r, "k_tested": "/".join(str(int(x)) for x in sorted(g["k"])),
                         "max_k_within_vram": int(best["k"]) if best is not None else 0,
                         "footprint_at_max_k_mb": float(best["device_footprint_mb"]) if best is not None else np.nan,
                         "latency_at_max_k_ms": float(best["latency_ms_median"]) if best is not None else np.nan,
                         "first_non_resident_k": int(fail["k"]) if fail is not None else None,
                         "status_first_non_resident": fail["status"] if fail is not None else None,
                         "footprint_first_non_resident_mb": float(fail["device_footprint_mb"]) if fail is not None else np.nan,
                         "latency_first_non_resident_ms": float(fail["latency_ms_median"]) if fail is not None
                         and pd.notna(fail["latency_ms_median"]) else np.nan})
        cap = pd.DataFrame(rows)
        cap.to_csv(AGG / f"{t}_reference_capacity.csv", index=False)

    # Pareto among VRAM-resident configs: AUROC vs latency (and vs VRAM)
    resident = tab[tab["residency"] == "VRAM-resident"]
    tol = {"auroc": 0.01, "total_latency_ms_p50": 0.03 * float(tab["total_latency_ms_p50"].min()), "vram_mb": 50.0}
    f_lat = {resident.index[i] for i in pareto_configs(resident.reset_index(drop=True), "auroc", ["total_latency_ms_p50"], tol)}
    f_mem = {resident.index[i] for i in pareto_configs(resident.reset_index(drop=True), "auroc", ["vram_mb"], tol)}
    par = tab.loc[sorted(f_lat | f_mem), ["k", "retention", "auroc", "auroc_ci_low", "auroc_ci_high", "f1_cal",
                                          "total_latency_ms_p50", "ttft_ms_p50", "vram_mb", "residency"]].copy()
    par["front_auroc_latency"] = [i in f_lat for i in par.index]
    par["front_auroc_vram"] = [i in f_mem for i in par.index]
    par.to_csv(AGG / f"{t}_pareto.csv", index=False)

    # paired contrasts (same queries): reference scaling, compression tolerance, capacity trade
    contrasts = []
    for r in (1.0, 0.75, 0.5, 0.25):
        contrasts.append(("more references (k=1 -> 8)", (0, 1, r), (0, 8, r)))
    for r in (1.0, 0.75, 0.5, 0.25):
        contrasts.append(("more references (k=4 -> 8)", (0, 4, r), (0, 8, r)))
    for k in (1, 2, 4, 8):
        for r in (0.75, 0.5, 0.25):
            contrasts.append((f"compression at k={k} (100% -> {int(r * 100)}%)", (0, k, 1.0), (0, k, r)))
    contrasts.append(("capacity trade: best full-token resident (k=4,100%) -> k=8,75%", (0, 4, 1.0), (0, 8, 0.75)))
    contrasts.append(("capacity trade: k=4,100% -> k=8,50%", (0, 4, 1.0), (0, 8, 0.5)))
    contrasts.append(("capacity trade: k=4,100% -> k=8,25%", (0, 4, 1.0), (0, 8, 0.25)))
    ctab = pd.DataFrame([{"contrast": name, **paired_macro_auroc_diff(df, a, b, args.n_boot)} for name, a, b in contrasts])
    ctab.to_csv(AGG / f"{t}_paired_contrasts.csv", index=False)

    # tables
    main_cols =["k", "retention", "auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "f1_cal", "precision_cal",
                 "recall_cal", "specificity_cal", "accuracy_cal", "f1", "pred_anomalous_ratio", "ttft_ms_p50",
                 "total_latency_ms_p50", "total_latency_ms_p95", "peak_allocated_mb_max", "vram_mb", "residency",
                 "visual_tokens_after"]
    type_cols = ["k", "retention", "n_structural", "n_logical", "auroc_structural", "auroc_structural_ci_low",
                 "auroc_structural_ci_high", "auroc_logical", "auroc_logical_ci_low", "auroc_logical_ci_high",
                 "auprc_structural", "auprc_logical", "f1_cal_structural", "f1_cal_logical"]
    typ = tab[type_cols].copy()
    typ["n_structural"] = typ["n_structural"] * 5
    typ["n_logical"] = typ["n_logical"] * 5
    pc0 = pc[pc["seed"] == 0][["category", "k", "retention", "auroc", "auroc_structural", "auroc_logical", "f1_cal",
                               "total_latency_ms_p50", "peak_reserved_mb_max"]]
    counts = df[df["role"] == "test"].groupby(["reference_seed", "category", "reference_count", "token_retention"]).size()
    expected = {0: 16 * 5, 1: None}
    with open(AGG / f"{t}_tables.md", "w", encoding="utf-8") as f:
        f.write(f"# EdgeInspect final results ({t})\n\n")
        f.write(f"Rows: {len(df)} ({(df['role'] == 'test').sum()} test, {(df['role'] == 'calibration').sum()} calibration); "
                f"status counts {df['status'].value_counts().to_dict()}; conditions (seed, category, k, r): "
                f"{len(counts)}; test queries per condition: {sorted(set(counts.tolist()))}.\n\n")
        f.write("## Table A - Main results, seed 0 (macro over 5 categories; 200 test queries per configuration)\n\n"
                "AUROC with 95% stratified-bootstrap CI. `*_cal` metrics use the normal-only calibrated threshold "
                "(90th percentile of 10 validation/good scores per category and configuration) - a deployment "
                "operating point, not an F1-optimal threshold. `f1` / `pred_anomalous_ratio` = raw greedy answer. "
                f"`vram_mb` = max per-query peak reserved + {context_mb:.0f} MB CUDA context/other processes.\n\n")
        f.write(md(tab[main_cols], ".3f") + "\n\n")
        f.write("## Table B - Structural vs logical (normals + that type; counts over 5 categories)\n\n" + md(typ, ".3f") + "\n\n")
        if not var.empty:
            f.write("## Table C - Reference-seed variability (seed 0 vs seed 1, macro AUROC)\n\n" + md(var, ".3f") + "\n\n")
        if not cap.empty:
            f.write("## Table D - Reference capacity (isolated probe, pushpins)\n\n" + md(cap, ".1f") + "\n\n")
        f.write("## Table E - Pareto-efficient VRAM-resident configurations\n\n" + md(par, ".3f") + "\n\n")
        f.write("## Table G - Paired contrasts of macro AUROC (same 200 queries; stratified paired bootstrap)\n\n"
                "`p_boot_le_0` = share of bootstrap replicates with difference <= 0 (one-sided).\n\n"
                + md(ctab, ".3f") + "\n\n")
        f.write("## Table F - Per category (seed 0)\n\n" + md(pc0, ".3f") + "\n")
    logger.info("wrote %s", AGG / f"{t}_tables.md")

    s1p = mac[mac["seed"] == 1]
    fig1_auroc_vs_k(tab, s1p, PLOTS / "fig1_auroc_vs_references.png")
    fig2_auroc_vs_vram(tab, PLOTS / "fig2_auroc_vs_vram.png")
    if not cap.empty:
        fig3_capacity(cap, PLOTS / "fig3_reference_capacity.png")
    fig4_types(tab, PLOTS / "fig4_structural_vs_logical.png")
    fig5_heatmap(pc, PLOTS / "fig5_category_heatmap.png")
    fig6_latency(tab, f_lat, PLOTS / "fig6_auroc_vs_latency.png")
    logger.info("figures in %s", PLOTS)


if __name__ == "__main__":
    main()
