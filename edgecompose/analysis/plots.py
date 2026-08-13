"""Figures for the report (matplotlib, static PNG).

Encoding: series identity = (attention backend, token method) with a validated
categorical palette (checked with the dataviz validator: CVD/normal-vision separation
pass; contrast relief via direct labels, distinct markers and the companion tables).
Latency/memory axes start at 0; quality axes that do not start at 0 say so on the axis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from edgecompose.analysis.pareto import pareto_front  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = {  # (backend, method, dispatch) -> (label, color, marker); categorical slots in fixed order
    ("sdpa", "visionzip", "upstream"): ("SDPA + VisionZip", "#2a78d6", "o"),
    ("eager", "visionzip", "upstream"): ("Eager + VisionZip", "#eb6834", "s"),
    ("sdpa", "visionzip", "tuned"): ("SDPA + VisionZip + tuned AWQ dispatch", "#1baf7a", "D"),
    ("sdpa", "uniform", "upstream"): ("SDPA + uniform (control)", "#eda100", "^"),
    ("flash_attention_2", "visionzip", "upstream"): ("FA2 + VisionZip", "#e87ba4", "P"),
}
STAGE_COLORS = {"preprocess_ms": "#e87ba4", "vision_ms": "#2a78d6", "compress_ms": "#eda100",
                "prefill_ms": "#eb6834", "decode_ms": "#1baf7a"}
STAGE_LABELS = {"preprocess_ms": "preprocess (CPU)", "vision_ms": "vision encoder", "compress_ms": "fusion + compression",
                "prefill_ms": "LLM prefill", "decode_ms": "decode"}
DS_TITLE = {"textvqa": "TextVQA (VQA accuracy)", "pope": "POPE (accuracy)"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
    })


def _series_key(row) -> tuple:
    """Series identity; the 100% (no compression) point belongs to its arm's VisionZip line."""
    method = "visionzip" if row["token_method"] == "none" else row["token_method"]
    thr = row.get("awq_dequant_threshold", 1024)
    dispatch = "upstream" if (thr is None or pd.isna(thr) or int(thr) == 1024) else "tuned"
    return row["attention_backend"], method, dispatch


def _series_rows(sub: pd.DataFrame, key: tuple) -> pd.DataFrame:
    """Rows of one series, plus the matching 100% baseline for the uniform control line."""
    s = sub[[_series_key(r) == key for _, r in sub.iterrows()]]
    if key[1] == "uniform" and not s.empty:
        base = sub[[_series_key(r) == (key[0], "visionzip", key[2]) and r["token_method"] == "none"
                    for _, r in sub.iterrows()]]
        s = pd.concat([base, s])
    return s


def _points(agg: pd.DataFrame) -> List[Dict]:
    pts = []
    for _, r in agg.iterrows():
        pts.append({**r.to_dict(), "series": _series_key(r)})
    return pts


def _quality_axis_note(ax, values) -> None:
    lo = min(values)
    if lo > 0.15:
        ax.set_ylabel(ax.get_ylabel() + "  (axis does not start at 0)")


def _scatter_tradeoff(agg: pd.DataFrame, xcol: str, xlabel: str, out: Path, title: str,
                      tols: Optional[Dict[str, Dict]] = None) -> None:
    _style()
    dsets = list(agg["dataset"].unique())
    fig, axes = plt.subplots(1, len(dsets), figsize=(5.8 * len(dsets), 4.4), squeeze=False)
    for ax, ds in zip(axes[0], dsets):
        sub = agg[agg["dataset"] == ds].reset_index(drop=True)
        rows = sub.to_dict(orient="records")
        tol = (tols or {}).get(ds, 0.0)
        front = set(pareto_front(rows, [("quality", "max"), (xcol, "min")], tol))
        for key, (label, color, marker) in SERIES.items():
            s = _series_rows(sub, key).sort_values("token_retention", ascending=False)
            if s.empty:
                continue
            ax.plot(s[xcol], s["quality"], color=color, lw=1.2, alpha=0.5, zorder=1)
            own = s if key[1] != "uniform" else s[s["token_method"] == "uniform"]
            ax.scatter(own[xcol], own["quality"], color=color, marker=marker, s=46, edgecolor=SURFACE, linewidth=1.5,
                       zorder=3, label=label)
            ax.errorbar(own[xcol], own["quality"], yerr=[own["quality"] - own["quality_ci_low"],
                                                         own["quality_ci_high"] - own["quality"]],
                        fmt="none", ecolor=color, alpha=0.35, lw=1, zorder=2)
            offset = {"sdpa": (5, 5), "eager": (5, -11), "flash_attention_2": (5, -11)}[key[0]]
            if key[2] == "tuned":
                offset = (-24, -11)
            if key[1] == "uniform":
                offset = (-26, 5)
            for _, r in s.iterrows():
                if key[1] == "uniform" and r["token_method"] == "none":
                    continue  # baseline already labelled by its own series
                ax.annotate(f"{int(round(r['token_retention'] * 100))}%", (r[xcol], r["quality"]), textcoords="offset points",
                            xytext=offset, fontsize=7.5, color=INK2)
        fr = sub.loc[sorted(front)].sort_values(xcol)
        ax.scatter(fr[xcol], fr["quality"], s=170, facecolors="none", edgecolors=INK, linewidths=1.1, zorder=4,
                   label="Pareto-efficient")
        ax.step(fr[xcol], fr["quality"], where="post", color=INK, lw=0.8, ls="--", alpha=0.6, zorder=0)
        ax.set_xlim(left=0)
        ax.set_title(DS_TITLE.get(ds, ds), loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("quality")
        _quality_axis_note(ax, sub["quality_ci_low"])
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.text(0.01, -0.02, "Labels = visual-token retention. Error bars = 95% bootstrap CI of quality. Ringed = "
             "non-dominated on (quality, x) with noise tolerance (0.5 pp quality; 3% latency/energy, 1% memory).",
             fontsize=7.5, color=MUTED)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_quality_vs_latency(agg: pd.DataFrame, out: Path, tols=None) -> None:
    _scatter_tradeoff(agg, "total_latency_ms_p50", "end-to-end latency, median (ms)", out,
                      "Figure 1 - Quality vs end-to-end latency (RTX 4060 Laptop 8 GB, batch 1)", tols)


def fig_quality_vs_memory(agg: pd.DataFrame, out: Path, tols=None) -> None:
    _scatter_tradeoff(agg, "peak_allocated_mb_p50", "peak CUDA memory allocated, median (MB)", out,
                      "Figure 2 - Quality vs peak GPU memory", tols)


def fig_quality_vs_energy(agg: pd.DataFrame, out: Path, tols=None) -> None:
    if agg["energy_j_median"].isna().all():
        return
    _scatter_tradeoff(agg, "energy_j_median", "GPU energy per query, median (J, NVML sampled)", out,
                      "Figure 6 - Quality vs energy per query", tols)


def fig_stage_breakdown(agg: pd.DataFrame, out: Path, configs: Optional[List[str]] = None) -> None:
    _style()
    dsets = list(agg["dataset"].unique())
    n_cfg = agg["config_name"].nunique() if not configs else len(configs)
    fig, axes = plt.subplots(len(dsets), 1, figsize=(8.6, 1.4 + 0.36 * n_cfg * len(dsets)), squeeze=False)
    for ax, ds in zip(axes[:, 0], dsets):
        sub = agg[agg["dataset"] == ds]
        if configs:
            sub = sub[sub["config_name"].isin(configs)]
        sub = sub.iloc[::-1]
        left = np.zeros(len(sub))
        y = np.arange(len(sub))
        for st, color in STAGE_COLORS.items():
            vals = sub[f"{st}_p50"].fillna(0).values
            ax.barh(y, vals, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.66,
                    label=STAGE_LABELS[st])
            left += vals
        for yi, tot, e2e in zip(y, left, sub["total_latency_ms_p50"]):
            ax.text(tot + 12, yi, f"{e2e:.0f} ms", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["config_name"], fontsize=8)
        ax.set_xlim(0, left.max() * 1.15)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{DS_TITLE.get(ds, ds).split(' (')[0]} - median stage latency (stacked medians)", loc="left")
        ax.set_xlabel("ms")
    axes[-1, 0].legend(ncol=5, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.16 / len(dsets) - 0.06))
    fig.suptitle("Figure 3 - Stage-wise latency", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.text(0.01, -0.03, "Bars stack per-stage medians; labels give the median end-to-end latency (the two differ slightly).",
             fontsize=7.5, color=MUTED)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _retention_lines(agg: pd.DataFrame, ycol: str, ylabel: str, out: Path, title: str, ci: bool = False,
                     zero: bool = True, lo_col: Optional[str] = None, hi_col: Optional[str] = None) -> None:
    _style()
    dsets = list(agg["dataset"].unique())
    fig, axes = plt.subplots(1, len(dsets), figsize=(5.2 * len(dsets), 3.9), squeeze=False)
    for ax, ds in zip(axes[0], dsets):
        sub = agg[agg["dataset"] == ds]
        for key, (label, color, marker) in SERIES.items():
            s = _series_rows(sub, key).sort_values("token_retention")
            if s.empty or (s["token_method"] == key[1]).sum() == 0:
                continue
            x = s["token_retention"] * 100
            ax.plot(x, s[ycol], color=color, marker=marker, markersize=6.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.4, label=label)
            if ci and lo_col and hi_col:
                ax.fill_between(x, s[lo_col], s[hi_col], color=color, alpha=0.12, linewidth=0)
        ax.set_xticks([25, 50, 75, 100])
        ax.set_xlabel("visual tokens retained (%)")
        ax.set_ylabel(ylabel)
        if zero:
            ax.set_ylim(bottom=0)
        else:
            _quality_axis_note(ax, sub[lo_col] if lo_col else sub[ycol])
        ax.set_title(DS_TITLE.get(ds, ds).split(" (")[0], loc="left")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_retention_vs_quality(agg: pd.DataFrame, out: Path) -> None:
    _retention_lines(agg, "quality", "quality", out, "Figure 4 - Visual-token retention vs benchmark score",
                     ci=True, zero=False, lo_col="quality_ci_low", hi_col="quality_ci_high")


def fig_retention_vs_ttft(agg: pd.DataFrame, out: Path) -> None:
    _retention_lines(agg, "ttft_ms_p50", "TTFT, median (ms)", out, "Figure 5 - Visual-token retention vs time-to-first-token")


def fig_interaction_heatmap(inter: pd.DataFrame, out: Path) -> None:
    """Rows = retention, cols = metric; I(A,B) with diverging blue(synergy)-gray-orange(interference)."""
    if inter.empty:
        return
    _style()
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
    dsets = list(inter["dataset"].unique())
    if "factor_B" not in inter.columns:
        inter = inter.assign(factor_B="SDPA (vs eager)")
    factors = list(dict.fromkeys(inter["factor_B"]))
    lim = max(0.05, float(np.nanmax(np.abs(inter["I"].values))))
    fig, axes = plt.subplots(len(factors), len(dsets), figsize=(5.2 * len(dsets), 2.5 * len(factors) + 0.6),
                             squeeze=False)
    im = None
    for fi, fb in enumerate(factors):
        for ax, ds in zip(axes[fi], dsets):
            sub = inter[(inter["dataset"] == ds) & (inter["factor_B"] == fb)]
            if sub.empty:
                ax.axis("off")
                continue
            pv = sub.pivot(index="retention", columns="metric", values="I").sort_index(ascending=False)
            pv = pv[[c for c in ("TTFT", "prefill", "vision", "E2E latency") if c in pv.columns]]
            im = ax.imshow(pv.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
            ax.set_xticks(range(len(pv.columns)))
            ax.set_xticklabels(pv.columns, fontsize=8)
            ax.set_yticks(range(len(pv.index)))
            ax.set_yticklabels([f"{int(r * 100)}%" for r in pv.index])
            ax.grid(False)
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    cell = sub[(sub["retention"] == pv.index[i]) & (sub["metric"] == pv.columns[j])].iloc[0]
                    sig = "" if (cell["I_ci_low"] <= 0 <= cell["I_ci_high"]) else "*"
                    ax.text(j, i, f"{pv.values[i, j]:+.3f}{sig}", ha="center", va="center", fontsize=8, color=INK)
            ax.set_title(f"{DS_TITLE.get(ds, ds).split(' (')[0]} | B = {fb}", loc="left", fontsize=9.5)
            ax.set_ylabel("A: VisionZip retention")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="I = R_AB - R_A R_B  (<0 synergy, >0 interference)")
    fig.suptitle("Figure 7 - Composition interaction of token compression (A) with attention / AWQ dispatch (B)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", y=1.0)
    fig.text(0.01, -0.03, "* = 95% paired-bootstrap CI of I excludes 0. Baseline = the arm without B at 100% tokens "
             "(eager for SDPA; upstream dispatch for tuned dispatch).", fontsize=7.5, color=MUTED)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_prefill_vs_length(df: pd.DataFrame, out: Path) -> None:
    """Per-query prefill latency vs prefill length; AutoAWQ switches kernels at batch*seq = 1024."""
    ok = df[(df["status"] == "ok")]
    if ok.empty:
        return
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for backend, (label, color, marker) in (("sdpa", ("SDPA", "#2a78d6", "o")), ("eager", ("eager", "#eb6834", "s"))):
        s = ok[ok["attention_backend"] == backend]
        ax.scatter(s["prefill_seq_len"], s["prefill_ms"], s=9, alpha=0.45, color=color, marker=marker,
                   linewidths=0, label=f"{label} ({len(s)} queries)")
    ax.set_xlim(0, max(1150, float(ok["prefill_seq_len"].max()) + 60))
    ax.set_ylim(bottom=0)
    thr = float(df.attrs.get("awq_threshold", 1024))
    ax.axvline(thr, color=INK2, lw=1, ls="--")
    top = ax.get_ylim()[1]
    ax.text(thr - 12, top * 0.97, "AutoAWQ Triton split-K GEMM", fontsize=7.5, color=INK2, va="top", ha="right")
    ax.text(thr + 12, top * 0.97, "dequant +\ncuBLAS", fontsize=7.5, color=INK2, va="top", ha="left")
    ax.set_xlabel("prefill sequence length (text + retained visual tokens)")
    ax.set_ylabel("LLM prefill latency (ms)")
    ax.legend(loc="center left", fontsize=8)
    ax.set_title("Figure 9 - Prefill latency vs prefill length (all configs, both datasets)", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_theory_vs_measured(tm: pd.DataFrame, out: Path) -> None:
    if tm.empty:
        return
    _style()
    dsets = list(tm["dataset"].unique())
    if "dispatch" not in tm.columns:
        tm = tm.assign(dispatch="upstream")
    fig, axes = plt.subplots(2, len(dsets), figsize=(5.2 * len(dsets), 7.0), squeeze=False)
    arms = [("upstream", "#2a78d6", "o", "measured, upstream AWQ dispatch"),
            ("tuned(64)", "#1baf7a", "D", "measured, tuned AWQ dispatch")]
    for col, ds in enumerate(dsets):
        for row, (flops_col, time_col, what) in enumerate((("prefill_flops_ratio", "prefill_time_ratio", "LLM prefill"),
                                                           ("ttft_flops_ratio", "ttft_time_ratio", "TTFT"))):
            ax = axes[row][col]
            s0 = tm[(tm["dataset"] == ds) & (tm["dispatch"] == "upstream")].sort_values("retention")
            ax.plot(s0["retention"] * 100, s0[flops_col], color=INK2, ls="--", marker="x",
                    label="theory (FLOPs)" + (" incl. vision encoder" if row else ""))
            for disp, color, marker, lab in arms:
                s = tm[(tm["dataset"] == ds) & (tm["dispatch"] == disp)].sort_values("retention")
                if not s.empty:
                    ax.plot(s["retention"] * 100, s[time_col], color=color, marker=marker, label=lab)
            ax.axhline(1.0, color=AXIS, lw=1)
            ax.set_ylim(bottom=0)
            ax.set_xticks([25, 50, 75, 100])
            ax.set_xlabel("visual tokens retained (%)")
            ax.set_ylabel(f"{what} cost relative to 100%")
            ax.set_title(f"{DS_TITLE.get(ds, ds).split(' (')[0]} - {what}", loc="left")
            ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Figure 8 - Theoretical compute reduction vs measured speedup (SDPA + VisionZip)", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
