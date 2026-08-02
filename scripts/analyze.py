"""Stage 7: aggregate raw logs into tables, Pareto/interaction analyses and figures.

    python scripts/analyze.py --n 200            # development subsets
    python scripts/analyze.py --n 1000           # final subsets (writes results/results.csv)

Outputs
    results/results.csv                     main aggregate table (one row per config x dataset)
    results/aggregate/aggregate_<n>.csv     same, per manifest size
    results/aggregate/per_query_<n>.csv     all per-query rows
    results/aggregate/*.md                  Tables 1-4, relative changes, interaction, theory
    plots/fig*.png                          Figures 1-8
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

import edgecompose  # noqa: F401
from edgecompose.analysis import plots
from edgecompose.analysis.aggregate import (
    BASELINE, STAGES, aggregate, llm_prefill_flops, load_arch, load_raw, relative_table, vit_flops,
)
from edgecompose.analysis.interaction import paired_bootstrap_interaction, summarize
from edgecompose.analysis.pareto import pareto_front
from edgecompose.utils import PLOTS_DIR, REPO_ROOT, RESULTS_DIR, setup_logging

logger = logging.getLogger("analyze")
RETENTIONS = (0.75, 0.5, 0.25)
INTERACTION_METRICS = {"E2E latency": "total_latency_ms", "TTFT": "ttft_ms", "prefill": "prefill_ms",
                       "vision": "vision_ms", "peak alloc": "peak_allocated_mb"}


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal markdown table writer (no tabulate dependency)."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if pd.isna(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def cfg_name(backend: str, r: float) -> str:
    prefix = {"sdpa": "C", "eager": "E", "flash_attention_2": "F"}[backend]
    idx = {1.0: 0, 0.75: 1, 0.5: 2, 0.25: 3}[r]
    return f"{prefix}{idx}_{backend if backend != 'flash_attention_2' else 'fa2'}_r{int(r * 100)}"


def interaction_table(df: pd.DataFrame) -> pd.DataFrame:
    """A = VisionZip at retention r, B = SDPA (vs eager); baseline = eager, 100% tokens."""
    rows = []
    for ds in df["dataset"].unique():
        d = df[(df["dataset"] == ds) & (df["status"] == "ok")]
        piv = {c: g.set_index("sample_id") for c, g in d.groupby("config_name")}
        base, b_only = cfg_name("eager", 1.0), cfg_name("sdpa", 1.0)
        if base not in piv or b_only not in piv:
            continue
        for r in RETENTIONS:
            a_only, ab = cfg_name("eager", r), cfg_name("sdpa", r)
            if a_only not in piv or ab not in piv:
                continue
            ids = piv[base].index
            for k in (b_only, a_only, ab):
                ids = ids.intersection(piv[k].index)
            for mname, col in INTERACTION_METRICS.items():
                it = paired_bootstrap_interaction(piv[base].loc[ids, col], piv[a_only].loc[ids, col],
                                                  piv[b_only].loc[ids, col], piv[ab].loc[ids, col], metric=mname)
                s = summarize(it)
                s["R_additive"] = s["R_A"] + s["R_B"] - 1
                rows.append({"dataset": ds, "retention": r, "n": len(ids), **s})
    return pd.DataFrame(rows)


def theory_table(agg: pd.DataFrame, arch: dict) -> pd.DataFrame:
    rows = []
    for ds in agg["dataset"].unique():
        sub = agg[(agg["dataset"] == ds) & (agg["attention_backend"] == "sdpa")]
        base = sub[sub["token_method"] == "none"]
        if base.empty:
            continue
        b = base.iloc[0]
        n_vis = b["visual_tokens_before_mean"]
        f_pre0 = llm_prefill_flops(b["prefill_seq_len_mean"], arch)
        f_vit = vit_flops(n_vis, arch)
        for _, r in sub[sub["token_method"] == "visionzip"].iterrows():
            f_pre = llm_prefill_flops(r["prefill_seq_len_mean"], arch)
            rows.append({
                "dataset": ds, "config_name": r["config_name"], "retention": r["token_retention"],
                "prefill_len": r["prefill_seq_len_mean"], "prefill_flops_ratio": f_pre / f_pre0,
                "prefill_time_ratio": r["prefill_ms_p50"] / b["prefill_ms_p50"],
                "ttft_flops_ratio": (f_vit + f_pre) / (f_vit + f_pre0),
                "ttft_time_ratio": r["ttft_ms_p50"] / b["ttft_ms_p50"],
                "vit_share_of_flops_at_100": f_vit / (f_vit + f_pre0),
            })
        rows.append({"dataset": ds, "config_name": b["config_name"], "retention": 1.0,
                     "prefill_len": b["prefill_seq_len_mean"], "prefill_flops_ratio": 1.0, "prefill_time_ratio": 1.0,
                     "ttft_flops_ratio": 1.0, "ttft_time_ratio": 1.0,
                     "vit_share_of_flops_at_100": f_vit / (f_vit + f_pre0)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tag", default="")
    ap.add_argument("--main", action="store_true", help="also write results/results.csv (final table)")
    args = ap.parse_args()
    setup_logging()

    agg_dir = RESULTS_DIR / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.n}{'_' + args.tag if args.tag else ''}"

    df = load_raw(RESULTS_DIR / "raw", args.n, args.tag)
    if df.empty:
        raise SystemExit(f"no raw results for n={args.n}")
    df.to_csv(agg_dir / f"per_query_{suffix}.csv", index=False)
    failed = df[df["status"] != "ok"]
    logger.info("%d rows, %d failed", len(df), len(failed))

    agg = aggregate(df)
    iso = RESULTS_DIR / "aggregate" / "memory_isolated.csv"
    if iso.exists():
        m = pd.read_csv(iso)[["config_name", "dataset", "peak_reserved_mb", "device_footprint_mb"]]
        m = m.rename(columns={"peak_reserved_mb": "isolated_peak_reserved_mb"})
        agg = agg.merge(m, on=["config_name", "dataset"], how="left")
    agg.to_csv(agg_dir / f"aggregate_{suffix}.csv", index=False)
    if args.main:
        agg.to_csv(RESULTS_DIR / "results.csv", index=False)

    # ---------------- Table 1: system
    sysinfo = json.loads((RESULTS_DIR / "system_info.json").read_text())
    load = json.loads((RESULTS_DIR / "model_load.json").read_text()) if (RESULTS_DIR / "model_load.json").exists() else {}
    t1 = pd.DataFrame([
        ("GPU", sysinfo.get("gpu_name")), ("VRAM", f"{sysinfo.get('gpu_vram_mb')} MB"),
        ("System RAM", f"{sysinfo.get('system_ram_gb')} GB"), ("NVIDIA driver", sysinfo.get("nvidia_driver")),
        ("CUDA (torch build)", sysinfo.get("torch_cuda_version")), ("PyTorch", sysinfo.get("torch_version")),
        ("Transformers", sysinfo.get("transformers_version")), ("AutoAWQ / Triton", f"{sysinfo.get('autoawq_version')} / triton-windows {sysinfo.get('triton-windows_version')}"),
        ("Python / OS", f"{sysinfo.get('python')} / {sysinfo.get('platform')}"),
        ("Model", "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"), ("Quantization", "AWQ INT4 (w4, g128) LLM; vision tower FP16"),
        ("Model weights resident", f"{load.get('model_load_allocated_mb', float('nan')):.0f} MB allocated after load"),
        ("SDPA kernels available", str(sysinfo.get("sdpa_kernels"))), ("flash-attn usable", str(sysinfo.get("flash_attn_usable"))),
    ], columns=["item", "value"])

    # ---------------- Table 2: main results
    t2 = agg[["dataset", "config_name", "quality", "quality_ci_low", "quality_ci_high", "ttft_ms_p50", "ttft_ms_p95",
              "total_latency_ms_p50", "total_latency_ms_p95", "tokens_per_second_p50", "peak_allocated_mb_p50",
              "peak_reserved_mb_max", "energy_j_median", "visual_tokens_after_mean", "n_ok", "n_failed"]].copy()

    # ---------------- Table 3: relative
    rel = relative_table(df, agg, BASELINE)
    rel.to_csv(agg_dir / f"relative_{suffix}.csv", index=False)

    # ---------------- Table 4: Pareto (quality max, latency min, VRAM min), per dataset
    pareto_rows = []
    for ds in agg["dataset"].unique():
        sub = agg[agg["dataset"] == ds].reset_index(drop=True)
        rows = sub.to_dict(orient="records")
        f3 = set(pareto_front(rows, [("quality", "max"), ("total_latency_ms_p50", "min"), ("peak_allocated_mb_p50", "min")]))
        f_lat = set(pareto_front(rows, [("quality", "max"), ("total_latency_ms_p50", "min")]))
        f_mem = set(pareto_front(rows, [("quality", "max"), ("peak_allocated_mb_p50", "min")]))
        f_en = set(pareto_front(rows, [("quality", "max"), ("energy_j_median", "min")]))
        for i in sorted(f3 | f_lat | f_mem):
            r = sub.loc[i]
            pareto_rows.append({"dataset": ds, "config_name": r["config_name"], "quality": r["quality"],
                                "latency_p50_ms": r["total_latency_ms_p50"], "ttft_p50_ms": r["ttft_ms_p50"],
                                "peak_alloc_mb": r["peak_allocated_mb_p50"], "energy_j": r["energy_j_median"],
                                "front_q_lat_mem": i in f3, "front_q_lat": i in f_lat, "front_q_mem": i in f_mem,
                                "front_q_energy": i in f_en})
    t4 = pd.DataFrame(pareto_rows)
    t4.to_csv(agg_dir / f"pareto_{suffix}.csv", index=False)

    # ---------------- stage shares
    stage = agg[["dataset", "config_name"] + [f"{s}_p50" for s in STAGES] + ["total_latency_ms_p50", "decode_ms_per_token_p50",
                                                                           "generated_tokens_p50", "prefill_seq_len_mean",
                                                                           "frac_prefill_ge_1024"]].copy()
    tot = stage[[f"{s}_p50" for s in STAGES]].sum(axis=1)
    for s in STAGES:
        stage[f"{s.replace('_ms', '')}_share"] = stage[f"{s}_p50"] / tot

    # ---------------- POPE detail
    pope_cols = [c for c in agg.columns if c.startswith("pope_")]
    pope = agg[agg["dataset"] == "pope"][["config_name"] + pope_cols] if pope_cols else pd.DataFrame()

    # ---------------- interaction + theory
    inter = interaction_table(df)
    inter.to_csv(agg_dir / f"interaction_{suffix}.csv", index=False)
    arch = load_arch(REPO_ROOT / "hf_assets" / "model" / "Qwen__Qwen2.5-VL-3B-Instruct-AWQ")
    theory = theory_table(agg, arch)
    theory.to_csv(agg_dir / f"theory_vs_measured_{suffix}.csv", index=False)

    with open(agg_dir / f"tables_{suffix}.md", "w", encoding="utf-8") as f:
        f.write(f"# EdgeCompose-VLM tables (n={args.n} per dataset)\n\n")
        f.write("## Table 1 - System configuration\n\n" + md(t1) + "\n\n")
        f.write("## Table 2 - Main results (medians unless noted; quality with 95% bootstrap CI)\n\n" + md(t2, ".1f").replace("| 0.", "| 0.") + "\n\n")
        f.write("Quality columns in Table 2 are fractions (shown with 1 decimal above; see CSV for full precision).\n\n")
        f.write("## Table 3 - Relative to C0 (SDPA, 100% tokens); paired bootstrap 95% CIs\n\n" + md(rel, ".2f") + "\n\n")
        f.write("## Table 4 - Pareto-efficient configurations\n\n" + md(t4, ".3f") + "\n\n")
        f.write("## Stage-wise medians and shares\n\n" + md(stage, ".2f") + "\n\n")
        if not pope.empty:
            f.write("## POPE detail (yes = positive class)\n\n" + md(pope, ".3f") + "\n\n")
        f.write("## Composition interaction (A = VisionZip r, B = SDPA vs eager, baseline = eager 100%)\n\n" + md(inter, ".3f") + "\n\n")
        f.write("## Theoretical FLOPs vs measured time (SDPA + VisionZip)\n\n" + md(theory, ".3f") + "\n")
    logger.info("wrote %s", agg_dir / f"tables_{suffix}.md")

    # ---------------- figures
    tag = f"_{suffix}"
    plots.fig_quality_vs_latency(agg, PLOTS_DIR / f"fig1_quality_vs_latency{tag}.png")
    plots.fig_quality_vs_memory(agg, PLOTS_DIR / f"fig2_quality_vs_memory{tag}.png")
    plots.fig_stage_breakdown(agg, PLOTS_DIR / f"fig3_stage_latency{tag}.png")
    plots.fig_retention_vs_quality(agg, PLOTS_DIR / f"fig4_retention_vs_quality{tag}.png")
    plots.fig_retention_vs_ttft(agg, PLOTS_DIR / f"fig5_retention_vs_ttft{tag}.png")
    plots.fig_quality_vs_energy(agg, PLOTS_DIR / f"fig6_quality_vs_energy{tag}.png")
    plots.fig_interaction_heatmap(inter[inter["metric"] != "peak alloc"] if not inter.empty else inter,
                                  PLOTS_DIR / f"fig7_interaction_heatmap{tag}.png")
    plots.fig_theory_vs_measured(theory, PLOTS_DIR / f"fig8_theory_vs_measured{tag}.png")
    logger.info("figures written to %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
