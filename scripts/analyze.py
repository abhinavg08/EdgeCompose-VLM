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
    BASELINE, STAGES, aggregate, llm_prefill_flops, load_arch, load_raw, paired_delta_ci, relative_table, vit_flops,
)
from edgecompose.analysis.interaction import paired_bootstrap_interaction, summarize
from edgecompose.analysis.pareto import pareto_front
from edgecompose.utils import PLOTS_DIR, REPO_ROOT, RESULTS_DIR, setup_logging

logger = logging.getLogger("analyze")
RETENTIONS = (0.75, 0.5, 0.25)
INTERACTION_METRICS = {"E2E latency": "total_latency_ms", "TTFT": "ttft_ms", "prefill": "prefill_ms",
                       "vision": "vision_ms", "peak alloc": "peak_allocated_mb"}


_MS_LIKE = ("_ms", "_mb", "latency_p", "ttft_p", "energy", "kv_len", "prefill_len", "_len_mean", "footprint")
_PCT_LIKE = ("_pct", "reduction", "retention_pct")


def _fmt_for(col: str, default: str) -> str:
    if any(k in col for k in _PCT_LIKE):
        return ".1f"
    if any(k in col for k in _MS_LIKE):
        return ".1f"
    return default


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal markdown table writer (no tabulate dependency); ms/MB/% columns get 1 decimal."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if pd.isna(v) else format(v, _fmt_for(c, floatfmt)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pareto_tolerances(sub: pd.DataFrame, baseline: str = BASELINE) -> dict:
    """Noise-aware tolerances: 0.5 pp quality; 3% latency/energy and 1% memory of the baseline.

    3% reflects the measured run-to-run stability of config-to-config latency ratios
    (repeatability table); absolute latencies drift more between sessions.
    """
    b = sub[sub["config_name"] == baseline]
    b = b.iloc[0] if len(b) else sub.iloc[0]
    return {"quality": 0.005, "total_latency_ms_p50": 0.03 * b["total_latency_ms_p50"],
            "ttft_ms_p50": 0.03 * b["ttft_ms_p50"], "peak_allocated_mb_p50": 0.01 * b["peak_allocated_mb_p50"],
            "energy_j_median": 0.03 * b["energy_j_median"] if pd.notna(b["energy_j_median"]) else 0.0}


def method_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """VisionZip vs uniform subsampling at equal retention (paired quality delta, overhead)."""
    rows = []
    for ds in df["dataset"].unique():
        d = df[(df["dataset"] == ds) & (df["status"] == "ok")]
        piv = {c: g.set_index("sample_id") for c, g in d.groupby("config_name")}
        for vz, un, r in (("C2_sdpa_r50", "U2_sdpa_uniform_r50", 0.5), ("C3_sdpa_r25", "U3_sdpa_uniform_r25", 0.25)):
            if vz not in piv or un not in piv:
                continue
            ids = piv[vz].index.intersection(piv[un].index)
            dq, lo, hi = paired_delta_ci(piv[un].loc[ids, "score"].astype(float), piv[vz].loc[ids, "score"].astype(float))
            rows.append({"dataset": ds, "retention": r, "n": len(ids),
                         "quality_visionzip": float(piv[vz].loc[ids, "score"].mean()),
                         "quality_uniform": float(piv[un].loc[ids, "score"].mean()),
                         "delta_vz_minus_uniform": dq, "delta_ci_low": lo, "delta_ci_high": hi,
                         "compress_ms_visionzip_p50": float(piv[vz].loc[ids, "compress_ms"].median()),
                         "compress_ms_uniform_p50": float(piv[un].loc[ids, "compress_ms"].median()),
                         "answers_identical_frac": float((piv[vz].loc[ids, "prediction"] == piv[un].loc[ids, "prediction"]).mean())})
    return pd.DataFrame(rows)


def cfg_name(arm: str, r: float) -> str:
    """Config names of the factorial arms: sdpa (C*), eager (E*), tuned AWQ dispatch (T*), fa2 (F*)."""
    idx = {1.0: 0, 0.75: 1, 0.5: 2, 0.25: 3}[r]
    pct = int(r * 100)
    return {"sdpa": f"C{idx}_sdpa_r{pct}", "eager": f"E{idx}_eager_r{pct}", "tuned": f"T{idx}_sdpa_awq64_r{pct}",
            "flash_attention_2": f"F{idx}_fa2_r{pct}"}[arm]


# Factor B definitions: (label, arm without B, arm with B). A = VisionZip at retention r.
FACTORS_B = [("SDPA (vs eager)", "eager", "sdpa"), ("tuned AWQ dispatch (vs upstream)", "sdpa", "tuned")]


def interaction_table(df: pd.DataFrame) -> pd.DataFrame:
    """I(A,B) for A = VisionZip r and each factor B; baseline = (without B, 100% tokens)."""
    rows = []
    for ds in df["dataset"].unique():
        d = df[(df["dataset"] == ds) & (df["status"] == "ok")]
        piv = {c: g.set_index("sample_id") for c, g in d.groupby("config_name")}
        for blabel, arm0, arm1 in FACTORS_B:
            base, b_only = cfg_name(arm0, 1.0), cfg_name(arm1, 1.0)
            if base not in piv or b_only not in piv:
                continue
            for r in RETENTIONS:
                a_only, ab = cfg_name(arm0, r), cfg_name(arm1, r)
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
                    rows.append({"dataset": ds, "factor_B": blabel, "baseline": base, "retention": r, "n": len(ids), **s})
    return pd.DataFrame(rows)


def theory_table(agg: pd.DataFrame, arch: dict) -> pd.DataFrame:
    rows = []
    for (ds, thr), sub in agg[agg["attention_backend"] == "sdpa"].groupby(["dataset", "awq_dequant_threshold"]):
        base = sub[sub["token_method"] == "none"]
        if base.empty:
            continue
        dispatch = "upstream" if thr == 1024 else f"tuned({thr})"
        b = base.iloc[0]
        n_vis = b["visual_tokens_before_mean"]
        f_pre0 = llm_prefill_flops(b["prefill_seq_len_mean"], arch)
        f_vit = vit_flops(n_vis, arch)
        for _, r in sub[sub["token_method"] == "visionzip"].iterrows():
            f_pre = llm_prefill_flops(r["prefill_seq_len_mean"], arch)
            rows.append({
                "dataset": ds, "dispatch": dispatch, "config_name": r["config_name"], "retention": r["token_retention"],
                "prefill_len": r["prefill_seq_len_mean"], "prefill_flops_ratio": f_pre / f_pre0,
                "prefill_time_ratio": r["prefill_ms_p50"] / b["prefill_ms_p50"],
                "ttft_flops_ratio": (f_vit + f_pre) / (f_vit + f_pre0),
                "ttft_time_ratio": r["ttft_ms_p50"] / b["ttft_ms_p50"],
                "vit_share_of_flops_at_100": f_vit / (f_vit + f_pre0),
            })
        rows.append({"dataset": ds, "dispatch": dispatch, "config_name": b["config_name"], "retention": 1.0,
                     "prefill_len": b["prefill_seq_len_mean"], "prefill_flops_ratio": 1.0, "prefill_time_ratio": 1.0,
                     "ttft_flops_ratio": 1.0, "ttft_time_ratio": 1.0,
                     "vit_share_of_flops_at_100": f_vit / (f_vit + f_pre0)})
    return pd.DataFrame(rows)


def supplementary(agg_dir: Path) -> str:
    """Decode-profile (forced 32 tokens) and run-to-run repeatability tables, if present."""
    text = ""
    dec = load_raw(RESULTS_DIR / "raw", 40, "decode32")
    if not dec.empty:
        dec = dec[dec["status"] == "ok"].copy()
        dec["ms_per_token"] = dec["decode_ms"] / (dec["generated_tokens"] - 1)
        t = dec.groupby("config_name").agg(
            kv_len=("prefill_seq_len", "mean"), prefill_ms_p50=("prefill_ms", "median"),
            decode_ms_p50=("decode_ms", "median"), ms_per_token_p50=("ms_per_token", "median"),
            ms_per_token_p95=("ms_per_token", lambda s: s.quantile(0.95)), e2e_ms_p50=("total_latency_ms", "median"),
            decode_share=("decode_ms", lambda s: float("nan")), n=("sample_id", "count")).reset_index()
        t["decode_share"] = t["decode_ms_p50"] / t["e2e_ms_p50"]
        t.to_csv(agg_dir / "decode_profile_32tok.csv", index=False)
        text += "## Decode profile (32 forced new tokens, TextVQA, 40 samples)\n\n" + md(t, ".1f") + "\n\n"
    rep = load_raw(RESULTS_DIR / "raw", 40, "rep")
    if not rep.empty:
        rep = rep[rep["status"] == "ok"]
        rows = []
        for (ds, cfg), g in rep.groupby(["dataset", "config_name"]):
            per_q = g.groupby("sample_id")["total_latency_ms"].agg(["mean", "std"])
            per_rep_median = g.groupby("repeat")["total_latency_ms"].median()
            ttft_rep = g.groupby("repeat")["ttft_ms"].median()
            base = rep[(rep["dataset"] == ds) & (rep["config_name"] == BASELINE)].groupby("repeat")["total_latency_ms"].median()
            ratio = (per_rep_median / base).dropna()
            rows.append({"dataset": ds, "config_name": cfg, "repeats": g["repeat"].nunique(),
                         "ratio_vs_C0_by_repeat": " / ".join(f"{v:.3f}" for v in ratio),
                         "ratio_spread_pp": float((ratio.max() - ratio.min()) * 100) if len(ratio) else float("nan"),
                         "per_query_cv_pct_p50": float((per_q["std"] / per_q["mean"]).median() * 100),
                         "e2e_p50_by_repeat": " / ".join(f"{v:.0f}" for v in per_rep_median),
                         "e2e_p50_spread_pct": float((per_rep_median.max() - per_rep_median.min()) / per_rep_median.mean() * 100),
                         "ttft_p50_spread_pct": float((ttft_rep.max() - ttft_rep.min()) / ttft_rep.mean() * 100),
                         "answers_identical_across_repeats": float(
                             g.groupby("sample_id")["prediction"].nunique().eq(1).mean())})
        t = pd.DataFrame(rows)
        t.to_csv(agg_dir / "repeatability.csv", index=False)
        text += "## Run-to-run repeatability (40 samples x 3 repeats)\n\n" + md(t, ".2f") + "\n\n"
    kb = RESULTS_DIR / "aggregate" / "awq_kernel_bench.csv"
    if kb.exists():
        k = pd.read_csv(kb)
        tot = k.assign(fused=k.triton_gemm_ms * k.count_per_layer * 36, dequant=k.dequant_cublas_ms * k.count_per_layer * 36)
        tot = tot.groupby("M")[["fused", "dequant"]].sum().reset_index()
        tot["faster"] = np.where(tot["dequant"] < tot["fused"], "dequant+cuBLAS", "Triton fused")
        text += "## AWQ INT4 linear-layer time vs rows M (36 layers, microbenchmark)\n\n" + md(tot, ".1f") + "\n\n"
    return text


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
    t2_cols = ["dataset", "config_name", "quality", "quality_ci_low", "quality_ci_high", "ttft_ms_p50", "ttft_ms_p95",
               "total_latency_ms_p50", "total_latency_ms_p95", "tokens_per_second_p50", "peak_allocated_mb_p50",
               "device_footprint_mb", "energy_j_median", "visual_tokens_after_mean", "n_ok", "n_failed"]
    t2 = agg[[c for c in t2_cols if c in agg.columns]].copy()

    # ---------------- Table 3: relative
    rel = relative_table(df, agg, BASELINE)
    rel.to_csv(agg_dir / f"relative_{suffix}.csv", index=False)

    # ---------------- Table 4: Pareto (quality max, latency min, VRAM min), per dataset
    pareto_rows = []
    for ds in agg["dataset"].unique():
        sub = agg[agg["dataset"] == ds].reset_index(drop=True)
        rows = sub.to_dict(orient="records")
        tol = pareto_tolerances(sub)
        q_lat_mem = [("quality", "max"), ("total_latency_ms_p50", "min"), ("peak_allocated_mb_p50", "min")]
        f3 = set(pareto_front(rows, q_lat_mem, tol))
        f3_strict = set(pareto_front(rows, q_lat_mem))
        f_lat = set(pareto_front(rows, [("quality", "max"), ("total_latency_ms_p50", "min")], tol))
        f_mem = set(pareto_front(rows, [("quality", "max"), ("peak_allocated_mb_p50", "min")], tol))
        f_en = set(pareto_front(rows, [("quality", "max"), ("energy_j_median", "min")], tol))
        for i in sorted(f3 | f_lat | f_mem):
            r = sub.loc[i]
            pareto_rows.append({"dataset": ds, "config_name": r["config_name"], "quality": r["quality"],
                                "latency_p50_ms": r["total_latency_ms_p50"], "ttft_p50_ms": r["ttft_ms_p50"],
                                "peak_alloc_mb": r["peak_allocated_mb_p50"], "energy_j": r["energy_j_median"],
                                "front_q_lat_mem": i in f3, "front_q_lat": i in f_lat, "front_q_mem": i in f_mem,
                                "front_q_energy": i in f_en, "strict_front_q_lat_mem": i in f3_strict})
    t4 = pd.DataFrame(pareto_rows)
    methods = method_comparison(df)
    methods.to_csv(agg_dir / f"visionzip_vs_uniform_{suffix}.csv", index=False)
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
        f.write("## Table 2 - Main results (medians unless noted; quality with 95% bootstrap CI)\n\n"
                "`peak_allocated_mb_p50` = per-query live-tensor peak (interleaved run); `device_footprint_mb` = isolated "
                "per-config peak reserved + CUDA context/other processes on the 10 largest images.\n\n" + md(t2, ".3f") + "\n\n")
        f.write("## Table 3 - Relative to C0 (SDPA, 100% tokens); paired bootstrap 95% CIs\n\n" + md(rel, ".2f") + "\n\n")
        f.write("## Table 4 - Pareto-efficient configurations\n\nNoise-aware dominance: differences within 0.5 pp "
                "quality, 3% latency/energy or 1% memory (relative to C0) count as ties; `strict_front_q_lat_mem` is "
                "the tolerance-free front.\n\n" + md(t4, ".3f") + "\n\n")
        if not methods.empty:
            f.write("## VisionZip vs uniform subsampling at equal retention (paired)\n\n" + md(methods, ".3f") + "\n\n")
        f.write("## Stage-wise medians and shares\n\n" + md(stage, ".2f") + "\n\n")
        if not pope.empty:
            f.write("## POPE detail (yes = positive class)\n\n" + md(pope, ".3f") + "\n\n")
        f.write("## Composition interaction (A = VisionZip r, B = SDPA vs eager, baseline = eager 100%)\n\n" + md(inter, ".3f") + "\n\n")
        f.write("## Theoretical FLOPs vs measured time (SDPA + VisionZip)\n\n" + md(theory, ".3f") + "\n\n")
        iso = RESULTS_DIR / "aggregate" / "memory_isolated.csv"
        if iso.exists():
            f.write("## Isolated per-config VRAM (10 largest images per dataset)\n\n" + md(pd.read_csv(iso), ".0f") + "\n\n")
        f.write(supplementary(agg_dir))
    logger.info("wrote %s", agg_dir / f"tables_{suffix}.md")

    # ---------------- figures
    tag = f"_{suffix}"
    tols = {ds: pareto_tolerances(agg[agg["dataset"] == ds]) for ds in agg["dataset"].unique()}
    plots.fig_quality_vs_latency(agg, PLOTS_DIR / f"fig1_quality_vs_latency{tag}.png", tols)
    plots.fig_quality_vs_memory(agg, PLOTS_DIR / f"fig2_quality_vs_memory{tag}.png", tols)
    plots.fig_stage_breakdown(agg, PLOTS_DIR / f"fig3_stage_latency{tag}.png")
    plots.fig_retention_vs_quality(agg, PLOTS_DIR / f"fig4_retention_vs_quality{tag}.png")
    plots.fig_retention_vs_ttft(agg, PLOTS_DIR / f"fig5_retention_vs_ttft{tag}.png")
    plots.fig_quality_vs_energy(agg, PLOTS_DIR / f"fig6_quality_vs_energy{tag}.png", tols)
    plots.fig_interaction_heatmap(inter[inter["metric"] != "peak alloc"] if not inter.empty else inter,
                                  PLOTS_DIR / f"fig7_interaction_heatmap{tag}.png")
    plots.fig_theory_vs_measured(theory, PLOTS_DIR / f"fig8_theory_vs_measured{tag}.png")
    plots.fig_prefill_vs_length(df, PLOTS_DIR / f"fig9_prefill_vs_length{tag}.png")
    logger.info("figures written to %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
