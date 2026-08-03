"""Deployment configuration selector over *measured* results (no learned model).

Input: the aggregate table (results/results.csv) - one row per (config, dataset) with
quality and resource metrics. The selector filters configurations by hard constraints
and ranks the feasible ones by an objective:

    quality  -> maximize quality        (tie-break: lower latency, then lower memory)
    latency  -> minimize latency        (tie-break: higher quality)
    memory   -> minimize peak VRAM      (tie-break: higher quality)
    energy   -> minimize energy/query   (tie-break: higher quality)

`min_quality` may be absolute (e.g. 0.70 accuracy) or relative to the uncompressed SDPA
baseline (`relative_quality=True`, e.g. 0.95 = keep 95% of baseline quality).
When several datasets are selected, constraints must hold on *every* dataset and the
objective is averaged across datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

OBJECTIVES = {
    "quality": ("quality", False),
    "latency": ("latency_ms", True),
    "memory": ("peak_vram_mb", True),
    "energy": ("energy_j", True),
}


@dataclass
class Constraints:
    max_vram_mb: Optional[float] = None
    max_latency_ms: Optional[float] = None
    max_ttft_ms: Optional[float] = None
    min_quality: Optional[float] = None
    relative_quality: bool = False
    datasets: List[str] = field(default_factory=list)
    latency_stat: str = "p50"  # p50 | p95 | mean
    baseline_config: str = "C0_sdpa_r100"


def _table(agg: pd.DataFrame, c: Constraints) -> pd.DataFrame:
    df = agg.copy()
    if c.datasets:
        df = df[df["dataset"].isin(c.datasets)]
    lat_col = {"p50": "total_latency_ms_p50", "p95": "total_latency_ms_p95", "mean": "total_latency_ms_mean"}[c.latency_stat]
    df["latency_ms"] = df[lat_col]
    df["ttft"] = df["ttft_ms_p50"]
    # Prefer the isolated per-config device footprint (scripts/memory_profile.py, merged
    # into results.csv by analyze.py); fall back to the per-query live-tensor peak.
    if "device_footprint_mb" in df.columns and df["device_footprint_mb"].notna().all():
        df["peak_vram_mb"] = df["device_footprint_mb"]
    elif "peak_allocated_mb_max" in df.columns:
        df["peak_vram_mb"] = df["peak_allocated_mb_max"]
    else:
        df["peak_vram_mb"] = df["peak_reserved_mb_max"]
    df["energy_j"] = df.get("energy_j_median")
    if c.relative_quality:
        base = df[df["config_name"] == c.baseline_config].set_index("dataset")["quality"]
        df["quality_eff"] = df.apply(lambda r: r["quality"] / base.get(r["dataset"], float("nan")), axis=1)
    else:
        df["quality_eff"] = df["quality"]
    return df


def select(agg: pd.DataFrame, constraints: Constraints, objective: str = "quality", top_k: int = 5) -> Dict:
    """Return the best feasible configuration, the ranked feasible list and rejections."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {list(OBJECTIVES)}")
    df = _table(agg, constraints)
    reasons: Dict[str, List[str]] = {}

    def reject(cfg: str, why: str) -> None:
        reasons.setdefault(cfg, []).append(why)

    for _, r in df.iterrows():
        cfg, ds = r["config_name"], r["dataset"]
        if constraints.max_vram_mb is not None and r["peak_vram_mb"] > constraints.max_vram_mb:
            reject(cfg, f"{ds}: VRAM {r['peak_vram_mb']:.0f} MB > {constraints.max_vram_mb:.0f}")
        if constraints.max_latency_ms is not None and r["latency_ms"] > constraints.max_latency_ms:
            reject(cfg, f"{ds}: latency {r['latency_ms']:.0f} ms > {constraints.max_latency_ms:.0f}")
        if constraints.max_ttft_ms is not None and r["ttft"] > constraints.max_ttft_ms:
            reject(cfg, f"{ds}: TTFT {r['ttft']:.0f} ms > {constraints.max_ttft_ms:.0f}")
        if constraints.min_quality is not None and not (r["quality_eff"] >= constraints.min_quality):
            reject(cfg, f"{ds}: quality {r['quality_eff']:.3f} < {constraints.min_quality:.3f}")
        if objective == "energy" and pd.isna(r.get("energy_j")):
            reject(cfg, f"{ds}: no energy measurement")

    per_cfg = (
        df.groupby("config_name")
        .agg(quality=("quality_eff", "mean"), latency_ms=("latency_ms", "mean"), ttft_ms=("ttft", "mean"),
             peak_vram_mb=("peak_vram_mb", "max"), energy_j=("energy_j", "mean"), n_datasets=("dataset", "nunique"))
        .reset_index()
    )
    feasible = per_cfg[~per_cfg["config_name"].isin(reasons)].copy()
    col, ascending = OBJECTIVES[objective]
    tiebreak = ["latency_ms", "peak_vram_mb"] if objective == "quality" else ["quality"]
    tb_asc = [True, True] if objective == "quality" else [False]
    feasible = feasible.sort_values([col] + tiebreak, ascending=[ascending] + tb_asc)
    best = feasible.iloc[0].to_dict() if len(feasible) else None
    return {
        "objective": objective,
        "constraints": constraints.__dict__,
        "best": best,
        "ranking": feasible.head(top_k).to_dict(orient="records"),
        "rejected": reasons,
    }
