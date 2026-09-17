"""Aggregation of EdgeInspect runs: per (category, seed, k, r) metrics, macro averages,
seed variability, OOM/feasibility tables, and Pareto fronts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from edgecompose.analysis.pareto import pareto_front
from edgeinspect.metrics.anomaly import aufc, evaluate

QUALITY_KEYS = ["f1", "precision", "recall", "accuracy", "balanced_accuracy", "auroc", "f1_max",
                "f1_structural", "f1_logical", "auroc_structural", "auroc_logical", "recall_structural",
                "recall_logical", "pred_anomalous_ratio",
                "f1_cal", "precision_cal", "recall_cal", "specificity_cal", "balanced_accuracy_cal",
                "f1_cal_structural", "f1_cal_logical", "f1_max_structural", "f1_max_logical"]
CAL_QUANTILE = 0.9  # threshold = 90th percentile of normal calibration scores (~10% FPR target)


def load_rows(raw_dir: Path, tag: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for p in sorted(raw_dir.glob(f"{tag}__*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["reference_seed", "reference_count", "token_retention", "sample_id"], keep="last")


def _llr(g: pd.DataFrame) -> pd.Series:
    """Log-likelihood ratio log P(ANOMALOUS) - log P(NORMAL) (same ranking as p_anomaly, no saturation)."""
    return g["logp_anomalous"].astype(float) - g["logp_normal"].astype(float)


def per_config(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (category, seed, k, r). Test rows are scored; calibration rows (normal-only
    validation images) set the calibrated decision threshold."""
    if "role" not in df:
        df = df.assign(role="test")
    df = df.assign(role=df["role"].fillna("test"))
    out = []
    for (cat, seed, k, r), g_all in df.groupby(["category", "reference_seed", "reference_count", "token_retention"]):
        g = g_all[g_all["role"] == "test"]
        cal = g_all[(g_all["role"] == "calibration") & (g_all["status"] == "ok")]
        ok = g[g["status"] == "ok"]
        row = {"category": cat, "seed": seed, "k": int(k), "retention": float(r), "n": len(g), "n_ok": len(ok),
               "n_oom": int((g["status"] == "OOM").sum()), "feasible": len(ok) == len(g) and len(g) > 0,
               "n_calibration": len(cal)}
        if len(ok):
            have = ok["logp_anomalous"].notna().all() and ok["logp_normal"].notna().all()
            scores = _llr(ok).tolist() if have else None
            m = evaluate(ok["binary_label"].astype(int), ok["prediction"].astype(int), scores, ok["anomaly_type"].tolist())
            if have and len(cal):
                tau = float(np.quantile(_llr(cal), CAL_QUANTILE))
                pred_cal = (_llr(ok) > tau).astype(int)
                mc = evaluate(ok["binary_label"].astype(int), pred_cal, None, ok["anomaly_type"].tolist())
                m.update({"f1_cal": mc["f1"], "precision_cal": mc["precision"], "recall_cal": mc["recall"],
                          "specificity_cal": mc["specificity"], "balanced_accuracy_cal": mc["balanced_accuracy"],
                          "f1_cal_structural": mc.get("f1_structural", np.nan),
                          "f1_cal_logical": mc.get("f1_logical", np.nan)})
                row["threshold_llr"] = tau
            row.update({key: m.get(key, np.nan) for key in QUALITY_KEYS})
            row["parse_fail_rate"] = float(ok["parse_failed"].mean())
            for c in ("ttft_ms", "total_latency_ms", "prefill_ms", "vision_ms", "decode_ms", "compress_ms"):
                row[f"{c}_p50"] = float(ok[c].median())
            row["total_latency_ms_p95"] = float(ok["total_latency_ms"].quantile(0.95))
            row["peak_allocated_mb_p50"] = float(ok["peak_allocated_mb"].median())
            row["peak_allocated_mb_max"] = float(ok["peak_allocated_mb"].max())
            # meaningful per config because the runner releases the allocator cache before each query
            row["peak_reserved_mb_max"] = float(ok["peak_reserved_mb"].max())
            row["visual_tokens_before"] = float(ok["visual_tokens_before"].mean())
            row["visual_tokens_after"] = float(ok["visual_tokens_after"].mean())
            row["prefill_seq_len"] = float(ok["prefill_seq_len"].mean())
            row["energy_j_median"] = float(ok["energy_j"].median()) if "energy_j" in ok and ok["energy_j"].notna().any() else np.nan
        out.append(row)
    return pd.DataFrame(out)


def macro(pc: pd.DataFrame) -> pd.DataFrame:
    """Macro average over categories per (seed, k, r); only configs feasible in every category."""
    cols = [c for c in pc.columns if c not in ("category", "seed", "k", "retention", "feasible")]
    rows = []
    for (seed, k, r), g in pc.groupby(["seed", "k", "retention"]):
        row = {"seed": seed, "k": k, "retention": r, "n_categories": g["category"].nunique(),
               "feasible_all": bool(g["feasible"].all()), "n_oom": int(g["n_oom"].sum())}
        for c in cols:
            if c in ("n", "n_ok", "n_oom"):
                row[c] = int(g[c].sum())
            elif pd.api.types.is_numeric_dtype(g[c]):
                row[c] = float(g[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def across_seeds(mac: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over reference seeds per (k, r)."""
    keys = [c for c in QUALITY_KEYS + ["ttft_ms_p50", "total_latency_ms_p50", "prefill_ms_p50", "vision_ms_p50",
                                       "peak_allocated_mb_max", "peak_reserved_mb_max", "peak_allocated_mb_p50",
                                       "visual_tokens_after", "energy_j_median", "total_latency_ms_p95"] if c in mac]
    g = mac.groupby(["k", "retention"])
    mean = g[keys].mean()
    std = g[keys].std(ddof=1).add_suffix("_std")
    out = pd.concat([mean, std, g["seed"].nunique().rename("n_seeds"), g["feasible_all"].all()], axis=1)
    return out.reset_index()


def aufc_table(mac_seed: pd.DataFrame, metric: str = "f1_max") -> pd.DataFrame:
    rows = []
    for r, g in mac_seed.groupby("retention"):
        g = g.dropna(subset=[metric]).sort_values("k")
        if len(g) >= 2:
            rows.append({"retention": r, f"aufc_{metric}": aufc(g["k"].tolist(), g[metric].tolist()),
                         "ks": "/".join(str(int(x)) for x in g["k"])})
    return pd.DataFrame(rows)


def pareto_configs(tab: pd.DataFrame, quality: str, costs: Sequence[str], tol: Dict[str, float]) -> List[int]:
    rows = tab.to_dict(orient="records")
    objs = [(quality, "max")] + [(c, "min") for c in costs]
    return pareto_front(rows, objs, tol)
