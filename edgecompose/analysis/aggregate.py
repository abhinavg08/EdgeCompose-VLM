"""Aggregate per-query rows into per-(config, dataset) statistics and derived tables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from edgecompose.evaluation.metrics import pope_metrics

STAGES = ["preprocess_ms", "vision_ms", "compress_ms", "prefill_ms", "decode_ms"]
LAT_COLS = STAGES + ["ttft_ms", "total_latency_ms"]
BASELINE = "C0_sdpa_r100"


def load_raw(raw_dir: Path, n_label: int, tag: str = "") -> pd.DataFrame:
    """Load all raw JSONL files for a manifest size (files `<cfg>__<ds>_<n>[__tag].jsonl`)."""
    suffix = f"__{tag}" if tag else ""
    rows: List[Dict] = []
    for p in sorted(raw_dir.glob(f"*_{n_label}{suffix}.jsonl")):
        if not tag and p.stem.count("__") > 1:
            continue  # tagged files belong to other experiments
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # partially written line while a run is active
                    continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # latest row wins if a (config, sample, repeat) was re-run
    df = df.drop_duplicates(subset=["config_name", "dataset", "sample_id", "repeat"], keep="last")
    return df


def bootstrap_ci(x: Sequence[float], stat=np.mean, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_delta_ci(a: Sequence[float], b: Sequence[float], stat=np.mean, n_boot: int = 2000, seed: int = 0):
    """Bootstrap CI of stat(b) - stat(a) with samples paired by index."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = stat(b[idx]) - stat(a[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(stat(b) - stat(a)), float(lo), float(hi)


def paired_ratio_ci(a: Sequence[float], b: Sequence[float], stat=np.median, n_boot: int = 2000, seed: int = 0):
    """Bootstrap CI of stat(b)/stat(a) with samples paired by index (latency ratios)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = stat(b[idx]) / stat(a[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(stat(b) / stat(a)), float(lo), float(hi)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (config, dataset) with quality, latency, memory and energy statistics."""
    out = []
    for (cfg, ds), g in df.groupby(["config_name", "dataset"], sort=False):
        ok = g[g["status"] == "ok"]
        r: Dict = {
            "config_name": cfg,
            "dataset": ds,
            "attention_backend": g["attention_backend"].iloc[0],
            "token_method": g["token_method"].iloc[0],
            "token_retention": float(g["token_retention"].iloc[0]),
            "awq_dequant_threshold": int(g["awq_dequant_threshold"].fillna(1024).iloc[0])
            if "awq_dequant_threshold" in g else 1024,
            "n": len(g),
            "n_ok": len(ok),
            "n_failed": int((g["status"] != "ok").sum()),
        }
        scores = ok["score"].astype(float).values
        if ds == "pope":
            m = pope_metrics(ok["parsed_prediction"], ok["ground_truth"])
            r.update({f"pope_{k}": v for k, v in m.items() if k != "n"})
            r["quality"] = m["accuracy"]
            r["quality_metric"] = "POPE accuracy"
            for cat, gc in ok.groupby("category"):
                r[f"pope_f1_{cat}"] = pope_metrics(gc["parsed_prediction"], gc["ground_truth"])["f1"]
        else:
            r["quality"] = float(np.mean(scores)) if len(scores) else float("nan")
            r["quality_metric"] = "TextVQA VQA-accuracy"
        r["quality_ci_low"], r["quality_ci_high"] = bootstrap_ci(scores)
        for c in LAT_COLS + ["generated_tokens", "tokens_per_second", "decode_tokens_per_second"]:
            v = ok[c].dropna().astype(float).values
            if len(v) == 0:
                continue
            r[f"{c}_mean"] = float(np.mean(v))
            r[f"{c}_p50"] = float(np.median(v))
            r[f"{c}_p95"] = float(np.percentile(v, 95))
            r[f"{c}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        decode_steps = (ok["generated_tokens"] - 1).clip(lower=0)
        mask = decode_steps > 0
        if mask.any():
            r["decode_ms_per_token_p50"] = float(np.median(ok.loc[mask, "decode_ms"] / decode_steps[mask]))
        for c in ("peak_allocated_mb", "peak_reserved_mb"):
            v = ok[c].dropna().astype(float)
            r[f"{c}_p50"] = float(v.median()) if len(v) else float("nan")
            r[f"{c}_max"] = float(v.max()) if len(v) else float("nan")
        for c in ("energy_j", "average_power_w"):
            v = ok[c].dropna().astype(float)
            r[f"{c}_median"] = float(v.median()) if len(v) else float("nan")
            r[f"{c}_mean"] = float(v.mean()) if len(v) else float("nan")
        r["visual_tokens_before_mean"] = float(ok["num_visual_tokens_before"].mean())
        r["visual_tokens_after_mean"] = float(ok["num_visual_tokens_after"].mean())
        r["achieved_retention"] = r["visual_tokens_after_mean"] / r["visual_tokens_before_mean"]
        r["prefill_seq_len_mean"] = float(ok["prefill_seq_len"].mean())
        r["frac_prefill_ge_1024"] = float((ok["prefill_seq_len"] >= 1024).mean())
        out.append(r)
    agg = pd.DataFrame(out)
    order = {"none": 0, "visionzip": 1, "uniform": 2}
    agg["_o"] = (agg["attention_backend"].map({"sdpa": 0, "eager": 1, "flash_attention_2": 2}) * 10
                 + (agg["awq_dequant_threshold"] != 1024) * 5 + agg["token_method"].map(order))
    agg = agg.sort_values(["dataset", "_o", "token_retention"], ascending=[True, True, False]).drop(columns="_o")
    return agg.reset_index(drop=True)


def relative_table(df: pd.DataFrame, agg: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    """Table 3: changes relative to the baseline with paired bootstrap CIs."""
    rows = []
    for ds in agg["dataset"].unique():
        base = df[(df["config_name"] == baseline) & (df["dataset"] == ds) & (df["status"] == "ok")].set_index("sample_id")
        base_q = agg[(agg["config_name"] == baseline) & (agg["dataset"] == ds)]["quality"].iloc[0]
        for cfg in agg[agg["dataset"] == ds]["config_name"]:
            cur = df[(df["config_name"] == cfg) & (df["dataset"] == ds) & (df["status"] == "ok")].set_index("sample_id")
            ids = base.index.intersection(cur.index)
            b, c = base.loc[ids], cur.loc[ids]
            q = agg[(agg["config_name"] == cfg) & (agg["dataset"] == ds)]["quality"].iloc[0]
            dq, dq_lo, dq_hi = paired_delta_ci(b["score"].astype(float), c["score"].astype(float))
            row = {"config_name": cfg, "dataset": ds, "n_paired": len(ids),
                   "quality_retention_pct": 100 * q / base_q if base_q else float("nan"),
                   "quality_delta": dq, "quality_delta_ci_low": dq_lo, "quality_delta_ci_high": dq_hi}
            for metric, col in (("latency", "total_latency_ms"), ("ttft", "ttft_ms"), ("prefill", "prefill_ms"),
                                ("vision", "vision_ms"), ("vram", "peak_allocated_mb")):
                ratio, lo, hi = paired_ratio_ci(b[col].astype(float), c[col].astype(float))
                row[f"{metric}_reduction_pct"] = 100 * (1 - ratio)
                row[f"{metric}_reduction_ci_low"] = 100 * (1 - hi)
                row[f"{metric}_reduction_ci_high"] = 100 * (1 - lo)
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ theoretical cost model
def load_arch(model_dir: Optional[Path]) -> Dict:
    """Architecture numbers for the FLOP model (from config.json if available)."""
    arch = {"llm_layers": 36, "llm_hidden": 2048, "llm_ff": 11008, "llm_heads": 16, "llm_kv_heads": 2,
            "vit_depth": 32, "vit_hidden": 1280, "vit_ff": 3420, "vit_heads": 16, "window": 112, "patch": 14,
            "fullatt_blocks": 4}
    if model_dir and (model_dir / "config.json").exists():
        cfg = json.loads((model_dir / "config.json").read_text())
        tc = cfg.get("text_config", cfg)
        vc = cfg.get("vision_config", {})
        arch.update(llm_layers=tc.get("num_hidden_layers", 36), llm_hidden=tc.get("hidden_size", 2048),
                    llm_ff=tc.get("intermediate_size", 11008), llm_heads=tc.get("num_attention_heads", 16),
                    llm_kv_heads=tc.get("num_key_value_heads", 2), vit_depth=vc.get("depth", 32),
                    vit_hidden=vc.get("hidden_size", 1280), vit_ff=vc.get("intermediate_size", 3420),
                    vit_heads=vc.get("num_heads", 16), window=vc.get("window_size", 112),
                    fullatt_blocks=len(vc.get("fullatt_block_indexes", [7, 15, 23, 31])))
    return arch


def llm_prefill_flops(seq_len: float, a: Dict) -> float:
    """~2*params*L for projections/MLP plus causal attention (QK^T and AV)."""
    d, ff, L = a["llm_hidden"], a["llm_ff"], seq_len
    kv = d // a["llm_heads"] * a["llm_kv_heads"]
    per_layer_params = 2 * d * d + 2 * d * kv + 3 * d * ff
    proj = 2 * per_layer_params * L
    attn = 2 * 2 * L * L * d / 2  # QK^T + AV, causal half
    return a["llm_layers"] * (proj + attn)


def vit_flops(n_llm_tokens: float, a: Dict) -> float:
    """Vision encoder FLOPs: 4 patches per LLM token; windowed attention except full blocks."""
    s = 4 * n_llm_tokens
    d, ff = a["vit_hidden"], a["vit_ff"]
    proj = 2 * (4 * d * d + 3 * d * ff) * s
    win = (a["window"] // a["patch"]) ** 2  # patches per window (8x8 = 64)
    full, windowed = a["fullatt_blocks"], a["vit_depth"] - a["fullatt_blocks"]
    attn_full = 4 * s * s * d
    attn_win = 4 * s * win * d
    return a["vit_depth"] * proj + full * attn_full + windowed * attn_win
