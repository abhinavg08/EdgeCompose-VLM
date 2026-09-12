"""Erratum check: fp16 vs bf16 vision tower on the same TextVQA samples (paired).

Compares the final sweep (fp16 ViT) with the targeted re-run (bf16 ViT, tag visionbf16):
degenerate '!!!!' outputs, accuracy on overflow vs healthy samples, answer agreement and
latency. Writes results/aggregate/vision_dtype_erratum.csv and prints a summary.

    python scripts/compare_vision_dtype.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import pandas as pd

from edgecompose.utils import RESULTS_DIR


def main() -> None:
    rows = []
    ids = [l.strip() for l in open(RESULTS_DIR / "aggregate" / "bf16_validation_ids.txt") if l.strip()]
    for cfg in ("C0_sdpa_r100", "T1_sdpa_awq64_r75", "C3_sdpa_r25", "U3_sdpa_uniform_r25"):
        a = pd.read_json(RESULTS_DIR / "raw" / f"{cfg}__textvqa_500.jsonl", lines=True).set_index("sample_id")
        b = pd.read_json(RESULTS_DIR / "raw" / f"{cfg}__textvqa_500__visionbf16.jsonl", lines=True).set_index("sample_id")
        a, b = a.loc[ids], b.loc[ids]
        over = a["prediction"].astype(str).str.contains("!!!")
        rows.append({
            "config_name": cfg, "n": len(ids),
            "degenerate_fp16": int(over.sum()),
            "degenerate_bf16": int(b["prediction"].astype(str).str.contains("!!!").sum()),
            "acc_fp16_on_overflow_samples": float(a.loc[over, "score"].mean()) if over.any() else float("nan"),
            "acc_bf16_on_overflow_samples": float(b.loc[over, "score"].mean()) if over.any() else float("nan"),
            "acc_fp16_on_healthy": float(a.loc[~over, "score"].mean()),
            "acc_bf16_on_healthy": float(b.loc[~over, "score"].mean()),
            "answers_identical_healthy": float((a.loc[~over, "prediction"] == b.loc[~over, "prediction"]).mean()),
            "vision_ms_fp16_p50": float(a["vision_ms"].median()), "vision_ms_bf16_p50": float(b["vision_ms"].median()),
            "e2e_ms_fp16_p50": float(a["total_latency_ms"].median()), "e2e_ms_bf16_p50": float(b["total_latency_ms"].median()),
            "peak_alloc_fp16_max": float(a["peak_allocated_mb"].max()), "peak_alloc_bf16_max": float(b["peak_allocated_mb"].max()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "aggregate" / "vision_dtype_erratum.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).T.to_string())

    # Corrected TextVQA quality of the final sweep: exclude samples that overflowed in ANY config
    import glob

    from edgecompose.analysis.aggregate import bootstrap_ci, paired_delta_ci

    allr = pd.concat([pd.read_json(f, lines=True) for f in glob.glob(str(RESULTS_DIR / "raw" / "*__textvqa_500.jsonl"))])
    overflow = set(allr[allr["prediction"].astype(str).str.contains("!!!")]["sample_id"])
    clean = allr[~allr["sample_id"].isin(overflow)]
    base = clean[clean["config_name"] == "C0_sdpa_r100"].set_index("sample_id")["score"]
    out = []
    for cfg, g in clean.groupby("config_name"):
        s = g.set_index("sample_id")["score"].loc[base.index]
        lo, hi = bootstrap_ci(s.values)
        d, dlo, dhi = paired_delta_ci(base.values, s.values)
        out.append({"config_name": cfg, "n_clean": len(s), "quality_clean": s.mean(), "ci_low": lo, "ci_high": hi,
                    "retention_vs_C0_pct": 100 * s.mean() / base.mean(), "delta_pp": 100 * d,
                    "delta_ci_low_pp": 100 * dlo, "delta_ci_high_pp": 100 * dhi,
                    "quality_as_reported_n500": allr[allr["config_name"] == cfg]["score"].mean()})
    q = pd.DataFrame(out).sort_values("config_name")
    for vz, un in (("C2_sdpa_r50", "U2_sdpa_uniform_r50"), ("C3_sdpa_r25", "U3_sdpa_uniform_r25")):
        a = clean[clean["config_name"] == un].set_index("sample_id")["score"].loc[base.index]
        b = clean[clean["config_name"] == vz].set_index("sample_id")["score"].loc[base.index]
        d, dlo, dhi = paired_delta_ci(a.values, b.values)
        print(f"clean VisionZip - uniform ({vz} vs {un}): {100 * d:+.1f} pp [{100 * dlo:+.1f}, {100 * dhi:+.1f}]")
    q.to_csv(RESULTS_DIR / "aggregate" / "textvqa_quality_clean484.csv", index=False)
    print(f"\n{len(overflow)} overflow samples excluded; corrected TextVQA quality (paired, n={len(base)}):")
    print(q.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
