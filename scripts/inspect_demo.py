"""EdgeInspect-VLM demo: inspect one image against k known-good references.

    python scripts/inspect_demo.py --category pushpins --references 4 --token-retention 0.75 --query path/to/image.png

(Named inspect_demo.py, not inspect.py: a scripts/inspect.py would shadow Python's standard
`inspect` module for every script in this directory.)

References are the saved seed-0 manifest (train/good). The decision uses the anomaly score
log P(ANOMALOUS) - log P(NORMAL) against the threshold calibrated on *normal validation images
only* for this (category, k, retention) in the final grid; the raw generated word is shown for
transparency (the model is biased towards "ANOMALOUS"). If the query is flagged, Stage B
produces a short explanation (timed separately). All numbers are measured on this run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import edgeinspect  # noqa: F401
import pandas as pd
import torch
from PIL import Image

from edgecompose.compression import build_compressor
from edgecompose.utils import RESULTS_DIR
from edgeinspect.inference.classify import classify
from edgeinspect.inference.explain import explain
from edgeinspect.references.sampler import load_references
from edgeinspect.runner import InspectConfig, build_runner, load_image


def calibrated_threshold(category: str, k: int, r: float) -> float | None:
    p = RESULTS_DIR / "edgeinspect" / "aggregate" / "final_per_category.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    row = t[(t["category"] == category) & (t["seed"] == 0) & (t["k"] == k) & (t["retention"] == r)]
    return float(row["threshold_llr"].iloc[0]) if len(row) and pd.notna(row["threshold_llr"].iloc[0]) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", required=True)
    ap.add_argument("--references", type=int, default=4)
    ap.add_argument("--token-retention", type=float, default=0.75)
    ap.add_argument("--query", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--warmup", type=int, default=1, help="warmup runs before the measured one")
    ap.add_argument("--json-out", default=None, help="also save the full result as JSON")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    runner = build_runner(InspectConfig())
    refs = load_references(args.category, args.references, args.seed)
    ref_imgs = [load_image(r) for r in refs]
    q = Image.open(Path(args.query)).convert("RGB")
    comp = build_compressor("visionzip", args.token_retention)
    for _ in range(args.warmup):
        classify(runner, args.category, ref_imgs, q, comp)
    torch.cuda.empty_cache()
    res = classify(runner, args.category, ref_imgs, q, comp)
    if res["status"] != "ok":
        print(f"FAILED: {res['status']} {res.get('error_message', '')}")
        sys.exit(1)
    llr = res["logp_anomalous"] - res["logp_normal"]
    tau = calibrated_threshold(args.category, args.references, args.token_retention)
    flagged = (llr > tau) if tau is not None else bool(res["prediction"])
    ex = explain(runner, args.category, ref_imgs, q, comp) if flagged and not args.no_explain else None
    gpu = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2

    lines = [
        "EdgeInspect-VLM",
        "",
        "Model:            Qwen2.5-VL-3B-Instruct-AWQ (INT4 LLM, bf16 vision tower, SDPA, tuned AWQ dispatch)",
        f"Hardware:         {gpu} ({total:.0f} MB)",
        f"Category:         {args.category}",
        f"Reference count:  {len(refs)}  (train/good: {', '.join(r.sample_id.split('/')[-1] for r in refs)}; seed {args.seed})",
        f"Token retention:  {args.token_retention:.0%}  (VisionZip per image; visual tokens {res['visual_tokens_before']} -> {res['visual_tokens_after']})",
        f"Query:            {args.query}",
        "",
        f"Prediction:       {'ANOMALOUS' if flagged else 'NORMAL'}",
        f"Anomaly score:    {llr:+.3f}  = log P(ANOMALOUS) - log P(NORMAL)"
        + (f"; normal-only calibrated threshold {tau:+.3f}" if tau is not None else "; (no calibrated threshold found - raw answer used)"),
        f"Raw model answer: {res['raw_output']!r}",
    ]
    if ex:
        lines += [f"Anomaly type:     {ex['anomaly_type'] or 'unparsed'}",
                  f"Issue:            {ex['issue']}",
                  f"Explanation:      {ex['explanation'] or ex['raw_explanation']}",
                  f"                  (explanation pass {ex['explain_latency_ms']:.0f} ms, not included below)"]
    lines += ["",
              "Performance (classification pass, measured):",
              f"  TTFT:           {res['ttft_ms']:.0f} ms  (vision {res['vision_ms']:.0f} ms, prefill {res['prefill_ms']:.0f} ms)",
              f"  Latency:        {res['total_latency_ms']:.0f} ms",
              f"  Peak VRAM:      {res['peak_allocated_mb']:.0f} MB allocated / {res['peak_reserved_mb']:.0f} MB reserved by the process"]
    print("\n".join(lines))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps({"stdout": lines, "classification": res, "explanation": ex,
                                                   "threshold_llr": tau, "score_llr": llr,
                                                   "references": [r.sample_id for r in refs]}, indent=2, default=str))


if __name__ == "__main__":
    main()
