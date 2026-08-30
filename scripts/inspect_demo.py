"""EdgeInspect-VLM demo: inspect one image against k known-good references.

    python scripts/inspect_demo.py --category pushpins --references 4 --token-retention 0.50 --query path/to/image.png
    python scripts/inspect_demo.py --category pushpins --references 4 --token-retention 0.50 --query path/to/image.png --json

References are the saved seed-0 manifest (train/good). Stage A (classification + anomaly
score) is always run; Stage B (explanation) runs when the query is classified ANOMALOUS
(disable with --no-explain). All reported numbers are measured on this run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import edgeinspect  # noqa: F401
from PIL import Image

from edgecompose.compression import build_compressor
from edgeinspect.inference.classify import classify
from edgeinspect.inference.explain import explain
from edgeinspect.references.sampler import load_references
from edgeinspect.runner import InspectConfig, build_runner, load_image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", required=True)
    ap.add_argument("--references", type=int, default=4)
    ap.add_argument("--token-retention", type=float, default=0.5)
    ap.add_argument("--query", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pixels-tokens", type=int, default=512)
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--warmup", type=int, default=1, help="warmup runs before the measured one")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = InspectConfig(max_pixels=args.max_pixels_tokens * 28 * 28)
    runner = build_runner(cfg)
    refs = load_references(args.category, args.references, args.seed)
    ref_imgs = [load_image(r) for r in refs]
    q = Image.open(Path(args.query)).convert("RGB")
    comp = build_compressor("visionzip", args.token_retention)
    for _ in range(args.warmup):
        classify(runner, args.category, ref_imgs, q, comp)
    res = classify(runner, args.category, ref_imgs, q, comp)
    if res["status"] != "ok":
        print(f"FAILED: {res['status']} {res.get('error_message', '')}")
        sys.exit(1)
    ex = None
    if res["prediction"] == 1 and not args.no_explain:
        ex = explain(runner, args.category, ref_imgs, q, comp)
    if args.json:
        print(json.dumps({"classification": res, "explanation": ex, "references": [r.sample_id for r in refs]},
                         indent=2, default=str))
        return
    print("EdgeInspect-VLM\n")
    print("Model:            Qwen2.5-VL-3B-Instruct-AWQ (INT4 AWQ, SDPA, tuned AWQ dispatch)")
    print(f"Category:         {args.category}")
    print(f"Reference images: {len(refs)}  ({', '.join(r.sample_id.split('/')[-1] for r in refs)} from train/good, seed {args.seed})")
    print(f"Token retention:  {args.token_retention:.0%}  (VisionZip; visual tokens {res['visual_tokens_before']} -> {res['visual_tokens_after']})")
    print(f"Query:            {args.query}\n")
    print(f"Prediction:       {res['prediction_label']}   (raw output: {res['raw_output']!r})")
    print(f"Anomaly score:    {res['anomaly_score']:.3f}   (P(ANOMALOUS) vs P(NORMAL) from answer likelihoods; not calibrated)")
    if ex:
        print(f"Type:             {ex['anomaly_type'] or 'unparsed'}")
        print(f"Issue:            {ex['issue']}")
        print(f"Explanation:      {ex['explanation'] or ex['raw_explanation']}")
        print(f"                  (explanation pass: {ex['explain_latency_ms']:.0f} ms, not included below)")
    print("\nPerformance (classification pass):")
    print(f"  TTFT:           {res['ttft_ms']:.0f} ms   (vision {res['vision_ms']:.0f} ms, prefill {res['prefill_ms']:.0f} ms)")
    print(f"  Latency:        {res['total_latency_ms']:.0f} ms")
    print(f"  Peak VRAM:      {res['peak_allocated_mb']:.0f} MB allocated / {res['peak_reserved_mb']:.0f} MB reserved")


if __name__ == "__main__":
    main()
