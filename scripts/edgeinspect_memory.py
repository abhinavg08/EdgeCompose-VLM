"""EdgeInspect RQ3: isolated VRAM / latency scaling with reference count, per token retention.

For each retention r and k in a list that deliberately extends past 8 (to find the real
limit on this 8 GB GPU), run a few queries in isolation (empty cache + peak reset) and
record peak allocated / reserved memory, device footprint (reserved + CUDA context and
other processes), latency and OOM. A configuration that OOMs is recorded and larger k for
the same r are still attempted once (memory is not guaranteed monotone in k), up to
`--stop-after` consecutive OOMs.

    python scripts/edgeinspect_memory.py --category pushpins --ks 1 2 4 8 12 16 24 --retentions 1.0 0.75 0.5 0.25
"""
from __future__ import annotations

import argparse
import gc
import logging

import _bootstrap  # noqa: F401

import edgeinspect  # noqa: F401
import pandas as pd
import torch

from edgecompose.compression import build_compressor
from edgecompose.profiling import memory
from edgecompose.utils import RESULTS_DIR, setup_logging
from edgeinspect.inference.classify import classify
from edgeinspect.references.sampler import load_queries, sample_references
from edgeinspect.runner import InspectConfig, build_runner, load_image

logger = logging.getLogger("edgeinspect_memory")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default="pushpins")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24])
    ap.add_argument("--retentions", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    ap.add_argument("--queries", type=int, default=3)
    ap.add_argument("--max-pixels-tokens", type=int, default=512)
    ap.add_argument("--stop-after", type=int, default=2, help="consecutive OOM k-values before stopping an r")
    ap.add_argument("--max-latency-s", type=float, default=60.0,
                    help="stop scanning larger k for an r once a query exceeds this (sysmem-spill guard)")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    setup_logging()
    cfg = InspectConfig(attention_backend=args.attn, max_pixels=args.max_pixels_tokens * 28 * 28)
    runner = build_runner(cfg)
    torch.cuda.empty_cache()
    context_mb = memory.device_used_mb() - torch.cuda.memory_reserved() / memory.MB
    total_mb = torch.cuda.get_device_properties(0).total_memory / memory.MB
    logger.info("CUDA context + other processes: %.0f MB; device total %.0f MB", context_mb, total_mb)
    refs = sample_references(args.category, max(args.ks), seed=0)
    ref_imgs = [load_image(r) for r in refs]
    qs = [q for q in load_queries(args.category, "dev") if q.label != "normal"][: args.queries]
    q_imgs = [load_image(q) for q in qs]
    rows = []
    for r in args.retentions:
        consecutive = 0
        for k in args.ks:
            comp = build_compressor("visionzip", r)
            gc.collect()
            torch.cuda.empty_cache()
            memory.reset_peak()
            status, lat, ttft, err = "ok", [], [], ""
            for qi in q_imgs:
                res = classify(runner, args.category, ref_imgs[:k], qi, comp, score=False)
                if res["status"] != "ok":
                    status, err = res["status"], res.get("error_message", "")
                    break
                lat.append(res["total_latency_ms"])
                ttft.append(res["ttft_ms"])
                vt_before, vt_after, seq = res["visual_tokens_before"], res["visual_tokens_after"], res["prefill_seq_len"]
            snap = memory.snapshot()
            row = {"category": args.category, "k": k, "retention": r, "status": status, "error_message": err,
                   "peak_allocated_mb": snap["peak_allocated_mb"], "peak_reserved_mb": snap["peak_reserved_mb"],
                   "context_overhead_mb": context_mb, "device_footprint_mb": snap["peak_reserved_mb"] + context_mb,
                   "latency_ms_median": float(pd.Series(lat).median()) if lat else None,
                   "ttft_ms_median": float(pd.Series(ttft).median()) if ttft else None,
                   "visual_tokens_before": vt_before if lat else None, "visual_tokens_after": vt_after if lat else None,
                   "prefill_seq_len": seq if lat else None, "max_pixels_tokens": args.max_pixels_tokens,
                   "attention_backend": args.attn, "device_total_mb": total_mb}
            # Windows' default "CUDA sysmem fallback" spills to shared system RAM instead of raising
            # OOM, so exceeding VRAM shows up as footprint > device total and/or a latency cliff.
            row["exceeds_vram"] = bool(row["device_footprint_mb"] > total_mb)
            if status == "ok" and row["exceeds_vram"]:
                row["status"] = status = "over_vram"
            rows.append(row)
            logger.info("r=%.2f k=%2d %-3s alloc %6.0f  reserved %6.0f  footprint %6.0f MB  lat %s ms  tokens %s->%s",
                        r, k, status, row["peak_allocated_mb"], row["peak_reserved_mb"], row["device_footprint_mb"],
                        None if not lat else round(row["latency_ms_median"]), row["visual_tokens_before"],
                        row["visual_tokens_after"])
            consecutive = consecutive + 1 if status in ("OOM", "over_vram") else 0
            too_slow = bool(lat) and max(lat) > args.max_latency_s * 1e3
            if consecutive >= args.stop_after or too_slow:
                logger.info("stopping k-scan for r=%.2f (%s)", r, "latency guard" if too_slow else "repeated OOM/over-VRAM")
                break
    out = RESULTS_DIR / "edgeinspect" / "aggregate" / f"memory_scaling_{args.attn}_px{args.max_pixels_tokens}{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
