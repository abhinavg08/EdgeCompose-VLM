"""EdgeInspect stage A3: one pushpins query with 1 reference (100% tokens), end to end.

Saves prompt, references, query, raw output, prediction, anomaly score, latency, TTFT and
VRAM to results/edgeinspect/smoke_test.json. Optionally probes larger k / pixel budgets
(--probe) to size the experiment grid.

    python scripts/inspect_smoke_test.py
    python scripts/inspect_smoke_test.py --probe
"""
from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

import edgeinspect  # noqa: F401
from edgecompose.compression import build_compressor
from edgecompose.utils import RESULTS_DIR, setup_logging, write_json
from edgeinspect.inference.classify import classify
from edgeinspect.inference.explain import explain
from edgeinspect.prompts.inspection import classification_content, render_text
from edgeinspect.references.sampler import load_queries, load_references
from edgeinspect.runner import InspectConfig, build_runner, load_image

logger = logging.getLogger("inspect_smoke_test")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default="pushpins")
    ap.add_argument("--max-pixels-tokens", type=int, default=512, help="per-image visual-token budget (x 28^2 px)")
    ap.add_argument("--probe", action="store_true", help="also time k in {1,4,8} x r in {1.0,0.5}")
    args = ap.parse_args()
    setup_logging()
    cfg = InspectConfig(max_pixels=args.max_pixels_tokens * 28 * 28)
    runner = build_runner(cfg)
    refs = load_references(args.category, 8, 0)
    queries = load_queries(args.category, "dev")
    q_anom = next(q for q in queries if q.label != "normal")
    q_norm = next(q for q in queries if q.label == "normal")
    ref_imgs = [load_image(r) for r in refs]
    report = {"config": cfg.__dict__, "prompt_k1": render_text(classification_content(args.category, 1))}

    for q in (q_norm, q_anom):  # warm up both paths once
        classify(runner, args.category, ref_imgs[:1], load_image(q), None)
    results = []
    for q in (q_norm, q_anom):
        res = classify(runner, args.category, ref_imgs[:1], load_image(q), None)
        res.update(query=q.sample_id, label=q.label, references=[refs[0].sample_id], k=1, retention=1.0)
        logger.info("k=1 r=1.00 | %-45s label=%-19s -> %-9s (raw %r) p_anom=%.3f | TTFT %.0f ms, E2E %.0f ms, "
                    "peak alloc %.0f MB | visual tokens %d", q.sample_id, q.label, res["prediction_label"],
                    res["raw_output"], res["anomaly_score"], res["ttft_ms"], res["total_latency_ms"],
                    res["peak_allocated_mb"], res["visual_tokens_before"])
        results.append(res)
    ex = explain(runner, args.category, ref_imgs[:1], load_image(q_anom))
    logger.info("explanation: %s", ex["raw_explanation"])
    report["single"] = results
    report["explanation_example"] = {**ex, "query": q_anom.sample_id, "label": q_anom.label}

    if args.probe:
        probe = []
        for k in (1, 4, 8):
            for r in (1.0, 0.5):
                comp = build_compressor("visionzip", r)
                classify(runner, args.category, ref_imgs[:k], load_image(q_anom), comp)  # warm
                res = classify(runner, args.category, ref_imgs[:k], load_image(q_anom), comp)
                row = {"k": k, "retention": r, **{x: res.get(x) for x in (
                    "status", "prediction_label", "anomaly_score", "vision_ms", "prefill_ms", "ttft_ms",
                    "total_latency_ms", "peak_allocated_mb", "peak_reserved_mb", "visual_tokens_before",
                    "visual_tokens_after", "prefill_seq_len", "error_message")}}
                probe.append(row)
                logger.info("probe k=%d r=%.2f: %s", k, r, {a: (round(b, 1) if isinstance(b, float) else b)
                                                            for a, b in row.items()})
        report["probe"] = probe
    out = RESULTS_DIR / "edgeinspect" / f"smoke_test_px{args.max_pixels_tokens}.json"
    write_json(report, out)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
