"""EdgeInspect grid runner (stages A4-A6).

    # A4: 20 normal + 20 anomalous, k=1, 100% tokens
    python scripts/run_edgeinspect.py --categories pushpins --ks 1 --retentions 1.0 --queries dev --limit 40 --tag a4
    # A5: development grid
    python scripts/run_edgeinspect.py --categories pushpins splicing_connectors --ks 1 4 --retentions 1.0 0.5 --queries dev --tag dev
    # A6: full grid
    python scripts/run_edgeinspect.py --categories breakfast_box juice_bottle pushpins screw_bag splicing_connectors \
        --ks 1 2 4 8 --retentions 1.0 0.75 0.5 0.25 --queries final --seeds 0 --tag final

Rows go to results/edgeinspect/raw/<tag>__<category>__seed<seed>.jsonl (resumable).
"""
from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

import edgeinspect  # noqa: F401
from edgecompose.utils import RESULTS_DIR, set_seed, setup_logging
from edgeinspect.runner import InspectConfig, build_runner, run_category, save_run_config

logger = logging.getLogger("run_edgeinspect")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categories", nargs="+", default=["pushpins", "splicing_connectors"])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--retentions", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--queries", default="dev", help="query manifest name (dev | final)")
    ap.add_argument("--limit", type=int, default=0, help="first N queries (label-interleaved)")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--max-pixels-tokens", type=int, default=512, help="per-image visual-token budget")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--awq-threshold", type=int, default=64)
    ap.add_argument("--no-compress-query", action="store_true", help="compress references only")
    ap.add_argument("--no-power", action="store_true")
    args = ap.parse_args()
    setup_logging(log_file=RESULTS_DIR / "logs" / "edgeinspect.log")
    set_seed(1234)
    cfg = InspectConfig(attention_backend=args.attn, awq_dequant_threshold=args.awq_threshold,
                        max_pixels=args.max_pixels_tokens * 28 * 28, compress_query=not args.no_compress_query)
    save_run_config(cfg, RESULTS_DIR / "edgeinspect" / "run_configs" / f"{args.tag}.json", categories=args.categories,
                    ks=args.ks, retentions=args.retentions, seeds=args.seeds, queries=args.queries, limit=args.limit)
    runner = build_runner(cfg)
    for seed in args.seeds:
        for cat in args.categories:
            p = run_category(runner, cfg, cat, args.ks, args.retentions, seed, args.queries, args.tag, args.limit,
                             measure_power=not args.no_power)
            logger.info("%s seed%d -> %s", cat, seed, p)


if __name__ == "__main__":
    main()
