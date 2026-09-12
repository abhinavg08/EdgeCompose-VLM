"""Run one or more configurations on one or more datasets (Stage 3/4 entry point).

    # baseline on 20 dev samples
    python scripts/benchmark.py --configs baseline.yaml --datasets textvqa --n 200 --limit 20
    # the retention sweep under SDPA, dev subsets
    python scripts/benchmark.py --configs baseline.yaml token_75.yaml token_50.yaml token_25.yaml --n 200

All configs passed together share one loaded model and are interleaved per sample
(see edgecompose/benchmark.py). Re-running resumes: completed (config, sample) pairs are
skipped. Aggregate tables are produced by scripts/analyze.py.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

import edgecompose  # noqa: F401
from edgecompose.benchmark import run_group
from edgecompose.datasets.base import load_manifest, manifest_path
from edgecompose.utils import RESULTS_DIR, load_config, set_seed, setup_logging, write_json

logger = logging.getLogger("benchmark")


def warmup_set(dataset: str, eval_ids: set, k: int):
    """Warmup samples drawn from a larger manifest but excluded from evaluation."""
    for n in (5000, 3000, 1000, 600, 500, 400, 300, 200):
        p = manifest_path(dataset, n)
        if p.exists():
            extra = [s for s in load_manifest(p) if s.sample_id not in eval_ids]
            if len(extra) >= k:
                return extra[-k:]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", required=True, help="YAML files (in configs/ or paths)")
    ap.add_argument("--datasets", nargs="+", default=["textvqa", "pope"])
    ap.add_argument("--n", type=int, default=200, help="manifest size to use (data/manifests/<ds>_<n>.jsonl)")
    ap.add_argument("--limit", type=int, default=None, help="only evaluate the first LIMIT samples")
    ap.add_argument("--repeats", type=int, default=1, help="repeat the full pass (latency variability)")
    ap.add_argument("--ignore-eos", action="store_true", help="force max_new_tokens (decode profiling)")
    ap.add_argument("--tag", default="", help="suffix for raw result files")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--no-power", action="store_true")
    ap.add_argument("--vision-dtype", default="bfloat16", choices=["float16", "bfloat16"],
                    help="vision-tower dtype (default bfloat16). The reported EdgeCompose sweeps ran with float16, "
                         "which overflows on ~3%% of TextVQA images - pass float16 to reproduce them (see report erratum)")
    ap.add_argument("--ids-file", default=None, help="optional text file of sample_ids to evaluate (one per line)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    setup_logging(log_file=results_dir / "logs" / "benchmark.log")
    configs = [load_config(c) for c in args.configs]
    base = configs[0]
    for c in configs[1:]:
        if (c.model_id, c.min_pixels, c.max_pixels) != (base.model_id, base.min_pixels, base.max_pixels):
            raise SystemExit("configs in one invocation must share model_id and pixel budget")
    set_seed(base.seed)

    from edgecompose.models.qwen_vl import QwenVLRunner

    import torch

    runner = QwenVLRunner(base.model_id, attention_backend=base.attention_backend,
                          min_pixels=base.min_pixels, max_pixels=base.max_pixels,
                          vision_dtype=getattr(torch, args.vision_dtype))
    if not args.tag:
        write_json(runner.describe(), results_dir / "model_load.json")
    n_label = args.n if args.limit is None else args.limit
    wanted = None
    if args.ids_file:
        wanted = {line.strip() for line in open(args.ids_file, encoding="utf-8") if line.strip()}
    for ds in args.datasets:
        samples = load_manifest(manifest_path(ds, args.n))
        if args.limit:
            samples = samples[: args.limit]
        if wanted is not None:
            samples = [s for s in samples if s.sample_id in wanted]
            if not samples:
                continue
        ids = {s.sample_id for s in samples}
        warm = warmup_set(ds, ids, base.warmup) or samples[: base.warmup]
        if warm and warm[0].sample_id in ids:
            logger.warning("no held-out warmup samples for %s; warming up on the first %d eval samples", ds, len(warm))
        paths = run_group(runner, configs, samples, warm, results_dir, n_label, ignore_eos=args.ignore_eos,
                          tag=args.tag, repeats=args.repeats, measure_power=not args.no_power)
        for name, p in paths.items():
            logger.info("%s -> %s", name, p)


if __name__ == "__main__":
    main()
