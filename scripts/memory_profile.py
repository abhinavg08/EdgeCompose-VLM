"""Isolated per-configuration VRAM footprint (complements the interleaved benchmark).

In the interleaved benchmark the caching allocator is shared by all configurations, so
`peak_reserved` is not attributable to one configuration. Here each configuration runs
alone after `torch.cuda.empty_cache()` + peak reset, on the K samples with the most
visual tokens, and we record:

    peak_allocated_mb   live-tensor peak
    peak_reserved_mb    allocator footprint (what the process actually holds)
    device_footprint_mb peak_reserved + CUDA context/runtime overhead (driver view), i.e. the
                        number that must fit in the 8 GB card next to the display/other apps

    python scripts/memory_profile.py --n 200 --k 10
"""
from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

import edgecompose  # noqa: F401
import pandas as pd
import torch

from edgecompose.compression import build_compressor
from edgecompose.datasets.base import load_manifest, manifest_path
from edgecompose.models.qwen_vl import QwenVLRunner
from edgecompose.profiling import memory
from edgecompose.utils import RESULTS_DIR, load_config, load_yaml, CONFIG_DIR, setup_logging

logger = logging.getLogger("memory_profile")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="sweep_main.yaml")
    ap.add_argument("--datasets", nargs="+", default=["textvqa", "pope"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    setup_logging()
    files = [f for g in load_yaml(CONFIG_DIR / args.sweep)["groups"].values() for f in g]
    cfgs = [load_config(f) for f in files]
    runner = QwenVLRunner(cfgs[0].model_id, attention_backend=cfgs[0].attention_backend,
                          min_pixels=cfgs[0].min_pixels, max_pixels=cfgs[0].max_pixels)
    torch.cuda.empty_cache()
    context_mb = memory.device_used_mb() - torch.cuda.memory_reserved() / memory.MB
    logger.info("CUDA context + non-PyTorch device usage: %.0f MB (includes other processes)", context_mb)
    rows = []
    for ds in args.datasets:
        samples = load_manifest(manifest_path(ds, args.n))
        # the largest images are the memory-critical ones
        samples = sorted(samples, key=lambda s: -(s.load_image().size[0] * s.load_image().size[1]))[: args.k]
        imgs = [s.load_image() for s in samples]
        for c in cfgs:
            runner.set_attention_backend(c.attention_backend)
            comp = build_compressor(c.token_method, c.token_retention,
                                    **({"contextual_ratio": c.contextual_ratio} if c.token_method == "visionzip" else {}))
            runner.run(imgs[0], samples[0].prompt, compressor=comp, max_new_tokens=c.max_new_tokens)  # warm kernels
            torch.cuda.empty_cache()
            memory.reset_peak()
            for s, im in zip(samples, imgs):
                runner.run(im, s.prompt, compressor=comp, max_new_tokens=c.max_new_tokens)
            snap = memory.snapshot()
            rows.append({"config_name": c.name, "dataset": ds, "k": len(samples),
                         "peak_allocated_mb": snap["peak_allocated_mb"], "peak_reserved_mb": snap["peak_reserved_mb"],
                         "context_overhead_mb": context_mb,
                         "device_footprint_mb": snap["peak_reserved_mb"] + context_mb})
            logger.info("%-22s %-8s alloc %.0f  reserved %.0f  footprint %.0f MB", c.name, ds,
                        snap["peak_allocated_mb"], snap["peak_reserved_mb"], snap["peak_reserved_mb"] + context_mb)
    out = RESULTS_DIR / "aggregate" / "memory_isolated.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
