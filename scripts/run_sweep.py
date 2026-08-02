"""Run a sweep file (groups of configs) over the datasets - Stage 5/6 entry point.

    python scripts/run_sweep.py --sweep sweep_main.yaml --n 200
    python scripts/run_sweep.py --sweep sweep_main.yaml --n 1000 --datasets textvqa pope

Each group is executed by scripts/benchmark.py in a fresh process (clean CUDA context and
allocator state per group). Groups whose configs need an unavailable backend (e.g.
flash_attention_2 without the flash_attn package) are skipped with a logged reason.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import edgecompose  # noqa: F401
from edgecompose.attention.backends import flash_attn_available
from edgecompose.utils import CONFIG_DIR, load_config, load_yaml, setup_logging

logger = logging.getLogger("run_sweep")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="sweep_main.yaml")
    ap.add_argument("--datasets", nargs="+", default=["textvqa", "pope"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--groups", nargs="*", default=None, help="subset of group names to run")
    args = ap.parse_args()
    setup_logging()

    sweep_path = Path(args.sweep) if Path(args.sweep).exists() else CONFIG_DIR / args.sweep
    sweep = load_yaml(sweep_path)
    script = Path(__file__).resolve().parent / "benchmark.py"
    for gname, files in sweep["groups"].items():
        if args.groups and gname not in args.groups:
            continue
        cfgs = [load_config(f) for f in files]
        if any(c.attention_backend == "flash_attention_2" for c in cfgs) and not flash_attn_available():
            logger.warning("skipping group %s: flash_attn not usable in this environment", gname)
            continue
        cmd = [sys.executable, str(script), "--configs", *files, "--datasets", *args.datasets, "--n", str(args.n),
               "--repeats", str(args.repeats)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.ignore_eos:
            cmd.append("--ignore-eos")
        if args.tag:
            cmd += ["--tag", args.tag]
        logger.info("group %s: %s", gname, " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            logger.error("group %s exited with %d (partial results are resumable)", gname, r.returncode)


if __name__ == "__main__":
    main()
