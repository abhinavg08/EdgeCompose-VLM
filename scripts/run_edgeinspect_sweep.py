"""EdgeInspect full sweep orchestrator (stage A6 + memory scaling + extra reference seeds).

Runs each step in a fresh process (clean CUDA state), sequentially, resumable:

  1. memory scaling probe (isolated VRAM/latency vs k up to 24, all retentions, SDPA)
  2. full grid, seed 0: 5 categories x k{1,2,4,8} x r{1.0,0.75,0.5,0.25} on the 'final' queries
  3. reference-seed variability: seeds 1,2 on k{1,4} x r{1.0,0.5}
  4. memory probe with eager attention (r 1.0 / 0.5) to show where the unoptimized stack fails

    python scripts/run_edgeinspect_sweep.py
    python scripts/run_edgeinspect_sweep.py --steps grid seeds
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from edgecompose.utils import setup_logging

logger = logging.getLogger("run_edgeinspect_sweep")
HERE = Path(__file__).resolve().parent
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

STEPS = {
    "memory": [HERE / "edgeinspect_memory.py", "--category", "pushpins", "--ks", "1", "2", "4", "8", "12", "16", "24",
               "--retentions", "1.0", "0.75", "0.5", "0.25"],
    "grid": [HERE / "run_edgeinspect.py", "--categories", *CATS, "--ks", "1", "2", "4", "8",
             "--retentions", "1.0", "0.75", "0.5", "0.25", "--queries", "final", "--seeds", "0", "--tag", "final"],
    "seeds": [HERE / "run_edgeinspect.py", "--categories", *CATS, "--ks", "1", "4", "--retentions", "1.0", "0.5",
              "--queries", "final", "--seeds", "1", "2", "--tag", "final"],
    "memory_eager": [HERE / "edgeinspect_memory.py", "--category", "pushpins", "--ks", "1", "2", "4", "8", "12",
                     "--retentions", "1.0", "0.5", "--attn", "eager", "--queries", "2"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", nargs="+", default=list(STEPS), choices=list(STEPS))
    args = ap.parse_args()
    setup_logging()
    for s in args.steps:
        cmd = [sys.executable, *[str(c) for c in STEPS[s]]]
        logger.info("step %s: %s", s, " ".join(cmd))
        r = subprocess.run(cmd)
        logger.info("step %s finished with exit code %d", s, r.returncode)


if __name__ == "__main__":
    main()
