"""Stage 8: pick a deployment configuration from *measured* results under constraints.

Examples
    python scripts/optimize.py --max-vram-mb 7000 --max-latency-ms 1500 --min-quality 0.90 --relative-quality
    python scripts/optimize.py --objective latency --min-quality 0.75 --datasets textvqa
    python scripts/optimize.py --objective memory --min-quality 0.95 --relative-quality
    python scripts/optimize.py --objective energy --max-ttft-ms 800

--min-quality is absolute (metric units) unless --relative-quality is given, in which case
it is the fraction of the C0 (SDPA, 100% tokens) baseline quality on each dataset.
Constraints must hold on every selected dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from edgecompose.optimizer.selector import OBJECTIVES, Constraints, select
from edgecompose.utils import RESULTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS_DIR / "results.csv"))
    ap.add_argument("--objective", default="quality", choices=list(OBJECTIVES))
    ap.add_argument("--max-vram-mb", type=float, default=None, help="peak CUDA memory reserved (run max)")
    ap.add_argument("--max-latency-ms", type=float, default=None)
    ap.add_argument("--max-ttft-ms", type=float, default=None)
    ap.add_argument("--min-quality", type=float, default=None)
    ap.add_argument("--relative-quality", action="store_true")
    ap.add_argument("--latency-stat", default="p50", choices=["p50", "p95", "mean"])
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        sys.exit(f"{path} not found - run scripts/analyze.py --main first")
    agg = pd.read_csv(path)
    c = Constraints(max_vram_mb=args.max_vram_mb, max_latency_ms=args.max_latency_ms, max_ttft_ms=args.max_ttft_ms,
                    min_quality=args.min_quality, relative_quality=args.relative_quality, datasets=args.datasets,
                    latency_stat=args.latency_stat)
    res = select(agg, c, args.objective, args.top_k)
    if args.json:
        print(json.dumps(res, indent=2, default=str))
        return
    qlabel = "quality/baseline" if args.relative_quality else "quality"
    print(f"objective: {args.objective} | datasets: {args.datasets or 'all'} | latency stat: {args.latency_stat}")
    if res["best"] is None:
        print("NO FEASIBLE CONFIGURATION. Rejections:")
        for cfg, why in res["rejected"].items():
            print(f"  {cfg}: " + "; ".join(why))
        sys.exit(2)
    b = res["best"]
    print(f"BEST: {b['config_name']}  {qlabel}={b['quality']:.3f}  latency={b['latency_ms']:.0f} ms  "
          f"TTFT={b['ttft_ms']:.0f} ms  peak VRAM={b['peak_vram_mb']:.0f} MB  energy={b['energy_j']:.1f} J")
    print("ranking (feasible):")
    for r in res["ranking"]:
        print(f"  {r['config_name']:22s} {qlabel}={r['quality']:.3f}  lat={r['latency_ms']:7.0f} ms  "
              f"ttft={r['ttft_ms']:6.0f} ms  vram={r['peak_vram_mb']:6.0f} MB  energy={r['energy_j']:.1f} J")
    if res["rejected"]:
        print(f"rejected: {len(res['rejected'])} configs (use --json for reasons)")


if __name__ == "__main__":
    main()
