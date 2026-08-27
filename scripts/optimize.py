"""Stage 8 / A9: pick a deployment configuration from *measured* results under constraints.

Generic VQA (EdgeCompose, results/results.csv):
    python scripts/optimize.py --max-vram-mb 7000 --max-latency-ms 1500 --min-quality 0.90 --relative-quality
    python scripts/optimize.py --objective latency --min-quality 0.75 --datasets textvqa

Industrial inspection (EdgeInspect, results/edgeinspect/aggregate/edgeinspect_results.csv):
    python scripts/optimize.py --task industrial_inspection --max-vram-mb 7000 --max-latency-ms 2000 --min-f1 0.80
    python scripts/optimize.py --task industrial_inspection --min-references 4 --objective latency

--min-quality is absolute (metric units) unless --relative-quality is given, in which case
it is the fraction of the C0 (SDPA, 100% tokens) baseline quality on each dataset.
Constraints must hold on every selected dataset (VQA) / every category (inspection).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from edgecompose.optimizer.selector import (
    OBJECTIVES, Constraints, InspectionConstraints, select, select_inspection,
)
from edgecompose.utils import RESULTS_DIR


def _inspection(args) -> None:
    path = Path(args.results or RESULTS_DIR / "edgeinspect" / "aggregate" / "edgeinspect_results.csv")
    if not path.exists():
        sys.exit(f"{path} not found - run scripts/analyze_edgeinspect.py first")
    tab = pd.read_csv(path)
    c = InspectionConstraints(max_vram_mb=args.max_vram_mb, max_latency_ms=args.max_latency_ms, min_f1=args.min_f1,
                              min_references=args.min_references, quality_metric=args.quality_metric)
    res = select_inspection(tab, c, args.objective, args.top_k)
    if args.json:
        print(json.dumps(res, indent=2, default=str))
        return
    q = args.quality_metric
    print(f"task: industrial_inspection | objective: {args.objective} | quality metric: {q} (macro over categories)")
    if res["best"] is None:
        print("NO FEASIBLE CONFIGURATION. Rejections:")
        for cfg, why in res["rejected"].items():
            print(f"  {cfg}: " + "; ".join(why))
        sys.exit(2)
    b = res["best"]
    print("RECOMMENDED:")
    print(f"  reference count:  {int(b['k'])}")
    print(f"  token retention:  {b['retention']:.0%} (VisionZip)")
    print(f"  attention:        {b.get('attention_backend', 'sdpa')}")
    print(f"  measured {q}:  {b[q]:.3f}")
    print(f"  measured VRAM:    {b['vram_mb']:.0f} MB ({b.get('vram_source', 'peak allocated')})")
    print(f"  measured latency: {b['total_latency_ms_p50']:.0f} ms median (TTFT {b['ttft_ms_p50']:.0f} ms)")
    print("ranking (feasible):")
    for r in res["ranking"]:
        print(f"  {r['config']:16s} {q}={r[q]:.3f}  lat={r['total_latency_ms_p50']:6.0f} ms  vram={r['vram_mb']:6.0f} MB")
    if res["rejected"]:
        print(f"rejected: {len(res['rejected'])} configs (use --json for reasons)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="vqa", choices=["vqa", "industrial_inspection"])
    ap.add_argument("--results", default=None)
    ap.add_argument("--objective", default="quality", choices=list(OBJECTIVES))
    ap.add_argument("--max-vram-mb", type=float, default=None)
    ap.add_argument("--max-latency-ms", type=float, default=None)
    ap.add_argument("--max-ttft-ms", type=float, default=None)
    ap.add_argument("--min-quality", type=float, default=None)
    ap.add_argument("--relative-quality", action="store_true")
    ap.add_argument("--min-f1", type=float, default=None, help="inspection: minimum macro quality metric")
    ap.add_argument("--min-references", type=int, default=None, help="inspection: require at least k references")
    ap.add_argument("--quality-metric", default="f1_cal", choices=["f1_cal", "f1", "f1_max", "auroc", "balanced_accuracy_cal"])
    ap.add_argument("--latency-stat", default="p50", choices=["p50", "p95", "mean"])
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = ap.parse_args()
    if args.task == "industrial_inspection":
        _inspection(args)
        return

    path = Path(args.results or RESULTS_DIR / "results.csv")
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
