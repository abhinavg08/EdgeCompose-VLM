"""Microbenchmark the two AutoAWQ W4A16 kernel paths vs activation rows M.

For every INT4 linear shape in Qwen2.5-VL-3B's LLM (per decoder layer: q/o 2048->2048,
k/v 2048->256, gate/up 2048->11008, down 11008->2048) and M in a sweep, time
  (a) Triton split-K fused GEMM   (AutoAWQ path for M < 1024)
  (b) Triton dequantize + cuBLAS  (AutoAWQ path for M >= 1024)
with CUDA events (median of repeats). Outputs results/aggregate/awq_kernel_bench.csv and
plots/fig10_awq_kernel_crossover.png, and prints the per-layer crossover M.

    python scripts/awq_kernel_bench.py
"""
from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

import edgecompose  # noqa: F401
import pandas as pd
import torch

from edgecompose.utils import PLOTS_DIR, RESULTS_DIR, setup_logging

logger = logging.getLogger("awq_kernel_bench")
SHAPES = {"q_proj/o_proj": (2048, 2048, 2), "k_proj/v_proj": (2048, 256, 2), "gate/up_proj": (2048, 11008, 2),
          "down_proj": (11008, 2048, 1)}  # (in, out, count per layer)


def _time(fn, reps: int) -> float:
    for _ in range(3):
        fn()
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, nargs="+", default=[1, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1023, 1024, 1536, 2048])
    ap.add_argument("--reps", type=int, default=15)
    args = ap.parse_args()
    setup_logging()
    from awq.modules.triton.gemm import awq_dequantize_triton, awq_gemm_triton

    rows = []
    g = 128
    for name, (k, n, cnt) in SHAPES.items():
        qweight = torch.randint(-2**31, 2**31 - 1, (k, n // 8), dtype=torch.int32, device="cuda")
        qzeros = torch.randint(-2**31, 2**31 - 1, (k // g, n // 8), dtype=torch.int32, device="cuda")
        scales = (torch.rand(k // g, n, dtype=torch.float16, device="cuda") * 0.01)
        for m in args.m:
            x = torch.randn(m, k, dtype=torch.float16, device="cuda")
            t_fused = _time(lambda: awq_gemm_triton(x, qweight, scales, qzeros, split_k_iters=8), args.reps)
            t_deq = _time(lambda: torch.matmul(x, awq_dequantize_triton(qweight, scales, qzeros)), args.reps)
            rows.append({"layer": name, "in": k, "out": n, "count_per_layer": cnt, "M": m,
                         "triton_gemm_ms": t_fused, "dequant_cublas_ms": t_deq})
            logger.info("%-14s M=%5d fused %.3f ms | dequant+cuBLAS %.3f ms", name, m, t_fused, t_deq)
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "aggregate" / "awq_kernel_bench.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # whole-LLM estimate: 36 layers x per-layer linear shapes
    tot = df.assign(f=df.triton_gemm_ms * df.count_per_layer * 36, d=df.dequant_cublas_ms * df.count_per_layer * 36)
    tot = tot.groupby("M")[["f", "d"]].sum().reset_index()
    cross = tot[tot.d < tot.f]["M"].min()
    logger.info("whole-LLM linear time (36 layers): crossover M where dequant+cuBLAS wins = %s", cross)
    for _, r in tot.iterrows():
        logger.info("  M=%5d  fused %.1f ms  dequant %.1f ms", r.M, r.f, r.d)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from edgecompose.analysis.plots import _style, INK2
    _style()
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.plot(tot.M, tot.f, color="#2a78d6", marker="o", label="Triton split-K fused GEMM")
    ax.plot(tot.M, tot.d, color="#eb6834", marker="s", label="Triton dequantize + cuBLAS GEMM")
    ax.axvline(1024, color=INK2, ls="--", lw=1)
    ax.text(1040, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 1, "AutoAWQ\nthreshold", fontsize=7.5, color=INK2)
    ax.set_xlabel("activation rows M (= prefill tokens at batch 1)")
    ax.set_ylabel("sum of INT4 linear-layer time, 36 layers (ms)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.set_title("Figure 10 - AWQ kernel crossover on RTX 4060 Laptop", loc="left")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig10_awq_kernel_crossover.png", dpi=160, bbox_inches="tight")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
