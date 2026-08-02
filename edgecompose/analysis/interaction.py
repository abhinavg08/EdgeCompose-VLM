"""Composition-interaction analysis of two optimizations A and B.

For a cost metric C (latency, TTFT, memory, ...), with the unoptimized baseline 0:

    R_A  = C_A  / C_0         (A alone)
    R_B  = C_B  / C_0         (B alone)
    R_AB = C_AB / C_0         (A and B composed)
    I(A, B) = R_AB - R_A * R_B

I < 0: composition is *better* than multiplicative independence (synergy);
I ~ 0: effects compose independently (multiplicatively);
I > 0: composition gives *less* benefit than predicted (interference, e.g. the removed
cost was already removed by the other technique, or a new bottleneck dominates).

Uncertainty: a paired bootstrap over samples (the same sample IDs are measured in all
four cells) gives a percentile CI for I on the per-configuration *median* cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class Interaction:
    metric: str
    r_a: float
    r_b: float
    r_ab: float
    predicted_ab: float
    interaction: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    def verdict(self, tol: float = 0.02) -> str:
        lo = self.ci_low if self.ci_low is not None else self.interaction
        hi = self.ci_high if self.ci_high is not None else self.interaction
        if lo > tol:
            return "interference (sub-multiplicative gain)"
        if hi < -tol:
            return "synergy (super-multiplicative gain)"
        return "approximately independent"


def interaction_from_costs(c0: float, ca: float, cb: float, cab: float, metric: str = "") -> Interaction:
    r_a, r_b, r_ab = ca / c0, cb / c0, cab / c0
    return Interaction(metric, r_a, r_b, r_ab, r_a * r_b, r_ab - r_a * r_b)


def paired_bootstrap_interaction(
    x0: Sequence[float],
    xa: Sequence[float],
    xb: Sequence[float],
    xab: Sequence[float],
    metric: str = "",
    n_boot: int = 2000,
    seed: int = 0,
    stat=np.median,
) -> Interaction:
    """Interaction on a robust statistic (median) with a paired-bootstrap 95% CI.

    All four arrays must be aligned by sample (same sample order).
    """
    arrs = [np.asarray(v, dtype=float) for v in (x0, xa, xb, xab)]
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        raise ValueError("paired bootstrap requires aligned arrays of equal length")
    point = interaction_from_costs(*(float(stat(a)) for a in arrs), metric=metric)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        c0, ca, cb, cab = (float(stat(a[idx])) for a in arrs)
        vals[b] = cab / c0 - (ca / c0) * (cb / c0)
    point.ci_low, point.ci_high = (float(v) for v in np.percentile(vals, [2.5, 97.5]))
    return point


def summarize(inter: Interaction) -> Dict:
    return {
        "metric": inter.metric,
        "R_A": inter.r_a,
        "R_B": inter.r_b,
        "R_AB": inter.r_ab,
        "R_A*R_B": inter.predicted_ab,
        "I": inter.interaction,
        "I_ci_low": inter.ci_low,
        "I_ci_high": inter.ci_high,
        "verdict": inter.verdict(),
    }
