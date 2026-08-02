"""Pareto-frontier computation over measured configurations.

A configuration A dominates B iff A is at least as good on every objective and strictly
better on at least one. Objectives are given as (column, direction) with direction
"max" (quality) or "min" (latency, memory, energy).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Objective = Tuple[str, str]  # (key, "max" | "min")


def _better_or_equal(a: float, b: float, direction: str, tol: float) -> bool:
    return a >= b - tol if direction == "max" else a <= b + tol


def _strictly_better(a: float, b: float, direction: str, tol: float) -> bool:
    return a > b + tol if direction == "max" else a < b - tol


def dominates(a: Dict, b: Dict, objectives: Sequence[Objective], tol: float = 0.0) -> bool:
    """True if `a` Pareto-dominates `b` on `objectives` (values compared within `tol`)."""
    if not all(_better_or_equal(a[k], b[k], d, tol) for k, d in objectives):
        return False
    return any(_strictly_better(a[k], b[k], d, tol) for k, d in objectives)


def pareto_front(rows: Sequence[Dict], objectives: Sequence[Objective], tol: float = 0.0) -> List[int]:
    """Indices of non-dominated rows (rows with missing objective values are excluded)."""
    valid = [i for i, r in enumerate(rows) if all(r.get(k) is not None for k, _ in objectives)]
    front = []
    for i in valid:
        if not any(dominates(rows[j], rows[i], objectives, tol) for j in valid if j != i):
            front.append(i)
    return front
