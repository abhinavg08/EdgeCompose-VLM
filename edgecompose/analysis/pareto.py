"""Pareto-frontier computation over measured configurations.

A configuration A dominates B iff A is at least as good on every objective and strictly
better on at least one. Objectives are given as (column, direction) with direction
"max" (quality) or "min" (latency, memory, energy).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple, Union

Objective = Tuple[str, str]  # (key, "max" | "min")
Tol = Union[float, Mapping[str, float]]


def _tol(tol: Tol, key: str) -> float:
    return float(tol.get(key, 0.0)) if isinstance(tol, Mapping) else float(tol)


def _better_or_equal(a: float, b: float, direction: str, tol: float) -> bool:
    return a >= b - tol if direction == "max" else a <= b + tol


def _strictly_better(a: float, b: float, direction: str, tol: float) -> bool:
    return a > b + tol if direction == "max" else a < b - tol


def dominates(a: Dict, b: Dict, objectives: Sequence[Objective], tol: Tol = 0.0) -> bool:
    """True if `a` Pareto-dominates `b` on `objectives`.

    `tol` (a scalar or a per-objective mapping, in objective units) makes the relation
    noise-aware: differences within tolerance count as ties, and a strict improvement must
    exceed the tolerance.
    """
    if not all(_better_or_equal(a[k], b[k], d, _tol(tol, k)) for k, d in objectives):
        return False
    return any(_strictly_better(a[k], b[k], d, _tol(tol, k)) for k, d in objectives)


def pareto_front(rows: Sequence[Dict], objectives: Sequence[Objective], tol: Tol = 0.0) -> List[int]:
    """Indices of non-dominated rows (rows with missing objective values are excluded)."""
    valid = [i for i, r in enumerate(rows) if all(r.get(k) is not None for k, _ in objectives)]
    front = []
    for i in valid:
        if not any(dominates(rows[j], rows[i], objectives, tol) for j in valid if j != i):
            front.append(i)
    return front
