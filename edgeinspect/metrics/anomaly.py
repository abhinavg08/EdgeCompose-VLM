"""Image-level anomaly-detection metrics (positive class = anomalous).

* precision / recall / F1 / accuracy of the discrete prediction;
* AUROC of the continuous anomaly score (Mann-Whitney U with tie correction);
* F1-max: best F1 over all score thresholds (as in VAND few-shot evaluation);
* per anomaly type (MVTec LOCO convention): metrics on {normal} U {that type}, so
  F1_structural uses normal + structural queries only, F1_logical normal + logical only;
* AUFC: normalised area under the F1-max-vs-shots curve over the evaluated k values,
  using log2(k) spacing (a VAND-style few-shot aggregate; reported as such, not as an
  official challenge number).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    n = len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": (tp + tn) / n if n else float("nan"),
            "specificity": spec, "balanced_accuracy": (rec + spec) / 2, "pred_anomalous_ratio": (tp + fp) / n if n else float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n}


def auroc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve via average ranks (ties get half credit)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auprc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Average precision (area under the precision-recall curve, step interpolation)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = int((y == 1).sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted, s_sorted = y[order], s[order]
    ap, tp, prev_recall = 0.0, 0, 0.0
    i = 0
    while i < len(s_sorted):  # process tied scores as one threshold
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        tp += int(y_sorted[i:j + 1].sum())
        precision = tp / (j + 1)
        recall = tp / pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
        i = j + 1
    return float(ap)


def f1_max(y_true: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    """Maximum F1 over thresholds 'score >= t' for all distinct t."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    best, best_t = 0.0, float("nan")
    for t in np.unique(s):
        m = binary_metrics(y, (s >= t).astype(int))
        if m["f1"] > best:
            best, best_t = m["f1"], float(t)
    return {"f1_max": best, "f1_max_threshold": best_t}


def evaluate(y_true: Sequence[int], y_pred: Sequence[int], scores: Optional[Sequence[float]] = None,
             types: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """All metrics, overall and per anomaly type ('structural' / 'logical'; normals have 'none')."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    out = binary_metrics(y, p)
    have_scores = scores is not None and not any(v is None or (isinstance(v, float) and np.isnan(v)) for v in scores)
    if have_scores:
        s = np.asarray(scores, dtype=float)
        out["auroc"] = auroc(y, s)
        out["auprc"] = auprc(y, s)
        out.update(f1_max(y, s))
    if types is not None:
        t = np.asarray(types)
        for kind in ("structural", "logical"):
            m = (t == "none") | (t == kind)
            if (t == kind).sum() == 0:
                continue
            bm = binary_metrics(y[m], p[m])
            out[f"f1_{kind}"] = bm["f1"]
            out[f"recall_{kind}"] = bm["recall"]
            out[f"n_{kind}"] = int((t == kind).sum())
            if have_scores:
                out[f"auroc_{kind}"] = auroc(y[m], s[m])
                out[f"auprc_{kind}"] = auprc(y[m], s[m])
                out[f"f1_max_{kind}"] = f1_max(y[m], s[m])["f1_max"]
    return out


def aufc(shots: Sequence[int], f1max: Sequence[float]) -> float:
    """Normalised area under F1-max vs log2(shots) (trapezoid); in [0, 1]."""
    k = np.log2(np.asarray(shots, dtype=float))
    f = np.asarray(f1max, dtype=float)
    order = np.argsort(k)
    k, f = k[order], f[order]
    if len(k) < 2 or k[-1] == k[0]:
        return float(f.mean()) if len(f) else float("nan")
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 renamed trapz
    return float(trapezoid(f, k) / (k[-1] - k[0]))
