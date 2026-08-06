import pandas as pd
import pytest

from edgecompose.optimizer.selector import Constraints, select


def _agg():
    rows = []
    for ds, qs in (("textvqa", [0.80, 0.79, 0.74, 0.60]), ("pope", [0.86, 0.86, 0.85, 0.83])):
        for name, q, lat, mem in zip(["C0_sdpa_r100", "C1", "C2", "C3"], qs, [1300, 1250, 1000, 800],
                                     [4000, 3900, 3700, 3600]):
            rows.append({"config_name": name, "dataset": ds, "quality": q, "total_latency_ms_p50": lat,
                         "total_latency_ms_p95": lat * 1.2, "total_latency_ms_mean": lat, "ttft_ms_p50": lat * 0.8,
                         "peak_reserved_mb_max": mem, "energy_j_median": lat / 20})
    return pd.DataFrame(rows)


def test_max_quality_under_latency_budget():
    res = select(_agg(), Constraints(max_latency_ms=1100), "quality")
    assert res["best"]["config_name"] == "C2"


def test_min_latency_under_relative_quality():
    res = select(_agg(), Constraints(min_quality=0.95, relative_quality=True), "latency")
    # C2 keeps 0.74/0.80 = 92.5% on textvqa -> infeasible; C1 feasible
    assert res["best"]["config_name"] == "C1"
    assert "C2" in res["rejected"]


def test_dataset_filter_and_memory_objective():
    res = select(_agg(), Constraints(datasets=["pope"], min_quality=0.84), "memory")
    assert res["best"]["config_name"] == "C2"


def test_infeasible_returns_none():
    res = select(_agg(), Constraints(max_vram_mb=1000), "quality")
    assert res["best"] is None
    assert len(res["rejected"]) == 4


def test_bad_objective():
    with pytest.raises(ValueError):
        select(_agg(), Constraints(), "speed")
