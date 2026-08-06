from edgecompose.analysis.pareto import dominates, pareto_front

OBJ = [("quality", "max"), ("latency", "min"), ("mem", "min")]


def test_dominates_requires_strict_improvement():
    a = {"quality": 0.8, "latency": 100, "mem": 10}
    assert not dominates(a, dict(a), OBJ)  # equal is not domination
    b = {"quality": 0.8, "latency": 120, "mem": 10}
    assert dominates(a, b, OBJ)
    assert not dominates(b, a, OBJ)


def test_tradeoff_points_are_all_efficient():
    rows = [
        {"quality": 0.9, "latency": 300, "mem": 10},
        {"quality": 0.8, "latency": 200, "mem": 10},
        {"quality": 0.7, "latency": 100, "mem": 10},
    ]
    assert pareto_front(rows, OBJ) == [0, 1, 2]


def test_dominated_point_is_removed():
    rows = [
        {"quality": 0.9, "latency": 100, "mem": 10},  # dominates everything below
        {"quality": 0.8, "latency": 200, "mem": 10},
        {"quality": 0.9, "latency": 100, "mem": 12},
    ]
    assert pareto_front(rows, OBJ) == [0]


def test_missing_values_excluded():
    rows = [{"quality": 0.9, "latency": None, "mem": 10}, {"quality": 0.5, "latency": 50, "mem": 10}]
    assert pareto_front(rows, OBJ) == [1]


def test_tolerance_treats_near_ties_as_equal():
    rows = [{"quality": 0.800, "latency": 100, "mem": 10}, {"quality": 0.801, "latency": 100, "mem": 10}]
    assert pareto_front(rows, OBJ) == [1]
    assert pareto_front(rows, OBJ, tol=0.005) == [0, 1]
