import numpy as np
import pytest

from edgecompose.analysis.interaction import interaction_from_costs, paired_bootstrap_interaction


def test_multiplicative_independence_gives_zero():
    it = interaction_from_costs(100, 80, 50, 40)  # 0.8 * 0.5 = 0.4
    assert it.interaction == pytest.approx(0.0)
    assert it.verdict() == "approximately independent"


def test_interference_positive():
    it = interaction_from_costs(100, 80, 50, 60)
    assert it.interaction == pytest.approx(0.2)
    assert "interference" in it.verdict()


def test_synergy_negative():
    it = interaction_from_costs(100, 80, 50, 20)
    assert it.interaction == pytest.approx(-0.2)
    assert "synergy" in it.verdict()


def test_bootstrap_ci_brackets_point_and_detects_interference():
    rng = np.random.default_rng(0)
    x0 = rng.normal(1000, 20, 200)
    xa, xb = 0.8 * x0, 0.5 * x0
    xab = 0.6 * x0  # expected 0.4 -> I = +0.2
    it = paired_bootstrap_interaction(x0, xa, xb, xab)
    assert it.ci_low <= it.interaction <= it.ci_high
    assert it.ci_low > 0.1


def test_bootstrap_requires_alignment():
    with pytest.raises(ValueError):
        paired_bootstrap_interaction([1, 2], [1], [1, 2], [1, 2])
