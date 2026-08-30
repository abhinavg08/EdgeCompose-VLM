import math

import pytest

from edgeinspect.inference.classify import p_anomaly_from_logprobs, parse_label
from edgeinspect.inference.explain import parse_json_object
from edgeinspect.metrics.anomaly import aufc, auroc, binary_metrics, evaluate, f1_max
from edgeinspect.prompts.inspection import classification_content


def test_parse_label():
    assert parse_label("ANOMALOUS") == 1
    assert parse_label("Anomalous.") == 1
    assert parse_label("NORMAL") == 0
    assert parse_label("normal") == 0
    assert parse_label("ABNORMAL") == 1
    assert parse_label("NOT NORMAL") == 1
    assert parse_label("I cannot tell") is None


def test_p_anomaly():
    assert p_anomaly_from_logprobs({"NORMAL": math.log(0.2), "ANOMALOUS": math.log(0.6)}) == pytest.approx(0.75)
    assert p_anomaly_from_logprobs({"NORMAL": -1.0}) is None
    assert p_anomaly_from_logprobs({"NORMAL": 0.0, "ANOMALOUS": -2000.0}) == pytest.approx(0.0)


def test_binary_metrics():
    m = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5 and m["accuracy"] == 0.5


def test_auroc_perfect_random_and_ties():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5
    assert math.isnan(auroc([1, 1], [0.1, 0.2]))


def test_f1_max_finds_best_threshold():
    r = f1_max([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert r["f1_max"] == 1.0 and r["f1_max_threshold"] == 0.7


def test_per_type_metrics_use_normals_plus_type():
    y = [0, 0, 1, 1]
    p = [0, 0, 1, 0]
    types = ["none", "none", "structural", "logical"]
    out = evaluate(y, p, [0.1, 0.2, 0.9, 0.3], types)
    assert out["f1_structural"] == 1.0
    assert out["f1_logical"] == 0.0
    assert out["auroc_structural"] == 1.0
    assert out["auroc_logical"] == 1.0  # 0.3 still ranks above both normals


def test_aufc():
    assert aufc([1, 2, 4, 8], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert aufc([1, 8], [0.0, 1.0]) == pytest.approx(0.5)


def test_prompt_has_k_plus_one_images():
    c = classification_content("pushpins", 4)
    assert sum(1 for x in c if x["type"] == "image") == 5
    assert "NORMAL or ANOMALOUS" in c[-1]["text"]


def test_parse_json_object():
    assert parse_json_object('```json\n{"anomaly_type": "logical", "issue": "missing pin"}\n```')["issue"] == "missing pin"
    assert parse_json_object("no json here") is None
