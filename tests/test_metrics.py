import math

import pytest

from edgecompose.evaluation.metrics import (
    normalize_vqa_answer, parse_yes_no, pope_metrics, score_sample, vqa_accuracy,
)


def test_normalization_rules():
    assert normalize_vqa_answer("The Mystery Machine.") == "mystery machine"
    assert normalize_vqa_answer("two") == "2"
    assert normalize_vqa_answer("dont") == "don't"
    assert normalize_vqa_answer("1,000") == "1000"
    assert normalize_vqa_answer("3.5") == "3.5"  # decimal point kept


def test_vqa_accuracy_leave_one_out():
    gts = ["a"] * 3 + ["b"] * 7
    # For "a": leaving out an "a" -> 2 matches -> 2/3; leaving out a "b" -> 3 matches -> 1
    expected = (3 * (2 / 3) + 7 * 1.0) / 10
    assert vqa_accuracy("a", gts) == pytest.approx(expected)
    assert vqa_accuracy("b", gts) == pytest.approx(1.0)
    assert vqa_accuracy("c", gts) == 0.0
    assert vqa_accuracy("A.", gts) == pytest.approx(expected)


def test_vqa_single_match_is_partial_credit():
    gts = ["x"] + ["y"] * 9
    # 1 human said x: 9 subsets contain it -> 1/3 each, 1 subset without -> 0
    assert vqa_accuracy("x", gts) == pytest.approx(9 / 10 / 3)


def test_parse_yes_no():
    assert parse_yes_no("Yes") == "yes"
    assert parse_yes_no("No") == "no"
    assert parse_yes_no("No, there is not.") == "no"
    assert parse_yes_no("There is not a dog in the image.") == "no"
    assert parse_yes_no("Yes. No other animals.") == "yes"  # only first sentence counts


def test_pope_metrics():
    preds = ["yes", "yes", "no", "no", "yes"]
    labels = ["yes", "no", "no", "yes", "yes"]
    m = pope_metrics(preds, labels)
    # tp=2 fp=1 fn=1 tn=1
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["yes_ratio"] == pytest.approx(3 / 5)


def test_pope_empty():
    assert math.isnan(pope_metrics([], [])["accuracy"])


def test_score_sample_dispatch():
    assert score_sample("pope", "No.", ["no"]) == 1.0
    assert score_sample("textvqa", "stop", ["stop"] * 10) == 1.0
    with pytest.raises(ValueError):
        score_sample("mmbench", "a", ["a"])
