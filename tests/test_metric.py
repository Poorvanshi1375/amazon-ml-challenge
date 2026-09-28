"""Macro F0.5 must behave exactly like the leaderboard metric."""
import pandas as pd
import pytest

from metric import entity_f05, macro_f05


def pairs(rows):
    return pd.DataFrame(rows, columns=["s1", "other"])


def test_readme_example_is_0714():
    # predicted 3 records, 2 of them true, both true records found -> P = 2/3, R = 1 -> F0.5 = 0.714
    pred = pairs([("S1-1", "S2-47"), ("S1-1", "S2-193"), ("S1-1", "S3-812")])
    true = pairs([("S1-1", "S2-47"), ("S1-1", "S3-812")])
    assert macro_f05(pred, true, ["S1-1"]) == pytest.approx(0.714, abs=1e-3)


def test_singleton_scores_one_only_when_predicted_empty():
    empty = pairs([])
    assert macro_f05(empty, empty, ["S1-1"]) == 1.0
    assert macro_f05(pairs([("S1-1", "S2-9")]), empty, ["S1-1"]) == 0.0


def test_missing_all_true_matches_scores_zero():
    assert macro_f05(pairs([]), pairs([("S1-1", "S2-1")]), ["S1-1"]) == 0.0


def test_macro_average_weights_every_entity_equally():
    # S1-1: perfect (1.0); S1-2: singleton predicted empty (1.0); S1-3: one wrong match (0.0)
    pred = pairs([("S1-1", "S2-1"), ("S1-1", "S2-2"), ("S1-3", "S2-9")])
    true = pairs([("S1-1", "S2-1"), ("S1-1", "S2-2"), ("S1-3", "S3-5")])
    assert macro_f05(pred, true, ["S1-1", "S1-2", "S1-3"]) == pytest.approx(2 / 3)


def test_false_positive_costs_more_than_a_miss():
    # 3 true matches: one extra wrong match vs one missed match
    fp = entity_f05([4], [3], [3])[0]
    fn = entity_f05([2], [3], [2])[0]
    assert fp == pytest.approx(0.789, abs=1e-3)
    assert fn == pytest.approx(0.909, abs=1e-3)
    assert fp < fn
