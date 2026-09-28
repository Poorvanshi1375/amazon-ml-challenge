"""Decoding: one-owner, look-alike correction and expected-F0.5 with an explicit empty option."""
import numpy as np
import pandas as pd
import pytest

from decode import lookalike_shift, one_owner
from finalize import ef_decode


def test_one_owner_keeps_only_the_best_s1_above_threshold():
    pred = pd.DataFrame({"q": ["S2-1", "S2-1", "S2-2"], "s1": ["A", "B", "A"], "p": [0.9, 0.8, 0.5]})
    out = one_owner(pred, 0.7)
    assert out.values.tolist() == [["A", "S2-1"]]


def test_lookalike_shift_only_lowers_flagged_pairs_by_delta_in_log_odds():
    pred = pd.DataFrame({"q": ["a", "b"], "s1": ["A", "A"], "p": [0.9, 0.9]})
    out = lookalike_shift(pred, [1, 0], 2.5)
    logit = lambda p: np.log(p / (1 - p))
    assert logit(out.p.iloc[0]) == pytest.approx(logit(0.9) - 2.5)
    assert out.p.iloc[1] == pytest.approx(0.9)


def kept(pc, alpha=1.0, m=0.15):
    k = pd.DataFrame({"s1": ["A"] * len(pc), "q": [f"q{i}" for i in range(len(pc))], "pc": pc})
    return sorted(ef_decode(k, alpha, m).other)


def test_confident_records_are_kept_and_weak_ones_dropped():
    assert kept([0.99, 0.97, 0.10]) == ["q0", "q1"]


def test_empty_prediction_wins_when_no_match_is_likely():
    assert kept([0.30]) == []                 # P(no match) = 0.7 beats the best expected F0.5
    assert kept([0.30], alpha=0.1) == ["q0"]  # a weaker empty option keeps it
