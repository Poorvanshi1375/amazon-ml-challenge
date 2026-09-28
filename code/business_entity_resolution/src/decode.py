"""Stage 6: turn pair probabilities into per-S1 match lists.

one_owner(pred, t)
    Each S2/S3 record keeps only its most probable S1, if p >= t.

expected_f05(pred, power, floor)
    After the one-owner step (records assigned to their most probable S1 with
    p >= floor), each S1 picks the k most probable of its assigned records that
    maximises a plug-in estimate of its expected F0.5:
        k = 0 : P(singleton) = prod(1 - p_i)
        k > 0 : 1.25 * sum_{i<=k} p_i / (0.25 * sum_all p_i + k)
    Probabilities are first sharpened/softened as p**power (a one-parameter
    calibration tuned on the holdout together with floor).
"""
import numpy as np
import pandas as pd


def one_owner(pred, t):
    """pred: DataFrame q, s1, p -> matched pairs DataFrame (s1, other)."""
    best = pred.loc[pred.groupby("q").p.idxmax()]
    best = best[best.p >= t]
    return best.rename(columns={"q": "other"})[["s1", "other"]]


def expected_f05(pred, power=1.0, floor=0.05):
    best = pred.loc[pred.groupby("q").p.idxmax()]
    best = best[best.p >= floor][["s1", "q", "p"]].copy()
    best["p"] = np.clip(best.p.values, 1e-6, 1 - 1e-6) ** power
    best = best.sort_values(["s1", "p"], ascending=[True, False])
    s1 = best.s1.values
    p = best.p.values
    starts = np.r_[0, np.flatnonzero(s1[1:] != s1[:-1]) + 1]
    ends = np.r_[starts[1:], len(s1)]
    keep = np.zeros(len(p), dtype=bool)
    for a, b in zip(starts, ends):
        pi = p[a:b]
        total = pi.sum()
        cum = np.cumsum(pi)
        k = np.arange(1, len(pi) + 1)
        f = 1.25 * cum / (0.25 * total + k)
        p_single = np.prod(1 - pi)
        j = int(np.argmax(f))
        if f[j] > p_single:
            keep[a:a + j + 1] = True
    out = best[keep]
    return out.rename(columns={"q": "other"})[["s1", "other"]]


def lookalike_shift(pred, flag, delta):
    """Lower the log-odds of flagged pairs (known, different house numbers) by delta."""
    if not delta:
        return pred
    p = np.clip(pred.p.values, 1e-7, 1 - 1e-7)
    lg = np.log(p) - np.log(1 - p) - delta * np.asarray(flag, dtype=float)
    return pred.assign(p=1 / (1 + np.exp(-lg)))


def decode(pred, cfg, flag=None):
    """Apply a saved decode config (decode.json); optional look-alike shift on flagged pairs."""
    if flag is not None and cfg.get("lookalike_shift"):
        pred = lookalike_shift(pred, flag, cfg["lookalike_shift"])
    if cfg.get("mode") == "expected_f05":
        return expected_f05(pred, cfg["power"], cfg["floor"])
    return one_owner(pred, cfg["threshold"])
