"""Macro F0.5, computed exactly like the leaderboard.

F0.5 is computed per S1 entity and averaged over *all* S1 entities in the
evaluation set. A singleton scores 1.0 if predicted empty, 0.0 otherwise.
"""
import numpy as np
import pandas as pd

BETA2 = 0.25  # beta = 0.5


def entity_f05(n_pred, n_true, tp):
    """Vectorised per-entity F0.5 from counts."""
    n_pred = np.asarray(n_pred, dtype=float)
    n_true = np.asarray(n_true, dtype=float)
    tp = np.asarray(tp, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, tp / n_pred, 0.0)
        r = np.where(n_true > 0, tp / n_true, 0.0)
        f = np.where(tp > 0, (1 + BETA2) * p * r / (BETA2 * p + r), 0.0)
    both_empty = (n_true == 0) & (n_pred == 0)
    return np.where(both_empty, 1.0, f)


def macro_f05(pred_pairs, true_pairs, s1_ids, return_frame=False):
    """pred_pairs / true_pairs: DataFrames with columns [s1, other].

    s1_ids: every S1 id in the evaluation set (defines the average, singletons included).
    """
    s1_ids = pd.Index(pd.unique(pd.Series(list(s1_ids))), name="s1")
    pred = pred_pairs[["s1", "other"]].drop_duplicates()
    true = true_pairs[["s1", "other"]].drop_duplicates()
    pred = pred[pred.s1.isin(s1_ids)]
    true = true[true.s1.isin(s1_ids)]
    n_pred = pred.groupby("s1").size().reindex(s1_ids, fill_value=0)
    n_true = true.groupby("s1").size().reindex(s1_ids, fill_value=0)
    tp = pred.merge(true, on=["s1", "other"]).groupby("s1").size().reindex(s1_ids, fill_value=0)
    f = entity_f05(n_pred.values, n_true.values, tp.values)
    if not return_frame:
        return float(f.mean())
    return float(f.mean()), pd.DataFrame({"n_pred": n_pred.values, "n_true": n_true.values, "tp": tp.values, "f05": f}, index=s1_ids)


def breakdown(frame, s1_country=None):
    """Macro F0.5 split into singletons / non-singletons (and per country if given)."""
    out = {
        "f05": frame.f05.mean(),
        "f05_singletons": frame.f05[frame.n_true == 0].mean(),
        "f05_non_singletons": frame.f05[frame.n_true > 0].mean(),
        "singleton_fp_rate": (frame.n_pred[frame.n_true == 0] > 0).mean(),
    }
    if s1_country is not None:
        c = s1_country.reindex(frame.index)
        for k, v in frame.f05.groupby(c).mean().items():
            out[f"f05_{k}"] = v
    return {k: round(float(v), 5) for k, v in out.items()}


if __name__ == "__main__":
    # README example: predicted 3, true 2, both true ones found -> 0.714
    pred = pd.DataFrame({"s1": ["S1-00001"] * 3, "other": ["S2-00047", "S2-00193", "S3-00812"]})
    true = pd.DataFrame({"s1": ["S1-00001"] * 2, "other": ["S2-00047", "S3-00812"]})
    score = macro_f05(pred, true, ["S1-00001"])
    print("README example:", round(score, 3))
    assert abs(score - 0.714) < 1e-3
    # singleton handling: empty/empty = 1, predicted-on-singleton = 0, missed-all = 0
    s, fr = macro_f05(pd.DataFrame({"s1": ["B"], "other": ["x"]}), pd.DataFrame({"s1": ["C"], "other": ["y"]}), ["A", "B", "C"], True)
    assert list(fr.f05) == [1.0, 0.0, 0.0], fr
    print("singleton checks OK; mean", round(s, 3))
