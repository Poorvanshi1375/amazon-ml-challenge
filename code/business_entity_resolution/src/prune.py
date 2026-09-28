"""Stage 2b (Tier 0): recall-safe candidate pruning with a simple margin rule.

A (query, S1) pair from blocking is kept if, in either index, it is the query's
top-ranked S1, or its score is within a margin of the query's top score:
    keep = rank_n == 1 or rank_b == 1
           or score_n >= top_n - m_n or score_b >= top_b - m_b
Queries with an empty address get their own margin for the name index.

Margins are chosen on the train holdout so that recall loss versus raw blocking
is at most MAX_LOSS; they are chosen on US and checked on India and vice versa,
and the stricter (larger) margin of the two is used.

Learned mode (--learned): a small LightGBM on blocking-only signals (scores, ranks,
margins, top scores, candidates per query, empty-address flag) scores every pair;
the cut-off is the highest one whose holdout recall loss vs raw blocking is
<= MAX_LOSS, checked US<->India like the margin rule. Its score is kept as a feature.

Usage: python prune.py --tune [--learned]   (train split: choose rule, save prune.json)
       python prune.py --split test         (apply the saved rule)
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import ART_DIR, N_JOBS, SEED
from tracking import log_run

MAX_LOSS = 0.001
GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 1.0]


def add_context(b):
    """Per-query top scores and margins from the raw blocking table.

    Grouping uses integer codes of the query id (computed once): same result as
    grouping by the string id, with far less memory on ~200M rows.
    """
    qi = pd.factorize(b.q.values)[0]
    for ix in [x for x in ("n", "b", "a") if f"score_{x}" in b]:
        s = b[f"score_{ix}"].fillna(0).astype(np.float32)
        b[f"score_{ix}"] = s
        top = s.groupby(qi).transform("max")
        # second best per query: max of scores strictly below the top (0 if none)
        second = s.where(s < top).groupby(qi).transform("max").fillna(0)
        b[f"top_{ix}"] = top.astype(np.float32)
        b[f"margin_{ix}"] = np.where(s >= top, s - second, s - top).astype(np.float32)
        del top, second
    b["n_raw_q"] = np.bincount(qi)[qi].astype(np.int16)
    return b


def keep_mask(b, m_n, m_b, m_n_empty, q_empty):
    mn = np.where(q_empty, m_n_empty, m_n)
    return ((b.rank_n == 1) | (b.rank_b == 1) | (b.score_n >= b.top_n - mn) | (b.score_b >= b.top_b - m_b)).values


BASE_PRUNE_FEATS = ["score_n", "rank_n", "score_b", "rank_b", "margin_n", "margin_b", "top_n", "top_b", "n_raw_q", "q_empty"]


def prune_feats(b):
    """Blocking-only pruning features; the address index columns are used when present."""
    extra = ["score_a", "rank_a", "margin_a", "top_a"] if "score_a" in b else []
    return BASE_PRUNE_FEATS + extra
FIT_ROWS = 20_000_000
PRUNE_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                    num_threads=N_JOBS, seed=SEED, verbose=-1)


def load(split):
    b = pd.read_parquet(ART_DIR / f"blocks_{split}.parquet")
    b = add_context(b)
    q_meta = pd.concat([pd.read_parquet(ART_DIR / f"norm_{split}_s{s}.parquet", columns=["entity_id", "country", "addr_clean"])
                        for s in (2, 3)]).set_index("entity_id")
    m = q_meta.reindex(b.q.values)
    b["q_empty"] = (m.addr_clean == "").values.astype(np.int8)
    return b, m.country.values, (m.addr_clean == "").values


def tune():
    t0 = time.time()
    b, country, q_empty = load("train")
    roles = pd.read_parquet(ART_DIR / "roles.parquet").set_index("s1").role
    tp = pd.read_parquet(ART_DIR / "truth_pairs.parquet")
    tp = tp[tp.s1.map(roles).eq("holdout")]
    key = pd.MultiIndex.from_arrays([b.q, b.s1])
    is_true = key.isin(pd.MultiIndex.from_arrays([tp.other, tp.s1]))
    base_found = is_true.sum()
    hold = b.s1.map(roles).eq("holdout").values

    def recall_loss(mask, sel):
        return 1 - (is_true & mask & sel).sum() / max((is_true & sel).sum(), 1)

    chosen = {}
    for fit_c, check_c in (("US", "India"), ("India", "US")):
        sel_fit = country == fit_c
        best = None
        for m in GRID:  # smallest common margin meeting the loss cap on the fitting country
            if recall_loss(keep_mask(b, m, m, m, q_empty), sel_fit) <= MAX_LOSS:
                best = m
                break
        best = GRID[-1] if best is None else best
        me = best
        for m in GRID:  # separate margin for empty-address queries (name index)
            if m >= best and recall_loss(keep_mask(b, best, best, m, q_empty), sel_fit & q_empty) <= MAX_LOSS:
                me = m
                break
        loss_check = recall_loss(keep_mask(b, best, best, me, q_empty), country == check_c)
        chosen[fit_c] = {"m": best, "m_empty": me, f"loss_on_{check_c}": round(float(loss_check), 5)}
    m = max(v["m"] for v in chosen.values())
    me = max(v["m_empty"] for v in chosen.values())
    mask = keep_mask(b, m, m, me, q_empty)
    params = {"m_n": m, "m_b": m, "m_n_empty": me}
    (ART_DIR / "prune.json").write_text(json.dumps(params))
    hold_ids = roles[roles == "holdout"].index
    cand = pd.Series(1, index=b.s1[mask & hold]).groupby(level=0).size().reindex(hold_ids, fill_value=0)
    raw = pd.Series(1, index=b.s1[hold]).groupby(level=0).size().reindex(hold_ids, fill_value=0)
    metrics = {**params, "per_country": chosen,
               "recall_raw": round(float(base_found / max(len(tp), 1)), 5),
               "recall_pruned": round(float((is_true & mask).sum() / max(len(tp), 1)), 5),
               "cand_per_s1_raw": round(float(raw.mean()), 2), "cand_per_s1_pruned": round(float(cand.mean()), 2),
               "pairs_kept_share": round(float(mask.mean()), 4)}
    log_run("prune_tune", {"max_loss": MAX_LOSS, "grid": GRID}, metrics, runtime_s=time.time() - t0, tier="0")
    b[mask].to_parquet(ART_DIR / "cands_train.parquet", index=False)


def tune_learned():
    t0 = time.time()
    b, country, _ = load("train")
    roles = pd.read_parquet(ART_DIR / "roles.parquet").set_index("s1").role
    tp = pd.read_parquet(ART_DIR / "truth_pairs.parquet")
    y = pd.MultiIndex.from_arrays([b.q, b.s1]).isin(pd.MultiIndex.from_arrays([tp.other, tp.s1]))
    r = b.s1.map(roles).values
    hold, fit = r == "holdout", r != "holdout"
    feats = prune_feats(b)
    fit_idx = np.flatnonzero(fit)
    if len(fit_idx) > FIT_ROWS:
        fit_idx = np.sort(np.random.default_rng(SEED).choice(fit_idx, FIT_ROWS, replace=False))
    model = lgb.train(PRUNE_PARAMS, lgb.Dataset(b.iloc[fit_idx][feats], y[fit_idx]), 300)
    model.save_model(str(ART_DIR / "prune_model.txt"))
    b["p_prune"] = model.predict(b[feats], num_threads=N_JOBS).astype(np.float32)
    grid = np.unique(np.quantile(b.p_prune[hold], np.linspace(0, 0.99, 400)))

    def loss(t, sel):
        return 1 - (y & hold & sel & (b.p_prune.values >= t)).sum() / max((y & hold & sel).sum(), 1)

    chosen = {}
    for fit_c, check_c in (("US", "India"), ("India", "US")):
        sel = country == fit_c
        ok = [t for t in grid if loss(t, sel) <= MAX_LOSS]
        t = max(ok) if ok else 0.0
        chosen[fit_c] = {"t": float(t), f"loss_on_{check_c}": round(float(loss(t, country == check_c)), 5)}
    t = min(v["t"] for v in chosen.values())
    mask = b.p_prune.values >= t
    (ART_DIR / "prune.json").write_text(json.dumps({"mode": "learned", "t": float(t)}))
    hold_ids = roles[roles == "holdout"].index
    cand = pd.Series(1, index=b.s1[mask & hold]).groupby(level=0).size().reindex(hold_ids, fill_value=0)
    raw = pd.Series(1, index=b.s1[hold]).groupby(level=0).size().reindex(hold_ids, fill_value=0)
    n_true_h = max((y & hold).sum(), 1)
    tp_h = tp[tp.s1.map(roles).eq("holdout")]
    metrics = {"t": float(t), "per_country": chosen,
               "recall_raw": round(float((y & hold).sum() / max(len(tp_h), 1)), 5),
               "recall_pruned": round(float((y & hold & mask).sum() / max(len(tp_h), 1)), 5),
               "loss_vs_raw": round(float(1 - (y & hold & mask).sum() / n_true_h), 5),
               "cand_per_s1_raw": round(float(raw.mean()), 2), "cand_per_s1_pruned": round(float(cand.mean()), 2),
               "pairs_kept_share": round(float(mask.mean()), 4)}
    log_run("prune_tune_learned", {"max_loss": MAX_LOSS, "features": feats, **PRUNE_PARAMS}, metrics, runtime_s=time.time() - t0, tier="1")
    kept = b[mask]
    # keep every query that touches a train/holdout S1 (all of its rows, so records owned
    # by "rest" S1s still compete for holdout S1s exactly as in test)
    touch = kept.q[kept.s1.map(roles).isin(["train", "holdout"]).values].unique()
    kept = kept[kept.q.isin(touch)]
    print(f"cands_train: {len(kept):,} pairs, {kept.q.nunique():,} queries", flush=True)
    kept.to_parquet(ART_DIR / "cands_train.parquet", index=False)


def apply(split):
    params = json.loads((ART_DIR / "prune.json").read_text())
    b, _, q_empty = load(split)
    if params.get("mode") == "learned":
        model = lgb.Booster(model_file=str(ART_DIR / "prune_model.txt"))
        b["p_prune"] = model.predict(b[model.feature_name()], num_threads=N_JOBS).astype(np.float32)
        mask = b.p_prune.values >= params["t"]
    else:
        mask = keep_mask(b, params["m_n"], params["m_b"], params["m_n_empty"], q_empty)
    b[mask].to_parquet(ART_DIR / f"cands_{split}.parquet", index=False)
    print(f"{split}: kept {mask.sum():,} of {len(b):,} pairs ({mask.mean():.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--learned", action="store_true")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    if a.tune:
        tune_learned() if a.learned else tune()
    else:
        apply(a.split)
