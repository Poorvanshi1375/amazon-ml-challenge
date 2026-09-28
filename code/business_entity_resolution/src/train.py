"""Stage 4 + 6 (Tier 0): LightGBM matcher, one-owner decoding, threshold tuning.

Train rows  : candidate pairs whose S1 is not in the holdout
Valid rows  : candidate pairs whose S1 is in the holdout
Decoding    : (a) one-owner + threshold, (b) one-owner + per-S1 expected-F0.5
              subset choice; parameters of both tuned for macro F0.5 on the
              holdout and the better mode is saved to decode.json
Outputs     : model.txt, decode.json, holdout_pred.parquet, errors/<run_id>.md
Usage       : python train.py
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import ART_DIR, N_JOBS, SEED
from decode import decode, expected_f05, one_owner  # noqa: F401  (one_owner re-exported)
from metric import breakdown, macro_f05
from tracking import log_run

PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=N_JOBS, seed=SEED, verbose=-1)
ROUNDS, EARLY = 3000, 100
TRAIN_ROLES = os.environ.get("ER_TRAIN_ROLES", "train").split(",")
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.025), 3)
POWERS = [0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
FLOORS = [0.02, 0.05, 0.1, 0.2, 0.3]


def feature_cols(f):
    return [c for c in f.columns if c not in ("q", "s1", "label")]


def main():
    t0 = time.time()
    f = pd.read_parquet(ART_DIR / "feats_train.parquet")
    roles = pd.read_parquet(ART_DIR / "roles.parquet").set_index("s1").role
    tp = pd.read_parquet(ART_DIR / "truth_pairs.parquet")
    f["label"] = pd.MultiIndex.from_arrays([f.q, f.s1]).isin(pd.MultiIndex.from_arrays([tp.other, tp.s1])).astype(np.int8)
    role = f.s1.map(roles).values
    is_hold = role == "holdout"
    # train only on S1 entities whose candidate groups are complete (role "train"): rows of the
    # "rest" entities have partial groups, which distorts the group/count features (see WORKLOG)
    is_fit = np.isin(role, TRAIN_ROLES)
    cols = feature_cols(f)
    dtr = lgb.Dataset(f.loc[is_fit, cols], f.label[is_fit], free_raw_data=True)
    dva = lgb.Dataset(f.loc[is_hold, cols], f.label[is_hold], reference=dtr)
    model = lgb.train(PARAMS, dtr, ROUNDS, valid_sets=[dva], valid_names=["holdout"],
                      callbacks=[lgb.early_stopping(EARLY, verbose=False), lgb.log_evaluation(100)])
    model.save_model(str(ART_DIR / "model.txt"))

    # decode on all candidate rows of queries that touch a holdout S1 (competition with non-holdout S1s kept)
    hq = f.q[is_hold].unique()
    rows = f[f.q.isin(hq)]
    pred = pd.DataFrame({"q": rows.q.values, "s1": rows.s1.values, "p": model.predict(rows[cols], num_threads=N_JOBS)})
    hold_ids = roles[roles == "holdout"].index
    tp_hold = tp[tp.s1.isin(hold_ids)]
    scores = {t: macro_f05(one_owner(pred, t), tp_hold, hold_ids) for t in THRESHOLDS}
    t_best = max(scores, key=scores.get)
    cfg = {"mode": "one_owner", "threshold": float(t_best)}
    best_score = scores[t_best]
    ef_scores = {}
    for pw in POWERS:
        for fl in FLOORS:
            ef_scores[(pw, fl)] = macro_f05(expected_f05(pred, pw, fl), tp_hold, hold_ids)
    (pw_b, fl_b), ef_best = max(ef_scores.items(), key=lambda kv: kv[1])
    print(f"decode: one-owner {best_score:.5f} (t={t_best}) vs expected-F0.5 {ef_best:.5f} (power={pw_b}, floor={fl_b})", flush=True)
    if ef_best > best_score:
        cfg = {"mode": "expected_f05", "power": pw_b, "floor": fl_b, "threshold": float(t_best)}
    best_pairs = decode(pred, cfg)
    score, frame = macro_f05(best_pairs, tp_hold, hold_ids, return_frame=True)
    s1_country = pd.read_parquet(ART_DIR / "norm_train_s1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
    bd = breakdown(frame, s1_country)
    (ART_DIR / "decode.json").write_text(json.dumps(cfg))
    pred.to_parquet(ART_DIR / "holdout_pred.parquet", index=False)

    imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
    metrics = {**bd, "decode": cfg, "f05_one_owner": round(best_score, 5), "f05_expected": round(ef_best, 5),
               "best_iter": model.best_iteration,
               "valid_auc_rows": int(is_hold.sum()), "train_rows": int(is_fit.sum()), "train_roles": TRAIN_ROLES,
               "f05_by_threshold": {str(k): round(v, 5) for k, v in scores.items()},
               "top_features": imp.head(12).round(0).to_dict()}
    run_id = log_run("train", {**PARAMS, "rounds": ROUNDS}, metrics, runtime_s=time.time() - t0, tier="1")
    error_report(run_id, best_pairs, tp_hold, frame, pred)
    print(f"HOLDOUT macro F0.5 = {score:.5f} with {cfg}", flush=True)


def error_report(run_id, pairs, tp_hold, frame, pred):
    """Counts + examples per error bucket, written to errors/<run_id>.md."""
    out = ART_DIR / "errors"
    out.mkdir(exist_ok=True)
    txt = ["entity_id", "business_name", "business_address"]
    names = pd.concat([pd.read_parquet(ART_DIR / f"norm_train_s{s}.parquet", columns=txt) for s in (1, 2, 3)]).set_index("entity_id")
    key = lambda d: set(zip(d.s1, d.other))
    pairs = pairs[pairs.s1.isin(frame.index)]
    P, T = key(pairs), key(tp_hold)
    cand = set(zip(pred.s1, pred.q))
    fp = pd.DataFrame(sorted(P - T), columns=["s1", "other"])
    fn = pd.DataFrame(sorted(T - P), columns=["s1", "other"])
    singles = set(frame.index[frame.n_true == 0])
    fp["bucket"] = np.where(fp.s1.isin(singles), "FP on singleton", "FP on non-singleton")
    fn["bucket"] = np.where([(s, o) in cand for s, o in zip(fn.s1, fn.other)], "missed by model", "missed by blocking/pruning")
    lines = [f"# Error report {run_id}\n"]
    for b, g in pd.concat([fp, fn]).groupby("bucket"):
        lines.append(f"\n## {b}: {len(g):,}\n")
        for _, r in g.sample(min(25, len(g)), random_state=SEED).iterrows():
            a, o = names.reindex([r.s1]).iloc[0], names.reindex([r.other]).iloc[0]
            lines.append(f"- S1 `{a.business_name}` | {a.business_address}\n  - {r.other[:2]} `{o.business_name}` | {o.business_address}")
    (out / f"{run_id}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"error report: errors/{run_id}.md  (FP {len(fp):,}, FN {len(fn):,})", flush=True)


if __name__ == "__main__":
    main()
