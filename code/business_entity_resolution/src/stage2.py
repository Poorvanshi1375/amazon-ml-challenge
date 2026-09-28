"""Tier 5: stage-2 sibling model.

Stage 1: Tier 3/4 features. Out-of-fold probabilities for training S1s (2 folds by S1 hash); holdout and
test get probabilities from a model trained on all training S1s.
Stage-2 features for each candidate (q, s1):
  acc_n         number of the S1's other candidates with p1 >= ACC ("accepted siblings")
  acc_nm_max    best name similarity (token set) of q to an accepted sibling
  acc_ad_max    best address similarity of q to an accepted sibling (-1 if q or sibling address empty)
  acc_both      1 if some accepted sibling has name >= 0.9 and (address >= 0.9 or either empty)
  p1, p1_rank_q, p1_gap_q     q's own probability, its rank among q's candidates, gap to q's best other S1
  comp_acc_nm   acc_nm_max of q's best competing S1 (the other S1 with the highest p1 for this q)
Stage-2 LightGBM on stage-1 features + these.
Outputs (in ART_DIR): stage1_full.txt, stage2.txt, holdout_pred_stage2.parquet, test_pred_stage2.parquet.
usage: python stage2.py
"""
import json
import os
import sys
import time
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from config import ART_DIR, N_JOBS
from decode import one_owner
from metric import breakdown, macro_f05

SUF = ""
A = str(ART_DIR) + "/"
ACC = 0.7
R1 = int(os.environ.get("S2_R1", 1200))      # stage-1 rounds (small for smoke tests)
R2 = int(os.environ.get("S2_R2", 3000))
P1 = dict(objective="binary", learning_rate=0.12, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=N_JOBS, seed=42, verbose=-1)
P2 = dict(P1, learning_rate=0.08)
t0 = time.time()


def log(msg):
    print(f"[{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)


roles = pd.read_parquet(A + "roles.parquet").set_index("s1").role
tp = pd.read_parquet(A + "truth_pairs.parquet")
s1c = pd.read_parquet(A + "norm_train_s1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
f = pd.read_parquet(A + f"feats_train{SUF}.parquet")
cols1 = [c for c in f.columns if c not in ("q", "s1")]
f["y"] = pd.MultiIndex.from_arrays([f.q, f.s1]).isin(pd.MultiIndex.from_arrays([tp.other, tp.s1])).astype(np.int8)
role = f.s1.map(roles).values
fit, hold = role == "train", role == "holdout"
fold = np.array([zlib.crc32(s.encode()) % 2 for s in f.s1.values], dtype=np.int8)
log(f"loaded {len(f):,} rows; train rows {fit.sum():,}")

# ---- stage 1: OOF for train rows, full model for everything else
p1 = np.full(len(f), np.nan, np.float32)
for k in (0, 1):
    tr = fit & (fold != k)
    m = lgb.train(P1, lgb.Dataset(f.loc[tr, cols1], f.y.values[tr]), R1)
    sel = fit & (fold == k)
    p1[sel] = m.predict(f.loc[sel, cols1], num_threads=N_JOBS)
    log(f"stage-1 fold {k} done")
full = lgb.train(P1, lgb.Dataset(f.loc[fit, cols1], f.y.values[fit]), R1)
full.save_model(A + f"stage1_full{SUF}.txt")
rest = ~fit
p1[rest] = full.predict(f.loc[rest, cols1], num_threads=N_JOBS)
f["p1"] = p1
log("stage-1 done")

NAMES = pd.concat([pd.read_parquet(A + f"norm_{sp}_s{s}.parquet", columns=["entity_id", "name_core", "addr_clean"])
                   for sp in ("train", "test") for s in (2, 3)]).drop_duplicates("entity_id").set_index("entity_id")


def stage2_features(d):
    d = d[["q", "s1", "p1"]].copy()
    d["i"] = np.arange(len(d))
    acc = d[d.p1 >= ACC][["s1", "q", "i"]].rename(columns={"q": "sib", "i": "j"})
    x = d[["i", "s1", "q"]].merge(acc, on="s1")
    x = x[x.q != x.sib]
    qn, sn = NAMES.name_core.reindex(x.q.values).values, NAMES.name_core.reindex(x.sib.values).values
    qa, sa = NAMES.addr_clean.reindex(x.q.values).values, NAMES.addr_clean.reindex(x.sib.values).values
    nm = cpdist(qn, sn, scorer=fuzz.token_set_ratio, workers=N_JOBS) / 100
    ad = cpdist(qa, sa, scorer=fuzz.token_set_ratio, workers=N_JOBS) / 100
    empty = (qa == "") | (sa == "")
    ad = np.where(empty, -1, ad)
    both = ((nm >= 0.9) & ((ad >= 0.9) | empty)).astype(np.int8)
    g = pd.DataFrame({"i": x.i.values, "nm": nm, "ad": ad, "both": both}).groupby("i").agg(
        acc_n=("nm", "size"), acc_nm_max=("nm", "max"), acc_ad_max=("ad", "max"), acc_both=("both", "max"))
    out = pd.DataFrame(index=np.arange(len(d)))
    out = out.join(g).fillna({"acc_n": 0, "acc_nm_max": -1, "acc_ad_max": -1, "acc_both": 0})
    p = d.p1.values
    gq = pd.Series(p).groupby(d.q.values)
    out["p1"] = p
    out["p1_rank_q"] = gq.rank(ascending=False, method="min").values
    top = gq.transform("max").values
    second = pd.Series(np.where(p < top, p, np.nan)).groupby(d.q.values).transform("max").fillna(0).values
    out["p1_gap_q"] = np.where(p >= top, p - second, p - top)
    # competing S1 for q: highest p1 among q's other candidates
    tmp = pd.DataFrame({"q": d.q.values, "p": p, "nm": out.acc_nm_max.values, "i": np.arange(len(d))})
    tmp = tmp.sort_values(["q", "p"], ascending=[True, False], kind="stable")
    tmp["r"] = tmp.groupby("q").cumcount()
    top2 = tmp[tmp.r < 2]
    first = top2[top2.r == 0].set_index("q")
    second = top2[top2.r == 1].set_index("q")
    comp = np.full(len(d), -2.0, np.float32)
    both_q = first.index.intersection(second.index)
    comp[first.loc[both_q, "i"].values] = second.loc[both_q, "nm"].values   # best S1 -> competitor = 2nd
    comp[second.loc[both_q, "i"].values] = first.loc[both_q, "nm"].values   # 2nd S1 -> competitor = best
    rest_i = tmp.i.values[tmp.r.values >= 2]
    comp[rest_i] = first.nm.reindex(tmp.q.values[tmp.r.values >= 2]).values   # others -> competitor = best
    out["comp_acc_nm"] = comp
    return out.astype(np.float32)


s2 = stage2_features(f)
cols2 = cols1 + list(s2.columns)
for c in s2.columns:
    f[c] = s2[c].values
del s2
log("stage-2 features (train) done")
m2 = lgb.train(P2, lgb.Dataset(f.loc[fit, cols2], f.y.values[fit]), R2,
               valid_sets=[lgb.Dataset(f.loc[hold, cols2], f.y.values[hold])],
               callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
m2.save_model(A + f"stage2{SUF}.txt")
log(f"stage-2 trained, best iter {m2.best_iteration}")
hold_ids = roles[roles == "holdout"].index
tph = tp[tp.s1.isin(hold_ids)]
rows = f[f.q.isin(f.q[hold].unique())]
hp = pd.DataFrame({"q": rows.q.values, "s1": rows.s1.values, "p": m2.predict(rows[cols2], num_threads=N_JOBS)})
hp.to_parquet(A + f"holdout_pred_stage2{SUF}.parquet", index=False)
sc = {t: macro_f05(one_owner(hp, t), tph, hold_ids) for t in [0.5, 0.6, 0.7, 0.8, 0.9]}
tb = max(sc, key=sc.get)
_, fr = macro_f05(one_owner(hp, tb), tph, hold_ids, return_frame=True)
log(f"HOLDOUT stage-2: best t {tb} -> {sc[tb]:.5f} | by t {({k: round(v, 5) for k, v in sc.items()})} | {breakdown(fr, s1c)}")
del f, rows

ft = pd.read_parquet(A + f"feats_test{SUF}.parquet")
ft["p1"] = full.predict(ft[cols1], num_threads=N_JOBS).astype(np.float32)
s2t = stage2_features(ft)
for c in s2t.columns:
    ft[c] = s2t[c].values
tpred = pd.DataFrame({"q": ft.q.values, "s1": ft.s1.values, "p": m2.predict(ft[cols2], num_threads=N_JOBS)})
flag = (ft.q_hn_eq_s1.values == 0).astype(float)
del ft, s2t
tpred.to_parquet(A + f"test_pred_stage2{SUF}.parquet", index=False)
log("test scored")
print("STAGE2_DONE", flush=True)
