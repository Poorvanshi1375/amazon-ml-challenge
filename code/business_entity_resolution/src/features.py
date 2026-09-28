"""Stage 3 (Tier 0): pair features for the pruned candidate pairs.

Input : cands_{split}.parquet (from prune.py) + norm_{split}_s*.parquet
Output: feats_{split}.parquet  (q, s1, feature columns; float32/int8)
Usage : python features.py --split train|test
"""
import argparse
import multiprocessing as mp
import re
import sys
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from config import ART_DIR, N_JOBS
from tracking import log_run

TXT = ["entity_id", "source", "name_clean", "name_core", "name_alt", "name_legal", "addr_clean",
       "house_no", "postcode", "num_tokens"]
CHUNK = 2_000_000
FIRST_INT = re.compile(r"\d+")
ALL_INT = re.compile(r"\d+")
DIGIT = re.compile(r"\d")
DIGIT_RUN = re.compile(r"\d+")
_DF = {}  # name-token document frequency among S1 names (set in main, shared with workers)
_N_DOCS = 1


SIM_WORKERS = N_JOBS  # set to 1 inside pool workers (parallelism comes from the pool)


def sim(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=SIM_WORKERS, dtype=np.float32) / (1 if scorer is JaroWinkler.normalized_similarity else 100)


def house_rel(a, b):
    """0 missing, 1 identical, 2 same leading number (802 vs 802-806, 00685 vs 685), 3 different."""
    out = np.zeros(len(a), dtype=np.int8)
    for i, (x, y) in enumerate(zip(a, b)):
        if not x or not y:
            continue
        if x == y:
            out[i] = 1
            continue
        mx, my = FIRST_INT.search(x), FIRST_INT.search(y)
        out[i] = 2 if (mx and my and int(mx.group()) == int(my.group())) else 3
    return out


def cat_rel(a, b):
    """0 unknown (either empty), 1 equal, 2 conflict."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    miss = (a == "") | (b == "")
    return np.where(miss, 0, np.where(a == b, 1, 2)).astype(np.int8)


def jaccard(a, b):
    out = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if sx or sy:
            out[i] = len(sx & sy) / len(sx | sy)
    return out


def _ints(x):
    return [int(m) for m in ALL_INT.findall(x)] if x else []


def house_detail(hq, hs, nq, ns):
    """Look-alike detection on numbers.

    True matches show number *noise* (00685 vs 685, 802-806 vs 802, 3-21 vs 21,
    one mistyped digit); distractors are *neighbours* (1385 vs 1386, 6154 vs 6167).
    Returns: abs gap of leading numbers (log1p, -1 missing), relative gap,
    digit edit distance of leading numbers, containment flag, smallest gap between
    any two numbers of the addresses (log1p, -1 missing).
    """
    n = len(hq)
    gap = np.full(n, -1, np.float32)
    rel = np.full(n, -1, np.float32)
    dlev = np.full(n, -1, np.int8)
    contain = np.zeros(n, np.int8)
    mingap = np.full(n, -1, np.float32)
    for i in range(n):
        a, b = _ints(hq[i]), _ints(hs[i])
        if a and b:
            x, y = a[0], b[0]
            gap[i] = np.log1p(abs(x - y))
            rel[i] = abs(x - y) / max(x, y, 1)
            dlev[i] = min(Levenshtein.distance(str(x), str(y)), 9)
            contain[i] = 1 if (x in b or y in a) else 0
        na, nb = _ints(nq[i]), _ints(ns[i])
        if na and nb:
            mingap[i] = np.log1p(min(abs(u - v) for u in na for v in nb))
    return gap, rel, dlev, contain, mingap


def name_diff(qc, sc):
    """What kind of words differ between the two name cores.

    Returns: #words only in query, #words only in S1, mean best spelling
    similarity of differing words to the other side's differing words (-1 if one
    side has none: a pure addition), log document frequency of the rarest and of
    the most common differing word (-1 if none).
    """
    n = len(qc)
    oq = np.zeros(n, np.int8)
    os_ = np.zeros(n, np.int8)
    typo = np.full(n, -1, np.float32)
    rare = np.full(n, -20, np.float32)  # -20 = no differing word (real values are > -15)
    common = np.full(n, -20, np.float32)
    for i in range(n):
        tq, ts = set(qc[i].split()), set(sc[i].split())
        dq, ds = tq - ts, ts - tq
        oq[i], os_[i] = min(len(dq), 99), min(len(ds), 99)
        if dq and ds:
            sims = [max(fuzz.ratio(u, v) for v in ds) for u in dq]
            typo[i] = float(np.mean(sims)) / 100
        diff = dq | ds
        if diff:
            dfs = [np.log1p(_DF.get(t, 0)) - np.log(_N_DOCS) for t in diff]
            rare[i], common[i] = min(dfs), max(dfs)
    return oq, os_, typo, rare, common


def nonum(texts):
    return [" ".join(t for t in x.split() if not DIGIT.search(t)) for x in texts]



def first_int_arr(values):
    out = np.full(len(values), -1, np.int64)
    for i, x in enumerate(values):
        m = FIRST_INT.search(x) if x else None
        if m:
            out[i] = int(m.group()[-12:])
    return out


def group_features(f, Q, S):
    """Consistency of a candidate with the other candidates of the same S1 / query.

    Uses only candidate texts and pair similarities (no labels), so it is safe for
    training and identical at test time.
    """
    q_hn = first_int_arr(Q.house_no.reindex(f.q.values).values)
    s_hn = first_int_arr(S.house_no.reindex(f.s1.values).values)
    f["q_hn_eq_s1"] = np.where((q_hn >= 0) & (s_hn >= 0), (q_hn == s_hn).astype(np.int8), -1).astype(np.int8)
    key = pd.DataFrame({"s1": f.s1.values, "h": q_hn})
    same = key.groupby(["s1", "h"]).h.transform("size").values - 1
    f["grp_hn_support"] = np.where(q_hn >= 0, same, -1).astype(np.int16)
    has = (q_hn >= 0).astype(np.int16)
    f["grp_hn_known"] = pd.Series(has).groupby(f.s1.values).transform("sum").values.astype(np.int16)
    eq = (f.q_hn_eq_s1.values == 1).astype(np.int16)
    f["grp_n_hn_eq_s1"] = pd.Series(eq).groupby(f.s1.values).transform("sum").values.astype(np.int16)
    for col in ("ad_tset", "nm_tset", "nm_jw"):
        v = f[col].astype(np.float32)
        g_s1, g_q = v.groupby(f.s1.values), v.groupby(f.q.values)
        f[f"{col}_rank_s1"] = g_s1.rank(ascending=False, method="min").values.astype(np.int16)
        f[f"{col}_rank_q"] = g_q.rank(ascending=False, method="min").values.astype(np.int16)
        top_q = g_q.transform("max").values
        second_q = v.where(v < top_q).groupby(f.q.values).transform("max").fillna(0).values
        f[f"{col}_gap_q"] = np.where(v.values >= top_q, v.values - second_q, v.values - top_q).astype(np.float32)
    close = (f.nm_tset.values >= 0.9).astype(np.int16)
    f["q_n_name_close"] = pd.Series(close).groupby(f.q.values).transform("sum").values.astype(np.int16)
    f["s1_n_name_close"] = pd.Series(close).groupby(f.s1.values).transform("sum").values.astype(np.int16)
    return f



SIB_K = 3


def sibling_features(f, Q):
    """Compare each candidate with the S1's other strongest candidates ("siblings").

    Siblings = the top SIB_K candidates of the S1 by blocking score (score_b), excluding
    the candidate itself. A true variant resembles its siblings; a look-alike neighbour
    (different house number, extra word) usually does not. Label-free.
    """
    n = len(f)
    order = np.lexsort((-f.score_b.values, f.s1.values))
    s1_sorted = f.s1.values[order]
    starts = np.r_[0, np.flatnonzero(s1_sorted[1:] != s1_sorted[:-1]) + 1]
    sizes = np.diff(np.r_[starts, n])
    grp_of = np.repeat(np.arange(len(starts)), sizes)
    top = np.full((len(starts), SIB_K), -1, np.int64)
    for k in range(SIB_K):
        ok = sizes > k
        top[ok, k] = order[starts[ok] + k]
    row_grp = np.empty(n, np.int64)
    row_grp[order] = grp_of
    cand = top[row_grp]                                  # (n, SIB_K) row indices of the S1's top candidates
    self_ = np.arange(n)[:, None]
    cand = np.where(cand == self_, -1, cand)             # exclude the row itself
    qn = Q.name_core.reindex(f.q.values).values
    qa = Q.addr_clean.reindex(f.q.values).values
    qh = first_int_arr(Q.house_no.reindex(f.q.values).values)
    best_nm = np.full(n, -1, np.float32)
    best_ad = np.full(n, -1, np.float32)
    hn_agree = np.zeros(n, np.int8)
    hn_known = np.zeros(n, np.int8)
    n_sib = (cand >= 0).sum(1).astype(np.int8)
    for k in range(SIB_K):
        idx = np.flatnonzero(cand[:, k] >= 0)
        if len(idx) == 0:
            continue
        j = cand[idx, k]
        best_nm[idx] = np.maximum(best_nm[idx], sim(qn[idx], qn[j], fuzz.token_set_ratio))
        best_ad[idx] = np.maximum(best_ad[idx], sim(qa[idx], qa[j], fuzz.token_set_ratio))
        both = (qh[idx] >= 0) & (qh[j] >= 0)
        hn_known[idx] += both.astype(np.int8)
        hn_agree[idx] += (both & (qh[idx] == qh[j])).astype(np.int8)
    f["sib_n"] = n_sib
    f["sib_nm_best"] = best_nm
    f["sib_ad_best"] = best_ad
    f["sib_hn_known"] = hn_known
    f["sib_hn_agree"] = hn_agree
    f["sib_hn_disagree"] = (hn_known - hn_agree).astype(np.int8)
    return f



def uniqueness_features(f, split):
    """How unique is the evidence? Counts over the whole S1 file of the split (same country, label-free).

    u_q_name_s1 / u_s1_name  : S1 records sharing the query's / this S1's name core
    u_q_addr_s1 / u_s1_addr  : S1 records sharing the query's / this S1's address (-1 if empty)
    u_q_street_s1            : S1 records sharing the query's address without numbers
    """
    cols = ["entity_id", "country", "name_core", "addr_clean"]
    s1 = pd.read_parquet(ART_DIR / f"norm_{split}_s1.parquet", columns=cols)
    q = pd.concat([pd.read_parquet(ART_DIR / f"norm_{split}_s{s}.parquet", columns=cols) for s in (2, 3)])
    q = q[q.entity_id.isin(set(f.q.values))]
    for d in (s1, q):
        d["street"] = [" ".join(DIGIT_RUN.sub(" ", x).split()) for x in d.addr_clean.values]

    def count(df, col, ref):
        key = df.country + "|" + df[col]
        cnt = (ref.country + "|" + ref[col]).value_counts()
        v = key.map(cnt).fillna(0).astype(np.int32).values
        return pd.Series(np.where(df[col].values == "", -1, v).astype(np.int32), index=df.entity_id.values)

    f["u_q_name_s1"] = count(q, "name_core", s1).reindex(f.q.values).values
    f["u_s1_name"] = count(s1, "name_core", s1).reindex(f.s1.values).values
    f["u_q_addr_s1"] = count(q, "addr_clean", s1).reindex(f.q.values).values
    f["u_s1_addr"] = count(s1, "addr_clean", s1).reindex(f.s1.values).values
    f["u_q_street_s1"] = count(q, "street", s1).reindex(f.q.values).values
    return f


def top1(d):
    """Pair is the query's rank-1 S1 in any index."""
    m = (d.rank_b == 1) | (d.rank_n == 1)
    return m | (d.rank_a == 1) if "rank_a" in d else m


def pair_features(c, Q, S):
    f = pd.DataFrame({"q": c.q.values, "s1": c.s1.values})
    for col in ("score_n", "rank_n", "score_b", "rank_b", "margin_n", "margin_b", "top_n", "top_b", "n_raw_q",
                "score_a", "rank_a", "margin_a", "top_a", "p_prune"):
        if col in c:
            f[col] = c[col].values
    f["n_cand_q"] = c.groupby("q").q.transform("size").values.astype(np.int16)
    f["n_cand_s1"] = c.groupby("s1").s1.transform("size").values.astype(np.int16)
    f["n_top1_s1"] = top1(c).groupby(c.s1).transform("sum").values.astype(np.int16)
    q, s = Q.reindex(c.q.values), S.reindex(c.s1.values)
    qc, sc = q.name_core.values, s.name_core.values
    f["nm_ratio"] = sim(qc, sc, fuzz.ratio)
    f["nm_tset"] = sim(qc, sc, fuzz.token_set_ratio)
    f["nm_tsort"] = sim(qc, sc, fuzz.token_sort_ratio)
    f["nm_partial"] = sim(qc, sc, fuzz.partial_ratio)
    f["nm_jw"] = sim(qc, sc, JaroWinkler.normalized_similarity)
    f["nm_full_tset"] = sim(q.name_clean.values, s.name_clean.values, fuzz.token_set_ratio)
    qa = np.where(q.name_alt.values == "", qc, q.name_alt.values)
    sa = np.where(s.name_alt.values == "", sc, s.name_alt.values)
    f["nm_alias_tset"] = np.maximum(sim(qa, sc, fuzz.token_set_ratio), sim(qc, sa, fuzz.token_set_ratio))
    f["nm_jacc"] = jaccard(qc, sc)
    f["nm_len_diff"] = np.abs(pd.Series(qc).str.len().values - pd.Series(sc).str.len().values).astype(np.int16)
    f["nm_first_eq"] = (pd.Series(qc).str.split().str[0].values == pd.Series(sc).str.split().str[0].values).astype(np.int8)
    f["legal_rel"] = cat_rel(q.name_legal.values, s.name_legal.values)
    qa_, sa_ = q.addr_clean.values, s.addr_clean.values
    f["ad_ratio"] = sim(qa_, sa_, fuzz.ratio)
    f["ad_tset"] = sim(qa_, sa_, fuzz.token_set_ratio)
    f["ad_partial"] = sim(qa_, sa_, fuzz.partial_ratio)
    f["ad_jacc"] = jaccard(qa_, sa_)
    f["q_addr_empty"] = (qa_ == "").astype(np.int8)
    f["s1_addr_empty"] = (sa_ == "").astype(np.int8)
    f["house_rel"] = house_rel(q.house_no.values, s.house_no.values)
    f["post_rel"] = cat_rel(q.postcode.values, s.postcode.values)
    f["num_jacc"] = jaccard(q.num_tokens.values, s.num_tokens.values)
    f["q_source"] = q.source.values.astype(np.int8)
    # look-alike features (same street / near-identical name, different business)
    (f["hn_gap"], f["hn_rel"], f["hn_dlev"], f["hn_contain"], f["num_mingap"]) = house_detail(
        q.house_no.values, s.house_no.values, q.num_tokens.values, s.num_tokens.values)
    f["ad_nonum_tset"] = sim(nonum(qa_), nonum(sa_), fuzz.token_set_ratio)
    (f["nd_only_q"], f["nd_only_s1"], f["nd_typo"], f["nd_rare_df"], f["nd_common_df"]) = name_diff(qc, sc)
    return f


TASK_ROWS = 400_000
_C = _Q = _S = None


def _init_worker():
    global SIM_WORKERS
    SIM_WORKERS = 1


def _work(task):
    a, b = task
    return pair_features(_C.iloc[a:b], _Q, _S)


def main(split):
    t0 = time.time()
    c = pd.read_parquet(ART_DIR / f"cands_{split}.parquet")
    S = pd.read_parquet(ART_DIR / f"norm_{split}_s1.parquet", columns=TXT).set_index("entity_id")
    Q = pd.concat([pd.read_parquet(ART_DIR / f"norm_{split}_s{s}.parquet", columns=TXT) for s in (2, 3)]).set_index("entity_id")
    Q = Q[Q.index.isin(c.q.unique())]
    S = S[S.index.isin(c.s1.unique())]
    # string features are computed in parallel on row ranges aligned to query boundaries
    c = c.sort_values("q", kind="stable").reset_index(drop=True)
    qv = c.q.values
    bounds = np.r_[0, np.flatnonzero(qv[1:] != qv[:-1]) + 1, len(c)]
    cuts = [0]
    for b in bounds[1:]:
        if b - cuts[-1] >= TASK_ROWS or b == len(c):
            cuts.append(int(b))
    tasks = list(zip(cuts[:-1], cuts[1:]))
    global _C, _Q, _S, _DF, _N_DOCS
    _C, _Q, _S = c, Q, S
    # name-word document frequency over all S1 names of this split (data-derived, no labels)
    all_s1 = pd.read_parquet(ART_DIR / f"norm_{split}_s1.parquet", columns=["name_core"]).name_core
    _DF = pd.Series([t for x in all_s1.values for t in set(x.split())]).value_counts().to_dict()
    _N_DOCS = len(all_s1)
    parts = []
    if sys.platform.startswith("linux") and N_JOBS > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(N_JOBS, initializer=_init_worker) as pool:
            for i, part in enumerate(pool.imap(_work, tasks)):
                parts.append(part)
                if i % 10 == 9 or i == len(tasks) - 1:
                    print(f"features {split}: {sum(len(p) for p in parts):,}/{len(c):,} pairs", flush=True)
    else:
        for t in tasks:
            parts.append(_work(t))
            print(f"features {split}: {sum(len(p) for p in parts):,}/{len(c):,} pairs", flush=True)
    f = pd.concat(parts, ignore_index=True)
    # recompute S1-level counts over all pairs (chunks split S1 groups)
    f["n_cand_s1"] = f.groupby("s1").s1.transform("size").astype(np.int16)
    f["n_top1_s1"] = top1(f).groupby(f.s1).transform("sum").astype(np.int16)
    f = group_features(f, Q, S)
    f = sibling_features(f, Q)
    f = uniqueness_features(f, split)
    f.to_parquet(ART_DIR / f"feats_{split}.parquet", index=False)
    log_run("features", {"split": split}, {"pairs": len(f), "n_features": f.shape[1] - 2}, runtime_s=time.time() - t0, tier="0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    main(ap.parse_args().split)
