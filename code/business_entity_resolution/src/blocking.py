"""Stage 2: candidate generation (blocking).

Reverse direction: every S2/S3 record (query) retrieves its top-K S1 records of
the same country, from two char-n-gram TF-IDF indexes:
  "n" - full cleaned name (includes alias words)          -> K_N candidates
  "b" - name core + cleaned address                       -> K_B candidates
  "a" - cleaned address only (queries with an address)     -> K_A candidates
        catches trade names / native-script names that share nothing with S1
Queries with an empty address keep K_EMPTY candidates from the name index.
Search = sparse matrix product with top-K selection (sparse_dot_topn), per
country and, where the country has training labels, per learned state partition:
  - an S1 record's key is the last address part that is a frequent final part
    of S1 addresses in its country (its state, or a city used as one);
  - a query's key is its last address part seen in the key table, which maps it
    (learned from non-holdout matched pairs) to the S1 keys covering COVER of its
    matches (e.g. "hyderabad" -> telangana, andhra pradesh);
  - queries without a usable key, and every country without labels (France),
    search the whole country; S1 records without a key are in every partition.

Output: blocks_{split}.parquet with one row per (query, s1) pair and columns
score_n, rank_n, score_b, rank_b (NaN / 0 when an index did not return the pair).

For the train split every S2/S3 record is a query (ER_TRAIN_QUERIES=all, the
default), so holdout S1 entities face the same competition as test entities.
ER_TRAIN_QUERIES=subset restores the faster Tier 0 behaviour (queries of
train/holdout S1s + a share of distractors), which under-counts competitors.

Usage:
  python blocking.py --split train [--bench 0.01]
  python blocking.py --split test
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
from scipy import sparse
from sparse_dot_topn import sp_matmul_topn

from config import ART_DIR, N_JOBS, SEED
from tracking import log_run

import os

K_N, K_B = int(os.environ.get("ER_K_N", 5)), int(os.environ.get("ER_K_B", 10))
K_A, K_EMPTY = int(os.environ.get("ER_K_A", 10)), int(os.environ.get("ER_K_EMPTY", 100))
INDEXES = os.environ.get("ER_INDEXES", "n,b,a").split(",")
COVER, MAX_KEYS, MIN_KEY_PAIRS = 0.997, 8, 200
PARTITION = os.environ.get("ER_PARTITION", "1") == "1"
# "all": every train S2/S3 record is a query, so holdout S1s face the same competition as test S1s
TRAIN_QUERIES = os.environ.get("ER_TRAIN_QUERIES", "all")
NGRAM = tuple(int(x) for x in os.environ.get("ER_NGRAM", "2,4").split(","))
MAX_DF = float(os.environ.get("ER_MAX_DF", "0.05"))
CHUNK = 200_000
FIT_SAMPLE = 1_000_000

COLS = ["entity_id", "country", "source", "name_clean", "name_core", "addr_clean", "addr_comps"]


def texts(df, index):
    if index == "n":
        return df.name_clean.values
    if index == "a":
        return df.addr_clean.values
    return (df.name_core + " " + df.addr_clean).str.strip().values


def par_transform(vec, txt, part=50_000):
    """TfidfVectorizer.transform on all cores (it is single-threaded)."""
    if len(txt) <= part:
        return vec.transform(txt)
    mats = Parallel(n_jobs=N_JOBS)(delayed(vec.transform)(txt[i:i + part]) for i in range(0, len(txt), part))
    return sparse.vstack(mats, format="csr")


def search(s1, q, index, groups, bench_log):
    """Top-k S1 rows for every query row of one country.

    groups: list of (s1_row_positions, q_row_positions, k); each query appears in
    exactly one group and is searched only against that group's S1 rows.
    """
    t = time.time()
    s1_txt, q_txt = texts(s1, index), texts(q, index)
    rng = np.random.default_rng(SEED)
    both = np.concatenate([s1_txt, q_txt])
    fit_txt = both if len(both) <= FIT_SAMPLE else both[rng.choice(len(both), FIT_SAMPLE, replace=False)]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM, min_df=2, max_df=MAX_DF,
                          sublinear_tf=True, dtype=np.float32)
    vec.fit(fit_txt)
    S_all = par_transform(vec, s1_txt)
    Q_all = par_transform(vec, q_txt, part=100_000)
    t_fit = time.time() - t
    out = []
    s1_ids, q_ids = s1.entity_id.values, q.entity_id.values
    for s_pos, q_pos, k in groups:
        if len(s_pos) == 0 or len(q_pos) == 0 or k == 0:
            continue
        S = S_all[s_pos].T.tocsr()
        for i in range(0, len(q_pos), CHUNK):
            qp = q_pos[i:i + CHUNK]
            C = sp_matmul_topn(Q_all[qp], S, top_n=k, threshold=0.05, sort=True, n_threads=N_JOBS).tocoo()
            if len(C.data) == 0:
                continue
            starts = np.r_[0, np.flatnonzero(np.diff(C.row)) + 1]
            rank = (1 + np.arange(len(C.data)) - np.repeat(starts, np.diff(np.r_[starts, len(C.data)]))).astype(np.int8)
            out.append(pd.DataFrame({"q": q_ids[qp[C.row]], "s1": s1_ids[s_pos[C.col]],
                                     f"score_{index}": C.data.astype(np.float32), f"rank_{index}": rank}))
    bench_log.append({"index": index, "n_s1": len(s1), "n_q": len(q), "groups": len(groups), "fit_s": round(t_fit, 1),
                      "search_s": round(time.time() - t - t_fit, 1), "vocab": len(vec.vocabulary_)})
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def last_key(comps_values, vocab):
    """Last address part (scanning from the end) that is in vocab, else empty string."""
    out = []
    for comps in comps_values:
        k = ""
        for p in reversed(comps.split("|")):
            if p in vocab:
                k = p
                break
        out.append(k)
    return np.array(out, dtype=object)


def learn_partitions():
    """Learn per-country S1 key vocabularies and query-key -> S1-keys tables (train only)."""
    roles = pd.read_parquet(ART_DIR / "roles.parquet").set_index("s1").role
    tp = pd.read_parquet(ART_DIR / "truth_pairs.parquet")
    tp = tp[tp.s1.map(roles).ne("holdout")]
    s1 = pd.read_parquet(ART_DIR / "norm_train_s1.parquet", columns=["entity_id", "country", "addr_comps"])
    q = pd.concat([pd.read_parquet(ART_DIR / f"norm_train_s{s}.parquet", columns=["entity_id", "addr_comps"]) for s in (2, 3)])
    tables = {}
    for country, g in s1.groupby("country"):
        last = g.addr_comps.str.split("|").str[-1]
        vc = last.value_counts()
        s1_vocab = set(vc[vc >= max(50, len(g) * 0.0005)].index)
        s1_key = pd.Series(last_key(g.addr_comps.values, s1_vocab), index=g.entity_id.values)
        pairs = tp[tp.s1.isin(s1_key.index)]
        pq = q[q.entity_id.isin(set(pairs.other))]
        parts = pd.Series([p for lst in pq.addr_comps.str.split("|") for p in set(lst)
                           if p and not any(ch.isdigit() for ch in p)])
        pc = parts.value_counts()
        q_vocab = set(pc[pc >= MIN_KEY_PAIRS].index)
        q_key = pd.Series(last_key(pq.addr_comps.values, q_vocab), index=pq.entity_id.values)
        m = pd.DataFrame({"qk": q_key.reindex(pairs.other.values).values, "sk": s1_key.reindex(pairs.s1.values).values})
        m = m[(m.qk != "") & (m.sk != "") & m.qk.notna() & m.sk.notna()]
        table = {}
        for qk, h in m.groupby("qk"):
            if len(h) < MIN_KEY_PAIRS:
                continue
            dist = h.sk.value_counts(normalize=True)
            keys = list(dist.index[:np.searchsorted(dist.cumsum().values, COVER) + 1])
            if len(keys) <= MAX_KEYS:
                table[qk] = keys
        tables[country] = {"s1_vocab": sorted(s1_vocab), "table": table}
        cov = m.qk.map(lambda k: k in table).mean()
        print(f"partitions {country}: {len(s1_vocab)} S1 keys, {len(table)} query keys, "
              f"{cov:.1%} of matched keyed queries routed to a partition", flush=True)
    (ART_DIR / "partitions.json").write_text(json.dumps(tables))
    return tables


def make_groups(s1c, qc, part):
    """Split one country's queries into groups searched against subsets of S1."""
    all_s1 = np.arange(len(s1c))
    all_q = np.arange(len(qc))
    if not PARTITION or part is None:
        return [(all_s1, all_q)]
    s1_key = last_key(s1c.addr_comps.values, set(part["s1_vocab"]))
    table = part["table"]
    q_key = last_key(qc.addr_comps.values, set(table))
    unkeyed_s1 = np.flatnonzero(s1_key == "")
    by_key = {k: np.flatnonzero(s1_key == k) for k in set(s1_key) if k}
    groups = []
    for qk, qpos in pd.Series(np.arange(len(qc))).groupby(q_key):
        if qk == "":
            groups.append((all_s1, qpos.values))
            continue
        s_pos = np.concatenate([by_key.get(k, np.array([], dtype=np.int64)) for k in table[qk]] + [unkeyed_s1])
        groups.append((np.sort(s_pos), qpos.values))
    return groups


def select_train_queries(q, roles, truth_pairs):
    """Matched queries of train/holdout S1s + an equal share of distractors."""
    keep_s1 = set(roles[roles.role.isin(["train", "holdout"])].s1)
    share = len(keep_s1) / len(roles)
    owner = truth_pairs.set_index("other").s1
    o = q.entity_id.map(owner)
    matched_keep = o.isin(keep_s1).values
    unmatched = o.isna().values
    rng = np.random.default_rng(SEED)
    pick = matched_keep | (unmatched & (rng.random(len(q)) < share))
    return q[pick]


def main(split, bench=None):
    t0 = time.time()
    s1 = pd.read_parquet(ART_DIR / f"norm_{split}_s1.parquet", columns=COLS)
    q = pd.concat([pd.read_parquet(ART_DIR / f"norm_{split}_s{s}.parquet", columns=COLS) for s in (2, 3)],
                  ignore_index=True)
    if split == "train" and TRAIN_QUERIES == "subset":
        q = select_train_queries(q, pd.read_parquet(ART_DIR / "roles.parquet"),
                                 pd.read_parquet(ART_DIR / "truth_pairs.parquet"))
    if bench:
        q = q.sample(frac=bench, random_state=SEED)
    only = os.environ.get("ER_ONLY_COUNTRY")  # re-run a single country (e.g. after a France-only normalisation change)
    if only:
        s1, q = s1[s1.country == only], q[q.country == only]
    print(f"{split}: {len(s1):,} S1, {len(q):,} queries", flush=True)

    if split == "train" and PARTITION and not (ART_DIR / "partitions.json").exists():
        learn_partitions()
    tables = json.loads((ART_DIR / "partitions.json").read_text()) if PARTITION else {}
    bench_log, parts = [], []
    for country in sorted(set(q.country) | set(s1.country)):
        s1c = s1[s1.country == country].reset_index(drop=True)
        qc = q[q.country == country].reset_index(drop=True)
        if len(s1c) == 0 or len(qc) == 0:
            continue
        groups = make_groups(s1c, qc, tables.get(country))
        full = sum(len(g[1]) for g in groups if len(g[0]) == len(s1c))
        avg = np.average([len(g[0]) for g in groups], weights=[len(g[1]) for g in groups])
        print(f"  {country}: {len(groups)} groups, {full / len(qc):.1%} of queries search the whole country, "
              f"avg S1 per query {avg:,.0f} of {len(s1c):,}", flush=True)
        q_empty = (qc.addr_clean.values == "")
        res = []
        for ix in INDEXES:
            g_ix = []
            for s_pos, q_pos in groups:
                e = q_empty[q_pos]
                if ix == "n":
                    g_ix += [(s_pos, q_pos[~e], K_N), (s_pos, q_pos[e], K_EMPTY)]
                elif ix == "b":
                    g_ix += [(s_pos, q_pos[~e], K_B)]
                else:
                    g_ix += [(s_pos, q_pos[~e], K_A)]
            res.append(search(s1c, qc, ix, g_ix, bench_log))
        res = [r for r in res if len(r)]
        m = res[0]
        for r in res[1:]:
            m = m.merge(r, on=["q", "s1"], how="outer")
        parts.append(m)
        print(f"  {country}: {len(qc):,} queries -> {len(m):,} pairs | {bench_log[-len(INDEXES):]}", flush=True)
    blocks = pd.concat(parts, ignore_index=True)
    for ix in ("n", "b", "a"):
        if f"rank_{ix}" not in blocks:
            blocks[f"rank_{ix}"], blocks[f"score_{ix}"] = 0, np.float32(0)
        blocks[f"rank_{ix}"] = blocks[f"rank_{ix}"].fillna(0).astype(np.int8)
    if bench:
        est = sum(b["search_s"] for b in bench_log) / bench
        fit = sum(b["fit_s"] for b in bench_log)
        tp = pd.read_parquet(ART_DIR / "truth_pairs.parquet") if split == "train" else None
        rec = None
        if tp is not None:
            t = tp[tp.other.isin(set(q.entity_id))]
            rec = round(float(t.merge(blocks[["q", "s1"]], left_on=["other", "s1"], right_on=["q", "s1"], how="left").q.notna().mean()), 5)
        print(f"BENCH ngram={NGRAM} max_df={MAX_DF} k={K_N},{K_B} partition={PARTITION}: projected full search ≈ {est / 60:.1f} min (+fit {fit / 60:.1f} min), "
              f"recall on sampled matched queries = {rec}, pairs/query = {len(blocks) / len(q):.2f}", flush=True)
        log_run("blocking_bench", {"split": split, "bench": bench, "ngram": NGRAM, "max_df": MAX_DF,
                                   "k": [K_N, K_B, K_A, K_EMPTY], "indexes": INDEXES, "partition": PARTITION}, {"projected_min": round(est / 60, 1), "recall": rec, "detail": bench_log},
                runtime_s=time.time() - t0, tier="0")
        return
    blocks.to_parquet(ART_DIR / f"blocks_{split}.parquet", index=False)
    metrics = {"pairs": len(blocks), "queries": int(blocks.q.nunique()), "detail": bench_log}
    if split == "train":
        metrics.update(recall_report(blocks))
    log_run("blocking", {"split": split, "ngram": NGRAM, "max_df": MAX_DF, "k": [K_N, K_B, K_A, K_EMPTY], "indexes": INDEXES, "partition": PARTITION}, metrics,
            runtime_s=time.time() - t0, tier="0")


def recall_report(blocks, role="holdout"):
    """Share of true (s1, other) pairs of holdout S1s found by blocking, by slice."""
    roles = pd.read_parquet(ART_DIR / "roles.parquet")
    tp = pd.read_parquet(ART_DIR / "truth_pairs.parquet")
    hold = set(roles[roles.role == role].s1)
    tp = tp[tp.s1.isin(hold)]
    found = tp.merge(blocks[["q", "s1"]], left_on=["other", "s1"], right_on=["q", "s1"], how="left").q.notna()
    info = pd.concat([pd.read_parquet(ART_DIR / f"norm_train_s{s}.parquet",
                                      columns=["entity_id", "country", "source", "addr_clean", "business_name"])
                      for s in (2, 3)]).set_index("entity_id")
    sl = info.reindex(tp.other.values)
    out = {"recall": round(found.mean(), 5), "true_pairs": len(tp)}
    groups = {
        "country": sl.country.values,
        "source": sl.source.values,
        "addr_empty": (sl.addr_clean == "").values,
        "non_ascii_name": (~sl.business_name.map(str.isascii)).values,
    }
    for g, v in groups.items():
        for k, r in pd.Series(found.values).groupby(v).mean().items():
            out[f"recall_{g}={k}"] = round(float(r), 5)
    cand = blocks[blocks.s1.isin(hold)].groupby("s1").size()
    out["cand_per_holdout_s1_raw"] = round(float(cand.reindex(list(hold), fill_value=0).mean()), 2)
    print("RECALL:", out, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--bench", type=float, default=None)
    a = ap.parse_args()
    main(a.split, a.bench)
