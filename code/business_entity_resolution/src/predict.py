"""Stage 7: test inference and the two submission files.

candidate_pairs.tsv  = the pruned candidate set the model scores (grouped by S1)
matching_results.tsv = one-owner decoding (threshold tuned on the holdout) after the look-alike shift
Every test S1 gets exactly one row in both files.
Usage: python predict.py
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import ART_DIR, LOOKALIKE_SHIFT, LOOKALIKE_SHIFT_EXTRA, N_JOBS, OUT_DIR
from io_utils import write_id_lists
from tracking import log_run
from decode import decode
from train import feature_cols


def main():
    t0 = time.time()
    f = pd.read_parquet(ART_DIR / "feats_test.parquet")
    model = lgb.Booster(model_file=str(ART_DIR / "model.txt"))
    dec = json.loads((ART_DIR / "decode.json").read_text())
    dec["lookalike_shift"] = LOOKALIKE_SHIFT
    cols = model.feature_name()
    assert cols == feature_cols(f), "feature columns differ between train and test"
    pred = pd.DataFrame({"q": f.q.values, "s1": f.s1.values, "p": model.predict(f[cols], num_threads=N_JOBS)})
    pred.to_parquet(ART_DIR / "test_pred.parquet", index=False)
    diff = (f.q_hn_eq_s1.values == 0).astype(float)
    weight = np.where(f.nd_only_q.values > 0, LOOKALIKE_SHIFT_EXTRA / LOOKALIKE_SHIFT, 1.0) if LOOKALIKE_SHIFT else 1.0
    matches = decode(pred, dec, flag=diff * weight)  # shift = LOOKALIKE_SHIFT, or _EXTRA with an extra name word

    s1_ids = pd.read_parquet(ART_DIR / "norm_test_s1.parquet", columns=["entity_id", "country"])
    cand_groups = pred.groupby("s1").q.apply(list).to_dict()
    match_groups = matches.groupby("s1").other.apply(list).to_dict()
    write_id_lists(OUT_DIR / "candidate_pairs.tsv", s1_ids.entity_id, cand_groups, "candidate_entity_ids")
    write_id_lists(OUT_DIR / "matching_results.tsv", s1_ids.entity_id, match_groups, "matched_entity_ids")

    n = s1_ids.set_index("entity_id")
    stats = {"test_s1": len(n), "pairs_scored": len(pred), "matches": len(matches),
             "cand_per_s1": round(len(pred) / len(n), 3), "matches_per_s1": round(len(matches) / len(n), 3)}
    m_per = matches.groupby("s1").size().reindex(n.index, fill_value=0)
    for c, g in m_per.groupby(n.country):  # unlabelled France check: compare with US/India
        stats[f"{c}_matches_per_s1"] = round(float(g.mean()), 3)
        stats[f"{c}_pred_singleton_rate"] = round(float((g == 0).mean()), 4)
    log_run("predict", dec, stats, runtime_s=time.time() - t0, tier="0")


if __name__ == "__main__":
    main()
