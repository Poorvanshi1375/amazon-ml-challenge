"""Stage 1 runner: normalise every source file, learn rewrites, save Parquet.

Outputs (in ART_DIR):
  roles.parquet                     train S1 id -> holdout / train / rest
  truth_pairs.parquet               train ground truth as (s1, other) pairs
  rewrites.json                     learned address rewrites (documented in methodology)
  norm_{split}_s{n}.parquet         normalised records, raw text kept

Usage: python prepare.py          (set ER_DATA_DIR to a smoke dataset from make_smoke.py for local tests)
"""
import json
import time

import pandas as pd

from config import ART_DIR, DATA_DIR, N_JOBS, SEED
from io_utils import read_source, read_truth_pairs
from normalize import apply_rewrites, mine_rewrites, mine_word_dict, normalize_df, set_word_dict
from split import assign_roles
from tracking import log_run

# Hand-written seed for countries without training data (France): common French
# street-type abbreviations. General language knowledge, documented in the methodology.
FRENCH_SEED = {"r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
               "pl": "place", "imp": "impasse", "all": "allee", "rte": "route", "ch": "chemin",
               "sq": "square", "fg": "faubourg", "qu": "quai", "st": "saint", "ste": "sainte"}

KEEP = ["entity_id", "country", "source", "business_name", "business_address", "name_clean", "name_core",
        "name_alt", "name_legal", "addr_clean", "addr_comps", "house_no", "postcode", "num_tokens"]


def main():
    t0 = time.time()
    tp = read_truth_pairs()
    raw = {(split, s): read_source(split, s) for split in ("train", "test") for s in (1, 2, 3)}
    roles = assign_roles(raw[("train", 1)].entity_id)

    # learn the native-script word dictionary from matched non-holdout training names
    wp = tp[tp.s1.map(roles).ne("holdout")]
    names_o = pd.concat([raw[("train", 2)], raw[("train", 3)]]).set_index("entity_id").business_name
    names_1 = raw[("train", 1)].set_index("entity_id").business_name
    nat = names_o.reindex(wp.other.values).fillna("").values
    keep = [not x.isascii() for x in nat]
    word_dict = mine_word_dict(nat[keep], names_1.reindex(wp.s1.values[keep]).fillna("").values)
    set_word_dict(word_dict)
    (ART_DIR / "word_dict.json").write_text(json.dumps(word_dict, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"word dictionary: {len(word_dict):,} native-script words learned from {sum(keep):,} pairs", flush=True)

    norm = {}
    for (split, s), df in raw.items():
        t = time.time()
        norm[(split, s)] = normalize_df(df, n_jobs=N_JOBS)[KEEP]
        print(f"normalised {split} s{s}: {len(df):,} rows in {time.time() - t:.0f}s", flush=True)
    del raw

    s1_train = norm[("train", 1)]
    roles.rename_axis("s1").reset_index().to_parquet(ART_DIR / "roles.parquet")
    tp = tp[tp.s1.isin(s1_train.entity_id)]
    tp.to_parquet(ART_DIR / "truth_pairs.parquet")

    # learn rewrites per country from matched pairs of non-holdout entities only
    # (a shared map turned India's "tn" = Tamil Nadu into "tennessee")
    mine = tp[tp.s1.map(roles).ne("holdout")]
    other = pd.concat([norm[("train", 2)], norm[("train", 3)]])[["entity_id", "addr_comps"]]
    mp = mine.merge(s1_train[["entity_id", "country", "addr_comps"]], left_on="s1", right_on="entity_id")              .merge(other, left_on="other", right_on="entity_id", suffixes=("_1", "_2"))
    maps = {}
    for country, g in mp.groupby("country"):
        g = g.sample(min(len(g), 1_000_000), random_state=SEED)
        min_count = max(10, len(g) // 7500)
        comp_map, token_map = mine_rewrites(g.addr_comps_1, g.addr_comps_2, min_count=min_count)
        maps[country] = {"min_count": min_count, "pairs_used": len(g), "comp_map": comp_map, "token_map": token_map}
        print(f"rewrites {country}: {len(comp_map)} part-level, {len(token_map)} word-level from {len(g):,} pairs", flush=True)
    maps["_unseen_country_seed"] = {"comp_map": {}, "token_map": FRENCH_SEED}
    (ART_DIR / "rewrites.json").write_text(json.dumps(maps, indent=1))

    for (split, s), df in norm.items():
        comps, clean = df.addr_comps.values.copy(), df.addr_clean.values.copy()
        for country, idx in df.groupby("country").indices.items():
            m = maps.get(country, maps["_unseen_country_seed"])
            c2, k2 = apply_rewrites(df.addr_comps.values[idx], m["comp_map"], m["token_map"])
            comps[idx], clean[idx] = c2, k2
        df = df.assign(addr_comps=comps, addr_clean=clean)
        df.to_parquet(ART_DIR / f"norm_{split}_s{s}.parquet", index=False)
    log_run("prepare", {"data_dir": str(DATA_DIR)},
            {"rows": {f"{k[0]}_s{k[1]}": len(v) for k, v in norm.items()},
             "rewrites": {c: [len(m["comp_map"]), len(m["token_map"])] for c, m in maps.items()},
             "word_dict": len(word_dict),
             "roles": roles.value_counts().to_dict()},
            runtime_s=time.time() - t0, tier="0")


if __name__ == "__main__":
    main()
