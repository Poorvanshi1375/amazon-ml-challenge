"""Reading the challenge TSVs and writing the two submission files."""
import pandas as pd

from config import DATA_DIR

READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=3)


def read_tsv(path, **kw):
    """Read a challenge TSV with every value as a string.

    keep_default_na=False matters: otherwise empty addresses and the literal
    string "null" are turned into NaN.
    """
    try:
        return pd.read_csv(path, engine="pyarrow", **{k: v for k, v in READ_KW.items() if k != "quoting"}, **kw)
    except Exception:
        return pd.read_csv(path, **READ_KW, **kw)


def read_source(split, source, nrows=None):
    """split in {"train", "test"}, source in {1, 2, 3}."""
    path = DATA_DIR / split / f"{split}_source{source}.tsv"
    df = read_tsv(path) if nrows is None else pd.read_csv(path, nrows=nrows, **READ_KW)
    df["source"] = source
    return df


def read_truth_pairs():
    """Train ground truth as a DataFrame of (s1, other) pairs (vectorised, low memory)."""
    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    gt = gt[gt.matched_entity_ids != ""]
    pairs = gt.assign(other=gt.matched_entity_ids.str.split(",")).explode("other")
    pairs = pairs.rename(columns={"source1_entity_id": "s1"})[["s1", "other"]]
    return pairs[pairs.other != ""].reset_index(drop=True)


def write_id_lists(path, s1_ids, groups, id_col):
    """Write one row per S1 id; `groups` maps s1 id -> iterable of S2/S3 ids.

    Every S1 id gets exactly one row (empty list allowed), ids are de-duplicated
    and never quoted, as the validator requires.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{id_col}\n")
        for s1 in s1_ids:
            ids = groups.get(s1, ())
            seen = list(dict.fromkeys(ids))
            f.write(f"{s1}\t{','.join(seen)}\n")
