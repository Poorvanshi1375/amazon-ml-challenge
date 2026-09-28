"""Build a small smoke-test dataset with the same layout as the real one.

train: a random ~1% of S1, their ground-truth rows, their true S2/S3 matches,
       plus a random ~1% of S2/S3 (distractors); test: a random ~1% of every file.
Reads in chunks so it runs on a low-memory laptop.

Usage: python make_smoke.py <out_dir> [frac]
"""
import sys
from pathlib import Path

import pandas as pd

from config import DATA_DIR, SEED
from io_utils import READ_KW

CH = 500_000


def chunks(path):
    return pd.read_csv(path, chunksize=CH, **READ_KW)


def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def main(out, frac):
    out = Path(out)
    s1 = pd.concat(c.sample(frac=frac, random_state=SEED) for c in chunks(DATA_DIR / "train/train_source1.tsv"))
    ids = set(s1.entity_id)
    gt = pd.concat(c[c.source1_entity_id.isin(ids)] for c in chunks(DATA_DIR / "train/train_ground_truth.tsv"))
    need = {m for x in gt.matched_entity_ids for m in x.split(",") if m}
    write(s1, out / "train/train_source1.tsv")
    write(gt, out / "train/train_ground_truth.tsv")
    for s in (2, 3):
        df = pd.concat(pd.concat([c[c.entity_id.isin(need)], c.sample(frac=frac, random_state=SEED)]).drop_duplicates("entity_id")
                       for c in chunks(DATA_DIR / f"train/train_source{s}.tsv"))
        write(df, out / f"train/train_source{s}.tsv")
    for s in (1, 2, 3):
        df = pd.concat(c.sample(frac=frac, random_state=SEED) for c in chunks(DATA_DIR / f"test/test_source{s}.tsv"))
        write(df, out / f"test/test_source{s}.tsv")
    print(f"smoke data in {out}: train S1 {len(s1):,}, gt rows {len(gt):,}, matched ids {len(need):,}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.01)
