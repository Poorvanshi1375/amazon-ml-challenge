"""Deterministic holdout / training-subset assignment of train S1 entities.

Assignment depends only on the entity id and SEED (crc32 hash), so it is the same
on every machine and run, without storing a file.
  role = "holdout" : ~HOLDOUT_FRAC of S1 entities, never used for training or mining
  role = "train"   : up to TRAIN_SUBSET_S1 entities used for training (Tier 0/1)
  role = "rest"    : remaining entities (used only in the Tier 2 full-data run)
"""
import zlib

import numpy as np
import pandas as pd

from config import HOLDOUT_FRAC, SEED, TRAIN_SUBSET_S1


def _unit_hash(ids, salt):
    return np.array([zlib.crc32(f"{salt}:{i}".encode()) for i in ids], dtype=np.uint64) / 2**32


def assign_roles(s1_ids, subset_size=TRAIN_SUBSET_S1):
    ids = pd.Series(list(s1_ids))
    u = _unit_hash(ids, f"holdout{SEED}")
    role = np.where(u < HOLDOUT_FRAC, "holdout", "rest").astype(object)
    rest = np.flatnonzero(role == "rest")
    order = rest[np.argsort(_unit_hash(ids.iloc[rest], f"subset{SEED}"))]
    role[order[:subset_size]] = "train"
    return pd.Series(role, index=ids.values, name="role")


if __name__ == "__main__":
    ids = [f"S1-{i}" for i in range(100_000)]
    r = assign_roles(ids, subset_size=20_000)
    print(r.value_counts().to_dict())
    assert (assign_roles(ids, subset_size=20_000) == r).all()  # deterministic
    print("split OK")
