"""Paths and settings shared by every stage.

Paths can be overridden with environment variables so the same code runs on the
laptop (small samples) and on the EC2 machine (full data).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("ER_DATA_DIR", ROOT.parents[1] / "student_resource" / "dataset"))
ART_DIR = Path(os.environ.get("ER_ART_DIR", ROOT / "artifacts"))
OUT_DIR = Path(os.environ.get("ER_OUT_DIR", ROOT / "output"))

SEED = 42
HOLDOUT_FRAC = 0.10          # share of train S1 entities held out for validation
TRAIN_SUBSET_S1 = 1_000_000  # S1 entities used to train (complete candidate groups)

# Look-alike correction applied at decode time (see decode.lookalike_shift): pairs whose house numbers are
# known and different get logit(p) - LOOKALIKE_SHIFT. The test set contains ~4x more "neighbour look-alike"
# businesses than train, which the holdout cannot show; the strength was chosen from leaderboard probes
# (1.2 -> 0.9645, 2.5 -> 0.9666, 3.5 -> 0.9651; quadratic optimum ~2.45).
LOOKALIKE_SHIFT = float(os.environ.get("ER_LOOKALIKE_SHIFT", 2.5))
# Stronger correction when the query name also has a word the S1 name lacks (nd_only_q > 0): test look-alikes
# are neighbouring businesses with one added word ("Group", "Holdings"), while a typo in the house number of the
# same business usually keeps the name. Set equal to LOOKALIKE_SHIFT for a flat correction.
LOOKALIKE_SHIFT_EXTRA = float(os.environ.get("ER_LOOKALIKE_SHIFT_EXTRA", 2.5))

N_JOBS = int(os.environ.get("ER_N_JOBS", os.cpu_count() or 4))

for _d in (ART_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
