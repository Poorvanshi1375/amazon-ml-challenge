"""Experiment log: one row per run in artifacts/experiments.csv."""
import csv
import datetime as dt
import hashlib
import json

from config import ART_DIR

LOG = ART_DIR / "experiments.csv"
FIELDS = ["run_id", "timestamp", "stage", "tier", "config_hash", "params", "metrics", "runtime_s", "note"]


def log_run(stage, params, metrics, runtime_s=None, tier="", note=""):
    """Append a run; returns its run_id. params/metrics are dicts (stored as JSON)."""
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8]
    run_id = f"{stage}-{dt.datetime.now().strftime('%m%d-%H%M%S')}-{cfg}"
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "run_id": run_id, "timestamp": ts, "stage": stage, "tier": tier, "config_hash": cfg,
            "params": json.dumps(params, sort_keys=True, default=str),
            "metrics": json.dumps(metrics, sort_keys=True, default=str),
            "runtime_s": "" if runtime_s is None else round(runtime_s, 1), "note": note,
        })
    print(f"[tracking] {run_id} {json.dumps(metrics, default=str)}")
    return run_id
