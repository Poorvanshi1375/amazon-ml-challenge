# Business Entity Resolution — reproducible pipeline

Amazon ML Challenge 2026. For every Source 1 (S1) record, find the Source 2 / Source 3 (S2/S3) records
describing the same business. Produces `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

CPU only, no GPU, no neural network, no external data or APIs. Only the provided training data is used
for learning; test data is only transformed and scored.

## 1. Environment

- Python 3.12, Linux (tested: Ubuntu 24.04, AWS EC2 r6i.2xlarge — 8 vCPU, 64 GB RAM).
- `pip install -r requirements.txt` (all libraries are MIT / BSD / Apache / ISC licensed).
- Disk: ~60 GB free for intermediate files.

## 2. Data layout

```
<ER_DATA_DIR>/
  train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
  test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```
By default `ER_DATA_DIR` is `../../student_resource/dataset` relative to this folder.

## 3. Run end to end

```bash
ER_DATA_DIR=/path/to/dataset bash run_all.sh
```
Runs 11 steps (about 8 h on 8 cores), writes intermediates to `artifacts/` and the two submission files to
`output/`. Check them with the validator from the challenge starter kit (not included here):
```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir <ER_DATA_DIR>/test
```

| Step | Script | What it does | Time |
|---|---|---|---|
| 1 | `prepare.py` | Normalise names/addresses; learn a native-script word dictionary and per-country address rewrites from matched training pairs; assign holdout/training roles | ~7 min |
| 2–3 | `blocking.py --split train/test` | Char 2–4-gram TF-IDF search (name, name+address, address), each S2/S3 record retrieves its top S1 candidates in the same country, split by learned state partitions | ~110 / ~90 min |
| 4–5 | `prune.py --tune --learned`, `prune.py --split test` | Learned meta-blocking: small LightGBM on blocking signals; cut-off loses ≤ 0.1 pt recall. The kept pairs are `candidate_pairs.tsv` | ~20 / ~10 min |
| 6, 8 | `features.py --split train/test` | 75 pair features: string similarity, house-number relations, look-alike and group/sibling consistency, evidence uniqueness | ~15 min each |
| 7 | `train.py` | LightGBM matcher trained on S1 entities with complete candidate groups; threshold tuned for macro F0.5 on the holdout | ~60 min |
| 9 | `predict.py` | Scores test pairs with the stage-1 matcher | ~25 min |
| 10 | `stage2.py` | Stage-2 sibling model: out-of-fold stage-1 probabilities → collective features (accepted siblings, competing S1) → second LightGBM; scores holdout and test | ~120 min |
| 11 | `finalize.py` | Blend of the two matchers with the look-alike shift, one-owner, isotonic calibration (holdout fold A), expected-F0.5 decoding with an explicit empty option (α, m tuned on fold A, checked on fold B); writes `matching_results.tsv` and `candidate_pairs.tsv` | ~15 min |

## 4. Settings (all defaults reproduce the submitted result)

| Setting | Default | Where |
|---|---|---|
| Holdout share / training entities | 10% / 1,000,000 | `config.py` |
| Look-alike correction (house numbers known and different) | 2.5 (log-odds) | `config.LOOKALIKE_SHIFT` |
| Look-alike correction for the stage-2 model | 3.5 (log-odds) | `finalize.SHIFT_STAGE2` |
| Expected-F0.5 decoder α, m | α 1.5 (leaderboard-chosen; fold-A choice 1.0 is equal on the holdout), m 0.15 | `finalize.py` |
| Blocking n-grams / max_df / indexes | 2–4 / 0.05 / name, name+address, address | `blocking.py` |
| Candidates kept per query | name 5 (100 if the address is empty), name+address 10, address 10 | `blocking.py` |
| Seed | 42 | `config.py` |

Environment variables `ER_DATA_DIR`, `ER_ART_DIR`, `ER_OUT_DIR`, `ER_N_JOBS` change paths and parallelism only;
`ER_LOOKALIKE_SHIFT` / `ER_LOOKALIKE_SHIFT_EXTRA` override the correction strengths.

## 5. Code map (`src/`)

| File | Role |
|---|---|
| `config.py` | Paths, seeds, look-alike correction strength |
| `io_utils.py` | String-safe TSV reading, submission writing |
| `normalize.py` | Cleaning, transliteration, learned word dictionary and rewrites, French address rules (France has no labels) |
| `prepare.py` | Stage 1 runner |
| `split.py` | Deterministic holdout / training roles (crc32 of the S1 id) |
| `blocking.py` | Candidate generation (TF-IDF + learned partitions) |
| `prune.py` | Learned pruning (meta-blocking) |
| `features.py` | Pair, look-alike, group, sibling and uniqueness features |
| `train.py` | LightGBM matcher, threshold search, error report |
| `decode.py` | One-owner decoding, look-alike correction, expected-F0.5 alternative |
| `predict.py` | Stage-1 test inference |
| `stage2.py` | Stage-2 sibling (collective) model |
| `finalize.py` | Blend, calibration, expected-F0.5 decoding, final output files |
| `metric.py` | Macro F0.5 exactly as the leaderboard (reproduces the README example 0.714) |
| `tracking.py` | Experiment log (`artifacts/experiments.csv`) |
| `make_smoke.py` | 1% smoke dataset for fast end-to-end tests |

The methodology is described in [`docs/METHODOLOGY.md`](../../docs/METHODOLOGY.md).
