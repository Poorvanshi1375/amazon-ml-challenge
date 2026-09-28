# Amazon ML Challenge 2026 — Business Entity Resolution

Given business records from three independent sources with noisy names and addresses, find every Source 2 / Source 3
record that refers to the same real-world business as each Source 1 entity — across the **US, India and France**
(France has no training labels).

**Public leaderboard: 0.967216 macro F0.5** · CPU-only (8 cores, 64 GB) · no GPU, no external data · ~35M candidate
pairs (20.4 per Source 1 entity) at 99.3% blocking recall.

Team **2AM CODERS** — Poorvanshi Kochar (lead), Leena Shah, Krisha Patel, Khushi Parekh.

---

## The problem

- **Source 1** is a deduplicated reference list (1.73M test entities); **Sources 2 and 3** hold ~10M noisy records.
- Noise: abbreviations (Pvt/Private, St/Street), legal-suffix changes, "f/k/a"/DBA aliases, typos, word-order swaps,
  missing or reordered address parts, landmark addresses, and Indian names written in Devanagari/Telugu scripts.
- **Metric:** F0.5 computed per Source 1 entity and macro-averaged, singletons included — an entity with no true match
  scores 1.0 for an empty prediction and 0.0 for any prediction. Precision counts roughly twice as much as recall.

## Approach

```
raw TSVs
 → normalisation        learned native-script word dictionary, per-country address rewrites, French address rules
 → blocking             each S2/S3 record retrieves its top-K S1: char 2–4-gram TF-IDF, 3 indexes
                        (name / name+address / address), sparse top-K search, learned state partitions
 → learned pruning      small LightGBM on blocking signals, ≤ 0.1 pt recall loss        → candidate_pairs.tsv
 → 75 pair features     string similarity, house-number relations, look-alike, group, sibling, uniqueness
 → stage-1 matcher      LightGBM
 → stage-2 matcher      collective "sibling" features from out-of-fold stage-1 probabilities
 → decoding             blend + look-alike correction + one-owner + isotonic calibration
                        + expected-F0.5 decoding with an explicit "no match" option       → matching_results.tsv
```

### Key ideas
- **Learned normalisation.** A 6,111-word native-script → English dictionary and per-country abbreviation rewrites are
  mined from matched training pairs (e.g. "सदर्न" → "southern", "oh" → "ohio"; per country because India's "TN" is Tamil
  Nadu, not Tennessee).
- **Reverse, partitioned blocking.** Every record retrieves its most similar reference entities with sparse TF-IDF
  top-K search, restricted to the geographic partitions that cover 99.7% of training matches.
- **Recall-safe learned pruning.** A meta-blocking model keeps ~20 candidates per entity while losing ≤ 0.1 point of
  recall (checked across countries).
- **Evidence-strength features.** How many reference entities share a name, address or street (the term-frequency idea
  from probabilistic record linkage).
- **Handling a train/test shift.** The test set contains many "neighbour look-alike" businesses (same street, nearby
  house number, one extra name word) that are rare in training. Pairs with known, different house numbers get a
  log-odds penalty at decode time; its strength was chosen from leaderboard probes and verified label-free by comparing
  per-class acceptance rates between the holdout and the test set.
- **Metric-aware decoding.** Calibrated probabilities feed an expected-F0.5 decoder that can also predict "no match".

## Results

| Version | Holdout F0.5 | Public leaderboard |
|---|---|---|
| Minimal pipeline | — | 0.919 |
| Learned normalisation, partitions, pruning | 0.98282 | 0.960243 |
| + look-alike and group features | 0.98340 | 0.962619 |
| + uniqueness features, look-alike correction 2.5 | 0.98492 (uncorrected) | 0.966559 |
| + stage-2 model blend, French rules, calibrated expected-F0.5 decoding | 0.98319 (unseen half) | **0.967216** |

The holdout (10% of training entities) is drawn from the training distribution; the gap to the leaderboard reflects the
train/test shift described above. Full methodology: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Repository layout

```
code/business_entity_resolution/
├── run_all.sh          # end-to-end pipeline (11 steps)
├── requirements.txt    # pinned dependencies (Python 3.12)
├── README.md           # detailed run instructions and settings
└── src/
    ├── prepare.py      # normalisation, learned dictionary and rewrites, holdout roles
    ├── normalize.py    # text cleaning, transliteration, French address rules
    ├── blocking.py     # TF-IDF candidate search with learned partitions
    ├── prune.py        # learned pruning (meta-blocking)
    ├── features.py     # 75 pair / group / sibling / uniqueness features
    ├── train.py        # stage-1 LightGBM matcher
    ├── predict.py      # stage-1 test scoring
    ├── stage2.py       # stage-2 collective model
    ├── finalize.py     # blend, calibration, expected-F0.5 decoding, output files
    ├── decode.py  metric.py  split.py  io_utils.py  config.py  tracking.py  make_smoke.py
docs/METHODOLOGY.md     # methodology write-up submitted with the solution
```

## Running it

The challenge data is not included (it belongs to the organisers). With the data in
`<data>/train/` and `<data>/test/`:

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
ER_DATA_DIR=/path/to/data bash run_all.sh      # ~8 h on 8 cores, 64 GB RAM
```

Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`. See
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md) for step-by-step details,
settings and environment variables.

## Tech stack

Python 3.12 · pandas · NumPy · PyArrow · scikit-learn · LightGBM · RapidFuzz · sparse_dot_topn · anyascii · SciPy ·
AWS EC2 (r6i.2xlarge) for compute.
