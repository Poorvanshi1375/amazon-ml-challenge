# Roadmap — what I would do next

Ordered by expected gain per unit of effort.

1. **Cross-encoder on uncertain pairs.** Fine-tune a small multilingual encoder (e.g. MiniLM) on raw
   "name | address" pairs and rescore only candidates with 0.02 < p < 0.98. A parallel pipeline in our team gained
   +0.004 on the leaderboard this way (0.975 → 0.980). Needs a GPU for training.
2. **A test-like validation set that is calibrated.** Our proxy (synthetic look-alikes injected into the holdout)
   matched the leaderboard's level but not its ranking. Fit its distractor rate to the known leaderboard points before
   using it for decisions.
3. **k-fold cross-fitting instead of one holdout**, plus a leave-one-country-out check to estimate performance on an
   unlabelled country.
4. **An entity-level "does this business exist in the other sources?" model**, trained with deliberately created
   no-match entities that have look-alike records, to decide empty predictions directly.
5. **Per-country pruning cut-offs** to shrink the candidate set further (the US could go below 10 candidates per entity
   at the same recall).
6. **Synthetic hard negatives that match the real generator.** Our first attempt at synthetic look-alikes did not
   transfer to the leaderboard; matching the house-number gap and name-change distributions of real test look-alikes is
   the key.
7. **Engineering:** a smoke test in CI on the 1% sample (`make_smoke.py`), and caching of blocking indexes between runs.
